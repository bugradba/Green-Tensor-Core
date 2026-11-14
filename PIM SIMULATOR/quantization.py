"""
Gerçek doğruluk ölçümü için küçük CNN + tamsayı (integer) quantized çıkarım.

Saf numpy ile yazıldı (PyTorch gerekmez). Veri seti: scikit-learn ile gelen
el yazısı rakamlar (8x8, 1797 örnek); internet bağlantısı gerektirmez.

Quantization semantiği, simülatördeki PIMCluster.mac_8bit ile birebir aynıdır:

    8-bit : a, b ∈ [0, 255]      →  a · b
    4-bit : sadece üst nibble'lar →  ((a >> 4) · (b >> 4)) << 8
                                   = (a & 0xF0) · (b & 0xF0)          ("truncate")

Ayrıca aynı donanım maliyetiyle (tek 4x4 LUT çarpımı) çalışan, ama 4-bit
kodu yuvarlayarak kendi ölçeğiyle seçen bir alternatif de vardır ("round").

Ağırlıklar işaretli olduğu için işaret-büyüklük (sign-magnitude) gösterimi
kullanılır: PIM büyüklükleri çarpar, işaret ayrıca uygulanır. Aktivasyonlar
(giriş pikselleri ve ReLU çıkışları) zaten negatif değildir.
"""
import os

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(_HERE, 'digits_cnn.npz')

# Hassasiyeti seçilebilen (MAC yapan) katmanlar, çalışma sırasıyla
QUANT_LAYERS = ('conv1', 'conv2', 'fc1', 'fc2')


# ---------------------------------------------------------------------------
# Veri
# ---------------------------------------------------------------------------

def load_digits_data(test_size=0.3, seed=0):
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    d = load_digits()
    X = (d.images / 16.0).astype(np.float64)[:, None, :, :]  # (N,1,8,8), [0,1]
    y = d.target.astype(np.int64)
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


# ---------------------------------------------------------------------------
# Yardımcı işlemler (3x3, stride 1, padding 1 conv)
# ---------------------------------------------------------------------------

def _im2col(x):
    """x: (N,C,H,W) -> (N*H*W, C*9)"""
    N, C, H, W = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    win = sliding_window_view(xp, (3, 3), axis=(2, 3))  # (N,C,H,W,3,3)
    return win.transpose(0, 2, 3, 1, 4, 5).reshape(N * H * W, C * 9)


def _col2im(cols, shape):
    """_im2col'un gradyan karşılığı."""
    N, C, H, W = shape
    c = cols.reshape(N, H, W, C, 3, 3).transpose(0, 3, 4, 5, 1, 2)  # N,C,3,3,H,W
    dxp = np.zeros((N, C, H + 2, W + 2))
    for i in range(3):
        for j in range(3):
            dxp[:, :, i:i + H, j:j + W] += c[:, :, i, j]
    return dxp[:, :, 1:-1, 1:-1]


def _conv_from_cols(cols, W, b, N, H, Wd):
    out = cols @ W.reshape(W.shape[0], -1).T + b  # (N*H*W, C_out)
    return out.reshape(N, H, Wd, -1).transpose(0, 3, 1, 2)


def _maxpool2(x):
    N, C, H, W = x.shape
    r = x.reshape(N, C, H // 2, 2, W // 2, 2).transpose(0, 1, 2, 4, 3, 5).reshape(N, C, H // 2, W // 2, 4)
    arg = r.argmax(-1)
    out = np.take_along_axis(r, arg[..., None], -1)[..., 0]
    # Eşitlik durumunda gradyan yalnızca tek bir elemana gider
    # (aksi halde düz arka planda gradyan 4 kat sayılır).
    mask = np.zeros_like(r, dtype=bool)
    np.put_along_axis(mask, arg[..., None], True, -1)
    mask = mask.reshape(N, C, H // 2, W // 2, 2, 2).transpose(0, 1, 2, 4, 3, 5)
    return out, mask


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DigitsCNN:
    """
    conv1 (1→16, 3x3) → ReLU → conv2 (16→32, 3x3) → ReLU → MaxPool 2
    → fc1 (512→64) → ReLU → fc2 (64→10)
    """

    def __init__(self, params=None, seed=0):
        if params is None:
            rng = np.random.default_rng(seed)

            def he(shape, fan_in):
                return rng.normal(0, np.sqrt(2.0 / fan_in), shape)

            params = {
                'conv1_w': he((16, 1, 3, 3), 9), 'conv1_b': np.zeros(16),
                'conv2_w': he((32, 16, 3, 3), 144), 'conv2_b': np.zeros(32),
                'fc1_w': he((64, 512), 512), 'fc1_b': np.zeros(64),
                'fc2_w': he((10, 64), 64), 'fc2_b': np.zeros(10),
            }
        self.p = {k: np.asarray(v, dtype=np.float64) for k, v in params.items()}
        self.act_max = None  # calibrate() doldurur

    # --- Simülatör formatında katman listesi -------------------------------
    @staticmethod
    def layer_configs():
        return [
            {'name': 'conv1', 'type': 'Conv2D', 'input': (1, 8, 8), 'kernel': (16, 1, 3, 3), 'padding': 1},
            {'name': 'relu1', 'type': 'ReLU', 'input': (16, 8, 8)},
            {'name': 'conv2', 'type': 'Conv2D', 'input': (16, 8, 8), 'kernel': (32, 16, 3, 3), 'padding': 1},
            {'name': 'relu2', 'type': 'ReLU', 'input': (32, 8, 8)},
            {'name': 'pool', 'type': 'MaxPool', 'input': (32, 8, 8)},
            {'name': 'fc1', 'type': 'Linear', 'in_features': 512, 'out_features': 64},
            {'name': 'relu3', 'type': 'ReLU', 'input': (64,)},
            {'name': 'fc2', 'type': 'Linear', 'in_features': 64, 'out_features': 10},
        ]

    # --- Float ileri geçiş (eğitim için ara değerleri de döndürür) --------
    def _forward(self, x, layer_fn=None):
        """
        layer_fn(name, inp, W, b) verilirse MAC katmanları onunla hesaplanır
        (quantized çıkarım için). Her MAC katmanının girişi `inputs`'ta tutulur.
        """
        p = self.p
        N = x.shape[0]
        inputs = {}

        def mac(name, inp, W, b, kind):
            inputs[name] = inp
            if layer_fn is not None:
                return layer_fn(name, inp, W, b, kind)
            if kind == 'conv':
                return _conv_from_cols(_im2col(inp), W, b, N, inp.shape[2], inp.shape[3])
            return inp @ W.T + b

        z1 = mac('conv1', x, p['conv1_w'], p['conv1_b'], 'conv')
        a1 = np.maximum(z1, 0)
        z2 = mac('conv2', a1, p['conv2_w'], p['conv2_b'], 'conv')
        a2 = np.maximum(z2, 0)
        pooled, mask = _maxpool2(a2)
        flat = pooled.reshape(N, -1)
        z3 = mac('fc1', flat, p['fc1_w'], p['fc1_b'], 'fc')
        a3 = np.maximum(z3, 0)
        logits = mac('fc2', a3, p['fc2_w'], p['fc2_b'], 'fc')
        cache = dict(x=x, z1=z1, a1=a1, z2=z2, a2=a2, mask=mask, flat=flat, z3=z3, a3=a3)
        return logits, inputs, cache

    def predict_float(self, x):
        return self._forward(x)[0].argmax(1)

    # --- Eğitim ------------------------------------------------------------
    def _grads(self, x, y):
        p = self.p
        logits, _, c = self._forward(x)
        N = x.shape[0]
        e = np.exp(logits - logits.max(1, keepdims=True))
        prob = e / e.sum(1, keepdims=True)
        loss = -np.log(prob[np.arange(N), y] + 1e-12).mean()

        g = {}
        d = prob
        d[np.arange(N), y] -= 1
        d /= N
        g['fc2_w'] = d.T @ c['a3']; g['fc2_b'] = d.sum(0)
        d = (d @ p['fc2_w']) * (c['z3'] > 0)
        g['fc1_w'] = d.T @ c['flat']; g['fc1_b'] = d.sum(0)
        d = (d @ p['fc1_w']).reshape(N, 32, 4, 4)
        d = (d[:, :, :, None, :, None] * c['mask']).reshape(N, 32, 8, 8)
        d = d * (c['z2'] > 0)
        dcol = d.transpose(0, 2, 3, 1).reshape(-1, 32)
        cols2 = _im2col(c['a1'])
        g['conv2_w'] = (dcol.T @ cols2).reshape(p['conv2_w'].shape); g['conv2_b'] = dcol.sum(0)
        d = _col2im(dcol @ p['conv2_w'].reshape(32, -1), c['a1'].shape) * (c['z1'] > 0)
        dcol = d.transpose(0, 2, 3, 1).reshape(-1, 16)
        g['conv1_w'] = (dcol.T @ _im2col(c['x'])).reshape(p['conv1_w'].shape); g['conv1_b'] = dcol.sum(0)
        return loss, g

    def train(self, X, y, epochs=25, batch=64, lr=2e-3, seed=0, verbose=False):
        """Adam ile eğitim."""
        rng = np.random.default_rng(seed)
        m = {k: np.zeros_like(v) for k, v in self.p.items()}
        v = {k: np.zeros_like(val) for k, val in self.p.items()}
        t = 0
        for ep in range(epochs):
            idx = rng.permutation(len(X))
            for s in range(0, len(X), batch):
                b = idx[s:s + batch]
                loss, g = self._grads(X[b], y[b])
                t += 1
                for k in self.p:
                    m[k] = 0.9 * m[k] + 0.1 * g[k]
                    v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
                    mh = m[k] / (1 - 0.9 ** t)
                    vh = v[k] / (1 - 0.999 ** t)
                    self.p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
            if verbose:
                print(f"epoch {ep + 1:>2}: loss {loss:.4f}")
        return self

    # --- Kayıt -------------------------------------------------------------
    def save(self, path=DEFAULT_WEIGHTS):
        np.savez(path, **self.p)

    @classmethod
    def load(cls, path=DEFAULT_WEIGHTS):
        with np.load(path) as f:
            return cls({k: f[k] for k in f.files})

    # --- Quantization ------------------------------------------------------
    def calibrate(self, X_calib):
        """Her MAC katmanının giriş aktivasyonu için maksimum değeri ölç."""
        _, inputs, _ = self._forward(X_calib)
        self.act_max = {k: float(v.max()) for k, v in inputs.items()}
        return self

    def evaluate_quantized(self, X, y, bits, scheme='truncate'):
        """
        Tamsayı aritmetiğiyle quantized çıkarım; doğruluk döndürür.

        Args:
            bits: {'conv1': 8|4, 'conv2': ..., 'fc1': ..., 'fc2': ...}
            scheme: 4-bit katmanlar için 'truncate' (simülatör/pPIM semantiği)
                    veya 'round' (4-bit kod yuvarlanarak, kendi ölçeğiyle)
        """
        if self.act_max is None:
            raise RuntimeError("Önce calibrate() çağrılmalı")
        N = X.shape[0]

        def qlayer(name, inp, W, b, kind):
            a_code, a_scale = quantize_activation(inp, self.act_max[name], bits[name], scheme)
            w_code, w_sign, w_scale = quantize_weight(W, bits[name], scheme)
            w_int = (w_sign * w_code).astype(np.int64)
            if kind == 'conv':
                cols = _im2col(a_code.astype(np.int64))  # int64: tam sayı birikimi
                acc = cols @ w_int.reshape(W.shape[0], -1).T
                acc = acc.reshape(N, inp.shape[2], inp.shape[3], -1).transpose(0, 3, 1, 2)
                bias = b[None, :, None, None]
            else:
                acc = a_code.astype(np.int64) @ w_int.T
                bias = b
            return acc * (a_scale * w_scale) + bias

        logits = self._forward(X, layer_fn=qlayer)[0]
        return float((logits.argmax(1) == y).mean())


def quantize_activation(a, a_max, bits, scheme='truncate'):
    """Negatif olmayan aktivasyonu tamsayı koda çevirir. Returns (code, scale)."""
    a_max = max(a_max, 1e-12)
    if bits == 8 or scheme == 'truncate':
        scale = a_max / 255
        code = np.clip(np.rint(a / scale), 0, 255).astype(np.int64)
        if bits == 4:
            code = code & 0xF0  # PIM: sadece üst nibble kullanılır
        return code, scale
    if scheme == 'round' and bits == 4:
        scale = a_max / 15
        return np.clip(np.rint(a / scale), 0, 15).astype(np.int64), scale
    raise ValueError(f"Desteklenmeyen bits/scheme: {bits}/{scheme}")


def quantize_weight(W, bits, scheme='truncate'):
    """İşaret-büyüklük quantization. Returns (magnitude_code, sign, scale)."""
    w_max = max(float(np.abs(W).max()), 1e-12)
    sign = np.sign(W).astype(np.int64)
    if bits == 8 or scheme == 'truncate':
        scale = w_max / 255
        code = np.clip(np.rint(np.abs(W) / scale), 0, 255).astype(np.int64)
        if bits == 4:
            code = code & 0xF0
        return code, sign, scale
    if scheme == 'round' and bits == 4:
        scale = w_max / 15
        return np.clip(np.rint(np.abs(W) / scale), 0, 15).astype(np.int64), sign, scale
    raise ValueError(f"Desteklenmeyen bits/scheme: {bits}/{scheme}")


def get_trained_model(path=DEFAULT_WEIGHTS, retrain=False, seed=0):
    """
    Eğitilmiş modeli ve veriyi döndürür (gerekirse eğitip kaydeder).
    Kalibrasyon eğitim verisiyle yapılır; doğruluk test verisinde ölçülür.
    """
    X_tr, X_te, y_tr, y_te = load_digits_data(seed=seed)
    if retrain or not os.path.exists(path):
        model = DigitsCNN(seed=seed).train(X_tr, y_tr, seed=seed)
        model.save(path)
    else:
        model = DigitsCNN.load(path)
    model.calibrate(X_tr)
    return model, (X_tr, X_te, y_tr, y_te)


if __name__ == "__main__":
    import time
    t = time.time()
    model, (X_tr, X_te, y_tr, y_te) = get_trained_model(retrain=True)
    print(f"Eğitim: {time.time() - t:.1f} s")
    print(f"Float doğruluk : %{(model.predict_float(X_te) == y_te).mean() * 100:.2f}")
    for scheme in ('truncate', 'round'):
        for b in (8, 4):
            acc = model.evaluate_quantized(X_te, y_te, {k: b for k in QUANT_LAYERS}, scheme)
            print(f"{scheme:<9} tüm katmanlar {b}-bit: %{acc * 100:.2f}")
