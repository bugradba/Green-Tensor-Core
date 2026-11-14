"""
Katman bazında cihaz + hassasiyet planlayıcı.

Her MAC katmanı için üç seçenek vardır:
    GPU   : Jetson Nano, INT8
    PIM8  : PIM, 8-bit
    PIM4  : PIM, 4-bit (daha az enerji, daha hızlı, doğruluk kaybı olabilir)

Amaç:  minimum enerji
Kısıtlar:
    doğruluk ≥ (tüm katmanlar 8-bit doğruluğu) − acc_budget   [gerçekten ölçülür]
    toplam gecikme ≤ max_latency_ms                           [opsiyonel]

Doğruluk toplanabilir bir büyüklük değildir: iki katmanı birlikte 4-bit yapmanın
etkisi, tek tek etkilerinin toplamı olmayabilir. Bu yüzden her aday plan için
doğruluk tamsayı çıkarımla gerçekten ölçülür (önbelleğe alınarak).

İki yöntem:
    exhaustive() : tüm 3^L planı dener (küçük ağlar için kesin optimum)
    greedy()     : büyük ağlar için ölçeklenebilir sezgisel yöntem
"""
import itertools

from hybrid_scheduler import simulate_layer
from quantization import QUANT_LAYERS

CHOICES = {
    'GPU': ('GPU', 8),
    'PIM8': ('PIM', 8),
    'PIM4': ('PIM', 4),
}


class PrecisionPlanner:
    def __init__(self, model, X_eval, y_eval, pim, gpu, scheme='truncate'):
        self.model = model
        self.X, self.y = X_eval, y_eval
        self.pim, self.gpu = pim, gpu
        self.scheme = scheme
        self.layers = model.layer_configs()
        self.quant_layers = [l['name'] for l in self.layers if l['name'] in QUANT_LAYERS]
        self._acc_cache = {}

        # Eleman bazlı katmanlar (ReLU, Pool) her planda PIM'de, sabit maliyet
        self.fixed_energy = 0.0
        self.fixed_latency = 0.0
        self._cost = {}
        for layer in self.layers:
            if layer['name'] in self.quant_layers:
                for c, (device, prec) in CHOICES.items():
                    self._cost[(layer['name'], c)] = simulate_layer(layer, device, pim, gpu, prec)
            else:
                e, l = simulate_layer(layer, 'PIM', pim, gpu)
                self.fixed_energy += e
                self.fixed_latency += l

        self.reference_accuracy = self.accuracy({n: 'PIM8' for n in self.quant_layers})

    # ------------------------------------------------------------------
    def layer_cost(self, name, choice):
        """(enerji mJ, gecikme ms)"""
        return self._cost[(name, choice)]

    def accuracy(self, plan):
        bits = {n: CHOICES[plan[n]][1] for n in self.quant_layers}
        key = tuple(bits[n] for n in self.quant_layers)
        if key not in self._acc_cache:
            self._acc_cache[key] = self.model.evaluate_quantized(self.X, self.y, bits, self.scheme)
        return self._acc_cache[key]

    def evaluate(self, plan):
        e = self.fixed_energy + sum(self.layer_cost(n, plan[n])[0] for n in self.quant_layers)
        l = self.fixed_latency + sum(self.layer_cost(n, plan[n])[1] for n in self.quant_layers)
        return {'plan': dict(plan), 'energy': e, 'latency': l, 'accuracy': self.accuracy(plan)}

    def _feasible(self, r, acc_budget, max_latency_ms):
        if r['accuracy'] < self.reference_accuracy - acc_budget - 1e-12:
            return False
        if max_latency_ms is not None and r['latency'] > max_latency_ms + 1e-12:
            return False
        return True

    # ------------------------------------------------------------------
    def all_plans(self):
        for combo in itertools.product(CHOICES, repeat=len(self.quant_layers)):
            yield self.evaluate(dict(zip(self.quant_layers, combo)))

    def exhaustive(self, acc_budget=0.01, max_latency_ms=None):
        """Kesin optimum. Uygun plan yoksa None."""
        feasible = [r for r in self.all_plans() if self._feasible(r, acc_budget, max_latency_ms)]
        if not feasible:
            return None
        return min(feasible, key=lambda r: (r['energy'], -r['accuracy']))

    def sensitivity(self):
        """Her katman tek başına 4-bit yapıldığında doğruluk kaybı."""
        base = {n: 'PIM8' for n in self.quant_layers}
        out = {}
        for n in self.quant_layers:
            p = dict(base)
            p[n] = 'PIM4'
            out[n] = self.reference_accuracy - self.accuracy(p)
        return out

    def greedy(self, acc_budget=0.01, max_latency_ms=None):
        """
        Büyük ağlar için ölçeklenebilir sezgisel yöntem (her adımda doğruluk ölçülür):

        1) Her katmanı en az enerjili 8-bit seçenekle başlat.
        2) Gecikme sınırı aşılıyorsa: (gecikme kazancı / enerji artışı) oranı en iyi
           olan tek-katman hamlesini uygula. Önce doğruluk bütçesi harcamayan
           8-bit hamleler (PIM8 -> GPU) denenir; yetmezse 4-bit hamlelere izin verilir.
        3) 4-bit'e geçişleri (enerji kazancı / hassasiyet) sırasıyla dene; her iki
           kısıt da sağlanıyorsa kabul et.
        4) Yerel iyileştirme: enerjiyi düşüren ve uygun kalan tek veya iki
           katmanlık değişiklik kalmayana kadar uygula.

        Uygun plan bulunamazsa None.
        """
        def ok_acc(r):
            return r['accuracy'] >= self.reference_accuracy - acc_budget - 1e-12

        def ok_lat(r):
            return max_latency_ms is None or r['latency'] <= max_latency_ms + 1e-12

        def neighbours(plan, choices=tuple(CHOICES)):
            for n in self.quant_layers:
                for c in choices:
                    if c != plan[n]:
                        trial = dict(plan)
                        trial[n] = c
                        yield trial

        def pair_neighbours(plan):
            for n1, n2 in itertools.combinations(self.quant_layers, 2):
                for c1 in CHOICES:
                    for c2 in CHOICES:
                        if c1 != plan[n1] and c2 != plan[n2]:
                            trial = dict(plan)
                            trial[n1], trial[n2] = c1, c2
                            yield trial

        def reduce_latency(plan, cur, choices):
            while not ok_lat(cur):
                best, best_score = None, None
                for trial in neighbours(plan, choices):
                    r = self.evaluate(trial)
                    dl = cur['latency'] - r['latency']
                    if dl <= 0 or not ok_acc(r):
                        continue
                    score = dl / (max(r['energy'] - cur['energy'], 0) + 1e-9)
                    if best_score is None or score > best_score:
                        best, best_score = (trial, r), score
                if best is None:
                    return plan, cur
                plan, cur = best
            return plan, cur

        plan = {n: min(('GPU', 'PIM8'), key=lambda c: self.layer_cost(n, c)[0])
                for n in self.quant_layers}

        # 2) Gecikme
        cur = self.evaluate(plan)
        plan, cur = reduce_latency(plan, cur, ('GPU', 'PIM8'))
        plan, cur = reduce_latency(plan, cur, tuple(CHOICES))
        if not ok_lat(cur):
            return None

        # 3) 4-bit geçişleri
        sens = self.sensitivity()

        def gain(n):
            return self.layer_cost(n, plan[n])[0] - self.layer_cost(n, 'PIM4')[0]

        for n in sorted(self.quant_layers, key=lambda n: -gain(n) / (max(sens[n], 0) + 1e-4)):
            if plan[n] == 'PIM4' or gain(n) <= 0:
                continue
            trial = dict(plan)
            trial[n] = 'PIM4'
            r = self.evaluate(trial)
            if ok_acc(r) and ok_lat(r):
                plan, cur = trial, r

        # 4) Yerel iyileştirme
        improved = True
        while improved:
            improved = False
            for trial in itertools.chain(neighbours(plan), pair_neighbours(plan)):
                r = self.evaluate(trial)
                if ok_acc(r) and ok_lat(r) and r['energy'] < cur['energy'] - 1e-15:
                    plan, cur, improved = trial, r, True
                    break

        return cur if (ok_acc(cur) and ok_lat(cur)) else None


def pareto_front(results):
    """Enerji (min) ve doğruluk (max) için Pareto-optimal planlar."""
    pts = sorted(results, key=lambda r: (r['energy'], -r['accuracy']))
    front, best_acc = [], -1
    for r in pts:
        if r['accuracy'] > best_acc + 1e-12:
            front.append(r)
            best_acc = r['accuracy']
    return front


def plan_str(plan):
    return ' '.join(f"{n}:{c}" for n, c in plan.items())
