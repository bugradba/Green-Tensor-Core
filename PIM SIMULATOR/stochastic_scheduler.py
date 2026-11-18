"""
Belirsizlik altında saatlik plan seçimi: dinamik programlama vs. Q-Learning.

carbon_scheduler.py günün tamamını önceden bilir. Gerçekte:
    - Saatin trafiği ancak o saat başladığında görülür (tahminden sapar).
    - Günün bulutluluğu (dolayısıyla güneş payı ve karbon yoğunluğu) sabah
      hava tahmininden bilinir.
    - Doğruluk bütçesi günlüktür: erken harcanan bütçe akşama kalmaz.

Bu, gerçek bir sıralı karar problemidir. Durum:
    (saat, kalan bütçe, bu saatin trafiği, günün bulutluluğu)

Karşılaştırılan politikalar (hepsi aynı ortamda, aynı test günlerinde):
    oracle      : günü önceden bilen MILP (ulaşılamaz alt sınır)
    myopic      : her saat kendi içinde bütçe ≤ B (gelecek düşünülmez)
    open_loop   : sabah tahminle MILP, gün boyu uygulanır (gerekirse güvenli geri dönüş)
    mpc         : her saat başı kalan gün için MILP yeniden çözülür (receding horizon)
    sdp         : stokastik dinamik programlama (gürültü dağılımını bilir)
    qlearning   : tablo tabanlı Q-Learning (modeli bilmez, simülasyonla öğrenir)

Kısıtlar her politikada maskeyle zorlanır, yani hiçbir politika bütçeyi veya
kapasiteyi ihlal edemez; politikalar yalnızca karbonda farklılaşır.
"""
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from carbon_model import _solar_shape, traffic_profile

H = 24
CLOUD_LEVELS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])   # 0 = açık, 1 = kapalı


def pareto_plans(plans):
    """Enerji, gecikme (min) ve doğruluk (max) açısından baskın olmayan planlar."""
    def dominates(a, b):
        no_worse = (a['energy'] <= b['energy'] and a['latency'] <= b['latency']
                    and a['accuracy'] >= b['accuracy'])
        better = (a['energy'] < b['energy'] or a['latency'] < b['latency']
                  or a['accuracy'] > b['accuracy'])
        return no_worse and better
    return [p for p in plans if not any(dominates(o, p) for o in plans)]


# ---------------------------------------------------------------------------
# Ortam
# ---------------------------------------------------------------------------

class DayEnvironment:
    """
    Bir günün rastgele gerçekleşmesi:
        trafik_h = tahmin_h × exp(σ z_h − σ²/2),  z_h ~ N(0,1) bağımsız
        bulut    ~ CLOUD_LEVELS içinden eşit olasılıkla, sabah bilinir
        güneş_h  = solar_peak × (1 − 0.8·bulut) × şekil_h
        CI_h     = non_solar × (1 − güneş_h)

    Doğruluk bütçesi mutlak birimdedir:
        Σ_h n_h (ref − acc_h) ≤ B × (beklenen günlük çıkarım sayısı)
    """

    def __init__(self, plans, reference_accuracy, budget=0.005, peak_rate=120_000,
                 sigma=0.25, rho_max=0.8, non_solar=460.0, solar_peak=0.30,
                 sunrise=5.5, sunset=20.0):
        self.plans = list(plans)
        self.E = np.array([p['energy'] for p in self.plans])           # mJ
        self.L = np.array([p['latency'] for p in self.plans]) / 1e3    # s
        self.loss_per_inf = reference_accuracy - np.array([p['accuracy'] for p in self.plans])
        self.ref = reference_accuracy
        self.A = len(self.plans)
        self.sigma = sigma
        self.rho_max = rho_max
        self.forecast_rate = peak_rate * traffic_profile()
        self.non_solar = non_solar
        self.solar = solar_peak * _solar_shape(sunrise, sunset)
        self.budget = budget
        self.loss_budget = budget * self.forecast_rate.sum() * 3600.0
        # g CO2 / (çıkarım · gCO2/kWh): mJ → kWh
        self.kwh_per_inf = self.E / 1e3 / 3.6e6
        # Kayıpsız (8-bit) planlar arasında en hızlısının kaldırabileceği hız
        zero_loss = self.loss_per_inf <= 1e-12
        if not zero_loss.any():
            raise ValueError("Plan kümesinde kayıpsız (tamamen 8-bit) bir plan olmalı")
        self.max_rate = 0.99 * rho_max / self.L[zero_loss].min()

    def ci(self, cloud):
        return self.non_solar * (1.0 - self.solar * (1.0 - 0.8 * cloud))

    def sample_day(self, rng):
        z = rng.standard_normal(H)
        rate = self.forecast_rate * np.exp(self.sigma * z - self.sigma ** 2 / 2)
        # Kabul kontrolü: en hızlı planın kaldıramayacağı fazla trafik AI dışı
        # yedek yola gider (her saat en az bir uygun plan olsun).
        rate = np.minimum(rate, self.max_rate)
        cloud_idx = int(rng.integers(len(CLOUD_LEVELS)))
        return {'rate': rate, 'z': z, 'cloud_idx': cloud_idx,
                'ci': self.ci(CLOUD_LEVELS[cloud_idx])}

    # --- adım yardımcıları -------------------------------------------------
    def carbon(self, rate, ci_h):
        """Her plan için o saatin karbonu (g)."""
        return rate * 3600.0 * self.kwh_per_inf * ci_h

    def loss(self, rate):
        return rate * 3600.0 * self.loss_per_inf

    def allowed(self, rate, used):
        """Kapasiteyi ve kalan bütçeyi aşmayan planlar."""
        return (rate * self.L <= self.rho_max + 1e-12) & \
               (used + self.loss(rate) <= self.loss_budget * (1 + 1e-12))

    def run(self, policy, day):
        """Politikayı bir günde çalıştırır. Returns: toplam g CO2, kullanılan bütçe, seçimler."""
        policy.start_day(day)
        used, total, choices = 0.0, 0.0, []
        for h in range(H):
            r = day['rate'][h]
            mask = self.allowed(r, used)
            a = policy.act(h, r, used, mask)
            if not mask[a]:
                raise RuntimeError(f"{type(policy).__name__} izin verilmeyen aksiyon seçti")
            total += self.carbon(r, day['ci'][h])[a]
            used += self.loss(r)[a]
            choices.append(a)
        return total, used, choices


# ---------------------------------------------------------------------------
# MILP yardımcı (oracle / open loop / MPC)
# ---------------------------------------------------------------------------

def solve_milp(env, rates, cis, loss_budget, first_hour_mask=None):
    """
    Verilen saatlik trafik ve karbon için kesin optimum (len(rates) saat).
    Bütçe kısıtı 1e4 ölçeklenir (çözücü toleransı). Returns: aksiyon listesi veya None.
    """
    T, A = len(rates), env.A
    cost = (np.asarray(rates)[:, None] * env.kwh_per_inf[None, :] * np.asarray(cis)[:, None]).ravel()
    cost = cost / max(cost.max(), 1e-30)
    loss = (np.asarray(rates)[:, None] * 3600.0 * env.loss_per_inf[None, :]).ravel()
    scale = 1e4 / max(env.loss_budget, 1e-30)

    A_eq = np.zeros((T, T * A))
    for t in range(T):
        A_eq[t, t * A:(t + 1) * A] = 1
    feas = (np.asarray(rates)[:, None] * env.L[None, :] <= env.rho_max + 1e-12)
    if first_hour_mask is not None:
        feas[0] &= first_hour_mask
    ub = feas.ravel().astype(float)

    margin = 0.0
    for _ in range(3):
        cons = [LinearConstraint(A_eq, 1, 1),
                LinearConstraint(scale * loss[None, :], -np.inf, scale * (loss_budget - margin))]
        res = milp(cost, constraints=cons, integrality=np.ones(T * A), bounds=Bounds(0, ub))
        if not res.success:
            return None
        acts = res.x.reshape(T, A).argmax(axis=1)
        if (loss.reshape(T, A)[np.arange(T), acts].sum() <= loss_budget * (1 + 1e-12)
                and feas[np.arange(T), acts].all()):
            return list(acts)
        margin += 1e-6 * env.loss_budget
    return None


# ---------------------------------------------------------------------------
# Politikalar
# ---------------------------------------------------------------------------

class Policy:
    name = 'policy'

    def __init__(self, env):
        self.env = env

    def start_day(self, day):
        self.day = day

    def _safe(self, h, r, mask):
        """İzin verilen en düşük karbonlu plan."""
        c = self.env.carbon(r, self.day['ci'][h])
        return int(np.argmin(np.where(mask, c, np.inf)))


class OraclePolicy(Policy):
    name = 'oracle'

    def start_day(self, day):
        super().start_day(day)
        self.plan = solve_milp(self.env, day['rate'], day['ci'], self.env.loss_budget)

    def act(self, h, r, used, mask):
        return self.plan[h]


class MyopicPolicy(Policy):
    """Her saat: o saatin çıkarım başına kaybı ≤ B olan en düşük karbonlu plan."""
    name = 'myopic'

    def act(self, h, r, used, mask):
        ok = mask & (self.env.loss_per_inf <= self.env.budget + 1e-12)
        return self._safe(h, r, ok if ok.any() else mask)


class OpenLoopPolicy(Policy):
    """Sabah tahminle planla; planlanan aksiyon gerçekleşen saatte izinli değilse güvenli seçim."""
    name = 'open_loop'

    def start_day(self, day):
        super().start_day(day)
        self.plan = solve_milp(self.env, self.env.forecast_rate, day['ci'], self.env.loss_budget)

    def act(self, h, r, used, mask):
        a = self.plan[h] if self.plan is not None else None
        if a is not None and mask[a]:
            return a
        # Planlanan plandan daha az kayıplı, izinli en düşük karbonlu plan
        ok = mask & (self.env.loss_per_inf <= (self.env.loss_per_inf[a] if a is not None else 0) + 1e-12)
        return self._safe(h, r, ok if ok.any() else mask)


class MPCPolicy(Policy):
    """Her saat başı: bu saat gerçekleşen trafikle, gelecek saatler tahminle MILP."""
    name = 'mpc'

    def act(self, h, r, used, mask):
        rates = np.concatenate([[r], self.env.forecast_rate[h + 1:]])
        plan = solve_milp(self.env, rates, self.day['ci'][h:], self.env.loss_budget - used,
                          first_hour_mask=mask)
        if plan is None:
            return self._safe(h, r, mask)
        return plan[0]


class SDPPolicy(Policy):
    """
    Stokastik dinamik programlama (geriye doğru tümevarım).

    V_h(b, c) = E_z [ min_a  carbon_h(a, z, c) + V_{h+1}(b − loss_h(a, z)) ]
    b: kalan bütçe (K aralığa bölünmüş), c: bulut seviyesi, z: trafik gürültüsü
    (M kuantil noktasıyla). Çalışırken, gözlenen gerçek trafikle tek adım
    ileriye bakılır. Kalan bütçe aşağı yuvarlanır (temkinli).
    """
    name = 'sdp'

    def __init__(self, env, K=200, M=21):
        super().__init__(env)
        self.K = K
        from scipy.stats import norm
        q = (np.arange(M) + 0.5) / M
        self.z = norm.ppf(q)
        t0 = time.perf_counter()
        self._solve()
        self.solve_seconds = time.perf_counter() - t0

    def _bin(self, remaining):
        """Kalan bütçe → bin (aşağı yuvarlama)."""
        return np.clip(np.floor(remaining / self.env.loss_budget * self.K + 1e-9), 0, self.K).astype(int)

    def _solve(self):
        env, K = self.env, self.K
        C = len(CLOUD_LEVELS)
        self.V = np.zeros((H + 1, C, K + 1))
        b_vals = np.arange(K + 1) / K * env.loss_budget            # kalan bütçe
        for h in range(H - 1, -1, -1):
            for c in range(C):
                ci_h = env.ci(CLOUD_LEVELS[c])[h]
                acc = np.zeros(K + 1)
                for z in self.z:
                    r = min(env.forecast_rate[h] * np.exp(env.sigma * z - env.sigma ** 2 / 2), env.max_rate)
                    carbon = env.carbon(r, ci_h)                    # (A,)
                    loss = env.loss(r)                              # (A,)
                    cap = r * env.L <= env.rho_max + 1e-12
                    rem = b_vals[:, None] - loss[None, :]           # (K+1, A)
                    ok = cap[None, :] & (rem >= -1e-9 * env.loss_budget)
                    nxt = self.V[h + 1, c][self._bin(np.maximum(rem, 0))]
                    q = np.where(ok, carbon[None, :] + nxt, np.inf)
                    acc += q.min(axis=1)
                self.V[h, c] = acc / len(self.z)

    def act(self, h, r, used, mask):
        env = self.env
        c = self.day['cloud_idx']
        rem = env.loss_budget - used - env.loss(r)
        nxt = self.V[h + 1, c][self._bin(np.maximum(rem, 0))]
        q = np.where(mask, env.carbon(r, self.day['ci'][h]) + nxt, np.inf)
        return int(np.argmin(q))


class QLearningPolicy(Policy):
    """
    Tablo tabanlı Q-Learning. Ortamın modelini bilmez; yalnızca simüle
    edilen günlerden öğrenir.

    Durum : (saat, kalan bütçe bini, trafik sapma bini, bulut seviyesi)
    Aksiyon: Pareto planları (izin verilmeyenler maskelenir)
    Ödül  : −karbon (günlük GPU-only karbonuna göre normalize)
    """
    name = 'qlearning'

    def __init__(self, env, budget_bins=40, rate_bins=3, seed=0):
        super().__init__(env)
        self.nb, self.nr = budget_bins, rate_bins
        self.Q = np.zeros((H, budget_bins, rate_bins, len(CLOUD_LEVELS), env.A))
        self.visits = np.zeros(self.Q.shape)
        self.rng = np.random.default_rng(seed)
        self.episodes = 0
        gpu = int(np.argmax(env.E))                                 # en pahalı plan ölçek için
        hourly = env.forecast_rate * 3600 * env.kwh_per_inf[gpu] * env.non_solar
        self.scale = hourly.sum()
        # Kötümser başlangıç: Q(s, a) = −(o saatten gün sonuna en pahalı planın maliyeti).
        # Sıfırla başlatmak hatalıdır: maliyetler negatif ödül olduğu için hiç
        # denenmemiş aksiyonlar "en iyi" görünür ve test sırasında seçilir.
        cost_to_go = np.cumsum(hourly[::-1])[::-1] / self.scale
        self.Q[:] = -cost_to_go[:, None, None, None, None]
        self.training_seconds = 0.0

    def _state(self, h, r, used, cloud_idx):
        rem = max(self.env.loss_budget - used, 0) / self.env.loss_budget
        b = min(int(rem * self.nb), self.nb - 1)   # tam dolu bütçe son kutuya düşer
        dev = np.log(r / self.env.forecast_rate[h]) / self.env.sigma   # ≈ z
        rb = int(np.clip(np.floor((dev + 1.5) / 3.0 * self.nr), 0, self.nr - 1))
        return h, b, rb, cloud_idx

    def train(self, n_episodes, alpha0=0.5, eps0=0.6, gamma=1.0, exploring_starts=False):
        """
        exploring_starts: her bölüm rastgele bir saatte ve rastgele kalan bütçeyle
        başlar. Denendi; bu problemde öğrenmeyi yavaşlattığı için varsayılan kapalı.
        """
        env = self.env
        t0 = time.perf_counter()
        for _ in range(n_episodes):
            self.episodes += 1
            eps = max(0.02, eps0 / (1 + self.episodes / 20000))
            day = env.sample_day(self.rng)
            if exploring_starts:
                h0 = int(self.rng.integers(H))
                used = 0.0 if self.rng.random() < 0.25 else float(self.rng.random()) * env.loss_budget
            else:
                h0, used = 0, 0.0
            s = self._state(h0, day['rate'][h0], used, day['cloud_idx'])
            for h in range(h0, H):
                r = day['rate'][h]
                mask = env.allowed(r, used)
                idx = np.flatnonzero(mask)
                if self.rng.random() < eps:
                    a = int(self.rng.choice(idx))
                else:
                    a = int(idx[np.argmax(self.Q[s][idx])])
                reward = -env.carbon(r, day['ci'][h])[a] / self.scale
                used += env.loss(r)[a]
                if h + 1 < H:
                    r2 = day['rate'][h + 1]
                    s2 = self._state(h + 1, r2, used, day['cloud_idx'])
                    m2 = env.allowed(r2, used)
                    target = reward + gamma * self.Q[s2][m2].max()
                else:
                    s2, target = None, reward
                self.visits[s + (a,)] += 1
                alpha = max(0.02, alpha0 / self.visits[s + (a,)] ** 0.6)
                self.Q[s + (a,)] += alpha * (target - self.Q[s + (a,)])
                s = s2
        self.training_seconds += time.perf_counter() - t0
        return self

    def act(self, h, r, used, mask):
        s = self._state(h, r, used, self.day['cloud_idx'])
        idx = np.flatnonzero(mask)
        return int(idx[np.argmax(self.Q[s][idx])])


def evaluate(env, policy, days):
    """Politikanın test günlerindeki karbonları ve karar süresi."""
    co2, used = [], []
    t0 = time.perf_counter()
    for d in days:
        c, u, _ = env.run(policy, d)
        co2.append(c)
        used.append(u)
    return {'co2_g': np.array(co2), 'budget_used': np.array(used) / env.loss_budget,
            'ms_per_decision': (time.perf_counter() - t0) / (len(days) * H) * 1e3}
