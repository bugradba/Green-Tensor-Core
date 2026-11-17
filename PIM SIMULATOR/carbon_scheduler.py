"""
Karbon farkında, saat bazlı plan seçimi.

Her saat h için katman bazında bir plan p (cihaz + hassasiyet, bkz.
precision_planner) seçilir. Bir günün toplam karbonu:

    CO2 = Σ_h  n_h · E_p(h) · CI_h          n_h = o saatteki çıkarım sayısı

Kısıtlar:
    kapasite : rate_h · latency_p ≤ ρ_max          (cihaz o saatin yükünü kaldırmalı)
    doğruluk : Σ_h n_h · acc_p(h) / Σ_h n_h ≥ acc_INT8 − B   (günlük ortalama)

Karşılaştırılan stratejiler (hepsi aynı kapasite kuralına uyar):
    gpu_only      : her saat tüm katmanlar GPU (INT8)
    static        : bütün gün tek plan (tepe saatin yükünü kaldırabilen)
    hourly_energy : her saat kendi içinde en az enerji, saat başına doğruluk ≥ ref − B
    daily_energy  : günlük bütçe, karbon yoğunluğu sabit sayılır (sadece trafik farkında)
    carbon_aware  : günlük bütçe + gerçek saatlik karbon yoğunluğu

daily_energy ile carbon_aware arasındaki fark, tam olarak "karbon farkındalığının"
katkısıdır; hourly_energy ile daily_energy arasındaki fark ise doğruluk
bütçesini gün içinde kaydırabilmenin katkısıdır.
"""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

STRATEGIES = ('gpu_only', 'static', 'hourly_energy', 'daily_energy', 'carbon_aware')


class CarbonAwareScheduler:
    def __init__(self, plans, reference_accuracy, rate_per_s, carbon_gco2_kwh, rho_max=0.8):
        """
        Args:
            plans: PrecisionPlanner.all_plans() çıktısı (energy mJ, latency ms, accuracy)
            reference_accuracy: tüm katmanlar INT8 doğruluğu
            rate_per_s: 24 saatlik çıkarım hızı (çıkarım/s)
            carbon_gco2_kwh: 24 saatlik şebeke karbon yoğunluğu
            rho_max: izin verilen en yüksek cihaz kullanım oranı
        """
        self.plans = list(plans)
        self.ref = reference_accuracy
        self.rate = np.asarray(rate_per_s, dtype=float)
        self.ci = np.asarray(carbon_gco2_kwh, dtype=float)
        if self.rate.shape != (24,) or self.ci.shape != (24,):
            raise ValueError("rate ve carbon 24 saatlik olmalı")
        self.rho_max = rho_max

        self.E = np.array([p['energy'] for p in self.plans])          # mJ / çıkarım
        self.L = np.array([p['latency'] for p in self.plans]) / 1e3   # s / çıkarım
        self.A = np.array([p['accuracy'] for p in self.plans])
        self.n = self.rate * 3600.0                                    # çıkarım / saat
        # feasible[h, p]
        self.feasible = self.rate[:, None] * self.L[None, :] <= rho_max + 1e-12
        self._gpu_idx = next(i for i, p in enumerate(self.plans)
                             if all(c == 'GPU' for c in p['plan'].values()))

    # ------------------------------------------------------------------
    def evaluate(self, choice):
        """choice: 24 plan indeksi → günlük metrikler."""
        choice = np.asarray(choice)
        energy_j = self.n * self.E[choice] / 1e3
        kwh = energy_j / 3.6e6
        co2_g = kwh * self.ci
        return {
            'choice': choice.tolist(),
            'plans': [self.plans[i]['plan'] for i in choice],
            'energy_kwh_day': float(kwh.sum()),
            'co2_kg_day': float(co2_g.sum() / 1e3),
            'co2_g_hour': co2_g.tolist(),
            'accuracy': float((self.n * self.A[choice]).sum() / self.n.sum()),
            'min_hour_accuracy': float(self.A[choice].min()),
            'max_utilization': float((self.rate * self.L[choice]).max()),
            'feasible': bool(self.feasible[np.arange(24), choice].all()),
        }

    # ------------------------------------------------------------------
    def gpu_only(self, budget=None):
        return self.evaluate([self._gpu_idx] * 24)

    def static(self, budget):
        ok = (self.A >= self.ref - budget - 1e-12) & self.feasible.all(axis=0)
        if not ok.any():
            return None
        p = int(np.argmin(np.where(ok, self.E, np.inf)))
        return self.evaluate([p] * 24)

    def hourly_energy(self, budget):
        choice = []
        for h in range(24):
            ok = (self.A >= self.ref - budget - 1e-12) & self.feasible[h]
            if not ok.any():
                return None
            choice.append(int(np.argmin(np.where(ok, self.E, np.inf))))
        return self.evaluate(choice)

    def _milp(self, budget, weights):
        """min Σ_h Σ_p n_h E_p w_h x_hp  (kesin çözüm, HiGHS)."""
        H, P = 24, len(self.plans)
        cost = (self.n[:, None] * self.E[None, :] * weights[:, None]).ravel()
        cost = cost / cost.max()

        # Her saat tam bir plan
        A_eq = np.zeros((H, H * P))
        for h in range(H):
            A_eq[h, h * P:(h + 1) * P] = 1
        # Günlük doğruluk: Σ n_h (ref − acc_p) x ≤ B Σ n_h   (n normalize edilir).
        # Çözücünün mutlak toleransı (~1e-6) bütçeye göre büyük kalmasın diye
        # kısıt 1e4 ile ölçeklenir; sonuç ayrıca doğrulanır.
        SCALE = 1e4
        w = self.n / self.n.sum()
        loss = (w[:, None] * (self.ref - self.A)[None, :]).ravel()
        ub = self.feasible.ravel().astype(float)

        margin = 0.0
        for _ in range(3):
            cons = [LinearConstraint(A_eq, 1, 1),
                    LinearConstraint(SCALE * loss[None, :], -np.inf, SCALE * (budget - margin))]
            res = milp(cost, constraints=cons, integrality=np.ones(H * P), bounds=Bounds(0, ub))
            if not res.success:
                return None
            r = self.evaluate(res.x.reshape(H, P).argmax(axis=1))
            if r['feasible'] and r['accuracy'] >= self.ref - budget - 1e-12:
                return r
            margin += 1e-6
        return None

    def daily_energy(self, budget):
        return self._milp(budget, np.ones(24))

    def carbon_aware(self, budget):
        return self._milp(budget, self.ci)

    def run_all(self, budget):
        return {s: getattr(self, s)(budget) for s in STRATEGIES}
