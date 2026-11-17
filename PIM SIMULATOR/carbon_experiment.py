"""
Karbon farkında 5G baz istasyonu senaryosu.

    python carbon_experiment.py     # → carbon_results.json, carbon_scenario.png

Senaryo (bir baz istasyonu):
    - İş yükü: digits CNN (doğruluğu ölçülen model), kullanıcı/slot başına çalışan
      küçük bir model yerine geçer. Tepe yük PEAK_RATE çıkarım/s
      (örn. 60 aktif kullanıcı × 2000 slot/s).
    - Gün içi yük: carbon_model.traffic_profile()
    - Şebeke: carbon_model.grid_carbon_intensity() (yaz / kış, stilize)
    - Yıl = 182,5 yaz günü + 182,5 kış günü

Sayılan enerji yalnızca çıkarım (dinamik) enerjisidir; her iki kurulumda da
bulunan cihazların boşta gücü hesaba katılmaz.
"""
import json
import os

import numpy as np

from baseline_models import GPUBaseline
from carbon_model import (EmbodiedCarbon, grid_carbon_intensity, payback_days,
                          traffic_profile)
from carbon_scheduler import STRATEGIES, CarbonAwareScheduler
from pim_simulator import PIMArray
from precision_planner import PrecisionPlanner
from quantization import get_trained_model

_HERE = os.path.dirname(os.path.abspath(__file__))

PEAK_RATE = 120_000        # çıkarım/s, tepe saat
RHO_MAX = 0.8              # en yüksek kullanım oranı
BUDGETS = (0.0, 0.0025, 0.005, 0.01)
MAIN_BUDGET = 0.005        # 0,5 puan
SOLAR_SHARES = (0.15, 0.30, 0.45, 0.60, 0.75)
DAYS = {'summer': 182.5, 'winter': 182.5}

LABELS = {
    'gpu_only': 'GPU-only',
    'static': 'Statik plan',
    'hourly_energy': 'Saatlik enerji',
    'daily_energy': 'Günlük bütçe\n(trafik farkında)',
    'carbon_aware': 'Karbon farkında',
}


def build_planner():
    model, (_, X_te, _, y_te) = get_trained_model()
    return PrecisionPlanner(model, X_te, y_te, PIMArray(), GPUBaseline())


def scheduler(P, plans, ci, peak_rate=PEAK_RATE):
    return CarbonAwareScheduler(plans, P.reference_accuracy, peak_rate * traffic_profile(), ci, RHO_MAX)


def _strip(r):
    if r is None:
        return None
    return {k: r[k] for k in ('energy_kwh_day', 'co2_kg_day', 'accuracy', 'max_utilization', 'plans')}


def run():
    P = build_planner()
    plans = list(P.all_plans())
    out = {'params': {'peak_rate_per_s': PEAK_RATE, 'rho_max': RHO_MAX,
                      'reference_accuracy': P.reference_accuracy}}

    # 1) Mevsim × bütçe × strateji
    print("=" * 78)
    print(f"1) Günlük CO2 (g/gün, tek baz istasyonu, tepe {PEAK_RATE:,} çıkarım/s)")
    print("=" * 78)
    seasons = {}
    for season in DAYS:
        ci = grid_carbon_intensity(season)
        S = scheduler(P, plans, ci)
        seasons[season] = {'ci': ci.tolist(), 'budgets': {}}
        print(f"\n  {season}: karbon yoğunluğu ort. {ci.mean():.0f}, aralık {ci.min():.0f}–{ci.max():.0f} gCO2/kWh")
        print(f"  {'bütçe':<8}" + ''.join(f"{s:>16}" for s in STRATEGIES))
        for b in BUDGETS:
            res = S.run_all(b)
            seasons[season]['budgets'][str(b)] = {k: _strip(v) for k, v in res.items()}
            cells = ''.join(f"{'—' if v is None else f'{v['co2_kg_day'] * 1e3:.2f}':>16}" for v in res.values())
            print(f"  {b * 100:.2f} pt" + cells)
    out['seasons'] = seasons

    # 2) Yıllık etki ve geri ödeme (ana bütçe)
    print("\n" + "=" * 78)
    print(f"2) Yıllık etki (bütçe {MAIN_BUDGET * 100:.1f} puan)")
    print("=" * 78)
    annual = {}
    for s in STRATEGIES:
        annual[s] = sum(DAYS[se] * seasons[se]['budgets'][str(MAIN_BUDGET)][s]['co2_kg_day'] for se in DAYS)
    base = annual['gpu_only']
    for s in STRATEGIES:
        print(f"  {s:<14}: {annual[s]:6.2f} kg CO2/yıl  (GPU-only'ye göre %{(1 - annual[s] / base) * 100:5.1f} az)")
    saving_day = (base - annual['carbon_aware']) / sum(DAYS.values())

    pay = []
    for cap in (4, 8, 16):
        for ov in (1.0, 2.0, 4.0):
            emb = EmbodiedCarbon(capacity_gb=cap, pim_overhead=ov).kgco2
            pay.append({'capacity_gb': cap, 'overhead': ov, 'embodied_kg': emb,
                        'payback_days': payback_days(emb, saving_day)})
    # Yalnızca PIM eklentisi (DRAM zaten gerekli): ek yük = overhead − 1
    pay_extra = []
    for cap in (4, 8, 16):
        for ov in (1.5, 2.0, 4.0):
            emb = EmbodiedCarbon(capacity_gb=cap, pim_overhead=ov - 1.0).kgco2
            pay_extra.append({'capacity_gb': cap, 'overhead': ov, 'embodied_kg': emb,
                              'payback_days': payback_days(emb, saving_day)})
    pd = [p['payback_days'] for p in pay]
    pe = [p['payback_days'] for p in pay_extra]
    print(f"\n  Günlük tasarruf: {saving_day * 1e3:.1f} g CO2")
    print(f"  Geri ödeme (tüm PIM modülü)       : {min(pd):.0f}–{max(pd):.0f} gün")
    print(f"  Geri ödeme (yalnızca PIM eklentisi): {min(pe):.0f}–{max(pe):.0f} gün")
    out['annual_kg'] = annual
    out['payback_full_module'] = pay
    out['payback_pim_addon'] = pay_extra
    out['daily_saving_kg'] = saving_day

    # 3) Güneş payı arttıkça karbon farkındalığının katkısı
    print("\n" + "=" * 78)
    print("3) Şebekede güneş payı arttıkça karbon farkındalığının ek katkısı (yaz)")
    print("=" * 78)
    solar = []
    for share in SOLAR_SHARES:
        ci = grid_carbon_intensity('summer', solar_peak_share=share)
        S = scheduler(P, plans, ci)
        row = {'solar_peak_share': share, 'ci_mean': float(ci.mean())}
        for b in (0.0025, 0.005, 0.01):
            d, c = S.daily_energy(b), S.carbon_aware(b)
            row[str(b)] = max(0.0, 1 - c['co2_kg_day'] / d['co2_kg_day'])  # sayısal gürültüyü kırp
        solar.append(row)
        print(f"  öğle güneş payı %{share * 100:3.0f} (ort. {row['ci_mean']:.0f} g/kWh): "
              + ', '.join(f"bütçe {b * 100:.2f} pt → %{row[str(b)] * 100:4.1f}" for b in (0.0025, 0.005, 0.01)))
    out['solar_sensitivity'] = solar

    # 4) Tepe yük duyarlılığı
    print("\n" + "=" * 78)
    print("4) Tepe yük duyarlılığı (yaz, bütçe 0,5 puan)")
    print("=" * 78)
    load = []
    ci = grid_carbon_intensity('summer')
    for rate in (30_000, 60_000, 120_000, 180_000, 240_000):
        S = scheduler(P, plans, ci, rate)
        r = S.run_all(MAIN_BUDGET)
        row = {'peak_rate': rate}
        for k, v in r.items():
            row[k] = None if v is None else v['co2_kg_day']
        load.append(row)
        g = r['gpu_only']
        txt = 'GPU-only kapasiteyi aşıyor' if not g['feasible'] else \
            f"karbon farkında GPU-only'ye göre %{(1 - r['carbon_aware']['co2_kg_day'] / g['co2_kg_day']) * 100:.1f} az"
        print(f"  {rate:>7,} çıkarım/s: {txt}")
    out['load_sensitivity'] = load

    path = os.path.join(_HERE, 'carbon_results.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Sonuçlar: {path}")
    plot(P, plans, out, os.path.join(_HERE, 'carbon_scenario.png'))
    return out


def plot(P, plans, out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator

    def plain_log(ax):
        ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ''))

    fig = plt.figure(figsize=(16, 9.5))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.42)
    hours = np.arange(24)

    # (a) Gün içi profil
    ax = fig.add_subplot(gs[0, :2])
    ci = grid_carbon_intensity('summer')
    S = scheduler(P, plans, ci)
    ca, de = S.carbon_aware(MAIN_BUDGET), S.daily_energy(MAIN_BUDGET)
    ax.bar(hours - 0.2, S.E[ca['choice']] * 1e3, 0.4, color='#2ecc71', label='Karbon farkında: µJ/çıkarım')
    ax.bar(hours + 0.2, S.E[de['choice']] * 1e3, 0.4, color='#95a5a6', label='Günlük bütçe: µJ/çıkarım')
    ax.set_ylabel('Enerji / çıkarım (µJ)')
    ax.set_ylim(0, S.E[ca['choice']].max() * 1e3 * 1.45)
    ax.set_xlabel('Saat')
    ax.set_xticks(hours)
    ax2 = ax.twinx()
    ax2.plot(hours, ci, 'o-', color='#e74c3c', label='Şebeke (gCO2/kWh)')
    ax2.plot(hours, traffic_profile() * ci.max(), 's--', color='#3498db', label='Trafik (ölçekli)')
    ax2.set_ylabel('gCO2/kWh')
    ax2.set_ylim(0, ci.max() * 1.45)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper center', fontsize=8, ncol=4, framealpha=0.95)
    ax.set_title('Yaz günü: saat başına seçilen plan (bütçe 0,5 puan)', fontweight='bold')

    # (b) Strateji karşılaştırması
    ax = fig.add_subplot(gs[0, 2])
    x = np.arange(len(STRATEGIES))
    for i, (season, color) in enumerate((('summer', '#f39c12'), ('winter', '#3498db'))):
        vals = [out['seasons'][season]['budgets'][str(MAIN_BUDGET)][s]['co2_kg_day'] * 1e3 for s in STRATEGIES]
        ax.bar(x + (i - 0.5) * 0.38, vals, 0.38, color=color, label={'summer': 'Yaz', 'winter': 'Kış'}[season],
               edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[s] for s in STRATEGIES], rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('g CO2 / gün')
    ax.set_yscale('log')
    plain_log(ax)
    ax.legend()
    ax.set_title('Günlük CO2 (bütçe 0,5 puan, log)', fontweight='bold')

    # (c) Güneş payı duyarlılığı
    ax = fig.add_subplot(gs[1, 0])
    sh = [r['solar_peak_share'] * 100 for r in out['solar_sensitivity']]
    for b, m in ((0.0025, 'o'), (0.005, 's'), (0.01, '^')):
        ax.plot(sh, [r[str(b)] * 100 for r in out['solar_sensitivity']], m + '-', label=f'bütçe {b * 100:.2f} pt')
    ax.axvline(30, color='gray', ls=':')
    ax.text(31, ax.get_ylim()[1] * 0.9, 'bugün\n(stilize)', fontsize=8, color='gray')
    ax.set_xlabel('Öğle saatinde güneş payı (%)')
    ax.set_ylabel('Ek CO2 azalması (%)')
    ax.set_title('Karbon farkındalığının ek katkısı', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (d) Bütçe → CO2
    ax = fig.add_subplot(gs[1, 1])
    bs = [b * 100 for b in BUDGETS]
    for s in ('static', 'hourly_energy', 'daily_energy', 'carbon_aware'):
        vals = [out['seasons']['summer']['budgets'][str(b)][s] for b in BUDGETS]
        ax.plot(bs, [np.nan if v is None else v['co2_kg_day'] * 1e3 for v in vals], 'o-',
                label=LABELS[s].replace('\n', ' '))
    ax.axhline(out['seasons']['summer']['budgets']['0.0']['gpu_only']['co2_kg_day'] * 1e3,
               color='#e74c3c', ls='--', label='GPU-only')
    ax.set_xlabel('Doğruluk bütçesi (puan)')
    ax.set_ylabel('g CO2 / gün (yaz)')
    ax.set_yscale('log')
    plain_log(ax)
    ax.set_title('Doğruluk bütçesi → karbon', fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (e) Geri ödeme
    ax = fig.add_subplot(gs[1, 2])
    pay = out['payback_full_module']
    for cap, m in ((4, 'o'), (8, 's'), (16, '^')):
        rows = [p for p in pay if p['capacity_gb'] == cap]
        ax.plot([p['overhead'] for p in rows], [p['payback_days'] for p in rows], m + '-', label=f'{cap} GB')
    ax.axhline(365, color='gray', ls=':')
    ax.text(1.02, 365, ' 1 yıl', va='bottom', fontsize=8, color='gray')
    ax.set_xlabel('PIM üretim ek yükü (DRAM\'e göre ×)')
    ax.set_ylabel('Gömülü karbon geri ödeme (gün)')
    ax.set_title('Tüm PIM modülü: geri ödeme süresi', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle('Karbon farkında hibrit PIM: 5G baz istasyonu senaryosu (stilize profiller)',
                 fontsize=14, fontweight='bold')
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Grafik: {path}")


if __name__ == "__main__":
    run()
