"""
Katman bazında hassasiyet seçimi deneyi.

    python precision_experiment.py            # tam deney (~1 dk), grafik + JSON
    python precision_experiment.py --quick    # sadece seed 0

Çıktılar:
    precision_pareto.png   — enerji / doğruluk / gecikme uzayı ve Pareto sınırı
    precision_results.json — README'deki tüm sayılar
"""
import argparse
import json
import os

import numpy as np

from baseline_models import GPUBaseline
from pim_simulator import PIMArray
from precision_planner import CHOICES, PrecisionPlanner, pareto_front, plan_str
from quantization import QUANT_LAYERS, DigitsCNN, get_trained_model, load_digits_data

_HERE = os.path.dirname(os.path.abspath(__file__))
UJ = 1000.0  # mJ -> µJ
US = 1000.0  # ms -> µs

BUDGETS = (0.0, 0.005, 0.01, 0.02)          # doğruluk bütçesi (oran)
DEADLINES_US = (None, 6.0, 4.0, 2.5)          # gecikme sınırı (µs)


def model_for_seed(seed):
    if seed == 0:
        return get_trained_model()
    X_tr, X_te, y_tr, y_te = load_digits_data(seed=seed)
    model = DigitsCNN(seed=seed).train(X_tr, y_tr, seed=seed).calibrate(X_tr)
    return model, (X_tr, X_te, y_tr, y_te)


def seed_summary(seed, pim, gpu):
    model, (_, X_te, _, y_te) = model_for_seed(seed)
    out = {'seed': seed, 'n_test': int(len(y_te)),
           'float': float((model.predict_float(X_te) == y_te).mean())}
    for scheme in ('truncate', 'round'):
        P = PrecisionPlanner(model, X_te, y_te, pim, gpu, scheme)
        all8 = P.evaluate({n: 'PIM8' for n in P.quant_layers})
        all4 = P.evaluate({n: 'PIM4' for n in P.quant_layers})
        s = {'int8': all8['accuracy'], 'all_4bit': all4['accuracy'],
             'all_pim8_energy_uj': all8['energy'] * UJ,
             'sensitivity': P.sensitivity(), 'budgets': {}}
        for b in BUDGETS:
            r = P.exhaustive(b)
            s['budgets'][str(b)] = {'plan': r['plan'], 'energy_uj': r['energy'] * UJ,
                                    'accuracy': r['accuracy'],
                                    'saving_vs_pim8': 1 - r['energy'] / all8['energy']}
        out[scheme] = s
    return out


def scenarios(P):
    rows = []
    for d in DEADLINES_US:
        for b in BUDGETS:
            dl = None if d is None else d / US
            ex, gr = P.exhaustive(b, dl), P.greedy(b, dl)
            rows.append({
                'deadline_us': d, 'budget': b,
                'exhaustive': None if ex is None else {
                    'plan': ex['plan'], 'energy_uj': ex['energy'] * UJ,
                    'latency_us': ex['latency'] * US, 'accuracy': ex['accuracy']},
                'greedy': None if gr is None else {
                    'plan': gr['plan'], 'energy_uj': gr['energy'] * UJ,
                    'latency_us': gr['latency'] * US, 'accuracy': gr['accuracy']},
            })
    return rows


def greedy_quality(planners):
    """Greedy, tam aramaya göre ne kadar iyi? (seed 0, geniş senaryo ızgarası)"""
    cases = optimal = missed = 0
    gaps = []
    for P in planners.values():
        for d_us in (None, 6.0, 5.0, 4.0, 3.0, 2.5, 2.0):
            for b in np.arange(0, 0.035, 0.0025):
                d = None if d_us is None else d_us / US
                ex, gr = P.exhaustive(b, d), P.greedy(b, d)
                if ex is None:
                    continue
                cases += 1
                if gr is None:
                    missed += 1
                    continue
                gap = gr['energy'] / ex['energy'] - 1
                gaps.append(gap)
                optimal += gap < 1e-9
    return {'cases': cases, 'optimal': int(optimal), 'not_found': missed,
            'mean_gap_when_found': float(np.mean(gaps)) if gaps else 0.0,
            'max_gap_when_found': float(np.max(gaps)) if gaps else 0.0}


def plot(planners, path):
    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(planners), figsize=(7 * len(planners), 5.5), sharey=True)
    axes = np.atleast_1d(axes)
    titles = {'truncate': '4-bit = üst nibble (pPIM / simülatör)',
              'round': '4-bit = yuvarlanmış kod (aynı donanım maliyeti)'}
    for ax, (scheme, P) in zip(axes, planners.items()):
        res = list(P.all_plans())
        e = np.array([r['energy'] * UJ for r in res])
        a = np.array([r['accuracy'] * 100 for r in res])
        l = np.array([r['latency'] * US for r in res])
        sc = ax.scatter(e, a, c=l, cmap='viridis', s=40, edgecolor='k', linewidth=0.3)
        front = pareto_front(res)
        ax.plot([r['energy'] * UJ for r in front], [r['accuracy'] * 100 for r in front],
                '-', color='#e74c3c', lw=2, label='Pareto sınırı')
        for name in CHOICES:
            r = P.evaluate({n: name for n in P.quant_layers})
            ax.annotate(f'Hepsi {name}', (r['energy'] * UJ, r['accuracy'] * 100),
                        textcoords='offset points', xytext=(-8, -14), ha='right' if name == 'GPU' else 'left',
                        fontsize=9, fontweight='bold')
        ax.axhline(P.reference_accuracy * 100 - 1, color='gray', ls=':', lw=1)
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f'{v:g}' if v in (2, 3, 5, 20) else ''))
        ax.text(0.98, P.reference_accuracy * 100 - 1, 'INT8 − 1 puan', va='bottom', ha='right',
                fontsize=8, color='gray', transform=ax.get_yaxis_transform())
        ax.set_xlabel('Enerji / çıkarım (µJ, log)')
        ax.set_title(titles[scheme], fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(loc='lower right')
    axes[0].set_ylabel('Test doğruluğu (%)')
    cb = fig.colorbar(sc, ax=axes.tolist(), pad=0.01)
    cb.set_label('Gecikme (µs)')
    fig.suptitle('Katman bazında cihaz + hassasiyet seçimi: 81 planın tamamı (digits CNN)',
                 fontweight='bold')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Grafik: {path}")


def _ms(vals):
    return f"{np.mean(vals) * 100:.2f} ± {np.std(vals) * 100:.2f}"


def main(seeds):
    pim, gpu = PIMArray(), GPUBaseline()

    print("=" * 78)
    print(f"1) Doğruluk ({len(seeds)} farklı veri bölmesi, ortalama ± std, %)")
    print("=" * 78)
    summaries = [seed_summary(s, pim, gpu) for s in seeds]
    print(f"  Float32               : {_ms([s['float'] for s in summaries])}")
    print(f"  INT8 (tüm katmanlar)  : {_ms([s['truncate']['int8'] for s in summaries])}")
    for sch in ('truncate', 'round'):
        print(f"  4-bit {sch:<9} (tümü) : {_ms([s[sch]['all_4bit'] for s in summaries])}")
    for sch in ('truncate', 'round'):
        print(f"\n  [{sch}] bütçe -> enerji tasarrufu (hepsi PIM8'e göre) / doğruluk kaybı")
        for b in BUDGETS:
            sv = [s[sch]['budgets'][str(b)]['saving_vs_pim8'] for s in summaries]
            drop = [s[sch]['int8'] - s[sch]['budgets'][str(b)]['accuracy'] for s in summaries]
            print(f"    ≤{b * 100:.1f} puan: tasarruf %{np.mean(sv) * 100:5.1f} ± {np.std(sv) * 100:4.1f}, "
                  f"kayıp {np.mean(drop) * 100:.2f} ± {np.std(drop) * 100:.2f} puan")

    model, (_, X_te, _, y_te) = model_for_seed(0)
    planners = {sch: PrecisionPlanner(model, X_te, y_te, pim, gpu, sch) for sch in ('truncate', 'round')}
    P = planners['truncate']

    print("\n" + "=" * 78)
    print("2) Katman hassasiyeti (seed 0, truncate): tek başına 4-bit -> doğruluk kaybı")
    print("=" * 78)
    for n, d in P.sensitivity().items():
        e8, e4 = P.layer_cost(n, 'PIM8')[0], P.layer_cost(n, 'PIM4')[0]
        print(f"  {n:<6} kayıp {d * 100:5.2f} puan | enerji {e8 * UJ:6.3f} -> {e4 * UJ:6.3f} µJ")

    print("\n" + "=" * 78)
    print("3) Senaryolar (seed 0, truncate): tam arama vs greedy")
    print("=" * 78)
    scen = scenarios(P)
    for row in scen:
        d = 'yok' if row['deadline_us'] is None else f"{row['deadline_us']} µs"
        ex, gr = row['exhaustive'], row['greedy']
        if ex is None:
            print(f"  sınır {d:<7} bütçe {row['budget'] * 100:.1f}: uygun plan yok")
            continue
        gtxt = 'bulunamadı' if gr is None else f"{gr['energy_uj']:.3f} µJ"
        print(f"  sınır {d:<7} bütçe {row['budget'] * 100:.1f}: {plan_str(ex['plan']):<44} "
              f"{ex['energy_uj']:.3f} µJ {ex['latency_us']:.2f} µs %{ex['accuracy'] * 100:.2f} | greedy {gtxt}")

    gq = greedy_quality(planners)
    print("\n" + "=" * 78)
    print("4) Greedy kalitesi (seed 0, iki şema, gecikme × bütçe ızgarası)")
    print("=" * 78)
    print(f"  {gq['cases']} senaryo: {gq['optimal']} optimum, {gq['not_found']} plan bulunamadı, "
          f"suboptimal {gq['cases'] - gq['optimal'] - gq['not_found']}")
    print(f"  Bulunduğunda fazla enerji: ortalama %{gq['mean_gap_when_found'] * 100:.1f}, "
          f"en kötü %{gq['max_gap_when_found'] * 100:.0f}")
    print("  Kaçırılan durumlar, birden fazla katmanın birlikte 4-bit yapılmasının tek tek")
    print("  etkilerinden daha az zarar verdiği (toplanabilir olmayan) durumlardır.")

    plot(planners, os.path.join(_HERE, 'precision_pareto.png'))

    out = {'seeds': summaries, 'scenarios_seed0_truncate': scen, 'greedy_quality': gq}
    path = os.path.join(_HERE, 'precision_results.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"✅ Sonuçlar: {path}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='sadece seed 0')
    args = ap.parse_args()
    main([0] if args.quick else [0, 1, 2, 3, 4])
