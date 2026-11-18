"""
Belirsizlik altında: dinamik programlama vs. Q-Learning.

    python stochastic_experiment.py           # tam deney (~11 dk)
    python stochastic_experiment.py --quick   # kısa sürüm (~1 dk)

Çıktılar: stochastic_results.json, stochastic_comparison.png
"""
import argparse
import json
import os
import time

import numpy as np

from baseline_models import GPUBaseline
from pim_simulator import PIMArray
from precision_planner import PrecisionPlanner
from quantization import get_trained_model
from stochastic_scheduler import (DayEnvironment, MPCPolicy, MyopicPolicy, OpenLoopPolicy,
                                  OraclePolicy, QLearningPolicy, SDPPolicy, evaluate, pareto_plans)

_HERE = os.path.dirname(os.path.abspath(__file__))

LABELS = {
    'oracle': 'Oracle (geleceği bilir)',
    'sdp': 'Stokastik DP',
    'mpc': 'MPC (saatlik MILP)',
    'open_loop': 'Açık döngü MILP',
    'myopic': 'Miyop (saat başı bütçe)',
    'qlearning': 'Q-Learning',
}


def setup(sigma=0.25):
    model, (_, X_te, _, y_te) = get_trained_model()
    P = PrecisionPlanner(model, X_te, y_te, PIMArray(), GPUBaseline())
    plans = pareto_plans(list(P.all_plans()))
    return DayEnvironment(plans, P.reference_accuracy, sigma=sigma), len(plans)


def test_days(env, n, seed=2026):
    rng = np.random.default_rng(seed)
    return [env.sample_day(rng) for _ in range(n)]


def summarize(res, oracle):
    gap = res['co2_g'] / oracle - 1
    return {'co2_g_mean': float(res['co2_g'].mean()), 'gap_mean': float(gap.mean()),
            'gap_p90': float(np.percentile(gap, 90)), 'budget_used': float(res['budget_used'].mean()),
            'ms_per_decision': res['ms_per_decision']}


def main(quick=False):
    n_days = 60 if quick else 200
    q_checkpoints = (5_000, 20_000) if quick else (10_000, 30_000, 100_000, 300_000)
    sigmas = (0.1, 0.25) if quick else (0.0, 0.1, 0.25, 0.4)
    out = {'n_test_days': n_days}

    env, n_plans = setup()
    days = test_days(env, n_days)
    out['n_actions'] = n_plans

    print("=" * 78)
    print(f"1) Politikalar (σ = {env.sigma}, {n_days} test günü, {n_plans} Pareto planı)")
    print("=" * 78)
    sdp = SDPPolicy(env)
    policies = [OraclePolicy(env), sdp, MPCPolicy(env), OpenLoopPolicy(env), MyopicPolicy(env)]
    raw = {p.name: evaluate(env, p, days) for p in policies}
    oracle = raw['oracle']['co2_g']
    summary = {k: summarize(v, oracle) for k, v in raw.items()}
    summary['sdp']['solve_seconds'] = sdp.solve_seconds

    q = QLearningPolicy(env)
    curve = []
    for n in q_checkpoints:
        q.train(n - q.episodes)
        r = summarize(evaluate(env, q, days), oracle)
        r.update(episodes=q.episodes, train_seconds=q.training_seconds)
        curve.append(r)
        print(f"  Q-Learning {q.episodes:>8,} bölüm ({q.training_seconds:5.0f} s eğitim): "
              f"oracle'dan %{r['gap_mean'] * 100:.1f} fazla")
    summary['qlearning'] = curve[-1]
    out['policies'] = summary
    out['qlearning_curve'] = curve

    print()
    for k in ('oracle', 'sdp', 'mpc', 'open_loop', 'myopic', 'qlearning'):
        s = summary[k]
        extra = f", çözüm {s['solve_seconds']:.2f} s" if 'solve_seconds' in s else \
            (f", eğitim {s['train_seconds']:.0f} s" if 'train_seconds' in s else '')
        print(f"  {LABELS[k]:<26} {s['co2_g_mean']:.3f} g/gün  oracle +%{s['gap_mean'] * 100:5.1f} "
              f"(p90 +%{s['gap_p90'] * 100:5.1f})  bütçe %{s['budget_used'] * 100:3.0f}  "
              f"{s['ms_per_decision']:.2f} ms/karar{extra}")

    # 2) Belirsizlik arttıkça uyarlanabilirliğin değeri
    print("\n" + "=" * 78)
    print("2) Trafik belirsizliği (σ) arttıkça oracle'a göre fark")
    print("=" * 78)
    sweep = []
    for sg in sigmas:
        e, _ = setup(sg)
        ds = test_days(e, n_days // 2, seed=7)
        orc = evaluate(e, OraclePolicy(e), ds)['co2_g']
        row = {'sigma': sg}
        for p in (SDPPolicy(e), MPCPolicy(e), OpenLoopPolicy(e), MyopicPolicy(e)):
            row[p.name] = summarize(evaluate(e, p, ds), orc)['gap_mean']
        sweep.append(row)
        print(f"  σ = {sg:.2f}: " + ', '.join(f"{k} +%{row[k] * 100:.1f}" for k in ('sdp', 'mpc', 'open_loop', 'myopic')))
    out['sigma_sweep'] = sweep

    path = os.path.join(_HERE, 'stochastic_results.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Sonuçlar: {path}")
    plot(out, os.path.join(_HERE, 'stochastic_comparison.png'))
    return out


def plot(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    pol = out['policies']
    order = ['sdp', 'mpc', 'open_loop', 'myopic', 'qlearning']
    colors = {'sdp': '#2ecc71', 'mpc': '#3498db', 'open_loop': '#9b59b6', 'myopic': '#f39c12', 'qlearning': '#e74c3c'}

    ax = axes[0]
    vals = [pol[k]['gap_mean'] * 100 for k in order]
    bars = ax.bar([LABELS[k].replace(' (', '\n(') for k in order], vals,
                  color=[colors[k] for k in order], edgecolor='black')
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f'+%{v:.1f}', ha='center', va='bottom', fontweight='bold')
    ax.set_ylabel("Oracle'a göre fazla CO2 (%)")
    ax.set_title(f"Belirsizlik altında politikalar (σ = 0,25, {out['n_test_days']} gün)", fontweight='bold')
    ax.tick_params(axis='x', labelsize=8)
    ax.set_ylim(0, max(vals) * 1.18)

    ax = axes[1]
    c = out['qlearning_curve']
    ax.plot([r['episodes'] for r in c], [r['gap_mean'] * 100 for r in c], 'o-', color=colors['qlearning'],
            label='Q-Learning')
    for k in ('sdp', 'myopic'):
        ax.axhline(pol[k]['gap_mean'] * 100, color=colors[k], ls='--', label=LABELS[k])
    ax.set_xscale('log')
    ax.set_xlabel('Eğitim bölümü (simüle gün, log)')
    ax.set_ylabel("Oracle'a göre fazla CO2 (%)")
    ax.set_title('Q-Learning öğrenme eğrisi', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    sw = out['sigma_sweep']
    for k in ('sdp', 'mpc', 'open_loop', 'myopic'):
        ax.plot([r['sigma'] for r in sw], [r[k] * 100 for r in sw], 'o-', color=colors[k], label=LABELS[k])
    ax.set_xlabel('Saatlik trafik belirsizliği σ')
    ax.set_ylabel("Oracle'a göre fazla CO2 (%)")
    ax.set_title('Belirsizlik arttıkça', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle('Belirsiz trafik ve hava koşullarında saatlik plan seçimi', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Grafik: {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    main(ap.parse_args().quick)
