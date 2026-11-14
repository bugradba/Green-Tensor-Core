"""
Raporlanan tüm sayıların tek kaynağı.

Grafikler (visualize_scheduler.py), demo (demo_simulator.py) ve README'deki
sonuçlar bu modülden üretilir; hiçbir yerde elle girilmiş sonuç kalmaz.

Çalıştırma:  python results.py
"""
import random

import numpy as np

from pim_simulator import PIMArray, PIMCluster, conv_stats, layer_workload
from baseline_models import GPUBaseline
from cnn_models import CNNModels
from hybrid_scheduler import simulate_layer, HybridSystem
from Q_Learning.adaptive_scheduler import AdaptiveQLearningScheduler


ALEXNET_CONV1 = {'input': (3, 227, 227), 'kernel': (96, 3, 11, 11), 'stride': 4, 'padding': 0}


def conv1_comparison(pim, gpu):
    """AlexNet conv1: PIM 8-bit / 4-bit ve GPU."""
    c = ALEXNET_CONV1
    p8 = pim.convolution_layer(c['input'], c['kernel'], 8, c['stride'], c['padding'])
    p4 = pim.convolution_layer(c['input'], c['kernel'], 4, c['stride'], c['padding'])
    cs = conv_stats(c['input'], c['kernel'], c['stride'], c['padding'])
    bits = (cs['input_elems'] + cs['weight_elems'] + cs['output_elems']) * 8
    g = gpu.model_inference(cs['total_macs'], bits)
    return {'pim8': p8, 'pim4': p4, 'gpu': g}


def mac_precision_stats(n_samples=2000, seed=0):
    """Tek MAC'te 8-bit vs 4-bit enerji/gecikme ve 4-bit'in gerçek hatası."""
    cl = PIMCluster()
    _, e8, l8 = cl.mac_8bit(128, 128, precision=8)
    _, e4, l4 = cl.mac_8bit(128, 128, precision=4)
    rng = np.random.default_rng(seed)
    a = rng.integers(16, 256, n_samples)
    b = rng.integers(16, 256, n_samples)
    approx = np.array([cl.mac_8bit(int(x), int(y), precision=4)[0] for x, y in zip(a, b)])
    exact = a * b
    rel_err = np.abs(exact - approx) / exact
    return {'energy_8bit_pj': e8, 'energy_4bit_pj': e4,
            'latency_8bit_ns': l8, 'latency_4bit_ns': l4,
            'mean_rel_error_4bit': float(rel_err.mean())}


def _plan(layers, choose, pim, gpu):
    rows = []
    for layer in layers:
        device = choose(layer)
        e, l = simulate_layer(layer, device, pim, gpu)
        rows.append({'layer': layer['name'], 'type': layer['type'], 'device': device,
                     'energy': e, 'latency': l})
    return {'layers': rows,
            'energy': sum(r['energy'] for r in rows),
            'latency': sum(r['latency'] for r in rows)}


def train_qlearning_on_model(layers, pim, gpu, episodes=40, seed=42):
    """
    Q-Learning'i verilen modelin gerçek katmanlarıyla eğitir.
    Returns: (scheduler, episode_rewards, q_states_per_episode)
    """
    random.seed(seed)
    np.random.seed(seed)
    sched = AdaptiveQLearningScheduler(learning_rate=0.15, epsilon=0.4)
    workloads = []
    for layer in layers:
        macs, data = layer_workload(layer)
        if macs is not None:
            workloads.append((macs, layer['type'], None, data))
    rewards, q_sizes = [], []
    for _ in range(episodes):
        ep = workloads * 3
        random.shuffle(ep)
        rewards.append(sched.train_episode(pim, gpu, ep))
        q_sizes.append(len(sched.q_table))
    return sched, rewards, q_sizes


def alexnet_strategies(pim, gpu, q_sched=None):
    """AlexNet'i farklı stratejilerle çalıştırır."""
    layers = CNNModels.get_model('alexnet')
    out = {
        'gpu_only': _plan(layers, lambda l: 'GPU', pim, gpu),
        'pim_only': _plan(layers, lambda l: 'PIM', pim, gpu),
        'rule_based': _plan(layers, lambda l: HybridSystem._layer_rule(l['type'])[0], pim, gpu),
    }
    if q_sched is not None:
        def q_choose(layer):
            macs, _ = layer_workload(layer)
            return q_sched.predict(macs if macs is not None else 1000000, layer['type'])[0]
        out['qlearning'] = _plan(layers, q_choose, pim, gpu)

    # Kaba kuvvet referansı: her katmanda en düşük enerjili cihaz (gecikme kısıtı yok)
    def best(layer):
        return min(('PIM', 'GPU', 'HYBRID'),
                   key=lambda d: simulate_layer(layer, d, pim, gpu)[0])
    out['oracle_min_energy'] = _plan(layers, best, pim, gpu)
    return out


def compute_results(episodes=40, seed=42):
    pim, gpu = PIMArray(), GPUBaseline()
    layers = CNNModels.get_model('alexnet')
    q_sched, rewards, q_sizes = train_qlearning_on_model(layers, pim, gpu, episodes, seed)
    return {
        'conv1': conv1_comparison(pim, gpu),
        'mac': mac_precision_stats(),
        'alexnet': alexnet_strategies(pim, gpu, q_sched),
        'qlearning': {'episode_rewards': rewards, 'q_states': q_sizes, 'scheduler': q_sched},
    }


def print_summary(r):
    c = r['conv1']
    print("AlexNet conv1 (stride 4)")
    print(f"  MAC sayısı        : {c['pim8']['total_macs']:,}")
    for k, name in (('pim8', 'PIM 8-bit'), ('pim4', 'PIM 4-bit')):
        print(f"  {name:<18}: {c[k]['energy_total_mj']:.3f} mJ, {c[k]['latency_ms']:.3f} ms")
    g = c['gpu']
    print(f"  {'GPU (Jetson Nano)':<18}: {g['total_energy_mj']:.3f} mJ, {g['total_latency_ms']:.3f} ms "
          f"({g['avg_power_mw']/1000:.1f} W, {g['bound']}-bound)")
    m = r['mac']
    print(f"\nTek MAC: 8-bit {m['energy_8bit_pj']:.2f} pJ / {m['latency_8bit_ns']} ns, "
          f"4-bit {m['energy_4bit_pj']:.2f} pJ / {m['latency_4bit_ns']} ns, "
          f"4-bit ortalama göreli hata %{m['mean_rel_error_4bit']*100:.1f}")
    a = r['alexnet']
    base = a['gpu_only']['energy']
    print("\nAlexNet (tam model)")
    for k, v in a.items():
        print(f"  {k:<18}: {v['energy']:8.2f} mJ  {v['latency']:8.2f} ms  "
              f"(GPU-only'ye göre enerji {(1 - v['energy']/base)*100:+.1f}%)")


if __name__ == "__main__":
    print_summary(compute_results())
