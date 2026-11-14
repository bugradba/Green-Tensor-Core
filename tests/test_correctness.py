"""
Simülatörün doğruluk testleri.  Çalıştırma (repo kökünden):  pytest
"""
import json
import math
import random

import numpy as np
import pytest

from pim_simulator import PIM_Core, PIMCluster, PIMArray, conv_output_hw, conv_stats, layer_workload
from baseline_models import GPUBaseline, CPUBaseline
from cnn_models import CNNModels
from hybrid_scheduler import simulate_layer, AdvancedHybridScheduler, HybridSystem
from Q_Learning.adaptive_scheduler import AdaptiveQLearningScheduler, normalize_layer_type


# --- Aritmetik -------------------------------------------------------------

def test_core_lut_multiplication_is_exact():
    core = PIM_Core()
    for a in range(16):
        for b in range(16):
            assert core.multiply_4bit(a, b)[0] == a * b


def test_cluster_8bit_mac_is_exact():
    cl = PIMCluster()
    rng = random.Random(0)
    for _ in range(500):
        a, b, acc = rng.randrange(256), rng.randrange(256), rng.randrange(1000)
        assert cl.mac_8bit(a, b, acc, precision=8)[0] == (a * b + acc) & 0xFFFF


def test_cluster_4bit_is_cheaper_and_uses_high_nibbles():
    cl = PIMCluster()
    r8, e8, l8 = cl.mac_8bit(255, 200, precision=8)
    r4, e4, l4 = cl.mac_8bit(255, 200, precision=4)
    assert r4 == ((255 >> 4) * (200 >> 4)) << 8
    assert e4 < e8 and l4 < l8


def test_invalid_precision_raises():
    with pytest.raises(ValueError):
        PIMCluster().mac_8bit(1, 1, precision=2)


# --- Conv geometrisi --------------------------------------------------------

@pytest.mark.parametrize("H,K,S,P,expected", [
    (227, 11, 4, 0, 55),   # AlexNet conv1
    (27, 5, 1, 2, 27),     # AlexNet conv2
    (224, 3, 1, 1, 224),   # VGG 'same'
    (224, 7, 2, 3, 112),   # ResNet conv1
])
def test_conv_output_size(H, K, S, P, expected):
    assert conv_output_hw(H, H, K, K, S, P) == (expected, expected)


def test_conv_output_uses_kernel_width():
    # Kh != Kw: genişlik Kw ile hesaplanmalı (eski hata: Kh kullanılıyordu)
    assert conv_output_hw(10, 10, 3, 1) == (8, 10)


def test_alexnet_conv1_mac_count():
    st = conv_stats((3, 227, 227), (96, 3, 11, 11), stride=4)
    assert st['total_macs'] == 96 * 55 * 55 * 3 * 11 * 11 == 105_415_200


def test_channel_mismatch_raises():
    with pytest.raises(ValueError):
        conv_stats((3, 32, 32), (8, 4, 3, 3))


def test_all_model_conv_layers_are_consistent():
    """cnn_models içindeki her conv katmanı geçerli bir çıkış boyutu vermeli."""
    for name in ('alexnet', 'vgg16', 'resnet50', 'mobilenet'):
        for layer in CNNModels.get_model(name):
            macs, data = layer_workload(layer)
            if layer['type'] == 'Conv2D' and 'input' in layer:
                assert macs > 0 and data > 0


# --- PIM maliyet modeli -----------------------------------------------------

def test_pim_energy_scales_linearly_with_macs():
    pim = PIMArray()
    a = pim.mac_workload(1_000_000, 0)
    b = pim.mac_workload(2_000_000, 0)
    assert math.isclose(b['energy_compute_mj'], 2 * a['energy_compute_mj'])


def test_pim_communication_is_charged_per_row_not_per_mac():
    pim = PIMArray()
    st = pim.convolution_layer((3, 227, 227), (96, 3, 11, 11), stride=4)
    # ~480 KB veri -> ~59 satır; iletişim enerjisi hesaplamanın çok altında olmalı
    assert st['rows_moved'] == math.ceil(
        (3 * 227 * 227 + 96 * 3 * 11 * 11 + 96 * 55 * 55) / PIMArray.ROW_SIZE_BYTES)
    assert st['energy_communication_mj'] < 0.01 * st['energy_compute_mj']


def test_pim_4bit_cheaper_than_8bit():
    pim = PIMArray()
    s8 = pim.convolution_layer((3, 227, 227), (96, 3, 11, 11), 8, stride=4)
    s4 = pim.convolution_layer((3, 227, 227), (96, 3, 11, 11), 4, stride=4)
    assert s4['energy_total_mj'] < s8['energy_total_mj']
    assert s4['latency_ms'] < s8['latency_ms']


# --- GPU/CPU baz modeli -----------------------------------------------------

@pytest.mark.parametrize("device", [GPUBaseline(), CPUBaseline()])
@pytest.mark.parametrize("macs,bits", [
    (105_415_200, 3_840_000),        # conv1, compute-bound
    (37_748_736, 37_748_736 * 8),    # fc1 batch=1, memory-bound
    (1_000_000, 0),                  # varsayılan veri yeniden kullanımı
])
def test_baseline_power_never_exceeds_tdp(device, macs, bits):
    st = device.model_inference(macs, bits)
    assert st['avg_power_mw'] <= device.tdp_watts * 1000 * 1.0001


def test_gpu_fc_batch1_is_memory_bound():
    st = GPUBaseline().model_inference(9216 * 4096, (9216 * 4096) * 8)
    assert st['bound'] == 'memory'


# --- Scheduler --------------------------------------------------------------

def test_layer_type_aliases():
    assert normalize_layer_type('Conv2D') == 'Conv'
    assert normalize_layer_type('Linear') == 'FC'
    assert normalize_layer_type('MaxPool') == 'Pool'


def test_state_is_shared_between_training_and_model_names():
    q = AdaptiveQLearningScheduler()
    assert q._discretize_state(5e6, 'Conv2D') == q._discretize_state(5e6, 'Conv')


def test_hybrid_is_not_simulated_as_gpu():
    pim, gpu = PIMArray(), GPUBaseline()
    layer = {'name': 'c', 'type': 'Conv2D', 'input': (64, 56, 56), 'kernel': (64, 64, 3, 3), 'padding': 1}
    h = simulate_layer(layer, 'HYBRID', pim, gpu)
    g = simulate_layer(layer, 'GPU', pim, gpu)
    p = simulate_layer(layer, 'PIM', pim, gpu)
    assert h != g
    # Enerji iki cihaz arasında kalmalı
    assert min(p[0], g[0]) <= h[0] <= max(p[0], g[0])


def test_train_episode_requires_real_simulators():
    q = AdaptiveQLearningScheduler()
    with pytest.raises(ValueError):
        q.train_episode(None, None, [(1000, 'Conv', None)])


def test_qlearning_learns_simulator_optimum():
    """Deadline yokken, enerji ağırlıklı ödülle küçük FC katmanında PIM ödülü GPU'dan yüksek olmalı."""
    np.random.seed(0)
    pim, gpu = PIMArray(), GPUBaseline()
    q = AdaptiveQLearningScheduler(epsilon=1.0)  # her aksiyonu dene
    work = [(4096 * 1000, 'Linear', None, 4096 * 1000 + 5096)] * 300
    q.train_episode(pim, gpu, work)
    state = q._discretize_state(4096 * 1000, 'FC')
    qv = q.q_table[state]
    assert qv[q.action_to_idx['PIM']] > qv[q.action_to_idx['GPU']]


def test_model_roundtrip_uses_safe_parsing(tmp_path):
    q = AdaptiveQLearningScheduler()
    q.q_table[('small', 'Conv', 'none')] = np.array([1.0, 2.0, 3.0])
    path = tmp_path / "m.json"
    q.save_model(str(path))

    data = json.loads(path.read_text())
    data['q_table']["__import__('os').system('echo pwned')"] = [0, 0, 0]
    path.write_text(json.dumps(data))
    with pytest.raises((ValueError, SyntaxError)):
        AdaptiveQLearningScheduler().load_model(str(path))


def test_advanced_scheduler_runs_alexnet_rule_based():
    sched = AdvancedHybridScheduler(PIMArray(), GPUBaseline(), use_qlearning=False)
    plan, e, l = sched.schedule_model(CNNModels.get_model('alexnet'))
    assert len(plan) == len(CNNModels.get_model('alexnet'))
    assert e > 0 and l > 0


def test_analyze_model_layers_totals_match_plan():
    hs = HybridSystem(PIMArray(), GPUBaseline())
    plan, e, l = hs.analyze_model_layers(CNNModels.get_model('alexnet'), 1.5)
    assert math.isclose(e, sum(p['energy'] for p in plan))
    assert math.isclose(l, sum(p['latency'] for p in plan))


def test_shipped_trained_model_loads():
    import os
    import hybrid_scheduler
    path = os.path.join(os.path.dirname(hybrid_scheduler.__file__), 'adaptive_scheduler_trained_v2.json')
    q = AdaptiveQLearningScheduler()
    q.load_model(path)
    assert len(q.q_table) > 0
    # Kanonik adlarla eğitilmiş olmalı
    assert all(state[1] not in ('Conv2D', 'Linear', 'MaxPool') for state in q.q_table)
