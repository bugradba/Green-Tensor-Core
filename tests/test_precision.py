"""
Katman bazında hassasiyet seçimi ve quantized çıkarım testleri.
"""
import itertools

import numpy as np
import pytest

from baseline_models import GPUBaseline
from pim_simulator import PIMArray, PIMCluster
from precision_planner import CHOICES, PrecisionPlanner, pareto_front
from quantization import (QUANT_LAYERS, DigitsCNN, get_trained_model, load_digits_data,
                          quantize_activation, quantize_weight)


@pytest.fixture(scope="module")
def trained():
    return get_trained_model()


@pytest.fixture(scope="module")
def planner(trained):
    model, (_, X_te, _, y_te) = trained
    return PrecisionPlanner(model, X_te, y_te, PIMArray(), GPUBaseline(), 'truncate')


# --- Aritmetik, simülatörle birebir aynı mı? ----------------------------------

def test_4bit_truncation_matches_pim_cluster():
    cl = PIMCluster()
    for a in range(0, 256, 7):
        for b in range(0, 256, 11):
            assert (a & 0xF0) * (b & 0xF0) == cl.mac_8bit(a, b, precision=4)[0]
            assert a * b == cl.mac_8bit(a, b, precision=8)[0]


@pytest.mark.parametrize("bits", [8, 4])
def test_quantized_dot_product_equals_pim_mac_sum(bits):
    """Quantized katmanın tamsayı birikimi = PIM cluster MAC'lerinin toplamı."""
    rng = np.random.default_rng(1)
    a = rng.random(32) * 3.0
    W = rng.normal(size=(4, 32))
    a_code, _ = quantize_activation(a, a.max(), bits, 'truncate')
    w_code, w_sign, _ = quantize_weight(W, bits, 'truncate')
    fast = (w_sign * w_code) @ a_code

    cl = PIMCluster()
    slow = [sum(int(w_sign[o, i]) * cl.mac_8bit(int(a_code[i]), int(w_code[o, i]), precision=bits)[0]
                for i in range(32)) for o in range(4)]
    assert list(fast) == slow


def test_code_ranges():
    a = np.linspace(0, 2, 101)
    c, _ = quantize_activation(a, 2, 4, 'truncate')
    assert np.all(c % 16 == 0) and c.max() <= 240
    c, _ = quantize_activation(a, 2, 4, 'round')
    assert c.min() == 0 and c.max() == 15
    c, _ = quantize_activation(a, 2, 8)
    assert c.max() == 255


# --- Model ---------------------------------------------------------------------

def test_backprop_matches_numerical_gradient():
    X, _, y, _ = load_digits_data()
    m = DigitsCNN(seed=3)
    rng = np.random.default_rng(0)
    for k in m.p:
        if k.endswith('_b'):
            m.p[k] += 0.05 * rng.standard_normal(m.p[k].shape)
    x, t = X[:6], y[:6]
    _, g = m._grads(x, t)
    for k in m.p:
        idx = tuple(rng.integers(0, s) for s in m.p[k].shape)
        old = m.p[k][idx]
        m.p[k][idx] = old + 1e-6
        lp = m._grads(x, t)[0]
        m.p[k][idx] = old - 1e-6
        lm = m._grads(x, t)[0]
        m.p[k][idx] = old
        num = (lp - lm) / 2e-6
        assert abs(num - g[k][idx]) <= 1e-4 * max(1.0, abs(num)), k


def test_trained_model_accuracy(trained):
    model, (_, X_te, _, y_te) = trained
    assert (model.predict_float(X_te) == y_te).mean() > 0.97


def test_int8_close_to_float(trained):
    model, (_, X_te, _, y_te) = trained
    f = (model.predict_float(X_te) == y_te).mean()
    q = model.evaluate_quantized(X_te, y_te, {k: 8 for k in QUANT_LAYERS})
    assert abs(f - q) <= 0.01


def test_evaluate_requires_calibration():
    with pytest.raises(RuntimeError):
        DigitsCNN().evaluate_quantized(np.zeros((1, 1, 8, 8)), np.zeros(1, int), {k: 8 for k in QUANT_LAYERS})


def test_layer_configs_match_weights(trained):
    model = trained[0]
    cfg = {l['name']: l for l in model.layer_configs()}
    for n in ('conv1', 'conv2'):
        assert cfg[n]['kernel'] == model.p[f'{n}_w'].shape
    for n in ('fc1', 'fc2'):
        assert (cfg[n]['out_features'], cfg[n]['in_features']) == model.p[f'{n}_w'].shape


# --- Planlayıcı ----------------------------------------------------------------

def test_pim4_cheaper_and_faster_than_pim8(planner):
    for n in planner.quant_layers:
        e8, l8 = planner.layer_cost(n, 'PIM8')
        e4, l4 = planner.layer_cost(n, 'PIM4')
        assert e4 < e8 and l4 < l8


@pytest.mark.parametrize("budget", [0.0, 0.005, 0.01, 0.02])
@pytest.mark.parametrize("deadline", [None, 0.006, 0.004])
def test_exhaustive_is_optimal_and_feasible(planner, budget, deadline):
    best = planner.exhaustive(budget, deadline)
    all_r = list(planner.all_plans())
    feas = [r for r in all_r if r['accuracy'] >= planner.reference_accuracy - budget - 1e-12
            and (deadline is None or r['latency'] <= deadline + 1e-12)]
    if not feas:
        assert best is None
        return
    assert best['energy'] == pytest.approx(min(r['energy'] for r in feas))


@pytest.mark.parametrize("budget", [0.0, 0.005, 0.01, 0.02])
@pytest.mark.parametrize("deadline", [None, 0.006, 0.004, 0.0025])
def test_greedy_feasible_and_never_better_than_optimum(planner, budget, deadline):
    gr = planner.greedy(budget, deadline)
    ex = planner.exhaustive(budget, deadline)
    if gr is None:
        return
    assert ex is not None
    assert gr['accuracy'] >= planner.reference_accuracy - budget - 1e-12
    if deadline is not None:
        assert gr['latency'] <= deadline + 1e-12
    assert gr['energy'] >= ex['energy'] - 1e-15


def test_larger_budget_never_costs_more_energy(planner):
    energies = [planner.exhaustive(b)['energy'] for b in (0, 0.0025, 0.005, 0.01, 0.02, 0.05)]
    assert all(a >= b - 1e-15 for a, b in zip(energies, energies[1:]))


def test_accuracy_depends_only_on_bit_pattern(planner):
    """GPU ve PIM8 aynı INT8 aritmetiğini kullandığı için aynı doğruluğu verir."""
    a = planner.accuracy({n: 'GPU' for n in planner.quant_layers})
    b = planner.accuracy({n: 'PIM8' for n in planner.quant_layers})
    assert a == b == planner.reference_accuracy


def test_pareto_front_is_non_dominated(planner):
    res = list(planner.all_plans())
    front = pareto_front(res)
    for f in front:
        assert not any(r['energy'] < f['energy'] - 1e-15 and r['accuracy'] > f['accuracy'] + 1e-12
                       for r in res)
    # Enerji arttıkça doğruluk artar
    assert all(x['energy'] <= y['energy'] and x['accuracy'] < y['accuracy']
               for x, y in zip(front, front[1:]))
    assert len(list(itertools.product(CHOICES, repeat=len(planner.quant_layers)))) == len(res)
