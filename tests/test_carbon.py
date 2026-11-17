"""
Karbon modeli ve karbon farkında zamanlayıcı testleri.
"""
import numpy as np
import pytest

from baseline_models import GPUBaseline
from carbon_model import (EmbodiedCarbon, grid_carbon_intensity, load_profile_csv,
                          payback_days, traffic_profile)
from carbon_scheduler import STRATEGIES, CarbonAwareScheduler
from pim_simulator import PIMArray
from precision_planner import PrecisionPlanner
from quantization import get_trained_model


@pytest.fixture(scope="module")
def planner():
    model, (_, X_te, _, y_te) = get_trained_model()
    return PrecisionPlanner(model, X_te, y_te, PIMArray(), GPUBaseline())


@pytest.fixture(scope="module")
def plans(planner):
    return list(planner.all_plans())


def make(planner, plans, ci=None, peak=120_000):
    ci = grid_carbon_intensity('summer') if ci is None else ci
    return CarbonAwareScheduler(plans, planner.reference_accuracy, peak * traffic_profile(), ci)


# --- Profiller ---------------------------------------------------------------

def test_carbon_profile_shape_and_range():
    for season in ('summer', 'winter'):
        ci = grid_carbon_intensity(season)
        assert ci.shape == (24,)
        assert ci.max() == pytest.approx(460.0)
        assert ci[12] < ci[0]                       # öğle gece temizdir
    # Yıllık ortalama yayımlanmış aralığa yakın
    annual = (grid_carbon_intensity('summer').mean() + grid_carbon_intensity('winter').mean()) / 2
    assert 400 <= annual <= 440


def test_more_solar_means_cleaner_grid():
    a = grid_carbon_intensity('summer', solar_peak_share=0.3).mean()
    b = grid_carbon_intensity('summer', solar_peak_share=0.6).mean()
    assert b < a


def test_traffic_profile():
    t = traffic_profile()
    assert t.max() == 1.0 and t.min() > 0 and t.argmin() == 4


def test_load_profile_csv(tmp_path):
    p = tmp_path / "ci.csv"
    p.write_text("hour,gco2_per_kwh\n" + "".join(f"{h},{400 + h}\n" for h in range(24)))
    assert load_profile_csv(str(p), 'gco2_per_kwh')[23] == 423
    p.write_text("hour,gco2_per_kwh\n0,1\n")
    with pytest.raises(ValueError):
        load_profile_csv(str(p), 'gco2_per_kwh')


def test_embodied_and_payback():
    assert EmbodiedCarbon(capacity_gb=8, dram_gco2_per_gb=48, pim_overhead=2).kgco2 == pytest.approx(0.768)
    assert payback_days(1.0, 0.1) == pytest.approx(10)
    assert payback_days(1.0, 0) == float('inf')


# --- Zamanlayıcı ---------------------------------------------------------------

@pytest.mark.parametrize("budget", [0.0, 0.0025, 0.005, 0.01])
def test_strategy_ordering_and_constraints(planner, plans, budget):
    S = make(planner, plans)
    r = S.run_all(budget)
    for name, v in r.items():
        assert v is not None and v['feasible'], name
        assert v['max_utilization'] <= S.rho_max + 1e-9
    for name in ('static', 'hourly_energy'):
        assert r[name]['min_hour_accuracy'] >= planner.reference_accuracy - budget - 1e-12
    for name in ('daily_energy', 'carbon_aware'):
        assert r[name]['accuracy'] >= planner.reference_accuracy - budget - 1e-9

    co2 = {k: v['co2_kg_day'] for k, v in r.items()}
    eps = 1e-12
    # Daha esnek strateji asla daha kötü olamaz
    assert co2['hourly_energy'] <= co2['static'] + eps
    assert co2['daily_energy'] <= co2['hourly_energy'] * (1 + 1e-9) or \
        r['daily_energy']['energy_kwh_day'] <= r['hourly_energy']['energy_kwh_day'] * (1 + 1e-9)
    assert co2['carbon_aware'] <= min(co2.values()) * (1 + 1e-9)
    assert co2['carbon_aware'] < co2['gpu_only']


def test_constant_grid_makes_carbon_awareness_irrelevant(planner, plans):
    S = make(planner, plans, ci=np.full(24, 420.0))
    assert S.carbon_aware(0.005)['co2_kg_day'] == pytest.approx(S.daily_energy(0.005)['co2_kg_day'])


def test_carbon_aware_is_locally_optimal(planner, plans):
    """Tek bir saatin planını değiştirmek, kısıtları bozmadan karbonu azaltamamalı."""
    S = make(planner, plans)
    b = 0.005
    best = S.carbon_aware(b)
    choice = np.array(best['choice'])
    for h in range(24):
        for p in range(len(plans)):
            if p == choice[h] or not S.feasible[h, p]:
                continue
            c = choice.copy()
            c[h] = p
            r = S.evaluate(c)
            if r['accuracy'] >= planner.reference_accuracy - b - 1e-9:
                assert r['co2_kg_day'] >= best['co2_kg_day'] * (1 - 1e-9)


def test_capacity_constraint_binds_at_high_load(planner, plans):
    S = make(planner, plans, peak=240_000)
    assert not S.gpu_only()['feasible'] or S.gpu_only()['max_utilization'] <= S.rho_max
    r = S.carbon_aware(0.005)
    assert r is not None and r['max_utilization'] <= S.rho_max + 1e-9


def test_impossible_load_returns_none(planner, plans):
    S = make(planner, plans, peak=10_000_000)
    assert S.static(0.01) is None
    assert S.hourly_energy(0.01) is None
    assert S.carbon_aware(0.01) is None


def test_energy_accounting(planner, plans):
    S = make(planner, plans)
    r = S.gpu_only()
    gpu_e_mj = plans[S._gpu_idx]['energy']
    expected_kwh = (S.n.sum() * gpu_e_mj / 1e3) / 3.6e6
    assert r['energy_kwh_day'] == pytest.approx(expected_kwh)
    assert set(STRATEGIES) == set(S.run_all(0.005))
