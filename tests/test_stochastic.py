"""
Belirsizlik altında plan seçimi (DP vs. Q-Learning) testleri.
"""
import numpy as np
import pytest

from baseline_models import GPUBaseline
from pim_simulator import PIMArray
from precision_planner import PrecisionPlanner
from quantization import get_trained_model
from stochastic_scheduler import (H, DayEnvironment, MPCPolicy, MyopicPolicy, OpenLoopPolicy,
                                  OraclePolicy, QLearningPolicy, SDPPolicy, evaluate, pareto_plans)


@pytest.fixture(scope="module")
def setup():
    model, (_, X_te, _, y_te) = get_trained_model()
    P = PrecisionPlanner(model, X_te, y_te, PIMArray(), GPUBaseline())
    all_plans = list(P.all_plans())
    return P, all_plans, pareto_plans(all_plans)


@pytest.fixture(scope="module")
def env(setup):
    P, _, plans = setup
    return DayEnvironment(plans, P.reference_accuracy)


@pytest.fixture(scope="module")
def days(env):
    rng = np.random.default_rng(11)
    return [env.sample_day(rng) for _ in range(12)]


def test_pareto_set_is_non_dominated_and_keeps_extremes(setup):
    _, all_plans, front = setup
    assert 0 < len(front) < len(all_plans)
    for p in front:
        assert not any(o['energy'] < p['energy'] and o['latency'] <= p['latency']
                       and o['accuracy'] >= p['accuracy'] for o in all_plans)
    assert min(p['energy'] for p in front) == min(p['energy'] for p in all_plans)
    assert min(p['latency'] for p in front) == min(p['latency'] for p in all_plans)


def test_every_hour_has_an_allowed_plan(env, days):
    for d in days:
        assert (d['rate'] <= env.max_rate + 1e-9).all()
        for h in range(H):
            # Bütçe tamamen bitmiş olsa bile kayıpsız bir plan kalmalı
            assert env.allowed(d['rate'][h], env.loss_budget).any()


@pytest.mark.parametrize("cls", [OraclePolicy, MyopicPolicy, OpenLoopPolicy, MPCPolicy, SDPPolicy])
def test_policies_respect_constraints(env, days, cls):
    res = evaluate(env, cls(env), days[:6])   # env.run izin verilmeyen aksiyonda hata fırlatır
    assert (res['budget_used'] <= 1 + 1e-9).all()


def test_oracle_is_a_lower_bound(env, days):
    orc = evaluate(env, OraclePolicy(env), days)['co2_g']
    for cls in (MyopicPolicy, OpenLoopPolicy, SDPPolicy):
        other = evaluate(env, cls(env), days)['co2_g']
        assert (other >= orc * (1 - 1e-6)).all(), cls.__name__


def test_without_noise_open_loop_equals_oracle(setup):
    P, _, plans = setup
    e = DayEnvironment(plans, P.reference_accuracy, sigma=0.0)
    ds = [e.sample_day(np.random.default_rng(s)) for s in range(3)]
    orc = evaluate(e, OraclePolicy(e), ds)['co2_g']
    ol = evaluate(e, OpenLoopPolicy(e), ds)['co2_g']
    assert ol == pytest.approx(orc, rel=1e-9)


def test_sdp_close_to_oracle_and_beats_myopic(env, days):
    orc = evaluate(env, OraclePolicy(env), days)['co2_g'].mean()
    sdp = evaluate(env, SDPPolicy(env), days)['co2_g'].mean()
    myo = evaluate(env, MyopicPolicy(env), days)['co2_g'].mean()
    assert sdp <= orc * 1.05
    assert sdp < myo


def test_sdp_value_function_is_monotone_in_budget(env):
    sdp = SDPPolicy(env, K=50, M=7)
    # Daha çok kalan bütçe asla daha pahalı olamaz
    assert (np.diff(sdp.V[:H], axis=-1) <= 1e-9).all()


def test_qlearning_improves_with_training(env, days):
    q = QLearningPolicy(env, seed=1)
    before = evaluate(env, q, days)['co2_g'].mean()
    q.train(5000)
    after = evaluate(env, q, days)['co2_g'].mean()
    assert after < before
    assert q.episodes == 5000


def test_qlearning_state_bins_in_range(env):
    q = QLearningPolicy(env)
    for used in (0.0, 0.5 * env.loss_budget, env.loss_budget):
        for r in (env.forecast_rate[0] * 0.2, env.forecast_rate[0], env.forecast_rate[0] * 5):
            h, b, rb, c = q._state(0, r, used, 2)
            assert 0 <= b < q.nb and 0 <= rb < q.nr
