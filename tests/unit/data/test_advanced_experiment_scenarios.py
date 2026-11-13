"""M7 scenario generators: determinism, ground-truth invariants and the planted defects."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from decision_intelligence.data import advanced_experiment_scenarios as ad

SMALL = ad.CupedScenarioConfig(n_units=3_000, seed=5)


def test_cuped_population_is_deterministic() -> None:
    a, b = ad.generate_cuped_population(SMALL), ad.generate_cuped_population(SMALL)
    pd.testing.assert_frame_equal(a.units, b.units)
    pd.testing.assert_frame_equal(a.potential, b.potential)
    assert a.truth == b.truth


def test_cuped_truth_invariants() -> None:
    pop = ad.generate_cuped_population(SMALL)
    pot = pop.potential
    assert pop.truth.ate_revenue == pytest.approx(SMALL.revenue_effect, abs=1e-12)
    assert pop.truth.ate_revenue == pytest.approx(float((pot["revenue1"] - pot["revenue0"]).mean()))
    assert 0.4 < pop.truth.pre_post_correlation < 0.8
    assert pop.truth.ate_conversion == pytest.approx(SMALL.conversion_effect, abs=1e-3)
    assert np.all(pot["converted1"] >= pot["converted0"])  # monotone coupling
    assert pop.units["arrival_day"].between(1, SMALL.n_days).all()


def test_new_users_have_no_pre_period_history() -> None:
    pop = ad.generate_cuped_population(SMALL)
    new = pop.units["is_new_user"]
    assert pop.units.loc[new, "pre_revenue"].isna().all()
    assert pop.units.loc[~new, "pre_revenue"].notna().all()
    assert abs(new.mean() - SMALL.new_user_share) < 0.03


def test_realisation_and_planted_covariate_defects() -> None:
    pop = ad.generate_cuped_population(SMALL)
    n = len(pop.units)
    all_t = ad.realise_cuped_experiment(pop, np.ones(n, dtype=int))
    all_c = ad.realise_cuped_experiment(pop, np.zeros(n, dtype=int))
    # Outcome and post-treatment covariate respond to treatment; the true pre-period does not.
    assert (all_t["revenue"] - all_c["revenue"]).mean() == pytest.approx(SMALL.revenue_effect,
                                                                         abs=0.01)
    assert (all_t["in_experiment_sessions"] >= all_c["in_experiment_sessions"]).all()
    pd.testing.assert_series_equal(all_t["pre_revenue"], all_c["pre_revenue"])
    leak = (all_t["pre_revenue_leaky"] - all_c["pre_revenue_leaky"]).dropna()
    assert leak.mean() > 0.1  # the leaky window absorbs part of the effect
    with pytest.raises(ValueError):
        ad.realise_cuped_experiment(pop, np.ones(n - 1, dtype=int))


def test_cuped_config_validation() -> None:
    with pytest.raises(ValueError):
        ad.CupedScenarioConfig(n_units=5)
    with pytest.raises(ValueError):
        ad.CupedScenarioConfig(new_user_share=1.0)
    with pytest.raises(ValueError):
        ad.CupedScenarioConfig(baseline_conversion=0.0)


def test_switchback_truth_equals_all_on_minus_all_off() -> None:
    config = ad.SwitchbackConfig(seed=3)
    market = ad.generate_switchback_market(config)
    on = ad.realise_switchback(market, np.ones(config.n_periods, dtype=int))
    off = ad.realise_switchback(market, np.zeros(config.n_periods, dtype=int))
    lag = len(config.carryover_effects)
    difference = (on["orders"] - off["orders"]).to_numpy()[lag:]
    assert difference == pytest.approx(np.full(difference.size, market.truth.global_effect),
                                       abs=2e-3)
    assert market.truth.global_effect == config.direct_effect + sum(config.carryover_effects)


def test_switchback_carryover_follows_a_switch() -> None:
    config = ad.SwitchbackConfig(seed=4, noise_sd=1e-9, daily_amplitude=0.0, weekend_lift=0.0)
    market = ad.generate_switchback_market(config)
    w = np.zeros(config.n_periods, dtype=int)
    w[0] = 1  # treated first period, then control
    frame = ad.realise_switchback(market, w)
    after = frame.loc[frame["period"] == 1, "orders"].to_numpy() - config.base_level
    # Hour 1 after the switch: both lags still point into the treated period.
    assert after[0] == pytest.approx(sum(config.carryover_effects), abs=1e-6)
    assert after[1] == pytest.approx(config.carryover_effects[1], abs=1e-6)
    assert after[2] == pytest.approx(0.0, abs=1e-6)


def test_switchback_validation_and_determinism() -> None:
    with pytest.raises(ValueError):
        ad.SwitchbackConfig(period_hours=5)
    with pytest.raises(ValueError):
        ad.SwitchbackConfig(carryover_effects=(1.0, 1.0, 1.0, 1.0))
    a = ad.generate_switchback_market(ad.SwitchbackConfig(seed=9))
    b = ad.generate_switchback_market(replace(ad.SwitchbackConfig(), seed=9))
    np.testing.assert_array_equal(a.baseline, b.baseline)
    with pytest.raises(ValueError):
        ad.realise_switchback(a, np.ones(3, dtype=int))
