"""Overlap, balance and weight diagnostics; E-values and bias grids; scenarios."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from decision_intelligence.causal_inference import diagnostics as dg
from decision_intelligence.causal_inference import estimators as es
from decision_intelligence.causal_inference import scenarios as sc
from decision_intelligence.causal_inference import sensitivity as sens
from decision_intelligence.common import seeded_generator


def _binary_confounded(n: int = 4000) -> es.CausalData:
    rng = np.random.default_rng(21)
    x = (rng.random(n) < 0.5).astype(float)
    t = (rng.random(n) < np.where(x == 1, 0.8, 0.2)).astype(float)
    return es.CausalData.from_arrays(t, x + t + rng.normal(size=n), pd.DataFrame({"x": x}))


def test_smd_formula_and_exact_balance_after_saturated_weighting() -> None:
    data = _binary_confounded()
    e = es.estimate_propensity(data)
    table = dg.balance_table(data, dg.balancing_weights(e, data.treatment))
    x, t = data.covariates["x"].to_numpy(), data.treatment == 1
    pooled = math.sqrt((x[t].var(ddof=1) + x[~t].var(ddof=1)) / 2)
    assert table.loc["x", "smd_raw"] == pytest.approx((x[t].mean() - x[~t].mean()) / pooled)
    assert abs(table.loc["x", "smd_raw"]) > 1.0
    assert table.loc["x", "smd_weighted"] == pytest.approx(0.0, abs=1e-10)


def test_att_weights_balance_controls_to_treated() -> None:
    data = _binary_confounded()
    e = es.estimate_propensity(data)
    table = dg.balance_table(data, dg.balancing_weights(e, data.treatment, "ATT"))
    assert table.loc["x", "smd_weighted"] == pytest.approx(0.0, abs=1e-10)


def test_kish_ess_and_weight_warnings() -> None:
    equal = dg.weight_diagnostics(np.ones(100), np.repeat([0.0, 1.0], 50))
    assert equal.ess_treated == pytest.approx(50) and not equal.warnings
    skewed = np.ones(100)
    skewed[0] = 100.0
    report = dg.weight_diagnostics(skewed, np.repeat([1.0, 0.0], 50))
    assert report.ess_treated < 25
    assert any("effective sample size" in w for w in report.warnings)
    assert report.n_weights_above_threshold == 1


def test_overlap_summary_and_trimming() -> None:
    e = np.array([0.01, 0.2, 0.5, 0.8, 0.99, 0.3])
    t = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    summary = dg.overlap_summary(e, t)
    assert summary.share_below == pytest.approx(1 / 6)
    assert summary.share_above == pytest.approx(1 / 6)
    assert not summary.adequate
    # Treated span [0.5, 0.99], controls [0.01, 0.3]: no common support at all.
    assert summary.common_support == (0.5, 0.3)
    assert summary.share_outside_common_support == 1.0
    np.testing.assert_array_equal(dg.trim_to_overlap(e, bound=0.1), [0, 1, 1, 1, 0, 1])
    with pytest.raises(ValueError):
        dg.trim_to_overlap(e, bound=0.6)
    with pytest.raises(ValueError, match="strictly inside"):
        dg.balancing_weights([0.0, 0.5], [0.0, 1.0])


# --- sensitivity --------------------------------------------------------------


def test_e_value_known_values() -> None:
    assert sens.e_value_from_rr(2.0) == pytest.approx(2 + math.sqrt(2))
    assert sens.e_value_from_rr(0.5) == pytest.approx(2 + math.sqrt(2))  # protective symmetric
    assert sens.e_value_from_rr(1.0) == 1.0
    result = sens.e_value(1.5, 1.2, 1.9)
    assert result.e_value_ci == pytest.approx(sens.e_value_from_rr(1.2))
    assert sens.e_value(1.3, 0.9, 1.8).e_value_ci == 1.0
    with pytest.raises(ValueError):
        sens.e_value(2.0, 2.5, 3.0)


def test_bias_grid_arithmetic_and_tipping_point() -> None:
    grid = sens.risk_difference_bias_grid(
        0.10, 0.05, 0.15, outcome_effects=(0.1, 0.5), prevalence_differences=(0.1, 0.2)
    )
    row = grid[(grid.outcome_effect_gamma == 0.5) & (grid.prevalence_difference_delta == 0.2)]
    assert float(row.bias.iloc[0]) == pytest.approx(0.1)
    assert float(row.corrected_estimate.iloc[0]) == pytest.approx(0.0)
    assert not bool(row.ci_excludes_zero.iloc[0])
    weak = grid[(grid.outcome_effect_gamma == 0.1) & (grid.prevalence_difference_delta == 0.1)]
    assert bool(weak.ci_excludes_zero.iloc[0])
    with pytest.raises(ValueError):
        sens.risk_difference_bias_grid(0.1, 0.0, 0.2, outcome_effects=(2.0,),
                                       prevalence_differences=(0.1,))


# --- scenarios ----------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda rng: sc.confounded(200, rng),
        lambda rng: sc.mediator(200, rng),
        lambda rng: sc.m_bias(200, rng),
        lambda rng: sc.positivity(200, rng),
        lambda rng: sc.misspecification(200, rng),
        lambda rng: sc.interference(50, 4, rng),
    ],
)
def test_scenarios_are_reproducible_and_consistent(build: object) -> None:
    first = build(seeded_generator(4))  # type: ignore[operator]
    second = build(seeded_generator(4))  # type: ignore[operator]
    pd.testing.assert_frame_equal(first.frame, second.frame)
    np.testing.assert_array_equal(first.frame["outcome"], first.potential_outcomes.observed)


def test_scenario_truths() -> None:
    rng = seeded_generator(5)
    assert sc.mediator(1000, rng).true_ate == pytest.approx(2.2)
    assert sc.m_bias(1000, rng).true_ate == pytest.approx(1.0)
    interference = sc.interference(100, 3, rng)
    assert interference.true_ate == pytest.approx(1.0)
    assert interference.truths["treat_all_vs_none_effect"] == 3.0
    positivity = sc.positivity(20000, rng)
    assert positivity.truths["trimmed_ate_0.1"] < positivity.truths["population_ate"]
    confounded = sc.confounded(20000, rng)
    assert confounded.true_att > confounded.true_ate


def test_bounding_factor_and_e_value_identity() -> None:
    """The E-value E is defined by bounding_factor(E, E) = RR (Ding & VanderWeele)."""
    for rr in (1.1, 1.29, 2.0, 4.5):
        e = sens.e_value_from_rr(rr)
        assert sens.bounding_factor(e, e) == pytest.approx(rr)
    assert sens.bounding_factor(1.0, 50.0) == pytest.approx(1.0)
    # Unequal associations: one below the E-value can still explain away the RR.
    rr = 1.29
    assert sens.bounding_factor(20.0, 1.4) > rr > sens.bounding_factor(1.4, 1.4)
    with pytest.raises(ValueError):
        sens.bounding_factor(0.9, 2.0)
