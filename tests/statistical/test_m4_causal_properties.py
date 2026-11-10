"""Repeated-sampling behaviour of M4 estimators against known causal truth.

Success cases: valid adjustment is unbiased with near-nominal coverage.
Failure cases: each violated assumption produces the documented failure.
Fixed seeds make every assertion deterministic; tolerances are expressed
in Monte Carlo standard errors so a correct estimator cannot fail by chance.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from decision_intelligence.causal_inference import diagnostics as dg
from decision_intelligence.causal_inference import estimators as es
from decision_intelligence.causal_inference import scenarios as sc
from decision_intelligence.causal_inference.campaign import campaign_monte_carlo
from decision_intelligence.common import seeded_generator
from decision_intelligence.data import SyntheticDataConfig


def _data(frame: pd.DataFrame, columns: list[str]) -> es.CausalData:
    return es.CausalData.from_arrays(frame["treatment"], frame["outcome"], frame[columns])


def _mc_check(values: list[float], truth: float, z: float = 4.0) -> bool:
    array = np.asarray(values)
    return abs(array.mean() - truth) <= z * array.std(ddof=1) / math.sqrt(array.size)


def test_valid_adjustment_recovers_ate_and_att_with_nominal_coverage() -> None:
    rng = seeded_generator(401)
    naive, ate, att, cover_ate, cover_att = [], [], [], 0, 0
    reps = 200
    for _ in range(reps):
        scenario = sc.confounded(1000, rng)
        data = _data(scenario.frame, ["x"])
        naive.append(es.naive_difference(data).estimate - scenario.true_ate)
        a = es.augmented_ipw(data)
        b = es.augmented_ipw(data, estimand="ATT")
        ate.append(a.estimate - scenario.true_ate)
        att.append(b.estimate - scenario.true_att)
        cover_ate += a.covers(scenario.true_ate)
        cover_att += b.covers(scenario.true_att)
    assert _mc_check(ate, 0.0) and _mc_check(att, 0.0)
    assert np.mean(naive) > 1.5  # strongly biased upwards
    assert cover_ate / reps >= 0.9 and cover_att / reps >= 0.88


def test_adjusting_for_a_pre_treatment_collider_creates_bias() -> None:
    rng = seeded_generator(402)
    unadjusted, adjusted = [], []
    for _ in range(150):
        scenario = sc.m_bias(1000, rng)
        unadjusted.append(es.naive_difference(_data(scenario.frame, [])).estimate)
        adjusted.append(es.augmented_ipw(_data(scenario.frame, ["collider"])).estimate)
    assert _mc_check(unadjusted, 1.0)
    assert np.mean(adjusted) < 0.5  # adjusting for C opens T <- U1 -> C <- U2 -> Y


def test_adjusting_for_a_mediator_targets_the_direct_effect_not_the_total() -> None:
    rng = seeded_generator(403)
    total, direct = [], []
    for _ in range(150):
        scenario = sc.mediator(1000, rng)
        total.append(es.naive_difference(_data(scenario.frame, [])).estimate)
        direct.append(es.augmented_ipw(_data(scenario.frame, ["mediator"])).estimate)
    assert _mc_check(total, 2.2)
    assert _mc_check(direct, 1.0)


def test_positivity_failure_destabilises_weighting_and_trimming_changes_the_estimand() -> None:
    rng = seeded_generator(404)
    full, trimmed, ess_shares = [], [], []
    for _ in range(80):
        scenario = sc.positivity(2000, rng, strength=6.0)
        data = _data(scenario.frame, ["x"])
        e = es.estimate_propensity(data)
        weights = dg.balancing_weights(e, data.treatment)
        report = dg.weight_diagnostics(weights, data.treatment)
        ess_shares.append(report.ess_treated / report.n_treated)
        full.append(
            es.inverse_probability_weighting(data).estimate - scenario.truths["population_ate"]
        )
        keep = np.flatnonzero(dg.trim_to_overlap(e, bound=0.1))
        # Truth for the population actually kept (trimmed on *estimated* propensities).
        kept_truth = float(scenario.potential_outcomes.individual_effects[keep].mean())
        trimmed.append(es.augmented_ipw(data.subset(keep)).estimate - kept_truth)
    assert np.median(ess_shares) < 0.2  # effective sample collapses
    assert np.sqrt(np.mean(np.square(full))) > 0.3  # IPW error large
    assert abs(np.mean(trimmed)) < 0.05  # trimmed estimand is recovered


def test_double_robustness() -> None:
    """Consistent if either nuisance model is correct; badly wrong if both are wrong."""
    rng = seeded_generator(405)
    results: dict[tuple[str, str], list[float]] = {}
    for _ in range(100):
        frame = sc.misspecification(1000, rng).frame
        for ps_cols, out_cols in (("z", "z"), ("z", "x"), ("x", "z"), ("x", "x")):
            p_vars = [c for c in frame if c.startswith(ps_cols) and c[1:].isdigit()]
            o_vars = [c for c in frame if c.startswith(out_cols) and c[1:].isdigit()]
            e = es.estimate_propensity(_data(frame, p_vars))
            estimate = es.augmented_ipw(_data(frame, o_vars), propensity=e).estimate
            results.setdefault((ps_cols, out_cols), []).append(estimate)
    for combo in (("z", "z"), ("z", "x"), ("x", "z")):
        assert _mc_check(results[combo], 10.0), combo
    both_wrong = np.asarray(results[("x", "x")])
    assert np.sqrt(np.mean((both_wrong - 10.0) ** 2)) > 5.0


def test_interference_breaks_the_policy_interpretation() -> None:
    rng = seeded_generator(406)
    estimates = [
        es.naive_difference(_data(sc.interference(500, 4, rng).frame, [])).estimate
        for _ in range(100)
    ]
    assert _mc_check(estimates, 1.0)  # direct effect
    assert abs(np.mean(estimates) - 3.0) > 1.5  # not the treat-everyone policy effect


@pytest.fixture(scope="module")
def campaign_replicates() -> dict[str, object]:
    summaries = campaign_monte_carlo(SyntheticDataConfig(), n_replications=60, first_seed=50_000)
    return {s.estimator: s for s in summaries}


def test_m2_campaign_valid_strategies_recover_the_known_effect(
    campaign_replicates: dict[str, object],
) -> None:
    for name in (
        "IPW (Hájek), regime A",
        "AIPW, regime A",
        "AIPW ATT, regime A",
        "IPW, regime C (logged propensity)",
        "AIPW, propensity correct / outcome model wrong",
        "AIPW, propensity wrong / outcome model correct",
        "negative control (log income), AIPW regime A",
    ):
        summary = campaign_replicates[name]
        assert abs(summary.bias) <= 4 * summary.bias_mc_se, name  # type: ignore[attr-defined]
        assert summary.coverage >= 0.85, name  # type: ignore[attr-defined]


def test_m2_campaign_invalid_strategies_are_biased(
    campaign_replicates: dict[str, object],
) -> None:
    expectations = {
        "naive": 1,
        "AIPW, regime B (proxies)": 1,
        "INVALID: AIPW regime A + post-treatment billing count": -1,
        "INVALID: AIPW regime A among billed customers": -1,
    }
    for name, sign in expectations.items():
        summary = campaign_replicates[name]
        assert sign * summary.bias > 5 * summary.bias_mc_se, name  # type: ignore[attr-defined]


def test_att_ipw_standard_error_is_calibrated_with_estimated_propensity() -> None:
    """Calibration check of the propensity-estimation-corrected ATT influence function
    under strong confounding and effect heterogeneity: the SE tracks the empirical SD
    and intervals cover the sample ATT at about 95%. (The statsmodels comparison in
    tests/unit/causal/test_estimators.py is the regression test for the correction.)"""
    rng = seeded_generator(407)
    errors, ses, hits = [], [], 0
    reps = 200
    for _ in range(reps):
        n = 1500
        x1, x2 = rng.normal(size=n), rng.normal(size=n)
        t = (rng.random(n) < 1 / (1 + np.exp(-(1.5 * x1 - 0.6 * x2)))).astype(float)
        effect = 2 - 2.0 * x1
        y = 1 + x1 + 0.5 * x2 + t * effect + rng.normal(size=n)
        data = es.CausalData.from_arrays(t, y, pd.DataFrame({"x1": x1, "x2": x2}))
        result = es.inverse_probability_weighting(data, estimand="ATT")
        att = float(effect[t == 1].mean())
        errors.append(result.estimate - att)
        ses.append(result.standard_error)
        hits += result.covers(att)
    ratio = float(np.mean(ses) / np.std(errors, ddof=1))
    assert 0.85 < ratio < 1.2
    assert hits / reps >= 0.91
