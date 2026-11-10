"""Estimators: exact standardisation identities, influence-function SEs, failure handling."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from decision_intelligence.causal_inference import estimators as es
from decision_intelligence.causal_inference.estimands import PotentialOutcomes
from decision_intelligence.common import seeded_generator


@pytest.fixture(scope="module")
def stratified() -> tuple[es.CausalData, float, float]:
    """One binary confounder: saturated models make every estimator equal the
    hand-computed standardised (stratified) estimate exactly."""
    rng = np.random.default_rng(11)
    n = 3000
    x = (rng.random(n) < 0.4).astype(float)
    t = (rng.random(n) < np.where(x == 1, 0.7, 0.3)).astype(float)
    y = (rng.random(n) < 0.2 + 0.3 * x + 0.1 * t).astype(float)
    frame = pd.DataFrame({"t": t, "y": y, "x": x})
    ate = att = 0.0
    for level in (0.0, 1.0):
        stratum = frame[frame.x == level]
        diff = stratum[stratum.t == 1].y.mean() - stratum[stratum.t == 0].y.mean()
        ate += len(stratum) / n * diff
        att += (stratum.t == 1).sum() / (frame.t == 1).sum() * diff
    return es.CausalData.from_arrays(t, y, frame[["x"]]), float(ate), float(att)


@pytest.mark.parametrize("estimand", ["ATE", "ATT"])
def test_saturated_estimators_equal_standardisation(
    stratified: tuple[es.CausalData, float, float], estimand: str
) -> None:
    data, ate, att = stratified
    target = ate if estimand == "ATE" else att
    aipw = es.augmented_ipw(data, estimand=estimand)  # type: ignore[arg-type]
    ipw = es.inverse_probability_weighting(data, estimand=estimand)  # type: ignore[arg-type]
    reg = es.regression_adjustment(
        data, estimand=estimand, rng=seeded_generator(1), n_bootstrap=100  # type: ignore[arg-type]
    )
    for estimate in (aipw, ipw, reg):
        assert estimate.estimate == pytest.approx(target, abs=1e-9)


def test_horvitz_thompson_differs_from_hajek_but_is_unbiased_formula(
    stratified: tuple[es.CausalData, float, float],
) -> None:
    data, _, _ = stratified
    e = es.estimate_propensity(data)
    t, y = data.treatment, data.outcome
    expected = float(np.mean(t * y / e - (1 - t) * y / (1 - e)))
    ht = es.inverse_probability_weighting(data, normalised=False)
    assert ht.estimate == pytest.approx(expected)


def test_no_covariates_reduces_to_difference_in_means() -> None:
    rng = np.random.default_rng(12)
    t = (rng.random(500) < 0.5).astype(float)
    y = rng.normal(size=500) + t
    data = es.CausalData.from_arrays(t, y)
    aipw = es.augmented_ipw(data)
    naive = es.naive_difference(data)
    assert aipw.estimate == pytest.approx(naive.estimate)
    unpooled = math.sqrt(y[t == 1].var(ddof=1) / t.sum() + y[t == 0].var(ddof=1) / (1 - t).sum())
    assert aipw.standard_error == pytest.approx(unpooled, rel=0.01)
    assert "not causal" in naive.estimand


def test_known_propensity_and_bootstrap_paths() -> None:
    rng = np.random.default_rng(13)
    x = rng.normal(size=2000)
    e = 1 / (1 + np.exp(-x))
    t = (rng.random(2000) < e).astype(float)
    y = x + 2 * t + rng.normal(size=2000)
    data = es.CausalData.from_arrays(t, y, pd.DataFrame({"x": x}))
    known = es.inverse_probability_weighting(data, propensity=e)
    assert known.adjustment_set == ("<known propensity>",)
    assert known.covers(2.0)
    boot = es.augmented_ipw(data, se_method="bootstrap", rng=seeded_generator(3), n_bootstrap=200)
    influence = es.augmented_ipw(data)
    assert boot.estimate == influence.estimate
    assert boot.standard_error == pytest.approx(influence.standard_error, rel=0.25)
    again = es.augmented_ipw(data, se_method="bootstrap", rng=seeded_generator(3), n_bootstrap=200)
    assert (boot.ci_lower, boot.ci_upper) == (again.ci_lower, again.ci_upper)
    with pytest.raises(ValueError, match="rng"):
        es.augmented_ipw(data, se_method="bootstrap")


def test_perfect_prediction_of_treatment_is_a_positivity_error() -> None:
    x = np.arange(20.0)
    t = (x >= 10).astype(float)
    data = es.CausalData.from_arrays(t, x + t, pd.DataFrame({"x": x}))
    with pytest.raises(es.PositivityError, match="positivity"):
        es.augmented_ipw(data)


def test_extreme_propensities_produce_warnings() -> None:
    rng = np.random.default_rng(14)
    x = rng.uniform(-3, 3, size=3000)
    t = (rng.random(3000) < 1 / (1 + np.exp(-4 * x))).astype(float)
    data = es.CausalData.from_arrays(t, x + t, pd.DataFrame({"x": x}))
    result = es.inverse_probability_weighting(data)
    assert any("positivity" in w for w in result.warnings)


@pytest.mark.parametrize(
    ("treatment", "message"),
    [([0, 1, 2, 1], "binary"), ([1, 1, 1, 0], "two treated and two untreated")],
)
def test_invalid_data(treatment: list[int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        es.CausalData.from_arrays(treatment, [1.0, 2.0, 3.0, 4.0])


def test_invalid_covariates_and_propensity() -> None:
    with pytest.raises(ValueError, match="NaN"):
        es.CausalData.from_arrays(
            [0, 1, 0, 1], [1.0, 2, 3, 4], pd.DataFrame({"x": [1.0, np.nan, 2, 3]})
        )
    data = es.CausalData.from_arrays([0, 1, 0, 1], [1.0, 2, 3, 4])
    with pytest.raises(es.PositivityError):
        es.inverse_probability_weighting(data, propensity=[0.5, 1.0, 0.5, 0.5])
    with pytest.raises(ValueError, match="estimand"):
        es.augmented_ipw(data, estimand="ATC")  # type: ignore[arg-type]


def test_aipw_risk_ratio_equals_standardised_ratio(
    stratified: tuple[es.CausalData, float, float],
) -> None:
    data, _, _ = stratified
    frame = pd.DataFrame({"t": data.treatment, "y": data.outcome, "x": data.covariates["x"]})
    weights = frame.x.value_counts(normalize=True)
    risk = {
        arm: sum(weights[x] * frame[(frame.x == x) & (frame.t == arm)].y.mean() for x in (0, 1))
        for arm in (0.0, 1.0)
    }
    rr = es.aipw_risk_ratio(data)
    assert rr.risk_ratio == pytest.approx(risk[1.0] / risk[0.0])
    assert rr.ci_lower < rr.risk_ratio < rr.ci_upper


def test_subgroup_effects_and_heterogeneity_test(
    stratified: tuple[es.CausalData, float, float],
) -> None:
    data, _, _ = stratified
    groups = np.where(np.arange(data.n) % 2 == 0, "even", "odd")
    subgroups = es.subgroup_effects(data, groups)
    assert set(subgroups) == {"even", "odd"}
    assert subgroups["even"].estimand.endswith("group = even")
    test = es.heterogeneity_test(subgroups)
    assert test.df == 1 and 0 <= test.p_value <= 1
    same = {k: subgroups["even"] for k in ("a", "b", "c")}
    assert es.heterogeneity_test(same).q_statistic == pytest.approx(0.0)
    with pytest.raises(ValueError):
        es.heterogeneity_test({"a": subgroups["even"]})


def test_potential_outcomes_decomposition_is_exact() -> None:
    rng = np.random.default_rng(15)
    x = rng.normal(size=1000)
    t = rng.random(1000) < 1 / (1 + np.exp(-x))
    po = PotentialOutcomes(x, x + 1 + x, t)
    decomposition = po.naive_decomposition()
    assert decomposition.naive_difference == pytest.approx(
        decomposition.att + decomposition.selection_bias
    )
    assert po.ate() == pytest.approx(float(np.mean(1 + x)))
    cate = po.cate(np.where(x > 0, "high", "low"))
    assert cate["high"] > cate["low"]
    with pytest.raises(ValueError, match="both treated and untreated"):
        PotentialOutcomes([1.0, 2.0], [2.0, 3.0], [1, 1])


def _statsmodels_design(seed: int, n: int, ps_coef: float, het: float):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    t = (rng.random(n) < 1 / (1 + np.exp(-(ps_coef * x1 - 0.6 * x2)))).astype(int)
    y = 1 + x1 + 0.5 * x2 + t * (2 + het * x1) + rng.normal(size=n)
    return x1, x2, t, y


@pytest.mark.parametrize(("seed", "ps_coef", "het"), [(3, 0.4, 0.5), (4, 1.5, 2.0)])
def test_estimates_and_ate_standard_errors_match_statsmodels_treatment_effect(
    seed: int, ps_coef: float, het: float
) -> None:
    """Independent reference: statsmodels' TreatmentEffect (joint GMM, accounts for
    propensity estimation). Regression test: IPW SEs used to ignore propensity
    estimation (≈1.6× too wide for the ATE; anti-conservative risk for the ATT)."""
    sm = pytest.importorskip("statsmodels.api")
    from statsmodels.treatment.treatment_effects import TreatmentEffect

    x1, x2, t, y = _statsmodels_design(seed, 3000, ps_coef, het)
    design = sm.add_constant(np.column_stack([x1, x2]))
    reference = TreatmentEffect(
        sm.OLS(y, design), t, results_select=sm.Logit(t, design).fit(disp=0)
    )
    data = es.CausalData.from_arrays(t, y, pd.DataFrame({"x1": x1, "x2": x2}))

    def first(result: object) -> tuple[float, float]:
        return (
            float(np.atleast_1d(result.effect)[0]),  # type: ignore[attr-defined]
            float(np.atleast_1d(result.sd)[0]),  # type: ignore[attr-defined]
        )

    for ours, theirs in (
        (es.inverse_probability_weighting(data), reference.ipw(return_results=True)),
        (es.augmented_ipw(data), reference.aipw(return_results=True)),
    ):
        effect, sd = first(theirs)
        assert ours.estimate == pytest.approx(effect, abs=5e-4)
        assert ours.standard_error == pytest.approx(sd, rel=0.03)
    att_effect, _ = first(reference.ipw(effect_group=1, return_results=True))
    assert es.inverse_probability_weighting(data, estimand="ATT").estimate == pytest.approx(
        att_effect, abs=5e-4
    )
    ra_ate, _ = first(reference.ra(return_results=True))
    ra_att, _ = first(reference.ra(effect_group=1, return_results=True))
    rng = seeded_generator(1)
    assert es.regression_adjustment(data, rng=rng, n_bootstrap=100).estimate == pytest.approx(
        ra_ate, abs=1e-10
    )
    assert es.regression_adjustment(
        data, estimand="ATT", rng=rng, n_bootstrap=100
    ).estimate == pytest.approx(ra_att, abs=1e-10)


def test_ipw_known_propensity_uses_uncorrected_influence_function() -> None:
    x1, x2, t, y = _statsmodels_design(5, 2000, 1.0, 1.0)
    e = 1 / (1 + np.exp(-(1.0 * x1 - 0.6 * x2)))
    data = es.CausalData.from_arrays(t, y, pd.DataFrame({"x1": x1, "x2": x2}))
    known = es.inverse_probability_weighting(data, propensity=e)
    m1 = np.sum(t * y / e) / np.sum(t / e)
    m0 = np.sum((1 - t) * y / (1 - e)) / np.sum((1 - t) / (1 - e))
    phi = t * (y - m1) / e - (1 - t) * (y - m0) / (1 - e)
    assert known.standard_error == pytest.approx(np.std(phi, ddof=1) / np.sqrt(t.size))
    assert "known propensity" in known.se_method
    assert "accounted for" in es.inverse_probability_weighting(data).se_method
