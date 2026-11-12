"""Planning and A/B analysis against independent references (closed forms, statsmodels, SciPy)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest

from decision_intelligence.experimentation import analysis as an
from decision_intelligence.experimentation import planning as pl


def _fleiss_n(p_c: float, p_t: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Closed-form per-arm n (pooled null SE, unpooled alternative SE; one tail)."""
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p_c + p_t) / 2
    root = za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p_c * (1 - p_c) + p_t * (1 - p_t))
    return root**2 / (p_t - p_c) ** 2


@pytest.mark.parametrize(("baseline", "mde"), [(0.05, 0.01), (0.10, 0.02), (0.30, 0.03)])
def test_binary_plan_matches_closed_form(baseline: float, mde: float) -> None:
    plan = pl.plan_binary(baseline, mde_absolute=mde)
    assert abs(plan.n_control - _fleiss_n(baseline, baseline + mde)) <= 1.0
    assert plan.n_control == plan.n_treatment and plan.achieved_power >= 0.8
    relative = pl.plan_binary(baseline, mde_relative=mde / baseline)
    assert relative.n_control == plan.n_control


def test_binary_plan_options() -> None:
    base = pl.plan_binary(0.10, mde_absolute=0.01, daily_eligible=2000)
    assert base.days == math.ceil(base.n_total / 2000)
    bonferroni = pl.plan_binary(0.10, mde_absolute=0.01, n_comparisons=3)
    assert bonferroni.alpha_per_comparison == pytest.approx(0.05 / 3)
    assert bonferroni.n_control > base.n_control
    clustered = pl.plan_binary(0.10, mde_absolute=0.01, design_effect_factor=2.5)
    assert clustered.n_control == math.ceil(base.n_control * 2.5 - 1e-9)
    unequal = pl.plan_binary(0.10, mde_absolute=0.01, treatment_share=0.2)
    assert unequal.n_control == pytest.approx(4 * unequal.n_treatment, rel=0.01)
    assert unequal.n_total > base.n_total  # unequal allocation is less efficient
    with pytest.raises(ValueError):
        pl.plan_binary(0.10)
    with pytest.raises(ValueError):
        pl.plan_binary(0.10, mde_absolute=0.01, mde_relative=0.1)


def test_mde_inverts_the_plan() -> None:
    plan = pl.plan_binary(0.10, mde_absolute=0.02)
    mde = pl.mde_binary(0.10, plan.n_control, plan.n_treatment)
    assert mde == pytest.approx(0.02, abs=2e-4) and mde <= 0.02


def test_continuous_plan_matches_statsmodels() -> None:
    plan = pl.plan_continuous(50.0, 10.0, mde_absolute=1.0, treatment_share=1 / 3)
    reference = TTestIndPower().solve_power(effect_size=0.1, power=0.8, alpha=0.05, ratio=2.0)
    assert abs(plan.n_treatment - reference) <= 1.0
    assert (
        plan.n_control == math.ceil(2 * plan.n_treatment)
        or abs(plan.n_control - 2 * plan.n_treatment) <= 1
    )
    mde = pl.mde_continuous(10.0, plan.n_control, plan.n_treatment)
    assert mde == pytest.approx(1.0, abs=0.01)


def test_design_effect() -> None:
    assert pl.design_effect(1, 0.3) == 1.0
    assert pl.design_effect(51, 0.02) == pytest.approx(2.0)
    assert pl.design_effect(51, 0.02, cluster_size_cv=0.5) > 2.0
    with pytest.raises(ValueError):
        pl.design_effect(10, 1.5)


@pytest.fixture
def binary_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(4)
    arm = np.repeat(["control", "treatment"], [1500, 1400])
    y = np.concatenate([rng.random(1500) < 0.10, rng.random(1400) < 0.13]).astype(float)
    return y, arm


def test_binary_analysis_matches_statsmodels(binary_data: tuple[np.ndarray, np.ndarray]) -> None:
    y, arm = binary_data
    result = an.analyse_binary(y, arm)
    s_t, s_c = int(y[arm == "treatment"].sum()), int(y[arm == "control"].sum())
    _, p = proportions_ztest([s_t, s_c], [1400, 1500])
    assert result.p_value == pytest.approx(p, rel=1e-10)
    lo, hi = confint_proportions_2indep(s_t, 1400, s_c, 1500, method="newcomb", compare="diff")
    assert (result.effect.ci_lower, result.effect.ci_upper) == pytest.approx((lo, hi), abs=1e-12)
    rr = (s_t / 1400) / (s_c / 1500)
    assert result.relative_lift == pytest.approx(rr - 1)
    assert result.effect.ci_lower < result.effect.estimate < result.effect.ci_upper
    assert result.relative_ci_lower < result.relative_lift < result.relative_ci_upper


def test_continuous_analysis_matches_scipy_and_delta_method() -> None:
    rng = np.random.default_rng(5)
    y_c, y_t = rng.lognormal(3, 1, 800), rng.lognormal(3.05, 1, 700)
    y = np.concatenate([y_c, y_t])
    arm = np.repeat(["control", "treatment"], [800, 700])
    result = an.analyse_continuous(y, arm, rng=np.random.default_rng(0), n_bootstrap=999)
    reference = stats.ttest_ind(y_t, y_c, equal_var=False)
    assert result.p_value == pytest.approx(reference.pvalue, rel=1e-10)
    # Delta-method check against a direct simulation of the log ratio's sampling variance.
    sims = [
        math.log(rng.choice(y_t, 700).mean() / rng.choice(y_c, 800).mean()) for _ in range(4000)
    ]
    se_log = math.log((1 + result.relative_ci_upper) / (1 + result.relative_ci_lower)) / (
        2 * stats.norm.ppf(0.975)
    )
    assert se_log == pytest.approx(float(np.std(sims)), rel=0.06)
    assert result.bootstrap_ci is not None and result.bootstrap_ci[0] < result.effect.estimate


def test_missing_outcomes_and_bad_labels_are_refused() -> None:
    with pytest.raises(ValueError, match="attrition"):
        an.analyse_binary([0, 1, np.nan, 1], ["control", "control", "treatment", "treatment"])
    with pytest.raises(ValueError, match="unexpected"):
        an.analyse_binary([0, 1, 0, 1], ["control", "control", "treatment", "other"])
    with pytest.raises(ValueError, match="0/1"):
        an.analyse_binary([0, 2, 0, 1], ["control", "control", "treatment", "treatment"])


def test_stratified_difference_matches_manual_computation() -> None:
    rng = np.random.default_rng(6)
    strata = np.repeat(["a", "b"], [600, 400])
    arm = np.tile(["control", "treatment"], 500)
    y = rng.normal(np.where(strata == "a", 0.0, 3.0) + 0.5 * (arm == "treatment"), 1.0)
    est = an.stratified_difference(y, arm, strata)
    manual, var = 0.0, 0.0
    for s, w in (("a", 0.6), ("b", 0.4)):
        t, c = y[(strata == s) & (arm == "treatment")], y[(strata == s) & (arm == "control")]
        manual += w * (t.mean() - c.mean())
        var += w**2 * (t.var(ddof=1) / t.size + c.var(ddof=1) / c.size)
    assert est.estimate == pytest.approx(manual) and est.standard_error == pytest.approx(
        math.sqrt(var)
    )
    plain = an.analyse_continuous(y, arm).effect
    assert est.standard_error < plain.standard_error  # blocking on a prognostic stratum helps


def test_clustered_difference_matches_statsmodels_cluster_ols() -> None:
    rng = np.random.default_rng(7)
    clusters = np.repeat(np.arange(20), 30)
    treated = np.isin(clusters, rng.permutation(20)[:10])
    y = rng.normal(size=20)[clusters] + rng.normal(size=600) + 0.2 * treated
    arm = np.where(treated, "treatment", "control")
    result = an.clustered_difference(y, arm, clusters)
    fit = sm.OLS(y, sm.add_constant(treated.astype(float))).fit(
        cov_type="cluster", cov_kwds={"groups": clusters}
    )
    assert result.cluster_robust.estimate == pytest.approx(fit.params[1])
    assert result.cluster_robust.standard_error == pytest.approx(fit.bse[1], rel=1e-8)
    assert result.naive.standard_error < 0.5 * result.cluster_robust.standard_error
    assert result.n_clusters == 20 and result.n_treated_clusters == 10
    mixed = clusters.copy()
    with pytest.raises(ValueError, match="both arms"):
        an.clustered_difference(y, np.where(np.arange(600) % 2, "treatment", "control"), mixed)


def test_multi_arm_comparison_adjusts_and_widens() -> None:
    rng = np.random.default_rng(8)
    arm = np.repeat(["control", "b", "c", "d"], 2000)
    y = (rng.random(8000) < np.where(arm == "d", 0.13, 0.10)).astype(float)
    result = an.multi_arm_comparison(y, arm, kind="binary")
    assert set(result.comparisons) == {"b", "c", "d"}
    raw = [result.comparisons[k].p_value for k in ("b", "c", "d")]
    assert all(result.adjusted_p_values[k] >= p for k, p in zip(("b", "c", "d"), raw, strict=True))
    single = an.analyse_binary(
        y[np.isin(arm, ["control", "d"])], arm[np.isin(arm, ["control", "d"])], treatment="d"
    )
    widened = result.comparisons["d"].effect
    assert widened.ci_upper - widened.ci_lower > single.effect.ci_upper - single.effect.ci_lower
    assert result.simultaneous_level == pytest.approx(1 - 0.05 / 3)
