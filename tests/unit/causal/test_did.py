"""DiD and event-study estimators: analytic identities, references and safeguards."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from decision_intelligence.causal_inference import did
from decision_intelligence.data.rollout_panel import RolloutConfig, generate_rollout_panel
from decision_intelligence.statistics.hypothesis_tests import two_sample_t_test


@pytest.fixture(scope="module")
def staggered() -> pd.DataFrame:
    return generate_rollout_panel().panel


@pytest.fixture(scope="module")
def noiseless():
    return generate_rollout_panel(RolloutConfig(noise_sd=1e-12))


def test_panel_validation() -> None:
    panel = generate_rollout_panel(RolloutConfig(n_units=8, n_periods=6, cohorts=((4, 0.5),),
                                                 cohort_effect_scales=(1.0,))).panel
    with pytest.raises(ValueError, match="unbalanced"):
        did.as_panel(panel.iloc[1:])
    with pytest.raises(ValueError, match="duplicate"):
        did.as_panel(pd.concat([panel, panel.iloc[:1]]))
    varying = panel.copy()
    varying.loc[varying.index[0], "cohort"] = 5
    with pytest.raises(ValueError, match="constant within unit"):
        did.as_panel(varying)
    early = panel.copy()
    early.loc[early["unit_id"] == early["unit_id"].iloc[0], "cohort"] = 1
    with pytest.raises(ValueError, match="first period"):
        did.as_panel(early)


def test_two_by_two_equals_twfe_and_welch_in_a_two_period_panel() -> None:
    full = generate_rollout_panel(
        RolloutConfig(n_units=40, cohorts=((9, 0.5),), cohort_effect_scales=(1.0,))
    ).panel
    panel = full[full["period"].isin([8, 9])]
    two_by_two = did.two_period_did(panel, 8, 9)
    twfe = did.twfe_did(panel).estimate
    assert two_by_two.estimate == pytest.approx(twfe.estimate, abs=1e-10)
    wide = panel.pivot(index="unit_id", columns="period", values="outcome")
    cohort = panel.groupby("unit_id")["cohort"].first()
    change = wide[9] - wide[8]
    welch = two_sample_t_test(change[cohort == 9], change[cohort == 0])
    assert two_by_two.ci_lower == pytest.approx(welch.ci_lower)
    with pytest.raises(ValueError):
        did.two_period_did(panel, 9, 8)


def test_twfe_matches_statsmodels_dummy_regression_with_clustered_se(
    staggered: pd.DataFrame,
) -> None:
    ours = did.twfe_did(staggered)
    design = pd.get_dummies(
        staggered[["unit_id", "period"]].astype(str), drop_first=True, dtype=float
    )
    design.insert(0, "treated", staggered["treated"].astype(float))
    reference = sm.OLS(staggered["outcome"], sm.add_constant(design)).fit(
        cov_type="cluster", cov_kwds={"groups": staggered["unit_id"]}
    )
    assert ours.estimate.estimate == pytest.approx(reference.params["treated"], rel=1e-10)
    assert ours.estimate.standard_error == pytest.approx(reference.bse["treated"], rel=1e-8)
    assert ours.staggered and any("staggered adoption" in w for w in ours.estimate.warnings)


def test_twfe_weight_decomposition_is_exact(noiseless) -> None:
    weights = did.twfe_weights(noiseless.panel)
    merged = weights.merge(noiseless.ground_truth.cell_effects, on=["unit_id", "period"])
    assert weights["weight"].sum() == pytest.approx(1.0)
    implied = float(np.sum(merged["weight"] * merged["true_direct_effect"]))
    assert did.twfe_did(noiseless.panel).estimate.estimate == pytest.approx(implied, abs=1e-8)
    assert (weights["weight"] < 0).any()  # staggered adoption: some effects enter negatively
    assert implied < noiseless.ground_truth.overall_att - 0.5


def test_common_timing_has_no_negative_weights() -> None:
    panel = generate_rollout_panel(
        RolloutConfig(cohorts=((13, 0.5),), cohort_effect_scales=(1.0,))
    ).panel
    assert (did.twfe_weights(panel)["weight"] > 0).all()
    assert not did.twfe_did(panel).staggered


@pytest.mark.parametrize("control_group", ["never", "not_yet"])
def test_group_time_att_recovers_noiseless_truth_exactly(noiseless, control_group: str) -> None:
    cells = did.group_time_att(did.as_panel(noiseless.panel), control_group=control_group)  # type: ignore[arg-type]
    truth = noiseless.ground_truth.att_by_cohort_period
    for cell in cells:
        assert cell.estimate == pytest.approx(truth.get((cell.cohort, cell.period), 0.0), abs=1e-8)
    result = did.staggered_did(noiseless.panel, control_group=control_group)  # type: ignore[arg-type]
    assert result.overall.estimate == pytest.approx(noiseless.ground_truth.overall_att, abs=1e-8)


def test_cell_influence_function_reproduces_welch_standard_error(staggered: pd.DataFrame) -> None:
    panel = did.as_panel(staggered)
    cell = next(c for c in did.group_time_att(panel) if c.cohort == 9 and c.period == 12)
    index = {int(p): k for k, p in enumerate(panel.periods)}
    change = panel.outcome[:, index[12]] - panel.outcome[:, index[8]]
    welch = two_sample_t_test(change[panel.cohort == 9], change[panel.cohort == 0])
    se_welch = math.sqrt(
        change[panel.cohort == 9].var(ddof=1) / 15 + change[panel.cohort == 0].var(ddof=1) / 15
    )
    assert cell.estimate == pytest.approx(welch.estimate)
    assert math.sqrt(float(np.sum(cell.influence**2))) == pytest.approx(se_welch)


def test_event_study_equals_saturated_dynamic_twfe_under_common_timing() -> None:
    """With one adoption date, never-treated controls and no binning, the CS event-time
    estimates with a universal base equal the dynamic TWFE coefficients exactly."""
    data = generate_rollout_panel(
        RolloutConfig(n_units=30, n_periods=10, cohorts=((6, 0.5),), cohort_effect_scales=(1.0,))
    )
    cs = did.staggered_did(data.panel)
    dynamic = did.event_study_twfe(data.panel, leads=5, lags=4)
    cs_by_event = {e.event_time: e.estimate for e in cs.event_study}
    for coefficient in dynamic.coefficients:
        assert coefficient.estimate == pytest.approx(cs_by_event[coefficient.event_time], abs=1e-9)


def test_anticipation_moves_the_base_period() -> None:
    data = generate_rollout_panel(RolloutConfig(noise_sd=1e-12, anticipation_periods=2))
    naive = did.staggered_did(data.panel)
    modelled = did.staggered_did(data.panel, anticipation=2)
    assert modelled.overall.estimate == pytest.approx(data.ground_truth.overall_att, abs=1e-8)
    assert naive.overall.estimate < data.ground_truth.overall_att - 0.5
    assert all(c.base_period == c.cohort - 3 for c in modelled.cells)


def test_pretrend_test_references_and_safeguards() -> None:
    b = np.array([0.5, -0.2])
    v = np.array([[0.04, 0.01], [0.01, 0.05]])
    chi = did.pretrend_wald_test(b, v, [-3, -2])
    f = did.pretrend_wald_test(b, v, [-3, -2], n_clusters=20)
    assert chi.statistic == pytest.approx(float(b @ np.linalg.solve(v, b)))
    assert f.p_value > chi.p_value  # finite-cluster reference is more conservative
    assert "F(2, 18)" in f.reference and "non-rejection" in f.note
    with pytest.raises(ValueError, match="singular"):
        did.pretrend_wald_test(b, np.ones((2, 2)), [-3, -2])
    with pytest.raises(ValueError, match="more clusters"):
        did.pretrend_wald_test(b, v, [-3, -2], n_clusters=2)


def test_staggered_did_bootstrap_option_and_warnings(staggered: pd.DataFrame) -> None:
    analytic = did.staggered_did(staggered)
    boot = did.staggered_did(
        staggered, se_method="bootstrap", rng=np.random.default_rng(1), n_bootstrap=199
    )
    assert boot.overall.estimate == analytic.overall.estimate
    assert boot.overall.standard_error == pytest.approx(analytic.overall.standard_error, rel=0.35)
    assert any("few clusters" in w for w in analytic.overall.warnings)
    with pytest.raises(ValueError, match="rng"):
        did.staggered_did(staggered, se_method="bootstrap")


def test_dynamic_twfe_flags_staggered_adoption(staggered: pd.DataFrame) -> None:
    result = did.event_study_twfe(staggered, leads=6, lags=8)
    assert any("Sun & Abraham" in w for w in result.warnings)
    assert not any("joint F-test" in w for w in result.warnings)
    with pytest.raises(ValueError):
        did.event_study_twfe(staggered, leads=2, lags=3, reference=-3)


def test_sup_t_critical_value_identities() -> None:
    from scipy import stats

    assert did.sup_t_critical_value(np.eye(1)) == pytest.approx(stats.norm.ppf(0.975), abs=0.01)
    sidak = stats.norm.ppf(1 - (1 - 0.95 ** (1 / 5)) / 2)  # 5 independent coefficients
    assert did.sup_t_critical_value(np.eye(5)) == pytest.approx(sidak, abs=0.01)
    collinear = np.full((4, 4), 1.0) + 1e-9 * np.eye(4)  # perfectly correlated: one test
    assert did.sup_t_critical_value(collinear) == pytest.approx(stats.norm.ppf(0.975), abs=0.01)
    with pytest.raises(ValueError):
        did.sup_t_critical_value(np.zeros((2, 2)))


def test_event_study_reports_simultaneous_band_over_window(staggered: pd.DataFrame) -> None:
    result = did.staggered_did(staggered)
    inside = [e for e in result.event_study if not math.isnan(e.band_lower)]
    assert min(e.event_time for e in inside) == -7 and max(e.event_time for e in inside) >= 8
    for e in inside:
        assert e.band_lower < e.ci_lower and e.band_upper > e.ci_upper
    assert all(math.isnan(e.band_lower) for e in result.event_study if e.event_time < -7)


def test_pretrend_warns_with_small_groups() -> None:
    small = generate_rollout_panel(RolloutConfig(n_units=20)).panel
    assert any("over-sized" in w for w in did.staggered_did(small).overall.warnings)
    assert not any("over-sized" in w for w in did.staggered_did(
        generate_rollout_panel().panel).overall.warnings)


@pytest.mark.parametrize("control_group", ["never", "not_yet"])
def test_callaway_santanna_matches_independent_pandas_implementation(
    staggered: pd.DataFrame, control_group: str
) -> None:
    """Independent reference on noisy data: ATT(g, t) by explicit group means of changes
    from the universal base period, and the cohort-size-weighted overall aggregate."""
    wide = staggered.pivot(index="unit_id", columns="period", values="outcome")
    cohort = staggered.groupby("unit_id")["cohort"].first()
    reference = {}
    for g in sorted(cohort[cohort > 0].unique()):
        base = g - 1
        for t in wide.columns:
            if t == base:
                continue
            if control_group == "never":
                controls = cohort == 0
            else:
                controls = (cohort == 0) | ((cohort > max(t, base)) & (cohort != g))
            change = wide[t] - wide[base]
            reference[(g, t)] = change[cohort == g].mean() - change[controls].mean()
    ours = did.group_time_att(did.as_panel(staggered), control_group=control_group)  # type: ignore[arg-type]
    assert len(ours) == len(reference)
    for cell in ours:
        assert cell.estimate == pytest.approx(reference[(cell.cohort, cell.period)], abs=1e-12)
    sizes = cohort[cohort > 0].value_counts()
    post = {k: v for k, v in reference.items() if k[1] >= k[0]}
    weights = np.array([sizes[k[0]] for k in post], dtype=float)
    expected = float(np.sum(weights * np.array(list(post.values()))) / weights.sum())
    result = did.staggered_did(staggered, control_group=control_group)  # type: ignore[arg-type]
    assert result.overall.estimate == pytest.approx(expected, abs=1e-12)
