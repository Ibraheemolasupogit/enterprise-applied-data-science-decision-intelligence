"""Repeated-sampling behaviour of M5 designs: recovery when valid, documented failure when not.

Fixed seeds make every assertion deterministic; tolerances are in Monte
Carlo standard errors (bias) or Wilson-interval terms (rates).
"""

from __future__ import annotations

from decision_intelligence.causal_inference import observational_cases as oc
from decision_intelligence.data.rollout_panel import RolloutConfig
from decision_intelligence.data.synthetic_control_panel import SyntheticControlConfig
from decision_intelligence.data.threshold_assignment import ThresholdConfig


def _unbiased(summary, z: float = 4.0) -> bool:
    return abs(summary.bias) <= z * summary.bias_mc_se


def test_callaway_santanna_recovers_staggered_att_with_nominal_coverage() -> None:
    result = oc.did_monte_carlo(RolloutConfig(), anticipation=0, n_replications=120)
    cs, twfe = result["callaway_santanna"], result["twfe"]
    assert _unbiased(cs)
    assert cs.coverage_ci_lower <= 0.95 <= cs.coverage_ci_upper or cs.coverage >= 0.9
    # Static TWFE is biased under staggered adoption with heterogeneous effects.
    assert twfe.bias < -10 * twfe.bias_mc_se
    assert twfe.coverage < 0.5
    # The windowed finite-cluster pre-trend test is close to its nominal size here,
    # while the dynamic TWFE "pre-trend" rejects although trends are parallel.
    assert result["cs_pretrend_rejection"].rate <= 0.15
    assert result["twfe_pretrend_rejection"].rate >= 0.8


def test_twfe_is_fine_under_common_timing() -> None:
    template = RolloutConfig(cohorts=((13, 0.5),), cohort_effect_scales=(1.0,))
    result = oc.did_monte_carlo(template, anticipation=0, n_replications=80)
    assert _unbiased(result["twfe"]) and _unbiased(result["callaway_santanna"])
    assert result["twfe"].coverage >= 0.85


def test_invalid_did_designs_bias_the_estimate() -> None:
    parallel = oc.did_monte_carlo(RolloutConfig(differential_trend=0.08), anticipation=0,
                                  n_replications=60)
    assert parallel["callaway_santanna"].bias > 5 * parallel["callaway_santanna"].bias_mc_se
    spill = oc.did_monte_carlo(RolloutConfig(spillover_share=0.5), anticipation=0,
                               n_replications=60)
    assert spill["callaway_santanna"].bias < -5 * spill["callaway_santanna"].bias_mc_se


def test_anticipation_biases_unless_modelled() -> None:
    template = RolloutConfig(anticipation_periods=2)
    ignored = oc.did_monte_carlo(template, anticipation=0, n_replications=60)
    modelled = oc.did_monte_carlo(template, anticipation=2, n_replications=60)
    assert ignored["callaway_santanna"].bias < -10 * ignored["callaway_santanna"].bias_mc_se
    assert ignored["cs_pretrend_rejection"].rate > 0.7  # the pre-test detects it
    assert _unbiased(modelled["callaway_santanna"])


def test_few_cluster_inference() -> None:
    rates = {r.procedure: r for r in oc.few_cluster_size_study(10, 5, n_replications=300)}
    assert rates["classical OLS SE (ignores clustering)"].rate > 0.15
    for name in ("wild cluster bootstrap (Rademacher)", "wild cluster bootstrap (Webb)"):
        assert rates[name].ci_lower <= 0.05 <= rates[name].ci_upper or rates[name].rate < 0.08


def test_synthetic_control_valid_and_invalid_designs() -> None:
    valid = oc.sc_monte_carlo(SyntheticControlConfig(), n_replications=30, with_placebo=False)
    assert abs(valid["synthetic_control"].bias) < 0.5
    assert valid["synthetic_control"].rmse < valid["naive_did"].rmse
    assert valid["share_flagged"] < 0.1
    outside = oc.sc_monte_carlo(SyntheticControlConfig(treated_in_convex_hull=False),
                                n_replications=20, with_placebo=False)
    assert outside["share_flagged"] == 1.0
    contaminated = oc.sc_monte_carlo(
        SyntheticControlConfig(contaminated_donors=4, contamination_share=0.6),
        n_replications=30, with_placebo=False,
    )
    assert contaminated["synthetic_control"].bias < valid["synthetic_control"].bias - 0.5
    assert contaminated["share_flagged"] < 0.1  # contamination is invisible to fit diagnostics


def test_rdd_valid_design_and_failures() -> None:
    valid = oc.rdd_monte_carlo(ThresholdConfig(), n_replications=120)
    assert _unbiased(valid["robust"]) and valid["robust"].coverage >= 0.88
    assert valid["mccrary_rejection"].rate <= 0.12
    manipulated = oc.rdd_monte_carlo(ThresholdConfig(manipulation_share=0.6), n_replications=40)
    assert manipulated["robust"].bias > 5 * manipulated["robust"].bias_mc_se
    assert manipulated["mccrary_rejection"].rate > 0.9
    assert manipulated["covariate_rejection"].rate > 0.8
    compound = oc.rdd_monte_carlo(ThresholdConfig(compound_jump=2.0), n_replications=40)
    assert compound["robust"].bias > 1.5
    assert compound["mccrary_rejection"].rate < 0.2  # invisible to the density test


def test_conventional_rdd_inference_over_rejects_at_a_curved_placebo_cutoff() -> None:
    conventional, robust = oc.placebo_cutoff_study(n_replications=150)
    assert conventional.rate > robust.rate
    assert robust.rate <= 0.1


def test_event_study_simultaneous_band_restores_joint_coverage() -> None:
    """Pointwise intervals cover the whole event path far less than 95% of the time;
    the nominal 95% sup-t band improves joint coverage but under-covers (not fully calibrated)."""
    from decision_intelligence.causal_inference import did
    from decision_intelligence.data.rollout_panel import generate_rollout_panel

    pointwise = band = 0
    reps = 150
    for r in range(reps):
        data = generate_rollout_panel(RolloutConfig(seed=60_000 + r))
        result = did.staggered_did(data.panel)
        truth = data.ground_truth.event_time_att
        window = [e for e in result.event_study if e.band_lower == e.band_lower]
        pointwise += all(e.ci_lower <= truth.get(e.event_time, 0.0) <= e.ci_upper for e in window)
        band += all(e.band_lower <= truth.get(e.event_time, 0.0) <= e.band_upper for e in window)
    assert pointwise / reps < 0.7
    assert band / reps >= 0.85
