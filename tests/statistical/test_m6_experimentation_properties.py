"""Repeated-sampling behaviour of M6 experiment analyses.

Fixed seeds make every assertion deterministic; tolerances are stated in
Monte Carlo standard errors (bias) or Wilson-interval terms (rates).
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from decision_intelligence.data.config import SyntheticDataConfig
from decision_intelligence.data.experiment_scenarios import ExperimentScenarioConfig
from decision_intelligence.data.pipeline import build_synthetic_dataset
from decision_intelligence.experimentation import cases as c


def _unbiased(summary, z: float = 4.0) -> bool:
    return abs(summary.bias) <= z * summary.bias_mc_se


def _nominal(rate, target: float = 0.95) -> bool:
    return rate.coverage_ci_lower <= target <= rate.coverage_ci_upper


def test_valid_design_recovers_effects_with_nominal_coverage() -> None:
    results = c.recovery_study(ExperimentScenarioConfig(), n_replications=150)
    for summary in results.values():
        assert _unbiased(summary), summary.estimator
        assert summary.coverage >= 0.9, summary.estimator
    plain = results["conversion: difference (Newcombe)"]
    assert plain.mean_reported_se == pytest.approx(plain.empirical_sd, rel=0.15)


def test_aa_tests_hold_their_size_at_scale() -> None:
    for rate in c.aa_study(n_units=20_000, n_replications=100):
        assert rate.ci_lower <= 0.05 <= rate.ci_upper or rate.rate <= 0.05, rate.procedure


def test_small_skewed_samples_distort_revenue_inference() -> None:
    welch, bootstrap = c.aa_study(n_units=200, n_replications=300)[1:]
    assert welch.rate < 0.05 and bootstrap.rate > 0.08  # conservative vs anti-conservative


def test_planned_sample_sizes_achieve_their_power() -> None:
    for check in c.planning_validation_study(n_replications=300):
        rate = check.empirical
        assert rate.ci_lower - 0.02 <= check.target_power <= rate.ci_upper + 0.02, check.design


def test_srm_test_size_and_cause_dependence() -> None:
    strict, loose = c.srm_size_study(n_units=10_000, n_replications=300)
    assert strict.rate <= 0.01 and loose.ci_lower <= 0.05 <= loose.ci_upper
    rows = {(r.mechanism, r.loss_share): r
            for r in c.srm_cause_study(loss_shares=(0.02, 0.05), n_replications=60)}
    # Equal detection for equal loss, whatever the cause ...
    for share in (0.02, 0.05):
        rates = [rows[(m, share)].detection.rate for m in ("random", "inactive")]
        assert abs(rates[0] - rates[1]) < 0.15
    assert rows[("random", 0.05)].detection.rate > 0.85
    # ... but only outcome-related loss biases the estimate.
    assert _unbiased(rows[("random", 0.05)].bias)
    inactive = rows[("inactive", 0.05)].bias
    assert inactive.bias > 6 * inactive.bias_mc_se
    # Damaging loss can be below the SRM test's detection threshold.
    assert rows[("inactive", 0.02)].detection.rate < 0.2
    assert rows[("inactive", 0.02)].bias.bias > 3 * rows[("inactive", 0.02)].bias.bias_mc_se


def test_noncompliance_itt_and_cace_are_valid_naive_comparisons_are_not() -> None:
    config = ExperimentScenarioConfig(never_taker_share=0.3, always_taker_share=0.1,
                                      compliance_engagement_link=2.0)
    results = c.noncompliance_study(config, n_replications=120)
    for key in ("ITT (vs ITT truth)", "CACE / Wald (vs complier effect)"):
        assert _unbiased(results[key]) and results[key].coverage >= 0.88, key
    for key in ("as-treated (vs ATE)", "per-protocol (vs ATE)"):
        assert results[key].bias > 10 * results[key].bias_mc_se and results[key].coverage < 0.2


def test_attrition_complete_case_bias_and_bounds() -> None:
    config = ExperimentScenarioConfig(attrition_control=0.1, attrition_treatment=0.25,
                                      attrition_engagement_link=1.5)
    study = c.attrition_study(config, n_replications=60, n_bootstrap=199)
    assert study.complete_case.bias > 4 * study.complete_case.bias_mc_se
    assert study.lee_contains_truth.rate >= 0.9 and study.manski_contains_truth.rate == 1.0
    assert study.lee_mean_width < study.manski_mean_width


def test_ignoring_cluster_randomisation_inflates_false_positives() -> None:
    naive, robust, means = c.clustering_study(n_clusters=40, cluster_sd=0.5, n_replications=200)
    assert naive.rate > 0.2
    for rate in (robust, means):
        assert rate.ci_lower <= 0.05 <= rate.ci_upper or rate.rate < 0.05


def test_post_randomisation_selection_biases_the_effect() -> None:
    summary = c.post_selection_study(n_replications=100)
    assert summary.bias < -6 * summary.bias_mc_se


def test_multiple_testing_controls_familywise_error() -> None:
    rows = {r.procedure: r for r in c.multiple_testing_study(n_replications=300)}
    assert rows["unadjusted"].familywise_error.rate > 0.2
    for name in ("bonferroni", "holm"):
        assert rows[name].familywise_error.ci_lower <= 0.05
    assert rows["holm"].power >= rows["bonferroni"].power
    assert rows["benjamini-hochberg"].power >= rows["holm"].power


def test_decision_rule_operating_characteristics() -> None:
    _, table = c.decision_operating_characteristics(effects=(0.0, 0.01, 0.02),
                                                    n_replications=300)
    by_effect = table.set_index("true effect")
    assert by_effect.loc[0.0, "LAUNCH"] == 0.0 and by_effect.loc[0.0, "STOP"] > 0.6
    assert 0.3 < by_effect.loc[0.01, "LAUNCH"] < 0.7
    assert by_effect.loc[0.02, "LAUNCH"] > 0.95


def test_m2_experiment_repeated_sampling() -> None:
    dataset = build_synthetic_dataset(SyntheticDataConfig())
    real = c.m2_replication_study(dataset, effect=0.04, n_replications=300)
    assert _unbiased(real.summary) and _nominal(real.summary)
    # The realised design is underpowered: about half of the replicates miss a real effect.
    assert abs(real.rejection.rate - real.planned_power) < 0.1
    assert real.decisions["HOLD"] > 0.3
    null = c.m2_replication_study(replace(dataset), effect=0.0, n_replications=300)
    assert null.rejection.ci_lower <= 0.05 <= null.rejection.ci_upper
    # Under no effect, LAUNCH requires an upward false positive (rate <= alpha / 2); in this
    # underpowered design such a false positive necessarily overstates the effect (type M error).
    assert null.decisions["LAUNCH"] <= 0.05
