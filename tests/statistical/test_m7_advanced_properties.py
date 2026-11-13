"""Repeated-sampling properties of the M7 methods (smaller replicate counts than the evidence).

Tolerances use Monte Carlo uncertainty: Wilson intervals for rates, bias within a few Monte
Carlo standard errors.
"""

from __future__ import annotations

import pytest

from decision_intelligence.data.advanced_experiment_scenarios import CupedScenarioConfig
from decision_intelligence.experimentation import advanced_cases as ac


@pytest.fixture(scope="module")
def cuped() -> ac.CupedStudy:
    return ac.cuped_study(n_replications=120, config=CupedScenarioConfig(n_units=6_000))


def test_cuped_and_lin_are_unbiased_and_reduce_variance(cuped: ac.CupedStudy) -> None:
    by = {s.estimator: s for s in cuped.summaries}
    for name in ("unadjusted (M6 Welch)", "CUPED (pre_revenue)", "Lin (pre_revenue)"):
        s = by[name]
        assert abs(s.bias) < 4 * s.bias_mc_se, name
        assert s.coverage_ci_lower <= 0.95 <= s.coverage_ci_upper, name
    assert by["CUPED (pre_revenue)"].empirical_sd < 0.88 * by["unadjusted (M6 Welch)"].empirical_sd
    # Realised variance reduction is close to the pooled R^2 (theory: 1 - rho^2 factor).
    assert cuped.mean_variance_reduction_cuped == pytest.approx(cuped.mean_r_squared, abs=0.03)
    assert cuped.mean_variance_reduction_lin == pytest.approx(cuped.mean_r_squared, abs=0.03)


def test_invalid_adjustments_are_biased(cuped: ac.CupedStudy) -> None:
    by = {s.estimator: s for s in cuped.summaries}
    post = by["INVALID: CUPED on in-experiment sessions"]
    leak = by["INVALID: CUPED on leaky window"]
    assert post.bias < -1.0 and post.coverage < 0.2
    # The leaky window absorbs about half the effect in treated users; CUPED subtracts
    # theta times that arm difference, a bias of roughly -0.1 here.
    assert leak.bias < -3 * leak.bias_mc_se


def test_cuped_keeps_size_and_raises_power(cuped: ac.CupedStudy) -> None:
    for rate in cuped.aa_size:
        assert rate.ci_lower <= 0.05 <= rate.ci_upper, rate.procedure
    power = {p.procedure: p.rate for p in cuped.power}
    assert power["CUPED (pre_revenue)"] > power["unadjusted (M6 Welch)"]


def test_peeking_inflates_and_valid_procedures_control_type_one_error() -> None:
    study = ac.peeking_study(n_replications=1_500, effects=[0.0, 0.015])
    rows = {(s.procedure, s.effect): s for s in study.summaries}
    naive = rows[("naive peeking (10 looks)", 0.0)].rejection
    assert naive.ci_lower > 0.15  # Armitage et al.: 0.193 for 10 looks
    for name in ("fixed horizon (M6 baseline)", "Lan–DeMets O'Brien–Fleming (10 looks)",
                 "Lan–DeMets Pocock (10 looks)"):
        r = rows[(name, 0.0)].rejection
        assert r.ci_lower <= 0.05 <= r.ci_upper + 0.01, name
    assert rows[("mSPRT (checked 100 times)", 0.0)].rejection.ci_lower <= 0.05
    obf = rows[("Lan–DeMets O'Brien–Fleming (10 looks)", 0.015)]
    fixed = rows[("fixed horizon (M6 baseline)", 0.015)]
    assert abs(obf.rejection.rate - fixed.rejection.rate) < 0.05
    assert obf.expected_sample_fraction < 0.85
    # Rejecting experiments exaggerate the estimate; early stopping adds to the significance
    # filter that already affects the fixed-horizon test.
    assert obf.mean_estimate_when_rejecting > fixed.mean_estimate_when_rejecting > 0.015
    # Repeated confidence intervals keep (conservative) coverage at stopping.
    assert obf.interval_coverage >= 0.94


def test_bayesian_rules_are_not_immune_to_peeking() -> None:
    rates = ac.bayesian_fixed_truth_study(n_replications=150)
    by = {(r.rule, r.effect, r.monitoring): r.launch_rate for r in rates}
    fixed = by[("launch if P(better) > 0.95", 0.0, "fixed horizon")]
    peeking = by[("launch if P(better) > 0.95", 0.0, "checked at 10 looks")]
    assert fixed.ci_lower <= 0.05 <= fixed.ci_upper
    assert peeking.rate > fixed.rate and peeking.ci_upper > 0.1


def test_bayesian_calibration_holds_under_the_true_prior_even_with_optional_stopping() -> None:
    rows = {r.scenario: r for r in ac.bayesian_calibration_study(n_replications=6_000)}
    for name in ("prior correct, analysed once",
                 "prior correct, optional stopping at P(theta > 0) > 0.95"):
        c = rows[name].coverage
        assert c.ci_lower <= 0.95 <= c.ci_upper, name
    wrong = rows["prior too wide (truths 3x narrower), analysed once"]
    assert wrong.mean_posterior_when_launched > 2 * wrong.mean_effect_when_launched


def test_switchback_burn_in_and_randomisation_inference() -> None:
    study = ac.switchback_study(n_replications=120, n_permutations=99)
    by = {s.estimator: s for s in study.by_burn_in}
    assert by["burn-in 0 h"].bias < -4 * by["burn-in 0 h"].bias_mc_se
    assert abs(by["burn-in 2 h"].bias) < 4 * by["burn-in 2 h"].bias_mc_se
    assert by["burn-in 2 h"].coverage_ci_upper >= 0.95
    assert study.aa_randomisation_test.ci_lower <= 0.05
    assert study.aa_naive_hourly_complete.ci_lower > 0.1  # hours are not independent units


def test_cluster_plan_delivers_its_power_and_the_naive_plan_does_not() -> None:
    case = ac.cluster_case()
    rows = {r.design: r for r in ac.cluster_power_study(case, n_replications=120)}
    planned = rows["planned with estimated ICC"]
    naive = rows["naive plan (ICC ignored)"]
    assert planned.cr1_power.ci_lower <= planned.planned_power + 0.05
    assert planned.cr1_power.ci_upper >= planned.planned_power - 0.08
    assert naive.cr1_power.rate < planned.cr1_power.rate - 0.15
