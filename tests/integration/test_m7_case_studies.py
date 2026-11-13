"""M7 case studies end to end: pre-specified checks, estimates against truth, decisions."""

from __future__ import annotations

import pytest

from decision_intelligence.experimentation import advanced_cases as ac


def test_cuped_case_rejects_invalid_covariates_and_shortens_the_experiment() -> None:
    case = ac.cuped_case()
    status = {c.check.split(": ")[1]: c.status for c in case.covariate_checks}
    assert status == {"pre_revenue": "PASS", "in_experiment_sessions": "FAIL",
                      "pre_revenue_leaky": "FAIL"}
    cuped, lin = case.results["CUPED"], case.results["Lin"]
    assert cuped.effect.covers(case.truth_ate) and lin.effect.covers(case.truth_ate)
    assert cuped.standard_error_ratio < 0.85
    assert not case.results["INVALID post-treatment"].effect.covers(case.truth_ate)
    assert case.plan_cuped.n_total < 0.75 * case.plan_unadjusted.n_total
    assert case.plan_cuped.days is not None and case.plan_unadjusted.days is not None
    assert case.plan_cuped.days < case.plan_unadjusted.days


def test_boundary_reference_reproduces_published_values() -> None:
    reference = ac.boundary_reference()
    for quantity, ours, published in reference.rows:
        assert ours == pytest.approx(published, abs=2e-3), quantity
    for quantity, ours, mvn in reference.mvn_check:
        assert ours == pytest.approx(mvn, abs=1e-6), quantity


def test_sequential_case_uses_completed_outcome_windows() -> None:
    case = ac.sequential_case()
    fractions = [look.information_fraction for look in case.looks]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0
    assert case.looks[0].n_analysed < case.looks[-1].n_analysed
    p_values = [look.always_valid_p_value for look in case.looks]
    assert p_values == sorted(p_values, reverse=True)
    if case.stopped_at_day is not None:
        stop = next(look for look in case.looks if look.calendar_day == case.stopped_at_day)
        assert abs(stop.z) >= stop.boundary
        assert stop.repeated_ci[0] <= case.truth <= stop.repeated_ci[1]


def test_bayesian_case_reports_priors_and_decisions() -> None:
    case = ac.bayesian_case()
    flat, p_one_sided = case.flat_prior_check
    assert flat == pytest.approx(p_one_sided, rel=1e-4)
    priors = [row.prior for row in case.revenue_rows]
    assert any("sceptical" in p for p in priors) and any("optimistic" in p for p in priors)
    sceptical = next(r for r in case.revenue_rows if "sceptical" in r.prior)
    vague = next(r for r in case.revenue_rows if "vague" in r.prior)
    assert sceptical.posterior_mean < vague.posterior_mean  # shrinkage toward zero
    for row in case.conversion_rows + case.revenue_rows:
        assert row.credible_interval[0] < row.posterior_mean < row.credible_interval[1]
        assert row.decision.decision in {"LAUNCH", "HOLD", "STOP"}


def test_cluster_and_switchback_cases() -> None:
    cluster = ac.cluster_case()
    assert cluster.plan.clusters_per_arm > cluster.plan.naive_clusters_per_arm
    assert cluster.naive_plan_power < 0.6
    assert cluster.icc.ci_lower is not None and cluster.icc.ci_lower > 0
    switchback = ac.switchback_case()
    assert switchback.analysis.burn_in_hours == ac.SWITCHBACK_BURN_IN
    assert switchback.analysis.effect.covers(switchback.truth)
    assert [row[0] for row in switchback.sensitivity] == [0, 1, 2, 3]


def test_cases_are_reproducible() -> None:
    a, b = ac.cuped_case(), ac.cuped_case()
    assert a.results["CUPED"].effect == b.results["CUPED"].effect
    s1, s2 = ac.switchback_case(), ac.switchback_case()
    assert s1.analysis == s2.analysis
