"""M11 P0: precision-planning formulas against simulation and brute force; selection and
fit-status rules."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from decision_intelligence.customer_value import precision as pr11
from decision_intelligence.customer_value import predictive_protocol as pp
from decision_intelligence.customer_value.model_selection import (
    CandidateResult,
    FitDiagnostics,
    classify_fit,
    gate,
    select_model,
)


def test_critical_sum_and_detectable_excess() -> None:
    k = pr11.critical_sum(0.05, 0.80)
    assert k == pytest.approx(1.959964 + 0.841621, abs=1e-5)
    mde = pr11.minimum_detectable_relative_excess(5000, 0.05, 0.80)
    assert mde == pytest.approx(4 * k * k / 5000)
    assert pr11.required_n_eff(mde, 0.05, 0.80) == pytest.approx(5000)
    # At the detectable excess, E[D] = tau^2 equals k standard errors.
    sigma = 3.0
    tau = math.sqrt(mde) * sigma
    assert tau**2 == pytest.approx(k * pr11.paired_mse_difference_se(tau, sigma, 5000))
    with pytest.raises(ValueError):
        pr11.critical_sum(0.0, 0.8)


def test_paired_difference_se_matches_simulation() -> None:
    rng = np.random.default_rng(11)
    n, reps, sigma, tau = 2000, 400, 2.0, 0.4
    means = np.empty(reps)
    for r in range(reps):
        mu = rng.normal(10, 3, n)
        y = mu + rng.normal(0, sigma, n)
        m2 = mu + rng.normal(0, tau, n)
        means[r] = np.mean((y - mu) ** 2 - (y - m2) ** 2)
    se = pr11.paired_mse_difference_se(tau, sigma, n)
    # Exact variance adds Var(delta^2) = 2 tau^4 (normal delta): a small correction here.
    exact = math.sqrt((4 * tau**2 * sigma**2 + 2 * tau**4) / n)
    assert float(np.std(means, ddof=1)) == pytest.approx(exact, rel=0.12)
    assert se == pytest.approx(exact, rel=0.03)
    assert float(np.mean(means)) == pytest.approx(-(tau**2), abs=4 * exact / math.sqrt(reps))


def test_design_effect_reduces_to_known_cases() -> None:
    equal = np.full(300, 3)
    assert pr11.design_effect(equal, 0.4) == pytest.approx(1 + 2 * 0.4)  # 1 + (m - 1) rho
    assert pr11.effective_n(equal, 1.0) == pytest.approx(300)
    unequal = np.array([2] * 40 + [1] * 60)
    assert pr11.effective_n(unequal, 0.0) == pytest.approx(140)
    assert pr11.effective_n(unequal, 1.0) == pytest.approx(140**2 / (40 * 4 + 60))
    assert pr11.effective_n(unequal, 1.0) < 100  # fewer than the number of customers
    assert pr11.design_effect(np.ones(50), 0.9) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        pr11.design_effect([2, 0], 0.5)
    with pytest.raises(ValueError):
        pr11.design_effect([2, 1], 1.5)


@pytest.mark.parametrize("rho", [0.5, 1.0])
def test_kish_effective_n_matches_simulation_with_unequal_clusters(rho: float) -> None:
    """Exchangeable rows within customer: Var(row mean) = 1 / n_eff (unit variance rows)."""
    rng = np.random.default_rng(7)
    sizes = np.array([2] * 600 + [1] * 500)
    n, reps = int(sizes.sum()), 20_000
    w = sizes / n
    u = rng.standard_normal((reps, sizes.size))
    mean = math.sqrt(rho) * (u @ w) + math.sqrt(1 - rho) * rng.standard_normal(reps) / math.sqrt(n)
    n_eff_mc = 1.0 / float(np.var(mean, ddof=1))
    assert pr11.effective_n(sizes, rho) == pytest.approx(n_eff_mc, rel=4 * math.sqrt(2 / reps))


def test_coverage_label_is_the_exact_binomial_classification() -> None:
    reps, level = 1000, 0.95
    lo, hi = pr11.coverage_acceptance_region(level, reps)
    assert pr11.classify_coverage(round(lo * reps), reps, level) == "consistent with nominal"
    assert pr11.classify_coverage(round(lo * reps) - 1, reps, level) == "below nominal"
    assert pr11.classify_coverage(round(hi * reps) + 1, reps, level) == "above nominal"
    with pytest.raises(ValueError):
        pr11.classify_coverage(reps + 1, reps, level)


@pytest.mark.parametrize("level", [0.80, 0.95])
def test_coverage_acceptance_region_is_the_exact_binomial_region(level: float) -> None:
    reps = 200
    lo, hi = pr11.coverage_acceptance_region(level, reps)
    klo, khi = round(lo * reps), round(hi * reps)
    assert stats.binomtest(klo, reps, level).pvalue >= 0.05
    assert stats.binomtest(khi, reps, level).pvalue >= 0.05
    assert stats.binomtest(klo - 1, reps, level).pvalue < 0.05
    if khi < reps:
        assert stats.binomtest(khi + 1, reps, level).pvalue < 0.05
    assert lo < level < hi
    assert pr11.coverage_mc_se(level, reps) == pytest.approx(math.sqrt(level * (1 - level) / reps))


def _c(mid: str, mse: float, *, ok: float = 1.0, citl: float = 0.0, slope: float = 1.0
       ) -> CandidateResult:
    return CandidateResult(mid, ok, citl, slope, mse)


def test_no_qualified_model_is_reported_and_e2_is_never_selected() -> None:
    cands = [_c("B1", 10.0, citl=0.20), _c("E2", 9.0, slope=0.5), _c("E3", 8.0, ok=0.5)]
    sel = select_model(cands, {})
    assert sel.status == pp.NO_QUALIFIED_MODEL == "NO QUALIFIED MODEL"
    assert sel.selected is None
    assert sel.reference_fallback == "E2"
    assert sel.reference_label == pp.REFERENCE_FALLBACK_LABEL
    assert not any(g.qualified for g in sel.gates)


def test_parsimony_requires_a_significant_improvement() -> None:
    cands = [_c("B1", 10.0), _c("E2", 9.0), _c("E3", 8.5)]
    not_sig = {("E2", "B1"): (-2.0, 0.3), ("E3", "B1"): (-3.0, 0.1), ("B1", "E2"): (-0.3, 2.0),
               ("E3", "E2"): (-1.0, 0.2), ("B1", "E3"): (-0.1, 3.0), ("E2", "E3"): (-0.2, 1.0)}
    assert select_model(cands, not_sig).selected == "B1"
    sig = dict(not_sig)
    sig[("E2", "B1")] = (-2.0, -0.1)
    sel = select_model(cands, sig)
    assert sel.status == "SELECTED" and sel.selected == "E2"


def test_complexity_ties_go_to_lower_mse_and_inputs_are_checked() -> None:
    cands = [_c("E1", 9.5), _c("E2", 9.0)]
    pairs = {("E1", "E2"): (-0.1, 1.0), ("E2", "E1"): (-1.0, 0.1)}
    assert select_model(cands, pairs).selected == "E2"
    with pytest.raises(ValueError):
        select_model([_c("S1", 1.0)], {})
    with pytest.raises(ValueError):
        select_model([_c("E1", 1.0), _c("E1", 2.0)], {})
    with pytest.raises(KeyError):
        select_model(cands, {})


def test_gates_use_the_frozen_bounds() -> None:
    lo, hi = pp.SELECTION.slope_bounds
    tol = pp.SELECTION.calibration_in_the_large_tolerance
    assert gate(_c("E2", 1.0, slope=lo, citl=tol)).qualified
    assert gate(_c("E2", 1.0, slope=hi, citl=-tol)).qualified
    assert not gate(_c("E2", 1.0, slope=float("nan"))).qualified
    assert not gate(_c("E2", 1.0, ok=pp.SELECTION.min_ok_cell_fraction - 1e-9)).qualified


def test_fit_status_rules() -> None:
    assert classify_fit(FitDiagnostics()) == ("ok", ())
    assert classify_fit(FitDiagnostics(has_data=False))[0] == "undefined"
    assert classify_fit(FitDiagnostics(divergent_infinite_sum=True, converged=False))[0] == \
        "undefined"
    status, reasons = classify_fit(FitDiagnostics(converged=False, finite_likelihood=False))
    assert status == "failed" and len(reasons) == 2
    assert classify_fit(FitDiagnostics(separated_without_penalty=True))[0] == "failed"
    rules = pp.STATUS_RULES
    for diag in (FitDiagnostics(hessian_positive_definite=False),
                 FitDiagnostics(max_abs_log_parameter=rules.log_parameter_bound + 0.1),
                 FitDiagnostics(start_log_likelihood_gap=1e-3),
                 FitDiagnostics(n_events=rules.min_events - 1),
                 FitDiagnostics(n_events=100, n_parameters=11)):
        assert classify_fit(diag)[0] == "unstable"
    assert classify_fit(FitDiagnostics(n_events=100, n_parameters=10))[0] == "ok"
