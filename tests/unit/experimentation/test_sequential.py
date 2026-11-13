"""Group-sequential boundaries and always-valid inference against published values and
independent numerical references (SciPy multivariate normal, quadrature, simulation)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate, stats

from decision_intelligence.experimentation import sequential as sq

EQUAL5 = np.arange(1, 6) / 5


def _mvn_rejection(boundaries: np.ndarray, fractions: np.ndarray, drift: float) -> float:
    cov = np.sqrt(np.minimum.outer(fractions, fractions) / np.maximum.outer(fractions, fractions))
    inside = stats.multivariate_normal.cdf(
        boundaries, mean=drift * np.sqrt(fractions), cov=cov, lower_limit=-boundaries,
        abseps=1e-9, releps=1e-9, maxpts=5_000_000,
    )
    return 1.0 - float(inside)


@pytest.mark.parametrize("family", ["obrien-fleming", "pocock", "power"])
def test_spending_functions_spend_alpha_monotonically(family: str) -> None:
    grid = np.linspace(0, 1, 21)
    spent = [sq.spending_function(float(t), family=family, rho=3.0) for t in grid]  # type: ignore[arg-type]
    assert spent[0] == 0.0 and spent[-1] == pytest.approx(0.05, abs=1e-12)
    assert np.all(np.diff(spent) >= 0)


def test_spending_function_validation() -> None:
    with pytest.raises(ValueError):
        sq.spending_function(1.2)
    with pytest.raises(ValueError):
        sq.spending_function(0.5, family="unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        sq.spending_function(0.5, family="power", rho=0.0)


@pytest.mark.parametrize(
    ("k", "pocock", "obf"),
    [(2, 2.178, 1.977), (3, 2.289, 2.004), (4, 2.361, 2.024), (5, 2.413, 2.040),
     (10, 2.555, 2.087)],
)
def test_classical_constants_match_jennison_turnbull(k: int, pocock: float, obf: float) -> None:
    """Jennison & Turnbull (2000) Table 2.3, two-sided alpha = 0.05 (published to 3 dp)."""
    t = np.arange(1, k + 1) / k
    assert sq.constant_boundaries(t, shape="pocock").boundaries[0] == pytest.approx(pocock,
                                                                                    abs=6e-4)
    assert sq.constant_boundaries(t, shape="obrien-fleming").boundaries[-1] == pytest.approx(
        obf, abs=6e-4)


def test_lan_demets_obrien_fleming_matches_software_convention() -> None:
    """gsDesign / rpact / ldbounds, five equal looks, two-sided alpha = 0.05."""
    design = sq.spending_boundaries(EQUAL5)
    np.testing.assert_allclose(design.boundaries, [4.8769, 3.3569, 2.6803, 2.2898, 2.0310],
                               atol=2e-4)


@pytest.mark.parametrize("family", ["obrien-fleming", "pocock", "power"])
def test_spending_boundaries_spend_exactly_the_planned_alpha(family: str) -> None:
    fractions = np.array([0.15, 0.4, 0.55, 0.8, 1.0])  # unequal spacing
    design = sq.spending_boundaries(fractions, family=family, rho=2.0)  # type: ignore[arg-type]
    planned = [sq.spending_function(float(t), family=family, rho=2.0)  # type: ignore[arg-type]
               for t in fractions]
    np.testing.assert_allclose(design.cumulative_alpha, planned, atol=1e-9)


@pytest.mark.parametrize("drift", [0.0, 1.5, sq.fixed_sample_drift()])
def test_recursive_integration_matches_multivariate_normal(drift: float) -> None:
    fractions = np.array([0.2, 0.45, 0.7, 1.0])
    design = sq.spending_boundaries(fractions, family="pocock")
    ours = sq.operating_characteristics(design, drift).rejection_probability
    assert ours == pytest.approx(_mvn_rejection(np.asarray(design.boundaries), fractions, drift),
                                 abs=1e-6)


def test_recursive_integration_matches_simulation() -> None:
    design = sq.spending_boundaries(EQUAL5)
    rng = np.random.default_rng(3)
    n = 200_000
    drift = 2.0
    increments = rng.normal(drift * np.diff(np.r_[0, EQUAL5]), np.sqrt(np.diff(np.r_[0, EQUAL5])),
                            size=(n, 5))
    z = np.cumsum(increments, axis=1) / np.sqrt(EQUAL5)
    simulated = float((np.abs(z) >= np.asarray(design.boundaries)).any(axis=1).mean())
    exact = sq.operating_characteristics(design, drift).rejection_probability
    assert abs(simulated - exact) < 4 * math.sqrt(exact * (1 - exact) / n)


@pytest.mark.parametrize(("k", "published"),
                         [(2, 0.083), (3, 0.107), (5, 0.142), (10, 0.193), (20, 0.248)])
def test_naive_peeking_reproduces_armitage_mcpherson_rowe(k: int, published: float) -> None:
    design = sq.naive_peeking_design(np.arange(1, k + 1) / k)
    assert design.cumulative_alpha[-1] == pytest.approx(published, abs=6e-4)


@pytest.mark.parametrize(("shape", "published"), [("pocock", 1.229), ("obrien-fleming", 1.028)])
def test_maximum_sample_inflation_matches_published(shape: str, published: float) -> None:
    """Jennison & Turnbull (2000) Tables 2.2 and 2.4: K = 5, alpha = 0.05, power 0.8."""
    design = sq.constant_boundaries(EQUAL5, shape=shape)  # type: ignore[arg-type]
    assert sq.maximum_information_inflation(design) == pytest.approx(published, abs=1.5e-3)


def test_expected_sample_size_and_stopping_probabilities_are_consistent() -> None:
    design = sq.spending_boundaries(EQUAL5)
    null = sq.operating_characteristics(design, 0.0)
    assert sum(null.stopping_probabilities) == pytest.approx(0.05, abs=1e-9)
    stop, t = np.asarray(null.stopping_probabilities), EQUAL5
    expected = float(np.sum(stop[:-1] * t[:-1]) + (1 - stop[:-1].sum()))
    assert null.expected_information_fraction == pytest.approx(expected, abs=1e-12)
    assert 0.99 < null.expected_information_fraction < 1.0  # early stops are rare under H0
    strong = sq.operating_characteristics(design, 2 * sq.fixed_sample_drift())
    assert strong.expected_information_fraction < 0.6 and strong.rejection_probability > 0.99


def test_monitor_stops_at_first_crossing_and_reports_repeated_ci() -> None:
    result = sq.monitor([0.5, 1.6, 2.5], [0.4, 0.3, 0.2], [0.3, 0.6, 1.0])
    z = [0.5 / 0.4, 1.6 / 0.3]
    assert result.design.boundaries[0] > z[0] and z[1] >= result.design.boundaries[1]
    assert result.stopped_at == 2 and len(result.looks) == 2
    assert result.decision == "reject H0"
    look = result.looks[1]
    assert look.repeated_ci == pytest.approx((1.6 - look.boundary * 0.3, 1.6 + look.boundary * 0.3))
    assert any("biased" in w for w in result.warnings)


def test_monitor_without_crossing_and_partial_information() -> None:
    result = sq.monitor([0.1, 0.2], [1.0, 0.8], [0.3, 0.6])
    # Review fix: before the final look the only valid outcome is to continue; "do not reject
    # H0" is a terminal decision.
    assert result.stopped_at is None and result.decision == "continue"
    assert any("maximum information" in w for w in result.warnings)
    complete = sq.monitor([0.1, 0.2, 0.3], [1.0, 0.8, 0.6], [0.3, 0.6, 1.0])
    assert complete.stopped_at is None and complete.decision == "do not reject H0"
    with pytest.raises(ValueError):
        sq.monitor([0.1], [0.0], [1.0])
    with pytest.raises(ValueError):
        sq.monitor([0.1, 0.2], [1.0], [0.5, 1.0])


def test_fraction_validation() -> None:
    for bad in ([0.5, 0.4, 1.0], [0.0, 1.0], [0.5, 0.9]):
        with pytest.raises(ValueError):
            sq.spending_boundaries(bad)


def test_msprt_matches_numerical_mixture_likelihood_ratio() -> None:
    """Lambda = int N(est; theta, V) N(theta; 0, tau^2) dtheta / N(est; 0, V) by quadrature."""
    for est, var, tau in ((0.3, 0.04, 0.2), (-0.05, 0.001, 0.05), (1.2, 0.5, 1.0)):
        numerator, _ = integrate.quad(
            lambda th, est=est, var=var, tau=tau: stats.norm.pdf(est, th, math.sqrt(var))
            * stats.norm.pdf(th, 0.0, tau), -10, 10, points=[est, 0.0], limit=200)
        reference = math.log(numerator / stats.norm.pdf(est, 0.0, math.sqrt(var)))
        ours = float(sq.msprt_log_likelihood_ratio(est, var, mixture_sd=tau))
        assert ours == pytest.approx(reference, abs=1e-7)


def test_always_valid_p_values_are_monotone_and_confidence_sequence_inverts_the_test() -> None:
    p = sq.always_valid_p_values([0.1, 0.4, 0.2, 0.6], [0.04, 0.02, 0.013, 0.01], mixture_sd=0.3)
    assert np.all(np.diff(p) <= 0) and np.all((p > 0) & (p <= 1))
    lo, hi = sq.confidence_sequence(0.6, 0.01, mixture_sd=0.3, alpha=0.05)
    for edge in (lo, hi):
        log_lr = float(sq.msprt_log_likelihood_ratio(0.6, 0.01, mixture_sd=0.3, null=edge))
        assert log_lr == pytest.approx(math.log(20), abs=1e-9)


def test_msprt_controls_type_one_error_under_continuous_monitoring() -> None:
    rng = np.random.default_rng(5)
    reps, n = 1_500, 2_000
    paths = np.cumsum(rng.standard_normal((reps, n)), axis=1)
    m = np.arange(1, n + 1)
    log_lr = sq.msprt_log_likelihood_ratio(paths / m, 1.0 / m, mixture_sd=0.2)
    rate = float((log_lr.max(axis=1) >= math.log(20)).mean())
    naive = float((np.abs(paths / np.sqrt(m)) >= 1.96).any(axis=1).mean())
    assert rate <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / reps)
    assert naive > 0.3  # peeking at every observation with a fixed-sample test


def test_monitor_refuses_non_finite_estimates() -> None:
    # Review fix: a NaN estimate gave Z = NaN, never "crossed", and was reported as a decision.
    with pytest.raises(ValueError, match="finite"):
        sq.monitor([float("nan")], [0.1], [1.0])
    with pytest.raises(ValueError, match="finite"):
        sq.monitor([0.1, float("inf")], [0.1, 0.1], [0.5, 1.0])
