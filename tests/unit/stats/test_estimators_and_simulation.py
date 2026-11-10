"""Estimator properties, CLT diagnostics, exact intervals and the Monte Carlo engine."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from decision_intelligence.common import seeded_generator
from decision_intelligence.statistics.estimators import (
    clt_diagnostics,
    estimator_properties,
    exponential_rate_mle,
    sampling_distribution,
    t_interval_for_mean,
)
from decision_intelligence.statistics.simulation import (
    SimulatedRate,
    monte_carlo_mean,
    simulate_interval_coverage,
    simulate_rejection_rate,
)


def _normal_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.normal(0.0, 2.0, n)


def test_variance_estimators_bias_matches_theory() -> None:
    """ddof=0 has bias -sigma^2/n; ddof=1 is unbiased; ddof=0 still has lower MSE at n=10."""
    n, sigma2 = 10, 4.0
    biased = estimator_properties(
        _normal_sampler, lambda x: float(np.var(x)), sigma2, n, 20000, seeded_generator(1)
    )
    unbiased = estimator_properties(
        _normal_sampler, lambda x: float(np.var(x, ddof=1)), sigma2, n, 20000, seeded_generator(1)
    )
    assert biased.bias == pytest.approx(-sigma2 / n, abs=4 * biased.bias_standard_error)
    assert abs(unbiased.bias) < 4 * unbiased.bias_standard_error
    assert biased.mse < unbiased.mse  # the bias–variance trade-off at the estimator level
    assert biased.mse == pytest.approx(biased.bias**2 + biased.variance, rel=0.01)


def test_sampling_distribution_of_mean_has_standard_error_sigma_over_root_n() -> None:
    means = sampling_distribution(
        _normal_sampler, lambda x: float(x.mean()), 25, 5000, seeded_generator(2)
    )
    assert means.std(ddof=1) == pytest.approx(2.0 / 5.0, rel=0.05)


def test_sampling_distribution_rejects_non_finite_estimates() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        sampling_distribution(_normal_sampler, lambda x: math.inf, 5, 10, seeded_generator(1))


def test_clt_diagnostics_track_theoretical_skewness_decay() -> None:
    """Exponential population: skewness of the mean is 2/sqrt(n); KS distance shrinks with n."""
    rows = clt_diagnostics(
        lambda rng, n: rng.exponential(1.0, n),
        [2, 10, 100],
        population_mean=1.0,
        population_sd=1.0,
        n_replications=8000,
        rng=seeded_generator(3),
        population_skewness=2.0,
    )
    for row in rows:
        assert row.theoretical_skewness_of_means is not None
        assert row.skewness_of_means == pytest.approx(row.theoretical_skewness_of_means, abs=0.2)
        assert row.sd_of_means == pytest.approx(row.theoretical_standard_error, rel=0.05)
    distances = [row.ks_distance_to_normal for row in rows]
    assert distances[0] > distances[1] > distances[2]


def test_clt_diagnostics_reject_infinite_variance() -> None:
    with pytest.raises(ValueError, match="finite"):
        clt_diagnostics(_normal_sampler, [5], 0.0, math.inf, 10, seeded_generator(1))


def test_exponential_rate_mle_exact_interval_matches_chi_square_pivot() -> None:
    data = seeded_generator(4).exponential(1 / 3.0, 25)
    estimate = exponential_rate_mle(data)
    assert estimate.rate == pytest.approx(1 / data.mean())
    total = data.sum()
    assert estimate.ci_lower == pytest.approx(stats.chi2.ppf(0.025, 50) / (2 * total))
    assert estimate.ci_upper == pytest.approx(stats.chi2.ppf(0.975, 50) / (2 * total))
    with pytest.raises(ValueError, match="positive"):
        exponential_rate_mle([1.0, 0.0])


def test_t_interval_matches_scipy_and_rejects_constant_data() -> None:
    data = seeded_generator(5).normal(10, 3, 15)
    reference = stats.t.interval(0.9, 14, loc=data.mean(), scale=stats.sem(data))
    assert t_interval_for_mean(data, confidence_level=0.9) == pytest.approx(reference)
    with pytest.raises(ValueError, match="constant"):
        t_interval_for_mean([1.0, 1.0, 1.0])


def test_monte_carlo_mean_reports_simulation_error() -> None:
    result = monte_carlo_mean(lambda rng: float(rng.normal() ** 2), 20000, seeded_generator(6))
    assert result.ci_lower <= 1.0 <= result.ci_upper  # E[Z^2] = 1
    assert result.standard_error == pytest.approx(math.sqrt(2 / 20000), rel=0.1)


def test_monte_carlo_is_reproducible_with_same_seed() -> None:
    def draw(rng: np.random.Generator) -> float:
        return float(rng.random())

    first = monte_carlo_mean(draw, 100, seeded_generator(9))
    second = monte_carlo_mean(draw, 100, seeded_generator(9))
    np.testing.assert_array_equal(first.draws, second.draws)


def test_simulate_rejection_rate_and_coverage_wiring() -> None:
    def generate(rng: np.random.Generator) -> tuple[np.ndarray, ...]:
        return (rng.normal(size=20),)

    rate = simulate_rejection_rate(
        generate,
        lambda x: float(stats.ttest_1samp(x, 0.0).pvalue),
        alpha=0.05,
        n_replications=2000,
        rng=seeded_generator(7),
    )
    assert rate.is_consistent_with(0.05)
    coverage = simulate_interval_coverage(
        generate, t_interval_for_mean, 0.0, n_replications=2000, rng=seeded_generator(8)
    )
    assert coverage.is_consistent_with(0.95)


def test_simulation_rejects_invalid_callbacks() -> None:
    with pytest.raises(ValueError, match="invalid p-value"):
        simulate_rejection_rate(
            lambda rng: (rng.normal(size=3),),
            lambda x: 1.5,
            alpha=0.05,
            n_replications=5,
            rng=seeded_generator(1),
        )
    with pytest.raises(ValueError, match="lower > upper"):
        simulate_interval_coverage(
            lambda rng: (rng.normal(size=3),),
            lambda x: (1.0, 0.0),
            0.0,
            n_replications=5,
            rng=seeded_generator(1),
        )


def test_simulated_rate_consistency_check_handles_boundaries() -> None:
    assert SimulatedRate(rate=0.0, standard_error=0.0, n_replications=1000).is_consistent_with(
        0.001
    )
    assert not SimulatedRate(rate=0.2, standard_error=0.01, n_replications=1000).is_consistent_with(
        0.05
    )
