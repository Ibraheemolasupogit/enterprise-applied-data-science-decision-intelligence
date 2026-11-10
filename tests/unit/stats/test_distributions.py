"""Sampling methods: distributional correctness, envelope checks, design-based estimation."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from decision_intelligence.common import seeded_generator
from decision_intelligence.statistics.distributions import (
    empirical_cdf,
    exponential_quantile,
    inverse_transform_sample,
    ks_distance,
    rejection_sample,
    simple_random_sample,
    stratified_mean,
    stratified_sample,
    weibull_mean,
    weibull_quantile,
    weibull_survival,
)


def test_inverse_transform_exponential_passes_ks_test() -> None:
    draws = inverse_transform_sample(exponential_quantile(0.5), 5000, seeded_generator(1))
    assert stats.kstest(draws, stats.expon(scale=2.0).cdf).pvalue > 0.01
    assert draws.mean() == pytest.approx(2.0, rel=0.05)


def test_inverse_transform_weibull_matches_m2_retention_survival() -> None:
    """The M2 retention generator solves scale so that S(90) = p exactly; check that
    inverse-transform draws reproduce that survival probability."""
    shape, p_target = 0.85, 0.6
    scale = 90 / (-math.log(p_target)) ** (1 / shape)
    assert float(weibull_survival([90.0], shape, scale)[0]) == pytest.approx(p_target)
    draws = inverse_transform_sample(weibull_quantile(shape, scale), 20000, seeded_generator(2))
    assert np.mean(draws > 90) == pytest.approx(p_target, abs=0.015)
    assert draws.mean() == pytest.approx(weibull_mean(shape, scale), rel=0.05)
    assert stats.kstest(draws, stats.weibull_min(shape, scale=scale).cdf).pvalue > 0.01


def test_inverse_transform_rejects_bad_quantile_function() -> None:
    with pytest.raises(ValueError, match="finite"), np.errstate(invalid="ignore"):
        inverse_transform_sample(lambda u: np.log(u - 1.0), 10, seeded_generator(1))


def _beta22_density(x: np.ndarray) -> np.ndarray:
    return np.where((x >= 0) & (x <= 1), 6 * x * (1 - x), 0.0)


def test_rejection_sampling_beta_target() -> None:
    """Beta(2,2) density peaks at 1.5, so M = 1.5 against a Uniform(0,1) proposal."""
    result = rejection_sample(
        _beta22_density,
        lambda rng, n: rng.random(n),
        lambda x: np.ones_like(x),
        1.5,
        5000,
        seeded_generator(3),
    )
    assert result.samples.size == 5000
    assert stats.kstest(result.samples, stats.beta(2, 2).cdf).pvalue > 0.01
    assert result.acceptance_rate == pytest.approx(result.theoretical_acceptance_rate, abs=0.02)


def test_rejection_sampling_detects_invalid_envelope() -> None:
    with pytest.raises(ValueError, match="envelope condition"):
        rejection_sample(
            _beta22_density,
            lambda rng, n: rng.random(n),
            lambda x: np.ones_like(x),
            1.2,
            100,
            seeded_generator(3),
        )
    with pytest.raises(ValueError, match=">= 1"):
        rejection_sample(
            _beta22_density,
            lambda rng, n: rng.random(n),
            np.ones_like,
            0.5,
            10,
            seeded_generator(1),
        )


def test_empirical_cdf_and_ks_distance_match_scipy() -> None:
    sample = seeded_generator(4).normal(size=300)
    cdf = empirical_cdf(sample)
    assert float(cdf(np.array([np.inf]))[0]) == 1.0
    assert float(cdf(np.array([-np.inf]))[0]) == 0.0
    assert ks_distance(sample, stats.norm.cdf) == pytest.approx(
        stats.kstest(sample, stats.norm.cdf).statistic
    )


def test_simple_random_sample_without_replacement() -> None:
    indices = simple_random_sample(100, 30, seeded_generator(5))
    assert indices.size == 30 and np.unique(indices).size == 30
    assert indices.min() >= 0 and indices.max() < 100
    with pytest.raises(ValueError, match="exceed"):
        simple_random_sample(10, 11, seeded_generator(5))


def test_stratified_sample_proportional_allocation_is_exact() -> None:
    strata = np.array(["a"] * 500 + ["b"] * 300 + ["c"] * 200)
    indices = stratified_sample(strata, 100, seeded_generator(6))
    chosen = strata[indices]
    assert indices.size == 100
    assert {s: int(np.sum(chosen == s)) for s in "abc"} == {"a": 50, "b": 30, "c": 20}
    with pytest.raises(ValueError, match="fewer than two"):
        stratified_sample(np.array(["a"] * 98 + ["b"] * 2), 10, seeded_generator(6))


def test_stratified_mean_is_unbiased_and_beats_srs_when_strata_explain_variance() -> None:
    rng = seeded_generator(7)
    strata = np.repeat(np.array(["low", "mid", "high"]), [600, 300, 100])
    population = np.concatenate(
        [rng.normal(10, 1, 600), rng.normal(20, 1, 300), rng.normal(50, 1, 100)]
    )
    sizes = {"low": 600, "mid": 300, "high": 100}
    stratified_estimates, srs_estimates, reported_se = [], [], []
    for _ in range(400):
        idx = stratified_sample(strata, 60, rng)
        estimate = stratified_mean(population[idx], strata[idx], sizes)
        stratified_estimates.append(estimate.mean)
        reported_se.append(estimate.standard_error)
        srs_estimates.append(population[simple_random_sample(1000, 60, rng)].mean())
    true_mean = population.mean()
    assert np.mean(stratified_estimates) == pytest.approx(true_mean, abs=0.1)
    assert np.std(stratified_estimates) < 0.25 * np.std(srs_estimates)
    assert np.mean(reported_se) == pytest.approx(np.std(stratified_estimates), rel=0.2)


def test_stratified_mean_validation() -> None:
    with pytest.raises(ValueError, match="at least two"):
        stratified_mean([1.0, 2.0, 3.0], ["a", "a", "b"], {"a": 10, "b": 10})
    with pytest.raises(ValueError, match="absent"):
        stratified_mean(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], ["a", "a", "b", "b", "z", "z"], {"a": 10, "b": 10}
        )
    with pytest.raises(ValueError, match="more sampled units"):
        stratified_mean([1.0, 2.0, 3.0], ["a", "a", "a"], {"a": 2})
