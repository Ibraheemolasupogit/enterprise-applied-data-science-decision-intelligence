"""Descriptive statistics: reference agreement, conventions, numerical stability, edge cases."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from decision_intelligence.common import seeded_generator
from decision_intelligence.statistics.descriptive import (
    RunningMoments,
    describe,
    excess_kurtosis,
    quantile_interval,
    skewness,
    standardise,
    trimmed_mean,
)


@pytest.fixture
def sample() -> np.ndarray:
    return seeded_generator(3).gamma(2.0, 3.0, 250)


def test_describe_matches_numpy_and_scipy(sample: np.ndarray) -> None:
    summary = describe(sample)
    assert summary.n == 250 and summary.n_missing == 0
    assert summary.mean == pytest.approx(np.mean(sample))
    assert summary.variance == pytest.approx(np.var(sample, ddof=1))
    assert summary.sem == pytest.approx(stats.sem(sample))
    assert summary.median == pytest.approx(np.median(sample))
    assert summary.iqr == pytest.approx(stats.iqr(sample))
    assert summary.mad == pytest.approx(stats.median_abs_deviation(sample))
    assert summary.mad_normal == pytest.approx(
        stats.median_abs_deviation(sample, scale="normal")
    )
    assert summary.skewness == pytest.approx(stats.skew(sample, bias=False))
    assert summary.excess_kurtosis == pytest.approx(stats.kurtosis(sample, bias=False))


def test_skewness_sign_and_kurtosis_of_known_shapes() -> None:
    rng = seeded_generator(4)
    assert skewness(rng.exponential(size=20000)) == pytest.approx(2.0, abs=0.15)
    assert excess_kurtosis(rng.normal(size=20000)) == pytest.approx(0.0, abs=0.15)
    assert skewness(-rng.exponential(size=5000)) < 0


def test_describe_counts_and_drops_missing() -> None:
    summary = describe([1.0, 2.0, math.nan, 4.0])
    assert summary.n == 3 and summary.n_missing == 1
    assert summary.mean == pytest.approx(7 / 3)
    with pytest.raises(ValueError, match="NaN"):
        describe([1.0, math.nan], drop_missing=False)


def test_describe_returns_nan_for_undefined_quantities() -> None:
    single = describe([5.0])
    assert math.isnan(single.variance) and math.isnan(single.sem)
    assert math.isnan(single.skewness) and math.isnan(single.excess_kurtosis)
    constant = describe([2.0, 2.0, 2.0, 2.0])
    assert constant.variance == 0.0
    assert math.isnan(constant.skewness)
    zero_mean = describe([-1.0, 1.0])
    assert math.isnan(zero_mean.coefficient_of_variation)


def test_describe_rejects_all_missing() -> None:
    with pytest.raises(ValueError, match="no non-missing"):
        describe([math.nan, math.nan])


def test_variance_is_stable_under_a_large_offset() -> None:
    """A 1e9 offset destroys the one-pass E[x^2]-E[x]^2 formula but not two-pass/Welford."""
    rng = seeded_generator(5)
    base = rng.normal(0.0, 1.0, 10_000)
    shifted = base + 1e9
    exact = float(np.var(base, ddof=1))
    naive = (np.sum(shifted**2) - shifted.size * np.mean(shifted) ** 2) / (shifted.size - 1)
    assert abs(naive - exact) / exact > 0.01  # the textbook formula really does fail here
    assert describe(shifted).variance == pytest.approx(exact, rel=1e-6)
    assert RunningMoments().update(shifted).variance == pytest.approx(exact, rel=1e-6)


def test_running_moments_merge_equals_single_pass(sample: np.ndarray) -> None:
    left = RunningMoments().update(sample[:97])
    right = RunningMoments().update(sample[97:])
    merged = left.merge(right)
    assert merged.n == sample.size
    assert merged.mean == pytest.approx(sample.mean())
    assert merged.variance == pytest.approx(sample.var(ddof=1))
    empty = RunningMoments()
    assert math.isnan(empty.mean) and math.isnan(empty.variance)
    assert empty.merge(RunningMoments()).n == 0


def test_trimmed_mean_matches_scipy_and_resists_outliers(sample: np.ndarray) -> None:
    assert trimmed_mean(sample, 0.1) == pytest.approx(stats.trim_mean(sample, 0.1))
    contaminated = np.append(sample, 1e6)
    assert trimmed_mean(contaminated, 0.1) < 2 * trimmed_mean(sample, 0.1)
    with pytest.raises(ValueError):
        trimmed_mean(sample, 0.5)


def test_quantile_interval_and_standardise() -> None:
    lower, upper = quantile_interval(np.arange(101.0), 0.9)
    assert (lower, upper) == pytest.approx((5.0, 95.0))
    z = standardise([1.0, 2.0, 3.0])
    assert z.mean() == pytest.approx(0.0) and z.std(ddof=1) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="constant"):
        standardise([1.0, 1.0])
