"""Bootstrap CI helper: agreement with SciPy, reproducibility and failure handling."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from decision_intelligence.common import bootstrap_ci, bootstrap_two_sample_ci, seeded_generator
from decision_intelligence.common.bootstrap import bca_acceleration


@pytest.fixture
def skewed_sample() -> np.ndarray:
    return seeded_generator(11).exponential(2.0, 40)


@pytest.mark.parametrize(
    ("method", "scipy_method"), [("percentile", "percentile"), ("basic", "basic"), ("bca", "BCa")]
)
def test_one_sample_matches_scipy_reference(
    skewed_sample: np.ndarray, method: str, scipy_method: str
) -> None:
    """With the same seed both implementations draw identical resample indices,
    so the intervals agree to floating-point precision."""
    ours = bootstrap_ci(
        skewed_sample, np.mean, rng=seeded_generator(1), method=method, n_resamples=5000
    )  # type: ignore[arg-type]
    reference = stats.bootstrap(
        (skewed_sample,),
        np.mean,
        method=scipy_method,
        n_resamples=5000,
        rng=np.random.default_rng(1),
    ).confidence_interval
    assert ours.ci_lower == pytest.approx(reference.low, rel=1e-10)
    assert ours.ci_upper == pytest.approx(reference.high, rel=1e-10)


def test_two_sample_bca_close_to_scipy(skewed_sample: np.ndarray) -> None:
    other = seeded_generator(12).exponential(1.0, 30)

    def difference(a: np.ndarray, b: np.ndarray) -> float:
        return float(a.mean() - b.mean())

    ours = bootstrap_two_sample_ci(
        skewed_sample, other, difference, rng=seeded_generator(1), n_resamples=20000
    )
    reference = stats.bootstrap(
        (skewed_sample, other),
        lambda a, b, axis=-1: a.mean(axis) - b.mean(axis),
        method="BCa",
        n_resamples=20000,
        rng=np.random.default_rng(1),
    ).confidence_interval
    # Different resampling streams: agreement only up to Monte Carlo error.
    assert ours.ci_lower == pytest.approx(reference.low, abs=0.05)
    assert ours.ci_upper == pytest.approx(reference.high, abs=0.05)
    assert ours.estimate == pytest.approx(difference(skewed_sample, other))


def test_same_seed_reproduces_interval_exactly(skewed_sample: np.ndarray) -> None:
    first = bootstrap_ci(skewed_sample, np.median, rng=seeded_generator(5), method="percentile")
    second = bootstrap_ci(skewed_sample, np.median, rng=seeded_generator(5), method="percentile")
    assert (first.ci_lower, first.ci_upper) == (second.ci_lower, second.ci_upper)
    np.testing.assert_array_equal(first.replicates, second.replicates)


def test_result_reports_standard_error_and_bias(skewed_sample: np.ndarray) -> None:
    result = bootstrap_ci(skewed_sample, np.mean, rng=seeded_generator(2), n_resamples=4000)
    analytic_se = skewed_sample.std(ddof=0) / np.sqrt(skewed_sample.size)
    assert result.standard_error == pytest.approx(analytic_se, rel=0.1)
    assert abs(result.bias) < 0.1 * result.standard_error + 0.05
    assert result.ci_lower < result.estimate < result.ci_upper


def test_constant_data_gives_degenerate_interval_not_an_error() -> None:
    result = bootstrap_ci([3.0, 3.0, 3.0, 3.0], np.mean, rng=seeded_generator(1), n_resamples=200)
    assert result.ci_lower == result.ci_upper == 3.0
    assert result.standard_error == 0.0


def test_bca_raises_when_bias_correction_is_infinite() -> None:
    # Number of distinct values: 20 in the sample, (almost surely) fewer in every resample,
    # so every replicate lies below the estimate and z0 = Phi^-1(1) is infinite.
    data = np.arange(1.0, 21.0)
    with pytest.raises(ValueError, match="BCa interval undefined"):
        bootstrap_ci(
            data, lambda x: float(np.unique(x).size), rng=seeded_generator(1), n_resamples=200
        )


def test_non_finite_statistic_on_a_resample_raises() -> None:
    data = np.array([0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="non-finite"):
        bootstrap_ci(
            data,
            lambda x: float(1.0 / x.sum()) if x.sum() else float("inf"),
            rng=seeded_generator(1),
            n_resamples=500,
            method="percentile",
        )


@pytest.mark.parametrize(
    "kwargs",
    [{"method": "studentized"}, {"n_resamples": 50}, {"confidence_level": 1.0}],
)
def test_invalid_arguments_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, 2.0, 3.0], np.mean, rng=seeded_generator(1), **kwargs)  # type: ignore[arg-type]


def test_requires_at_least_two_observations() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        bootstrap_ci([1.0], np.mean, rng=seeded_generator(1))


def _standard_multi_sample_acceleration(samples, statistic) -> float:
    """Independent statement of the Efron–Tibshirani multi-sample acceleration
    (per-sample jackknife centring, influence U = (n - 1)(mean - T_(i)))."""
    numerator = denominator = 0.0
    for j, sample in enumerate(samples):
        values = []
        for i in range(sample.size):
            reduced = list(samples)
            reduced[j] = np.delete(sample, i)
            values.append(statistic(*reduced))
        jack = np.asarray(values)
        n = sample.size
        u = (n - 1) * (jack.mean() - jack)
        numerator += np.sum(u**3) / n**3
        denominator += np.sum(u**2) / n**2
    return float(numerator / (6 * denominator**1.5))


def test_two_sample_bca_acceleration_uses_per_sample_influence() -> None:
    """Regression test: acceleration was once computed from jackknife values pooled
    around a single grand mean, which is wrong for unequal sizes / non-linear statistics."""
    rng = seeded_generator(3)
    x, y = rng.exponential(2.0, 10), rng.exponential(1.0, 30)

    def ratio(a: np.ndarray, b: np.ndarray) -> float:
        return float(a.mean() / b.mean())

    ours = bca_acceleration((x, y), lambda arrays: ratio(arrays[0], arrays[1]))
    assert ours == pytest.approx(_standard_multi_sample_acceleration((x, y), ratio), rel=1e-12)


def test_one_sample_acceleration_reduces_to_classic_formula(skewed_sample: np.ndarray) -> None:
    jack = np.array([np.delete(skewed_sample, i).mean() for i in range(skewed_sample.size)])
    d = jack.mean() - jack
    classic = np.sum(d**3) / (6 * np.sum(d**2) ** 1.5)
    ours = bca_acceleration((skewed_sample,), lambda arrays: float(arrays[0].mean()))
    assert ours == pytest.approx(classic, rel=1e-12)
