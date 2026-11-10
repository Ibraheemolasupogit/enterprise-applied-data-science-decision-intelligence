"""Descriptive statistics with explicit conventions and numerically stable moments.

Conventions (stated because libraries disagree and silent mismatches cause
real reporting errors):

* ``variance``/``std`` use the unbiased ``n - 1`` denominator (ddof=1).
  ``std`` is *not* unbiased for sigma — only the variance is.
* ``skewness`` is the adjusted Fisher–Pearson coefficient
  ``G1 = sqrt(n(n-1))/(n-2) * m3 / m2^{3/2}`` (Excel/SAS/SPSS ``SKEW``;
  ``scipy.stats.skew(bias=False)``). Requires ``n >= 3``.
* ``excess_kurtosis`` is the bias-adjusted ``G2`` (normal = 0;
  ``scipy.stats.kurtosis(bias=False)``). Requires ``n >= 4``.
* Quantiles use linear interpolation between order statistics
  (Hyndman–Fan type 7, NumPy/R default).
* ``mad`` is the raw median absolute deviation; multiply by 1.4826 for a
  consistent estimator of sigma under normality (``mad_normal``).

Undefined quantities (variance of one observation, skewness of constant
data, coefficient of variation with zero mean) are returned as NaN rather
than an arbitrary number, so a downstream consumer cannot mistake "not
defined" for "zero".

Numerical stability: moments are computed by a two-pass algorithm (centre
first, then accumulate), and :class:`RunningMoments` implements Welford's
streaming update with Chan et al.'s parallel merge. The textbook one-pass
formula ``E[x^2] - E[x]^2`` suffers catastrophic cancellation when the mean
is large relative to the spread and is deliberately not used anywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from decision_intelligence.common.validation import FloatArray, as_float_array, require_probability

MAD_NORMAL_CONSISTENCY = 1.482602218505602
"""1 / Phi^-1(3/4): scales the MAD to estimate sigma for normal data."""


@dataclass(frozen=True)
class DescriptiveSummary:
    """Location, spread and shape of one numeric sample."""

    n: int
    n_missing: int
    mean: float
    variance: float
    std: float
    sem: float
    minimum: float
    q1: float
    median: float
    q3: float
    maximum: float
    iqr: float
    mad: float
    mad_normal: float
    skewness: float
    excess_kurtosis: float
    coefficient_of_variation: float


def describe(values: ArrayLike, *, drop_missing: bool = True) -> DescriptiveSummary:
    """Summarise a numeric sample.

    Args:
        values: One-dimensional numeric data.
        drop_missing: If ``True`` (the default) NaNs are removed and counted
            in ``n_missing`` — the caller must still judge whether the
            missingness is ignorable (see :mod:`.missing_data`). If
            ``False``, any NaN raises.

    Raises:
        ValueError: If no non-missing observations remain.
    """
    raw = as_float_array(values, name="values", min_size=1, allow_nan=drop_missing)
    missing = np.isnan(raw)
    x = raw[~missing]
    n = int(x.size)
    if n == 0:
        raise ValueError("values contains no non-missing observations")

    mean = float(np.mean(x))
    centred = x - mean
    m2 = float(np.mean(centred**2))
    variance = m2 * n / (n - 1) if n > 1 else math.nan
    std = math.sqrt(variance) if n > 1 else math.nan
    q1, median, q3 = (float(q) for q in np.quantile(x, [0.25, 0.5, 0.75]))
    mad = float(np.median(np.abs(x - median)))

    return DescriptiveSummary(
        n=n,
        n_missing=int(missing.sum()),
        mean=mean,
        variance=variance,
        std=std,
        sem=std / math.sqrt(n) if n > 1 else math.nan,
        minimum=float(np.min(x)),
        q1=q1,
        median=median,
        q3=q3,
        maximum=float(np.max(x)),
        iqr=q3 - q1,
        mad=mad,
        mad_normal=MAD_NORMAL_CONSISTENCY * mad,
        skewness=skewness(x) if n >= 3 else math.nan,
        excess_kurtosis=excess_kurtosis(x) if n >= 4 else math.nan,
        coefficient_of_variation=std / mean if n > 1 and mean != 0 else math.nan,
    )


def skewness(values: ArrayLike) -> float:
    """Adjusted Fisher–Pearson sample skewness ``G1`` (NaN for constant data)."""
    x = as_float_array(values, name="values", min_size=3)
    n = x.size
    centred = x - x.mean()
    m2 = float(np.mean(centred**2))
    if m2 == 0.0:
        return math.nan
    m3 = float(np.mean(centred**3))
    g1 = m3 / math.pow(m2, 1.5)
    return math.sqrt(n * (n - 1)) / (n - 2) * g1


def excess_kurtosis(values: ArrayLike) -> float:
    """Bias-adjusted sample excess kurtosis ``G2`` (NaN for constant data)."""
    x = as_float_array(values, name="values", min_size=4)
    n = x.size
    centred = x - x.mean()
    m2 = float(np.mean(centred**2))
    if m2 == 0.0:
        return math.nan
    m4 = float(np.mean(centred**4))
    g2 = m4 / m2**2 - 3.0
    return (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g2 + 6.0)


def trimmed_mean(values: ArrayLike, proportion_to_cut: float) -> float:
    """Mean after removing ``proportion_to_cut`` of observations from *each* tail.

    The number cut from each tail is ``floor(proportion_to_cut * n)``
    (matching ``scipy.stats.trim_mean``). A robust location estimate whose
    breakdown point equals ``proportion_to_cut``; it estimates the
    population trimmed mean, which equals the mean only for symmetric
    distributions.
    """
    x = as_float_array(values, name="values", min_size=1)
    if not 0.0 <= proportion_to_cut < 0.5:
        raise ValueError("proportion_to_cut must be in [0, 0.5)")
    n_cut = int(math.floor(proportion_to_cut * x.size))
    ordered = np.sort(x)
    kept = ordered[n_cut : x.size - n_cut]
    return float(np.mean(kept))


def quantile_interval(values: ArrayLike, coverage: float) -> tuple[float, float]:
    """Central empirical interval containing ``coverage`` of the observations.

    This describes the *spread of the data*, not uncertainty about a
    parameter — it must not be reported as a confidence interval.
    """
    x = as_float_array(values, name="values", min_size=2)
    coverage = require_probability(coverage, "coverage")
    tail = (1 - coverage) / 2
    lower, upper = np.quantile(x, [tail, 1 - tail])
    return float(lower), float(upper)


class RunningMoments:
    """Streaming mean/variance (Welford) with an exact parallel merge (Chan et al.).

    Suitable for data too large to hold in memory or arriving in batches.
    The update is numerically stable: it accumulates squared deviations
    from the running mean rather than raw squares.
    """

    def __init__(self) -> None:
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0

    @property
    def n(self) -> int:
        return self._n

    @property
    def mean(self) -> float:
        return self._mean if self._n > 0 else math.nan

    @property
    def variance(self) -> float:
        """Unbiased (ddof=1) variance; NaN with fewer than two observations."""
        return self._m2 / (self._n - 1) if self._n > 1 else math.nan

    def update(self, values: ArrayLike) -> RunningMoments:
        """Add observations one at a time; returns ``self`` for chaining."""
        for value in as_float_array(values, name="values", min_size=0):
            self._n += 1
            delta = value - self._mean
            self._mean += delta / self._n
            self._m2 += delta * (value - self._mean)
        return self

    def merge(self, other: RunningMoments) -> RunningMoments:
        """Return a new accumulator equal to having seen both streams."""
        merged = RunningMoments()
        n = self._n + other._n
        if n == 0:
            return merged
        delta = other._mean - self._mean
        merged._n = n
        merged._mean = self._mean + delta * other._n / n
        merged._m2 = self._m2 + other._m2 + delta**2 * self._n * other._n / n
        return merged


def standardise(values: ArrayLike) -> FloatArray:
    """Z-scores ``(x - mean) / std`` (ddof=1). Raises for constant data."""
    x = as_float_array(values, name="values", min_size=2)
    std = float(np.std(x, ddof=1))
    if std == 0.0:
        raise ValueError("cannot standardise constant data (standard deviation is zero)")
    return (x - x.mean()) / std
