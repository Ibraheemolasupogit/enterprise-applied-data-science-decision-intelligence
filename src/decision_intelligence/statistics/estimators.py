"""Estimators, sampling distributions and the Central Limit Theorem.

An estimator is a function of a random sample, so it has a distribution —
its *sampling distribution* — over hypothetical repeated samples. Its
quality is a property of that distribution, not of any one realised value:

* ``bias = E[theta_hat] - theta``
* ``variance = Var(theta_hat)``
* ``MSE = E[(theta_hat - theta)^2] = bias^2 + variance``

:func:`estimator_properties` measures these by simulation against a known
``theta``, which is how this repository checks estimators before they are
trusted on data where ``theta`` is unknown.

Central Limit Theorem: for i.i.d. draws with mean ``mu`` and finite
variance ``sigma^2``, ``sqrt(n) (xbar - mu) / sigma`` converges in
distribution to N(0, 1). It says nothing about *how fast*: the skewness of
the sample mean is ``gamma / sqrt(n)`` (``gamma`` = population skewness), so
strongly skewed populations need much larger ``n`` before normal-theory
intervals are accurate, and for infinite-variance populations (e.g.
Pareto with tail index <= 2, Cauchy) the classical CLT does not apply at
all. :func:`clt_diagnostics` makes that convergence rate visible.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from decision_intelligence.common.validation import (
    FloatArray,
    as_float_array,
    require_positive_int,
    require_probability,
)
from decision_intelligence.statistics.descriptive import skewness
from decision_intelligence.statistics.distributions import ks_distance

Sampler = Callable[[np.random.Generator, int], FloatArray]
Estimator = Callable[[FloatArray], float]


def sampling_distribution(
    sampler: Sampler,
    estimator: Estimator,
    sample_size: int,
    n_replications: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Draw ``n_replications`` values of ``estimator`` on fresh samples of ``sample_size``."""
    sample_size = require_positive_int(sample_size, "sample_size")
    n_replications = require_positive_int(n_replications, "n_replications", minimum=2)
    values = np.empty(n_replications, dtype=np.float64)
    for r in range(n_replications):
        values[r] = float(estimator(np.asarray(sampler(rng, sample_size), dtype=np.float64)))
    if not np.isfinite(values).all():
        raise ValueError("estimator returned a non-finite value on a simulated sample")
    return values


@dataclass(frozen=True)
class EstimatorProperties:
    """Simulated bias, variance and MSE of an estimator at one sample size.

    ``bias_standard_error`` is the Monte Carlo standard error of ``bias``:
    a bias within ~2 of these of zero is indistinguishable from unbiased at
    this simulation size.
    """

    sample_size: int
    true_value: float
    mean_estimate: float
    bias: float
    bias_standard_error: float
    variance: float
    mse: float
    estimates: FloatArray = field(repr=False)


def estimator_properties(
    sampler: Sampler,
    estimator: Estimator,
    true_value: float,
    sample_size: int,
    n_replications: int,
    rng: np.random.Generator,
) -> EstimatorProperties:
    """Monte Carlo bias/variance/MSE of ``estimator`` for a known ``true_value``."""
    values = sampling_distribution(sampler, estimator, sample_size, n_replications, rng)
    bias = float(values.mean() - true_value)
    variance = float(values.var(ddof=1))
    return EstimatorProperties(
        sample_size=sample_size,
        true_value=true_value,
        mean_estimate=float(values.mean()),
        bias=bias,
        bias_standard_error=math.sqrt(variance / values.size),
        variance=variance,
        mse=float(np.mean((values - true_value) ** 2)),
        estimates=values,
    )


def _sample_mean(sample: FloatArray) -> float:
    return float(np.mean(sample))


@dataclass(frozen=True)
class CLTDiagnostic:
    """How close the sampling distribution of the mean is to normal at one ``n``."""

    sample_size: int
    mean_of_means: float
    sd_of_means: float
    theoretical_standard_error: float
    skewness_of_means: float
    theoretical_skewness_of_means: float | None
    ks_distance_to_normal: float


def clt_diagnostics(
    sampler: Sampler,
    sample_sizes: Sequence[int],
    population_mean: float,
    population_sd: float,
    n_replications: int,
    rng: np.random.Generator,
    *,
    population_skewness: float | None = None,
) -> list[CLTDiagnostic]:
    """Measure convergence of the standardised sample mean to N(0, 1).

    For each ``n`` the KS distance between the simulated distribution of
    ``(xbar - mu) / (sigma / sqrt(n))`` and the standard normal is
    reported, together with the skewness of the sample means (theory:
    ``population_skewness / sqrt(n)`` when supplied). Requires finite
    ``population_sd`` — the diagnostic is meaningless otherwise.
    """
    if not population_sd > 0 or not math.isfinite(population_sd):
        raise ValueError("population_sd must be positive and finite")
    results: list[CLTDiagnostic] = []
    for n in sample_sizes:
        means = sampling_distribution(sampler, _sample_mean, n, n_replications, rng)
        standard_error = population_sd / math.sqrt(n)
        standardised = (means - population_mean) / standard_error
        results.append(
            CLTDiagnostic(
                sample_size=n,
                mean_of_means=float(means.mean()),
                sd_of_means=float(means.std(ddof=1)),
                theoretical_standard_error=standard_error,
                skewness_of_means=skewness(means),
                theoretical_skewness_of_means=(
                    None if population_skewness is None else population_skewness / math.sqrt(n)
                ),
                ks_distance_to_normal=ks_distance(standardised, stats.norm.cdf),
            )
        )
    return results


@dataclass(frozen=True)
class RateEstimate:
    """Maximum-likelihood rate estimate with an exact confidence interval."""

    rate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float


def exponential_rate_mle(values: ArrayLike, *, confidence_level: float = 0.95) -> RateEstimate:
    """MLE ``lambda_hat = 1 / xbar`` of an exponential rate with an exact CI.

    Uses the pivot ``2 lambda sum(x) ~ chi^2_{2n}``, exact for any ``n`` when
    the data really are exponential. The MLE is biased upwards in small
    samples (``E[lambda_hat] = n lambda / (n - 1)``); ``(n - 1) / sum(x)`` is the
    unbiased alternative. Assumes i.i.d., uncensored, strictly positive
    observations — censored durations (customers still active) need
    survival methods instead (M11).
    """
    x = as_float_array(values, name="values", min_size=1)
    confidence_level = require_probability(confidence_level, "confidence_level")
    if (x <= 0).any():
        raise ValueError("exponential data must be strictly positive")
    n = x.size
    total = float(x.sum())
    alpha = 1 - confidence_level
    return RateEstimate(
        rate=n / total,
        ci_lower=float(stats.chi2.ppf(alpha / 2, 2 * n)) / (2 * total),
        ci_upper=float(stats.chi2.ppf(1 - alpha / 2, 2 * n)) / (2 * total),
        confidence_level=confidence_level,
    )


def t_interval_for_mean(
    values: ArrayLike, *, confidence_level: float = 0.95
) -> tuple[float, float]:
    """Student-t interval ``xbar +- t_{n-1} s / sqrt(n)`` for a population mean.

    Exact under normality; asymptotically valid otherwise by the CLT, with
    accuracy degrading for small ``n`` from skewed populations. Raises for
    constant data, where the interval would collapse to a point and
    misrepresent the uncertainty.
    """
    x = as_float_array(values, name="values", min_size=2)
    confidence_level = require_probability(confidence_level, "confidence_level")
    sd = float(x.std(ddof=1))
    if sd == 0.0:
        raise ValueError("t interval undefined for constant data (zero sample variance)")
    half_width = float(stats.t.ppf(0.5 + confidence_level / 2, x.size - 1)) * sd / math.sqrt(
        x.size
    )
    mean = float(x.mean())
    return mean - half_width, mean + half_width
