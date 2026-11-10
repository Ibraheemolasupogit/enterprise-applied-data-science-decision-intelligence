"""Nonparametric bootstrap confidence intervals — the shared, owned implementation.

Per docs/architecture/overview.md ("Shared-utility ownership"), this module
is the single bootstrap/confidence-interval helper for every pillar
(statistics now; experimentation and decision science later). Pillars must
call it rather than resample locally.

Mathematical summary
--------------------
Given i.i.d. data ``x_1..x_n`` and a statistic ``T``, the bootstrap
approximates the sampling distribution of ``T`` by the distribution of
``T(x*)`` where ``x*`` is drawn with replacement from the observed data
(the empirical distribution ``F_n`` stands in for the unknown ``F``).

Three interval constructions are provided, all at level ``1 - alpha``:

* ``percentile``: ``[q_{alpha/2}, q_{1-alpha/2}]`` of the replicates. Simple,
  transformation-respecting, but first-order accurate only and biased when
  ``T`` is biased or its sampling distribution is skewed.
* ``basic`` (a.k.a. reverse percentile): ``[2T - q_{1-alpha/2}, 2T - q_{alpha/2}]``.
  Pivots on ``T* - T``; can leave the parameter space (e.g. negative
  variance) for skewed statistics.
* ``bca`` (bias-corrected and accelerated, Efron 1987): adjusts the
  percentile levels using a bias correction ``z0 = Phi^-1(P*(T* < T))``
  (ties counted half) and an acceleration ``a`` estimated by the jackknife.
  Second-order accurate; the recommended default for smooth statistics.

Assumptions and failure modes
-----------------------------
* Observations are independent and identically distributed *within each
  sample*. Clustered, time-series or otherwise dependent data violate this
  and produce intervals that are too narrow (use cluster-robust or block
  methods instead).
* The statistic is a smooth functional of ``F``. The bootstrap is
  inconsistent for extremes (the sample maximum), and percentile-type
  intervals for medians of small or heavily tied samples are coarse.
* With small ``n`` the empirical distribution poorly approximates ``F``;
  bootstrap intervals then under-cover, typically more than a t-interval
  would for near-normal data. The bootstrap does not create information.
* BCa is undefined when every replicate lies on one side of the estimate
  (``z0`` infinite); this raises rather than silently degrading.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy import special, stats

from decision_intelligence.common.validation import (
    FloatArray,
    as_float_array,
    require_positive_int,
    require_probability,
)

BootstrapMethod = Literal["percentile", "basic", "bca"]
_METHODS: tuple[str, ...] = ("percentile", "basic", "bca")


@dataclass(frozen=True)
class BootstrapResult:
    """A bootstrap confidence interval plus the evidence behind it.

    ``bias`` is ``mean(replicates) - estimate`` — the bootstrap estimate of
    the statistic's bias. It is reported for diagnosis, *not* subtracted
    from the estimate: bias-correcting a point estimate this way roughly
    doubles its variance and is rarely advisable.
    """

    estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    method: BootstrapMethod
    n_resamples: int
    standard_error: float
    bias: float
    replicates: FloatArray = field(repr=False)


def bootstrap_ci(
    data: ArrayLike,
    statistic: Callable[[FloatArray], float],
    *,
    rng: np.random.Generator,
    confidence_level: float = 0.95,
    n_resamples: int = 2000,
    method: BootstrapMethod = "bca",
) -> BootstrapResult:
    """One-sample nonparametric bootstrap confidence interval.

    Args:
        data: One-dimensional sample (at least two observations, no NaN).
        statistic: Maps a 1-D float array to a scalar, e.g. ``np.mean``.
        rng: Generator for resampling — pass a seeded generator (see
            :func:`decision_intelligence.common.seeded_generator`) for
            reproducible intervals.
        confidence_level: Two-sided coverage target in (0, 1).
        n_resamples: Number of bootstrap replicates (>= 100; 2000+ is
            advisable for BCa, whose tail quantiles need more replicates).
        method: ``"percentile"``, ``"basic"`` or ``"bca"``.
    """
    sample = as_float_array(data, name="data", min_size=2)
    return _bootstrap(
        (sample,),
        lambda arrays: statistic(arrays[0]),
        rng=rng,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        method=method,
    )


def bootstrap_two_sample_ci(
    x: ArrayLike,
    y: ArrayLike,
    statistic: Callable[[FloatArray, FloatArray], float],
    *,
    rng: np.random.Generator,
    confidence_level: float = 0.95,
    n_resamples: int = 2000,
    method: BootstrapMethod = "bca",
) -> BootstrapResult:
    """Two independent-sample bootstrap interval for ``statistic(x, y)``.

    Each group is resampled separately with its own size held fixed
    (stratified resampling), which is correct when the group sizes are
    fixed by design — e.g. a randomised experiment's arms — and conservative
    otherwise. The BCa acceleration combines per-group jackknife influence
    values (see :func:`bca_acceleration`).
    """
    sample_x = as_float_array(x, name="x", min_size=2)
    sample_y = as_float_array(y, name="y", min_size=2)
    return _bootstrap(
        (sample_x, sample_y),
        lambda arrays: statistic(arrays[0], arrays[1]),
        rng=rng,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        method=method,
    )


def _bootstrap(
    samples: tuple[FloatArray, ...],
    statistic: Callable[[Sequence[FloatArray]], float],
    *,
    rng: np.random.Generator,
    confidence_level: float,
    n_resamples: int,
    method: str,
) -> BootstrapResult:
    confidence_level = require_probability(confidence_level, "confidence_level")
    n_resamples = require_positive_int(n_resamples, "n_resamples", minimum=100)
    checked_method = _require_method(method)

    estimate = _evaluate(statistic, samples)

    replicates = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        resampled = tuple(s[rng.integers(0, s.size, size=s.size)] for s in samples)
        replicates[b] = _evaluate(statistic, resampled)
    if not np.isfinite(replicates).all():
        raise ValueError(
            "statistic returned a non-finite value on a bootstrap resample; it is not "
            "defined on every resample (e.g. a ratio whose denominator can resample to zero)"
        )

    alpha = 1.0 - confidence_level
    if checked_method == "percentile":
        lower, upper = np.quantile(replicates, [alpha / 2, 1 - alpha / 2])
    elif checked_method == "basic":
        q_low, q_high = np.quantile(replicates, [alpha / 2, 1 - alpha / 2])
        lower, upper = 2 * estimate - q_high, 2 * estimate - q_low
    else:
        lower, upper = _bca_interval(samples, statistic, estimate, replicates, alpha)

    return BootstrapResult(
        estimate=estimate,
        ci_lower=float(lower),
        ci_upper=float(upper),
        confidence_level=confidence_level,
        method=checked_method,
        n_resamples=n_resamples,
        standard_error=float(np.std(replicates, ddof=1)),
        bias=float(np.mean(replicates) - estimate),
        replicates=replicates,
    )


def _require_method(method: str) -> BootstrapMethod:
    if method == "percentile":
        return "percentile"
    if method == "basic":
        return "basic"
    if method == "bca":
        return "bca"
    raise ValueError(f"method must be one of {_METHODS}, got {method!r}")


def _evaluate(
    statistic: Callable[[Sequence[FloatArray]], float], samples: Sequence[FloatArray]
) -> float:
    return float(statistic(samples))


def bca_acceleration(
    samples: Sequence[FloatArray], statistic: Callable[[Sequence[FloatArray]], float]
) -> float:
    """Jackknife estimate of the BCa acceleration for one or more independent samples.

    For sample ``j`` of size ``n_j``, let ``T_(i)`` be the statistic with
    observation ``i`` of that sample left out, ``T_(.)`` their mean, and the
    empirical influence ``U_ji = (n_j - 1)(T_(.) - T_(i))``. Then

        a = sum_j sum_i U_ji^3 / n_j^3  /  (6 [sum_j sum_i U_ji^2 / n_j^2]^{3/2})

    (Efron & Tibshirani 1993, ch. 14; the form SciPy uses). Each sample is
    centred on *its own* jackknife mean. For one sample the ``(n - 1)/n``
    factors cancel and this reduces to Efron's classic
    ``sum d^3 / (6 (sum d^2)^{3/2})``.
    """
    numerator = 0.0
    denominator = 0.0
    for index, sample in enumerate(samples):
        values = np.empty(sample.size, dtype=np.float64)
        for i in range(sample.size):
            reduced = list(samples)
            reduced[index] = np.delete(sample, i)
            values[i] = _evaluate(statistic, reduced)
        n = sample.size
        influence = (n - 1) * (float(values.mean()) - values)
        numerator += float(np.sum(influence**3)) / n**3
        denominator += float(np.sum(influence**2)) / n**2
    if denominator <= 0.0:
        return 0.0
    return numerator / (6.0 * math.pow(denominator, 1.5))


def _bca_interval(
    samples: tuple[FloatArray, ...],
    statistic: Callable[[Sequence[FloatArray]], float],
    estimate: float,
    replicates: FloatArray,
    alpha: float,
) -> tuple[float, float]:
    # Bias correction: proportion of replicates below the estimate, ties half.
    below = float(np.mean(replicates < estimate))
    ties = float(np.mean(replicates == estimate))
    proportion = below + 0.5 * ties
    if proportion <= 0.0 or proportion >= 1.0:
        raise ValueError(
            "BCa interval undefined: every bootstrap replicate lies on one side of the "
            "estimate (bias-correction z0 is infinite); use method='percentile' and "
            "investigate the statistic's behaviour on resamples"
        )
    z0 = float(special.ndtri(proportion))
    acceleration = bca_acceleration(samples, statistic)

    z_levels = stats.norm.ppf([alpha / 2, 1 - alpha / 2])
    adjusted = z0 + (z0 + z_levels) / (1 - acceleration * (z0 + z_levels))
    levels = np.clip(stats.norm.cdf(adjusted), 0.0, 1.0)
    lower, upper = np.quantile(replicates, levels)
    return float(lower), float(upper)
