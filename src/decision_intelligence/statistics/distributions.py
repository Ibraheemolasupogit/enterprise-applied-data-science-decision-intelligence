"""Probability distributions and sampling methods.

Two families of tools live here:

1. **Random-variate generation** from a specified distribution:
   inverse-transform sampling and rejection (accept–reject) sampling. NumPy
   already provides samplers for standard families; these exist because
   real modelling work routinely needs distributions NumPy lacks (a fitted
   empirical quantile function, a truncated or custom density), and because
   their correctness conditions are worth making explicit and testable.

2. **Survey-style sampling from a finite population** — simple random
   sampling and stratified sampling with the design-based stratified mean
   estimator and its variance (with finite-population correction).

Closed-form quantile functions for the exponential and Weibull families
are included; the Weibull matches the time-to-churn model used by the M2
synthetic retention generator (``S(t) = exp(-(t/scale)^shape)``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from decision_intelligence.common.validation import (
    FloatArray,
    as_float_array,
    require_positive_int,
)

QuantileFunction = Callable[[FloatArray], FloatArray]
DensityFunction = Callable[[FloatArray], FloatArray]


# ---------------------------------------------------------------------------
# Closed-form distribution helpers
# ---------------------------------------------------------------------------


def exponential_quantile(rate: float) -> QuantileFunction:
    """Quantile function ``Q(u) = -ln(1 - u) / rate`` of Exponential(rate)."""
    if not rate > 0:
        raise ValueError("rate must be positive")

    def quantile(u: FloatArray) -> FloatArray:
        return np.asarray(-np.log1p(-u) / rate, dtype=np.float64)

    return quantile


def weibull_quantile(shape: float, scale: float) -> QuantileFunction:
    """Quantile function of Weibull(shape k, scale lambda): ``lambda (-ln(1-u))^{1/k}``."""
    if not shape > 0 or not scale > 0:
        raise ValueError("shape and scale must be positive")

    def quantile(u: FloatArray) -> FloatArray:
        return np.asarray(scale * np.power(-np.log1p(-u), 1.0 / shape), dtype=np.float64)

    return quantile


def weibull_survival(t: ArrayLike, shape: float, scale: float) -> FloatArray:
    """Weibull survival function ``S(t) = exp(-(t/scale)^shape)`` for ``t >= 0``."""
    if not shape > 0 or not scale > 0:
        raise ValueError("shape and scale must be positive")
    times = np.asarray(t, dtype=np.float64)
    if (times < 0).any():
        raise ValueError("t must be non-negative")
    return np.asarray(np.exp(-np.power(times / scale, shape)), dtype=np.float64)


def weibull_mean(shape: float, scale: float) -> float:
    """``E[T] = scale * Gamma(1 + 1/shape)``."""
    if not shape > 0 or not scale > 0:
        raise ValueError("shape and scale must be positive")
    return scale * math.gamma(1.0 + 1.0 / shape)


# ---------------------------------------------------------------------------
# Random-variate generation
# ---------------------------------------------------------------------------


def inverse_transform_sample(
    quantile: QuantileFunction, size: int, rng: np.random.Generator
) -> FloatArray:
    """Draw ``size`` variates as ``Q(U)`` with ``U ~ Uniform(0, 1)``.

    Correct for *any* distribution whose (generalised) quantile function is
    supplied: if ``U`` is uniform then ``Q(U)`` has CDF ``F``. Failure mode:
    an incorrect or numerically unstable ``Q`` in the tails silently
    produces the wrong distribution — the result is checked for finiteness
    but not for correctness, which is what the accompanying tests do.
    """
    size = require_positive_int(size, "size")
    u = rng.random(size)
    draws = np.asarray(quantile(u), dtype=np.float64)
    if draws.shape != (size,) or not np.isfinite(draws).all():
        raise ValueError("quantile function must return finite values with the input's shape")
    return draws


@dataclass(frozen=True)
class RejectionSampleResult:
    """Samples from accept–reject plus the efficiency evidence.

    ``acceptance_rate`` should be close to ``theoretical_acceptance_rate``
    (``1 / envelope_constant`` when both densities are normalised); a large
    gap indicates a mis-specified density or envelope.
    """

    samples: FloatArray = field(repr=False)
    n_proposed: int
    acceptance_rate: float
    theoretical_acceptance_rate: float


def rejection_sample(
    target_density: DensityFunction,
    proposal_sampler: Callable[[np.random.Generator, int], FloatArray],
    proposal_density: DensityFunction,
    envelope_constant: float,
    size: int,
    rng: np.random.Generator,
    *,
    max_proposals: int = 10_000_000,
) -> RejectionSampleResult:
    """Accept–reject sampling from ``target_density``.

    Requires an envelope ``target(x) <= M * proposal(x)`` for all ``x``.
    A candidate ``x ~ proposal`` is accepted with probability
    ``target(x) / (M * proposal(x))``; accepted draws are exact samples from
    the (normalised) target. The expected acceptance rate is ``1/M`` for
    normalised densities, so a loose envelope wastes proportionally more
    proposals.

    Raises:
        ValueError: If the envelope condition is observed to fail on any
            proposed point — continuing would silently sample from the wrong
            distribution — or if ``max_proposals`` is exhausted.
    """
    size = require_positive_int(size, "size")
    if not envelope_constant >= 1.0:
        raise ValueError("envelope_constant must be >= 1 for normalised densities")

    accepted: list[FloatArray] = []
    n_accepted = 0
    n_proposed = 0
    batch = max(size, 1024)
    while n_accepted < size:
        if n_proposed >= max_proposals:
            raise ValueError(
                f"rejection sampling exhausted {max_proposals} proposals after accepting "
                f"{n_accepted}; the envelope is too loose or the target density is ~zero"
            )
        candidates = np.asarray(proposal_sampler(rng, batch), dtype=np.float64)
        target = np.asarray(target_density(candidates), dtype=np.float64)
        envelope = envelope_constant * np.asarray(proposal_density(candidates), dtype=np.float64)
        if (target > envelope * (1 + 1e-12)).any():
            raise ValueError(
                "envelope condition target(x) <= M * proposal(x) violated; increase "
                "envelope_constant or choose a heavier-tailed proposal"
            )
        keep = rng.random(batch) * envelope < target
        accepted.append(candidates[keep])
        n_accepted += int(keep.sum())
        n_proposed += batch

    samples = np.concatenate(accepted)[:size]
    return RejectionSampleResult(
        samples=samples,
        n_proposed=n_proposed,
        acceptance_rate=n_accepted / n_proposed,
        theoretical_acceptance_rate=1.0 / envelope_constant,
    )


def empirical_cdf(sample: ArrayLike) -> Callable[[ArrayLike], FloatArray]:
    """Return the empirical CDF ``F_n(t) = #{x_i <= t} / n`` as a callable."""
    ordered = np.sort(as_float_array(sample, name="sample", min_size=1))

    def cdf(t: ArrayLike) -> FloatArray:
        points = np.asarray(t, dtype=np.float64)
        return np.asarray(
            np.searchsorted(ordered, points, side="right") / ordered.size, dtype=np.float64
        )

    return cdf


def ks_distance(sample: ArrayLike, cdf: Callable[[FloatArray], FloatArray]) -> float:
    """Kolmogorov–Smirnov distance ``sup_t |F_n(t) - F(t)|`` to a continuous CDF.

    Evaluated exactly at the jump points of ``F_n`` (both sides of each
    jump). Used as a goodness-of-fit *distance* (e.g. for CLT diagnostics);
    for a formal test with estimated parameters the standard KS p-value is
    invalid (Lilliefors), so none is reported here.
    """
    ordered = np.sort(as_float_array(sample, name="sample", min_size=1))
    n = ordered.size
    theoretical = np.asarray(cdf(ordered), dtype=np.float64)
    upper = np.arange(1, n + 1) / n - theoretical
    lower = theoretical - np.arange(0, n) / n
    return float(max(np.max(upper), np.max(lower)))


# ---------------------------------------------------------------------------
# Finite-population sampling designs
# ---------------------------------------------------------------------------


def simple_random_sample(
    population_size: int, sample_size: int, rng: np.random.Generator
) -> NDArray[np.int64]:
    """Indices of a simple random sample *without* replacement (each subset equally likely)."""
    population_size = require_positive_int(population_size, "population_size")
    sample_size = require_positive_int(sample_size, "sample_size")
    if sample_size > population_size:
        raise ValueError("sample_size cannot exceed population_size without replacement")
    return np.sort(rng.choice(population_size, size=sample_size, replace=False)).astype(np.int64)


Allocation = Literal["proportional", "equal"]


def stratified_sample(
    strata: ArrayLike,
    sample_size: int,
    rng: np.random.Generator,
    *,
    allocation: Allocation = "proportional",
) -> NDArray[np.int64]:
    """Indices of a stratified random sample (SRS without replacement within each stratum).

    ``proportional`` allocation gives stratum ``h`` ``round(n * N_h / N)``
    units (largest-remainder rounding so the total is exactly ``n``);
    ``equal`` gives each stratum ``n / H``. Raises if any stratum would
    receive fewer than two units (its within-stratum variance would then be
    inestimable) or more units than it contains.
    """
    labels = np.asarray(strata)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("strata must be a non-empty one-dimensional array of labels")
    sample_size = require_positive_int(sample_size, "sample_size")
    unique, counts = np.unique(labels, return_counts=True)
    if sample_size < 2 * unique.size:
        raise ValueError("sample_size must allow at least two units per stratum")
    if (counts < 2).any():
        raise ValueError("every stratum must contain at least two population units")

    if allocation == "proportional":
        ideal = sample_size * counts / counts.sum()
    elif allocation == "equal":
        ideal = np.full(unique.size, sample_size / unique.size)
    else:
        raise ValueError("allocation must be 'proportional' or 'equal'")
    allocated = np.floor(ideal).astype(int)
    remainder = sample_size - int(allocated.sum())
    allocated[np.argsort(-(ideal - allocated), kind="stable")[:remainder]] += 1
    if (allocated < 2).any():
        raise ValueError(
            "allocation gives some stratum fewer than two units; increase sample_size "
            "or use allocation='equal'"
        )
    if (allocated > counts).any():
        raise ValueError("allocation exceeds a stratum's population size; reduce sample_size")

    chosen: list[NDArray[np.int64]] = []
    for label, n_h in zip(unique, allocated, strict=True):
        members = np.flatnonzero(labels == label)
        chosen.append(rng.choice(members, size=int(n_h), replace=False))
    return np.sort(np.concatenate(chosen)).astype(np.int64)


@dataclass(frozen=True)
class StratifiedEstimate:
    """Design-based estimate of a population mean from a stratified sample."""

    mean: float
    standard_error: float
    stratum_weights: dict[str, float]
    stratum_sample_sizes: dict[str, int]


def stratified_mean(
    values: ArrayLike,
    sample_strata: ArrayLike,
    population_stratum_sizes: Mapping[str, int],
) -> StratifiedEstimate:
    """Stratified estimator of a population mean, with finite-population correction.

    ``ybar_st = sum_h W_h ybar_h`` with ``W_h = N_h / N`` and
    ``Var(ybar_st) = sum_h W_h^2 (1 - n_h/N_h) s_h^2 / n_h``.
    Unbiased under stratified SRS without replacement regardless of how the
    outcome is distributed; it gains precision over SRS when strata
    explain outcome variation.
    """
    y = as_float_array(values, name="values", min_size=2)
    labels = np.asarray(sample_strata).astype(str)
    if labels.shape != y.shape:
        raise ValueError("sample_strata must align with values")
    total = sum(population_stratum_sizes.values())
    if total <= 0:
        raise ValueError("population_stratum_sizes must be positive")

    mean = 0.0
    variance = 0.0
    weights: dict[str, float] = {}
    sizes: dict[str, int] = {}
    for label, population_size in population_stratum_sizes.items():
        stratum_values = y[labels == str(label)]
        n_h = stratum_values.size
        if n_h < 2:
            raise ValueError(f"stratum {label!r} needs at least two sampled units")
        if n_h > population_size:
            raise ValueError(f"stratum {label!r} has more sampled units than its population")
        weight = population_size / total
        mean += weight * float(stratum_values.mean())
        fpc = 1.0 - n_h / population_size
        variance += weight**2 * fpc * float(stratum_values.var(ddof=1)) / n_h
        weights[str(label)] = weight
        sizes[str(label)] = int(n_h)
    if set(np.unique(labels)) - {str(k) for k in population_stratum_sizes}:
        raise ValueError("sample contains strata absent from population_stratum_sizes")
    return StratifiedEstimate(
        mean=mean,
        standard_error=math.sqrt(variance),
        stratum_weights=weights,
        stratum_sample_sizes=sizes,
    )
