"""Reproducible Monte Carlo simulation with explicit simulation error.

A Monte Carlo result is itself an estimate. Every function here reports
the Monte Carlo standard error alongside the estimate, so a reader can
tell whether "coverage 0.93" means "under-covers" or "0.95 within
simulation noise". Rule of thumb: for a proportion near ``p`` estimated
from ``R`` replications, the MC standard error is ``sqrt(p(1-p)/R)`` —
about 0.007 for ``p=0.95, R=1000``.

Reproducibility: every function takes an explicit ``numpy.random.Generator``.
Passing a generator from :func:`decision_intelligence.common.seeded_generator`
makes the whole simulation deterministic; the functions never touch global
random state.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from decision_intelligence.common.validation import (
    FloatArray,
    require_positive_int,
    require_probability,
)


@dataclass(frozen=True)
class MonteCarloEstimate:
    """Mean of ``R`` simulated draws with its Monte Carlo uncertainty."""

    estimate: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    n_replications: int
    draws: FloatArray = field(repr=False)


def monte_carlo_mean(
    simulate_once: Callable[[np.random.Generator], float],
    n_replications: int,
    rng: np.random.Generator,
    *,
    confidence_level: float = 0.95,
) -> MonteCarloEstimate:
    """Estimate ``E[g]`` where ``simulate_once(rng)`` returns one draw of ``g``.

    The interval is a normal-approximation interval for the *simulation*
    error only (valid by the CLT for large ``R`` and finite-variance ``g``);
    it says nothing about modelling error in ``simulate_once`` itself.
    """
    n_replications = require_positive_int(n_replications, "n_replications", minimum=2)
    confidence_level = require_probability(confidence_level, "confidence_level")
    draws = np.fromiter(
        (float(simulate_once(rng)) for _ in range(n_replications)),
        dtype=np.float64,
        count=n_replications,
    )
    if not np.isfinite(draws).all():
        raise ValueError("simulate_once returned a non-finite value")
    estimate = float(draws.mean())
    standard_error = float(draws.std(ddof=1)) / math.sqrt(n_replications)
    z = float(stats.norm.ppf(0.5 + confidence_level / 2))
    return MonteCarloEstimate(
        estimate=estimate,
        standard_error=standard_error,
        ci_lower=estimate - z * standard_error,
        ci_upper=estimate + z * standard_error,
        n_replications=n_replications,
        draws=draws,
    )


@dataclass(frozen=True)
class SimulatedRate:
    """A simulated probability (rejection rate, coverage) with its MC standard error."""

    rate: float
    standard_error: float
    n_replications: int

    def is_consistent_with(self, target: float, *, z: float = 3.0) -> bool:
        """``|rate - target| <= z * SE`` using the SE implied by ``target``.

        Using the target's binomial SE (rather than the observed one) keeps
        the check well-defined when the observed rate is exactly 0 or 1.
        """
        target_se = math.sqrt(target * (1 - target) / self.n_replications)
        return abs(self.rate - target) <= z * max(target_se, 1e-12)


def simulate_rejection_rate(
    generate_data: Callable[[np.random.Generator], tuple[FloatArray, ...]],
    compute_p_value: Callable[..., float],
    *,
    alpha: float,
    n_replications: int,
    rng: np.random.Generator,
) -> SimulatedRate:
    """Proportion of simulated datasets on which a test rejects at level ``alpha``.

    Under a data-generating process satisfying H0 this estimates the
    actual type I error rate (which should be <= ``alpha`` for a valid
    test); under an alternative it estimates power.
    """
    alpha = require_probability(alpha, "alpha")
    n_replications = require_positive_int(n_replications, "n_replications")
    rejections = 0
    for _ in range(n_replications):
        p_value = float(compute_p_value(*generate_data(rng)))
        if not 0.0 <= p_value <= 1.0:
            raise ValueError(f"compute_p_value returned an invalid p-value {p_value}")
        rejections += p_value <= alpha
    rate = rejections / n_replications
    return SimulatedRate(
        rate=rate,
        standard_error=math.sqrt(rate * (1 - rate) / n_replications),
        n_replications=n_replications,
    )


def simulate_interval_coverage(
    generate_data: Callable[[np.random.Generator], tuple[FloatArray, ...]],
    compute_interval: Callable[..., tuple[float, float]],
    true_value: float,
    *,
    n_replications: int,
    rng: np.random.Generator,
) -> SimulatedRate:
    """Proportion of simulated intervals that contain ``true_value``.

    The defining property of a ``(1 - alpha)`` confidence procedure is that
    this rate equals ``1 - alpha`` over repeated sampling; a single realised
    interval either contains the parameter or it does not.
    """
    n_replications = require_positive_int(n_replications, "n_replications")
    hits = 0
    for _ in range(n_replications):
        lower, upper = compute_interval(*generate_data(rng))
        if not lower <= upper:
            raise ValueError("compute_interval returned lower > upper")
        hits += lower <= true_value <= upper
    rate = hits / n_replications
    return SimulatedRate(
        rate=rate,
        standard_error=math.sqrt(rate * (1 - rate) / n_replications),
        n_replications=n_replications,
    )
