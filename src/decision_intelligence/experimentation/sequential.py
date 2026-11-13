"""Sequential experimentation: valid inference when results are monitored repeatedly.

**The problem.** A fixed-horizon test controls the type I error only if it is
performed *once*, at the planned sample size. Testing the accumulating data at
several looks and stopping at the first ``p <= 0.05`` ("peeking") inflates the
false-positive rate: with five equally spaced looks to 0.142, with ten to 0.193
and with twenty to 0.248 (Armitage, McPherson & Rowe 1969; reproduced exactly
by :func:`crossing_probabilities` and by simulation in the M7 evidence).

Two valid families are implemented; they answer different monitoring needs:

* **Group-sequential designs** (:func:`spending_boundaries`,
  :func:`constant_boundaries`) — a pre-specified, small number of interim
  analyses. Boundaries ``c_k`` on the standardised statistic ``Z_k`` are chosen
  so that ``P(|Z_k| >= c_k`` for some ``k | H0) = alpha``. Lan–DeMets
  alpha-spending functions fix how much of ``alpha`` may be used by
  information fraction ``t``, so looks need not be equally spaced or planned
  exactly. O'Brien–Fleming-type spending is very conservative early and loses
  almost nothing at the final look (maximum sample size +2.8% for five looks);
  Pocock-type spending stops earlier for large effects but needs a larger
  maximum sample size (+23%).
* **Always-valid inference** (:func:`msprt_log_likelihood_ratio`,
  :func:`always_valid_p_values`, :func:`confidence_sequence`) — the normal
  mixture sequential probability ratio test (mSPRT; Robbins 1970; Johari,
  Pekelis & Walsh 2017). Valid under continuous monitoring and data-dependent
  stopping at *any* time, at the price of wider intervals and lower power than
  a group-sequential design at the same sample size.

Boundary probabilities are computed exactly (to quadrature accuracy) with the
recursive numerical integration of Armitage, McPherson & Rowe (1969), as set
out in Jennison & Turnbull (2000, ch. 19): the score ``S_k = Z_k sqrt(I_k)`` is
a Brownian motion with drift ``theta`` observed at information ``I_k``.

**Inference after stopping.** A trial that stops early for efficacy tends to do
so when the estimate is high: the naive estimate at stopping is biased upward
and its fixed-sample interval does not have nominal coverage. The repeated
confidence intervals :func:`repeated_confidence_interval` (Jennison & Turnbull
1989) have simultaneous coverage ``>= 1 - alpha`` over all looks and remain
valid whatever the stopping rule. Both effects are quantified in the evidence.

Assumptions: independent units, outcomes observed promptly (no delayed
outcomes), a statistic that is approximately normal with independent increments
(true for differences in means of i.i.d. units by the CLT), and the variance
estimated consistently. Delayed outcomes, novelty effects and time-varying
treatment effects violate them (see the methodology).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize, stats

from decision_intelligence.common.validation import FloatArray, require_probability

SpendingFamily = Literal["obrien-fleming", "pocock", "power"]
ConstantShape = Literal["pocock", "obrien-fleming"]

_GRID_POINTS = 801
"""Odd number of Simpson nodes per stage; boundary probabilities are accurate to ~1e-7."""


# ---------------------------------------------------------------------------
# Alpha spending
# ---------------------------------------------------------------------------


def spending_function(
    t: float, *, alpha: float = 0.05, family: SpendingFamily = "obrien-fleming", rho: float = 1.0
) -> float:
    """Cumulative two-sided type I error spent by information fraction ``t`` in ``[0, 1]``.

    Boundaries are symmetric, so each side spends half of the value returned.

    * ``obrien-fleming``: each side spends ``2 - 2 Phi(z_{1-a/2} / sqrt(t))`` with ``a =
      alpha / 2`` — the one-sided Lan & DeMets (1983) function applied per side, the convention
      of gsDesign, rpact and ldbounds (five equal looks at alpha = 0.05: 4.877, 3.357, 2.680,
      2.290, 2.031). Applying the two-sided form to the total alpha instead is also seen in the
      literature; it spends more early (first boundary 4.38) and is not used here;
    * ``pocock``: ``alpha log(1 + (e - 1) t)`` (Lan & DeMets 1983);
    * ``power``: ``alpha t^rho`` (Kim & DeMets 1987; ``rho = 3`` resembles O'Brien–Fleming,
      ``rho = 1`` Pocock).
    """
    alpha = require_probability(alpha, "alpha")
    if not 0.0 <= t <= 1.0:
        raise ValueError("information fraction t must lie in [0, 1]")
    if t == 0.0:
        return 0.0
    if family == "obrien-fleming":
        z = float(stats.norm.ppf(1 - alpha / 4))  # per side: a = alpha / 2, z_{1 - a/2}
        return float(2 * (2 - 2 * stats.norm.cdf(z / math.sqrt(t))))
    if family == "pocock":
        return float(alpha * math.log(1 + (math.e - 1) * t))
    if family == "power":
        if rho <= 0:
            raise ValueError("rho must be positive")
        return float(alpha * t**rho)
    raise ValueError("family must be 'obrien-fleming', 'pocock' or 'power'")


# ---------------------------------------------------------------------------
# Recursive numerical integration (Armitage, McPherson & Rowe 1969)
# ---------------------------------------------------------------------------


def _validate_fractions(fractions: ArrayLike) -> FloatArray:
    t = np.asarray(fractions, dtype=np.float64)
    if t.ndim != 1 or t.size < 1:
        raise ValueError("information fractions must be a non-empty 1-D sequence")
    if not (np.all(np.diff(t) > 0) and t[0] > 0 and abs(t[-1] - 1.0) < 1e-12):
        raise ValueError("information fractions must increase strictly from > 0 to exactly 1")
    return t


def _simpson_weights(n: int, width: float) -> FloatArray:
    w = np.ones(n)
    w[1:-1:2], w[2:-1:2] = 4.0, 2.0
    return np.asarray(w * width / 3.0, dtype=np.float64)


@dataclass(frozen=True)
class _Stage:
    """Sub-density of S_k on the continuation region at stage k (on a Simpson grid)."""

    nodes: FloatArray
    density: FloatArray
    weights: FloatArray


def _first_stage(bound: float, information: float, drift: float) -> _Stage:
    half = bound * math.sqrt(information)
    nodes = np.linspace(-half, half, _GRID_POINTS)
    sd = math.sqrt(information)
    density = stats.norm.pdf((nodes - drift * information) / sd) / sd
    return _Stage(nodes, np.asarray(density), _simpson_weights(nodes.size, nodes[1] - nodes[0]))


def _next_stage(stage: _Stage, bound: float, information: float, increment: float,
                drift: float) -> _Stage:
    half = bound * math.sqrt(information)
    nodes = np.linspace(-half, half, _GRID_POINTS)
    sd = math.sqrt(increment)
    kernel = stats.norm.pdf((nodes[:, None] - stage.nodes[None, :] - drift * increment) / sd) / sd
    density = kernel @ (stage.weights * stage.density)
    return _Stage(nodes, np.asarray(density), _simpson_weights(nodes.size, nodes[1] - nodes[0]))


def _exit_probabilities(
    stage: _Stage | None, bound: float, information: float, increment: float, drift: float
) -> tuple[float, float]:
    """P(cross upper) and P(cross lower) at this stage, continuing from ``stage``."""
    level = bound * math.sqrt(information)
    if stage is None:
        sd = math.sqrt(information)
        mean = drift * information
        return (float(stats.norm.sf((level - mean) / sd)),
                float(stats.norm.cdf((-level - mean) / sd)))
    sd = math.sqrt(increment)
    shifted = stage.nodes + drift * increment
    upper = stats.norm.sf((level - shifted) / sd)
    lower = stats.norm.cdf((-level - shifted) / sd)
    mass = stage.weights * stage.density
    return float(mass @ upper), float(mass @ lower)


@dataclass(frozen=True)
class CrossingProbabilities:
    """Per-look probabilities of first crossing the upper / lower boundary."""

    upper: tuple[float, ...]
    lower: tuple[float, ...]

    @property
    def total(self) -> float:
        return float(sum(self.upper) + sum(self.lower))

    @property
    def stopping(self) -> tuple[float, ...]:
        """Probability of stopping at each look (either boundary)."""
        return tuple(u + lo for u, lo in zip(self.upper, self.lower, strict=True))


def crossing_probabilities(
    boundaries: ArrayLike, fractions: ArrayLike, *, drift: float = 0.0
) -> CrossingProbabilities:
    """Exact first-crossing probabilities for two-sided symmetric boundaries ``|Z_k| >= c_k``.

    ``drift`` is the expected final-look statistic, ``E[Z_K] = theta sqrt(I_max)``, so that
    ``E[Z_k] = drift sqrt(t_k)``. ``drift = 0`` gives the type I error spent at each look.
    The information scale is normalised to ``I_max = 1``.
    """
    c = np.asarray(boundaries, dtype=np.float64)
    t = _validate_fractions(fractions)
    if c.shape != t.shape or (c <= 0).any():
        raise ValueError("one positive boundary per look is required")
    upper, lower = [], []
    stage: _Stage | None = None
    previous = 0.0
    for k in range(t.size):
        increment = t[k] - previous
        u, lo = _exit_probabilities(stage, c[k], t[k], increment, drift)
        upper.append(u)
        lower.append(lo)
        if k < t.size - 1:
            stage = (_first_stage(c[k], t[k], drift) if stage is None
                     else _next_stage(stage, c[k], t[k], increment, drift))
        previous = t[k]
    return CrossingProbabilities(tuple(upper), tuple(lower))


# ---------------------------------------------------------------------------
# Boundaries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupSequentialDesign:
    """Two-sided symmetric efficacy boundaries on ``Z_k`` at information fractions ``t_k``."""

    information_fractions: tuple[float, ...]
    boundaries: tuple[float, ...]
    alpha: float
    method: str
    cumulative_alpha: tuple[float, ...]
    """Type I error spent by each look, recomputed exactly from the boundaries."""

    @property
    def nominal_p_values(self) -> tuple[float, ...]:
        """Two-sided fixed-sample p-value equivalent to each boundary (not an adjusted p-value)."""
        return tuple(float(2 * stats.norm.sf(c)) for c in self.boundaries)

    @property
    def n_looks(self) -> int:
        return len(self.boundaries)


def _design(fractions: FloatArray, bounds: Sequence[float], alpha: float,
            method: str) -> GroupSequentialDesign:
    probs = crossing_probabilities(bounds, fractions)
    cumulative = np.cumsum(np.asarray(probs.upper) + np.asarray(probs.lower))
    return GroupSequentialDesign(
        information_fractions=tuple(float(x) for x in fractions),
        boundaries=tuple(float(b) for b in bounds),
        alpha=alpha,
        method=method,
        cumulative_alpha=tuple(float(x) for x in cumulative),
    )


def spending_boundaries(
    information_fractions: ArrayLike,
    *,
    alpha: float = 0.05,
    family: SpendingFamily = "obrien-fleming",
    rho: float = 1.0,
) -> GroupSequentialDesign:
    """Lan–DeMets boundaries: look ``k`` spends ``alpha(t_k) - alpha(t_{k-1})`` exactly.

    Fractions are the *observed* information fractions (``n_k / n_max`` for a difference in
    means), so unplanned or unequally spaced looks are handled by recomputing the boundaries
    with the fractions actually reached. A look that would spend (numerically) nothing gets an
    infinite-like boundary of 8.
    """
    alpha = require_probability(alpha, "alpha")
    t = _validate_fractions(information_fractions)
    spent = [spending_function(float(x), alpha=alpha, family=family, rho=rho) for x in t]
    bounds: list[float] = []
    stage: _Stage | None = None
    previous_t = previous_spent = 0.0
    for k in range(t.size):
        increment = t[k] - previous_t
        target = spent[k] - previous_spent

        def excess(c: float, stage: _Stage | None = stage, k: int = k,
                   increment: float = increment, target: float = target) -> float:
            u, lo = _exit_probabilities(stage, c, t[k], increment, 0.0)
            return u + lo - target

        if target <= 1e-15 or excess(8.0) > 0:
            bound = 8.0
        else:
            bound = float(optimize.brentq(excess, 1e-6, 8.0, xtol=1e-10))
        bounds.append(bound)
        if k < t.size - 1:
            stage = (_first_stage(bound, t[k], 0.0) if stage is None
                     else _next_stage(stage, bound, t[k], increment, 0.0))
        previous_t, previous_spent = float(t[k]), spent[k]
    return _design(t, bounds, alpha, f"Lan-DeMets {family} spending"
                   + (f" (rho = {rho:g})" if family == "power" else ""))


def constant_boundaries(
    information_fractions: ArrayLike, *, alpha: float = 0.05, shape: ConstantShape = "pocock"
) -> GroupSequentialDesign:
    """Classical boundaries ``c_k = C`` (Pocock 1977) or ``c_k = C / sqrt(t_k)``
    (O'Brien & Fleming 1979), with ``C`` solved so the total type I error is ``alpha``."""
    alpha = require_probability(alpha, "alpha")
    t = _validate_fractions(information_fractions)
    if shape == "pocock":
        profile = np.ones_like(t)
    elif shape == "obrien-fleming":
        profile = 1.0 / np.sqrt(t)
    else:
        raise ValueError("shape must be 'pocock' or 'obrien-fleming'")

    def excess(constant: float) -> float:
        return crossing_probabilities(constant * profile, t).total - alpha

    constant = float(optimize.brentq(excess, 1.0, 6.0, xtol=1e-9))
    return _design(t, list(constant * profile), alpha, f"{shape} constant boundary")


def naive_peeking_design(information_fractions: ArrayLike, *, alpha: float = 0.05
                         ) -> GroupSequentialDesign:
    """The *invalid* practice: a fixed-sample test at level ``alpha`` at every look."""
    t = _validate_fractions(information_fractions)
    c = float(stats.norm.ppf(1 - alpha / 2))
    return _design(t, [c] * t.size, alpha, "naive repeated fixed-sample tests (invalid)")


# ---------------------------------------------------------------------------
# Operating characteristics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OperatingCharacteristics:
    drift: float
    rejection_probability: float
    """Two-sided rejection probability (power when ``drift != 0``, size when 0)."""
    expected_information_fraction: float
    """Expected sample size as a fraction of the maximum (average sample number / n_max)."""
    stopping_probabilities: tuple[float, ...]


def operating_characteristics(design: GroupSequentialDesign, drift: float
                              ) -> OperatingCharacteristics:
    probs = crossing_probabilities(design.boundaries, design.information_fractions, drift=drift)
    stop = np.asarray(probs.stopping)
    t = np.asarray(design.information_fractions)
    # Units that reach the last look use the full sample whether or not they reject.
    expected = float(np.sum(stop[:-1] * t[:-1]) + (1.0 - stop[:-1].sum()) * t[-1])
    return OperatingCharacteristics(drift, probs.total, expected, tuple(float(s) for s in stop))


def fixed_sample_drift(*, alpha: float = 0.05, power: float = 0.8) -> float:
    """``E[Z]`` a two-sided fixed-sample test needs for ``power``: ``z_{1-alpha/2} + z_power``."""
    return float(stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power))


def maximum_information_inflation(design: GroupSequentialDesign, *, power: float = 0.8) -> float:
    """``R = I_max / I_fixed``: how much larger the maximum sample must be than a fixed-sample
    design with the same alpha and power (Jennison & Turnbull 2000, Tables 2.2 and 2.4)."""
    power = require_probability(power, "power")
    fixed = fixed_sample_drift(alpha=design.alpha, power=power)

    def gap(drift: float) -> float:
        return operating_characteristics(design, drift).rejection_probability - power

    drift = float(optimize.brentq(gap, fixed * 0.9, fixed * 2.0, xtol=1e-9))
    return (drift / fixed) ** 2


# ---------------------------------------------------------------------------
# Monitoring data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SequentialLook:
    look: int
    information_fraction: float
    estimate: float
    standard_error: float
    z: float
    boundary: float
    crossed: bool
    repeated_ci: tuple[float, float]


@dataclass(frozen=True)
class SequentialMonitoringResult:
    design: GroupSequentialDesign
    looks: tuple[SequentialLook, ...]
    stopped_at: int | None
    """1-based look at which a boundary was crossed (``None``: no boundary crossed yet)."""
    decision: Literal["reject H0", "do not reject H0", "continue"]
    """``continue`` while no boundary is crossed and the maximum information is not reached:
    "do not reject H0" is a terminal decision, available only at the final look."""
    warnings: tuple[str, ...] = ()

    @property
    def final(self) -> SequentialLook:
        return self.looks[-1]


def repeated_confidence_interval(estimate: float, standard_error: float, boundary: float
                                 ) -> tuple[float, float]:
    """``estimate +/- c_k SE_k``: simultaneous coverage ``>= 1 - alpha`` over all looks."""
    return estimate - boundary * standard_error, estimate + boundary * standard_error


def monitor(
    estimates: Sequence[float],
    standard_errors: Sequence[float],
    information_fractions: Sequence[float],
    *,
    alpha: float = 0.05,
    family: SpendingFamily = "obrien-fleming",
    rho: float = 1.0,
) -> SequentialMonitoringResult:
    """Apply a spending-function design to the statistics observed at each look.

    Boundaries are computed for the information fractions actually observed (``n_k /
    n_max``). Monitoring stops at the first look with ``|Z_k| >= c_k``; looks after it are not
    used. Without a crossing the decision is ``continue`` until the final look (fraction 1)
    has been analysed. The estimate reported at stopping is *not* bias-corrected (see the module
    docstring); the repeated confidence interval at that look remains valid.
    """
    if not (len(estimates) == len(standard_errors) == len(information_fractions) >= 1):
        raise ValueError("one estimate, SE and information fraction per look")
    if not all(math.isfinite(float(e)) for e in estimates):
        raise ValueError("estimates must be finite")
    fractions = [float(f) for f in information_fractions]
    reached_maximum = abs(fractions[-1] - 1.0) <= 1e-12
    warnings: list[str] = []
    if not reached_maximum:
        warnings.append("the final look has not reached the maximum information: boundaries "
                        "for the remaining looks will be recomputed when they occur")
        fractions = fractions + [1.0]
    design = spending_boundaries(fractions, alpha=alpha, family=family, rho=rho)
    looks: list[SequentialLook] = []
    stopped: int | None = None
    for k, (estimate, se) in enumerate(zip(estimates, standard_errors, strict=True)):
        if se <= 0 or not math.isfinite(se):
            raise ValueError("standard errors must be positive and finite")
        z = estimate / se
        c = design.boundaries[k]
        crossed = abs(z) >= c
        looks.append(SequentialLook(k + 1, fractions[k], float(estimate), float(se), z, c,
                                    crossed, repeated_confidence_interval(estimate, se, c)))
        if crossed:
            stopped = k + 1
            break
    decision: Literal["reject H0", "do not reject H0", "continue"]
    if stopped is not None:
        decision = "reject H0"
    elif reached_maximum:
        decision = "do not reject H0"
    else:
        decision = "continue"
    if stopped is not None and stopped < design.n_looks:
        warnings.append("stopped early: the estimate at stopping is biased away from zero; "
                        "report the repeated confidence interval")
    return SequentialMonitoringResult(design, tuple(looks), stopped, decision, tuple(warnings))


# ---------------------------------------------------------------------------
# Always-valid inference: normal-mixture SPRT
# ---------------------------------------------------------------------------


def msprt_log_likelihood_ratio(
    estimate: ArrayLike, variance: ArrayLike, *, mixture_sd: float, null: float = 0.0
) -> FloatArray:
    """``log Lambda_n`` of the normal-mixture SPRT for ``H0: theta = null``.

    With ``theta_hat ~ N(theta, V_n)`` and mixing distribution ``theta ~ N(null, tau^2)``:
    ``Lambda_n = sqrt(V_n / (V_n + tau^2)) exp(tau^2 (theta_hat - null)^2 / (2 V_n (V_n +
    tau^2)))``. Under ``H0`` it is a non-negative martingale, so by Ville's inequality
    ``P(sup_n Lambda_n >= 1 / alpha) <= alpha`` (Robbins 1970).
    """
    if mixture_sd <= 0:
        raise ValueError("mixture_sd must be positive")
    theta = np.asarray(estimate, dtype=np.float64) - null
    v = np.asarray(variance, dtype=np.float64)
    if (v <= 0).any():
        raise ValueError("variances must be positive")
    tau2 = mixture_sd**2
    return np.asarray(0.5 * np.log(v / (v + tau2)) + tau2 * theta**2 / (2 * v * (v + tau2)),
                      dtype=np.float64)


def always_valid_p_values(
    estimates: ArrayLike, variances: ArrayLike, *, mixture_sd: float
) -> FloatArray:
    """``p_n = min(1, min_{m <= n} 1 / Lambda_m)``: ``P(p_n <= alpha`` for some ``n | H0) <=
    alpha`` under continuous monitoring and any stopping rule."""
    log_lr = msprt_log_likelihood_ratio(estimates, variances, mixture_sd=mixture_sd)
    p = np.minimum(1.0, np.exp(-log_lr))
    return np.asarray(np.minimum.accumulate(p), dtype=np.float64)


def confidence_sequence(
    estimate: float, variance: float, *, mixture_sd: float, alpha: float = 0.05
) -> tuple[float, float]:
    """``{theta: Lambda_n(theta) < 1 / alpha}`` — an interval valid at every ``n``
    simultaneously: ``theta_hat +/- sqrt(V (V + tau^2) / tau^2 * (2 log(1/alpha) +
    log((V + tau^2) / V)))``. Intersecting the intervals over looks is also valid."""
    alpha = require_probability(alpha, "alpha")
    if variance <= 0 or mixture_sd <= 0:
        raise ValueError("variance and mixture_sd must be positive")
    tau2 = mixture_sd**2
    radius = math.sqrt(variance * (variance + tau2) / tau2
                       * (2 * math.log(1 / alpha) + math.log((variance + tau2) / variance)))
    return estimate - radius, estimate + radius


__all__ = [
    "ConstantShape",
    "CrossingProbabilities",
    "GroupSequentialDesign",
    "OperatingCharacteristics",
    "SequentialLook",
    "SequentialMonitoringResult",
    "SpendingFamily",
    "always_valid_p_values",
    "confidence_sequence",
    "constant_boundaries",
    "crossing_probabilities",
    "fixed_sample_drift",
    "maximum_information_inflation",
    "monitor",
    "msprt_log_likelihood_ratio",
    "naive_peeking_design",
    "operating_characteristics",
    "repeated_confidence_interval",
    "spending_boundaries",
    "spending_function",
]
