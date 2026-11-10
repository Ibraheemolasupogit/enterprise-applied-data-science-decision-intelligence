"""Statistical power, sample size and minimum detectable effect.

Power is ``P(reject H0 | a specified true effect)``. It is a property of a
*design* (sample size, alpha, the effect you care about), computed before
data are collected. Two misuses are explicitly out of bounds here:

* **Post-hoc ("observed") power** — plugging the observed effect into a
  power formula after the fact — is a monotone function of the p-value
  and adds no information. These functions take the effect size as a
  design input, never from the data being analysed.
* **Powering for the effect you hope for** rather than the smallest effect
  that would change the decision. The effect size should be the minimum
  decision-relevant effect.

Means use the exact noncentral-t distribution (not the normal
approximation, which overstates power at small ``n``). Proportions use the
standard normal approximation with the null variance under H0 and the
alternative variance under H1; it is accurate when every expected cell
count is reasonably large.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from scipy import optimize, stats

from decision_intelligence.common.validation import (
    require_alternative,
    require_positive_int,
    require_probability,
)

_MAX_SAMPLE_SIZE = 100_000_000


def _t_power(ncp: float, df: float, alpha: float, alternative: str) -> float:
    alt = require_alternative(alternative)
    if alt == "two-sided":
        critical = float(stats.t.ppf(1 - alpha / 2, df))
        return float(stats.nct.sf(critical, df, ncp) + stats.nct.cdf(-critical, df, ncp))
    critical = float(stats.t.ppf(1 - alpha, df))
    if alt == "greater":
        return float(stats.nct.sf(critical, df, ncp))
    return float(stats.nct.cdf(-critical, df, ncp))


def power_two_sample_t(
    effect_size: float,
    n_x: int,
    n_y: int | None = None,
    *,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> float:
    """Power of the pooled two-sample t-test for standardised effect ``delta``.

    Noncentrality ``delta * sqrt(n_x n_y / (n_x + n_y))``, df ``n_x + n_y - 2``.
    ``n_y`` defaults to ``n_x``. With ``effect_size=0`` this returns ``alpha``.
    """
    n_x = require_positive_int(n_x, "n_x", minimum=2)
    n_y = n_x if n_y is None else require_positive_int(n_y, "n_y", minimum=2)
    alpha = require_probability(alpha, "alpha")
    _require_finite(effect_size, "effect_size")
    ncp = effect_size * math.sqrt(n_x * n_y / (n_x + n_y))
    return _t_power(ncp, n_x + n_y - 2, alpha, alternative)


def power_one_sample_t(
    effect_size: float, n: int, *, alpha: float = 0.05, alternative: str = "two-sided"
) -> float:
    """Power of the one-sample (or paired, with ``d_z``) t-test.

    Noncentrality ``delta * sqrt(n)``, df ``n - 1``.
    """
    n = require_positive_int(n, "n", minimum=2)
    alpha = require_probability(alpha, "alpha")
    _require_finite(effect_size, "effect_size")
    return _t_power(effect_size * math.sqrt(n), n - 1, alpha, alternative)


def sample_size_two_sample_t(
    effect_size: float,
    *,
    power: float = 0.8,
    alpha: float = 0.05,
    ratio: float = 1.0,
    alternative: str = "two-sided",
) -> tuple[int, int]:
    """Smallest ``(n_x, n_y)`` with ``n_y = ceil(ratio * n_x)`` achieving ``power``.

    Searches integer ``n_x`` directly against the exact noncentral-t power,
    so the returned design genuinely attains the target (a continuous
    solution rounded up can miss it for unequal allocation).
    """
    power = require_probability(power, "power")
    alpha = require_probability(alpha, "alpha")
    _require_nonzero_effect(effect_size, alternative)
    if not ratio > 0:
        raise ValueError("ratio must be positive")

    def achieved(n_x: int) -> float:
        n_y = max(2, math.ceil(ratio * n_x))
        return power_two_sample_t(effect_size, n_x, n_y, alpha=alpha, alternative=alternative)

    n_x = _smallest_n(achieved, power, start=2)
    return n_x, max(2, math.ceil(ratio * n_x))


def sample_size_one_sample_t(
    effect_size: float,
    *,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> int:
    """Smallest ``n`` for the one-sample/paired t-test to reach ``power``."""
    power = require_probability(power, "power")
    alpha = require_probability(alpha, "alpha")
    _require_nonzero_effect(effect_size, alternative)
    return _smallest_n(
        lambda n: power_one_sample_t(effect_size, n, alpha=alpha, alternative=alternative),
        power,
        start=2,
    )


def minimum_detectable_effect_two_sample_t(
    n_x: int,
    n_y: int | None = None,
    *,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> float:
    """Smallest standardised effect ``delta > 0`` detectable with ``power`` at these sizes.

    For ``alternative="less"`` the returned magnitude applies to a negative
    effect.
    """
    power = require_probability(power, "power")
    alpha = require_probability(alpha, "alpha")
    if power <= alpha:
        raise ValueError("power must exceed alpha")
    sign = -1.0 if alternative == "less" else 1.0

    def gap(delta: float) -> float:
        return (
            power_two_sample_t(sign * delta, n_x, n_y, alpha=alpha, alternative=alternative)
            - power
        )

    high = 1.0
    while gap(high) < 0:
        high *= 2
        if high > 1e6:
            raise ValueError("no finite effect reaches the requested power")
    return float(optimize.brentq(gap, 0.0, high, xtol=1e-10))


def power_two_proportions(
    p_x: float,
    p_y: float,
    n_x: int,
    n_y: int | None = None,
    *,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> float:
    """Approximate power of the pooled two-proportion z-test.

    ``power = Phi((|p_x - p_y| - z_crit * SE_0) / SE_1)`` (two-sided adds the
    negligible opposite tail), with ``SE_0`` the pooled null SE at the
    weighted average proportion and ``SE_1`` the unpooled SE under H1.
    """
    p_x = require_probability(p_x, "p_x")
    p_y = require_probability(p_y, "p_y")
    n_x = require_positive_int(n_x, "n_x")
    n_y = n_x if n_y is None else require_positive_int(n_y, "n_y")
    alpha = require_probability(alpha, "alpha")
    alt = require_alternative(alternative)
    pooled = (n_x * p_x + n_y * p_y) / (n_x + n_y)
    se_null = math.sqrt(pooled * (1 - pooled) * (1 / n_x + 1 / n_y))
    se_alt = math.sqrt(p_x * (1 - p_x) / n_x + p_y * (1 - p_y) / n_y)
    difference = p_x - p_y
    if alt == "two-sided":
        z = float(stats.norm.ppf(1 - alpha / 2))
        return float(
            stats.norm.cdf((difference - z * se_null) / se_alt)
            + stats.norm.cdf((-difference - z * se_null) / se_alt)
        )
    z = float(stats.norm.ppf(1 - alpha))
    if alt == "greater":
        return float(stats.norm.cdf((difference - z * se_null) / se_alt))
    return float(stats.norm.cdf((-difference - z * se_null) / se_alt))


def sample_size_two_proportions(
    p_x: float,
    p_y: float,
    *,
    power: float = 0.8,
    alpha: float = 0.05,
    ratio: float = 1.0,
    alternative: str = "two-sided",
) -> tuple[int, int]:
    """Smallest ``(n_x, n_y)``, ``n_y = ceil(ratio * n_x)``, reaching ``power`` (normal approx)."""
    power = require_probability(power, "power")
    if p_x == p_y:
        raise ValueError("p_x equals p_y: no sample size detects a zero difference")
    if not ratio > 0:
        raise ValueError("ratio must be positive")
    _require_nonzero_effect(p_x - p_y, alternative)

    def achieved(n_x: int) -> float:
        return power_two_proportions(
            p_x, p_y, n_x, max(1, math.ceil(ratio * n_x)), alpha=alpha, alternative=alternative
        )

    n_x = _smallest_n(achieved, power, start=2)
    return n_x, max(1, math.ceil(ratio * n_x))


def _smallest_n(func: Callable[[int], float], target: float, *, start: int) -> int:
    """Smallest integer ``n >= start`` with ``achieved(n) >= target``.

    Exponential search then bisection; assumes power is non-decreasing in
    ``n`` (true for every design in this module).
    """
    if func(start) >= target:
        return start
    low, high = start, start * 2
    while func(high) < target:
        low, high = high, high * 2
        if high > _MAX_SAMPLE_SIZE:
            raise ValueError("required sample size exceeds 100 million; effect too small")
    while high - low > 1:
        middle = (low + high) // 2
        if func(middle) >= target:
            high = middle
        else:
            low = middle
    return high


def _require_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _require_nonzero_effect(effect_size: float, alternative: str) -> None:
    _require_finite(effect_size, "effect_size")
    alt = require_alternative(alternative)
    if effect_size == 0:
        raise ValueError("effect_size must be non-zero to compute a sample size")
    if alt == "greater" and effect_size < 0 or alt == "less" and effect_size > 0:
        raise ValueError("effect_size sign contradicts the one-sided alternative")
