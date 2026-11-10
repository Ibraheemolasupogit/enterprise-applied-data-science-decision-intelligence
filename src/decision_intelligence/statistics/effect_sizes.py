"""Effect sizes: *how large* a difference is, with uncertainty.

A p-value conflates effect size with sample size; an effect size with a
confidence interval separates them. Which effect size is decision-relevant
depends on the decision: an absolute risk difference (percentage points)
drives expected-value calculations, a ratio is portable across baselines,
and a standardised mean difference (Cohen's d) is useful mainly for power
analysis and comparing across differently scaled outcomes — it is not, by
itself, a business quantity. Cohen's "small/medium/large" labels (0.2/0.5/0.8)
are generic conventions, not decision thresholds.

The quantities here are descriptive comparisons between groups. They
estimate a *causal* effect only when the groups were formed by
randomisation (or an identification strategy from M4/M5 holds).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize, special, stats

from decision_intelligence.common.validation import (
    as_float_array,
    require_positive_int,
    require_probability,
)
from decision_intelligence.statistics.hypothesis_tests import two_proportion_z_test


@dataclass(frozen=True)
class EffectSize:
    """A named effect size with an optional confidence interval."""

    name: str
    estimate: float
    ci_lower: float | None = None
    ci_upper: float | None = None
    confidence_level: float | None = None
    note: str = ""


def hedges_correction(df: float) -> float:
    """Exact small-sample bias correction ``J(df) = Gamma(df/2) / (sqrt(df/2) Gamma((df-1)/2))``.

    ``E[d] ~= delta / J``, so ``g = J d`` is (nearly) unbiased for the
    population standardised difference ``delta``. ``J -> 1`` as ``df`` grows.
    """
    if df <= 1:
        raise ValueError("df must exceed 1")
    return math.exp(
        special.gammaln(df / 2) - 0.5 * math.log(df / 2) - special.gammaln((df - 1) / 2)
    )


_EXACT_NOTE = "CI by noncentral-t inversion"
_FALLBACK_NOTE = (
    "CI by large-sample normal approximation (noncentral-t CDF numerically unavailable "
    "for this t and df)"
)
_SMALL_DF_CAVEAT = "; the normal approximation is poor at small df, so treat this CI as rough"


class _NonFiniteCDF(Exception):
    """SciPy returned NaN for the noncentral t CDF by every available route."""


def _noncentral_t_cdf(t_observed: float, df: float, ncp: float) -> float:
    """``F(t; df, ncp)``, using the exact identity ``F(t; df, ncp) = S(-t; df, -ncp)``
    when SciPy's direct CDF evaluation returns NaN (it does for some small-df,
    large-|t| combinations where the survival-function route is finite)."""
    value = float(stats.nct.cdf(t_observed, df, ncp))
    if math.isnan(value):
        value = float(stats.nct.sf(-t_observed, df, -ncp))
    if math.isnan(value):
        raise _NonFiniteCDF
    return value


def _noncentrality_interval(
    t_observed: float, df: float, confidence_level: float
) -> tuple[float, float, str]:
    """Interval for the noncentrality parameter of an observed t statistic.

    Primary method: invert the noncentral t CDF in its noncentrality
    parameter (exact under the model). The root is bracketed around the
    large-sample approximation ``ncp ~ N(t, 1 + t^2 / (2 df))`` and the CDF is
    evaluated through :func:`_noncentral_t_cdf`. Only if SciPy cannot
    evaluate the CDF by either route is the normal approximation returned,
    and the note says so (with an explicit caveat below 30 df, where that
    approximation is poor).
    """
    alpha = 1 - confidence_level
    approx_sd = math.sqrt(1 + t_observed**2 / (2 * df))
    z = float(stats.norm.ppf(1 - alpha / 2))
    note = _FALLBACK_NOTE + (_SMALL_DF_CAVEAT if df < 30 else "")
    fallback = (t_observed - z * approx_sd, t_observed + z * approx_sd, note)

    def equation(ncp: float, target: float) -> float:
        return _noncentral_t_cdf(t_observed, df, ncp) - target

    def solve(target: float) -> float:
        low, high = t_observed - 4 * approx_sd, t_observed + 4 * approx_sd
        for _ in range(60):
            f_low, f_high = equation(low, target), equation(high, target)
            if f_low >= 0 >= f_high:
                return float(optimize.brentq(equation, low, high, args=(target,), xtol=1e-10))
            if f_low < 0:
                low -= approx_sd
            if f_high > 0:
                high += approx_sd
        raise _NonFiniteCDF

    try:
        return solve(1 - alpha / 2), solve(alpha / 2), _EXACT_NOTE
    except _NonFiniteCDF:
        return fallback


def cohens_d(x: ArrayLike, y: ArrayLike, *, confidence_level: float = 0.95) -> EffectSize:
    """Cohen's d with pooled SD: ``(xbar - ybar) / s_pooled``.

    The CI is for the population ``delta`` by noncentral-t inversion —
    exact under normality with equal variances, approximate otherwise.
    """
    a = as_float_array(x, name="x", min_size=2)
    b = as_float_array(y, name="y", min_size=2)
    confidence_level = require_probability(confidence_level, "confidence_level")
    n1, n2 = a.size, b.size
    df = n1 + n2 - 2
    pooled_sd = math.sqrt(
        ((n1 - 1) * float(a.var(ddof=1)) + (n2 - 1) * float(b.var(ddof=1))) / df
    )
    if pooled_sd == 0.0:
        raise ValueError("standardised difference undefined: zero pooled standard deviation")
    d = float(a.mean() - b.mean()) / pooled_sd
    scale = math.sqrt(1 / n1 + 1 / n2)
    low_ncp, high_ncp, method_note = _noncentrality_interval(d / scale, df, confidence_level)
    return EffectSize(
        name="Cohen's d (pooled SD)",
        estimate=d,
        ci_lower=low_ncp * scale,
        ci_upper=high_ncp * scale,
        confidence_level=confidence_level,
        note=f"{method_note}; assumes normality and equal variances",
    )


def hedges_g(x: ArrayLike, y: ArrayLike, *, confidence_level: float = 0.95) -> EffectSize:
    """Hedges' g: Cohen's d times the exact small-sample correction ``J``.

    Shares Cohen's d's noncentral-t interval for ``delta`` (the correction
    targets the point estimate's bias; the interval is already exact).
    """
    d = cohens_d(x, y, confidence_level=confidence_level)
    n_total = np.asarray(x).size + np.asarray(y).size
    return EffectSize(
        name="Hedges' g",
        estimate=hedges_correction(n_total - 2) * d.estimate,
        ci_lower=d.ci_lower,
        ci_upper=d.ci_upper,
        confidence_level=confidence_level,
        note=f"bias-corrected d; CI for delta from Cohen's d ({d.note})",
    )


def cohens_dz(x: ArrayLike, y: ArrayLike, *, confidence_level: float = 0.95) -> EffectSize:
    """Paired standardised difference ``mean(diff) / sd(diff)`` with noncentral-t CI.

    Not comparable to independent-groups d: it is scaled by the SD of the
    differences, which shrinks as the within-pair correlation grows.
    """
    a = as_float_array(x, name="x", min_size=2)
    b = as_float_array(y, name="y", min_size=2)
    if a.shape != b.shape:
        raise ValueError("paired samples must have the same length")
    confidence_level = require_probability(confidence_level, "confidence_level")
    diff = a - b
    sd = float(diff.std(ddof=1))
    if sd == 0.0:
        raise ValueError("paired standardised difference undefined: constant differences")
    dz = float(diff.mean()) / sd
    scale = 1 / math.sqrt(diff.size)
    low_ncp, high_ncp, method_note = _noncentrality_interval(
        dz / scale, diff.size - 1, confidence_level
    )
    return EffectSize(
        name="Cohen's d_z (paired)",
        estimate=dz,
        ci_lower=low_ncp * scale,
        ci_upper=high_ncp * scale,
        confidence_level=confidence_level,
        note=f"{method_note}; scaled by SD of differences, not comparable to "
        "independent-groups d",
    )


def cliffs_delta(x: ArrayLike, y: ArrayLike) -> EffectSize:
    """Cliff's delta ``P(X > Y) - P(X < Y)`` in [-1, 1] (equals the rank-biserial correlation).

    Ordinal and robust to outliers and monotone transformations; the
    natural effect size to report alongside a Mann–Whitney test.
    """
    a = as_float_array(x, name="x", min_size=1)
    b = as_float_array(y, name="y", min_size=1)
    combined = np.concatenate([a, b])
    ranks = stats.rankdata(combined)
    u1 = float(ranks[: a.size].sum() - a.size * (a.size + 1) / 2)
    return EffectSize(
        name="Cliff's delta",
        estimate=2 * u1 / (a.size * b.size) - 1,
        note="P(X > Y) - P(X < Y); ordinal, scale-free",
    )


@dataclass(frozen=True)
class BinaryEffectSizes:
    """Absolute and relative comparisons of two proportions (group x vs group y)."""

    risk_difference: EffectSize
    risk_ratio: EffectSize
    odds_ratio: EffectSize
    cohens_h: EffectSize
    zero_cell_correction_applied: bool


def binary_effect_sizes(
    successes_x: int,
    n_x: int,
    successes_y: int,
    n_y: int,
    *,
    confidence_level: float = 0.95,
) -> BinaryEffectSizes:
    """Risk difference (Newcombe CI), risk ratio (Katz log CI), odds ratio (Woolf log CI), h.

    When any cell of the 2x2 table is zero, the ratio measures are computed
    with the Haldane–Anscombe correction (0.5 added to every cell) and the
    flag ``zero_cell_correction_applied`` is set — the uncorrected ratio is
    0 or infinite and its log-scale interval undefined.
    """
    for name, (s, n) in {"x": (successes_x, n_x), "y": (successes_y, n_y)}.items():
        require_positive_int(n, f"n_{name}")
        require_positive_int(s, f"successes_{name}", minimum=0)
        if s > n:
            raise ValueError(f"successes_{name} cannot exceed n_{name}")
    confidence_level = require_probability(confidence_level, "confidence_level")
    z = float(stats.norm.ppf(0.5 + confidence_level / 2))

    difference = two_proportion_z_test(
        successes_x, n_x, successes_y, n_y, confidence_level=confidence_level
    )

    a, b = float(successes_x), float(n_x - successes_x)
    c, d = float(successes_y), float(n_y - successes_y)
    corrected = min(a, b, c, d) == 0
    if corrected:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5

    risk_x, risk_y = a / (a + b), c / (c + d)
    log_rr = math.log(risk_x / risk_y)
    se_log_rr = math.sqrt(1 / a - 1 / (a + b) + 1 / c - 1 / (c + d))
    log_or = math.log((a * d) / (b * c))
    se_log_or = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)

    p1, p2 = successes_x / n_x, successes_y / n_y
    h = 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))
    note = " (Haldane–Anscombe 0.5 correction applied)" if corrected else ""
    return BinaryEffectSizes(
        risk_difference=EffectSize(
            name="risk difference (p_x - p_y)",
            estimate=difference.estimate,
            ci_lower=difference.ci_lower,
            ci_upper=difference.ci_upper,
            confidence_level=confidence_level,
            note="Newcombe hybrid score interval",
        ),
        risk_ratio=EffectSize(
            name="risk ratio (p_x / p_y)",
            estimate=math.exp(log_rr),
            ci_lower=math.exp(log_rr - z * se_log_rr),
            ci_upper=math.exp(log_rr + z * se_log_rr),
            confidence_level=confidence_level,
            note="Katz log interval" + note,
        ),
        odds_ratio=EffectSize(
            name="odds ratio",
            estimate=math.exp(log_or),
            ci_lower=math.exp(log_or - z * se_log_or),
            ci_upper=math.exp(log_or + z * se_log_or),
            confidence_level=confidence_level,
            note="Woolf log interval" + note,
        ),
        cohens_h=EffectSize(
            name="Cohen's h",
            estimate=h,
            note="arcsine-transformed difference; used for power analysis",
        ),
        zero_cell_correction_applied=corrected,
    )
