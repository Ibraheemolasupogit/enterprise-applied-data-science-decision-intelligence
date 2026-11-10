"""Parametric and non-parametric hypothesis tests with explicit assumptions.

Every test returns a :class:`HypothesisTestResult` carrying not only the
statistic and p-value but the estimand, a confidence interval where one is
defined, the assumptions the p-value relies on, and any data-specific
warnings (small expected counts, ties, dropped zero differences). The
intent is that a result can never be read without its conditions.

Interpretation rules that apply to every function here:

* A p-value is ``P(data at least this extreme | H0 and the assumptions)``.
  It is not the probability that H0 is true, and ``p > alpha`` is not
  evidence that the effect is zero — read the confidence interval.
* Statistical significance is not practical significance: pair every
  test with an effect size (:mod:`.effect_sizes`) and a decision-relevant
  threshold.
* A test chosen *after* inspecting the data (e.g. "normality test failed,
  so use Mann–Whitney") changes the procedure's error rates. Choose the
  test from the design and the estimand of interest beforehand. In
  particular, Welch's t-test (the default here) is preferred over
  "test variances, then pick Student or Welch".
* Mann–Whitney and Wilcoxon test *different hypotheses* from the t-tests
  (stochastic ordering / symmetry about zero, not equality of means). They
  are not drop-in "assumption-free" substitutes for a mean comparison.

Confidence intervals: for ``alternative="two-sided"`` a two-sided interval
at ``confidence_level`` is returned; for one-sided alternatives the
matching one-sided bound is returned with the other end infinite, so the
interval and the test agree.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from decision_intelligence.common.validation import (
    Alternative,
    FloatArray,
    as_float_array,
    require_alternative,
    require_positive_int,
    require_probability,
)

_EXACT_MAX_GROUP_SIZE = 50


@dataclass(frozen=True)
class HypothesisTestResult:
    """Outcome of one hypothesis test, with the context needed to read it."""

    method: str
    estimand: str
    estimate: float
    statistic: float
    p_value: float
    alternative: Alternative
    sample_sizes: tuple[int, ...]
    df: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    confidence_level: float | None = None
    assumptions: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _p_value_from_distribution(
    statistic: float, alternative: Alternative, sf: Callable[[float], float],
    cdf: Callable[[float], float],
) -> float:
    if alternative == "greater":
        return float(sf(statistic))
    if alternative == "less":
        return float(cdf(statistic))
    return float(min(1.0, 2.0 * sf(abs(statistic))))


def _interval(
    estimate: float,
    standard_error: float,
    critical: Callable[[float], float],
    alternative: Alternative,
    confidence_level: float,
) -> tuple[float, float]:
    """Wald-type interval ``estimate +- q * se`` respecting the alternative."""
    if alternative == "two-sided":
        q = critical(0.5 + confidence_level / 2)
        return estimate - q * standard_error, estimate + q * standard_error
    q = critical(confidence_level)
    if alternative == "greater":
        return estimate - q * standard_error, math.inf
    return -math.inf, estimate + q * standard_error


# ---------------------------------------------------------------------------
# t-tests
# ---------------------------------------------------------------------------

_T_ASSUMPTIONS = (
    "observations are independent",
    "sample mean(s) approximately normal: exact for normal data, otherwise by the CLT "
    "(inaccurate for small n from skewed or heavy-tailed populations)",
)


def one_sample_t_test(
    values: ArrayLike,
    *,
    popmean: float = 0.0,
    alternative: str = "two-sided",
    confidence_level: float = 0.95,
) -> HypothesisTestResult:
    """Test ``H0: mu = popmean`` with ``t = (xbar - popmean) / (s / sqrt(n))``, df ``n - 1``."""
    x = as_float_array(values, name="values", min_size=2)
    alt = require_alternative(alternative)
    confidence_level = require_probability(confidence_level, "confidence_level")
    sd = float(x.std(ddof=1))
    if sd == 0.0:
        raise ValueError("t-test undefined for constant data (zero sample variance)")
    n = x.size
    df = n - 1
    mean = float(x.mean())
    se = sd / math.sqrt(n)
    t = (mean - popmean) / se
    lower, upper = _interval(
        mean, se, lambda q: float(stats.t.ppf(q, df)), alt, confidence_level
    )
    return HypothesisTestResult(
        method="one-sample t-test",
        estimand="population mean",
        estimate=mean,
        statistic=t,
        p_value=_p_value_from_distribution(
            t, alt, lambda v: float(stats.t.sf(v, df)), lambda v: float(stats.t.cdf(v, df))
        ),
        alternative=alt,
        sample_sizes=(n,),
        df=float(df),
        ci_lower=lower,
        ci_upper=upper,
        confidence_level=confidence_level,
        assumptions=_T_ASSUMPTIONS,
    )


def two_sample_t_test(
    x: ArrayLike,
    y: ArrayLike,
    *,
    equal_var: bool = False,
    alternative: str = "two-sided",
    confidence_level: float = 0.95,
) -> HypothesisTestResult:
    """Independent two-sample t-test of ``H0: mu_x = mu_y``.

    ``equal_var=False`` (default) is Welch's test with Welch–Satterthwaite
    degrees of freedom; it loses almost nothing when variances are equal
    and stays valid when they are not. ``equal_var=True`` is Student's
    pooled test, whose type I error can be badly wrong when the smaller
    group has the larger variance.
    """
    a = as_float_array(x, name="x", min_size=2)
    b = as_float_array(y, name="y", min_size=2)
    alt = require_alternative(alternative)
    confidence_level = require_probability(confidence_level, "confidence_level")
    n1, n2 = a.size, b.size
    v1, v2 = float(a.var(ddof=1)), float(b.var(ddof=1))
    if v1 == 0.0 and v2 == 0.0:
        raise ValueError("t-test undefined: both samples are constant (zero variance)")
    difference = float(a.mean() - b.mean())

    if equal_var:
        df = float(n1 + n2 - 2)
        pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
        se = math.sqrt(pooled * (1 / n1 + 1 / n2))
        method = "Student two-sample t-test (pooled variance)"
        assumptions = (*_T_ASSUMPTIONS, "equal population variances")
    else:
        q1, q2 = v1 / n1, v2 / n2
        se = math.sqrt(q1 + q2)
        df = (q1 + q2) ** 2 / (q1**2 / (n1 - 1) + q2**2 / (n2 - 1))
        method = "Welch two-sample t-test"
        assumptions = (*_T_ASSUMPTIONS, "the two samples are independent of each other")

    t = difference / se
    lower, upper = _interval(
        difference, se, lambda q: float(stats.t.ppf(q, df)), alt, confidence_level
    )
    return HypothesisTestResult(
        method=method,
        estimand="difference in population means (x - y)",
        estimate=difference,
        statistic=t,
        p_value=_p_value_from_distribution(
            t, alt, lambda v: float(stats.t.sf(v, df)), lambda v: float(stats.t.cdf(v, df))
        ),
        alternative=alt,
        sample_sizes=(n1, n2),
        df=df,
        ci_lower=lower,
        ci_upper=upper,
        confidence_level=confidence_level,
        assumptions=assumptions,
    )


def paired_t_test(
    x: ArrayLike,
    y: ArrayLike,
    *,
    alternative: str = "two-sided",
    confidence_level: float = 0.95,
) -> HypothesisTestResult:
    """Paired t-test: a one-sample t-test on the within-pair differences ``x - y``.

    Valid only when observations are genuinely paired (same unit measured
    twice, matched pairs). Treating paired data as independent wastes the
    within-pair correlation; treating independent data as paired is invalid.
    """
    a = as_float_array(x, name="x", min_size=2)
    b = as_float_array(y, name="y", min_size=2)
    if a.shape != b.shape:
        raise ValueError("paired samples must have the same length")
    result = one_sample_t_test(
        a - b, popmean=0.0, alternative=alternative, confidence_level=confidence_level
    )
    return HypothesisTestResult(
        method="paired t-test",
        estimand="mean of within-pair differences (x - y)",
        estimate=result.estimate,
        statistic=result.statistic,
        p_value=result.p_value,
        alternative=result.alternative,
        sample_sizes=(a.size,),
        df=result.df,
        ci_lower=result.ci_lower,
        ci_upper=result.ci_upper,
        confidence_level=confidence_level,
        assumptions=(
            "pairs are independent of each other",
            "mean difference approximately normal (exact if differences are normal)",
        ),
    )


# ---------------------------------------------------------------------------
# Proportions and contingency tables
# ---------------------------------------------------------------------------


def wilson_interval(
    successes: int, n: int, *, confidence_level: float = 0.95
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the Wald interval ``p +- z sqrt(p(1-p)/n)``, which has
    poor coverage near 0 or 1 and collapses to a point when ``p`` is 0 or 1.
    """
    n = require_positive_int(n, "n")
    successes = require_positive_int(successes, "successes", minimum=0)
    if successes > n:
        raise ValueError("successes cannot exceed n")
    confidence_level = require_probability(confidence_level, "confidence_level")
    z = float(stats.norm.ppf(0.5 + confidence_level / 2))
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def two_proportion_z_test(
    successes_x: int,
    n_x: int,
    successes_y: int,
    n_y: int,
    *,
    alternative: str = "two-sided",
    confidence_level: float = 0.95,
) -> HypothesisTestResult:
    """Pooled two-proportion z-test of ``H0: p_x = p_y`` with a Newcombe interval.

    The test uses the pooled SE (correct under H0); the interval for
    ``p_x - p_y`` is Newcombe's hybrid score interval (method 10), which
    has far better small-sample coverage than the Wald interval. Because
    the two are built differently they can, rarely and only near the
    boundary, disagree about whether 0 is excluded.
    """
    alt = require_alternative(alternative)
    confidence_level = require_probability(confidence_level, "confidence_level")
    for name, (s, n) in {"x": (successes_x, n_x), "y": (successes_y, n_y)}.items():
        require_positive_int(n, f"n_{name}")
        require_positive_int(s, f"successes_{name}", minimum=0)
        if s > n:
            raise ValueError(f"successes_{name} cannot exceed n_{name}")

    p1, p2 = successes_x / n_x, successes_y / n_y
    difference = p1 - p2
    pooled = (successes_x + successes_y) / (n_x + n_y)
    warnings: list[str] = []
    expected_counts = (n_x * pooled, n_x * (1 - pooled), n_y * pooled, n_y * (1 - pooled))
    if min(expected_counts) < 5:
        warnings.append(
            "an expected cell count is below 5; the normal approximation is unreliable — "
            "prefer fisher_exact_2x2"
        )

    se_pooled = math.sqrt(pooled * (1 - pooled) * (1 / n_x + 1 / n_y))
    if se_pooled == 0.0:
        # Both groups all-success or all-failure: no evidence of any difference.
        warnings.append("no variation in either group; test statistic undefined, p set to 1")
        z, p_value = 0.0, 1.0
    else:
        z = difference / se_pooled
        p_value = _p_value_from_distribution(
            z, alt, lambda v: float(stats.norm.sf(v)), lambda v: float(stats.norm.cdf(v))
        )

    level = confidence_level if alt == "two-sided" else 2 * confidence_level - 1
    l1, u1 = wilson_interval(successes_x, n_x, confidence_level=level)
    l2, u2 = wilson_interval(successes_y, n_y, confidence_level=level)
    lower = difference - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    upper = difference + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    if alt == "greater":
        upper = math.inf
    elif alt == "less":
        lower = -math.inf

    return HypothesisTestResult(
        method="two-proportion z-test (pooled) with Newcombe hybrid score interval",
        estimand="difference in proportions (p_x - p_y)",
        estimate=difference,
        statistic=z,
        p_value=p_value,
        alternative=alt,
        sample_sizes=(n_x, n_y),
        ci_lower=lower,
        ci_upper=upper,
        confidence_level=confidence_level,
        assumptions=(
            "independent Bernoulli trials within and between groups",
            "large-sample normal approximation for the test statistic",
        ),
        warnings=tuple(warnings),
    )


def fisher_exact_2x2(
    table: ArrayLike, *, alternative: str = "two-sided"
) -> HypothesisTestResult:
    """Fisher's exact test for a 2x2 table (thin wrapper over SciPy's validated implementation).

    Conditions on both margins; exact at any sample size, conservative for
    two-sided tests. The reported estimate is the sample (unconditional)
    odds ratio ``(a d) / (b c)`` (may be 0 or infinite).
    """
    counts = _validate_table(table)
    if counts.shape != (2, 2):
        raise ValueError("fisher_exact_2x2 requires a 2x2 table")
    alt = require_alternative(alternative)
    result = stats.fisher_exact(counts, alternative=alt)
    return HypothesisTestResult(
        method="Fisher exact test",
        estimand="odds ratio (row 1 vs row 2, column 1 as 'success')",
        estimate=float(result.statistic),
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        alternative=alt,
        sample_sizes=(int(counts.sum()),),
        assumptions=("independent observations", "inference conditional on both table margins"),
    )


def _validate_table(table: ArrayLike) -> FloatArray:
    counts = np.asarray(table, dtype=np.float64)
    if counts.ndim != 2 or min(counts.shape) < 2:
        raise ValueError("contingency table must be two-dimensional with at least 2x2 cells")
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("contingency table must contain finite, non-negative counts")
    if not np.allclose(counts, np.round(counts)):
        raise ValueError("contingency table must contain integer counts")
    if (counts.sum(axis=0) == 0).any() or (counts.sum(axis=1) == 0).any():
        raise ValueError("contingency table has an all-zero row or column")
    return counts


def expected_frequencies(table: ArrayLike) -> FloatArray:
    """Expected counts under independence: ``E_ij = row_i * col_j / N``."""
    counts = _validate_table(table)
    return np.asarray(
        np.outer(counts.sum(axis=1), counts.sum(axis=0)) / counts.sum(), dtype=np.float64
    )


def chi_square_independence(table: ArrayLike) -> HypothesisTestResult:
    """Pearson chi-square test of independence for an r x c table (no continuity correction).

    The estimate is Cramér's V ``sqrt(X^2 / (N (min(r, c) - 1)))`` — a
    measure of association strength in [0, 1]; the p-value alone says
    nothing about strength, and with large ``N`` trivially weak
    associations become "significant".
    """
    counts = _validate_table(table)
    expected = expected_frequencies(counts)
    statistic = float(np.sum((counts - expected) ** 2 / expected))
    rows, cols = counts.shape
    df = (rows - 1) * (cols - 1)
    total = float(counts.sum())
    warnings: list[str] = []
    if (expected < 1).any() or np.mean(expected < 5) > 0.2:
        warnings.append(
            "expected counts too small (some < 1 or more than 20% < 5); the chi-square "
            "approximation is unreliable — combine categories or use an exact test"
        )
    return HypothesisTestResult(
        method="Pearson chi-square test of independence",
        estimand="Cramér's V (strength of association)",
        estimate=math.sqrt(statistic / (total * (min(rows, cols) - 1))),
        statistic=statistic,
        p_value=float(stats.chi2.sf(statistic, df)),
        alternative="two-sided",
        sample_sizes=(int(total),),
        df=float(df),
        assumptions=(
            "each observation falls in exactly one cell, independently",
            "expected counts large enough for the chi-square approximation",
        ),
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# Rank-based (non-parametric) tests
# ---------------------------------------------------------------------------


def _tie_sizes(values: FloatArray) -> FloatArray:
    _, counts = np.unique(values, return_counts=True)
    return counts[counts > 1].astype(np.float64)


def _mann_whitney_exact_counts(n1: int, n2: int) -> list[int]:
    """Null frequencies of ``U`` (0..n1*n2): coefficients of the Gaussian binomial [n1+n2, n1]_q.

    Built as prod_{i=1..n1} (1 - q^{n2+i}) / (1 - q^i) in exact integer
    arithmetic, so no floating-point error accumulates in the tail counts.
    """
    size = n1 * n2 + 1
    coefficients = [0] * size
    coefficients[0] = 1
    for i in range(1, n1 + 1):
        shift = n2 + i
        for k in range(size - 1, shift - 1, -1):
            coefficients[k] -= coefficients[k - shift]
        for k in range(i, size):
            coefficients[k] += coefficients[k - i]
    return coefficients


def _signed_rank_exact_counts(n: int) -> list[int]:
    """Null frequencies of the signed-rank statistic ``T+`` (0..n(n+1)/2): prod (1 + q^i)."""
    size = n * (n + 1) // 2 + 1
    coefficients = [0] * size
    coefficients[0] = 1
    for i in range(1, n + 1):
        for k in range(size - 1, i - 1, -1):
            coefficients[k] += coefficients[k - i]
    return coefficients


def _exact_p_value(counts: list[int], observed: int, alternative: Alternative) -> float:
    total = sum(counts)
    upper = sum(counts[observed:]) / total
    lower = sum(counts[: observed + 1]) / total
    if alternative == "greater":
        return upper
    if alternative == "less":
        return lower
    return min(1.0, 2.0 * min(upper, lower))


def mann_whitney_u(
    x: ArrayLike,
    y: ArrayLike,
    *,
    alternative: str = "two-sided",
    method: str = "auto",
    use_continuity: bool = True,
) -> HypothesisTestResult:
    """Mann–Whitney U (Wilcoxon rank-sum) test.

    ``H0``: the two distributions are identical. The test is sensitive to
    ``P(X > Y) != 1/2`` (stochastic dominance), which is the estimand
    reported: ``U_x / (n_x n_y) = P(X > Y) + P(X = Y) / 2``. It is a test of
    a location *shift* only under the additional assumption that the two
    distributions have the same shape.

    ``method``: ``"exact"`` (no ties, both groups <= 50), ``"asymptotic"``
    (normal approximation with tie-corrected variance and optional
    continuity correction) or ``"auto"`` (exact when allowed).
    """
    a = as_float_array(x, name="x", min_size=1)
    b = as_float_array(y, name="y", min_size=1)
    alt = require_alternative(alternative)
    n1, n2 = a.size, b.size
    combined = np.concatenate([a, b])
    ranks = stats.rankdata(combined)
    u1 = float(ranks[:n1].sum() - n1 * (n1 + 1) / 2)
    u2 = n1 * n2 - u1
    ties = _tie_sizes(combined)
    has_ties = ties.size > 0
    exact_allowed = not has_ties and max(n1, n2) <= _EXACT_MAX_GROUP_SIZE

    if method not in ("auto", "exact", "asymptotic"):
        raise ValueError("method must be 'auto', 'exact' or 'asymptotic'")
    if method == "exact" and not exact_allowed:
        raise ValueError(
            f"exact Mann–Whitney requires no ties and group sizes <= {_EXACT_MAX_GROUP_SIZE}"
        )
    use_exact = method == "exact" or (method == "auto" and exact_allowed)
    warnings: list[str] = []
    if has_ties:
        warnings.append(f"{int(ties.sum())} tied observations; tie-corrected variance used")

    if use_exact:
        p_value = _exact_p_value(_mann_whitney_exact_counts(n1, n2), int(round(u1)), alt)
        statistic = u1
        method_name = "Mann–Whitney U test (exact)"
    else:
        total = n1 + n2
        mean_u = n1 * n2 / 2
        tie_term = float(np.sum(ties**3 - ties)) / (total * (total - 1))
        variance = n1 * n2 / 12 * ((total + 1) - tie_term)
        if variance <= 0:
            raise ValueError("Mann–Whitney variance is zero: all observations are tied")
        sd = math.sqrt(variance)
        cc = 0.5 if use_continuity else 0.0
        if alt == "greater":
            p_value = float(stats.norm.sf((u1 - mean_u - cc) / sd))
        elif alt == "less":
            p_value = float(stats.norm.sf((u2 - mean_u - cc) / sd))
        else:
            p_value = min(1.0, 2.0 * float(stats.norm.sf((max(u1, u2) - mean_u - cc) / sd)))
        statistic = u1
        method_name = "Mann–Whitney U test (normal approximation)"
        if min(n1, n2) < 10:
            warnings.append("small group(s) with the normal approximation; p-value approximate")

    return HypothesisTestResult(
        method=method_name,
        estimand="probability of superiority P(X > Y) + P(X = Y)/2",
        estimate=u1 / (n1 * n2),
        statistic=statistic,
        p_value=p_value,
        alternative=alt,
        sample_sizes=(n1, n2),
        assumptions=(
            "observations independent within and between groups",
            "ordinal or continuous outcome; tests stochastic ordering, not means",
        ),
        warnings=tuple(warnings),
    )


def wilcoxon_signed_rank(
    x: ArrayLike,
    y: ArrayLike | None = None,
    *,
    alternative: str = "two-sided",
    method: str = "auto",
    use_continuity: bool = False,
) -> HypothesisTestResult:
    """Wilcoxon signed-rank test on ``x`` (or on paired differences ``x - y``).

    ``H0``: the differences are symmetric about zero. Zero differences are
    dropped (Wilcoxon's convention) and reported in the warnings. The
    estimand is the matched-pairs rank-biserial correlation
    ``(T+ - T-) / (T+ + T-)`` in [-1, 1].
    """
    a = as_float_array(x, name="x", min_size=1)
    if y is not None:
        b = as_float_array(y, name="y", min_size=1)
        if a.shape != b.shape:
            raise ValueError("paired samples must have the same length")
        differences = a - b
    else:
        differences = a
    alt = require_alternative(alternative)
    if method not in ("auto", "exact", "asymptotic"):
        raise ValueError("method must be 'auto', 'exact' or 'asymptotic'")

    warnings: list[str] = []
    n_zero = int(np.sum(differences == 0))
    nonzero = differences[differences != 0]
    if n_zero:
        warnings.append(f"{n_zero} zero difference(s) dropped (Wilcoxon zero_method)")
    n = nonzero.size
    if n == 0:
        raise ValueError("all differences are zero; signed-rank test undefined")

    ranks = stats.rankdata(np.abs(nonzero))
    t_plus = float(ranks[nonzero > 0].sum())
    t_minus = float(ranks[nonzero < 0].sum())
    ties = _tie_sizes(np.abs(nonzero))
    exact_allowed = ties.size == 0 and n <= _EXACT_MAX_GROUP_SIZE
    if method == "exact" and not exact_allowed:
        raise ValueError("exact signed-rank test requires no tied |differences| and n <= 50")
    use_exact = method == "exact" or (method == "auto" and exact_allowed)

    if use_exact:
        p_value = _exact_p_value(_signed_rank_exact_counts(n), int(round(t_plus)), alt)
        method_name = "Wilcoxon signed-rank test (exact)"
    else:
        mean_t = n * (n + 1) / 4
        variance = n * (n + 1) * (2 * n + 1) / 24 - float(np.sum(ties**3 - ties)) / 48
        sd = math.sqrt(variance)
        cc = 0.5 if use_continuity else 0.0
        deviation = t_plus - mean_t
        if alt == "greater":
            z = (deviation - cc) / sd
            p_value = float(stats.norm.sf(z))
        elif alt == "less":
            z = (deviation + cc) / sd
            p_value = float(stats.norm.cdf(z))
        else:
            z = (abs(deviation) - cc) / sd
            p_value = min(1.0, 2.0 * float(stats.norm.sf(z)))
        method_name = "Wilcoxon signed-rank test (normal approximation)"
        if ties.size:
            warnings.append("tied |differences|; tie-corrected variance used")

    return HypothesisTestResult(
        method=method_name,
        estimand="matched-pairs rank-biserial correlation",
        estimate=(t_plus - t_minus) / (t_plus + t_minus),
        statistic=t_plus,
        p_value=p_value,
        alternative=alt,
        sample_sizes=(n,),
        assumptions=(
            "pairs independent of each other",
            "differences symmetric about their centre under H0 (interval-scale differences)",
        ),
        warnings=tuple(warnings),
    )


def permutation_test(
    x: ArrayLike,
    y: ArrayLike,
    *,
    statistic: Callable[[FloatArray, FloatArray], float] | None = None,
    alternative: str = "two-sided",
    n_permutations: int = 9999,
    rng: np.random.Generator,
) -> HypothesisTestResult:
    """Two-sample permutation test (default statistic: difference in means).

    Under ``H0`` (group labels exchangeable — e.g. random assignment with no
    effect) every relabelling is equally likely. Enumerates all
    ``C(n, n_x)`` relabellings when that is at most ``n_permutations``;
    otherwise draws ``n_permutations`` random relabellings and reports the
    valid Monte Carlo p-value ``(b + 1) / (m + 1)``. Exchangeability fails
    when the groups differ in spread but not location, so the test is not
    an exact test of equal means under unequal variances.
    """
    a = as_float_array(x, name="x", min_size=1)
    b = as_float_array(y, name="y", min_size=1)
    alt = require_alternative(alternative)
    n_permutations = require_positive_int(n_permutations, "n_permutations", minimum=99)
    stat: Callable[[FloatArray, FloatArray], float] = statistic or _mean_difference
    combined = np.concatenate([a, b])
    n1, total = a.size, combined.size
    observed = float(stat(a, b))
    tolerance = 1e-12 * max(1.0, abs(observed))

    n_combinations = math.comb(total, n1)
    if n_combinations <= n_permutations:
        null = np.empty(n_combinations, dtype=np.float64)
        indices = np.arange(total)
        for k, chosen in enumerate(itertools.combinations(range(total), n1)):
            mask = np.zeros(total, dtype=bool)
            mask[list(chosen)] = True
            null[k] = stat(combined[indices[mask]], combined[indices[~mask]])
        exact = True
    else:
        null = np.empty(n_permutations, dtype=np.float64)
        for k in range(n_permutations):
            permuted = rng.permutation(combined)
            null[k] = stat(permuted[:n1], permuted[n1:])
        exact = False

    def tail_probability(count: int, size: int) -> float:
        return count / size if exact else (count + 1) / (size + 1)

    greater = tail_probability(int(np.sum(null >= observed - tolerance)), null.size)
    less = tail_probability(int(np.sum(null <= observed + tolerance)), null.size)
    if alt == "greater":
        p_value = greater
    elif alt == "less":
        p_value = less
    else:
        p_value = min(1.0, 2.0 * min(greater, less))

    return HypothesisTestResult(
        method=f"two-sample permutation test ({'exact' if exact else 'Monte Carlo'})",
        estimand="statistic(x, y); default difference in means (x - y)",
        estimate=observed,
        statistic=observed,
        p_value=p_value,
        alternative=alt,
        sample_sizes=(n1, b.size),
        assumptions=("group labels exchangeable under H0",),
    )


def _mean_difference(a: FloatArray, b: FloatArray) -> float:
    return float(a.mean() - b.mean())


# ---------------------------------------------------------------------------
# k-sample tests
# ---------------------------------------------------------------------------


def _validate_groups(groups: Sequence[ArrayLike], min_size: int) -> list[FloatArray]:
    if len(groups) < 2:
        raise ValueError("at least two groups are required")
    return [as_float_array(g, name=f"group {i}", min_size=min_size) for i, g in enumerate(groups)]


def one_way_anova(*groups: ArrayLike) -> HypothesisTestResult:
    """One-way ANOVA F-test of equal group means; estimate is eta-squared ``SSB / SST``.

    Assumes normal errors with *equal variances*; with unequal variances
    and unequal group sizes the F-test's type I error is distorted (Welch's
    ANOVA is then preferable). Eta-squared is upwardly biased in small
    samples.
    """
    samples = _validate_groups(groups, min_size=2)
    all_values = np.concatenate(samples)
    grand_mean = float(all_values.mean())
    k, total = len(samples), all_values.size
    ss_between = float(sum(s.size * (s.mean() - grand_mean) ** 2 for s in samples))
    ss_within = float(sum(np.sum((s - s.mean()) ** 2) for s in samples))
    if ss_within == 0.0:
        raise ValueError("ANOVA undefined: zero within-group variance")
    df_between, df_within = k - 1, total - k
    f = (ss_between / df_between) / (ss_within / df_within)
    return HypothesisTestResult(
        method="one-way ANOVA",
        estimand="eta-squared (share of variance between groups)",
        estimate=ss_between / (ss_between + ss_within),
        statistic=f,
        p_value=float(stats.f.sf(f, df_between, df_within)),
        alternative="two-sided",
        sample_sizes=tuple(s.size for s in samples),
        df=float(df_between),
        assumptions=(
            "independent observations",
            "normally distributed errors",
            "equal variances across groups",
        ),
    )


def kruskal_wallis(*groups: ArrayLike) -> HypothesisTestResult:
    """Kruskal–Wallis H test (rank-based k-sample test), tie-corrected, chi-square approx.

    Estimate is the rank epsilon-squared ``H / (N - 1)``.
    """
    samples = _validate_groups(groups, min_size=1)
    all_values = np.concatenate(samples)
    total = all_values.size
    ranks = stats.rankdata(all_values)
    position = 0
    h = 0.0
    for sample in samples:
        group_ranks = ranks[position : position + sample.size]
        h += group_ranks.sum() ** 2 / sample.size
        position += sample.size
    h = 12.0 / (total * (total + 1)) * h - 3 * (total + 1)
    ties = _tie_sizes(all_values)
    correction = 1.0 - float(np.sum(ties**3 - ties)) / (total**3 - total)
    if correction == 0.0:
        raise ValueError("Kruskal–Wallis undefined: all observations are tied")
    h /= correction
    df = len(samples) - 1
    warnings: tuple[str, ...] = ()
    if min(s.size for s in samples) < 5:
        warnings = ("group(s) smaller than 5; chi-square approximation is rough",)
    return HypothesisTestResult(
        method="Kruskal–Wallis H test",
        estimand="rank epsilon-squared",
        estimate=h / (total - 1),
        statistic=h,
        p_value=float(stats.chi2.sf(h, df)),
        alternative="two-sided",
        sample_sizes=tuple(s.size for s in samples),
        df=float(df),
        assumptions=(
            "independent observations",
            "ordinal or continuous outcome; tests stochastic ordering across groups",
        ),
        warnings=warnings,
    )


def brown_forsythe_levene(*groups: ArrayLike) -> HypothesisTestResult:
    """Brown–Forsythe (median-centred Levene) test of equal variances.

    ANOVA on ``|x_ij - median_j|``; robust to non-normality, unlike
    Bartlett's test. Intended as a diagnostic for describing the data, not
    as a gate for choosing a subsequent test (see module docstring).
    """
    samples = _validate_groups(groups, min_size=2)
    deviations = [np.abs(s - np.median(s)) for s in samples]
    result = one_way_anova(*deviations)
    return HypothesisTestResult(
        method="Brown–Forsythe test of equal variances",
        estimand="eta-squared of absolute deviations from group medians",
        estimate=result.estimate,
        statistic=result.statistic,
        p_value=result.p_value,
        alternative="two-sided",
        sample_sizes=result.sample_sizes,
        df=result.df,
        assumptions=("independent observations",),
    )


def shapiro_wilk(values: ArrayLike) -> HypothesisTestResult:
    """Shapiro–Wilk normality test (SciPy's validated implementation; 3 <= n <= 5000).

    A descriptive diagnostic only. At large ``n`` it rejects for trivial
    departures that do not matter for CLT-based inference; at small ``n``
    it has little power to detect departures that do matter. Prefer a
    Q–Q plot and judgment about the population.
    """
    x = as_float_array(values, name="values", min_size=3)
    if x.size > 5000:
        raise ValueError("Shapiro–Wilk p-values are unreliable above n = 5000")
    if float(x.std()) == 0.0:
        raise ValueError("Shapiro–Wilk undefined for constant data")
    result = stats.shapiro(x)
    return HypothesisTestResult(
        method="Shapiro–Wilk normality test",
        estimand="W statistic (1 = perfectly normal order statistics)",
        estimate=float(result.statistic),
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        alternative="two-sided",
        sample_sizes=(x.size,),
        assumptions=("independent observations",),
    )
