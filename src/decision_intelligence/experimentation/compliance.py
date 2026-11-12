"""Assignment vs treatment received, and missing outcomes (attrition).

**Noncompliance.** Randomisation fixes the *assigned* arm ``Z``; the
treatment actually received ``D`` can differ (users in the treatment arm
never see the feature; control users are exposed through a leak). Only
comparisons by ``Z`` are protected by randomisation:

* **ITT** (by ``Z``) — unbiased for the effect of *assignment*, the effect
  of the policy "offer the feature", diluted by noncompliance;
* **as-treated** (by ``D``) and **per-protocol** (compliant units only) —
  compare groups selected *after* randomisation; biased whenever compliance
  is related to outcomes;
* **CACE / LATE** (Wald: ITT on the outcome / ITT on exposure) — the effect
  of exposure among compliers, under monotonicity (no defiers) and the
  exclusion restriction (assignment affects the outcome only through
  exposure) (Angrist, Imbens & Rubin 1996). An instrumental-variable
  estimand, reported *alongside* the ITT, never instead of it.

**Attrition.** When outcomes are missing for some randomised units,
complete-case comparisons lose the protection of randomisation if response
differs by arm or is related to outcomes. Two bounding approaches keep
the analysis honest:

* **Manski bounds** — no assumptions beyond known outcome bounds; for the
  ITT on all randomised units; usually wide;
* **Lee (2009) bounds** — assume monotone response (assignment moves
  response in one direction only); bound the effect among units that would
  respond in either arm ("always-observed") by trimming the arm with higher
  response.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from decision_intelligence.causal_inference.estimators import CausalEstimate
from decision_intelligence.common.bootstrap import bootstrap_two_sample_ci
from decision_intelligence.common.validation import FloatArray, require_probability
from decision_intelligence.statistics.hypothesis_tests import (
    two_proportion_z_test,
    two_sample_t_test,
)


def _binary(values: ArrayLike, name: str) -> FloatArray:
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 1 or not np.isin(v, (0.0, 1.0)).all():
        raise ValueError(f"{name} must be a 1-D 0/1 vector")
    return v


def _welch_estimate(
    y1: FloatArray, y0: FloatArray, *, estimand: str, estimator: str, level: float,
    warnings: tuple[str, ...] = (), assumptions: tuple[str, ...] = (),
) -> CausalEstimate:
    test = two_sample_t_test(y1, y0, confidence_level=level)
    assert test.ci_lower is not None and test.ci_upper is not None
    se = math.sqrt(y1.var(ddof=1) / y1.size + y0.var(ddof=1) / y0.size)
    return CausalEstimate(
        estimand=estimand, estimator=estimator, estimate=test.estimate, standard_error=se,
        ci_lower=test.ci_lower, ci_upper=test.ci_upper, confidence_level=level,
        se_method="Welch", n=y1.size + y0.size, n_treated=y1.size, adjustment_set=(),
        assumptions=assumptions, warnings=warnings,
    )


@dataclass(frozen=True)
class NoncomplianceAnalysis:
    itt: CausalEstimate
    first_stage: CausalEstimate
    """Effect of assignment on exposure = complier share under monotonicity."""
    cace: CausalEstimate
    as_treated: CausalEstimate
    per_protocol: CausalEstimate
    exposure_rate_treatment: float
    exposure_rate_control: float
    first_stage_f: float


def noncompliance_analysis(
    outcome: ArrayLike, assigned: ArrayLike, exposed: ArrayLike, *, confidence_level: float = 0.95
) -> NoncomplianceAnalysis:
    """ITT, first stage, CACE (Wald, delta-method SE) and the two biased comparisons."""
    y = np.asarray(outcome, dtype=np.float64)
    z = _binary(assigned, "assigned")
    d = _binary(exposed, "exposed")
    if not (y.shape == z.shape == d.shape) or not np.isfinite(y).all():
        raise ValueError("outcome, assigned and exposed must be finite and of equal length")
    level = require_probability(confidence_level, "confidence_level")
    t, c = z == 1, z == 0
    itt = _welch_estimate(
        y[t], y[c], estimand="ITT: effect of assignment", estimator="difference in means by "
        "assigned arm", level=level, assumptions=("random assignment",),
    )
    if float(d[t].mean() - d[c].mean()) <= 0:
        raise ValueError("assignment does not increase exposure: the CACE is not identified")
    first = _welch_estimate(
        d[t], d[c], estimand="effect of assignment on exposure (complier share)",
        estimator="difference in exposure rates by assigned arm", level=level,
        assumptions=("random assignment",),
    )
    warnings: list[str] = []
    f_stat = (first.estimate / first.standard_error) ** 2 if first.standard_error > 0 else math.inf
    if f_stat < 10:
        warnings.append(f"weak first stage (F = {f_stat:.1f} < 10): the CACE is unreliable")
    beta = itt.estimate / first.estimate
    residual = y - beta * d
    variance = (residual[t].var(ddof=1) / t.sum() + residual[c].var(ddof=1) / c.sum())
    se = math.sqrt(variance) / first.estimate
    q = float(stats.norm.ppf(0.5 + level / 2))
    cace = CausalEstimate(
        estimand="CACE/LATE: effect of exposure among compliers",
        estimator="Wald (ITT on outcome / ITT on exposure)",
        estimate=beta, standard_error=se, ci_lower=beta - q * se, ci_upper=beta + q * se,
        confidence_level=level, se_method="delta method (equals 2SLS HC sandwich up to n/(n-1))",
        n=y.size, n_treated=int(t.sum()), adjustment_set=(),
        assumptions=("random assignment", "monotonicity (no defiers)",
                     "exclusion restriction (assignment acts only through exposure)"),
        warnings=tuple(warnings),
    )
    as_treated = _welch_estimate(
        y[d == 1], y[d == 0], estimand="association of exposure with outcome (not causal)",
        estimator="as-treated: difference in means by exposure", level=level,
        warnings=("compares groups formed after randomisation: biased if compliance relates "
                  "to outcomes",),
    )
    compliant = d == z
    per_protocol = _welch_estimate(
        y[compliant & t], y[compliant & c],
        estimand="association among protocol-compliant units (not causal)",
        estimator="per-protocol: difference in means among units with exposure == assignment",
        level=level,
        warnings=("drops units by a post-randomisation behaviour: biased if compliance "
                  "relates to outcomes",),
    )
    return NoncomplianceAnalysis(
        itt=itt, first_stage=first, cace=cace, as_treated=as_treated, per_protocol=per_protocol,
        exposure_rate_treatment=float(d[t].mean()), exposure_rate_control=float(d[c].mean()),
        first_stage_f=float(f_stat),
    )


@dataclass(frozen=True)
class AttritionSummary:
    response_control: float
    response_treatment: float
    n_control: int
    n_treatment: int
    differential_p_value: float

    @property
    def differential(self) -> float:
        return self.response_treatment - self.response_control


def attrition_summary(outcome: ArrayLike, assigned: ArrayLike) -> AttritionSummary:
    """Response (outcome observed) rate by arm and a test of differential attrition.

    A non-significant difference does not make complete-case analysis safe:
    equal rates can hide different *kinds* of units dropping out in each arm.
    """
    y = np.asarray(outcome, dtype=np.float64)
    z = _binary(assigned, "assigned")
    observed = np.isfinite(y)
    r_t, n_t = int(observed[z == 1].sum()), int((z == 1).sum())
    r_c, n_c = int(observed[z == 0].sum()), int((z == 0).sum())
    test = two_proportion_z_test(r_t, n_t, r_c, n_c)
    return AttritionSummary(r_c / n_c, r_t / n_t, n_c, n_t, test.p_value)


def complete_case_difference(
    outcome: ArrayLike, assigned: ArrayLike, *, confidence_level: float = 0.95
) -> CausalEstimate:
    """Difference among units with observed outcomes — valid only if missingness is ignorable."""
    y = np.asarray(outcome, dtype=np.float64)
    z = _binary(assigned, "assigned")
    keep = np.isfinite(y)
    return _welch_estimate(
        y[keep & (z == 1)], y[keep & (z == 0)],
        estimand="ITT among respondents (biased unless attrition is unrelated to outcomes)",
        estimator="complete-case difference in means", level=confidence_level,
        warnings=("conditions on a post-randomisation variable (response)",),
    )


def manski_bounds(
    outcome: ArrayLike, assigned: ArrayLike, *, lower: float, upper: float
) -> tuple[float, float]:
    """Worst-case bounds on the ITT for all randomised units (outcomes known to lie in
    ``[lower, upper]``); missing outcomes are set to the extremes."""
    if not lower < upper:
        raise ValueError("lower must be below upper")
    y = np.asarray(outcome, dtype=np.float64)
    z = _binary(assigned, "assigned")
    observed = y[np.isfinite(y)]
    if observed.size and (float(np.min(observed)) < lower or float(np.max(observed)) > upper):
        raise ValueError("observed outcomes fall outside the stated bounds")

    def arm_bounds(mask: np.ndarray) -> tuple[float, float]:
        values = y[mask]
        seen = np.isfinite(values)
        share = float(seen.mean())
        mean_seen = float(values[seen].mean()) if seen.any() else 0.0
        return share * mean_seen + (1 - share) * lower, share * mean_seen + (1 - share) * upper

    t_lo, t_hi = arm_bounds(z == 1)
    c_lo, c_hi = arm_bounds(z == 0)
    return t_lo - c_hi, t_hi - c_lo


def _trimmed_mean(values: FloatArray, share: float, *, drop: str) -> float:
    """Mean after removing a (fractional) ``share`` of the largest or smallest values."""
    v = np.sort(values)
    keep = (1.0 - share) * v.size
    whole = int(math.floor(keep + 1e-12))
    fraction = keep - whole
    if drop == "top":
        kept = v[:whole]
        extra = v[whole] * fraction if whole < v.size else 0.0
    else:
        kept = v[v.size - whole:] if whole else v[:0]
        extra = v[v.size - whole - 1] * fraction if whole < v.size else 0.0
    return float((kept.sum() + extra) / keep)


class LeeBoundsUndefinedError(ValueError):
    """Lee bounds (or their bootstrap) are undefined: an arm has no observed outcomes."""


# Largest tolerated probability that at least one bootstrap resample has an arm with no
# observed outcomes (see ``lee_bounds``).
LEE_BOOTSTRAP_FAILURE_TOLERANCE = 1e-6


def _lee_point(y_t: FloatArray, y_c: FloatArray) -> tuple[float, float, float, str]:
    """Bounds from arrays with NaN for missing outcomes."""
    seen_t, seen_c = y_t[np.isfinite(y_t)], y_c[np.isfinite(y_c)]
    if seen_t.size == 0 or seen_c.size == 0:
        empty = "treatment" if seen_t.size == 0 else "control"
        raise LeeBoundsUndefinedError(
            f"the {empty} arm has no observed outcomes: the trimming share and the "
            "always-observed mean are undefined"
        )
    q_t, q_c = seen_t.size / y_t.size, seen_c.size / y_c.size
    if q_t >= q_c:
        p = (q_t - q_c) / q_t
        lower = _trimmed_mean(seen_t, p, drop="top") - seen_c.mean()
        upper = _trimmed_mean(seen_t, p, drop="bottom") - seen_c.mean()
        return float(lower), float(upper), p, "treatment"
    p = (q_c - q_t) / q_c
    lower = seen_t.mean() - _trimmed_mean(seen_c, p, drop="bottom")
    upper = seen_t.mean() - _trimmed_mean(seen_c, p, drop="top")
    return float(lower), float(upper), p, "control"


@dataclass(frozen=True)
class LeeBounds:
    lower: float
    upper: float
    trimming_share: float
    trimmed_arm: str
    ci_lower: float | None
    ci_upper: float | None
    """Percentile bootstrap: 2.5% quantile of the lower bound, 97.5% of the upper
    (covers the identified set with at least the nominal probability; conservative)."""


def lee_bounds(
    outcome: ArrayLike,
    assigned: ArrayLike,
    *,
    rng: np.random.Generator | None = None,
    n_bootstrap: int = 999,
    confidence_level: float = 0.95,
) -> LeeBounds:
    """Lee (2009) trimming bounds on the ITT among always-observed units (monotone response).

    With ``rng`` a percentile bootstrap interval is added (arms resampled
    separately). **Failure policy.** The bounds are undefined on a resample
    in which an arm has no observed outcomes. Such draws are never dropped
    or redrawn — that would condition the bootstrap distribution on response,
    the very variable whose selectivity the bounds address — and no interval
    is fabricated. Instead:

    * before resampling, the probability that *any* of the ``n_bootstrap``
      draws empties an arm, bounded by
      ``n_bootstrap * sum_arms (1 - q_arm) ** n_arm``, is computed from the
      data; if it exceeds :data:`LEE_BOOTSTRAP_FAILURE_TOLERANCE` (1e-6) the
      request is refused with :class:`LeeBoundsUndefinedError`, so the
      outcome depends on the data, not on the seed;
    * if an undefined draw nevertheless occurs (probability <= 1e-6), the
      same error is raised.

    The point bounds remain available by calling without ``rng``; with so
    few observed outcomes in an arm the bootstrap would not be reliable anyway.
    """
    y = np.asarray(outcome, dtype=np.float64)
    z = _binary(assigned, "assigned")
    y_t, y_c = y[z == 1], y[z == 0]
    lower, upper, share, arm = _lee_point(y_t, y_c)
    ci_lo = ci_hi = None
    if rng is not None:
        empty_per_draw = sum(
            (1.0 - float(np.isfinite(v).mean())) ** v.size for v in (y_t, y_c)
        )
        failure_bound = min(1.0, n_bootstrap * empty_per_draw)
        if failure_bound > LEE_BOOTSTRAP_FAILURE_TOLERANCE:
            raise LeeBoundsUndefinedError(
                f"bootstrap refused: with {int(np.isfinite(y_t).sum())}/{y_t.size} treatment and "
                f"{int(np.isfinite(y_c).sum())}/{y_c.size} control outcomes observed, a resample "
                f"leaves an arm with no observed outcomes with probability up to "
                f"{failure_bound:.2g} over {n_bootstrap} draws (tolerance "
                f"{LEE_BOOTSTRAP_FAILURE_TOLERANCE:g}); undefined draws are not discarded. "
                "Report the point bounds (call without rng)"
            )
        # Resample unit indices within each arm (M3 stratified bootstrap) and recompute bounds.
        index_t = np.arange(y_t.size, dtype=np.float64)
        index_c = np.arange(y_c.size, dtype=np.float64)

        def lower_stat(it: FloatArray, ic: FloatArray) -> float:
            return _lee_point(y_t[it.astype(int)], y_c[ic.astype(int)])[0]

        def upper_stat(it: FloatArray, ic: FloatArray) -> float:
            return _lee_point(y_t[it.astype(int)], y_c[ic.astype(int)])[1]

        try:
            low = bootstrap_two_sample_ci(index_t, index_c, lower_stat, rng=rng,
                                          confidence_level=confidence_level,
                                          n_resamples=n_bootstrap, method="percentile")
            high = bootstrap_two_sample_ci(index_t, index_c, upper_stat, rng=rng,
                                           confidence_level=confidence_level,
                                           n_resamples=n_bootstrap, method="percentile")
        except LeeBoundsUndefinedError as error:
            raise LeeBoundsUndefinedError(
                f"a bootstrap resample is undefined ({error}); undefined draws are not "
                "discarded, so no interval is reported. Report the point bounds (call without rng)"
            ) from error
        ci_lo, ci_hi = low.ci_lower, high.ci_upper
    return LeeBounds(lower, upper, share, arm, ci_lo, ci_hi)


__all__ = [
    "LEE_BOOTSTRAP_FAILURE_TOLERANCE",
    "AttritionSummary",
    "LeeBounds",
    "LeeBoundsUndefinedError",
    "NoncomplianceAnalysis",
    "attrition_summary",
    "complete_case_difference",
    "lee_bounds",
    "manski_bounds",
    "noncompliance_analysis",
]
