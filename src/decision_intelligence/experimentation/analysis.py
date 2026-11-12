"""A/B analysis: intention-to-treat effects for binary and continuous metrics.

Every function compares units **as randomised** (by assigned arm), so the
difference estimates the intention-to-treat (ITT) effect of *assignment*.
Randomisation makes arm membership independent of the potential outcomes,
which is what licenses the causal reading — no adjustment is needed for
identification.

Statistical machinery is reused, not re-implemented:

* binary: M3 pooled two-proportion z-test with the Newcombe hybrid score
  interval; relative lift from the M3 Katz log risk-ratio interval;
* continuous: M3 Welch t-test; relative lift by the delta method on the log
  ratio of means; optional M3 stratified bootstrap (BCa) for heavy tails;
* clustered: CR1 cluster-robust inference with a t(G - 1) reference (M5);
* A/B/n and metric families: M3/M6 multiple-testing corrections.

Missing outcomes are refused: how they are handled (attrition bounds,
see :mod:`.compliance`) is an explicit analysis decision, never a silent drop.

**Test and interval at the significance boundary.** For binary metrics the
pooled z-test (variance estimated under ``H0: p_t = p_c``) and the Newcombe
hybrid score interval (built from the two single-proportion Wilson
intervals) are different constructions; they are *not* dual, so near
``p = alpha`` one can reject while the other still contains 0, in either
direction (e.g. 25/100 vs 14/100: p = 0.0496, interval [-0.0004, 0.218];
168/200 vs 36/50: p = 0.0502, interval [0.0002, 0.264]). Both are
retained unchanged — each is the better-calibrated choice for its own job —
and the policy is:

* :attr:`MetricResult.significant` reports the pre-specified *test*
  (``p <= alpha``; the multiplicity corrections operate on these p-values);
* :attr:`MetricResult.interval_excludes_zero` reports the *interval*;
* when they disagree the result carries a warning and
  :attr:`MetricResult.test_and_interval_agree` is ``False``; the decision
  rule (:func:`.decision.recommend`) then treats a non-zero effect as **not
  established** (HOLD), because the evidence sits on the boundary and
  neither construction is privileged after the fact. This is conservative:
  it can only move a borderline LAUNCH/ITERATE to HOLD.

The Welch test and interval are dual, so continuous metrics disagree only
through floating-point rounding exactly at the boundary.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from decision_intelligence.causal_inference.cluster_inference import cluster_robust_t_test
from decision_intelligence.causal_inference.estimators import CausalEstimate
from decision_intelligence.common.bootstrap import bootstrap_two_sample_ci
from decision_intelligence.common.validation import FloatArray, require_probability
from decision_intelligence.experimentation.design import MetricKind
from decision_intelligence.statistics.effect_sizes import binary_effect_sizes
from decision_intelligence.statistics.hypothesis_tests import (
    two_proportion_z_test,
    two_sample_t_test,
)
from decision_intelligence.statistics.multiple_testing import adjust_p_values, rejects

ITT_ESTIMAND = (
    "ITT: average effect of assignment to treatment vs control (units analysed as randomised)"
)
_RANDOMISATION = ("random assignment (independent of potential outcomes)",
                  "no interference between units (SUTVA)")


@dataclass(frozen=True)
class MetricResult:
    """Two-arm comparison of one metric; ``effect`` is treatment minus control."""

    metric: str
    kind: MetricKind
    n_control: int
    n_treatment: int
    mean_control: float
    mean_treatment: float
    effect: CausalEstimate
    relative_lift: float
    relative_ci_lower: float
    relative_ci_upper: float
    p_value: float
    test: str
    bootstrap_ci: tuple[float, float] | None = None
    warnings: tuple[str, ...] = ()

    @property
    def alpha(self) -> float:
        """Two-sided level matching the interval: ``1 - confidence_level``."""
        return 1 - self.effect.confidence_level

    @property
    def significant(self) -> bool:
        """The pre-specified test rejects ``H0: no effect`` (``p <= alpha``)."""
        return rejects(self.p_value, self.alpha)

    @property
    def interval_excludes_zero(self) -> bool:
        """The reported (closed) interval does not contain 0."""
        return self.effect.ci_lower > 0 or self.effect.ci_upper < 0

    @property
    def test_and_interval_agree(self) -> bool:
        """``False`` only at the significance boundary for non-dual test/interval pairs."""
        return self.significant == self.interval_excludes_zero


def split_by_arm(
    outcome: ArrayLike, arm: ArrayLike, *, control: str = "control", treatment: str = "treatment"
) -> tuple[FloatArray, FloatArray]:
    """Outcomes of the control and treatment arms; refuses missing outcomes and unknown arms."""
    y = np.asarray(outcome, dtype=np.float64)
    labels = np.asarray(arm).astype(str)
    if y.shape != labels.shape or y.ndim != 1:
        raise ValueError("outcome and arm must be 1-D and of equal length")
    if not np.isfinite(y).all():
        raise ValueError(
            "outcomes contain NaN/inf: handle attrition explicitly (see compliance.py) "
            "rather than dropping units silently"
        )
    unknown = set(labels.tolist()) - {control, treatment}
    if unknown:
        raise ValueError(f"unexpected arm labels {sorted(unknown)}")
    y_c, y_t = y[labels == control], y[labels == treatment]
    if y_c.size < 2 or y_t.size < 2:
        raise ValueError("each arm needs at least two units")
    return y_c, y_t


def _estimate(
    estimate: float, se: float, lower: float, upper: float, level: float, method: str,
    n: int, n_treated: int, warnings: Sequence[str] = (),
) -> CausalEstimate:
    return CausalEstimate(
        estimand=ITT_ESTIMAND,
        estimator="difference in means (treatment - control)",
        estimate=estimate,
        standard_error=se,
        ci_lower=lower,
        ci_upper=upper,
        confidence_level=level,
        se_method=method,
        n=n,
        n_treated=n_treated,
        adjustment_set=(),
        assumptions=_RANDOMISATION,
        warnings=tuple(warnings),
    )


def analyse_binary(
    outcome: ArrayLike,
    arm: ArrayLike,
    *,
    metric: str = "conversion",
    control: str = "control",
    treatment: str = "treatment",
    confidence_level: float = 0.95,
) -> MetricResult:
    """Risk difference (Newcombe interval, pooled z-test) and relative lift (Katz interval)."""
    y_c, y_t = split_by_arm(outcome, arm, control=control, treatment=treatment)
    if not (np.isin(y_c, (0.0, 1.0)).all() and np.isin(y_t, (0.0, 1.0)).all()):
        raise ValueError("a binary metric must be coded 0/1")
    s_t, n_t, s_c, n_c = int(y_t.sum()), y_t.size, int(y_c.sum()), y_c.size
    test = two_proportion_z_test(s_t, n_t, s_c, n_c, confidence_level=confidence_level)
    sizes = binary_effect_sizes(s_t, n_t, s_c, n_c, confidence_level=confidence_level)
    p_t, p_c = s_t / n_t, s_c / n_c
    se = math.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c)
    assert test.ci_lower is not None and test.ci_upper is not None
    rr = sizes.risk_ratio
    warnings = list(test.warnings)
    if sizes.zero_cell_correction_applied:
        warnings.append("zero cell: relative lift uses the Haldane–Anscombe correction")
    excludes_zero = test.ci_lower > 0 or test.ci_upper < 0
    if rejects(test.p_value, 1 - confidence_level) != excludes_zero:
        warnings.append(
            f"boundary disagreement: pooled z-test p = {test.p_value:.4f} but the Newcombe "
            f"interval [{test.ci_lower:+.5f}, {test.ci_upper:+.5f}] "
            f"{'excludes' if excludes_zero else 'contains'} 0 (the two are not dual); a non-zero "
            "effect is treated as not established"
        )
    return MetricResult(
        metric=metric,
        kind="binary",
        n_control=n_c,
        n_treatment=n_t,
        mean_control=p_c,
        mean_treatment=p_t,
        effect=_estimate(test.estimate, se, test.ci_lower, test.ci_upper, confidence_level,
                         "Newcombe hybrid score interval; unpooled SE", n_c + n_t, n_t, warnings),
        relative_lift=rr.estimate - 1.0,
        relative_ci_lower=(rr.ci_lower if rr.ci_lower is not None else math.nan) - 1.0,
        relative_ci_upper=(rr.ci_upper if rr.ci_upper is not None else math.nan) - 1.0,
        p_value=test.p_value,
        test="pooled two-proportion z-test",
        warnings=tuple(warnings),
    )


def _mean_difference(a: FloatArray, b: FloatArray) -> float:
    return float(a.mean() - b.mean())


def analyse_continuous(
    outcome: ArrayLike,
    arm: ArrayLike,
    *,
    metric: str = "revenue per user",
    control: str = "control",
    treatment: str = "treatment",
    confidence_level: float = 0.95,
    rng: np.random.Generator | None = None,
    n_bootstrap: int = 2000,
) -> MetricResult:
    """Welch difference in means; relative lift by the delta method on ``log(mean_t / mean_c)``.

    With ``rng`` a stratified (per-arm) BCa bootstrap interval for the
    difference is added as a check for heavy-tailed metrics.
    """
    y_c, y_t = split_by_arm(outcome, arm, control=control, treatment=treatment)
    test = two_sample_t_test(y_t, y_c, confidence_level=confidence_level)
    assert test.ci_lower is not None and test.ci_upper is not None and test.df is not None
    se = math.sqrt(y_t.var(ddof=1) / y_t.size + y_c.var(ddof=1) / y_c.size)
    m_t, m_c = float(y_t.mean()), float(y_c.mean())
    warnings: list[str] = []
    if m_t > 0 and m_c > 0:
        z = float(stats.norm.ppf(0.5 + confidence_level / 2))
        se_log = math.sqrt(
            y_t.var(ddof=1) / (y_t.size * m_t**2) + y_c.var(ddof=1) / (y_c.size * m_c**2)
        )
        log_ratio = math.log(m_t / m_c)
        lift = m_t / m_c - 1
        lift_lo = math.exp(min(log_ratio - z * se_log, 700.0)) - 1
        lift_hi = math.exp(min(log_ratio + z * se_log, 700.0)) - 1
    else:
        lift = lift_lo = lift_hi = math.nan
        warnings.append("relative lift undefined: an arm mean is not positive")
    boot = None
    if rng is not None:
        result = bootstrap_two_sample_ci(
            y_t, y_c, _mean_difference, rng=rng, confidence_level=confidence_level,
            n_resamples=n_bootstrap, method="bca",
        )
        boot = (result.ci_lower, result.ci_upper)
    return MetricResult(
        metric=metric,
        kind="continuous",
        n_control=y_c.size,
        n_treatment=y_t.size,
        mean_control=m_c,
        mean_treatment=m_t,
        effect=_estimate(test.estimate, se, test.ci_lower, test.ci_upper, confidence_level,
                         f"Welch (df = {test.df:.1f})", y_c.size + y_t.size, y_t.size, warnings),
        relative_lift=lift,
        relative_ci_lower=lift_lo,
        relative_ci_upper=lift_hi,
        p_value=test.p_value,
        test="Welch two-sample t-test",
        bootstrap_ci=boot,
        warnings=tuple(warnings),
    )


def stratified_difference(
    outcome: ArrayLike,
    arm: ArrayLike,
    strata: ArrayLike,
    *,
    control: str = "control",
    treatment: str = "treatment",
    confidence_level: float = 0.95,
) -> CausalEstimate:
    """Design-based estimator for a stratified (blocked) experiment.

    ``tau = sum_s (N_s / N) (mean_t,s - mean_c,s)`` with the Neyman variance
    ``sum_s (N_s / N)^2 (v_t,s / n_t,s + v_c,s / n_c,s)``. It analyses the
    experiment *as randomised*: with stratified assignment the unstratified
    difference in means is still unbiased, but its Welch variance ignores
    the blocking and is conservative. This is not covariate adjustment
    (CUPED/regression adjustment are M7 scope).
    """
    y = np.asarray(outcome, dtype=np.float64)
    labels = np.asarray(arm).astype(str)
    blocks = np.asarray(strata)
    split_by_arm(y, labels, control=control, treatment=treatment)  # validation
    level = require_probability(confidence_level, "confidence_level")
    total = y.size
    estimate = variance = 0.0
    for stratum in np.unique(blocks):
        in_s = blocks == stratum
        y_t, y_c = y[in_s & (labels == treatment)], y[in_s & (labels == control)]
        if y_t.size < 2 or y_c.size < 2:
            raise ValueError(f"stratum {stratum!r} needs at least two units per arm")
        weight = in_s.sum() / total
        estimate += weight * (y_t.mean() - y_c.mean())
        variance += weight**2 * (y_t.var(ddof=1) / y_t.size + y_c.var(ddof=1) / y_c.size)
    se = math.sqrt(variance)
    z = float(stats.norm.ppf(0.5 + level / 2))
    return _estimate(float(estimate), se, estimate - z * se, estimate + z * se, level,
                     "Neyman stratified variance (normal)", total,
                     int((labels == treatment).sum()))


@dataclass(frozen=True)
class ClusteredAnalysis:
    """Naive unit-level vs cluster-aware inference for the same difference in means."""

    naive: CausalEstimate
    cluster_robust: CausalEstimate
    cluster_means: CausalEstimate
    n_clusters: int
    n_treated_clusters: int


def clustered_difference(
    outcome: ArrayLike,
    arm: ArrayLike,
    clusters: ArrayLike,
    *,
    control: str = "control",
    treatment: str = "treatment",
    confidence_level: float = 0.95,
) -> ClusteredAnalysis:
    """Three analyses of a cluster-randomised experiment.

    * naive Welch on units — **invalid** when whole clusters were assigned:
      it treats correlated units as independent and understates the SE;
    * OLS of outcome on treatment with CR1 cluster-robust SE, t(G - 1);
    * Welch on cluster means (one observation per cluster; estimates the
      cluster-weighted effect, which differs from the unit-weighted one when
      cluster sizes and effects co-vary).
    """
    y_c, y_t = split_by_arm(outcome, arm, control=control, treatment=treatment)
    y = np.asarray(outcome, dtype=np.float64)
    labels = np.asarray(arm).astype(str)
    groups = np.asarray(clusters)
    level = require_probability(confidence_level, "confidence_level")
    treated = (labels == treatment).astype(float)
    for g in np.unique(groups):
        if np.unique(treated[groups == g]).size > 1:
            raise ValueError(f"cluster {g!r} contains both arms: not a cluster-randomised design")
    naive_test = two_sample_t_test(y_t, y_c, confidence_level=level)
    assert naive_test.ci_lower is not None and naive_test.ci_upper is not None
    naive_se = math.sqrt(y_t.var(ddof=1) / y_t.size + y_c.var(ddof=1) / y_c.size)
    naive = _estimate(naive_test.estimate, naive_se, naive_test.ci_lower, naive_test.ci_upper,
                      level, "Welch on units (ignores clustering)", y.size, int(treated.sum()),
                      ("invalid under cluster randomisation: SE understated",))

    design = np.column_stack([np.ones(y.size), treated])
    beta, se, _ = cluster_robust_t_test(design, y, groups, 1)
    unique = np.unique(groups)
    q = float(stats.t.ppf(0.5 + level / 2, unique.size - 1))
    robust = _estimate(beta, se, beta - q * se, beta + q * se, level,
                       f"CR1 cluster-robust, t({unique.size - 1})", y.size, int(treated.sum()))

    means = np.array([y[groups == g].mean() for g in unique])
    cluster_treated = np.array([treated[groups == g][0] for g in unique])
    mt = two_sample_t_test(means[cluster_treated == 1], means[cluster_treated == 0],
                           confidence_level=level)
    assert mt.ci_lower is not None and mt.ci_upper is not None
    m1, m0 = means[cluster_treated == 1], means[cluster_treated == 0]
    mse = math.sqrt(m1.var(ddof=1) / m1.size + m0.var(ddof=1) / m0.size)
    cluster_means = _estimate(mt.estimate, mse, mt.ci_lower, mt.ci_upper, level,
                              "Welch on cluster means (cluster-weighted estimand)", unique.size,
                              int(m1.size))
    return ClusteredAnalysis(naive, robust, cluster_means, int(unique.size), int(m1.size))


@dataclass(frozen=True)
class MultiArmResult:
    """Each treatment arm against one control, with multiplicity-adjusted inference."""

    comparisons: dict[str, MetricResult]
    adjusted_p_values: dict[str, float]
    correction: str
    simultaneous_level: float


def multi_arm_comparison(
    outcome: ArrayLike,
    arm: ArrayLike,
    *,
    kind: MetricKind,
    control: str = "control",
    correction: str = "holm",
    alpha: float = 0.05,
    metric: str = "conversion",
) -> MultiArmResult:
    """A/B/n: every treatment arm vs control.

    p-values are adjusted (default Holm, FWER). Intervals use the Bonferroni
    level ``1 - alpha / m`` so they hold simultaneously (Holm has no matching
    interval). Picking the best arm and reporting its unadjusted estimate
    overstates its effect (winner's curse); the adjusted analysis does not
    remove that selection bias in the point estimate.
    """
    labels = np.asarray(arm).astype(str)
    y = np.asarray(outcome, dtype=np.float64)
    treatments = sorted(set(labels.tolist()) - {control})
    if not treatments:
        raise ValueError("need at least one treatment arm besides control")
    level = 1 - alpha / len(treatments)
    comparisons: dict[str, MetricResult] = {}
    for name in treatments:
        keep = (labels == control) | (labels == name)
        analyse = analyse_binary if kind == "binary" else analyse_continuous
        comparisons[name] = analyse(y[keep], labels[keep], metric=metric, control=control,
                                    treatment=name, confidence_level=level)
    adjusted = adjust_p_values([comparisons[t].p_value for t in treatments], method=correction,
                               alpha=alpha)
    return MultiArmResult(
        comparisons=comparisons,
        adjusted_p_values=dict(zip(treatments, adjusted.adjusted_p_values.tolist(), strict=True)),
        correction=correction,
        simultaneous_level=level,
    )


__all__ = [
    "ClusteredAnalysis",
    "ITT_ESTIMAND",
    "MetricResult",
    "MultiArmResult",
    "analyse_binary",
    "analyse_continuous",
    "clustered_difference",
    "multi_arm_comparison",
    "split_by_arm",
    "stratified_difference",
]
