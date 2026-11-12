"""Experiment planning: sample size, minimum detectable effect, duration and design effect.

Thin, explicit wrappers around the M3 power primitives
(:mod:`decision_intelligence.statistics.power`) that speak the language of
an experiment: a baseline conversion rate, an absolute or relative minimum
detectable effect (MDE), a treatment share, daily eligible traffic and —
for cluster-randomised designs — a design effect.

Power is a *design* quantity, computed before the experiment for an effect
size chosen on business grounds. "Observed power" computed from the
estimate after the fact is a transformation of the p-value and is not
reported anywhere in this package.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import optimize

from decision_intelligence.common.validation import require_probability
from decision_intelligence.statistics.power import (
    minimum_detectable_effect_two_sample_t,
    power_two_proportions,
    power_two_sample_t,
    sample_size_two_proportions,
    sample_size_two_sample_t,
)


@dataclass(frozen=True)
class ExperimentPlan:
    """A planned two-arm comparison (treatment vs one control)."""

    metric: str
    baseline: float
    """Control mean (a rate for binary metrics)."""
    mde_absolute: float
    mde_relative: float
    alpha: float
    alpha_per_comparison: float
    power: float
    treatment_share: float
    n_control: int
    n_treatment: int
    design_effect: float
    achieved_power: float
    daily_eligible: float | None = None

    @property
    def n_total(self) -> int:
        return self.n_control + self.n_treatment

    @property
    def days(self) -> int | None:
        """Days of traffic needed (rounded up), if daily eligible traffic was given."""
        if self.daily_eligible is None:
            return None
        return math.ceil(self.n_total / self.daily_eligible)


def design_effect(mean_cluster_size: float, icc: float, *, cluster_size_cv: float = 0.0) -> float:
    """Variance inflation from randomising clusters: ``1 + ((cv^2 + 1) m - 1) * icc``.

    Kish's ``1 + (m - 1) icc`` for equal cluster sizes; the Eldridge, Ashby &
    Kerry (2006) correction for unequal sizes with coefficient of variation ``cv``.
    """
    if mean_cluster_size < 1:
        raise ValueError("mean_cluster_size must be >= 1")
    if not 0.0 <= icc <= 1.0:
        raise ValueError("icc must lie in [0, 1]")
    if cluster_size_cv < 0:
        raise ValueError("cluster_size_cv must be >= 0")
    return 1.0 + ((cluster_size_cv**2 + 1.0) * mean_cluster_size - 1.0) * icc


def _resolve_mde(baseline: float, mde_absolute: float | None, mde_relative: float | None) -> float:
    if (mde_absolute is None) == (mde_relative is None):
        raise ValueError("give exactly one of mde_absolute or mde_relative")
    if mde_absolute is not None:
        delta = mde_absolute
    else:
        assert mde_relative is not None
        delta = baseline * mde_relative
    if delta == 0:
        raise ValueError("the MDE must be non-zero")
    return float(delta)


def _inflate(n: int, factor: float) -> int:
    return math.ceil(n * factor - 1e-9)


def plan_binary(
    baseline: float,
    *,
    mde_absolute: float | None = None,
    mde_relative: float | None = None,
    alpha: float = 0.05,
    power: float = 0.8,
    treatment_share: float = 0.5,
    n_comparisons: int = 1,
    design_effect_factor: float = 1.0,
    daily_eligible: float | None = None,
    metric: str = "conversion",
) -> ExperimentPlan:
    """Sample sizes for the pooled two-proportion z-test (two-sided).

    ``n_comparisons > 1`` (A/B/n against one control) splits ``alpha`` by
    Bonferroni. ``design_effect_factor`` inflates both arms for clustered
    randomisation (sizes are then in *units*, randomised in clusters).
    """
    baseline = require_probability(baseline, "baseline")
    treatment_share = require_probability(treatment_share, "treatment_share")
    if n_comparisons < 1 or design_effect_factor < 1:
        raise ValueError("n_comparisons must be >= 1 and design_effect_factor >= 1")
    delta = _resolve_mde(baseline, mde_absolute, mde_relative)
    treated = require_probability(baseline + delta, "baseline + MDE")
    alpha_each = alpha / n_comparisons
    ratio = (1 - treatment_share) / treatment_share  # n_control / n_treatment
    n_t, n_c = sample_size_two_proportions(
        treated, baseline, power=power, alpha=alpha_each, ratio=ratio
    )
    n_t, n_c = _inflate(n_t, design_effect_factor), _inflate(n_c, design_effect_factor)
    achieved = power_two_proportions(
        treated,
        baseline,
        max(2, math.floor(n_t / design_effect_factor)),
        max(2, math.floor(n_c / design_effect_factor)),
        alpha=alpha_each,
    )
    return ExperimentPlan(
        metric=metric,
        baseline=baseline,
        mde_absolute=delta,
        mde_relative=delta / baseline,
        alpha=alpha,
        alpha_per_comparison=alpha_each,
        power=power,
        treatment_share=treatment_share,
        n_control=n_c,
        n_treatment=n_t,
        design_effect=design_effect_factor,
        achieved_power=achieved,
        daily_eligible=daily_eligible,
    )


def plan_continuous(
    baseline_mean: float,
    standard_deviation: float,
    *,
    mde_absolute: float | None = None,
    mde_relative: float | None = None,
    alpha: float = 0.05,
    power: float = 0.8,
    treatment_share: float = 0.5,
    n_comparisons: int = 1,
    design_effect_factor: float = 1.0,
    daily_eligible: float | None = None,
    metric: str = "revenue per user",
) -> ExperimentPlan:
    """Sample sizes for a two-sample t-test with a common SD (normal-theory planning).

    For heavy-tailed metrics (revenue) the t-test is valid for large n by the
    CLT, but the SD must come from historical data on the same population and
    window; an underestimated SD is the commonest planning error.
    """
    if standard_deviation <= 0:
        raise ValueError("standard_deviation must be positive")
    treatment_share = require_probability(treatment_share, "treatment_share")
    if n_comparisons < 1 or design_effect_factor < 1:
        raise ValueError("n_comparisons must be >= 1 and design_effect_factor >= 1")
    delta = _resolve_mde(baseline_mean, mde_absolute, mde_relative)
    alpha_each = alpha / n_comparisons
    ratio = (1 - treatment_share) / treatment_share
    n_t, n_c = sample_size_two_sample_t(
        abs(delta) / standard_deviation, power=power, alpha=alpha_each, ratio=ratio
    )
    n_t, n_c = _inflate(n_t, design_effect_factor), _inflate(n_c, design_effect_factor)
    achieved = power_two_sample_t(
        abs(delta) / standard_deviation,
        max(2, math.floor(n_t / design_effect_factor)),
        max(2, math.floor(n_c / design_effect_factor)),
        alpha=alpha_each,
    )
    return ExperimentPlan(
        metric=metric,
        baseline=baseline_mean,
        mde_absolute=delta,
        mde_relative=delta / baseline_mean if baseline_mean else math.nan,
        alpha=alpha,
        alpha_per_comparison=alpha_each,
        power=power,
        treatment_share=treatment_share,
        n_control=n_c,
        n_treatment=n_t,
        design_effect=design_effect_factor,
        achieved_power=achieved,
        daily_eligible=daily_eligible,
    )


def mde_binary(
    baseline: float,
    n_control: int,
    n_treatment: int,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    design_effect_factor: float = 1.0,
) -> float:
    """Smallest absolute increase over ``baseline`` detectable with ``power`` (two-sided)."""
    baseline = require_probability(baseline, "baseline")
    n_c = max(2, math.floor(n_control / design_effect_factor))
    n_t = max(2, math.floor(n_treatment / design_effect_factor))

    def gap(delta: float) -> float:
        return power_two_proportions(baseline + delta, baseline, n_t, n_c, alpha=alpha) - power

    upper = (1 - baseline) - 1e-9
    if gap(upper) < 0:
        raise ValueError("no increase over the baseline reaches the requested power")
    return float(optimize.brentq(gap, 1e-12, upper, xtol=1e-12))


def mde_continuous(
    standard_deviation: float,
    n_control: int,
    n_treatment: int,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    design_effect_factor: float = 1.0,
) -> float:
    """Smallest absolute mean difference detectable with ``power`` (two-sided t-test)."""
    if standard_deviation <= 0:
        raise ValueError("standard_deviation must be positive")
    d = minimum_detectable_effect_two_sample_t(
        max(2, math.floor(n_treatment / design_effect_factor)),
        max(2, math.floor(n_control / design_effect_factor)),
        power=power,
        alpha=alpha,
    )
    return d * standard_deviation


__all__ = [
    "ExperimentPlan",
    "design_effect",
    "mde_binary",
    "mde_continuous",
    "plan_binary",
    "plan_continuous",
]
