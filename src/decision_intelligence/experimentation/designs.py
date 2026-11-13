"""Selected advanced designs: cluster-randomised planning and switchback experiments.

Both designs randomise something coarser than the unit whose outcome is
measured, so the *randomisation unit* — not the number of rows — determines
the information in the experiment.

**Cluster-randomised experiments** (whole cities, stores or classrooms are
assigned). M6 provides the assignment and the valid analyses (CR1 with
``t(G - 1)``, cluster means) and the design-effect formula. M7 adds planning
from *estimated* intra-cluster correlation:

* :func:`estimate_icc` — the one-way ANOVA estimator of the ICC on historical
  (pre-experiment) data, with a cluster bootstrap interval (M3 bootstrap). The
  ICC of a binary outcome is small (often 0.001–0.05) but multiplies with the
  cluster size: ``DE = 1 + ((cv^2 + 1) m - 1) ICC``;
* :func:`plan_cluster_randomised` — clusters per arm for a target power with a
  ``t(G - 1)`` critical value matching the CR1 analysis. Ignoring the ICC
  ("naive" plan) under-sizes the experiment, and the ICC's own uncertainty
  propagates into the achieved power (both quantified in the evidence).

**Switchback experiments** (the treatment is switched on and off for the
whole marketplace over time periods, used when units interact through a
shared market — drivers, inventory, prices — so user-level randomisation would
violate SUTVA):

* the randomisation unit is the **period**; hours or orders within a period
  share the same assignment and correlated shocks, so an hour- or order-level
  test that treats them as independent is invalid;
* the estimand is the **global** effect (all-on vs all-off). Carryover — the
  treatment still acting after it is switched off — contaminates the hours
  just after a switch, so a **burn-in** at the start of each period is
  excluded from the analysis. The burn-in must cover the carryover and be
  fixed in advance (a sensitivity table is reported, not used to choose it);
* periods are randomised **within time-of-day blocks** (M6 stratified
  randomisation): time of day is the dominant, pre-treatment source of
  variance in a marketplace, and blocking on it removes that variance by
  design; :func:`analyse_switchback` then analyses as randomised (M6
  ``stratified_difference``) and reports a randomisation p-value that
  re-draws the assignment within blocks — exact under the sharp null of no
  effect, regardless of serial correlation between periods.

Not covered: geo experiments with synthetic-control analysis, interference
modelling and exposure mappings, and optimal switchback period length
(see the methodology's limitations).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats

from decision_intelligence.causal_inference.estimators import CausalEstimate
from decision_intelligence.common.bootstrap import bootstrap_ci
from decision_intelligence.common.validation import FloatArray, require_probability
from decision_intelligence.experimentation.analysis import (
    analyse_continuous,
    stratified_difference,
)
from decision_intelligence.experimentation.assignment import (
    complete_randomisation,
    stratified_randomisation,
)
from decision_intelligence.experimentation.planning import (
    ExperimentPlan,
    design_effect,
    plan_binary,
)
from decision_intelligence.statistics.hypothesis_tests import permutation_test

# ---------------------------------------------------------------------------
# Intra-cluster correlation
# ---------------------------------------------------------------------------


def _icc_from_sums(sizes: FloatArray, sums: FloatArray, sums_sq: FloatArray) -> float:
    n_total = float(sizes.sum())
    g = sizes.size
    grand = float(sums.sum()) / n_total
    means = sums / sizes
    ss_between = float(np.sum(sizes * (means - grand) ** 2))
    ss_within = float(np.sum(sums_sq - sizes * means**2))
    msb = ss_between / (g - 1)
    msw = ss_within / (n_total - g)
    n0 = (n_total - float(np.sum(sizes**2)) / n_total) / (g - 1)
    denominator = msb + (n0 - 1) * msw
    return (msb - msw) / denominator if denominator > 0 else 0.0


@dataclass(frozen=True)
class ICCEstimate:
    icc: float
    """ANOVA estimate; may be slightly negative by chance (true ICC >= 0 here)."""
    ci_lower: float | None
    ci_upper: float | None
    n_clusters: int
    mean_cluster_size: float
    cluster_size_cv: float

    @property
    def icc_for_planning(self) -> float:
        """The estimate truncated at 0."""
        return max(0.0, self.icc)


def estimate_icc(
    outcome: ArrayLike,
    clusters: ArrayLike,
    *,
    rng: np.random.Generator | None = None,
    n_bootstrap: int = 999,
    confidence_level: float = 0.95,
) -> ICCEstimate:
    """One-way ANOVA ICC ``(MSB - MSW) / (MSB + (n0 - 1) MSW)`` (unbalanced ``n0``).

    With ``rng`` a percentile cluster-bootstrap interval is added (clusters resampled with
    replacement; M3 bootstrap). Use pre-experiment data: the experiment's own outcomes contain
    the treatment effect between clusters.
    """
    y = np.asarray(outcome, dtype=np.float64)
    labels = np.asarray(clusters)
    if y.shape != labels.shape or y.ndim != 1 or not np.isfinite(y).all():
        raise ValueError("outcome and clusters must be finite 1-D arrays of equal length")
    unique, inverse = np.unique(labels, return_inverse=True)
    if unique.size < 3:
        raise ValueError("need at least three clusters")
    if y.size <= unique.size:
        raise ValueError("within-cluster variance undefined: every cluster has a single unit")
    sizes = np.bincount(inverse).astype(float)
    # Explicit float64: NumPy 2.4 stubs type a weighted bincount as integer (it is float64).
    sums = np.asarray(np.bincount(inverse, weights=y), dtype=np.float64)
    sums_sq = np.asarray(np.bincount(inverse, weights=y * y), dtype=np.float64)
    icc = _icc_from_sums(sizes, sums, sums_sq)
    lo = hi = None
    if rng is not None:
        def statistic(index: FloatArray) -> float:
            chosen = index.astype(int)
            return _icc_from_sums(sizes[chosen], sums[chosen], sums_sq[chosen])

        boot = bootstrap_ci(np.arange(unique.size, dtype=float), statistic, rng=rng,
                            confidence_level=confidence_level, n_resamples=n_bootstrap,
                            method="percentile")
        lo, hi = boot.ci_lower, boot.ci_upper
    return ICCEstimate(icc, lo, hi, int(unique.size), float(sizes.mean()),
                       float(sizes.std(ddof=0) / sizes.mean()))


# ---------------------------------------------------------------------------
# Cluster-randomised planning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClusterPlan:
    clusters_per_arm: int
    mean_cluster_size: float
    icc: float
    cluster_size_cv: float
    design_effect: float
    individual_plan: ExperimentPlan
    """The M6 plan if units were randomised individually (no clustering)."""
    planned_power: float
    """Normal-theory power with a ``t(2G - 1)`` critical value at ``clusters_per_arm``."""
    naive_clusters_per_arm: int
    """Clusters per arm if the ICC were (wrongly) taken as zero."""

    @property
    def units_per_arm(self) -> float:
        return self.clusters_per_arm * self.mean_cluster_size


def cluster_power(
    baseline: float, mde_absolute: float, *, clusters_per_arm: int, mean_cluster_size: float,
    icc: float, cluster_size_cv: float = 0.0, alpha: float = 0.05,
) -> float:
    """Approximate power of the CR1 analysis: effective size ``G m / DE`` per arm, unpooled
    binomial variance, two-sided ``t(2G - 1)`` critical value."""
    if clusters_per_arm < 2:
        raise ValueError("need at least two clusters per arm")
    p_c, p_t = baseline, baseline + mde_absolute
    deff = design_effect(mean_cluster_size, icc, cluster_size_cv=cluster_size_cv)
    n_eff = clusters_per_arm * mean_cluster_size / deff
    se = math.sqrt((p_t * (1 - p_t) + p_c * (1 - p_c)) / n_eff)
    critical = float(stats.t.ppf(1 - alpha / 2, 2 * clusters_per_arm - 1))
    shift = abs(mde_absolute) / se
    return float(stats.norm.sf(critical - shift) + stats.norm.cdf(-critical - shift))


def plan_cluster_randomised(
    baseline: float,
    *,
    mde_absolute: float,
    mean_cluster_size: float,
    icc: float,
    cluster_size_cv: float = 0.0,
    alpha: float = 0.05,
    power: float = 0.8,
    max_clusters_per_arm: int = 10_000,
) -> ClusterPlan:
    """Smallest number of clusters per arm reaching ``power`` (see :func:`cluster_power`)."""
    power = require_probability(power, "power")
    individual = plan_binary(baseline, mde_absolute=mde_absolute, alpha=alpha, power=power)
    deff = design_effect(mean_cluster_size, icc, cluster_size_cv=cluster_size_cv)

    def smallest(icc_value: float) -> int:
        g = max(2, math.ceil(individual.n_treatment
                             * design_effect(mean_cluster_size, icc_value,
                                             cluster_size_cv=cluster_size_cv)
                             / mean_cluster_size))
        while cluster_power(baseline, mde_absolute, clusters_per_arm=g,
                            mean_cluster_size=mean_cluster_size, icc=icc_value,
                            cluster_size_cv=cluster_size_cv, alpha=alpha) < power:
            g += 1
            if g > max_clusters_per_arm:
                raise ValueError("target power not reachable within max_clusters_per_arm")
        return g

    g = smallest(icc)
    planned = cluster_power(baseline, mde_absolute, clusters_per_arm=g,
                            mean_cluster_size=mean_cluster_size, icc=icc,
                            cluster_size_cv=cluster_size_cv, alpha=alpha)
    return ClusterPlan(g, mean_cluster_size, icc, cluster_size_cv, deff, individual, planned,
                       smallest(0.0))


# ---------------------------------------------------------------------------
# Switchback experiments
# ---------------------------------------------------------------------------


def switchback_assignment(
    n_periods: int, *, rng: np.random.Generator, strata: ArrayLike | None = None
) -> np.ndarray:
    """0/1 treatment state per period.

    With ``strata`` (e.g. the time-of-day block of each period) periods are randomised within
    each stratum (M6 stratified randomisation), so every block has (as near as possible) half
    its periods treated; otherwise complete randomisation (M6). Stratifying on time of day
    removes the largest source of variance in a marketplace series by design.
    """
    if strata is None:
        labels = complete_randomisation(n_periods, rng=rng)
    else:
        blocks = np.asarray(strata)
        if blocks.shape != (n_periods,):
            raise ValueError("one stratum label per period is required")
        labels = stratified_randomisation(blocks, rng=rng)
    return np.asarray(np.asarray(labels) == "treatment", dtype=np.int64)


@dataclass(frozen=True)
class SwitchbackAnalysis:
    effect: CausalEstimate
    """Global-effect estimate from period means after the burn-in (stratified when strata
    are given)."""
    randomisation_p_value: float
    """Permutation of period labels (within strata), statistic = the estimator itself: exact
    under the sharp null of no effect, whatever the serial correlation."""
    burn_in_hours: int
    n_periods_treated: int
    n_periods_control: int
    naive_hourly: CausalEstimate
    """INVALID reference: every hour an independent unit, no burn-in, no strata."""


def period_means(
    hourly: pd.DataFrame, *, outcome: str, burn_in_hours: int, period_hours: int
) -> pd.DataFrame:
    """Mean outcome per period over hours ``>= burn_in_hours`` into the period, with the
    period's time-of-day block (``hour_of_day // period_hours`` at the period start, also for
    periods that cross midnight)."""
    required = {"period", "hour_in_period", "hour_of_day", "arm", outcome}
    if not required.issubset(hourly.columns):
        raise ValueError(f"hourly frame needs columns {sorted(required)}")
    if not 0 <= burn_in_hours < period_hours:
        raise ValueError("burn_in_hours must leave at least one hour per period")
    kept = hourly.loc[hourly["hour_in_period"] >= burn_in_hours]
    if (kept.groupby("period")["arm"].nunique() > 1).any():
        raise ValueError("a period contains both arms: the period is the randomisation unit")
    start_hour = (kept["hour_of_day"] - kept["hour_in_period"]) % 24
    grouped = kept.assign(_start_hour=start_hour).groupby("period", sort=True)
    return pd.DataFrame({
        "arm": grouped["arm"].first(),
        "mean": grouped[outcome].mean(),
        "block": grouped["_start_hour"].first() // period_hours,
    }).reset_index()


def _difference(y: FloatArray, treated: np.ndarray, strata: np.ndarray | None) -> float:
    if strata is None:
        return float(y[treated].mean() - y[~treated].mean())
    total = 0.0
    for s in np.unique(strata):
        in_s = strata == s
        total += in_s.mean() * (y[in_s & treated].mean() - y[in_s & ~treated].mean())
    return float(total)


def randomisation_p_value(
    y: ArrayLike, treated: ArrayLike, *, rng: np.random.Generator, strata: ArrayLike | None = None,
    n_permutations: int = 1999,
) -> float:
    """Two-sided Monte Carlo randomisation p-value ``(b + 1) / (m + 1)`` re-drawing the
    assignment as the design did (within strata when given)."""
    values = np.asarray(y, dtype=np.float64)
    z = np.asarray(treated, dtype=bool)
    if strata is not None:
        blocks = np.asarray(strata)
        for s in np.unique(blocks):
            if np.unique(z[blocks == s]).size < 2:
                raise ValueError(f"stratum {s} lacks a treated or a control unit: the "
                                 "within-stratum contrast is undefined")
    if strata is None:  # complete randomisation: the M3 permutation test
        return permutation_test(values[z], values[~z], n_permutations=n_permutations,
                                rng=rng).p_value
    blocks = np.asarray(strata)
    observed = abs(_difference(values, z, blocks))
    tolerance = 1e-12 * max(1.0, observed)
    groups = [np.flatnonzero(blocks == s) for s in np.unique(blocks)]
    exceed = 0
    for _ in range(n_permutations):
        permuted = z.copy()
        for index in groups:
            permuted[index] = rng.permutation(z[index])
        exceed += abs(_difference(values, permuted, blocks)) >= observed - tolerance
    return (exceed + 1) / (n_permutations + 1)


def analyse_switchback(
    hourly: pd.DataFrame,
    *,
    outcome: str = "orders",
    period_hours: int,
    burn_in_hours: int,
    rng: np.random.Generator,
    stratified: bool = True,
    n_permutations: int = 1999,
    confidence_level: float = 0.95,
) -> SwitchbackAnalysis:
    """Period-level analysis of a switchback experiment (see the module docstring).

    ``stratified=True`` analyses as randomised by time-of-day block: M6
    ``stratified_difference`` (Neyman variance) and a within-block permutation test.
    """
    level = require_probability(confidence_level, "confidence_level")
    periods = period_means(hourly, outcome=outcome, burn_in_hours=burn_in_hours,
                           period_hours=period_hours)
    y = periods["mean"].to_numpy(float)
    arm = periods["arm"].to_numpy(str)
    treated = arm == "treatment"
    blocks = periods["block"].to_numpy() if stratified else None
    assumptions = ("periods randomised (within time-of-day blocks)" if stratified
                   else "periods randomised", "carryover shorter than the burn-in",
                   "no anticipation")
    if stratified:
        base = stratified_difference(y, arm, periods["block"].to_numpy(),
                                     confidence_level=level)
        se_method = f"Neyman stratified variance over {y.size} period means"
    else:
        base = analyse_continuous(y, arm, confidence_level=level).effect
        se_method = f"Welch on {y.size} period means"
    effect = CausalEstimate(
        estimand="global effect: all-on vs all-off mean outcome per hour",
        estimator=f"difference in period means after a {burn_in_hours}-hour burn-in"
        + (" (stratified by time of day)" if stratified else ""),
        estimate=base.estimate, standard_error=base.standard_error, ci_lower=base.ci_lower,
        ci_upper=base.ci_upper, confidence_level=level, se_method=se_method, n=y.size,
        n_treated=int(treated.sum()), adjustment_set=(), assumptions=assumptions,
    )
    p_value = randomisation_p_value(y, treated, rng=rng, strata=blocks,
                                    n_permutations=n_permutations)
    naive = analyse_continuous(hourly[outcome].to_numpy(float), hourly["arm"].to_numpy(str),
                               metric=f"{outcome} per hour (hourly, invalid)",
                               confidence_level=level).effect
    return SwitchbackAnalysis(effect, p_value, burn_in_hours, int(treated.sum()),
                              int((~treated).sum()), naive)


__all__ = [
    "ClusterPlan",
    "ICCEstimate",
    "SwitchbackAnalysis",
    "analyse_switchback",
    "cluster_power",
    "estimate_icc",
    "period_means",
    "plan_cluster_randomised",
    "randomisation_p_value",
    "switchback_assignment",
]
