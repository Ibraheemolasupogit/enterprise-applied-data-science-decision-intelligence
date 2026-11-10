"""Difference-in-differences and event studies for (possibly staggered) panel data.

Notation and timing follow :mod:`decision_intelligence.data.rollout_panel`:
periods ``1..T``, cohort ``g`` = first treated period (0 = never treated),
absorbing treatment ``D_it = 1{g > 0, t >= g}``, event time ``e = t - g``.

Estimands
---------
* ``ATT(g, t) = E[Y_t(g) - Y_t(0) | G = g]`` — the effect at period ``t`` for
  units first treated at ``g`` (Callaway & Sant'Anna 2021).
* Overall ATT: the average of ``ATT(g, t)`` over post-treatment cells weighted
  by cohort size = the average effect over all treated unit-periods.
* Event-time ``ATT(e)``: cohort-size-weighted average of ``ATT(g, g + e)``
  over cohorts observed ``e`` periods after adoption. Composition changes
  with ``e`` in an unbalanced event window.

Identification (for the post-treatment ATT(g, t))
-------------------------------------------------
1. **Parallel trends**: absent treatment, cohort ``g`` and the comparison
   group would have had the same average change between the base period and
   ``t``. Untestable for post periods. Pre-period estimates (leads) test an
   *implication* of it before treatment; a non-rejection is not proof
   (low power, and conditioning on passing a pre-test distorts inference —
   Roth 2022).
2. **No anticipation** (or a known anticipation window ``k``, in which case the
   base period moves to ``g - 1 - k``).
3. **No spillovers (SUTVA)**: comparison units are unaffected by others'
   treatment.
4. Absorbing treatment and a balanced panel (enforced).

Estimators
----------
* :func:`two_period_did` — the canonical 2×2: difference in mean changes,
  Welch interval on unit-level changes (correct clustering by construction).
* :func:`twfe_did` — static two-way fixed effects, cluster-robust (unit) SEs.
  With **staggered adoption and heterogeneous effects the coefficient is a
  weighted average of ATT(i, t) with possibly negative weights**
  (de Chaisemartin & D'Haultfœuille 2020; Goodman-Bacon 2021) — it is
  flagged, and :func:`twfe_weights` exposes the weights exactly.
* :func:`staggered_did` — Callaway & Sant'Anna group-time ATTs with
  never-treated or not-yet-treated comparisons, aggregated to the overall and
  event-time ATT; unit-level (cluster) bootstrap via the shared M3 bootstrap.
* :func:`event_study_twfe` — dynamic TWFE with relative-time indicators;
  interpretable under common timing, contaminated across cohorts under
  staggered heterogeneous effects (Sun & Abraham 2021) and flagged so.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from decision_intelligence.causal_inference.estimators import CausalEstimate
from decision_intelligence.common.bootstrap import bootstrap_ci
from decision_intelligence.common.validation import (
    FloatArray,
    require_positive_int,
    require_probability,
)
from decision_intelligence.statistics.hypothesis_tests import two_sample_t_test
from decision_intelligence.statistics.linear_regression import (
    LinearRegressionResult,
    fit_ols,
)

ControlGroup = Literal["never", "not_yet"]

DID_ASSUMPTIONS = (
    "parallel trends between treated cohort and comparison group (untestable for post periods)",
    "no anticipation before the stated base period",
    "no spillovers onto comparison units (SUTVA)",
    "absorbing treatment; balanced panel",
)


# ---------------------------------------------------------------------------
# Panel handling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Panel:
    """A validated balanced panel in wide form."""

    units: np.ndarray
    periods: np.ndarray
    cohort: np.ndarray
    """Cohort per unit (aligned with ``units``)."""
    outcome: FloatArray
    """``n_units × n_periods`` matrix."""

    @property
    def treated(self) -> np.ndarray:
        return (self.cohort[:, None] > 0) & (self.periods[None, :] >= self.cohort[:, None])

    def subset(self, rows: np.ndarray) -> Panel:
        return Panel(self.units[rows], self.periods, self.cohort[rows], self.outcome[rows])


def as_panel(
    frame: pd.DataFrame,
    *,
    unit: str = "unit_id",
    period: str = "period",
    cohort: str = "cohort",
    outcome: str = "outcome",
) -> Panel:
    """Validate a long panel and pivot it: balanced, one row per unit-period, constant cohort."""
    for column in (unit, period, cohort, outcome):
        if column not in frame.columns:
            raise ValueError(f"panel is missing column {column!r}")
    if frame[[unit, period]].duplicated().any():
        raise ValueError("duplicate unit-period rows")
    if frame[outcome].isna().any() or not np.isfinite(frame[outcome].to_numpy(float)).all():
        raise ValueError("outcome contains missing or non-finite values")
    wide = frame.pivot(index=unit, columns=period, values=outcome)
    if wide.isna().any().any():
        raise ValueError("panel is unbalanced: every unit must be observed in every period")
    cohorts = frame.groupby(unit)[cohort].nunique()
    if (cohorts > 1).any():
        raise ValueError("cohort must be constant within unit")
    cohort_by_unit = frame.groupby(unit)[cohort].first().reindex(wide.index).to_numpy(int)
    periods = wide.columns.to_numpy(int)
    if not np.array_equal(periods, np.arange(periods[0], periods[0] + periods.size)):
        raise ValueError("periods must be consecutive integers")
    if ((cohort_by_unit > 0) & (cohort_by_unit <= periods[0])).any():
        raise ValueError("units treated in the first period have no pre-treatment observation")
    return Panel(
        units=wide.index.to_numpy(),
        periods=periods,
        cohort=cohort_by_unit,
        outcome=wide.to_numpy(dtype=np.float64),
    )


def _fit_fixed_effects(
    design: pd.DataFrame, y: FloatArray, unit_index: np.ndarray, confidence_level: float
) -> tuple[LinearRegressionResult, list[str]]:
    """OLS with unit and period dummies, SEs clustered by unit (M3 ``fit_ols``, CR1).

    Unit dummies are nested within the unit clusters, so the cluster-robust
    covariance of those nuisance coefficients is degenerate (their cluster
    scores sum to zero); rounding can make their variances slightly negative,
    giving NaN standard errors for the dummies only. They are never reported,
    so that numerical warning is suppressed here. The regression's overall F
    test (over hundreds of dummies) is likewise meaningless for this purpose
    and its warning is dropped.
    """
    with np.errstate(invalid="ignore"):
        regression = fit_ols(
            design, y, covariance="cluster", clusters=unit_index,
            confidence_level=confidence_level,
        )
    warnings = [w for w in regression.warnings if not w.startswith("joint F-test not computed")]
    return regression, warnings


def _long_design(panel: Panel) -> tuple[pd.DataFrame, FloatArray, np.ndarray]:
    """Unit and period dummies (first level dropped) for a long-form regression."""
    n, t = panel.outcome.shape
    unit_index = np.repeat(np.arange(n), t)
    period_index = np.tile(np.arange(t), n)
    columns: dict[str, FloatArray] = {}
    for k in range(1, n):
        columns[f"unit_{k}"] = (unit_index == k).astype(float)
    for k in range(1, t):
        columns[f"period_{k}"] = (period_index == k).astype(float)
    return pd.DataFrame(columns), panel.outcome.ravel(), unit_index


# ---------------------------------------------------------------------------
# 2×2 and static TWFE
# ---------------------------------------------------------------------------


def two_period_did(
    frame: pd.DataFrame, pre_period: int, post_period: int, *, confidence_level: float = 0.95
) -> CausalEstimate:
    """Canonical 2×2 DiD between two periods.

    Treated group: units untreated at ``pre_period`` and treated at
    ``post_period``. Comparison: units untreated at both. Units already
    treated at ``pre_period`` are excluded. Estimand: ATT at ``post_period`` for
    the units switching in between. Inference: Welch interval on unit-level
    changes ``Y_post - Y_pre`` — one observation per unit, so arbitrary serial
    correlation within unit is handled.
    """
    panel = as_panel(frame)
    p = {int(v): k for k, v in enumerate(panel.periods)}
    if pre_period not in p or post_period not in p or pre_period >= post_period:
        raise ValueError("need pre_period < post_period, both in the panel")
    treated = panel.treated
    switch = ~treated[:, p[pre_period]] & treated[:, p[post_period]]
    control = ~treated[:, p[pre_period]] & ~treated[:, p[post_period]]
    if switch.sum() < 2 or control.sum() < 2:
        raise ValueError("need at least two switching and two comparison units")
    change = panel.outcome[:, p[post_period]] - panel.outcome[:, p[pre_period]]
    test = two_sample_t_test(change[switch], change[control], confidence_level=confidence_level)
    if test.ci_lower is None or test.ci_upper is None or test.df is None:  # pragma: no cover
        raise RuntimeError("Welch test returned no interval")
    se = (test.ci_upper - test.ci_lower) / (
        2 * float(stats.t.ppf(0.5 + confidence_level / 2, test.df))
    )
    return CausalEstimate(
        estimand=f"ATT at period {post_period} (2×2 DiD vs period {pre_period})",
        estimator="difference in mean changes",
        estimate=test.estimate,
        standard_error=se,
        ci_lower=test.ci_lower,
        ci_upper=test.ci_upper,
        confidence_level=confidence_level,
        se_method="Welch on unit-level changes",
        n=int(switch.sum() + control.sum()),
        n_treated=int(switch.sum()),
        adjustment_set=("unit fixed effect (differenced)", "common period shock"),
        assumptions=DID_ASSUMPTIONS,
    )


@dataclass(frozen=True)
class TWFEResult:
    estimate: CausalEstimate
    regression: LinearRegressionResult = field(repr=False)
    staggered: bool


def twfe_did(frame: pd.DataFrame, *, confidence_level: float = 0.95) -> TWFEResult:
    """Static TWFE ``Y_it = alpha_i + lambda_t + beta D_it + e_it``; SEs clustered by unit (CR1).

    Interpretable as an ATT only with a single adoption date or homogeneous
    effects. Under staggered adoption the result carries a warning; use
    :func:`staggered_did` for the estimand and :func:`twfe_weights` to see
    what the coefficient actually averages.
    """
    panel = as_panel(frame)
    dummies, y, unit_index = _long_design(panel)
    design = pd.concat(
        [pd.DataFrame({"treated": panel.treated.ravel().astype(float)}), dummies], axis=1
    )
    regression, warnings = _fit_fixed_effects(design, y, unit_index, confidence_level)
    cohorts = sorted({int(g) for g in panel.cohort if g > 0})
    staggered = len(cohorts) > 1
    if staggered:
        warnings.append(
            "staggered adoption: the TWFE coefficient averages unit-period effects with weights "
            "that can be negative (later-treated units use earlier-treated ones as "
            "controls); it is not an ATT under heterogeneous or dynamic effects — use "
            "staggered_did"
        )
    k = regression.feature_names.index("treated")
    return TWFEResult(
        estimate=CausalEstimate(
            estimand=(
                "TWFE coefficient (equals the ATT only with common timing or homogeneous "
                "effects)"
            ),
            estimator="two-way fixed effects OLS",
            estimate=float(regression.coefficients[k]),
            standard_error=float(regression.standard_errors[k]),
            ci_lower=float(regression.ci_lower[k]),
            ci_upper=float(regression.ci_upper[k]),
            confidence_level=confidence_level,
            se_method=f"cluster-robust by unit (CR1, t with {regression.df_resid:g} df)",
            n=int(y.size),
            n_treated=int(panel.treated.sum()),
            adjustment_set=("unit fixed effects", "period fixed effects"),
            assumptions=(*DID_ASSUMPTIONS, "homogeneous effects if adoption is staggered"),
            warnings=tuple(warnings),
        ),
        regression=regression,
        staggered=staggered,
    )


def two_way_demean(matrix: FloatArray) -> FloatArray:
    """Residual from unit and period fixed effects in a balanced panel (exact FWL step)."""
    return np.asarray(
        matrix - matrix.mean(axis=1, keepdims=True) - matrix.mean(axis=0, keepdims=True)
        + matrix.mean(),
        dtype=np.float64,
    )


def twfe_weights(frame: pd.DataFrame) -> pd.DataFrame:
    """Weights ``w_it`` with ``beta_TWFE = sum_{treated (i,t)} w_it tau_it + noise term``.

    By Frisch–Waugh–Lovell, ``beta = sum D~_it Y_it / sum D~_it^2`` with ``D~``
    the two-way demeaned treatment. If ``Y_it = alpha_i + lambda_t +
    tau_it D_it + e_it`` then ``beta = sum_{D=1} w_it tau_it + sum D~ e / sum D~^2``
    with ``w_it = D~_it / sum D~^2`` (the weights sum to 1 over treated cells).
    Negative weights mean some treatment effects enter the coefficient with
    the wrong sign. Exact for a balanced panel (de Chaisemartin & D'Haultfœuille 2020).
    """
    panel = as_panel(frame)
    treated = panel.treated.astype(float)
    residual = two_way_demean(treated)
    denominator = float(np.sum(residual**2))
    if denominator == 0:
        raise ValueError("treatment has no variation after removing unit and period effects")
    rows, cols = np.nonzero(treated)
    return pd.DataFrame(
        {
            "unit_id": panel.units[rows],
            "period": panel.periods[cols],
            "cohort": panel.cohort[rows],
            "event_time": panel.periods[cols] - panel.cohort[rows],
            "weight": residual[rows, cols] / denominator,
        }
    )


# ---------------------------------------------------------------------------
# Callaway & Sant'Anna group-time ATTs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupTimeATT:
    cohort: int
    period: int
    event_time: int
    estimate: float
    n_treated: int
    n_control: int
    base_period: int
    influence: FloatArray = field(repr=False, compare=False)
    """Per-unit influence function; ``sum(influence**2)`` is the cell's Welch variance."""


def group_time_att(
    panel: Panel, *, control_group: ControlGroup = "never", anticipation: int = 0
) -> list[GroupTimeATT]:
    """All identifiable ATT(g, t), including pre-period placebo cells.

    Base period: ``g - 1 - anticipation`` for every ``t`` (a "universal" base,
    so pre-period estimates are cumulative placebo differences relative to
    the same base as the post-period effects). Comparison units at ``t``:
    never treated, or (``not_yet``) also those whose cohort starts after both
    ``t`` and the base period + anticipation.
    """
    if anticipation < 0:
        raise ValueError("anticipation must be non-negative")
    index = {int(v): k for k, v in enumerate(panel.periods)}
    cells: list[GroupTimeATT] = []
    for g in sorted({int(c) for c in panel.cohort if c > 0}):
        base = g - 1 - anticipation
        if base not in index:
            continue
        members = panel.cohort == g
        for t in panel.periods:
            t = int(t)
            if t == base:
                continue
            horizon = max(t, base) + anticipation
            if control_group == "never":
                controls = panel.cohort == 0
            elif control_group == "not_yet":
                controls = (panel.cohort == 0) | (panel.cohort > horizon)
                controls &= panel.cohort != g
            else:
                raise ValueError("control_group must be 'never' or 'not_yet'")
            if members.sum() == 0 or controls.sum() == 0:
                continue
            change = panel.outcome[:, index[t]] - panel.outcome[:, index[base]]
            n_t, n_c = int(members.sum()), int(controls.sum())
            influence = np.zeros(panel.units.size)
            # Difference of two independent means; the n/(n-1) factors make the
            # squared influence sum equal the Welch (unbiased) variance.
            if n_t > 1:
                influence[members] = (change[members] - change[members].mean()) / math.sqrt(
                    n_t * (n_t - 1)
                )
            if n_c > 1:
                influence[controls] = -(change[controls] - change[controls].mean()) / math.sqrt(
                    n_c * (n_c - 1)
                )
            cells.append(
                GroupTimeATT(
                    cohort=g,
                    period=t,
                    event_time=t - g,
                    estimate=float(change[members].mean() - change[controls].mean()),
                    n_treated=n_t,
                    n_control=n_c,
                    base_period=base,
                    influence=influence,
                )
            )
    if not cells:
        raise ValueError("no identifiable group-time cells (need treated cohorts and comparisons)")
    return cells


@dataclass(frozen=True)
class _Aggregates:
    overall: float
    overall_influence: FloatArray
    by_event: dict[int, float]
    event_influence: dict[int, FloatArray]


def _aggregate(
    cells: Sequence[GroupTimeATT], cohort_sizes: dict[int, int]
) -> _Aggregates:
    """Overall post-treatment ATT and event-time ATT(e), weights ∝ cohort size.

    The aggregates are fixed linear combinations of the cell estimates
    (weights treat cohort sizes as fixed), so their influence functions are
    the same combinations of the cells' influence functions.
    """
    post = [c for c in cells if c.event_time >= 0]
    if not post:
        raise ValueError("no post-treatment cells")
    weights = np.array([cohort_sizes[c.cohort] for c in post], dtype=float)
    weights /= weights.sum()
    overall = float(sum(w * c.estimate for w, c in zip(weights, post, strict=True)))
    overall_if = np.sum([w * c.influence for w, c in zip(weights, post, strict=True)], axis=0)
    by_event: dict[int, float] = {}
    event_if: dict[int, FloatArray] = {}
    for e in sorted({c.event_time for c in cells}):
        group = [c for c in cells if c.event_time == e]
        w = np.array([cohort_sizes[c.cohort] for c in group], dtype=float)
        w /= w.sum()
        by_event[e] = float(sum(wk * c.estimate for wk, c in zip(w, group, strict=True)))
        event_if[e] = np.sum([wk * c.influence for wk, c in zip(w, group, strict=True)], axis=0)
    return _Aggregates(overall, np.asarray(overall_if), by_event, event_if)


@dataclass(frozen=True)
class PretrendTest:
    """Joint Wald test that all pre-period (lead) effects are zero.

    A rejection is evidence against parallel trends (or of anticipation).
    A non-rejection is *not* evidence that trends are parallel: the test can
    have little power against violations large enough to bias the estimate.
    """

    statistic: float
    """Wald statistic ``W = b' V^-1 b``."""
    df: int
    p_value: float
    leads_tested: tuple[int, ...]
    reference: str
    note: str = (
        "non-rejection does not establish parallel trends; it fails to detect a violation of "
        "the tested form with this sample"
    )


def pretrend_wald_test(
    estimates: FloatArray,
    covariance: FloatArray,
    leads: Sequence[int],
    *,
    n_clusters: int | None = None,
) -> PretrendTest:
    """Joint Wald test of ``b = 0`` for pre-period estimates.

    With a cluster-robust covariance from ``G`` clusters the statistic is
    referred to the Hotelling-type finite-cluster distribution
    ``(G - q) / (q (G - 1)) W ~ F(q, G - q)``; the chi-square(q) reference
    over-rejects badly when ``q`` is not small relative to ``G`` (generated
    evidence, M5 results §2b: all leads with chi-square rejects 34% at nominal
    5% with 60 units; the windowed F test 5.4%). Without
    ``n_clusters`` the chi-square reference is used. Refuses a singular
    covariance and ``q >= G``.
    """
    b = np.asarray(estimates, dtype=np.float64)
    v = np.asarray(covariance, dtype=np.float64)
    q = int(b.size)
    if q == 0:
        raise ValueError("no pre-period estimates to test")
    if int(np.linalg.matrix_rank(v)) < q:
        raise ValueError("pre-period covariance is singular; the joint test is not identified")
    statistic = float(b @ np.linalg.solve(v, b))
    if n_clusters is None:
        p_value = float(stats.chi2.sf(statistic, q))
        reference = f"chi-square({q})"
    else:
        if n_clusters <= q:
            raise ValueError("need more clusters than tested leads for the finite-cluster F test")
        scaled = (n_clusters - q) / (q * (n_clusters - 1)) * statistic
        p_value = float(stats.f.sf(scaled, q, n_clusters - q))
        reference = f"F({q}, {n_clusters - q}) finite-cluster (Hotelling) reference"
    return PretrendTest(
        statistic=statistic,
        df=q,
        p_value=p_value,
        leads_tested=tuple(int(e) for e in leads),
        reference=reference,
    )


@dataclass(frozen=True)
class EventTimeEstimate:
    """One event-time effect with a pointwise interval and (optionally) a simultaneous band.

    ``ci_*`` are *pointwise* 95% intervals: over the reported event window they
    all cover the truth simultaneously in only 0.492 of panels (M5 results §2b).
    ``band_*`` are *nominal* 95% sup-t simultaneous bands over that window (NaN
    outside it); use them to read the whole path at once, but note they
    under-cover in the evidence (observed joint coverage 0.920, Monte Carlo 95%
    interval [0.889, 0.943]) — nominal coverage is not guaranteed.
    """

    event_time: int
    estimate: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    band_lower: float = math.nan
    band_upper: float = math.nan


def sup_t_critical_value(
    covariance: FloatArray, confidence_level: float = 0.95, *, n_draws: int = 200_000
) -> float:
    """Critical value ``c`` with ``P(max_k |Z_k| <= c) = confidence`` for ``Z ~ N(0, corr(V))``.

    The sup-t simultaneous band of Montiel Olea & Plagborg-Møller (2019):
    ``estimate_k ± c · se_k`` covers all coefficients jointly. Computed by
    simulation from a fixed seed, so it is deterministic; with 200,000 draws
    the Monte Carlo error in ``c`` is ~0.005.
    """
    v = np.asarray(covariance, dtype=np.float64)
    sd = np.sqrt(np.diag(v))
    if (sd <= 0).any():
        raise ValueError("all event-time variances must be positive")
    correlation = v / np.outer(sd, sd)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    root = eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))
    draws = np.random.default_rng(20260509).standard_normal((n_draws, v.shape[0])) @ root.T
    return float(np.quantile(np.max(np.abs(draws), axis=1), confidence_level))


@dataclass(frozen=True)
class StaggeredDiDResult:
    overall: CausalEstimate
    cells: tuple[GroupTimeATT, ...]
    event_study: tuple[EventTimeEstimate, ...]
    """Pointwise intervals plus a sup-t simultaneous band over the reported window."""
    pretrend: PretrendTest | None
    control_group: ControlGroup
    anticipation: int
    event_covariance: FloatArray = field(repr=False, default_factory=lambda: np.empty((0, 0)))
    """Covariance of the event-time estimates, in the order of ``event_study``."""


def staggered_did(
    frame: pd.DataFrame,
    *,
    rng: np.random.Generator | None = None,
    control_group: ControlGroup = "never",
    anticipation: int = 0,
    se_method: Literal["influence_function", "bootstrap"] = "influence_function",
    n_bootstrap: int = 999,
    confidence_level: float = 0.95,
    max_pretrend_leads: int = 6,
) -> StaggeredDiDResult:
    """Callaway–Sant'Anna estimator with unit-level (cluster-robust) inference.

    ``se_method="influence_function"`` (default): every ATT(g, t) is a
    difference of means of unit-level changes, so its influence function is
    exact and units are independent; aggregates and the joint covariance of
    the event-time estimates follow linearly. Inference is *conditional on
    cohort membership* (cohort sizes treated as fixed — exactly so in the
    synthetic rollout design) and uses normal critical values.

    ``se_method="bootstrap"``: units resampled with replacement through the
    shared M3 bootstrap (percentile intervals). Resampling also randomises
    cohort sizes, so it is conservative when they are fixed by design.

    Both are unreliable with few units per cohort or few comparison units
    (warning below 20).

    The pre-trend test uses the leads ``-max_pretrend_leads - 1 - anticipation``
    to ``-2 - anticipation`` (the analysis window) with the finite-cluster F
    reference; testing every available lead with a chi-square reference is
    badly over-sized in small panels.
    """
    confidence_level = require_probability(confidence_level, "confidence_level")
    panel = as_panel(frame)

    def compute(sample: Panel) -> tuple[_Aggregates, list[GroupTimeATT]]:
        cells = group_time_att(sample, control_group=control_group, anticipation=anticipation)
        sizes = {g: int(np.sum(sample.cohort == g)) for g in {c.cohort for c in cells}}
        return _aggregate(cells, sizes), cells

    aggregates, cells = compute(panel)
    event_times = sorted(aggregates.by_event)
    z = float(stats.norm.ppf(0.5 + confidence_level / 2))

    if se_method == "influence_function":
        overall_se = float(np.sqrt(np.sum(aggregates.overall_influence**2)))
        overall_ci = (aggregates.overall - z * overall_se, aggregates.overall + z * overall_se)
        influence_matrix = np.column_stack([aggregates.event_influence[e] for e in event_times])
        event_cov = influence_matrix.T @ influence_matrix
        event_se = np.sqrt(np.diag(event_cov))
        event_ci = [
            (aggregates.by_event[e] - z * se, aggregates.by_event[e] + z * se)
            for e, se in zip(event_times, event_se, strict=True)
        ]
        method = "unit-level influence function (cluster-robust; conditional on cohorts)"
    elif se_method == "bootstrap":
        if rng is None:
            raise ValueError("bootstrap inference requires an explicit rng")
        n_bootstrap = require_positive_int(n_bootstrap, "n_bootstrap", minimum=100)
        replicate_rows: list[FloatArray] = []

        def statistic(indices: FloatArray) -> float:
            try:
                sample_aggregates, _ = compute(panel.subset(indices.astype(int)))
            except ValueError:
                return math.nan
            if any(e not in sample_aggregates.by_event for e in event_times):
                return math.nan
            replicate_rows.append(
                np.array([sample_aggregates.overall,
                          *[sample_aggregates.by_event[e] for e in event_times]])
            )
            return sample_aggregates.overall

        result = bootstrap_ci(
            np.arange(panel.units.size, dtype=np.float64), statistic, rng=rng,
            n_resamples=n_bootstrap, confidence_level=confidence_level, method="percentile",
        )
        # bootstrap_ci evaluates the statistic on the original data first, then on each resample.
        replicates = np.vstack(replicate_rows[1:])
        overall_se = result.standard_error
        overall_ci = (result.ci_lower, result.ci_upper)
        event_cov = np.cov(replicates[:, 1:], rowvar=False).reshape(
            len(event_times), len(event_times)
        )
        event_se = np.sqrt(np.diag(event_cov))
        alpha = 1 - confidence_level
        event_ci = []
        for k in range(1, len(event_times) + 1):
            low, high = np.quantile(replicates[:, k], [alpha / 2, 1 - alpha / 2])
            event_ci.append((float(low), float(high)))
        method = f"unit (cluster) bootstrap ({n_bootstrap}, percentile)"
    else:
        raise ValueError("se_method must be 'influence_function' or 'bootstrap'")

    reference_lead = -1 - anticipation
    window = [e for e in event_times if e >= reference_lead - max_pretrend_leads]
    window_idx = [event_times.index(e) for e in window]
    critical = sup_t_critical_value(event_cov[np.ix_(window_idx, window_idx)], confidence_level)
    event_estimates = tuple(
        EventTimeEstimate(
            e, aggregates.by_event[e], float(se), float(lo), float(hi),
            band_lower=aggregates.by_event[e] - critical * float(se) if e in window else math.nan,
            band_upper=aggregates.by_event[e] + critical * float(se) if e in window else math.nan,
        )
        for e, se, (lo, hi) in zip(event_times, event_se, event_ci, strict=True)
    )
    leads = [
        e for e in event_times if reference_lead - max_pretrend_leads <= e < reference_lead
    ]
    pretrend = None
    if leads:
        idx = [event_times.index(e) for e in leads]
        try:
            pretrend = pretrend_wald_test(
                np.array([aggregates.by_event[e] for e in leads]),
                event_cov[np.ix_(idx, idx)],
                leads,
                n_clusters=int(panel.units.size),
            )
        except ValueError:
            pretrend = None

    warnings: list[str] = []
    smallest = min(min(c.n_treated, c.n_control) for c in cells)
    if smallest < 20:
        warnings.append(
            f"smallest cohort or comparison group has {smallest} units: unit-level inference "
            "rests on few clusters"
        )
    lead_cells = [c for c in cells if c.event_time in leads]
    if pretrend is not None and lead_cells:
        smallest_lead = min(min(c.n_treated, c.n_control) for c in lead_cells)
        if smallest_lead < 10:
            warnings.append(
                f"pre-trend test uses groups of only {smallest_lead} units: it is over-sized in "
                "this regime (≈10% rejection at nominal 5% with 5 units per group in the evidence)"
            )
    if anticipation:
        warnings.append(f"base period set {anticipation} period(s) before adoption (anticipation)")
    return StaggeredDiDResult(
        overall=CausalEstimate(
            estimand="overall ATT (average effect over treated unit-periods)",
            estimator=f"Callaway–Sant'Anna, {control_group}-treated comparisons",
            estimate=aggregates.overall,
            standard_error=overall_se,
            ci_lower=float(overall_ci[0]),
            ci_upper=float(overall_ci[1]),
            confidence_level=confidence_level,
            se_method=method,
            n=int(panel.outcome.size),
            n_treated=int(panel.treated.sum()),
            adjustment_set=("unit-level differencing", "comparison-group period change"),
            assumptions=DID_ASSUMPTIONS,
            warnings=tuple(warnings),
        ),
        cells=tuple(cells),
        event_study=event_estimates,
        pretrend=pretrend,
        control_group=control_group,
        anticipation=anticipation,
        event_covariance=np.asarray(event_cov, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Dynamic TWFE event study
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventStudyTWFE:
    coefficients: tuple[EventTimeEstimate, ...]
    pretrend: PretrendTest | None
    reference_period: int
    regression: LinearRegressionResult = field(repr=False)
    warnings: tuple[str, ...] = ()


def event_study_twfe(
    frame: pd.DataFrame,
    *,
    leads: int,
    lags: int,
    reference: int = -1,
    confidence_level: float = 0.95,
) -> EventStudyTWFE:
    """Dynamic TWFE: ``Y = alpha_i + lambda_t + sum_{e != ref} beta_e 1{t - g = e} + u``.

    Event times below ``-leads`` and above ``lags`` are binned into the end
    points; never-treated units have all indicators zero. SEs clustered by
    unit. Under staggered adoption with heterogeneous effects each
    ``beta_e`` is contaminated by effects at other event times (Sun & Abraham
    2021) — flagged; use :func:`staggered_did` there.
    """
    if leads < 1 or lags < 0:
        raise ValueError("need leads >= 1 and lags >= 0")
    if not -leads <= reference < 0:
        raise ValueError("reference must be a pre-period event time within the window")
    panel = as_panel(frame)
    dummies, y, unit_index = _long_design(panel)
    ever = (panel.cohort > 0)[:, None]
    event = np.where(ever, panel.periods[None, :] - panel.cohort[:, None], 10**6).ravel()
    binned = np.where(event < 10**6, np.clip(event, -leads, lags), 10**6)
    columns: dict[str, FloatArray] = {}
    window = [e for e in range(-leads, lags + 1) if e != reference]
    for e in window:
        columns[f"event_{e}"] = (binned == e).astype(float)
    design = pd.concat([pd.DataFrame(columns), dummies], axis=1)
    regression, warnings = _fit_fixed_effects(design, y, unit_index, confidence_level)
    estimates = []
    for e in window:
        k = regression.feature_names.index(f"event_{e}")
        estimates.append(
            EventTimeEstimate(
                e,
                float(regression.coefficients[k]),
                float(regression.standard_errors[k]),
                float(regression.ci_lower[k]),
                float(regression.ci_upper[k]),
            )
        )
    lead_terms = [e for e in window if e < reference]
    pretrend = None
    if lead_terms:
        idx = [regression.feature_names.index(f"event_{e}") for e in lead_terms]
        pretrend = pretrend_wald_test(
            regression.coefficients[idx], regression.covariance[np.ix_(idx, idx)], lead_terms,
            n_clusters=int(panel.units.size),
        )
    if len({int(g) for g in panel.cohort if g > 0}) > 1:
        warnings.append(
            "staggered adoption: dynamic TWFE coefficients mix effects across event times and "
            "cohorts under heterogeneous effects (Sun & Abraham 2021); prefer staggered_did"
        )
    return EventStudyTWFE(
        coefficients=tuple(estimates),
        pretrend=pretrend,
        reference_period=reference,
        regression=regression,
        warnings=tuple(warnings),
    )


def twfe_design(frame: pd.DataFrame) -> tuple[FloatArray, FloatArray, np.ndarray, int]:
    """Full TWFE design ``[intercept, treated, unit dummies, period dummies]``, the outcome,
    unit cluster labels and the treated column index — the exact regression
    :func:`twfe_did` fits, for use with
    :func:`decision_intelligence.causal_inference.cluster_inference.wild_cluster_bootstrap`.
    """
    panel = as_panel(frame)
    dummies, y, unit_index = _long_design(panel)
    design = np.column_stack(
        [np.ones(y.size), panel.treated.ravel().astype(float), dummies.to_numpy(dtype=float)]
    )
    return design, y, unit_index, 1
