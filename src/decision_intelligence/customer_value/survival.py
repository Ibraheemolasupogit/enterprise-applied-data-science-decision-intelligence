"""Survival-regression LTV models for Milestone 11 (P4): E3 and S1.

**Landmark data** (protocol, van Houwelingen 2007). At evaluation cutoff ``C`` the training rows
are P2's landmark stacks (:func:`~.landmarks.training_stack`): for every protocol landmark ``L``
(``L + W <= C - 2``) each customer active at ``L`` with point-in-time features frozen at ``L``.
Covariates are never updated after ``L``; prediction at ``C`` uses features frozen at ``C``.

**Window outcome.** ``N`` = paid months among ``L+1 .. L+W``, from the cancellation status as
recorded by ``C`` (M10's convention; cancellations of months ``< C`` are always recorded by
``C``, and ``L + W <= C - 2``): ``N = min(W, last_paid - L)`` for a churned customer, ``W``
otherwise. It equals P2's ``paid_in_window`` label (tested).

**E3 — discrete-time landmark hazard x contribution by age.** Decision ``k = 1 .. W`` is
whether month ``L + k`` is paid, taken after month ``L + k - 1`` at tenure (age)
``a = a_L + k - 1``. A customer contributes decisions ``k = 1 .. min(N + 1, W)``, the last one
an event when ``N < W``. ``logit h(k | x_L) = gamma_band(a) + x_L' beta``: an unpenalised
piecewise-constant tenure basis on ``TENURE_BAND_EDGES`` (bands with fewer than
``TENURE_BAND_MIN_EVENTS`` training events merged into their lower neighbour — the lowest band
into its upper one — and reported) plus the tier's registry features **except
``E3_EXCLUDED_COVARIATES``** (amendment A8: ``acquisition_cohort``, ``tenure_months``,
``paid_months``, which as frozen linear covariates were extrapolated beyond the landmarks'
range; tenure enters only through the band basis), L2-penalised under
``PENALTY_CONVENTION`` (:mod:`decision_intelligence.statistics.penalised_regression`): every
penalised column, one-hot indicators included (reference = the most frequent training level),
standardised on the person-month rows of the fit; columns with SD 0, and exact duplicates of an
earlier column after standardisation (``DESIGN_DUPLICATE_RULE``, A8), are dropped and
reported. The objective is strictly convex for ``lambda > 0``, so a single Newton start suffices
(the multi-start rule has nothing to disagree on).
``P(month a_C + k paid | x_C) = prod_{i <= k} (1 - h(i | x_C))``; the value multiplies it by the
expected contribution of a paid month at that age (P3's cell x promotion-state
:class:`~.contractual.ContributionModel`, fitted on completely recorded months at ``C``).

**Penalty** (``TUNING_RULE``). If at least ``MIN_TUNING_LANDMARKS`` tuning landmarks exist,
each ``lambda`` in ``PENALTY_GRID`` is fitted on the tuning landmarks using only the tables
as recorded by the validation landmark ``L_v`` and scored on ``L_v``'s completely recorded
labels by ``E3_TUNING_LOSS`` (squared error of window value; A8): the smallest loss wins, ties to
the larger ``lambda``. The final model is refitted with that ``lambda`` on all training
landmarks. Otherwise ``DEFAULT_PENALTY`` is used, labelled
``"fixed penalty: temporal tuning infeasible"``. The same tuning fit at ``L_v`` gives the
selection inputs (:func:`validate_e3`); with no tuning landmark there is no validation fit and
E3 is **not selection-eligible** at that cutoff (``E3_SELECTION_ELIGIBILITY``, A8; no censored
validation windows).

**S1 — Cox proportional hazards** (secondary). Time = months from the landmark to churn
(``N + 1``), censored at ``W``; covariates frozen at the landmark (the same standardised
registry design — all registry features, A8 does not apply — with no tenure basis, so tenure
is a linear covariate and S1 keeps the extrapolation limitation A8 removed from E3); Efron's
method for the heavily tied
monthly times; Breslow baseline ``dH0(t) = d_t / sum_{risk} exp(x' beta)``. Unpenalised.
``S(k | x) = exp(-H0(k) exp(x' beta))``.

Estimation code only: observable tables, never truth, oracles or generator parameters (AST
tested). Repeated customers across landmarks make the returned model-based covariances
optimistic; customer-clustered inference is P7's.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from decision_intelligence.common.validation import FloatArray
from decision_intelligence.customer_value import predictive_protocol as pp
from decision_intelligence.customer_value.contractual import (
    ContributionModel,
    calibration_slope,
    fit_contribution,
)
from decision_intelligence.customer_value.features import (
    PromotionPolicy,
    contractual_features,
    feature_names,
)
from decision_intelligence.customer_value.landmarks import training_stack, view_at
from decision_intelligence.customer_value.model_selection import (
    CandidateResult,
    FitDiagnostics,
    classify_fit,
)
from decision_intelligence.customer_value.panel import EconomicsTables, observe
from decision_intelligence.statistics.penalised_regression import (
    PenalisedLogisticResult,
    Standardiser,
    expit,
    fit_penalised_logistic,
)

FIXED_PENALTY_LABEL = "fixed penalty: temporal tuning infeasible"
TUNED_LABEL = "temporal tuning"
NOT_ELIGIBLE = ("not selection-eligible: no landmark has labels completely recorded before the "
                "validation landmark (E3_SELECTION_ELIGIBILITY)")
CATEGORICAL = ("segment", "channel")


# --- window outcomes and person-months ---------------------------------------------------------
def window_paid(status: pd.DataFrame, customer_id: np.ndarray, landmark: np.ndarray,
                horizon: int) -> np.ndarray:
    """Paid months among ``L+1 .. L+horizon`` for customers active at ``L`` (status as recorded
    by a cutoff at or after ``L + horizon``)."""
    st = status.reindex(customer_id)
    last = np.where(st["churned"].to_numpy(dtype=bool),
                    st["acquisition_month"].to_numpy() + st["paid_months"].to_numpy() - 1,
                    np.iinfo(np.int64).max // 2)
    return np.asarray(np.clip(last - landmark, 0, horizon), dtype=np.int64)


@dataclass(frozen=True)
class PersonMonths:
    row: np.ndarray
    """Index of the landmark row."""
    k: np.ndarray
    """Decision ``1 .. W`` (month ``L + k`` paid?)."""
    age: np.ndarray
    """Tenure at the decision: ``a_L + k - 1``."""
    event: np.ndarray


def person_months(n_paid: np.ndarray, age_at_landmark: np.ndarray, horizon: int
                  ) -> PersonMonths:
    n = np.asarray(n_paid, dtype=np.int64)
    a0 = np.asarray(age_at_landmark, dtype=np.int64)
    m = np.minimum(n + 1, horizon)
    row = np.repeat(np.arange(n.size), m)
    start = np.repeat(np.cumsum(m) - m, m)
    k = np.arange(row.size) - start + 1
    event = (k == n[row] + 1) & (n[row] < horizon)
    return PersonMonths(row, k, a0[row] + k - 1, event)


# --- tenure bands --------------------------------------------------------------------------------
def tenure_band(age: np.ndarray, edges: Sequence[int] = pp.TENURE_BAND_EDGES) -> np.ndarray:
    """Index of the band ``[edges[j], edges[j+1])`` (last band open-ended) containing ``age``."""
    return np.asarray(np.searchsorted(np.asarray(edges), age, side="right") - 1, dtype=np.int64)


def merge_bands(events: np.ndarray, min_events: int = pp.TENURE_BAND_MIN_EVENTS
                ) -> np.ndarray:
    """Map original band -> merged group: a band with fewer than ``min_events`` events joins its
    lower neighbour (repeatedly, from the top); a sparse lowest group joins the next one up."""
    groups = [[j] for j in range(len(events))]
    ev = [int(e) for e in events]
    j = len(groups) - 1
    while j >= 1:
        if ev[j] < min_events:
            groups[j - 1] += groups[j]
            ev[j - 1] += ev[j]
            del groups[j], ev[j]
        j -= 1
    while len(groups) > 1 and ev[0] < min_events:
        groups[1] = groups[0] + groups[1]
        ev[1] += ev[0]
        del groups[0], ev[0]
    out = np.empty(len(events), dtype=np.int64)
    for g, members in enumerate(groups):
        out[members] = g
    return out


def band_labels(mapping: np.ndarray, edges: Sequence[int] = pp.TENURE_BAND_EDGES
                ) -> tuple[str, ...]:
    labels = []
    for g in range(int(np.max(mapping)) + 1):
        members = np.flatnonzero(mapping == g)
        lo = edges[int(np.min(members))]
        top = int(np.max(members))
        hi = edges[top + 1] if top + 1 < len(edges) else None
        labels.append(f"[{lo}, {hi})" if hi is not None else f"[{lo}, inf)")
    return tuple(labels)


def band_basis(group: np.ndarray, n_groups: int) -> FloatArray:
    """One unpenalised indicator column per (merged) tenure band."""
    basis = np.zeros((group.size, n_groups))
    basis[np.arange(group.size), group] = 1.0
    return basis


def tenure_hazards(age: np.ndarray, event: np.ndarray, edges: Sequence[int], *,
                   gradient_tolerance: float = pp.STATUS_RULES.gradient_tolerance
                   ) -> FloatArray:
    """Tenure-only E3: the unpenalised band-basis logistic hazard per band (no merging) —
    E3's own basis and solver with no covariates. Its MLE is ``events / at risk`` per band;
    a band with no decisions has no estimate (NaN)."""
    band = tenure_band(np.asarray(age), edges)
    present, group = np.unique(band, return_inverse=True)
    res = fit_penalised_logistic(band_basis(group, present.size),
                                 np.asarray(event, dtype=np.float64), penalty=0.0,
                                 penalised=np.zeros(present.size, dtype=bool),
                                 gradient_tolerance=gradient_tolerance,
                                 relative_change_tolerance=0.0, max_iterations=500)
    out = np.full(len(edges), np.nan)
    out[present] = expit(res.coefficients)
    return out


# --- design (PENALTY_CONVENTION) -----------------------------------------------------------------
@dataclass(frozen=True)
class Design:
    tier: str
    raw_columns: tuple[str, ...]
    """Numeric features and one-hot indicator names, before dropping."""
    levels: dict[str, tuple[tuple[str, ...], str]]
    """Categorical: (kept non-reference levels, reference level)."""
    standardiser: Standardiser
    duplicates: tuple[str, ...]
    """Columns dropped as exact duplicates of an earlier one after standardisation."""
    columns: tuple[str, ...]
    """Final penalised columns, in order."""

    def raw(self, frame: pd.DataFrame) -> FloatArray:
        parts = []
        for name in self.raw_columns:
            if "=" in name:
                var, level = name.split("=", 1)
                parts.append((frame[var].to_numpy() == level).astype(np.float64))
            else:
                parts.append(frame[name].to_numpy(dtype=np.float64))
        return np.column_stack(parts) if parts else np.zeros((len(frame), 0))

    def matrix(self, frame: pd.DataFrame) -> FloatArray:
        z = self.standardiser.transform(self.raw(frame), self.raw_columns)
        keep = [self.standardiser.names.index(c) for c in self.columns]
        return np.asarray(z[:, keep], dtype=np.float64)


def build_design(frame: pd.DataFrame, tier: str, weights: np.ndarray | None = None,
                 exclude: Sequence[str] = ()) -> Design:
    """Fit the penalised design on the rows of a fit (``weights``: rows per landmark row).
    ``exclude``: registry features a model does not use (E3: ``E3_EXCLUDED_COVARIATES``, A8);
    the registry itself is never changed. Exact duplicates follow ``DESIGN_DUPLICATE_RULE``."""
    names = [n for n in feature_names("contractual", tier) if n not in exclude]
    numeric = [n for n in names if n not in CATEGORICAL]
    levels: dict[str, tuple[tuple[str, ...], str]] = {}
    raw_cols: list[str] = list(numeric)
    w = np.ones(len(frame)) if weights is None else np.asarray(weights, dtype=np.float64)
    for var in CATEGORICAL:
        if var not in names:
            continue
        counts = pd.Series(w).groupby(frame[var].to_numpy()).sum()
        ref = str(sorted(counts.index, key=lambda v: (-counts[v], str(v)))[0])
        kept = tuple(sorted(str(v) for v in counts.index if v != ref))
        levels[var] = (kept, ref)
        raw_cols += [f"{var}={lv}" for lv in kept]
    probe = Design(tier, tuple(raw_cols), levels, Standardiser((), np.zeros(0), np.zeros(0), ()),
                   (), ())
    raw = probe.raw(frame)
    std = Standardiser.fit(raw, raw_cols, weights=w)
    z = std.transform(raw, raw_cols)
    cols: list[str] = []
    dups: list[str] = []
    kept_z: list[np.ndarray] = []
    for j, name in enumerate(std.names):
        col = z[:, j]
        if any(np.max(np.abs(col - other)) <= 1e-10 for other in kept_z):
            dups.append(name)
            continue
        cols.append(name)
        kept_z.append(col)
    return Design(tier, tuple(raw_cols), levels, std, tuple(dups), tuple(cols))


# --- E3: fitting ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class HazardFit:
    design: Design
    band_map: np.ndarray
    """Original tenure band -> merged group."""
    bands: tuple[str, ...]
    result: PenalisedLogisticResult
    n_person_months: int
    n_events: int
    status: str
    reasons: tuple[str, ...]

    @property
    def gamma(self) -> FloatArray:
        return np.asarray(self.result.coefficients[:len(self.bands)], dtype=np.float64)

    @property
    def beta(self) -> FloatArray:
        return np.asarray(self.result.coefficients[len(self.bands):], dtype=np.float64)

    def hazard(self, features: pd.DataFrame, age: np.ndarray) -> FloatArray:
        """``h`` for row-aligned ``features`` (repeated per decision) and decision ages."""
        group = self.band_map[tenure_band(age)]
        eta = self.gamma[group] + (self.design.matrix(features) @ self.beta
                                   if self.beta.size else 0.0)
        return expit(eta)


def fit_hazard(features: pd.DataFrame, n_paid: np.ndarray, horizon: int, tier: str,
               penalty: float, exclude: Sequence[str] = ()) -> HazardFit:
    """Fit E3's hazard on landmark rows ``features`` (registry features + ``tenure_months``)
    with window outcomes ``n_paid``."""
    pm = person_months(n_paid, features["tenure_months"].to_numpy(), horizon)
    rows = features.iloc[pm.row].reset_index(drop=True)
    raw_band = tenure_band(pm.age)
    n_bands = len(pp.TENURE_BAND_EDGES)
    ev_band = np.bincount(raw_band[pm.event], minlength=n_bands)
    band_map = merge_bands(ev_band)
    group = band_map[raw_band]
    n_groups = int(band_map.max()) + 1
    design = build_design(rows, tier, exclude=exclude)
    X = np.column_stack([band_basis(group, n_groups), design.matrix(rows)])
    mask = np.r_[np.zeros(n_groups, dtype=bool), np.ones(len(design.columns), dtype=bool)]
    rules = pp.STATUS_RULES
    res = fit_penalised_logistic(X, pm.event.astype(np.float64), penalty=penalty,
                                 penalised=mask, gradient_tolerance=rules.gradient_tolerance,
                                 relative_change_tolerance=rules.relative_change_tolerance)
    n_events = int(pm.event.sum())
    d = FitDiagnostics(
        has_data=pm.row.size > 0 and n_events > 0, converged=res.converged,
        finite_likelihood=math.isfinite(res.log_likelihood),
        separated_without_penalty=res.separated,
        hessian_positive_definite=res.hessian_positive_definite, n_events=n_events,
        n_parameters=X.shape[1])
    status, reasons = classify_fit(d)
    return HazardFit(design, band_map, band_labels(band_map), res, int(pm.row.size), n_events,
                     status, reasons)


def _landmark_data(tables: EconomicsTables, cutoff: int, horizon: int, tier: str,
                   policy: PromotionPolicy, landmarks: Sequence[int] | None
                   ) -> tuple[pd.DataFrame, np.ndarray]:
    stack = training_stack(tables, cutoff, horizon, "contractual", tier, policy,
                           landmarks=landmarks)
    status = observe(view_at(tables, cutoff), cutoff).status()
    n = window_paid(status, stack["customer_id"].to_numpy(), stack["landmark"].to_numpy(),
                    horizon)
    return stack, n


def _paid(fit: HazardFit, feats: pd.DataFrame, horizon: int) -> FloatArray:
    n = len(feats)
    a0 = feats["tenure_months"].to_numpy()
    k = np.tile(np.arange(1, horizon + 1), n)
    rep = feats.iloc[np.repeat(np.arange(n), horizon)].reset_index(drop=True)
    h = fit.hazard(rep, np.repeat(a0, horizon) + k - 1).reshape(n, horizon)
    return np.asarray(np.cumprod(1.0 - h, axis=1), dtype=np.float64)


def _value_frame(feats: pd.DataFrame, paid: FloatArray, contribution: ContributionModel,
                 status: str, monthly_rate: float) -> pd.DataFrame:
    horizon = paid.shape[1]
    a0 = feats["tenure_months"].to_numpy()
    ages = a0[:, None] + np.arange(1, horizon + 1)[None, :]
    m = contribution.expected(feats["segment"].to_numpy(), feats["channel"].to_numpy(), ages)
    d = (1.0 + monthly_rate) ** -np.arange(1, horizon + 1)
    frame = pd.DataFrame({"segment": feats["segment"], "channel": feats["channel"],
                          "age": a0, "status": status, "prediction": (paid * m).sum(axis=1),
                          "prediction_discounted": (paid * m * d).sum(axis=1)},
                         index=feats.index)
    return frame.join(pd.DataFrame(paid, index=feats.index,
                                   columns=[f"paid_{k}" for k in range(1, horizon + 1)]))


@dataclass(frozen=True)
class TuningRecord:
    penalty: float
    validation_mse: float
    status: str


@dataclass(frozen=True)
class E3Model:
    tier: str
    cutoff: int
    horizon: int
    penalty: float
    penalty_source: str
    tuning: tuple[TuningRecord, ...]
    hazard: HazardFit
    contribution: ContributionModel
    landmarks: tuple[int, ...]


def _validation_labels(tables: EconomicsTables, cutoff: int, horizon: int, tier: str,
                       policy: PromotionPolicy, lv: int) -> pd.Series:
    st = training_stack(tables, cutoff, horizon, "contractual", tier, policy, landmarks=[lv])
    return st.set_index("customer_id")["realised"]


def _tune(tables: EconomicsTables, cutoff: int, horizon: int, tier: str,
          policy: PromotionPolicy, exclude: Sequence[str] = ()
          ) -> tuple[float, str, tuple[TuningRecord, ...]]:
    if not pp.temporal_tuning_feasible(cutoff, horizon):
        return pp.DEFAULT_PENALTY, FIXED_PENALTY_LABEL, ()
    lv = pp.validation_landmark(cutoff, horizon)
    assert lv is not None
    records = []
    for lam in pp.PENALTY_GRID:
        frame = _fit_at_validation(tables, cutoff, horizon, tier, policy, lam, exclude)
        records.append(TuningRecord(lam, frame[1], frame[2]))
    best = choose_penalty(records)
    if best is None:
        return pp.DEFAULT_PENALTY, "fixed penalty: no tuning fit was ok", tuple(records)
    return best, TUNED_LABEL, tuple(records)


def choose_penalty(records: Sequence[TuningRecord]) -> float | None:
    """``E3_TUNING_LOSS``: the ``ok`` record with the smallest validation MSE, ties to the larger
    penalty; None if no record is usable."""
    usable = [r for r in records if r.status == "ok" and math.isfinite(r.validation_mse)]
    if not usable:
        return None
    return min(usable, key=lambda r: (r.validation_mse, -r.penalty)).penalty


def _fit_at_validation(tables: EconomicsTables, cutoff: int, horizon: int, tier: str,
                       policy: PromotionPolicy, penalty: float, exclude: Sequence[str] = ()
                       ) -> tuple[pd.DataFrame, float, str]:
    """Fit on the tuning landmarks with the tables as recorded by ``L_v``; predict ``L_v``."""
    lv = pp.validation_landmark(cutoff, horizon)
    assert lv is not None
    landmarks = pp.tuning_landmarks(cutoff, horizon)
    stack, n = _landmark_data(tables, lv, horizon, tier, policy, landmarks)
    fit = fit_hazard(stack, n, horizon, tier, penalty, exclude)
    view = view_at(tables, lv)
    feats = contractual_features(view, lv, tier, policy)
    contrib = fit_contribution(view, lv, policy)
    frame = _value_frame(feats, _paid(fit, feats, horizon), contrib, fit.status,
                         pp.MONTHLY_DISCOUNT_RATE)
    realised = _validation_labels(tables, cutoff, horizon, tier, policy, lv)
    if not realised.index.equals(frame.index):
        raise RuntimeError("validation population differs from the prediction population")
    frame = frame.assign(realised=realised)
    mse = float(np.mean((frame["realised"] - frame["prediction"]) ** 2))
    return frame, mse, fit.status


def fit_e3(tables: EconomicsTables, cutoff: int, horizon: int, tier: str,
           policy: PromotionPolicy, *, penalty: float | None = None,
           exclude_features: Sequence[str] = pp.E3_EXCLUDED_COVARIATES) -> E3Model:
    """E3 at ``cutoff`` (Tier ``A`` or ``B``): temporal penalty tuning (unless ``penalty`` is
    given), then the final fit on every training landmark (tables as recorded by ``cutoff``).

    ``exclude_features`` defaults to the protocol's ``E3_EXCLUDED_COVARIATES`` (A8);
    ``()`` reproduces the pre-A8 model for the before/after record only."""
    if tier not in ("A", "B"):
        raise ValueError("tier must be 'A' or 'B'")
    if penalty is None:
        lam, source, records = _tune(tables, cutoff, horizon, tier, policy, exclude_features)
    else:
        lam, source, records = float(penalty), "given", ()
    stack, n = _landmark_data(tables, cutoff, horizon, tier, policy, None)
    fit = fit_hazard(stack, n, horizon, tier, lam, exclude_features)
    return E3Model(tier, cutoff, horizon, lam, source, records, fit,
                   fit_contribution(tables, cutoff, policy),
                   tuple(int(v) for v in np.unique(stack["landmark"])))


@dataclass(frozen=True)
class SurvivalPrediction:
    model: str
    cutoff: int
    frame: pd.DataFrame
    """Per customer: ``segment, channel, age, status, prediction, prediction_discounted,
    paid_1 .. paid_H`` (the contractual population at ``cutoff``)."""


def predict_e3(model: E3Model, tables: EconomicsTables, *,
               monthly_rate: float = pp.MONTHLY_DISCOUNT_RATE) -> SurvivalPrediction:
    """Predict months ``C+1 .. C+W`` from features frozen at ``C = model.cutoff``."""
    feats = contractual_features(tables, model.cutoff, model.tier, model.contribution.policy)
    paid = _paid(model.hazard, feats, model.horizon)
    return SurvivalPrediction("E3", model.cutoff,
                              _value_frame(feats, paid, model.contribution, model.hazard.status,
                                           monthly_rate))


@dataclass(frozen=True)
class E3Validation:
    tier: str
    cutoff: int
    landmark: int | None
    penalty: float
    candidate: CandidateResult | None
    """None when no landmark has labels complete before ``L_v`` (no validation fit exists)."""
    frame: pd.DataFrame | None
    selection_eligible: bool
    """``E3_SELECTION_ELIGIBILITY`` (A8): False when no validation fit exists."""
    eligibility: str


def validate_e3(tables: EconomicsTables, cutoff: int, horizon: int, tier: str,
                policy: PromotionPolicy, *, penalty: float | None = None,
                exclude_features: Sequence[str] = pp.E3_EXCLUDED_COVARIATES
                ) -> E3Validation:
    """Selection inputs at the validation landmark: the fit on the tuning landmarks (tables as
    recorded by ``L_v``) with the chosen penalty, scored on ``L_v``'s labels."""
    lv = pp.validation_landmark(cutoff, horizon)
    lam = (_tune(tables, cutoff, horizon, tier, policy, exclude_features)[0]
           if penalty is None else float(penalty))
    if lv is None or not pp.tuning_landmarks(cutoff, horizon):
        return E3Validation(tier, cutoff, lv, lam, None, None, False, NOT_ELIGIBLE)
    frame, mse, status = _fit_at_validation(tables, cutoff, horizon, tier, policy, lam,
                                            exclude_features)
    p, r = frame["prediction"].to_numpy(np.float64), frame["realised"].to_numpy(np.float64)
    cand = CandidateResult("E3", 1.0 if status == "ok" else 0.0,
                           float(np.sum(p)) / float(np.sum(r)) - 1.0, calibration_slope(p, r),
                           mse)
    return E3Validation(tier, cutoff, lv, lam, cand, frame, True, "eligible")


# --- S1: Cox proportional hazards (Efron ties, Breslow baseline) ----------------------------------
@dataclass(frozen=True)
class CoxResult:
    coefficients: FloatArray
    covariance: FloatArray
    log_likelihood: float
    converged: bool
    n_iterations: int
    max_gradient_per_row: float
    hessian_positive_definite: bool
    times: FloatArray
    """Distinct event times."""
    baseline_cumulative_hazard: FloatArray
    """Breslow ``H0`` at ``times``."""

    def cumulative_hazard_at(self, t: FloatArray) -> FloatArray:
        idx = np.searchsorted(self.times, np.asarray(t, dtype=np.float64), side="right") - 1
        h = np.where(idx >= 0, self.baseline_cumulative_hazard[np.maximum(idx, 0)], 0.0)
        return np.asarray(h, dtype=np.float64)


def _efron(beta: FloatArray, time: np.ndarray, event: np.ndarray, X: FloatArray
           ) -> tuple[float, FloatArray, FloatArray]:
    """Efron partial log-likelihood, score and (negative-definite) Hessian."""
    eta = X @ beta
    shift = float(np.max(eta, initial=0.0))
    r = np.exp(eta - shift)
    times = np.unique(time[event])
    p = X.shape[1]
    ll, grad, hess = 0.0, np.zeros(p), np.zeros((p, p))
    order = np.argsort(-time, kind="stable")  # descending: risk sets are prefixes
    t_sorted = time[order]
    Xs, rs, es, etas = X[order], r[order], event[order], eta[order]
    cum0 = np.cumsum(rs)
    cum1 = np.cumsum(rs[:, None] * Xs, axis=0)
    xr = Xs * rs[:, None]
    for t in times:
        end = int(np.searchsorted(-t_sorted, -t, side="right"))  # rows with time >= t
        s0, s1 = cum0[end - 1], cum1[end - 1]
        s2 = xr[:end].T @ Xs[:end]
        dmask = (t_sorted[:end] == t) & es[:end]
        d = int(dmask.sum())
        xd, rd = Xs[:end][dmask], rs[:end][dmask]
        d0, d1 = float(rd.sum()), (rd[:, None] * xd).sum(axis=0)
        d2 = (xd * rd[:, None]).T @ xd
        phi = np.arange(d) / d
        den = s0 - phi * d0
        ll += float(etas[:end][dmask].sum()) - float(np.sum(np.log(den)))
        a, b = float(np.sum(1 / den)), float(np.sum(phi / den))
        c0, c1, c2 = (float(np.sum(1 / den ** 2)), float(np.sum(phi / den ** 2)),
                      float(np.sum(phi ** 2 / den ** 2)))
        grad += xd.sum(axis=0) - (s1 * a - d1 * b)
        hess -= (s2 * a - d2 * b) - (np.outer(s1, s1) * c0
                                     - (np.outer(s1, d1) + np.outer(d1, s1)) * c1
                                     + np.outer(d1, d1) * c2)
    # the shift multiplies every risk-set sum by exp(-shift): restore it in the log terms
    ll -= float(event.sum()) * shift
    return ll, grad, hess


def fit_cox_efron(time: np.ndarray, event: np.ndarray, X: FloatArray, *,
                  max_iterations: int = 100,
                  gradient_tolerance: float = pp.STATUS_RULES.gradient_tolerance,
                  relative_change_tolerance: float = pp.STATUS_RULES.relative_change_tolerance
                  ) -> CoxResult:
    """Cox proportional hazards by Newton on Efron's partial likelihood; Breslow baseline."""
    t = np.asarray(time, dtype=np.float64)
    e = np.asarray(event, dtype=bool)
    Xa = np.asarray(X, dtype=np.float64)
    n, p = Xa.shape
    beta = np.zeros(p)
    ll, g, h = _efron(beta, t, e, Xa)
    rel, n_iter = math.inf, 0
    for it in range(1, max_iterations + 1):
        n_iter = it
        if float(np.max(np.abs(g), initial=0.0)) / n <= gradient_tolerance and \
                rel <= relative_change_tolerance:
            break
        try:
            step = np.linalg.solve(-h, g)
        except np.linalg.LinAlgError:
            break
        moved = False
        for _ in range(60):
            cand = beta + step
            llc, gc, hc = _efron(cand, t, e, Xa)
            if math.isfinite(llc) and llc >= ll - 1e-12 * max(1.0, abs(ll)):
                rel = abs(llc - ll) / max(1.0, abs(ll))
                beta, ll, g, h = cand, llc, gc, hc
                moved = True
                break
            step = step / 2.0
        if not moved:
            break
    max_g = float(np.max(np.abs(g), initial=0.0)) / n
    try:
        eig = np.linalg.eigvalsh(-h)
        pd_ok = bool(float(np.min(eig)) > 1e-12 * max(abs(float(np.max(eig))), 1e-300))
    except np.linalg.LinAlgError:
        pd_ok = False
    cov = np.linalg.inv(-h) if pd_ok else np.full((p, p), np.nan)
    times = np.unique(t[e])
    r = np.exp(Xa @ beta)
    dh = np.array([float(np.sum(e & (t == u))) / float(np.sum(r[t >= u])) for u in times])
    return CoxResult(np.asarray(beta, dtype=np.float64), np.asarray(cov, dtype=np.float64),
                     float(ll), max_g <= gradient_tolerance and rel <= relative_change_tolerance,
                     n_iter, max_g, pd_ok, np.asarray(times, dtype=np.float64),
                     np.asarray(np.cumsum(dh), dtype=np.float64))


@dataclass(frozen=True)
class S1Model:
    tier: str
    cutoff: int
    horizon: int
    design: Design
    cox: CoxResult
    contribution: ContributionModel
    n_rows: int
    n_events: int
    status: str
    reasons: tuple[str, ...]


def fit_s1(tables: EconomicsTables, cutoff: int, horizon: int, tier: str,
           policy: PromotionPolicy, *, exclude_features: Sequence[str] = ()) -> S1Model:
    """S1 on every training landmark (tables as recorded by ``cutoff``), with the full registry
    (A8 applies to E3 only); ``exclude_features`` is a diagnostic hook, never used for S1's
    reported results."""
    stack, n = _landmark_data(tables, cutoff, horizon, tier, policy, None)
    design = build_design(stack, tier, exclude=exclude_features)
    time = np.minimum(n + 1, horizon).astype(np.float64)
    event = n < horizon
    cox = fit_cox_efron(time, event, design.matrix(stack))
    d = FitDiagnostics(has_data=bool(event.any()), converged=cox.converged,
                       finite_likelihood=math.isfinite(cox.log_likelihood),
                       hessian_positive_definite=cox.hessian_positive_definite,
                       n_events=int(event.sum()), n_parameters=len(design.columns))
    status, reasons = classify_fit(d)
    return S1Model(tier, cutoff, horizon, design, cox, fit_contribution(tables, cutoff, policy),
                   len(stack), int(event.sum()), status, reasons)


def predict_s1(model: S1Model, tables: EconomicsTables, *,
               monthly_rate: float = pp.MONTHLY_DISCOUNT_RATE) -> SurvivalPrediction:
    feats = contractual_features(tables, model.cutoff, model.tier, model.contribution.policy)
    lp = model.design.matrix(feats) @ model.cox.coefficients
    h0 = model.cox.cumulative_hazard_at(np.arange(1, model.horizon + 1, dtype=np.float64))
    paid = np.exp(-np.outer(np.exp(lp), h0))
    return SurvivalPrediction("S1", model.cutoff,
                              _value_frame(feats, paid, model.contribution, model.status,
                                           monthly_rate))
