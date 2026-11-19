"""Predictive (direct-value) LTV model for Milestone 11 (P6): E5 and S2.

**E5 — two-part (hurdle) penalised GLM.** Trained on P2's landmark stacks
(:func:`~.landmarks.training_stack`: features frozen at each landmark ``L``, the window value
``R`` over ``L+1 .. L+W``, completely recorded by ``C``), for the contractual (Tier A or B) and
non-contractual (Tier A) scenarios, with every registry feature of the tier (A8's covariate
exclusion is E3's only; E5's design is as specified):

* part 1: ``P(R > 0 | x)`` by L2-penalised logistic regression
  (:func:`~decision_intelligence.statistics.penalised_regression.fit_penalised_logistic`);
* part 2: ``E[R | R > 0, x]`` by an L2-penalised Gamma GLM with a log link, the shape estimated
  jointly (:func:`~decision_intelligence.statistics.gamma_regression.fit_penalised_gamma`);
* prediction ``P(R > 0 | x) x E[R | R > 0, x]``.

A window with ``R <= 0`` counts as "not positive": contractual windows can be slightly negative
(a month whose cost to serve exceeds its revenue); their small negative mass is omitted from the
mean and reported (:attr:`E5Model.negative_mass`). Both parts follow ``PENALTY_CONVENTION``: an
unpenalised intercept; every other column, one-hot indicators included (reference = most
frequent training level), standardised on the rows of *that part's* fit; SD-0 columns dropped and
``DESIGN_DUPLICATE_RULE`` applied, all reported. Fit status follows ``STATUS_RULES`` (events: the
minority class for part 1, positive windows for part 2).

**Penalty** (``TUNING_RULE`` with ``E5_TUNING_LOSS``, amendment A9): one ``lambda`` shared by
both parts, chosen on the validation landmark ``L_v`` from the tuning landmarks (tables as
recorded by ``L_v``) by squared error of window value, ties to the larger penalty; if no tuning
fit has both parts ``ok``, ``DEFAULT_PENALTY`` labelled :data:`NO_OK_FIT_LABEL`; with fewer than
``MIN_TUNING_LANDMARKS`` tuning landmarks, ``DEFAULT_PENALTY`` labelled
:data:`FIXED_PENALTY_LABEL`. The final model is refitted on every training landmark. Without a
tuning landmark there is no validation fit and E5 is *not selection-eligible*
(``E5_SELECTION_ELIGIBILITY``, A9).

**S2 — split conformal intervals for E5** (``CLUSTERED_INFERENCE``). Customers in the training
stack are split deterministically by a hash of ``customer_id`` (about
``conformal_calibration_fraction`` to calibration; no random seed). E5 (same ``lambda``) is fitted
on the other customers' rows; each calibration customer contributes **one landmark, their latest**
(the calibration set closest in time to ``C``); scores are absolute residuals of window value;
intervals are ``m(x) -/+ q`` at ``CALIBRATION.interval_levels``
(:mod:`decision_intelligence.statistics.conformal`), centred on that half-sample fit. The
finite-sample guarantee needs exchangeability, which landmarks before ``C`` and the window after
``C`` do not have under drift; coverage at ``C`` is reported, never asserted
(``COVERAGE_REPORTING_RULE``).

Estimation code only: observable tables; never truth, oracles or generator parameters (AST
tested).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from decision_intelligence.common.validation import FloatArray
from decision_intelligence.customer_value import predictive_protocol as pp
from decision_intelligence.customer_value.contractual import calibration_slope
from decision_intelligence.customer_value.features import (
    PromotionPolicy,
    contractual_features,
    feature_names,
    noncontractual_features,
)
from decision_intelligence.customer_value.landmarks import training_stack, view_at
from decision_intelligence.customer_value.model_selection import (
    CandidateResult,
    FitDiagnostics,
    classify_fit,
)
from decision_intelligence.customer_value.panel import EconomicsTables
from decision_intelligence.customer_value.survival import TuningRecord, choose_penalty
from decision_intelligence.statistics.conformal import conformal_quantile
from decision_intelligence.statistics.gamma_regression import (
    GammaRegressionResult,
    fit_penalised_gamma,
)
from decision_intelligence.statistics.penalised_regression import (
    PenalisedLogisticResult,
    Standardiser,
    expit,
    fit_penalised_logistic,
)

CATEGORICAL = ("segment", "channel")
FIXED_PENALTY_LABEL = "fixed penalty: temporal tuning infeasible"
TUNED_LABEL = "temporal tuning"
NO_OK_FIT_LABEL = "fixed penalty: no tuning fit was ok"
NOT_ELIGIBLE = ("not selection-eligible: no landmark has labels completely recorded before the "
                "validation landmark")


# --- design (PENALTY_CONVENTION, DESIGN_DUPLICATE_RULE) ------------------------------------------
@dataclass(frozen=True)
class Design:
    kind: str
    tier: str
    raw_columns: tuple[str, ...]
    levels: dict[str, tuple[tuple[str, ...], str]]
    standardiser: Standardiser
    duplicates: tuple[str, ...]
    columns: tuple[str, ...]

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
        """``[1, standardised columns]``."""
        z = self.standardiser.transform(self.raw(frame), self.raw_columns)
        keep = [self.standardiser.names.index(c) for c in self.columns]
        return np.column_stack([np.ones(len(frame)), z[:, keep]])


def build_design(frame: pd.DataFrame, kind: str, tier: str, exclude: Sequence[str] = ()
                 ) -> Design:
    """``exclude`` is a diagnostic hook only (default: the tier's full registry, as specified
    for E5); using it for a reported E5 result would need a protocol amendment."""
    names = [n for n in feature_names(kind, tier) if n not in exclude]
    numeric = [n for n in names if n not in CATEGORICAL]
    levels: dict[str, tuple[tuple[str, ...], str]] = {}
    raw_cols: list[str] = list(numeric)
    for var in CATEGORICAL:
        if var not in names:
            continue
        counts = frame[var].value_counts()
        ref = str(sorted(counts.index, key=lambda v: (-counts[v], str(v)))[0])
        kept = tuple(sorted(str(v) for v in counts.index if v != ref))
        levels[var] = (kept, ref)
        raw_cols += [f"{var}={lv}" for lv in kept]
    probe = Design(kind, tier, tuple(raw_cols), levels,
                   Standardiser((), np.zeros(0), np.zeros(0), ()), (), ())
    raw = probe.raw(frame)
    std = Standardiser.fit(raw, raw_cols)
    z = std.transform(raw, raw_cols)
    cols: list[str] = []
    dups: list[str] = []
    kept_z: list[np.ndarray] = []
    for j, name in enumerate(std.names):
        col = z[:, j]
        if any(float(np.max(np.abs(col - other))) <= 1e-10 for other in kept_z):
            dups.append(name)
            continue
        cols.append(name)
        kept_z.append(col)
    return Design(kind, tier, tuple(raw_cols), levels, std, tuple(dups), tuple(cols))


# --- the two parts --------------------------------------------------------------------------------
@dataclass(frozen=True)
class PartFit:
    part: str
    """``positive`` (logistic) or ``amount`` (Gamma)."""
    design: Design
    result: PenalisedLogisticResult | GammaRegressionResult
    n_rows: int
    n_events: int
    status: str
    reasons: tuple[str, ...]


def _mask(k: int) -> np.ndarray:
    return np.asarray(np.r_[False, np.ones(k - 1, dtype=bool)], dtype=bool)


def fit_parts(stack: pd.DataFrame, kind: str, tier: str, penalty: float,
              exclude: Sequence[str] = ()) -> tuple[PartFit, PartFit]:
    """E5's two parts on landmark rows with window values ``stack['realised']``."""
    rules = pp.STATUS_RULES
    r = stack["realised"].to_numpy(np.float64)
    pos = r > 0
    d1 = build_design(stack, kind, tier, exclude)
    X1 = d1.matrix(stack)
    res1 = fit_penalised_logistic(X1, pos.astype(np.float64), penalty=penalty,
                                  penalised=_mask(X1.shape[1]),
                                  gradient_tolerance=rules.gradient_tolerance,
                                  relative_change_tolerance=rules.relative_change_tolerance)
    ev1 = int(min(pos.sum(), (~pos).sum()))
    diag1 = FitDiagnostics(has_data=len(r) > 0 and ev1 > 0, converged=res1.converged,
                           finite_likelihood=math.isfinite(res1.log_likelihood),
                           separated_without_penalty=res1.separated,
                           hessian_positive_definite=res1.hessian_positive_definite,
                           n_events=ev1, n_parameters=X1.shape[1])
    s1, why1 = classify_fit(diag1)
    rows2 = stack.loc[pos]
    d2 = build_design(rows2, kind, tier, exclude)
    X2 = d2.matrix(rows2)
    res2 = fit_penalised_gamma(X2, r[pos], penalty=penalty, penalised=_mask(X2.shape[1]),
                               gradient_tolerance=rules.gradient_tolerance,
                               relative_change_tolerance=rules.relative_change_tolerance)
    diag2 = FitDiagnostics(has_data=bool(pos.any()), converged=res2.converged,
                           finite_likelihood=math.isfinite(res2.log_likelihood),
                           hessian_positive_definite=res2.hessian_positive_definite,
                           n_events=int(pos.sum()), n_parameters=X2.shape[1])
    s2, why2 = classify_fit(diag2)
    return (PartFit("positive", d1, res1, len(r), ev1, s1, why1),
            PartFit("amount", d2, res2, int(pos.sum()), int(pos.sum()), s2, why2))


def _predict_parts(part1: PartFit, part2: PartFit, feats: pd.DataFrame
                   ) -> tuple[FloatArray, FloatArray]:
    p = expit(part1.design.matrix(feats) @ part1.result.coefficients)
    m = np.exp(part2.design.matrix(feats) @ part2.result.coefficients)
    return np.asarray(p, dtype=np.float64), np.asarray(m, dtype=np.float64)


def _features(tables: EconomicsTables, cutoff: int, kind: str, tier: str,
              policy: PromotionPolicy | None) -> pd.DataFrame:
    if kind == "contractual":
        if policy is None:
            raise ValueError("contractual features need the promotion policy")
        return contractual_features(tables, cutoff, tier, policy)
    if kind == "non_contractual":
        if tier != "A":
            raise ValueError("non-contractual scenarios have Tier A only")
        return noncontractual_features(tables, cutoff)
    raise ValueError("kind must be 'contractual' or 'non_contractual'")


# --- E5 -------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class E5Model:
    kind: str
    tier: str
    cutoff: int
    horizon: int
    penalty: float
    penalty_source: str
    tuning: tuple[TuningRecord, ...]
    positive: PartFit
    amount: PartFit
    landmarks: tuple[int, ...]
    policy: PromotionPolicy | None
    negative_windows: int
    negative_mass: float
    """Mean of ``min(R, 0)`` over training rows: the part omitted from the two-part mean."""

    @property
    def status(self) -> str:
        """The worse of the two parts' statuses (``ok`` only if both are)."""
        order = ("undefined", "failed", "unstable", "ok")
        return min((self.positive.status, self.amount.status), key=order.index)


def _fit_at_validation(tables: EconomicsTables, cutoff: int, horizon: int, kind: str,
                       tier: str, policy: PromotionPolicy | None, penalty: float
                       ) -> tuple[pd.DataFrame, float, str]:
    lv = pp.validation_landmark(cutoff, horizon)
    assert lv is not None
    stack = training_stack(tables, lv, horizon, kind, tier, policy,
                           landmarks=pp.tuning_landmarks(cutoff, horizon))
    p1, p2 = fit_parts(stack, kind, tier, penalty)
    feats = _features(view_at(tables, lv), lv, kind, tier, policy)
    p, m = _predict_parts(p1, p2, feats)
    labels = training_stack(tables, cutoff, horizon, kind, tier, policy,
                            landmarks=[lv]).set_index("customer_id")["realised"]
    if not labels.index.equals(feats.index):
        raise RuntimeError("validation population differs from the prediction population")
    frame = pd.DataFrame({"prediction": p * m, "realised": labels}, index=feats.index)
    status = "ok" if p1.status == "ok" and p2.status == "ok" else "not ok"
    return frame, float(np.mean((frame["realised"] - frame["prediction"]) ** 2)), status


def _tune(tables: EconomicsTables, cutoff: int, horizon: int, kind: str, tier: str,
          policy: PromotionPolicy | None) -> tuple[float, str, tuple[TuningRecord, ...]]:
    if not pp.temporal_tuning_feasible(cutoff, horizon):
        return pp.DEFAULT_PENALTY, FIXED_PENALTY_LABEL, ()
    records = []
    for lam in pp.PENALTY_GRID:
        _, mse, status = _fit_at_validation(tables, cutoff, horizon, kind, tier, policy, lam)
        records.append(TuningRecord(lam, mse, status))
    best = choose_penalty(records)
    if best is None:
        return pp.DEFAULT_PENALTY, NO_OK_FIT_LABEL, tuple(records)
    return best, TUNED_LABEL, tuple(records)


def fit_e5(tables: EconomicsTables, cutoff: int, horizon: int, kind: str, tier: str,
           policy: PromotionPolicy | None = None, *, penalty: float | None = None) -> E5Model:
    """E5 at ``cutoff``: temporal penalty tuning (unless ``penalty`` is given), then both parts
    fitted on every training landmark (tables as recorded by ``cutoff``)."""
    if penalty is None:
        lam, source, records = _tune(tables, cutoff, horizon, kind, tier, policy)
    else:
        lam, source, records = float(penalty), "given", ()
    stack = training_stack(tables, cutoff, horizon, kind, tier, policy)
    p1, p2 = fit_parts(stack, kind, tier, lam)
    r = stack["realised"].to_numpy(np.float64)
    return E5Model(kind, tier, cutoff, horizon, lam, source, records, p1, p2,
                   tuple(int(v) for v in np.unique(stack["landmark"])), policy,
                   int((r < 0).sum()), float(np.mean(np.minimum(r, 0.0))))


@dataclass(frozen=True)
class E5Prediction:
    cutoff: int
    frame: pd.DataFrame
    """Per customer: ``prediction`` (= ``p_positive x mean_positive``), ``p_positive``,
    ``mean_positive``, ``status``."""


def predict_e5(model: E5Model, tables: EconomicsTables) -> E5Prediction:
    feats = _features(tables, model.cutoff, model.kind, model.tier, model.policy)
    p, m = _predict_parts(model.positive, model.amount, feats)
    frame = pd.DataFrame({"prediction": p * m, "p_positive": p, "mean_positive": m,
                          "status": model.status}, index=feats.index)
    return E5Prediction(model.cutoff, frame)


@dataclass(frozen=True)
class E5Validation:
    kind: str
    tier: str
    cutoff: int
    landmark: int | None
    penalty: float
    candidate: CandidateResult | None
    frame: pd.DataFrame | None
    selection_eligible: bool
    eligibility: str


def validate_e5(tables: EconomicsTables, cutoff: int, horizon: int, kind: str, tier: str,
                policy: PromotionPolicy | None = None, *, penalty: float | None = None
                ) -> E5Validation:
    """Selection inputs at ``L_v``: the fit on the tuning landmarks (tables as recorded by
    ``L_v``) with the chosen penalty, scored on ``L_v``'s labels."""
    lv = pp.validation_landmark(cutoff, horizon)
    lam = (_tune(tables, cutoff, horizon, kind, tier, policy)[0] if penalty is None
           else float(penalty))
    if lv is None or not pp.tuning_landmarks(cutoff, horizon):
        return E5Validation(kind, tier, cutoff, lv, lam, None, None, False, NOT_ELIGIBLE)
    frame, mse, status = _fit_at_validation(tables, cutoff, horizon, kind, tier, policy, lam)
    p, r = frame["prediction"].to_numpy(np.float64), frame["realised"].to_numpy(np.float64)
    cand = CandidateResult("E5", 1.0 if status == "ok" else 0.0,
                           float(np.sum(p)) / float(np.sum(r)) - 1.0, calibration_slope(p, r),
                           mse)
    return E5Validation(kind, tier, cutoff, lv, lam, cand, frame, True, "eligible")


# --- S2: split conformal intervals ----------------------------------------------------------------
def calibration_customers(customer_id: Sequence[int] | np.ndarray,
                          fraction: float = pp.CLUSTERED_INFERENCE.conformal_calibration_fraction
                          ) -> np.ndarray:
    """Deterministic customer split (no random seed): calibration iff the first 8 bytes of
    ``blake2b(str(customer_id))``, as a fraction of ``2**64``, fall below ``fraction``."""
    out = np.empty(len(customer_id), dtype=bool)
    for i, cid in enumerate(customer_id):
        h = hashlib.blake2b(str(int(cid)).encode("ascii"), digest_size=8).digest()
        out[i] = int.from_bytes(h, "big") / 2.0 ** 64 < fraction
    return out


@dataclass(frozen=True)
class S2Intervals:
    cutoff: int
    penalty: float
    n_training_customers: int
    n_calibration: int
    quantiles: dict[float, float]
    """Level -> conformal half-width ``q``."""
    frame: pd.DataFrame
    """Per customer at ``cutoff``: ``centre`` (the half-sample E5 prediction) and
    ``lower_<level>``, ``upper_<level>``."""
    half_model: E5Model


def fit_s2(tables: EconomicsTables, cutoff: int, horizon: int, kind: str, tier: str,
           policy: PromotionPolicy | None = None, *, penalty: float | None = None
           ) -> S2Intervals:
    """S2 at ``cutoff`` (penalty: E5's, tuned unless given)."""
    lam = (_tune(tables, cutoff, horizon, kind, tier, policy)[0] if penalty is None
           else float(penalty))
    stack = training_stack(tables, cutoff, horizon, kind, tier, policy)
    ids = stack["customer_id"].to_numpy()
    cal = calibration_customers(ids)
    train = stack.loc[~cal]
    p1, p2 = fit_parts(train, kind, tier, lam)
    r_train = train["realised"].to_numpy(np.float64)
    half = E5Model(kind, tier, cutoff, horizon, lam, "S2 training half", (), p1, p2,
                   tuple(int(v) for v in np.unique(train["landmark"])), policy,
                   int((r_train < 0).sum()), float(np.mean(np.minimum(r_train, 0.0))))
    calib = stack.loc[cal].sort_values(["customer_id", "landmark"]).groupby(
        "customer_id").tail(1)  # one landmark per calibration customer: the latest
    p, m = _predict_parts(p1, p2, calib)
    scores = np.abs(calib["realised"].to_numpy(np.float64) - p * m)
    feats = _features(tables, cutoff, kind, tier, policy)
    pc, mc = _predict_parts(p1, p2, feats)
    centre = pc * mc
    frame = pd.DataFrame({"centre": centre}, index=feats.index)
    quantiles: dict[float, float] = {}
    for level in pp.CALIBRATION.interval_levels:
        q = conformal_quantile(scores, level)
        quantiles[level] = q
        frame[f"lower_{level}"] = centre - q
        frame[f"upper_{level}"] = centre + q
    return S2Intervals(cutoff, lam, int(np.unique(train["customer_id"]).size), len(calib),
                       quantiles, frame, half)
