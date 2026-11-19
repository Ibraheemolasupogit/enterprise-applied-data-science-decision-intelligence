"""Milestone 11 baselines (P7, amendment A10): B1 (contractual) and B2 (non-contractual).

**B1** (``B1_DEFINITION``): M10's customer method *Kaplan-Meier by cell*, computed by M10's own
:func:`~.backtest.customer_projections` (unchanged) with ``W = PRIMARY_HORIZON``, restricted to
the M11 contractual population (active at ``C``, acquired by ``complete_through(C)``). Cell
status uses M10's thresholds: ``undefined`` without a usable Kaplan-Meier curve (fewer than two
ages with ``MIN_AT_RISK`` at risk), ``unstable`` with fewer than ``MIN_CHURN_EVENTS`` churn events,
otherwise ``ok``. **No fallback**: a customer without a finite prediction is ``undefined`` and
prevents B1 from qualifying at that cutoff.

**B2** (``B2_DEFINITION``): ``f x s x k`` for every customer acquired by ``T_C`` (E4's
population): ``f`` = purchases (acquisition included) in the complete months
``max(acquisition, T_C - H + 1) .. T_C``, carried forward unchanged (no churn, no annualisation);
``s`` = mean revenue per completely recorded purchase (acquisition included); ``k`` = E4's
recorded contribution-to-revenue ratio. Status ``ok`` where finite, otherwise ``undefined``.

**Selection inputs** for both: computed with the information of the validation landmark ``L_v``
only (tables as recorded by ``C``, observed at ``L_v``) and scored on ``L_v``'s completely
recorded labels, exactly as E1/E2 and E4; eligible wherever a validation landmark exists.

Estimation code only: observable tables; never truth, oracles or generator parameters (AST
tested).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from decision_intelligence.customer_value import predictive_protocol as pp
from decision_intelligence.customer_value.backtest import customer_projections
from decision_intelligence.customer_value.btyd import purchase_data
from decision_intelligence.customer_value.contractual import calibration_slope
from decision_intelligence.customer_value.features import PromotionPolicy
from decision_intelligence.customer_value.heuristic import empirical_curve
from decision_intelligence.customer_value.landmarks import training_stack, view_at
from decision_intelligence.customer_value.model_selection import CandidateResult
from decision_intelligence.customer_value.panel import EconomicsTables, observe
from decision_intelligence.customer_value.protocol import MIN_CHURN_EVENTS

B1_METHOD = "Kaplan–Meier by cell"
NOT_ELIGIBLE = "not selection-eligible: no validation landmark"


@dataclass(frozen=True)
class BaselinePrediction:
    model: str
    cutoff: int
    frame: pd.DataFrame
    """Per customer (index ``customer_id``): ``segment, channel, status, prediction``."""
    cell_status: dict[tuple[str, str], str]

    @property
    def ok_cell_fraction(self) -> float:
        if not self.cell_status:
            return math.nan
        return sum(s == "ok" for s in self.cell_status.values()) / len(self.cell_status)

    @property
    def undefined_customers(self) -> int:
        return int((~np.isfinite(self.frame["prediction"].to_numpy(np.float64))).sum())


def predict_b1(tables: EconomicsTables, cutoff: int, horizon: int = pp.PRIMARY_HORIZON
               ) -> BaselinePrediction:
    """B1 at ``cutoff`` from the information recorded by then."""
    obs = observe(tables, cutoff)
    st = obs.status()
    pop = st.loc[~st["churned"] & (st["acquisition_month"] <= obs.complete_through())]
    proj = customer_projections(obs, horizon)
    pred = proj[B1_METHOD].reindex(pop.index)
    cells: dict[tuple[str, str], str] = {}
    customer_status = pd.Series("undefined", index=pop.index, dtype=object)
    for (seg, ch), g in pop.groupby(["segment", "channel"]):
        key = (str(seg), str(ch))
        mask = ((st["segment"] == seg) & (st["channel"] == ch)).to_numpy()
        curve = empirical_curve(obs, mask)
        if curve is None:
            cells[key] = "undefined"
        elif curve.events < MIN_CHURN_EVENTS:
            cells[key] = "unstable"
        else:
            cells[key] = "ok"
        customer_status.loc[g.index] = cells[key]
    finite = np.isfinite(pred.to_numpy(np.float64))
    customer_status[~finite] = "undefined"
    frame = pd.DataFrame({"segment": pop["segment"], "channel": pop["channel"],
                          "status": customer_status, "prediction": pred}, index=pop.index)
    return BaselinePrediction("B1", cutoff, frame, cells)


def predict_b2(tables: EconomicsTables, cutoff: int, horizon: int = pp.PRIMARY_HORIZON
               ) -> BaselinePrediction:
    """B2 at ``cutoff`` from the information recorded by then."""
    data = purchase_data(tables, cutoff, horizon)
    frame = data.frame
    obs = observe(tables, cutoff)
    tx = obs.transactions.drop_duplicates("transaction_id")
    lo = data.through - pp.PRIMARY_HORIZON + 1
    recent = tx.loc[(tx["billing_month"] >= lo) & (tx["billing_month"] <= data.through)
                    & tx["customer_id"].isin(frame.index)]
    f = recent.groupby("customer_id").size().reindex(frame.index, fill_value=0)
    s = frame["S"] / frame["n"]
    pred = f.to_numpy(np.float64) * s.to_numpy(np.float64) * data.contribution_ratio
    finite = np.isfinite(pred)
    out = pd.DataFrame({"segment": frame["segment"], "channel": frame["channel"],
                        "status": np.where(finite, "ok", "undefined"), "prediction": pred,
                        "purchases_carried": f.to_numpy(np.float64),
                        "mean_revenue": s.to_numpy(np.float64)}, index=frame.index)
    cells: dict[tuple[str, str], str] = {}
    for key, rows in out.groupby(["segment", "channel"]).indices.items():
        seg, ch = cast(tuple[object, object], key)
        cells[(str(seg), str(ch))] = "ok" if bool(np.all(finite[rows])) else "undefined"
    return BaselinePrediction("B2", cutoff, out, cells)


@dataclass(frozen=True)
class BaselineValidation:
    model: str
    cutoff: int
    landmark: int | None
    candidate: CandidateResult | None
    frame: pd.DataFrame | None
    """``prediction``, ``realised`` per customer at ``L_v``."""
    undefined_customers: int
    selection_eligible: bool
    eligibility: str


def validate_baseline(model: str, tables: EconomicsTables, cutoff: int,
                      horizon: int = pp.PRIMARY_HORIZON, policy: PromotionPolicy | None = None
                      ) -> BaselineValidation:
    """Selection inputs at ``L_v`` (information of ``L_v`` only; labels completely recorded
    by ``cutoff``). ``policy`` is needed for B1's label stack only (it is not a B1 input)."""
    if model not in ("B1", "B2"):
        raise ValueError("model must be 'B1' or 'B2'")
    lv = pp.validation_landmark(cutoff, horizon)
    if lv is None:
        return BaselineValidation(model, cutoff, None, None, None, 0, False, NOT_ELIGIBLE)
    view = view_at(tables, cutoff)
    if model == "B1":
        if policy is None:
            raise ValueError("B1's validation labels need the promotion policy")
        pred = predict_b1(view, lv, horizon)
        stack = training_stack(tables, cutoff, horizon, "contractual", "A", policy,
                               landmarks=[lv])
    else:
        pred = predict_b2(view, lv, horizon)
        stack = training_stack(tables, cutoff, horizon, "non_contractual", "A",
                               landmarks=[lv])
    stack = stack.set_index("customer_id")
    if not stack.index.equals(pred.frame.index):
        raise RuntimeError("validation population differs from the prediction population")
    frame = pd.DataFrame({"prediction": pred.frame["prediction"], "realised": stack["realised"]})
    p = frame["prediction"].to_numpy(np.float64)
    r = frame["realised"].to_numpy(np.float64)
    ok = np.isfinite(p)
    cand = CandidateResult(
        model, pred.ok_cell_fraction,
        float(np.sum(p[ok])) / float(np.sum(r[ok])) - 1.0 if ok.any() else math.nan,
        calibration_slope(p[ok], r[ok]) if ok.any() else math.nan,
        float(np.mean((r - p) ** 2)) if ok.all() else math.nan)
    return BaselineValidation(model, cutoff, lv, cand, frame, int((~ok).sum()), True,
                              "eligible")
