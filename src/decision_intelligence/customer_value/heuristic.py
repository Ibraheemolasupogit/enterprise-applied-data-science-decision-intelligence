"""Heuristic LTV: projected and expected-lifetime contribution per new customer (M10).

All are *heuristics* estimated from observable data at a cutoff — not predictive models (M11).
Each returns an :class:`~decision_intelligence.customer_value.economics.Estimate` with a status:
``"undefined"`` when there is no data or the infinite-horizon sum diverges, ``"unstable"`` when
the churn rate rests on fewer than ``MIN_CHURN_EVENTS`` events.

* **H0 (legacy baseline)** — per customer: historical *revenue* to date + current monthly
  revenue x ``sum_{t=1..12} (1 - c)^t``; undiscounted, revenue-based (replicates the earlier
  Product Growth method); averaged over the group's customers.
* **H1 / H2** — constant churn ``c`` and run-rate contribution ``m`` (mean contribution per paid
  month over the trailing window), geometric value from acquisition: H1 infinite horizon, H2
  finite horizons.
* **H3** — H1/H2 computed per segment x channel cell and mixed by the group's recent
  acquisition mix.
* **H4** — empirical curve: ``sum_a m_a S(a) d^a`` with Kaplan–Meier ``S`` and mean contribution
  by age ``m_a`` for ages with at least ``MIN_AT_RISK`` customers at risk; beyond, a geometric
  tail at the KM hazard of the last ``KM_TAIL_MONTHS`` observed ages and the last ``m_a``.

Infinite-horizon values are *expected lifetime* contribution under the heuristic's assumptions;
finite-horizon values are *projected* contribution over that horizon.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from decision_intelligence.customer_value.economics import (
    Estimate,
    discount_factors,
    geometric_value,
)
from decision_intelligence.customer_value.panel import Observable
from decision_intelligence.customer_value.protocol import (
    CHURN_WINDOW_MONTHS,
    KM_TAIL_MONTHS,
    MIN_AT_RISK,
    MIN_CHURN_EVENTS,
)
from decision_intelligence.customer_value.retention import (
    at_risk_curve,
    constant_churn,
    paid_month_km,
)


def _mask(obs: Observable, segment: str | None, channel: str | None) -> np.ndarray:
    st = obs.status()
    m = np.ones(len(st), dtype=bool)
    if segment is not None:
        m &= (st["segment"] == segment).to_numpy()
    if channel is not None:
        m &= (st["channel"] == channel).to_numpy()
    return m


def run_rate(obs: Observable, mask: np.ndarray, window: int = CHURN_WINDOW_MONTHS) -> float:
    """Mean contribution per paid month over complete billing months in the trailing window."""
    ids = obs.status().index[mask]
    through = obs.complete_through()
    tx = obs.transactions
    tx = tx.loc[tx["customer_id"].isin(ids) & (tx["billing_month"] <= through)
                & (tx["billing_month"] > through - window)]
    return float(tx["contribution"].mean()) if len(tx) else float("nan")


def _status_for(events: int, value: Estimate) -> Estimate:
    if value.status == "undefined":
        return value
    if events < MIN_CHURN_EVENTS:
        return Estimate(value.value, "unstable", f"only {events} churn events")
    return value


def h1_h2(obs: Observable, rate: float, horizon: int | None, *, segment: str | None = None,
          channel: str | None = None) -> Estimate:
    mask = _mask(obs, segment, channel)
    if not mask.any():
        return Estimate(float("nan"), "undefined", "no customers")
    ch = constant_churn(obs, CHURN_WINDOW_MONTHS, mask)
    m = run_rate(obs, mask)
    if ch.trials == 0 or not np.isfinite(m):
        return Estimate(float("nan"), "undefined", "no trials in the churn window")
    return _status_for(ch.events, geometric_value(m, 1.0 - ch.churn, rate, horizon))


def h3(obs: Observable, rate: float, horizon: int | None, *, segment: str | None = None,
       channel: str | None = None) -> Estimate:
    """Cell-level H1/H2 mixed by the group's acquisitions over the churn window."""
    st = obs.status()
    sel = _mask(obs, segment, channel) & (
        st["acquisition_month"] > obs.cutoff_month - CHURN_WINDOW_MONTHS).to_numpy()
    if not sel.any():
        return Estimate(float("nan"), "undefined", "no recent acquisitions")
    cells = st.loc[sel, ["segment", "channel"]].astype(str)
    keys, counts = np.unique(cells["segment"] + "|" + cells["channel"], return_counts=True)
    total, statuses, reasons = 0.0, [], []
    for key, n in zip(keys.tolist(), counts.tolist(), strict=True):
        seg, chn = str(key).split("|")
        e = h1_h2(obs, rate, horizon, segment=seg, channel=chn)
        if e.status == "undefined":
            return Estimate(float("nan"), "undefined", f"cell {seg}/{chn}: {e.reason}")
        total += n * e.value
        statuses.append(e.status)
        if e.reason:
            reasons.append(f"{seg}/{chn}: {e.reason}")
    status = "unstable" if "unstable" in statuses else "ok"
    return Estimate(float(total / counts.sum()), status, "; ".join(reasons))


@dataclass(frozen=True)
class EmpiricalCurve:
    contribution: np.ndarray
    survival: np.ndarray
    last_age: int
    tail_continuation: float
    events: int


def empirical_curve(obs: Observable, mask: np.ndarray) -> EmpiricalCurve | None:
    st = obs.status().loc[mask]
    if st.empty:
        return None
    km = paid_month_km(st["paid_months"].to_numpy(), st["churned"].to_numpy())
    n_ages = int(st["paid_months"].max())
    at_risk = at_risk_curve(st["paid_months"].to_numpy(), n_ages)
    usable = np.flatnonzero(at_risk >= MIN_AT_RISK)
    if usable.size < 2:
        return None
    last = int(usable[-1])
    through = obs.complete_through()
    tx = obs.transactions
    tx = tx.loc[tx["customer_id"].isin(st.index) & (tx["billing_month"] <= through)
                & (tx["age"] <= last)]
    sums = np.bincount(tx["age"].to_numpy(), weights=tx["contribution"].to_numpy(),
                       minlength=last + 1)
    counts = np.bincount(tx["age"].to_numpy(), minlength=last + 1)
    contrib = np.divide(sums, counts, out=np.full(last + 1, np.nan), where=counts > 0)
    contrib = pd.Series(contrib).ffill().bfill().to_numpy()
    surv = km.at(np.arange(last + 1))
    lo = max(0, last - KM_TAIL_MONTHS)
    q = float((surv[last] / surv[lo]) ** (1.0 / (last - lo))) if surv[lo] > 0 else 0.0
    return EmpiricalCurve(np.asarray(contrib, dtype=np.float64), surv, last, q,
                          int(km.events.sum()))


def h4(obs: Observable, rate: float, horizon: int | None, *, segment: str | None = None,
       channel: str | None = None) -> Estimate:
    curve = empirical_curve(obs, _mask(obs, segment, channel))
    if curve is None:
        return Estimate(float("nan"), "undefined", "too few customers at risk")
    last, q = curve.last_age, curve.tail_continuation
    n_obs = last + 1 if horizon is None else min(horizon, last + 1)
    d = discount_factors(n_obs, rate)
    head = float(np.sum(curve.contribution[:n_obs] * curve.survival[:n_obs] * d))
    if horizon is not None and horizon <= last + 1:
        return _status_for(curve.events, Estimate(head, "ok"))
    s = q / (1.0 + rate)
    base = curve.contribution[last] * curve.survival[last] * (1.0 + rate) ** -last
    if horizon is None:
        if s >= 1:
            return Estimate(float("inf"), "undefined", "tail diverges")
        return _status_for(curve.events, Estimate(head + base * s / (1.0 - s), "ok"))
    k = horizon - (last + 1)
    tail = base * s * (1.0 - s ** k) / (1.0 - s) if s != 1 else base * k
    return _status_for(curve.events, Estimate(head + tail, "ok"))


def h0_legacy(obs: Observable, *, segment: str | None = None, channel: str | None = None,
              months: int = 12) -> Estimate:
    """Historical revenue + churn-adjusted 12-month revenue projection (undiscounted)."""
    mask = _mask(obs, segment, channel)
    st = obs.status().loc[mask]
    if st.empty:
        return Estimate(float("nan"), "undefined", "no customers")
    ch = constant_churn(obs, CHURN_WINDOW_MONTHS, mask)
    tx = obs.transactions.loc[obs.transactions["customer_id"].isin(st.index)]
    hist = tx.groupby("customer_id")["revenue"].sum().reindex(st.index, fill_value=0.0)
    last_rev = (tx.sort_values("billing_month").groupby("customer_id")["revenue"].last()
                .reindex(st.index, fill_value=0.0))
    factor = float(sum((1.0 - ch.churn) ** t for t in range(1, months + 1)))
    projection = np.where(st["churned"].to_numpy(), 0.0, last_rev.to_numpy() * factor)
    return _status_for(ch.events, Estimate(float(np.mean(hist.to_numpy() + projection)), "ok"))
