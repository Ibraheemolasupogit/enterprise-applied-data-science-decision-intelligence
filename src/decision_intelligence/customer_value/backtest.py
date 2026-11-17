"""Temporal backtests with complete observation windows (Milestone 10).

**Complete-window rule.** A backtest at month-end cutoff ``C`` with window ``W`` uses billing
months ``C+1 .. C+W``; it is run only if ``C + W`` is within the data and the realised values
are read from an observable view at ``C + W`` plus ``MAX_RECORDING_LAG_DAYS`` (so every payment of
the window is recorded). Otherwise :class:`IncompleteWindow` is raised — never a partial window.

**Customer backtest (existing customers).** For customers active at ``C`` (no cancellation
recorded; paid month ``C``), projections of their undiscounted contribution over the window:

* *run rate (no churn)* — mean of their last up-to-3 complete recorded months x ``W``;
* *historical average (no churn)* — their mean contribution per recorded month x ``W``;
* *constant churn, pooled* — run rate x ``sum_{t=1..W} (1 - c)^t``, ``c`` over all customers;
* *constant churn by cell* — the same with ``c`` per segment x channel;
* *Kaplan–Meier by cell* — run rate x ``sum_t S(a0 + t) / S(a0)`` with the cell's KM curve
  and geometric tail (``a0`` = paid age at ``C``).

Customers with no completely recorded month at the cutoff (acquired within the last
``MAX_RECORDING_LAG_DAYS``) have no run rate; every metric is computed on the customers that
every method can project, and the excluded count is reported. Customer-level accuracy here is a
**baseline** (predictive LTV is M11).

**Cohort backtest (new customers).** Per channel, H1/H3/H4 projected ``W``-month discounted
value per acquired customer at ``C``, against the realised cohort-matched value of the cohorts
acquired in the following months whose windows are complete.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from decision_intelligence.customer_value.heuristic import empirical_curve, h1_h2, h3, h4
from decision_intelligence.customer_value.historical import cohort_matched
from decision_intelligence.customer_value.panel import EconomicsTables, Observable, observe
from decision_intelligence.customer_value.protocol import (
    CHURN_WINDOW_MONTHS,
    MAX_RECORDING_LAG_DAYS,
    TOP_FRACTION,
)
from decision_intelligence.customer_value.retention import constant_churn

CUSTOMER_METHODS = ("run rate (no churn)", "historical average (no churn)",
                    "constant churn, pooled", "constant churn by cell", "Kaplan–Meier by cell")


class IncompleteWindow(ValueError):
    """The backtest window is not completely observed in the data."""


def check_window(n_months: int, cutoff_month: int, window: int) -> None:
    if window < 1 or cutoff_month < 0:
        raise ValueError("window must be >= 1 and cutoff non-negative")
    if cutoff_month + window > n_months - 1:
        raise IncompleteWindow(
            f"cutoff {cutoff_month} + window {window} needs billing months up to "
            f"{cutoff_month + window}; data end at month {n_months - 1}")


def _recent_mean(tx: pd.DataFrame, through: int, k: int = 3) -> pd.Series:
    t = tx.loc[tx["billing_month"] <= through].sort_values(["customer_id", "billing_month"])
    return t.groupby("customer_id")["contribution"].apply(lambda s: s.tail(k).mean())


def customer_projections(obs: Observable, window: int) -> pd.DataFrame:
    st = obs.status()
    active = st.loc[~st["churned"]]
    through = obs.complete_through()
    tx = obs.transactions.loc[obs.transactions["customer_id"].isin(active.index)]
    rr = _recent_mean(tx, through).reindex(active.index)
    hist = tx.loc[tx["billing_month"] <= through].groupby("customer_id")["contribution"].mean()
    hist = hist.reindex(active.index)
    rr = rr.fillna(hist)
    out = pd.DataFrame(index=active.index)
    out["segment"], out["channel"] = active["segment"], active["channel"]
    out["paid_age"] = obs.cutoff_month - active["acquisition_month"]
    out["run rate (no churn)"] = rr * window
    out["historical average (no churn)"] = hist * window
    pooled = constant_churn(obs, CHURN_WINDOW_MONTHS).churn
    t = np.arange(1, window + 1)
    out["constant churn, pooled"] = rr * float(np.sum((1 - pooled) ** t))
    cell = np.full(len(out), np.nan)
    km = np.full(len(out), np.nan)
    for (seg, chn), g in out.groupby(["segment", "channel"]):
        mask = ((st["segment"] == seg) & (st["channel"] == chn)).to_numpy()
        c = constant_churn(obs, CHURN_WINDOW_MONTHS, mask).churn
        rows = out.index.get_indexer(g.index)
        cell[rows] = rr.loc[g.index].to_numpy() * float(np.sum((1 - c) ** t))
        curve = empirical_curve(obs, mask)
        if curve is None:
            continue
        a0 = g["paid_age"].to_numpy()
        ages = a0[:, None] + np.concatenate([[0], t])[None, :]
        surv = np.where(ages <= curve.last_age, curve.survival[np.minimum(ages, curve.last_age)],
                        curve.survival[curve.last_age]
                        * curve.tail_continuation ** (ages - curve.last_age))
        cond = surv[:, 1:] / np.maximum(surv[:, :1], 1e-300)
        km[rows] = rr.loc[g.index].to_numpy() * cond.sum(axis=1)
    out["constant churn by cell"] = cell
    out["Kaplan–Meier by cell"] = km
    return out


@dataclass(frozen=True)
class CustomerBacktest:
    cutoff_month: int
    window: int
    frame: pd.DataFrame
    """Per active customer: projections, ``realised`` and (if given) ``expected``."""
    summary: pd.DataFrame


def customer_backtest(tables: EconomicsTables, n_months: int, cutoff_month: int, window: int,
                      expected: Callable[[pd.DataFrame], np.ndarray] | None = None
                      ) -> CustomerBacktest:
    check_window(n_months, cutoff_month, window)
    obs = observe(tables, cutoff_month)
    proj = customer_projections(obs, window)
    later = observe(tables, cutoff_month + window, lag_days=MAX_RECORDING_LAG_DAYS)
    tx = later.transactions
    win = tx.loc[(tx["billing_month"] > cutoff_month)
                 & (tx["billing_month"] <= cutoff_month + window)]
    proj["realised"] = win.groupby("customer_id")["contribution"].sum().reindex(
        proj.index, fill_value=0.0)
    if expected is not None:
        proj["expected"] = expected(proj)
    # Common support: every method must have a projection for the customer (customers with no
    # completely recorded month yet have none); totals and ranks use the same customers.
    finite = np.all(np.isfinite(proj[list(CUSTOMER_METHODS)].to_numpy(dtype=np.float64)), axis=1)
    sub = proj.loc[finite]
    real = sub["realised"].to_numpy(dtype=np.float64)
    n_top = max(1, int(np.ceil(TOP_FRACTION * len(sub))))
    rows = []
    for m in CUSTOMER_METHODS:
        p = sub[m].to_numpy(dtype=np.float64)
        top = np.argsort(-p, kind="stable")[:n_top]
        row = {"method": m, "customers": int(len(sub)),
               "excluded_no_complete_month": int((~finite).sum()),
               "projected_total": float(p.sum()), "realised_total": float(real.sum()),
               "bias_vs_realised": float(np.sum(p)) / float(np.sum(real)) - 1,
               "spearman_vs_realised": float(stats.spearmanr(p, real).statistic),
               "top_decile_capture": float(np.sum(real[top])) / float(np.sum(real))}
        if expected is not None:
            ex = sub["expected"].to_numpy(dtype=np.float64)
            row["expected_total"] = float(ex.sum())
            row["bias_vs_expected"] = float(np.sum(p)) / float(np.sum(ex)) - 1
        rows.append(row)
    return CustomerBacktest(cutoff_month, window, proj, pd.DataFrame(rows))


def cohort_backtest(tables: EconomicsTables, n_months: int, cutoff_month: int, window: int,
                    rate: float, channels: tuple[str, ...]) -> pd.DataFrame:
    """Projected ``window``-month value per new customer (H1/H3/H4 at the cutoff) vs the
    realised cohort-matched value of later cohorts with complete windows."""
    check_window(n_months, cutoff_month, window)
    obs = observe(tables, cutoff_month)
    final = observe(tables, n_months - 1, lag_days=MAX_RECORDING_LAG_DAYS)
    real = cohort_matched(final, window, rate)
    real = real.loc[real["acquisition_month"] > cutoff_month]
    if real.empty:
        raise IncompleteWindow("no later cohort has a complete window")
    rows = []
    for ch in channels:
        r = real.loc[real["channel"] == ch, "value"]
        row = {"channel": ch, "cohorts": f"{int(real['acquisition_month'].min())}–"
                                         f"{int(real['acquisition_month'].max())}",
               "customers": int(r.size), "realised": float(r.mean()),
               "realised_se": float(r.std(ddof=1) / np.sqrt(r.size))}
        for name, fn in (("H2 constant churn", h1_h2), ("H3 by cell", h3),
                         ("H4 Kaplan–Meier", h4)):
            e = fn(obs, rate, window, channel=ch)
            row[name] = e.value
            row[f"{name} status"] = e.status
        rows.append(row)
    return pd.DataFrame(rows)
