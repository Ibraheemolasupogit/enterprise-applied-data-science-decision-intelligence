"""Realised (historical) contribution with censoring-aware cohort rules (Milestone 10).

**Realised** contribution is what was actually recorded — never a projection. The
*cohort-matched* ``A``-month value of a customer is the discounted contribution of ages
``0 .. A-1``, with zeros after churn, computed **only** for customers whose whole ``A``-month
window is completely observed (``acquisition + A - 1 <=`` the last completely recorded month).
Averaging it over every such acquired customer — churned ones included — is an unbiased
estimate of the expected ``A``-month value per acquired customer for those cohorts.

Two naive versions are provided as counterexamples: *all customers* (windows truncated by the
cutoff are treated as if complete — a truncation bias, downward when monthly contribution is
positive) and *survivors only* (customers with a complete window who paid every month of it —
pure survivorship bias, upward when later months contribute positively). Neither direction is a
theorem here (promotional months can have negative contribution); the evidence reports the
realised directions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from decision_intelligence.customer_value.economics import discount_factors
from decision_intelligence.customer_value.panel import Observable


def _value_by_customer(obs: Observable, age: int, rate: float, through_month: int
                       ) -> pd.Series:
    tx = obs.transactions
    tx = tx.loc[(tx["age"] < age) & (tx["billing_month"] <= through_month)]
    d = discount_factors(age, rate)
    v = (tx["contribution"] * d[tx["age"].to_numpy()]).groupby(tx["customer_id"]).sum()
    return pd.Series(v.reindex(obs.customers["customer_id"], fill_value=0.0), dtype=float)


def cohort_matched(obs: Observable, age: int, rate: float) -> pd.DataFrame:
    """Per eligible customer: ``customer_id, acquisition_month, channel, segment, value``."""
    if age < 1:
        raise ValueError("age must be >= 1")
    through = obs.complete_through()
    cust = obs.customers.loc[obs.customers["acquisition_month"] + age - 1 <= through]
    if cust.empty:
        raise ValueError(f"no cohort has a complete {age}-month window at this cutoff")
    v = _value_by_customer(obs, age, rate, through)
    out = cust[["customer_id", "acquisition_month", "channel", "segment"]].copy()
    out["value"] = v.reindex(out["customer_id"]).to_numpy()
    return out.reset_index(drop=True)


def cumulative_curve(obs: Observable, age: int, rate: float, mask: np.ndarray | None = None
                     ) -> np.ndarray:
    """Mean cumulative discounted contribution by age (0..age-1) per acquired customer, over
    customers whose ``age``-month window is complete."""
    through = obs.complete_through()
    keep = (obs.customers["acquisition_month"] + age - 1 <= through).to_numpy()
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool)  # aligned with obs.customers rows
    cust = obs.customers.loc[keep]
    if cust.empty:
        raise ValueError("no complete windows")
    tx = obs.transactions
    tx = tx.loc[tx["customer_id"].isin(cust["customer_id"]) & (tx["age"] < age)]
    d = discount_factors(age, rate)
    by_age = np.bincount(tx["age"].to_numpy(), weights=tx["contribution"].to_numpy()
                         * d[tx["age"].to_numpy()], minlength=age)
    return np.asarray(np.cumsum(by_age) / len(cust), dtype=np.float64)


def naive_all_customers(obs: Observable, age: int, rate: float) -> pd.DataFrame:
    """Counterexample: every acquired customer, truncated windows treated as complete."""
    v = _value_by_customer(obs, age, rate, obs.cutoff_month)
    out = obs.customers[["customer_id", "acquisition_month", "channel", "segment"]].copy()
    out["value"] = v.reindex(out["customer_id"]).to_numpy()
    return out.reset_index(drop=True)


def naive_survivors(obs: Observable, age: int, rate: float) -> pd.DataFrame:
    """Counterexample: complete-window customers who paid all ``age`` months (survivorship)."""
    out = cohort_matched(obs, age, rate)
    paid = obs.status()["paid_months"].reindex(out["customer_id"]).to_numpy()
    return out.loc[paid >= age].reset_index(drop=True)
