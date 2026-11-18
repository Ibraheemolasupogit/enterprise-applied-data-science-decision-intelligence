"""Point-in-time customer features for Milestone 11 (P2).

Implements the feature registry of :mod:`.predictive_protocol` (``FEATURES``). Every value is
computed from what was *recorded* by the cutoff ``C``: transactions and cancellations through
M10's :func:`~.panel.observe` (``recorded_at <= C``), behavioural signals through
:func:`decision_intelligence.data.point_in_time.as_of` on ``recorded_at`` and restricted to
months ``<= C - SIGNAL_COMPLETE_OFFSET``. Transaction-based features use completely recorded
months only (``billing_month <= C - COMPLETE_MONTH_OFFSET``). No feature reads a forbidden
source (the account-closure summary), the generator's parameters or its latent truth.

**Population.** Contractual: customers active at ``C`` (no cancellation recorded) with at
least one completely recorded month — M10's comparison-set rule. Non-contractual: every
customer acquired by ``complete_through(C)`` (dropout is never observed).

**Definitions** (see ``FEATURES`` for the registry):

* ``promo_months_remaining`` — months of the business's own promotion policy (passed in, not
  read from the generator) still to run after month ``C``: ``max(0, months - (tenure + 1))``
  for promoted channels.
* ``usage_change_3m`` — ``log(1 + mean usage, last 3 usable months) - log(1 + mean usage,
  previous 3)``; ``0`` when the previous window is empty.
* ``months_since_payment_failure`` — ``(C - 1) - month of the last usable failure``, capped at
  ``FAILURE_CAP_MONTHS`` (also the value when no failure was recorded).
* Non-contractual times are in weeks (``NONCONTRACTUAL_TIME_UNIT_DAYS``): ``recency`` = time
  from the acquisition purchase to the last repeat purchase (0 if none), ``age`` = time from
  acquisition to the end of the last complete month; ``mean_spend`` = mean revenue of repeat
  purchases (0 if none).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
import pandas as pd

from decision_intelligence.customer_value import predictive_protocol as pp
from decision_intelligence.customer_value.panel import EconomicsTables, Observable, observe
from decision_intelligence.data.point_in_time import as_of

FAILURE_CAP_MONTHS = 24
NONCONTRACTUAL_TIME_UNIT_DAYS = 7.0


class SignalTables(EconomicsTables, Protocol):
    @property
    def signals(self) -> pd.DataFrame: ...


@dataclass(frozen=True)
class PromotionPolicy:
    """The business's own promotion policy (known to it; not a generator parameter read)."""

    months: int
    channels: tuple[str, ...]


def feature_names(kind: str, tier: str) -> list[str]:
    return [f.name for f in pp.features(kind, tier)]


def _population(obs: Observable) -> pd.DataFrame:
    st = obs.status()
    return st.loc[~st["churned"] & (st["acquisition_month"] <= obs.complete_through())]


def usable_signals(tables: SignalTables, cutoff: int) -> pd.DataFrame:
    """Signals recorded by the end of ``cutoff`` for months ``<= cutoff - offset``."""
    end = tables.start + pd.DateOffset(months=cutoff + 1) - pd.to_timedelta(1, unit="s")
    seen = as_of(tables.signals, "recorded_at", end)
    return seen.loc[seen["month"] <= cutoff - pp.SIGNAL_COMPLETE_OFFSET]


def contractual_features(tables: EconomicsTables, cutoff: int, tier: str,
                         policy: PromotionPolicy) -> pd.DataFrame:
    """Features at ``cutoff`` for the contractual population, indexed by ``customer_id``."""
    if tier not in ("A", "B"):
        raise ValueError("tier must be 'A' or 'B'")
    obs = observe(tables, cutoff)
    pop = _population(obs)
    through = obs.complete_through()
    tx = obs.transactions.loc[obs.transactions["billing_month"] <= through]
    tx = tx.sort_values(["customer_id", "billing_month"], kind="stable")
    by = tx.groupby("customer_id")["contribution"]
    tenure = cutoff - pop["acquisition_month"]
    promoted = pop["channel"].isin(policy.channels)
    frame = pd.DataFrame({
        "segment": pop["segment"], "channel": pop["channel"],
        "tenure_months": tenure, "acquisition_cohort": pop["acquisition_month"],
        "promo_months_remaining": np.where(promoted, np.maximum(0, policy.months
                                                                  - (tenure + 1)), 0),
        "paid_months": pop["paid_months"],
        "mean_contribution": by.mean().reindex(pop.index),
        "last_contribution": by.last().reindex(pop.index),
    }, index=pop.index)
    if tier == "B":
        if not hasattr(tables, "signals"):
            raise ValueError("Tier B needs a signals table")
        sig = usable_signals(cast(SignalTables, tables), cutoff)
        frame = frame.join(_signal_features(sig, cutoff, pop.index))
    return frame[feature_names("contractual", tier)]


def _signal_features(sig: pd.DataFrame, cutoff: int, index: pd.Index) -> pd.DataFrame:
    last = cutoff - pp.SIGNAL_COMPLETE_OFFSET  # latest usable month
    w3 = sig.loc[sig["month"] > last - 3]
    prev = sig.loc[(sig["month"] <= last - 3) & (sig["month"] > last - 6)]
    w6 = sig.loc[sig["month"] > last - 6]
    u3 = w3.groupby("customer_id")["usage"].mean().reindex(index)
    u_prev = prev.groupby("customer_id")["usage"].mean().reindex(index)
    change = (np.log1p(u3) - np.log1p(u_prev)).fillna(0.0)
    fails = sig.loc[sig["payment_failures"] > 0].groupby("customer_id")["month"].max()
    since = (last - fails).reindex(index).clip(upper=FAILURE_CAP_MONTHS)
    return pd.DataFrame({
        "usage_mean_3m": u3,
        "usage_change_3m": change,
        "support_contacts_3m": w3.groupby("customer_id")["support_contacts"].sum()
        .reindex(index, fill_value=0),
        "payment_failures_6m": w6.groupby("customer_id")["payment_failures"].sum()
        .reindex(index, fill_value=0),
        "months_since_payment_failure": since.fillna(FAILURE_CAP_MONTHS),
    }, index=index)


def noncontractual_features(tables: EconomicsTables, cutoff: int) -> pd.DataFrame:
    """Features at ``cutoff`` for every customer acquired by ``complete_through(cutoff)``."""
    obs = observe(tables, cutoff)
    through = obs.complete_through()
    cust = obs.customers.set_index("customer_id")
    cust = cust.loc[cust["acquisition_month"] <= through]
    tx = obs.transactions.loc[obs.transactions["billing_month"] <= through]
    tx = tx.loc[tx["customer_id"].isin(cust.index)].sort_values(
        ["customer_id", "billing_date"], kind="stable")
    unit_s = NONCONTRACTUAL_TIME_UNIT_DAYS * 86400.0
    acq = cust["acquisition_date"]
    since = (tx["billing_date"].to_numpy() - acq.reindex(tx["customer_id"]).to_numpy())
    tx = tx.assign(t=pd.to_timedelta(since).total_seconds() / unit_s)
    repeat = tx.loc[tx["t"] > 0]
    end = tables.start + pd.DateOffset(months=through + 1) - pd.to_timedelta(1, unit="s")
    frame = pd.DataFrame({
        "segment": cust["segment"], "channel": cust["channel"],
        "frequency": repeat.groupby("customer_id").size().reindex(cust.index, fill_value=0),
        "recency": repeat.groupby("customer_id")["t"].max().reindex(cust.index, fill_value=0.0),
        "age": (end - acq).dt.total_seconds() / unit_s,
        "mean_spend": repeat.groupby("customer_id")["revenue"].mean().reindex(
            cust.index, fill_value=0.0),
        "purchases_6m": tx.loc[tx["billing_month"] > through - 6].groupby("customer_id")
        .size().reindex(cust.index, fill_value=0),
    }, index=cust.index)
    return frame[feature_names("non_contractual", "A")]
