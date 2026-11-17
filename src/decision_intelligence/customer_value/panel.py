"""Point-in-time observable data and the customer panel (Milestone 10).

:class:`Observable` holds only what a business would have recorded by ``cutoff``: customers
acquired by then, transactions and cancellations *recorded* by then (``recorded_at``, via
:func:`decision_intelligence.data.point_in_time.as_of`), marketing spend for completed months.
Duplicated transaction rows are dropped. It carries no ground truth.

**Status at a month-end cutoff ``C``.** A customer acquired by ``C`` is *churned* if a
cancellation is recorded by the cutoff (its last paid month is then known), otherwise *active*:
presumed to have paid month ``C`` (cancellations are recorded within days of the end of the last
paid month, so a churn effective before ``C`` is always recorded by the cutoff). Paid months:
``last_paid - acquisition + 1`` for churned customers (an event), ``C - acquisition + 1`` for
active ones (censored).

**Complete months.** Payments can be recorded up to ``MAX_RECORDING_LAG_DAYS`` after billing, so
at cutoff ``C`` the contribution of a recent month may be incomplete; :meth:`Observable.
complete_through` is the last billing month whose payments are certainly all recorded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from decision_intelligence.customer_value.economics import contribution
from decision_intelligence.customer_value.protocol import MAX_RECORDING_LAG_DAYS
from decision_intelligence.data.point_in_time import as_of


@dataclass(frozen=True)
class Observable:
    cutoff_month: int
    cutoff: pd.Timestamp
    start: pd.Timestamp
    customers: pd.DataFrame
    transactions: pd.DataFrame
    """Recorded transactions with ``age`` (billing month - acquisition month) and
    ``contribution``."""
    cancellations: pd.DataFrame
    marketing: pd.DataFrame
    overhead: pd.DataFrame

    def complete_through(self) -> int:
        """Last billing month all of whose payments are recorded by the cutoff."""
        m = self.cutoff_month
        while m >= 0:
            month_end = self.start + pd.DateOffset(months=m + 1) - pd.to_timedelta(1, unit="s")
            if month_end + pd.to_timedelta(MAX_RECORDING_LAG_DAYS, unit="D") <= self.cutoff:
                return m
            m -= 1
        return -1

    def status(self) -> pd.DataFrame:
        """Per customer: ``acquisition_month, channel, segment, churned, paid_months``."""
        c = self.customers.set_index("customer_id")
        last = self.cancellations.set_index("customer_id")["last_paid_month"]
        churned = c.index.isin(last.index)
        paid = np.where(churned, last.reindex(c.index).to_numpy() - c["acquisition_month"] + 1,
                        self.cutoff_month - c["acquisition_month"] + 1)
        return pd.DataFrame({"acquisition_month": c["acquisition_month"],
                             "channel": c["channel"], "segment": c["segment"],
                             "churned": churned, "paid_months": paid.astype(np.int64)})


class EconomicsTables(Protocol):
    """The observable tables of a scenario (structural; ground truth is not part of it)."""

    @property
    def start(self) -> pd.Timestamp: ...
    @property
    def customers(self) -> pd.DataFrame: ...
    @property
    def transactions(self) -> pd.DataFrame: ...
    @property
    def cancellations(self) -> pd.DataFrame: ...
    @property
    def marketing(self) -> pd.DataFrame: ...
    @property
    def overhead(self) -> pd.DataFrame: ...


def month_end(start: pd.Timestamp, month: int) -> pd.Timestamp:
    return start + pd.DateOffset(months=int(month) + 1) - pd.to_timedelta(1, unit="s")


def observe(tables: EconomicsTables, cutoff_month: int, *, lag_days: int = 0) -> Observable:
    """Observable view of a scenario's tables at the end of ``cutoff_month`` (+ ``lag_days``);
    only the observable tables are read."""
    start = tables.start
    cutoff = month_end(start, cutoff_month) + pd.to_timedelta(lag_days, unit="D")
    customers = tables.customers
    customers = customers.loc[customers["acquisition_month"] <= cutoff_month].copy()
    tx = tables.transactions.drop_duplicates("transaction_id")
    tx = as_of(tx, "recorded_at", cutoff)
    tx = tx.loc[tx["customer_id"].isin(customers["customer_id"])
                & (tx["billing_month"] <= cutoff_month)]
    tx = tx.merge(customers[["customer_id", "acquisition_month"]], on="customer_id")
    tx = tx.assign(age=tx["billing_month"] - tx["acquisition_month"],
                   contribution=contribution(tx))
    cancels = as_of(tables.cancellations, "recorded_at", cutoff)
    cancels = cancels.loc[cancels["last_paid_month"] < cutoff_month]
    mk, oh = tables.marketing, tables.overhead
    return Observable(cutoff_month, cutoff, start, customers, tx.reset_index(drop=True),
                      cancels.reset_index(drop=True),
                      mk.loc[mk["month"] <= cutoff_month].reset_index(drop=True),
                      oh.loc[oh["month"] <= cutoff_month].reset_index(drop=True))
