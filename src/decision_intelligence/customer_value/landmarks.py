"""Leakage-safe landmark datasets for Milestone 11 (P2).

**Training stack at evaluation cutoff ``C``.** Everything is computed from :func:`view_at`
``(tables, C)``: the tables truncated to what was recorded by the end of month ``C``. For each
protocol landmark ``L`` (``predictive_protocol.training_landmarks``: ``L + W <= C - 2``) a row
holds the point-in-time features at ``L`` (:mod:`.features`) and the label over months
``L+1 .. L+W``, which is completely recorded by ``C`` because of the complete-month rule. No
value recorded after ``C`` can enter a feature or a label (perturbation-tested).

**Holdout frame at ``C``.** Features at ``C``; labels over ``C+1 .. C+W`` read from
:func:`~.panel.observe` at ``C + W`` plus the recording lag, refused by M10's
``check_window`` when the window is incomplete. Holdout labels are for evaluation only.

**Labels.** ``realised`` = undiscounted contribution over the window (M10's contribution;
churned customers contribute zeros) — the protocol's primary target; ``discounted`` = sum of
``d^k`` x contribution of month ``L + k``; ``paid_in_window`` = paid months (contractual) or
purchases (non-contractual) in the window.

**Repeated customers.** A customer appears in every landmark at which they are in the
population, so stack rows are clustered by ``customer_id`` (protocol amendment 2). Inference
resamples customers with all their rows (:func:`resample_customers`); effective sample sizes
use the cluster sizes (:func:`cluster_sizes`, Kish design effect in :mod:`.precision`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from decision_intelligence.customer_value import predictive_protocol as pp
from decision_intelligence.customer_value.backtest import check_window
from decision_intelligence.customer_value.economics import contribution
from decision_intelligence.customer_value.features import (
    PromotionPolicy,
    contractual_features,
    noncontractual_features,
)
from decision_intelligence.customer_value.panel import EconomicsTables, month_end, observe
from decision_intelligence.data.point_in_time import as_of


@dataclass(frozen=True)
class TablesView:
    """Observable tables as recorded by ``cutoff`` (satisfies ``EconomicsTables``)."""

    start: pd.Timestamp
    customers: pd.DataFrame
    transactions: pd.DataFrame
    cancellations: pd.DataFrame
    marketing: pd.DataFrame
    overhead: pd.DataFrame
    signals: pd.DataFrame
    cutoff_month: int


def view_at(tables: EconomicsTables, cutoff: int) -> TablesView:
    end = month_end(tables.start, cutoff)
    cust = tables.customers.loc[tables.customers["acquisition_month"] <= cutoff]
    signals = getattr(tables, "signals", None)
    sig = (as_of(signals, "recorded_at", end) if signals is not None
           else pd.DataFrame(columns=["customer_id", "month", "recorded_at"]))
    return TablesView(tables.start, cust.copy(), as_of(tables.transactions, "recorded_at", end),
                      as_of(tables.cancellations, "recorded_at", end),
                      tables.marketing.loc[tables.marketing["month"] <= cutoff].copy(),
                      tables.overhead.loc[tables.overhead["month"] <= cutoff].copy(), sig,
                      cutoff)


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


def _labels(transactions: pd.DataFrame, index: pd.Index, landmark: int, horizon: int
            ) -> pd.DataFrame:
    tx = transactions.drop_duplicates("transaction_id")
    win = tx.loc[(tx["billing_month"] > landmark) & (tx["billing_month"] <= landmark + horizon)
                 & tx["customer_id"].isin(index)]
    k = (win["billing_month"] - landmark).to_numpy()
    c = contribution(win).to_numpy()
    d = (1.0 + pp.MONTHLY_DISCOUNT_RATE) ** -k
    by = pd.DataFrame({"customer_id": win["customer_id"].to_numpy(), "c": c, "dc": c * d})
    g = by.groupby("customer_id")
    return pd.DataFrame({
        "realised": g["c"].sum().reindex(index, fill_value=0.0),
        "discounted": g["dc"].sum().reindex(index, fill_value=0.0),
        "paid_in_window": g.size().reindex(index, fill_value=0).astype(np.int64),
    }, index=index)


def training_stack(tables: EconomicsTables, cutoff: int, horizon: int, kind: str, tier: str,
                   policy: PromotionPolicy | None = None,
                   landmarks: Sequence[int] | None = None) -> pd.DataFrame:
    """Rows ``customer_id, landmark, <features>, realised, discounted, paid_in_window`` for the
    protocol landmarks of ``(cutoff, horizon)`` (or a given subset of them), computed only from
    :func:`view_at` ``(tables, cutoff)``."""
    allowed = pp.training_landmarks(cutoff, horizon)
    chosen = tuple(allowed if landmarks is None else landmarks)
    if not chosen:
        raise ValueError(f"no landmarks to stack at cutoff {cutoff} (horizon {horizon})")
    if not set(chosen) <= set(allowed):
        raise ValueError(f"landmarks {sorted(set(chosen) - set(allowed))} have labels that "
                         f"are not completely recorded by cutoff {cutoff}")
    view = view_at(tables, cutoff)
    parts = []
    for landmark in chosen:
        feats = _features(view, landmark, kind, tier, policy)
        labels = _labels(view.transactions, feats.index, landmark, horizon)
        parts.append(feats.join(labels).assign(landmark=landmark).reset_index())
    stack = pd.concat(parts, ignore_index=True)
    first = ["customer_id", "landmark"]
    return stack[first + [c for c in stack.columns if c not in first]]


def holdout_frame(tables: EconomicsTables, n_months: int, cutoff: int, horizon: int,
                  kind: str, tier: str, policy: PromotionPolicy | None = None
                  ) -> pd.DataFrame:
    """Features at ``cutoff`` and labels over ``cutoff+1 .. cutoff+horizon`` (read after the
    recording lag). Evaluation only."""
    check_window(n_months, cutoff, horizon)
    feats = _features(tables, cutoff, kind, tier, policy)
    later = observe(tables, cutoff + horizon, lag_days=pp.RECORDING_LAG_DAYS)
    labels = _labels(later.transactions, feats.index, cutoff, horizon)
    return feats.join(labels).assign(landmark=cutoff).reset_index()


def cluster_sizes(frame: pd.DataFrame) -> np.ndarray:
    """Rows per customer (the cluster sizes for the Kish design effect)."""
    return frame["customer_id"].value_counts().to_numpy()


def resample_customers(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """One customer-level bootstrap resample: customers drawn with replacement, each carrying
    all of their rows; ``cluster`` identifies the draw (a customer drawn twice appears as two
    clusters)."""
    ids = frame["customer_id"].unique()
    draw = rng.choice(ids, size=ids.size, replace=True)
    rows = {c: np.asarray(r, dtype=np.int64)
            for c, r in frame.groupby("customer_id").indices.items()}
    parts = [np.column_stack([rows[c], np.full(rows[c].size, j)]) for j, c in enumerate(draw)]
    idx = np.concatenate(parts)
    return frame.iloc[idx[:, 0]].assign(cluster=idx[:, 1]).reset_index(drop=True)
