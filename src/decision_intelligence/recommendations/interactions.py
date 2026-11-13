"""Interaction matrices built from logged data only, as of a cutoff.

Implicit and explicit feedback are kept apart: :class:`InteractionData` holds *implicit*
feedback (clicks and purchases on logged impressions); explicit ratings are a separate
table handled by :mod:`.explicit`. Impressions without a click are not used as negative
feedback — under exposure bias "shown and not clicked" mixes "not liked" with "not
examined" (position bias).

Point-in-time discipline: every table is truncated with
:func:`decision_intelligence.data.point_in_time.as_of` at the cutoff, the one sanctioned
truncation in this repository, and the result records its cutoff. Nothing here imports
or accepts ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from decision_intelligence.common.validation import FloatArray
from decision_intelligence.data.point_in_time import as_of
from decision_intelligence.data.recommendation_scenarios import RecommendationLogs

IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


def deduplicate_impressions(impressions: pd.DataFrame) -> pd.DataFrame:
    """Drop repeated ``impression_id`` rows (keep the first). Logged duplicates are a
    common pipeline defect that silently double-counts clicks."""
    if impressions["impression_id"].is_unique:
        return impressions
    return impressions.drop_duplicates("impression_id", keep="first").reset_index(drop=True)


def item_feature_matrix(items: pd.DataFrame) -> tuple[FloatArray, tuple[str, ...]]:
    """Content features: binary tags plus a one-hot category, rows L2-normalised.

    Price and margin are deliberately excluded: they are commercial attributes, not
    descriptions of what the item is.
    """
    tags = [c for c in items.columns if c.startswith("tag_")]
    categories = tuple(sorted(items["category"].unique()))
    onehot = (items["category"].to_numpy()[:, None] == np.array(categories)[None, :])
    raw = np.hstack([items[tags].to_numpy(dtype=np.float64), onehot.astype(np.float64)])
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    features = np.divide(raw, norms, out=np.zeros_like(raw), where=norms > 0)
    return np.asarray(features, dtype=np.float64), tuple(tags) + tuple(
        f"category={c}" for c in categories)


@dataclass(frozen=True)
class InteractionData:
    """Implicit feedback observable at ``cutoff`` (rows: all users, columns: all items).

    ``user_ids`` and ``item_ids`` are the full, sorted user and item tables, so indices are
    stable across cutoffs. Items not yet launched at the cutoff have no interactions.
    """

    cutoff: pd.Timestamp
    start: pd.Timestamp
    user_ids: IntArray
    item_ids: IntArray
    user_segment: NDArray[np.str_]
    user_region: NDArray[np.str_]
    item_category: NDArray[np.str_]
    item_features: FloatArray
    item_launched: BoolArray
    """``launch_time <= cutoff``: the items that exist at the cutoff."""
    clicks: FloatArray
    """1 if the user clicked the item at least once at or before the cutoff."""
    click_events: pd.DataFrame
    """``user_index, item_index, day`` for every click (days since ``start``)."""

    @property
    def n_users(self) -> int:
        return int(self.user_ids.size)

    @property
    def n_items(self) -> int:
        return int(self.item_ids.size)

    def user_interaction_counts(self) -> IntArray:
        return np.asarray(self.clicks.sum(axis=1), dtype=np.int64)


def index_of(ids: IntArray, values: NDArray[np.int64] | pd.Series) -> IntArray:
    """Map ids to positions in the sorted ``ids``; raises on unknown ids."""
    arr = np.asarray(values, dtype=np.int64)
    pos = np.searchsorted(ids, arr)
    if arr.size and (np.any(pos >= ids.size) or np.any(ids[np.minimum(pos, ids.size - 1)]
                                                         != arr)):
        raise ValueError("unknown id")
    return np.asarray(pos, dtype=np.int64)


def build_interaction_data(
    logs: RecommendationLogs, cutoff: pd.Timestamp, *, clicks: pd.DataFrame | None = None,
) -> InteractionData:
    """Implicit-feedback matrix from clicks at or before ``cutoff``.

    ``clicks`` overrides the click events (``user_id, item_id, timestamp``) — used only
    to construct deliberately invalid splits for the leakage counterexample; the default
    takes them from the de-duplicated impression log, truncated with ``as_of``.
    """
    users = logs.users.sort_values("user_id", kind="stable")
    items = logs.items.sort_values("item_id", kind="stable")
    user_ids = users["user_id"].to_numpy(dtype=np.int64)
    item_ids = items["item_id"].to_numpy(dtype=np.int64)
    if clicks is None:
        imp = deduplicate_impressions(logs.impressions)
        events = as_of(imp.loc[imp["clicked"], ["user_id", "item_id", "timestamp"]],
                       "timestamp", cutoff)
    else:
        events = clicks[["user_id", "item_id", "timestamp"]]
    u = index_of(user_ids, events["user_id"].to_numpy())
    i = index_of(item_ids, events["item_id"].to_numpy())
    matrix = np.zeros((user_ids.size, item_ids.size), dtype=np.float64)
    matrix[u, i] = 1.0
    day = ((pd.to_datetime(events["timestamp"]) - logs.start).dt.total_seconds()
           / 86_400.0).to_numpy()
    features, _ = item_feature_matrix(items)
    return InteractionData(
        cutoff=cutoff, start=logs.start, user_ids=user_ids, item_ids=item_ids,
        user_segment=users["segment"].to_numpy(dtype=str),
        user_region=users["region"].to_numpy(dtype=str),
        item_category=items["category"].to_numpy(dtype=str),
        item_features=features,
        item_launched=(items["launch_time"] <= cutoff).to_numpy(),
        clicks=matrix,
        click_events=pd.DataFrame({"user_index": u, "item_index": i, "day": day}),
    )
