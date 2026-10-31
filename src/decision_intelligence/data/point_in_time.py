"""Point-in-time utilities shared by every pillar that must avoid leakage.

Forecasting, recommenders and LTV each need "don't use data from after the
prediction date" logic. Per docs/architecture/overview.md, this module is
the single owner of that logic — pillars must call these functions rather
than reimplementing an ``as of`` filter locally.
"""

from __future__ import annotations

import pandas as pd


def as_of(df: pd.DataFrame, timestamp_column: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Return only the rows of ``df`` observable at or before ``cutoff``.

    This is the one sanctioned way to truncate a table for a
    temporal train/test split or a feature computed "as of" some date.
    """
    return df.loc[df[timestamp_column] <= cutoff].copy()


def latest_as_of(
    df: pd.DataFrame,
    entity_column: str,
    timestamp_column: str,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """Return each entity's most recent row at or before ``cutoff``.

    Used for point-in-time feature joins: e.g. "each customer's latest
    known segment as of the prediction date", without accidentally picking
    up a later row.
    """
    observable = as_of(df, timestamp_column, cutoff)
    if observable.empty:
        return observable
    return (
        observable.sort_values(timestamp_column)
        .groupby(entity_column, as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
