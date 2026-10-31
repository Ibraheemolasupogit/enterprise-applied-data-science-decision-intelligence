"""Deterministic, configurable data-quality imperfections.

Clean synthetic data would let every later pillar skip the robustness
checks real pipelines actually need. These functions inject specific,
rate-controlled imperfections so those checks have something real to
catch — see :class:`decision_intelligence.data.config.DataQualityConfig`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def inject_missingness(
    df: pd.DataFrame, column: str, rate: float, rng: np.random.Generator
) -> pd.DataFrame:
    """Return a copy of ``df`` with ``rate`` of ``column`` set to null."""
    if rate <= 0:
        return df
    result = df.copy()
    mask = rng.random(len(result)) < rate
    result.loc[mask, column] = np.nan
    return result


def inject_duplicate_events(
    df: pd.DataFrame, rate: float, id_column: str, rng: np.random.Generator
) -> pd.DataFrame:
    """Append duplicate rows (same business key, a fresh ``id_column``) for ``rate`` of rows.

    Duplicates get a new, unique ``id_column`` value — exactly how a
    retried delivery would actually appear in an events log — so the
    primary-key contract still holds while a business-key uniqueness check
    can still find them.
    """
    if rate <= 0 or len(df) == 0:
        return df
    mask = rng.random(len(df)) < rate
    duplicates = df.loc[mask].copy()
    if duplicates.empty:
        return df
    next_id = int(df[id_column].max()) + 1
    duplicates[id_column] = np.arange(next_id, next_id + len(duplicates))
    return pd.concat([df, duplicates], ignore_index=True)


def inject_late_arrivals(
    df: pd.DataFrame,
    event_timestamp_column: str,
    recorded_at_column: str,
    rate: float,
    max_delay_days: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Delay ``recorded_at_column`` behind ``event_timestamp_column`` for ``rate`` of rows."""
    if rate <= 0 or len(df) == 0:
        return df
    result = df.copy()
    mask = rng.random(len(result)) < rate
    delay_days = rng.integers(1, max_delay_days + 1, size=int(mask.sum()))
    result.loc[mask, recorded_at_column] = result.loc[
        mask, event_timestamp_column
    ] + pd.to_timedelta(delay_days, unit="D")
    return result
