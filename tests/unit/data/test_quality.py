"""Unit tests for deterministic data-quality injection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from decision_intelligence.common import seeded_generator
from decision_intelligence.data.quality import (
    inject_duplicate_events,
    inject_late_arrivals,
    inject_missingness,
)


def test_inject_missingness_rate_is_approximately_correct() -> None:
    df = pd.DataFrame({"x": np.arange(5000, dtype=float)})
    result = inject_missingness(df, "x", 0.2, seeded_generator(1))
    observed_rate = result["x"].isna().mean()
    assert 0.15 < observed_rate < 0.25


def test_inject_missingness_zero_rate_is_a_no_op() -> None:
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    result = inject_missingness(df, "x", 0.0, seeded_generator(1))
    assert result["x"].isna().sum() == 0


def test_inject_duplicate_events_preserves_business_key_and_adds_new_ids() -> None:
    df = pd.DataFrame(
        {
            "event_id": [1, 2, 3, 4, 5],
            "exposure_id": [10, 11, 12, 13, 14],
            "event_type": ["click"] * 5,
        }
    )
    result = inject_duplicate_events(df, 0.5, "event_id", seeded_generator(1))
    assert len(result) > len(df)
    assert result["event_id"].is_unique
    # Every duplicated business key must still appear more than once.
    duplicated_business_keys = result.duplicated(subset=["exposure_id", "event_type"]).sum()
    assert duplicated_business_keys == len(result) - len(df)


def test_inject_late_arrivals_only_delays_the_selected_rows() -> None:
    df = pd.DataFrame(
        {
            "event_timestamp": pd.to_datetime(["2024-01-01"] * 1000),
            "recorded_at": pd.to_datetime(["2024-01-01"] * 1000),
        }
    )
    result = inject_late_arrivals(
        df, "event_timestamp", "recorded_at", 0.3, 5, seeded_generator(1)
    )
    delayed = result["recorded_at"] > result["event_timestamp"]
    assert 0.2 < delayed.mean() < 0.4
    max_delay = (result.loc[delayed, "recorded_at"] - result.loc[delayed, "event_timestamp"]).max()
    assert max_delay <= pd.Timedelta(5, unit="D")
