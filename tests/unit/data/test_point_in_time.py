"""Unit tests for the shared point-in-time utilities."""

from __future__ import annotations

import pandas as pd

from decision_intelligence.data.point_in_time import as_of, latest_as_of


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [1, 1, 1, 2, 2],
            "ts": pd.to_datetime(
                ["2024-01-01", "2024-02-01", "2024-03-01", "2024-01-15", "2024-02-15"]
            ),
            "value": ["a", "b", "c", "x", "y"],
        }
    )


def test_as_of_excludes_future_rows() -> None:
    result = as_of(_frame(), "ts", pd.Timestamp("2024-02-01"))
    assert set(result["value"]) == {"a", "b", "x"}


def test_as_of_mutation_does_not_affect_source() -> None:
    source = _frame()
    result = as_of(source, "ts", pd.Timestamp("2024-02-01"))
    result.loc[result.index[0], "value"] = "mutated"
    assert "mutated" not in source["value"].to_numpy()


def test_latest_as_of_returns_one_row_per_entity() -> None:
    result = latest_as_of(_frame(), "customer_id", "ts", pd.Timestamp("2024-02-10"))
    result = result.set_index("customer_id")
    assert result.loc[1, "value"] == "b"
    assert result.loc[2, "value"] == "x"


def test_latest_as_of_empty_before_any_observation() -> None:
    result = latest_as_of(_frame(), "customer_id", "ts", pd.Timestamp("2023-12-31"))
    assert result.empty
