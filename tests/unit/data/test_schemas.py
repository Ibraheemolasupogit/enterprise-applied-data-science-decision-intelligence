"""Unit tests for the data-contract validator itself (not the generators).

These use small, hand-built DataFrames so a violation can be injected on
purpose and checked for — proving the validator actually detects broken
contracts, not just that it stays quiet on clean data.
"""

from __future__ import annotations

import pandas as pd

from decision_intelligence.data.schemas import (
    TableSchema,
    validate_referential_integrity,
    validate_table,
)

_SCHEMA = TableSchema(
    name="widgets",
    columns={"widget_id": "int", "name": "str", "price": "float"},
    primary_key=("widget_id",),
    not_null=("widget_id", "name"),
)


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"widget_id": [1, 2, 3], "name": ["a", "b", "c"], "price": [1.0, 2.0, 3.0]}
    )


def test_valid_table_has_no_violations() -> None:
    assert validate_table(_valid_frame(), _SCHEMA) == []


def test_missing_column_is_detected() -> None:
    df = _valid_frame().drop(columns=["price"])
    violations = validate_table(df, _SCHEMA)
    assert any("missing columns" in v for v in violations)


def test_wrong_dtype_is_detected() -> None:
    df = _valid_frame()
    df["price"] = df["price"].astype(str)
    violations = validate_table(df, _SCHEMA)
    assert any("expected dtype kind 'float'" in v for v in violations)


def test_null_in_not_null_column_is_detected() -> None:
    df = _valid_frame()
    df.loc[0, "name"] = None
    violations = validate_table(df, _SCHEMA)
    assert any("contains null values" in v for v in violations)


def test_duplicate_primary_key_is_detected() -> None:
    df = _valid_frame()
    df.loc[2, "widget_id"] = 1
    violations = validate_table(df, _SCHEMA)
    assert any("duplicate row" in v for v in violations)


def test_referential_integrity_detects_orphan_foreign_key() -> None:
    parent_schema = TableSchema(
        name="parents", columns={"parent_id": "int"}, primary_key=("parent_id",)
    )
    child_schema = TableSchema(
        name="children",
        columns={"child_id": "int", "parent_id": "int"},
        primary_key=("child_id",),
        foreign_keys={"parent_id": ("parents", "parent_id")},
    )
    tables = {
        "parents": pd.DataFrame({"parent_id": [1, 2]}),
        "children": pd.DataFrame({"child_id": [10, 11], "parent_id": [1, 99]}),
    }
    violations = validate_referential_integrity(
        tables, {"parents": parent_schema, "children": child_schema}
    )
    assert any("no matching parent row" in v for v in violations)
