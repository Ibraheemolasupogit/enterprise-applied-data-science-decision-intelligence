"""Data contracts for every synthetic table: columns, dtypes, keys.

A table "passes" its contract when :func:`validate_table` returns an empty
list of violations. Referential integrity across tables is checked
separately by :func:`validate_referential_integrity` because it requires
the full set of generated tables, not just one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

_DTYPE_CHECKS = {
    "int": pd.api.types.is_integer_dtype,
    "float": pd.api.types.is_float_dtype,
    "str": pd.api.types.is_object_dtype,
    "bool": pd.api.types.is_bool_dtype,
    "datetime": pd.api.types.is_datetime64_any_dtype,
}


@dataclass(frozen=True)
class TableSchema:
    """The data contract for a single synthetic table."""

    name: str
    columns: dict[str, str]
    primary_key: tuple[str, ...]
    foreign_keys: dict[str, tuple[str, str]] = field(default_factory=dict)
    not_null: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for column in (*self.primary_key, *self.not_null, *self.foreign_keys):
            if column not in self.columns:
                raise ValueError(f"{self.name}: '{column}' is not a declared column")
        for dtype in self.columns.values():
            if dtype not in _DTYPE_CHECKS:
                raise ValueError(f"{self.name}: unknown dtype kind '{dtype}'")


def validate_table(df: pd.DataFrame, schema: TableSchema) -> list[str]:
    """Return a list of contract violations for ``df`` against ``schema``.

    An empty list means the table satisfies its own contract (columns,
    dtypes, not-null constraints and primary-key uniqueness). Foreign-key
    satisfaction is checked separately — see
    :func:`validate_referential_integrity`.
    """
    violations: list[str] = []

    missing_columns = set(schema.columns) - set(df.columns)
    extra_columns = set(df.columns) - set(schema.columns)
    if missing_columns:
        violations.append(f"{schema.name}: missing columns {sorted(missing_columns)}")
    if extra_columns:
        violations.append(f"{schema.name}: undeclared columns {sorted(extra_columns)}")

    for column, dtype_kind in schema.columns.items():
        if column not in df.columns:
            continue
        check = _DTYPE_CHECKS[dtype_kind]
        if not check(df[column]):
            violations.append(
                f"{schema.name}.{column}: expected dtype kind '{dtype_kind}', "
                f"got '{df[column].dtype}'"
            )

    for column in schema.not_null:
        if column in df.columns and df[column].isna().any():
            violations.append(f"{schema.name}.{column}: contains null values")

    if schema.primary_key and set(schema.primary_key).issubset(df.columns):
        key_frame = df[list(schema.primary_key)]
        if key_frame.isna().any().any():
            violations.append(f"{schema.name}: primary key {schema.primary_key} contains nulls")
        duplicate_count = int(key_frame.duplicated().sum())
        if duplicate_count:
            violations.append(
                f"{schema.name}: primary key {schema.primary_key} has "
                f"{duplicate_count} duplicate row(s)"
            )

    return violations


def validate_referential_integrity(
    tables: dict[str, pd.DataFrame], schemas: dict[str, TableSchema]
) -> list[str]:
    """Check every foreign key in every schema against the referenced table."""
    violations: list[str] = []
    for name, schema in schemas.items():
        if name not in tables:
            continue
        df = tables[name]
        for column, (ref_table, ref_column) in schema.foreign_keys.items():
            if column not in df.columns or ref_table not in tables:
                continue
            valid_values = set(tables[ref_table][ref_column].dropna())
            observed_values = set(df[column].dropna())
            orphans = observed_values - valid_values
            if orphans:
                violations.append(
                    f"{name}.{column} -> {ref_table}.{ref_column}: "
                    f"{len(orphans)} value(s) have no matching parent row"
                )
    return violations


TABLE_SCHEMAS: dict[str, TableSchema] = {
    "customers": TableSchema(
        name="customers",
        columns={
            "customer_id": "int",
            "segment": "str",
            "signup_date": "datetime",
            "acquisition_channel": "str",
            "region": "str",
            "latent_value": "float",
            "price_sensitivity": "float",
            "household_income_proxy": "float",
        },
        primary_key=("customer_id",),
        not_null=(
            "customer_id",
            "segment",
            "signup_date",
            "acquisition_channel",
            "latent_value",
            "price_sensitivity",
        ),
    ),
    "products": TableSchema(
        name="products",
        columns={
            "product_id": "int",
            "category": "str",
            "monthly_price": "float",
            "margin_rate": "float",
            "launch_date": "datetime",
        },
        primary_key=("product_id",),
        not_null=("product_id", "category", "monthly_price", "margin_rate", "launch_date"),
    ),
    "marketing_campaigns": TableSchema(
        name="marketing_campaigns",
        columns={
            "campaign_id": "int",
            "campaign_name": "str",
            "start_date": "datetime",
        },
        primary_key=("campaign_id",),
        not_null=("campaign_id", "campaign_name", "start_date"),
    ),
    "customer_treatments": TableSchema(
        name="customer_treatments",
        columns={
            "customer_id": "int",
            "campaign_id": "int",
            "treated": "bool",
            "assignment_propensity": "float",
        },
        primary_key=("customer_id", "campaign_id"),
        foreign_keys={
            "customer_id": ("customers", "customer_id"),
            "campaign_id": ("marketing_campaigns", "campaign_id"),
        },
        not_null=("customer_id", "campaign_id", "treated", "assignment_propensity"),
    ),
    "customer_retention": TableSchema(
        name="customer_retention",
        columns={
            "customer_id": "int",
            "tenure_days": "int",
            "churn_date": "datetime",
            "churned": "bool",
            "retained_90d": "bool",
        },
        primary_key=("customer_id",),
        foreign_keys={"customer_id": ("customers", "customer_id")},
        not_null=("customer_id", "tenure_days", "churned", "retained_90d"),
    ),
    "subscriptions": TableSchema(
        name="subscriptions",
        columns={
            "subscription_id": "int",
            "customer_id": "int",
            "product_id": "int",
            "start_date": "datetime",
            "end_date": "datetime",
            "status": "str",
            "monthly_price": "float",
        },
        primary_key=("subscription_id",),
        foreign_keys={
            "customer_id": ("customers", "customer_id"),
            "product_id": ("products", "product_id"),
        },
        not_null=(
            "subscription_id",
            "customer_id",
            "product_id",
            "start_date",
            "status",
            "monthly_price",
        ),
    ),
    "transactions": TableSchema(
        name="transactions",
        columns={
            "transaction_id": "int",
            "customer_id": "int",
            "product_id": "int",
            "transaction_date": "datetime",
            "amount": "float",
            "margin": "float",
        },
        primary_key=("transaction_id",),
        foreign_keys={
            "customer_id": ("customers", "customer_id"),
            "product_id": ("products", "product_id"),
        },
        not_null=(
            "transaction_id",
            "customer_id",
            "product_id",
            "transaction_date",
            "amount",
            "margin",
        ),
    ),
    "experiment_assignments": TableSchema(
        name="experiment_assignments",
        columns={
            "customer_id": "int",
            "experiment_name": "str",
            "variant": "str",
            "assignment_date": "datetime",
        },
        primary_key=("customer_id", "experiment_name"),
        foreign_keys={"customer_id": ("customers", "customer_id")},
        not_null=("customer_id", "experiment_name", "variant", "assignment_date"),
    ),
    "experiment_outcomes": TableSchema(
        name="experiment_outcomes",
        columns={
            "customer_id": "int",
            "experiment_name": "str",
            "converted": "bool",
            "observation_date": "datetime",
        },
        primary_key=("customer_id", "experiment_name"),
        foreign_keys={"customer_id": ("customers", "customer_id")},
        not_null=("customer_id", "experiment_name", "converted", "observation_date"),
    ),
    "sessions": TableSchema(
        name="sessions",
        columns={
            "session_id": "int",
            "customer_id": "int",
            "session_timestamp": "datetime",
        },
        primary_key=("session_id",),
        foreign_keys={"customer_id": ("customers", "customer_id")},
        not_null=("session_id", "customer_id", "session_timestamp"),
    ),
    "recommendation_exposures": TableSchema(
        name="recommendation_exposures",
        columns={
            "exposure_id": "int",
            "session_id": "int",
            "customer_id": "int",
            "product_id": "int",
            "position": "int",
            "exposure_mechanism": "str",
        },
        primary_key=("exposure_id",),
        foreign_keys={
            "session_id": ("sessions", "session_id"),
            "customer_id": ("customers", "customer_id"),
            "product_id": ("products", "product_id"),
        },
        not_null=(
            "exposure_id",
            "session_id",
            "customer_id",
            "product_id",
            "position",
            "exposure_mechanism",
        ),
    ),
    "events": TableSchema(
        name="events",
        columns={
            "event_id": "int",
            "exposure_id": "int",
            "event_type": "str",
            "event_timestamp": "datetime",
            "recorded_at": "datetime",
        },
        primary_key=("event_id",),
        foreign_keys={"exposure_id": ("recommendation_exposures", "exposure_id")},
        not_null=("event_id", "exposure_id", "event_type", "event_timestamp", "recorded_at"),
    ),
    "optimisation_reference_campaigns": TableSchema(
        name="optimisation_reference_campaigns",
        columns={
            "campaign_name": "str",
            "cost": "float",
            "value": "float",
        },
        primary_key=("campaign_name",),
        not_null=("campaign_name", "cost", "value"),
    ),
}
