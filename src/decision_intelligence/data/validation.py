"""Full data-contract validation: schema, referential and temporal integrity.

:func:`validate_dataset` is the single entry point the pipeline (and every
test) should call — it is the one place that knows all three kinds of
contract a synthetic build must satisfy.
"""

from __future__ import annotations

import pandas as pd

from decision_intelligence.data.schemas import (
    TABLE_SCHEMAS,
    validate_referential_integrity,
    validate_table,
)


def validate_temporal_integrity(tables: dict[str, pd.DataFrame]) -> list[str]:
    """Check business-specific ordering rules that span one or two tables.

    Schema contracts (schemas.py) check columns, dtypes and keys. This
    checks the domain-specific "X must happen at or after Y" rules that
    make the dataset a believable timeline rather than merely well-typed.
    """
    violations: list[str] = []

    customers = tables.get("customers")
    subscriptions = tables.get("subscriptions")
    transactions = tables.get("transactions")
    sessions = tables.get("sessions")
    events = tables.get("events")
    exposures = tables.get("recommendation_exposures")
    retention = tables.get("customer_retention")
    assignments = tables.get("experiment_assignments")
    outcomes = tables.get("experiment_outcomes")

    if subscriptions is not None and customers is not None:
        merged = subscriptions.merge(
            customers[["customer_id", "signup_date"]], on="customer_id", how="left"
        )
        if (merged["start_date"] < merged["signup_date"]).any():
            violations.append("subscriptions: start_date before customer signup_date")
        ended = merged.dropna(subset=["end_date"])
        if (ended["end_date"] < ended["start_date"]).any():
            violations.append("subscriptions: end_date before start_date")

    if transactions is not None and subscriptions is not None:
        merged = transactions.merge(
            subscriptions[["customer_id", "product_id", "start_date", "end_date"]],
            on=["customer_id", "product_id"],
            how="left",
            suffixes=("", "_subscription"),
        )
        if (merged["transaction_date"] < merged["start_date"]).any():
            violations.append("transactions: transaction_date before subscription start_date")
        ended = merged.dropna(subset=["end_date"])
        if (ended["transaction_date"] > ended["end_date"]).any():
            violations.append("transactions: transaction_date after subscription end_date")

    if sessions is not None and customers is not None:
        merged = sessions.merge(
            customers[["customer_id", "signup_date"]], on="customer_id", how="left"
        )
        if (merged["session_timestamp"] < merged["signup_date"]).any():
            violations.append("sessions: session_timestamp before customer signup_date")

    if events is not None:
        if (events["recorded_at"] < events["event_timestamp"]).any():
            violations.append("events: recorded_at before event_timestamp")

    if events is not None and exposures is not None and sessions is not None:
        merged = events.merge(
            exposures[["exposure_id", "session_id"]], on="exposure_id", how="left"
        ).merge(
            sessions[["session_id", "session_timestamp"]], on="session_id", how="left"
        )
        if (merged["event_timestamp"] < merged["session_timestamp"]).any():
            violations.append("events: event_timestamp before its session_timestamp")

    if retention is not None and customers is not None:
        merged = retention.merge(
            customers[["customer_id", "signup_date"]], on="customer_id", how="left"
        )
        churned = merged.dropna(subset=["churn_date"])
        if (churned["churn_date"] < churned["signup_date"]).any():
            violations.append("customer_retention: churn_date before customer signup_date")

    if assignments is not None and outcomes is not None:
        merged = assignments.merge(
            outcomes, on=["customer_id", "experiment_name"], how="inner"
        )
        if (merged["observation_date"] < merged["assignment_date"]).any():
            violations.append(
                "experiment_outcomes: observation_date before "
                "experiment_assignments.assignment_date"
            )

    return violations


def validate_dataset(tables: dict[str, pd.DataFrame]) -> list[str]:
    """Run every schema, referential and temporal check over a full dataset build."""
    violations: list[str] = []
    for name, schema in TABLE_SCHEMAS.items():
        if name in tables:
            violations.extend(validate_table(tables[name], schema))
    violations.extend(validate_referential_integrity(tables, TABLE_SCHEMAS))
    violations.extend(validate_temporal_integrity(tables))
    return violations
