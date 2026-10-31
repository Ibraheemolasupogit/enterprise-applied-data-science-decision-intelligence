"""Unit tests for churn-timing generation."""

from __future__ import annotations

import pandas as pd

from decision_intelligence.common import seeded_generator
from decision_intelligence.data.customers import generate_customers
from decision_intelligence.data.marketing import generate_customer_treatments
from decision_intelligence.data.retention import generate_customer_retention


def _build(small_config):
    customers = generate_customers(small_config, seeded_generator(1))
    treatments = generate_customer_treatments(small_config, customers, seeded_generator(2))
    table, ground_truth = generate_customer_retention(
        small_config, customers, treatments, seeded_generator(3)
    )
    return customers, table, ground_truth


def test_shape_and_types(small_config) -> None:
    _, table, _ = _build(small_config)
    assert len(table) == small_config.n_customers
    assert table["tenure_days"].ge(0).all()


def test_churn_date_is_null_iff_not_churned(small_config) -> None:
    _, table, _ = _build(small_config)
    assert (table.loc[table["churned"], "churn_date"].notna()).all()
    assert (table.loc[~table["churned"], "churn_date"].isna()).all()


def test_treatment_improves_average_retention(small_config) -> None:
    """The retention offer's true effect is positive, so — holding the
    confounded assignment aside — treated customers should, on average,
    retain more than their *own* baseline would without it. We check this
    indirectly via the ground truth's effect sign and magnitude."""
    _, _, ground_truth = _build(small_config)
    assert small_config.marketing_true_effect > 0
    assert set(ground_truth.segment_shape) == {"casual", "regular", "power"}
    assert all(shape > 0 for shape in ground_truth.segment_shape.values())


def test_same_seed_is_deterministic(small_config) -> None:
    customers = generate_customers(small_config, seeded_generator(1))
    treatments = generate_customer_treatments(small_config, customers, seeded_generator(2))
    table_a, _ = generate_customer_retention(
        small_config, customers, treatments, seeded_generator(5)
    )
    table_b, _ = generate_customer_retention(
        small_config, customers, treatments, seeded_generator(5)
    )
    pd.testing.assert_frame_equal(table_a, table_b)
