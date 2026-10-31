"""Unit tests for customer generation."""

from __future__ import annotations

import pandas as pd

from decision_intelligence.common import seeded_generator
from decision_intelligence.data.customers import SEGMENTS, generate_customers


def test_shape_and_uniqueness(small_config) -> None:
    df = generate_customers(small_config, seeded_generator(1))
    assert len(df) == small_config.n_customers
    assert df["customer_id"].is_unique


def test_values_are_within_expected_ranges(small_config) -> None:
    df = generate_customers(small_config, seeded_generator(1))
    assert set(df["segment"].unique()) <= set(SEGMENTS)
    assert df["latent_value"].between(0.0, 1.0).all()
    assert df["price_sensitivity"].between(0.0, 1.0).all()
    assert (df["household_income_proxy"] > 0).all()
    assert df["signup_date"].min() >= pd.Timestamp(small_config.start_date)


def test_same_seed_is_deterministic(small_config) -> None:
    first = generate_customers(small_config, seeded_generator(99))
    second = generate_customers(small_config, seeded_generator(99))
    assert first.equals(second)


def test_different_seed_changes_output(small_config) -> None:
    first = generate_customers(small_config, seeded_generator(1))
    second = generate_customers(small_config, seeded_generator(2))
    assert not first["latent_value"].equals(second["latent_value"])
