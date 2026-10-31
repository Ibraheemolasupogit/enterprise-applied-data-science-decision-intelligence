"""Unit tests for product catalogue generation."""

from __future__ import annotations

from decision_intelligence.common import seeded_generator
from decision_intelligence.data.products import (
    _CATEGORY_PRICE_RANGE,
    CATEGORIES,
    generate_products,
)


def test_shape_and_uniqueness(small_config) -> None:
    df = generate_products(small_config, seeded_generator(1))
    assert len(df) == small_config.n_products
    assert df["product_id"].is_unique


def test_prices_within_category_range(small_config) -> None:
    df = generate_products(small_config, seeded_generator(1))
    assert set(df["category"].unique()) <= set(CATEGORIES)
    for row in df.itertuples(index=False):
        low, high = _CATEGORY_PRICE_RANGE[row.category]
        assert low <= row.monthly_price <= high


def test_margin_rate_is_bounded(small_config) -> None:
    df = generate_products(small_config, seeded_generator(1))
    assert df["margin_rate"].between(0.2, 0.85).all()
