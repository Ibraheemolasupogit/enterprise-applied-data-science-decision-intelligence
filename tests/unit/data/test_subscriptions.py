"""Unit tests for subscription generation."""

from __future__ import annotations

from decision_intelligence.common import seeded_generator
from decision_intelligence.data.customers import generate_customers
from decision_intelligence.data.marketing import generate_customer_treatments
from decision_intelligence.data.products import generate_products
from decision_intelligence.data.retention import generate_customer_retention
from decision_intelligence.data.subscriptions import generate_subscriptions


def _build(small_config):
    customers = generate_customers(small_config, seeded_generator(1))
    products = generate_products(small_config, seeded_generator(2))
    treatments = generate_customer_treatments(small_config, customers, seeded_generator(3))
    retention, _ = generate_customer_retention(
        small_config, customers, treatments, seeded_generator(4)
    )
    subscriptions = generate_subscriptions(
        small_config, customers, products, retention, seeded_generator(5)
    )
    return customers, products, retention, subscriptions


def test_one_subscription_per_customer(small_config) -> None:
    customers, _, _, subscriptions = _build(small_config)
    assert len(subscriptions) == len(customers)
    assert subscriptions["subscription_id"].is_unique
    assert set(subscriptions["customer_id"]) == set(customers["customer_id"])


def test_product_ids_are_valid(small_config) -> None:
    _, products, _, subscriptions = _build(small_config)
    assert set(subscriptions["product_id"]) <= set(products["product_id"])


def test_status_matches_retention_churn_flag(small_config) -> None:
    _, _, retention, subscriptions = _build(small_config)
    merged = subscriptions.merge(retention, on="customer_id")
    assert (merged.loc[merged["churned"], "status"] == "churned").all()
    assert (merged.loc[~merged["churned"], "status"] == "active").all()
