"""Unit tests for transaction generation."""

from __future__ import annotations

from decision_intelligence.common import seeded_generator
from decision_intelligence.data.customers import generate_customers
from decision_intelligence.data.marketing import generate_customer_treatments
from decision_intelligence.data.products import generate_products
from decision_intelligence.data.retention import generate_customer_retention
from decision_intelligence.data.subscriptions import generate_subscriptions
from decision_intelligence.data.transactions import attach_margin, generate_transactions


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
    transactions = generate_transactions(
        small_config, customers, subscriptions, seeded_generator(6)
    )
    transactions = attach_margin(transactions, products)
    return subscriptions, transactions


def test_transactions_are_non_negative_and_within_subscription_window(small_config) -> None:
    subscriptions, transactions = _build(small_config)
    assert (transactions["amount"] >= 0).all()
    assert (transactions["margin"] >= 0).all()

    merged = transactions.merge(
        subscriptions[["customer_id", "product_id", "start_date", "end_date"]],
        on=["customer_id", "product_id"],
    )
    assert (merged["transaction_date"] >= merged["start_date"]).all()
    ended = merged.dropna(subset=["end_date"])
    assert (ended["transaction_date"] <= ended["end_date"]).all()


def test_transaction_ids_are_unique(small_config) -> None:
    _, transactions = _build(small_config)
    assert transactions["transaction_id"].is_unique
