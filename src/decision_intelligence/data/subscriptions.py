"""Subscription plan choice and lifecycle, linked to the retention process.

Plan choice is price-sensitivity-weighted (more price-sensitive customers
are more likely to pick a cheaper plan), and the subscription's lifecycle
(``start_date``/``end_date``/``status``) is driven entirely by the churn
timing already decided in retention.py, so the two tables never disagree
about when a customer left.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from decision_intelligence.data.config import SyntheticDataConfig


def _choose_products(
    customers: pd.DataFrame, products: pd.DataFrame, rng: np.random.Generator
) -> np.ndarray:
    prices = products["monthly_price"].to_numpy()
    # Softmax over negative price, scaled by each customer's price sensitivity:
    # a highly price-sensitive customer concentrates probability mass on
    # cheaper plans; an insensitive customer is closer to a uniform choice.
    sensitivity = customers["price_sensitivity"].to_numpy().reshape(-1, 1)
    scores = -sensitivity * prices.reshape(1, -1)
    scores -= scores.max(axis=1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=1, keepdims=True)

    chosen_index = np.array(
        [rng.choice(len(products), p=weights[i]) for i in range(len(customers))]
    )
    return np.asarray(products["product_id"].to_numpy()[chosen_index])


def generate_subscriptions(
    config: SyntheticDataConfig,
    customers: pd.DataFrame,
    products: pd.DataFrame,
    retention: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate one subscription per customer, aligned to its churn outcome."""
    product_id = _choose_products(customers, products, rng)
    monthly_price = (
        products.set_index("product_id").loc[product_id, "monthly_price"].to_numpy()
    )

    retention_by_customer = retention.set_index("customer_id")
    churned = retention_by_customer.loc[customers["customer_id"], "churned"].to_numpy()
    churn_date = retention_by_customer.loc[customers["customer_id"], "churn_date"].to_numpy()

    end_date = np.where(churned, churn_date, np.datetime64("NaT", "ns"))
    status = np.where(churned, "churned", "active")

    return pd.DataFrame(
        {
            "subscription_id": np.arange(1, len(customers) + 1),
            "customer_id": customers["customer_id"].to_numpy(),
            "product_id": product_id,
            "start_date": customers["signup_date"].to_numpy(),
            "end_date": end_date,
            "status": status,
            "monthly_price": monthly_price,
        }
    )
