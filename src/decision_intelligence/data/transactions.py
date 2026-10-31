"""Monthly billing transactions with seasonality, trend and price sensitivity.

Transactions exist only while a subscription is active (between its
``start_date`` and ``end_date``/the data horizon), at roughly 30-day
intervals, so a customer's transaction history and their retention record
in retention.py are always mutually consistent. Amounts vary with calendar
seasonality and a mild time trend to give forecasting (M16) something
non-trivial to decompose.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from decision_intelligence.data.config import SyntheticDataConfig
from decision_intelligence.data.customers import horizon_end

_BILLING_PERIOD_DAYS = 30
# A mild holiday bump (Nov/Dec) and a winter dip, applied by calendar month.
_SEASONAL_MULTIPLIER = {
    1: 0.98,
    2: 0.97,
    3: 0.99,
    4: 1.00,
    5: 1.00,
    6: 1.01,
    7: 1.00,
    8: 0.99,
    9: 1.00,
    10: 1.02,
    11: 1.06,
    12: 1.08,
}
_TREND_PER_DAY = 0.0003


def generate_transactions(
    config: SyntheticDataConfig,
    customers: pd.DataFrame,
    subscriptions: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate monthly billing transactions for every active subscription."""
    customer_sensitivity = customers.set_index("customer_id")["price_sensitivity"]
    end_of_data = horizon_end(config)

    rows: list[dict[str, object]] = []
    for sub in subscriptions.itertuples(index=False):
        customer_id = int(sub.customer_id)  # type: ignore[arg-type]
        product_id = int(sub.product_id)  # type: ignore[arg-type]
        start_date = pd.Timestamp(sub.start_date)  # type: ignore[arg-type]
        monthly_price = float(sub.monthly_price)  # type: ignore[arg-type]
        end_date_value = sub.end_date
        active_end = (
            pd.Timestamp(end_date_value)  # type: ignore[arg-type]
            if pd.notna(end_date_value)  # type: ignore[arg-type]
            else end_of_data
        )

        n_periods = max(int((active_end - start_date).days // _BILLING_PERIOD_DAYS), 0)
        if n_periods == 0:
            continue

        period_index = np.arange(n_periods)
        billing_dates = start_date + pd.to_timedelta(
            period_index * _BILLING_PERIOD_DAYS, unit="D"
        )
        seasonal = np.array([_SEASONAL_MULTIPLIER[d.month] for d in billing_dates])
        day_offset = (billing_dates - pd.Timestamp(config.start_date)).days.to_numpy()
        trend = 1.0 + _TREND_PER_DAY * day_offset

        sensitivity = float(customer_sensitivity.loc[customer_id])
        noise = rng.normal(loc=1.0, scale=0.03 + 0.04 * sensitivity, size=n_periods)
        amount = np.clip(monthly_price * seasonal * trend * noise, 0.0, None)

        for i in range(n_periods):
            rows.append(
                {
                    "customer_id": customer_id,
                    "product_id": product_id,
                    "transaction_date": billing_dates[i],
                    "amount": round(float(amount[i]), 2),
                }
            )

    transactions = pd.DataFrame(rows)
    transactions.insert(0, "transaction_id", np.arange(1, len(transactions) + 1))
    return transactions


def attach_margin(transactions: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Attach ``margin`` using each product's margin rate; kept as a separate,
    explicit step so the margin computation is easy to audit independently
    of transaction-amount generation."""
    margin_rate = products.set_index("product_id")["margin_rate"]
    transactions = transactions.copy()
    transactions["margin"] = (
        transactions["amount"] * transactions["product_id"].map(margin_rate)
    ).round(2)
    return transactions
