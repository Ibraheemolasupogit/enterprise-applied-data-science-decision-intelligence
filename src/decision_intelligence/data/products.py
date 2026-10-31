"""Synthetic product catalogue for NexaMarket's subscription plans."""

from __future__ import annotations

import numpy as np
import pandas as pd

from decision_intelligence.data.config import SyntheticDataConfig

CATEGORIES = (
    "streaming_basic",
    "streaming_premium",
    "productivity",
    "learning",
    "fitness",
    "gaming",
)
_CATEGORY_PRICE_RANGE = {
    "streaming_basic": (4.99, 9.99),
    "streaming_premium": (12.99, 19.99),
    "productivity": (6.99, 14.99),
    "learning": (9.99, 24.99),
    "fitness": (7.99, 17.99),
    "gaming": (9.99, 29.99),
}


def generate_products(config: SyntheticDataConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Generate the product/plan catalogue for one synthetic build."""
    n = config.n_products
    product_id = np.arange(1, n + 1)
    category = rng.choice(CATEGORIES, size=n)

    low = np.array([_CATEGORY_PRICE_RANGE[c][0] for c in category])
    high = np.array([_CATEGORY_PRICE_RANGE[c][1] for c in category])
    monthly_price = np.round(rng.uniform(low, high), 2)

    margin_rate = np.clip(rng.normal(loc=0.55, scale=0.08, size=n), 0.2, 0.85)

    launch_offset_days = rng.integers(0, max(config.horizon_days // 2, 1), size=n)
    launch_date = pd.to_datetime(config.start_date) - pd.to_timedelta(
        launch_offset_days, unit="D"
    )

    return pd.DataFrame(
        {
            "product_id": product_id,
            "category": category,
            "monthly_price": monthly_price,
            "margin_rate": margin_rate,
            "launch_date": launch_date,
        }
    )
