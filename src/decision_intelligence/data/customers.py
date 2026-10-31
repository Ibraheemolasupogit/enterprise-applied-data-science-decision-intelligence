"""Synthetic customer population with realistic, segment-driven heterogeneity.

``latent_value`` is the one hidden driver behind this module: it is drawn
once per customer and then used — directly or with added noise — by every
downstream generator (retention, transactions, marketing treatment
assignment). That shared root is what makes the later marketing-campaign
confounding genuine rather than cosmetic: the covariate that predicts
treatment assignment is the *same* covariate that predicts the outcome.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from decision_intelligence.data.config import SyntheticDataConfig

SEGMENTS = ("casual", "regular", "power")
_SEGMENT_PROBABILITIES = (0.50, 0.35, 0.15)
_SEGMENT_LATENT_VALUE_MEAN = {"casual": 0.30, "regular": 0.55, "power": 0.80}
_SEGMENT_PRICE_SENSITIVITY_MEAN = {"casual": 0.70, "regular": 0.45, "power": 0.20}

_ACQUISITION_CHANNELS = ("organic", "paid_search", "social", "referral")
_ACQUISITION_PROBABILITIES = (0.40, 0.25, 0.20, 0.15)
_REGIONS = ("north", "south", "east", "west")


def generate_customers(config: SyntheticDataConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Generate the customer population for one synthetic build.

    Args:
        config: Build-wide configuration (size, horizon, seed-derived rng).
        rng: A generator dedicated to this stage (see pipeline.py), so this
            stage's draws are independent of every other stage's draws.
    """
    n = config.n_customers
    customer_id = np.arange(1, n + 1)

    segment = rng.choice(SEGMENTS, size=n, p=_SEGMENT_PROBABILITIES)

    latent_value_mean = np.array([_SEGMENT_LATENT_VALUE_MEAN[s] for s in segment])
    latent_value = np.clip(rng.normal(loc=latent_value_mean, scale=0.15, size=n), 0.0, 1.0)

    price_sensitivity_mean = np.array([_SEGMENT_PRICE_SENSITIVITY_MEAN[s] for s in segment])
    # Price sensitivity is anchored to segment, then nudged down slightly by an
    # individual's latent value: higher-value customers tend to be a little
    # less price sensitive even within their own segment.
    price_sensitivity = np.clip(
        rng.normal(loc=price_sensitivity_mean - 0.1 * latent_value, scale=0.12, size=n),
        0.0,
        1.0,
    )

    household_income_proxy = np.clip(
        rng.lognormal(mean=10.5 + 0.6 * latent_value, sigma=0.35, size=n), 10_000, None
    )

    signup_offset_days = rng.integers(0, config.horizon_days, size=n)
    signup_date = pd.to_datetime(config.start_date) + pd.to_timedelta(
        signup_offset_days, unit="D"
    )

    acquisition_channel = rng.choice(
        _ACQUISITION_CHANNELS, size=n, p=_ACQUISITION_PROBABILITIES
    )
    region = rng.choice(_REGIONS, size=n)

    return pd.DataFrame(
        {
            "customer_id": customer_id,
            "segment": segment,
            "signup_date": signup_date,
            "acquisition_channel": acquisition_channel,
            "region": region,
            "latent_value": latent_value,
            "price_sensitivity": price_sensitivity,
            "household_income_proxy": household_income_proxy,
        }
    )


def horizon_end(config: SyntheticDataConfig) -> pd.Timestamp:
    """The last calendar day covered by one synthetic build."""
    return pd.to_datetime(config.start_date) + timedelta(days=config.horizon_days)
