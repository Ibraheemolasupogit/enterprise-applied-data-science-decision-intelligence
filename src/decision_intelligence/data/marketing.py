"""Confounded observational marketing-campaign assignment.

This is the deliberately *non-randomised* counterpart to experiments.py.
Treatment propensity is a logistic function of ``latent_value`` — the same
hidden covariate that independently drives retention in retention.py — so a
naive treated-vs-control comparison is expected to be biased. Recovering
``marketing_true_effect`` requires the adjustment methods introduced in
Pillar 1 (M4/M5); this table exists specifically to make that demonstration
possible instead of merely described.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from decision_intelligence.data.config import SyntheticDataConfig

CAMPAIGN_ID = 1
CONFOUNDING_COVARIATE = "latent_value"

# Calibrated so assignment propensity ranges roughly 10%-85% across the
# observed latent_value range, giving the eventual propensity model
# something real to estimate without any customer being near-deterministically
# treated or untreated (positivity holds by construction).
_PROPENSITY_INTERCEPT = -2.0
_PROPENSITY_SLOPE = 4.0


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def generate_marketing_campaigns(config: SyntheticDataConfig) -> pd.DataFrame:
    """Generate the (single, for M2) marketing campaign record."""
    start_date = pd.to_datetime(config.start_date) + pd.Timedelta(
        config.horizon_days // 2, unit="D"
    )
    return pd.DataFrame(
        {
            "campaign_id": [CAMPAIGN_ID],
            "campaign_name": [config.marketing_campaign_name],
            "start_date": [start_date],
        }
    )


def generate_customer_treatments(
    config: SyntheticDataConfig, customers: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Assign the retention offer with propensity confounded on ``latent_value``."""
    latent_value = customers["latent_value"].to_numpy()
    logit = _PROPENSITY_INTERCEPT + _PROPENSITY_SLOPE * latent_value
    assignment_propensity = _sigmoid(logit)
    treated = rng.random(len(customers)) < assignment_propensity

    return pd.DataFrame(
        {
            "customer_id": customers["customer_id"].to_numpy(),
            "campaign_id": np.full(len(customers), CAMPAIGN_ID),
            "treated": treated,
            "assignment_propensity": assignment_propensity,
        }
    )
