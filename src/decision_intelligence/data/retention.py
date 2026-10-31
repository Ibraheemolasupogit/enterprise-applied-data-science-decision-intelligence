"""Time-to-churn generation with a known, individually-exact 90-day retention effect.

Design: each customer gets a *known* probability of surviving past 90 days,
``p_i`` — built from segment and ``latent_value`` for untreated customers,
and from the same baseline plus ``config.marketing_true_effect`` for
customers treated in marketing.py. A Weibull time-to-churn is then drawn so
that ``P(T_i > 90) == p_i`` exactly, which gives every downstream pillar two
compatible views of the same ground truth: a clean probability-space effect
(for M4/M5/M6-style mean-difference estimators) and genuine time-to-event
data with a per-segment hazard shape (for M11 survival analysis).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from decision_intelligence.data.config import SyntheticDataConfig
from decision_intelligence.data.customers import horizon_end
from decision_intelligence.data.ground_truth import RetentionGroundTruth

RETENTION_WINDOW_DAYS = 90

_SEGMENT_BASELINE_LOGIT_OFFSET = {"casual": -0.4, "regular": 0.0, "power": 0.5}
_SEGMENT_SHAPE = {"casual": 0.75, "regular": 0.85, "power": 0.95}
_BASELINE_INTERCEPT = -0.3
_LATENT_VALUE_SLOPE = 1.6
_MIN_PROBABILITY = 0.05
_MAX_PROBABILITY = 0.97


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def _baseline_retention_probability(customers: pd.DataFrame) -> np.ndarray:
    segment_offset = customers["segment"].map(_SEGMENT_BASELINE_LOGIT_OFFSET).to_numpy()
    logit = (
        _BASELINE_INTERCEPT
        + segment_offset
        + _LATENT_VALUE_SLOPE * customers["latent_value"].to_numpy()
    )
    return np.asarray(np.clip(_sigmoid(logit), _MIN_PROBABILITY, _MAX_PROBABILITY))


def generate_customer_retention(
    config: SyntheticDataConfig,
    customers: pd.DataFrame,
    treatments: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, RetentionGroundTruth]:
    """Generate per-customer churn timing and the retention ground truth.

    Args:
        customers: Output of :func:`customers.generate_customers`.
        treatments: Output of
            :func:`marketing.generate_customer_treatments`, aligned by
            ``customer_id`` with ``customers``.
        rng: A generator dedicated to this stage.
    """
    baseline_p = _baseline_retention_probability(customers)
    treated = (
        treatments.set_index("customer_id").loc[customers["customer_id"], "treated"].to_numpy()
    )
    true_p = np.clip(
        baseline_p + np.where(treated, config.marketing_true_effect, 0.0),
        _MIN_PROBABILITY,
        _MAX_PROBABILITY,
    )

    shape = customers["segment"].map(_SEGMENT_SHAPE).to_numpy()
    # Scale solved so that P(T > RETENTION_WINDOW_DAYS) == true_p exactly:
    # Weibull survival S(t) = exp(-(t/scale)^shape)  =>  scale = t / (-ln(p))^(1/shape).
    scale = RETENTION_WINDOW_DAYS / np.power(-np.log(true_p), 1.0 / shape)

    tenure_draw = rng.weibull(shape) * scale
    retained_90d = tenure_draw > RETENTION_WINDOW_DAYS

    remaining_horizon_days = (horizon_end(config) - customers["signup_date"]).dt.days.to_numpy()
    remaining_horizon_days = np.clip(remaining_horizon_days, 0, None)
    churned = tenure_draw <= remaining_horizon_days
    tenure_days = np.where(churned, tenure_draw, remaining_horizon_days).round().astype(int)
    churn_date = pd.to_datetime(customers["signup_date"]) + pd.to_timedelta(
        np.where(churned, tenure_draw, np.nan), unit="D"
    )

    table = pd.DataFrame(
        {
            "customer_id": customers["customer_id"].to_numpy(),
            "tenure_days": tenure_days,
            "churn_date": churn_date,
            "churned": churned,
            "retained_90d": retained_90d,
        }
    )

    representative_scale = {
        segment: float(
            RETENTION_WINDOW_DAYS
            / np.power(
                -np.log(np.clip(_sigmoid(_BASELINE_INTERCEPT + offset + _LATENT_VALUE_SLOPE * 0.5),
                                 _MIN_PROBABILITY, _MAX_PROBABILITY)),
                1.0 / _SEGMENT_SHAPE[segment],
            )
        )
        for segment, offset in _SEGMENT_BASELINE_LOGIT_OFFSET.items()
    }
    ground_truth = RetentionGroundTruth(
        segment_shape=dict(_SEGMENT_SHAPE),
        segment_scale=representative_scale,
    )
    return table, ground_truth
