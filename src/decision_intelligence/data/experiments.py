"""Unconfounded randomised experiment: NexaMarket's recommendation-feature test.

Unlike marketing.py, assignment here is an independent coin flip —
deliberately *not* a function of any customer covariate — so this is the
clean randomised-controlled contrast to the confounded observational
campaign. Baseline conversion still varies by segment and latent value
(real RCTs have heterogeneous baselines too); only the *assignment
mechanism* is unconfounded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from decision_intelligence.data.config import SyntheticDataConfig
from decision_intelligence.data.ground_truth import ExperimentGroundTruth

OBSERVATION_WINDOW_DAYS = 14
_ASSIGNMENT_HORIZON_FRACTION = 0.6

_SEGMENT_BASE_CONVERSION = {"casual": 0.08, "regular": 0.14, "power": 0.22}
_MIN_PROBABILITY = 0.01
_MAX_PROBABILITY = 0.95


def generate_experiment(
    config: SyntheticDataConfig, customers: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame, ExperimentGroundTruth]:
    """Generate assignments and outcomes for the randomised experiment.

    Eligible population: customers who had signed up at least
    ``OBSERVATION_WINDOW_DAYS`` before the experiment's fixed assignment
    date, so every assigned customer has a fully observed outcome window
    within the data horizon.
    """
    assignment_date = pd.Timestamp(config.start_date) + pd.Timedelta(
        int(config.horizon_days * _ASSIGNMENT_HORIZON_FRACTION), unit="D"
    )
    eligible_cutoff = assignment_date - pd.Timedelta(OBSERVATION_WINDOW_DAYS, unit="D")
    eligible = customers[customers["signup_date"] <= eligible_cutoff].reset_index(drop=True)

    n = len(eligible)
    is_treatment = rng.random(n) < config.experiment_assignment_probability
    variant = np.where(is_treatment, "treatment", "control")

    assignments = pd.DataFrame(
        {
            "customer_id": eligible["customer_id"].to_numpy(),
            "experiment_name": config.experiment_name,
            "variant": variant,
            "assignment_date": assignment_date,
        }
    )

    base_conversion = np.clip(
        eligible["segment"].map(_SEGMENT_BASE_CONVERSION).to_numpy()
        + 0.15 * (eligible["latent_value"].to_numpy() - 0.5),
        _MIN_PROBABILITY,
        _MAX_PROBABILITY,
    )
    true_conversion = np.clip(
        base_conversion + np.where(is_treatment, config.experiment_true_effect, 0.0),
        _MIN_PROBABILITY,
        _MAX_PROBABILITY,
    )
    converted = rng.random(n) < true_conversion

    outcomes = pd.DataFrame(
        {
            "customer_id": eligible["customer_id"].to_numpy(),
            "experiment_name": config.experiment_name,
            "converted": converted,
            "observation_date": assignment_date + pd.Timedelta(OBSERVATION_WINDOW_DAYS, unit="D"),
        }
    )

    ground_truth = ExperimentGroundTruth(
        experiment_name=config.experiment_name,
        assignment_probability=config.experiment_assignment_probability,
        true_absolute_effect=config.experiment_true_effect,
    )
    return assignments, outcomes, ground_truth
