"""Configuration for deterministic synthetic data generation.

Every synthetic dataset in this repository — for every pillar, and for the
NexaMarket flagship — is produced from one :class:`SyntheticDataConfig`.
Rebuilding with the same ``seed`` and the same other fields must reproduce
the dataset exactly; changing a field is the only sanctioned way to change
the data. See docs/methodology/synthetic-data.md for the full rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class DataQualityConfig:
    """Deterministic, configurable data-quality imperfections.

    Real analytics pipelines face missing fields, duplicated events and
    late-arriving data. These rates let later milestones test robustness to
    exactly those failure modes instead of only ever seeing clean data.
    """

    missing_income_rate: float = 0.05
    missing_region_rate: float = 0.02
    duplicate_event_rate: float = 0.01
    late_arrival_rate: float = 0.03
    late_arrival_max_days: int = 2


@dataclass(frozen=True)
class SyntheticDataConfig:
    """Top-level configuration for one deterministic synthetic dataset build.

    Defaults are deliberately small (``n_customers=2000``) so the full build
    runs in a few seconds locally and in CI, per the Milestone 2 requirement
    for bounded sample generation.
    """

    seed: int = 20260101
    n_customers: int = 2000
    n_products: int = 25
    start_date: date = date(2024, 1, 1)
    horizon_days: int = 365

    sessions_per_customer_mean: float = 6.0
    items_per_session_mean: float = 5.0
    n_latent_categories: int = 8

    experiment_name: str = "recommendation_feature_experiment"
    experiment_assignment_probability: float = 0.5
    experiment_true_effect: float = 0.04
    """Absolute lift in conversion probability caused by the treatment arm."""

    marketing_campaign_name: str = "retention_offer_q1"
    marketing_true_effect: float = 0.08
    """Absolute lift in 90-day retention probability truly caused by the offer."""

    quality: DataQualityConfig = field(default_factory=DataQualityConfig)

    def __post_init__(self) -> None:
        if self.n_customers <= 0:
            raise ValueError("n_customers must be positive")
        if self.n_products <= 0:
            raise ValueError("n_products must be positive")
        if self.horizon_days <= 0:
            raise ValueError("horizon_days must be positive")
        if not 0.0 < self.experiment_assignment_probability < 1.0:
            raise ValueError("experiment_assignment_probability must be in (0, 1)")
