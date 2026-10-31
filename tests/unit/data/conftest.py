"""Shared fixtures for synthetic-data generator unit tests.

Sizes are kept small here purely for test speed — the pipeline's own
defaults (SyntheticDataConfig()) are the bounded-but-realistic sizes used
by the integration and statistical tests.
"""

from __future__ import annotations

import pytest

from decision_intelligence.common import seeded_generator
from decision_intelligence.data.config import SyntheticDataConfig


@pytest.fixture
def small_config() -> SyntheticDataConfig:
    return SyntheticDataConfig(seed=123, n_customers=200, n_products=10, horizon_days=180)


@pytest.fixture
def rng():
    return seeded_generator(123)
