"""Unit tests for SyntheticDataConfig validation."""

from __future__ import annotations

import pytest

from decision_intelligence.data.config import SyntheticDataConfig


def test_defaults_are_valid() -> None:
    config = SyntheticDataConfig()
    assert config.n_customers > 0
    assert config.n_products > 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_customers": 0},
        {"n_products": -1},
        {"horizon_days": 0},
        {"experiment_assignment_probability": 0.0},
        {"experiment_assignment_probability": 1.0},
    ],
)
def test_invalid_configs_are_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SyntheticDataConfig(**overrides)
