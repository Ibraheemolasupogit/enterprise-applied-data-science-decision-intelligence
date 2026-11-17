"""Small development scenarios (never the protocol's main, pilot or fresh seeds)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from decision_intelligence.data.customer_economics_scenarios import (
    EconomicsScenario,
    EconomicsScenarioConfig,
    generate_economics_scenario,
)

SMALL = replace(EconomicsScenarioConfig(seed=4242),
                acquisitions_per_month=(25.0, 30.0, 35.0, 15.0))


@pytest.fixture(scope="session")
def small() -> EconomicsScenario:
    return generate_economics_scenario(SMALL)
