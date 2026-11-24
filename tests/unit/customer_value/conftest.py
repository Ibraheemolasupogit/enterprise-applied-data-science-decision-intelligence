"""Small development scenarios (never the protocol's main, pilot or fresh seeds)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from decision_intelligence.customer_value import p8_private
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


@pytest.fixture
def p8_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> p8_private.PrivateEvidence:
    root = tmp_path / "p8-private"
    for name in ("", "history", "history/m11-p8-attempts", "staging",
                 "staging/m11-p8-attempts", "staging/supervisor", "reviewed"):
        (root / name).mkdir(mode=0o700)
    monkeypatch.setattr(p8_private, "_ROOT", root)
    return p8_private.open_private_evidence(Path(__file__).resolve().parents[3])
