"""M6 case studies end to end: the M2 experiment and the guardrailed scenario."""

from __future__ import annotations

import numpy as np
import pytest

from decision_intelligence.common.seeding import seeded_generator
from decision_intelligence.data.config import SyntheticDataConfig
from decision_intelligence.data.experiment_scenarios import generate_experiment_population
from decision_intelligence.data.pipeline import build_synthetic_dataset
from decision_intelligence.experimentation import cases as c


@pytest.fixture(scope="module")
def m2() -> c.M2ExperimentAnalysis:
    return c.analyse_m2_experiment(build_synthetic_dataset(SyntheticDataConfig()),
                                   rng=seeded_generator(20260601))


def test_m2_experiment_is_valid_but_inconclusive(m2: c.M2ExperimentAnalysis) -> None:
    statuses = {check.check: check.status for check in m2.checks}
    assert statuses["sample ratio mismatch"] == "PASS"
    assert statuses["duplicate units"] == "PASS"
    assert "FAIL" not in statuses.values()
    primary = m2.primary
    assert primary.n_control + primary.n_treatment == 1135
    assert primary.p_value > 0.05 and primary.effect.ci_lower < 0 < primary.effect.ci_upper
    assert m2.recommendation.decision == "HOLD"
    assert "not evidence of no effect" in " ".join(m2.recommendation.reasons)
    # Evaluation-only: the interval covers the true effect; the design was underpowered for it.
    assert primary.effect.covers(m2.true_effect)
    assert m2.plan_for_true_effect.n_control > primary.n_control
    assert m2.mde_at_realised_n > m2.true_effect


def test_m2_secondary_metrics_and_m3_consistency(m2: c.M2ExperimentAnalysis) -> None:
    assert [s.metric for s in m2.secondary] == ["revenue_14d", "sessions_14d"]
    for result in m2.secondary:
        assert result.effect.covers(m2.true_secondary_effects[result.metric])
        assert m2.secondary_holm[result.metric] >= result.p_value
    # Same z-test as the M3 applied analysis of this experiment (p = 0.109 there).
    assert m2.primary.p_value == pytest.approx(0.109, abs=0.001)


def test_scenario_clean_run_launches_and_defective_run_holds() -> None:
    clean, defective = c.analyse_scenario_case()
    assert clean.recommendation.decision == "LAUNCH"
    assert all(g.status == "PASS" for g in clean.guardrails)
    assert {ch.status for ch in clean.checks} == {"PASS"}
    failed = {ch.check for ch in defective.checks if ch.status == "FAIL"}
    assert failed == {"duplicate units", "sample ratio mismatch"}
    assert defective.recommendation.decision == "HOLD"
    profile = defective.lost_unit_profile
    assert profile is not None
    missing = profile.loc[("treatment", "missing"), "pre_sessions"]
    logged = profile.loc[("treatment", "logged"), "pre_sessions"]
    assert missing < logged  # diagnosis aid: the lost units were less engaged beforehand
    assert clean.primary.effect.covers(clean.true_effect)


def test_case_studies_are_reproducible(m2: c.M2ExperimentAnalysis) -> None:
    again = c.analyse_m2_experiment(build_synthetic_dataset(SyntheticDataConfig()),
                                    rng=seeded_generator(20260601))
    assert again.primary == m2.primary and again.secondary == m2.secondary
    first, _ = c.analyse_scenario_case()
    second, _ = c.analyse_scenario_case()
    assert first.primary == second.primary and first.recommendation == second.recommendation


def test_m2_foundation_is_unaffected_by_m6_generator() -> None:
    first = build_synthetic_dataset(SyntheticDataConfig())
    generate_experiment_population()
    second = build_synthetic_dataset(SyntheticDataConfig())
    for name, table in first.tables.items():
        assert table.equals(second.tables[name]), name
    assert np.array_equal(first.tables["experiment_assignments"]["customer_id"],
                          second.tables["experiment_assignments"]["customer_id"])
