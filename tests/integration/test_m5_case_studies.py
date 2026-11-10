"""M5 case studies on the canonical additive datasets: structure, honesty, reproducibility."""

from __future__ import annotations

import math

import numpy as np
import pytest

from decision_intelligence.causal_inference import observational_cases as oc
from decision_intelligence.common import seeded_generator
from decision_intelligence.data import SyntheticDataConfig, build_synthetic_dataset
from decision_intelligence.data.rollout_panel import generate_rollout_panel
from decision_intelligence.data.synthetic_control_panel import generate_synthetic_control_panel
from decision_intelligence.data.threshold_assignment import generate_threshold_data


@pytest.fixture(scope="module")
def rollout() -> oc.RolloutAnalysis:
    return oc.analyse_rollout(generate_rollout_panel(), rng=seeded_generator(1))


def test_rollout_analysis(rollout: oc.RolloutAnalysis) -> None:
    assert rollout.staggered.overall.covers(rollout.true_overall_att)
    assert rollout.twfe.estimate == pytest.approx(rollout.twfe_implied_estimand, abs=1.0)
    assert rollout.twfe_negative_weight_share > 0
    assert rollout.twfe.estimate < rollout.true_overall_att
    assert rollout.twfe_wild_bootstrap.p_value < 0.05
    assert rollout.twfe_wild_bootstrap.ci_lower is not None
    assert any("Sun & Abraham" in w for w in rollout.event_study_twfe.warnings)
    assert rollout.staggered.pretrend is not None
    assert "F(" in rollout.staggered.pretrend.reference


def test_synthetic_control_analysis() -> None:
    analysis = oc.analyse_synthetic_control(generate_synthetic_control_panel())
    assert not analysis.fit.warnings
    assert abs(analysis.fit.average_post_gap - analysis.true_average_post_effect) < 1.0
    assert analysis.placebo_all.rank == 1
    assert analysis.placebo_all.permutation_p_value == pytest.approx(1 / 31)
    assert abs(analysis.in_time_placebo.average_post_gap) < 2.0
    assert len(analysis.leave_one_out) == len(analysis.fit.weights)


def test_threshold_analysis_reports_local_estimand() -> None:
    analysis = oc.analyse_threshold(generate_threshold_data())
    # No single-draw coverage assertion: the canonical seed happens to be a ~2.9-SE draw
    # (an independent rdrobust fit agrees), reported as such in the evidence; calibration
    # is established by the repeated-sampling tests.
    assert analysis.robust.ci_lower < analysis.robust.estimate < analysis.robust.ci_upper
    # The design makes the local effect differ from population-wide averages.
    assert analysis.true_tau_at_cutoff - analysis.true_population_ate > 0.5
    assert analysis.true_population_ate < analysis.true_tau_at_cutoff < analysis.true_att
    assert analysis.robust.estimand.startswith("tau(c)")
    assert any("not a population ATE" in a for a in analysis.robust.assumptions)
    assert analysis.mccrary.p_value > 0.05
    assert set(analysis.global_polynomials) == {1, 2, 4, 6}
    assert len(analysis.bandwidth_sensitivity) == 5 and math.isfinite(analysis.bandwidth)


def test_analyses_are_reproducible() -> None:
    a = oc.analyse_rollout(generate_rollout_panel(), rng=seeded_generator(9))
    b = oc.analyse_rollout(generate_rollout_panel(), rng=seeded_generator(9))
    assert a.staggered.overall.ci_lower == b.staggered.overall.ci_lower
    assert a.twfe_wild_bootstrap.p_value == b.twfe_wild_bootstrap.p_value
    c = oc.analyse_threshold(generate_threshold_data())
    d = oc.analyse_threshold(generate_threshold_data())
    assert c.bandwidth == d.bandwidth and c.robust.estimate == d.robust.estimate


def test_m2_foundation_is_unaffected_by_m5_generators() -> None:
    """The additive generators do not touch the M2 pipeline: rebuilding is byte-identical."""
    first = build_synthetic_dataset(SyntheticDataConfig())
    generate_rollout_panel()
    generate_synthetic_control_panel()
    generate_threshold_data()
    second = build_synthetic_dataset(SyntheticDataConfig())
    for name, table in first.tables.items():
        assert table.equals(second.tables[name]), name
    assert np.array_equal(
        first.tables["customers"]["customer_id"], second.tables["customers"]["customer_id"]
    )
