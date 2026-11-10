"""M5 additive synthetic datasets: determinism, contracts and ground-truth invariants."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from decision_intelligence.data.rollout_panel import (
    RolloutConfig,
    effect_path,
    generate_rollout_panel,
    validate_rollout_panel,
)
from decision_intelligence.data.synthetic_control_panel import (
    SyntheticControlConfig,
    generate_synthetic_control_panel,
)
from decision_intelligence.data.threshold_assignment import (
    ThresholdConfig,
    baseline_spend,
    generate_threshold_data,
)


def test_rollout_is_deterministic_and_satisfies_contract() -> None:
    first, second = generate_rollout_panel(), generate_rollout_panel()
    pd.testing.assert_frame_equal(first.panel, second.panel)
    assert validate_rollout_panel(first.panel) == []
    counts = first.panel.groupby("cohort")["unit_id"].nunique().to_dict()
    assert counts == {0: 15, 9: 15, 13: 15, 17: 15}
    assert not generate_rollout_panel(RolloutConfig(seed=1)).panel.equals(first.panel)


def test_rollout_ground_truth_matches_effect_path() -> None:
    data = generate_rollout_panel()
    config, truth = data.config, data.ground_truth
    for k, (g, _) in enumerate(config.cohorts):
        for t in range(g, config.n_periods + 1):
            expected = effect_path(config, k, t - g)
            assert truth.att_by_cohort_period[(g, t)] == pytest.approx(expected)
    cells = truth.cell_effects.merge(data.panel, on=["unit_id", "period"])
    assert (cells.loc[cells["treated"] == 0, "true_direct_effect"] == 0).all()
    assert truth.overall_att == pytest.approx(
        cells.loc[cells["treated"] == 1, "true_direct_effect"].mean()
    )
    assert truth.parallel_trends_hold and truth.no_anticipation and truth.no_spillovers


def test_rollout_failure_switches() -> None:
    anticipating = generate_rollout_panel(RolloutConfig(anticipation_periods=2))
    cells = anticipating.ground_truth.cell_effects.merge(
        anticipating.panel, on=["unit_id", "period"]
    )
    pre = cells[(cells["cohort"] > 0) & (cells["period"] == cells["cohort"] - 1)]
    assert (pre["true_direct_effect"] > 0).all()  # effect before adoption
    spill = generate_rollout_panel(RolloutConfig(spillover_share=0.5))
    s_cells = spill.ground_truth.cell_effects.merge(spill.panel, on=["unit_id", "period"])
    assert (s_cells.loc[s_cells["cohort"] > 0, "true_spillover"] == 0).all()
    exposed = (s_cells["cohort"] == 0) & (s_cells["period"] >= 9)
    assert (s_cells.loc[exposed, "true_spillover"] > 0).all()
    trending = generate_rollout_panel(RolloutConfig(differential_trend=0.1))
    assert not trending.ground_truth.parallel_trends_hold


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cohort_effect_scales": (1.0,)},
        {"cohorts": ((1, 0.5),), "cohort_effect_scales": (1.0,)},
        {"cohorts": ((9, 0.6), (13, 0.6)), "cohort_effect_scales": (1.0, 1.0)},
        {"ar1_rho": 1.0},
        {"n_units": 3},
    ],
)
def test_rollout_config_validation(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RolloutConfig(**kwargs)  # type: ignore[arg-type]


def test_rollout_validator_catches_contract_breaks() -> None:
    panel = generate_rollout_panel().panel
    assert validate_rollout_panel(panel.drop(columns="cohort"))
    broken = panel.copy()
    broken.loc[broken.index[5], "treated"] = 1 - broken.loc[broken.index[5], "treated"]
    assert validate_rollout_panel(broken)
    assert validate_rollout_panel(panel.iloc[1:])


def test_synthetic_control_panel_structure_and_truth() -> None:
    data = generate_synthetic_control_panel()
    config = data.config
    assert data.panel["unit_id"].nunique() == config.n_donors + 1
    assert set(data.panel.loc[data.panel["treated_unit"] == 1, "unit_id"]) == {0}
    assert sum(data.ground_truth.true_donor_weights.values()) == pytest.approx(1.0)
    effects = data.ground_truth.effect_by_period
    assert min(effects) == config.n_pre + 1 and len(effects) == config.n_post
    assert effects[config.n_pre + 1] == pytest.approx(config.effect_size / config.effect_ramp)
    pd.testing.assert_frame_equal(data.panel, generate_synthetic_control_panel().panel)


def test_synthetic_control_noiseless_treated_is_exact_combination_before_treatment() -> None:
    data = generate_synthetic_control_panel(SyntheticControlConfig(noise_sd=1e-12))
    wide = data.panel.pivot(index="period", columns="unit_id", values="outcome")
    weights = pd.Series(data.ground_truth.true_donor_weights)
    combination = wide[weights.index].to_numpy() @ weights.to_numpy()
    pre = wide.index <= data.config.n_pre
    np.testing.assert_allclose(wide.loc[pre, 0].to_numpy(), combination[pre], atol=1e-8)


def test_contaminated_donors_receive_part_of_the_effect() -> None:
    clean = generate_synthetic_control_panel()
    dirty = generate_synthetic_control_panel(SyntheticControlConfig(contaminated_donors=2))
    ids = dirty.ground_truth.contaminated_donor_ids
    assert len(ids) == 2
    diff = dirty.panel["outcome"] - clean.panel["outcome"]
    affected = dirty.panel.loc[diff.abs() > 1e-9, "unit_id"].unique()
    assert set(affected) == set(ids)


def test_threshold_sharp_assignment_and_truth() -> None:
    data = generate_threshold_data()
    frame, config = data.frame, data.config
    assert ((frame["score"] >= config.cutoff) == (frame["tier"] == 1)).all()
    assert data.ground_truth.n_manipulated == 0
    # Cutoff above the median score: local effect differs from the population average.
    truth = data.ground_truth
    assert truth.population_ate < truth.tau_at_cutoff < truth.att
    assert baseline_spend(np.array([0.0]))[0] == pytest.approx(20.0)


def test_threshold_manipulation_and_compound_jump() -> None:
    manipulated = generate_threshold_data(ThresholdConfig(manipulation_share=0.6))
    c = manipulated.config.cutoff
    assert manipulated.ground_truth.n_manipulated > 0
    just_above = manipulated.frame["score"].between(c, c + 0.5).sum()
    baseline_above = generate_threshold_data().frame["score"].between(c, c + 0.5).sum()
    assert just_above > baseline_above + manipulated.ground_truth.n_manipulated * 0.9
    jump = generate_threshold_data(ThresholdConfig(compound_jump=2.0))
    base = generate_threshold_data()
    delta = jump.frame["spend"] - base.frame["spend"]
    np.testing.assert_allclose(delta[base.frame["tier"] == 1], 2.0)
    np.testing.assert_allclose(delta[base.frame["tier"] == 0], 0.0)


@pytest.mark.parametrize(
    "kwargs", [{"cutoff": 0.0}, {"n_customers": 10}, {"manipulation_share": 1.5}]
)
def test_threshold_config_validation(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ThresholdConfig(**kwargs)  # type: ignore[arg-type]
