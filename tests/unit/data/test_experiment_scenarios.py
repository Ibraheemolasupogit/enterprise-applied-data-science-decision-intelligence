"""M6 experiment-population generator: determinism, truth invariants and defect switches."""

from __future__ import annotations

import numpy as np
import pytest

from decision_intelligence.data.experiment_scenarios import (
    ExperimentScenarioConfig,
    generate_experiment_population,
    realise_experiment,
)


def _alternating(n: int) -> np.ndarray:
    return np.arange(n) % 2


def test_population_is_deterministic() -> None:
    a = generate_experiment_population(ExperimentScenarioConfig(n_units=500))
    b = generate_experiment_population(ExperimentScenarioConfig(n_units=500))
    assert a.units.equals(b.units) and a.potential.equals(b.potential) and a.truth == b.truth
    c = generate_experiment_population(ExperimentScenarioConfig(n_units=500, seed=1))
    assert not a.potential.equals(c.potential)


def test_clean_truths_coincide_and_effects_are_exact() -> None:
    pop = generate_experiment_population()
    t = pop.truth
    assert t.itt_conversion == t.ate_conversion == t.cace_conversion
    assert t.complier_share == 1.0
    pot = pop.potential
    assert np.all(pot["y1"] >= pot["y0"])  # coupled potential outcomes: monotone effect
    for segment, effect in zip(("casual", "regular", "power"), (0.01, 0.02, 0.04), strict=True):
        assert t.ate_by_segment[segment] == pytest.approx(effect, abs=1e-12)


def test_noncompliance_truths() -> None:
    config = ExperimentScenarioConfig(n_units=4000, never_taker_share=0.3, always_taker_share=0.1)
    pop = generate_experiment_population(config)
    t = pop.truth
    assert t.complier_share == pytest.approx(0.6, abs=1e-3)
    pot = pop.potential
    compliers = (pot["d1"] == 1) & (pot["d0"] == 0)
    # ITT = sum over compliers of their effect / n  =  complier share x CACE.
    assert t.itt_conversion == pytest.approx(t.complier_share * t.cace_conversion, rel=1e-12)
    assert compliers.sum() == pytest.approx(0.6 * 4000, abs=2)


def test_realisation_is_deterministic_and_respects_assignment() -> None:
    pop = generate_experiment_population(ExperimentScenarioConfig(n_units=1000))
    z = _alternating(1000)
    first = realise_experiment(pop, z)
    second = realise_experiment(pop, z)
    assert first.analysis_frame.equals(second.analysis_frame)
    frame = first.analysis_frame
    treated = frame["arm"] == "treatment"
    assert np.array_equal(frame.loc[treated, "converted"], pop.potential.loc[treated, "y1"])
    assert np.array_equal(frame.loc[~treated, "converted"], pop.potential.loc[~treated, "y0"])
    assert "engagement" not in frame.columns and "p0" not in frame.columns
    with pytest.raises(ValueError):
        realise_experiment(pop, np.full(1000, 2))


def test_loss_affects_only_the_treatment_arm_and_log_keeps_everyone() -> None:
    for mechanism in ("random", "low_engagement", "inactive"):
        config = ExperimentScenarioConfig(n_units=4000, treatment_loss_share=0.1,
                                          loss_mechanism=mechanism)  # type: ignore[arg-type]
        realised = realise_experiment(generate_experiment_population(config), _alternating(4000))
        counts = realised.analysis_frame["arm"].value_counts()
        assert counts["control"] == 2000 and counts["treatment"] < 1900
        assert len(realised.assignment_log) == 4000
    inactive = ExperimentScenarioConfig(n_units=4000, treatment_loss_share=0.1,
                                        loss_mechanism="inactive")
    pop = generate_experiment_population(inactive)
    lost = pop.potential["lost_if_treated"]
    assert (pop.potential.loc[lost, "y1"] == 0).all()  # only non-converters are filtered


def test_attrition_is_monotone_and_differential() -> None:
    config = ExperimentScenarioConfig(n_units=4000, attrition_control=0.05,
                                      attrition_treatment=0.2, attrition_engagement_link=1.0)
    pop = generate_experiment_population(config)
    assert (pop.potential["r1"] <= pop.potential["r0"]).all()
    frame = realise_experiment(pop, _alternating(4000)).analysis_frame
    missing = frame["converted"].isna().groupby(frame["arm"]).mean()
    assert missing["treatment"] > missing["control"] + 0.1


def test_duplicates_and_clusters() -> None:
    config = ExperimentScenarioConfig(n_units=2000, duplicate_share=0.05, n_clusters=20,
                                      cluster_sd=0.5)
    pop = generate_experiment_population(config)
    realised = realise_experiment(pop, _alternating(2000))
    log = realised.assignment_log
    assert len(log) > 2000
    assert log["unit_id"].duplicated().sum() == pop.potential["duplicated"].sum()
    conflicting = log.groupby("unit_id")["arm"].nunique()
    assert (conflicting > 1).sum() == pop.potential["duplicate_conflicting"].sum()
    assert pop.units["cluster_id"].nunique() == 20
    assert np.array_equal(pop.randomisation_units, pop.units["cluster_id"].to_numpy())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_units": 10},
        {"segment_shares": (0.5, 0.5, 0.5)},
        {"baseline_conversion": (0.0, 0.1, 0.2)},
        {"never_taker_share": 0.6, "always_taker_share": 0.5},
        {"attrition_control": 0.3, "attrition_treatment": 0.1},
        {"n_clusters": 2},
        {"duplicate_share": 1.0},
    ],
)
def test_config_validation(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ExperimentScenarioConfig(**kwargs)  # type: ignore[arg-type]
