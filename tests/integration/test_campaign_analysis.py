"""M4 analysis of the M2 retention-offer campaign: structure, honesty, no leakage."""

from __future__ import annotations

import numpy as np
import pytest

from decision_intelligence.causal_inference import campaign as cp
from decision_intelligence.common import seeded_generator
from decision_intelligence.data import SyntheticDataConfig, build_synthetic_dataset
from decision_intelligence.data.dataset import SyntheticDataset


@pytest.fixture(scope="module")
def dataset() -> SyntheticDataset:
    return build_synthetic_dataset(SyntheticDataConfig())


@pytest.fixture(scope="module")
def analysis(dataset: SyntheticDataset) -> cp.CampaignAnalysis:
    return cp.analyse_campaign(dataset, rng=seeded_generator(1), n_bootstrap=200)


def test_billing_rule_reproduces_pipeline_transaction_counts(dataset: SyntheticDataset) -> None:
    frame = (
        dataset.tables["customers"]
        .merge(dataset.tables["customer_retention"], on="customer_id")
    )
    counts = (
        dataset.tables["transactions"].groupby("customer_id").size()
        .reindex(frame["customer_id"], fill_value=0).to_numpy()
    )
    np.testing.assert_array_equal(cp.billing_periods(frame, dataset.config), counts)


def test_replicate_generator_reuses_m2_code_exactly() -> None:
    """Same stage generators as the pipeline: a replicate's columns obey M2's contracts."""
    frame = cp.simulate_campaign_replicate(SyntheticDataConfig(n_customers=500), seed=3)
    assert len(frame) == 500
    assert frame["logged_propensity"].between(0.11, 0.89).all()
    assert set(np.unique(frame[cp.TREATMENT])) == {0.0, 1.0}
    again = cp.simulate_campaign_replicate(SyntheticDataConfig(n_customers=500), seed=3)
    assert frame.equals(again)


def test_strategy_validity_flags_and_direction_of_bias(analysis: cp.CampaignAnalysis) -> None:
    by_name = {s.strategy: s for s in analysis.strategies}
    naive = by_name["none (naive)"].estimates[0].estimate
    regime_a = by_name["regime A: latent_value + segment"]
    assert regime_a.dag_valid
    assert not by_name["regime B: observed proxies"].dag_valid
    assert "post-treatment" in by_name[
        "INVALID: regime A + post-treatment billing count"
    ].dag_explanation
    for estimate in regime_a.estimates:
        assert estimate.estimate < naive  # adjustment removes most of the confounding
        assert estimate.adjustment_set == ("latent_value", "segment_regular", "segment_power")


def test_ground_truth_attached_only_for_evaluation(analysis: cp.CampaignAnalysis) -> None:
    assert analysis.true_effect == pytest.approx(0.08)
    # Nothing used as an adjustment variable is evaluation-only knowledge.
    for strategy in analysis.strategies:
        for estimate in strategy.estimates:
            assert "logged_propensity" not in estimate.adjustment_set
            assert "true" not in " ".join(estimate.adjustment_set)


def test_diagnostics_on_m2_data(analysis: cp.CampaignAnalysis) -> None:
    assert analysis.overlap.adequate  # propensities stay within [0.05, 0.95]
    assert not analysis.weights.warnings
    assert analysis.balance["smd_raw"].abs().max() > 0.5  # latent_value strongly imbalanced
    assert analysis.balance["smd_weighted"].abs().max() < 0.1
    assert analysis.estimated_vs_logged_propensity_max_abs_diff < 0.15
    assert analysis.falsification.p_value > 0.05  # DAG implication not rejected


def test_sensitivity_and_negative_control_outputs(analysis: cp.CampaignAnalysis) -> None:
    assert analysis.regime_b_e_value.e_value_point > 1
    assert analysis.regime_b_risk_ratio.risk_ratio > 1
    assert set(analysis.subgroup_effects) == {"casual", "regular", "power"}
    assert analysis.subgroup_heterogeneity.df == 2
    naive_nc = analysis.negative_control[0]
    assert naive_nc.ci_lower > 0  # treated customers have higher income: confounding visible
    assert len(analysis.bias_grid) == 16


def test_analysis_is_reproducible(dataset: SyntheticDataset) -> None:
    first = cp.analyse_campaign(dataset, rng=seeded_generator(7), n_bootstrap=100)
    second = cp.analyse_campaign(dataset, rng=seeded_generator(7), n_bootstrap=100)
    assert [e.ci_lower for s in first.strategies for e in s.estimates] == [
        e.ci_lower for s in second.strategies for e in s.estimates
    ]


def test_confounder_benchmarks_use_prevalence_ratio_and_bounding_factor(
    analysis: cp.CampaignAnalysis,
) -> None:
    frame = cp.campaign_frame(build_synthetic_dataset(SyntheticDataConfig()))
    median_split = analysis.confounder_benchmarks[0]
    assert median_split.n_categories == 2
    high = (frame["latent_value"] > frame["latent_value"].median()).to_numpy()
    treated = frame[cp.TREATMENT].to_numpy() == 1
    prevalence_ratio = max(
        np.mean(high[treated]) / np.mean(high[~treated]),
        np.mean(~high[treated]) / np.mean(~high[~treated]),
    )
    assert median_split.rr_exposure_confounder == pytest.approx(prevalence_ratio)
    # Finer categories reveal a stronger confounder: the bound then exceeds the observed RR.
    finest = analysis.confounder_benchmarks[-1]
    assert finest.bounding_factor > analysis.regime_b_risk_ratio.risk_ratio
