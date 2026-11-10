"""M3 methods applied to the M2 synthetic data: correctness, reproducibility, no leakage."""

from __future__ import annotations

import numpy as np
import pytest

from decision_intelligence.common import seeded_generator
from decision_intelligence.data import SyntheticDataConfig, build_synthetic_dataset
from decision_intelligence.data.dataset import SyntheticDataset
from decision_intelligence.statistics import applied_analyses as aa


@pytest.fixture(scope="module")
def dataset() -> SyntheticDataset:
    return build_synthetic_dataset(SyntheticDataConfig())


def test_experiment_analysis_is_internally_consistent(dataset: SyntheticDataset) -> None:
    result = aa.experiment_conversion_analysis(dataset, rng=seeded_generator(1))
    merged = dataset.tables["experiment_assignments"].merge(
        dataset.tables["experiment_outcomes"], on=["customer_id", "experiment_name"]
    )
    assert result.n_control + result.n_treatment == len(merged)
    assert result.test.estimate == pytest.approx(result.treatment_rate - result.control_rate)
    assert result.effect_sizes.risk_difference.estimate == pytest.approx(result.test.estimate)
    # Bootstrap and analytic intervals answer the same question; they should broadly agree.
    assert result.bootstrap_difference.ci_lower == pytest.approx(result.test.ci_lower, abs=0.01)
    assert result.bootstrap_difference.ci_upper == pytest.approx(result.test.ci_upper, abs=0.01)
    assert 0 < result.design_power < 1
    assert result.required_n_per_arm > 0


def test_experiment_interval_covers_known_effect(dataset: SyntheticDataset) -> None:
    """Ground truth is used only to *check* the estimate, never to produce it."""
    result = aa.experiment_conversion_analysis(dataset, rng=seeded_generator(1))
    assert result.true_effect == dataset.config.experiment_true_effect
    assert result.ci_covers_true_effect


def test_revenue_analysis_shows_skew_and_consistent_intervals(dataset: SyntheticDataset) -> None:
    result = aa.revenue_analysis(dataset, rng=seeded_generator(2))
    assert result.revenue_summary.n == dataset.config.n_customers
    assert result.revenue_summary.skewness > 1.0  # revenue is right-skewed
    assert result.revenue_summary.median < result.revenue_summary.mean
    t_low, t_high = result.mean_t_interval
    assert result.mean_bootstrap.ci_lower == pytest.approx(t_low, rel=0.05)
    assert result.mean_bootstrap.ci_upper == pytest.approx(t_high, rel=0.05)
    assert result.power_vs_casual_welch.estimate > 0
    assert result.power_vs_casual_cliffs_delta.estimate > 0


def test_retention_model_beats_base_rate_on_held_out_customers(
    dataset: SyntheticDataset,
) -> None:
    result = aa.retention_model_analysis(dataset, rng=seeded_generator(3))
    assert result.n_train + result.n_test == dataset.config.n_customers
    assert "latent_value" not in result.model.feature_names
    diagnostics = result.model_test_diagnostics
    assert diagnostics.n == result.n_test
    assert diagnostics.brier_score < result.baseline_test_brier
    assert diagnostics.log_loss < result.baseline_test_log_loss
    assert diagnostics.auc > 0.6


def test_retention_features_exclude_hidden_and_post_outcome_columns(
    dataset: SyntheticDataset,
) -> None:
    frame = aa.retention_feature_frame(dataset)
    forbidden = {"latent_value", "treated", "assignment_propensity", "tenure_days", "churned"}
    assert forbidden.isdisjoint(frame.columns)
    assert set(aa.RETENTION_FEATURES) <= set(frame.columns)


def test_clustered_transactions_inflate_standard_errors(dataset: SyntheticDataset) -> None:
    result = aa.clustered_transaction_analysis(dataset)
    assert result.n_transactions > result.n_customers
    ratios = np.array(result.cluster_se) / np.array(result.classical_se)
    assert np.all(ratios > 1.3)


def test_income_missingness_known_truth_check() -> None:
    result = aa.income_missingness_analysis(SyntheticDataConfig(), rng=seeded_generator(4))
    assert result.m2_missing_rate == pytest.approx(0.05, abs=0.02)
    # M2 injects MCAR missingness: complete-case mean is unbiased.
    assert abs(result.m2_complete_case_mean - result.full_mean_log_income) < (
        3 * result.m2_complete_case_se
    )
    # Imposed MAR: complete case is biased; regression imputation repairs most of it.
    cc_error = abs(result.mar_complete_case_mean - result.full_mean_log_income)
    imputation_error = abs(result.mar_regression_imputation_mean - result.full_mean_log_income)
    assert cc_error > 3 * result.m2_complete_case_se
    assert imputation_error < 0.5 * cc_error


def test_applied_analyses_are_reproducible(dataset: SyntheticDataset) -> None:
    first = aa.experiment_conversion_analysis(dataset, rng=seeded_generator(9))
    second = aa.experiment_conversion_analysis(dataset, rng=seeded_generator(9))
    assert first.bootstrap_difference.ci_lower == second.bootstrap_difference.ci_lower
    a = aa.retention_model_analysis(dataset, rng=seeded_generator(9))
    b = aa.retention_model_analysis(dataset, rng=seeded_generator(9))
    np.testing.assert_array_equal(a.model.coefficients, b.model.coefficients)
