"""Logistic regression versus statsmodels; separation; classification diagnostics."""

from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm

from decision_intelligence.common import seeded_generator
from decision_intelligence.statistics.logistic_regression import (
    SeparationError,
    classification_diagnostics,
    fit_logistic,
    is_separated,
    roc_auc,
)


@pytest.fixture
def logistic_data() -> tuple[np.ndarray, np.ndarray]:
    rng = seeded_generator(51)
    X = rng.normal(size=(400, 3))
    probability = 1 / (1 + np.exp(-(0.3 + X @ np.array([1.0, -0.5, 0.2]))))
    return X, (rng.random(400) < probability).astype(float)


def test_matches_statsmodels_logit(logistic_data: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = logistic_data
    ours = fit_logistic(X, y)
    reference = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
    np.testing.assert_allclose(ours.coefficients, reference.params, rtol=1e-8)
    np.testing.assert_allclose(ours.standard_errors, reference.bse, rtol=1e-8)
    np.testing.assert_allclose(ours.p_values, reference.pvalues, rtol=1e-6)
    assert ours.log_likelihood == pytest.approx(reference.llf)
    assert ours.null_log_likelihood == pytest.approx(reference.llnull, rel=1e-8)
    assert ours.aic == pytest.approx(reference.aic)
    assert ours.mcfadden_r_squared == pytest.approx(reference.prsquared, rel=1e-8)
    assert ours.likelihood_ratio_p_value == pytest.approx(reference.llr_pvalue, rel=1e-6)


def test_robust_covariance_matches_statsmodels_hc0(
    logistic_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = logistic_data
    ours = fit_logistic(X, y, covariance="robust")
    reference = sm.Logit(y, sm.add_constant(X)).fit(disp=0, cov_type="HC0")
    np.testing.assert_allclose(ours.standard_errors, reference.bse, rtol=1e-7)


def test_recovers_true_coefficients_in_large_sample() -> None:
    rng = seeded_generator(52)
    X = rng.normal(size=(20000, 2))
    beta = np.array([-1.0, 0.8, -0.4])
    y = (rng.random(20000) < 1 / (1 + np.exp(-(beta[0] + X @ beta[1:])))).astype(float)
    result = fit_logistic(X, y)
    assert np.all(np.abs(result.coefficients - beta) < 3 * result.standard_errors)
    np.testing.assert_allclose(result.odds_ratios, np.exp(result.coefficients))


def test_prediction_and_extreme_linear_predictor_are_numerically_stable(
    logistic_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = logistic_data
    result = fit_logistic(X, y)
    np.testing.assert_allclose(result.predict_probability(X), result.fitted_probabilities)
    extreme = result.predict_probability(np.array([[1e4, 0.0, 0.0], [-1e4, 0.0, 0.0]]))
    assert np.all(np.isfinite(extreme)) and extreme[0] == pytest.approx(1.0)


def test_complete_separation_raises() -> None:
    x = np.arange(1.0, 9.0)
    with pytest.raises(SeparationError, match="separation"):
        fit_logistic(x, (x > 4).astype(float))


def test_quasi_complete_separation_raises() -> None:
    """Every x=1 case is a success: the MLE for that coefficient is +infinity."""
    x = np.array([0, 0, 0, 0, 1, 1, 1, 1.0])
    y = np.array([0, 1, 0, 1, 1, 1, 1, 1.0])
    with pytest.raises(SeparationError):
        fit_logistic(x, y)


@pytest.mark.parametrize(
    ("y", "message"),
    [([0, 1, 2, 1], "binary"), ([1, 1, 1, 1], "single class")],
)
def test_invalid_outcomes_raise(y: list[int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        fit_logistic(np.arange(4.0), y)


def test_roc_auc_matches_pairwise_definition() -> None:
    rng = seeded_generator(53)
    y = (rng.random(300) < 0.4).astype(float)
    scores = y + rng.normal(size=300)
    positives, negatives = scores[y == 1], scores[y == 0]
    pairwise = np.mean(
        (positives[:, None] > negatives[None, :]) + 0.5 * (positives[:, None] == negatives[None, :])
    )
    assert roc_auc(y, scores) == pytest.approx(pairwise)
    assert roc_auc([0, 0, 1, 1], [0.1, 0.1, 0.1, 0.1]) == 0.5
    with pytest.raises(ValueError, match="single class"):
        roc_auc([1, 1], [0.2, 0.3])


def test_calibration_diagnostics_distinguish_calibrated_from_miscalibrated() -> None:
    rng = seeded_generator(54)
    true_p = rng.uniform(0.05, 0.95, 20000)
    y = (rng.random(20000) < true_p).astype(float)
    calibrated = classification_diagnostics(y, true_p)
    overconfident = classification_diagnostics(
        y, 1 / (1 + np.exp(-3 * np.log(true_p / (1 - true_p))))
    )
    assert calibrated.calibration_slope == pytest.approx(1.0, abs=0.05)
    assert abs(calibrated.calibration_in_the_large) < 0.01
    assert calibrated.hosmer_lemeshow_p_value > 0.01
    assert overconfident.calibration_slope == pytest.approx(1 / 3, abs=0.05)
    assert overconfident.brier_score > calibrated.brier_score
    assert len(calibrated.calibration_table) == 10


def test_calibration_rejects_probabilities_at_bounds() -> None:
    with pytest.raises(ValueError, match="strictly inside"):
        classification_diagnostics([0, 1, 0], [0.0, 0.5, 0.5])


def test_rare_event_data_with_extreme_probabilities_is_not_flagged_as_separated() -> None:
    """Regression test: a former |eta| > 18 heuristic rejected this valid model
    (68 events in 200,000; true fitted probabilities reach ~5e-9)."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=200_000)
    y = (rng.random(200_000) < 1 / (1 + np.exp(-(-10 + 2 * x)))).astype(float)
    ours = fit_logistic(x, y)
    reference = sm.Logit(y, sm.add_constant(x)).fit(disp=0)
    np.testing.assert_allclose(ours.coefficients, reference.params, rtol=1e-8)
    np.testing.assert_allclose(ours.standard_errors, reference.bse, rtol=1e-6)
    assert ours.fitted_probabilities.min() < 1e-8


def test_is_separated_exact_criterion() -> None:
    with_intercept = np.column_stack([np.ones(8), np.arange(8.0)])
    assert is_separated(with_intercept, (np.arange(8.0) > 3.5).astype(float))  # complete
    dummy = np.column_stack([np.ones(8), [0, 0, 0, 0, 1, 1, 1, 1.0]])
    assert is_separated(dummy, np.array([0, 1, 0, 1, 1, 1, 1, 1.0]))  # quasi-complete
    assert not is_separated(dummy, np.array([0, 1, 0, 1, 1, 0, 1, 1.0]))  # overlap


def test_large_but_finite_mle_is_fitted_not_rejected() -> None:
    """Nearly separated (one overlapping pair): the MLE exists and must be returned."""
    x = np.array([1, 2, 3, 4, 4.5, 5, 6, 7.0])
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1.0])
    result = fit_logistic(x, y)
    reference = sm.Logit(y, sm.add_constant(x)).fit(disp=0)
    np.testing.assert_allclose(result.coefficients, reference.params, rtol=1e-6)
