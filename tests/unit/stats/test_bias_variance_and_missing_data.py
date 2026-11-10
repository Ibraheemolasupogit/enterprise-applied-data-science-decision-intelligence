"""Bias–variance decomposition and missing-data mechanisms/strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from decision_intelligence.common import seeded_generator
from decision_intelligence.statistics.bias_variance import (
    BiasVarianceRow,
    bias_variance_decomposition,
    polynomial_fitter,
    ridge_polynomial_fitter,
)
from decision_intelligence.statistics.missing_data import (
    compare_missing_data_strategies,
    impute,
    missingness_mask,
    missingness_summary,
)


def _true_function(x: np.ndarray) -> np.ndarray:
    return np.sin(2.5 * x)


@pytest.fixture(scope="module")
def polynomial_rows() -> dict[str, BiasVarianceRow]:
    rows = bias_variance_decomposition(
        _true_function,
        {f"degree_{d}": polynomial_fitter(d) for d in (1, 3, 10)},
        noise_sd=0.3,
        train_size=30,
        x_test=np.linspace(-0.9, 0.9, 25),
        n_replications=400,
        rng=seeded_generator(61),
    )
    return {row.model: row for row in rows}


def test_bias_variance_trade_off_across_polynomial_degree(
    polynomial_rows: dict[str, BiasVarianceRow],
) -> None:
    low, mid, high = (polynomial_rows[f"degree_{d}"] for d in (1, 3, 10))
    # The underfit model has by far the largest bias.
    assert low.bias_squared > 10 * mid.bias_squared
    # The overfit model has by far the largest variance. (Variance is *not* monotone in
    # complexity here: with a random training design the misspecified straight line also
    # varies with where the x's land, so degree 1 and degree 3 have similar variance.)
    assert high.variance > 5 * max(low.variance, mid.variance)
    # U-shaped test error: the middle model wins.
    assert mid.expected_test_mse < low.expected_test_mse
    assert mid.expected_test_mse < high.expected_test_mse
    # Training error is optimistic and falls monotonically with complexity.
    assert low.mean_training_mse > mid.mean_training_mse > high.mean_training_mse
    assert high.mean_training_mse < high.noise < high.expected_test_mse


def test_decomposition_identity_holds_within_monte_carlo_error(
    polynomial_rows: dict[str, BiasVarianceRow],
) -> None:
    for row in polynomial_rows.values():
        assert abs(row.decomposition_gap) < 0.05 * row.expected_test_mse + 0.005


def test_ridge_penalty_trades_variance_for_bias() -> None:
    rows = bias_variance_decomposition(
        _true_function,
        {
            "light": ridge_polynomial_fitter(10, 1e-4),
            "heavy": ridge_polynomial_fitter(10, 100.0),
        },
        noise_sd=0.3,
        train_size=30,
        x_test=np.linspace(-0.9, 0.9, 25),
        n_replications=200,
        rng=seeded_generator(62),
    )
    light, heavy = rows
    assert heavy.variance < light.variance
    assert heavy.bias_squared > light.bias_squared


def test_fitter_validation() -> None:
    with pytest.raises(ValueError, match="too small"):
        polynomial_fitter(5)(np.arange(3.0), np.arange(3.0))
    with pytest.raises(ValueError, match="non-negative"):
        ridge_polynomial_fitter(3, -1.0)


# --- missing data -----------------------------------------------------------


def test_masks_hit_target_rate_and_mechanism_direction() -> None:
    rng = seeded_generator(63)
    x = rng.normal(size=20000)
    y = 0.7 * x + rng.normal(size=20000)
    for mechanism in ("MCAR", "MAR", "MNAR"):
        mask = missingness_mask(y, mechanism, 0.3, rng, covariate=x)
        assert mask.mean() == pytest.approx(0.3, abs=0.015)
    mar = missingness_mask(y, "MAR", 0.3, rng, covariate=x)
    mnar = missingness_mask(y, "MNAR", 0.3, rng)
    assert x[mar].mean() > x[~mar].mean() + 0.5  # high-x values go missing under MAR
    assert y[mnar].mean() > y[~mnar].mean() + 0.5  # high-y values go missing under MNAR


def test_mask_validation() -> None:
    rng = seeded_generator(64)
    with pytest.raises(ValueError, match="covariate"):
        missingness_mask([1.0, 2.0, 3.0], "MAR", 0.2, rng)
    with pytest.raises(ValueError, match="mechanism"):
        missingness_mask([1.0, 2.0, 3.0], "random", 0.2, rng)
    with pytest.raises(ValueError, match="constant"):
        missingness_mask([1.0, 1.0, 1.0], "MNAR", 0.2, rng)


def test_impute_strategies() -> None:
    rng = seeded_generator(65)
    x = np.arange(10.0)
    y = 2 * x + 1
    mask = np.zeros(10, dtype=bool)
    mask[[2, 7]] = True
    observed = np.where(mask, np.nan, y)
    assert impute(observed, mask, "complete_case", rng).size == 8
    mean_filled = impute(observed, mask, "mean_imputation", rng)
    assert mean_filled[2] == pytest.approx(np.nanmean(observed))
    regression = impute(observed, mask, "regression_imputation", rng, covariate=x)
    np.testing.assert_allclose(regression[[2, 7]], y[[2, 7]])
    with pytest.raises(ValueError, match="covariate"):
        impute(observed, mask, "regression_imputation", rng)
    with pytest.raises(ValueError, match="strategy"):
        impute(observed, mask, "hot_deck", rng, covariate=x)


def test_missingness_summary() -> None:
    frame = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan], "b": [1, 2, 3, 4]})
    summary = missingness_summary(frame)
    assert summary.index[0] == "a"
    assert summary.loc["a", "missing_rate"] == 0.5 and summary.loc["b", "n_missing"] == 0


def test_strategy_comparison_reproduces_textbook_pattern() -> None:
    """MCAR: complete-case unbiased. MAR: complete-case biased, regression imputation
    repairs the bias. Mean imputation's naive SE is too small under every mechanism."""
    results = {
        (r.mechanism, r.strategy): r
        for mechanism in ("MCAR", "MAR")
        for r in compare_missing_data_strategies(
            mechanism=mechanism,
            missing_rate=0.4,
            sample_size=200,
            n_replications=300,
            rng=seeded_generator(66),
        )
    }
    mcar_cc = results[("MCAR", "complete_case")]
    assert abs(mcar_cc.bias) < 4 * mcar_cc.bias_standard_error
    mar_cc = results[("MAR", "complete_case")]
    assert mar_cc.bias < -8 * mar_cc.bias_standard_error
    mar_reg = results[("MAR", "regression_imputation")]
    assert abs(mar_reg.bias) < 4 * mar_reg.bias_standard_error
    for mechanism in ("MCAR", "MAR"):
        assert results[(mechanism, "mean_imputation")].naive_se_ratio < 0.85
        assert results[(mechanism, "complete_case")].naive_se_ratio == pytest.approx(1.0, abs=0.15)
