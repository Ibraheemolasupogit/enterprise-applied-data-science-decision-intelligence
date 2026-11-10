"""OLS estimates, every covariance estimator and diagnostics versus statsmodels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera

from decision_intelligence.common import seeded_generator
from decision_intelligence.statistics.linear_regression import fit_ols, ols_diagnostics


@pytest.fixture
def heteroskedastic_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = seeded_generator(41)
    X = rng.normal(size=(120, 3))
    # Error SD increases monotonically in x0 — the form Breusch–Pagan is built to detect.
    noise = rng.normal(size=120) * np.exp(0.6 * X[:, 0])
    y = 1.0 + X @ np.array([0.5, -1.0, 0.2]) + noise
    clusters = np.repeat(np.arange(30), 4)
    return X, y, clusters


@pytest.mark.parametrize("covariance", ["classical", "HC0", "HC1", "HC2", "HC3"])
def test_coefficients_and_standard_errors_match_statsmodels(
    heteroskedastic_data: tuple[np.ndarray, np.ndarray, np.ndarray], covariance: str
) -> None:
    X, y, _ = heteroskedastic_data
    ours = fit_ols(X, y, covariance=covariance)
    reference = sm.OLS(y, sm.add_constant(X)).fit(
        cov_type="nonrobust" if covariance == "classical" else covariance
    )
    np.testing.assert_allclose(ours.coefficients, reference.params, rtol=1e-10)
    np.testing.assert_allclose(ours.standard_errors, reference.bse, rtol=1e-10)


def test_classical_inference_and_fit_statistics_match_statsmodels(
    heteroskedastic_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    X, y, _ = heteroskedastic_data
    ours = fit_ols(X, y, confidence_level=0.9)
    reference = sm.OLS(y, sm.add_constant(X)).fit()
    np.testing.assert_allclose(ours.p_values, reference.pvalues, rtol=1e-8)
    np.testing.assert_allclose(
        np.column_stack([ours.ci_lower, ours.ci_upper]), reference.conf_int(0.1), rtol=1e-10
    )
    assert ours.r_squared == pytest.approx(reference.rsquared)
    assert ours.adj_r_squared == pytest.approx(reference.rsquared_adj)
    assert ours.f_statistic == pytest.approx(reference.fvalue)
    assert ours.f_p_value == pytest.approx(reference.f_pvalue)
    assert ours.residual_std_error == pytest.approx(np.sqrt(reference.scale))
    assert ours.condition_number == pytest.approx(reference.condition_number, rel=1e-6)


def test_cluster_robust_matches_statsmodels(
    heteroskedastic_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    X, y, clusters = heteroskedastic_data
    ours = fit_ols(X, y, covariance="cluster", clusters=clusters)
    reference = sm.OLS(y, sm.add_constant(X)).fit(
        cov_type="cluster", cov_kwds={"groups": clusters}
    )
    np.testing.assert_allclose(ours.standard_errors, reference.bse, rtol=1e-10)
    assert ours.n_clusters == 30 and ours.df_resid == 29
    assert not ours.warnings


def test_few_clusters_warns_and_cluster_arguments_are_validated(
    heteroskedastic_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    X, y, _ = heteroskedastic_data
    few = fit_ols(X, y, covariance="cluster", clusters=np.repeat(np.arange(6), 20))
    assert any("clusters" in w for w in few.warnings)
    with pytest.raises(ValueError, match="requires cluster labels"):
        fit_ols(X, y, covariance="cluster")
    with pytest.raises(ValueError, match="not 'cluster'"):
        fit_ols(X, y, clusters=np.zeros(120))
    with pytest.raises(ValueError, match="at least two clusters"):
        fit_ols(X, y, covariance="cluster", clusters=np.zeros(120))


def test_no_intercept_model_matches_statsmodels(
    heteroskedastic_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    X, y, _ = heteroskedastic_data
    ours = fit_ols(X, y, add_intercept=False)
    reference = sm.OLS(y, X).fit()
    np.testing.assert_allclose(ours.coefficients, reference.params, rtol=1e-10)
    assert ours.r_squared == pytest.approx(reference.rsquared)
    assert ours.adj_r_squared == pytest.approx(reference.rsquared_adj)


def test_dataframe_input_names_and_prediction() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [2.0, 1.0, 4.0, 3.0, 6.0]})
    y = 2 + 3 * frame["a"] - frame["b"]
    result = fit_ols(frame, y)
    assert result.feature_names == ("intercept", "a", "b")
    np.testing.assert_allclose(result.coefficients, [2.0, 3.0, -1.0], atol=1e-10)
    np.testing.assert_allclose(result.predict([[10.0, 1.0]]), [31.0])
    assert list(result.summary_frame().columns)[:2] == ["estimate", "std_error"]


def test_rank_deficient_and_underdetermined_designs_raise() -> None:
    x = np.arange(10.0)
    with pytest.raises(ValueError, match="rank-deficient"):
        fit_ols(np.column_stack([x, 2 * x]), x + 1)
    with pytest.raises(ValueError, match="more observations"):
        fit_ols(np.eye(3), [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="NaN"):
        fit_ols([[1.0], [np.nan], [3.0]], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="observations"):
        fit_ols(np.arange(5.0), np.arange(4.0))


def test_hc3_refuses_leverage_one_observation() -> None:
    X = np.column_stack([np.arange(6.0), [0, 0, 0, 0, 0, 1.0]])
    y = np.array([1.0, 2.0, 2.5, 4.0, 5.5, 9.0])
    with pytest.raises(ValueError, match="leverage 1"):
        fit_ols(X, y, covariance="HC3")
    assert fit_ols(X, y, covariance="HC1").standard_errors.size == 3


def test_near_collinearity_is_flagged() -> None:
    rng = seeded_generator(42)
    x = rng.normal(size=200)
    X = np.column_stack([x, x + 1e-9 * rng.normal(size=200)])
    result = fit_ols(X, x + rng.normal(size=200))
    assert any("condition number" in w for w in result.warnings)


def test_diagnostics_match_statsmodels(
    heteroskedastic_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    X, y, _ = heteroskedastic_data
    ours = ols_diagnostics(fit_ols(X, y))
    reference = sm.OLS(y, sm.add_constant(X)).fit()
    bp_stat, bp_p, _, _ = het_breuschpagan(reference.resid, reference.model.exog)
    assert ours.breusch_pagan_statistic == pytest.approx(bp_stat)
    assert ours.breusch_pagan_p_value == pytest.approx(bp_p)
    assert ours.breusch_pagan_p_value < 0.05  # the data really are heteroskedastic
    assert ours.durbin_watson == pytest.approx(durbin_watson(reference.resid))
    assert ours.jarque_bera_statistic == pytest.approx(jarque_bera(reference.resid)[0])
    influence = OLSInfluence(reference)
    np.testing.assert_allclose(ours.cooks_distance, influence.cooks_distance[0], rtol=1e-8)
    np.testing.assert_allclose(
        ours.studentized_residuals, influence.resid_studentized_internal, rtol=1e-8
    )
    for j, name in enumerate(("x0", "x1", "x2"), start=1):
        assert ours.vif[name] == pytest.approx(variance_inflation_factor(reference.model.exog, j))


def test_diagnostics_flag_an_influential_point() -> None:
    rng = seeded_generator(43)
    x = rng.normal(size=50)
    y = 2 * x + rng.normal(size=50)
    x[0], y[0] = 8.0, -20.0
    diagnostics = ols_diagnostics(fit_ols(x, y))
    assert 0 in diagnostics.influential_indices
    assert 0 in diagnostics.high_leverage_indices


@pytest.mark.parametrize("covariance", ["classical", "HC3"])
def test_exact_fit_and_constant_response_report_undefined_inference(covariance: str) -> None:
    """Regression test: residuals at rounding level used to yield p-values ~1e-120
    (and p = 0.03 for a slope on a constant response)."""
    x = np.arange(10.0)
    for y in (2 + 3 * x, np.full(10, 4.0)):
        result = fit_ols(x, y, covariance=covariance)
        assert np.all(np.isnan(result.standard_errors))
        assert np.all(np.isnan(result.p_values)) and result.f_statistic is None
        assert any("numerically zero" in w for w in result.warnings)
    np.testing.assert_allclose(fit_ols(x, 2 + 3 * x).coefficients, [2.0, 3.0], atol=1e-10)


def test_joint_f_test_skipped_when_cluster_covariance_is_singular() -> None:
    """Regression test: 3 clusters cannot identify a 4-restriction Wald test; F used
    to come out as a huge negative number."""
    rng = seeded_generator(44)
    X = rng.normal(size=(40, 4))
    result = fit_ols(
        X, X.sum(axis=1) + rng.normal(size=40), covariance="cluster",
        clusters=np.repeat([0, 1, 2], [14, 13, 13]),
    )
    assert result.f_statistic is None and result.f_p_value is None
    assert any("F-test not computed" in w for w in result.warnings)
