"""M11 P4 (gate carried from P2): L2-penalised logistic regression and standardisation.
At penalty 0 it reproduces M3's ``fit_logistic`` and statsmodels' GLM; the analytic gradient
and Hessian match finite differences; the penalty mask, weights and stationarity hold."""

from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm

from decision_intelligence.statistics.logistic_regression import fit_logistic
from decision_intelligence.statistics.penalised_regression import (
    Standardiser,
    fit_penalised_logistic,
    penalised_objective,
)


def _data(seed: int, n: int = 600) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    eta = 0.3 + X @ np.array([0.5, -1.0, 0.2])
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
    return np.column_stack([np.ones(n), X]), y


def test_penalty_zero_reproduces_m3_and_statsmodels() -> None:
    Xi, y = _data(4921)
    r = fit_penalised_logistic(Xi, y, penalty=0.0, gradient_tolerance=1e-12,
                               relative_change_tolerance=0.0)
    m3 = fit_logistic(Xi[:, 1:], y)
    glm = sm.GLM(y, Xi, family=sm.families.Binomial()).fit(tol=1e-14)
    np.testing.assert_allclose(r.coefficients, m3.coefficients, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(r.coefficients, glm.params, rtol=1e-9, atol=1e-12)
    assert r.log_likelihood == pytest.approx(m3.log_likelihood, rel=1e-12)
    np.testing.assert_allclose(np.sqrt(np.diag(r.covariance)), glm.bse, rtol=1e-7)
    assert r.converged and r.hessian_positive_definite


def test_gradient_and_hessian_match_finite_differences() -> None:
    Xi, y = _data(4922, 200)
    w = np.random.default_rng(1).uniform(0.5, 2.0, y.size)
    mask = np.array([False, True, True, True])
    beta = np.array([0.2, -0.4, 0.7, 0.1])
    f, g, h = penalised_objective(beta, Xi, y, w, mask, 3.0)
    for j in range(4):
        e = np.zeros(4)
        e[j] = 1e-6
        fp, gp, _ = penalised_objective(beta + e, Xi, y, w, mask, 3.0)
        fm, gm, _ = penalised_objective(beta - e, Xi, y, w, mask, 3.0)
        assert g[j] == pytest.approx((fp - fm) / 2e-6, rel=1e-6)
        np.testing.assert_allclose(h[:, j], (gp - gm) / 2e-6, rtol=1e-6, atol=1e-8)


def test_penalised_optimum_is_stationary_and_respects_the_mask() -> None:
    Xi, y = _data(4923)
    mask = np.array([False, True, True, True])
    r = fit_penalised_logistic(Xi, y, penalty=25.0, penalised=mask, gradient_tolerance=1e-12,
                               relative_change_tolerance=0.0)
    p = 1 / (1 + np.exp(-Xi @ r.coefficients))
    score = Xi.T @ (y - p)
    np.testing.assert_allclose(score[1:], 25.0 * r.coefficients[1:], atol=1e-8)
    assert abs(score[0]) < 1e-8  # the unpenalised intercept is a pure MLE condition
    big = fit_penalised_logistic(Xi, y, penalty=1e9, penalised=mask)
    assert np.all(np.abs(big.coefficients[1:]) < 1e-5)
    assert big.coefficients[0] == pytest.approx(np.log(y.mean() / (1 - y.mean())), abs=1e-4)


def test_frequency_weights_equal_replicated_rows_and_separation_is_finite() -> None:
    Xi, y = _data(4924, 80)
    w = np.random.default_rng(2).integers(1, 4, y.size)
    rep = np.repeat(np.arange(y.size), w)
    a = fit_penalised_logistic(Xi, y, penalty=2.0, weights=w)
    b = fit_penalised_logistic(Xi[rep], y[rep], penalty=2.0)
    np.testing.assert_allclose(a.coefficients, b.coefficients, rtol=1e-8, atol=1e-10)
    x = np.linspace(-2, 2, 40)
    sep = np.column_stack([np.ones(40), x])
    ys = (x > 0).astype(float)
    unpen = fit_penalised_logistic(sep, ys, penalty=0.0, max_iterations=50)
    assert unpen.separated and not unpen.converged  # a vanishing gradient is not convergence
    pen = fit_penalised_logistic(sep, ys, penalty=1.0, penalised=np.array([False, True]))
    assert pen.converged and not pen.separated and np.all(np.isfinite(pen.coefficients))
    # separation inside the unpenalised block still has no finite optimum
    block = fit_penalised_logistic(sep, ys, penalty=1.0, penalised=np.array([True, False]),
                                   max_iterations=50)
    assert block.separated and not block.converged
    with pytest.raises(ValueError):
        fit_penalised_logistic(sep, ys + 0.5, penalty=1.0)
    with pytest.raises(ValueError):
        fit_penalised_logistic(sep, ys, penalty=-1.0)


def test_standardiser_uses_fit_rows_and_drops_constant_columns() -> None:
    X = np.array([[1.0, 5.0, 2.0], [3.0, 5.0, 4.0], [5.0, 5.0, 9.0]])
    s = Standardiser.fit(X, ["a", "b", "c"])
    assert s.names == ("a", "c") and s.dropped == ("b",)
    z = s.transform(X, ["a", "b", "c"])
    np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(z.std(axis=0, ddof=0), 1.0)
    new = s.transform(np.array([[7.0, 0.0, 2.0]]), ["a", "b", "c"])
    assert new[0, 0] == pytest.approx((7.0 - 3.0) / np.std([1.0, 3.0, 5.0]))
