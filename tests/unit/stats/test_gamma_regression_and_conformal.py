"""M11 P6 gate (statistics primitives): the penalised Gamma log-link GLM matches statsmodels at
penalty 0 and is internally consistent; the split-conformal quantile has the exact
finite-sample rank and coverage."""

from __future__ import annotations

import math

import numpy as np
import pytest
import statsmodels.api as sm
from scipy import optimize, special

from decision_intelligence.statistics.conformal import (
    conformal_interval,
    conformal_quantile,
    conformal_rank,
)
from decision_intelligence.statistics.gamma_regression import (
    fit_penalised_gamma,
    gamma_objective,
)


def _data(seed: int, n: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=(n, 3))])
    mu = np.exp(X @ np.array([3.0, 0.4, -0.2, 0.0]))
    return X, rng.gamma(2.5, mu / 2.5)


def test_penalty_zero_matches_statsmodels_glm_gamma_log_link() -> None:
    X, y = _data(4971)
    r = fit_penalised_gamma(X, y, penalty=0.0, gradient_tolerance=1e-12,
                            relative_change_tolerance=1e-15)
    glm = sm.GLM(y, X, family=sm.families.Gamma(sm.families.links.Log())).fit(tol=1e-14)
    np.testing.assert_allclose(r.coefficients, glm.params, rtol=1e-9, atol=1e-12)
    assert float(np.sum(2 * (-np.log(y / r.mean(X)) + (y - r.mean(X)) / r.mean(X)))) == \
        pytest.approx(glm.deviance, rel=1e-10)
    # the shape is the exact profile MLE (an independent 1-D search)
    eta = X @ r.coefficients

    def neg(lk: float) -> float:
        k = math.exp(lk)
        return -float(np.sum(k * math.log(k) - special.gammaln(k) + (k - 1) * np.log(y)
                             - k * eta - k * y * np.exp(-eta)))

    res = optimize.minimize_scalar(neg, bounds=(-5, 5), method="bounded",
                                   options={"xatol": 1e-12})
    # a 1-D search on this flat log-likelihood (|f| ~ 1e4) is accurate to ~1e-6 only ...
    assert r.shape == pytest.approx(math.exp(res.x), rel=1e-6)
    # ... so the exact check is the first-order condition, written independently
    k = r.shape
    score = float(np.mean(math.log(k) + 1 - special.digamma(k) + np.log(y) - eta
                          - y * np.exp(-eta)))
    assert abs(score) < 1e-12


def test_gradient_and_hessian_match_finite_differences() -> None:
    X, y = _data(4972, 300)
    w = np.random.default_rng(1).uniform(0.5, 2.0, y.size)
    mask = np.array([False, True, True, True])
    beta, lk = np.array([2.9, 0.3, -0.1, 0.05]), math.log(2.0)
    _, g, h, gk = gamma_objective(beta, lk, X, y, w, mask, 4.0)
    for j in range(4):
        e = np.zeros(4)
        e[j] = 1e-6
        fp, gp, _, _ = gamma_objective(beta + e, lk, X, y, w, mask, 4.0)
        fm, gm, _, _ = gamma_objective(beta - e, lk, X, y, w, mask, 4.0)
        assert g[j] == pytest.approx((fp - fm) / 2e-6, rel=1e-6)
        np.testing.assert_allclose(h[:, j], (gp - gm) / 2e-6, rtol=1e-6, atol=1e-6)
    fp = gamma_objective(beta, lk + 1e-6, X, y, w, mask, 4.0)[0]
    fm = gamma_objective(beta, lk - 1e-6, X, y, w, mask, 4.0)[0]
    assert gk == pytest.approx((fp - fm) / 2e-6, rel=1e-6)


def test_penalised_fit_is_stationary_intercept_only_is_the_mean_and_weights_replicate(
        ) -> None:
    X, y = _data(4973, 800)
    mask = np.array([False, True, True, True])
    r = fit_penalised_gamma(X, y, penalty=30.0, penalised=mask, gradient_tolerance=1e-12,
                            relative_change_tolerance=1e-15)
    score = X.T @ (r.shape * (y / r.mean(X) - 1.0))
    np.testing.assert_allclose(score[1:], 30.0 * r.coefficients[1:], atol=1e-7)
    assert abs(score[0]) < 1e-7
    one = fit_penalised_gamma(np.ones((y.size, 1)), y, penalty=0.0)
    assert math.exp(one.coefficients[0]) == pytest.approx(float(y.mean()), rel=1e-9)
    wts = np.random.default_rng(3).integers(1, 4, y.size)
    rep = np.repeat(np.arange(y.size), wts)
    a = fit_penalised_gamma(X, y, penalty=5.0, penalised=mask, weights=wts)
    b = fit_penalised_gamma(X[rep], y[rep], penalty=5.0, penalised=mask)
    np.testing.assert_allclose(a.coefficients, b.coefficients, rtol=1e-7, atol=1e-9)
    assert a.shape == pytest.approx(b.shape, rel=1e-7)
    with pytest.raises(ValueError):
        fit_penalised_gamma(X, np.r_[y[:-1], 0.0], penalty=1.0)


def test_conformal_rank_quantile_and_interval() -> None:
    assert conformal_rank(99, 0.80) == 80 and conformal_rank(99, 0.95) == 95
    assert conformal_rank(9, 0.95) == 10
    s = np.arange(1.0, 100.0)  # 1 .. 99
    assert conformal_quantile(s, 0.80) == 80.0
    assert conformal_quantile(s[:9], 0.95) == math.inf  # rank 10 > n = 9: unbounded
    lo, hi = conformal_interval(np.array([10.0, 20.0]), s, 0.80)
    np.testing.assert_array_equal(lo, [-70.0, -60.0])
    np.testing.assert_array_equal(hi, [90.0, 100.0])
    with pytest.raises(ValueError):
        conformal_quantile(np.array([-1.0, 2.0]), 0.8)


@pytest.mark.parametrize("level", [0.80, 0.95])
def test_split_conformal_has_exact_finite_sample_coverage_under_exchangeability(level: float
                                                                                ) -> None:
    """Exchangeable (i.i.d.) data, any fixed predictor: P(cover) = ceil((n+1) level)/(n+1)
    exactly; checked over 20,000 replications within 4 Monte Carlo SE."""
    rng = np.random.default_rng(4974)
    n, reps = 49, 20_000
    x = rng.normal(size=(reps, n + 1))
    y = 2.0 * x + rng.standard_t(3, size=(reps, n + 1))  # heavy-tailed noise
    pred = 1.7 * x  # a deliberately imperfect, fixed predictor
    scores = np.abs(y - pred)
    covered = np.array([scores[i, -1] <= conformal_quantile(scores[i, :n], level)
                        for i in range(reps)])
    exact = conformal_rank(n, level) / (n + 1)
    se = math.sqrt(exact * (1 - exact) / reps)
    assert abs(float(covered.mean()) - exact) <= 4 * se
    assert exact >= level and exact <= level + 1 / (n + 1)
