"""L2-penalised Gamma regression with a log link (added in M11 P6).

Owned here (regression primitives live in :mod:`decision_intelligence.statistics`). For
``y > 0`` with ``mu = exp(X beta)`` and shape ``k`` (dispersion ``phi = 1 / k``),

``log f(y) = k log k - lnGamma(k) + (k - 1) log y - k log mu - k y / mu``.

**Objective.** ``minimise -sum_i w_i log f(y_i) + (lambda / 2) sum_{j in P} beta_j^2`` over
``beta`` and ``k`` jointly — the *total* weighted log-likelihood, as in
:mod:`.penalised_regression` (the shape is never penalised). At ``lambda = 0`` the coefficient
MLE does not depend on ``k`` and equals the GLM estimate (statsmodels' ``GLM(Gamma(log))``);
with a penalty, ``k`` sets the likelihood's weight against the penalty, so it is estimated
jointly rather than by a moment estimator.

**Algorithm.** Alternate (i) a damped Newton step in ``beta`` with the observed Hessian
``k X' diag(w y / mu) X + lambda diag(P)`` (positive definite for ``y > 0``) and (ii) the exact
1-D maximisation of the shape given ``beta`` (Newton on ``log k``). Converged when the largest
``beta``-gradient component per unit weight and the shape score per unit weight are at most
``gradient_tolerance`` and the last relative change of the objective is at most
``relative_change_tolerance``. The covariance returned is the inverse penalised Hessian in
``beta`` at the estimated shape.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import special

from decision_intelligence.common.validation import FloatArray


def gamma_objective(beta: FloatArray, log_k: float, X: FloatArray, y: FloatArray,
                    weights: FloatArray, penalised: np.ndarray, penalty: float
                    ) -> tuple[float, FloatArray, FloatArray, float]:
    """Objective ``f``, its gradient and Hessian in ``beta``, and its derivative in ``log k``."""
    k = math.exp(log_k)
    eta = X @ beta
    ratio = y * np.exp(-eta)
    loglik_terms = (k * math.log(k) - float(special.gammaln(k)) + (k - 1.0) * np.log(y)
                    - k * eta - k * ratio)
    mask = penalised.astype(np.float64)
    f = -float(np.sum(weights * loglik_terms)) + 0.5 * penalty * float(np.sum(mask * beta ** 2))
    grad = -(X.T @ (weights * k * (ratio - 1.0))) + penalty * mask * beta
    hess = (X * (weights * k * ratio)[:, None]).T @ X + penalty * np.diag(mask)
    dk = float(np.sum(weights * (math.log(k) + 1.0 - float(special.digamma(k)) + np.log(y)
                                 - eta - ratio)))
    return f, np.asarray(grad, dtype=np.float64), np.asarray(hess, dtype=np.float64), -dk * k


def _shape_mle(eta: FloatArray, y: FloatArray, weights: FloatArray, log_k: float) -> float:
    """Exact maximiser of the likelihood in ``log k`` given ``eta`` (Newton on ``log k``)."""
    tw = float(weights.sum())
    c = float(np.sum(weights * (np.log(y) - eta - y * np.exp(-eta)))) / tw  # <= -1
    lk = log_k
    for _ in range(100):
        k = math.exp(lk)
        score = math.log(k) + 1.0 - float(special.digamma(k)) + c
        d_score = 1.0 / k - float(special.polygamma(1, k))
        step = score / (d_score * k)  # d score / d log k = k d_score
        new = min(max(lk - step, -30.0), 30.0)
        if abs(new - lk) < 1e-14:
            lk = new
            break
        lk = new
    return lk


@dataclass(frozen=True)
class GammaRegressionResult:
    coefficients: FloatArray
    shape: float
    covariance: FloatArray
    log_likelihood: float
    objective: float
    penalty: float
    converged: bool
    n_iterations: int
    max_gradient_per_weight: float
    last_relative_change: float
    hessian_positive_definite: bool

    def mean(self, X: ArrayLike) -> FloatArray:
        return np.asarray(np.exp(np.asarray(X, dtype=np.float64) @ self.coefficients),
                          dtype=np.float64)


def fit_penalised_gamma(X: ArrayLike, y: ArrayLike, *, penalty: float,
                        penalised: ArrayLike | None = None, weights: ArrayLike | None = None,
                        max_iterations: int = 200, gradient_tolerance: float = 1e-6,
                        relative_change_tolerance: float = 1e-9) -> GammaRegressionResult:
    """Fit ``y > 0`` on the columns of ``X`` (include an intercept column; leave it out of
    ``penalised``)."""
    Xa = np.asarray(X, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if Xa.ndim != 2 or ya.shape != (Xa.shape[0],):
        raise ValueError("X must be 2-D with one row per element of y")
    if not np.all(np.isfinite(Xa)) or not np.all(ya > 0) or not np.all(np.isfinite(ya)):
        raise ValueError("X must be finite and y finite and strictly positive")
    if penalty < 0 or not math.isfinite(penalty):
        raise ValueError("penalty must be finite and non-negative")
    n, p = Xa.shape
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (n,) or np.any(w < 0):
        raise ValueError("weights must be non-negative, one per row")
    mask = np.ones(p, dtype=bool) if penalised is None else np.asarray(penalised, dtype=bool)
    tw = max(float(w.sum()), 1.0)
    beta = np.zeros(p)
    # start: intercept-like least squares on log y
    try:
        beta = np.asarray(np.linalg.lstsq(Xa, np.log(ya), rcond=None)[0], dtype=np.float64)
    except np.linalg.LinAlgError:
        pass
    lk = _shape_mle(Xa @ beta, ya, w, 0.0)
    f, g, h, gk = gamma_objective(beta, lk, Xa, ya, w, mask, penalty)
    rel = math.inf
    n_iter = 0
    for it in range(1, max_iterations + 1):
        n_iter = it
        if max(float(np.max(np.abs(g), initial=0.0)), abs(gk)) / tw <= gradient_tolerance \
                and rel <= relative_change_tolerance:
            break
        try:
            step = np.linalg.solve(h, g)
        except np.linalg.LinAlgError:
            break
        moved = False
        for _ in range(60):
            cand = beta - step
            fc, gc, hc, _gk = gamma_objective(cand, lk, Xa, ya, w, mask, penalty)
            if math.isfinite(fc) and fc <= f + 1e-12 * max(1.0, abs(f)):
                beta = cand
                moved = True
                break
            step = step / 2.0
        if not moved:
            break
        lk = _shape_mle(Xa @ beta, ya, w, lk)
        fn, g, h, gk = gamma_objective(beta, lk, Xa, ya, w, mask, penalty)
        rel = abs(f - fn) / max(1.0, abs(f))
        f = fn
    max_g = max(float(np.max(np.abs(g), initial=0.0)), abs(gk)) / tw
    try:
        eig = np.linalg.eigvalsh(h)
        pd_ok = bool(float(np.min(eig)) > 1e-12 * max(abs(float(np.max(eig))), 1e-300))
    except np.linalg.LinAlgError:
        pd_ok = False
    cov = np.linalg.inv(h) if pd_ok else np.full((p, p), np.nan)
    k = math.exp(lk)
    eta = Xa @ beta
    loglik = float(np.sum(w * (k * math.log(k) - float(special.gammaln(k))
                               + (k - 1.0) * np.log(ya) - k * eta - k * ya * np.exp(-eta))))
    return GammaRegressionResult(np.asarray(beta, dtype=np.float64), float(k),
                                 np.asarray(cov, dtype=np.float64), loglik, float(f),
                                 float(penalty),
                                 max_g <= gradient_tolerance and rel <= relative_change_tolerance,
                                 n_iter, max_g, float(rel), pd_ok)
