"""L2-penalised logistic regression and column standardisation (added in M11 P4).

Owned here, per the shared-primitive rule (regression lives in :mod:`decision_intelligence.
statistics`); consumers such as ``customer_value`` import it. M3's
:func:`~.logistic_regression.fit_logistic` is unchanged: this module adds a penalty, frequency
weights and an explicit penalty mask, and at ``penalty = 0`` reproduces ``fit_logistic``.

**Objective.** ``minimise f(beta) = -sum_i w_i loglik_i(beta) + (lambda / 2) sum_{j in P}
beta_j^2`` — the *total* (not mean) weighted log-likelihood, so ``lambda`` does not depend on
``n``; ``P`` is the set of penalised columns (an intercept or baseline basis is left
unpenalised by excluding it from ``P``). ``loglik_i = y_i eta_i - log(1 + exp(eta_i))``,
``eta = X beta``.

**Algorithm.** Damped Newton: ``H = X' diag(w p (1 - p)) X + lambda diag(P)``, step
``H^{-1} grad f`` halved until ``f`` does not increase. Converged when the largest absolute
gradient component *per unit weight* is at most ``gradient_tolerance`` and the last relative
change of ``f`` is at most ``relative_change_tolerance``. The covariance returned is the
inverse penalised Hessian (the Bayesian posterior covariance under the matching Gaussian
prior); cluster-robust variances are the caller's responsibility.

**Standardisation** (:class:`Standardiser`): centre each column at its weighted mean and
divide by its weighted population SD (``ddof = 0``) computed on the rows of the fit; columns
with SD 0 are dropped and reported; new rows reuse the fitted statistics unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from decision_intelligence.common.validation import FloatArray
from decision_intelligence.statistics.logistic_regression import is_separated


def _weights(w: ArrayLike | None, n: int) -> FloatArray:
    if w is None:
        return np.ones(n)
    out = np.asarray(w, dtype=np.float64)
    if out.shape != (n,) or np.any(out < 0) or not np.all(np.isfinite(out)):
        raise ValueError("weights must be finite, non-negative and one per row")
    return out


def expit(eta: FloatArray) -> FloatArray:
    """Logistic function without overflow: ``exp(-log(1 + exp(-eta)))``."""
    return np.asarray(np.exp(-np.logaddexp(0.0, -np.asarray(eta, dtype=np.float64))),
                      dtype=np.float64)


def penalised_objective(beta: FloatArray, X: FloatArray, y: FloatArray, weights: FloatArray,
                        penalised: np.ndarray, penalty: float
                        ) -> tuple[float, FloatArray, FloatArray]:
    """``f``, its gradient and its Hessian at ``beta`` (see the module docstring)."""
    eta = X @ beta
    p = expit(eta)
    loglik = float(np.sum(weights * (y * eta - np.logaddexp(0.0, eta))))
    mask = penalised.astype(np.float64)
    f = -loglik + 0.5 * penalty * float(np.sum(mask * beta * beta))
    grad = -(X.T @ (weights * (y - p))) + penalty * mask * beta
    hess = (X * (weights * p * (1.0 - p))[:, None]).T @ X + penalty * np.diag(mask)
    return f, np.asarray(grad, dtype=np.float64), np.asarray(hess, dtype=np.float64)


@dataclass(frozen=True)
class PenalisedLogisticResult:
    coefficients: FloatArray
    covariance: FloatArray
    """Inverse penalised Hessian at the optimum (NaN if it is not positive definite)."""
    log_likelihood: float
    """Unpenalised weighted log-likelihood at the optimum."""
    objective: float
    penalty: float
    converged: bool
    n_iterations: int
    max_gradient_per_weight: float
    last_relative_change: float
    hessian_positive_definite: bool
    n_positive: float
    """Weighted number of ``y = 1`` rows."""
    separated: bool
    """No finite optimum: a separating direction exists within the unpenalised columns (all
    columns when ``penalty = 0``), by M3's exact linear-programme check on the distinct
    ``(row, y)`` pairs. A vanishing gradient is then not convergence."""


def unpenalised_separation(X: FloatArray, y: FloatArray, weights: FloatArray,
                           penalised: np.ndarray, penalty: float) -> bool:
    """Whether the penalised objective has no finite minimiser (see :attr:`separated`)."""
    free = ~penalised if penalty > 0 else np.ones(X.shape[1], dtype=bool)
    if not free.any():
        return False
    rows = weights > 0
    # Select both axes together so penalised columns never enter the row copy.
    pairs = np.unique(np.column_stack([X[np.ix_(rows, free)], y[rows]]), axis=0)
    return bool(is_separated(pairs[:, :-1], pairs[:, -1]))


def fit_penalised_logistic(X: ArrayLike, y: ArrayLike, *, penalty: float,
                           penalised: ArrayLike | None = None,
                           weights: ArrayLike | None = None,
                           start: ArrayLike | None = None, max_iterations: int = 200,
                           gradient_tolerance: float = 1e-6,
                           relative_change_tolerance: float = 1e-9
                           ) -> PenalisedLogisticResult:
    """Fit the L2-penalised logistic regression of ``y`` (0/1) on the columns of ``X``.

    ``X`` must already contain any intercept column; ``penalised`` is a boolean mask over
    columns (default: every column). With ``penalty = 0`` this is ordinary maximum
    likelihood; under separation no finite optimum exists, ``separated`` is True and
    ``converged`` False.
    """
    Xa = np.asarray(X, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if Xa.ndim != 2 or ya.shape != (Xa.shape[0],):
        raise ValueError("X must be 2-D with one row per element of y")
    if not np.isin(ya, (0.0, 1.0)).all():
        raise ValueError("y must be binary")
    if not np.all(np.isfinite(Xa)):
        raise ValueError("X must be finite")
    if penalty < 0 or not math.isfinite(penalty):
        raise ValueError("penalty must be finite and non-negative")
    n, k = Xa.shape
    w = _weights(weights, n)
    mask = (np.ones(k, dtype=bool) if penalised is None
            else np.asarray(penalised, dtype=bool))
    if mask.shape != (k,):
        raise ValueError("penalised must be a boolean mask over the columns of X")
    beta = np.zeros(k) if start is None else np.asarray(start, dtype=np.float64).copy()
    total_w = max(float(w.sum()), 1.0)
    f, g, h = penalised_objective(beta, Xa, ya, w, mask, penalty)
    rel = math.inf
    n_iter = 0
    for it in range(1, max_iterations + 1):
        n_iter = it
        if float(np.max(np.abs(g), initial=0.0)) / total_w <= gradient_tolerance and \
                rel <= relative_change_tolerance:
            break
        try:
            step = np.linalg.solve(h, g)
        except np.linalg.LinAlgError:
            break
        moved = False
        for _ in range(60):
            cand = beta - step
            fc, gc, hc = penalised_objective(cand, Xa, ya, w, mask, penalty)
            if math.isfinite(fc) and fc <= f + 1e-12 * max(1.0, abs(f)):
                rel = abs(f - fc) / max(1.0, abs(f))
                beta, f, g, h = cand, fc, gc, hc
                moved = True
                break
            step = step / 2.0
        if not moved:
            break
    max_g = float(np.max(np.abs(g), initial=0.0)) / total_w
    separated = unpenalised_separation(Xa, ya, w, mask, penalty)
    converged = (max_g <= gradient_tolerance and rel <= relative_change_tolerance
                 and not separated)
    try:
        eig = np.linalg.eigvalsh(h)
        pd_ok = bool(float(np.min(eig)) > 1e-12 * max(abs(float(np.max(eig))), 1e-300))
    except np.linalg.LinAlgError:
        pd_ok = False
    cov = np.linalg.inv(h) if pd_ok else np.full((k, k), np.nan)
    eta = Xa @ beta
    loglik = float(np.sum(w * (ya * eta - np.logaddexp(0.0, eta))))
    return PenalisedLogisticResult(np.asarray(beta, dtype=np.float64),
                                   np.asarray(cov, dtype=np.float64), loglik, float(f),
                                   float(penalty), bool(converged), n_iter, max_g, float(rel),
                                   pd_ok, float(np.sum(w * ya)), separated)


@dataclass(frozen=True)
class Standardiser:
    """Column centring and scaling fitted on one set of rows (weighted, ``ddof = 0``)."""

    names: tuple[str, ...]
    """Kept column names, in order."""
    mean: FloatArray
    scale: FloatArray
    dropped: tuple[str, ...]
    """Columns with SD 0 on the fitting rows (dropped)."""

    @classmethod
    def fit(cls, X: ArrayLike, names: tuple[str, ...] | list[str],
            weights: ArrayLike | None = None) -> Standardiser:
        Xa = np.asarray(X, dtype=np.float64)
        if Xa.ndim != 2 or Xa.shape[1] != len(names):
            raise ValueError("X must be 2-D with one name per column")
        w = _weights(weights, Xa.shape[0])
        tw = float(w.sum())
        if tw <= 0:
            raise ValueError("need positive total weight")
        mean = (w[:, None] * Xa).sum(axis=0) / tw
        sd = np.sqrt((w[:, None] * (Xa - mean) ** 2).sum(axis=0) / tw)
        keep = sd > 1e-12 * np.maximum(1.0, np.abs(mean))
        return cls(tuple(n for n, k in zip(names, keep, strict=True) if k),
                   np.asarray(mean[keep], dtype=np.float64),
                   np.asarray(sd[keep], dtype=np.float64),
                   tuple(n for n, k in zip(names, keep, strict=True) if not k))

    def transform(self, X: ArrayLike, names: tuple[str, ...] | list[str]) -> FloatArray:
        """Standardise the kept columns of ``X`` (columns located by name)."""
        Xa = np.asarray(X, dtype=np.float64)
        idx = [list(names).index(n) for n in self.names]
        return np.asarray((Xa[:, idx] - self.mean) / self.scale, dtype=np.float64)
