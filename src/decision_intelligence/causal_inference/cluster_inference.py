"""Wild cluster bootstrap for one OLS coefficient (small-cluster inference).

Cluster-robust (CR1) standard errors are consistent as the number of
clusters ``G`` grows; with few clusters — and especially few *treated*
clusters — CR1 t-tests over-reject. The wild cluster restricted bootstrap
(WCR; Cameron, Gelbach & Miller 2008) is the standard remedy:

1. Impose the null ``beta_j = beta0``: regress ``y - beta0 x_j`` on the other
   regressors, keeping restricted fitted values and residuals ``u~``.
2. For each draw, flip residuals cluster by cluster:
   ``y* = fitted_r + beta0 x_j + v_g u~`` with ``v_g`` Rademacher (±1) or Webb's
   six-point weights (better when ``G`` is very small).
3. Re-estimate and form the CR1 t-statistic ``t* = (beta*_j - beta0) / se*``.
4. Two-sided p-value: share of ``|t*| >= |t_obs|`` (symmetric).

When ``2^G`` Rademacher sign patterns (``6^G`` for Webb) do not exceed the
requested number of draws, every pattern is enumerated and the bootstrap
distribution is exact. Confidence intervals invert the test over ``beta0``.

Limitations (MacKinnon & Webb 2017): with very few *treated* clusters the
WCR test is badly mis-sized. In this repository's evidence it never
rejects with one treated cluster of ten (size 0.000, i.e. no power at all)
and rejects 0.2–1.2% with two, while CR1 rejects 15.8% with two. The
result reports how many clusters carry variation in the tested regressor
(for a treatment dummy: the treated clusters) and warns when that is 3 or
fewer, or when the untreated side is that small.

Implementation: Frisch–Waugh–Lovell. With ``x~`` = ``x_j`` residualised on the
other regressors, ``beta*_j - beta0 = x~'(v ⊙ u~) / x~'x~`` and the full-model
residual is ``M_X (v ⊙ u~)``, so all draws are evaluated with matrix products.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from decision_intelligence.common.validation import (
    FloatArray,
    as_float_array,
    as_float_matrix,
    require_positive_int,
    require_probability,
)

WildWeights = Literal["rademacher", "webb"]
_WEBB = np.array(
    [-math.sqrt(1.5), -1.0, -math.sqrt(0.5), math.sqrt(0.5), 1.0, math.sqrt(1.5)]
)


@dataclass(frozen=True)
class WildBootstrapResult:
    coefficient: float
    t_statistic: float
    """CR1 t-statistic for H0: coefficient = beta0."""
    p_value: float
    beta0: float
    n_clusters: int
    n_draws: int
    enumerated: bool
    weights: WildWeights
    clusters_with_regressor: int = 0
    """Clusters in which the tested regressor is non-zero (treated clusters for a dummy)."""
    ci_lower: float | None = None
    ci_upper: float | None = None
    confidence_level: float | None = None
    warnings: tuple[str, ...] = ()


class _WildProblem:
    """Pre-computed pieces of one regression for repeated null-imposed bootstraps."""

    def __init__(
        self, design: FloatArray, y: FloatArray, clusters: np.ndarray, column: int
    ) -> None:
        n, k = design.shape
        self.y = y
        self.x = design[:, column]
        others = np.delete(design, column, axis=1)
        self.q_full, _ = np.linalg.qr(design)
        self.q_other, _ = np.linalg.qr(others)
        self.x_tilde = self.x - self.q_other @ (self.q_other.T @ self.x)
        self.xtx = float(self.x_tilde @ self.x_tilde)
        if self.xtx <= 1e-12 * max(1.0, float(self.x @ self.x)):
            raise ValueError("tested regressor is collinear with the others")
        _, self.codes = np.unique(clusters, return_inverse=True)
        self.n_clusters = int(np.max(self.codes)) + 1
        if self.n_clusters < 2:
            raise ValueError("need at least two clusters")
        self.correction = self.n_clusters / (self.n_clusters - 1) * (n - 1) / (n - k)

    def cluster_sums(self, values: FloatArray) -> FloatArray:
        out = np.zeros((self.n_clusters,) + values.shape[1:])
        np.add.at(out, self.codes, values)
        return out

    def observed(self, beta0: float) -> tuple[float, float]:
        beta = float(self.x_tilde @ self.y) / self.xtx
        residual = self.y - self.q_full @ (self.q_full.T @ self.y)
        scores = self.cluster_sums(self.x_tilde * residual)
        se = math.sqrt(self.correction * float(np.sum(scores**2))) / self.xtx
        return beta, (beta - beta0) / se

    def bootstrap_t(self, beta0: float, signs: FloatArray) -> FloatArray:
        restricted_y = self.y - beta0 * self.x
        restricted_residual = restricted_y - self.q_other @ (self.q_other.T @ restricted_y)
        draws = restricted_residual[:, None] * signs[self.codes, :]
        numerator = (self.x_tilde @ draws) / self.xtx
        residual = draws - self.q_full @ (self.q_full.T @ draws)
        scores = self.cluster_sums(self.x_tilde[:, None] * residual)
        se = np.sqrt(self.correction * np.sum(scores**2, axis=0)) / self.xtx
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.asarray(numerator / se, dtype=np.float64)


def _sign_matrix(
    n_clusters: int, n_draws: int, weights: WildWeights, rng: np.random.Generator
) -> tuple[FloatArray, bool]:
    support = np.array([-1.0, 1.0]) if weights == "rademacher" else _WEBB
    if support.size**n_clusters <= n_draws:
        patterns = np.array(list(itertools.product(support, repeat=n_clusters)), dtype=float)
        return patterns.T, True
    return support[rng.integers(0, support.size, size=(n_clusters, n_draws))], False


def wild_cluster_bootstrap(
    design: ArrayLike,
    y: ArrayLike,
    clusters: ArrayLike,
    column: int,
    *,
    rng: np.random.Generator,
    beta0: float = 0.0,
    n_draws: int = 9999,
    weights: WildWeights = "rademacher",
    confidence_level: float | None = 0.95,
) -> WildBootstrapResult:
    """WCR bootstrap test (and optionally CI) for coefficient ``column`` of ``design``.

    ``design`` must already contain every regressor, including the intercept
    and any fixed-effect dummies; ``clusters`` gives one label per row.
    """
    x = as_float_matrix(design, name="design")
    response = as_float_array(y, name="y")
    labels = np.asarray(clusters)
    if response.size != x.shape[0] or labels.shape != (x.shape[0],):
        raise ValueError("design, y and clusters must have the same number of rows")
    if not 0 <= column < x.shape[1]:
        raise ValueError("column out of range")
    if int(np.linalg.matrix_rank(x)) < x.shape[1]:
        raise ValueError("design matrix is rank-deficient")
    if weights not in ("rademacher", "webb"):
        raise ValueError("weights must be 'rademacher' or 'webb'")
    n_draws = require_positive_int(n_draws, "n_draws", minimum=99)
    problem = _WildProblem(x, response, labels, column)
    signs, enumerated = _sign_matrix(problem.n_clusters, n_draws, weights, rng)

    def p_value(null: float) -> tuple[float, float, float]:
        beta, t_obs = problem.observed(null)
        t_star = problem.bootstrap_t(null, signs)
        valid = np.isfinite(t_star)
        return float(np.mean(np.abs(t_star[valid]) >= abs(t_obs) - 1e-12)), beta, t_obs

    p, beta, t_obs = p_value(beta0)
    lower = upper = None
    if confidence_level is not None:
        confidence_level = require_probability(confidence_level, "confidence_level")
        alpha = 1 - confidence_level
        # Step size: the CR1 standard error (|beta - beta0| / |t| at the tested null).
        scale = abs(beta - beta0) / abs(t_obs) if t_obs != 0 else 1.0
        if not math.isfinite(scale) or scale == 0:
            scale = 1.0
        lower = _invert(lambda b: p_value(b)[0], beta, -scale, alpha)
        upper = _invert(lambda b: p_value(b)[0], beta, scale, alpha)
    nonzero = problem.cluster_sums(np.abs(x[:, column])[:, None])[:, 0] > 0
    with_regressor = int(nonzero.sum())
    warnings: list[str] = []
    smaller_side = min(with_regressor, problem.n_clusters - with_regressor)
    if 0 < smaller_side <= 3:
        warnings.append(
            f"only {smaller_side} cluster(s) on one side of the tested regressor: the wild "
            "cluster bootstrap is unreliable here (with 1 treated cluster it cannot reject; with "
            "2 it is extremely conservative) — no cluster-based method is trustworthy"
        )
    return WildBootstrapResult(
        coefficient=beta,
        t_statistic=t_obs,
        p_value=p,
        beta0=beta0,
        n_clusters=problem.n_clusters,
        n_draws=int(signs.shape[1]),
        enumerated=enumerated,
        weights=weights,
        clusters_with_regressor=with_regressor,
        ci_lower=lower,
        ci_upper=upper,
        confidence_level=confidence_level,
        warnings=tuple(warnings),
    )


def _invert(
    func: Callable[[float], float], estimate: float, step: float, alpha: float
) -> float:
    """Find the null value on one side of ``estimate`` where the p-value crosses ``alpha``.

    Expands the bracket geometrically, then bisects. The same bootstrap weight
    draws are used for every null value, so the p-value function is smooth
    enough in ``beta0`` for bisection.
    """
    inside, outside = estimate, estimate + step
    for _ in range(60):
        if func(outside) < alpha:
            break
        inside, outside = outside, outside + step
        step *= 2
    else:
        raise ValueError("confidence bound not reached: the bootstrap test never rejects")
    for _ in range(40):
        middle = (inside + outside) / 2
        if func(middle) < alpha:
            outside = middle
        else:
            inside = middle
    return (inside + outside) / 2


def cluster_robust_t_test(
    design: ArrayLike, y: ArrayLike, clusters: ArrayLike, column: int
) -> tuple[float, float, float]:
    """CR1 coefficient, SE and two-sided p-value with t(G-1) reference (FWL form; matches M3)."""
    x = as_float_matrix(design, name="design")
    problem = _WildProblem(x, as_float_array(y, name="y"), np.asarray(clusters), column)
    beta, t_obs = problem.observed(0.0)
    se = abs(beta / t_obs) if t_obs != 0 else math.inf
    p = float(2 * stats.t.sf(abs(t_obs), problem.n_clusters - 1))
    return beta, se, p


__all__ = [
    "WildBootstrapResult",
    "WildWeights",
    "cluster_robust_t_test",
    "wild_cluster_bootstrap",
]
