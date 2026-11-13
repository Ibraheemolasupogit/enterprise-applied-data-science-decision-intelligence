"""Implicit-feedback matrix factorisation: implicit ALS and BPR (Milestone 9).

**Implicit ALS** (Hu, Koren & Volinsky 2008). With click counts ``r_ui``, preference
``p_ui = 1{r_ui > 0}`` and confidence ``c_ui = 1 + alpha r_ui``, minimise over user factors
``X`` and item factors ``Y``::

    sum_{u,i} c_ui (p_ui - x_u'y_i)^2 + reg (|X|_F^2 + |Y|_F^2)

over *all* user-item pairs (unclicked pairs have p = 0, c = 1). Each half-step is an exact
weighted ridge solve per user (per item), using ``Y'C_uY = Y'Y + Y'(C_u - I)Y`` so the cost is
linear in the number of clicks. The objective never increases (tested). Users with no clicks
get ``x_u = 0``; :meth:`ImplicitALS.fold_in` computes factors for a new click vector with the
item factors fixed (the exact user half-step).

**BPR** (Rendle et al. 2009). Maximise ``sum ln sigma(x_uij) - (reg/2)|theta|^2`` over
triples (user, clicked item i, unclicked item j), ``x_uij = u.(v_i - v_j) + b_i - b_j``, by
seeded mini-batch stochastic gradient ascent. Negatives are uniform over items the user has not
clicked (rejection sampling; any residual collision after 10 redraws is counted and reported)
or, for the exposure-aware variant, uniform over items the logging policy *showed* the user
before the cutoff without a click — negatives the user demonstrably saw. Each step applies
exactly ``learning_rate x`` the gradient of the per-triple objective (tested against finite
differences).

Both operate on scipy sparse matrices (so they scale) and are wrapped as
:class:`~decision_intelligence.recommendations.baselines.Recommender` for the M8 protocol.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from decision_intelligence.common.validation import FloatArray
from decision_intelligence.data.point_in_time import as_of
from decision_intelligence.data.recommendation_scenarios import RecommendationLogs
from decision_intelligence.recommendations.baselines import Recommender
from decision_intelligence.recommendations.interactions import (
    IntArray,
    InteractionData,
    deduplicate_impressions,
    index_of,
)


def count_matrix(history: InteractionData) -> sparse.csr_matrix:
    """Click counts per (user, item) at the history cutoff."""
    ev = history.click_events
    data = np.ones(len(ev), dtype=np.float64)
    m = sparse.coo_matrix((data, (ev["user_index"].to_numpy(), ev["item_index"].to_numpy())),
                          shape=(history.n_users, history.n_items))
    return m.tocsr()


# ---------------------------------------------------------------------------
# Implicit ALS
# ---------------------------------------------------------------------------


def _als_half_step(counts: sparse.csr_matrix, fixed: FloatArray, reg: float,
                   alpha: float) -> FloatArray:
    """Exact solve of every row's weighted ridge problem given the other side's factors."""
    d = fixed.shape[1]
    gram = fixed.T @ fixed
    out = np.zeros((counts.shape[0], d), dtype=np.float64)
    eye = reg * np.eye(d)
    indptr, indices, data = counts.indptr, counts.indices, counts.data
    for row in range(counts.shape[0]):
        lo, hi = indptr[row], indptr[row + 1]
        if lo == hi:
            continue
        idx = indices[lo:hi]
        conf = 1.0 + alpha * data[lo:hi]
        f = fixed[idx]
        a = gram + (f.T * (conf - 1.0)) @ f + eye
        out[row] = np.linalg.solve(a, f.T @ conf)
    return out


def als_objective(counts: sparse.csr_matrix, x: FloatArray, y: FloatArray, reg: float,
                  alpha: float) -> float:
    """The full ALS objective over all pairs (dense; for small problems and tests)."""
    r = counts.toarray()
    p = (r > 0).astype(np.float64)
    c = 1.0 + alpha * r
    return float(np.sum(c * (p - x @ y.T) ** 2) + reg * (np.sum(x ** 2) + np.sum(y ** 2)))


@dataclass(frozen=True)
class ALSFactors:
    user: FloatArray
    item: FloatArray
    objective_trace: tuple[float, ...]


def fit_als(counts: sparse.csr_matrix, *, factors: int, reg: float, alpha: float,
            iterations: int, seed: int = 0, track_objective: bool = False) -> ALSFactors:
    if factors < 1 or reg <= 0 or alpha < 0 or iterations < 1:
        raise ValueError("need factors >= 1, reg > 0, alpha >= 0, iterations >= 1")
    if counts.nnz and counts.data.min() < 0:
        raise ValueError("counts must be non-negative")
    rng = np.random.default_rng(seed)
    n_u, n_i = counts.shape
    y = rng.normal(0.0, 0.01, size=(n_i, factors))
    x = np.zeros((n_u, factors))
    transposed = counts.T.tocsr()
    trace: list[float] = []
    for _ in range(iterations):
        x = _als_half_step(counts, y, reg, alpha)
        y = _als_half_step(transposed, x, reg, alpha)
        if track_objective:
            trace.append(als_objective(counts, x, y, reg, alpha))
    return ALSFactors(x, y, tuple(trace))


class ImplicitALS(Recommender):
    def __init__(self, factors: int = 32, reg: float = 1.0, alpha: float = 20.0,
                 iterations: int = 15, seed: int = 0) -> None:
        super().__init__()
        self.factors, self.reg, self.alpha = factors, reg, alpha
        self.iterations, self.seed = iterations, seed
        self.name = f"implicit ALS (k={factors}, λ={reg:g}, α={alpha:g})"
        self.user_factors: FloatArray = np.zeros((0, 0))
        self.item_factors: FloatArray = np.zeros((0, 0))

    def _fit(self, history: InteractionData) -> None:
        f = fit_als(count_matrix(history), factors=self.factors, reg=self.reg,
                    alpha=self.alpha, iterations=self.iterations, seed=self.seed)
        self.user_factors, self.item_factors = f.user, f.item

    def score(self, users: IntArray) -> FloatArray:
        return np.asarray(self.user_factors[users] @ self.item_factors.T, dtype=np.float64)

    def fold_in(self, counts_row: FloatArray) -> FloatArray:
        """Factors for a new user's click-count vector (item factors fixed)."""
        row = sparse.csr_matrix(np.asarray(counts_row, dtype=np.float64)[None, :])
        return np.asarray(_als_half_step(row, self.item_factors, self.reg, self.alpha)[0],
                          dtype=np.float64)


# ---------------------------------------------------------------------------
# BPR
# ---------------------------------------------------------------------------


@dataclass
class BPRParams:
    user: FloatArray
    item: FloatArray
    bias: FloatArray


def bpr_objective(p: BPRParams, triples: IntArray, reg: float) -> float:
    u, i, j = triples[:, 0], triples[:, 1], triples[:, 2]
    x = np.einsum("nd,nd->n", p.user[u], p.item[i] - p.item[j]) + p.bias[i] - p.bias[j]
    penalty = (np.sum(p.user[u] ** 2, axis=1) + np.sum(p.item[i] ** 2, axis=1)
               + np.sum(p.item[j] ** 2, axis=1) + p.bias[i] ** 2 + p.bias[j] ** 2)
    return float(np.sum(-np.logaddexp(0.0, -x) - 0.5 * reg * penalty))


def bpr_gradient(p: BPRParams, triples: IntArray, reg: float) -> BPRParams:
    """Gradient of :func:`bpr_objective` (summed over triples)."""
    u, i, j = triples[:, 0], triples[:, 1], triples[:, 2]
    x = np.einsum("nd,nd->n", p.user[u], p.item[i] - p.item[j]) + p.bias[i] - p.bias[j]
    g = 1.0 / (1.0 + np.exp(x))  # sigma(-x) = d ln sigma(x) / dx
    gu, gi, gb = np.zeros_like(p.user), np.zeros_like(p.item), np.zeros_like(p.bias)
    np.add.at(gu, u, g[:, None] * (p.item[i] - p.item[j]) - reg * p.user[u])
    np.add.at(gi, i, g[:, None] * p.user[u] - reg * p.item[i])
    np.add.at(gi, j, -g[:, None] * p.user[u] - reg * p.item[j])
    np.add.at(gb, i, g - reg * p.bias[i])
    np.add.at(gb, j, -g - reg * p.bias[j])
    return BPRParams(gu, gi, gb)


def bpr_step(p: BPRParams, triples: IntArray, reg: float, learning_rate: float) -> None:
    """In place: ``theta += learning_rate * bpr_gradient(theta)``, touching only the rows in
    ``triples`` (the gradient of every other row is zero)."""
    u, i, j = triples[:, 0], triples[:, 1], triples[:, 2]
    pu, pi, pj, bi, bj = p.user[u], p.item[i], p.item[j], p.bias[i], p.bias[j]
    x = np.einsum("nd,nd->n", pu, pi - pj) + bi - bj
    g = 1.0 / (1.0 + np.exp(x))
    np.add.at(p.user, u, learning_rate * (g[:, None] * (pi - pj) - reg * pu))
    np.add.at(p.item, i, learning_rate * (g[:, None] * pu - reg * pi))
    np.add.at(p.item, j, learning_rate * (-g[:, None] * pu - reg * pj))
    np.add.at(p.bias, i, learning_rate * (g - reg * bi))
    np.add.at(p.bias, j, learning_rate * (-g - reg * bj))


def _positive_keys(binary: sparse.csr_matrix) -> IntArray:
    coo = binary.tocoo()
    return np.sort(coo.row.astype(np.int64) * binary.shape[1] + coo.col.astype(np.int64))


def _is_positive(keys: IntArray, u: IntArray, j: IntArray, n_items: int) -> np.ndarray:
    q = u * n_items + j
    pos = np.searchsorted(keys, q)
    pos = np.minimum(pos, max(keys.size - 1, 0))
    return np.asarray((keys.size > 0) & (keys[pos] == q), dtype=np.bool_)


@dataclass(frozen=True)
class BPRFit:
    params: BPRParams
    residual_collisions: int
    """Sampled negatives that were in fact positives after 10 redraws (left in place)."""


def fit_bpr(binary: sparse.csr_matrix, *, factors: int, reg: float, learning_rate: float,
            epochs: int, seed: int = 0, batch_size: int = 1024,
            exposure_negatives: sparse.csr_matrix | None = None) -> BPRFit:
    """Mini-batch SGA on BPR; ``exposure_negatives`` (shown, not clicked) if given."""
    if factors < 1 or reg < 0 or learning_rate <= 0 or epochs < 1:
        raise ValueError("invalid BPR hyperparameters")
    rng = np.random.default_rng(seed)
    n_u, n_i = binary.shape
    params = BPRParams(rng.normal(0.0, 0.1, (n_u, factors)), rng.normal(0.0, 0.1, (n_i, factors)),
                       np.zeros(n_i))
    coo = binary.tocoo()
    pos_u, pos_i = coo.row.astype(np.int64), coo.col.astype(np.int64)
    if pos_u.size == 0:
        return BPRFit(params, 0)
    keys = _positive_keys(binary)
    exp_ptr = exp_idx = None
    if exposure_negatives is not None:
        exp_csr = exposure_negatives.tocsr()
        exp_ptr, exp_idx = exp_csr.indptr, exp_csr.indices.astype(np.int64)
    collisions = 0
    for _ in range(epochs):
        order = rng.permutation(pos_u.size)
        for b in range(0, order.size, batch_size):
            sel = order[b:b + batch_size]
            u, i = pos_u[sel], pos_i[sel]
            j = rng.integers(0, n_i, size=sel.size)
            if exp_ptr is not None and exp_idx is not None:
                counts = exp_ptr[u + 1] - exp_ptr[u]
                has = counts > 0
                pick = exp_ptr[u[has]] + (rng.random(int(has.sum())) * counts[has]).astype(
                    np.int64)
                j[has] = exp_idx[pick]
            for _redraw in range(10):
                bad = _is_positive(keys, u, j, n_i)
                if not bad.any():
                    break
                j[bad] = rng.integers(0, n_i, size=int(bad.sum()))
            collisions += int(_is_positive(keys, u, j, n_i).sum())
            bpr_step(params, np.column_stack([u, i, j]), reg, learning_rate)
    return BPRFit(params, collisions)


@dataclass(frozen=True)
class ExposureData:
    """Items shown to each user up to ``cutoff`` without a click (logging-policy exposure)."""

    cutoff: pd.Timestamp
    shown_not_clicked: sparse.csr_matrix


def exposure_data(logs: RecommendationLogs, history: InteractionData) -> ExposureData:
    imp = as_of(deduplicate_impressions(logs.impressions), "timestamp", history.cutoff)
    u = index_of(history.user_ids, imp["user_id"].to_numpy())
    i = index_of(history.item_ids, imp["item_id"].to_numpy())
    shown = sparse.coo_matrix((np.ones(u.size), (u, i)),
                              shape=(history.n_users, history.n_items)).tocsr()
    shown.data[:] = 1.0
    clicked = sparse.csr_matrix(history.clicks > 0)
    neg = (shown - shown.multiply(clicked)).tocsr()
    neg.eliminate_zeros()
    return ExposureData(history.cutoff, neg)


class BPR(Recommender):
    def __init__(self, factors: int = 32, reg: float = 0.01, *, learning_rate: float = 0.05,
                 epochs: int = 30, seed: int = 0,
                 exposure_logs: RecommendationLogs | None = None) -> None:
        """``exposure_logs``: draw negatives from items shown but not clicked, computed from
        these logs at the fitted history's own cutoff (so they cannot leak past it)."""
        super().__init__()
        self.factors, self.reg, self.learning_rate = factors, reg, learning_rate
        self.epochs, self.seed, self.exposure_logs = epochs, seed, exposure_logs
        self.exposure: ExposureData | None = None
        kind = "BPR, exposure-aware negatives" if exposure_logs is not None else "BPR"
        self.name = f"{kind} (k={factors}, λ={reg:g})"
        self.params: BPRParams | None = None
        self.residual_collisions = 0

    def _fit(self, history: InteractionData) -> None:
        self.exposure = (None if self.exposure_logs is None
                         else exposure_data(self.exposure_logs, history))
        fit = fit_bpr(sparse.csr_matrix(history.clicks > 0, dtype=np.float64),
                      factors=self.factors, reg=self.reg, learning_rate=self.learning_rate,
                      epochs=self.epochs, seed=self.seed,
                      exposure_negatives=None if self.exposure is None
                      else self.exposure.shown_not_clicked)
        self.params, self.residual_collisions = fit.params, fit.residual_collisions

    def score(self, users: IntArray) -> FloatArray:
        assert self.params is not None
        return np.asarray(self.params.user[users] @ self.params.item.T + self.params.bias[None, :],
                          dtype=np.float64)


# ---------------------------------------------------------------------------
# EASE
# ---------------------------------------------------------------------------


def fit_ease(binary: sparse.csr_matrix, reg: float) -> FloatArray:
    """EASE (Steck 2019): ``B = I - P diag(1 / diag P)``, ``P = (X'X + reg I)^-1``; diag B = 0.

    The closed form solves ``min |X - XB|_F^2 + reg |B|_F^2`` subject to ``diag(B) = 0``.
    """
    if reg <= 0:
        raise ValueError("reg must be positive")
    gram = np.asarray((binary.T @ binary).toarray(), dtype=np.float64)
    gram[np.diag_indices_from(gram)] += reg
    p = np.linalg.inv(gram)
    b = -p / np.diag(p)[None, :]
    np.fill_diagonal(b, 0.0)
    return np.asarray(b, dtype=np.float64)


class EASE(Recommender):
    def __init__(self, reg: float = 500.0) -> None:
        super().__init__()
        self.reg = reg
        self.name = f"EASE (λ={reg:g})"
        self.weights: FloatArray = np.zeros((0, 0))

    def _fit(self, history: InteractionData) -> None:
        self.weights = fit_ease(sparse.csr_matrix(history.clicks > 0, dtype=np.float64), self.reg)

    def score(self, users: IntArray) -> FloatArray:
        return np.asarray((self.history.clicks[users] > 0).astype(np.float64) @ self.weights,
                          dtype=np.float64)
