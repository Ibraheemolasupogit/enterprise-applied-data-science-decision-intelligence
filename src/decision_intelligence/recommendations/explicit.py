"""Explicit feedback: a regularised bias model for 1-5 ratings.

``r_hat_ui = mu + b_u + b_i``, minimising
``sum (r - mu - b_u - b_i)^2 + lam_u sum b_u^2 + lam_i sum b_i^2`` (``mu`` unpenalised).
The objective is a ridge least-squares problem with a unique minimiser; it is solved
exactly from its sparse normal equations (``scipy.sparse.linalg.spsolve``). An earlier
block-coordinate-descent version stopped at its iteration cap without converging on
weakly regularised, chain-coupled data (M8 assurance review); tests compare the fit with a
dense least-squares solve, including that case.

Explicit ratings here are **missing not at random**: users rate items they liked more
often. RMSE on held-out *observed* ratings therefore measures accuracy on the kind of
pairs users choose to rate, not on the pairs a recommender must rank; the evidence
compares it with the oracle error on uniformly random pairs. Unknown users or items get
a zero bias.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

from decision_intelligence.common.validation import FloatArray
from decision_intelligence.data.point_in_time import as_of
from decision_intelligence.recommendations.interactions import IntArray, index_of


@dataclass(frozen=True)
class RatingData:
    user_index: IntArray
    item_index: IntArray
    rating: FloatArray
    n_users: int
    n_items: int


def rating_data(ratings: pd.DataFrame, user_ids: IntArray, item_ids: IntArray,
                *, after: pd.Timestamp | None, until: pd.Timestamp) -> RatingData:
    """Ratings in ``(after, until]`` (``after=None``: everything up to ``until``)."""
    frame = as_of(ratings, "timestamp", until)
    if after is not None:
        frame = frame.loc[frame["timestamp"] > after]
    return RatingData(index_of(user_ids, frame["user_id"].to_numpy()),
                      index_of(item_ids, frame["item_id"].to_numpy()),
                      frame["rating"].to_numpy(dtype=np.float64), int(user_ids.size),
                      int(item_ids.size))


class BiasRatingModel:
    def __init__(self, reg_user: float = 5.0, reg_item: float = 5.0) -> None:
        if reg_user <= 0 or reg_item <= 0:
            raise ValueError("regularisation must be positive")
        self.reg_user, self.reg_item = reg_user, reg_item
        self.mu = 0.0
        self.user_bias: FloatArray = np.zeros(0)
        self.item_bias: FloatArray = np.zeros(0)

    def fit(self, data: RatingData) -> BiasRatingModel:
        """Exact minimiser: solve ``(A'A + P) beta = A'r`` with ``A = [1 | users | items]``
        and ``P = diag(0, lam_u, ..., lam_i, ...)``. ``P`` makes the system positive definite
        whenever there is at least one rating; unrated users and items get a zero bias."""
        n = data.rating.size
        if n == 0:
            raise ValueError("no ratings to fit")
        if not np.all(np.isfinite(data.rating)):
            raise ValueError("ratings must be finite")
        n_u, n_i = data.n_users, data.n_items
        rows = np.repeat(np.arange(n), 3)
        cols = np.column_stack([np.zeros(n, dtype=np.int64), 1 + data.user_index,
                                1 + n_u + data.item_index]).ravel()
        design = sparse.csr_matrix((np.ones(3 * n), (rows, cols)), shape=(n, 1 + n_u + n_i))
        penalty = sparse.diags(np.concatenate([[0.0], np.full(n_u, self.reg_user),
                                               np.full(n_i, self.reg_item)]))
        normal = (design.T @ design + penalty).tocsc()
        beta = np.asarray(spsolve(normal, design.T @ data.rating), dtype=np.float64)
        if not np.all(np.isfinite(beta)):
            raise RuntimeError("bias model solve failed")
        self.mu = float(beta[0])
        self.user_bias = beta[1:1 + n_u]
        self.item_bias = beta[1 + n_u:]
        return self

    def objective(self, data: RatingData) -> float:
        resid = data.rating - self.predict(data.user_index, data.item_index)
        return float(np.sum(resid ** 2) + self.reg_user * np.sum(self.user_bias ** 2)
                     + self.reg_item * np.sum(self.item_bias ** 2))

    def predict(self, users: IntArray, items: IntArray) -> FloatArray:
        return np.asarray(self.mu + self.user_bias[users] + self.item_bias[items],
                          dtype=np.float64)
