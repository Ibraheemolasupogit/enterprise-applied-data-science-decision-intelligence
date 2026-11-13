"""Cold-start strategies (Milestone 9).

**Users** are grouped by the number of distinct items they clicked before the cutoff —
``0``, ``1-2``, ``3-5``, ``6+`` — so each strategy's quality can be read as a function of how
much history it has. Users in the ``0`` bucket are scored by each model's own cold policy: the
segment-popularity fallback for collaborative models, the user tower for the two-tower model.

**Items** launched after the cutoff have no clicks. Strategies compared: content-based scores,
the two-tower item tower (features only), and :class:`ContentToFactor`, which regresses ALS
item factors on item content features over the items that do have clicks (ridge, closed form)
and predicts factors for new items, scored with the users' ALS factors.
"""

from __future__ import annotations

import numpy as np

from decision_intelligence.common.validation import FloatArray
from decision_intelligence.recommendations.baselines import Recommender
from decision_intelligence.recommendations.factorisation import ImplicitALS
from decision_intelligence.recommendations.interactions import IntArray, InteractionData

HISTORY_BUCKETS = ((0, 0, "0"), (1, 2, "1-2"), (3, 5, "3-5"), (6, None, "6+"))


def history_bucket(counts: IntArray) -> np.ndarray:
    labels = np.empty(counts.size, dtype=object)
    for lo, hi, name in HISTORY_BUCKETS:
        sel = (counts >= lo) & (counts <= (hi if hi is not None else np.iinfo(np.int64).max))
        labels[sel] = name
    return labels


def ridge_map(x: FloatArray, y: FloatArray, reg: float) -> FloatArray:
    """``argmin_W |XW - Y|^2 + reg |W|^2`` = ``(X'X + reg I)^-1 X'Y``."""
    if reg <= 0:
        raise ValueError("reg must be positive")
    return np.asarray(np.linalg.solve(x.T @ x + reg * np.eye(x.shape[1]), x.T @ y),
                      dtype=np.float64)


class ContentToFactor(Recommender):
    """ALS user factors x content-predicted item factors (for items without clicks)."""

    def __init__(self, als: ImplicitALS, reg: float = 1.0) -> None:
        super().__init__()
        self.als, self.reg = als, reg
        self.name = "content → ALS factors"
        self.mapping: FloatArray = np.zeros((0, 0))
        self.predicted_items: FloatArray = np.zeros((0, 0))

    @staticmethod
    def _design(history: InteractionData) -> FloatArray:
        return np.hstack([history.item_features, np.ones((history.n_items, 1))])

    def _fit(self, history: InteractionData) -> None:
        self.als.fit(history)
        warm = history.clicks.sum(axis=0) > 0
        x = self._design(history)
        self.mapping = ridge_map(x[warm], self.als.item_factors[warm], self.reg)
        self.predicted_items = x @ self.mapping

    def score(self, users: IntArray) -> FloatArray:
        return np.asarray(self.als.user_factors[users] @ self.predicted_items.T, dtype=np.float64)
