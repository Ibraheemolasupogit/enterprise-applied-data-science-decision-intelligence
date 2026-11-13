"""Baseline recommenders (Brief §8.3): the benchmarks every later model must beat.

All models share one interface: ``fit(history)`` on an :class:`InteractionData` (logged
implicit feedback as of a cutoff — never ground truth) and ``score(users)``, which returns
a ``(len(users), n_items)`` score matrix. Ranking, eligibility and seen-item exclusion are
applied by :func:`decision_intelligence.recommendations.evaluation.top_k`, identically for
every model, with ties broken by item id (ascending) so rankings are deterministic.

=====================  =========================================================
Model                  Score for user ``u`` and item ``i``
=====================  =========================================================
Random                 i.i.d. uniform (seeded): the floor
Global popularity      number of clicks on ``i`` (optionally exponentially
                       time-decayed with a half-life in days)
Segment popularity     ``(n_gi + lam * s_i) / (n_g + lam)``: the item's share of
                       clicks in the user's segment ``g``, shrunk towards the
                       global share ``s_i`` (empirical-Bayes style; ``lam`` in clicks)
Item-item kNN          ``sum_j x_uj S_ji``; ``S`` = cosine similarity of item click
                       columns with shrinkage ``x_i.x_j / (|x_i||x_j| + h)``, zero
                       diagonal, top-``k`` neighbours per item
User-user kNN          ``sum_v S_uv x_vi`` over the top-``k`` most similar users
                       (same shrunk cosine on user rows, self excluded)
Content-based          cosine between the item's content vector and the user's
                       profile (sum of clicked items' content vectors)
PureSVD                ``x_u V_k V_k^T``: projection of the user's click row onto
                       the top-``k`` right singular vectors (Cremonesi et al. 2010)
=====================  =========================================================

Collaborative models cannot score a user with no history (all-zero scores, so the
ranking degenerates to item-id order) or an item with no clicks. :class:`Fallback`
routes users with fewer than ``min_interactions`` training clicks to a fallback model
(segment popularity by default in the case studies) — the cold-start policy evaluated
in the evidence.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np

from decision_intelligence.common.validation import FloatArray
from decision_intelligence.recommendations.interactions import IntArray, InteractionData

_USER_BATCH = 512


class Recommender(ABC):
    """Fit on logged implicit feedback; score every item for the requested users."""

    name: str = "recommender"

    def __init__(self) -> None:
        self._history: InteractionData | None = None

    @property
    def history(self) -> InteractionData:
        if self._history is None:
            raise RuntimeError(f"{self.name} has not been fitted")
        return self._history

    def fit(self, history: InteractionData) -> Recommender:
        self._history = history
        self._fit(history)
        return self

    @abstractmethod
    def _fit(self, history: InteractionData) -> None: ...

    @abstractmethod
    def score(self, users: IntArray) -> FloatArray: ...


class RandomRecommender(Recommender):
    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self.seed = seed
        self.name = "random"
        self._scores: FloatArray | None = None

    def _fit(self, history: InteractionData) -> None:
        rng = np.random.default_rng(self.seed)
        self._scores = rng.random((history.n_users, history.n_items))

    def score(self, users: IntArray) -> FloatArray:
        assert self._scores is not None
        return np.asarray(self._scores[users], dtype=np.float64)


def _click_weights(history: InteractionData, half_life_days: float | None) -> FloatArray:
    events = history.click_events
    if half_life_days is None:
        return np.ones(len(events), dtype=np.float64)
    cutoff_day = (history.cutoff - history.start).total_seconds() / 86_400.0
    age = cutoff_day - events["day"].to_numpy(dtype=np.float64)
    return np.asarray(np.exp(-math.log(2.0) * age / half_life_days), dtype=np.float64)


class GlobalPopularity(Recommender):
    def __init__(self, half_life_days: float | None = None) -> None:
        super().__init__()
        if half_life_days is not None and half_life_days <= 0:
            raise ValueError("half_life_days must be positive")
        self.half_life_days = half_life_days
        self.name = ("global popularity" if half_life_days is None
                     else f"global popularity (half-life {half_life_days:g} d)")
        self.item_scores: FloatArray = np.zeros(0)

    def _fit(self, history: InteractionData) -> None:
        w = _click_weights(history, self.half_life_days)
        self.item_scores = np.bincount(history.click_events["item_index"].to_numpy(),
                                       weights=w, minlength=history.n_items).astype(np.float64)

    def score(self, users: IntArray) -> FloatArray:
        return np.asarray(np.broadcast_to(self.item_scores, (users.size, self.item_scores.size)),
                          dtype=np.float64).copy()


class SegmentPopularity(Recommender):
    def __init__(self, shrinkage: float = 100.0) -> None:
        super().__init__()
        if shrinkage < 0:
            raise ValueError("shrinkage must be non-negative")
        self.shrinkage = shrinkage
        self.name = f"segment popularity (λ={shrinkage:g})"
        self.segments: tuple[str, ...] = ()
        self.segment_scores: FloatArray = np.zeros((0, 0))

    def _fit(self, history: InteractionData) -> None:
        events = history.click_events
        items = events["item_index"].to_numpy()
        seg_of_event = history.user_segment[events["user_index"].to_numpy()]
        self.segments = tuple(sorted(set(history.user_segment.tolist())))
        total = np.bincount(items, minlength=history.n_items).astype(np.float64)
        share = total / total.sum() if total.sum() > 0 else total
        rows = []
        for g in self.segments:
            n_gi = np.bincount(items[seg_of_event == g], minlength=history.n_items).astype(float)
            rows.append((n_gi + self.shrinkage * share) / (n_gi.sum() + self.shrinkage)
                        if n_gi.sum() + self.shrinkage > 0 else n_gi)
        self.segment_scores = np.asarray(rows, dtype=np.float64)

    def score(self, users: IntArray) -> FloatArray:
        seg = np.searchsorted(np.array(self.segments), self.history.user_segment[users])
        return np.asarray(self.segment_scores[seg], dtype=np.float64)


def shrunk_cosine(a: FloatArray, b: FloatArray, shrink: float) -> FloatArray:
    """``a_r . b_c / (|a_r| |b_c| + shrink)`` for every row ``r`` of ``a`` and ``b``."""
    num = a @ b.T
    den = np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=1)[None, :] + shrink
    return np.asarray(np.divide(num, den, out=np.zeros_like(num), where=den > 0),
                      dtype=np.float64)


def keep_top_k_per_row(matrix: FloatArray, k: int) -> FloatArray:
    """Zero all but the ``k`` largest entries of each row (ties: lower column index kept)."""
    if k >= matrix.shape[1]:
        return matrix.copy()
    order = np.argsort(-matrix, axis=1, kind="stable")[:, :k]
    out = np.zeros_like(matrix)
    rows = np.arange(matrix.shape[0])[:, None]
    out[rows, order] = matrix[rows, order]
    return out


class ItemKNN(Recommender):
    def __init__(self, n_neighbours: int = 50, shrink: float = 10.0) -> None:
        super().__init__()
        if n_neighbours < 1 or shrink < 0:
            raise ValueError("n_neighbours must be >= 1 and shrink >= 0")
        self.n_neighbours, self.shrink = n_neighbours, shrink
        self.name = f"item-kNN (k={n_neighbours})"
        self.similarity: FloatArray = np.zeros((0, 0))

    def _fit(self, history: InteractionData) -> None:
        cols = history.clicks.T
        sim = shrunk_cosine(cols, cols, self.shrink)
        np.fill_diagonal(sim, 0.0)
        # Keep, for each target item (column), its k most similar source items.
        self.similarity = keep_top_k_per_row(sim.T, self.n_neighbours).T

    def score(self, users: IntArray) -> FloatArray:
        return np.asarray(self.history.clicks[users] @ self.similarity, dtype=np.float64)


class UserKNN(Recommender):
    def __init__(self, n_neighbours: int = 50, shrink: float = 10.0) -> None:
        super().__init__()
        if n_neighbours < 1 or shrink < 0:
            raise ValueError("n_neighbours must be >= 1 and shrink >= 0")
        self.n_neighbours, self.shrink = n_neighbours, shrink
        self.name = f"user-kNN (k={n_neighbours})"

    def _fit(self, history: InteractionData) -> None:
        return None

    def score(self, users: IntArray) -> FloatArray:
        x = self.history.clicks
        out = np.empty((users.size, x.shape[1]), dtype=np.float64)
        for b in range(0, users.size, _USER_BATCH):
            batch = users[b:b + _USER_BATCH]
            sim = shrunk_cosine(x[batch], x, self.shrink)
            sim[np.arange(batch.size), batch] = 0.0
            out[b:b + batch.size] = keep_top_k_per_row(sim, self.n_neighbours) @ x
        return out


class ContentBased(Recommender):
    def __init__(self) -> None:
        super().__init__()
        self.name = "content-based"

    def _fit(self, history: InteractionData) -> None:
        return None

    def score(self, users: IntArray) -> FloatArray:
        h = self.history
        profile = h.clicks[users] @ h.item_features
        norm = np.linalg.norm(profile, axis=1, keepdims=True)
        profile = np.divide(profile, norm, out=np.zeros_like(profile), where=norm > 0)
        return np.asarray(profile @ h.item_features.T, dtype=np.float64)


class PureSVD(Recommender):
    def __init__(self, n_factors: int = 32) -> None:
        super().__init__()
        if n_factors < 1:
            raise ValueError("n_factors must be >= 1")
        self.n_factors = n_factors
        self.name = f"PureSVD (k={n_factors})"
        self.item_factors: FloatArray = np.zeros((0, 0))
        self.singular_values: FloatArray = np.zeros(0)

    def _fit(self, history: InteractionData) -> None:
        if self.n_factors > min(history.clicks.shape):
            raise ValueError("n_factors exceeds the matrix rank bound")
        _, sigma, vt = np.linalg.svd(history.clicks, full_matrices=False)
        self.singular_values = sigma
        self.item_factors = np.asarray(vt[: self.n_factors].T, dtype=np.float64)

    def score(self, users: IntArray) -> FloatArray:
        v = self.item_factors
        return np.asarray((self.history.clicks[users] @ v) @ v.T, dtype=np.float64)


class Fallback(Recommender):
    """``primary`` for users with at least ``min_interactions`` training clicks, else
    ``fallback`` (the cold-start policy)."""

    def __init__(self, primary: Recommender, fallback: Recommender,
                 min_interactions: int = 1) -> None:
        super().__init__()
        if min_interactions < 1:
            raise ValueError("min_interactions must be >= 1")
        self.primary, self.fallback, self.min_interactions = primary, fallback, min_interactions
        self.name = primary.name

    def _fit(self, history: InteractionData) -> None:
        self.primary.fit(history)
        self.fallback.fit(history)
        self._counts = history.user_interaction_counts()

    def score(self, users: IntArray) -> FloatArray:
        cold = self._counts[users] < self.min_interactions
        out = np.empty((users.size, self.history.n_items), dtype=np.float64)
        if (~cold).any():
            out[~cold] = self.primary.score(users[~cold])
        if cold.any():
            out[cold] = self.fallback.score(users[cold])
        return out

    def uses_fallback(self, users: IntArray) -> np.ndarray:
        return np.asarray(self._counts[users] < self.min_interactions)
