"""Baseline recommenders against brute-force and closed-form references."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from decision_intelligence.recommendations import baselines as bl
from decision_intelligence.recommendations.evaluation import top_k
from decision_intelligence.recommendations.interactions import InteractionData

RNG = np.random.default_rng(20)
X = (RNG.random((12, 9)) < 0.35).astype(float)
X[0] = 0.0  # a user with no history
ALL = np.arange(12)


def test_global_popularity_matches_groupby(make_history: Callable[..., InteractionData]) -> None:
    h = make_history(X)
    counts = pd.Series(h.click_events["item_index"]).value_counts()
    expected = counts.reindex(range(9), fill_value=0).to_numpy(dtype=float)
    model = bl.GlobalPopularity().fit(h)
    assert np.array_equal(model.score(ALL), np.tile(expected, (12, 1)))


def test_time_decayed_popularity_closed_form(make_history: Callable[..., InteractionData]) -> None:
    clicks = np.array([[1.0, 1.0], [1.0, 0.0]])
    days = np.array([90.0, 30.0, 100.0])  # events in np.nonzero order: (0,0), (0,1), (1,0)
    h = make_history(clicks, days=days)
    scores = bl.GlobalPopularity(half_life_days=10.0).fit(h).score(np.array([0]))[0]
    assert scores[0] == pytest.approx(2 ** -1.0 + 2 ** 0.0)
    assert scores[1] == pytest.approx(2 ** -7.0)


def test_segment_popularity_shrinkage_closed_form(
    make_history: Callable[..., InteractionData],
) -> None:
    segs = ["a"] * 6 + ["b"] * 6
    h = make_history(X, segments=segs)
    lam = 4.0
    model = bl.SegmentPopularity(lam).fit(h)
    total = X.sum(axis=0)
    share = total / total.sum()
    for g, rows in (("a", slice(0, 6)), ("b", slice(6, 12))):
        n_gi = X[rows].sum(axis=0)
        expected = (n_gi + lam * share) / (n_gi.sum() + lam)
        user = 0 if g == "a" else 6
        assert np.allclose(model.score(np.array([user]))[0], expected, atol=1e-15)
    assert np.allclose(bl.SegmentPopularity(0.0).fit(h).score(np.array([6]))[0],
                       X[6:].sum(axis=0) / X[6:].sum())


def _cosine(a: np.ndarray, b: np.ndarray, shrink: float) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + shrink)) if (
        np.linalg.norm(a) * np.linalg.norm(b) + shrink) > 0 else 0.0


def test_item_knn_matches_brute_force(make_history: Callable[..., InteractionData]) -> None:
    k, shrink = 3, 2.0
    model = bl.ItemKNN(k, shrink).fit(make_history(X))
    n = X.shape[1]
    sim = np.array([[0.0 if i == j else _cosine(X[:, i], X[:, j], shrink) for j in range(n)]
                    for i in range(n)])
    pruned = np.zeros_like(sim)
    for j in range(n):  # keep, for target item j, its k most similar items (ties: lower index)
        order = sorted(range(n), key=lambda i, j=j: (-sim[i, j], i))[:k]
        pruned[order, j] = sim[order, j]
    assert np.allclose(model.similarity, pruned, atol=1e-12)
    assert np.allclose(model.score(ALL), X @ pruned, atol=1e-12)


def test_user_knn_matches_brute_force(make_history: Callable[..., InteractionData]) -> None:
    k, shrink = 4, 1.0
    model = bl.UserKNN(k, shrink).fit(make_history(X))
    got = model.score(ALL)
    for u in range(X.shape[0]):
        sims = [0.0 if v == u else _cosine(X[u], X[v], shrink) for v in range(X.shape[0])]
        top = sorted(range(X.shape[0]), key=lambda v: (-sims[v], v))[:k]
        expected = sum(sims[v] * X[v] for v in top)
        assert np.allclose(got[u], expected, atol=1e-12)


def test_content_based_matches_brute_force(make_history: Callable[..., InteractionData]) -> None:
    f = RNG.random((9, 4))
    f = f / np.linalg.norm(f, axis=1, keepdims=True)
    model = bl.ContentBased().fit(make_history(X, features=f))
    got = model.score(ALL)
    for u in range(1, X.shape[0]):
        profile = X[u] @ f
        profile = profile / np.linalg.norm(profile)
        assert np.allclose(got[u], f @ profile, atol=1e-12)
    assert np.all(got[0] == 0.0)  # no history: no profile


def test_pure_svd_eckart_young_and_reconstruction(
    make_history: Callable[..., InteractionData],
) -> None:
    k = 3
    model = bl.PureSVD(k).fit(make_history(X))
    v = model.item_factors
    residual = np.linalg.norm(X - X @ v @ v.T) ** 2
    assert residual == pytest.approx(np.sum(model.singular_values[k:] ** 2), rel=1e-10)
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    assert np.allclose(model.score(ALL), (u[:, :k] * s[:k]) @ vt[:k], atol=1e-10)
    # Sign-invariant: flipping a singular vector leaves scores unchanged.
    flipped = v * np.array([-1.0, 1.0, -1.0])
    assert np.allclose(X @ flipped @ flipped.T, model.score(ALL), atol=1e-12)


def test_fallback_routes_users_by_history(make_history: Callable[..., InteractionData]) -> None:
    h = make_history(X)
    fb = bl.Fallback(bl.ItemKNN(3), bl.GlobalPopularity(), min_interactions=1).fit(h)
    scores = fb.score(ALL)
    pop = bl.GlobalPopularity().fit(h).score(ALL)
    knn = bl.ItemKNN(3).fit(h).score(ALL)
    assert np.array_equal(scores[0], pop[0])
    assert np.array_equal(scores[1:], knn[1:])
    assert fb.uses_fallback(ALL).tolist() == [True] + [False] * 11


def test_collaborative_models_cannot_rank_a_user_without_history(
    make_history: Callable[..., InteractionData],
) -> None:
    h = make_history(X)
    for model in (bl.ItemKNN(3), bl.UserKNN(3), bl.PureSVD(2)):
        row = model.fit(h).score(np.array([0]))[0]
        assert np.ptp(row) == 0.0  # constant: the ranking degenerates to item-id order


def test_rankings_are_deterministic_and_ties_go_to_lower_item_ids() -> None:
    scores = np.array([[1.0, 2.0, 2.0, 0.5, 2.0]])
    allowed = np.ones_like(scores, dtype=bool)
    assert top_k(scores, allowed, 3).tolist() == [[1, 2, 4]]
    allowed[0, 2] = False
    assert top_k(scores, allowed, 3).tolist() == [[1, 4, 0]]


def test_top_k_refuses_unsafe_inputs() -> None:
    scores = np.array([[1.0, np.nan, 0.0]])
    with pytest.raises(ValueError, match="finite"):
        top_k(scores, np.ones_like(scores, dtype=bool), 2)
    with pytest.raises(ValueError, match="allowed items"):
        top_k(np.zeros((1, 3)), np.array([[True, False, False]]), 2)


def test_random_recommender_is_seeded(make_history: Callable[..., InteractionData]) -> None:
    h = make_history(X)
    a = bl.RandomRecommender(3).fit(h).score(ALL)
    assert np.array_equal(a, bl.RandomRecommender(3).fit(h).score(ALL))
    assert not np.array_equal(a, bl.RandomRecommender(4).fit(h).score(ALL))


def test_refitting_is_deterministic(make_history: Callable[..., InteractionData]) -> None:
    h = make_history(X)
    for make in (lambda: bl.ItemKNN(4), lambda: bl.UserKNN(4), lambda: bl.PureSVD(3),
                 lambda: bl.SegmentPopularity(10.0), bl.ContentBased):
        assert np.array_equal(make().fit(h).score(ALL), make().fit(h).score(ALL))


def test_invalid_hyperparameters_and_unfitted_use() -> None:
    for bad in (lambda: bl.ItemKNN(0), lambda: bl.UserKNN(5, -1.0), lambda: bl.PureSVD(0),
                lambda: bl.SegmentPopularity(-1.0), lambda: bl.GlobalPopularity(0.0)):
        with pytest.raises(ValueError):
            bad()
    with pytest.raises(RuntimeError, match="not been fitted"):
        _ = bl.ContentBased().history
    with pytest.raises(ValueError):
        bl.Fallback(bl.ContentBased(), bl.GlobalPopularity(), min_interactions=0)


def test_pure_svd_refuses_too_many_factors(make_history: Callable[..., InteractionData]) -> None:
    with pytest.raises(ValueError, match="rank"):
        bl.PureSVD(10).fit(make_history(X))


def test_shrunk_cosine_handles_zero_vectors() -> None:
    sim = bl.shrunk_cosine(np.zeros((2, 3)), np.ones((2, 3)), 0.0)
    assert np.all(sim == 0.0) and not np.any(np.isnan(sim))
    assert math.isclose(float(bl.shrunk_cosine(np.ones((1, 2)), np.ones((1, 2)), 0.0)[0, 0]), 1.0)
