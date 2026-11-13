"""Two-tower gradients and training; exact and approximate retrieval (Milestone 9)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from decision_intelligence.data.recommendation_scenarios import RecommendationScenario
from decision_intelligence.recommendations.cases import perturb_after
from decision_intelligence.recommendations.interactions import build_interaction_data
from decision_intelligence.recommendations.retrieval import (
    KMeansIndex,
    exact_top_k,
    recall_of_exact,
)
from decision_intelligence.recommendations.two_tower import (
    Tower,
    TwoTower,
    TwoTowerParams,
    softmax_loss_and_grads,
)


@pytest.mark.parametrize("hidden", [0, 4])
def test_two_tower_gradients_match_finite_differences(hidden: int) -> None:
    rng = np.random.default_rng(1)
    params = TwoTowerParams(Tower.initialise(5, 3, hidden, rng),
                            Tower.initialise(4, 3, hidden, rng))
    fu, fi = rng.normal(size=(6, 5)), rng.normal(size=(8, 4))
    pos = np.array([0, 3, 3, 7, 2, 5])
    allowed = np.ones(8, dtype=bool)
    allowed[1] = False
    _, gu, gi = softmax_loss_and_grads(params, fu, fi, pos, allowed, 0.01)
    eps = 1e-6
    for weights, grads in ((params.user.weights, gu), (params.item.weights, gi)):
        for w, g in zip(weights, grads, strict=True):
            for idx in list(np.ndindex(w.shape))[:12]:
                old = w[idx]
                w[idx] = old + eps
                up = softmax_loss_and_grads(params, fu, fi, pos, allowed, 0.01)[0]
                w[idx] = old - eps
                down = softmax_loss_and_grads(params, fu, fi, pos, allowed, 0.01)[0]
                w[idx] = old
                assert g[idx] == pytest.approx((up - down) / (2 * eps), abs=1e-6)


def test_two_tower_trains_scores_cold_users_and_new_items(
    small_scenario: RecommendationScenario,
) -> None:
    logs = small_scenario.logs
    h = build_interaction_data(logs, logs.start + pd.to_timedelta(98, unit="D"))
    model = TwoTower(8, 0, epochs=10, seed=3).fit(h)
    assert model.loss_trace[-1] < model.loss_trace[0]
    cold = np.flatnonzero(h.user_interaction_counts() == 0)[:5]
    s = model.score(cold)
    assert np.all(np.isfinite(s)) and np.all(np.ptp(s, axis=1) > 0)
    new = np.flatnonzero(~h.item_launched)
    assert new.size and np.ptp(model.item_embeddings[new], axis=0).max() > 0
    again = TwoTower(8, 0, epochs=10, seed=3).fit(h)
    assert np.array_equal(again.score(cold), s)


def test_two_tower_is_invariant_to_post_cutoff_data(small_scenario: RecommendationScenario) -> None:
    logs = small_scenario.logs
    cutoff = logs.start + pd.to_timedelta(98, unit="D")
    users = np.arange(len(logs.users))
    a = TwoTower(4, 4, epochs=2).fit(build_interaction_data(logs, cutoff)).score(users)
    p2 = perturb_after(logs, cutoff, seed=9)
    b = TwoTower(4, 4, epochs=2).fit(build_interaction_data(p2, cutoff)).score(users)
    assert np.array_equal(a, b)


def test_exact_top_k_matches_a_brute_force_sort() -> None:
    rng = np.random.default_rng(2)
    q, items = rng.normal(size=(7, 4)), rng.normal(size=(30, 4))
    allowed = rng.random((7, 30)) < 0.7
    got = exact_top_k(q, items, allowed, 5)
    for r in range(7):
        cand = np.flatnonzero(allowed[r])
        ref = sorted(cand, key=lambda i: (-(items[i] @ q[r]), i))[:5]
        assert got[r].tolist() == ref


def test_kmeans_index_is_exact_when_every_cluster_is_probed() -> None:
    rng = np.random.default_rng(3)
    q, items = rng.normal(size=(10, 4)), rng.normal(size=(60, 4))
    allowed = np.ones((10, 60), dtype=bool)
    exact = exact_top_k(q, items, allowed, 5)
    index = KMeansIndex(6, seed=1).build(items)
    full, scored = index.query(q, items, allowed, 5, n_probe=6)
    assert np.array_equal(full, exact) and np.all(scored == 60)
    assert np.allclose(recall_of_exact(full, exact), 1.0)
    one, scored1 = index.query(q, items, allowed, 5, n_probe=1)
    assert np.all(scored1 < 60) and recall_of_exact(one, exact).mean() <= 1.0
    with pytest.raises(RuntimeError):
        KMeansIndex(3).query(q, items, allowed, 5, 1)
    with pytest.raises(ValueError):
        KMeansIndex(100).build(items)
