"""Implicit ALS, BPR and EASE against exact dense references (Milestone 9)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from scipy import sparse

from decision_intelligence.data.recommendation_scenarios import RecommendationScenario
from decision_intelligence.recommendations.baselines import Recommender
from decision_intelligence.recommendations.cases import perturb_after
from decision_intelligence.recommendations.factorisation import (
    BPR,
    EASE,
    BPRParams,
    ImplicitALS,
    _als_half_step,
    bpr_gradient,
    bpr_objective,
    bpr_step,
    exposure_data,
    fit_als,
    fit_bpr,
    fit_ease,
)
from decision_intelligence.recommendations.interactions import build_interaction_data

RNG = np.random.default_rng(42)
_DENSE = np.where(RNG.random((9, 7)) < 0.35, RNG.integers(1, 4, (9, 7)), 0).astype(float)
_DENSE[0] = 0.0  # a user with no clicks
COUNTS = sparse.csr_matrix(_DENSE)


def _dense_user_solve(r_row: np.ndarray, y: np.ndarray, reg: float, alpha: float) -> np.ndarray:
    c = 1.0 + alpha * r_row
    p = (r_row > 0).astype(float)
    cy = y * c[:, None]
    return np.linalg.solve(y.T @ cy + reg * np.eye(y.shape[1]), cy.T @ p)


def test_als_half_step_is_the_exact_weighted_ridge_solution() -> None:
    y = RNG.normal(size=(7, 3))
    reg, alpha = 0.5, 4.0
    got = _als_half_step(COUNTS, y, reg, alpha)
    dense = COUNTS.toarray()
    for u in range(dense.shape[0]):
        expected = np.zeros(3) if dense[u].sum() == 0 else _dense_user_solve(dense[u], y, reg,
                                                                             alpha)
        assert np.allclose(got[u], expected, atol=1e-10)


def test_als_objective_never_increases_and_fold_in_is_exact() -> None:
    f = fit_als(COUNTS, factors=3, reg=0.5, alpha=4.0, iterations=8, track_objective=True)
    trace = np.array(f.objective_trace)
    assert np.all(np.diff(trace) <= 1e-9)
    assert np.all(f.user[0] == 0.0)  # no clicks -> zero factors
    model = ImplicitALS(3, 0.5, 4.0, 8)
    model.item_factors = f.item
    row = COUNTS.toarray()[4]
    assert np.allclose(model.fold_in(row), _dense_user_solve(row, f.item, 0.5, 4.0), atol=1e-10)


def test_als_is_deterministic_and_validates() -> None:
    a = fit_als(COUNTS, factors=2, reg=1.0, alpha=1.0, iterations=3, seed=5)
    b = fit_als(COUNTS, factors=2, reg=1.0, alpha=1.0, iterations=3, seed=5)
    assert np.array_equal(a.user, b.user)
    with pytest.raises(ValueError):
        fit_als(COUNTS, factors=0, reg=1.0, alpha=1.0, iterations=3)
    with pytest.raises(ValueError):
        fit_als(COUNTS, factors=2, reg=0.0, alpha=1.0, iterations=3)


def _params(rng: np.random.Generator) -> BPRParams:
    return BPRParams(rng.normal(size=(5, 3)), rng.normal(size=(6, 3)), rng.normal(size=6))


def test_bpr_gradient_matches_finite_differences() -> None:
    rng = np.random.default_rng(3)
    p = _params(rng)
    triples = np.array([[0, 1, 2], [1, 3, 4], [0, 5, 1], [4, 2, 0], [0, 1, 3]])
    grad = bpr_gradient(p, triples, reg=0.1)
    eps = 1e-6
    for attr in ("user", "item", "bias"):
        arr = getattr(p, attr)
        g = getattr(grad, attr)
        for idx in np.ndindex(arr.shape):
            old = arr[idx]
            arr[idx] = old + eps
            up = bpr_objective(p, triples, 0.1)
            arr[idx] = old - eps
            down = bpr_objective(p, triples, 0.1)
            arr[idx] = old
            assert g[idx] == pytest.approx((up - down) / (2 * eps), abs=1e-6)


def test_bpr_step_is_learning_rate_times_gradient_including_repeated_rows() -> None:
    rng = np.random.default_rng(4)
    p = _params(rng)
    triples = np.array([[0, 1, 2], [0, 1, 3], [2, 1, 0]])  # user 0 and item 1 repeat
    grad = bpr_gradient(p, triples, reg=0.05)
    before = BPRParams(p.user.copy(), p.item.copy(), p.bias.copy())
    bpr_step(p, triples, 0.05, 0.1)
    assert np.allclose(p.user - before.user, 0.1 * grad.user, atol=1e-14)
    assert np.allclose(p.item - before.item, 0.1 * grad.item, atol=1e-14)
    assert np.allclose(p.bias - before.bias, 0.1 * grad.bias, atol=1e-14)


def test_bpr_is_seeded_and_ranks_positives_above_negatives() -> None:
    binary = COUNTS.copy()
    binary.data[:] = 1.0
    a = fit_bpr(binary, factors=4, reg=0.01, learning_rate=0.05, epochs=200, seed=1)
    b = fit_bpr(binary, factors=4, reg=0.01, learning_rate=0.05, epochs=200, seed=1)
    assert np.array_equal(a.params.item, b.params.item)
    assert a.residual_collisions == 0
    s = a.params.user @ a.params.item.T + a.params.bias
    dense = binary.toarray()
    auc = []
    for u in range(1, dense.shape[0]):
        pos, neg = s[u, dense[u] > 0], s[u, dense[u] == 0]
        if pos.size and neg.size:
            auc.append(np.mean(pos[:, None] > neg[None, :]))
    assert np.mean(auc) > 0.9  # training-set AUC: the optimiser fits the pairwise objective


def test_ease_matches_a_column_by_column_constrained_ridge_solve() -> None:
    binary = COUNTS.copy()
    binary.data[:] = 1.0
    x = binary.toarray()
    reg = 2.0
    b = fit_ease(binary, reg)
    assert np.all(np.diag(b) == 0.0)
    for j in range(x.shape[1]):
        others = [i for i in range(x.shape[1]) if i != j]
        xo = x[:, others]
        col = np.linalg.solve(xo.T @ xo + reg * np.eye(len(others)), xo.T @ x[:, j])
        assert np.allclose(b[others, j], col, atol=1e-10)


def test_exposure_negatives_are_shown_unclicked_items_at_the_cutoff(
    small_scenario: RecommendationScenario,
) -> None:
    logs = small_scenario.logs
    cutoff = logs.start + np.timedelta64(98, "D")
    h = build_interaction_data(logs, cutoff)
    ex = exposure_data(logs, h)
    shown = logs.impressions.loc[logs.impressions["timestamp"] <= cutoff]
    pairs = set(zip(shown["user_id"] - 1, shown["item_id"] - 1, strict=True))
    r, c = ex.shown_not_clicked.nonzero()
    assert all((u, i) in pairs and h.clicks[u, i] == 0 for u, i in zip(r, c, strict=True))
    perturbed = perturb_after(logs, cutoff, seed=2)
    ex2 = exposure_data(perturbed, build_interaction_data(perturbed, cutoff))
    assert (ex.shown_not_clicked != ex2.shown_not_clicked).nnz == 0


@pytest.mark.parametrize("make", [
    lambda: ImplicitALS(4, 1.0, 5.0, 3), lambda: BPR(4, 0.01, epochs=3), lambda: EASE(10.0)])
def test_m9_factor_models_are_invariant_to_post_cutoff_data(
    small_scenario: RecommendationScenario, make: Callable[[], Recommender],
) -> None:
    logs = small_scenario.logs
    cutoff = logs.start + np.timedelta64(98, "D")
    users = np.arange(len(logs.users))
    a = make().fit(build_interaction_data(logs, cutoff)).score(users)
    perturbed = perturb_after(logs, cutoff, seed=4)
    b = make().fit(build_interaction_data(perturbed, cutoff)).score(users)
    assert np.array_equal(a, b)


def test_exposure_bpr_is_leakage_safe(small_scenario: RecommendationScenario) -> None:
    logs = small_scenario.logs
    cutoff = logs.start + np.timedelta64(98, "D")
    users = np.arange(len(logs.users))
    a = BPR(4, 0.01, epochs=3, exposure_logs=logs).fit(build_interaction_data(logs, cutoff))
    perturbed = perturb_after(logs, cutoff, seed=6)
    b = BPR(4, 0.01, epochs=3, exposure_logs=perturbed).fit(
        build_interaction_data(perturbed, cutoff))
    assert np.array_equal(a.score(users), b.score(users))


def test_invalid_hyperparameters_are_refused() -> None:
    with pytest.raises(ValueError):
        fit_ease(COUNTS, 0.0)
    with pytest.raises(ValueError):
        fit_bpr(COUNTS, factors=2, reg=0.01, learning_rate=0.0, epochs=1)
