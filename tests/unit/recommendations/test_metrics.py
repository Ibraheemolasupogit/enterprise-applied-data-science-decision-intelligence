"""Ranking and beyond-accuracy metrics against hand-worked and brute-force references."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from decision_intelligence.recommendations import metrics as m

# One user, six items; ranked list (0, 1, 2, 3); relevant items {1, 3, 5}.
RECS = np.array([[0, 1, 2, 3]])
REL = np.zeros((1, 6), dtype=bool)
REL[0, [1, 3, 5]] = True


def test_precision_recall_hit_rate_hand_worked() -> None:
    assert m.precision_at_k(RECS, REL)[0] == pytest.approx(2 / 4)
    assert m.recall_at_k(RECS, REL)[0] == pytest.approx(2 / 3)
    assert m.hit_rate_at_k(RECS, REL)[0] == 1.0


def test_average_precision_hand_worked() -> None:
    # Hits at ranks 2 and 4: (1 / min(3, 4)) * (1/2 + 2/4) = 1/3.
    assert m.average_precision_at_k(RECS, REL)[0] == pytest.approx(1 / 3)


def test_reciprocal_rank_hand_worked() -> None:
    assert m.reciprocal_rank_at_k(RECS, REL)[0] == pytest.approx(0.5)
    none = np.zeros_like(REL)
    assert m.reciprocal_rank_at_k(RECS, none)[0] == 0.0


def test_ndcg_hand_worked() -> None:
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert m.ndcg_at_k(RECS, REL.astype(float))[0] == pytest.approx(dcg / idcg, abs=1e-12)


def _ap_by_definition(ranked: list[int], relevant: set[int], k: int) -> float:
    hits, total = 0, 0.0
    for rank, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            hits += 1
            total += hits / rank
    return total / min(len(relevant), k)


def test_average_precision_matches_definition_over_all_permutations() -> None:
    relevant = {1, 3, 4}
    rel = np.zeros((1, 6), dtype=bool)
    rel[0, list(relevant)] = True
    for perm in itertools.permutations(range(6), 3):
        got = m.average_precision_at_k(np.array([perm]), rel)[0]
        assert got == pytest.approx(_ap_by_definition(list(perm), relevant, 3), abs=1e-12)


def test_ndcg_ideal_equals_best_permutation_by_brute_force() -> None:
    gains = np.array([[0.1, 0.9, 0.0, 0.5, 0.3, 0.7]])
    k = 3
    d = 1 / np.log2(np.arange(2, k + 2))
    best = max(float(gains[0, list(p)] @ d) for p in itertools.permutations(range(6), k))
    values = [m.ndcg_at_k(np.array([p]), gains)[0] for p in itertools.permutations(range(6), k)]
    assert max(values) == pytest.approx(1.0)
    assert min(values) >= 0.0
    ideal = np.array([[1, 5, 3]])
    assert float(gains[0, [1, 5, 3]] @ d) == pytest.approx(best)
    assert m.ndcg_at_k(ideal, gains)[0] == pytest.approx(1.0)


def test_ndcg_ideal_respects_allowed_items() -> None:
    gains = np.array([[1.0, 0.5, 0.2, 0.0]])
    allowed = np.array([[False, True, True, True]])  # item 0 was already seen
    assert m.ndcg_at_k(np.array([[1, 2]]), gains, allowed)[0] == pytest.approx(1.0)
    assert m.ndcg_at_k(np.array([[1, 2]]), gains)[0] < 1.0


def test_expected_precision_of_a_random_ranking_is_relevant_share() -> None:
    # Exact: average over every ordered K-subset of 6 items; E[P@K] = |R| / n.
    values = [m.precision_at_k(np.array([p]), REL)[0]
              for p in itertools.permutations(range(6), 3)]
    assert np.mean(values) == pytest.approx(3 / 6, abs=1e-12)


def test_recall_is_monotone_in_k_and_undefined_without_relevant_items() -> None:
    ranked = np.array([[5, 0, 1, 2, 3, 4]])
    recalls = [m.recall_at_k(ranked[:, :k], REL)[0] for k in range(1, 7)]
    assert all(b >= a for a, b in zip(recalls, recalls[1:], strict=False))
    assert recalls[-1] == pytest.approx(1.0)
    assert math.isnan(m.recall_at_k(RECS, np.zeros_like(REL))[0])


def test_gini_closed_forms_and_mean_difference_identity() -> None:
    assert m.gini(np.ones(5)) == pytest.approx(0.0)
    assert m.gini(np.array([0.0, 0.0, 0.0, 1.0])) == pytest.approx(3 / 4)
    x = np.random.default_rng(1).gamma(0.5, size=40)
    brute = np.abs(x[:, None] - x[None, :]).sum() / (2 * x.size ** 2 * x.mean())
    assert m.gini(x) == pytest.approx(brute, abs=1e-12)


def test_entropy_top_share_and_coverage() -> None:
    assert m.normalised_entropy(np.ones(8)) == pytest.approx(1.0)
    assert m.top_share(np.array([5.0, 1, 1, 1, 1, 1, 0, 0, 0, 0]), 0.1) == pytest.approx(0.5)
    eligible = np.array([True] * 5 + [False])
    assert m.catalogue_coverage(np.array([[0, 1], [1, 2]]), eligible) == pytest.approx(3 / 5)


def test_exposure_counts_and_position_weights() -> None:
    recs = np.array([[0, 1], [1, 2]])
    assert m.exposure(recs, 4).tolist() == [1, 2, 1, 0]
    w = m.exposure(recs, 4, position_weighted=True)
    assert w[1] == pytest.approx(1 + 1 / math.log2(3))


def test_novelty_hand_worked() -> None:
    counts = np.array([9.0, 0.0, 4.0])
    # n_users = 9: item 0 -> -log2(10/10) = 0; item 1 -> -log2(1/10).
    assert m.novelty(np.array([[0, 1]]), counts, 9)[0] == pytest.approx(math.log2(10) / 2)


def test_intra_list_diversity_matches_pairwise_loop() -> None:
    rng = np.random.default_rng(3)
    f = rng.random((12, 5))
    recs = np.array([rng.choice(12, size=4, replace=False) for _ in range(6)])
    fn = f / np.linalg.norm(f, axis=1, keepdims=True)
    for row, got in zip(recs, m.intra_list_diversity(recs, f), strict=True):
        sims = [fn[a] @ fn[b] for a, b in itertools.combinations(row, 2)]
        assert got == pytest.approx(1 - np.mean(sims), abs=1e-12)


def test_personalisation_matches_pairwise_jaccard_loop() -> None:
    rng = np.random.default_rng(4)
    recs = np.array([rng.choice(10, size=3, replace=False) for _ in range(7)])
    jac = [len(set(a) & set(b)) / len(set(a) | set(b))
           for a, b in itertools.combinations(recs.tolist(), 2)]
    assert m.personalisation(recs, 10) == pytest.approx(1 - np.mean(jac), abs=1e-12)
    same = np.tile(np.array([[1, 2, 3]]), (4, 1))
    assert m.personalisation(same, 10) == pytest.approx(0.0)


def test_rmse() -> None:
    assert m.rmse(np.array([1.0, 3.0]), np.array([2.0, 5.0])) == pytest.approx(math.sqrt(2.5))


@pytest.mark.parametrize("recs", [np.array([[0, 0]]), np.array([[0, 9]]), np.array([[-1, 0]])])
def test_invalid_lists_are_refused(recs: np.ndarray) -> None:
    with pytest.raises(ValueError):
        m.precision_at_k(recs, np.zeros((1, 6), dtype=bool))


def test_invalid_gains_and_shapes_are_refused() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        m.ndcg_at_k(RECS, -REL.astype(float))
    with pytest.raises(ValueError, match="same number of rows"):
        m.precision_at_k(RECS, np.zeros((2, 6), dtype=bool))
    with pytest.raises(ValueError):
        m.gini(np.array([-1.0, 1.0]))
    with pytest.raises(ValueError):
        m.intra_list_diversity(np.array([[0]]), np.ones((3, 2)))
