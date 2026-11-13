"""Candidates, point-in-time features and rankers against independent references (M9)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from decision_intelligence.data.recommendation_scenarios import (
    RecommendationLogs,
    RecommendationScenario,
)
from decision_intelligence.recommendations import baselines as bl
from decision_intelligence.recommendations import off_policy as op
from decision_intelligence.recommendations.candidates import (
    allowed_mask,
    candidate_recall,
    generate_candidates,
)
from decision_intelligence.recommendations.cases import perturb_after
from decision_intelligence.recommendations.evaluation import top_k
from decision_intelligence.recommendations.features import ranking_features
from decision_intelligence.recommendations.interactions import (
    InteractionData,
    build_interaction_data,
)
from decision_intelligence.recommendations.ranking import (
    NON_CANDIDATE_SCORE,
    LabelledCandidates,
    LambdaRankRanker,
    PointwiseRanker,
    TwoStageRecommender,
    delta_ndcg,
    ranks_within,
    training_data,
)
from decision_intelligence.statistics.logistic_regression import fit_logistic

Setup = tuple[RecommendationLogs, InteractionData, dict[str, bl.Recommender], np.ndarray,
              pd.Timestamp]


def _ndcg(gains: np.ndarray, order: np.ndarray) -> float:
    disc = 1.0 / np.log2(2.0 + np.arange(order.size))
    ideal = (np.sort(gains)[::-1] * disc).sum()
    return float((gains[order] * disc).sum() / ideal)


def test_delta_ndcg_matches_brute_force_swaps() -> None:
    rng = np.random.default_rng(1)
    gains = (rng.random(9) < 0.4).astype(float)
    gains[0] = 1.0
    scores = rng.normal(size=9)
    order = np.argsort(-scores, kind="stable")
    ranks = ranks_within(scores)
    pos, neg = np.flatnonzero(gains > 0), np.flatnonzero(gains == 0)
    d = delta_ndcg(gains, ranks, pos, neg)
    base = _ndcg(gains, order)
    for a, i in enumerate(pos):
        for b, j in enumerate(neg):
            swapped = order.copy()
            pi, pj = np.flatnonzero(order == i)[0], np.flatnonzero(order == j)[0]
            swapped[pi], swapped[pj] = j, i
            assert d[a, b] == pytest.approx(abs(_ndcg(gains, swapped) - base), abs=1e-12)


def test_ranks_within_breaks_ties_by_index() -> None:
    assert ranks_within(np.array([1.0, 2.0, 2.0, 0.0])).tolist() == [3, 1, 2, 4]


@pytest.fixture(scope="module")
def setup(small_scenario: RecommendationScenario) -> Setup:
    logs = small_scenario.logs
    h70 = build_interaction_data(logs, logs.start + pd.to_timedelta(70, unit="D"))
    gens = {"pop": bl.GlobalPopularity().fit(h70),
            "knn": bl.Fallback(bl.ItemKNN(20), bl.SegmentPopularity(10.0)).fit(h70)}
    theta = op.estimate_examination(logs, h70.cutoff).relative
    end = logs.start + pd.to_timedelta(84, unit="D")
    return logs, h70, gens, theta, end


def test_candidates_are_the_union_of_allowed_top_n(setup: Setup) -> None:
    logs, h, gens, _, _ = setup
    users = np.arange(0, 300, 7)
    c = generate_candidates(h, users, gens, 15)
    allowed = allowed_mask(h, np.sort(users))
    member = c.membership(h.n_items)
    union = np.zeros_like(member)
    for g in gens.values():
        t = top_k(g.score(np.sort(users)), allowed, 15)
        union[np.arange(users.size)[:, None], t] = True
    assert np.array_equal(member, union)
    assert not np.any(member & ~allowed)
    rel = np.zeros_like(member)
    rel[:, :40] = True
    rec = candidate_recall(c, rel)
    assert np.allclose(rec["union"], (member & rel).sum(1) / rel.sum(1))
    other = {"x": bl.GlobalPopularity().fit(build_interaction_data(logs, logs.end))}
    with pytest.raises(ValueError, match="different cutoff"):
        generate_candidates(h, users, other, 5)


def test_features_ignore_post_cutoff_data(setup: Setup) -> None:
    logs, h, gens, _, _ = setup
    users = np.arange(0, 300, 11)
    a = ranking_features(logs, h, generate_candidates(h, users, gens, 10))
    p2 = perturb_after(logs, h.cutoff, seed=3)
    h2 = build_interaction_data(p2, h.cutoff)
    gens2 = {"pop": bl.GlobalPopularity().fit(h2),
             "knn": bl.Fallback(bl.ItemKNN(20), bl.SegmentPopularity(10.0)).fit(h2)}
    b = ranking_features(p2, h2, generate_candidates(h2, users, gens2, 10))
    pd.testing.assert_frame_equal(a.frame, b.frame)
    assert (a.frame["item_age"] >= 0).all()


def test_labels_and_propensity_weights(setup: Setup) -> None:
    logs, h, gens, theta, end = setup
    naive = training_data(logs, h, gens, end, theta, 15, propensity_weighted=False)
    ipw = training_data(logs, h, gens, end, theta, 15, propensity_weighted=True)
    assert np.array_equal(naive.label, ipw.label) and np.all(naive.weight == 1.0)
    assert np.all(ipw.weight[ipw.label == 0] == 1.0)
    assert np.all(ipw.weight[ipw.label > 0] >= 1.0)
    # each positive's weight is 1 / theta at the position of its first click in the window
    imp = logs.impressions
    win = imp.loc[imp["clicked"] & (imp["timestamp"] > h.cutoff) & (imp["timestamp"] <= end)]
    first = win.groupby(["user_id", "item_id"])["position"].min()
    f = ipw.features.frame
    for row in np.flatnonzero(ipw.label > 0)[:30]:
        key = (int(h.user_ids[f["user_index"].iloc[row]]),
               int(h.item_ids[f["item_index"].iloc[row]]))
        assert ipw.weight[row] == pytest.approx(1.0 / theta[first[key] - 1])


def test_pointwise_ranker_matches_m3_logistic_regression(setup: Setup) -> None:
    logs, h, gens, theta, end = setup
    data = training_data(logs, h, gens, end, theta, 15, propensity_weighted=False)
    raw = data.features.matrix()
    keep = raw.std(axis=0) > 0
    data_k = LabelledCandidates(
        type(data.features)(data.features.cutoff, data.features.frame,
                            tuple(n for n, k in zip(data.features.feature_names, keep,
                                                    strict=True) if k)),
        data.label, data.weight, data.query)
    ranker = PointwiseRanker(0.0).fit(data_k)
    ref = fit_logistic(raw[:, keep], data.label)
    eta_ref = ref.coefficients[0] + raw[:, keep] @ ref.coefficients[1:]
    assert ranker.converged
    assert np.allclose(ranker.predict(raw[:, keep]), eta_ref, atol=1e-6)


def test_lambdarank_gradient_matches_finite_differences(setup: Setup) -> None:
    logs, h, gens, theta, end = setup
    data = training_data(logs, h, gens, end, theta, 10, propensity_weighted=True)
    r = LambdaRankRanker(0.1)
    x = (data.features.matrix() - data.features.matrix().mean(0)) / (
        data.features.matrix().std(0) + 1e-12)
    groups = r._groups(data)
    w = np.random.default_rng(5).normal(size=x.shape[1]) * 0.3
    _, grad = r.gradient(w, x, data, groups)
    eps = 1e-7
    for k in range(x.shape[1]):
        e = np.zeros_like(w)
        e[k] = eps
        up = r.gradient(w + e, x, data, groups)[0]
        down = r.gradient(w - e, x, data, groups)[0]
        assert grad[k] == pytest.approx((up - down) / (2 * eps), rel=1e-4, abs=1e-6)


def test_lambdarank_weight_rescaling_is_scale_only(setup: Setup) -> None:
    logs, h, gens, theta, end = setup
    data = training_data(logs, h, gens, end, theta, 10, propensity_weighted=True)
    scaled = LabelledCandidates(data.features, data.label, data.weight * 7.0, data.query)
    a = LambdaRankRanker(0.1, epochs=5).fit(data)
    b = LambdaRankRanker(0.1, epochs=5).fit(scaled)
    assert np.allclose(a.weights, b.weights, atol=1e-12)
    assert a.ndcg_trace[-1] >= a.ndcg_trace[0]


def test_two_stage_scores_candidates_and_floors_the_rest(setup: Setup) -> None:
    logs, h, gens, theta, end = setup
    data = training_data(logs, h, gens, end, theta, 10, propensity_weighted=False)
    ranker = PointwiseRanker(1.0).fit(data)
    factories = {"pop": bl.GlobalPopularity,
                 "knn": lambda: bl.Fallback(bl.ItemKNN(20), bl.SegmentPopularity(10.0))}
    model = TwoStageRecommender(factories, ranker, logs, 10).fit(h)
    users = np.array([5, 1, 9])
    s = model.score(users)
    member = generate_candidates(h, users, model.fitted, 10).membership(h.n_items)
    order = np.argsort(users)
    assert np.all(s[order][~member] == NON_CANDIDATE_SCORE)
    assert np.all(s[order][member] > NON_CANDIDATE_SCORE)
    assert np.array_equal(s, model.score(users))


def test_ranker_validation() -> None:
    with pytest.raises(ValueError):
        PointwiseRanker(-1.0)
    with pytest.raises(ValueError):
        LambdaRankRanker(0.1, epochs=0)
