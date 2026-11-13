"""Blend hybrid, cold start, the simulated online period and the operational harness (M9)."""

from __future__ import annotations

import ast
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse, stats

from decision_intelligence.data.recommendation_experiments import (
    OnlineSessions,
    draw_sessions,
    simulate_online_period,
)
from decision_intelligence.data.recommendation_scenarios import RecommendationScenario
from decision_intelligence.recommendations import baselines as bl
from decision_intelligence.recommendations import online, operational
from decision_intelligence.recommendations import protocol as pr
from decision_intelligence.recommendations.candidates import allowed_mask
from decision_intelligence.recommendations.cold_start import (
    ContentToFactor,
    history_bucket,
    ridge_map,
)
from decision_intelligence.recommendations.evaluation import top_k
from decision_intelligence.recommendations.factorisation import ImplicitALS
from decision_intelligence.recommendations.hybrid import (
    BlendRecommender,
    percentile_ranks,
    simplex_grid,
)
from decision_intelligence.recommendations.interactions import build_interaction_data

PACKAGE = Path(online.__file__).resolve().parent


@pytest.mark.parametrize("module", ["factorisation", "two_tower", "retrieval", "candidates",
                                    "features", "ranking", "hybrid", "cold_start",
                                    "off_policy_advanced", "operational", "protocol"])
def test_m9_training_and_selection_modules_never_touch_ground_truth(module: str) -> None:
    tree = ast.parse((PACKAGE / f"{module}.py").read_text())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert not names & {"RecommendationTruth", "RecommendationScenario", "oracle"}


def test_percentile_ranks_and_simplex_grid() -> None:
    scores = np.array([[3.0, 1.0, 1.0, 9.0], [0.0, 0.0, 5.0, 2.0]])
    allowed = np.array([[True, True, True, False], [True, True, True, True]])
    r = percentile_ranks(scores, allowed)
    assert np.allclose(r[0, :3], stats.rankdata([3.0, 1.0, 1.0]) / 3) and r[0, 3] == 0
    assert np.allclose(r[1], stats.rankdata(scores[1]) / 4)
    grid = simplex_grid(4, pr.BLEND_STEP)
    assert len(grid) == comb(4 + 3, 3) == 35
    assert all(abs(sum(w) - 1) < 1e-12 for w in grid)
    with pytest.raises(ValueError):
        simplex_grid(3, 0.3)


def test_blend_with_one_component_reproduces_its_ranking(
    small_scenario: RecommendationScenario,
) -> None:
    logs = small_scenario.logs
    h = build_interaction_data(logs, logs.start + pd.to_timedelta(98, unit="D"))
    users = np.arange(0, 300, 13)
    allowed = allowed_mask(h, users)
    blend = BlendRecommender({"pop": bl.GlobalPopularity,
                              "knn": lambda: bl.ItemKNN(20)}, (0.0, 1.0)).fit(h)
    knn = bl.ItemKNN(20).fit(h)
    a = top_k(blend.score(users), allowed, 10)
    b = top_k(percentile_ranks(knn.score(users), allowed), allowed, 10)
    assert np.array_equal(a, b)
    with pytest.raises(ValueError):
        BlendRecommender({"pop": bl.GlobalPopularity}, (0.5,))


def test_history_buckets_ridge_map_and_content_to_factor(
    small_scenario: RecommendationScenario,
) -> None:
    labels = history_bucket(np.array([0, 1, 2, 3, 5, 6, 40]))
    assert labels.tolist() == ["0", "1-2", "1-2", "3-5", "3-5", "6+", "6+"]
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(30, 4)), rng.normal(size=(30, 2))
    ref = np.linalg.lstsq(np.vstack([x, np.sqrt(2.0) * np.eye(4)]),
                          np.vstack([y, np.zeros((4, 2))]), rcond=None)[0]
    assert np.allclose(ridge_map(x, y, 2.0), ref, atol=1e-10)
    logs = small_scenario.logs
    h = build_interaction_data(logs, logs.start + pd.to_timedelta(98, unit="D"))
    model = ContentToFactor(ImplicitALS(4, 1.0, 5.0, 3)).fit(h)
    new = np.flatnonzero(~h.item_launched)
    s = model.score(np.array([10, 20]))
    assert np.ptp(s[:, new], axis=1).min() > 0  # new items are distinguished


def _period(sc: RecommendationScenario, k: int = 5
            ) -> tuple[np.ndarray, dict[str, np.ndarray], OnlineSessions, np.ndarray]:
    users = np.arange(0, 200)
    rng = np.random.default_rng(1)
    lists = {"control": np.array([rng.choice(sc.truth.item_ids.size, k, replace=False)
                                  for _ in users]),
             "treatment": np.array([rng.choice(sc.truth.item_ids.size, k, replace=False)
                                    for _ in users])}
    sessions = draw_sessions(sc, users, 120.0, 14.0, seed=3)
    arm = np.where(users % 2 == 0, "control", "treatment")
    return users, lists, sessions, arm


def test_online_period_expected_outcomes_match_a_session_loop(
    small_scenario: RecommendationScenario,
) -> None:
    sc = small_scenario
    users, lists, sessions, arm = _period(sc)
    period = simulate_online_period(sc, users, arm, lists, sessions, click_seed=4)
    t = sc.truth
    exp = np.zeros(users.size)
    for row, day in zip(sessions.user_row, sessions.day, strict=True):
        a = t.attractiveness(np.array([users[row]]), np.array([day]))[0]
        exp[row] += float(np.sum(t.examination[:5] * a[lists["treatment"][row]]))
    assert np.allclose(period.expected["treatment"]["clicks"], exp, atol=1e-12)
    assert np.array_equal(period.sessions, np.bincount(sessions.user_row, minlength=users.size))
    same = simulate_online_period(sc, users, arm, {"control": lists["control"],
                                                   "treatment": lists["control"]}, sessions, 4)
    assert same.sate("treatment", "control", "clicks") == 0.0
    big = simulate_online_period(sc, users, np.full(users.size, "treatment"), lists,
                                 sessions, 5)
    z = (big.clicks.sum() - exp.sum()) / np.sqrt(exp.sum())
    assert abs(z) < 4  # observed clicks agree with the expectation


def test_online_period_validates_inputs(small_scenario: RecommendationScenario) -> None:
    sc = small_scenario
    users, lists, sessions, arm = _period(sc)
    with pytest.raises(ValueError):
        draw_sessions(sc, users, 120.0, 0.0, seed=1)
    with pytest.raises(ValueError):
        simulate_online_period(sc, users, arm[:-1], lists, sessions, 1)
    with pytest.raises(ValueError):
        simulate_online_period(sc, users, np.full(users.size, "other"), lists, sessions, 1)


def test_run_ab_uses_bonferroni_intervals_and_m6_decisions(
    small_scenario: RecommendationScenario,
) -> None:
    sc = small_scenario
    h = build_interaction_data(sc.logs, sc.logs.end)
    users = online.experiment_users(sc, h.cutoff)
    pop = online.policy_lists(bl.GlobalPopularity().fit(h), h, users, pr.K)
    knn = online.policy_lists(bl.Fallback(bl.ItemKNN(20), bl.SegmentPopularity(10.0)).fit(h),
                              h, users, pr.K)
    res = online.run_ab(sc, h, {"control": pop, "single_model": knn, "hybrid": pop}, users,
                        pr.AB_DESIGN)
    assert sum(res.arm_counts.values()) == users.size
    for c in res.comparisons:
        primary = c.results[pr.AB_DESIGN.primary_metric]
        assert primary.effect.confidence_level == pytest.approx(1 - pr.AB_DESIGN.alpha / 2)
        assert c.bonferroni_p == pytest.approx(min(1.0, 2 * c.p_value))
        assert c.holm_p >= c.p_value
        assert c.decision in {"LAUNCH", "ITERATE", "HOLD", "STOP"}
    hybrid = next(c for c in res.comparisons if c.treatment == "hybrid")
    assert hybrid.sate["clicks per user"] == 0.0  # identical lists to control


def test_pre_period_clicks_use_only_the_pre_period(small_scenario: RecommendationScenario
                                                   ) -> None:
    sc = small_scenario
    h = build_interaction_data(sc.logs, sc.logs.end)
    users = np.arange(len(sc.logs.users))
    got = online.pre_period_clicks(sc, h, users, 14.0)
    imp = sc.logs.impressions
    win = imp.loc[imp["clicked"] & (imp["timestamp"] > h.cutoff - pd.to_timedelta(14, unit="D"))]
    ref = win.groupby("user_id").size().reindex(h.user_ids, fill_value=0).to_numpy()
    assert np.array_equal(got, ref)


def test_scale_matrix_and_preflight_skips_unsafe_workloads() -> None:
    x = operational.scale_click_matrix(200, 50, seed=1)
    assert isinstance(x, sparse.csr_matrix) and x.shape == (200, 50)
    assert (x != operational.scale_click_matrix(200, 50, seed=1)).nnz == 0
    assert operational.planned_bytes("item-kNN (dense M8)", 30_000, 5000) == (
        30_000 * 5000 * 8 + 3 * 5000 * 5000 * 8 + 1000 * 5000 * 8)
    row = operational.benchmark_model("PureSVD (dense M8)", x, budget=1000)
    assert row.status.startswith("SKIPPED") and np.isnan(row.fit_seconds)
    ok = operational.benchmark_model("EASE", x, budget=10**9, requests=5)
    assert ok.status == "RUN" and ok.model_bytes > 0 and ok.latency_p95_ms >= ok.latency_p50_ms
    with pytest.raises(ValueError):
        operational.planned_bytes("unknown", 1, 1)


def test_retrieval_benchmark_reports_exact_recall_one() -> None:
    rng = np.random.default_rng(2)
    rows = operational.retrieval_benchmark(rng.normal(size=(20, 4)), rng.normal(size=(100, 4)),
                                           k=5, clusters=(4,), probes=(1, 4))
    assert rows[0].method == "exact" and rows[0].recall_of_exact == 1.0
    full = [r for r in rows if r.n_probe == 4][0]
    assert full.recall_of_exact == pytest.approx(1.0)
