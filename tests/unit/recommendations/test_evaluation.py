"""Evaluation harness, IPS, oracle value and the explicit-ratings model against references."""

from __future__ import annotations

import ast
import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from decision_intelligence.data.recommendation_scenarios import (
    LoggingPolicySpec,
    RecommendationScenario,
)
from decision_intelligence.recommendations import baselines as bl
from decision_intelligence.recommendations import evaluation as ev
from decision_intelligence.recommendations import off_policy as op
from decision_intelligence.recommendations import oracle as orc
from decision_intelligence.recommendations.explicit import BiasRatingModel, RatingData
from decision_intelligence.recommendations.interactions import (
    InteractionData,
    build_interaction_data,
    index_of,
)
from decision_intelligence.recommendations.splits import (
    ChronologicalSplit,
    RankingTask,
    chronological_split,
    ranking_task,
)
from decision_intelligence.statistics.hypothesis_tests import paired_t_test

PACKAGE = Path(ev.__file__).resolve().parent
Fitted = tuple[RecommendationScenario, ChronologicalSplit, InteractionData, RankingTask,
               np.ndarray]


@pytest.fixture(scope="module")
def fitted(small_scenario: RecommendationScenario) -> Fitted:
    logs = small_scenario.logs
    sp = chronological_split(logs, train_days=84, validation_days=14)
    h = build_interaction_data(logs, sp.validation_end)
    task = ranking_task(logs, h, sp.test_end)
    recs = ev.recommend(bl.Fallback(bl.ItemKNN(20), bl.SegmentPopularity(10.0)).fit(h), task, 5)
    return small_scenario, sp, h, task, recs


@pytest.mark.parametrize("module", ["baselines", "interactions", "splits", "evaluation",
                                    "off_policy", "explicit", "metrics"])
def test_training_tuning_and_selection_modules_never_touch_ground_truth(module: str) -> None:
    tree = ast.parse((PACKAGE / f"{module}.py").read_text())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert "RecommendationTruth" not in names
    assert "oracle" not in names and "RecommendationScenario" not in names


def test_tuning_interface_accepts_only_logged_data() -> None:
    params = set(inspect.signature(ev.tune).parameters)
    assert params == {"candidates", "history", "validation", "k", "metric"}


def test_tune_selects_the_best_validation_configuration(fitted: Fitted) -> None:
    sc, sp, _, _, _ = fitted
    train = build_interaction_data(sc.logs, sp.train_end)
    val = ranking_task(sc.logs, train, sp.validation_end)
    grid = {"pop": [bl.GlobalPopularity, lambda: bl.RandomRecommender(1)]}
    res = ev.tune(grid, train, val, 5)
    table = res.table
    assert table["selected"].sum() == 1
    best = table.loc[table["validation_ndcg"].idxmax(), "model"]
    assert table.loc[table["selected"], "model"].item() == best == "global popularity"
    with pytest.raises(ValueError, match="history cutoff"):
        ev.tune(grid, build_interaction_data(sc.logs, sp.validation_end), val, 5)


def test_logged_metrics_only_score_users_with_relevant_items(fitted: Fitted) -> None:
    _, _, _, task, recs = fitted
    frame = ev.logged_metrics(recs, task)
    assert len(frame) == task.users_with_relevant().sum()
    assert frame[list(ev.LOGGED_METRICS)].notna().all().all()
    assert ((frame[list(ev.LOGGED_METRICS)] >= 0) & (frame[list(ev.LOGGED_METRICS)] <= 1)
            ).all().all()


def test_paired_comparisons_match_m3_and_holm_never_lowers_p_values() -> None:
    rng = np.random.default_rng(5)
    base = rng.random(200)
    per_user = {"ref": base, "a": base + 0.05 + rng.normal(0, 0.1, 200),
                "b": base + rng.normal(0, 0.1, 200), "c": base - 0.02 + rng.normal(0, 0.1, 200)}
    table = ev.paired_comparisons(per_user, "ref")
    for _, row in table.iterrows():
        res = paired_t_test(per_user[row["model"]], base)
        assert row["p_value"] == pytest.approx(res.p_value)
        assert row["difference"] == pytest.approx(res.estimate)
    assert np.all(table["adjusted_p_value"] >= table["p_value"])
    with pytest.raises(ValueError):
        ev.paired_comparisons(per_user, "missing")


def test_bootstrap_summary_brackets_the_mean() -> None:
    x = np.random.default_rng(6).random(300)
    s = ev.summarise(x, rng=np.random.default_rng(7))
    assert s.ci_lower < s.mean < s.ci_upper and s.n == 300


def _ips_by_loop(sc: RecommendationScenario, h: InteractionData, task: RankingTask,
                 recs: np.ndarray, after: pd.Timestamp, until: pd.Timestamp, theta: np.ndarray,
                 clip: float | None = None) -> float:
    logs = sc.logs
    lists = {int(u): list(r) for u, r in zip(task.users, recs, strict=True)}
    sessions = logs.sessions.loc[(logs.sessions["timestamp"] > after)
                                 & (logs.sessions["timestamp"] <= until)]
    imp = logs.impressions.set_index("session_id")
    total = 0.0
    for sid, uid in zip(sessions["session_id"], sessions["user_id"], strict=True):
        u = int(index_of(h.user_ids, np.array([uid]))[0])
        value = 0.0
        for _, row in imp.loc[[sid]].iterrows():
            i = int(index_of(h.item_ids, np.array([row["item_id"]]))[0])
            if i in lists[u]:
                k = lists[u].index(i) + 1
                w = theta[k - 1] / theta[int(row["position"]) - 1] / row["logging_propensity"]
                if clip is not None:
                    w = min(w, clip)
                value += float(row["clicked"]) * w
        total += value / logs.logging_policy.slate_size
    return total / len(sessions)


def test_ips_matches_an_explicit_loop_over_logged_slots(fitted: Fitted) -> None:
    sc, sp, h, task, recs = fitted
    theta = op.estimate_examination(sc.logs, sp.validation_end).relative
    for clip in (None, 30.0):
        res = op.ips_policy_value(sc.logs, h, task.users, recs, sp.validation_end, sp.test_end,
                                  theta, clip=clip)
        expected = _ips_by_loop(sc, h, task, recs, sp.validation_end, sp.test_end, theta, clip)
        assert res.estimate == pytest.approx(expected, rel=1e-10)
    assert res.unsupported_share == 0.0
    assert 0 < res.effective_sample_size <= res.n_slots


def test_clipping_never_raises_the_estimate(fitted: Fitted) -> None:
    sc, sp, h, task, recs = fitted
    theta = op.estimate_examination(sc.logs, sp.validation_end).relative
    values = [op.ips_policy_value(sc.logs, h, task.users, recs, sp.validation_end, sp.test_end,
                                  theta, clip=c).estimate for c in (None, 1e9, 100.0, 10.0, 1.0)]
    assert values[0] == pytest.approx(values[1])
    assert all(b <= a + 1e-15 for a, b in zip(values, values[1:], strict=False))


def test_ips_refuses_invalid_settings(fitted: Fitted) -> None:
    sc, sp, h, task, recs = fitted
    theta = op.estimate_examination(sc.logs, sp.validation_end).relative
    with pytest.raises(ValueError, match="positive"):
        op.ips_policy_value(sc.logs, h, task.users, recs, sp.validation_end, sp.test_end, theta,
                            clip=0.0)
    with pytest.raises(ValueError, match="slate"):
        op.ips_policy_value(sc.logs, h, task.users, np.tile(np.arange(11), (task.users.size, 1)),
                            sp.validation_end, sp.test_end, theta)
    with pytest.raises(ValueError, match="target list"):
        op.ips_policy_value(sc.logs, h, task.users[:5], recs[:5], sp.validation_end,
                            sp.test_end, theta)
    unknown = replace(sc.logs, logging_policy=LoggingPolicySpec(0.1, 10, None, "unknown"))
    with pytest.raises(ValueError, match="independent"):
        op.estimate_examination(unknown, sp.validation_end)
    bad = sc.logs.impressions.copy()
    bad["logging_propensity"] = 0.0
    with pytest.raises(ValueError, match="positive"):
        op.ips_policy_value(sc.logs, h, task.users, recs, sp.validation_end, sp.test_end, theta,
                            impressions=bad)


def test_examination_estimate_is_relative_to_position_one(fitted: Fitted) -> None:
    sc, sp, _, _, _ = fitted
    est = op.estimate_examination(sc.logs, sp.validation_end)
    assert est.relative[0] == 1.0
    assert est.positions.tolist() == list(range(1, 11))


def test_simulated_online_value_matches_a_loop_over_sessions(fitted: Fitted) -> None:
    sc, sp, h, task, recs = fitted
    truth = sc.truth
    val = orc.simulated_online_value(sc.logs, h, truth, task.users, recs, sp.validation_end,
                                     sp.test_end)
    lists = {int(u): r for u, r in zip(task.users, recs, strict=True)}
    s = sc.logs.sessions
    s = s.loc[(s["timestamp"] > sp.validation_end) & (s["timestamp"] <= sp.test_end)]
    expected = []
    for uid, ts in zip(s["user_id"], s["timestamp"], strict=True):
        u = int(index_of(h.user_ids, np.array([uid]))[0])
        t = truth.time_in_days(pd.Timestamp(ts))
        a = truth.attractiveness(np.array([u]), t)[0]
        expected.append(float(np.sum(truth.examination[:5] * a[lists[u]])))
    assert np.allclose(val.per_session, expected, atol=1e-12)
    assert val.estimate == pytest.approx(np.mean(expected))
    sup = orc.simulated_online_value(sc.logs, h, truth, task.users, recs, sp.validation_end,
                                     sp.test_end, supported_only=True)
    assert sup.estimate == pytest.approx(val.estimate)  # full logging support


def test_purchase_and_margin_values_are_ordered_consistently(fitted: Fitted) -> None:
    sc, sp, h, task, recs = fitted
    vals = {r: orc.simulated_online_value(sc.logs, h, sc.truth, task.users, recs,
                                          sp.validation_end, sp.test_end, reward=r).estimate
            for r in ("clicked", "purchased", "margin")}
    assert 0 < vals["purchased"] < vals["clicked"]
    assert vals["margin"] > 0


def test_oracle_ndcg_is_one_for_the_ideal_list(fitted: Fitted) -> None:
    sc, _, h, task, _ = fitted
    gains = orc.oracle_gains(sc.truth, h, task)
    ideal = ev.top_k(gains, task.allowed, 5)
    frame = orc.oracle_metrics(ideal, task, sc.truth, h)
    assert np.allclose(frame["oracle_ndcg"], 1.0) and np.allclose(frame["oracle_precision"], 1.0)


def test_oracle_refuses_misaligned_truth(fitted: Fitted) -> None:
    sc, _, h, task, recs = fitted
    bad = replace(sc.truth, user_ids=sc.truth.user_ids + 1)
    with pytest.raises(ValueError, match="different users"):
        orc.oracle_metrics(recs, task, bad, h)


def test_bias_model_matches_direct_least_squares() -> None:
    rng = np.random.default_rng(8)
    n_users, n_items, n = 15, 10, 120
    u, i = rng.integers(0, n_users - 2, n), rng.integers(0, n_items - 2, n)
    r = np.clip(np.round(3 + rng.normal(size=n)), 1, 5)
    data = RatingData(u, i, r.astype(float), n_users, n_items)
    lam_u, lam_i = 3.0, 2.0
    model = BiasRatingModel(lam_u, lam_i).fit(data)
    # Augmented least squares: [1 | U | I] beta = r, plus ridge rows on the biases only.
    design = np.zeros((n, 1 + n_users + n_items))
    design[:, 0] = 1.0
    design[np.arange(n), 1 + u] = 1.0
    design[np.arange(n), 1 + n_users + i] = 1.0
    ridge = np.zeros((n_users + n_items, 1 + n_users + n_items))
    ridge[:n_users, 1:1 + n_users] = np.sqrt(lam_u) * np.eye(n_users)
    ridge[n_users:, 1 + n_users:] = np.sqrt(lam_i) * np.eye(n_items)
    beta = np.linalg.lstsq(np.vstack([design, ridge]),
                           np.concatenate([r, np.zeros(n_users + n_items)]), rcond=None)[0]
    assert model.mu == pytest.approx(beta[0], abs=1e-8)
    assert np.allclose(model.user_bias, beta[1:1 + n_users], atol=1e-8)
    assert np.allclose(model.item_bias, beta[1 + n_users:], atol=1e-8)
    # Users and items without ratings get a zero bias (prediction = mu).
    assert model.user_bias[-1] == 0.0 and model.item_bias[-1] == 0.0
    with pytest.raises(ValueError):
        BiasRatingModel(0.0, 1.0)


def test_bias_model_is_exact_on_weakly_regularised_chain_coupled_ratings() -> None:
    # Regression (M8 assurance review): block coordinate descent stopped at its iteration cap
    # with bias errors of ~5e-3 here (user u rates items u and u + 1, so every bias is coupled
    # to every other one), without reporting non-convergence.
    n, lam = 60, 1e-3
    u = np.repeat(np.arange(n), 2).astype(np.int64)
    i = np.concatenate([[k, k + 1] for k in range(n)]).astype(np.int64)
    r = (1.0 + np.arange(2 * n) % 5).astype(float)
    data = RatingData(u, i, r, n, n + 1)
    model = BiasRatingModel(lam, lam).fit(data)
    design = np.zeros((r.size, 1 + n + n + 1))
    design[:, 0] = 1.0
    design[np.arange(r.size), 1 + u] = 1.0
    design[np.arange(r.size), 1 + n + i] = 1.0
    normal = design.T @ design + np.diag(np.r_[0.0, np.full(2 * n + 1, lam)])
    beta = np.linalg.solve(normal, design.T @ r)
    assert model.mu == pytest.approx(beta[0], abs=1e-9)
    assert np.allclose(model.user_bias, beta[1:1 + n], atol=1e-9)
    assert np.allclose(model.item_bias, beta[1 + n:], atol=1e-9)
    with pytest.raises(ValueError, match="no ratings"):
        BiasRatingModel().fit(RatingData(u[:0], i[:0], r[:0], n, n + 1))
