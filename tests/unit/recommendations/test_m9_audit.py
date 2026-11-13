"""Coverage added by the M9 scientific assurance audit (gaps against the acceptance criteria)."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from decision_intelligence.data.recommendation_scenarios import (
    RecommendationScenario,
    generate_recommendation_scenario,
)
from decision_intelligence.recommendations import baselines as bl
from decision_intelligence.recommendations import evaluation as ev
from decision_intelligence.recommendations import off_policy as op
from decision_intelligence.recommendations import off_policy_advanced as opa
from decision_intelligence.recommendations.cases import perturb_after
from decision_intelligence.recommendations.cold_start import ContentToFactor
from decision_intelligence.recommendations.factorisation import ImplicitALS
from decision_intelligence.recommendations.hybrid import BlendRecommender
from decision_intelligence.recommendations.interactions import (
    InteractionData,
    build_interaction_data,
)
from decision_intelligence.recommendations.protocol import FRESH_SEED as RESERVED_SEED
from decision_intelligence.recommendations.ranking import (
    PointwiseRanker,
    TwoStageRecommender,
    training_data,
)
from decision_intelligence.recommendations.splits import RankingTask, ranking_task

from .conftest import SMALL

TESTS = Path(__file__).resolve().parents[2]


def _day(sc: RecommendationScenario, d: float) -> pd.Timestamp:
    return sc.logs.start + pd.to_timedelta(d, unit="D")


def test_two_stage_blend_and_content_to_factor_ignore_post_cutoff_data(
    small_scenario: RecommendationScenario,
) -> None:
    """The multi-window pipeline: ranker trained on labels (70, 84] with generators at 70,
    applied with generators at 84. Rewriting everything after day 84 must change nothing."""
    gens = {"pop": bl.GlobalPopularity,
            "knn": lambda: bl.Fallback(bl.ItemKNN(20), bl.SegmentPopularity(10.0))}

    def scores(sc: RecommendationScenario) -> list[np.ndarray]:
        logs = sc.logs
        h70 = build_interaction_data(logs, _day(sc, 70))
        h84 = build_interaction_data(logs, _day(sc, 84))
        theta = op.estimate_examination(logs, h70.cutoff).relative
        data = training_data(logs, h70, {n: f().fit(h70) for n, f in gens.items()},
                             h84.cutoff, theta, 15, propensity_weighted=True)
        users = np.arange(len(logs.users))
        two = TwoStageRecommender(gens, PointwiseRanker(1.0).fit(data), logs, 15).fit(h84)
        blend = BlendRecommender(gens, (0.5, 0.5)).fit(h84)
        c2f = ContentToFactor(ImplicitALS(4, 1.0, 5.0, 3)).fit(h84)
        return [two.score(users), blend.score(users), c2f.score(users)]

    perturbed = replace(small_scenario,
                        logs=perturb_after(small_scenario.logs, _day(small_scenario, 84), 5))
    for a, b in zip(scores(small_scenario), scores(perturbed), strict=True):
        assert np.array_equal(a, b)


def _loop(sc: RecommendationScenario, h: InteractionData, task: RankingTask, recs: np.ndarray,
          theta: np.ndarray, a: pd.Timestamp, b: pd.Timestamp, q: np.ndarray
          ) -> tuple[float, float]:
    """SNIPS and DR by explicit loops, with the logging support and launch dates."""
    logs = sc.logs
    support = set(logs.logging_policy.support_item_ids or [])
    launch = logs.items.set_index("item_id")["launch_time"]
    lists = {int(u): list(r) for u, r in zip(task.users, recs, strict=True)}
    s = logs.sessions.loc[(logs.sessions["timestamp"] > a) & (logs.sessions["timestamp"] <= b)]
    imp = logs.impressions.set_index("session_id")
    L = logs.logging_policy.slate_size
    num = den = dr = 0.0
    for sid, uid, ts in zip(s["session_id"], s["user_id"], s["timestamp"], strict=True):
        u = int(uid) - 1
        lst = lists[u]
        m_s = sum(1 for it in lst if (it + 1) in support and launch[it + 1] <= ts)
        a_s = rho_sum = corr = 0.0
        for _, row in imp.loc[[sid]].iterrows():
            i = int(row["item_id"]) - 1
            p = int(row["position"])
            if i in lst:
                k = lst.index(i) + 1
                rho = 1.0 / row["logging_propensity"]
                r = theta[k - 1] / theta[p - 1]
                a_s += float(row["clicked"]) * r * rho / L
                rho_sum += rho / L
                corr += rho * r * (float(row["clicked"]) - theta[p - 1] * q[u, i]) / L
        num += a_s
        den += rho_sum / m_s if m_s > 0 else 1.0
        dr += sum(theta[k] * q[u, it] for k, it in enumerate(lst)) + corr
    return num / den, dr / len(s)


def test_snips_and_dr_match_loops_when_positivity_fails() -> None:
    sc = generate_recommendation_scenario(replace(SMALL, logging_support_size=25))
    logs = sc.logs
    h = build_interaction_data(logs, _day(sc, 98))
    task = ranking_task(logs, h, logs.end)
    recs = ev.recommend(bl.Fallback(bl.ContentBased(), bl.SegmentPopularity(10.0)).fit(h),
                        task, 5)
    theta = op.estimate_examination(logs, h.cutoff).relative
    q = np.random.default_rng(0).uniform(0.0, 0.5, size=(h.n_users, h.n_items))
    slots = opa.extract_slots(logs, h, task.users, recs, h.cutoff, logs.end)
    assert not slots.supported.all() and (slots.supported.sum(axis=1) == 0).any()
    sn_ref, dr_ref = _loop(sc, h, task, recs, theta, h.cutoff, logs.end, q)
    assert opa.snips(slots, theta).estimate == pytest.approx(sn_ref, rel=1e-10)
    assert opa.doubly_robust(slots, theta, q).estimate == pytest.approx(dr_ref, rel=1e-10)


def test_no_test_evaluates_the_m9_fresh_seed() -> None:
    """The fresh seed is reserved for the final evidence run (protocol)."""
    offenders = []
    for path in TESTS.rglob("test_*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == RESERVED_SEED:
                offenders.append(str(path))
            if isinstance(node, ast.Attribute) and node.attr == "FRESH_SEED":
                value = node.value
                if isinstance(value, ast.Name) and value.id in {"pr", "protocol"}:
                    offenders.append(str(path))
    assert offenders == []
