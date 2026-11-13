"""SNIPS, DR, cluster-bootstrap intervals and the out-of-sample click model (Milestone 9)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from decision_intelligence.data.recommendation_scenarios import RecommendationScenario
from decision_intelligence.recommendations import baselines as bl
from decision_intelligence.recommendations import evaluation as ev
from decision_intelligence.recommendations import off_policy as op
from decision_intelligence.recommendations import off_policy_advanced as opa
from decision_intelligence.recommendations.factorisation import ImplicitALS
from decision_intelligence.recommendations.interactions import (
    InteractionData,
    build_interaction_data,
)
from decision_intelligence.recommendations.splits import RankingTask, ranking_task

Fixture = tuple[RecommendationScenario, InteractionData, RankingTask, np.ndarray, np.ndarray,
                pd.Timestamp, pd.Timestamp]


@pytest.fixture(scope="module")
def setup(small_scenario: RecommendationScenario) -> Fixture:
    logs = small_scenario.logs
    after = logs.start + pd.to_timedelta(98, unit="D")
    h = build_interaction_data(logs, after)
    task = ranking_task(logs, h, logs.end)
    recs = ev.recommend(bl.Fallback(bl.ItemKNN(20), bl.SegmentPopularity(10.0)).fit(h), task, 5)
    theta = op.estimate_examination(logs, after).relative
    return small_scenario, h, task, recs, theta, after, logs.end


def test_ips_reproduces_m8(setup: Fixture) -> None:
    sc, h, task, recs, theta, a, b = setup
    m8 = op.ips_policy_value(sc.logs, h, task.users, recs, a, b, theta)
    new = opa.ips(opa.extract_slots(sc.logs, h, task.users, recs, a, b), theta)
    assert new.estimate == pytest.approx(m8.estimate, rel=1e-12)
    assert new.standard_error == pytest.approx(m8.standard_error, rel=1e-12)


def _loop(sc: RecommendationScenario, h: InteractionData, task: RankingTask, recs: np.ndarray,
          theta: np.ndarray, a: pd.Timestamp, b: pd.Timestamp, q: np.ndarray
          ) -> tuple[float, float]:
    """SNIPS and DR by explicit loops over sessions and slots."""
    logs = sc.logs
    lists = {int(u): list(r) for u, r in zip(task.users, recs, strict=True)}
    s = logs.sessions.loc[(logs.sessions["timestamp"] > a) & (logs.sessions["timestamp"] <= b)]
    imp = logs.impressions.set_index("session_id")
    L = logs.logging_policy.slate_size
    num = den = dr = 0.0
    for sid, uid in zip(s["session_id"], s["user_id"], strict=True):
        u = int(uid) - 1
        lst = lists[u]
        a_s = d_s = corr = 0.0
        for _, row in imp.loc[[sid]].iterrows():
            i = int(row["item_id"]) - 1
            p = int(row["position"])
            if i in lst:
                k = lst.index(i) + 1
                rho = 1.0 / row["logging_propensity"]
                r = theta[k - 1] / theta[p - 1]
                a_s += float(row["clicked"]) * r * rho / L
                d_s += rho / L
                corr += rho * r * (float(row["clicked"]) - theta[p - 1] * q[u, i]) / L
        num += a_s
        den += d_s / len(lst)  # full support: every list item can be shown
        dr += sum(theta[k] * q[u, it] for k, it in enumerate(lst)) + corr
    return num / den, dr / len(s)


def test_snips_and_dr_match_explicit_loops(setup: Fixture) -> None:
    sc, h, task, recs, theta, a, b = setup
    q = np.random.default_rng(0).uniform(0.0, 0.5, size=(h.n_users, h.n_items))
    slots = opa.extract_slots(sc.logs, h, task.users, recs, a, b)
    sn_ref, dr_ref = _loop(sc, h, task, recs, theta, a, b, q)
    assert opa.snips(slots, theta).estimate == pytest.approx(sn_ref, rel=1e-10)
    assert opa.doubly_robust(slots, theta, q).estimate == pytest.approx(dr_ref, rel=1e-10)
    zero = opa.doubly_robust(slots, theta, np.zeros_like(q))
    assert zero.estimate == pytest.approx(opa.ips(slots, theta).estimate, rel=1e-12)


def test_clustered_ratio_standard_error_and_intervals() -> None:
    rng = np.random.default_rng(1)
    a, d = rng.gamma(2.0, size=50), rng.integers(1, 4, size=50).astype(float)
    cr = opa.ClusteredRatio(a, d)
    v = a.sum() / d.sum()
    se = np.sqrt(50 / 49 * np.sum((a - v * d) ** 2)) / d.sum()
    assert cr.estimate == pytest.approx(v) and cr.standard_error == pytest.approx(se)
    boot = cr.bootstrap_intervals(np.random.default_rng(2), 999)
    for lo, hi in boot.values():
        assert lo < v < hi
    nlo, nhi = cr.normal_interval()
    assert nlo == pytest.approx(v - 1.959964 * se, rel=1e-6)


def test_bootstrap_is_reproducible_with_a_seed() -> None:
    cr = opa.ClusteredRatio(np.arange(1.0, 21.0), np.ones(20))
    a = cr.bootstrap_intervals(np.random.default_rng(3), 199)
    b = cr.bootstrap_intervals(np.random.default_rng(3), 199)
    assert a == b


def test_click_model_is_fitted_out_of_sample(setup: Fixture) -> None:
    sc, h, task, recs, theta, a, b = setup
    logs = sc.logs
    h84 = build_interaction_data(logs, logs.start + pd.to_timedelta(84, unit="D"))
    cm = opa.fit_click_model(logs, h84, ImplicitALS(8, 1.0, 20.0, 5).fit(h84), theta,
                             until=a)
    assert cm.feature_cutoff == h84.cutoff and cm.label_until == a
    q = cm.predict(h, ImplicitALS(8, 1.0, 20.0, 5).fit(h))
    t_mid = sc.truth.time_in_days(a + (b - a) / 2)[0]
    truth = sc.truth.attractiveness(np.arange(h.n_users), np.full(h.n_users, t_mid))
    # Regression (M9 development): the in-sample version collapsed q on unseen items.
    unseen = h.clicks == 0
    assert np.corrcoef(q[unseen], truth[unseen])[0, 1] > 0.2
    assert 0.5 < q[unseen].mean() / truth[unseen].mean() < 2.0
    with pytest.raises(ValueError, match="after the feature cutoff"):
        opa.fit_click_model(logs, h84, ImplicitALS(8).fit(h84), theta, until=h84.cutoff)
    with pytest.raises(ValueError, match="fitted at the history cutoff"):
        opa.fit_click_model(logs, h84, ImplicitALS(8).fit(h), theta, until=a)
    const = opa.fit_click_model(logs, h84, ImplicitALS(8).fit(h84), theta, until=a,
                                kind="constant")
    assert const.coefficients.size == 1


def test_extract_slots_refuses_unknown_logging(setup: Fixture) -> None:
    from dataclasses import replace

    from decision_intelligence.data.recommendation_scenarios import LoggingPolicySpec

    sc, h, task, recs, theta, a, b = setup
    bad = replace(sc.logs, logging_policy=LoggingPolicySpec(0.1, 10, None, "unknown"))
    with pytest.raises(ValueError, match="independently"):
        opa.extract_slots(bad, h, task.users, recs, a, b)
