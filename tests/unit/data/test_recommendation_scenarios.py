"""The M8 recommendation scenario: determinism, schemas, exact propensities and truth."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from decision_intelligence.data.recommendation_scenarios import (
    RecommendationScenario,
    RecommendationScenarioConfig,
    generate_recommendation_scenario,
    resimulate_logging,
)

CFG = RecommendationScenarioConfig(seed=11, n_users=250, n_items=60)


@pytest.fixture(scope="module")
def scenario() -> RecommendationScenario:
    return generate_recommendation_scenario(CFG)


def test_same_seed_reproduces_and_a_new_seed_changes(scenario: RecommendationScenario) -> None:
    again = generate_recommendation_scenario(CFG)
    for name in ("users", "items", "sessions", "impressions", "ratings"):
        pd.testing.assert_frame_equal(getattr(scenario.logs, name), getattr(again.logs, name))
    other = generate_recommendation_scenario(replace(CFG, seed=12))
    assert not scenario.logs.impressions["item_id"].equals(other.logs.impressions["item_id"])


def test_logging_seed_redraws_only_slates_clicks_and_ratings(
    scenario: RecommendationScenario,
) -> None:
    other = generate_recommendation_scenario(replace(CFG, logging_seed=5))
    for name in ("users", "items", "sessions"):
        pd.testing.assert_frame_equal(getattr(scenario.logs, name), getattr(other.logs, name))
    assert np.array_equal(scenario.truth.user_factors, other.truth.user_factors)
    assert not scenario.logs.impressions["item_id"].equals(other.logs.impressions["item_id"])


def test_schemas_and_slate_structure(scenario: RecommendationScenario) -> None:
    logs = scenario.logs
    assert list(logs.impressions.columns) == [
        "impression_id", "session_id", "user_id", "item_id", "position", "timestamp",
        "logging_propensity", "clicked", "purchased"]
    assert "utility" not in logs.impressions.columns  # no ground truth in the logs
    assert logs.impressions["impression_id"].is_unique
    per_session = logs.impressions.groupby("session_id")["position"].apply(list)
    assert all(p == list(range(1, CFG.slate_size + 1)) for p in per_session)
    assert set(per_session.index) == set(logs.sessions["session_id"])
    assert not (logs.impressions["purchased"] & ~logs.impressions["clicked"]).any()
    assert logs.ratings["rating"].between(1, 5).all()


def test_sessions_and_impressions_respect_signup_and_launch_times(
    scenario: RecommendationScenario,
) -> None:
    logs = scenario.logs
    s = logs.sessions.merge(logs.users[["user_id", "signup_time"]], on="user_id")
    assert (s["timestamp"] >= s["signup_time"]).all()
    assert (s["timestamp"] >= logs.start).all() and (s["timestamp"] <= logs.end).all()
    imp = logs.impressions.merge(logs.items[["item_id", "launch_time"]], on="item_id")
    assert (imp["launch_time"] <= imp["timestamp"]).all()


def test_logged_propensities_are_the_exact_logging_distribution(
    scenario: RecommendationScenario,
) -> None:
    logs, truth, lm = scenario.logs, scenario.truth, scenario.logging_model
    imp = logs.impressions
    for sid in logs.sessions["session_id"].to_numpy()[::97]:
        rows = imp.loc[imp["session_id"] == sid]
        u = int(np.searchsorted(truth.user_ids, rows["user_id"].iloc[0]))
        day = float(truth.time_in_days(rows["timestamp"].iloc[[0]])[0])
        allowed = lm.supported & (lm.launch_days <= day)
        z = np.where(allowed, np.exp(lm.logits[u] - lm.logits[u][allowed].max()), 0.0)
        mu = (1 - CFG.exploration) * z / z.sum() + CFG.exploration * allowed / allowed.sum()
        assert mu.sum() == pytest.approx(1.0)
        items = np.searchsorted(truth.item_ids, rows["item_id"].to_numpy())
        assert np.allclose(rows["logging_propensity"].to_numpy(), mu[items], rtol=1e-12)


def test_slot_frequencies_match_propensities_in_repeated_sampling(
    scenario: RecommendationScenario,
) -> None:
    after = scenario.logs.end - pd.to_timedelta(2, unit="D")
    sid = scenario.logs.sessions.loc[scenario.logs.sessions["timestamp"] > after,
                                     "session_id"].iloc[0]
    draws = pd.concat([resimulate_logging(scenario, after, seed) for seed in range(300)])
    draws = draws.loc[draws["session_id"] == sid]
    freq = draws["item_id"].value_counts(normalize=True)
    mu = draws.groupby("item_id")["logging_propensity"].first()
    top = mu.sort_values(ascending=False).index[:3]
    n = len(draws)
    for item in top:  # within 4 binomial SE
        se = np.sqrt(mu[item] * (1 - mu[item]) / n)
        assert abs(freq[item] - mu[item]) < 4 * se


def test_restricted_logging_support_never_shows_other_items() -> None:
    sc = generate_recommendation_scenario(replace(CFG, logging_support_size=20))
    support = sc.logs.logging_policy.support_item_ids
    assert support is not None and len(support) == 20
    assert set(sc.logs.impressions["item_id"]) <= set(support)


def test_truth_is_well_formed(scenario: RecommendationScenario) -> None:
    truth = scenario.truth
    assert truth.examination[0] == 1.0 and np.all(np.diff(truth.examination) < 0)
    users = np.arange(5)
    times = np.full(5, 30.0)
    a = truth.attractiveness(users, times)
    assert a.shape == (5, CFG.n_items) and np.all((a > 0) & (a < 1))
    er = truth.expected_rating(users, times)
    assert np.all((er >= 1) & (er <= 5))


def test_expected_rating_matches_monte_carlo(scenario: RecommendationScenario) -> None:
    truth = scenario.truth
    rng = np.random.default_rng(0)
    s = truth.utility(np.array([3]), np.array([40.0]))[0, :4]
    exact = truth.expected_rating(np.array([3]), np.array([40.0]))[0, :4]
    for j in range(4):
        raw = 3 + truth.rating_slope * (s[j] - truth.rating_centre) + rng.normal(
            0, truth.rating_noise_sd, 200_000)
        assert np.clip(np.round(raw), 1, 5).mean() == pytest.approx(exact[j], abs=0.01)


def test_ratings_are_missing_not_at_random(scenario: RecommendationScenario) -> None:
    truth = scenario.truth
    observed = scenario.logs.ratings["rating"].mean()
    everywhere = truth.expected_rating(np.arange(CFG.n_users), np.full(CFG.n_users, 60.0)).mean()
    assert observed > everywhere + 1.0


def test_duplicate_defect_is_planted_only_when_requested(scenario: RecommendationScenario) -> None:
    dup = generate_recommendation_scenario(replace(CFG, duplicate_impression_rate=0.1))
    assert not dup.logs.impressions["impression_id"].is_unique
    assert scenario.logs.impressions["impression_id"].is_unique


@pytest.mark.parametrize("changes", [
    {"n_users": 1}, {"n_items": 5}, {"exploration": 1.5}, {"new_item_share": 1.0},
    {"content_weight": 0.9, "category_weight": 0.9}, {"logging_support_size": 3},
    {"duplicate_impression_rate": 1.0}, {"popularity_quality_correlation": 2.0},
    {"sessions_per_day": 0.0}, {"examination_decay": -1.0},
])
def test_invalid_configurations_are_refused(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        RecommendationScenarioConfig(**changes)
