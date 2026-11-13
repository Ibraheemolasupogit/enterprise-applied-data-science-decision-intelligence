"""Chronological protocol, point-in-time invariance and the leakage counterexample."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from decision_intelligence.data.recommendation_scenarios import (
    RecommendationScenario,
    generate_recommendation_scenario,
)
from decision_intelligence.recommendations import baselines as bl
from decision_intelligence.recommendations.cases import perturb_after
from decision_intelligence.recommendations.explicit import BiasRatingModel, rating_data
from decision_intelligence.recommendations.interactions import (
    build_interaction_data,
    deduplicate_impressions,
)
from decision_intelligence.recommendations.splits import (
    ChronologicalSplit,
    chronological_split,
    random_holdout_tasks,
    ranking_task,
)

from .conftest import SMALL


def _split(sc: RecommendationScenario) -> ChronologicalSplit:
    return chronological_split(sc.logs, train_days=84, validation_days=14)


def test_split_cutoffs_are_ordered_and_validated(small_scenario: RecommendationScenario) -> None:
    sp = _split(small_scenario)
    assert sp.start < sp.train_end < sp.validation_end < sp.test_end
    with pytest.raises(ValueError):
        chronological_split(small_scenario.logs, train_days=100, validation_days=30)


def test_history_ignores_everything_after_the_cutoff(
    small_scenario: RecommendationScenario,
) -> None:
    sp = _split(small_scenario)
    a = build_interaction_data(small_scenario.logs, sp.validation_end)
    perturbed = perturb_after(small_scenario.logs, sp.validation_end, seed=1)
    b = build_interaction_data(perturbed, sp.validation_end)
    assert np.array_equal(a.clicks, b.clicks)
    pd.testing.assert_frame_equal(a.click_events, b.click_events)
    cutoff_day = (sp.validation_end - small_scenario.logs.start).total_seconds() / 86_400
    assert a.click_events["day"].max() <= cutoff_day


@pytest.mark.parametrize("make", [
    bl.GlobalPopularity, lambda: bl.GlobalPopularity(7.0), lambda: bl.SegmentPopularity(10.0),
    lambda: bl.ItemKNN(10), lambda: bl.UserKNN(10), bl.ContentBased, lambda: bl.PureSVD(8),
    lambda: bl.Fallback(bl.ItemKNN(10), bl.SegmentPopularity(10.0)),
])
def test_every_model_is_invariant_to_post_cutoff_data(
    small_scenario: RecommendationScenario, make: Callable[[], bl.Recommender],
) -> None:
    sp = _split(small_scenario)
    users = np.arange(len(small_scenario.logs.users))
    a = make().fit(build_interaction_data(small_scenario.logs, sp.validation_end)).score(users)
    perturbed = perturb_after(small_scenario.logs, sp.validation_end, seed=2)
    b = make().fit(build_interaction_data(perturbed, sp.validation_end)).score(users)
    assert np.array_equal(a, b)


def test_rating_model_is_invariant_to_post_cutoff_ratings(
    small_scenario: RecommendationScenario,
) -> None:
    sp = _split(small_scenario)
    h = build_interaction_data(small_scenario.logs, sp.validation_end)
    perturbed = perturb_after(small_scenario.logs, sp.validation_end, seed=3)
    fits = [BiasRatingModel().fit(rating_data(logs.ratings, h.user_ids, h.item_ids, after=None,
                                              until=sp.validation_end))
            for logs in (small_scenario.logs, perturbed)]
    assert np.array_equal(fits[0].item_bias, fits[1].item_bias)


def test_ranking_task_relevance_eligibility_and_seen_exclusion(
    small_scenario: RecommendationScenario,
) -> None:
    logs = small_scenario.logs
    sp = _split(small_scenario)
    h = build_interaction_data(logs, sp.validation_end)
    task = ranking_task(logs, h, sp.test_end)
    launch = logs.items.sort_values("item_id")["launch_time"].to_numpy()
    assert np.array_equal(task.eligible, launch <= np.datetime64(sp.validation_end))
    assert not np.any(task.relevant & task.seen)
    assert not np.any(task.relevant & ~task.eligible[None, :])
    assert np.array_equal(task.seen, h.clicks[task.users] > 0)
    # Every relevant pair is a click inside the window.
    imp = logs.impressions
    window = imp.loc[imp["clicked"] & (imp["timestamp"] > sp.validation_end)]
    pairs = set(zip(window["user_id"], window["item_id"], strict=True))
    u, i = np.nonzero(task.relevant)
    assert all((int(h.user_ids[task.users[a]]), int(h.item_ids[b])) in pairs
               for a, b in zip(u, i, strict=True))


def test_ranking_task_refuses_a_window_before_the_cutoff(
    small_scenario: RecommendationScenario,
) -> None:
    sp = _split(small_scenario)
    h = build_interaction_data(small_scenario.logs, sp.validation_end)
    with pytest.raises(ValueError, match="after the history cutoff"):
        ranking_task(small_scenario.logs, h, sp.train_end)


def test_users_who_sign_up_after_the_cutoff_have_no_history(
    small_scenario: RecommendationScenario,
) -> None:
    logs = small_scenario.logs
    sp = _split(small_scenario)
    h = build_interaction_data(logs, sp.validation_end)
    late = (logs.users.sort_values("user_id")["signup_time"] > sp.validation_end).to_numpy()
    assert late.any()
    assert np.all(h.clicks[late] == 0)


def test_duplicated_impressions_are_removed_before_counting() -> None:
    dup = generate_recommendation_scenario(replace(SMALL, duplicate_impression_rate=0.05))
    clean = generate_recommendation_scenario(SMALL)
    assert not dup.logs.impressions["impression_id"].is_unique
    pd.testing.assert_frame_equal(deduplicate_impressions(dup.logs.impressions),
                                  clean.logs.impressions)
    cutoff = clean.logs.start + pd.to_timedelta(98, unit="D")
    a = build_interaction_data(dup.logs, cutoff)
    b = build_interaction_data(clean.logs, cutoff)
    assert np.array_equal(a.clicks, b.clicks)


def test_random_holdout_leaks_future_clicks_into_training(
    small_scenario: RecommendationScenario,
) -> None:
    logs = small_scenario.logs
    sp = _split(small_scenario)
    h = build_interaction_data(logs, sp.validation_end)
    leaky, leaky_task, proper_task = random_holdout_tasks(
        logs, h, sp.test_end, holdout_share=0.5, rng=np.random.default_rng(0))
    assert np.array_equal(leaky_task.relevant, proper_task.relevant)
    assert np.all(leaky.clicks >= h.clicks)
    assert leaky.clicks.sum() > h.clicks.sum()  # post-cutoff clicks entered training
    assert leaky.cutoff > h.cutoff
    with pytest.raises(ValueError):
        random_holdout_tasks(logs, h, sp.test_end, holdout_share=1.0,
                             rng=np.random.default_rng(0))
