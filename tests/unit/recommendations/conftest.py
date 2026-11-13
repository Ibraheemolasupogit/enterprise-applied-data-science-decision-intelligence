"""Shared fixtures: a small synthetic scenario and hand-built interaction histories."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from decision_intelligence.data.recommendation_scenarios import (
    RecommendationScenario,
    RecommendationScenarioConfig,
    generate_recommendation_scenario,
)
from decision_intelligence.recommendations.interactions import InteractionData

SMALL = RecommendationScenarioConfig(seed=7, n_users=300, n_items=80)


@pytest.fixture(scope="session")
def small_scenario() -> RecommendationScenario:
    return generate_recommendation_scenario(SMALL)


def history_from_matrix(
    clicks: np.ndarray, *, segments: list[str] | None = None,
    features: np.ndarray | None = None, days: np.ndarray | None = None,
) -> InteractionData:
    """A hand-built InteractionData (cutoff day 100) for exact reference checks."""
    n_users, n_items = clicks.shape
    u, i = np.nonzero(clicks)
    if days is None:
        days = np.full(u.size, 50.0)
    start = pd.Timestamp("2025-01-01")
    if features is None:
        features = np.eye(n_items)
    return InteractionData(
        cutoff=start + pd.to_timedelta(100, unit="D"), start=start,
        user_ids=np.arange(1, n_users + 1, dtype=np.int64),
        item_ids=np.arange(1, n_items + 1, dtype=np.int64),
        user_segment=np.array(segments or ["a"] * n_users),
        user_region=np.array(["north"] * n_users),
        item_category=np.array(["c"] * n_items),
        item_features=features.astype(np.float64),
        item_launched=np.ones(n_items, dtype=bool),
        clicks=(clicks > 0).astype(np.float64),
        click_events=pd.DataFrame({"user_index": u.astype(np.int64),
                                   "item_index": i.astype(np.int64), "day": days}),
    )


@pytest.fixture
def make_history() -> Callable[..., InteractionData]:
    return history_from_matrix
