"""Chronological train / validation / test protocol and ranking tasks.

Protocol (enforced here, documented in docs/methodology/recommender-baselines.md):

1. **Chronological split** at two cutoffs ``train_end < validation_end < test_end``. A
   global time cutoff, not a per-user "leave last out" split, so no model ever sees an
   interaction that happened after any interaction it is evaluated on.
2. **Tuning**: fit on interactions ``<= train_end``; evaluate on the validation window
   ``(train_end, validation_end]``; choose hyperparameters by a logged metric.
3. **Testing**: refit the chosen configuration on interactions ``<= validation_end``;
   evaluate once on ``(validation_end, test_end]``.
4. **Eligible catalogue**: every item launched by the cutoff. Recommendations are ranked
   over the full eligible catalogue (no sampled negatives), minus items the user already
   clicked before the cutoff ("seen" items are excluded from both the ranking and the
   relevant set: the task is discovery of new items).
5. **Logged relevance**: an item is relevant if the user clicked it in the evaluation
   window, it was eligible at the cutoff and it was not seen.

:func:`random_holdout_tasks` builds the classic *invalid* protocol (a random split of
click events ignoring time) only for the leakage counterexample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from decision_intelligence.data.recommendation_scenarios import RecommendationLogs
from decision_intelligence.recommendations.interactions import (
    BoolArray,
    IntArray,
    InteractionData,
    build_interaction_data,
    deduplicate_impressions,
    index_of,
)


@dataclass(frozen=True)
class ChronologicalSplit:
    start: pd.Timestamp
    train_end: pd.Timestamp
    validation_end: pd.Timestamp
    test_end: pd.Timestamp


def chronological_split(
    logs: RecommendationLogs, *, train_days: float, validation_days: float,
) -> ChronologicalSplit:
    """Cutoffs ``start + train_days`` and ``+ validation_days``; the test runs to the end."""
    train_end = logs.start + pd.to_timedelta(train_days, unit="D")
    validation_end = train_end + pd.to_timedelta(validation_days, unit="D")
    if not logs.start < train_end < validation_end < logs.end:
        raise ValueError("need start < train_end < validation_end < end of the logs")
    return ChronologicalSplit(logs.start, train_end, validation_end, logs.end)


@dataclass(frozen=True)
class RankingTask:
    """An offline evaluation task at one cutoff.

    Rows of ``seen`` and ``relevant`` correspond to ``users`` (indices into the history's
    user table). ``allowed = eligible & ~seen`` is what any model may recommend.
    """

    name: str
    cutoff: pd.Timestamp
    window_end: pd.Timestamp
    users: IntArray
    eligible: BoolArray
    seen: BoolArray
    relevant: BoolArray

    @property
    def allowed(self) -> BoolArray:
        return np.asarray(self.eligible[None, :] & ~self.seen, dtype=np.bool_)

    def users_with_relevant(self) -> BoolArray:
        return np.asarray(self.relevant.any(axis=1), dtype=np.bool_)


def _window_clicks(
    logs: RecommendationLogs, after: pd.Timestamp, until: pd.Timestamp,
) -> pd.DataFrame:
    imp = deduplicate_impressions(logs.impressions)
    mask = imp["clicked"] & (imp["timestamp"] > after) & (imp["timestamp"] <= until)
    return imp.loc[mask, ["user_id", "item_id", "timestamp"]]


def active_users(logs: RecommendationLogs, history: InteractionData,
                 after: pd.Timestamp, until: pd.Timestamp) -> IntArray:
    """Users with at least one session in ``(after, until]`` (sorted indices)."""
    s = logs.sessions
    ids = s.loc[(s["timestamp"] > after) & (s["timestamp"] <= until), "user_id"].unique()
    return np.sort(index_of(history.user_ids, ids))


def ranking_task(
    logs: RecommendationLogs, history: InteractionData, window_end: pd.Timestamp,
    *, name: str = "", exclude_seen: bool = True,
) -> RankingTask:
    """Evaluate on ``(history.cutoff, window_end]`` for every user active in the window."""
    if window_end <= history.cutoff:
        raise ValueError("the evaluation window must end after the history cutoff")
    users = active_users(logs, history, history.cutoff, window_end)
    clicks = _window_clicks(logs, history.cutoff, window_end)
    relevant_full = np.zeros((history.n_users, history.n_items), dtype=np.bool_)
    relevant_full[index_of(history.user_ids, clicks["user_id"].to_numpy()),
                  index_of(history.item_ids, clicks["item_id"].to_numpy())] = True
    seen = (history.clicks[users] > 0) if exclude_seen else np.zeros(
        (users.size, history.n_items), dtype=np.bool_)
    relevant = relevant_full[users] & history.item_launched[None, :] & ~seen
    return RankingTask(name, history.cutoff, window_end, users, history.item_launched.copy(),
                       np.asarray(seen, dtype=np.bool_), np.asarray(relevant, dtype=np.bool_))


def random_holdout_tasks(
    logs: RecommendationLogs, proper_history: InteractionData, window_end: pd.Timestamp,
    *, holdout_share: float, rng: np.random.Generator,
) -> tuple[InteractionData, RankingTask, RankingTask]:
    """INVALID protocol, for the leakage counterexample only.

    The clicks in the evaluation window ``(proper_history.cutoff, window_end]`` are split
    at random, ignoring time: a held-out share ``H`` is the relevant set; the rest ``R``
    is *added to the training history* — so training contains interactions that happen
    after held-out ones (future popularity and tastes), and items launched during the
    window enter the catalogue. Returns ``(leaky_history, leaky_task, proper_task)``: both
    tasks score exactly the same held-out clicks; ``proper_task`` uses the chronological
    history ``proper_history``. Any metric gap is therefore due to the leaked information.
    """
    if not 0.0 < holdout_share < 1.0:
        raise ValueError("holdout_share must lie in (0, 1)")
    cutoff = proper_history.cutoff
    window = _window_clicks(logs, cutoff, window_end)
    held = rng.random(len(window)) < holdout_share
    imp = deduplicate_impressions(logs.impressions)
    before = imp.loc[imp["clicked"] & (imp["timestamp"] <= cutoff),
                     ["user_id", "item_id", "timestamp"]]
    leaky = build_interaction_data(
        logs, window_end, clicks=pd.concat([before, window.loc[~held]], ignore_index=True))
    users = active_users(logs, proper_history, cutoff, window_end)
    held_matrix = np.zeros((proper_history.n_users, proper_history.n_items), dtype=np.bool_)
    test = window.loc[held]
    held_matrix[index_of(proper_history.user_ids, test["user_id"].to_numpy()),
                index_of(proper_history.item_ids, test["item_id"].to_numpy())] = True
    proper_seen = proper_history.clicks[users] > 0
    leaky_seen = leaky.clicks[users] > 0
    relevant = (held_matrix[users] & proper_history.item_launched[None, :] & ~leaky_seen
                & ~proper_seen)
    leaky_task = RankingTask("random holdout (invalid)", window_end, window_end, users,
                             leaky.item_launched.copy(), np.asarray(leaky_seen, dtype=np.bool_),
                             np.asarray(relevant, dtype=np.bool_))
    proper_task = RankingTask("chronological", cutoff, window_end, users,
                              proper_history.item_launched.copy(),
                              np.asarray(proper_seen, dtype=np.bool_),
                              np.asarray(relevant, dtype=np.bool_))
    return leaky, leaky_task, proper_task
