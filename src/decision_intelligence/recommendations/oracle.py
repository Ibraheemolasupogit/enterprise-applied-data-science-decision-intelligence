"""Synthetic-oracle evaluation and simulated online value — evaluation only.

This is the only recommendations module that reads :class:`RecommendationTruth`. It is
never called during fitting, tuning or model selection; the separation is structural
(:mod:`.baselines`, :mod:`.interactions`, :mod:`.splits`, :mod:`.evaluation` and
:mod:`.off_policy` do not import the truth type) and is tested.

Three distinct things are computed, and must not be confused:

* **Oracle ranking metrics** — how well a list matches the user's *true* attractiveness
  ``a_ui(t)`` at the middle of the evaluation window, over the same allowed items the
  model ranked (graded NDCG with gains ``a_ui``; mean attractiveness of the list; share
  of the list inside the user's true top-K). Free of exposure bias, but only available
  because the data are synthetic.
* **Simulated online value** — the expected clicks (or purchases, or margin) per session
  had each logged test session shown the target list, under the generator's
  position-based click model. It is the estimand ``V(pi)`` of :mod:`.off_policy`
  computed exactly, i.e. a *simulation* of an online test, not an online test.
* **Supported value** — the same, counting only items the logging policy could have
  shown; this is what IPS can estimate when positivity fails.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from decision_intelligence.common.validation import FloatArray
from decision_intelligence.data.recommendation_scenarios import (
    RecommendationLogs,
    RecommendationTruth,
)
from decision_intelligence.recommendations import metrics as m
from decision_intelligence.recommendations.interactions import IntArray, InteractionData, index_of
from decision_intelligence.recommendations.off_policy import (
    clustered_mean,
    supported_mask,
    window_sessions,
)
from decision_intelligence.recommendations.splits import RankingTask


def _check_alignment(truth: RecommendationTruth, history: InteractionData) -> None:
    if not (np.array_equal(truth.user_ids, history.user_ids)
            and np.array_equal(truth.item_ids, history.item_ids)):
        raise ValueError("truth and history index different users or items")


def window_midpoint(task: RankingTask) -> pd.Timestamp:
    return task.cutoff + (task.window_end - task.cutoff) / 2


def oracle_gains(truth: RecommendationTruth, history: InteractionData, task: RankingTask
                 ) -> FloatArray:
    """True attractiveness ``a_ui`` at the window midpoint (rows: task users)."""
    _check_alignment(truth, history)
    t = truth.time_in_days(window_midpoint(task))[0]
    return truth.attractiveness(task.users, np.full(task.users.size, t))


def oracle_metrics(recs: IntArray, task: RankingTask, truth: RecommendationTruth,
                   history: InteractionData) -> pd.DataFrame:
    """Per-user oracle metrics for every active user (no click needed)."""
    gains = oracle_gains(truth, history, task)
    allowed = task.allowed
    k = recs.shape[1]
    masked = np.where(allowed, gains, -np.inf)
    ideal = np.argsort(-masked, axis=1, kind="stable")[:, :k]
    ideal_set = np.zeros_like(allowed)
    ideal_set[np.arange(ideal.shape[0])[:, None], ideal] = True
    return pd.DataFrame({
        "user_index": task.users,
        "oracle_ndcg": m.ndcg_at_k(recs, gains, allowed),
        "oracle_attractiveness": np.take_along_axis(gains, recs, axis=1).mean(axis=1),
        "oracle_precision": m.precision_at_k(recs, ideal_set),
    })


@dataclass(frozen=True)
class OnlineValue:
    estimate: float
    standard_error: float
    """Clustered by user: variation over the realised session sample, not over clicks."""
    ci_lower: float
    ci_upper: float
    per_session: FloatArray
    session_users: IntArray


def simulated_online_value(
    logs: RecommendationLogs, history: InteractionData, truth: RecommendationTruth,
    users: IntArray, recs: IntArray, after: pd.Timestamp, until: pd.Timestamp,
    *, reward: str = "clicked", supported_only: bool = False,
) -> OnlineValue:
    """Exact ``V(pi)`` under the generator's click model (see the module docstring)."""
    _check_alignment(truth, history)
    k = recs.shape[1]
    full = np.full((history.n_users, k), -1, dtype=np.int64)
    full[users] = recs
    sessions = window_sessions(logs, after, until)
    s_user = index_of(history.user_ids, sessions["user_id"].to_numpy())
    if np.any(full[s_user, 0] < 0):
        raise ValueError("a session's user has no target list")
    t = truth.time_in_days(sessions["timestamp"])
    lists = full[s_user]
    u = (truth.user_factors[s_user]
         + (t / truth.horizon_days)[:, None] * truth.user_drift[s_user])
    s_ui = np.einsum("nd,nkd->nk", u, truth.item_factors[lists]) + truth.item_bias[lists]
    value = 1.0 / (1.0 + np.exp(-(truth.click_intercept + s_ui)))
    if reward in ("purchased", "margin"):
        value = value / (1.0 + np.exp(-(truth.purchase_intercept + truth.purchase_preference
                                        * s_ui - truth.purchase_price
                                        * truth.price_z[lists])))
        if reward == "margin":
            value = value * truth.item_value[lists]
    elif reward != "clicked":
        raise ValueError("reward must be clicked, purchased or margin")
    value = value * truth.examination[:k][None, :]
    if supported_only:
        value = value * supported_mask(logs, history, s_user, sessions["timestamp"], full)
    per_session = np.asarray(value.sum(axis=1), dtype=np.float64)
    est, se, lo, hi = clustered_mean(per_session, s_user)
    return OnlineValue(est, se, lo, hi, per_session, s_user)
