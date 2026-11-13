"""Inverse-propensity (IPS) estimates of a recommendation list's value from logged data.

**Estimand.** For a target policy that shows user ``u`` the fixed list
``pi_u = (pi_u1, ..., pi_uK)`` (computed at the cutoff), its value over the evaluation
window is the expected reward per logged session if, in each of those sessions, the
target list had been shown instead of the logged slate::

    V(pi) = (1 / N) sum_s sum_{k<=K} E[Y | item pi_{u_s k} at position k in session s]

with ``Y`` a click, a purchase or the margin of a purchase. Sessions are held at their
logged users and times (the estimand is conditional on the realised session sample).

**Assumptions** under which the estimator below is unbiased for ``V(pi)``:

1. *Position-based examination (PBM)*: ``E[Y | i at k] = theta_k * r_ui(t)`` — position
   only scales the reward through an examination probability, with no interaction
   between items in the slate (no cascade, no slate-level effects).
2. *Known propensities*: each logged slot's item was drawn with the logged probability
   ``mu_j`` and slots were filled independently (``slot_sampling == "independent"``).
3. *Positivity*: every item in every target list had ``mu > 0`` in that session. Items
   outside the logging support (or not yet launched) violate it; their contribution is
   unestimable and the IPS estimate then targets only the *supported* part of ``V``
   (a downward bias for non-negative rewards). :attr:`IPSResult.unsupported_share`
   reports the examination mass on such items.
4. *Known relative examination* ``theta_k / theta_j``. It is identified from the logs
   (:func:`estimate_examination`) because independent slots make logging
   position-exchangeable: ``CTR_k / CTR_1 = theta_k / theta_1``. Only ratios enter.

**Estimator.** For slot ``j`` of a logged session (item ``i_j`` at position ``p_j``,
reward ``y_j``), with ``k(i)`` the item's position in the target list::

    w_j = (theta_{k(i_j)} / theta_{p_j}) / mu_j   if i_j is in pi_u, else 0
    V_hat_s = (1 / L) sum_j y_j w_j,              V_hat = mean_s V_hat_s

Each slot is unbiased for ``V_s`` under 1-4, so the average is too. **Clipping** the
weights at ``M`` reduces variance and biases the estimate downwards (rewards are
non-negative); it is not unbiased and is reported as such. The standard error is
clustered by user (a user's sessions share one list). The **effective sample size**
``(sum w)^2 / sum w^2`` over all logged slots shows how few slots carry the estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from decision_intelligence.common.validation import FloatArray
from decision_intelligence.data.recommendation_scenarios import RecommendationLogs
from decision_intelligence.recommendations.interactions import (
    IntArray,
    InteractionData,
    deduplicate_impressions,
    index_of,
)

REWARDS = ("clicked", "purchased", "margin")


@dataclass(frozen=True)
class ExaminationEstimate:
    positions: IntArray
    ctr: FloatArray
    relative: FloatArray
    """``theta_k / theta_1`` estimated by ``CTR_k / CTR_1``."""
    n_impressions: IntArray


def estimate_examination(logs: RecommendationLogs, until: pd.Timestamp) -> ExaminationEstimate:
    """Relative examination by position from impressions at or before ``until``.

    Valid only for position-exchangeable logging (independent slots); refused otherwise.
    """
    if logs.logging_policy.slot_sampling != "independent":
        raise ValueError("relative examination is identified only for independent slots")
    imp = deduplicate_impressions(logs.impressions)
    imp = imp.loc[imp["timestamp"] <= until]
    grouped = imp.groupby("position")["clicked"].agg(["mean", "size"]).sort_index()
    ctr = grouped["mean"].to_numpy(dtype=np.float64)
    if ctr[0] <= 0:
        raise ValueError("no clicks at position 1")
    return ExaminationEstimate(grouped.index.to_numpy(dtype=np.int64), ctr, ctr / ctr[0],
                               grouped["size"].to_numpy(dtype=np.int64))


@dataclass(frozen=True)
class IPSResult:
    estimate: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    n_sessions: int
    n_users: int
    n_slots: int
    effective_sample_size: float
    max_weight: float
    clip: float | None
    clipped_share: float
    """Share of slots with a positive weight whose weight was clipped."""
    unsupported_share: float
    """Mean share of the target list's examination mass on unsupported items."""
    per_session: FloatArray = field(repr=False)
    session_users: IntArray = field(repr=False)


def item_rewards(logs: RecommendationLogs, impressions: pd.DataFrame, reward: str) -> FloatArray:
    if reward == "clicked":
        return impressions["clicked"].to_numpy(dtype=np.float64)
    if reward == "purchased":
        return impressions["purchased"].to_numpy(dtype=np.float64)
    if reward == "margin":
        items = logs.items.set_index("item_id")
        value = (items["price"] * items["margin_rate"]).reindex(impressions["item_id"])
        return np.asarray(impressions["purchased"].to_numpy() * value.to_numpy(),
                          dtype=np.float64)
    raise ValueError(f"reward must be one of {REWARDS}")


def clustered_mean(values: FloatArray, clusters: IntArray, confidence_level: float = 0.95
                   ) -> tuple[float, float, float, float]:
    """Mean of ``values`` with a cluster-robust SE (clusters = users) and normal CI."""
    estimate = float(values.mean())
    _, inverse = np.unique(clusters, return_inverse=True)
    totals = np.bincount(inverse, weights=values)
    sizes = np.bincount(inverse).astype(np.float64)
    g, n = totals.size, values.size
    if g < 2:
        raise ValueError("need at least two clusters")
    var = g / (g - 1) * np.sum((totals - sizes * estimate) ** 2) / n ** 2
    se = math.sqrt(var)
    z = float(stats.norm.ppf(0.5 + confidence_level / 2))
    return estimate, se, estimate - z * se, estimate + z * se


def supported_mask(
    logs: RecommendationLogs, history: InteractionData, session_users: IntArray,
    session_times: pd.Series, recs_by_user: IntArray,
) -> np.ndarray:
    """``(n_sessions, K)``: whether each target item could have been logged in the session."""
    items = logs.items.set_index("item_id").reindex(history.item_ids)
    launch = items["launch_time"].to_numpy()
    lists = recs_by_user[session_users]
    ok = launch[lists] <= session_times.to_numpy()[:, None]
    support = logs.logging_policy.support_item_ids
    if support is not None:
        in_support = np.isin(history.item_ids, np.array(support, dtype=np.int64))
        ok &= in_support[lists]
    return np.asarray(ok)


def window_sessions(logs: RecommendationLogs, after: pd.Timestamp, until: pd.Timestamp
                    ) -> pd.DataFrame:
    s = logs.sessions
    return s.loc[(s["timestamp"] > after) & (s["timestamp"] <= until)].reset_index(drop=True)


def ips_policy_value(
    logs: RecommendationLogs,
    history: InteractionData,
    users: IntArray,
    recs: IntArray,
    after: pd.Timestamp,
    until: pd.Timestamp,
    relative_examination: FloatArray,
    *,
    reward: str = "clicked",
    clip: float | None = None,
    impressions: pd.DataFrame | None = None,
    confidence_level: float = 0.95,
) -> IPSResult:
    """IPS estimate of ``V(pi)`` for the lists ``recs`` (rows aligned with ``users``).

    ``impressions`` overrides the logged impressions of the window (used to re-simulate
    logging in repeated-sampling checks); by default the de-duplicated log is used.
    """
    k = recs.shape[1]
    slate = logs.logging_policy.slate_size
    if logs.logging_policy.slot_sampling != "independent":
        raise ValueError("this estimator requires independently filled slots")
    if k > slate or relative_examination.size < slate:
        raise ValueError("list length must not exceed the slate size")
    if clip is not None and clip <= 0:
        raise ValueError("clip must be positive")
    full = np.full((history.n_users, k), -1, dtype=np.int64)
    full[users] = recs
    rank = np.zeros((history.n_users, history.n_items), dtype=np.int64)
    rank[users[:, None], recs] = np.arange(1, k + 1)[None, :]

    sessions = window_sessions(logs, after, until)
    s_user = index_of(history.user_ids, sessions["user_id"].to_numpy())
    if np.any(full[s_user, 0] < 0):
        raise ValueError("a session's user has no target list")
    imp = deduplicate_impressions(logs.impressions if impressions is None else impressions)
    imp = imp.loc[(imp["timestamp"] > after) & (imp["timestamp"] <= until)]
    imp = imp.loc[imp["session_id"].isin(sessions["session_id"])]
    u = index_of(history.user_ids, imp["user_id"].to_numpy())
    i = index_of(history.item_ids, imp["item_id"].to_numpy())
    pos = imp["position"].to_numpy(dtype=np.int64)
    target_k = rank[u, i]
    mu = imp["logging_propensity"].to_numpy(dtype=np.float64)
    if np.any(mu <= 0):
        raise ValueError("logged propensities must be positive")
    theta = relative_examination
    w = np.where(target_k > 0, theta[np.maximum(target_k, 1) - 1] / theta[pos - 1] / mu, 0.0)
    positive = w > 0
    clipped_share = 0.0
    if clip is not None:
        clipped_share = float((w[positive] > clip).mean()) if positive.any() else 0.0
        w = np.minimum(w, clip)
    y = item_rewards(logs, imp, reward)
    contrib = y * w / slate
    session_index = pd.Index(sessions["session_id"]).get_indexer(imp["session_id"])
    per_session = np.asarray(np.bincount(session_index, weights=contrib,
                                         minlength=len(sessions)), dtype=np.float64)
    est, se, lo, hi = clustered_mean(per_session, s_user, confidence_level)
    ess = float(w.sum() ** 2 / np.sum(w ** 2)) if np.any(w > 0) else 0.0
    sup = supported_mask(logs, history, s_user, sessions["timestamp"], full)
    exam = theta[:k]
    unsupported = float(((~sup) * exam[None, :]).sum(axis=1).mean() / exam.sum())
    return IPSResult(est, se, lo, hi, len(sessions), int(np.unique(s_user).size), int(w.size),
                     ess, float(w.max()) if w.size else 0.0, clip, clipped_share, unsupported,
                     np.asarray(per_session, dtype=np.float64), s_user)
