"""Simulated online period under chosen recommendation policies (Milestone 9) — SIMULATED.

Additive: reads the M8 scenario's ground truth and configuration; changes nothing in M2 or M8.

Each experiment user receives sessions over ``(start, start + duration]`` (Poisson with the
scenario's ``sessions_per_day``, times uniform; users must have signed up by ``start``). In
every session the user is shown their arm's **fixed** top-K list (computed at ``start``); clicks
follow the M8 position-based model, ``P(click at k) = theta_k a_ui(t)``, and purchases follow a
click with the M8 purchase model; margin = purchase x price x margin rate.

Assumptions (stated, not tested against reality): the policy does not change how often users
visit (sessions are drawn independently of the arm, so the same sessions are used to compute
every arm's expected outcome); users do not interact (no interference); the list does not adapt
during the period (no feedback loop); no long-term effects (retention is out of scope).

Because sessions are arm-independent, the **exact** expected outcome of every user under every
arm, given their sessions, is available: the sample average treatment effect (SATE) of arm ``b``
versus ``a`` is the mean over experiment users of the difference — the target a simulated A/B
analysis is checked against.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from decision_intelligence.common.validation import FloatArray
from decision_intelligence.data.recommendation_scenarios import RecommendationScenario

IntArray = np.ndarray


@dataclass(frozen=True)
class OnlineSessions:
    user_row: IntArray
    """Row of the experiment user (index into ``users``) per session."""
    day: FloatArray


@dataclass(frozen=True)
class OnlinePeriod:
    users: IntArray
    """Truth/user index of each experiment user."""
    arm: np.ndarray
    sessions: IntArray
    clicks: FloatArray
    purchases: FloatArray
    margin: FloatArray
    expected: dict[str, dict[str, FloatArray]]
    """``expected[arm][metric]``: each user's exact expected outcome had they been in ``arm``
    (metrics: clicks, purchases, margin), given their simulated sessions."""

    def sate(self, treatment: str, control: str, metric: str) -> float:
        return float(np.mean(self.expected[treatment][metric] - self.expected[control][metric]))


def draw_sessions(scenario: RecommendationScenario, users: IntArray, start_day: float,
                  duration_days: float, seed: int) -> OnlineSessions:
    if duration_days <= 0:
        raise ValueError("duration_days must be positive")
    rng = np.random.default_rng(seed)
    counts = rng.poisson(scenario.config.sessions_per_day * duration_days, size=users.size)
    row = np.repeat(np.arange(users.size), counts)
    day = start_day + rng.random(row.size) * duration_days
    return OnlineSessions(row.astype(np.int64), day)


def _session_values(scenario: RecommendationScenario, users: IntArray, sessions: OnlineSessions,
                    lists: IntArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Per session and list position: attractiveness, purchase-given-click, item value."""
    t = scenario.truth
    u = users[sessions.user_row]
    items = lists[sessions.user_row]
    factors = (t.user_factors[u]
               + (sessions.day / t.horizon_days)[:, None] * t.user_drift[u])
    s = np.einsum("nd,nkd->nk", factors, t.item_factors[items]) + t.item_bias[items]
    attract = 1.0 / (1.0 + np.exp(-(t.click_intercept + s)))
    buy = 1.0 / (1.0 + np.exp(-(t.purchase_intercept + t.purchase_preference * s
                                - t.purchase_price * t.price_z[items])))
    return attract, buy, t.item_value[items]


def simulate_online_period(
    scenario: RecommendationScenario, users: IntArray, arm: np.ndarray,
    arm_lists: Mapping[str, IntArray], sessions: OnlineSessions, click_seed: int,
) -> OnlinePeriod:
    """Outcomes per user under the assigned arm, plus exact expected outcomes for every arm.

    ``arm_lists[a]`` is the (n_users x K) list of item indices arm ``a`` would show each user.
    """
    if arm.shape != users.shape:
        raise ValueError("one arm label per user")
    if set(np.unique(arm)) - set(arm_lists):
        raise ValueError("an assigned arm has no lists")
    theta = scenario.truth.examination
    k = next(iter(arm_lists.values())).shape[1]
    if k > theta.size:
        raise ValueError("list longer than the slate")
    expected: dict[str, dict[str, FloatArray]] = {}
    values = {}
    for name, lists in arm_lists.items():
        if lists.shape != (users.size, k):
            raise ValueError("every arm needs one K-list per user")
        attract, buy, value = _session_values(scenario, users, sessions, lists)
        values[name] = (attract, buy, value)
        click_p = theta[:k][None, :] * attract
        per = {"clicks": click_p.sum(axis=1), "purchases": (click_p * buy).sum(axis=1),
               "margin": (click_p * buy * value).sum(axis=1)}
        expected[name] = {m: np.asarray(np.bincount(sessions.user_row, weights=v,
                                                    minlength=users.size), dtype=np.float64)
                          for m, v in per.items()}
    rng = np.random.default_rng(click_seed)
    n_s = sessions.user_row.size
    session_arm = arm[sessions.user_row]
    attract = np.empty((n_s, k))
    buy = np.empty((n_s, k))
    value = np.empty((n_s, k))
    for name, (a_, b_, v_) in values.items():
        sel = session_arm == name
        attract[sel], buy[sel], value[sel] = a_[sel], b_[sel], v_[sel]
    clicked = rng.random((n_s, k)) < theta[:k][None, :] * attract
    bought = clicked & (rng.random((n_s, k)) < buy)
    row = sessions.user_row
    return OnlinePeriod(
        users, arm, np.bincount(row, minlength=users.size),
        np.bincount(row, weights=clicked.sum(axis=1), minlength=users.size).astype(float),
        np.bincount(row, weights=bought.sum(axis=1), minlength=users.size).astype(float),
        np.asarray(np.bincount(row, weights=(bought * value).sum(axis=1),
                               minlength=users.size), dtype=np.float64),
        expected)
