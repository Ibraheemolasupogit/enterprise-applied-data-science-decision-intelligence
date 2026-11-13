"""Self-normalised IPS, doubly robust estimation and bootstrap intervals (Milestone 9).

Estimand, notation and the IPS estimator are those of M8
(:mod:`decision_intelligence.recommendations.off_policy`): ``V(pi)`` is the expected reward per
logged test session had the fixed target list ``pi_u`` been shown, under the position-based
model (PBM), in units where ``theta_1 = 1`` (relative examination). For logged slot ``j`` (item
``i_j`` at position ``p_j``, propensity ``mu_j``, reward ``y_j``) with target position ``k``:
``rho_j = 1{i_j in pi_u} / mu_j`` and ``r_j = theta_k / theta_{p_j}``.

**IPS** (M8): ``V_s = (1/L) sum_j y_j r_j rho_j``. Unbiased under (1) PBM, (2) known
propensities with independently filled slots, (3) positivity for every list item, (4) known
relative examination.

**SNIPS** (self-normalised IPS; Swaminathan & Joachims 2015, adapted to slots). The control
variate is the item-selection weight: ``E[rho_j] = m_s``, the number of the session's list items
the logging policy could show (known from the logging support and launch dates). With
``D_s = (1/L) sum_j rho_j / m_s`` (``E[D_s] = 1``; ``D_s = 1`` when ``m_s = 0``)::

    V_SN = sum_s A_s / sum_s D_s,    A_s = IPS term of session s.

A ratio estimator: **biased** in finite samples (bias O(1/n)), consistent under assumptions
(1)-(4), usually lower variance. Like IPS it estimates only the supported part of ``V``.

**Doubly robust** (Dudik, Langford & Li 2011, for the PBM slot model). With a click model
``q(u, i)`` for the position-1 click probability::

    V_DR,s = sum_k theta_k q(u, pi_k)  +  (1/L) sum_j rho_j r_j (y_j - theta_{p_j} q(u, i_j)).

Under (1), (2), (4), the correction is unbiased for ``sum_{k supported} theta_k (a_uk - q_uk)``,
so DR is unbiased for the supported part of ``V`` **whatever** ``q`` is, and for unsupported
list items it relies on ``q`` alone (correct only if ``q`` is). Its variance is small when ``q``
is close to the truth. Clicks only (``q`` models clicks).

**Intervals.** Every estimator is a ratio of user-level sums ``sum_u A_u / sum_u D_u`` (IPS and
DR: ``D_u`` = sessions; SNIPS: ``D_u = sum D_s``). The standard error is the cluster-robust
(linearised) one, ``sqrt(G/(G-1) sum_u (A_u - V D_u)^2) / sum_u D_u`` (for IPS identical to
M8). Intervals: normal; percentile cluster bootstrap; and **studentised** cluster bootstrap
(bootstrap-t): resample users, ``t* = (V* - V) / SE*``, interval
``[V - t*_(1-a/2) SE, V - t*_(a/2) SE]``, which can be asymmetric and is designed to correct
skewness. All treat the estimated examination as known.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, stats

from decision_intelligence.common.validation import FloatArray
from decision_intelligence.data.point_in_time import as_of
from decision_intelligence.data.recommendation_scenarios import RecommendationLogs
from decision_intelligence.recommendations.baselines import Recommender
from decision_intelligence.recommendations.interactions import (
    IntArray,
    InteractionData,
    deduplicate_impressions,
    index_of,
)
from decision_intelligence.recommendations.off_policy import (
    item_rewards,
    supported_mask,
    window_sessions,
)


@dataclass(frozen=True)
class SlotData:
    """Logged slots of a window, aligned with one set of target lists."""

    slate_size: int
    session_user: IntArray
    """User index per session (sessions in window order)."""
    lists: IntArray
    """Target list per session (n_sessions x K)."""
    supported: np.ndarray
    """(n_sessions x K): could the logging policy show the list item in the session?"""
    slot_session: IntArray
    slot_user: IntArray
    slot_item: IntArray
    slot_position: IntArray
    slot_target_k: IntArray
    """1-based position of the slot's item in the target list, 0 if absent."""
    slot_mu: FloatArray
    slot_reward: FloatArray


def extract_slots(logs: RecommendationLogs, history: InteractionData, users: IntArray,
                  recs: IntArray, after: pd.Timestamp, until: pd.Timestamp, *,
                  reward: str = "clicked", impressions: pd.DataFrame | None = None) -> SlotData:
    if logs.logging_policy.slot_sampling != "independent":
        raise ValueError("these estimators require independently filled slots")
    k = recs.shape[1]
    slate = logs.logging_policy.slate_size
    if k > slate:
        raise ValueError("list length must not exceed the slate size")
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
    mu = imp["logging_propensity"].to_numpy(dtype=np.float64)
    if np.any(mu <= 0):
        raise ValueError("logged propensities must be positive")
    sup = supported_mask(logs, history, s_user, sessions["timestamp"], full)
    return SlotData(
        slate, s_user, full[s_user], np.asarray(sup),
        np.asarray(pd.Index(sessions["session_id"]).get_indexer(imp["session_id"]),
                   dtype=np.int64),
        u, i, imp["position"].to_numpy(dtype=np.int64), rank[u, i], mu,
        item_rewards(logs, imp, reward))


@dataclass(frozen=True)
class ClusteredRatio:
    """``sum A_u / sum D_u`` with cluster-robust and bootstrap intervals."""

    a: FloatArray
    d: FloatArray

    @property
    def estimate(self) -> float:
        return float(self.a.sum() / self.d.sum())

    @property
    def standard_error(self) -> float:
        g = self.a.size
        v = self.estimate
        return float(np.sqrt(g / (g - 1) * np.sum((self.a - v * self.d) ** 2)) / self.d.sum())

    def normal_interval(self, level: float = 0.95) -> tuple[float, float]:
        z = float(stats.norm.ppf(0.5 + level / 2))
        return self.estimate - z * self.standard_error, self.estimate + z * self.standard_error

    def bootstrap_intervals(self, rng: np.random.Generator, n_resamples: int,
                            level: float = 0.95) -> dict[str, tuple[float, float]]:
        """Percentile and studentised (bootstrap-t) cluster-bootstrap intervals."""
        g = self.a.size
        counts = rng.multinomial(g, np.full(g, 1.0 / g), size=n_resamples).astype(np.float64)
        sa, sd = counts @ self.a, counts @ self.d
        est_b = sa / sd
        ss = (counts @ (self.a * self.a) - 2 * est_b * (counts @ (self.a * self.d))
              + est_b ** 2 * (counts @ (self.d * self.d)))
        se_b = np.sqrt(np.maximum(g / (g - 1) * ss, 0.0)) / sd
        ok = se_b > 0
        t = (est_b[ok] - self.estimate) / se_b[ok]
        lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
        v, se = self.estimate, self.standard_error
        return {"percentile": (float(np.quantile(est_b, lo_q)), float(np.quantile(est_b, hi_q))),
                "studentised": (v - float(np.quantile(t, hi_q)) * se,
                                v - float(np.quantile(t, lo_q)) * se)}


def _clusters(per_session_a: FloatArray, per_session_d: FloatArray, session_user: IntArray
              ) -> ClusteredRatio:
    _, inv = np.unique(session_user, return_inverse=True)
    return ClusteredRatio(np.asarray(np.bincount(inv, weights=per_session_a), dtype=np.float64),
                          np.asarray(np.bincount(inv, weights=per_session_d), dtype=np.float64))


def _ratio_weights(slots: SlotData, theta: FloatArray) -> tuple[FloatArray, FloatArray]:
    k = slots.slot_target_k
    inlist = k > 0
    rho = np.where(inlist, 1.0 / slots.slot_mu, 0.0)
    r = np.where(inlist, theta[np.maximum(k, 1) - 1] / theta[slots.slot_position - 1], 0.0)
    return rho, r


def ips(slots: SlotData, theta: FloatArray) -> ClusteredRatio:
    rho, r = _ratio_weights(slots, theta)
    n_s = slots.session_user.size
    a = np.bincount(slots.slot_session, weights=slots.slot_reward * r * rho,
                    minlength=n_s) / slots.slate_size
    return _clusters(a, np.ones(n_s), slots.session_user)


def snips(slots: SlotData, theta: FloatArray) -> ClusteredRatio:
    rho, r = _ratio_weights(slots, theta)
    n_s = slots.session_user.size
    a = np.bincount(slots.slot_session, weights=slots.slot_reward * r * rho,
                    minlength=n_s) / slots.slate_size
    m = slots.supported.sum(axis=1).astype(np.float64)
    raw = np.bincount(slots.slot_session, weights=rho, minlength=n_s) / slots.slate_size
    d = np.where(m > 0, raw / np.maximum(m, 1.0), 1.0)
    return _clusters(a, d, slots.session_user)


def doubly_robust(slots: SlotData, theta: FloatArray, click_model: FloatArray) -> ClusteredRatio:
    """``click_model``: (n_users x n_items) position-1 click probability ``q(u, i)``."""
    rho, r = _ratio_weights(slots, theta)
    n_s = slots.session_user.size
    k = slots.lists.shape[1]
    dm = (click_model[slots.session_user[:, None], slots.lists] * theta[:k][None, :]).sum(axis=1)
    q_slot = click_model[slots.slot_user, slots.slot_item]
    resid = slots.slot_reward - theta[slots.slot_position - 1] * q_slot
    corr = np.bincount(slots.slot_session, weights=rho * r * resid,
                       minlength=n_s) / slots.slate_size
    return _clusters(dm + corr, np.ones(n_s), slots.session_user)


# ---------------------------------------------------------------------------
# Click (reward) model for DR
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClickModel:
    """``q(u, i) = expit(c0 + c1 z_ui + c2 log(1 + pop_i))``: the position-1 click probability,
    with ``z`` a standardised model score and ``pop`` the item's click count, both computed at a
    cutoff.

    **Fitted out of sample.** The coefficients are learned on impressions in a window
    ``(t0, t1]`` with features from a scorer fitted at ``t0`` (maximum likelihood with
    ``P(click | i at p) = theta_p q(u, i)``), then applied with features from a scorer refitted
    at the application cutoff. Fitting on impressions the scorer was itself trained on makes
    in-sample scores of already-clicked items dominate the fit (found and fixed during M9
    development; see the M9 evidence). ``kind="constant"`` keeps only ``c0`` (a deliberately
    misspecified model).
    """

    feature_cutoff: pd.Timestamp
    label_until: pd.Timestamp
    coefficients: FloatArray
    kind: str

    def predict(self, history: InteractionData, scorer: Recommender) -> FloatArray:
        """``q`` for all users and items with features at ``history.cutoff``."""
        if scorer.history.cutoff != history.cutoff:
            raise ValueError("the scoring model must be fitted at the history cutoff")
        design = click_features(history, scorer)[..., _columns(self.kind)]
        return _expit(design @ self.coefficients)


def _expit(x: FloatArray) -> FloatArray:
    return np.asarray(1.0 / (1.0 + np.exp(-x)), dtype=np.float64)


def _columns(kind: str) -> list[int]:
    if kind not in ("score", "constant"):
        raise ValueError("kind must be 'score' or 'constant'")
    return [0] if kind == "constant" else [0, 1, 2]


def click_features(history: InteractionData, scorer: Recommender) -> FloatArray:
    """(n_users x n_items x 3): constant, standardised score, ``log(1 + item clicks)``."""
    s = scorer.score(np.arange(history.n_users))
    z = (s - s.mean()) / (s.std() if s.std() > 0 else 1.0)
    logpop = np.broadcast_to(np.log1p(history.clicks.sum(axis=0)), s.shape)
    return np.asarray(np.stack([np.ones_like(z), z, logpop], axis=-1), dtype=np.float64)


def fit_click_model(logs: RecommendationLogs, history: InteractionData, scorer: Recommender,
                    theta: FloatArray, *, until: pd.Timestamp, kind: str = "score"
                    ) -> ClickModel:
    """Fit on impressions in ``(history.cutoff, until]`` with features at ``history.cutoff``."""
    if scorer.history.cutoff != history.cutoff:
        raise ValueError("the scoring model must be fitted at the history cutoff")
    if until <= history.cutoff:
        raise ValueError("the label window must end after the feature cutoff")
    cols = _columns(kind)
    imp = as_of(deduplicate_impressions(logs.impressions), "timestamp", until)
    imp = imp.loc[imp["timestamp"] > history.cutoff]
    u = index_of(history.user_ids, imp["user_id"].to_numpy())
    i = index_of(history.item_ids, imp["item_id"].to_numpy())
    y = imp["clicked"].to_numpy(dtype=np.float64)
    th = theta[imp["position"].to_numpy() - 1]
    x = click_features(history, scorer)[u, i][:, cols]

    def nll(c: FloatArray) -> tuple[float, FloatArray]:
        q = _expit(x @ c)
        p = np.clip(th * q, 1e-12, 1 - 1e-12)
        val = -float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
        dp = y / p - (1 - y) / (1 - p)
        grad = -(x.T @ (dp * th * q * (1 - q)))
        return val / y.size, grad / y.size

    res = optimize.minimize(nll, np.zeros(len(cols)), jac=True, method="L-BFGS-B")
    return ClickModel(history.cutoff, until, np.asarray(res.x), kind)
