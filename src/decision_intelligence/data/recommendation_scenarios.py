"""Synthetic recommendation logs with known preferences (Milestone 8).

Additive: nothing here reads or modifies an M2 generator, table or contract. The
six product categories are the M2 ones (:data:`decision_intelligence.data.products.CATEGORIES`).

Why a dedicated generator: the M2 catalogue has 25 products and a category-level
preference, so items within a category are interchangeable and a 25-item ranking
is too short for @K metrics. Here preference is item-level and the catalogue large
enough for item-item similarity and matrix factorisation to be meaningful.

The generating process (all times in days since ``start``):

* **Users** carry a segment and a region (both observable) and a latent taste vector
  ``u_u = m_segment + e_u`` that drifts linearly, ``u_u(t) = u_u + (t / horizon) d_u``.
  Users sign up uniformly over ``[-signup_window_days, horizon)``; users who sign up
  late in the horizon have no history — **cold-start users**.
* **Items** carry a category, binary content tags, a price and a margin (observable)
  and a latent vector ``v_i`` that is partly a function of the tags and category
  (so content-based filtering can work) and partly idiosyncratic (so it cannot be
  perfect), plus a quality bias ``b_i``. A share of items launches during the
  horizon — those launched after a training cutoff are **cold-start items**.
* **True attractiveness** of item ``i`` to user ``u`` at time ``t``:
  ``a_ui(t) = expit(alpha + b_i + u_u(t) . v_i)``.
* **Position-based examination** (PBM): position ``k`` of a slate is examined with
  probability ``theta_k = k ** -examination_decay``; a click happens with probability
  ``theta_k * a_ui(t)``. A purchase follows a click with probability
  ``expit(q0 + q1 * s_ui - q2 * price_z_i)``, where ``s_ui = b_i + u_u(t) . v_i``.
* **Logging policy** (the production system that generated the logs, *not* ground
  truth about preferences): each session shows ``slate_size`` slots, each filled
  **independently** from ``mu(i | u, t) = (1 - eps) softmax(l_ui) + eps / |S_t|`` over
  the supported items ``S_t`` available at ``t`` (launched, and inside the logging
  support if one is set). ``l_ui = beta_pop * pop_i + beta_pers * (u_u + noise) . v_i``
  mixes a popularity score correlated with — but not equal to — quality and a stale,
  noisy personal estimate. Popular items are therefore over-exposed relative to their
  true appeal (exposure bias). The shown item's propensity ``mu`` is logged with each
  impression, as a production system would log it. Independent slots make the
  propensities exact and the logging position-exchangeable; the price is that a slate
  can repeat an item (each slot is then a separate impression).
* **Explicit ratings** (1-5) are left on some clicked items, more often when the user
  liked the item (missing not at random): ``P(rate) = expit(r0 + r1 * s_ui)``, rating
  ``= clip(round(3 + c * (s_ui - s0) + noise), 1, 5)``.

Ground truth (:class:`RecommendationTruth`) is returned separately from the logs
(:class:`RecommendationLogs`). Training, tuning and model selection in
:mod:`decision_intelligence.recommendations` accept only logs; the truth is read by
:mod:`decision_intelligence.recommendations.oracle` after estimation, and establishes
nothing about real-world validity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from decision_intelligence.common.seeding import spawn_generators
from decision_intelligence.common.validation import FloatArray
from decision_intelligence.data.products import CATEGORIES

IntArray = NDArray[np.int64]

SEGMENTS = ("value", "mainstream", "premium", "enthusiast")
REGIONS = ("north", "south", "east", "west")
_SEGMENT_SHARES = (0.3, 0.4, 0.2, 0.1)
_TAG_RATE = 0.2
_N_STAGES = 6  # users, items, preferences, sessions, logging, ratings


def _expit(x: FloatArray) -> FloatArray:
    return np.asarray(1.0 / (1.0 + np.exp(-x)), dtype=np.float64)


@dataclass(frozen=True)
class RecommendationScenarioConfig:
    """Scenario parameters. Defaults: 3,000 users, 600 items, 120 days, exposure-biased logs."""

    seed: int = 20260801
    logging_seed: int | None = None
    """Seed for the logging/click and rating stages only; ``None`` derives it from ``seed``.
    Changing it re-draws slates and clicks while users, items, preferences and sessions stay
    fixed — used for repeated-sampling checks of off-policy estimators."""
    n_users: int = 3000
    n_items: int = 600
    n_factors: int = 8
    n_tags: int = 12
    start: str = "2025-01-01"
    horizon_days: float = 120.0
    signup_window_days: float = 100.0
    new_item_share: float = 0.2
    sessions_per_day: float = 0.2
    slate_size: int = 10
    examination_decay: float = 0.8
    click_intercept: float = -3.0
    item_bias_sd: float = 0.5
    segment_sd: float = 0.5
    user_sd: float = 0.5
    item_factor_scale: float = 0.6
    content_weight: float = 0.5
    category_weight: float = 0.5
    drift_sd: float = 0.3
    logging_popularity_weight: float = 1.5
    logging_personal_weight: float = 1.0
    logging_noise_sd: float = 0.5
    popularity_quality_correlation: float = 0.7
    """Correlation between the logging popularity score and item quality ``b_i``; the rest
    is item-level hype unrelated to appeal."""
    exploration: float = 0.1
    logging_support_size: int | None = None
    """If set, the logging policy only ever shows the ``n`` items with the highest logging
    popularity score (plus nothing else): items outside have zero propensity."""
    purchase_intercept: float = -1.0
    purchase_preference: float = 0.8
    purchase_price: float = 0.5
    rating_centre: float = 1.5
    rating_slope: float = 0.8
    rating_noise_sd: float = 0.7
    rate_intercept: float = -3.0
    rate_slope: float = 1.5
    duplicate_impression_rate: float = 0.0
    """Planted defect: this share of impression rows is written twice (same ``impression_id``)."""

    def __post_init__(self) -> None:
        if self.n_users < 2 or self.n_items < self.slate_size + 1:
            raise ValueError("need at least two users and more items than slate positions")
        if self.n_factors < 1 or self.n_tags < 1 or self.slate_size < 1:
            raise ValueError("n_factors, n_tags and slate_size must be positive")
        if self.horizon_days <= 0 or self.signup_window_days < 0 or self.sessions_per_day <= 0:
            raise ValueError("horizon_days and sessions_per_day must be positive")
        if not 0.0 <= self.new_item_share < 1.0:
            raise ValueError("new_item_share must lie in [0, 1)")
        if not 0.0 <= self.exploration <= 1.0:
            raise ValueError("exploration must lie in [0, 1]")
        if not 0.0 <= self.duplicate_impression_rate < 1.0:
            raise ValueError("duplicate_impression_rate must lie in [0, 1)")
        if self.content_weight ** 2 + self.category_weight ** 2 > 1.0:
            raise ValueError("content_weight**2 + category_weight**2 must not exceed 1")
        if self.logging_support_size is not None and not (
            self.slate_size <= self.logging_support_size <= self.n_items
        ):
            raise ValueError("logging_support_size must lie in [slate_size, n_items]")
        if not -1.0 <= self.popularity_quality_correlation <= 1.0:
            raise ValueError("popularity_quality_correlation must lie in [-1, 1]")
        if self.examination_decay < 0:
            raise ValueError("examination_decay must be non-negative")


@dataclass(frozen=True)
class LoggingPolicySpec:
    """What the platform knows about its own logging policy (no preference information).

    ``support_item_ids`` is ``None`` when every launched item can be shown; otherwise only
    those items can ever be logged. ``slot_sampling`` documents that slate slots are filled
    independently, which makes logging position-exchangeable.
    """

    exploration: float
    slate_size: int
    support_item_ids: tuple[int, ...] | None
    slot_sampling: str = "independent"


@dataclass(frozen=True)
class RecommendationLogs:
    """Everything a real platform would observe. Model inputs come only from here."""

    start: pd.Timestamp
    end: pd.Timestamp
    users: pd.DataFrame
    """``user_id, segment, region, signup_time``."""
    items: pd.DataFrame
    """``item_id, category, tag_00.., price, margin_rate, launch_time``."""
    sessions: pd.DataFrame
    """``session_id, user_id, timestamp``."""
    impressions: pd.DataFrame
    """``impression_id, session_id, user_id, item_id, position, timestamp,
    logging_propensity, clicked, purchased``."""
    ratings: pd.DataFrame
    """``user_id, item_id, timestamp, rating`` (explicit feedback, missing not at random)."""
    logging_policy: LoggingPolicySpec


@dataclass(frozen=True)
class RecommendationTruth:
    """Evaluation-only ground truth. Never a model, tuning or selection input.

    Arrays are indexed by position in ``user_ids`` / ``item_ids`` (both sorted).
    """

    start: pd.Timestamp
    horizon_days: float
    user_ids: IntArray
    item_ids: IntArray
    user_factors: FloatArray
    user_drift: FloatArray
    item_factors: FloatArray
    item_bias: FloatArray
    click_intercept: float
    examination: FloatArray
    """``theta_k`` for positions 1..slate_size (absolute, not relative)."""
    price_z: FloatArray
    item_value: FloatArray
    """Margin per purchase: ``price * margin_rate``."""
    purchase_intercept: float
    purchase_preference: float
    purchase_price: float
    rating_centre: float
    rating_slope: float
    rating_noise_sd: float

    def time_in_days(self, timestamp: pd.Timestamp | pd.Series) -> FloatArray:
        delta = pd.to_datetime(timestamp) - self.start
        if isinstance(delta, pd.Series):
            return np.asarray(delta.dt.total_seconds().to_numpy() / 86_400.0, dtype=np.float64)
        duration = pd.Timedelta(cast(pd.Timedelta, delta))
        return np.asarray([duration.total_seconds() / 86_400.0], dtype=np.float64)

    def utility(self, users: IntArray, times: FloatArray) -> FloatArray:
        """``s_ui(t) = b_i + u_u(t) . v_i`` for each requested (user, time) row, all items."""
        u = self.user_factors[users] + (times / self.horizon_days)[:, None] * self.user_drift[users]
        return np.asarray(u @ self.item_factors.T + self.item_bias[None, :], dtype=np.float64)

    def attractiveness(self, users: IntArray, times: FloatArray) -> FloatArray:
        """``a_ui(t)``: click probability if examined (rows: requests, columns: items)."""
        return _expit(self.click_intercept + self.utility(users, times))

    def purchase_given_click(self, users: IntArray, times: FloatArray) -> FloatArray:
        s = self.utility(users, times)
        return _expit(self.purchase_intercept + self.purchase_preference * s
                      - self.purchase_price * self.price_z[None, :])

    def expected_rating(self, users: IntArray, times: FloatArray) -> FloatArray:
        """``E[rating]`` of the rounded, clipped normal rating model (exact)."""
        mean = 3.0 + self.rating_slope * (self.utility(users, times) - self.rating_centre)
        edges = np.array([1.5, 2.5, 3.5, 4.5])
        cdf = stats.norm.cdf((edges[None, None, :] - mean[..., None]) / self.rating_noise_sd)
        # E[R] = 1 + sum_k P(R > k) for R in {1..5}
        return np.asarray(1.0 + (1.0 - cdf).sum(axis=-1), dtype=np.float64)


@dataclass(frozen=True)
class LoggingModel:
    """The generator's internal logging-policy scores, kept to re-simulate logging."""

    logits: FloatArray
    """``l_ui`` (users x items)."""
    supported: NDArray[np.bool_]
    """Items the logging policy may show (before the launch-time restriction)."""
    launch_days: FloatArray


@dataclass(frozen=True)
class RecommendationScenario:
    config: RecommendationScenarioConfig
    logs: RecommendationLogs
    truth: RecommendationTruth = field(repr=False)
    logging_model: LoggingModel = field(repr=False)


def _users(cfg: RecommendationScenarioConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = cfg.n_users
    segment = rng.choice(SEGMENTS, size=n, p=_SEGMENT_SHARES)
    region = rng.choice(REGIONS, size=n)
    signup = rng.uniform(-cfg.signup_window_days, cfg.horizon_days, size=n)
    return pd.DataFrame({"user_id": np.arange(1, n + 1, dtype=np.int64), "segment": segment,
                         "region": region, "signup_day": signup})


def _items(cfg: RecommendationScenarioConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = cfg.n_items
    category = rng.choice(CATEGORIES, size=n)
    tags = (rng.random((n, cfg.n_tags)) < _TAG_RATE).astype(np.int64)
    price = np.round(np.exp(rng.normal(math.log(12.0), 0.4, size=n)), 2)
    margin = np.clip(rng.normal(0.5, 0.12, size=n), 0.1, 0.9)
    new = rng.random(n) < cfg.new_item_share
    launch = np.where(new, rng.uniform(0.0, cfg.horizon_days, size=n),
                      rng.uniform(-365.0, 0.0, size=n))
    frame = pd.DataFrame({"item_id": np.arange(1, n + 1, dtype=np.int64), "category": category})
    for j in range(cfg.n_tags):
        frame[f"tag_{j:02d}"] = tags[:, j]
    frame["price"] = price
    frame["margin_rate"] = margin
    frame["launch_day"] = launch
    return frame


def _preferences(
    cfg: RecommendationScenarioConfig, users: pd.DataFrame, items: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    d = cfg.n_factors
    segment_means = rng.normal(0.0, cfg.segment_sd, size=(len(SEGMENTS), d))
    seg_index = np.array([SEGMENTS.index(s) for s in users["segment"]])
    user_factors = segment_means[seg_index] + rng.normal(0.0, cfg.user_sd, size=(len(users), d))
    user_drift = rng.normal(0.0, cfg.drift_sd, size=(len(users), d))
    tags = items[[c for c in items.columns if c.startswith("tag_")]].to_numpy(dtype=float)
    mixing = rng.normal(0.0, 1.0, size=(cfg.n_tags, d))
    content = (tags - _TAG_RATE) @ mixing / math.sqrt(cfg.n_tags * _TAG_RATE * (1 - _TAG_RATE))
    category_emb = rng.normal(0.0, 1.0, size=(len(CATEGORIES), d))
    cat_index = np.array([CATEGORIES.index(c) for c in items["category"]])
    idio_weight = math.sqrt(1.0 - cfg.content_weight ** 2 - cfg.category_weight ** 2)
    item_factors = cfg.item_factor_scale * (
        cfg.content_weight * content + cfg.category_weight * category_emb[cat_index]
        + idio_weight * rng.normal(0.0, 1.0, size=(len(items), d)))
    item_bias = rng.normal(0.0, cfg.item_bias_sd, size=len(items))
    # Logging popularity: correlated with quality, plus item-level hype unrelated to appeal.
    rho = cfg.popularity_quality_correlation
    pop = (rho * item_bias / cfg.item_bias_sd
           + math.sqrt(1.0 - rho ** 2) * rng.normal(0.0, 1.0, size=len(items)))
    pop = (pop - pop.mean()) / pop.std()
    stale = user_factors + rng.normal(0.0, cfg.logging_noise_sd, size=user_factors.shape)
    logits = (cfg.logging_popularity_weight * pop[None, :]
              + cfg.logging_personal_weight * stale @ item_factors.T)
    return user_factors, user_drift, item_factors, item_bias, logits, pop


def _sessions(cfg: RecommendationScenarioConfig, users: pd.DataFrame,
              rng: np.random.Generator) -> pd.DataFrame:
    first = np.maximum(users["signup_day"].to_numpy(), 0.0)
    exposure = cfg.horizon_days - first
    counts = rng.poisson(cfg.sessions_per_day * exposure)
    user_index = np.repeat(np.arange(len(users)), counts)
    day = first[user_index] + rng.random(user_index.size) * exposure[user_index]
    order = np.lexsort((user_index, day))
    user_index, day = user_index[order], day[order]
    return pd.DataFrame({"session_id": np.arange(1, user_index.size + 1, dtype=np.int64),
                         "user_index": user_index, "day": day})


def simulate_impressions(
    scenario_parts: tuple[RecommendationTruth, LoggingModel, RecommendationScenarioConfig],
    sessions: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Fill each session's slate from the logging policy and draw clicks and purchases.

    ``sessions`` needs ``session_id, user_index, day``. Returns one row per slot with
    ``user_index, item_index, position, day, logging_propensity, clicked, purchased``.
    """
    truth, logging, cfg = scenario_parts
    n_items = logging.logits.shape[1]
    L = cfg.slate_size
    user_index = sessions["user_index"].to_numpy(dtype=np.int64)
    day = sessions["day"].to_numpy(dtype=np.float64)
    items_out = np.empty((user_index.size, L), dtype=np.int64)
    prop_out = np.empty((user_index.size, L), dtype=np.float64)
    for s in range(user_index.size):
        allowed = logging.supported & (logging.launch_days <= day[s])
        logits = np.where(allowed, logging.logits[user_index[s]], -np.inf)
        z = np.exp(logits - logits[allowed].max())
        mu = (1.0 - cfg.exploration) * z / z.sum() + cfg.exploration * allowed / allowed.sum()
        cdf = np.cumsum(mu)
        draws = np.searchsorted(cdf, rng.random(L) * cdf[-1], side="right")
        draws = np.minimum(draws, n_items - 1)
        items_out[s] = draws
        prop_out[s] = mu[draws]
    rows = np.repeat(np.arange(user_index.size), L)
    item_flat = items_out.ravel()
    users_flat = user_index[rows]
    day_flat = day[rows]
    position = np.tile(np.arange(1, L + 1), user_index.size)
    s_ui = (np.einsum("nd,nd->n", truth.user_factors[users_flat]
                      + (day_flat / truth.horizon_days)[:, None] * truth.user_drift[users_flat],
                      truth.item_factors[item_flat]) + truth.item_bias[item_flat])
    attract = _expit(truth.click_intercept + s_ui)
    clicked = rng.random(item_flat.size) < truth.examination[position - 1] * attract
    buy_p = _expit(truth.purchase_intercept + truth.purchase_preference * s_ui
                   - truth.purchase_price * truth.price_z[item_flat])
    purchased = clicked & (rng.random(item_flat.size) < buy_p)
    return pd.DataFrame({
        "session_id": sessions["session_id"].to_numpy()[rows], "user_index": users_flat,
        "item_index": item_flat, "position": position, "day": day_flat,
        "logging_propensity": prop_out.ravel(), "clicked": clicked, "purchased": purchased,
        "utility": s_ui,
    })


def _ratings(cfg: RecommendationScenarioConfig, impressions: pd.DataFrame,
             rng: np.random.Generator) -> pd.DataFrame:
    clicked = impressions.loc[impressions["clicked"]]
    s = clicked["utility"].to_numpy()
    rated = rng.random(s.size) < _expit(cfg.rate_intercept + cfg.rate_slope * s)
    raw = (3.0 + cfg.rating_slope * (s - cfg.rating_centre)
           + rng.normal(0.0, cfg.rating_noise_sd, size=s.size))
    rating = np.clip(np.round(raw), 1, 5).astype(np.int64)
    out = clicked.loc[rated, ["user_index", "item_index", "day"]].copy()
    out["rating"] = rating[rated]
    return out


def generate_recommendation_scenario(
    config: RecommendationScenarioConfig | None = None,
) -> RecommendationScenario:
    """Build users, items, sessions, logged impressions and ratings, plus the ground truth."""
    cfg = config or RecommendationScenarioConfig()
    rngs = spawn_generators(cfg.seed, _N_STAGES)
    if cfg.logging_seed is not None:
        rngs[4], rngs[5] = spawn_generators(cfg.logging_seed, 2)
    start = pd.Timestamp(cfg.start)
    users = _users(cfg, rngs[0])
    items = _items(cfg, rngs[1])
    u, drift, v, b, logits, pop = _preferences(cfg, users, items, rngs[2])
    supported = np.ones(cfg.n_items, dtype=bool)
    if cfg.logging_support_size is not None:
        supported = np.zeros(cfg.n_items, dtype=bool)
        supported[np.argsort(-pop, kind="stable")[: cfg.logging_support_size]] = True
    price = items["price"].to_numpy()
    price_z = (np.log(price) - np.log(price).mean()) / np.log(price).std()
    truth = RecommendationTruth(
        start=start, horizon_days=cfg.horizon_days,
        user_ids=users["user_id"].to_numpy(dtype=np.int64),
        item_ids=items["item_id"].to_numpy(dtype=np.int64),
        user_factors=u, user_drift=drift, item_factors=v, item_bias=b,
        click_intercept=cfg.click_intercept,
        examination=np.arange(1, cfg.slate_size + 1, dtype=np.float64) ** -cfg.examination_decay,
        price_z=price_z, item_value=price * items["margin_rate"].to_numpy(),
        purchase_intercept=cfg.purchase_intercept, purchase_preference=cfg.purchase_preference,
        purchase_price=cfg.purchase_price, rating_centre=cfg.rating_centre,
        rating_slope=cfg.rating_slope,
        rating_noise_sd=cfg.rating_noise_sd)
    logging = LoggingModel(logits=logits, supported=supported,
                           launch_days=items["launch_day"].to_numpy())
    sessions = _sessions(cfg, users, rngs[3])
    imp = simulate_impressions((truth, logging, cfg), sessions, rngs[4])
    ratings = _ratings(cfg, imp, rngs[5])
    logs = _assemble_logs(cfg, start, users, items, sessions, imp, ratings, supported, rngs[4])
    return RecommendationScenario(cfg, logs, truth, logging)


def _to_time(start: pd.Timestamp, days: NDArray[np.float64] | pd.Series) -> pd.Series:
    return pd.Series(start + pd.to_timedelta(np.asarray(days, dtype=np.float64), unit="D"))


def impressions_frame(
    start: pd.Timestamp, user_ids: IntArray, item_ids: IntArray, raw: pd.DataFrame,
    first_impression_id: int = 1,
) -> pd.DataFrame:
    """Convert :func:`simulate_impressions` output to the logged impression schema."""
    return pd.DataFrame({
        "impression_id": np.arange(first_impression_id, first_impression_id + len(raw),
                                   dtype=np.int64),
        "session_id": raw["session_id"].to_numpy(dtype=np.int64),
        "user_id": user_ids[raw["user_index"].to_numpy()],
        "item_id": item_ids[raw["item_index"].to_numpy()],
        "position": raw["position"].to_numpy(dtype=np.int64),
        "timestamp": _to_time(start, raw["day"].to_numpy()).to_numpy(),
        "logging_propensity": raw["logging_propensity"].to_numpy(),
        "clicked": raw["clicked"].to_numpy(dtype=bool),
        "purchased": raw["purchased"].to_numpy(dtype=bool),
    })


def _assemble_logs(
    cfg: RecommendationScenarioConfig, start: pd.Timestamp, users: pd.DataFrame,
    items: pd.DataFrame, sessions: pd.DataFrame, imp: pd.DataFrame, ratings: pd.DataFrame,
    supported: NDArray[np.bool_], rng: np.random.Generator,
) -> RecommendationLogs:
    user_ids = users["user_id"].to_numpy(dtype=np.int64)
    item_ids = items["item_id"].to_numpy(dtype=np.int64)
    users_out = users[["user_id", "segment", "region"]].copy()
    users_out["signup_time"] = _to_time(start, users["signup_day"]).to_numpy()
    items_out = items.drop(columns="launch_day").copy()
    items_out["launch_time"] = _to_time(start, items["launch_day"]).to_numpy()
    sessions_out = pd.DataFrame({
        "session_id": sessions["session_id"].to_numpy(dtype=np.int64),
        "user_id": user_ids[sessions["user_index"].to_numpy()],
        "timestamp": _to_time(start, sessions["day"]).to_numpy()})
    impressions = impressions_frame(start, user_ids, item_ids, imp)
    if cfg.duplicate_impression_rate > 0:
        dup = impressions.loc[rng.random(len(impressions)) < cfg.duplicate_impression_rate]
        impressions = pd.concat([impressions, dup]).sort_values(
            ["impression_id"], kind="stable").reset_index(drop=True)
    ratings_out = pd.DataFrame({
        "user_id": user_ids[ratings["user_index"].to_numpy()],
        "item_id": item_ids[ratings["item_index"].to_numpy()],
        "timestamp": _to_time(start, ratings["day"]).to_numpy(),
        "rating": ratings["rating"].to_numpy(dtype=np.int64)})
    support = None if supported.all() else tuple(int(i) for i in item_ids[supported])
    return RecommendationLogs(
        start=start, end=start + pd.to_timedelta(cfg.horizon_days, unit="D"),
        users=users_out, items=items_out, sessions=sessions_out, impressions=impressions,
        ratings=ratings_out,
        logging_policy=LoggingPolicySpec(cfg.exploration, cfg.slate_size, support))


def resimulate_logging(
    scenario: RecommendationScenario, after: pd.Timestamp, logging_seed: int,
) -> pd.DataFrame:
    """Re-draw slates and clicks for the sessions after ``after``, everything else fixed.

    Used to check off-policy estimators in repeated sampling: the users, items, preferences,
    sessions and the logging policy are unchanged; only the logging randomness is new.
    """
    truth, cfg = scenario.truth, scenario.config
    sessions = scenario.logs.sessions
    keep = sessions["timestamp"] > after
    day = truth.time_in_days(sessions.loc[keep, "timestamp"])
    user_index = np.searchsorted(truth.user_ids, sessions.loc[keep, "user_id"].to_numpy())
    frame = pd.DataFrame({"session_id": sessions.loc[keep, "session_id"].to_numpy(),
                          "user_index": user_index, "day": day})
    raw = simulate_impressions((truth, scenario.logging_model, cfg), frame,
                               np.random.default_rng(logging_seed))
    return impressions_frame(truth.start, truth.user_ids, truth.item_ids, raw)
