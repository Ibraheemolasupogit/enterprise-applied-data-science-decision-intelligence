"""Point-in-time ranking features for (user, candidate item) pairs (Milestone 9).

Every feature is computed from data observable at the candidate set's cutoff ``t``: the history
(clicks at or before ``t``, via :mod:`.interactions`) and static item attributes. Item launch
dates are attributes known at launch, and only items launched by ``t`` are candidates.

=========================  =================================================================
Feature                    Definition
=========================  =================================================================
``score_<generator>``      the generator's score (from :class:`.candidates.CandidateSet`)
``log_item_popularity``    ``log(1 + users who clicked the item by t)``
``item_trend``             ``log(1 + clicks in (t-14, t]) - log(1 + clicks in (t-28, t-14])``
``item_age``               ``log(1 + days since launch)`` at ``t``
``category_affinity``      share of the user's clicks (by ``t``) in the item's category
``price_z``, ``margin``    standardised log price; margin rate
``log_user_history``       ``log(1 + user's clicks by t)``
=========================  =================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from decision_intelligence.data.recommendation_scenarios import RecommendationLogs
from decision_intelligence.recommendations.candidates import CandidateSet
from decision_intelligence.recommendations.interactions import InteractionData

TREND_DAYS = 14.0
BASE_FEATURES = ("log_item_popularity", "item_trend", "item_age", "category_affinity",
                 "price_z", "margin", "log_user_history")


@dataclass(frozen=True)
class FeatureFrame:
    cutoff: pd.Timestamp
    frame: pd.DataFrame
    """``user_index, item_index`` and one column per feature."""
    feature_names: tuple[str, ...]

    def matrix(self) -> np.ndarray:
        return np.asarray(self.frame[list(self.feature_names)].to_numpy(dtype=np.float64))


def ranking_features(logs: RecommendationLogs, history: InteractionData,
                     candidates: CandidateSet) -> FeatureFrame:
    if candidates.cutoff != history.cutoff:
        raise ValueError("candidates and history must share the cutoff")
    f = candidates.frame.copy()
    u, i = f["user_index"].to_numpy(), f["item_index"].to_numpy()
    clicks = history.clicks
    f["log_item_popularity"] = np.log1p(clicks.sum(axis=0))[i]
    ev = history.click_events
    t = (history.cutoff - history.start).total_seconds() / 86_400.0
    recent = np.bincount(ev.loc[ev["day"] > t - TREND_DAYS, "item_index"],
                         minlength=history.n_items)
    prior = np.bincount(ev.loc[(ev["day"] > t - 2 * TREND_DAYS) & (ev["day"] <= t - TREND_DAYS),
                               "item_index"], minlength=history.n_items)
    f["item_trend"] = (np.log1p(recent) - np.log1p(prior))[i]
    items = logs.items.set_index("item_id").reindex(history.item_ids)
    age = ((history.cutoff - pd.to_datetime(items["launch_time"])).dt.total_seconds()
           / 86_400.0).to_numpy()
    if np.any(age[i] < 0):
        raise ValueError("a candidate item was launched after the cutoff")
    f["item_age"] = np.log1p(age[i])
    cats = np.array(sorted(set(history.item_category.tolist())))
    cat_idx = np.searchsorted(cats, history.item_category)
    onehot = (cat_idx[:, None] == np.arange(cats.size)[None, :]).astype(np.float64)
    per_user_cat = clicks @ onehot
    n = clicks.sum(axis=1)
    share = np.divide(per_user_cat, n[:, None], out=np.zeros_like(per_user_cat),
                      where=n[:, None] > 0)
    f["category_affinity"] = share[u, cat_idx[i]]
    logp = np.log(items["price"].to_numpy())
    f["price_z"] = ((logp - logp.mean()) / logp.std())[i]
    f["margin"] = items["margin_rate"].to_numpy()[i]
    f["log_user_history"] = np.log1p(n)[u]
    names = tuple(f"score_{g}" for g in candidates.generators) + BASE_FEATURES
    return FeatureFrame(history.cutoff, f, names)
