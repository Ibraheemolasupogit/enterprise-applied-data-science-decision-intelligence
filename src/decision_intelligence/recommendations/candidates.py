"""Candidate generation for two-stage ranking (Milestone 9).

Each fitted generator proposes its top ``n`` *allowed* items per user (launched by the cutoff,
not clicked before it — the M8 rules); the candidate set is their union, with each generator's
score for every candidate kept as a feature and a provenance flag per generator. A second-stage
ranker can only reorder what the first stage proposes, so **candidate recall** — the share of
the evaluation window's relevant items that are in the candidate set — is reported separately
from ranking quality.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from decision_intelligence.recommendations.baselines import Recommender
from decision_intelligence.recommendations.evaluation import top_k
from decision_intelligence.recommendations.interactions import BoolArray, IntArray, InteractionData


def allowed_mask(history: InteractionData, users: IntArray) -> BoolArray:
    """Launched by the cutoff and not clicked before it (as M8 ``RankingTask.allowed``)."""
    return np.asarray(history.item_launched[None, :] & ~(history.clicks[users] > 0),
                      dtype=np.bool_)


@dataclass(frozen=True)
class CandidateSet:
    cutoff: pd.Timestamp
    users: IntArray
    frame: pd.DataFrame
    """``user_index, item_index``, ``score_<generator>`` and ``from_<generator>`` columns."""
    generators: tuple[str, ...]

    def membership(self, n_items: int) -> BoolArray:
        m = np.zeros((self.users.size, n_items), dtype=np.bool_)
        row = np.searchsorted(self.users, self.frame["user_index"].to_numpy())
        m[row, self.frame["item_index"].to_numpy()] = True
        return m


def generate_candidates(history: InteractionData, users: IntArray,
                        generators: Mapping[str, Recommender], n_per_generator: int
                        ) -> CandidateSet:
    """Union of each fitted generator's top ``n`` allowed items per user (users sorted)."""
    users = np.sort(np.asarray(users, dtype=np.int64))
    allowed = allowed_mask(history, users)
    n = min(n_per_generator, int(np.min(allowed.sum(axis=1))))
    scores = {}
    member = np.zeros_like(allowed)
    flags = {}
    for name, model in generators.items():
        if model.history.cutoff != history.cutoff:
            raise ValueError(f"generator {name} was fitted at a different cutoff")
        s = model.score(users)
        scores[name] = s
        top = top_k(s, allowed, n)
        f = np.zeros_like(allowed)
        f[np.arange(users.size)[:, None], top] = True
        flags[name] = f
        member |= f
    r, i = np.nonzero(member)
    frame = pd.DataFrame({"user_index": users[r], "item_index": i.astype(np.int64)})
    for name in generators:
        frame[f"score_{name}"] = scores[name][r, i]
        frame[f"from_{name}"] = flags[name][r, i]
    return CandidateSet(history.cutoff, users, frame, tuple(generators))


def candidate_recall(candidates: CandidateSet, relevant: BoolArray) -> pd.DataFrame:
    """Per user (rows of ``relevant`` align with ``candidates.users``): share of relevant items
    inside the candidate set, overall and per generator; users without relevant items dropped."""
    member = candidates.membership(relevant.shape[1])
    n_rel = relevant.sum(axis=1)
    keep = n_rel > 0
    out = pd.DataFrame({"user_index": candidates.users[keep],
                        "union": (member & relevant).sum(axis=1)[keep] / n_rel[keep]})
    row = np.searchsorted(candidates.users, candidates.frame["user_index"].to_numpy())
    for name in candidates.generators:
        g = np.zeros_like(member)
        sel = candidates.frame[f"from_{name}"].to_numpy()
        g[row[sel], candidates.frame["item_index"].to_numpy()[sel]] = True
        out[name] = (g & relevant).sum(axis=1)[keep] / n_rel[keep]
    out["candidates_per_user"] = member.sum(axis=1)[keep]
    return out
