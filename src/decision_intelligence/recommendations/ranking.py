"""Learning-to-rank over candidate sets: pointwise and LambdaRank-style rankers (Milestone 9).

**Labels.** For candidates generated at cutoff ``t``, the label is whether the user clicked the
item in the label window ``(t, t_end]`` (the M8 relevance rule: launched by ``t``, not clicked
before ``t``). Candidates the logging policy never showed in the window are labelled 0 — they
were not *observed*, not necessarily disliked (exposure bias; not corrected here).

**Propensity weighting** corrects *position* bias among shown items. Under the position-based
model a click at position ``p`` is observed with probability ``theta_p`` times relevance, so
``click / theta_p`` is unbiased for relevance of a shown item (Joachims, Swaminathan & Schnabel
2017). Weights use the relative examination ``theta_p / theta_1`` estimated from logs at or
before ``t`` (M8 :func:`~decision_intelligence.recommendations.off_policy.estimate_examination`),
at the position of the item's first click in the window:

* **pointwise**: positives weighted ``1 / theta_p``, negatives 1 — a common heuristic; it does
  not make the logistic loss unbiased (negatives are not reweighted);
* **LambdaRank-style**: each (clicked, unclicked) pair weighted ``1 / theta_p`` of the clicked
  item, the propensity-weighted pairwise objective of Joachims et al. — unbiased for the
  pairwise loss over pairs whose negative was examined, under the position-based model.

**Pointwise ranker**: L2-penalised logistic regression on standardised features (intercept not
penalised), ``(1/N) sum w l(y, x'b) + (l2/2)|b|^2``, minimised by L-BFGS with the analytic
gradient. With ``l2 = 0`` and unit weights it is the maximum-likelihood fit (tested against the
M3 Newton–Raphson :func:`~decision_intelligence.statistics.logistic_regression.fit_logistic`).

**LambdaRank-style ranker** (Burges 2010): linear score ``s = x'w``; per user, every (clicked
``i``, unclicked ``j``) pair contributes the pairwise logistic loss ``log(1 + exp(-(s_i - s_j)))``
weighted by ``|Delta NDCG_ij|`` — the change in NDCG over the user's full candidate list if ``i``
and ``j`` swapped places in the current ranking (gains 1/0, discount ``1/log2(1 + rank)``) —
and by the pair weight (rescaled by one constant so clicked items average 1 — scale only). The
weights ``|Delta NDCG|`` are recomputed each epoch from the current ranking (so the update is
not the gradient of a fixed function and the weighted loss can rise while the ranking improves;
``ndcg_trace`` records training NDCG instead); full-batch gradient descent, averaged over users,
plus ``l2 w``. Deterministic.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize

from decision_intelligence.common.validation import FloatArray
from decision_intelligence.data.recommendation_scenarios import RecommendationLogs
from decision_intelligence.recommendations.baselines import Recommender
from decision_intelligence.recommendations.candidates import CandidateSet, generate_candidates
from decision_intelligence.recommendations.features import FeatureFrame, ranking_features
from decision_intelligence.recommendations.interactions import (
    IntArray,
    InteractionData,
    deduplicate_impressions,
    index_of,
)
from decision_intelligence.recommendations.splits import ranking_task


@dataclass(frozen=True)
class LabelledCandidates:
    features: FeatureFrame
    label: FloatArray
    weight: FloatArray
    """1 for naive training; ``1 / theta_p`` for clicked items with propensity weighting."""
    query: IntArray
    """User index per row (rows are grouped by user)."""


def label_candidates(
    logs: RecommendationLogs, history: InteractionData, candidates: CandidateSet,
    features: FeatureFrame, window_end: pd.Timestamp, relative_examination: FloatArray,
    *, propensity_weighted: bool,
) -> LabelledCandidates:
    """Labels from clicks in ``(cutoff, window_end]``; IPW weights from first-click position."""
    task = ranking_task(logs, history, window_end)
    rel = np.zeros((history.n_users, history.n_items), dtype=np.bool_)
    rel[task.users] = task.relevant
    f = features.frame
    u, i = f["user_index"].to_numpy(), f["item_index"].to_numpy()
    label = rel[u, i].astype(np.float64)
    weight = np.ones_like(label)
    if propensity_weighted:
        imp = deduplicate_impressions(logs.impressions)
        win = imp.loc[imp["clicked"] & (imp["timestamp"] > history.cutoff)
                      & (imp["timestamp"] <= window_end)]
        first = win.groupby(["user_id", "item_id"])["position"].min()
        pos = np.full((history.n_users, history.n_items), 0, dtype=np.int64)
        pos[index_of(history.user_ids, first.index.get_level_values(0).to_numpy()),
            index_of(history.item_ids, first.index.get_level_values(1).to_numpy())] = (
                first.to_numpy(dtype=np.int64))
        p = pos[u, i]
        clicked = label > 0
        weight[clicked] = 1.0 / relative_examination[p[clicked] - 1]
    return LabelledCandidates(features, label, weight, u)


@dataclass
class Standardiser:
    mean: FloatArray
    scale: FloatArray

    @staticmethod
    def fit(x: FloatArray) -> Standardiser:
        sd = x.std(axis=0)
        return Standardiser(x.mean(axis=0), np.where(sd > 0, sd, 1.0))

    def transform(self, x: FloatArray) -> FloatArray:
        return np.asarray((x - self.mean) / self.scale, dtype=np.float64)


def _logistic_objective(beta: FloatArray, x: FloatArray, y: FloatArray, w: FloatArray,
                        l2: float) -> tuple[float, FloatArray]:
    eta = beta[0] + x @ beta[1:]
    n = y.size
    loss = float(np.sum(w * (np.logaddexp(0.0, eta) - y * eta)) / n
                 + 0.5 * l2 * np.sum(beta[1:] ** 2))
    resid = w * (1.0 / (1.0 + np.exp(-eta)) - y) / n
    grad = np.concatenate([[resid.sum()], x.T @ resid + l2 * beta[1:]])
    return loss, grad


class PointwiseRanker:
    def __init__(self, l2: float = 1.0) -> None:
        if l2 < 0:
            raise ValueError("l2 must be non-negative")
        self.l2 = l2
        self.standardiser: Standardiser | None = None
        self.coefficients: FloatArray = np.zeros(0)
        self.converged = False

    def fit(self, data: LabelledCandidates) -> PointwiseRanker:
        raw = data.features.matrix()
        self.standardiser = Standardiser.fit(raw)
        x = self.standardiser.transform(raw)
        res = optimize.minimize(_logistic_objective, np.zeros(x.shape[1] + 1),
                                args=(x, data.label, data.weight, self.l2), jac=True,
                                method="L-BFGS-B",
                                options={"maxiter": 5000, "gtol": 1e-10, "ftol": 1e-15})
        self.coefficients, self.converged = np.asarray(res.x), bool(res.success)
        return self

    def predict(self, raw: FloatArray) -> FloatArray:
        assert self.standardiser is not None
        x = self.standardiser.transform(raw)
        return np.asarray(self.coefficients[0] + x @ self.coefficients[1:], dtype=np.float64)


def delta_ndcg(gains: FloatArray, ranks: IntArray, pos: IntArray, neg: IntArray) -> FloatArray:
    """``|Delta NDCG|`` for swapping each (pos, neg) pair in the current ranking (1-based
    ``ranks``) over the full list; ideal DCG from the list's own gains."""
    disc = 1.0 / np.log2(1.0 + ranks)
    ideal = np.sort(gains)[::-1] / np.log2(2.0 + np.arange(gains.size))
    idcg = ideal.sum()
    if idcg == 0:
        return np.zeros((pos.size, neg.size))
    return np.asarray(np.abs((gains[pos][:, None] - gains[neg][None, :])
                             * (disc[pos][:, None] - disc[neg][None, :])) / idcg)


def ranks_within(scores: FloatArray) -> IntArray:
    """1-based ranks by descending score, ties to the lower index (stable)."""
    order = np.argsort(-scores, kind="stable")
    r = np.empty(scores.size, dtype=np.int64)
    r[order] = np.arange(1, scores.size + 1)
    return r


class LambdaRankRanker:
    def __init__(self, l2: float = 0.1, *, epochs: int = 30, learning_rate: float = 0.05) -> None:
        if l2 < 0 or epochs < 1 or learning_rate <= 0:
            raise ValueError("invalid LambdaRank hyperparameters")
        self.l2, self.epochs, self.learning_rate = l2, epochs, learning_rate
        self.standardiser: Standardiser | None = None
        self.weights: FloatArray = np.zeros(0)
        self.loss_trace: list[float] = []
        self.ndcg_trace: list[float] = []
        """Mean training NDCG (full candidate list) at the start of each epoch. The
        ``loss_trace`` is not a fixed objective (its weights follow the ranking) and can rise
        while the ranking improves; ``ndcg_trace`` is what LambdaRank aims to raise."""

    def _groups(self, data: LabelledCandidates) -> list[IntArray]:
        q = data.query
        bounds = np.flatnonzero(np.diff(q)) + 1
        groups = np.split(np.arange(q.size), bounds)
        return [g for g in groups if 0 < data.label[g].sum() < g.size]

    def gradient(self, w: FloatArray, x: FloatArray, data: LabelledCandidates,
                 groups: list[IntArray]) -> tuple[float, FloatArray]:
        """Weighted pairwise loss and its gradient with ``|Delta NDCG|`` fixed at ``w``."""
        s = x @ w
        grad = np.zeros_like(w)
        loss = 0.0
        for g in groups:
            y = data.label[g]
            pos, neg = np.flatnonzero(y > 0), np.flatnonzero(y == 0)
            delta = delta_ndcg(y, ranks_within(s[g]), pos, neg)
            diff = s[g][pos][:, None] - s[g][neg][None, :]
            pw = data.weight[g][pos][:, None] * delta
            loss += float(np.sum(pw * np.logaddexp(0.0, -diff)))
            lam = pw / (1.0 + np.exp(diff))  # = pw * sigma(-diff)
            xg = x[g]
            grad -= lam.sum(axis=1) @ xg[pos] - lam.sum(axis=0) @ xg[neg]
        n = max(len(groups), 1)
        return loss / n + 0.5 * self.l2 * float(w @ w), grad / n + self.l2 * w

    def fit(self, data: LabelledCandidates) -> LambdaRankRanker:
        # Scale-only normalisation of the pair weights (clicked items average 1): the same
        # weighted objective up to a constant, so the frozen step size behaves as for naive
        # labels. Without it, propensity weights of up to ~6 made full-batch gradient descent
        # oscillate (observed on a development seed during M9 implementation; disclosed).
        positive = data.label > 0
        if positive.any():
            scale = float(np.mean(data.weight[positive]))
            data = LabelledCandidates(data.features, data.label, data.weight / scale, data.query)
        raw = data.features.matrix()
        self.standardiser = Standardiser.fit(raw)
        x = self.standardiser.transform(raw)
        groups = self._groups(data)
        w = np.zeros(x.shape[1])
        self.loss_trace, self.ndcg_trace = [], []
        for _ in range(self.epochs):
            self.ndcg_trace.append(training_ndcg(x @ w, data.label, groups))
            loss, grad = self.gradient(w, x, data, groups)
            self.loss_trace.append(loss)
            w = w - self.learning_rate * grad
        self.weights = w
        return self

    def predict(self, raw: FloatArray) -> FloatArray:
        assert self.standardiser is not None
        return np.asarray(self.standardiser.transform(raw) @ self.weights, dtype=np.float64)


def training_ndcg(scores: FloatArray, labels: FloatArray, groups: list[IntArray]) -> float:
    """Mean NDCG over the full candidate list of each group (binary gains)."""
    vals = []
    for g in groups:
        y = labels[g]
        disc = 1.0 / np.log2(1.0 + ranks_within(scores[g]))
        ideal = np.sort(y)[::-1] / np.log2(2.0 + np.arange(y.size))
        vals.append(float((y * disc).sum() / ideal.sum()))
    return float(np.mean(vals)) if vals else float("nan")


Ranker = PointwiseRanker | LambdaRankRanker
NON_CANDIDATE_SCORE = -1e6


class TwoStageRecommender(Recommender):
    """Fit candidate generators on the history; score candidates with a pre-trained ranker.

    Non-candidates get :data:`NON_CANDIDATE_SCORE` (finite, below every candidate), so the
    M8 ``top_k`` returns ranked candidates first and the ranking stays deterministic.
    """

    def __init__(self, generators: Mapping[str, Callable[[], Recommender]], ranker: Ranker,
                 logs: RecommendationLogs, n_per_generator: int, name: str = "two-stage") -> None:
        super().__init__()
        self.factories = dict(generators)
        self.ranker, self.logs, self.n_per_generator = ranker, logs, n_per_generator
        self.name = name
        self.fitted: dict[str, Recommender] = {}

    def _fit(self, history: InteractionData) -> None:
        self.fitted = {n: f().fit(history) for n, f in self.factories.items()}

    def score(self, users: IntArray) -> FloatArray:
        h = self.history
        uniq = np.unique(users)
        cands = generate_candidates(h, uniq, self.fitted, self.n_per_generator)
        feats = ranking_features(self.logs, h, cands)
        s = self.ranker.predict(feats.matrix())
        out = np.full((uniq.size, h.n_items), NON_CANDIDATE_SCORE)
        row = np.searchsorted(uniq, feats.frame["user_index"].to_numpy())
        out[row, feats.frame["item_index"].to_numpy()] = s
        return np.asarray(out[np.searchsorted(uniq, users)], dtype=np.float64)


def training_data(
    logs: RecommendationLogs, history: InteractionData,
    generators: Mapping[str, Recommender], window_end: pd.Timestamp,
    relative_examination: FloatArray, n_per_generator: int, *, propensity_weighted: bool,
) -> LabelledCandidates:
    """Candidates and features at ``history.cutoff`` for users active in the label window."""
    users = ranking_task(logs, history, window_end).users
    cands = generate_candidates(history, users, generators, n_per_generator)
    feats = ranking_features(logs, history, cands)
    return label_candidates(logs, history, cands, feats, window_end, relative_examination,
                            propensity_weighted=propensity_weighted)
