"""Logged-feedback evaluation, uncertainty, paired comparisons and tuning.

Nothing here accepts ground truth: this module is everything a real team could compute
from its own logs. Synthetic-oracle scores live in :mod:`.oracle`, off-policy estimates
in :mod:`.off_policy`.

Uncertainty is over *users*: per-user metrics are averaged, and intervals are percentile
bootstrap intervals (M3 :func:`~decision_intelligence.common.bootstrap.bootstrap_ci`)
resampling users with the fitted models held fixed — they ignore training-data
variability, which the evidence reports separately across seeds. Model comparisons are
paired over the same users (M3 paired t-test on per-user differences) and the family of
comparisons against a reference is Holm-adjusted (M6
:func:`~decision_intelligence.statistics.multiple_testing.adjust_p_values`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from decision_intelligence.common.bootstrap import bootstrap_ci
from decision_intelligence.common.validation import FloatArray
from decision_intelligence.recommendations import metrics as m
from decision_intelligence.recommendations.baselines import Recommender
from decision_intelligence.recommendations.interactions import IntArray, InteractionData
from decision_intelligence.recommendations.splits import RankingTask
from decision_intelligence.statistics.hypothesis_tests import paired_t_test
from decision_intelligence.statistics.multiple_testing import adjust_p_values

LOGGED_METRICS = ("precision", "recall", "hit_rate", "map", "mrr", "ndcg")


def top_k(scores: FloatArray, allowed: m.BoolArray, k: int) -> IntArray:
    """The ``k`` highest-scoring allowed items per row; ties go to the lower item index.

    Raises if a row has fewer than ``k`` allowed items or a non-finite allowed score —
    a silent ``-inf`` in the list would put an ineligible item into the evaluation.
    """
    if scores.shape != allowed.shape:
        raise ValueError("scores and allowed must have the same shape")
    if k < 1 or np.any(allowed.sum(axis=1) < k):
        raise ValueError(f"every user needs at least k={k} allowed items")
    if not np.all(np.isfinite(scores[allowed])):
        raise ValueError("allowed items must have finite scores")
    masked = np.where(allowed, scores, -np.inf)
    return np.asarray(np.argsort(-masked, axis=1, kind="stable")[:, :k], dtype=np.int64)


def recommend(model: Recommender, task: RankingTask, k: int) -> IntArray:
    return top_k(model.score(task.users), task.allowed, k)


def logged_metrics(recs: IntArray, task: RankingTask) -> pd.DataFrame:
    """Per-user logged metrics for users with at least one relevant item (rows: users)."""
    keep = task.users_with_relevant()
    r, rel, allowed = recs[keep], task.relevant[keep], task.allowed[keep]
    return pd.DataFrame({
        "user_index": task.users[keep],
        "precision": m.precision_at_k(r, rel),
        "recall": m.recall_at_k(r, rel),
        "hit_rate": m.hit_rate_at_k(r, rel),
        "map": m.average_precision_at_k(r, rel),
        "mrr": m.reciprocal_rank_at_k(r, rel),
        "ndcg": m.ndcg_at_k(r, rel.astype(np.float64), allowed),
    })


@dataclass(frozen=True)
class MetricSummary:
    mean: float
    ci_lower: float
    ci_upper: float
    n: int


def summarise(values: FloatArray, *, rng: np.random.Generator, n_resamples: int = 2000,
              confidence_level: float = 0.95) -> MetricSummary:
    """Mean with a percentile bootstrap interval over users."""
    x = np.asarray(values, dtype=np.float64)
    res = bootstrap_ci(x, lambda a: float(np.mean(a)), rng=rng, n_resamples=n_resamples,
                       confidence_level=confidence_level, method="percentile")
    return MetricSummary(float(x.mean()), res.ci_lower, res.ci_upper, int(x.size))


def paired_comparisons(
    per_user: Mapping[str, FloatArray], reference: str, *, alpha: float = 0.05,
    correction: str = "holm",
) -> pd.DataFrame:
    """Each model minus ``reference`` on the same users: paired t, Holm-adjusted."""
    if reference not in per_user:
        raise ValueError("reference model missing")
    base = np.asarray(per_user[reference], dtype=np.float64)
    rows = []
    for name, values in per_user.items():
        if name == reference:
            continue
        res = paired_t_test(np.asarray(values, dtype=np.float64), base)
        rows.append({"model": name, "difference": res.estimate, "ci_lower": res.ci_lower,
                     "ci_upper": res.ci_upper, "p_value": res.p_value})
    frame = pd.DataFrame(rows)
    adj = adjust_p_values(frame["p_value"].to_numpy(), method=correction, alpha=alpha)
    frame["adjusted_p_value"] = adj.adjusted_p_values
    frame["significant"] = adj.rejected
    return frame


def item_user_counts(history: InteractionData) -> FloatArray:
    return np.asarray(history.clicks.sum(axis=0), dtype=np.float64)


def beyond_accuracy(recs: IntArray, task: RankingTask, history: InteractionData
                    ) -> dict[str, float]:
    """Coverage, exposure concentration, novelty, diversity, personalisation."""
    exp = m.exposure(recs, history.n_items, position_weighted=True)[task.eligible]
    return {
        "coverage": m.catalogue_coverage(recs, task.eligible),
        "exposure_gini": m.gini(exp),
        "exposure_entropy": m.normalised_entropy(exp),
        "top10pct_share": m.top_share(exp, 0.1),
        "novelty": float(m.novelty(recs, item_user_counts(history), history.n_users).mean()),
        "diversity": float(m.intra_list_diversity(recs, history.item_features).mean()),
        "personalisation": m.personalisation(recs, history.n_items),
    }


@dataclass(frozen=True)
class TuningResult:
    table: pd.DataFrame
    """``family, model, validation_<metric>, selected``."""
    selected: dict[str, Callable[[], Recommender]]


def tune(
    candidates: Mapping[str, Sequence[Callable[[], Recommender]]],
    history: InteractionData,
    validation: RankingTask,
    k: int,
    *,
    metric: str = "ndcg",
) -> TuningResult:
    """Pick each family's configuration by the mean logged validation ``metric``.

    Fits on ``history`` (data up to the training cutoff) and scores the validation
    window; ties go to the first candidate. Accepts only logged data — ground truth
    cannot enter model selection through this interface.
    """
    if metric not in LOGGED_METRICS:
        raise ValueError(f"metric must be one of {LOGGED_METRICS}")
    if validation.cutoff != history.cutoff:
        raise ValueError("the validation task must start at the history cutoff")
    rows: list[dict[str, object]] = []
    selected: dict[str, Callable[[], Recommender]] = {}
    for family, factories in candidates.items():
        values = []
        for factory in factories:
            model = factory().fit(history)
            values.append(float(logged_metrics(recommend(model, validation, k), validation)[
                metric].mean()))
            rows.append({"family": family, "model": model.name,
                         f"validation_{metric}": values[-1]})
        best = int(np.argmax(values))  # first maximum on ties
        selected[family] = factories[best]
        for j in range(len(factories)):
            rows[len(rows) - len(factories) + j]["selected"] = j == best
    return TuningResult(pd.DataFrame(rows), selected)
