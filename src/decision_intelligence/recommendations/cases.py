"""Milestone 8 case studies: the computations behind the M8 evidence report.

Logging regimes:

* **default** (:func:`default_config`) — moderate exposure bias: the logging policy's
  popularity score is correlated 0.7 with item quality. Fixed before any model was evaluated
  (only the click intercept and rating parameters were calibrated, on descriptive log
  statistics).
* **hype** (:func:`hype_config`) — strong exposure bias: popularity uncorrelated with quality
  and weighted 2.5. **Selected after inspecting model results** in four exploratory regimes,
  as the one that showed a logged-vs-online reversal; it is an illustration, not a
  pre-registered test. :func:`logging_regime_sweep` reports a grid fixed during the M8
  assurance review (before its results were seen), and :func:`fresh_seed_replication`
  repeats the frozen protocol on a generator seed never inspected before.

Every case keeps the three evaluation views apart: logged feedback (observable),
synthetic oracle (ground truth, evaluation only) and simulated online value (the
generator's click model, not a real online test). Tuning uses logged validation data
only.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from decision_intelligence.common.seeding import spawn_generators
from decision_intelligence.data import SyntheticDataConfig, build_synthetic_dataset
from decision_intelligence.data.customers import horizon_end
from decision_intelligence.data.recommendation_scenarios import (
    LoggingPolicySpec,
    RecommendationLogs,
    RecommendationScenario,
    RecommendationScenarioConfig,
    generate_recommendation_scenario,
    resimulate_logging,
)
from decision_intelligence.recommendations import baselines as bl
from decision_intelligence.recommendations import evaluation as ev
from decision_intelligence.recommendations import metrics as m
from decision_intelligence.recommendations import off_policy as op
from decision_intelligence.recommendations import oracle as orc
from decision_intelligence.recommendations.explicit import BiasRatingModel, rating_data
from decision_intelligence.recommendations.interactions import (
    IntArray,
    InteractionData,
    build_interaction_data,
    index_of,
)
from decision_intelligence.recommendations.splits import (
    ChronologicalSplit,
    RankingTask,
    chronological_split,
    random_holdout_tasks,
    ranking_task,
)
from decision_intelligence.statistics.hypothesis_tests import (
    kruskal_wallis,
    one_sample_t_test,
    paired_t_test,
    wilson_interval,
)
from decision_intelligence.statistics.multiple_testing import adjust_p_values

K = 10
TRAIN_DAYS, VALIDATION_DAYS = 84.0, 14.0
FALLBACK_SHRINKAGE = 100.0
REFERENCE = "global popularity"


def default_config(**changes: Any) -> RecommendationScenarioConfig:
    return replace(RecommendationScenarioConfig(), **changes)


def hype_config(**changes: Any) -> RecommendationScenarioConfig:
    return replace(RecommendationScenarioConfig(popularity_quality_correlation=0.0,
                                                logging_popularity_weight=2.5), **changes)


Factory = Callable[[], bl.Recommender]


def _seg() -> bl.Recommender:
    return bl.SegmentPopularity(FALLBACK_SHRINKAGE)


def _factories(named: dict[str, Factory]) -> list[tuple[str, Factory]]:
    return list(named.items())


def _with_fallback(make: Callable[[], bl.Recommender]) -> Factory:
    return lambda: bl.Fallback(make(), _seg())


def candidate_grid(seed: int = 0) -> dict[str, list[Factory]]:
    """Hyperparameter grid per family (personalised models fall back to segment popularity)."""
    return {
        "random": [partial(bl.RandomRecommender, seed)],
        "global popularity": [partial(bl.GlobalPopularity, h) for h in (None, 7.0, 28.0)],
        "segment popularity": [partial(bl.SegmentPopularity, s) for s in (10.0, 100.0, 1000.0)],
        "item-kNN": [_with_fallback(partial(bl.ItemKNN, k)) for k in (20, 50, 100)],
        "user-kNN": [_with_fallback(partial(bl.UserKNN, k)) for k in (25, 50, 100)],
        "content-based": [_with_fallback(bl.ContentBased)],
        "PureSVD": [_with_fallback(partial(bl.PureSVD, f)) for f in (8, 16, 32, 64)],
    }


@dataclass(frozen=True)
class Prepared:
    scenario: RecommendationScenario
    split: ChronologicalSplit
    train_history: InteractionData
    validation: RankingTask
    history: InteractionData
    test: RankingTask
    examination: op.ExaminationEstimate


def prepare(config: RecommendationScenarioConfig) -> Prepared:
    scenario = generate_recommendation_scenario(config)
    logs = scenario.logs
    split = chronological_split(logs, train_days=TRAIN_DAYS, validation_days=VALIDATION_DAYS)
    train_history = build_interaction_data(logs, split.train_end)
    validation = ranking_task(logs, train_history, split.validation_end, name="validation")
    history = build_interaction_data(logs, split.validation_end)
    test = ranking_task(logs, history, split.test_end, name="test")
    examination = op.estimate_examination(logs, split.validation_end)
    return Prepared(scenario, split, train_history, validation, history, test, examination)


# ---------------------------------------------------------------------------
# 1. Data audit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataAudit:
    counts: dict[str, float]
    position: pd.DataFrame
    """``position, impressions, ctr, relative_examination_estimate, relative_truth``."""


def data_audit(p: Prepared) -> DataAudit:
    logs, truth = p.scenario.logs, p.scenario.truth
    imp = logs.impressions
    shown = np.asarray(np.bincount(index_of(p.history.item_ids, imp["item_id"].to_numpy()),
                                   minlength=p.history.n_items), dtype=np.float64)
    top = np.argsort(-shown, kind="stable")[: max(1, int(math.ceil(0.1 * shown.size)))]
    mid = truth.time_in_days(logs.start + (logs.end - logs.start) / 2)[0]
    mean_attr = truth.attractiveness(np.arange(p.history.n_users),
                                     np.full(p.history.n_users, mid)).mean(axis=0)
    train_counts = p.history.user_interaction_counts()
    cold_users = int(np.sum(train_counts[p.test.users] == 0))
    new_items = int(np.sum((logs.items["launch_time"] > p.split.validation_end)
                           & (logs.items["launch_time"] <= p.split.test_end)))
    counts = {
        "users": float(len(logs.users)), "items": float(len(logs.items)),
        "sessions": float(len(logs.sessions)), "impressions": float(len(imp)),
        "clicks": float(imp["clicked"].sum()), "purchases": float(imp["purchased"].sum()),
        "ratings": float(len(logs.ratings)), "click_rate": float(imp["clicked"].mean()),
        "train_density": float(p.history.clicks.mean()),
        "top10pct_items_impression_share": float(shown[top].sum() / shown.sum()),
        "top10pct_items_attractiveness_share": float(mean_attr[top].sum() / mean_attr.sum()),
        "exposure_gini_logging": m.gini(shown),
        "test_users": float(p.test.users.size),
        "test_users_with_clicks": float(p.test.users_with_relevant().sum()),
        "cold_start_test_users": float(cold_users),
        "eligible_items_at_cutoff": float(p.test.eligible.sum()),
        "items_launched_in_test_window": float(new_items),
    }
    est = p.examination
    position = pd.DataFrame({
        "position": est.positions, "impressions": est.n_impressions, "ctr": est.ctr,
        "relative_examination_estimate": est.relative,
        "relative_truth": truth.examination / truth.examination[0]})
    return DataAudit(counts, position)


# ---------------------------------------------------------------------------
# 2. Leakage
# ---------------------------------------------------------------------------


def _models_for_leakage() -> dict[str, Callable[[], bl.Recommender]]:
    return {"global popularity": lambda: bl.GlobalPopularity(),
            "item-kNN (k=50)": lambda: bl.Fallback(bl.ItemKNN(50), _seg()),
            "PureSVD (k=16)": lambda: bl.Fallback(bl.PureSVD(16), _seg())}


@dataclass(frozen=True)
class LeakageStudy:
    random_vs_chronological: pd.DataFrame
    perturbation: pd.DataFrame
    """Max absolute score change after rewriting every post-cutoff log row (should be 0)."""
    n_users: int


def perturb_after(logs: RecommendationLogs, cutoff: pd.Timestamp, seed: int
                  ) -> RecommendationLogs:
    """Rewrite every impression and rating after ``cutoff`` (items, clicks, ratings)."""
    rng = np.random.default_rng(seed)
    imp = logs.impressions.copy()
    late = imp["timestamp"] > cutoff
    imp.loc[late, "item_id"] = rng.choice(logs.items["item_id"].to_numpy(), size=int(late.sum()))
    imp.loc[late, "clicked"] = rng.random(int(late.sum())) < 0.5
    ratings = logs.ratings.copy()
    rl = ratings["timestamp"] > cutoff
    ratings.loc[rl, "rating"] = rng.integers(1, 6, size=int(rl.sum()))
    return replace(logs, impressions=imp, ratings=ratings)


def leakage_study(p: Prepared, *, seed: int = 8) -> LeakageStudy:
    logs = p.scenario.logs
    leaky_history, leaky_task, proper_task = random_holdout_tasks(
        logs, p.history, p.split.test_end, holdout_share=0.5, rng=np.random.default_rng(seed))
    rows = []
    for name, factory in _models_for_leakage().items():
        per = {}
        for label, history, task in (("chronological", p.history, proper_task),
                                     ("random holdout (invalid)", leaky_history, leaky_task)):
            recs = ev.recommend(factory().fit(history), task, K)
            per[label] = ev.logged_metrics(recs, task)
        diff = per["random holdout (invalid)"]["ndcg"].to_numpy() - per["chronological"][
            "ndcg"].to_numpy()
        t = one_sample_t_test(diff, popmean=0.0)
        rows.append({"model": name,
                     "ndcg_chronological": float(per["chronological"]["ndcg"].mean()),
                     "ndcg_random_holdout": float(per["random holdout (invalid)"]["ndcg"].mean()),
                     "recall_chronological": float(per["chronological"]["recall"].mean()),
                     "recall_random_holdout": float(
                         per["random holdout (invalid)"]["recall"].mean()),
                     "inflation": float(diff.mean()), "p_value": t.p_value})
    perturbed = perturb_after(logs, p.split.validation_end, seed)
    h2 = build_interaction_data(perturbed, p.split.validation_end)
    prow = []
    for name, factory in _factories({**_models_for_leakage(),
                          "content-based": lambda: bl.Fallback(bl.ContentBased(), _seg()),
                          "user-kNN (k=50)": lambda: bl.Fallback(bl.UserKNN(50), _seg())}):
        a = factory().fit(p.history).score(p.test.users)
        b = factory().fit(h2).score(p.test.users)
        prow.append({"model": name, "max_abs_score_change": float(np.max(np.abs(a - b)))})
    n = int(proper_task.users_with_relevant().sum())
    return LeakageStudy(pd.DataFrame(rows), pd.DataFrame(prow), n)


# ---------------------------------------------------------------------------
# 3-5. Tuning, benchmark and evaluation views
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Benchmark:
    tuning: pd.DataFrame
    models: dict[str, bl.Recommender]
    recs: dict[str, IntArray]
    logged: pd.DataFrame
    """Per model: mean and 95% bootstrap CI of each logged metric."""
    logged_comparisons: pd.DataFrame
    oracle: pd.DataFrame
    oracle_comparisons: pd.DataFrame
    views: pd.DataFrame
    """Per model: logged NDCG, oracle NDCG, simulated online clicks/purchases/margin, IPS."""
    rank_agreement: pd.DataFrame
    beyond: pd.DataFrame
    per_user_logged: dict[str, pd.DataFrame]
    per_user_oracle: dict[str, pd.DataFrame]
    online: dict[str, orc.OnlineValue]
    ips: dict[str, op.IPSResult]
    fit_seconds: dict[str, float]
    score_seconds: dict[str, float]


def _summary_rows(per_user: dict[str, pd.DataFrame], columns: tuple[str, ...],
                  rng_seed: int) -> pd.DataFrame:
    rows = []
    rngs = spawn_generators(rng_seed, len(per_user) * len(columns))
    j = 0
    for name, frame in per_user.items():
        row: dict[str, object] = {"model": name, "n_users": len(frame)}
        for c in columns:
            s = ev.summarise(frame[c].to_numpy(), rng=rngs[j])
            j += 1
            row[c], row[f"{c}_lo"], row[f"{c}_hi"] = s.mean, s.ci_lower, s.ci_upper
        rows.append(row)
    return pd.DataFrame(rows)


def benchmark(p: Prepared, *, seed: int = 0) -> Benchmark:
    logs, truth = p.scenario.logs, p.scenario.truth
    tuned = ev.tune(candidate_grid(seed), p.train_history, p.validation, K)
    models: dict[str, bl.Recommender] = {}
    recs: dict[str, IntArray] = {}
    fit_s: dict[str, float] = {}
    score_s: dict[str, float] = {}
    for family, factory in tuned.selected.items():
        start = time.perf_counter()
        model = factory().fit(p.history)  # refit on train + validation
        fit_s[family] = time.perf_counter() - start
        start = time.perf_counter()
        scores = model.score(p.test.users)
        score_s[family] = time.perf_counter() - start
        models[family] = model
        recs[family] = ev.top_k(scores, p.test.allowed, K)
    per_logged = {f: ev.logged_metrics(r, p.test) for f, r in recs.items()}
    per_oracle = {f: orc.oracle_metrics(r, p.test, truth, p.history) for f, r in recs.items()}
    logged = _summary_rows(per_logged, ev.LOGGED_METRICS, seed + 1)
    oracle = _summary_rows(per_oracle, ("oracle_ndcg", "oracle_attractiveness",
                                        "oracle_precision"), seed + 2)
    logged_cmp = ev.paired_comparisons({f: d["ndcg"].to_numpy() for f, d in per_logged.items()},
                                       REFERENCE)
    oracle_cmp = ev.paired_comparisons(
        {f: d["oracle_ndcg"].to_numpy() for f, d in per_oracle.items()}, REFERENCE)
    online, ips, rows = {}, {}, []
    after, until = p.split.validation_end, p.split.test_end
    for f, r in recs.items():
        values = {rw: orc.simulated_online_value(logs, p.history, truth, p.test.users, r, after,
                                                 until, reward=rw)
                  for rw in ("clicked", "purchased", "margin")}
        online[f] = values["clicked"]
        ips[f] = op.ips_policy_value(logs, p.history, p.test.users, r, after, until,
                                     p.examination.relative)
        rows.append({"model": f, "logged_ndcg": float(per_logged[f]["ndcg"].mean()),
                     "oracle_ndcg": float(per_oracle[f]["oracle_ndcg"].mean()),
                     "online_clicks": values["clicked"].estimate,
                     "online_purchases": values["purchased"].estimate,
                     "online_margin": values["margin"].estimate,
                     "ips_clicks": ips[f].estimate, "ips_lo": ips[f].ci_lower,
                     "ips_hi": ips[f].ci_upper, "ips_ess": ips[f].effective_sample_size})
    views = pd.DataFrame(rows)
    agree = []
    for col in ("logged_ndcg", "oracle_ndcg", "ips_clicks", "online_margin"):
        tau = stats.kendalltau(views[col], views["online_clicks"])
        best = views["model"].to_numpy()[int(np.argmax(views[col].to_numpy()))]
        best_online = views["model"].to_numpy()[int(np.argmax(views["online_clicks"].to_numpy()))]
        agree.append({"view": col, "kendall_tau_vs_online_clicks": float(tau.statistic),
                      "same_best_model": bool(best == best_online)})
    beyond = pd.DataFrame([{"model": f, **ev.beyond_accuracy(r, p.test, p.history)}
                           for f, r in recs.items()])
    return Benchmark(tuned.table, models, recs, logged, logged_cmp, oracle, oracle_cmp, views,
                     pd.DataFrame(agree), beyond, per_logged, per_oracle, online, ips, fit_s,
                     score_s)


@dataclass(frozen=True)
class PairContrast:
    """One pair of models compared in all three views (A minus B)."""

    model_a: str
    model_b: str
    logged: pd.Series
    oracle: pd.Series
    online: tuple[float, float, float]
    ips: tuple[float, float, float]


def contrast(p: Prepared, bench: Benchmark, a: str, b: str) -> PairContrast:
    def paired(da: np.ndarray, db: np.ndarray) -> pd.Series:
        t = paired_t_test(da, db)
        return pd.Series({"difference": t.estimate, "ci_lower": t.ci_lower,
                          "ci_upper": t.ci_upper, "p_value": t.p_value})

    logged = paired(bench.per_user_logged[a]["ndcg"].to_numpy(),
                    bench.per_user_logged[b]["ndcg"].to_numpy())
    oracle = paired(bench.per_user_oracle[a]["oracle_ndcg"].to_numpy(),
                    bench.per_user_oracle[b]["oracle_ndcg"].to_numpy())
    on = op.clustered_mean(bench.online[a].per_session - bench.online[b].per_session,
                           bench.online[a].session_users)
    ip = op.clustered_mean(bench.ips[a].per_session - bench.ips[b].per_session,
                           bench.ips[a].session_users)
    return PairContrast(a, b, logged, oracle, (on[0], on[2], on[3]), (ip[0], ip[2], ip[3]))


# ---------------------------------------------------------------------------
# 6. Fairness and segments
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FairnessStudy:
    category_exposure: pd.DataFrame
    """Per model and category: position-weighted exposure share, merit share
    (share of the users' oracle top-K lists), and their ratio."""
    groups: pd.DataFrame
    """Per model, attribute (segment/region) and group: mean oracle and logged NDCG."""
    heterogeneity: pd.DataFrame
    """Kruskal–Wallis test of oracle NDCG across groups, per model and attribute, with
    Holm-adjusted p-values over the whole table."""


def fairness_study(p: Prepared, bench: Benchmark, models: tuple[str, ...]) -> FairnessStudy:
    gains = orc.oracle_gains(p.scenario.truth, p.history, p.test)
    masked = np.where(p.test.allowed, gains, -np.inf)
    ideal = np.argsort(-masked, axis=1, kind="stable")[:, :K]
    categories = np.array(sorted(set(p.history.item_category.tolist())))
    w = m.discounts(K)

    def shares(lists: IntArray) -> np.ndarray:
        cat = np.searchsorted(categories, p.history.item_category[lists])
        return np.asarray(np.bincount(cat.ravel(), weights=np.tile(w, lists.shape[0]),
                                      minlength=categories.size) / (w.sum() * lists.shape[0]),
                          dtype=np.float64)

    merit = shares(ideal)
    rows = []
    for name in models:
        expo = shares(bench.recs[name])
        for c, e, mt in zip(categories, expo, merit, strict=True):
            rows.append({"model": name, "category": str(c), "exposure_share": float(e),
                         "merit_share": float(mt), "ratio": float(e / mt) if mt > 0 else math.nan})
    group_rows, het_rows = [], []
    for name in models:
        po = bench.per_user_oracle[name]
        pl = bench.per_user_logged[name].set_index("user_index")
        for attr, values in (("segment", p.history.user_segment),
                             ("region", p.history.user_region)):
            groups = values[po["user_index"].to_numpy()]
            samples = []
            for g in sorted(set(groups.tolist())):
                sel = groups == g
                users_g = po["user_index"].to_numpy()[sel]
                logged_g = pl["ndcg"].reindex(users_g).dropna()
                samples.append(po["oracle_ndcg"].to_numpy()[sel])
                group_rows.append({"model": name, "attribute": attr, "group": g,
                                   "n_users": int(sel.sum()),
                                   "oracle_ndcg": float(samples[-1].mean()),
                                   "logged_ndcg": float(logged_g.mean())})
            kw = kruskal_wallis(*samples)
            means = [float(s.mean()) for s in samples]
            het_rows.append({"model": name, "attribute": attr, "max_gap": max(means) - min(means),
                             "kruskal_wallis_p": kw.p_value})
    het = pd.DataFrame(het_rows)
    het["holm_adjusted_p"] = adjust_p_values(het["kruskal_wallis_p"].to_numpy(),
                                             method="holm").adjusted_p_values
    return FairnessStudy(pd.DataFrame(rows), pd.DataFrame(group_rows), het)


# ---------------------------------------------------------------------------
# 7. Cold start
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColdStartStudy:
    users: pd.DataFrame
    """Per model and variant (with / without fallback): oracle NDCG for warm and cold users."""
    items: pd.DataFrame
    """Ranking only the items launched after the cutoff: oracle NDCG@5 per model."""
    n_cold_users: int
    n_new_items: int


def cold_start_study(p: Prepared) -> ColdStartStudy:
    truth = p.scenario.truth
    counts = p.history.user_interaction_counts()[p.test.users]
    cold = counts == 0
    gains = orc.oracle_gains(truth, p.history, p.test)
    variants: dict[str, Callable[[], bl.Recommender]] = {
        "global popularity": lambda: bl.GlobalPopularity(),
        "segment popularity": lambda: bl.SegmentPopularity(FALLBACK_SHRINKAGE),
        "item-kNN, no fallback": lambda: bl.ItemKNN(50),
        "item-kNN → segment popularity": lambda: bl.Fallback(bl.ItemKNN(50), _seg()),
        "PureSVD, no fallback": lambda: bl.PureSVD(16),
        "PureSVD → segment popularity": lambda: bl.Fallback(bl.PureSVD(16), _seg()),
        "content-based, no fallback": lambda: bl.ContentBased(),
        "random": lambda: bl.RandomRecommender(0),
    }
    rows = []
    for name, factory in variants.items():
        model = factory().fit(p.history)
        scores = model.score(p.test.users)
        recs = ev.top_k(scores, p.test.allowed, K)
        nd = m.ndcg_at_k(recs, gains, p.test.allowed)
        constant = np.ptp(scores[cold], axis=1) == 0 if cold.any() else np.zeros(0, dtype=bool)
        rows.append({"model": name, "warm_oracle_ndcg": float(nd[~cold].mean()),
                     "cold_oracle_ndcg": float(nd[cold].mean()) if cold.any() else math.nan,
                     "cold_users_with_constant_scores": float(constant.mean())
                     if cold.any() else math.nan})
    # New items: launched after the cutoff, before the end of the test window.
    launch = p.scenario.logs.items.set_index("item_id").reindex(p.history.item_ids)["launch_time"]
    new = ((launch > p.split.validation_end) & (launch <= p.split.test_end)).to_numpy()
    warm_users = p.test.users[~cold]
    allowed = np.broadcast_to(new, (warm_users.size, new.size))
    end_t = truth.time_in_days(p.split.test_end)[0]
    new_gains = truth.attractiveness(warm_users, np.full(warm_users.size, end_t))
    k_new = min(5, int(new.sum()))
    item_rows = []
    for name, factory in _factories({"content-based": lambda: bl.ContentBased(),
                          "item-kNN": lambda: bl.ItemKNN(50),
                          "global popularity": lambda: bl.GlobalPopularity(),
                          "random": lambda: bl.RandomRecommender(0)}):
        scores = factory().fit(p.history).score(warm_users)
        recs = ev.top_k(scores, np.asarray(allowed), k_new)
        item_rows.append({"model": name, "oracle_ndcg_new_items": float(
            m.ndcg_at_k(recs, new_gains, np.asarray(allowed)).mean()),
            "scores_constant_on_new_items": bool(np.all(np.ptp(scores[:, new], axis=1) == 0))})
    return ColdStartStudy(pd.DataFrame(rows), pd.DataFrame(item_rows), int(cold.sum()),
                          int(new.sum()))


# ---------------------------------------------------------------------------
# 8. Off-policy diagnostics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IPSRepeated:
    rows: pd.DataFrame
    """Per policy: exact simulated value; IPS with examination estimated from the training
    logs (mean, MC SE, bias); nominal-95% interval coverage with its Wilson interval and an
    exact binomial test against 0.95, the direction of misses, the calibration of the
    standard error (mean SE vs SD of the estimates), skewness and the SE-estimate
    correlation, and coverage with an i.i.d.-session SE; the same with the true
    examination; the clipped estimator; and the bias when position is ignored."""
    n_replications: int


def ips_repeated_sampling(config: RecommendationScenarioConfig, n_replications: int,
                          *, clip: float = 50.0, seed: int = 9000) -> IPSRepeated:
    """Re-draw only the test-window logging; the policy lists and sessions stay fixed.

    The estimand is therefore fixed across replications, and the examination estimate
    (from the fixed training logs) is held fixed too: coverage here is conditional on it.
    """
    p = prepare(config)
    truth = p.scenario.truth
    theta = p.examination.relative
    true_theta = truth.examination / truth.examination[0]
    after, until = p.split.validation_end, p.split.test_end
    z = float(stats.norm.ppf(0.975))
    rows = []
    for name, factory in _factories({"global popularity": lambda: bl.GlobalPopularity(),
                          "item-kNN (k=50)": lambda: bl.Fallback(bl.ItemKNN(50), _seg())}):
        recs = ev.recommend(factory().fit(p.history), p.test, K)
        exact = orc.simulated_online_value(p.scenario.logs, p.history, truth, p.test.users,
                                           recs, after, until).estimate
        est, se, iid_se, lo, hi, true_lo, true_hi = [], [], [], [], [], [], []
        clipped, ignored, oracle_theta = [], [], []
        for r in range(n_replications):
            imp = resimulate_logging(p.scenario, after, seed + r)
            res = op.ips_policy_value(p.scenario.logs, p.history, p.test.users, recs, after,
                                      until, theta, impressions=imp)
            est.append(res.estimate)
            se.append(res.standard_error)
            lo.append(res.ci_lower)
            hi.append(res.ci_upper)
            ps = res.per_session
            iid_se.append(float(ps.std(ddof=1) / math.sqrt(ps.size)))
            clipped.append(op.ips_policy_value(p.scenario.logs, p.history, p.test.users, recs,
                                               after, until, theta, impressions=imp,
                                               clip=clip).estimate)
            ignored.append(op.ips_policy_value(p.scenario.logs, p.history, p.test.users, recs,
                                               after, until, np.ones_like(theta),
                                               impressions=imp).estimate)
            tr = op.ips_policy_value(p.scenario.logs, p.history, p.test.users, recs, after,
                                     until, true_theta, impressions=imp)
            oracle_theta.append(tr.estimate)
            true_lo.append(tr.ci_lower)
            true_hi.append(tr.ci_upper)
        e, s_, lo_a, hi_a = np.array(est), np.array(se), np.array(lo), np.array(hi)
        covered = int(np.sum((lo_a <= exact) & (exact <= hi_a)))
        covered_true = int(np.sum((np.array(true_lo) <= exact) & (exact <= np.array(true_hi))))
        iid = np.array(iid_se)
        covered_iid = int(np.sum((e - z * iid <= exact) & (exact <= e + z * iid)))
        wl, wh = wilson_interval(covered, n_replications)
        rows.append({
            "policy": name, "exact_value": exact, "ips_mean": float(e.mean()),
            "mc_se": float(e.std(ddof=1) / math.sqrt(e.size)),
            "bias": float(e.mean() - exact),
            "coverage": covered / n_replications, "coverage_wilson_lo": wl,
            "coverage_wilson_hi": wh,
            "coverage_binomial_p": float(stats.binomtest(covered, n_replications, 0.95).pvalue),
            "misses_estimate_too_low": int(np.sum(hi_a < exact)),
            "misses_estimate_too_high": int(np.sum(lo_a > exact)),
            "sd_of_estimates": float(e.std(ddof=1)), "mean_se": float(s_.mean()),
            "skewness": float(stats.skew(e)),
            "corr_estimate_se": float(np.corrcoef(e, s_)[0, 1]),
            "coverage_iid_session_se": covered_iid / n_replications,
            "coverage_true_examination": covered_true / n_replications,
            "clipped_mean": float(np.mean(clipped)),
            "clipped_bias": float(np.mean(clipped) - exact), "clip": clip,
            "true_examination_bias": float(np.mean(oracle_theta) - exact),
            "true_examination_mc_se": float(np.std(oracle_theta, ddof=1)
                                            / math.sqrt(len(oracle_theta))),
            "position_ignored_bias": float(np.mean(ignored) - exact),
            "position_ignored_mc_se": float(np.std(ignored, ddof=1) / math.sqrt(len(ignored)))})
    return IPSRepeated(pd.DataFrame(rows), n_replications)


def examination_uncertainty(p: Prepared, bench: Benchmark, models: tuple[str, ...],
                            *, n_bootstrap: int = 200, seed: int = 21) -> pd.DataFrame:
    """Variability of IPS due to *estimating* relative examination, which the IPS interval
    treats as known: re-estimate it on session-level bootstrap resamples of the training
    impressions (test logs fixed) and compare the spread of IPS with its standard error."""
    imp = p.scenario.logs.impressions
    imp = imp.loc[imp["timestamp"] <= p.split.validation_end]
    clicks = imp.pivot_table(index="session_id", columns="position", values="clicked",
                             aggfunc="sum").to_numpy(dtype=np.float64)
    shown = imp.pivot_table(index="session_id", columns="position", values="clicked",
                            aggfunc="size").to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = [rng.integers(0, clicks.shape[0], clicks.shape[0]) for _ in range(n_bootstrap)]
    rows = []
    for name in models:
        args = (p.scenario.logs, p.history, p.test.users, bench.recs[name],
                p.split.validation_end, p.split.test_end)
        base = op.ips_policy_value(*args, p.examination.relative)
        values = []
        for idx in draws:
            ctr = clicks[idx].sum(axis=0) / shown[idx].sum(axis=0)
            values.append(op.ips_policy_value(*args, ctr / ctr[0]).estimate)
        sd = float(np.std(values, ddof=1))
        rows.append({"model": name, "ips": base.estimate, "ips_se": base.standard_error,
                     "sd_from_examination": sd, "ratio_to_se": sd / base.standard_error,
                     "share_of_total_variance": sd ** 2 / (sd ** 2 + base.standard_error ** 2)})
    return pd.DataFrame(rows)


def clipping_sweep(p: Prepared, bench: Benchmark, model: str,
                   clips: tuple[float | None, ...] = (None, 1000.0, 100.0, 10.0)) -> pd.DataFrame:
    exact = bench.online[model].estimate
    rows = []
    for c in clips:
        r = op.ips_policy_value(p.scenario.logs, p.history, p.test.users, bench.recs[model],
                                p.split.validation_end, p.split.test_end,
                                p.examination.relative, clip=c)
        rows.append({"clip": "none" if c is None else c, "estimate": r.estimate,
                     "ci_lower": r.ci_lower, "ci_upper": r.ci_upper,
                     "error_vs_exact": r.estimate - exact, "ess": r.effective_sample_size,
                     "ess_share": r.effective_sample_size / r.n_slots,
                     "clipped_share": r.clipped_share, "max_weight": r.max_weight})
    return pd.DataFrame(rows)


def examination_sensitivity(p: Prepared, bench: Benchmark, model: str) -> pd.DataFrame:
    """IPS with estimated, true and (wrongly) ignored position bias."""
    truth = p.scenario.truth
    exact = bench.online[model].estimate
    rows = []
    for label, theta in (("estimated from logs", p.examination.relative),
                         ("true (oracle)", truth.examination / truth.examination[0]),
                         ("ignored (theta = 1)", np.ones_like(p.examination.relative))):
        r = op.ips_policy_value(p.scenario.logs, p.history, p.test.users, bench.recs[model],
                                p.split.validation_end, p.split.test_end, theta)
        rows.append({"examination": label, "estimate": r.estimate,
                     "error_vs_exact": r.estimate - exact})
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class PositivityStudy:
    rows: pd.DataFrame
    support_size: int


def positivity_study(support_size: int = 150) -> PositivityStudy:
    """Logging restricted to the top ``support_size`` items: IPS sees only that support."""
    p = prepare(default_config(logging_support_size=support_size))
    truth = p.scenario.truth
    after, until = p.split.validation_end, p.split.test_end
    rows = []
    for name, factory in _factories({"global popularity": lambda: bl.GlobalPopularity(),
                          "item-kNN (k=50)": lambda: bl.Fallback(bl.ItemKNN(50), _seg()),
                          "content-based": lambda: bl.Fallback(bl.ContentBased(), _seg())
                          }):
        recs = ev.recommend(factory().fit(p.history), p.test, K)
        full = orc.simulated_online_value(p.scenario.logs, p.history, truth, p.test.users, recs,
                                          after, until)
        sup = orc.simulated_online_value(p.scenario.logs, p.history, truth, p.test.users, recs,
                                         after, until, supported_only=True)
        ips = op.ips_policy_value(p.scenario.logs, p.history, p.test.users, recs, after, until,
                                  p.examination.relative)
        rows.append({"policy": name, "unsupported_share": ips.unsupported_share,
                     "exact_value": full.estimate, "exact_supported_value": sup.estimate,
                     "ips": ips.estimate, "ips_lo": ips.ci_lower, "ips_hi": ips.ci_upper})
    return PositivityStudy(pd.DataFrame(rows), support_size)


# ---------------------------------------------------------------------------
# 9. Explicit ratings (missing not at random)
# ---------------------------------------------------------------------------


def explicit_ratings_case(p: Prepared, *, seed: int = 11, n_random_pairs: int = 20_000
                          ) -> pd.DataFrame:
    logs, truth, h = p.scenario.logs, p.scenario.truth, p.history
    train = rating_data(logs.ratings, h.user_ids, h.item_ids, after=None,
                        until=p.split.validation_end)
    test = rating_data(logs.ratings, h.user_ids, h.item_ids, after=p.split.validation_end,
                       until=p.split.test_end)
    model = BiasRatingModel(5.0, 5.0).fit(train)
    rng = np.random.default_rng(seed)
    users = rng.integers(0, h.n_users, size=n_random_pairs)
    items = rng.choice(np.flatnonzero(h.item_launched), size=n_random_pairs)
    t_mid = truth.time_in_days(orc.window_midpoint(p.test))[0]
    expected = truth.expected_rating(users, np.full(users.size, t_mid))[
        np.arange(users.size), items]
    pred_random = model.predict(users, items)
    global_mean = float(train.rating.mean())
    rows = [
        {"evaluation": "observed held-out ratings (logged, MNAR)", "n": int(test.rating.size),
         "mean_actual": float(test.rating.mean()),
         "mean_predicted": float(model.predict(test.user_index, test.item_index).mean()),
         "rmse_bias_model": m.rmse(model.predict(test.user_index, test.item_index),
                                   test.rating),
         "rmse_global_mean": m.rmse(np.full(test.rating.size, global_mean), test.rating)},
        {"evaluation": "random user-item pairs vs E[rating] (oracle)", "n": n_random_pairs,
         "mean_actual": float(expected.mean()), "mean_predicted": float(pred_random.mean()),
         "rmse_bias_model": m.rmse(pred_random, expected),
         "rmse_global_mean": m.rmse(np.full(expected.size, global_mean), expected)},
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 10. Seed stability
# ---------------------------------------------------------------------------


def _fixed_models() -> dict[str, Factory]:
    return {
        "global popularity": lambda: bl.GlobalPopularity(),
        "segment popularity": lambda: bl.SegmentPopularity(100.0),
        "item-kNN (k=50)": lambda: bl.Fallback(bl.ItemKNN(50), _seg()),
        "user-kNN (k=50)": lambda: bl.Fallback(bl.UserKNN(50), _seg()),
        "content-based": lambda: bl.Fallback(bl.ContentBased(), _seg()),
        "PureSVD (k=16)": lambda: bl.Fallback(bl.PureSVD(16), _seg()),
    }


def seed_stability(seeds: tuple[int, ...], *, hype: bool = False) -> pd.DataFrame:
    """Fixed configurations (no re-tuning) across generator seeds: logged and oracle NDCG."""
    rows = []
    for s in seeds:
        p = prepare(hype_config(seed=s) if hype else default_config(seed=s))
        for name, factory in _fixed_models().items():
            recs = ev.recommend(factory().fit(p.history), p.test, K)
            rows.append({"seed": s, "model": name,
                         "logged_ndcg": float(ev.logged_metrics(recs, p.test)["ndcg"].mean()),
                         "oracle_ndcg": float(orc.oracle_metrics(
                             recs, p.test, p.scenario.truth, p.history)["oracle_ndcg"].mean()),
                         "online_clicks": orc.simulated_online_value(
                             p.scenario.logs, p.history, p.scenario.truth, p.test.users, recs,
                             p.split.validation_end, p.split.test_end).estimate})
    return pd.DataFrame(rows)


# Fixed during the M8 assurance review, before any result of the sweep was seen.
REGIME_CORRELATIONS = (0.7, 0.35, 0.0)
REGIME_WEIGHTS = (1.5, 2.0, 2.5)
FRESH_SEED = 20261114


@dataclass(frozen=True)
class RegimeSweep:
    summary: pd.DataFrame
    """Per regime: Kendall tau of logged NDCG and of IPS with simulated online clicks, the
    best model under each view, and the content-based minus global-popularity gap."""
    per_model: pd.DataFrame


def logging_regime_sweep(
    correlations: tuple[float, ...] = REGIME_CORRELATIONS,
    weights: tuple[float, ...] = REGIME_WEIGHTS,
    **changes: Any,
) -> RegimeSweep:
    """Fixed model configurations across a grid of logging regimes (same users, items,
    preferences and sessions; only the logging policy changes). ``changes`` override other
    scenario parameters (used to run the sweep on a reduced scenario in tests)."""
    rows: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for rho in correlations:
        for weight in weights:
            p = prepare(default_config(popularity_quality_correlation=rho,
                                       logging_popularity_weight=weight, **changes))
            after, until = p.split.validation_end, p.split.test_end
            block: dict[str, dict[str, float]] = {}
            for name, factory in _fixed_models().items():
                recs = ev.recommend(factory().fit(p.history), p.test, K)
                block[name] = {
                    "logged_ndcg": float(ev.logged_metrics(recs, p.test)["ndcg"].mean()),
                    "online_clicks": orc.simulated_online_value(
                        p.scenario.logs, p.history, p.scenario.truth, p.test.users, recs,
                        after, until).estimate,
                    "ips_clicks": op.ips_policy_value(
                        p.scenario.logs, p.history, p.test.users, recs, after, until,
                        p.examination.relative).estimate}
                rows.append({"correlation": rho, "weight": weight, "model": name,
                             **block[name]})
            g = pd.DataFrame.from_dict(block, orient="index")
            gap = {col: block["content-based"][col] - block["global popularity"][col]
                   for col in ("logged_ndcg", "online_clicks", "ips_clicks")}
            summary.append({
                "correlation": rho, "weight": weight,
                "tau_logged_vs_online": float(stats.kendalltau(g["logged_ndcg"],
                                                               g["online_clicks"]).statistic),
                "tau_ips_vs_online": float(stats.kendalltau(g["ips_clicks"],
                                                            g["online_clicks"]).statistic),
                "best_logged": str(g["logged_ndcg"].idxmax()),
                "best_online": str(g["online_clicks"].idxmax()),
                "content_minus_pop_logged": gap["logged_ndcg"],
                "content_minus_pop_online": gap["online_clicks"],
                "content_minus_pop_ips": gap["ips_clicks"],
                "logged_sign_reversed": bool(np.sign(gap["logged_ndcg"])
                                             != np.sign(gap["online_clicks"]))})
    return RegimeSweep(pd.DataFrame(summary), pd.DataFrame(rows))


@dataclass(frozen=True)
class FreshSeedReplication:
    seed: int
    views: dict[str, pd.DataFrame]
    """Per regime (default, hype): the tuned benchmark's views table."""
    contrasts: pd.DataFrame
    """Per regime: content-based minus global popularity in every view."""


def fresh_seed_replication(seed: int = FRESH_SEED, **changes: Any) -> FreshSeedReplication:
    """The frozen protocol (grid, tuning, K, splits, both regimes) on an uninspected seed."""
    views, rows = {}, []
    for regime, cfg in (("default", default_config(seed=seed, **changes)),
                        ("hype", hype_config(seed=seed, **changes))):
        p = prepare(cfg)
        b = benchmark(p)
        views[regime] = b.views
        pc = contrast(p, b, "content-based", REFERENCE)
        rows.append({"regime": regime, "logged": float(pc.logged["difference"]),
                     "logged_lo": float(pc.logged["ci_lower"]),
                     "logged_hi": float(pc.logged["ci_upper"]),
                     "online": pc.online[0], "online_lo": pc.online[1], "online_hi": pc.online[2],
                     "ips": pc.ips[0], "ips_lo": pc.ips[1], "ips_hi": pc.ips[2]})
    return FreshSeedReplication(seed, views, pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# 11. M2 compatibility
# ---------------------------------------------------------------------------


def m2_logs(config: SyntheticDataConfig | None = None) -> tuple[RecommendationLogs, pd.DataFrame]:
    """The M2 recommendation tables mapped (read-only) to the M8 log schema.

    M2 does not log propensities (``logging_propensity`` is NaN), so IPS is unavailable;
    M2 has no content tags (content features are the category alone). Returns the logs
    and the M2 latent affinities (category level; evaluation only).
    """
    data = build_synthetic_dataset(config)
    t = data.tables
    customers, products = t["customers"], t["products"]
    start = pd.Timestamp(data.config.start_date)
    users = pd.DataFrame({"user_id": customers["customer_id"].to_numpy(dtype=np.int64),
                          "segment": customers["segment"].astype(str).to_numpy(),
                          "region": customers["region"].fillna("unknown").astype(str).to_numpy(),
                          "signup_time": pd.to_datetime(customers["signup_date"]).to_numpy()})
    items = pd.DataFrame({"item_id": products["product_id"].to_numpy(dtype=np.int64),
                          "category": products["category"].astype(str).to_numpy(),
                          "price": products["monthly_price"].to_numpy(),
                          "margin_rate": products["margin_rate"].to_numpy(),
                          "launch_time": pd.to_datetime(products["launch_date"]).to_numpy()})
    sessions = t["sessions"].rename(columns={"customer_id": "user_id",
                                             "session_timestamp": "timestamp"})
    exp = t["recommendation_exposures"].merge(sessions[["session_id", "timestamp"]],
                                              on="session_id")
    ev_ = t["events"]
    clicked = set(ev_.loc[ev_["event_type"] == "click", "exposure_id"])
    bought = set(ev_.loc[ev_["event_type"] == "purchase", "exposure_id"])
    impressions = pd.DataFrame({
        "impression_id": exp["exposure_id"].to_numpy(dtype=np.int64),
        "session_id": exp["session_id"].to_numpy(dtype=np.int64),
        "user_id": exp["customer_id"].to_numpy(dtype=np.int64),
        "item_id": exp["product_id"].to_numpy(dtype=np.int64),
        "position": exp["position"].to_numpy(dtype=np.int64),
        "timestamp": pd.to_datetime(exp["timestamp"]).to_numpy(),
        "logging_propensity": np.nan,
        "clicked": exp["exposure_id"].isin(clicked).to_numpy(),
        "purchased": exp["exposure_id"].isin(bought).to_numpy()})
    logs = RecommendationLogs(
        start=start, end=horizon_end(data.config), users=users, items=items,
        sessions=sessions[["session_id", "user_id", "timestamp"]], impressions=impressions,
        ratings=pd.DataFrame(columns=["user_id", "item_id", "timestamp", "rating"]),
        logging_policy=LoggingPolicySpec(math.nan, 0, None, slot_sampling="unknown"))
    return logs, data.latent_affinities


def m2_compatibility_case(k: int = 5) -> pd.DataFrame:
    logs, affinities = m2_logs()
    span = (logs.end - logs.start).total_seconds() / 86_400.0
    split = chronological_split(logs, train_days=0.6 * span, validation_days=0.2 * span)
    history = build_interaction_data(logs, split.validation_end)
    task = ranking_task(logs, history, split.test_end)
    aff = affinities.set_index("customer_id").reindex(history.user_ids)
    cat_cols = list(affinities.columns.drop("customer_id"))
    item_cat = np.array([cat_cols.index(c) for c in history.item_category])
    gains = aff[cat_cols].to_numpy(dtype=np.float64)[task.users][:, item_cat]
    rows = []
    for name, factory in _factories({"random": lambda: bl.RandomRecommender(0),
                          "global popularity": lambda: bl.GlobalPopularity(),
                          "segment popularity": lambda: bl.SegmentPopularity(100.0),
                          "item-kNN (k=10)": lambda: bl.Fallback(bl.ItemKNN(10), _seg()),
                          "content-based (category)": lambda: bl.Fallback(bl.ContentBased(),
                                                                          _seg())}):
        recs = ev.recommend(factory().fit(history), task, k)
        logged = ev.logged_metrics(recs, task)
        rows.append({"model": name, "n_users": len(logged),
                     "logged_hit_rate": float(logged["hit_rate"].mean()),
                     "logged_ndcg": float(logged["ndcg"].mean()),
                     "oracle_ndcg": float(m.ndcg_at_k(recs, gains, task.allowed).mean()),
                     "eligible_items": int(task.eligible.sum())})
    return pd.DataFrame(rows)
