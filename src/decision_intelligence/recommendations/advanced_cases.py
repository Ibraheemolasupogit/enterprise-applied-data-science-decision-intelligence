"""Milestone 9 case studies: the computations behind the M9 evidence report.

Everything follows the frozen :mod:`.protocol`: grids, windows, selection rules, the A/B design
and the repeated-sampling sizes. Model selection uses logged validation data only; ground truth
is read only through :mod:`.oracle`-style evaluation here (never by models, tuning or selection);
simulated online results are labelled SIMULATED.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from decision_intelligence.data.recommendation_scenarios import (
    RecommendationLogs,
    RecommendationScenarioConfig,
    resimulate_logging,
)
from decision_intelligence.recommendations import baselines as bl
from decision_intelligence.recommendations import cases as m8
from decision_intelligence.recommendations import evaluation as ev
from decision_intelligence.recommendations import metrics as m
from decision_intelligence.recommendations import off_policy as op
from decision_intelligence.recommendations import off_policy_advanced as opa
from decision_intelligence.recommendations import online
from decision_intelligence.recommendations import oracle as orc
from decision_intelligence.recommendations import protocol as pr
from decision_intelligence.recommendations.candidates import candidate_recall, generate_candidates
from decision_intelligence.recommendations.cold_start import ContentToFactor, history_bucket
from decision_intelligence.recommendations.factorisation import BPR, EASE, ImplicitALS
from decision_intelligence.recommendations.hybrid import (
    BlendRecommender,
    percentile_ranks,
    simplex_grid,
)
from decision_intelligence.recommendations.interactions import (
    IntArray,
    InteractionData,
    build_interaction_data,
)
from decision_intelligence.recommendations.ranking import (
    LambdaRankRanker,
    PointwiseRanker,
    Ranker,
    TwoStageRecommender,
    training_data,
)
from decision_intelligence.recommendations.two_tower import TwoTower
from decision_intelligence.statistics.hypothesis_tests import wilson_interval

Factory = Callable[[], bl.Recommender]
M9_SINGLE = ("implicit ALS", "BPR", "BPR (exposure negatives)", "EASE", "two-tower")
RANKER_CELLS = (("pointwise", False), ("pointwise", True), ("LambdaRank", False),
                ("LambdaRank", True))


def _seg() -> bl.Recommender:
    return bl.SegmentPopularity(m8.FALLBACK_SHRINKAGE)


def _fb(make: Callable[[], bl.Recommender]) -> Factory:
    return lambda: bl.Fallback(make(), _seg())


def m9_grid(logs: RecommendationLogs, seed: int = 0) -> dict[str, list[Factory]]:
    """The frozen M9 single-stage grids (protocol); personalised models use the M8 fallback."""
    return {
        "implicit ALS": [_fb(partial(ImplicitALS, f, reg, a, pr.ALS_ITERATIONS, seed))
                         for f, reg, a in pr.ALS_GRID],
        "BPR": [_fb(partial(BPR, f, reg, learning_rate=pr.BPR_LEARNING_RATE,
                            epochs=pr.BPR_EPOCHS, seed=seed)) for f, reg in pr.BPR_GRID],
        "BPR (exposure negatives)": [
            _fb(partial(BPR, f, reg, learning_rate=pr.BPR_LEARNING_RATE, epochs=pr.BPR_EPOCHS,
                        seed=seed, exposure_logs=logs)) for f, reg in pr.BPR_GRID],
        "EASE": [_fb(partial(EASE, reg)) for reg in pr.EASE_GRID],
        "two-tower": [partial(TwoTower, d, h, epochs=pr.TWO_TOWER_EPOCHS,
                              learning_rate=pr.TWO_TOWER_LEARNING_RATE, seed=seed)
                      for d, h in pr.TWO_TOWER_GRID],
    }


def history_at(logs: RecommendationLogs, day: float) -> InteractionData:
    return build_interaction_data(logs, logs.start + pd.to_timedelta(day, unit="D"))


# ---------------------------------------------------------------------------
# Selection on the validation window (frozen rules)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Selection:
    tuning: pd.DataFrame
    """Every validation result: ``family, model, validation_ndcg, selected``."""
    single: dict[str, Factory]
    """Selected configuration per single-stage family (M8 and M9)."""
    rankers: dict[str, Callable[[], Ranker]]
    """Selected ranker per cell (pointwise / LambdaRank x naive / IPW)."""
    best_ranker_cell: str
    blend_weights: tuple[float, ...]
    best_single_m9: str
    hybrid: str
    """``"blend"`` or the best two-stage cell."""
    candidate_recall_validation: pd.DataFrame


def _ranker_factory(kind: str, l2: float) -> Callable[[], Ranker]:
    if kind == "pointwise":
        return lambda: PointwiseRanker(l2)
    return lambda: LambdaRankRanker(l2, epochs=pr.LAMBDARANK_EPOCHS,
                                    learning_rate=pr.LAMBDARANK_LEARNING_RATE)


def cell_name(kind: str, ipw: bool) -> str:
    return f"two-stage: {kind}{' IPW' if ipw else ''}"


BLEND_COMPONENTS = ("implicit ALS", "item-kNN", "content-based", "segment popularity")
GENERATORS = ("implicit ALS", "item-kNN", "EASE", "content-based", "segment popularity")


def select(p: m8.Prepared, *, seed: int = 0) -> Selection:
    """Apply the frozen selection rules on the validation window only."""
    logs = p.scenario.logs
    grid = {**m8.candidate_grid(seed), **m9_grid(logs, seed)}
    tuned = ev.tune(grid, p.train_history, p.validation, pr.K, metric=pr.SELECTION_METRIC)
    single = tuned.selected
    rows = [tuned.table]
    # Two-stage: generators at day 70, labels (70, 84], validated with generators at 84.
    gens = {g: single[g] for g in GENERATORS}
    h70 = history_at(logs, pr.RANKER_TRAIN_START)
    fitted70 = {g: f().fit(h70) for g, f in gens.items()}
    theta70 = op.estimate_examination(logs, h70.cutoff).relative
    rankers: dict[str, Callable[[], Ranker]] = {}
    cell_scores: dict[str, float] = {}
    for kind, ipw in RANKER_CELLS:
        data = training_data(logs, h70, fitted70, p.validation.cutoff, theta70,
                             pr.CANDIDATES_PER_GENERATOR, propensity_weighted=ipw)
        grid_l2 = pr.POINTWISE_L2_GRID if kind == "pointwise" else pr.LAMBDARANK_L2_GRID
        best, best_val = None, -np.inf
        for l2 in grid_l2:
            make = _ranker_factory(kind, l2)
            ranker = make().fit(data)
            model = TwoStageRecommender(gens, ranker, logs, pr.CANDIDATES_PER_GENERATOR)
            model.fit(p.train_history)
            val = float(ev.logged_metrics(ev.recommend(model, p.validation, pr.K),
                                          p.validation)["ndcg"].mean())
            rows.append(pd.DataFrame([{"family": cell_name(kind, ipw),
                                       "model": f"{kind} (L2={l2:g})", "validation_ndcg": val,
                                       "selected": False}]))
            if val > best_val:
                best, best_val = make, val
        assert best is not None
        rankers[cell_name(kind, ipw)] = best
        cell_scores[cell_name(kind, ipw)] = best_val
    # Blend: weights on the simplex, components fitted once on the training history.
    comps = {c: single[c] for c in BLEND_COMPONENTS}
    users = p.validation.users
    allowed = p.validation.allowed
    ranks = [percentile_ranks(f().fit(p.train_history).score(users), allowed)
             for f in comps.values()]
    blend_best, blend_val = None, -np.inf
    for w in simplex_grid(len(comps), pr.BLEND_STEP):
        s = sum(wi * r for wi, r in zip(w, ranks, strict=True))
        val = float(ev.logged_metrics(ev.top_k(np.asarray(s), allowed, pr.K),
                                      p.validation)["ndcg"].mean())
        rows.append(pd.DataFrame([{"family": "blend", "model": f"weights {w}",
                                   "validation_ndcg": val, "selected": False}]))
        if val > blend_val:
            blend_best, blend_val = w, val
    assert blend_best is not None
    table = pd.concat(rows, ignore_index=True)
    best_cell = max(cell_scores, key=lambda c: cell_scores[c])
    for fam in [*cell_scores, "blend"]:  # first maximum within each family is selected
        sub = table["family"] == fam
        table.loc[sub, "selected"] = False
        table.loc[table.loc[sub, "validation_ndcg"].idxmax(), "selected"] = True
    m9_val = table.loc[table["family"].isin(M9_SINGLE) & table["selected"]]
    best_single = str(m9_val["family"].to_numpy()[int(np.argmax(
        m9_val["validation_ndcg"].to_numpy()))])
    hybrid = "blend" if blend_val >= cell_scores[best_cell] else best_cell
    # Candidate recall on validation (generators fitted at 84).
    fitted84 = {g: f().fit(p.train_history) for g, f in gens.items()}
    cands = generate_candidates(p.train_history, users, fitted84, pr.CANDIDATES_PER_GENERATOR)
    rec = candidate_recall(cands, p.validation.relevant)
    return Selection(table, single, rankers, best_cell, blend_best, best_single, hybrid,
                     rec.drop(columns="user_index").mean().to_frame("mean").T)


# ---------------------------------------------------------------------------
# Final models on the test window
# ---------------------------------------------------------------------------


def final_models(p: m8.Prepared, sel: Selection, history: InteractionData,
                 ranker_history: InteractionData, ranker_window_end: pd.Timestamp
                 ) -> dict[str, bl.Recommender]:
    """Selected configurations fitted on ``history``; rankers retrained on labels in
    ``(ranker_history.cutoff, ranker_window_end]`` with generators at ``ranker_history``."""
    logs = p.scenario.logs
    models: dict[str, bl.Recommender] = {f: make().fit(history) for f, make in sel.single.items()}
    gens = {g: sel.single[g] for g in GENERATORS}
    fitted_r = {g: f().fit(ranker_history) for g, f in gens.items()}
    theta_r = op.estimate_examination(logs, ranker_history.cutoff).relative
    for kind, ipw in RANKER_CELLS:
        name = cell_name(kind, ipw)
        data = training_data(logs, ranker_history, fitted_r, ranker_window_end, theta_r,
                             pr.CANDIDATES_PER_GENERATOR, propensity_weighted=ipw)
        ranker = sel.rankers[name]().fit(data)
        models[name] = TwoStageRecommender(gens, ranker, logs, pr.CANDIDATES_PER_GENERATOR,
                                           name=name).fit(history)
    models["blend"] = BlendRecommender({c: sel.single[c] for c in BLEND_COMPONENTS},
                                       sel.blend_weights).fit(history)
    return models


@dataclass(frozen=True)
class Views:
    summary: pd.DataFrame
    """Per model: logged and oracle NDCG@10 (bootstrap CIs), logged recall, simulated online
    clicks/purchases/margin (exact), IPS / SNIPS / DR estimates with normal CIs."""
    logged_comparisons: pd.DataFrame
    oracle_comparisons: pd.DataFrame
    agreement: pd.DataFrame
    per_user_logged: dict[str, pd.DataFrame]
    per_user_oracle: dict[str, pd.DataFrame]
    recs: dict[str, IntArray]
    online: dict[str, orc.OnlineValue]
    ope: dict[str, dict[str, opa.ClusteredRatio]]


def out_of_sample_click_model(p: m8.Prepared, theta: np.ndarray, *, kind: str = "score"
                              ) -> np.ndarray:
    """DR click model: ALS (k=32, reg 1, alpha 20) fitted at day 84, click model fitted on
    impressions in (84, 98], applied with ALS refitted at day 98 (test-window logs unused)."""
    logs = p.scenario.logs

    def als() -> ImplicitALS:
        return ImplicitALS(32, 1.0, 20.0, pr.ALS_ITERATIONS)

    cm = opa.fit_click_model(logs, p.train_history, als().fit(p.train_history), theta,
                             until=p.split.validation_end, kind=kind)
    return cm.predict(p.history, als().fit(p.history))


def evaluate(p: m8.Prepared, models: Mapping[str, bl.Recommender], *, seed: int = 0,
             with_ope: bool = True, reference: str = pr.REFERENCE_MODEL) -> Views:
    logs, truth, h, task = p.scenario.logs, p.scenario.truth, p.history, p.test
    after, until = p.split.validation_end, p.split.test_end
    theta = p.examination.relative
    recs = {n: ev.recommend(mod, task, pr.K) for n, mod in models.items()}
    per_l = {n: ev.logged_metrics(r, task) for n, r in recs.items()}
    per_o = {n: orc.oracle_metrics(r, task, truth, h) for n, r in recs.items()}
    q_model = q_oracle = None
    if with_ope:
        q_model = out_of_sample_click_model(p, theta)
        t_mid = truth.time_in_days(orc.window_midpoint(task))[0]
        q_oracle = truth.attractiveness(np.arange(h.n_users), np.full(h.n_users, t_mid))
    rows, online_v, ope = [], {}, {}
    rng = np.random.default_rng(seed)
    for n, r in recs.items():
        lg = ev.summarise(per_l[n]["ndcg"].to_numpy(), rng=rng)
        og = ev.summarise(per_o[n]["oracle_ndcg"].to_numpy(), rng=rng)
        vals = {rw: orc.simulated_online_value(logs, h, truth, task.users, r, after, until,
                                               reward=rw)
                for rw in ("clicked", "purchased", "margin")}
        online_v[n] = vals["clicked"]
        row: dict[str, Any] = {
            "model": n, "logged_ndcg": lg.mean, "logged_lo": lg.ci_lower,
            "logged_hi": lg.ci_upper, "logged_recall": float(per_l[n]["recall"].mean()),
            "oracle_ndcg": og.mean, "oracle_lo": og.ci_lower, "oracle_hi": og.ci_upper,
            "online_clicks": vals["clicked"].estimate,
            "online_purchases": vals["purchased"].estimate,
            "online_margin": vals["margin"].estimate,
            "coverage": m.catalogue_coverage(r, task.eligible)}
        if with_ope:
            assert q_model is not None and q_oracle is not None
            slots = opa.extract_slots(logs, h, task.users, r, after, until)
            est = {"IPS": opa.ips(slots, theta), "SNIPS": opa.snips(slots, theta),
                   "DR": opa.doubly_robust(slots, theta, q_model),
                   "DR (true click model)": opa.doubly_robust(slots, theta, q_oracle)}
            ope[n] = est
            for k, e in est.items():
                lo, hi = e.normal_interval()
                row[k] = e.estimate
                row[f"{k} lo"], row[f"{k} hi"] = lo, hi
        rows.append(row)
    summary = pd.DataFrame(rows)
    lc = ev.paired_comparisons({n: d["ndcg"].to_numpy() for n, d in per_l.items()}, reference)
    oc = ev.paired_comparisons({n: d["oracle_ndcg"].to_numpy() for n, d in per_o.items()},
                               reference)
    agree = []
    cols = ["logged_ndcg", "oracle_ndcg"] + (["IPS", "SNIPS", "DR"] if with_ope else [])
    for col in cols:
        tau = stats.kendalltau(summary[col], summary["online_clicks"])
        agree.append({"view": col, "kendall_tau_vs_online_clicks": float(tau.statistic),
                      "best_model": argmax_model(summary, col)})
    agree.append({"view": "online_clicks (SIMULATED)", "kendall_tau_vs_online_clicks": 1.0,
                  "best_model": argmax_model(summary, "online_clicks")})
    return Views(summary, lc, oc, pd.DataFrame(agree), per_l, per_o, recs, online_v, ope)


def argmax_model(frame: pd.DataFrame, col: str) -> str:
    """Model with the largest ``col`` (first on ties)."""
    return str(frame["model"].to_numpy()[int(np.argmax(frame[col].to_numpy()))])


def test_candidate_recall(p: m8.Prepared, sel: Selection) -> pd.DataFrame:
    """Candidate recall on the test window (generators at 98): logged relevance and the
    users' oracle top-10 (allowed items)."""
    gens = {g: sel.single[g]().fit(p.history) for g in GENERATORS}
    cands = generate_candidates(p.history, p.test.users, gens, pr.CANDIDATES_PER_GENERATOR)
    logged = candidate_recall(cands, p.test.relevant).drop(columns="user_index").mean()
    gains = orc.oracle_gains(p.scenario.truth, p.history, p.test)
    top = ev.top_k(gains, p.test.allowed, pr.K)
    ideal = np.zeros_like(p.test.allowed)
    ideal[np.arange(top.shape[0])[:, None], top] = True
    oracle = candidate_recall(cands, ideal).drop(columns="user_index").mean()
    return pd.DataFrame({"logged relevant": logged, "oracle top-10": oracle})


# ---------------------------------------------------------------------------
# Cold start
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColdStart:
    users: pd.DataFrame
    """Mean oracle NDCG@10 per model and training-history bucket (+ bucket sizes)."""
    items: pd.DataFrame


def cold_start(p: m8.Prepared, sel: Selection, views: Views,
               models: tuple[str, ...]) -> ColdStart:
    counts = p.history.user_interaction_counts()[p.test.users]
    bucket = history_bucket(counts)
    rows = []
    for n in models:
        nd = views.per_user_oracle[n]["oracle_ndcg"].to_numpy()
        for b in ("0", "1-2", "3-5", "6+"):
            sel_b = bucket == b
            rows.append({"model": n, "bucket": b, "n_users": int(sel_b.sum()),
                         "oracle_ndcg": float(nd[sel_b].mean()) if sel_b.any() else math.nan})
    users = pd.DataFrame(rows).pivot(index="model", columns="bucket", values="oracle_ndcg")
    users = users[["0", "1-2", "3-5", "6+"]].reset_index()
    sizes = {b: int((bucket == b).sum()) for b in ("0", "1-2", "3-5", "6+")}
    users.columns = ["model"] + [f"{b} (n={sizes[b]})" for b in ("0", "1-2", "3-5", "6+")]
    # New items launched in the test window, ranked for warm users at the window end.
    truth, h = p.scenario.truth, p.history
    launch = p.scenario.logs.items.set_index("item_id").reindex(h.item_ids)["launch_time"]
    new = ((launch > p.split.validation_end) & (launch <= p.split.test_end)).to_numpy()
    warm = p.test.users[counts > 0]
    allowed = np.broadcast_to(new, (warm.size, new.size))
    end_t = truth.time_in_days(p.split.test_end)[0]
    gains = truth.attractiveness(warm, np.full(warm.size, end_t))
    k_new = min(5, int(new.sum()))
    wrapped = sel.single["implicit ALS"]()
    assert isinstance(wrapped, bl.Fallback) and isinstance(wrapped.primary, ImplicitALS)
    als_cfg = wrapped.primary
    items_rows = []
    new_item_models: dict[str, Factory] = {
        "content-based": bl.ContentBased,
        "two-tower (item tower)": sel.single["two-tower"],
        "content → ALS factors": lambda: ContentToFactor(
            ImplicitALS(als_cfg.factors, als_cfg.reg, als_cfg.alpha, als_cfg.iterations)),
        "implicit ALS": sel.single["implicit ALS"],
        "global popularity": bl.GlobalPopularity,
        "random": partial(bl.RandomRecommender, 0),
    }
    for name, make in new_item_models.items():
        scores = make().fit(h).score(warm)
        recs = ev.top_k(scores, np.asarray(allowed), k_new)
        items_rows.append({"model": name, "oracle_ndcg_new_items": float(
            m.ndcg_at_k(recs, gains, np.asarray(allowed)).mean()),
            "constant_on_new_items": bool(np.all(np.ptp(scores[:, new], axis=1) == 0))})
    return ColdStart(users, pd.DataFrame(items_rows))


# ---------------------------------------------------------------------------
# Off-policy calibration in repeated sampling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OPECalibration:
    rows: pd.DataFrame
    n_replications: int
    n_resamples: int


def ope_calibration(n_replications: int = pr.OPE_REPLICATIONS,
                    n_resamples: int = pr.BOOTSTRAP_RESAMPLES, *, seed: int = 31_000,
                    **changes: Any) -> OPECalibration:
    """Re-draw only the test-window logging (estimand fixed, as M8); for IPS, SNIPS and DR
    (fitted click model, and a deliberately constant one) record bias and the coverage of the
    normal, percentile-bootstrap and studentised-bootstrap 95% intervals."""
    cfg = m8.default_config(**{**pr.OPE_SCENARIO, **changes})
    p = m8.prepare(cfg)
    logs, truth, h = p.scenario.logs, p.scenario.truth, p.history
    after, until = p.split.validation_end, p.split.test_end
    theta = p.examination.relative
    q_fit = out_of_sample_click_model(p, theta)
    q_const = out_of_sample_click_model(p, theta, kind="constant")
    policies: dict[str, bl.Recommender] = {
        "global popularity": bl.GlobalPopularity(),
        "item-kNN (k=50)": bl.Fallback(bl.ItemKNN(50), _seg())}
    rng = np.random.default_rng(seed)
    records = []
    for name, model in policies.items():
        recs = ev.recommend(model.fit(h), p.test, pr.K)
        exact = orc.simulated_online_value(logs, h, truth, p.test.users, recs, after,
                                           until).estimate
        for r in range(n_replications):
            imp = resimulate_logging(p.scenario, after, seed + r)
            slots = opa.extract_slots(logs, h, p.test.users, recs, after, until,
                                      impressions=imp)
            for est_name, est in (("IPS", opa.ips(slots, theta)),
                                  ("SNIPS", opa.snips(slots, theta)),
                                  ("DR (fitted click model)",
                                   opa.doubly_robust(slots, theta, q_fit)),
                                  ("DR (constant click model)",
                                   opa.doubly_robust(slots, theta, q_const))):
                nlo, nhi = est.normal_interval()
                boot = est.bootstrap_intervals(rng, n_resamples)
                records.append({
                    "policy": name, "estimator": est_name, "exact": exact,
                    "estimate": est.estimate, "se": est.standard_error,
                    "normal": nlo <= exact <= nhi,
                    "percentile": boot["percentile"][0] <= exact <= boot["percentile"][1],
                    "studentised": boot["studentised"][0] <= exact <= boot["studentised"][1],
                    "normal_low_miss": nhi < exact,
                    "studentised_low_miss": boot["studentised"][1] < exact})
    df = pd.DataFrame(records)
    out = []
    for key, g in df.groupby(["policy", "estimator"], sort=False):
        pol, est_name = (str(x) for x in key)
        n = len(g)
        row: dict[str, Any] = {"policy": pol, "estimator": est_name,
                               "exact": float(g["exact"].iloc[0]),
                               "mean_estimate": float(g["estimate"].mean()),
                               "bias": float(g["estimate"].mean() - g["exact"].iloc[0]),
                               "mc_se": float(g["estimate"].std(ddof=1) / math.sqrt(n)),
                               "sd": float(g["estimate"].std(ddof=1)),
                               "mean_se": float(g["se"].mean())}
        for iv in ("normal", "percentile", "studentised"):
            k = int(g[iv].sum())
            lo, hi = wilson_interval(k, n)
            row[f"{iv}_coverage"] = k / n
            row[f"{iv}_wilson"] = f"[{lo:.3f}, {hi:.3f}]"
            row[f"{iv}_binom_p"] = float(stats.binomtest(k, n, 0.95).pvalue)
        row["normal_misses_low/high"] = (f"{int(g['normal_low_miss'].sum())}/"
                                         f"{int((~g['normal'] & ~g['normal_low_miss']).sum())}")
        row["studentised_misses_low/high"] = (
            f"{int(g['studentised_low_miss'].sum())}/"
            f"{int((~g['studentised'] & ~g['studentised_low_miss']).sum())}")
        out.append(row)
    return OPECalibration(pd.DataFrame(out), n_replications, n_resamples)


# ---------------------------------------------------------------------------
# SIMULATED A/B test
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ABStudy:
    policies: dict[str, str]
    result: online.ABResult
    table: pd.DataFrame
    aa: online.RepeatedAB
    coverage: online.RepeatedAB
    prediction: pd.DataFrame
    """Predicted per-session click gap vs control from the test window (exact SIMULATED value,
    IPS, SNIPS, DR) beside the A/B estimate per session."""


def ab_study(p: m8.Prepared, sel: Selection, test_views: Views, *,
             aa_replications: int = pr.AA_REPLICATIONS,
             coverage_replications: int = pr.AB_COVERAGE_REPLICATIONS) -> ABStudy:
    """SIMULATED: policies refitted on all logs (<= day 120), lists fixed at day 120."""
    design = pr.AB_DESIGN
    logs = p.scenario.logs
    h120 = history_at(logs, design.start_day)
    arm_model = {"control": pr.REFERENCE_MODEL, "single_model": sel.best_single_m9,
                 "hybrid": sel.hybrid}
    h84 = history_at(logs, pr.TRAIN_DAYS)
    models = final_models(p, sel, h120, h84, p.split.validation_end)
    users = online.experiment_users(p.scenario, h120.cutoff)
    lists = {a: online.policy_lists(models[name], h120, users, pr.K)
             for a, name in arm_model.items()}
    res = online.run_ab(p.scenario, h120, lists, users, design)
    rows = []
    for c in res.comparisons:
        pm = c.results[design.primary_metric]
        rows.append({"comparison": f"{c.treatment} vs control",
                     "effect (clicks/user)": pm.effect.estimate,
                     "interval (97.5%)": f"[{pm.effect.ci_lower:.4f}, {pm.effect.ci_upper:.4f}]",
                     "p": c.p_value, "Bonferroni p": c.bonferroni_p, "Holm p": c.holm_p,
                     "CUPED effect": c.cuped_effect[0],
                     "CUPED interval": f"[{c.cuped_effect[1]:.4f}, {c.cuped_effect[2]:.4f}]",
                     "CUPED variance reduction": c.cuped_variance_reduction,
                     "purchases effect": c.results["purchases per user"].effect.estimate,
                     "margin effect": c.results["margin per user"].effect.estimate,
                     "guardrails": "; ".join(c.guardrails), "decision": c.decision,
                     "SATE clicks (exact)": c.sate["clicks per user"],
                     "covers SATE": c.covered["clicks per user"]})
    aa = online.aa_study(p.scenario, h120, lists["control"], users, design, aa_replications)
    cov = online.coverage_study(p.scenario, h120, lists, users, design, "single_model",
                                coverage_replications)
    # Offline/off-policy predictions of the per-session gap (test window, lists at day 98).
    sessions_per_user = float(np.mean(res.period.sessions))
    pred = []
    for a, name in arm_model.items():
        if a == "control":
            continue
        ref = pr.REFERENCE_MODEL
        gap = {"exact on test window (SIMULATED)":
               test_views.online[name].estimate - test_views.online[ref].estimate}
        for k in ("IPS", "SNIPS", "DR"):
            gap[k] = test_views.ope[name][k].estimate - test_views.ope[ref][k].estimate
        comp = next(c for c in res.comparisons if c.treatment == a)
        gap["A/B estimate per session (SIMULATED)"] = (
            comp.results[design.primary_metric].effect.estimate / sessions_per_user)
        gap["A/B exact SATE per session (SIMULATED)"] = (
            comp.sate["clicks per user"] / sessions_per_user)
        pred.append({"comparison": f"{a} ({name}) vs control ({ref})", **gap})
    return ABStudy(arm_model, res, pd.DataFrame(rows), aa, cov, pd.DataFrame(pred))


# ---------------------------------------------------------------------------
# Seeds, regimes and the fresh seed
# ---------------------------------------------------------------------------


def _fixed(sel: Selection) -> dict[str, Factory]:
    keep = (pr.REFERENCE_MODEL, "global popularity", *M9_SINGLE)
    out: dict[str, Factory] = {n: sel.single[n] for n in keep}
    comps = {c: sel.single[c] for c in BLEND_COMPONENTS}
    out["blend"] = lambda: BlendRecommender(comps, sel.blend_weights)
    return out


def fixed_config_study(sel: Selection, configs: Mapping[str, RecommendationScenarioConfig]
                       ) -> pd.DataFrame:
    """Selected configurations (no re-tuning, no ranker) on other scenarios: logged and oracle
    NDCG and SIMULATED online clicks; Kendall tau of logged vs online per scenario."""
    rows = []
    for label, cfg in configs.items():
        p = m8.prepare(cfg)
        models = {n: f().fit(p.history) for n, f in _fixed(sel).items()}
        v = evaluate(p, models, with_ope=False)
        tau = float(stats.kendalltau(v.summary["logged_ndcg"],
                                     v.summary["online_clicks"]).statistic)
        for r in v.summary.itertuples():
            rows.append({"scenario": label, "model": r.model, "logged_ndcg": r.logged_ndcg,
                         "oracle_ndcg": r.oracle_ndcg, "online_clicks": r.online_clicks,
                         "tau_logged_vs_online": tau})
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class FullRun:
    prepared: m8.Prepared
    selection: Selection
    models: dict[str, bl.Recommender]
    views: Views


def full_run(config: RecommendationScenarioConfig, *, with_ope: bool = True) -> FullRun:
    """The complete frozen protocol on one scenario: select on validation, fit, test."""
    p = m8.prepare(config)
    sel = select(p)
    h84 = p.train_history
    models = final_models(p, sel, p.history, h84, p.split.validation_end)
    return FullRun(p, sel, models, evaluate(p, models, with_ope=with_ope))
