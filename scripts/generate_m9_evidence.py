"""Generate the Milestone 9 (advanced recommendation and ranking) evidence report.

Writes ``docs/evidence/m9-advanced-recommendation-results.md``. All randomness is seeded; with
the recorded dependency versions every number reproduces exactly, except the operational
timings (§11) and the runtime table. The fresh seed (protocol ``FRESH_SEED``) is evaluated
only here, in the last section, after the protocol and code were frozen.

Usage:
    python scripts/generate_m9_evidence.py
    python scripts/generate_m9_evidence.py --dry-run OUT.md   # skip the fresh seed
"""

from __future__ import annotations

import math
import platform
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy

from decision_intelligence.recommendations import advanced_cases as ac
from decision_intelligence.recommendations import cases as m8
from decision_intelligence.recommendations import operational as opn
from decision_intelligence.recommendations import protocol as pr
from decision_intelligence.recommendations.factorisation import fit_als
from decision_intelligence.recommendations.off_policy import estimate_examination
from decision_intelligence.recommendations.ranking import LambdaRankRanker, training_data
from decision_intelligence.statistics.hypothesis_tests import wilson_interval

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "docs" / "evidence" / "m9-advanced-recommendation-results.md"
_CACHE: dict[str, Any] = {}
M8_COVERAGE = {"global popularity": 0.921, "item-kNN (k=50)": 0.930}  # M8 results §11


def table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    def fmt(value: object) -> str:
        if isinstance(value, bool | np.bool_):
            return "yes" if value else "no"
        if isinstance(value, float | np.floating):
            value = float(value)
            if math.isnan(value):
                return "—"
            if math.isinf(value):
                return "inf"
            if value.is_integer() and abs(value) >= 1e4:
                return str(int(value))
            return f"{value:.4g}" if abs(value) >= 1e-3 or value == 0 else f"{value:.2e}"
        return str(value)

    def cell(value: object) -> str:
        return fmt(value).replace("|", "\\|")

    lines = ["| " + " | ".join(cell(h) for h in headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def frame_table(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> str:
    cols = list(columns) if columns is not None else list(frame.columns)
    return table(cols, frame[cols].itertuples(index=False, name=None))


def ci(lo: float, hi: float, digits: int = 4) -> str:
    return f"[{lo:.{digits}f}, {hi:.{digits}f}]"


def main_run() -> ac.FullRun:
    if "main" not in _CACHE:
        _CACHE["main"] = ac.full_run(m8.default_config(seed=pr.MAIN_SEED))
    run = _CACHE["main"]
    assert isinstance(run, ac.FullRun)
    return run


def ab() -> ac.ABStudy:
    if "ab" not in _CACHE:
        r = main_run()
        _CACHE["ab"] = ac.ab_study(r.prepared, r.selection, r.views)
    out = _CACHE["ab"]
    assert isinstance(out, ac.ABStudy)
    return out


# ---------------------------------------------------------------------------


def protocol_section() -> str:
    d = pr.AB_DESIGN
    rows = [
        ["seeds", f"main {pr.MAIN_SEED} (the M8 scenario); stability {pr.STABILITY_SEEDS}; "
                  f"fresh {pr.FRESH_SEED} (§12 only)"],
        ["K, selection metric", f"{pr.K}; mean logged NDCG@{pr.K} on validation"],
        ["windows", "single-stage: fit ≤ 84, validate (84, 98], refit ≤ 98, test (98, 120]; "
                    "ranker: features at 70, labels (70, 84], validated at 84; final ranker "
                    "labels (84, 98] with features at 84, applied at 98"],
        ["grids", f"ALS {len(pr.ALS_GRID)}, BPR {len(pr.BPR_GRID)} (x2 negative samplers), "
                  f"EASE {len(pr.EASE_GRID)}, two-tower {len(pr.TWO_TOWER_GRID)}, pointwise "
                  f"L2 {pr.POINTWISE_L2_GRID}, LambdaRank L2 {pr.LAMBDARANK_L2_GRID}, blend "
                  f"simplex step {pr.BLEND_STEP}"],
        ["reference for paired tests", pr.REFERENCE_MODEL],
        ["A/B arms and selection", d.selection_rule],
        ["A/B primary / secondary", f"{d.primary_metric} / {', '.join(d.secondary_metrics)}"],
        ["A/B guardrails (non-inferiority margins)",
         ", ".join(f"{n} −{m:g}" for n, m in d.guardrails)],
        ["A/B multiplicity", f"{d.multiplicity}: each comparison at α/2 = {d.alpha / 2:g}, "
                             f"{1 - d.alpha / 2:.1%} intervals; Holm p also reported"],
        ["A/B minimum worthwhile effect", f"{d.minimum_worthwhile_effect:g} clicks per user"],
        ["repeated sampling", f"off-policy {pr.OPE_REPLICATIONS} × {pr.BOOTSTRAP_RESAMPLES} "
                              f"bootstrap; A/A {pr.AA_REPLICATIONS}; A/B coverage "
                              f"{pr.AB_COVERAGE_REPLICATIONS}"],
    ]
    return (
        "All evaluation choices were fixed in `recommendations/protocol.py` before any M9 model "
        "was evaluated.\n\n" + table(["item", "frozen value"], rows)
        + "\n\n**Changes after the protocol was written, all before any main-seed validation "
        "or test result was seen** (each prompted by a development-seed diagnostic, never by "
        "a validation or test score):\n\n"
        "1. Added to the protocol before any evaluation: the paired-test reference model, the "
        "fallback rule, and how the four ranker cells are tuned.\n"
        "2. DR click model fitted **out of sample** (scorer at an earlier cutoff, labels in the "
        "next window): the first, in-sample version learned that previously clicked items have "
        "high scores and predicted near-zero clicks for the unseen items lists contain, making "
        "DR collapse to IPS.\n"
        "3. LambdaRank pair weights rescaled by one constant (clicked items average 1) — scale "
        "only — after the propensity-weighted variant's training NDCG oscillated at the frozen "
        "step size; a training-NDCG trace replaced the (non-fixed) loss as the monitor.\n"
        "4. Pointwise ranker solver tolerances tightened (L-BFGS stopped early; found by the "
        "comparison with the M3 logistic regression).\n"
    )


def selection_section() -> str:
    sel = main_run().selection
    t = sel.tuning
    chosen = t[t["selected"]].sort_values("validation_ndcg", ascending=False)
    return (
        f"Validation window (84, 98]; {len(t)} configurations evaluated. Selected configuration "
        "per family:\n\n" + frame_table(chosen, ["family", "model", "validation_ndcg"])
        + f"\n\nBest M9 single-stage family: **{sel.best_single_m9}**. Best ranker cell: "
        f"**{sel.best_ranker_cell}**. Blend weights (ALS, item-kNN, content-based, segment "
        f"popularity): {sel.blend_weights}. Hybrid for the A/B test: **{sel.hybrid}**.\n\n"
        "Candidate recall on validation (generators at day 84; share of each user's "
        "validation clicks in the candidate set):\n\n"
        + frame_table(sel.candidate_recall_validation.round(6))
    )


def recall_section() -> str:
    r = main_run()
    rec = ac.test_candidate_recall(r.prepared, r.selection)
    return (
        "Candidate recall on the test window — reported separately from ranking quality: a "
        "ranker can only reorder what the generators propose. Columns: share of each user's "
        "logged test clicks, and of their oracle top-10, inside the candidate set.\n\n"
        + frame_table(rec.reset_index().rename(columns={"index": "generator"}))
        + f"\n\nReading: the union of {len(ac.GENERATORS)} generators × "
        f"{pr.CANDIDATES_PER_GENERATOR} contains {rec.loc['union', 'logged relevant']:.1%} of "
        f"logged test clicks and {rec.loc['union', 'oracle top-10']:.1%} of oracle top-10 "
        "items; the rest are unreachable for every two-stage model.\n"
    )


def benchmark_section() -> str:
    v = main_run().views
    s = v.summary
    rows = [[r.model, f"{r.logged_ndcg:.4f} {ci(r.logged_lo, r.logged_hi)}", r.logged_recall,
             f"{r.oracle_ndcg:.4f} {ci(r.oracle_lo, r.oracle_hi)}", r.online_clicks,
             r.online_purchases, r.online_margin, r.coverage] for r in s.itertuples()]
    lc = v.logged_comparisons
    oc = v.oracle_comparisons
    cmp_rows = []
    for (_, a), (_, b) in zip(lc.iterrows(), oc.iterrows(), strict=True):
        cmp_rows.append([a["model"], f"{a['difference']:+.4f} {ci(a['ci_lower'], a['ci_upper'])}",
                         a["adjusted_p_value"],
                         f"{b['difference']:+.4f} {ci(b['ci_lower'], b['ci_upper'])}",
                         b["adjusted_p_value"]])
    best_l = ac.argmax_model(s, "logged_ndcg")
    best_o = ac.argmax_model(s, "oracle_ndcg")
    best_on = ac.argmax_model(s, "online_clicks")
    return (
        "Test window (98, 120], selected configurations refitted on everything ≤ day 98. "
        "Logged NDCG@10 over users with a test click; oracle NDCG@10 over all active users "
        "(95% bootstrap intervals over users, models fixed). Simulated online value is the "
        "exact expected reward per test session under the generator's click model "
        "(**SIMULATED**).\n\n"
        + table(["model", "logged NDCG@10", "logged Recall@10", "oracle NDCG@10",
                 "sim. clicks/session", "sim. purchases/session", "sim. margin/session",
                 "catalogue coverage"], rows)
        + f"\n\nPaired comparisons with {pr.REFERENCE_MODEL} (per-user differences, paired t, "
        "Holm across all models):\n\n"
        + table(["model", "logged NDCG Δ [95% CI]", "Holm p", "oracle NDCG Δ [95% CI]",
                 "Holm p"], cmp_rows)
        + f"\n\nReading: best logged NDCG — {best_l}; best oracle NDCG — {best_o}; best "
        f"simulated online clicks — {best_on}.\n"
    )


def views_section() -> str:
    v = main_run().views
    s = v.summary
    rows = [[r["model"], r["online_clicks"],
             *[f"{r[k]:.4f} {ci(r[k + ' lo'], r[k + ' hi'])}"
               for k in ("IPS", "SNIPS", "DR", "DR (true click model)")]]
            for _, r in s.iterrows()]
    err = {k: float(np.mean(np.abs(s[k] - s["online_clicks"])))
           for k in ("IPS", "SNIPS", "DR", "DR (true click model)")}
    width = {k: float(np.mean(s[k + " hi"] - s[k + " lo"]))
             for k in ("IPS", "SNIPS", "DR", "DR (true click model)")}
    return (
        "Off-policy estimates of each model's simulated online click value from the test-window "
        "logs (examination estimated from logs ≤ day 98; normal 95% intervals). The DR click "
        "model is fitted out of sample (ALS at day 84, labels (84, 98], applied with ALS at "
        "day 98). *DR (true click model)* plugs in the generator's attractiveness — a "
        "diagnostic of DR's best case, not an available estimator.\n\n"
        + table(["model", "exact (SIMULATED)", "IPS", "SNIPS", "DR", "DR (true click model)"],
                rows)
        + "\n\nMean absolute error vs exact / mean interval width: "
        + "; ".join(f"{k} {err[k]:.4f} / {width[k]:.4f}" for k in err)
        + ".\n\nRank agreement with the simulated online click ranking:\n\n"
        + frame_table(v.agreement) + "\n"
    )


def ranker_section() -> str:
    r = main_run()
    s = r.views.summary.set_index("model")
    cells = [ac.cell_name(k, w) for k, w in ac.RANKER_CELLS]
    rows = [[c, s.loc[c, "logged_ndcg"], s.loc[c, "oracle_ndcg"], s.loc[c, "online_clicks"]]
            for c in cells]
    # Training-NDCG traces of the final LambdaRank rankers (labels (84, 98]).
    p, sel = r.prepared, r.selection
    logs = p.scenario.logs
    gens = {g: sel.single[g]().fit(p.train_history) for g in ac.GENERATORS}
    theta = estimate_examination(logs, p.train_history.cutoff).relative
    traces = []
    for ipw in (False, True):
        data = training_data(logs, p.train_history, gens, p.split.validation_end, theta,
                             pr.CANDIDATES_PER_GENERATOR, propensity_weighted=ipw)
        ranker = sel.rankers[ac.cell_name("LambdaRank", ipw)]()
        assert isinstance(ranker, LambdaRankRanker)
        ranker.fit(data)
        tr = ranker.ndcg_trace
        traces.append([f"LambdaRank{' IPW' if ipw else ''}", tr[0], tr[len(tr) // 2], tr[-1],
                       max(tr)])
    return (
        "The four ranker cells (each with its validation-selected L2), on the test window:\n\n"
        + table(["ranker", "logged NDCG@10", "oracle NDCG@10", "sim. clicks/session"], rows)
        + "\n\nFinal LambdaRank training NDCG (full candidate list) at the start, middle and end "
        "of training, and its maximum:\n\n"
        + table(["ranker", "epoch 0", "middle", "last", "max"], traces)
        + "\n\nPropensity weighting corrects position bias among *shown* items only; candidates "
        "the logging policy never showed in the label window remain labelled 0.\n"
    )


def cold_start_section() -> str:
    r = main_run()
    cs = ac.cold_start(r.prepared, r.selection, r.views,
                       ("segment popularity", "item-kNN", "implicit ALS", "BPR", "EASE",
                        "two-tower", "blend", ac.cell_name("pointwise", False)))
    return (
        "**Users** by distinct training clicks (oracle NDCG@10 on the test window). Bucket "
        "'0' users are served by each model's cold policy: segment popularity for the "
        "single-stage personalised models (identical numbers), user features for the "
        "two-tower model; the two-stage ranker reorders their candidates (so it differs).\n\n"
        + frame_table(cs.users)
        + "\n\n**New items** (launched during the test window; warm users; ranking only those "
        "items; oracle NDCG@5 at the window end):\n\n" + frame_table(cs.items) + "\n"
    )


def ope_section() -> str:
    cal = ac.ope_calibration()
    rows = cal.rows
    cols = ["policy", "estimator", "exact", "bias", "mc_se", "sd", "mean_se",
            "normal_coverage", "normal_wilson", "normal_binom_p", "percentile_coverage",
            "studentised_coverage", "studentised_wilson", "studentised_binom_p",
            "normal_misses_low/high", "studentised_misses_low/high"]
    ips = rows[rows["estimator"] == "IPS"].set_index("policy")
    compare = [[pol, M8_COVERAGE[pol], ips.loc[pol, "normal_coverage"],
                ips.loc[pol, "studentised_coverage"]] for pol in ips.index]
    better = int((rows["studentised_coverage"] > rows["normal_coverage"]).sum())
    sig_normal = rows.loc[rows["normal_binom_p"] < 0.05, ["policy", "estimator"]]
    sig_stud = rows.loc[rows["studentised_binom_p"] < 0.05, ["policy", "estimator"]]
    return (
        f"Repeated sampling as in M8 §11: {cal.n_replications} re-draws of the test-window "
        f"logging ({pr.OPE_SCENARIO['n_users']} users × {pr.OPE_SCENARIO['n_items']} items; "
        "the estimand is fixed). Each interval nominally 95%; bootstrap "
        f"{cal.n_resamples} cluster (user) resamples. Monte Carlo SE of a coverage near 0.95: "
        f"{math.sqrt(0.95 * 0.05 / cal.n_replications):.4f}. Different logging seeds from M8, "
        "so the IPS normal coverage is a fresh replicate of M8's.\n\n"
        + frame_table(rows, cols)
        + "\n\nIPS normal-interval coverage, M8 vs this replicate, and the studentised "
        "bootstrap:\n\n"
        + table(["policy", "M8 normal", "M9 normal", "M9 studentised"], compare)
        + f"\n\nReading: the studentised interval covers more often than the normal one in "
        f"{better} of {len(rows)} estimator–policy cells. Coverage is significantly below 95% "
        f"(exact binomial p < 0.05) for the normal interval in {len(sig_normal)} cells and for "
        f"the studentised interval in {len(sig_stud)} cells. Bias is reported in absolute terms "
        "with its Monte Carlo SE; DR with a deliberately constant click model remains "
        "unbiased (its correction term), as theory predicts.\n"
    )


def ab_section() -> str:
    a = ab()
    res = a.result
    aa = a.aa.rows
    k = int((aa["p_value"] <= 0.05).sum())
    lo, hi = wilson_interval(k, len(aa))
    cov = a.coverage.rows
    return (
        "**SIMULATED.** A user-randomised test of 14 days from day 120, arms fixed by the "
        "frozen selection rule before any outcome was simulated: "
        + ", ".join(f"{arm} = {name}" for arm, name in a.policies.items())
        + f". Lists fixed at day 120 (models refitted on all logs ≤ 120). Users: "
        f"{sum(res.arm_counts.values())}; arm counts {res.arm_counts}; SRM p = "
        f"{res.srm_p_value:.3g} ({'passed' if res.srm_passed else 'FAILED'}).\n\n"
        + frame_table(a.table)
        + f"\n\n**A/A** ({len(aa)} re-randomisations, both arms the control lists): "
        f"false-positive rate at α = 0.05 {k / len(aa):.3f} [{lo:.3f}, {hi:.3f}].\n\n"
        f"**Coverage** ({len(cov)} re-randomisations and click draws, sessions fixed so the "
        f"exact SATE is fixed): single_model vs control, 95% interval "
        f"{cov['covered_95'].mean():.3f}, Bonferroni interval "
        f"{cov['covered_bonferroni'].mean():.3f}.\n\n"
        "**Offline and off-policy predictions vs the simulated A/B test** — per-session click "
        "gap of each treatment over control (test-window predictions use lists fitted at day "
        "98; the A/B uses lists fitted at day 120, so they differ slightly by construction):\n\n"
        + frame_table(a.prediction) + "\n"
    )


def robustness_section() -> str:
    sel = main_run().selection
    seeds = ac.fixed_config_study(sel, {f"seed {s}": m8.default_config(seed=s)
                                        for s in pr.STABILITY_SEEDS})
    grid = {f"corr {c:g}, weight {w:g}": m8.default_config(popularity_quality_correlation=c,
                                                          logging_popularity_weight=w)
            for c in m8.REGIME_CORRELATIONS for w in m8.REGIME_WEIGHTS}
    regimes = ac.fixed_config_study(sel, grid)
    agg = seeds.groupby("model")[["logged_ndcg", "oracle_ndcg", "online_clicks"]].agg(
        ["mean", "std"])
    agg.columns = [f"{a} {b}" for a, b in agg.columns]
    wide = seeds.pivot(index="scenario", columns="model", values="oracle_ndcg")
    ref = wide[pr.REFERENCE_MODEL]
    wins = {m_: int((wide[m_] > ref).sum()) for m_ in wide.columns if m_ != pr.REFERENCE_MODEL}
    tau = regimes.groupby("scenario")["tau_logged_vs_online"].first().reset_index()
    best = regimes.loc[regimes.groupby("scenario")["online_clicks"].idxmax(),
                       ["scenario", "model"]].rename(columns={"model": "best online"})
    bestl = regimes.loc[regimes.groupby("scenario")["logged_ndcg"].idxmax(),
                        ["scenario", "model"]].rename(columns={"model": "best logged"})
    tau = tau.merge(best, on="scenario").merge(bestl, on="scenario")
    return (
        f"Selected configurations (no re-tuning; blend weights fixed; no two-stage ranker) on "
        f"generator seeds {pr.STABILITY_SEEDS}, default logging:\n\n"
        + frame_table(agg.reset_index())
        + f"\n\nSeeds (of {len(wide)}) in which each model beats {pr.REFERENCE_MODEL} on oracle "
        "NDCG: " + ", ".join(f"{m_} {n}" for m_, n in wins.items())
        + ".\n\nThe M8 3 × 3 logging-regime grid (main seed's users, items and sessions): "
        "Kendall τ between logged NDCG and SIMULATED online clicks over the eight models, and "
        "the best model under each view:\n\n" + frame_table(tau) + "\n"
    )


def operational_section() -> str:
    rows = []
    for n_u, n_i in pr.SCALE_TIERS:
        x = opn.scale_click_matrix(n_u, n_i, seed=1)
        for name in ("global popularity", "implicit ALS", "BPR", "EASE", "item-kNN (dense M8)",
                     "PureSVD (dense M8)"):
            rows.append(opn.benchmark_model(name, x, budget=pr.MEMORY_BUDGET_BYTES,
                                            requests=pr.LATENCY_REQUESTS))
    bench = pd.DataFrame([r.__dict__ for r in rows])
    bench["planned_GB"] = bench["planned_bytes"] / 1e9
    bench["peak_traced_MB"] = bench["peak_traced_bytes"] / 1e6
    bench["model_MB"] = bench["model_bytes"] / 1e6
    n_u, n_i = pr.SCALE_TIERS[-1]
    f = fit_als(opn.scale_click_matrix(n_u, n_i, seed=1), factors=32, reg=1.0, alpha=20.0,
                iterations=pr.ALS_ITERATIONS)
    q = f.user[np.random.default_rng(0).integers(0, n_u, 200)]
    retr = opn.retrieval_benchmark(q, f.item, k=pr.K, clusters=pr.ANN_CLUSTERS,
                                   probes=pr.ANN_PROBES)
    tt = main_run().models["two-tower"]
    start = time.perf_counter()
    type(tt)(tt.dim, tt.hidden, epochs=tt.epochs, learning_rate=tt.learning_rate,
             seed=tt.seed).fit(main_run().prepared.history)
    tt_fit = time.perf_counter() - start
    skipped = bench.loc[bench["status"] != "RUN", ["model", "n_users", "n_items", "status"]]
    return (
        "**Indicative, machine-dependent timings** (not reproducible numbers); model size and "
        "planned allocations are deterministic. Scale data: seeded sparse click matrices "
        f"(about 20 items per user, Zipf popularity). Memory budget per workload "
        f"{pr.MEMORY_BUDGET_BYTES / 1e9:.1f} GB — workloads above it are skipped, not run.\n\n"
        + frame_table(bench, ["model", "n_users", "n_items", "status", "planned_GB",
                              "fit_seconds", "peak_traced_MB", "model_MB", "latency_p50_ms",
                              "latency_p95_ms", "throughput_users_per_s"])
        + f"\n\nSkipped by the preflight check: {len(skipped)} workload(s)"
        + ("" if skipped.empty else " — " + "; ".join(
            f"{r.model} at {r.n_users} × {r.n_items}" for r in skipped.itertuples()))
        + f".\n\n**Retrieval** at {n_u} × {n_i} (ALS item factors, 200 queries, top-{pr.K}): "
        "exact search vs a k-means inverted-file index.\n\n"
        + frame_table(pd.DataFrame([r.__dict__ for r in retr]))
        + f"\n\nTwo-tower (selected configuration) fit time at the main scale: {tt_fit:.1f} s.\n"
    )


def fresh_section() -> str:
    run = ac.full_run(m8.default_config(seed=pr.FRESH_SEED))
    a = ac.ab_study(run.prepared, run.selection, run.views)
    s = run.views.summary
    sel = run.selection
    return (
        f"The frozen protocol, unchanged, on seed {pr.FRESH_SEED} — evaluated once, here, "
        "after all choices were frozen. Selection on its validation window: best M9 "
        f"single-stage **{sel.best_single_m9}**, hybrid **{sel.hybrid}**.\n\n"
        + frame_table(s, ["model", "logged_ndcg", "oracle_ndcg", "online_clicks", "IPS",
                          "SNIPS", "DR"])
        + "\n\nRank agreement with simulated online clicks:\n\n" + frame_table(run.views.agreement)
        + "\n\nSIMULATED A/B on this seed ("
        + ", ".join(f"{k} = {v}" for k, v in a.policies.items()) + "):\n\n"
        + frame_table(a.table, ["comparison", "effect (clicks/user)", "interval (97.5%)", "p",
                                "Bonferroni p", "guardrails", "decision", "SATE clicks (exact)",
                                "covers SATE"])
        + f"\n\nA/A false-positive rate: {(a.aa.rows['p_value'] <= 0.05).mean():.3f}.\n"
    )


def main() -> None:
    sections: list[tuple[str, Callable[[], str]]] = [
        ("1. Frozen protocol and disclosed changes", protocol_section),
        ("2. Selection on the validation window", selection_section),
        ("3. Candidate recall (separate from ranking quality)", recall_section),
        ("4. Test benchmark", benchmark_section),
        ("5. Off-policy estimates and agreement between views", views_section),
        ("6. Naive vs propensity-weighted rankers", ranker_section),
        ("7. Cold-start users and items", cold_start_section),
        ("8. Off-policy calibration: SNIPS, DR and bootstrap intervals", ope_section),
        ("9. SIMULATED A/B test with the M6/M7 machinery", ab_section),
        ("10. Seeds and logging regimes", robustness_section),
        ("11. Operational benchmarks (indicative)", operational_section),
        ("12. Fresh-seed replication", fresh_section),
    ]
    dry_run = "--dry-run" in sys.argv
    output = Path(sys.argv[sys.argv.index("--dry-run") + 1]) if dry_run else OUTPUT
    rendered, runtimes = [], []
    for title, build in sections:
        if dry_run and title.startswith("12."):
            rendered.append(f"## {title}\n\nNot run (dry run: the fresh seed stays untouched).\n")
            continue
        start = time.perf_counter()
        rendered.append(f"## {title}\n\n{build()}\n")
        runtimes.append((title, time.perf_counter() - start))
        print(f"{title}: {runtimes[-1][1]:.0f} s", flush=True)
    environment = table(
        ["component", "version"],
        [["python", platform.python_version()], ["numpy", np.__version__],
         ["scipy", scipy.__version__], ["pandas", pd.__version__]])
    header = (
        "# Milestone 9 evidence — numerical results\n\n"
        "Generated by `python scripts/generate_m9_evidence.py`; do not edit by hand. Seeded; "
        "with the versions below every number reproduces exactly except §11 timings and the "
        "runtime table. Narrative and limitations: "
        "[m9-advanced-recommendation.md](m9-advanced-recommendation.md); methodology: "
        "[../methodology/advanced-recommendation.md](../methodology/advanced-recommendation.md)"
        ".\n\nViews are kept apart: **logged** (observable), **oracle** (synthetic ground "
        "truth, evaluation only) and **SIMULATED online** (the generator's click model — not "
        "an online experiment). Intervals are 95% unless stated.\n"
    )
    output.write_text(
        header + "\n" + "\n".join(rendered) + "\n## Environment\n\n" + environment
        + "\n\n## Runtime\n\n"
        + table(["section", "runtime (s, indicative)"], [[t, s] for t, s in runtimes]) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
