"""Generate the Milestone 8 (recommendation baselines) evidence report.

Writes ``docs/evidence/m8-recommendation-baselines-results.md``. All randomness is seeded;
with the recorded dependency versions every number reproduces exactly (the operational
timings and runtimes excepted). Prose numbers are computed, not typed.

Usage:
    python scripts/generate_m8_evidence.py
"""

from __future__ import annotations

import math
import platform
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from decision_intelligence.recommendations import cases as c

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "docs" / "evidence" / "m8-recommendation-baselines-results.md"
IPS_REPS = 1000
STABILITY_SEEDS = (1, 2, 3, 4, 5)

_CACHE: dict[str, object] = {}


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


def prepared(name: str) -> c.Prepared:
    if name not in _CACHE:
        _CACHE[name] = c.prepare(c.default_config() if name == "default" else c.hype_config())
    out = _CACHE[name]
    assert isinstance(out, c.Prepared)
    return out


def bench(name: str) -> c.Benchmark:
    key = f"bench-{name}"
    if key not in _CACHE:
        _CACHE[key] = c.benchmark(prepared(name))
    out = _CACHE[key]
    assert isinstance(out, c.Benchmark)
    return out


def audit_section() -> str:
    d, h = c.data_audit(prepared("default")), c.data_audit(prepared("hype"))
    keys = list(d.counts)
    rows = [[k, d.counts[k], h.counts[k]] for k in keys]
    pos = d.position
    return (
        "Two logging regimes share users, items, preferences and sessions (same seed); only "
        "the logging policy differs. *Default*: the logging popularity score is correlated "
        "0.7 with item quality (weight 1.5), fixed before any model was evaluated. *Hype*: "
        "popularity uncorrelated with quality (weight 2.5) — **selected after inspecting "
        "model results** in four exploratory regimes, as one showing a logged-vs-online "
        "reversal; it is an illustration, not a pre-specified test. §16 reports a grid of "
        "regimes fixed before its results were seen, and §17 a never-inspected seed.\n\n"
        + table(["quantity", "default", "hype"], rows)
        + "\n\nPosition effect (default logs up to the test cutoff). Slots are filled "
        "independently, so CTR ratios identify relative examination:\n\n"
        + frame_table(pos.round(6))
        + f"\n\nReading: the most-shown 10% of items take "
        f"{d.counts['top10pct_items_impression_share']:.1%} of impressions but "
        f"{d.counts['top10pct_items_attractiveness_share']:.1%} of mean true attractiveness "
        f"(default); under hype {h.counts['top10pct_items_impression_share']:.1%} vs "
        f"{h.counts['top10pct_items_attractiveness_share']:.1%}. The largest error in "
        f"estimated relative examination is "
        f"{np.max(np.abs(pos['relative_examination_estimate'] - pos['relative_truth'])):.4f}.\n"
    )


def leakage_section() -> str:
    s = c.leakage_study(prepared("default"))
    rv = s.random_vs_chronological
    assert (rv["inflation"] > 0).all(), "reading below assumes a positive inflation"
    sig = rv.loc[rv["p_value"] < 0.05, "model"].tolist()
    nonsig = rv.loc[rv["p_value"] >= 0.05, "model"].tolist()
    return (
        "Both protocols score **the same held-out clicks** (a random half of the test-window "
        f"clicks; {s.n_users} users). The invalid protocol adds the other half of the "
        "test-window clicks — interactions that happen after some held-out ones — to "
        "training. Inflation = mean per-user NDCG@10 difference (one-sample t, M3).\n\n"
        + frame_table(s.random_vs_chronological)
        + "\n\nPerturbation check: every impression, click and rating after the cutoff is "
        "rewritten at random and each model refitted; the largest change in any score:\n\n"
        + frame_table(s.perturbation)
        + f"\n\nReading: the invalid protocol raises mean NDCG@10 for every model; the "
        f"increase is significant at 5% for {', '.join(sig) or 'none'}"
        f"{' and not for ' + ', '.join(nonsig) if nonsig else ''} (the size depends on how "
        "much a model exploits co-occurrence with the leaked clicks). Models fitted under the "
        "chronological protocol are exactly invariant to post-cutoff data.\n"
    )


def tuning_section() -> str:
    b = bench("default")
    return (
        "Hyperparameters chosen by mean logged NDCG@10 on the validation window "
        "(fit ≤ day 84, evaluate days 84–98); the selected configuration is refitted on "
        "everything ≤ day 98 and evaluated once on days 98–120. Ground truth is not an input "
        "to this step.\n\n" + frame_table(b.tuning)
        + "\n\nDisclosure: before this grid was written, exploratory runs scored PureSVD "
        "(16 and 64 factors) and other fixed configurations on this seed's test window; the "
        "8-factor option was then added to the grid and is the one validation selected. The "
        "test window of this seed is therefore not a pristine holdout. §17 repeats the frozen "
        "protocol on a seed that was never inspected.\n"
    )


def benchmark_section() -> str:
    b = bench("default")
    lg = b.logged
    rows = [[r.model, r.n_users, f"{r.ndcg:.4f} {ci(r.ndcg_lo, r.ndcg_hi)}",
             f"{r.recall:.4f} {ci(r.recall_lo, r.recall_hi)}", r.precision, r.map, r.mrr,
             r.hit_rate] for r in lg.itertuples()]
    cmp_ = b.logged_comparisons
    crow = [[r.model, r.difference, ci(r.ci_lower, r.ci_upper), r.p_value, r.adjusted_p_value,
             r.significant] for r in cmp_.itertuples()]
    return (
        "Logged-feedback metrics at K = 10 on the test window, users with at least one "
        "relevant click (95% percentile bootstrap over users, 2,000 resamples; models held "
        "fixed).\n\n"
        + table(["model", "users", "NDCG@10 [95% CI]", "Recall@10 [95% CI]", "Precision@10",
                 "MAP@10", "MRR@10", "HitRate@10"], rows)
        + f"\n\nPaired comparisons with {c.REFERENCE} (per-user NDCG@10 differences, paired "
        "t; Holm across the six comparisons):\n\n"
        + table(["model", "difference", "95% CI", "p", "Holm p", "significant"], crow) + "\n"
    )


def oracle_section() -> str:
    b = bench("default")
    rows = [[r.model, r.n_users,
             f"{r.oracle_ndcg:.4f} {ci(r.oracle_ndcg_lo, r.oracle_ndcg_hi)}",
             r.oracle_attractiveness, r.oracle_precision] for r in b.oracle.itertuples()]
    cmp_ = b.oracle_comparisons
    crow = [[r.model, r.difference, ci(r.ci_lower, r.ci_upper), r.adjusted_p_value]
            for r in cmp_.itertuples()]
    return (
        "**Synthetic oracle** (evaluation only): graded NDCG@10 with gains = true "
        "attractiveness at the middle of the test window over the same allowed items, mean "
        "true attractiveness of the list, and the share of the list inside the user's true "
        "top 10. All active test users (clicks not required).\n\n"
        + table(["model", "users", "oracle NDCG@10 [95% CI]", "mean attractiveness",
                 "oracle precision@10"], rows)
        + f"\n\nPaired comparisons with {c.REFERENCE} (Holm):\n\n"
        + table(["model", "difference", "95% CI", "Holm p"], crow) + "\n"
    )


def _views_block(name: str) -> str:
    b = bench(name)
    v = b.views
    rows = [[r.model, r.logged_ndcg, r.oracle_ndcg, r.online_clicks, r.online_purchases,
             r.online_margin, f"{r.ips_clicks:.4f} {ci(r.ips_lo, r.ips_hi)}", r.ips_ess]
            for r in v.itertuples()]
    agree = b.rank_agreement
    return (
        table(["model", "logged NDCG", "oracle NDCG", "sim. online clicks/session",
               "sim. purchases/session", "sim. margin/session (currency units)",
               "IPS clicks [95% CI]", "IPS ESS"], rows)
        + "\n\nAgreement with the simulated online click ranking (Kendall τ over the seven "
        "models):\n\n" + frame_table(agree) + "\n"
    )


def views_section() -> str:
    return (
        "Three views of the same tuned models. *Logged*: observable. *Oracle*: ground truth. "
        "*Simulated online value*: the expected reward per test session had the list been "
        "shown in that session, under the generator's position-based click model (a "
        "simulation, not an online test). *IPS*: the estimate of the simulated online click "
        "value from the logs alone (examination estimated from the logs; no clipping).\n\n"
        "### Default logging\n\n" + _views_block("default")
        + "\n### Hype logging (strong exposure bias)\n\n" + _views_block("hype")
    )


def counterexample_section() -> str:
    p, b = prepared("hype"), bench("hype")
    rows = []
    for a, bb in (("content-based", "global popularity"), ("PureSVD", "item-kNN")):
        pc = c.contrast(p, b, a, bb)
        lg, orc_ = pc.logged, pc.oracle
        rows.append([f"{a} − {bb}",
                     f"{lg['difference']:.4f} {ci(lg['ci_lower'], lg['ci_upper'])}",
                     f"{orc_['difference']:.4f} {ci(orc_['ci_lower'], orc_['ci_upper'])}",
                     f"{pc.online[0]:.4f} {ci(pc.online[1], pc.online[2])}",
                     f"{pc.ips[0]:.4f} {ci(pc.ips[1], pc.ips[2])}"])
    v = b.views.set_index("model")
    best_logged = v["logged_ndcg"].idxmax()
    best_online = v["online_clicks"].idxmax()
    svd = c.contrast(p, b, "PureSVD", "item-kNN")
    content = c.contrast(p, b, "content-based", "global popularity")
    assert content.online[0] > 0 and svd.online[0] > 0, "reading assumes positive online gaps"
    assert content.ips[0] > 0 and svd.ips[0] > 0, "reading assumes IPS has the online sign"
    logged_rank = int(v["logged_ndcg"].rank(ascending=False)["content-based"])
    ips_note = (
        f"the content-based − popularity IPS interval "
        f"{'excludes' if content.ips[1] > 0 else 'includes'} 0 and the PureSVD − item-kNN "
        f"IPS interval {'excludes' if svd.ips[1] > 0 or svd.ips[2] < 0 else 'includes'} 0")
    return (
        "Offline (logged) improvement need not be online improvement. Under hype logging "
        "(selected after inspection; §16–17 test the pattern without that choice), paired "
        "differences (A − B) in each view; logged and oracle: per-user paired t; online and "
        "IPS: per-session differences with user-clustered SEs.\n\n"
        + table(["contrast", "logged NDCG@10", "oracle NDCG@10", "sim. online clicks/session",
                 "IPS clicks/session"], rows)
        + f"\n\nReading: the logged winner is {best_logged}; the simulated-online winner is "
        f"{best_online}. Logged metrics reward agreeing with what the logging policy showed: "
        "popular items collect clicks because they are shown, so a popularity-like model "
        f"looks good offline. Content-based lists, ranked {logged_rank} of {len(v)} by logged "
        "NDCG, beat global popularity on simulated online value. IPS estimates from the "
        f"same logs have the online sign in both contrasts but are noisy: {ips_note}. "
        "On this seed under default logging the logged, oracle and simulated online "
        "rankings agree (§6), but that is seed-specific: the content-based vs popularity "
        "reversal also appears under default logging for other seeds (§13, §17).\n"
    )


def beyond_section() -> str:
    b = bench("default")
    return (
        "Catalogue and exposure properties of the tuned models' top-10 lists (test users; "
        "exposure position-weighted 1/log2(k+1) over eligible items).\n\n"
        + frame_table(b.beyond) + "\n"
    )


def fairness_section() -> str:
    p, b = prepared("default"), bench("default")
    f = c.fairness_study(p, b, ("global popularity", "segment popularity", "item-kNN",
                                "PureSVD", "content-based"))
    ce = f.category_exposure.pivot(index="category", columns="model", values="ratio")
    merit = f.category_exposure.groupby("category")["merit_share"].first()
    ce.insert(0, "merit share", merit)
    ce = ce.reset_index()
    models_only = ce.drop(columns=["category", "merit share"])
    over = ", ".join(ce.loc[(models_only > 1).all(axis=1), "category"]) or "no category"
    het = f.heterogeneity
    seg, reg = het[het["attribute"] == "segment"], het[het["attribute"] == "region"]
    n_seg_sig = int((seg["holm_adjusted_p"] < 0.05).sum())
    n_reg_sig = int((reg["holm_adjusted_p"] < 0.05).sum())
    return (
        "**Item side** — exposure-to-merit ratio by category: position-weighted exposure "
        "share in the recommendations ÷ the category's share of the users' oracle top-10 "
        "lists (1 = proportional; merit is defined by synthetic ground truth).\n\n"
        + frame_table(ce)
        + "\n\n**User side** — oracle NDCG@10 by segment and region; Kruskal–Wallis (M3) "
        "across groups, Holm across the table. Segments differ in true taste by "
        "construction; region has no effect on preference in the generator, so region "
        "differences are noise.\n\n"
        + frame_table(f.heterogeneity)
        + f"\n\nReading: every model over-exposes {over} relative to its merit share. "
        f"After Holm, segment differences are significant for {n_seg_sig} of "
        f"{len(seg)} models and region differences for {n_reg_sig} of {len(reg)} (region "
        "has no effect in the generator). Segment gaps reflect different tastes and how "
        "well a segment's taste is served by the shared popularity signal; they are a "
        "prompt for review, not by themselves evidence of unfair treatment.\n\n"
        "Group means:\n\n" + frame_table(f.groups) + "\n"
    )


def cold_start_section() -> str:
    s = c.cold_start_study(prepared("default"))
    return (
        f"**Cold-start users** ({s.n_cold_users} active test users with no click before the "
        "cutoff). Oracle NDCG@10; a collaborative model without a fallback gives such users "
        "constant scores, so its ranking degenerates to item-id order.\n\n"
        + frame_table(s.users)
        + f"\n\n**Cold-start items** ({s.n_new_items} items launched during the test window; "
        "warm users; ranking only those items; oracle NDCG@5 at the end of the window). "
        "Collaborative and popularity models have no data on them.\n\n"
        + frame_table(s.items)
        + "\n\nReading: compare models within a column, not warm with cold: seen-item "
        "exclusion removes warm users' already-clicked items (often their most attractive) "
        "from the allowed set, while cold users keep them, so the two columns rank different "
        "item sets. Without a fallback, collaborative and content models give cold users an "
        "item-id ordering; the segment-popularity fallback restores segment-level quality. "
        "For new items only content features carry information; collaborative and "
        "popularity scores are constant there.\n"
    )


def off_policy_section() -> str:
    rep = c.ips_repeated_sampling(c.default_config(n_users=600, n_items=150), IPS_REPS)
    p, b = prepared("default"), bench("default")
    sweep = c.clipping_sweep(p, b, "item-kNN")
    sens = c.examination_sensitivity(p, b, "item-kNN")
    pos = c.positivity_study()
    rows = rep.rows
    max_true = float((rows["true_examination_bias"].abs()
                      / rows["true_examination_mc_se"]).max())
    max_est = float((rows["bias"].abs() / rows["mc_se"]).max())
    ignored_z = (rows["position_ignored_bias"] / rows["position_ignored_mc_se"]).tolist()
    exam = c.examination_uncertainty(p, b, ("global popularity", "item-kNN"))
    cov_text = "; ".join(
        f"{r.policy}: {r.coverage:.3f} [{r.coverage_wilson_lo:.3f}, {r.coverage_wilson_hi:.3f}], "
        f"exact binomial p = {r.coverage_binomial_p:.3g}, misses {r.misses_estimate_too_low} "
        f"with the estimate too low vs {r.misses_estimate_too_high} too high, mean SE "
        f"{r.mean_se:.4f} vs SD {r.sd_of_estimates:.4f}, skewness {r.skewness:.2f}, "
        f"corr(estimate, SE) {r.corr_estimate_se:.2f}, i.i.d.-session-SE coverage "
        f"{r.coverage_iid_session_se:.3f}, true-examination coverage "
        f"{r.coverage_true_examination:.3f}" for r in rows.itertuples())
    below = rows.loc[rows["coverage_binomial_p"] < 0.05, "policy"].tolist()
    coverage_frame = rows[["policy", "exact_value", "ips_mean", "mc_se", "bias", "coverage",
                           "coverage_wilson_lo", "coverage_wilson_hi", "coverage_binomial_p",
                           "misses_estimate_too_low", "misses_estimate_too_high",
                           "sd_of_estimates", "mean_se", "skewness", "corr_estimate_se",
                           "coverage_iid_session_se", "coverage_true_examination"]]
    other_frame = rows[["policy", "true_examination_bias", "true_examination_mc_se",
                        "clip", "clipped_mean", "clipped_bias", "position_ignored_bias",
                        "position_ignored_mc_se"]]
    return (
        f"**Repeated sampling** ({rep.n_replications} re-draws of the test-window logging; "
        "600 users × 150 items; users, items, preferences, sessions and the target lists "
        "fixed, so the estimand is fixed; the examination estimate from the fixed training "
        "logs is fixed too). Monte Carlo SE of a coverage near 0.95: "
        f"{math.sqrt(0.95 * 0.05 / rep.n_replications):.4f}.\n\n"
        + frame_table(coverage_frame) + "\n\n" + frame_table(other_frame)
        + f"\n\nCoverage findings — {cov_text}.\n\n"
        "Reading: the standard error is calibrated (its mean matches the spread of the "
        "estimates, and clustering by user changes it little because users rarely have many "
        "test sessions). Coverage is "
        + (f"significantly below 95% for {', '.join(below)}" if below else
           "not significantly different from 95% for either policy")
        + ". The misses are almost all on one side — the estimate falls low while its SE "
        "falls with it (positive skew, estimate–SE correlation) — the known weakness of a "
        "normal interval for an importance-weighted mean, not an error in the variance "
        f"formula. Bias with the true examination: at most {max_true:.1f} MC SE; with the "
        f"estimated examination: at most {max_est:.1f} MC SE. Clipping at 50 biases the "
        "estimate downwards heavily. Ignoring position bias gives a bias of "
        + ", ".join(f"{r.position_ignored_bias / r.exact_value:+.1%} of the value "
                    f"({z:+.1f} MC SE) for {r.policy}"
                    for r, z in zip(rows.itertuples(), ignored_z, strict=True))
        + ": small for a list whose items are similarly attractive, a larger underestimate "
        "for a list that puts its most attractive items first.\n\n"
        "**Uncertainty from estimating the examination** (ignored by the IPS interval): "
        "session-level bootstrap of the training impressions (200 resamples), test logs "
        "fixed, default scenario, tuned models.\n\n" + frame_table(exam)
        + "\n\nReading: estimating the examination adds at most "
        f"{exam['share_of_total_variance'].max():.1%} to the variance of the IPS estimate, "
        "so treating it as known has a negligible effect on interval width.\n\n"
        "**Clipping sweep** (item-kNN, default logs; error vs the exact simulated value):\n\n"
        + frame_table(sweep)
        + "\n\n**Examination assumption** (single realisation; the repeated-sampling table "
        "above is the reliable comparison):\n\n" + frame_table(sens)
        + f"\n\n**Positivity**: logging restricted to the {pos.support_size} items with the "
        "highest logging popularity. IPS can only see that support:\n\n"
        + frame_table(pos.rows)
        + "\n\nReading: when a list puts examination mass on items the logging policy never "
        "shows, IPS estimates only the supported part of the value (its interval covers "
        "the exact supported value, not the full value) — a downward bias no amount of data "
        "removes.\n"
    )


def explicit_section() -> str:
    t = c.explicit_ratings_case(prepared("default"))
    return (
        "Regularised bias model (λ_user = λ_item = 5) fitted on ratings ≤ the cutoff. "
        "Ratings are explicit feedback and missing not at random (users rate liked items "
        "more often).\n\n" + frame_table(t)
        + "\n\nReading: held-out observed ratings understate the error a rating model makes "
        "on the pairs a recommender must rank; the model also over-predicts on random pairs "
        "because it learnt from a sample skewed towards liked items.\n"
    )


def stability_section() -> str:
    s = c.seed_stability(STABILITY_SEEDS)
    agg = s.groupby("model")[["logged_ndcg", "oracle_ndcg", "online_clicks"]].agg(
        ["mean", "std", "min", "max"])
    agg.columns = [f"{a} {b}" for a, b in agg.columns]
    agg = agg.reset_index()
    wide = s.pivot(index="seed", columns="model")
    gap_logged = wide["logged_ndcg"]["content-based"] - wide["logged_ndcg"]["global popularity"]
    gap_online = (wide["online_clicks"]["content-based"]
                  - wide["online_clicks"]["global popularity"])
    main = bench("default").views.set_index("model")["online_clicks"]
    exception = ("exception" if main["content-based"] < main["global popularity"]
                 else "not an exception")
    knn_wins = int((wide["oracle_ndcg"]["item-kNN (k=50)"]
                    > wide["oracle_ndcg"]["global popularity"]).sum())
    return (
        f"Fixed configurations (no re-tuning), default logging, generator seeds "
        f"{', '.join(map(str, STABILITY_SEEDS))} — variation from the data-generating draw, "
        "which the user-level bootstrap intervals do not include.\n\n" + frame_table(agg)
        + f"\n\nReading: item-kNN beats global popularity on oracle NDCG in {knn_wins} of "
        f"{len(wide)} seeds. Content-based minus global popularity is negative on logged NDCG "
        f"in {int((gap_logged < 0).sum())} of {len(wide)} seeds "
        f"({', '.join(f'{x:+.3f}' for x in gap_logged)}) but positive on simulated online "
        f"clicks in {int((gap_online > 0).sum())} of {len(wide)} "
        f"({', '.join(f'{x:+.3f}' for x in gap_online)}): the logged-vs-online reversal is "
        "common under default logging too; the main seed, where content-based is also "
        "below popularity on simulated online value, is the "
        f"{exception}.\n"
    )


def m2_section() -> str:
    t = c.m2_compatibility_case()
    return (
        "The M2 recommendation tables (read-only; 25 products; category-level latent "
        "affinities; no logged propensities, so no IPS) mapped to the M8 log schema. "
        "Chronological 60/20/20 split; K = 5.\n\n" + frame_table(t)
        + "\n\nReading: with 25 items, hit rates are high even for a random ranking, and "
        "with a category-level preference every item in a category has the same oracle gain, "
        "so the oracle cannot distinguish item-level models — why M8 uses its own item-level "
        "scenario.\n"
    )


def operational_section() -> str:
    b = bench("default")
    rows = [[f, b.fit_seconds[f], b.score_seconds[f] * 1000 / prepared("default").test.users.size
             * 1000] for f in b.models]
    return (
        "Indicative only (single run, this machine; not reproducible numbers): refit time "
        "on train + validation and scoring time per 1,000 users for the full eligible "
        "catalogue.\n\n" + table(["model", "fit (s)", "score (ms / 1,000 users)"], rows) + "\n"
    )


def regime_section() -> str:
    sw = c.logging_regime_sweep()
    t = sw.summary
    n_rev = int(t["logged_sign_reversed"].sum())
    ips_wrong = int((np.sign(t["content_minus_pop_ips"])
                     != np.sign(t["content_minus_pop_online"])).sum())
    tau_by_corr = ", ".join(f"{k:g}: {v:.2f}" for k, v in
                            t.groupby("correlation")["tau_logged_vs_online"].mean()
                            .sort_index(ascending=False).items())
    return (
        "A grid of logging regimes **fixed during the assurance review before any of its "
        f"results were seen**: popularity–quality correlation {c.REGIME_CORRELATIONS} × "
        f"logging popularity weight {c.REGIME_WEIGHTS}; same users, items, preferences and "
        "sessions; the six fixed configurations of §13 (no tuning). Kendall τ is with the "
        "simulated online click ranking; the last columns are content-based minus global "
        "popularity.\n\n" + frame_table(t)
        + f"\n\nReading: logged NDCG reverses the sign of the content-based vs popularity "
        f"comparison in {n_rev} of {len(t)} regimes, and its rank agreement with simulated "
        f"online value ranges from τ = {t['tau_logged_vs_online'].min():.2f} to "
        f"{t['tau_logged_vs_online'].max():.2f}; it tends to fall as popularity becomes less "
        f"related to quality (mean τ by correlation: {tau_by_corr}), though not "
        "monotonically across every cell. IPS ranks the models in close agreement with the "
        f"online ranking (τ ≥ {t['tau_ips_vs_online'].min():.2f}), but as single "
        "realisations its content-based vs popularity gap has the wrong sign in "
        f"{ips_wrong} of {len(t)} regimes and can be far from the online gap. The hype "
        "regime of §6–7 is one cell of this grid, not an isolated case.\n"
    )


def fresh_seed_section() -> str:
    fr = c.fresh_seed_replication()
    blocks = []
    for regime, v in fr.views.items():
        blocks.append(f"### {regime.capitalize()} logging\n\n"
                      + frame_table(v, ["model", "logged_ndcg", "oracle_ndcg",
                                        "online_clicks", "ips_clicks", "ips_lo", "ips_hi"]))
    rows = [[r.regime, f"{r.logged:.4f} {ci(r.logged_lo, r.logged_hi)}",
             f"{r.online:.4f} {ci(r.online_lo, r.online_hi)}",
             f"{r.ips:.4f} {ci(r.ips_lo, r.ips_hi)}"] for r in fr.contrasts.itertuples()]
    d = fr.views["default"].set_index("model")
    k = fr.contrasts
    reversed_in = k.loc[(k.logged_hi < 0) & (k.online_lo > 0), "regime"].tolist()
    ips_sign = bool((np.sign(k.ips) == np.sign(k.online)).all())
    cf_beats = all(d.loc[m_, col] > d.loc[c.REFERENCE, col]
                   for m_ in ("item-kNN", "user-kNN", "PureSVD")
                   for col in ("logged_ndcg", "oracle_ndcg", "online_clicks"))
    return (
        f"The frozen protocol (grid, tuning on validation, K = 10, splits, both regimes) on "
        f"generator seed {fr.seed}, fixed during the assurance review and never inspected "
        "before this run.\n\n" + "\n\n".join(blocks)
        + "\n\nContent-based minus global popularity:\n\n"
        + table(["regime", "logged NDCG@10", "sim. online clicks/session",
                 "IPS clicks/session"], rows)
        + "\n\nReading: collaborative baselines (item-kNN, user-kNN, PureSVD) "
        + ("beat" if cf_beats else "do not all beat")
        + " global popularity in the logged, oracle and simulated online views under default "
        "logging. The logged-vs-online reversal for content-based vs popularity (logged "
        "interval below 0, online interval above 0) "
        + (f"replicates under {' and '.join(reversed_in)} logging" if reversed_in
           else "does not replicate")
        + " on this seed. IPS "
        + ("has the online sign in both regimes" if ips_sign else "does not always have the "
           "online sign")
        + "; its intervals "
        + ("exclude" if (k.ips_lo > 0).all() else "do not exclude")
        + " 0.\n"
    )


def main() -> None:
    sections: list[tuple[str, Callable[[], str]]] = [
        ("1. Data audit: exposure and position bias in the logs", audit_section),
        ("2. Leakage: chronological vs random holdout, and invariance", leakage_section),
        ("3. Tuning on the validation window", tuning_section),
        ("4. Test benchmark: logged feedback", benchmark_section),
        ("5. Test benchmark: synthetic oracle", oracle_section),
        ("6. Logged, oracle, simulated online and IPS views", views_section),
        ("7. Counterexample: offline winner ≠ online winner", counterexample_section),
        ("8. Coverage, novelty, diversity and exposure concentration", beyond_section),
        ("9. Exposure fairness and segment-level performance", fairness_section),
        ("10. Cold-start users and items", cold_start_section),
        ("11. Off-policy (IPS) diagnostics", off_policy_section),
        ("12. Explicit ratings missing not at random", explicit_section),
        ("13. Stability across generator seeds", stability_section),
        ("14. M2 compatibility", m2_section),
        ("15. Operational cost (indicative)", operational_section),
        ("16. Logging-regime grid (pre-specified in the assurance review)", regime_section),
        ("17. Fresh-seed replication of the frozen protocol", fresh_seed_section),
    ]
    rendered, runtimes = [], []
    for title, build in sections:
        start = time.perf_counter()
        rendered.append(f"## {title}\n\n{build()}\n")
        runtimes.append((title, time.perf_counter() - start))
    environment = table(
        ["component", "version"],
        [["python", platform.python_version()], ["numpy", np.__version__],
         ["scipy", scipy.__version__], ["pandas", pd.__version__]])
    header = (
        "# Milestone 8 evidence — numerical results\n\n"
        "Generated by `python scripts/generate_m8_evidence.py`; do not edit by hand. All "
        "randomness is seeded; with the versions below every number reproduces exactly "
        "(§15 timings and the runtime table excepted). Narrative and limitations: "
        "[m8-recommendation-baselines.md](m8-recommendation-baselines.md); methodology: "
        "[../methodology/recommender-baselines.md](../methodology/recommender-baselines.md)."
        "\n\nThree views are kept apart throughout: **logged feedback** (observable), the "
        "**synthetic oracle** (ground truth, read only after estimation) and **simulated "
        "online value** (the generator's click model — not an online experiment). Ground "
        "truth establishes nothing about real-world validity. Intervals are 95% confidence "
        "intervals.\n"
    )
    OUTPUT.write_text(
        header + "\n" + "\n".join(rendered) + "\n## Environment\n\n" + environment
        + "\n\n## Runtime\n\n"
        + table(["section", "runtime (s, indicative)"], [[t, s] for t, s in runtimes]) + "\n")
    print(f"Wrote {OUTPUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
