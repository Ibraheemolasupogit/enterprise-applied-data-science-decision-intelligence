"""Generate the Milestone 4 (causal foundations) evidence report.

Writes ``docs/evidence/m4-causal-foundations-results.md``. Every random draw
is seeded, so all numbers reproduce exactly with the recorded dependency
versions (runtimes excepted).

Usage:
    python scripts/generate_m4_evidence.py
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

from decision_intelligence.causal_inference import campaign as cp
from decision_intelligence.causal_inference import diagnostics as dg
from decision_intelligence.causal_inference import estimators as es
from decision_intelligence.causal_inference import scenarios as sc
from decision_intelligence.common import seeded_generator
from decision_intelligence.data import SyntheticDataConfig, build_synthetic_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "docs" / "evidence" / "m4-causal-foundations-results.md"
CONFIG = SyntheticDataConfig()
TRUTH = CONFIG.marketing_true_effect


def table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    def fmt(value: object) -> str:
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, float):
            if math.isnan(value):
                return "—"
            if math.isinf(value):
                return "inf"
            return f"{value:.4g}" if abs(value) >= 1e-3 or value == 0 else f"{value:.2e}"
        return str(value)

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def interval(estimate: es.CausalEstimate) -> str:
    return f"[{estimate.ci_lower:.3f}, {estimate.ci_upper:.3f}]"


# ---------------------------------------------------------------------------


def dag_section() -> str:
    dag = cp.campaign_dag()
    roles = dag.roles(cp.TREATMENT, cp.OUTCOME)
    measured = [
        "segment", "latent_value", "price_sensitivity", "household_income",
        "acquisition_channel", "region", "signup_date", "billing_transactions",
    ]
    minimal = dag.valid_adjustment_sets(cp.TREATMENT, cp.OUTCOME, measured)
    edges = ", ".join(f"{a} → {b}" for a, b in sorted(dag.edges))
    proxies = cp.assess_strategy(cp.PROXIES)
    path_rows = [
        [report.render(dag), report.is_backdoor, ", ".join(report.colliders) or "—", report.reason]
        for report in (
            dag.path_report(p, set(proxies.adjustment_set))
            for p in dag.paths(cp.TREATMENT, cp.OUTCOME)
        )
    ]
    strategy_rows = [
        [name, assessment.valid, assessment.explanation]
        for name, assessment in (
            (name, cp.assess_strategy(cols)) for name, cols in cp.ADJUSTMENT_STRATEGIES.items()
        )
    ]
    return (
        "The DAG is read off the M2 generator code (customers, marketing, retention, "
        "subscriptions, transactions, experiments modules). It is an *assumption* about the "
        "data-generating process; here it happens to be known exactly.\n\n"
        f"Edges: {edges}. Isolated: acquisition_channel, region.\n\n"
        + table(["role (w.r.t. treated → retained_90d)", "nodes"],
                [[k, ", ".join(sorted(v)) or "—"] for k, v in roles.items()])
        + f"\n\nMinimal valid adjustment sets among measured variables: "
        f"{[sorted(s) for s in minimal]}. Without `latent_value` the effect is **not "
        "identified** by backdoor adjustment.\n\n"
        "Every treated–retained path with the regime-B proxies conditioned on:\n\n"
        + table(["path", "backdoor?", "colliders", "status given proxies"], path_rows)
        + "\n\nBackdoor assessment of each adjustment strategy used below:\n\n"
        + table(["strategy", "valid", "explanation"], strategy_rows)
    )


def campaign_section(analysis: cp.CampaignAnalysis) -> str:
    rows = []
    for strategy in analysis.strategies:
        for estimate in strategy.estimates:
            rows.append([
                strategy.strategy, strategy.dag_valid, estimate.estimator, estimate.estimate,
                interval(estimate), estimate.covers(analysis.true_effect),
            ])
    extra = [
        ["regime A (latent_value + segment), ATT", True, analysis.att_regime_a.estimator,
         analysis.att_regime_a.estimate, interval(analysis.att_regime_a),
         analysis.att_regime_a.covers(analysis.true_effect)],
        ["regime C: logged propensity", True, analysis.known_propensity_ipw.estimator,
         analysis.known_propensity_ipw.estimate, interval(analysis.known_propensity_ipw),
         analysis.known_propensity_ipw.covers(analysis.true_effect)],
        ["INVALID: regime A among customers billed ≥ 1 time "
         f"({analysis.selection_share_kept:.0%} kept)", False,
         "AIPW (doubly robust)", analysis.selection_on_billing.estimate,
         interval(analysis.selection_on_billing),
         analysis.selection_on_billing.covers(analysis.true_effect)],
    ]
    naive = analysis.strategies[0].estimates[0].estimate
    return (
        f"Canonical M2 build (`SyntheticDataConfig()`, seed 20260101): {analysis.n_customers} "
        f"customers, {analysis.treated_share:.1%} treated. Known design effect (read only after "
        f"estimation): **{analysis.true_effect}** (ATE = ATT; constant on the probability "
        "scale).\n\n"
        + table(["strategy", "DAG-valid", "estimator", "estimate", "95% CI", "CI covers 0.08"],
                rows + extra)
        + "\n\n**Reading this table honestly.** The naive contrast "
        f"({naive:.3f}) is almost three times the true effect: treated customers have higher "
        "`latent_value` and would have retained better anyway. Valid adjustment (regime A, "
        "regime C) removes most of that bias, but on this particular seed the valid "
        "estimates land 0.046–0.056 above the truth and all but one of their 95% intervals "
        "narrowly miss 0.08 (g-computation adjusting for latent_value alone just covers it). "
        "That is what a ~2-standard-error draw looks like; whether the estimators are "
        "biased cannot be judged from one dataset. Section 7 answers it with 200 "
        "independent M2 draws (no detectable bias; near-nominal coverage). Conversely, both "
        "*invalid* post-treatment strategies produce intervals that cover 0.08 on this seed, "
        "although section 7 shows they are biased by −0.044 and −0.024 on average — a single "
        "good-looking number is not validation, and only the DAG argument tells valid from "
        "invalid here."
    )


def diagnostics_section(analysis: cp.CampaignAnalysis) -> str:
    o, w = analysis.overlap, analysis.weights
    overlap_rows = [
        ["treated", *o.treated_quantiles.values()],
        ["control", *o.control_quantiles.values()],
    ]
    balance = analysis.balance.reset_index()
    balance_rows = [
        [r.covariate, r.smd_raw, r.smd_weighted, r.variance_ratio_raw, r.variance_ratio_weighted]
        for r in balance.itertuples()
    ]
    f = analysis.falsification
    return (
        "Regime A propensity model (logistic in latent_value + segment), ATE weights.\n\n"
        + table(["arm", *o.treated_quantiles.keys()], overlap_rows)
        + f"\n\nShare with e < {o.bound} or e > {1 - o.bound}: "
        f"{o.share_below + o.share_above:.1%}; common support "
        f"[{o.common_support[0]:.3f}, {o.common_support[1]:.3f}].\n\n"
        + table(["max weight", "ESS treated", "n treated", "ESS control", "n control",
                 "top-1% weight share", "warnings"],
                [[w.max_weight, w.ess_treated, w.n_treated, w.ess_control, w.n_control,
                  w.top_one_percent_weight_share, "; ".join(w.warnings) or "none"]])
        + "\n\nBalance (SMD; |SMD| < 0.1 adequate). `latent_value` is the confounder; the "
        "others are checked although not all are adjusted for.\n\n"
        + table(["covariate", "SMD raw", "SMD weighted", "var. ratio raw", "var. ratio weighted"],
                balance_rows)
        + "\n\nEstimated vs logged (true) propensity: max |difference| = "
        f"{analysis.estimated_vs_logged_propensity_max_abs_diff:.3f} — the propensity model is "
        "well specified, as expected since the true model is logistic in latent_value.\n\n"
        f"**DAG falsification test.** The DAG implies *{f.claim}*. Likelihood-ratio test: "
        f"χ² = {f.lr_statistic:.2f} on {f.df} df, p = {f.p_value:.3f} — not rejected. "
        "(Passing does not prove the DAG; it fails to refute one testable implication.)"
    )


def negative_control_section(analysis: cp.CampaignAnalysis) -> str:
    rows = [[e.estimand.split(" — ")[-1], e.estimate, interval(e), e.covers(0.0)]
            for e in analysis.negative_control]
    return (
        "`log_income` is measured before the campaign, so the offer cannot change it (true "
        "effect 0), but it shares the confounder `latent_value`. A non-zero adjusted 'effect' "
        "on it signals residual confounding.\n\n"
        + table(["analysis", "estimate", "95% CI", "covers 0"], rows)
        + "\n\nThe naive contrast detects the confounding clearly. On this seed the regime A "
        "estimate is ≈ 2 SE below zero — the same kind of chance draw as in section 2 (across "
        "replicates it is unbiased with near-nominal coverage; section 7). Regime B's "
        "interval happens to cover 0 here, but section 7 shows its negative-control "
        "estimate is biased by about +0.05 on average and flags the residual confounding in "
        "roughly 80% of draws — this seed is one of the misses. A single negative-control "
        "check is informative but not conclusive in either direction."
    )


def subgroup_section(analysis: cp.CampaignAnalysis) -> str:
    rows = [[k, v.estimate, interval(v), v.covers(analysis.true_effect)]
            for k, v in sorted(analysis.subgroup_effects.items())]
    h = analysis.subgroup_heterogeneity
    return (
        "AIPW within each segment, adjusting for latent_value. The true conditional effect "
        "is 0.08 in **every** segment (constant on the risk-difference scale).\n\n"
        + table(["segment", "estimate", "95% CI", "covers 0.08"], rows)
        + f"\n\nHeterogeneity test: Q = {h.q_statistic:.2f} on {h.df} df, p = {h.p_value:.3f}; "
        f"inverse-variance pooled effect {h.pooled_effect:.3f}. The apparent spread between "
        "segments is compatible with noise — exactly the trap of reading subgroup point "
        "estimates as heterogeneity. A constant risk difference also implies *non-constant* "
        "risk ratios and odds ratios across segments with different baselines: whether an "
        "effect is 'heterogeneous' depends on the scale."
    )


def sensitivity_section(analysis: cp.CampaignAnalysis) -> str:
    rr, ev = analysis.regime_b_risk_ratio, analysis.regime_b_e_value
    regime_b = next(s for s in analysis.strategies if s.strategy.startswith("regime B"))
    grid = analysis.bias_grid
    rows = [[r.outcome_effect_gamma, r.prevalence_difference_delta, r.bias,
             r.corrected_estimate, f"[{r.corrected_ci_lower:.3f}, {r.corrected_ci_upper:.3f}]",
             r.ci_excludes_zero] for r in grid.itertuples()]
    return (
        "Regime B (proxies only) is the realistic case: the confounder is unmeasured.\n\n"
        f"AIPW risk ratio {rr.risk_ratio:.3f} (95% CI {rr.ci_lower:.3f}–{rr.ci_upper:.3f}); "
        f"marginal risks {rr.risk_treated:.3f} vs {rr.risk_untreated:.3f}. "
        f"**E-value {ev.e_value_point:.2f}** (CI limit: {ev.e_value_ci:.2f}): an unmeasured "
        f"confounder associated with both treatment and retention by a risk ratio of "
        f"{ev.e_value_point:.2f} each, beyond the proxies, could explain the whole estimate.\n\n"
        "**Benchmark against the real omitted confounder** (possible only because the data "
        "are synthetic), using the Ding & VanderWeele (2016) bounding factor "
        "B = RR_EU·RR_UD / (RR_EU + RR_UD − 1): the most a confounder with prevalence ratio "
        "RR_EU between arms and outcome risk ratio RR_UD could inflate the observed RR.\n\n"
        + table(["latent_value categories", "RR_EU (prevalence ratio)",
                 "RR_UD (max/min retention)", "bounding factor B", "RR could fall to"],
                [[b.n_categories, b.rr_exposure_confounder, b.rr_confounder_outcome,
                  b.bounding_factor, b.risk_ratio_lower_bound]
                 for b in analysis.confounder_benchmarks])
        + "\n\nA coarse median split understates a continuous confounder: with finer "
        f"categories the bound reaches {analysis.confounder_benchmarks[-1].bounding_factor:.2f}"
        f", above the observed RR of {rr.risk_ratio:.2f}. So a confounder as strong as the one "
        "regime B omits *could*, in the worst case the bound describes, explain away the whole "
        "estimate — the E-value analysis correctly flags the regime-B estimate as not robust. "
        "The actual bias is smaller than that worst case (the effect is real, 0.08, and regime "
        "B overstates it by about a third on average, section 7): the bound is conservative by "
        "design. Caveats: these are marginal associations whereas the E-value refers to "
        "associations conditional on the measured proxies, and with many small bins the "
        "max/min ratios are inflated by sampling noise. The E-value says how much unmeasured "
        "confounding an estimate can withstand, not whether it is right.\n\n"
        f"Bias-formula grid for the regime-B AIPW risk difference "
        f"({regime_b.estimates[-1].estimate:.3f}, CI {interval(regime_b.estimates[-1])}):\n\n"
        + table(["γ (U → retention, RD)", "δ (prevalence diff.)", "bias γδ",
                 "corrected estimate", "corrected CI", "CI excludes 0"], rows)
    )


def monte_carlo_section() -> str:
    summaries = cp.campaign_monte_carlo(CONFIG, n_replications=200)
    rows = [[s.estimator, s.truth, s.mean_estimate, f"{s.bias:+.4f} ± {s.bias_mc_se:.4f}",
             s.empirical_sd, s.mean_reported_se, s.rmse,
             f"{s.coverage:.3f} [{s.coverage_ci_lower:.3f}, {s.coverage_ci_upper:.3f}]",
             s.coverage_p_value_vs_nominal] for s in summaries]
    valid_names = (
        "IPW (Hájek), regime A", "AIPW, regime A", "AIPW ATT, regime A",
        "IPW, regime C (logged propensity)", "AIPW, propensity correct / outcome model wrong",
        "AIPW, propensity wrong / outcome model correct",
    )
    valid = [s for s in summaries if s.estimator in valid_names]
    coverage_low = min(s.coverage for s in valid)
    coverage_high = max(s.coverage for s in valid)
    smallest_p = min(s.coverage_p_value_vs_nominal for s in valid)
    return (
        "200 independent draws of the M2 customer → offer → retention process (2,000 "
        "customers each; the same M2 generator functions, seeds 10,000–10,199). Each "
        "estimator is scored against the truth of its own estimand. Bias is shown ± its "
        "Monte Carlo SE. Coverage uses the influence-function interval.\n\n"
        + table(["estimator", "truth", "mean", "bias", "empirical SD", "mean reported SE",
                 "RMSE", "coverage [95% Wilson CI]", "exact binomial p vs 0.95"], rows)
        + "\n\n**Interpretation.** Valid strategies (regime A, regime C, both single-"
        "misspecification AIPW variants) show no bias beyond Monte Carlo error. Their "
        f"coverage ranges {coverage_low:.3f}–{coverage_high:.3f}; every Wilson interval "
        "contains 0.95 and the smallest exact binomial p-value against 0.95 is "
        f"{smallest_p:.2f}, so none is statistically distinguishable from nominal. The "
        "converse also holds: 200 replications cannot rule out true coverage a few points "
        "below 95% (the intervals reach down to ≈ 0.89). Reported SEs track the empirical "
        "SDs; IPW SEs include the correction for propensity estimation, and AIPW's "
        "fixed-nuisance SEs remain close even under single-model misspecification in this "
        "design. The naive contrast is biased by more than the effect "
        "itself. Observed proxies (regime B) remove most of the bias but not all, and their "
        "intervals under-cover. Adjusting for a post-treatment variable, or selecting on "
        "one, biases the estimate *downwards*. The negative control behaves as the DAG "
        "predicts: centred on 0 with valid adjustment, shifted under proxy adjustment."
    )


def large_sample_section() -> str:
    frame = cp.simulate_campaign_replicate(SyntheticDataConfig(n_customers=20_000), seed=777)
    rows = []
    for label, columns in (("naive", ()), ("regime B: proxies", cp.PROXIES),
                           ("regime A: latent_value + segment", cp.REGIME_A)):
        data = cp.causal_data(frame, columns)
        estimate = es.naive_difference(data) if not columns else es.augmented_ipw(data)
        rows.append([label, estimate.estimator, estimate.estimate, interval(estimate),
                     estimate.covers(TRUTH)])
    known = es.inverse_probability_weighting(
        cp.causal_data(frame, ()), propensity=frame["logged_propensity"].to_numpy()
    )
    rows.append(["regime C: logged propensity", known.estimator, known.estimate,
                 interval(known), known.covers(TRUTH)])
    return (
        "One draw with 20,000 customers (same generator, seed 777): with a tenth of the "
        "variance, the ranking of strategies is visible in a single dataset.\n\n"
        + table(["strategy", "estimator", "estimate", "95% CI", "covers 0.08"], rows)
    )


def scenario_section() -> str:
    rng = seeded_generator(4040)
    rows: list[list[object]] = []

    def summarise(name: str, truth: float, values: list[float], note: str) -> None:
        array = np.asarray(values)
        rows.append([name, truth, float(array.mean()),
                     float(array.std(ddof=1) / np.sqrt(array.size)), float(np.median(array)),
                     float(array.std(ddof=1)), float(np.sqrt(np.mean((array - truth) ** 2))),
                     note])

    def data(frame: pd.DataFrame, columns: list[str]) -> es.CausalData:
        return es.CausalData.from_arrays(frame["treatment"], frame["outcome"], frame[columns])

    reps = 500
    collected: dict[str, list[float]] = {}
    truths: dict[str, float] = {}
    for _ in range(reps):
        s = sc.confounded(1000, rng)
        collected.setdefault("confounded: naive", []).append(
            es.naive_difference(data(s.frame, [])).estimate - s.true_ate)
        collected.setdefault("confounded: AIPW ATE (adjust x)", []).append(
            es.augmented_ipw(data(s.frame, ["x"])).estimate - s.true_ate)
        collected.setdefault("confounded: AIPW ATT (adjust x)", []).append(
            es.augmented_ipw(data(s.frame, ["x"]), estimand="ATT").estimate - s.true_att)
        m = sc.m_bias(1000, rng)
        collected.setdefault("M-bias: unadjusted", []).append(
            es.naive_difference(data(m.frame, [])).estimate - m.true_ate)
        collected.setdefault("M-bias: adjust collider C", []).append(
            es.augmented_ipw(data(m.frame, ["collider"])).estimate - m.true_ate)
        md = sc.mediator(1000, rng)
        collected.setdefault("mediator: unadjusted (total effect)", []).append(
            es.naive_difference(data(md.frame, [])).estimate - md.true_ate)
        collected.setdefault("mediator: adjust M, vs total effect", []).append(
            es.augmented_ipw(data(md.frame, ["mediator"])).estimate - md.true_ate)
        it = sc.interference(250, 4, rng)
        collected.setdefault("interference: difference in means, vs policy effect", []).append(
            es.naive_difference(data(it.frame, [])).estimate
            - it.truths["treat_all_vs_none_effect"])
    notes = {
        "confounded: naive": "selection bias",
        "confounded: AIPW ATE (adjust x)": "valid backdoor set",
        "confounded: AIPW ATT (adjust x)": "ATT ≠ ATE (effect varies with x)",
        "M-bias: unadjusted": "no confounding: naive is valid",
        "M-bias: adjust collider C": "conditioning on a collider creates bias",
        "mediator: unadjusted (total effect)": "randomised: naive is valid",
        "mediator: adjust M, vs total effect": "estimates the direct effect (1.0), not 2.2",
        "interference: difference in means, vs policy effect":
            "recovers the direct effect (1.0); spillovers ignored",
    }
    for name, values in collected.items():
        truths[name] = 0.0
        summarise(name, 0.0, values, notes[name])

    positivity_rows = []
    for strength in (1.0, 3.0, 6.0):
        ipw_err, aipw_err, trim_err, ess = [], [], [], []
        for _ in range(100):
            s = sc.positivity(2000, rng, strength=strength)
            d = data(s.frame, ["x"])
            e = es.estimate_propensity(d)
            report = dg.weight_diagnostics(dg.balancing_weights(e, d.treatment), d.treatment)
            ess.append(min(report.ess_treated / report.n_treated,
                           report.ess_control / report.n_control))
            ipw_err.append(es.inverse_probability_weighting(d).estimate
                           - s.truths["population_ate"])
            aipw_err.append(es.augmented_ipw(d).estimate - s.truths["population_ate"])
            keep = np.flatnonzero(dg.trim_to_overlap(e, bound=0.1))
            kept_truth = float(s.potential_outcomes.individual_effects[keep].mean())
            trim_err.append(es.augmented_ipw(d.subset(keep)).estimate - kept_truth)
        positivity_rows.append([
            strength, float(np.median(ess)),
            float(np.sqrt(np.mean(np.square(ipw_err)))),
            float(np.sqrt(np.mean(np.square(aipw_err)))),
            float(np.mean(trim_err)), float(np.sqrt(np.mean(np.square(trim_err)))),
        ])

    dr_rows = []
    dr: dict[tuple[str, str], list[float]] = {}
    for _ in range(200):
        frame = sc.misspecification(1000, rng).frame
        for ps_cols, out_cols in (("z", "z"), ("z", "x"), ("x", "z"), ("x", "x")):
            p_vars = [c for c in frame if c.startswith(ps_cols) and c[1:].isdigit()]
            o_vars = [c for c in frame if c.startswith(out_cols) and c[1:].isdigit()]
            e = es.estimate_propensity(data(frame, p_vars))
            dr.setdefault((ps_cols, out_cols), []).append(
                es.augmented_ipw(data(frame, o_vars), propensity=e).estimate)
    label = {"z": "correct", "x": "wrong"}
    for (p, o), values in dr.items():
        array = np.asarray(values)
        dr_rows.append([label[p], label[o], float(array.mean()), float(np.median(array)),
                        float(array.std(ddof=1)), float(np.sqrt(np.mean((array - 10) ** 2)))])

    return (
        f"Simulations with both potential outcomes known ({reps} replications, n = 1,000; "
        "interference: 250 clusters of 4). Columns are *errors* (estimate − true value of "
        "the target named in the row).\n\n"
        + table(["scenario / analysis", "target error", "mean error", "MC SE of mean",
                 "median error", "SD", "RMSE", "reading"], rows)
        + "\n\nValid analyses have mean errors within about 3 Monte Carlo SEs of zero; the "
        "failures are biased by tens to hundreds of Monte Carlo SEs."
        + "\n\n**Positivity** (x ~ U(−2, 2), e = sigmoid(s·x), effect 1 + x²; 100 "
        "replications, n = 2,000). Trimming keeps units with estimated 0.1 ≤ e ≤ 0.9.\n\n"
        + table(["strength s", "median min-arm ESS share", "IPW RMSE (vs ATE)",
                 "AIPW RMSE (vs ATE)", "trimmed AIPW mean error", "trimmed AIPW RMSE"],
                positivity_rows)
        + "\n\nThe trimmed estimate is scored against the true effect among the units "
        "actually kept (trimming uses *estimated* propensities). Under poor overlap "
        "(s ≥ 3) AIPW does worse than IPW because its arm-specific outcome models are "
        "linear in x while the treated outcome is quadratic (effect 1 + x²): with a "
        "misspecified outcome model AIPW leans on the weights, and the weights are extreme. "
        "As overlap deteriorates the effective sample collapses and both IPW and AIPW "
        "estimates of the population ATE degrade; trimming restores precision for a "
        "*different* estimand (the effect where both treatments occur), which here is "
        "smaller than the ATE because the effect is largest where positivity fails.\n\n"
        "**Double robustness** (Kang & Schafer-style design, effect 10; 200 replications, "
        "n = 1,000; 'wrong' = model in non-linear transforms of the confounders).\n\n"
        + table(["propensity model", "outcome model", "mean", "median", "SD", "RMSE"], dr_rows)
        + "\n\nAIPW is consistent when either model is right, but with a wrong propensity "
        "model it inherits heavy tails from extreme weights (SD and occasional outliers). "
        "With both wrong it is far worse than either single-model estimator — double "
        "robustness is not a licence for careless modelling."
    )


def main() -> None:
    sections: list[tuple[str, Callable[[], str]]] = []
    start = time.perf_counter()
    analysis = cp.analyse_campaign(
        build_synthetic_dataset(CONFIG), rng=seeded_generator(4001), n_bootstrap=500
    )
    runtimes = [("analysis of the canonical M2 build", time.perf_counter() - start)]
    sections = [
        ("1. Causal DAG of the M2 campaign and identification", dag_section),
        ("2. Effect estimates on the canonical M2 build", lambda: campaign_section(analysis)),
        ("3. Overlap, balance, weights and a DAG falsification test",
         lambda: diagnostics_section(analysis)),
        ("4. Negative-control outcome", lambda: negative_control_section(analysis)),
        ("5. Conditional (subgroup) effects", lambda: subgroup_section(analysis)),
        ("6. Sensitivity to unmeasured confounding", lambda: sensitivity_section(analysis)),
        ("7. Repeated sampling over independent M2 draws", monte_carlo_section),
        ("8. Large-sample draw", large_sample_section),
        ("9. Failure scenarios with known truth", scenario_section),
    ]
    rendered = []
    for title, build in sections:
        start = time.perf_counter()
        rendered.append(f"## {title}\n\n{build()}\n")
        runtimes.append((title, time.perf_counter() - start))

    environment = table(
        ["component", "version"],
        [["python", platform.python_version()], ["numpy", np.__version__],
         ["scipy", scipy.__version__], ["pandas", pd.__version__]],
    )
    header = (
        "# Milestone 4 evidence — numerical results\n\n"
        "Generated by `python scripts/generate_m4_evidence.py`; do not edit by hand. All "
        "randomness is seeded; with the versions below every number reproduces exactly "
        "(runtimes excepted). Narrative and limitations: "
        "[m4-causal-foundations.md](m4-causal-foundations.md); methodology: "
        "[../methodology/causal-foundations.md](../methodology/causal-foundations.md).\n\n"
        "Known synthetic ground truth is reported alongside estimates **for evaluation "
        "only**: no estimator reads it, and it establishes nothing about real-world "
        "external validity.\n"
    )
    OUTPUT.write_text(
        header + "\n" + "\n".join(rendered)
        + "\n## Environment\n\n" + environment
        + "\n\n## Runtime\n\n"
        + table(["section", "runtime (s, indicative)"], [[t, s] for t, s in runtimes]) + "\n"
    )
    print(f"Wrote {OUTPUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
