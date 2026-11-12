"""Generate the Milestone 6 (experimentation fundamentals) evidence report.

Writes ``docs/evidence/m6-experimentation-fundamentals-results.md``. All
randomness is seeded; with the recorded dependency versions every number
reproduces exactly (runtimes excepted). Prose numbers are computed, not typed.

Usage:
    python scripts/generate_m6_evidence.py
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

from decision_intelligence.causal_inference.campaign import ReplicateSummary
from decision_intelligence.causal_inference.observational_cases import RejectionRate
from decision_intelligence.common import seeded_generator
from decision_intelligence.data.config import SyntheticDataConfig
from decision_intelligence.data.experiment_scenarios import ExperimentScenarioConfig
from decision_intelligence.data.pipeline import build_synthetic_dataset
from decision_intelligence.experimentation import cases as c
from decision_intelligence.experimentation.analysis import MetricResult
from decision_intelligence.experimentation.decision import business_impact

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "docs" / "evidence" / "m6-experimentation-fundamentals-results.md"
M2_REPS, RECOVERY_REPS, AA_LARGE_REPS, AA_SMALL_REPS = 1000, 300, 300, 1000
PLAN_REPS, SRM_SIZE_REPS, SRM_CAUSE_REPS = 1000, 1000, 300
NONCOMPLIANCE_REPS, ATTRITION_REPS, CLUSTER_REPS, SELECTION_REPS = 300, 200, 500, 300
MULTIPLE_REPS, DECISION_REPS = 1000, 1000

# Illustrative M2 economics (assumptions, stated in the report): the worthwhile effect is
# the break-even effect, annual feature cost / (eligible customers x value per conversion).
M2_ELIGIBLE_PER_YEAR, M2_VALUE_PER_CONVERSION, M2_ANNUAL_COST = 50_000, 30.0, 30_000.0

NONCOMPLIANCE = ExperimentScenarioConfig(never_taker_share=0.3, always_taker_share=0.1,
                                         compliance_engagement_link=2.0)
ATTRITION = ExperimentScenarioConfig(attrition_control=0.1, attrition_treatment=0.25,
                                     attrition_engagement_link=1.5)


def table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    def fmt(value: object) -> str:
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, float | np.floating):
            value = float(value)
            if math.isnan(value):
                return "—"
            if math.isinf(value):
                return "inf"
            return f"{value:.4g}" if abs(value) >= 1e-3 or value == 0 else f"{value:.2e}"
        return str(value)

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def ci(lower: float | None, upper: float | None, digits: int = 4) -> str:
    if lower is None or upper is None:
        return "—"
    return f"[{lower:.{digits}f}, {upper:.{digits}f}]"


def money(value: float) -> str:
    return f"−£{-value:,.0f}" if value < 0 else f"£{value:,.0f}"


def yes_no(flag: bool) -> str:
    return "yes" if flag else "no"


def reasons(text: Sequence[str]) -> str:
    return "; ".join(text) + "."


def rate_cell(r: RejectionRate) -> str:
    return f"{r.rate:.3f} [{r.ci_lower:.3f}, {r.ci_upper:.3f}]"


def summary_row(s: ReplicateSummary, coverage: bool = True) -> list[object]:
    row: list[object] = [s.estimator, s.truth, s.mean_estimate,
                         f"{s.bias:+.4f} ± {s.bias_mc_se:.4f}", s.rmse]
    if coverage:
        row += [s.mean_reported_se, s.empirical_sd,
                f"{s.coverage:.3f} [{s.coverage_ci_lower:.3f}, {s.coverage_ci_upper:.3f}]"]
    return row


SUMMARY_HEADERS = ["estimator", "truth (mean)", "mean estimate", "bias ± MC SE", "RMSE"]
COVERAGE_HEADERS = [*SUMMARY_HEADERS, "mean reported SE", "empirical SD",
                    "95% CI coverage [Wilson]"]


def metric_row(r: MetricResult) -> list[object]:
    e = r.effect
    return [r.metric, r.n_control, r.n_treatment, r.mean_control, r.mean_treatment, e.estimate,
            ci(e.ci_lower, e.ci_upper), f"{r.relative_lift:+.1%}",
            ci(r.relative_ci_lower, r.relative_ci_upper, 3), r.p_value]


METRIC_HEADERS = ["metric", "n control", "n treatment", "mean control", "mean treatment",
                  "difference", "95% CI", "relative lift", "lift 95% CI", "p-value"]


# ---------------------------------------------------------------------------
# 1–2. The M2 experiment
# ---------------------------------------------------------------------------

_DATASET = build_synthetic_dataset(SyntheticDataConfig())


def m2_section() -> str:
    a = c.analyse_m2_experiment(_DATASET, rng=seeded_generator(20260601))
    checks = [[ch.check, ch.status, ch.detail] for ch in a.checks]
    balance = a.balance.reset_index()
    balance_rows = [[r.covariate, r.mean_treated, r.mean_control, r.smd_raw]
                    for r in balance.itertuples()]
    worst = balance.loc[balance["smd_raw"].abs().idxmax()]
    balance_check_p = next(ch.value for ch in a.checks if ch.check == "pre-treatment balance")
    assert balance_check_p is not None
    p = a.primary
    impact = business_impact(p.effect.estimate, (p.effect.ci_lower, p.effect.ci_upper),
                             units_per_year=M2_ELIGIBLE_PER_YEAR,
                             value_per_unit_effect=M2_VALUE_PER_CONVERSION,
                             annual_cost=M2_ANNUAL_COST)
    plan_rows = [
        ["true effect, +4 points (known only in simulation)", a.plan_for_true_effect.n_control,
         a.plan_for_true_effect.n_total],
        ["worthwhile effect, +2 points (break-even)", a.plan_for_worthwhile_effect.n_control,
         a.plan_for_worthwhile_effect.n_total],
    ]
    secondary_rows = [[*metric_row(s)[:-1], s.p_value, a.secondary_holm[s.metric],
                       a.true_secondary_effects[s.metric]] for s in a.secondary]
    return (
        "**Pre-specified design.**\n\n"
        + table(["item", "specification"], a.design.summary_rows())
        + f"\n\nThe worthwhile effect is the break-even effect under illustrative economics "
        f"(assumptions): {M2_ELIGIBLE_PER_YEAR:,} eligible customers a year, "
        f"£{M2_VALUE_PER_CONVERSION:g} margin per incremental conversion, "
        f"£{M2_ANNUAL_COST:,.0f} annual cost → break-even "
        f"{impact.break_even_effect:.3f}.\n\n"
        "**Validity checks** (before looking at the effect).\n\n"
        + table(["check", "status", "detail"], checks)
        + "\n\nPre-treatment balance (standardised mean differences; the checks were run "
        "before outcomes were analysed):\n\n"
        + table(["covariate", "mean treatment", "mean control", "SMD"], balance_rows)
        + f"\n\nThe largest imbalance is {worst['covariate']} (SMD {worst['smd_raw']:+.3f}). "
        f"The joint test (p = {balance_check_p:.3f}) does not reach the pre-specified failure "
        "threshold of 0.001, so the assignment mechanism is not called into question; the "
        "imbalance is reported, not 'fixed' (no re-randomisation, no post-hoc covariate "
        "selection). With 11 covariates and 1,135 customers, an imbalance of this size can "
        "arise by chance (in M2, assignment is an independent coin flip — evaluation-only "
        "knowledge). Pre-specified covariate adjustment is M7 scope.\n\n"
        "**Primary metric** (ITT; units analysed as randomised):\n\n"
        + table(METRIC_HEADERS, [metric_row(p)])
        + "\n\n**Secondary metrics** (Holm-adjusted across the secondary family; the true "
        "effects are zero because M2 generates transactions and sessions independently of "
        "the experiment):\n\n"
        + table([*METRIC_HEADERS[:-1], "p-value", "Holm p", "truth"], secondary_rows)
        + "\n\n**Planning.** Assumed pre-experiment baseline 0.10. Per-arm sample sizes for 80% "
        "power at α = 0.05 (two-sided):\n\n"
        + table(["effect to detect", "n per arm", "n total"], plan_rows)
        + f"\n\nWith the realised {p.n_control} / {p.n_treatment} customers the minimum "
        f"detectable effect is {a.mde_at_realised_n:.3f} — larger than both the true and the "
        "worthwhile effect: the experiment was underpowered by design.\n\n"
        f"**Decision: {a.recommendation.decision}.** "
        + reasons(a.recommendation.reasons)
        + f" Annual net value at the estimate: {money(impact.annual_value)} (95% CI "
        f"{money(impact.annual_value_ci[0])} to {money(impact.annual_value_ci[1])}) — the "
        "interval spans loss and substantial gain. Recommendation: rerun with at least "
        f"{a.plan_for_worthwhile_effect.n_control:,} customers per arm.\n\n"
        f"**Evaluation only.** The true effect is +{a.true_effect:.2f}; the interval covers "
        f"it ({yes_no(p.effect.covers(a.true_effect))}). 'Not significant' here does not mean "
        "'no effect': the feature works, and a correct conclusion requires more data."
    )


def m2_replication_section() -> str:
    rows, decision_rows = [], []
    for effect in (0.04, 0.0):
        s = c.m2_replication_study(_DATASET, effect=effect, n_replications=M2_REPS)
        rows.append([*summary_row(s.summary), rate_cell(s.rejection), s.planned_power,
                     s.mean_significant_estimate])
        decision_rows.append([f"{effect:+.2f}", *[s.decisions[k] for k in
                                                   ("LAUNCH", "ITERATE", "HOLD", "STOP")]])
    real = rows[0]
    return (
        f"The unchanged M2 generator re-run {M2_REPS} times on the same 1,135 eligible "
        "customers (new assignment and outcomes each time).\n\n"
        + table([*COVERAGE_HEADERS, "rejection rate at 5% [Wilson]",
                 "planned power (normal approx.)", "mean estimate when significant (+)"], rows)
        + "\n\nDecision frequencies (worthwhile effect 0.02):\n\n"
        + table(["true effect", "LAUNCH", "ITERATE", "HOLD", "STOP"], decision_rows)
        + f"\n\nWith a real +0.04 effect the test detects it in only about half of the "
        "experiments (matching the planned power), so a non-significant result is common "
        "even though the feature works. Significant replicates overstate the effect "
        f"(mean {real[-1]:.3f} vs truth 0.04; type M error) — a consequence of "
        "underpowering. Under no effect the interval keeps nominal coverage and LAUNCH "
        "occurs only through an upward false positive (at most α/2), which in this design "
        "always comes with an exaggerated estimate."
    )


# ---------------------------------------------------------------------------
# 3. Scenario case
# ---------------------------------------------------------------------------


def scenario_section() -> str:
    clean, defective = c.analyse_scenario_case()
    plan = clean.plan
    out = [
        "**Design.**\n\n" + table(["item", "specification"], c.SCENARIO_DESIGN.summary_rows()),
        f"**Plan.** Baseline 0.10, worthwhile effect +0.01, 80% power, α = 0.05: "
        f"{plan.n_control:,} per arm ({plan.n_total:,} total), {plan.days} days at 2,500 "
        "eligible users a day. Assignment: stratified by segment (complete randomisation "
        "within each segment).",
    ]
    for run in (clean, defective):
        g_rows = [[g.metric, g.deterioration, ci(*g.deterioration_ci), g.margin, g.status]
                  for g in run.guardrails]
        block = (
            f"### {run.label.capitalize()} run\n\n"
            + table(["check", "status", "detail"], [[ch.check, ch.status, ch.detail]
                                                     for ch in run.checks])
            + "\n\n" + table(METRIC_HEADERS, [metric_row(run.primary), metric_row(run.revenue)])
            + f"\n\nStratified (design-based) estimate of the primary effect: "
            f"{run.stratified_primary.estimate:.4f} "
            f"{ci(run.stratified_primary.ci_lower, run.stratified_primary.ci_upper)}, SE "
            f"{run.stratified_primary.standard_error:.5f} vs "
            f"{run.primary.effect.standard_error:.5f} "
            "unstratified (segment is only weakly prognostic here, so blocking gains little).\n\n"
            "Guardrails (deterioration = change in the undesirable direction; non-inferiority "
            "margin):\n\n"
            + table(["guardrail", "deterioration", "95% CI", "margin", "status"], g_rows)
            + f"\n\n**Decision: {run.recommendation.decision}.** "
            + reasons(run.recommendation.reasons)
            + f"\n\nEvaluation only: true ITT effect {run.true_effect:.4f}; covered by the "
            f"primary interval: {yes_no(run.primary.effect.covers(run.true_effect))}."
        )
        if run.lost_unit_profile is not None:
            profile = run.lost_unit_profile.reset_index()
            by_segment = run.srm_by_segment
            block += (
                "\n\n**Detection is not diagnosis.** The SRM test says the analysed sample is "
                "inconsistent with a 50/50 design. Two diagnostic aids narrow the cause "
                "without establishing it. By segment (Holm-adjusted):\n\n"
                + table(list(by_segment.columns), by_segment.to_numpy().tolist())
                + "\n\nThe loss is spread across segments rather than confined to one. "
                "Comparing the server-side assignment log with the analysed telemetry "
                "shows which units are missing and their pre-treatment profile:\n\n"
                + table(["arm", "status", "mean pre-sessions", "mean pre-revenue", "n"],
                        profile.to_numpy().tolist())
                + "\n\nOnly treatment units are missing, and they were less engaged before "
                "the experiment — consistent with a treatment-side failure affecting "
                "low-engagement users (in this simulation, the injected mechanism). An "
                "engineering investigation must confirm it; until then the comparison is "
                "not interpreted, even though the point estimate looks plausible."
            )
        out.append(block)
    sel = clean.selection_on_engaged
    out.append(
        "**Post-randomisation selection (clean run).** Restricting the analysis to users who "
        f"'engaged' (a behaviour the treatment itself changes) gives "
        f"{sel.effect.estimate:.4f} {ci(sel.effect.ci_lower, sel.effect.ci_upper)} for the "
        f"same experiment whose ITT is {clean.primary.effect.estimate:.4f}. The engaged "
        "subgroups of the two arms are different populations; repeated-sampling bias is in "
        "section 8."
    )
    return "\n\n".join(out)


# ---------------------------------------------------------------------------
# 4–8. Repeated-sampling studies
# ---------------------------------------------------------------------------


def calibration_section() -> str:
    recovery = c.recovery_study(ExperimentScenarioConfig(), n_replications=RECOVERY_REPS)
    aa = (c.aa_study(n_units=20_000, n_replications=AA_LARGE_REPS)
          + c.aa_study(n_units=200, n_replications=AA_SMALL_REPS))
    return (
        f"**Effect recovery and coverage** ({RECOVERY_REPS} independently generated "
        "populations of 20,000 users; heterogeneous effects 0.01 / 0.02 / 0.04 by segment).\n\n"
        + table(COVERAGE_HEADERS, [summary_row(s) for s in recovery.values()])
        + "\n\n**A/A tests** (no effect on any metric): false-positive rates at nominal 5%.\n\n"
        + table(["analysis", "false-positive rate [Wilson]"],
                [[r.procedure, rate_cell(r)] for r in aa])
        + "\n\nAt scale all three analyses hold their size. With 100 users per arm the "
        "zero-inflated, heavy-tailed revenue metric breaks the large-sample approximations "
        "in opposite directions: Welch's t-test becomes conservative and the BCa bootstrap "
        "anti-conservative. The bootstrap is not a small-sample cure for heavy tails."
    )


def planning_section() -> str:
    rows = [[p.design, p.n_per_arm, p.target_power, rate_cell(p.empirical)]
            for p in c.planning_validation_study(n_replications=PLAN_REPS)]
    return (
        f"Simulated power at the planned per-arm sample size ({PLAN_REPS} experiments per "
        "design; analysed with the same tests used in practice).\n\n"
        + table(["design", "n per arm", "target power", "empirical power [Wilson]"], rows)
    )


def srm_section() -> str:
    size = c.srm_size_study(n_units=10_000, n_replications=SRM_SIZE_REPS)
    causes = c.srm_cause_study(loss_shares=(0.01, 0.02, 0.05), n_replications=SRM_CAUSE_REPS)
    rows = [[r.mechanism, f"{r.loss_share:.0%}", rate_cell(r.detection),
             f"{r.bias.bias:+.4f} ± {r.bias.bias_mc_se:.4f}"] for r in causes]
    truth = ExperimentScenarioConfig()
    by = {(r.mechanism, r.loss_share): r for r in causes}
    effect = 0.5 * 0.01 + 0.35 * 0.02 + 0.15 * 0.04
    small = by[("inactive", 0.02)]
    return (
        f"**Size** ({SRM_SIZE_REPS} hash-assigned experiments of 10,000 units, new salt each):\n\n"
        + table(["threshold", "false-alarm rate [Wilson]"],
                [[r.procedure, rate_cell(r)] for r in size])
        + f"\n\n**Same detection, different damage** ({SRM_CAUSE_REPS} experiments of 20,000 "
        "users per row; treatment-arm units lost after randomisation). Bias of the "
        f"conversion difference against the true effect (≈{effect:.3f}):\n\n"
        + table(["loss mechanism", "loss share", "SRM detected at p <= 0.001 [Wilson]",
                 "bias ± MC SE"], rows)
        + "\n\nAt 5% loss, random loss (a logging outage) is detected about as often "
        f"({by[('random', 0.05)].detection.rate:.3f}) as a filter that drops treated users who "
        f"did not convert ({by[('inactive', 0.05)].detection.rate:.3f}), but only "
        "outcome-related loss biases the estimate "
        f"({by[('random', 0.05)].bias.bias:+.4f} vs {by[('inactive', 0.05)].bias.bias:+.4f}). "
        f"At 2% outcome-related loss the SRM test fired in {small.detection.rate:.1%} of "
        f"experiments while the estimate was biased by {small.bias.bias:+.4f} "
        f"({small.bias.bias / effect:.0%} of the effect): passing the SRM check is not proof "
        "that no damaging loss occurred. Low-engagement loss (a client failure correlated with "
        "a pre-treatment trait) is in between. (Configuration defaults: "
        f"{truth.n_units:,} units, 50/50 allocation.)"
    )


def noncompliance_attrition_section() -> str:
    nc = c.noncompliance_study(NONCOMPLIANCE, n_replications=NONCOMPLIANCE_REPS)
    ratios = [nc[k].mean_estimate / nc[k].truth for k in ("as-treated (vs ATE)",
                                                          "per-protocol (vs ATE)")]
    at = c.attrition_study(ATTRITION, n_replications=ATTRITION_REPS)
    bounds_rows = [
        [at.lee_contains_truth.procedure, rate_cell(at.lee_contains_truth), at.lee_mean_width],
        [at.lee_ci_contains_truth.procedure, rate_cell(at.lee_ci_contains_truth), "—"],
        [at.manski_contains_truth.procedure, rate_cell(at.manski_contains_truth),
         at.manski_mean_width],
    ]
    return (
        f"**Noncompliance** ({NONCOMPLIANCE_REPS} experiments): 30% never-takers, 10% "
        "always-takers (control users exposed — leakage), compliance more likely among "
        "engaged users.\n\n"
        + table(COVERAGE_HEADERS, [summary_row(s) for s in nc.values()])
        + "\n\nThe ITT (by assignment) and the Wald complier effect recover their own "
        "estimands. As-treated and per-protocol comparisons select on a post-randomisation "
        "behaviour correlated with outcomes and overstate the effect "
        f"{min(ratios):.1f}–{max(ratios):.1f}-fold.\n\n"
        f"**Attrition** ({ATTRITION_REPS} experiments): outcomes missing for "
        f"{1 - at.mean_response_control:.0%} of control and "
        f"{1 - at.mean_response_treatment:.0%} of treatment users, more often among the less "
        "engaged (monotone: treatment never prevents an outcome from being observed).\n\n"
        + table(COVERAGE_HEADERS, [summary_row(at.complete_case)])
        + "\n\n" + table(["bounds", "contain their target [Wilson]", "mean width"], bounds_rows)
        + "\n\nComplete-case analysis is biased upwards because treated respondents are a more "
        "engaged subset. Lee bounds target the effect among always-observed users (under "
        "monotone response); Manski bounds target the ITT for everyone with no assumptions "
        f"and are wider still. Both always contain their targets here, but at widths of "
        f"{at.lee_mean_width:.3f} and {at.manski_mean_width:.3f} against an effect of about "
        f"{at.complete_case.truth:.3f} they cannot even sign the effect: with this much "
        "differential attrition, honest uncertainty is large. Neither is a point estimate."
    )


def design_failures_section() -> str:
    clusters = (c.clustering_study(n_clusters=40, cluster_sd=0.5, n_replications=CLUSTER_REPS)
                + c.clustering_study(n_clusters=10, cluster_sd=0.5,
                                     n_replications=CLUSTER_REPS))
    selection = c.post_selection_study(n_replications=SELECTION_REPS)
    few_cr1, few_means = clusters[4], clusters[5]
    return (
        f"**Cluster randomisation analysed per user** ({CLUSTER_REPS} A/A experiments per "
        "configuration; 6,000 users in cities with a random city effect on conversion):\n\n"
        + table(["analysis", "false-positive rate at 5% [Wilson]"],
                [[r.procedure, rate_cell(r)] for r in clusters])
        + "\n\nTreating users as independent when whole cities were randomised understates "
        "the standard error; the analysis unit must respect the randomisation unit "
        "(enforced by `ExperimentDesign`). With 10 clusters CR1 rejects "
        f"{few_cr1.rate:.3f} [{few_cr1.ci_lower:.3f}, {few_cr1.ci_upper:.3f}] and the "
        f"cluster-means test {few_means.rate:.3f}: few clusters strain the robust variance "
        "(see M5 for the wild cluster bootstrap).\n\n"
        f"**Post-randomisation selection** ({SELECTION_REPS} experiments): analysing only "
        "users who 'engaged' after assignment.\n\n"
        + table(SUMMARY_HEADERS, [summary_row(selection, coverage=False)])
    )


def multiplicity_decision_section() -> str:
    mt = c.multiple_testing_study(n_replications=MULTIPLE_REPS)
    mt_rows = [[r.procedure, rate_cell(r.familywise_error), r.false_discovery_rate, r.power]
               for r in mt]
    mt_by = {r.procedure: r for r in mt}
    plan, oc_table = c.decision_operating_characteristics(
        effects=(-0.005, 0.0, 0.005, 0.01, 0.015, 0.02), n_replications=DECISION_REPS
    )
    launch = oc_table.set_index("true effect")["LAUNCH"]
    launch_null, launch_harm, launch_delta = launch[0.0], launch[-0.005], launch[0.01]
    return (
        f"**Multiple metrics** ({MULTIPLE_REPS} experiments, 10 metrics of which 3 have a "
        "true effect of 0.1 SD, 1,000 users per arm, independent metrics):\n\n"
        + table(["procedure", "family-wise error [Wilson]", "false discovery rate",
                 "average power"], mt_rows)
        + "\n\nUnadjusted testing makes at least one false claim in "
        f"{mt_by['unadjusted'].familywise_error.rate:.1%} of experiments. Holm controls the "
        "family-wise error with at least Bonferroni's power; Benjamini–Hochberg controls the "
        "false discovery rate (not the family-wise error, which it exceeds here) with more "
        "power still.\n\n"
        f"**Decision-rule operating characteristics** ({DECISION_REPS} experiments per row; "
        f"baseline 0.10, worthwhile effect +0.01, {plan.n_control:,} per arm = 80% power for "
        "+0.01):\n\n"
        + table(["true effect", "LAUNCH", "ITERATE", "HOLD", "STOP"],
                oc_table[["true effect", "LAUNCH", "ITERATE", "HOLD", "STOP"]].to_numpy()
                .tolist())
        + f"\n\nA useless change (true effect 0) is launched in {launch_null:.1%} of experiments "
        f"and a harmful one (−0.005) in {launch_harm:.1%}; a change exactly at the worthwhile "
        f"effect launches {launch_delta:.1%} of the time (the estimate must reach the "
        "threshold), and clearly worthwhile effects launch reliably. ITERATE captures real "
        "but sub-threshold effects. The thresholds are a business choice; these frequencies "
        "are what that choice implies."
    )


def detection_section() -> str:
    rows = [
        ["sample ratio mismatch", "yes, when large enough", "detection says nothing about "
         "cause or bias; small damaging losses pass undetected (section 5)"],
        ["duplicate / conflicting assignment", "yes (assignment log)", "first assignment "
         "kept; contamination reported"],
        ["assignment leakage / noncompliance", "yes, if exposure is logged", "ITT stays valid "
         "for assignment; complier effect needs IV assumptions"],
        ["pre-treatment imbalance", "yes (chance vs mechanism)", "chance imbalance is "
         "expected; never re-randomise post hoc"],
        ["post-treatment exclusions / selection", "only if variable timing is declared",
         "the data cannot prove a variable is unaffected by treatment"],
        ["differential attrition", "rates yes; mechanism no", "bounds, not complete-case "
         "estimates"],
        ["metric integrity", "domain and missingness checks", "silent definition changes "
         "are not detectable from the data"],
        ["interference between units", "no (not in these checks)", "requires design "
         "knowledge or cluster/switchback designs (M7)"],
    ]
    return table(["validity threat", "detectable from the data?", "limits"], rows)


def main() -> None:
    sections: list[tuple[str, Callable[[], str]]] = [
        ("1. The M2 recommendation-feature experiment", m2_section),
        ("2. The M2 experiment in repeated sampling", m2_replication_section),
        ("3. Scenario case: guardrails, clean and defective runs", scenario_section),
        ("4. Calibration: effect recovery, coverage and A/A tests", calibration_section),
        ("5. Sample ratio mismatch: size, detection and damage", srm_section),
        ("6. Planning: simulated power at planned sample sizes", planning_section),
        ("7. Noncompliance and attrition", noncompliance_attrition_section),
        ("8. Design failures: clustering and post-randomisation selection",
         design_failures_section),
        ("9. Multiple metrics and decision rules", multiplicity_decision_section),
        ("10. What the validity checks can and cannot detect", detection_section),
    ]
    rendered, runtimes = [], []
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
        "# Milestone 6 evidence — numerical results\n\n"
        "Generated by `python scripts/generate_m6_evidence.py`; do not edit by hand. All "
        "randomness is seeded; with the versions below every number reproduces exactly "
        "(runtimes excepted). Narrative and limitations: "
        "[m6-experimentation-fundamentals.md](m6-experimentation-fundamentals.md); "
        "methodology: [../methodology/experimentation-fundamentals.md]"
        "(../methodology/experimentation-fundamentals.md).\n\n"
        "Ground truth is known only because the data are synthetic; it is read after "
        "estimation and establishes nothing about real-world validity.\n"
    )
    OUTPUT.write_text(
        header + "\n" + "\n".join(rendered) + "\n## Environment\n\n" + environment
        + "\n\n## Runtime\n\n"
        + table(["section", "runtime (s, indicative)"], [[t, s] for t, s in runtimes]) + "\n"
    )
    print(f"Wrote {OUTPUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
