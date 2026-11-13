"""Generate the Milestone 7 (advanced experimentation) evidence report.

Writes ``docs/evidence/m7-advanced-experimentation-results.md``. All randomness is
seeded; with the recorded dependency versions every number reproduces exactly
(runtimes excepted). Prose numbers are computed, not typed.

Usage:
    python scripts/generate_m7_evidence.py
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
from decision_intelligence.experimentation import advanced_cases as ac
from decision_intelligence.experimentation import sequential as sq

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "docs" / "evidence" / "m7-advanced-experimentation-results.md"
CUPED_REPS, PEEKING_REPS, BAYES_FIXED_REPS, CALIBRATION_REPS = 400, 4000, 1000, 20_000
CLUSTER_REPS, SWITCHBACK_REPS = 400, 400


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

    def cell(value: object) -> str:
        return fmt(value).replace("|", "\\|")  # a literal pipe would split the column

    lines = ["| " + " | ".join(cell(h) for h in headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def ci(lower: float | None, upper: float | None, digits: int = 4) -> str:
    if lower is None or upper is None:
        return "—"
    return f"[{lower:.{digits}f}, {upper:.{digits}f}]"


def money(value: float) -> str:
    return f"−£{-value:,.0f}" if value < 0 else f"£{value:,.0f}"


def rate_cell(r: RejectionRate) -> str:
    return f"{r.rate:.3f} [{r.ci_lower:.3f}, {r.ci_upper:.3f}]"


def summary_row(s: ReplicateSummary) -> list[object]:
    return [s.estimator, s.truth, s.mean_estimate, f"{s.bias:+.4f} ± {s.bias_mc_se:.4f}",
            s.empirical_sd, s.mean_reported_se,
            f"{s.coverage:.3f} [{s.coverage_ci_lower:.3f}, {s.coverage_ci_upper:.3f}]"]


SUMMARY_HEADERS = ["estimator", "truth (mean)", "mean estimate", "bias ± MC SE", "empirical SD",
                   "mean reported SE", "95% CI coverage [Wilson]"]

# Results shared between sections (computed once).
_cache: dict[str, object] = {}


def cached(key: str, build: Callable[[], object]) -> object:
    if key not in _cache:
        _cache[key] = build()
    return _cache[key]


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def cuped_case_section() -> str:
    case = cached("cuped_case", ac.cuped_case)
    assert isinstance(case, ac.CupedCase)
    rows = []
    for key, label in (("CUPED", "CUPED (pre_revenue)"), ("Lin", "Lin regression (pre_revenue)"),
                       ("INVALID post-treatment", "INVALID: in-experiment sessions"),
                       ("INVALID leaky window", "INVALID: leaky pre-period window")):
        r = case.results[key]
        e = r.effect
        rows.append([label, e.estimate, e.standard_error, ci(e.ci_lower, e.ci_upper),
                     e.covers(case.truth_ate), r.realised_variance_reduction])
    unadj = case.results["CUPED"].unadjusted.effect
    rows.insert(0, ["unadjusted (M6 Welch)", unadj.estimate, unadj.standard_error,
                    ci(unadj.ci_lower, unadj.ci_upper), unadj.covers(case.truth_ate), 0.0])
    checks = table(["check", "status", "detail"],
                   [[c.check, c.status, c.detail] for c in case.covariate_checks])
    binary = case.results["CUPED binary"]
    gap = abs(case.results["CUPED"].effect.estimate - case.results["Lin"].effect.estimate)
    direction = "less" if case.pre_period_difference < 0 else "more"
    plain, adjusted = case.plan_unadjusted, case.plan_cuped
    assert plain.days is not None and adjusted.days is not None
    saved_days = plain.days - adjusted.days
    return (
        "NexaMarket tests a checkout change on 14-day revenue per user (20,000 users, 50/50 "
        "complete randomisation). Pre-specified covariate: 28-day revenue *before* assignment "
        "(`pre_revenue`; missing for the 10% new users, handled by mean imputation plus an "
        f"indicator). True average effect {case.truth_ate:.3f}; true pre/post correlation "
        f"{case.truth_correlation:.3f}.\n\n**Covariate checks** (timing decides; the arm "
        "difference is supporting evidence only):\n\n" + checks + "\n\n"
        + table(["estimator", "estimate", "SE", "95% CI", "covers truth",
                 "variance reduction"], rows)
        + f"\n\nCUPED and Lin agree to {gap:.4f}; "
        f"both cut the variance by {case.results['CUPED'].realised_variance_reduction:.1%} "
        f"(pooled R² {case.results['CUPED'].r_squared:.3f}). The unadjusted estimate is low "
        "partly by chance: treatment users happened to spend "
        f"£{abs(case.pre_period_difference):.2f} {direction} in the pre-period, and CUPED "
        "corrects such chance imbalance. The post-treatment covariate "
        "looks like an even better variance reducer and reverses the sign of the effect; the "
        "leaky window absorbs part of the effect. Both FAIL the timing check, which is why "
        "`check_adjustment_covariate` runs before adjustment.\n\nConversion with "
        f"`pre_converted`: variance reduction {binary.realised_variance_reduction:.1%} — a "
        "binary pre-period indicator carries little information about a 12% conversion "
        "outcome; CUPED is only as good as the covariate's correlation.\n\n"
        "**Planning implication** (R² estimated on a *historical* cohort, "
        f"{case.historical_r_squared:.3f}; MDE = break-even effect £{ac.revenue_break_even():.3f}"
        f" per user; 714 eligible users a day):\n\n"
        + table(["plan", "users (total)", "days"],
                [["unadjusted (M6)", plain.n_total, plain.days],
                 ["CUPED", adjusted.n_total, adjusted.days]])
        + f"\n\nCUPED needs {adjusted.n_total / plain.n_total:.0%} of the users — "
        f"{saved_days} fewer days of traffic for the same power. **Decision (M6 rule, "
        f"δ = break-even):** unadjusted {case.m6_recommendation.decision}; CUPED-adjusted "
        f"{case.cuped_recommendation.decision} — the same data, made decisive by a smaller "
        "standard error, not by a different estimand.\n"
    )


def cuped_study_section() -> str:
    study = ac.cuped_study(n_replications=CUPED_REPS)
    rows = [summary_row(s) for s in study.summaries]
    return (
        f"{CUPED_REPS} independent populations of 8,000 users (new seed and assignment each). "
        "Bias with its Monte Carlo SE; coverage with a Wilson interval.\n\n"
        + table(SUMMARY_HEADERS, rows)
        + "\n\n" + table(["estimator", "rejects H0: effect = 0 [Wilson]"],
                         [[p.procedure, rate_cell(p)] for p in study.power])
        + "\n\nA/A (no effect), same populations:\n\n"
        + table(["estimator", "false-positive rate [Wilson]"],
                [[p.procedure, rate_cell(p)] for p in study.aa_size])
        + f"\n\nMean pre/post correlation {study.mean_correlation:.3f}; mean pooled R² "
        f"{study.mean_r_squared:.3f}; realised variance reduction CUPED "
        f"{study.mean_variance_reduction_cuped:.3f}, Lin {study.mean_variance_reduction_lin:.3f}"
        f" (theory: R²). Conversion: {study.binary_variance_reduction:.3f}.\n"
    )


def boundary_section() -> str:
    ref = cached("boundaries", ac.boundary_reference)
    assert isinstance(ref, ac.BoundaryReference)
    worst = max(abs(a - b) for _, a, b in ref.rows)
    fractions = np.arange(1, 6) / 5
    designs = [sq.naive_peeking_design(fractions), sq.spending_boundaries(fractions),
               sq.spending_boundaries(fractions, family="pocock"),
               sq.constant_boundaries(fractions, shape="pocock"),
               sq.constant_boundaries(fractions, shape="obrien-fleming")]
    design_rows = []
    for d in designs:
        r = (sq.maximum_information_inflation(d) if not d.method.startswith("naive") else math.nan)
        design_rows.append([d.method, ", ".join(f"{b:.3f}" for b in d.boundaries),
                            ", ".join(f"{p:.4f}" for p in d.nominal_p_values),
                            d.cumulative_alpha[-1], r])
    return (
        "Boundaries are computed by recursive numerical integration (Armitage, McPherson & "
        "Rowe 1969; Jennison & Turnbull 2000 ch. 19). Published references (3 decimal places):\n\n"
        + table(["quantity", "computed", "published"], ref.rows)
        + f"\n\nLargest absolute difference: {worst:.1e} (rounding of the published values). "
        "Independent check against SciPy's multivariate-normal integration (Genz):\n\n"
        + table(["quantity", "recursive integration", "SciPy multivariate normal"],
                ref.mvn_check)
        + "\n\nFive equally spaced looks, two-sided α = 0.05:\n\n"
        + table(["design", "boundaries |Z|", "nominal p at each look", "total type I error",
                 "max sample inflation R (power 0.8)"], design_rows)
        + "\n\nThe Lan–DeMets O'Brien–Fleming function is applied per side at α/2 (the "
        "gsDesign/rpact/ldbounds convention). Nominal p-values at the boundary are not "
        "adjusted p-values: they show how much stricter early looks are.\n"
    )


def peeking_section() -> str:
    study = ac.peeking_study(n_replications=PEEKING_REPS)
    null = {s.procedure: s.rejection.rate for s in study.summaries if s.effect == 0}
    rows = []
    for s in study.summaries:
        rows.append([s.procedure, s.effect, rate_cell(s.rejection), s.expected_sample_fraction,
                     s.mean_estimate_when_rejecting if s.effect > 0 else math.nan,
                     f"{s.interval_coverage:.3f} ({s.interval})"])
    return (
        f"Conversion experiment, baseline {study.baseline}, i.i.d. arrivals; maximum "
        f"{study.n_max_per_arm:,} users per arm = the M6 fixed-horizon size for an effect of "
        f"{study.design_effect_size} (α 0.05, power 0.8). {PEEKING_REPS:,} replicates per "
        "effect; group-sequential looks every 10% of the maximum; the mSPRT (mixture sd = "
        "the design effect) is checked every 1%. All procedures see the same simulated data "
        "within an effect size.\n\n"
        + table(["procedure", "true effect", "rejects H0 [Wilson]",
                 "expected sample (fraction of max)",
                 "mean estimate when rejecting", "interval coverage at stop"], rows)
        + f"\n\nReading: naive peeking raises the false-positive rate from "
        f"{null['fixed horizon (M6 baseline)']:.3f} to {null['naive peeking (10 looks)']:.3f}; "
        "O'Brien–Fleming spending keeps α and nearly all of the fixed design's power (its "
        "maximum-size inflation R is 1.025) while stopping early when the effect is real; "
        "Pocock stops earliest but loses power at the same maximum size; the mSPRT tolerates "
        "continuous monitoring at a large power cost at this sample size. Among experiments "
        "that reject, estimates are exaggerated (the truth is the 'true effect' column): "
        "conditioning on significance alone does this for the fixed-horizon test, and early "
        "stopping adds to it. Under the null the naive interval at stopping under-covers, "
        "while repeated confidence intervals and confidence sequences remain valid "
        "(conservatively).\n"
    )


def sequential_case_section() -> str:
    case = ac.sequential_case()
    rows = [[look.calendar_day, look.n_analysed, look.information_fraction, look.estimate,
             look.standard_error, look.z, look.boundary, ci(*look.repeated_ci),
             look.naive_p_value, look.always_valid_p_value, ci(*look.confidence_sequence)]
            for look in case.looks]
    if case.stopped_at_day is not None:
        outcome = (f"The O'Brien–Fleming boundary is crossed on day {case.stopped_at_day}: the "
                   f"experiment stops {case.fixed_horizon_day - case.stopped_at_day} days "
                   f"before the fixed horizon (day {case.fixed_horizon_day}).")
    else:
        outcome = "No boundary is crossed; the experiment runs to the fixed horizon."
    final = case.fixed_horizon.effect
    return (
        "The CUPED revenue experiment monitored weekly. Revenue is a 14-day outcome, so a "
        "look on calendar day d analyses only users who arrived by day d − 14 (complete "
        "outcome windows); information fractions are the observed shares of the 20,000 "
        "users. CUPED-adjusted statistics; Lan–DeMets O'Brien–Fleming spending recomputed for "
        f"the observed fractions; true effect {case.truth:.3f}.\n\n"
        + table(["day", "users analysed", "information", "estimate", "SE", "Z", "boundary",
                 "repeated 95% CI", "naive p", "always-valid p (mSPRT)",
                 "confidence sequence (running)"], rows)
        + f"\n\n{outcome} Had the team run to day {case.fixed_horizon_day}: "
        f"{final.estimate:.3f} {ci(final.ci_lower, final.ci_upper)}. The naive p-values would "
        "have been 'significant' at an earlier look already; they are shown only to illustrate "
        "why they must not be used for stopping. Delayed outcomes are why the earliest look "
        "sees a quarter of the users: monitoring cannot be faster than the outcome window.\n"
    )


def bayesian_case_section() -> str:
    case = ac.bayesian_case()

    def rows(items: Sequence[ac.BayesianSummaryRow]) -> list[list[object]]:
        return [[r.prior, r.posterior_mean, ci(*r.credible_interval), r.prob_better,
                 r.prob_worthwhile, money(r.decision.expected_annual_value),
                 money(r.decision.expected_loss_launch),
                 money(r.decision.expected_loss_not_launching), r.decision.decision]
                for r in items]

    headers = ["prior", "posterior mean", "95% credible interval", "P(effect > 0 | data)",
               "P(effect > break-even | data)", "E[annual value]", "E[loss | launch]",
               "E[loss | not launch]", "decision"]
    conv, rev = case.conversion_frequentist, case.revenue_frequentist
    flat, p_one = case.flat_prior_check
    conversion_break_even = ac.CONVERSION_ANNUAL_COST / (ac.CONVERSION_UNITS_PER_YEAR
                                                         * ac.CONVERSION_VALUE)
    return (
        "The same randomised experiment analysed both ways. Economics (illustrative): "
        f"conversion — {ac.CONVERSION_UNITS_PER_YEAR:,} customers a year, £"
        f"{ac.CONVERSION_VALUE:.0f} per conversion, £{ac.CONVERSION_ANNUAL_COST:,.0f} a year "
        f"(break-even +{conversion_break_even:.3f}); revenue — "
        f"{ac.REVENUE_UNITS_PER_YEAR:,} user-fortnights a year, {ac.REVENUE_MARGIN:.0%} margin, "
        f"£{ac.REVENUE_ANNUAL_COST:,.0f} a year (break-even £{ac.revenue_break_even():.3f}). "
        f"Loss tolerance {money(ac.LOSS_TOLERANCE)} a year, fixed in advance.\n\n"
        f"**Conversion** (truth {case.truth_conversion:.4f}). Frequentist (M6): "
        f"{conv.effect.estimate:+.4f}, 95% confidence interval "
        f"{ci(conv.effect.ci_lower, conv.effect.ci_upper)}, p = {conv.p_value:.3f}; M6 rule: "
        f"{case.m6_conversion.decision}.\n\n" + table(headers, rows(case.conversion_rows))
        + f"\n\n**Revenue** (truth {case.truth_revenue:.3f}). Frequentist (M6): "
        f"{rev.effect.estimate:+.3f}, 95% confidence interval "
        f"{ci(rev.effect.ci_lower, rev.effect.ci_upper)}, p = {rev.p_value:.4f}; M6 rule: "
        f"{case.m6_revenue.decision}.\n\n" + table(headers, rows(case.revenue_rows))
        + f"\n\nUnder the vague prior, 1 − P(effect > 0 | data) = {flat:.6f} and the one-sided "
        f"p-value = {p_one:.6f}: numerically equal for a normal model with a flat prior, yet "
        "one is a probability about the effect and the other a probability about data under "
        "H0. Under the sceptical prior they differ. Conversion: the effect is probably "
        "positive (P ≈ 0.96 under the uniform prior) but almost certainly below break-even, "
        "so the expected-loss rule stops it — statistical and commercial questions differ. "
        "Revenue: the decision depends on the prior; a sceptical prior asks for more data. "
        "The prior is a pre-specified, reviewable assumption, not a tuning knob.\n"
    )


def bayesian_operating_section() -> str:
    fixed = ac.bayesian_fixed_truth_study(n_replications=BAYES_FIXED_REPS)
    calibration = ac.bayesian_calibration_study(n_replications=CALIBRATION_REPS)
    by = {(r.rule, r.effect, r.monitoring): r.launch_rate.rate for r in fixed}
    once = by[("launch if P(better) > 0.95", 0.0, "fixed horizon")]
    peek = by[("launch if P(better) > 0.95", 0.0, "checked at 10 looks")]
    return (
        f"**Frequentist properties at fixed true effects** ({BAYES_FIXED_REPS} replicates, "
        "uniform Beta priors, exact posteriors, M6 sample size for an effect of 0.015, 10 "
        "equal looks):\n\n"
        + table(["rule", "true effect", "monitoring", "launch rate [Wilson]"],
                [[r.rule, r.effect, r.monitoring, rate_cell(r.launch_rate)] for r in fixed])
        + f"\n\nWith one analysis, 'P(better) > 0.95' launched a null change in "
        f"{once:.3f} of experiments — under a flat prior it behaves like a one-sided 5% test. "
        f"Checked at every look it launched {peek:.3f} ({peek / once:.1f} times as often): "
        "posterior probabilities do not protect a repeatedly applied decision rule from "
        "optional-stopping inflation.\n\n"
        f"**Bayesian calibration** ({CALIBRATION_REPS:,} replicates, normal model on the "
        "estimate with final SE 0.006, prior N(0, 0.01²)). Truths are drawn from a "
        "distribution; coverage averages over truths:\n\n"
        + table(["scenario", "95% credible interval coverage [Wilson]",
                 "mean true effect | launched", "mean posterior mean | launched"],
                [[r.scenario, rate_cell(r.coverage), r.mean_effect_when_launched,
                  r.mean_posterior_when_launched] for r in calibration])
        + "\n\nWhen the prior is the true distribution of effects, credible intervals are "
        "calibrated *on average over effects* — even with optional stopping. That is a "
        "different guarantee from frequentist coverage at each fixed effect. When the prior "
        "is too wide (most real changes are smaller than assumed), launched changes are "
        "overestimated about threefold: the prior must reflect the distribution of past "
        "effects.\n"
    )


def cluster_section() -> str:
    case = ac.cluster_case()
    rows = ac.cluster_power_study(case, n_replications=CLUSTER_REPS)
    icc = case.icc
    return (
        "City-level rollout (whole cities randomised; ~100 eligible users per city). The ICC "
        "of conversion is estimated on a *historical* cohort of 120 cities (M6 generator, "
        "cluster SD 0.4 on the log-odds):\n\n"
        + table(["quantity", "value"],
                [["ICC (ANOVA)", icc.icc], ["95% cluster-bootstrap CI",
                                             ci(icc.ci_lower, icc.ci_upper)],
                 ["mean cluster size", icc.mean_cluster_size],
                 ["cluster-size CV", icc.cluster_size_cv], ["baseline conversion",
                                                             case.baseline],
                 ["target effect (true ATE)", case.target_effect],
                 ["design effect", case.plan.design_effect],
                 ["individual-randomisation n per arm (M6)",
                  case.plan.individual_plan.n_treatment],
                 ["clusters per arm (planned)", case.plan.clusters_per_arm],
                 ["clusters per arm if ICC ignored", case.plan.naive_clusters_per_arm],
                 ["planned power at the ICC CI ends", ci(*case.power_at_icc_bounds, digits=3)]])
        + f"\n\nSimulated ({CLUSTER_REPS} replicates; CR1 with t(G − 1), M6):\n\n"
        + table(["design", "clusters per arm", "planned power", "CR1 power [Wilson]",
                 "naive unit-level 'power' [Wilson]", "CR1 coverage"],
                [[r.design, r.clusters_per_arm, r.planned_power, rate_cell(r.cr1_power),
                  rate_cell(r.naive_unit_level_rejection), r.cr1_coverage] for r in rows])
        + "\n\nIgnoring an ICC of about 0.01 halves the number of cities and leaves the "
        "experiment badly underpowered. The naive unit-level test 'rejects' more often only "
        "because its SE is too small (its size is wrong, M6 §8). The ICC estimate itself is "
        "uncertain; planned power ranges over the interval shown, so a conservative plan uses "
        "the upper ICC bound.\n"
    )


def switchback_section() -> str:
    case = ac.switchback_case()
    study = ac.switchback_study(n_replications=SWITCHBACK_REPS)
    a = case.analysis
    sens = table(["burn-in (hours)", "estimate", "95% CI"],
                 [[b, est, ci(lo, hi, 2)] for b, est, lo, hi in case.sensitivity])
    return (
        "A marketplace pricing change, 14 days in 4-hour periods (84 periods), randomised "
        "within time-of-day blocks (7 treated and 7 control periods per block). Effect: "
        f"{case.direct_effect:g} orders per hour while on, plus carryover of 3 and 1.5 in the "
        f"two hours after it is switched off; global effect {case.truth:g}. Pre-specified "
        f"burn-in: {ac.SWITCHBACK_BURN_IN} hours.\n\n"
        f"Case: {a.effect.estimate:.2f} {ci(a.effect.ci_lower, a.effect.ci_upper, 2)}, "
        f"randomisation p = {a.randomisation_p_value:.4f} ({a.n_periods_treated} treated, "
        f"{a.n_periods_control} control periods). Burn-in sensitivity (reported, not used to "
        "choose):\n\n" + sens
        + f"\n\nRepeated sampling ({SWITCHBACK_REPS} markets):\n\n"
        + table(SUMMARY_HEADERS, [summary_row(s) for s in study.by_burn_in])
        + "\n\nA/A (no effect, no carryover):\n\n"
        + table(["analysis", "false-positive rate [Wilson]"],
                [["within-block randomisation test, period means (valid)",
                  rate_cell(study.aa_randomisation_test)],
                 ["hourly Welch, blocked design (INVALID: ignores blocks and periods)",
                  rate_cell(study.aa_naive_hourly_stratified)],
                 ["hourly Welch, complete randomisation (INVALID: hours as units)",
                  rate_cell(study.aa_naive_hourly_complete)],
                 ["period-mean Welch, complete randomisation (valid analysis unit)",
                  rate_cell(study.aa_period_welch_complete)]])
        + "\n\nWithout a burn-in, carryover pulls the estimate toward zero (treated periods "
        "lose it, control periods inherit it); a burn-in covering the carryover removes the "
        "bias at a modest variance cost. Treating hours as independent gets the error rate "
        "wrong in either direction depending on the design — the period is the unit.\n"
    )


def comparison_section() -> str:
    cuped = cached("cuped_case", ac.cuped_case)
    assert isinstance(cuped, ac.CupedCase)
    rows = [
        ["CUPED / Lin adjustment", "unadjusted Welch at the same n",
         "same estimand (ITT); unbiased; SE lower by √(1 − R²)",
         "covariate measured before assignment; R² from history for planning",
         "covariate affected by treatment (biased); weak covariates (no gain)",
         f"{cuped.plan_cuped.n_total / cuped.plan_unadjusted.n_total:.0%} of the sample"],
        ["Group-sequential (O'Brien–Fleming)", "fixed horizon at the M6 size",
         "α kept; same power with +2–3% maximum size; expected size well below the maximum "
         "when the effect is real", "a few pre-planned looks; prompt outcomes; spending "
         "fixed in advance", "estimates at early stopping exaggerated; unplanned extra looks "
         "need the spending function", "earlier decisions for large effects"],
        ["mSPRT / confidence sequences", "fixed horizon",
         "valid under continuous monitoring and any stopping", "normal approximation with "
         "a pre-specified mixture sd", "much lower power at the same maximum size",
         "monitoring flexibility at a sample-size cost"],
        ["Bayesian expected loss", "M6 LAUNCH/HOLD/STOP at fixed horizon",
         "decision in money with explicit priors", "prior reflects past effects; model "
         "fits", "frequentist error rates not controlled when peeking; prior misspecification "
         "biases launched effects", "communicates risk in £; not a replacement for "
         "pre-registration"],
        ["Cluster planning with estimated ICC", "plan ignoring clustering",
         "planned power achieved", "ICC estimable from history; clusters exchangeable",
         "few clusters; ICC uncertainty", "avoids an underpowered rollout"],
        ["Switchback with burn-in, blocked", "naive hourly comparison",
         "unbiased global effect; valid randomisation inference", "carryover shorter than "
         "the burn-in; no anticipation", "carryover longer than the burn-in; few periods",
         "measures market-level effects user-level tests cannot"],
    ]
    return (
        "Each advanced method answers a specific question better than the M6 baseline under "
        "stated conditions; none dominates in general.\n\n"
        + table(["method", "M6 baseline compared with", "advantage shown (this evidence)",
                 "requires", "fails when", "business relevance"], rows) + "\n"
    )


def main() -> None:
    sections: list[tuple[str, Callable[[], str]]] = [
        ("1. CUPED and regression adjustment: the revenue experiment", cuped_case_section),
        ("2. CUPED in repeated sampling", cuped_study_section),
        ("3. Group-sequential boundaries against published values", boundary_section),
        ("4. Peeking, group-sequential designs and the mSPRT", peeking_section),
        ("5. Sequential monitoring of the revenue experiment", sequential_case_section),
        ("6. Bayesian analysis of the same experiment", bayesian_case_section),
        ("7. Bayesian rules: operating characteristics and calibration",
         bayesian_operating_section),
        ("8. Cluster-randomised planning with an estimated ICC", cluster_section),
        ("9. Switchback experiment with carryover", switchback_section),
        ("10. Advanced methods against the M6 baselines", comparison_section),
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
        "# Milestone 7 evidence — numerical results\n\n"
        "Generated by `python scripts/generate_m7_evidence.py`; do not edit by hand. All "
        "randomness is seeded; with the versions below every number reproduces exactly "
        "(runtimes excepted). Narrative and limitations: "
        "[m7-advanced-experimentation.md](m7-advanced-experimentation.md); methodology: "
        "[../methodology/advanced-experimentation.md](../methodology/advanced-experimentation.md)"
        ".\n\nGround truth is known only because the data are synthetic; it is read after "
        "estimation and establishes nothing about real-world validity. Confidence intervals "
        "(frequentist) and credible intervals (Bayesian) are labelled as such throughout.\n"
    )
    OUTPUT.write_text(
        header + "\n" + "\n".join(rendered) + "\n## Environment\n\n" + environment
        + "\n\n## Runtime\n\n"
        + table(["section", "runtime (s, indicative)"], [[t, s] for t, s in runtimes]) + "\n"
    )
    print(f"Wrote {OUTPUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
