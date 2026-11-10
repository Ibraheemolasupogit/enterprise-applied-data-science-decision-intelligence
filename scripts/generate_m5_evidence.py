"""Generate the Milestone 5 (observational causal inference) evidence report.

Writes ``docs/evidence/m5-observational-causal-inference-results.md``. All
randomness is seeded; with the recorded dependency versions every number
reproduces exactly (runtimes excepted). Prose numbers are computed, not typed.

Usage:
    python scripts/generate_m5_evidence.py
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

from decision_intelligence.causal_inference import did
from decision_intelligence.causal_inference import observational_cases as oc
from decision_intelligence.causal_inference import synthetic_control as sc
from decision_intelligence.causal_inference.campaign import ReplicateSummary
from decision_intelligence.common import seeded_generator
from decision_intelligence.data.rollout_panel import generate_rollout_panel
from decision_intelligence.data.synthetic_control_panel import generate_synthetic_control_panel
from decision_intelligence.data.threshold_assignment import generate_threshold_data

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "docs" / "evidence" / "m5-observational-causal-inference-results.md"
DID_REPS, FEW_CLUSTER_REPS, SC_REPS, RDD_REPS = 150, 500, 60, 300
PRETREND_REPS, BAND_REPS, CS_INFERENCE_REPS, ORACLE_REPS = 1000, 400, 100, 300


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


def ci(lower: float | None, upper: float | None) -> str:
    return "—" if lower is None or upper is None else f"[{lower:.3f}, {upper:.3f}]"


def summary_row(s: ReplicateSummary, coverage: bool = True) -> list[object]:
    bias = f"{s.bias:+.3f} ± {s.bias_mc_se:.3f}"
    row: list[object] = [s.estimator, s.truth, s.mean_estimate, bias,
                         s.rmse]
    if coverage:
        row += [s.mean_reported_se, s.empirical_sd,
                f"{s.coverage:.3f} [{s.coverage_ci_lower:.3f}, {s.coverage_ci_upper:.3f}]"]
    return row


SUMMARY_HEADERS = ["estimator", "truth (mean)", "mean estimate", "bias ± MC SE", "RMSE"]
COVERAGE_HEADERS = [*SUMMARY_HEADERS, "mean reported SE", "empirical SD",
                    "95% CI coverage [Wilson]"]


def rate_cell(r: object) -> str:
    if not isinstance(r, oc.RejectionRate):
        return "—"
    return f"{r.rate:.3f} [{r.ci_lower:.3f}, {r.ci_upper:.3f}]"


# ---------------------------------------------------------------------------
# Difference-in-differences
# ---------------------------------------------------------------------------


def did_canonical_section() -> str:
    data = generate_rollout_panel()
    a = oc.analyse_rollout(data, rng=seeded_generator(5001))
    cs, wild = a.staggered, a.twfe_wild_bootstrap
    cohorts = data.panel.groupby("cohort")["unit_id"].nunique().to_dict()
    rows = [
        ["static TWFE", a.twfe.estimate, ci(a.twfe.ci_lower, a.twfe.ci_upper), a.twfe.se_method,
         a.twfe.covers(a.true_overall_att)],
        ["static TWFE, wild cluster bootstrap", wild.coefficient,
         ci(wild.ci_lower, wild.ci_upper), f"WCR Rademacher, {wild.n_draws} draws",
         wild.ci_lower is not None and wild.ci_upper is not None
         and wild.ci_lower <= a.true_overall_att <= wild.ci_upper],
        ["Callaway–Sant'Anna (never-treated)", cs.overall.estimate,
         ci(cs.overall.ci_lower, cs.overall.ci_upper), cs.overall.se_method,
         cs.overall.covers(a.true_overall_att)],
    ]
    two_by_two = did.two_period_did(data.panel, 8, 12)
    truth_2x2 = data.ground_truth.att_by_cohort_period[(9, 12)]
    event_rows = []
    dynamic = {c.event_time: c for c in a.event_study_twfe.coefficients}
    for e in cs.event_study:
        if -6 <= e.event_time <= 8:
            dyn = dynamic.get(e.event_time)
            event_rows.append([
                e.event_time, a.true_event_time_att.get(e.event_time, 0.0), e.estimate,
                ci(e.ci_lower, e.ci_upper), ci(e.band_lower, e.band_upper),
                "—" if dyn is None else dyn.estimate,
            ])
    pre = cs.pretrend
    dyn_pre = a.event_study_twfe.pretrend
    return (
        f"Canonical rollout (`RolloutConfig()`): {data.config.n_units} markets × "
        f"{data.config.n_periods} periods; cohorts (first treated period: markets) {cohorts}; "
        f"AR(1) errors ρ = {data.config.ar1_rho}. Effects grow with exposure and are larger for "
        "early adopters. Timing: cohort g = first treated period, e = t − g, e = −1 is the "
        "reference.\n\n"
        f"**Estimand:** overall ATT = average effect over treated market-weeks = "
        f"**{a.true_overall_att:.3f}** (known by construction; read after estimation).\n\n"
        + table(["estimator", "estimate", "95% CI", "inference", "CI covers truth"], rows)
        + f"\n\n**Why TWFE misses.** If `Y_it = α_i + λ_t + τ_it·D_it + ε_it` in a balanced "
        "panel, the TWFE coefficient is exactly Σ w_it·τ_it over treated market-periods plus "
        "a mean-zero noise term, with weights fixed by the adoption dates and panel length "
        f"alone (not by the effects). Here {a.twfe_negative_weight_share:.0%} of treated "
        f"market-periods (a count share) carry negative weight, totalling "
        f"{a.twfe_negative_weight_sum:.3f} of weight mass; with the true effects the weights "
        f"imply {a.twfe_implied_estimand:.3f}, a *different estimand* from the overall ATT "
        f"{a.true_overall_att:.3f} — later adopters are compared with already-treated early "
        "adopters whose effects are still growing. The decomposition does not hold under "
        "non-parallel trends or spillovers, which add terms of their own. The wild bootstrap "
        "fixes the *uncertainty* of the coefficient, not its *meaning*.\n\n"
        f"2×2 DiD (cohort 9 vs not-yet/never treated, periods 8 → 12): {two_by_two.estimate:.3f} "
        f"{ci(two_by_two.ci_lower, two_by_two.ci_upper)}; truth {truth_2x2:.3f}.\n\n"
        "**Event study** (CS event-time ATT with a universal base; dynamic TWFE for "
        "comparison):\n\n"
        + table(["e", "truth", "CS estimate", "pointwise 95% CI", "sup-t band (nominal 95%)",
                 "dynamic TWFE"], event_rows)
        + "\n\nPointwise intervals are for one event time at a time; reading the whole path "
        "requires the simultaneous band (section 2b)."
        + (f"\n\nPre-trend test, CS leads {list(pre.leads_tested)}: W = {pre.statistic:.2f}, "
           f"{pre.reference}, p = {pre.p_value:.3f}." if pre else "")
        + (f" Dynamic TWFE leads: p = {dyn_pre.p_value:.4f} — a spurious 'pre-trend' produced "
           "by heterogeneous effects leaking across event times under staggered adoption (Sun & "
           "Abraham 2021), although trends are parallel by construction." if dyn_pre else "")
        + "\n\nA non-rejected pre-trend test is consistent with parallel trends; it does not "
        "establish them."
    )


def did_monte_carlo_section() -> str:
    rows, pretrend_rows = [], []
    for name, (template, anticipation, valid) in oc.DID_DESIGNS.items():
        result = oc.did_monte_carlo(template, anticipation=anticipation,
                                    n_replications=DID_REPS)
        for key in ("callaway_santanna", "twfe"):
            s = result[key]
            assert isinstance(s, ReplicateSummary)
            rows.append([name, *summary_row(s)])
        pretrend_rows.append([
            name, valid,
            rate_cell(result["cs_pretrend_rejection"]),
            rate_cell(result["twfe_pretrend_rejection"]),
        ])
    return (
        f"{DID_REPS} independent panels per design (seeds 50,000+). Truth = each panel's "
        "overall ATT. Bias ± Monte Carlo SE; coverage with a Wilson interval.\n\n"
        + table(["design", *COVERAGE_HEADERS], rows)
        + "\n\nPre-trend tests: rejection rate at 5% [Wilson]. Under valid designs this is "
        "the test's size; under violations it is its power.\n\n"
        + table(["design", "valid design", "CS pre-trend test", "dynamic TWFE pre-trend test"],
                pretrend_rows)
        + f"\n\nWith {DID_REPS} panels these rates carry Monte Carlo error of about ±2 points; "
        "the precise size of the CS test is in section 2b (1,000 panels per design)."
    )


def few_cluster_section() -> str:
    rows = []
    for g, k in ((10, 5), (10, 2), (10, 1), (20, 10), (40, 20)):
        for r in oc.few_cluster_size_study(g, k, n_replications=FEW_CLUSTER_REPS):
            rows.append([g, k, r.procedure, f"{r.rate:.3f} [{r.ci_lower:.3f}, {r.ci_upper:.3f}]"])
    return (
        f"Null effect, common timing, 12 periods, AR(1) ρ = 0.7; {FEW_CLUSTER_REPS} panels per "
        "row. A valid 5% test rejects 5% of the time.\n\n"
        + table(["clusters G", "treated clusters", "procedure", "rejection rate [Wilson]"], rows)
        + "\n\n**Unreliable regimes.** With one or two treated clusters CR1 over-rejects "
        "severely (with one treated cluster it rejects a true null about half the time) and the "
        "wild bootstrap becomes so conservative that it essentially never rejects — it has "
        "almost no power. No cluster-based method is trustworthy there. `wild_cluster_bootstrap` "
        "reports the number of clusters carrying the tested regressor and warns at three or "
        "fewer."
    )


def did_calibration_section() -> str:
    pretrend_rows = []
    for n_units in (20, 40, 60, 120):
        for r in oc.pretrend_calibration_study(n_units, n_replications=PRETREND_REPS):
            pretrend_rows.append([r.procedure, rate_cell(r)])
    band_rows = [[r.procedure, rate_cell(r)]
                 for r in oc.event_band_coverage_study(n_replications=BAND_REPS)]
    inference_rows = [summary_row(s) for s in oc.cs_inference_comparison(
        n_replications=CS_INFERENCE_REPS)]
    return (
        f"**Pre-trend test size** under valid parallel trends ({PRETREND_REPS} panels per "
        "row; 4 cohorts of equal size, so 20 units = 5 per group). Nominal 5%.\n\n"
        + table(["test", "rejection rate [Wilson]"], pretrend_rows)
        + "\n\nThe naive all-leads χ² test is badly over-sized; the windowed finite-cluster F "
        "test is calibrated from 40 units (10 per group) but still over-sized with 5 per group, "
        "where `staggered_did` now warns.\n\n"
        f"**Event-study multiplicity** ({BAND_REPS} panels): share in which every event time "
        "in the reported window (6 leads and all lags) is covered at once.\n\n"
        + table(["intervals", "observed joint coverage [Monte Carlo 95% Wilson interval]"],
                band_rows)
        + "\n\nPointwise intervals must not be read jointly. The sup-t band (Montiel Olea & "
        "Plagborg-Møller 2019) is a *nominal* 95% simultaneous band; in this design it exhibits "
        "**undercoverage** — its observed joint coverage is below 0.95 and nominal coverage is "
        "not guaranteed. The shortfall is plausibly concentrated at the long horizons, which "
        "rest on one cohort of 15 markets (not separately verified).\n\n"
        f"**Callaway–Sant'Anna inference** ({CS_INFERENCE_REPS} panels): the unit bootstrap "
        "randomises cohort sizes that are fixed by design, so it is conservative here.\n\n"
        + table(COVERAGE_HEADERS, inference_rows)
    )


# ---------------------------------------------------------------------------
# Synthetic control
# ---------------------------------------------------------------------------


def sc_canonical_section() -> str:
    data = generate_synthetic_control_panel()
    a = oc.analyse_synthetic_control(data)
    f = a.fit
    weight_rows = [[u, w, data.ground_truth.true_donor_weights.get(u, 0.0)]
                   for u, w in sorted(f.weights.items(), key=lambda kv: -kv[1])]
    loo = list(a.leave_one_out.values())
    path_rows = [[t, data.ground_truth.effect_by_period[t], gap]
                 for t, gap in f.post_gaps.items()]
    invalid_rows = []
    for name, (template, _) in oc.SC_DESIGNS.items():
        fit = sc.fit_synthetic_control(
            generate_synthetic_control_panel(template).panel, 0, template.n_pre + 1
        )
        invalid_rows.append([name, fit.average_post_gap, fit.relative_pre_fit,
                             fit.share_pre_outside_donor_range, "; ".join(fit.warnings) or "none"])
    pa, pe = a.placebo_all, a.placebo_excluding_poor_fits
    strict = sc.placebo_in_space(data.panel, 0, data.config.n_pre + 1, exclusion_multiple=2.0)
    return (
        f"Canonical design: 1 treated region, {data.config.n_donors} donors, "
        f"{data.config.n_pre} pre- and {data.config.n_post} post-treatment months, factor model "
        f"with {data.config.n_factors} factors. **Estimand:** the treated region's effect path; "
        f"its post-period average is **{a.true_average_post_effect:.3f}**.\n\n"
        f"Average post-period gap {f.average_post_gap:.3f}; pre-period RMSPE "
        f"{f.pre_rmspe:.3f} ({f.relative_pre_fit:.2f} × pre-period SD); effective donors "
        f"{f.effective_number_of_donors:.1f}. Naive DiD against the donor average: "
        f"{a.naive_did:.3f}. Leave-one-out range of the average gap: "
        f"[{min(loo):.3f}, {max(loo):.3f}]. In-time placebo (fake start 10 periods early): "
        f"{a.in_time_placebo.average_post_gap:.3f}.\n\n"
        + table(["period", "true effect", "estimated gap"], path_rows)
        + "\n\nEstimated vs generating donor weights (the generating weights are one of many "
        "combinations that reproduce the treated loadings — weights are not identified, the "
        "counterfactual path is what matters):\n\n"
        + table(["donor", "estimated weight", "generating weight"], weight_rows)
        + f"\n\nIn-space placebos: treated post/pre RMSPE ratio {pa.treated_ratio:.2f}, rank "
        f"{pa.rank} of {pa.n_compared} → p = {pa.permutation_p_value:.3f} (minimum attainable "
        f"{pa.minimum_attainable_p:.3f}); excluding placebos with pre-period MSPE > "
        f"{pe.exclusion_multiple:g}× the treated unit's ({len(pe.excluded_for_poor_fit)} "
        f"excluded): rank {pe.rank} of {pe.n_compared}; at the stricter 2× MSPE rule of Abadie "
        f"et al. (2010) ({len(strict.excluded_for_poor_fit)} excluded): rank {strict.rank} of "
        f"{strict.n_compared}. "
        "This is an exact permutation p-value only if the treated region had been chosen at "
        "random; here it is a descriptive rank.\n\n"
        "Single fits of each design:\n\n"
        + table(["design", "average gap", "pre-fit / SD", "share pre outside donor range",
                 "warnings"], invalid_rows)
    )


def sc_monte_carlo_section() -> str:
    rows, other = [], []
    for name, (template, valid) in oc.SC_DESIGNS.items():
        result = oc.sc_monte_carlo(template, n_replications=SC_REPS)
        for key in ("synthetic_control", "naive_did"):
            s = result[key]
            assert isinstance(s, ReplicateSummary)
            rows.append([name, *summary_row(s, coverage=False)])
        other.append([name, valid, result["share_flagged"],
                      rate_cell(result["placebo_rank_one"])])
    return (
        f"{SC_REPS} panels per design. Synthetic control has no sampling-based interval, so "
        "bias and RMSE are reported; 'placebo rank 1' is how often the treated region has "
        "the most extreme post/pre RMSPE ratio among 31 units (nominal p = 1/31 ≈ 0.032).\n\n"
        + table(["design", *SUMMARY_HEADERS], rows)
        + "\n\n"
        + table(["design", "valid design", "share of fits flagged by diagnostics",
                 "placebo rank-1 rate [Wilson]"], other)
    )


# ---------------------------------------------------------------------------
# Regression discontinuity
# ---------------------------------------------------------------------------


def rdd_canonical_section() -> str:
    data = generate_threshold_data()
    a = oc.analyse_threshold(data)
    z = (a.robust.estimate - a.true_tau_at_cutoff) / a.robust.standard_error
    sens = a.bandwidth_sensitivity
    sens_rows = [[r.bandwidth, r.local_linear, ci(r.conventional_ci_lower, r.conventional_ci_upper),
                  r.bias_corrected, ci(r.robust_ci_lower, r.robust_ci_upper), r.n_left, r.n_right]
                 for r in sens.itertuples()]
    placebo_rows = [[r.placebo_cutoff, r.estimate, ci(r.ci_lower, r.ci_upper), r.covers_zero]
                    for r in a.placebo_cutoffs.itertuples()]
    poly_rows = [
        [p, e.estimate, ci(e.ci_lower, e.ci_upper)] for p, e in a.global_polynomials.items()
    ]
    cov = a.covariates.iloc[0]
    invalid_rows = []
    for name, (template, _) in oc.RDD_DESIGNS.items():
        b = oc.analyse_threshold(generate_threshold_data(template))
        donut_cells = [f"{d.estimate:.2f} {ci(d.ci_lower, d.ci_upper)}" for d in b.donuts.values()]
        invalid_rows.append([
            name, b.robust.estimate, ci(b.robust.ci_lower, b.robust.ci_upper),
            b.mccrary.p_value, b.binomial.p_value, bool(b.covariates["covers_zero"].iloc[0]),
            *donut_cells,
        ])
    return (
        f"Canonical loyalty-tier data: {data.config.n_customers} customers, sharp cutoff at "
        f"score {data.config.cutoff:g}. **Estimand:** τ(c), the effect for customers at the "
        f"cutoff = **{a.true_tau_at_cutoff:.3f}**. For contrast (not targeted by RDD): "
        f"population-average effect {a.true_population_ate:.3f}; average effect among tier "
        f"members {a.true_att:.3f}.\n\n"
        f"IK bandwidth h = {a.bandwidth:.2f}. Local linear (conventional): "
        f"{a.conventional.estimate:.3f} "
        f"{ci(a.conventional.ci_lower, a.conventional.ci_upper)}. Robust bias-corrected: "
        f"{a.robust.estimate:.3f} {ci(a.robust.ci_lower, a.robust.ci_upper)}.\n\n"
        f"**This canonical draw is unlucky:** the robust estimate is {z:.1f} standard errors from "
        "the truth. An independent `rdrobust` fit on the same data also excludes it (development "
        "cross-check in m5-observational-causal-inference.md). The seed is reported as "
        "generated; across 300 independent draws (section 7) the same procedure is unbiased with "
        "near-nominal coverage.\n\n"
        + table(["h", "local linear", "conventional CI", "bias-corrected", "robust CI", "n left",
                 "n right"], sens_rows)
        + "\n\nManipulation checks: McCrary log-density jump "
        f"{a.mccrary.log_density_difference:.3f}"
        f" (p = {a.mccrary.p_value:.3f}); {a.binomial.method}, p = {a.binomial.p_value:.3f}. "
        f"Pre-determined covariate (tenure) jump {cov.jump:.3f} {ci(cov.ci_lower, cov.ci_upper)}."
        f"\n\nPlacebo cutoffs (robust inference, data from one side only):\n\n"
        + table(["placebo cutoff", "estimate", "robust CI", "covers 0"], placebo_rows)
        + "\n\nGlobal polynomial fits (all data; shown as the cautionary comparison):\n\n"
        + table(["order", "estimate", "95% CI"], poly_rows)
        + "\n\nEach design on its canonical seed:\n\n"
        + table(["design", "robust estimate", "robust CI", "McCrary p", "binomial p",
                 "tenure continuous", "donut ±0.5", f"donut ±{data.config.manipulation_window:g}"],
                invalid_rows)
        + "\n\nDonut estimates extrapolate across the excluded hole with a local quadratic, so "
        "they are imprecise and, with a wide hole, unstable even in the valid design; a donut "
        "removes the bunching just above the cutoff but not the deficit of sorted units below "
        "it, so it does not repair manipulation here."
    )


def rdd_monte_carlo_section() -> str:
    rows, rates = [], []
    for name, (template, valid) in oc.RDD_DESIGNS.items():
        result = oc.rdd_monte_carlo(template, n_replications=RDD_REPS)
        for key in ("conventional", "robust"):
            s = result[key]
            assert isinstance(s, ReplicateSummary)
            rows.append([name, *summary_row(s)])
        quartic = result["global_quartic"]
        assert isinstance(quartic, ReplicateSummary)
        rows.append([name, *summary_row(quartic, coverage=False), "—", "—", "—"])
        rates.append([name, valid, rate_cell(result["mccrary_rejection"]),
                      rate_cell(result["covariate_rejection"])])
    placebo = oc.placebo_cutoff_study(n_replications=RDD_REPS)
    oracle, ik_mean, ik_sd = oc.bandwidth_oracle_study(n_replications=ORACLE_REPS)
    best = oracle.loc[oracle["mse"].idxmin()]
    oracle_rows = [[r.bandwidth, r.bias, r.variance, r.mse] for r in oracle.itertuples()]
    return (
        f"{RDD_REPS} independent datasets per design; truth τ(c) = 3.\n\n"
        + table(["design", *COVERAGE_HEADERS], rows)
        + "\n\nValidity checks: rejection rate at 5% [Wilson].\n\n"
        + table(["design", "valid design", "McCrary density test", "covariate continuity"], rates)
        + "\n\nFalse discoveries at a placebo cutoff (50, where the outcome curve bends "
        "sharply) — conventional vs robust inference:\n\n"
        + table(["procedure", "rejection rate [Wilson]"],
                [[r.procedure, f"{r.rate:.3f} [{r.ci_lower:.3f}, {r.ci_upper:.3f}]"]
                 for r in placebo])
        + f"\n\n**Bandwidth check against the oracle** ({ORACLE_REPS} datasets; possible "
        "only because τ(c) is known): Monte Carlo MSE of the local-linear estimate at fixed "
        "bandwidths.\n\n"
        + table(["h", "bias", "variance", "MSE"], oracle_rows)
        + f"\n\nThe lowest MSE on this grid is at h = {best.bandwidth:g} (MSE {best.mse:.4f}; "
        "the curve is flat at the top of the grid, so the optimum may lie beyond it — the "
        "outcome is nearly linear around this cutoff). The IK selector averages "
        f"{ik_mean:.2f} (SD {ik_sd:.2f}): smaller than the oracle, trading some MSE for less "
        "smoothing bias; IK's regularisation term deliberately prevents very large bandwidths "
        "when the estimated curvature difference is small. (The PyPI "
        "`rdd` package's IK implementation returns about half this bandwidth because it floors "
        "the third-derivative estimate at m₃² ≥ 0.01, an ad-hoc and scale-dependent constant; "
        "development check.)"
    )


def main() -> None:
    sections: list[tuple[str, Callable[[], str]]] = [
        ("1. Difference-in-differences on the canonical rollout", did_canonical_section),
        ("2. Difference-in-differences: repeated sampling, valid and invalid designs",
         did_monte_carlo_section),
        ("2b. Pre-trend size, event-study multiplicity and CS inference",
         did_calibration_section),
        ("3. Few-cluster inference", few_cluster_section),
        ("4. Synthetic control on the canonical design", sc_canonical_section),
        ("5. Synthetic control: repeated sampling", sc_monte_carlo_section),
        ("6. Regression discontinuity on the canonical data", rdd_canonical_section),
        ("7. Regression discontinuity: repeated sampling", rdd_monte_carlo_section),
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
        "# Milestone 5 evidence — numerical results\n\n"
        "Generated by `python scripts/generate_m5_evidence.py`; do not edit by hand. All "
        "randomness is seeded; with the versions below every number reproduces exactly "
        "(runtimes excepted). Narrative and limitations: "
        "[m5-observational-causal-inference.md](m5-observational-causal-inference.md); "
        "methodology: [../methodology/observational-causal-inference.md]"
        "(../methodology/observational-causal-inference.md).\n\n"
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
