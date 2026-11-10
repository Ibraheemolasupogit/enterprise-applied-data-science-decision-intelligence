"""Generate the Milestone 3 (statistical foundations) evidence report.

Runs every M3 applied analysis on the default M2 synthetic build and every
repeated-sampling study (type I error, coverage, power, CLT, bias–variance,
missing data, dependence), then writes the numbers to
``docs/evidence/m3-statistical-foundations-results.md``.

All randomness flows from fixed seeds, so every number in the report is
reproduced exactly by rerunning this script with the same dependency
versions (recorded in the report). Only the runtime column varies by
machine.

Usage:
    python scripts/generate_m3_evidence.py
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

from decision_intelligence.common import bootstrap_ci, seeded_generator
from decision_intelligence.data import SyntheticDataConfig, build_synthetic_dataset
from decision_intelligence.statistics import applied_analyses as aa
from decision_intelligence.statistics import hypothesis_tests as ht
from decision_intelligence.statistics.bias_variance import (
    bias_variance_decomposition,
    polynomial_fitter,
    ridge_polynomial_fitter,
)
from decision_intelligence.statistics.estimators import clt_diagnostics, t_interval_for_mean
from decision_intelligence.statistics.linear_regression import fit_ols
from decision_intelligence.statistics.missing_data import compare_missing_data_strategies
from decision_intelligence.statistics.power import power_two_proportions, power_two_sample_t
from decision_intelligence.statistics.simulation import (
    SimulatedRate,
    simulate_interval_coverage,
    simulate_rejection_rate,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "docs" / "evidence" / "m3-statistical-foundations-results.md"


def table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    def fmt(value: object) -> str:
        if isinstance(value, float):
            if math.isinf(value):
                return "inf" if value > 0 else "-inf"
            return f"{value:.4g}" if abs(value) >= 1e-3 or value == 0 else f"{value:.2e}"
        return str(value)

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def rate_cell(rate: SimulatedRate) -> str:
    return f"{rate.rate:.3f} (± {rate.standard_error:.3f})"


class Report:
    def __init__(self) -> None:
        self.sections: list[str] = []
        self.runtimes: list[tuple[str, float]] = []

    def section(self, title: str, build: Callable[[], str]) -> None:
        start = time.perf_counter()
        body = build()
        self.runtimes.append((title, time.perf_counter() - start))
        self.sections.append(f"## {title}\n\n{body}\n")


# ---------------------------------------------------------------------------
# Applied analyses on the M2 data
# ---------------------------------------------------------------------------

CONFIG = SyntheticDataConfig()


def experiment_section() -> str:
    result = aa.experiment_conversion_analysis(
        build_synthetic_dataset(CONFIG), rng=seeded_generator(301)
    )
    effects = result.effect_sizes
    rows = [
        ["control conversion", result.control_rate, *result.control_rate_ci, "Wilson"],
        ["treatment conversion", result.treatment_rate, *result.treatment_rate_ci, "Wilson"],
        [
            "risk difference",
            result.test.estimate,
            result.test.ci_lower,
            result.test.ci_upper,
            "Newcombe hybrid score",
        ],
        [
            "risk difference",
            result.bootstrap_difference.estimate,
            result.bootstrap_difference.ci_lower,
            result.bootstrap_difference.ci_upper,
            "stratified percentile bootstrap (2,000)",
        ],
        [
            "risk ratio",
            effects.risk_ratio.estimate,
            effects.risk_ratio.ci_lower,
            effects.risk_ratio.ci_upper,
            "Katz log",
        ],
        [
            "odds ratio",
            effects.odds_ratio.estimate,
            effects.odds_ratio.ci_lower,
            effects.odds_ratio.ci_upper,
            "Woolf log",
        ],
    ]
    return (
        "**Question.** Did the recommendation feature raise 14-day conversion? Assignment was "
        "randomised (M2 `experiments.py`), so the difference in proportions estimates the "
        "average treatment effect on conversion for the eligible population.\n\n"
        f"Arms: control n = {result.n_control}, treatment n = {result.n_treatment}.\n\n"
        + table(["quantity", "estimate", "95% lower", "95% upper", "method"], rows)
        + f"\n\nPooled two-proportion z = {result.test.statistic:.3f}, two-sided p = "
        f"{result.test.p_value:.3f}; Cohen's h = {effects.cohens_h.estimate:.3f}.\n\n"
        f"**Design check (not post-hoc power).** For the pre-specified minimum effect of "
        f"{result.minimum_effect:.2f} (absolute) at the observed control rate, these arm sizes "
        f"give power {result.design_power:.2f}; 80% power needs about "
        f"{result.required_n_per_arm} customers per arm.\n\n"
        f"**Known-truth check (evaluation only).** The design value is "
        f"{result.true_effect:.2f}; the 95% interval "
        f"{'covers' if result.ci_covers_true_effect else 'does NOT cover'} it.\n\n"
        "**Interpretation.** The point estimate is close to the true effect, but the interval "
        "includes zero: this experiment is underpowered for an effect of this size. "
        "'Not significant' here means 'not enough data', not 'no effect' — the correct "
        "decision-relevant output is the interval, and the design check shows the sample "
        "needed. (Experimentation design and sequential decision rules are M6/M7 scope.)"
    )


def revenue_section() -> str:
    dataset = build_synthetic_dataset(CONFIG)
    result = aa.revenue_analysis(dataset, rng=seeded_generator(302))
    s = result.revenue_summary
    customers_with_revenue = dataset.tables["transactions"]["customer_id"].nunique()
    summary_rows = [
        ["n", s.n],
        ["mean", s.mean],
        ["median", s.median],
        ["std", s.std],
        ["skewness (G1)", s.skewness],
        ["excess kurtosis (G2)", s.excess_kurtosis],
        ["IQR", s.iqr],
        ["share with zero revenue", 1 - customers_with_revenue / s.n],
    ]
    interval_rows = [
        ["mean", "Student t", *result.mean_t_interval],
        [
            "mean",
            "BCa bootstrap (2,000)",
            result.mean_bootstrap.ci_lower,
            result.mean_bootstrap.ci_upper,
        ],
        [
            "median",
            "percentile bootstrap (2,000)",
            result.median_bootstrap.ci_lower,
            result.median_bootstrap.ci_upper,
        ],
    ]
    welch = result.power_vs_casual_welch
    mwu = result.power_vs_casual_mann_whitney
    comparison_rows = [
        ["Welch t", "difference in means", welch.estimate, welch.ci_lower, welch.ci_upper,
         welch.p_value],
        ["Mann–Whitney U", "P(power > casual)", mwu.estimate, None, None, mwu.p_value],
        [
            "Hedges' g",
            "standardised difference",
            result.power_vs_casual_hedges_g.estimate,
            result.power_vs_casual_hedges_g.ci_lower,
            result.power_vs_casual_hedges_g.ci_upper,
            None,
        ],
        [
            "Cliff's delta",
            "P(X>Y) − P(X<Y)",
            result.power_vs_casual_cliffs_delta.estimate,
            None,
            None,
            None,
        ],
    ]
    return (
        "**Question.** What is a typical customer's total billed revenue over the horizon, "
        "and do power-segment customers bill more than casual ones?\n\n"
        + table(["statistic", "value"], summary_rows)
        + "\n\n"
        + table(["target", "interval method", "95% lower", "95% upper"], interval_rows)
        + "\n\n"
        + table(["method", "estimand", "estimate", "95% lower", "95% upper", "p"],
                [[("—" if v is None else v) for v in row] for row in comparison_rows])
        + "\n\n**Interpretation.** Revenue is strongly right-skewed with a point mass at zero "
        "(customers active for less than one 30-day billing period), so mean and median "
        "answer different questions; with n = 2,000 "
        "the t and BCa intervals for the mean nearly coincide (the CLT is adequate at this "
        "n), which would not be true at small n (see the coverage study below). The "
        "segment comparison is associational: segments differ in plan price, tenure and "
        "latent value simultaneously, so the gap must not be read as the effect of moving a "
        "customer between segments."
    )


def retention_section() -> str:
    result = aa.retention_model_analysis(
        build_synthetic_dataset(CONFIG), rng=seeded_generator(303)
    )
    frame = result.model.summary_frame()
    coefficient_rows = [
        [name, row.estimate, row.std_error, row.odds_ratio, row.ci_lower, row.ci_upper,
         row.p_value]
        for name, row in frame.iterrows()
    ]
    d = result.model_test_diagnostics
    evaluation_rows = [
        ["Brier score", d.brier_score, result.baseline_test_brier],
        ["log loss", d.log_loss, result.baseline_test_log_loss],
        ["ROC AUC", d.auc, 0.5],
        ["calibration-in-the-large", d.calibration_in_the_large, "≈0 by construction"],
        ["calibration slope", d.calibration_slope, "undefined"],
        ["Hosmer–Lemeshow p (10 bins)", d.hosmer_lemeshow_p_value, "—"],
    ]
    return (
        "**Question.** Which observable, pre-outcome customer attributes are associated with "
        "90-day retention, and does a simple model predict retention better than the base "
        "rate on customers it has not seen?\n\n"
        f"Random customer split: {result.n_train} train / {result.n_test} test. Features: "
        f"segment (ref. casual), price sensitivity, acquisition channel (ref. organic). "
        "Excluded: `latent_value` (hidden generator driver), campaign treatment (confounded "
        "exposure — M4/M5), income (has injected missingness).\n\n"
        + table(["feature", "log-odds", "SE", "odds ratio", "95% lower (log-odds)",
                 "95% upper (log-odds)", "p"], coefficient_rows)
        + "\n\nHeld-out evaluation against the base-rate baseline (training prevalence for "
        "everyone):\n\n"
        + table(["metric", "logistic model", "base-rate baseline"], evaluation_rows)
        + "\n\n**Interpretation.** The model is better than the baseline on every proper "
        "scoring rule, with modest discrimination. Coefficients are conditional associations, "
        "not effects: e.g. the power-segment odds ratio also carries segment differences in "
        "the unobserved `latent_value` (which raises baseline retention) and in exposure to "
        "the retention campaign, which is targeted on that value. A calibration slope above 1 "
        "on held-out data indicates predictions slightly *under*-dispersed (not overfit), "
        "although with 600 test customers the slope itself is imprecise."
    )


def clustered_section() -> str:
    result = aa.clustered_transaction_analysis(build_synthetic_dataset(CONFIG))
    rows = [
        [name, coef, classical, hc3, cluster, cluster / classical]
        for name, coef, classical, hc3, cluster in zip(
            result.feature_names,
            result.coefficients,
            result.classical_se,
            result.hc3_se,
            result.cluster_se,
            strict=True,
        )
    ]
    return (
        "**Question.** How strongly are transaction amounts associated with customer segment "
        "and price sensitivity — and how uncertain is that, given each customer contributes "
        "many transactions?\n\n"
        f"{result.n_transactions} transactions from {result.n_customers} customers.\n\n"
        + table(["feature", "coefficient", "classical SE", "HC3 SE",
                 "cluster (customer) SE", "cluster / classical"], rows)
        + "\n\n**Interpretation.** Point estimates are identical across estimators; only the "
        "uncertainty differs. Transactions of one customer share a plan price, so errors are "
        "strongly correlated within customer and the effective sample size is closer to the "
        "number of customers than the number of rows. Classical and HC3 SEs (which assume "
        "independent rows) understate uncertainty by a factor of about two; reporting them "
        "would overstate precision. The repeated-sampling study below confirms the "
        "consequence for interval coverage."
    )


def missingness_section() -> str:
    r = aa.income_missingness_analysis(CONFIG, rng=seeded_generator(304))
    rows = [
        ["full data (twin build, no injected missingness)", r.full_mean_log_income, "—"],
        [f"M2 complete case (MCAR, {r.m2_missing_rate:.1%} missing)", r.m2_complete_case_mean,
         r.m2_complete_case_mean - r.full_mean_log_income],
        [f"imposed MAR complete case ({r.mar_missing_rate:.1%} missing)",
         r.mar_complete_case_mean, r.mar_complete_case_mean - r.full_mean_log_income],
        ["imposed MAR, regression imputation on price sensitivity",
         r.mar_regression_imputation_mean,
         r.mar_regression_imputation_mean - r.full_mean_log_income],
    ]
    return (
        "**Question.** Can mean log household income be estimated from a column with missing "
        "values?\n\n"
        + table(["estimate", "mean log income", "error vs full data"], rows)
        + f"\n\nStandard error of the M2 complete-case mean: {r.m2_complete_case_se:.4f}. "
        f"Correlation of log income with price sensitivity: {r.covariate_correlation:.3f}.\n\n"
        "**Interpretation.** M2 injects missingness completely at random, so dropping the "
        "missing rows costs precision but not validity (error well within one SE). When "
        "missingness instead depends on an observed covariate associated with income (MAR), "
        "the complete-case mean is biased by several standard errors; regressing on that "
        "covariate removes most of the bias. Whether real missingness is MCAR, MAR or MNAR "
        "cannot be tested from the observed data — this check is possible only because the "
        "data are synthetic."
    )


# ---------------------------------------------------------------------------
# Repeated-sampling studies
# ---------------------------------------------------------------------------


def type_one_error_section() -> str:
    def pair(n_x: int, n_y: int, sd_x: float):
        return lambda rng: (rng.normal(0, sd_x, n_x), rng.normal(0, 1, n_y))

    def proportions(rng: np.random.Generator) -> tuple[np.ndarray, ...]:
        return (np.array([rng.binomial(500, 0.2), rng.binomial(500, 0.2)], dtype=float),)

    studies: list[tuple[str, str, Callable[..., float], Callable, int]] = [
        ("Welch t", "N(0,3²), n=10 vs N(0,1), n=40",
         lambda x, y: ht.two_sample_t_test(x, y).p_value, pair(10, 40, 3.0), 10000),
        ("Student pooled t", "N(0,3²), n=10 vs N(0,1), n=40",
         lambda x, y: ht.two_sample_t_test(x, y, equal_var=True).p_value, pair(10, 40, 3.0),
         10000),
        ("Welch t", "N(0,1), n=20 vs N(0,1), n=20",
         lambda x, y: ht.two_sample_t_test(x, y).p_value, pair(20, 20, 1.0), 10000),
        ("Mann–Whitney U", "Exp(1), n=15 vs Exp(1), n=15",
         lambda x, y: ht.mann_whitney_u(x, y).p_value,
         lambda rng: (rng.exponential(1, 15), rng.exponential(1, 15)), 10000),
        ("two-proportion z", "Bin(500, 0.2) vs Bin(500, 0.2)",
         lambda c: ht.two_proportion_z_test(int(c[0]), 500, int(c[1]), 500).p_value,
         proportions, 10000),
    ]
    rows = []
    for index, (method, scenario, test, generate, reps) in enumerate(studies):
        rate = simulate_rejection_rate(
            generate, test, alpha=0.05, n_replications=reps, rng=seeded_generator(400 + index)
        )
        rows.append([method, scenario, reps, rate_cell(rate)])
    return (
        "Rejection rate at α = 0.05 when H0 is true (Monte Carlo SE in brackets).\n\n"
        + table(["test", "data-generating process (H0 true)", "replications",
                 "type I error"], rows)
        + "\n\n**Interpretation.** Welch's test holds its level under unequal variances; "
        "Student's pooled test does not when the smaller group has the larger variance — the "
        "reason Welch is the default. The exact Mann–Whitney test is slightly conservative "
        "(discrete null distribution)."
    )


def coverage_section() -> str:
    lognormal_mean = math.exp(1.5**2 / 2)
    bootstrap_rng = seeded_generator(509)

    def bootstrap_interval(method: str) -> Callable[[np.ndarray], tuple[float, float]]:
        def compute(x: np.ndarray) -> tuple[float, float]:
            result = bootstrap_ci(x, np.mean, rng=bootstrap_rng, n_resamples=1000,
                                  method=method)  # type: ignore[arg-type]
            return result.ci_lower, result.ci_upper

        return compute

    def wald(successes: float) -> tuple[float, float]:
        p = successes / 40
        half = 1.959964 * math.sqrt(p * (1 - p) / 40)
        return p - half, p + half

    def binomial(rng: np.random.Generator) -> tuple[float]:
        return (float(rng.binomial(40, 0.05)),)

    studies: list[tuple[str, str, Callable, Callable, float, int]] = [
        ("t interval", "N(5, 2²), n=12", t_interval_for_mean,
         lambda rng: (rng.normal(5, 2, 12),), 5.0, 10000),
        ("t interval", "LogNormal(0, 1.5), n=10", t_interval_for_mean,
         lambda rng: (rng.lognormal(0, 1.5, 10),), lognormal_mean, 10000),
        ("t interval", "LogNormal(0, 1.5), n=200", t_interval_for_mean,
         lambda rng: (rng.lognormal(0, 1.5, 200),), lognormal_mean, 10000),
        ("percentile bootstrap", "Exp(mean 2), n=20", bootstrap_interval("percentile"),
         lambda rng: (rng.exponential(2, 20),), 2.0, 1000),
        ("BCa bootstrap", "Exp(mean 2), n=20", bootstrap_interval("bca"),
         lambda rng: (rng.exponential(2, 20),), 2.0, 1000),
        ("t interval", "Exp(mean 2), n=20", t_interval_for_mean,
         lambda rng: (rng.exponential(2, 20),), 2.0, 10000),
        ("Wilson", "Bin(40, 0.05)", lambda s: ht.wilson_interval(int(s), 40), binomial, 0.05,
         10000),
        ("Wald", "Bin(40, 0.05)", wald, binomial, 0.05, 10000),
    ]
    rows = []
    for index, (method, scenario, interval, generate, truth, reps) in enumerate(studies):
        rate = simulate_interval_coverage(
            generate, interval, truth, n_replications=reps, rng=seeded_generator(500 + index)
        )
        rows.append([method, scenario, reps, rate_cell(rate)])
    return (
        "Proportion of nominal 95% intervals containing the true parameter.\n\n"
        + table(["interval", "data-generating process", "replications", "coverage"], rows)
        + "\n\n**Interpretation.** The t interval is exact for normal data. For heavy-tailed "
        "data it under-covers badly at small n and improves only slowly with n — log-normal "
        "(σ = 1.5) data still give well under 95% at n = 200, because the sample mean's "
        "skewness decays like 1/√n (section 9). The bootstrap does not rescue small samples: "
        "at n = 20 from an exponential population both bootstrap intervals under-cover, BCa "
        "slightly less than percentile, and neither beats the plain t interval here. Wald "
        "binomial intervals fail near 0; Wilson intervals hold their level."
    )


def power_section() -> str:
    rows = []
    designs = [(0.2, 50), (0.5, 40), (0.5, 64), (0.8, 20)]
    for index, (effect, n) in enumerate(designs):
        simulated = simulate_rejection_rate(
            lambda rng, e=effect, size=n: (rng.normal(e, 1, size), rng.normal(0, 1, size)),
            lambda x, y: ht.two_sample_t_test(x, y, equal_var=True).p_value,
            alpha=0.05,
            n_replications=5000,
            rng=seeded_generator(600 + index),
        )
        rows.append([f"two-sample t, d={effect}", n, power_two_sample_t(effect, n),
                     rate_cell(simulated)])
    for index, (p1, p2, n) in enumerate([(0.14, 0.10, 800), (0.146, 0.106, 566)]):
        simulated = simulate_rejection_rate(
            lambda rng, a=p1, b=p2, size=n: (
                np.array([rng.binomial(size, a), rng.binomial(size, b)], dtype=float),
            ),
            lambda c, size=n: ht.two_proportion_z_test(
                int(c[0]), size, int(c[1]), size
            ).p_value,
            alpha=0.05,
            n_replications=5000,
            rng=seeded_generator(610 + index),
        )
        rows.append([f"two proportions, {p1} vs {p2}", n, power_two_proportions(p1, p2, n),
                     rate_cell(simulated)])
    return (
        "Analytic power (noncentral t; normal approximation for proportions) versus the "
        "simulated rejection rate of the implemented test (5,000 replications each, α = 0.05, "
        "two-sided).\n\n"
        + table(["design", "n per group", "analytic power", "simulated power"], rows)
        + "\n\nThe last row approximates the M2 experiment (control ≈ 0.106, +0.04, ≈566 per "
        "arm): roughly a coin-flip chance of detecting the true effect."
    )


def clt_section() -> str:
    rows = []
    for name, sampler, sd, skew in [
        ("Exponential(1)", lambda rng, n: rng.exponential(1.0, n), 1.0, 2.0),
        ("LogNormal(0, 1)", lambda rng, n: rng.lognormal(0.0, 1.0, n),
         math.sqrt((math.e - 1) * math.e), (math.e + 2) * math.sqrt(math.e - 1)),
    ]:
        mean = 1.0 if name.startswith("Exp") else math.exp(0.5)
        for diagnostic in clt_diagnostics(sampler, [2, 10, 30, 100, 1000], mean, sd, 5000,
                                          seeded_generator(700), population_skewness=skew):
            rows.append([name, diagnostic.sample_size, diagnostic.sd_of_means,
                         diagnostic.theoretical_standard_error, diagnostic.skewness_of_means,
                         diagnostic.theoretical_skewness_of_means,
                         diagnostic.ks_distance_to_normal])
    return (
        "Sampling distribution of the mean (5,000 replications per n).\n\n"
        + table(["population", "n", "SD of means", "σ/√n", "skewness of means",
                 "theory γ/√n", "KS distance to N(0,1)"], rows)
        + "\n\n**Interpretation.** The standard error is σ/√n at every n — that part of the "
        "theory is exact. Normality is the asymptotic part: skewness decays like γ/√n, so the "
        "log-normal population (γ ≈ 6.2) needs far larger samples than the exponential "
        "(γ = 2) before normal-theory inference is accurate. 'n ≥ 30' is not a universal rule. "
        "Simulation noise: with 5,000 replications the KS distance cannot fall much below "
        "≈0.012 even for an exactly normal statistic, and sample skewness has a Monte Carlo SE "
        "of ≈0.035 (larger for heavy-tailed populations), so differences at n = 100 vs 1,000 "
        "are within noise."
    )


def bias_variance_section() -> str:
    def truth(x: np.ndarray) -> np.ndarray:
        return np.sin(2.5 * x)

    models = {f"polynomial degree {d}": polynomial_fitter(d) for d in (1, 3, 5, 8)}
    models.update({f"ridge degree 8, λ={lam:g}": ridge_polynomial_fitter(8, lam)
                   for lam in (0.01, 1.0, 10.0)})
    rows = [
        [r.model, r.bias_squared, r.variance, r.noise, r.noise + r.bias_squared + r.variance,
         r.expected_test_mse, r.mean_training_mse]
        for r in bias_variance_decomposition(
            truth, models, noise_sd=0.3, train_size=30,
            x_test=np.linspace(-0.9, 0.9, 25), n_replications=1000, rng=seeded_generator(800),
        )
    ]
    return (
        "f(x) = sin(2.5x), x ~ U(−1, 1), σ = 0.3, n = 30 training points, 1,000 training "
        "sets; test grid on [−0.9, 0.9].\n\n"
        + table(["model", "bias²", "variance", "noise σ²", "sum", "simulated test MSE",
                 "mean training MSE"], rows)
        + "\n\n**Interpretation.** Test error is U-shaped in complexity while training error "
        "falls monotonically — training error cannot select complexity. The simulated test "
        "MSE matches bias² + variance + σ² up to Monte Carlo error, an independent check of "
        "the decomposition. Ridge reaches a similar trade-off continuously at fixed degree. "
        "Variance is not strictly monotone in complexity under a random design: the "
        "misspecified straight line also varies with where training points fall."
    )


def missing_data_section() -> str:
    rows = []
    for index, mechanism in enumerate(("MCAR", "MAR", "MNAR")):
        for r in compare_missing_data_strategies(
            mechanism=mechanism, missing_rate=0.4, sample_size=200, n_replications=2000,
            rng=seeded_generator(900 + index),
        ):
            rows.append([mechanism, r.strategy.replace("_", " "), r.bias,
                         r.bias_standard_error, r.naive_se_ratio, r.coverage])
    return (
        "(X, Y) bivariate normal, ρ = 0.6, n = 200, 40% of Y missing, target E[Y] = 0; 2,000 "
        "replications. Naive SE = sd/√(rows analysed).\n\n"
        + table(["mechanism", "strategy", "bias", "MC SE of bias", "naive SE / true SD",
                 "95% CI coverage"], rows)
        + "\n\n**Interpretation.** Under MCAR, complete-case analysis is unbiased and its SE "
        "honest. Under MAR it is biased; conditioning on the observed driver (regression "
        "imputation) removes the bias, but single imputation treats imputed values as data, "
        "so naive SEs are too small and coverage falls. Mean imputation is the worst of both: "
        "biased under MAR and overconfident everywhere. Under MNAR no strategy based only on "
        "observed data is unbiased."
    )


def dependence_section() -> str:
    n_clusters, per_cluster, beta = 60, 10, 0.5

    def generate(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cluster = np.repeat(np.arange(n_clusters), per_cluster)
        x = rng.normal(size=n_clusters)[cluster] + 0.3 * rng.normal(size=cluster.size)
        error = rng.normal(size=n_clusters)[cluster] + 0.5 * rng.normal(size=cluster.size)
        return x, beta * x + error, cluster.astype(float)

    rows = []
    for covariance in ("classical", "HC3", "cluster"):
        def interval(x: np.ndarray, y: np.ndarray, cluster: np.ndarray,
                     cov: str = covariance) -> tuple[float, float]:
            fit = fit_ols(x, y, covariance=cov,
                          clusters=cluster if cov == "cluster" else None)
            return float(fit.ci_lower[1]), float(fit.ci_upper[1])

        coverage = simulate_interval_coverage(
            generate, interval, beta, n_replications=2000, rng=seeded_generator(1000)
        )
        rows.append([covariance, rate_cell(coverage)])
    return (
        "60 clusters × 10 observations; the regressor and the error both have a cluster "
        "component (intra-cluster error correlation 0.8). Coverage of nominal 95% intervals "
        "for the slope, 2,000 replications (same datasets for every estimator).\n\n"
        + table(["covariance estimator", "coverage"], rows)
        + "\n\n**Interpretation.** Heteroskedasticity-robust (HC3) SEs do not address "
        "dependence; only the cluster-robust estimator restores near-nominal coverage."
    )


def main() -> None:
    report = Report()
    report.section("1. Randomised experiment: conversion lift (M2 data)", experiment_section)
    report.section("2. Revenue distribution and segment comparison (M2 data)", revenue_section)
    report.section("3. Retention model vs base-rate baseline (M2 data)", retention_section)
    report.section("4. Dependence: clustered transactions (M2 data)", clustered_section)
    report.section("5. Missing income: known-truth check (M2 data)", missingness_section)
    report.section("6. Type I error under H0 (simulation)", type_one_error_section)
    report.section("7. Confidence-interval coverage (simulation)", coverage_section)
    report.section("8. Analytic vs simulated power (simulation)", power_section)
    report.section("9. Central Limit Theorem diagnostics (simulation)", clt_section)
    report.section("10. Bias–variance decomposition (simulation)", bias_variance_section)
    report.section("11. Missing-data strategies by mechanism (simulation)", missing_data_section)
    report.section("12. Dependence and interval coverage (simulation)", dependence_section)

    environment = table(
        ["component", "version"],
        [["python", platform.python_version()], ["numpy", np.__version__],
         ["scipy", scipy.__version__], ["pandas", pd.__version__]],
    )
    runtimes = table(["section", "runtime (s, indicative)"],
                     [[title, seconds] for title, seconds in report.runtimes])
    header = (
        "# Milestone 3 evidence — numerical results\n\n"
        "Generated by `python scripts/generate_m3_evidence.py`. Do not edit by hand: rerun "
        "the script. All randomness is seeded; with the dependency versions below every "
        "number is reproduced exactly (runtimes excepted). Narrative, methodology and "
        "limitations: [m3-statistical-foundations.md](m3-statistical-foundations.md) and "
        "[../methodology/statistical-foundations.md](../methodology/statistical-foundations.md)."
        "\n\nM2 data: default `SyntheticDataConfig()` (seed 20260101, 2,000 customers). "
        "Ground truth is read only after estimation, to check recovery.\n\n"
        "Monte Carlo results are shown as `estimate (± Monte Carlo standard error)`.\n"
    )
    OUTPUT.write_text(
        header
        + "\n"
        + "\n".join(report.sections)
        + "\n## Environment\n\n"
        + environment
        + "\n\n## Runtime\n\n"
        + runtimes
        + "\n"
    )
    print(f"Wrote {OUTPUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
