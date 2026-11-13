"""M7 case studies and repeated-sampling studies (advanced experimentation).

Every quantitative claim in the M7 evidence comes from a function here. Each
advanced method is compared with the M6 fixed-horizon baseline *on the same
data and assumptions*: the unadjusted Welch / two-proportion analysis at the
planned sample size. Ground truth is read only after estimation.

* CUPED / regression adjustment — :func:`cuped_case`, :func:`cuped_study`;
* sequential monitoring — :func:`boundary_reference`, :func:`peeking_study`,
  :func:`sequential_case`;
* Bayesian analysis — :func:`bayesian_case`, :func:`bayesian_fixed_truth_study`,
  :func:`bayesian_calibration_study`;
* designs — :func:`cluster_case`, :func:`cluster_power_study`,
  :func:`switchback_case`, :func:`switchback_study`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import stats

from decision_intelligence.causal_inference.campaign import ReplicateSummary
from decision_intelligence.causal_inference.observational_cases import (
    RejectionRate,
    rejection_rate,
    summarise_replicates,
)
from decision_intelligence.common.seeding import spawn_generators
from decision_intelligence.data.advanced_experiment_scenarios import (
    ASSIGNMENT_TIME,
    LEAKY_WINDOW_OVERLAP_DAYS,
    OUTCOME_WINDOW_DAYS,
    CupedScenarioConfig,
    SwitchbackConfig,
    generate_cuped_population,
    generate_switchback_market,
    realise_cuped_experiment,
    realise_switchback,
)
from decision_intelligence.data.experiment_scenarios import (
    ExperimentScenarioConfig,
    generate_experiment_population,
    realise_experiment,
)
from decision_intelligence.experimentation import bayesian as ba
from decision_intelligence.experimentation import designs as de
from decision_intelligence.experimentation import sequential as sq
from decision_intelligence.experimentation import variance_reduction as vr
from decision_intelligence.experimentation.analysis import (
    MetricResult,
    analyse_binary,
    analyse_continuous,
    clustered_difference,
    stratified_difference,
)
from decision_intelligence.experimentation.assignment import (
    cluster_randomisation,
    complete_randomisation,
)
from decision_intelligence.experimentation.decision import Recommendation, recommend
from decision_intelligence.experimentation.planning import ExperimentPlan, plan_binary
from decision_intelligence.experimentation.validity import CheckResult
from decision_intelligence.statistics.multiple_testing import rejects

ALPHA = 0.05

# Illustrative NexaMarket economics (assumptions stated in the evidence).
# Revenue: effect in £ per user per 14 days; 40,000 active users x 26 fortnights; 30% margin.
REVENUE_UNITS_PER_YEAR = 40_000 * 26
REVENUE_MARGIN = 0.30
REVENUE_ANNUAL_COST = 100_000.0
# Conversion: as in M6 — 50,000 eligible customers a year, £30 per conversion, £30,000 a year.
CONVERSION_UNITS_PER_YEAR = 50_000
CONVERSION_VALUE = 30.0
CONVERSION_ANNUAL_COST = 30_000.0
LOSS_TOLERANCE = 10_000.0
DAILY_ELIGIBLE = 20_000 / 28  # the scenario's accrual rate


def _streams(seed: int) -> tuple[np.random.Generator, np.random.Generator]:
    first, second = spawn_generators(seed, 2)
    return first, second


def _assign(n: int, seed: int) -> np.ndarray:
    labels = complete_randomisation(n, rng=_streams(seed)[0])
    return np.asarray(labels == "treatment", dtype=np.int64)


def revenue_break_even() -> float:
    return REVENUE_ANNUAL_COST / (REVENUE_UNITS_PER_YEAR * REVENUE_MARGIN)


# ---------------------------------------------------------------------------
# 1. CUPED / regression adjustment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CupedCase:
    truth_ate: float
    truth_correlation: float
    covariate_checks: tuple[CheckResult, ...]
    results: dict[str, vr.VarianceReductionResult]
    """Keys: 'CUPED', 'Lin', 'INVALID post-treatment', 'INVALID leaky window',
    'CUPED binary'."""
    historical_r_squared: float
    plan_unadjusted: ExperimentPlan
    plan_cuped: ExperimentPlan
    m6_recommendation: Recommendation
    cuped_recommendation: Recommendation
    pre_period_difference: float
    """Treatment - control difference in mean pre_revenue (chance imbalance; pre-treatment)."""


def cuped_case(config: CupedScenarioConfig | None = None, *, seed: int = 11) -> CupedCase:
    config = config or CupedScenarioConfig()
    population = generate_cuped_population(config)
    frame = realise_cuped_experiment(population, _assign(config.n_units, seed))
    z = (frame["arm"] == "treatment").astype(float).to_numpy()

    pre_end = ASSIGNMENT_TIME
    leaky_end = ASSIGNMENT_TIME + pd.Timedelta(LEAKY_WINDOW_OVERLAP_DAYS, unit="D")
    sessions_end = ASSIGNMENT_TIME + pd.Timedelta(OUTCOME_WINDOW_DAYS, unit="D")
    checks = (
        vr.check_adjustment_covariate("pre_revenue", frame["pre_revenue"], z,
                                      window_end=pre_end, assignment_time=ASSIGNMENT_TIME),
        vr.check_adjustment_covariate("in_experiment_sessions", frame["in_experiment_sessions"],
                                      z, window_end=sessions_end,
                                      assignment_time=ASSIGNMENT_TIME),
        vr.check_adjustment_covariate("pre_revenue_leaky", frame["pre_revenue_leaky"], z,
                                      window_end=leaky_end, assignment_time=ASSIGNMENT_TIME),
    )
    y, arm = frame["revenue"], frame["arm"]
    results = {
        "CUPED": vr.cuped(y, arm, frame[["pre_revenue"]]),
        "Lin": vr.regression_adjustment(y, arm, frame[["pre_revenue"]]),
        "INVALID post-treatment": vr.cuped(y, arm, frame[["in_experiment_sessions"]]),
        "INVALID leaky window": vr.cuped(y, arm, frame[["pre_revenue_leaky"]]),
        "CUPED binary": vr.cuped(frame["converted"], arm, frame[["pre_converted"]],
                                 metric="conversion"),
    }
    # Planning: R^2 and SD from a *historical* cohort (different users, no treatment).
    historical = generate_cuped_population(replace(config, seed=config.seed + 1))
    h_units, h_pot = historical.units, historical.potential
    h_frame = pd.DataFrame({"y": h_pot["revenue0"], "x": h_units["pre_revenue"]})
    x, _, _ = vr.prepare_covariates(h_frame[["x"]])
    fit = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), h_frame["y"], rcond=None)[0]
    residual = h_frame["y"].to_numpy() - np.column_stack([np.ones(len(x)), x]) @ fit
    r2 = 1.0 - float(residual.var() / h_frame["y"].var(ddof=0))
    plain, adjusted = vr.plan_with_covariate(
        float(h_pot["revenue0"].mean()), float(h_pot["revenue0"].std(ddof=1)),
        historical_r_squared=r2, mde_absolute=revenue_break_even(),
        daily_eligible=DAILY_ELIGIBLE,
    )
    m6 = recommend(results["CUPED"].unadjusted, minimum_worthwhile_effect=revenue_break_even())
    cuped_metric = _as_metric(results["CUPED"])
    adjusted_rec = recommend(cuped_metric, minimum_worthwhile_effect=revenue_break_even())
    pre, treated = frame["pre_revenue"], frame["arm"] == "treatment"
    imbalance = float(pre[treated].mean() - pre[~treated].mean())
    return CupedCase(population.truth.ate_revenue, population.truth.pre_post_correlation,
                     checks, results, r2, plain, adjusted, m6, adjusted_rec, imbalance)


def _as_metric(result: vr.VarianceReductionResult) -> MetricResult:
    """The adjusted estimate in the M6 MetricResult shape (for the M6 decision rule)."""
    e = result.effect
    p = float(2 * stats.norm.sf(abs(e.estimate / e.standard_error)))
    return replace(result.unadjusted, effect=e, p_value=p, test=result.method)


@dataclass(frozen=True)
class CupedStudy:
    summaries: list[ReplicateSummary]
    power: list[RejectionRate]
    aa_size: list[RejectionRate]
    mean_correlation: float
    mean_variance_reduction_cuped: float
    mean_variance_reduction_lin: float
    mean_r_squared: float
    binary_variance_reduction: float


def cuped_study(*, n_replications: int, first_seed: int = 70_000,
                config: CupedScenarioConfig | None = None) -> CupedStudy:
    """Bias, SE, coverage and power of unadjusted (M6), CUPED, Lin and the two invalid
    adjustments; A/A size with no effect."""
    config = config or CupedScenarioConfig(n_units=8_000)
    names = ["unadjusted (M6 Welch)", "CUPED (pre_revenue)", "Lin (pre_revenue)",
             "INVALID: CUPED on in-experiment sessions", "INVALID: CUPED on leaky window",
             "conversion unadjusted (M6)", "conversion CUPED (pre_converted)"]
    est: dict[str, list[float]] = {n: [] for n in names}
    truth: dict[str, list[float]] = {n: [] for n in names}
    cover: dict[str, list[bool]] = {n: [] for n in names}
    ses: dict[str, list[float]] = {n: [] for n in names}
    reject: dict[str, list[bool]] = {n: [] for n in names}
    aa: dict[str, list[bool]] = {n: [] for n in names[:3]}
    rho, vr_c, vr_l, r2s, vr_b = [], [], [], [], []
    for r in range(n_replications):
        population = generate_cuped_population(replace(config, seed=first_seed + r))
        frame = realise_cuped_experiment(population, _assign(config.n_units, first_seed + r))
        y, arm = frame["revenue"], frame["arm"]
        c = vr.cuped(y, arm, frame[["pre_revenue"]])
        lin = vr.regression_adjustment(y, arm, frame[["pre_revenue"]])
        post = vr.cuped(y, arm, frame[["in_experiment_sessions"]])
        leak = vr.cuped(y, arm, frame[["pre_revenue_leaky"]])
        cb = vr.cuped(frame["converted"], arm, frame[["pre_converted"]], metric="conversion")
        revenue_truth, conversion_truth = (population.truth.ate_revenue,
                                           population.truth.ate_conversion)
        rows = [(c.unadjusted.effect, revenue_truth), (c.effect, revenue_truth),
                (lin.effect, revenue_truth), (post.effect, revenue_truth),
                (leak.effect, revenue_truth), (cb.unadjusted.effect, conversion_truth),
                (cb.effect, conversion_truth)]
        for name, (e, t) in zip(names, rows, strict=True):
            est[name].append(e.estimate)
            truth[name].append(t)
            cover[name].append(e.covers(t))
            ses[name].append(e.standard_error)
            reject[name].append(not e.covers(0.0))
        rho.append(population.truth.pre_post_correlation)
        vr_c.append(c.realised_variance_reduction)
        vr_l.append(lin.realised_variance_reduction)
        r2s.append(c.r_squared)
        vr_b.append(cb.realised_variance_reduction)
        null = generate_cuped_population(replace(config, seed=first_seed + r, revenue_effect=0.0))
        nf = realise_cuped_experiment(null, _assign(config.n_units, first_seed + r))
        nc = vr.cuped(nf["revenue"], nf["arm"], nf[["pre_revenue"]])
        nl = vr.regression_adjustment(nf["revenue"], nf["arm"], nf[["pre_revenue"]])
        for name, e in zip(names[:3], (nc.unadjusted.effect, nc.effect, nl.effect), strict=True):
            aa[name].append(not e.covers(0.0))
    return CupedStudy(
        summaries=[summarise_replicates(n, est[n], truth[n], cover[n], ses[n]) for n in names],
        power=[rejection_rate(n, reject[n]) for n in names],
        aa_size=[rejection_rate(n, aa[n]) for n in names[:3]],
        mean_correlation=float(np.mean(rho)),
        mean_variance_reduction_cuped=float(np.mean(vr_c)),
        mean_variance_reduction_lin=float(np.mean(vr_l)),
        mean_r_squared=float(np.mean(r2s)),
        binary_variance_reduction=float(np.mean(vr_b)),
    )


# ---------------------------------------------------------------------------
# 2. Sequential monitoring
# ---------------------------------------------------------------------------

# Published references: Jennison & Turnbull (2000) Table 2.3 (constants, two-sided 0.05) and
# Tables 2.2/2.4 (R, power 0.8); Armitage, McPherson & Rowe (1969) (repeated 5% tests);
# gsDesign / rpact / ldbounds Lan-DeMets O'Brien-Fleming, five equal looks.
PUBLISHED_POCOCK = {2: 2.178, 3: 2.289, 4: 2.361, 5: 2.413, 10: 2.555}
PUBLISHED_OBF = {2: 1.977, 3: 2.004, 4: 2.024, 5: 2.040, 10: 2.087}
PUBLISHED_PEEKING = {2: 0.083, 3: 0.107, 4: 0.126, 5: 0.142, 10: 0.193, 20: 0.248}
PUBLISHED_LD_OBF_5 = (4.877, 3.357, 2.680, 2.290, 2.031)
PUBLISHED_R = {("pocock", 5): 1.229, ("obrien-fleming", 5): 1.028}


@dataclass(frozen=True)
class BoundaryReference:
    rows: list[tuple[str, float, float]]
    """(quantity, computed, published)."""
    mvn_check: list[tuple[str, float, float]]
    """(quantity, recursive integration, SciPy multivariate normal)."""


def boundary_reference() -> BoundaryReference:
    rows: list[tuple[str, float, float]] = []
    for k, value in PUBLISHED_POCOCK.items():
        d = sq.constant_boundaries(np.arange(1, k + 1) / k, shape="pocock")
        rows.append((f"Pocock constant, K = {k}", d.boundaries[0], value))
    for k, value in PUBLISHED_OBF.items():
        d = sq.constant_boundaries(np.arange(1, k + 1) / k, shape="obrien-fleming")
        rows.append((f"O'Brien–Fleming constant, K = {k}", d.boundaries[-1], value))
    ld = sq.spending_boundaries(np.arange(1, 6) / 5)
    for i, value in enumerate(PUBLISHED_LD_OBF_5):
        rows.append((f"Lan–DeMets O'Brien–Fleming, K = 5, look {i + 1}", ld.boundaries[i], value))
    for k, value in PUBLISHED_PEEKING.items():
        d = sq.naive_peeking_design(np.arange(1, k + 1) / k)
        rows.append((f"type I error of {k} repeated 5% tests", d.cumulative_alpha[-1], value))
    for (shape, k), value in PUBLISHED_R.items():
        d = sq.constant_boundaries(np.arange(1, k + 1) / k, shape=shape)  # type: ignore[arg-type]
        rows.append((f"max sample size inflation R, {shape}, K = {k}",
                     sq.maximum_information_inflation(d), value))
    # Independent check: rectangle probabilities from SciPy's multivariate normal (Genz).
    mvn: list[tuple[str, float, float]] = []
    fractions = np.arange(1, 6) / 5
    cov = np.sqrt(np.minimum.outer(fractions, fractions) / np.maximum.outer(fractions, fractions))
    drift = sq.fixed_sample_drift()
    for label, design in (("Lan–DeMets O'Brien–Fleming", ld),
                          ("Lan–DeMets Pocock", sq.spending_boundaries(fractions,
                                                                      family="pocock"))):
        b = np.asarray(design.boundaries)
        for name, mean in (("size", np.zeros(5)), ("power at fixed-sample drift",
                                                     drift * np.sqrt(fractions))):
            # No ``rng`` argument: it is not accepted by SciPy 1.13, the declared minimum. Genz
            # integration error (~1e-7) is far below the 4 significant digits reported.
            inside = stats.multivariate_normal.cdf(b, mean=mean, cov=cov, lower_limit=-b,
                                                   abseps=1e-9, releps=1e-9, maxpts=5_000_000)
            ours = sq.operating_characteristics(design, 0.0 if name == "size" else drift)
            mvn.append((f"{label}: {name}", ours.rejection_probability, 1.0 - float(inside)))
    return BoundaryReference(rows, mvn)


@dataclass(frozen=True)
class ProcedureSummary:
    procedure: str
    effect: float
    rejection: RejectionRate
    expected_sample_fraction: float
    mean_estimate_when_rejecting: float
    """Mean estimate at stopping among rejecting replicates (truth = ``effect``)."""
    interval_coverage: float
    """Coverage of the interval reported at the stopping (or final) look."""
    interval: str


@dataclass(frozen=True)
class PeekingStudy:
    baseline: float
    design_effect_size: float
    n_max_per_arm: int
    n_looks: int
    summaries: list[ProcedureSummary]
    designs: dict[str, sq.GroupSequentialDesign]


def _cumulative_binary(rng: np.random.Generator, reps: int, block: int, n_blocks: int,
                       p: float) -> np.ndarray:
    """Cumulative successes after each block of ``block`` units (binomial increments)."""
    return np.cumsum(rng.binomial(block, p, size=(reps, n_blocks)), axis=1)


def peeking_study(
    *,
    n_replications: int,
    baseline: float = 0.12,
    effect_size: float = 0.015,
    n_looks: int = 10,
    sub_looks: int = 100,
    effects: Sequence[float] | None = None,
    seed: int = 71_000,
) -> PeekingStudy:
    """Repeated monitoring of a two-arm conversion experiment, i.i.d. arrivals.

    ``n_max`` per arm is the M6 fixed-horizon sample size for ``effect_size`` (alpha 0.05,
    power 0.8). Procedures: the M6 fixed-horizon test at ``n_max`` (baseline), naive peeking at
    every look, Lan–DeMets O'Brien–Fleming and Pocock spending at ``n_looks`` equal looks, and
    the mSPRT checked after every ``n_max / sub_looks`` units per arm (mixture sd =
    ``effect_size``). Statistics use the unpooled SE.
    """
    plan = plan_binary(baseline, mde_absolute=effect_size)
    n_max = int(math.ceil(plan.n_treatment / sub_looks) * sub_looks)
    block = n_max // sub_looks
    look_index = np.arange(1, n_looks + 1) * (sub_looks // n_looks) - 1
    fractions = (look_index + 1) / sub_looks
    designs = {
        "naive peeking (10 looks)": sq.naive_peeking_design(fractions),
        "Lan–DeMets O'Brien–Fleming (10 looks)": sq.spending_boundaries(fractions),
        "Lan–DeMets Pocock (10 looks)": sq.spending_boundaries(fractions, family="pocock"),
    }
    effect_list = list(effects) if effects is not None else [0.0, effect_size, 2 * effect_size]
    summaries: list[ProcedureSummary] = []
    z_crit = float(stats.norm.ppf(1 - ALPHA / 2))
    for e_index, effect in enumerate(effect_list):
        rng = spawn_generators(seed + e_index, 1)[0]
        ct = _cumulative_binary(rng, n_replications, block, sub_looks, baseline + effect)
        cc = _cumulative_binary(rng, n_replications, block, sub_looks, baseline)
        n = block * np.arange(1, sub_looks + 1)
        pt, pc = ct / n, cc / n
        var = (np.maximum(pt * (1 - pt), 1e-12) + np.maximum(pc * (1 - pc), 1e-12)) / n
        estimate = pt - pc
        z = estimate / np.sqrt(var)

        def record(name: str, stop_look: np.ndarray, rejected: np.ndarray,
                   lower: np.ndarray, upper: np.ndarray, interval: str,
                   fraction: np.ndarray, est_at_stop: np.ndarray, effect: float = effect) -> None:
            covered = (lower <= effect) & (effect <= upper)
            mean_rej = float(est_at_stop[rejected].mean()) if rejected.any() else math.nan
            summaries.append(ProcedureSummary(
                name, effect, rejection_rate(name, rejected.tolist()),
                float(fraction.mean()), mean_rej, float(covered.mean()), interval))

        rows = np.arange(n_replications)
        # Fixed horizon (M6 baseline): one test at n_max.
        e_f, s_f = estimate[:, -1], np.sqrt(var[:, -1])
        record("fixed horizon (M6 baseline)", np.full(n_replications, sub_looks - 1),
               np.abs(z[:, -1]) >= z_crit, e_f - z_crit * s_f, e_f + z_crit * s_f,
               "95% CI", np.ones(n_replications), e_f)
        # Group-sequential procedures at the 10 looks.
        z_looks, e_looks, s_looks = (z[:, look_index], estimate[:, look_index],
                                     np.sqrt(var[:, look_index]))
        for name, design in designs.items():
            c = np.asarray(design.boundaries)
            crossed = np.abs(z_looks) >= c
            any_cross = crossed.any(axis=1)
            stop = np.where(any_cross, crossed.argmax(axis=1), n_looks - 1)
            est_stop, se_stop = e_looks[rows, stop], s_looks[rows, stop]
            if name.startswith("naive"):
                lo, hi, label = est_stop - z_crit * se_stop, est_stop + z_crit * se_stop, \
                    "naive 95% CI at stop"
            else:
                lo, hi, label = est_stop - c[stop] * se_stop, est_stop + c[stop] * se_stop, \
                    "repeated CI at stop"
            record(name, stop, any_cross, lo, hi, label, fractions[stop], est_stop)
        # mSPRT: always-valid p-value checked at every sub-look.
        log_lr = sq.msprt_log_likelihood_ratio(estimate, var, mixture_sd=effect_size)
        crossed = log_lr >= math.log(1 / ALPHA)
        any_cross = crossed.any(axis=1)
        stop = np.where(any_cross, crossed.argmax(axis=1), sub_looks - 1)
        est_stop, var_stop = estimate[rows, stop], var[rows, stop]
        bounds = np.array([sq.confidence_sequence(float(a), float(b), mixture_sd=effect_size)
                           for a, b in zip(est_stop, var_stop, strict=True)])
        record("mSPRT (checked 100 times)", stop, any_cross, bounds[:, 0], bounds[:, 1],
               "confidence sequence at stop", (stop + 1) / sub_looks, est_stop)
    return PeekingStudy(baseline, effect_size, n_max, n_looks, summaries, designs)


@dataclass(frozen=True)
class SequentialCaseLook:
    calendar_day: int
    n_analysed: int
    information_fraction: float
    estimate: float
    standard_error: float
    z: float
    boundary: float
    repeated_ci: tuple[float, float]
    naive_p_value: float
    always_valid_p_value: float
    confidence_sequence: tuple[float, float]


@dataclass(frozen=True)
class SequentialCase:
    truth: float
    looks: list[SequentialCaseLook]
    stopped_at_day: int | None
    fixed_horizon_day: int
    fixed_horizon: MetricResult
    monitoring: sq.SequentialMonitoringResult


def sequential_case(config: CupedScenarioConfig | None = None, *, seed: int = 12,
                    look_days: Sequence[int] = (21, 28, 35, 42),
                    mixture_sd: float | None = None) -> SequentialCase:
    """Weekly looks at the revenue experiment, using only users whose 14-day outcome window
    has closed by the look (arrival day <= look day - 14): outcomes are delayed, so early looks
    see the earliest cohorts only. CUPED-adjusted statistics at every look; O'Brien–Fleming
    spending on the observed information fractions; mSPRT alongside."""
    config = config or CupedScenarioConfig()
    population = generate_cuped_population(config)
    frame = realise_cuped_experiment(population, _assign(config.n_units, seed))
    tau = mixture_sd if mixture_sd is not None else revenue_break_even()
    estimates, ses, fractions, sizes = [], [], [], []
    for day in look_days:
        cohort = frame.loc[frame["arrival_day"] <= day - OUTCOME_WINDOW_DAYS]
        result = vr.cuped(cohort["revenue"], cohort["arm"], cohort[["pre_revenue"]])
        estimates.append(result.effect.estimate)
        ses.append(result.effect.standard_error)
        sizes.append(len(cohort))
        fractions.append(len(cohort) / len(frame))
    monitoring = sq.monitor(estimates, ses, fractions)
    p_av = sq.always_valid_p_values(estimates, np.square(ses), mixture_sd=tau)
    looks = []
    running_lo, running_hi = -math.inf, math.inf
    for k, day in enumerate(look_days):
        lo, hi = sq.confidence_sequence(estimates[k], ses[k] ** 2, mixture_sd=tau)
        running_lo, running_hi = max(running_lo, lo), min(running_hi, hi)
        boundary = (monitoring.design.boundaries[k] if k < len(monitoring.design.boundaries)
                    else math.nan)
        looks.append(SequentialCaseLook(
            day, sizes[k], fractions[k], estimates[k], ses[k], estimates[k] / ses[k], boundary,
            sq.repeated_confidence_interval(estimates[k], ses[k], boundary),
            float(2 * stats.norm.sf(abs(estimates[k] / ses[k]))), float(p_av[k]),
            (running_lo, running_hi)))
    stopped_day = look_days[monitoring.stopped_at - 1] if monitoring.stopped_at else None
    final = vr.cuped(frame["revenue"], frame["arm"], frame[["pre_revenue"]])
    return SequentialCase(population.truth.ate_revenue, looks, stopped_day, look_days[-1],
                          _as_metric(final), monitoring)


# ---------------------------------------------------------------------------
# 3. Bayesian analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BayesianSummaryRow:
    prior: str
    posterior_mean: float
    credible_interval: tuple[float, float]
    prob_better: float
    prob_worthwhile: float
    decision: ba.ExpectedLossDecision


@dataclass(frozen=True)
class BayesianCase:
    truth_conversion: float
    truth_revenue: float
    conversion_frequentist: MetricResult
    revenue_frequentist: MetricResult
    conversion_rows: list[BayesianSummaryRow]
    revenue_rows: list[BayesianSummaryRow]
    m6_conversion: Recommendation
    m6_revenue: Recommendation
    flat_prior_check: tuple[float, float]
    """(1 - P(theta > 0 | data) under a vague normal prior, one-sided p-value) — equal for a
    normal model with a flat prior; different meaning."""


def _bayes_row(label: str, posterior: ba.EffectPosterior, *, worthwhile: float,
               units: float, value: float, cost: float) -> BayesianSummaryRow:
    return BayesianSummaryRow(
        label, posterior.mean, ba.credible_interval(posterior),
        ba.probability_above(posterior, 0.0), ba.probability_above(posterior, worthwhile),
        ba.expected_loss_decision(posterior, units_per_year=units, value_per_unit_effect=value,
                                  annual_cost=cost, loss_tolerance=LOSS_TOLERANCE),
    )


def bayesian_case(config: CupedScenarioConfig | None = None, *, seed: int = 13) -> BayesianCase:
    config = config or CupedScenarioConfig()
    population = generate_cuped_population(config)
    frame = realise_cuped_experiment(population, _assign(config.n_units, seed))
    conv = analyse_binary(frame["converted"], frame["arm"])
    rev = analyse_continuous(frame["revenue"], frame["arm"])
    n_t, n_c = conv.n_treatment, conv.n_control
    s_t, s_c = int(round(conv.mean_treatment * n_t)), int(round(conv.mean_control * n_c))
    conversion_break_even = CONVERSION_ANNUAL_COST / (CONVERSION_UNITS_PER_YEAR * CONVERSION_VALUE)
    historical = float(frame["pre_converted"].mean())
    priors = [
        ("uniform Beta(1, 1) in each arm", ba.BetaPrior.uniform()),
        ("sceptical: Beta centred on the historical rate, weight 5,000, both arms",
         ba.BetaPrior.from_mean(historical, 5_000)),
    ]
    conversion_rows = [
        _bayes_row(label, ba.beta_binomial_posterior(s_t, n_t, s_c, n_c, prior_treatment=prior,
                                                     prior_control=prior),
                   worthwhile=conversion_break_even, units=CONVERSION_UNITS_PER_YEAR,
                   value=CONVERSION_VALUE, cost=CONVERSION_ANNUAL_COST)
        for label, prior in priors
    ]
    revenue_priors = [
        ba.NormalPrior(0.0, 100.0, "vague N(0, 100^2)"),
        ba.NormalPrior(0.0, 0.3, "sceptical N(0, 0.3^2): most changes do little"),
        ba.NormalPrior(1.0, 0.3, "optimistic N(1, 0.3^2): the team's expectation"),
    ]
    revenue_rows = [
        _bayes_row(prior.label, ba.normal_effect_posterior(rev.effect.estimate,
                                                           rev.effect.standard_error, prior),
                   worthwhile=revenue_break_even(), units=REVENUE_UNITS_PER_YEAR,
                   value=REVENUE_MARGIN, cost=REVENUE_ANNUAL_COST)
        for prior in revenue_priors
    ]
    vague = ba.normal_effect_posterior(rev.effect.estimate, rev.effect.standard_error,
                                       revenue_priors[0])
    one_sided_p = float(stats.norm.sf(rev.effect.estimate / rev.effect.standard_error))
    return BayesianCase(
        population.truth.ate_conversion, population.truth.ate_revenue, conv, rev,
        conversion_rows, revenue_rows,
        recommend(conv, minimum_worthwhile_effect=conversion_break_even),
        recommend(rev, minimum_worthwhile_effect=revenue_break_even()),
        (1.0 - ba.probability_above(vague, 0.0), one_sided_p),
    )


@dataclass(frozen=True)
class BayesianRuleRate:
    rule: str
    effect: float
    monitoring: str
    launch_rate: RejectionRate


def bayesian_fixed_truth_study(
    *, n_replications: int, baseline: float = 0.12, effect_size: float = 0.015,
    n_looks: int = 10, seed: int = 72_000,
) -> list[BayesianRuleRate]:
    """Frequentist operating characteristics of Bayesian launch rules at fixed true effects.

    Rules (uniform Beta priors, exact posterior): "launch if P(better) > 0.95" and "launch
    if P(better) > 0.975". Fixed horizon (one analysis at the M6 sample size) versus
    checking at each of ``n_looks`` equal looks and launching at the first look the rule fires.
    """
    plan = plan_binary(baseline, mde_absolute=effect_size)
    n_max = int(math.ceil(plan.n_treatment / n_looks) * n_looks)
    block = n_max // n_looks
    out: list[BayesianRuleRate] = []
    thresholds = (0.95, 0.975)
    for e_index, effect in enumerate((0.0, effect_size)):
        rng = spawn_generators(seed + e_index, 1)[0]
        ct = _cumulative_binary(rng, n_replications, block, n_looks, baseline + effect)
        cc = _cumulative_binary(rng, n_replications, block, n_looks, baseline)
        prob = np.empty((n_replications, n_looks))
        for r in range(n_replications):
            for k in range(n_looks):
                n = block * (k + 1)
                post = ba.beta_binomial_posterior(int(ct[r, k]), n, int(cc[r, k]), n)
                prob[r, k] = ba.probability_above(post, 0.0)
        for threshold in thresholds:
            rule = f"launch if P(better) > {threshold}"
            out.append(BayesianRuleRate(rule, effect, "fixed horizon",
                                        rejection_rate(rule, (prob[:, -1] > threshold).tolist())))
            out.append(BayesianRuleRate(rule, effect, f"checked at {n_looks} looks",
                                        rejection_rate(rule,
                                                       (prob > threshold).any(axis=1).tolist())))
    return out


@dataclass(frozen=True)
class CalibrationRow:
    scenario: str
    coverage: RejectionRate
    """Share of replicates whose 95% credible interval contains the true effect."""
    mean_effect_when_launched: float
    mean_posterior_when_launched: float


def bayesian_calibration_study(
    *, n_replications: int, standard_error_final: float = 0.006, prior_sd: float = 0.01,
    n_looks: int = 10, seed: int = 73_000,
) -> list[CalibrationRow]:
    """Bayesian calibration: truths drawn from a distribution; normal model on the estimate.

    The estimate path is a Brownian motion: ``theta_hat_k ~ N(theta, SE_k^2)`` with independent
    increments, ``SE_k^2 = SE_final^2 / t_k``. Scenarios: (a) the prior equals the distribution
    the truths are drawn from, analysed once; (b) the same, stopping at the first look where
    ``P(theta > 0) > 0.95`` (optional stopping); (c) the prior is wrong — truths drawn from a
    distribution 3 times narrower than the prior (most changes do nothing), analysed once.
    """
    rng = spawn_generators(seed, 1)[0]
    t = np.arange(1, n_looks + 1) / n_looks
    var_k = standard_error_final**2 / t
    prior = ba.NormalPrior(0.0, prior_sd, "N(0, prior_sd^2)")
    rows: list[CalibrationRow] = []

    def path(theta: np.ndarray) -> np.ndarray:
        # Score S_k = sum of independent increments; theta_hat_k = S_k / I_k with I_k = 1/var_k.
        info = 1.0 / var_k
        increments = rng.normal(np.outer(theta, np.diff(np.r_[0.0, info])),
                                np.sqrt(np.diff(np.r_[0.0, info])))
        return np.asarray(np.cumsum(increments, axis=1) / info)

    def posterior(est: np.ndarray, var: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        precision = 1 / prior.sd**2 + 1 / var
        return (prior.mean / prior.sd**2 + est / var) / precision, np.sqrt(1 / precision)

    for label, truth_sd, stopping in (
        ("prior correct, analysed once", prior_sd, False),
        ("prior correct, optional stopping at P(theta > 0) > 0.95", prior_sd, True),
        ("prior too wide (truths 3x narrower), analysed once", prior_sd / 3, False),
    ):
        theta = rng.normal(0.0, truth_sd, n_replications)
        est = path(theta)
        mean, sd = posterior(est, np.broadcast_to(var_k, est.shape))
        prob = stats.norm.sf(0.0, mean, sd)
        if stopping:
            fired = prob > 0.95
            stop = np.where(fired.any(axis=1), fired.argmax(axis=1), n_looks - 1)
        else:
            stop = np.full(n_replications, n_looks - 1)
        rows_idx = np.arange(n_replications)
        m, s = mean[rows_idx, stop], sd[rows_idx, stop]
        z = float(stats.norm.ppf(0.975))
        covered = (m - z * s <= theta) & (theta <= m + z * s)
        launched = prob[rows_idx, stop] > 0.95
        rows.append(CalibrationRow(
            label, rejection_rate(label, covered.tolist()),
            float(theta[launched].mean()) if launched.any() else math.nan,
            float(m[launched].mean()) if launched.any() else math.nan,
        ))
    return rows


# ---------------------------------------------------------------------------
# 4. Cluster-randomised planning
# ---------------------------------------------------------------------------

CLUSTER_CONFIG = ExperimentScenarioConfig(n_units=12_000, n_clusters=120, cluster_sd=0.4)


@dataclass(frozen=True)
class ClusterCase:
    icc: de.ICCEstimate
    mean_cluster_size: float
    baseline: float
    target_effect: float
    plan: de.ClusterPlan
    naive_plan_power: float
    """Planned (approximate) power of the naive plan given the estimated ICC."""
    power_at_icc_bounds: tuple[float, float]


def cluster_case(*, seed: int = 14, mean_cluster_size: int = 100) -> ClusterCase:
    """Estimate the ICC on a historical (pre-experiment) cohort of the same cities and plan."""
    historical = generate_experiment_population(replace(CLUSTER_CONFIG, seed=CLUSTER_CONFIG.seed
                                                        + 1))
    icc = de.estimate_icc(historical.potential["y0"], historical.units["cluster_id"],
                          rng=_streams(seed)[0])
    baseline = float(historical.potential["y0"].mean())
    target = historical.truth.ate_conversion
    plan = de.plan_cluster_randomised(baseline, mde_absolute=target,
                                      mean_cluster_size=mean_cluster_size,
                                      icc=icc.icc_for_planning,
                                      cluster_size_cv=icc.cluster_size_cv)
    naive_power = de.cluster_power(baseline, target, clusters_per_arm=plan.naive_clusters_per_arm,
                                   mean_cluster_size=mean_cluster_size, icc=icc.icc_for_planning,
                                   cluster_size_cv=icc.cluster_size_cv)
    assert icc.ci_lower is not None and icc.ci_upper is not None
    bounds = tuple(
        de.cluster_power(baseline, target, clusters_per_arm=plan.clusters_per_arm,
                         mean_cluster_size=mean_cluster_size, icc=max(0.0, value),
                         cluster_size_cv=icc.cluster_size_cv)
        for value in (icc.ci_upper, icc.ci_lower)
    )
    return ClusterCase(icc, mean_cluster_size, baseline, target, plan, naive_power,
                       (bounds[0], bounds[1]))


@dataclass(frozen=True)
class ClusterPowerRow:
    design: str
    clusters_per_arm: int
    planned_power: float
    cr1_power: RejectionRate
    naive_unit_level_rejection: RejectionRate
    cr1_coverage: float


def cluster_power_study(case: ClusterCase, *, n_replications: int, first_seed: int = 74_000
                        ) -> list[ClusterPowerRow]:
    """Simulated power of the CR1 analysis (M6) at the planned and the naive cluster counts."""
    rows: list[ClusterPowerRow] = []
    for label, g in (("planned with estimated ICC", case.plan.clusters_per_arm),
                     ("naive plan (ICC ignored)", case.plan.naive_clusters_per_arm)):
        cr1, naive, covered = [], [], []
        for r in range(n_replications):
            config = replace(CLUSTER_CONFIG, seed=first_seed + r, n_clusters=2 * g,
                             n_units=int(2 * g * case.mean_cluster_size))
            population = generate_experiment_population(config)
            clusters = population.units["cluster_id"].to_numpy()
            labels = cluster_randomisation(clusters, rng=_streams(first_seed + r)[0])
            z = (labels == "treatment").astype(int)
            frame = realise_experiment(population, z).analysis_frame
            analysis = clustered_difference(frame["converted"], frame["arm"],
                                            frame["cluster_id"])
            truth = population.truth.ate_conversion
            cr1.append(not analysis.cluster_robust.covers(0.0))
            naive.append(not analysis.naive.covers(0.0))
            covered.append(analysis.cluster_robust.covers(truth))
        planned = de.cluster_power(case.baseline, case.target_effect, clusters_per_arm=g,
                                   mean_cluster_size=case.mean_cluster_size,
                                   icc=case.icc.icc_for_planning,
                                   cluster_size_cv=case.icc.cluster_size_cv)
        rows.append(ClusterPowerRow(label, g, planned, rejection_rate(label, cr1),
                                    rejection_rate(label, naive), float(np.mean(covered))))
    return rows


# ---------------------------------------------------------------------------
# 5. Switchback
# ---------------------------------------------------------------------------

SWITCHBACK_BURN_IN = 2
"""Pre-specified from operational knowledge: driver repositioning settles within two hours."""


def _blocks(config: SwitchbackConfig) -> np.ndarray:
    start_hour = np.arange(config.n_periods) * config.period_hours % 24
    return start_hour // config.period_hours


@dataclass(frozen=True)
class SwitchbackCase:
    truth: float
    direct_effect: float
    analysis: de.SwitchbackAnalysis
    sensitivity: list[tuple[int, float, float, float]]
    """(burn-in hours, estimate, CI lower, CI upper) — reported, not used to choose."""


def switchback_case(config: SwitchbackConfig | None = None, *, seed: int = 15) -> SwitchbackCase:
    config = config or SwitchbackConfig()
    market = generate_switchback_market(config)
    rng_assign, rng_test = _streams(seed)
    w = de.switchback_assignment(config.n_periods, rng=rng_assign, strata=_blocks(config))
    hourly = realise_switchback(market, w)
    analysis = de.analyse_switchback(hourly, period_hours=config.period_hours,
                                     burn_in_hours=SWITCHBACK_BURN_IN, rng=rng_test)
    sensitivity = []
    for b in range(config.period_hours):
        a = de.analyse_switchback(hourly, period_hours=config.period_hours, burn_in_hours=b,
                                  rng=rng_test, n_permutations=199)
        sensitivity.append((b, a.effect.estimate, a.effect.ci_lower, a.effect.ci_upper))
    return SwitchbackCase(market.truth.global_effect, market.truth.direct_effect, analysis,
                          sensitivity)


@dataclass(frozen=True)
class SwitchbackStudy:
    by_burn_in: list[ReplicateSummary]
    aa_randomisation_test: RejectionRate
    aa_naive_hourly_stratified: RejectionRate
    aa_naive_hourly_complete: RejectionRate
    aa_period_welch_complete: RejectionRate


def switchback_study(*, n_replications: int, first_seed: int = 75_000,
                     n_permutations: int = 199) -> SwitchbackStudy:
    config = SwitchbackConfig()
    est: dict[int, list[float]] = {b: [] for b in range(config.period_hours)}
    cov: dict[int, list[bool]] = {b: [] for b in range(config.period_hours)}
    ses: dict[int, list[float]] = {b: [] for b in range(config.period_hours)}
    aa_perm, aa_naive_s, aa_naive_c, aa_period_c = [], [], [], []
    null_config = replace(config, direct_effect=0.0,
                          carryover_effects=tuple(0.0 for _ in config.carryover_effects))
    for r in range(n_replications):
        rng_assign, rng_test = _streams(first_seed + r)
        market = generate_switchback_market(replace(config, seed=first_seed + r))
        w = de.switchback_assignment(config.n_periods, rng=rng_assign, strata=_blocks(config))
        hourly = realise_switchback(market, w)
        for b in est:
            periods = de.period_means(hourly, outcome="orders", burn_in_hours=b,
                                      period_hours=config.period_hours)
            e = stratified_difference(periods["mean"], periods["arm"], periods["block"])
            est[b].append(e.estimate)
            cov[b].append(e.covers(market.truth.global_effect))
            ses[b].append(e.standard_error)
        null_market = generate_switchback_market(replace(null_config, seed=first_seed + r))
        null_hourly = realise_switchback(null_market, w)
        a = de.analyse_switchback(null_hourly, period_hours=config.period_hours,
                                  burn_in_hours=SWITCHBACK_BURN_IN, rng=rng_test,
                                  n_permutations=n_permutations)
        aa_perm.append(rejects(a.randomisation_p_value, ALPHA))
        aa_naive_s.append(not a.naive_hourly.covers(0.0))
        w_complete = de.switchback_assignment(config.n_periods, rng=rng_assign)
        complete_hourly = realise_switchback(null_market, w_complete)
        naive = analyse_continuous(complete_hourly["orders"], complete_hourly["arm"])
        aa_naive_c.append(naive.significant)
        periods = de.period_means(complete_hourly, outcome="orders",
                                  burn_in_hours=SWITCHBACK_BURN_IN,
                                  period_hours=config.period_hours)
        aa_period_c.append(analyse_continuous(periods["mean"], periods["arm"]).significant)
    truth = config.direct_effect + sum(config.carryover_effects)
    return SwitchbackStudy(
        by_burn_in=[summarise_replicates(f"burn-in {b} h", est[b], [truth] * len(est[b]),
                                         cov[b], ses[b]) for b in est],
        aa_randomisation_test=rejection_rate("within-block randomisation test", aa_perm),
        aa_naive_hourly_stratified=rejection_rate("hourly Welch, blocked design", aa_naive_s),
        aa_naive_hourly_complete=rejection_rate("hourly Welch, complete randomisation",
                                                aa_naive_c),
        aa_period_welch_complete=rejection_rate("period-mean Welch, complete randomisation",
                                                aa_period_c),
    )


__all__ = [
    "ALPHA",
    "BayesianCase",
    "BayesianRuleRate",
    "BoundaryReference",
    "CalibrationRow",
    "ClusterCase",
    "ClusterPowerRow",
    "CupedCase",
    "CupedStudy",
    "PeekingStudy",
    "ProcedureSummary",
    "SequentialCase",
    "SwitchbackCase",
    "SwitchbackStudy",
    "bayesian_calibration_study",
    "bayesian_case",
    "bayesian_fixed_truth_study",
    "boundary_reference",
    "cluster_case",
    "cluster_power_study",
    "cuped_case",
    "cuped_study",
    "peeking_study",
    "revenue_break_even",
    "sequential_case",
    "switchback_case",
    "switchback_study",
]
