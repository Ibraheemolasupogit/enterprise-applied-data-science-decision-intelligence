"""M5 case studies: observational designs on synthetic data with known effects.

Three NexaMarket questions, each with its own additive synthetic dataset:

* **Regional delivery-policy rollout** (staggered adoption across markets) —
  difference-in-differences and event studies
  (:mod:`decision_intelligence.data.rollout_panel`).
* **Flagship regional pricing policy** (one treated region) — synthetic
  control (:mod:`decision_intelligence.data.synthetic_control_panel`).
* **Loyalty tier at a score threshold** — regression discontinuity
  (:mod:`decision_intelligence.data.threshold_assignment`).

Every analysis estimates first and attaches ground truth last. Every
method is also run on deliberately invalid designs, and repeated-sampling
studies report bias with Monte Carlo SEs and interval coverage with Wilson
intervals (via the M4 :class:`ReplicateSummary`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy import stats

from decision_intelligence.causal_inference import did as did_methods
from decision_intelligence.causal_inference import rdd as rdd_methods
from decision_intelligence.causal_inference import synthetic_control as sc_methods
from decision_intelligence.causal_inference.campaign import ReplicateSummary
from decision_intelligence.causal_inference.cluster_inference import (
    WildBootstrapResult,
    WildWeights,
    cluster_robust_t_test,
    wild_cluster_bootstrap,
)
from decision_intelligence.causal_inference.estimators import CausalEstimate
from decision_intelligence.common.seeding import seeded_generator
from decision_intelligence.data.rollout_panel import (
    RolloutConfig,
    RolloutPanel,
    generate_rollout_panel,
)
from decision_intelligence.data.synthetic_control_panel import (
    SyntheticControlConfig,
    SyntheticControlPanel,
    generate_synthetic_control_panel,
)
from decision_intelligence.data.threshold_assignment import (
    ThresholdConfig,
    ThresholdData,
    generate_threshold_data,
)
from decision_intelligence.statistics.hypothesis_tests import wilson_interval
from decision_intelligence.statistics.linear_regression import fit_ols


def summarise_replicates(
    name: str, estimates: Sequence[float], truths: Sequence[float],
    covered: Sequence[bool] | None = None, reported_se: Sequence[float] | None = None,
) -> ReplicateSummary:
    """Bias (± Monte Carlo SE), RMSE and coverage with a Wilson interval and exact test."""
    values = np.asarray(estimates, dtype=float)
    errors = values - np.asarray(truths, dtype=float)
    sd = float(errors.std(ddof=1))
    hits = int(sum(covered)) if covered is not None else 0
    n = values.size
    low, high = wilson_interval(hits, n) if covered is not None else (math.nan, math.nan)
    return ReplicateSummary(
        estimator=name,
        truth=float(np.mean(truths)),
        mean_estimate=float(values.mean()),
        bias=float(errors.mean()),
        bias_mc_se=sd / math.sqrt(n),
        empirical_sd=float(values.std(ddof=1)),
        mean_reported_se=float(np.mean(reported_se)) if reported_se is not None else math.nan,
        rmse=float(np.sqrt(np.mean(errors**2))),
        coverage=hits / n if covered is not None else math.nan,
        coverage_ci_lower=low,
        coverage_ci_upper=high,
        coverage_p_value_vs_nominal=float(stats.binomtest(hits, n, 0.95).pvalue)
        if covered is not None else math.nan,
    )


@dataclass(frozen=True)
class RejectionRate:
    """Share of replications rejecting at 5%, with a Wilson interval."""

    procedure: str
    rate: float
    ci_lower: float
    ci_upper: float
    n_replications: int


def rejection_rate(procedure: str, rejections: Sequence[bool]) -> RejectionRate:
    hits, n = int(sum(rejections)), len(rejections)
    low, high = wilson_interval(hits, n)
    return RejectionRate(procedure, hits / n, low, high, n)


# ---------------------------------------------------------------------------
# Difference-in-differences: rollout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RolloutAnalysis:
    twfe: CausalEstimate
    twfe_wild_bootstrap: WildBootstrapResult
    twfe_negative_weight_share: float
    twfe_negative_weight_sum: float
    staggered: did_methods.StaggeredDiDResult
    event_study_twfe: did_methods.EventStudyTWFE
    # evaluation-only (attached after estimation)
    true_overall_att: float = math.nan
    true_event_time_att: dict[int, float] = field(default_factory=dict)
    twfe_implied_estimand: float = math.nan
    """sum_it w_it tau_it with the *true* effects: what TWFE converges to."""


def analyse_rollout(
    data: RolloutPanel, *, rng: np.random.Generator, anticipation: int = 0,
) -> RolloutAnalysis:
    panel = data.panel
    twfe = did_methods.twfe_did(panel)
    weights = did_methods.twfe_weights(panel)
    design, y, clusters, column = did_methods.twfe_design(panel)
    wild = wild_cluster_bootstrap(design, y, clusters, column, rng=rng, n_draws=1999)
    staggered = did_methods.staggered_did(panel, anticipation=anticipation)
    event_study = did_methods.event_study_twfe(panel, leads=6, lags=8)
    analysis = RolloutAnalysis(
        twfe=twfe.estimate,
        twfe_wild_bootstrap=wild,
        twfe_negative_weight_share=float(np.mean(weights["weight"] < 0)),
        twfe_negative_weight_sum=float(weights.loc[weights["weight"] < 0, "weight"].sum()),
        staggered=staggered,
        event_study_twfe=event_study,
    )
    truth = data.ground_truth
    merged = weights.merge(truth.cell_effects, on=["unit_id", "period"])
    return replace(
        analysis,
        true_overall_att=truth.overall_att,
        true_event_time_att=truth.event_time_att,
        twfe_implied_estimand=float(np.sum(merged["weight"] * merged["true_direct_effect"])),
    )


#: Rollout designs: name -> (template config, anticipation assumed by the analyst, valid?).
DID_DESIGNS: dict[str, tuple[RolloutConfig, int, bool]] = {
    "valid: staggered, heterogeneous effects": (RolloutConfig(), 0, True),
    "valid: common timing": (
        RolloutConfig(cohorts=((13, 0.5),), cohort_effect_scales=(1.0,)), 0, True
    ),
    "INVALID: parallel trends violated (adopters on steeper path)": (
        RolloutConfig(differential_trend=0.08), 0, False
    ),
    "INVALID: 2-period anticipation ignored": (RolloutConfig(anticipation_periods=2), 0, False),
    "valid: 2-period anticipation modelled (base period moved back)": (
        RolloutConfig(anticipation_periods=2), 2, True
    ),
    "INVALID: spillovers onto never-treated markets": (
        RolloutConfig(spillover_share=0.5), 0, False
    ),
    "INVALID: spillovers, common timing": (
        RolloutConfig(cohorts=((13, 0.5),), cohort_effect_scales=(1.0,), spillover_share=0.5),
        0,
        False,
    ),
}


def did_monte_carlo(
    template: RolloutConfig, *, anticipation: int, n_replications: int,
    first_seed: int = 50_000,
) -> dict[str, object]:
    """Repeated draws of one rollout design: CS and TWFE bias/coverage, pre-trend test rates."""
    cs_est, cs_cov, cs_se, twfe_est, twfe_cov, twfe_se, truths = [], [], [], [], [], [], []
    cs_pretrend, twfe_pretrend = [], []
    for r in range(n_replications):
        config = replace(template, seed=first_seed + r)
        data = generate_rollout_panel(config)
        cs = did_methods.staggered_did(data.panel, anticipation=anticipation)
        twfe = did_methods.twfe_did(data.panel).estimate
        truth = data.ground_truth.overall_att
        truths.append(truth)
        cs_est.append(cs.overall.estimate)
        cs_cov.append(cs.overall.covers(truth))
        cs_se.append(cs.overall.standard_error)
        twfe_est.append(twfe.estimate)
        twfe_cov.append(twfe.covers(truth))
        twfe_se.append(twfe.standard_error)
        if cs.pretrend is not None:
            cs_pretrend.append(cs.pretrend.p_value < 0.05)
        es = did_methods.event_study_twfe(data.panel, leads=6, lags=8)
        if es.pretrend is not None:
            twfe_pretrend.append(es.pretrend.p_value < 0.05)
    return {
        "callaway_santanna": summarise_replicates(
            "Callaway–Sant'Anna overall ATT", cs_est, truths, cs_cov, cs_se
        ),
        "twfe": summarise_replicates(
            "static TWFE (deviation from the overall ATT)", twfe_est, truths, twfe_cov, twfe_se
        ),
        "cs_pretrend_rejection": rejection_rate("CS pre-trend Wald test", cs_pretrend)
        if cs_pretrend else None,
        "twfe_pretrend_rejection": rejection_rate("dynamic TWFE pre-trend test", twfe_pretrend)
        if twfe_pretrend else None,
    }


_WILD_VARIANTS: tuple[tuple[WildWeights, str], ...] = (
    ("rademacher", "wild cluster bootstrap (Rademacher)"),
    ("webb", "wild cluster bootstrap (Webb)"),
)


def few_cluster_size_study(
    n_units: int, n_treated: int, *, n_replications: int, n_periods: int = 12,
    ar1_rho: float = 0.7, n_draws: int = 999, first_seed: int = 70_000,
) -> list[RejectionRate]:
    """Null-effect common-timing panels: how often each procedure falsely rejects at 5%.

    Procedures: classical OLS SEs (ignore serial correlation), CR1 cluster SEs
    with t(G-1), wild cluster restricted bootstrap with Rademacher and Webb
    weights. A valid 5% test rejects ~5% of the time.
    """
    share = n_treated / n_units
    adoption = n_periods // 2 + 1
    results: dict[str, list[bool]] = {
        "classical OLS SE (ignores clustering)": [],
        "CR1 cluster SE, t(G-1)": [],
        "wild cluster bootstrap (Rademacher)": [],
        "wild cluster bootstrap (Webb)": [],
    }
    for r in range(n_replications):
        config = RolloutConfig(
            seed=first_seed + r, n_units=n_units, n_periods=n_periods,
            cohorts=((adoption, share),), cohort_effect_scales=(1.0,), base_effect=0.0,
            ar1_rho=ar1_rho, unit_effect_sd=1.0,
        )
        design, y, clusters, column = did_methods.twfe_design(generate_rollout_panel(config).panel)
        classical = fit_ols(design, y, add_intercept=False)
        results["classical OLS SE (ignores clustering)"].append(
            float(classical.p_values[column]) < 0.05
        )
        results["CR1 cluster SE, t(G-1)"].append(
            cluster_robust_t_test(design, y, clusters, column)[2] < 0.05
        )
        rng = seeded_generator(first_seed + r)
        for weights, key in _WILD_VARIANTS:
            wild = wild_cluster_bootstrap(
                design, y, clusters, column, rng=rng, n_draws=n_draws,
                weights=weights, confidence_level=None,
            )
            results[key].append(wild.p_value < 0.05)
    return [rejection_rate(name, values) for name, values in results.items()]


# ---------------------------------------------------------------------------
# Synthetic control
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticControlAnalysis:
    fit: sc_methods.SyntheticControlFit
    placebo_all: sc_methods.PlaceboInference
    placebo_excluding_poor_fits: sc_methods.PlaceboInference
    in_time_placebo: sc_methods.SyntheticControlFit
    leave_one_out: dict[int, float]
    naive_did: float
    """Treated pre/post change minus donor-average change: ignores differential factor loadings."""
    true_average_post_effect: float = math.nan
    true_effect_by_period: dict[int, float] = field(default_factory=dict)


def analyse_synthetic_control(data: SyntheticControlPanel) -> SyntheticControlAnalysis:
    panel, start = data.panel, data.config.n_pre + 1
    fit = sc_methods.fit_synthetic_control(panel, 0, start)
    wide = panel.pivot(index="period", columns="unit_id", values="outcome")
    pre = wide.index < start
    treated_change = wide.loc[~pre, 0].mean() - wide.loc[pre, 0].mean()
    donors = wide.drop(columns=0)
    donor_change = donors.loc[~pre].mean().mean() - donors.loc[pre].mean().mean()
    analysis = SyntheticControlAnalysis(
        fit=fit,
        placebo_all=sc_methods.placebo_in_space(panel, 0, start),
        placebo_excluding_poor_fits=sc_methods.placebo_in_space(
            panel, 0, start, exclusion_multiple=5.0
        ),
        in_time_placebo=sc_methods.placebo_in_time(panel, 0, start, start - 10),
        leave_one_out=sc_methods.leave_one_out(panel, 0, start),
        naive_did=float(treated_change - donor_change),
    )
    return replace(
        analysis,
        true_average_post_effect=data.ground_truth.average_post_effect,
        true_effect_by_period=data.ground_truth.effect_by_period,
    )


SC_DESIGNS: dict[str, tuple[SyntheticControlConfig, bool]] = {
    "valid: treated inside donors' convex hull": (SyntheticControlConfig(), True),
    "valid, no effect (calibration of the placebo rank)": (
        SyntheticControlConfig(effect_size=0.0), True
    ),
    "INVALID: treated outside the convex hull": (
        SyntheticControlConfig(treated_in_convex_hull=False), False
    ),
    "INVALID: contaminated donors (spillovers)": (
        SyntheticControlConfig(contaminated_donors=4, contamination_share=0.6), False
    ),
}


def sc_monte_carlo(
    template: SyntheticControlConfig, *, n_replications: int, with_placebo: bool = True,
    first_seed: int = 60_000,
) -> dict[str, object]:
    """Bias/RMSE of the average post gap, share of fits flagged, placebo rank-1 rate."""
    estimates, truths, naive, flagged, rank_one = [], [], [], [], []
    for r in range(n_replications):
        config = replace(template, seed=first_seed + r)
        data = generate_synthetic_control_panel(config)
        start = config.n_pre + 1
        fit = sc_methods.fit_synthetic_control(data.panel, 0, start)
        estimates.append(fit.average_post_gap)
        truths.append(data.ground_truth.average_post_effect)
        flagged.append(bool(fit.warnings))
        wide = data.panel.pivot(index="period", columns="unit_id", values="outcome")
        pre = wide.index < start
        donors = wide.drop(columns=0)
        naive.append(
            float(wide.loc[~pre, 0].mean() - wide.loc[pre, 0].mean()
                  - (donors.loc[~pre].mean().mean() - donors.loc[pre].mean().mean()))
        )
        if with_placebo:
            placebo = sc_methods.placebo_in_space(data.panel, 0, start)
            rank_one.append(placebo.rank == 1)
    return {
        "synthetic_control": summarise_replicates(
            "synthetic control average gap", estimates, truths
        ),
        "naive_did": summarise_replicates("naive DiD vs donor average", naive, truths),
        "share_flagged": float(np.mean(flagged)),
        "placebo_rank_one": rejection_rate("placebo rank 1 (p = 1/(J+1))", rank_one)
        if rank_one else None,
    }


# ---------------------------------------------------------------------------
# Regression discontinuity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdAnalysis:
    bandwidth: float
    conventional: rdd_methods.RDDEstimate
    robust: rdd_methods.RDDEstimate
    bandwidth_sensitivity: pd.DataFrame
    mccrary: rdd_methods.DensityTest
    binomial: rdd_methods.DensityTest
    covariates: pd.DataFrame
    placebo_cutoffs: pd.DataFrame
    donuts: dict[float, rdd_methods.RDDEstimate]
    """Robust estimates excluding ``|x - c| <`` each hole size."""
    global_polynomials: dict[int, rdd_methods.RDDEstimate]
    true_tau_at_cutoff: float = math.nan
    true_population_ate: float = math.nan
    true_att: float = math.nan


def analyse_threshold(data: ThresholdData) -> ThresholdAnalysis:
    x = data.frame["score"].to_numpy(float)
    y = data.frame["spend"].to_numpy(float)
    c = data.config.cutoff
    h = rdd_methods.ik_bandwidth(x, y, c)
    analysis = ThresholdAnalysis(
        bandwidth=h,
        conventional=rdd_methods.local_linear_rdd(x, y, c, h),
        robust=rdd_methods.robust_bias_corrected_rdd(x, y, c, h),
        bandwidth_sensitivity=rdd_methods.bandwidth_sensitivity(
            x, y, c, [h * f for f in (0.5, 0.75, 1.0, 1.5, 2.0)]
        ),
        mccrary=rdd_methods.mccrary_density_test(x, c),
        binomial=rdd_methods.binomial_density_test(x, c, 2.0),
        covariates=rdd_methods.covariate_continuity(x, data.frame[["tenure_months"]], c, h),
        placebo_cutoffs=rdd_methods.placebo_cutoffs(x, y, c, [c - 10, c - 5, c + 5, c + 10], h),
        donuts={
            hole: rdd_methods.donut_rdd(x, y, c, h, hole)
            for hole in (0.5, data.config.manipulation_window)
        },
        global_polynomials={p: rdd_methods.global_polynomial_rdd(x, y, c, p) for p in (1, 2, 4, 6)},
    )
    truth = data.ground_truth
    return replace(
        analysis, true_tau_at_cutoff=truth.tau_at_cutoff,
        true_population_ate=truth.population_ate, true_att=truth.att,
    )


RDD_DESIGNS: dict[str, tuple[ThresholdConfig, bool]] = {
    "valid: sharp threshold, no sorting": (ThresholdConfig(), True),
    "INVALID: manipulation (60% of engaged near-misses push over the cutoff)": (
        ThresholdConfig(manipulation_share=0.6), False
    ),
    "INVALID: compound treatment (+2 from another rule at the same cutoff)": (
        ThresholdConfig(compound_jump=2.0), False
    ),
}


def rdd_monte_carlo(
    template: ThresholdConfig, *, n_replications: int, first_seed: int = 80_000,
) -> dict[str, object]:
    """Conventional vs robust bias-corrected inference, density and covariate test rates,
    and the global-polynomial comparison."""
    conv, conv_cov, rbc, rbc_cov, glob, truths = [], [], [], [], [], []
    density_reject, covariate_reject, rbc_se, conv_se = [], [], [], []
    for r in range(n_replications):
        config = replace(template, seed=first_seed + r)
        data = generate_threshold_data(config)
        x = data.frame["score"].to_numpy(float)
        y = data.frame["spend"].to_numpy(float)
        c = config.cutoff
        h = rdd_methods.ik_bandwidth(x, y, c)
        a = rdd_methods.local_linear_rdd(x, y, c, h)
        b = rdd_methods.robust_bias_corrected_rdd(x, y, c, h)
        truth = data.ground_truth.tau_at_cutoff
        truths.append(truth)
        conv.append(a.estimate)
        conv_cov.append(a.covers(truth))
        conv_se.append(a.standard_error)
        rbc.append(b.estimate)
        rbc_cov.append(b.covers(truth))
        rbc_se.append(b.standard_error)
        glob.append(rdd_methods.global_polynomial_rdd(x, y, c, 4).estimate)
        density_reject.append(rdd_methods.mccrary_density_test(x, c).p_value < 0.05)
        cov = rdd_methods.covariate_continuity(x, data.frame[["tenure_months"]], c, h)
        covariate_reject.append(not bool(cov["covers_zero"].iloc[0]))
    return {
        "conventional": summarise_replicates("local linear, conventional CI", conv, truths,
                                             conv_cov, conv_se),
        "robust": summarise_replicates("robust bias-corrected (CCT, rho = 1)", rbc, truths,
                                       rbc_cov, rbc_se),
        "global_quartic": summarise_replicates("global quartic polynomial", glob, truths),
        "mccrary_rejection": rejection_rate("McCrary density test", density_reject),
        "covariate_rejection": rejection_rate("covariate (tenure) continuity", covariate_reject),
    }


def placebo_cutoff_study(
    *, n_replications: int, placebo: float = 50.0, first_seed: int = 85_000
) -> list[RejectionRate]:
    """False-positive rate at a placebo cutoff for conventional vs robust inference."""
    conventional, robust = [], []
    for r in range(n_replications):
        data = generate_threshold_data(ThresholdConfig(seed=first_seed + r))
        x = data.frame["score"].to_numpy(float)
        y = data.frame["spend"].to_numpy(float)
        c = data.config.cutoff
        h = rdd_methods.ik_bandwidth(x, y, c)
        side = x < c
        conventional.append(
            not rdd_methods.local_linear_rdd(x[side], y[side], placebo, h).covers(0.0)
        )
        robust.append(
            not rdd_methods.robust_bias_corrected_rdd(x[side], y[side], placebo, h).covers(0.0)
        )
    return [
        rejection_rate(f"conventional local linear at placebo cutoff {placebo:g}", conventional),
        rejection_rate(f"robust bias-corrected at placebo cutoff {placebo:g}", robust),
    ]



# ---------------------------------------------------------------------------
# Calibration studies (scientific assurance review)
# ---------------------------------------------------------------------------


def pretrend_calibration_study(
    n_units: int, *, n_replications: int, first_seed: int = 70_000
) -> list[RejectionRate]:
    """Size of the pre-trend test under valid parallel trends, two variants:

    * all available leads, chi-square reference (the naive choice);
    * leads in the 6-period analysis window, finite-cluster F reference (default).
    """
    naive, default = [], []
    for r in range(n_replications):
        data = generate_rollout_panel(RolloutConfig(seed=first_seed + r, n_units=n_units))
        result = did_methods.staggered_did(data.panel)
        times = [e.event_time for e in result.event_study]
        leads = [e for e in times if e < -1]
        idx = [times.index(e) for e in leads]
        estimates = np.array([result.event_study[i].estimate for i in idx])
        covariance = result.event_covariance[np.ix_(idx, idx)]
        naive.append(did_methods.pretrend_wald_test(estimates, covariance, leads).p_value < 0.05)
        if result.pretrend is not None:
            default.append(result.pretrend.p_value < 0.05)
    return [
        rejection_rate(f"all leads, chi-square ({n_units} units)", naive),
        rejection_rate(f"6-lead window, finite-cluster F ({n_units} units)", default),
    ]


def event_band_coverage_study(
    *, n_replications: int, first_seed: int = 60_000
) -> list[RejectionRate]:
    """Share of panels in which *every* event-time interval in the window covers the truth."""
    pointwise, band = [], []
    for r in range(n_replications):
        data = generate_rollout_panel(RolloutConfig(seed=first_seed + r))
        result = did_methods.staggered_did(data.panel)
        truth = data.ground_truth.event_time_att
        window = [e for e in result.event_study if not math.isnan(e.band_lower)]
        pointwise.append(
            all(e.ci_lower <= truth.get(e.event_time, 0.0) <= e.ci_upper for e in window)
        )
        band.append(
            all(e.band_lower <= truth.get(e.event_time, 0.0) <= e.band_upper for e in window)
        )
    return [
        rejection_rate("joint coverage: pointwise 95% intervals", pointwise),
        rejection_rate("joint coverage: sup-t simultaneous band (nominal 95%)", band),
    ]


def cs_inference_comparison(
    *, n_replications: int, n_bootstrap: int = 199, first_seed: int = 50_000
) -> list[ReplicateSummary]:
    """Callaway–Sant'Anna overall ATT: influence-function vs unit-bootstrap intervals."""
    rows: dict[str, tuple[list[float], list[float], list[bool], list[float]]] = {
        "influence function (conditional on cohorts)": ([], [], [], []),
        f"unit bootstrap ({n_bootstrap}, percentile)": ([], [], [], []),
    }
    for r in range(n_replications):
        data = generate_rollout_panel(RolloutConfig(seed=first_seed + r))
        truth = data.ground_truth.overall_att
        analytic = did_methods.staggered_did(data.panel).overall
        boot = did_methods.staggered_did(
            data.panel, se_method="bootstrap", rng=seeded_generator(first_seed + r),
            n_bootstrap=n_bootstrap,
        ).overall
        for key, estimate in zip(rows, (analytic, boot), strict=True):
            rows[key][0].append(estimate.estimate)
            rows[key][1].append(truth)
            rows[key][2].append(estimate.covers(truth))
            rows[key][3].append(estimate.standard_error)
    return [summarise_replicates(k, *v) for k, v in rows.items()]


def bandwidth_oracle_study(
    *, n_replications: int, grid: Sequence[float] = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14),
    first_seed: int = 40_000,
) -> tuple[pd.DataFrame, float, float]:
    """Monte Carlo MSE of the local-linear estimator over fixed bandwidths (the oracle,
    possible only with known truth), and the mean and SD of the data-driven IK bandwidth."""
    estimates: dict[float, list[float]] = {h: [] for h in grid}
    chosen: list[float] = []
    truth = ThresholdConfig().tau_at_cutoff
    for r in range(n_replications):
        data = generate_threshold_data(ThresholdConfig(seed=first_seed + r))
        x = data.frame["score"].to_numpy(float)
        y = data.frame["spend"].to_numpy(float)
        chosen.append(rdd_methods.ik_bandwidth(x, y, data.config.cutoff))
        for h in grid:
            estimates[h].append(rdd_methods.local_linear_rdd(x, y, data.config.cutoff, h).estimate)
    table = pd.DataFrame(
        {
            "bandwidth": list(grid),
            "bias": [float(np.mean(estimates[h]) - truth) for h in grid],
            "variance": [float(np.var(estimates[h])) for h in grid],
            "mse": [float(np.mean((np.array(estimates[h]) - truth) ** 2)) for h in grid],
        }
    )
    return table, float(np.mean(chosen)), float(np.std(chosen))
