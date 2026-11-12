"""M6 case studies and repeated-sampling studies.

* :func:`analyse_m2_experiment` — the M2 recommendation-feature experiment
  through the full lifecycle (design, integrity checks, analysis, decision).
* :func:`analyse_scenario_case` — a larger synthetic experiment with
  guardrails, run once cleanly and once with injected defects.
* Monte Carlo studies (``*_study``) — every calibration, coverage or error
  rate claim in the M6 evidence comes from one of these, with Wilson
  intervals (rates) or Monte Carlo standard errors (bias).

Ground truth is read only after estimation, to score estimates.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from decision_intelligence.causal_inference.campaign import ReplicateSummary
from decision_intelligence.causal_inference.estimators import CausalEstimate
from decision_intelligence.causal_inference.observational_cases import (
    RejectionRate,
    rejection_rate,
    summarise_replicates,
)
from decision_intelligence.common.seeding import seeded_generator, spawn_generators
from decision_intelligence.data.config import SyntheticDataConfig
from decision_intelligence.data.dataset import SyntheticDataset
from decision_intelligence.data.experiment_scenarios import (
    ExperimentPopulation,
    ExperimentScenarioConfig,
    generate_experiment_population,
    realise_experiment,
)
from decision_intelligence.data.experiments import generate_experiment
from decision_intelligence.experimentation import analysis as an
from decision_intelligence.experimentation import compliance as co
from decision_intelligence.experimentation import validity as va
from decision_intelligence.experimentation.assignment import (
    cluster_randomisation,
    complete_randomisation,
    hash_assignment,
    stratified_randomisation,
)
from decision_intelligence.experimentation.decision import (
    GuardrailAssessment,
    Recommendation,
    assess_guardrail,
    recommend,
)
from decision_intelligence.experimentation.design import ExperimentDesign, MetricSpec
from decision_intelligence.experimentation.planning import (
    ExperimentPlan,
    mde_binary,
    plan_binary,
    plan_continuous,
)
from decision_intelligence.experimentation.srm import SRMResult, srm_breakdown, srm_from_labels
from decision_intelligence.statistics.multiple_testing import adjust_p_values, rejects
from decision_intelligence.statistics.power import power_two_proportions

HALF = {"control": 0.5, "treatment": 0.5}

# ---------------------------------------------------------------------------
# 1. The M2 randomised experiment
# ---------------------------------------------------------------------------

M2_DESIGN = ExperimentDesign(
    name="recommendation_feature_experiment",
    hypothesis="The new recommendation feature raises 14-day conversion among eligible customers",
    population="customers who signed up at least 14 days before the assignment date",
    experimental_unit="customer",
    randomisation_unit="customer",
    analysis_unit="customer",
    estimand="ITT: difference in 14-day conversion probability between assignment to the "
    "feature and to control, averaged over the eligible population",
    primary=MetricSpec(
        "conversion_14d",
        "binary",
        "converted within 14 days of assignment (M2 experiment_outcomes)",
    ),
    minimum_worthwhile_effect=0.02,
    secondary=(
        MetricSpec(
            "revenue_14d", "continuous", "transaction amount in [assignment, assignment + 14 days)"
        ),
        MetricSpec("sessions_14d", "continuous", "sessions in [assignment, assignment + 14 days)"),
    ),
    alpha=0.05,
    power=0.8,
    treatment_share=0.5,
    notes=(
        "M2 has no latency/error telemetry, so no guardrails can be evaluated here; the "
        "scenario case demonstrates them",
    ),
)


@dataclass(frozen=True)
class M2ExperimentAnalysis:
    design: ExperimentDesign
    checks: tuple[va.CheckResult, ...]
    balance: pd.DataFrame
    srm: SRMResult
    primary: an.MetricResult
    secondary: tuple[an.MetricResult, ...]
    secondary_holm: dict[str, float]
    plan_for_true_effect: ExperimentPlan
    plan_for_worthwhile_effect: ExperimentPlan
    mde_at_realised_n: float
    recommendation: Recommendation
    # evaluation-only
    true_effect: float
    true_secondary_effects: dict[str, float]


def m2_experiment_frame(dataset: SyntheticDataset) -> pd.DataFrame:
    """One row per assigned customer: arm, outcomes and pre-treatment covariates only."""
    tables = dataset.tables
    frame = tables["experiment_assignments"].merge(
        tables["experiment_outcomes"], on=["customer_id", "experiment_name"]
    )
    start = frame["assignment_date"].iloc[0]
    end = start + pd.Timedelta(14, unit="D")
    customers = tables["customers"].drop(columns=["latent_value"])  # evaluation-only driver
    frame = frame.merge(customers, on="customer_id")
    tx = tables["transactions"]
    in_window = tx[(tx["transaction_date"] >= start) & (tx["transaction_date"] < end)]
    before = tx[
        (tx["transaction_date"] >= start - pd.Timedelta(90, unit="D"))
        & (tx["transaction_date"] < start)
    ]
    sessions = tables["sessions"]
    s_window = sessions[
        (sessions["session_timestamp"] >= start) & (sessions["session_timestamp"] < end)
    ]
    frame = frame.assign(
        revenue_14d=frame["customer_id"]
        .map(in_window.groupby("customer_id")["amount"].sum())
        .fillna(0.0),
        sessions_14d=frame["customer_id"]
        .map(s_window.groupby("customer_id").size())
        .fillna(0)
        .astype(float),
        pre_revenue_90d=frame["customer_id"]
        .map(before.groupby("customer_id")["amount"].sum())
        .fillna(0.0),
        tenure_days=(start - frame["signup_date"]).dt.days.astype(float),
        income_missing=frame["household_income_proxy"].isna().astype(float),
        converted=frame["converted"].astype(float),
        window_days=(frame["observation_date"] - frame["assignment_date"]).dt.days,
    )
    return frame


def analyse_m2_experiment(
    dataset: SyntheticDataset, *, rng: np.random.Generator
) -> M2ExperimentAnalysis:
    design = M2_DESIGN
    frame = m2_experiment_frame(dataset)
    assigned = (frame["variant"] == "treatment").astype(float).to_numpy()
    srm = srm_from_labels(frame["variant"], HALF, alpha=design.srm_alpha)
    balance_check, balance = va.check_pre_treatment_balance(
        frame[
            [
                "segment",
                "acquisition_channel",
                "region",
                "tenure_days",
                "pre_revenue_90d",
                "income_missing",
            ]
        ],
        assigned,
    )
    window_ok = bool((frame["window_days"] == 14).all())
    checks = (
        va.check_duplicates(
            dataset.tables["experiment_assignments"], unit="customer_id", arm="variant"
        ),
        va.check_srm(frame["variant"], HALF, alpha=design.srm_alpha),
        balance_check,
        va.check_metric_integrity(frame["converted"], metric="conversion_14d", kind="binary"),
        va.check_metric_integrity(
            frame["revenue_14d"], metric="revenue_14d", kind="continuous", lower=0.0
        ),
        va.CheckResult(
            "outcome window",
            "PASS" if window_ok else "FAIL",
            "every outcome observed over exactly 14 days"
            if window_ok
            else "outcome windows differ between units",
        ),
    )
    primary = an.analyse_binary(frame["converted"], frame["variant"], metric="conversion_14d")
    secondary = (
        an.analyse_continuous(
            frame["revenue_14d"], frame["variant"], metric="revenue_14d", rng=rng
        ),
        an.analyse_continuous(frame["sessions_14d"], frame["variant"], metric="sessions_14d"),
    )
    holm = adjust_p_values([s.p_value for s in secondary], method="holm")
    baseline = 0.10  # assumed pre-experiment conversion rate (planning input, not the result)
    failures = [c.check for c in checks if c.status == "FAIL"]
    recommendation = recommend(
        primary,
        minimum_worthwhile_effect=design.minimum_worthwhile_effect,
        validity_failures=failures,
    )
    truth = dataset.ground_truth.experiment.true_absolute_effect
    return M2ExperimentAnalysis(
        design=design,
        checks=checks,
        balance=balance,
        srm=srm,
        primary=primary,
        secondary=secondary,
        secondary_holm=dict(
            zip([s.metric for s in secondary], holm.adjusted_p_values.tolist(), strict=True)
        ),
        plan_for_true_effect=plan_binary(baseline, mde_absolute=0.04),
        plan_for_worthwhile_effect=plan_binary(
            baseline, mde_absolute=design.minimum_worthwhile_effect
        ),
        mde_at_realised_n=mde_binary(baseline, primary.n_control, primary.n_treatment),
        recommendation=recommendation,
        true_effect=truth,
        # Transactions and sessions are generated independently of the experiment in M2.
        true_secondary_effects={"revenue_14d": 0.0, "sessions_14d": 0.0},
    )


@dataclass(frozen=True)
class M2ReplicationStudy:
    effect: float
    summary: ReplicateSummary
    rejection: RejectionRate
    decisions: dict[str, float]
    planned_power: float
    mean_significant_estimate: float
    """Mean estimate among replicates significant at 5% in the positive direction
    (exaggeration / type M error when the design is underpowered)."""


def m2_replication_study(
    dataset: SyntheticDataset, *, effect: float, n_replications: int, first_seed: int = 90_000
) -> M2ReplicationStudy:
    """Re-run the M2 experiment generator (unchanged) on the same eligible customers.

    Each replicate draws a new assignment and new outcomes; the individual
    effect is exactly ``effect`` for every customer (M2 adds it to every
    conversion probability, which never reaches the clip bounds).
    """
    config: SyntheticDataConfig = replace(dataset.config, experiment_true_effect=effect)
    customers = dataset.tables["customers"]
    estimates, covered, rejected, ses = [], [], [], []
    decisions: dict[str, int] = {"LAUNCH": 0, "HOLD": 0, "ITERATE": 0, "STOP": 0}
    n_c = n_t = 0
    control_rates: list[float] = []
    for r in range(n_replications):
        assignments, outcomes, _ = generate_experiment(
            config, customers, seeded_generator(first_seed + r)
        )
        merged = assignments.merge(outcomes, on=["customer_id", "experiment_name"])
        result = an.analyse_binary(merged["converted"].astype(float), merged["variant"])
        estimates.append(result.effect.estimate)
        covered.append(result.effect.covers(effect))
        rejected.append(result.significant)
        ses.append(result.effect.standard_error)
        decisions[recommend(result, minimum_worthwhile_effect=0.02).decision] += 1
        n_c += result.n_control
        control_rates.append(result.mean_control)
        n_t += result.n_treatment
    base = float(np.mean(control_rates))
    planned = (
        power_two_proportions(
            base + effect, base, round(n_t / n_replications), round(n_c / n_replications)
        )
        if effect
        else 0.05
    )
    return M2ReplicationStudy(
        effect=effect,
        summary=summarise_replicates(
            f"Newcombe CI, true effect {effect:g}",
            estimates,
            [effect] * n_replications,
            covered,
            ses,
        ),
        rejection=rejection_rate(f"z-test rejects at 5% (true effect {effect:g})", rejected),
        decisions={k: v / n_replications for k, v in decisions.items()},
        planned_power=planned,
        mean_significant_estimate=float(
            np.mean([e for e, r in zip(estimates, rejected, strict=True) if r and e > 0])
        )
        if any(r and e > 0 for e, r in zip(estimates, rejected, strict=True))
        else math.nan,
    )


# ---------------------------------------------------------------------------
# 2. Scenario case: full lifecycle with guardrails, clean and defective runs
# ---------------------------------------------------------------------------

SCENARIO_DESIGN = ExperimentDesign(
    name="ranking_algorithm_v2",
    hypothesis="Ranking algorithm v2 raises 14-day conversion without slowing pages or raising "
    "errors",
    population="active users eligible on the assignment date",
    experimental_unit="user",
    randomisation_unit="user",
    analysis_unit="user",
    estimand="ITT: difference in 14-day conversion probability, assignment to v2 vs current",
    primary=MetricSpec("conversion_14d", "binary", "converted within 14 days of assignment"),
    minimum_worthwhile_effect=0.01,
    secondary=(MetricSpec("revenue_14d", "continuous", "spend in the 14-day window"),),
    guardrails=(
        MetricSpec(
            "latency_ms", "continuous", "mean page latency", direction="decrease", margin=10.0
        ),
        MetricSpec(
            "error_rate",
            "binary",
            "share of users with a failed request",
            direction="decrease",
            margin=0.003,
        ),
    ),
    srm_alpha=0.001,
    exclusion_rules=("internal test accounts (flagged before assignment)",),
)

SCENARIO_CLEAN = ExperimentScenarioConfig(latency_shift_ms=2.0)
SCENARIO_DEFECTIVE = replace(
    SCENARIO_CLEAN,
    treatment_loss_share=0.05,
    loss_mechanism="low_engagement",
    duplicate_share=0.01,
    always_taker_share=0.03,
)


@dataclass(frozen=True)
class ScenarioRun:
    label: str
    plan: ExperimentPlan
    checks: tuple[va.CheckResult, ...]
    srm_by_segment: pd.DataFrame
    lost_unit_profile: pd.DataFrame | None
    primary: an.MetricResult
    stratified_primary: CausalEstimate
    revenue: an.MetricResult
    guardrails: tuple[GuardrailAssessment, ...]
    recommendation: Recommendation
    selection_on_engaged: an.MetricResult
    # evaluation-only
    true_effect: float
    true_revenue_effect: float


def _scenario_run(
    label: str, population: ExperimentPopulation, plan: ExperimentPlan, seed: int
) -> ScenarioRun:
    design = SCENARIO_DESIGN
    labels = stratified_randomisation(population.units["segment"], rng=seeded_generator(seed))
    assigned = (labels == "treatment").astype(int)
    realised = realise_experiment(population, assigned)
    log = va.first_assignment(realised.assignment_log)
    frame = realised.analysis_frame.drop_duplicates(subset=["unit_id"], keep="first")
    # Use the server-side first assignment as the arm of record (preserves randomisation).
    frame = frame.drop(columns=["arm"]).merge(log[["unit_id", "arm"]], on="unit_id")
    z = (frame["arm"] == "treatment").astype(float).to_numpy()
    checks = [
        va.check_duplicates(realised.assignment_log),
        va.check_srm(frame["arm"], HALF, alpha=design.srm_alpha),
        va.check_pre_treatment_balance(frame[["segment", "pre_sessions", "pre_revenue"]], z)[0],
        va.check_assignment_leakage(z, frame["exposed"].to_numpy()),
        va.check_metric_integrity(frame["converted"], metric="conversion_14d", kind="binary"),
        va.check_metric_integrity(
            frame["latency_ms"], metric="latency_ms", kind="continuous", lower=0.0
        ),
        va.check_exclusions(
            np.zeros(len(frame), dtype=bool), z, rule_timing={"internal test accounts": "pre"}
        ),
    ]
    by_segment = srm_breakdown(frame, arm_column="arm", by="segment", expected_shares=HALF)
    lost_profile = None
    missing = ~log["unit_id"].isin(frame["unit_id"])
    if missing.any():
        # Diagnosis aid: compare pre-treatment covariates of units missing from telemetry.
        units = population.units.set_index("unit_id")
        profile = units.loc[log["unit_id"]].assign(
            arm=log["arm"].to_numpy(), status=np.where(missing, "missing", "logged")
        )
        lost_profile = profile.groupby(["arm", "status"])[["pre_sessions", "pre_revenue"]].mean()
        lost_profile["n"] = profile.groupby(["arm", "status"]).size()
    primary = an.analyse_binary(frame["converted"], frame["arm"], metric="conversion_14d")
    stratified = an.stratified_difference(frame["converted"], frame["arm"], frame["segment"])
    revenue = an.analyse_continuous(frame["revenue"], frame["arm"], metric="revenue_14d")
    guardrail_results = (
        assess_guardrail(
            an.analyse_continuous(frame["latency_ms"], frame["arm"], metric="latency_ms"),
            design.guardrails[0],
        ),
        assess_guardrail(
            an.analyse_binary(frame["error"], frame["arm"], metric="error_rate"),
            design.guardrails[1],
        ),
    )
    engaged = frame["engaged"] == 1
    selected = an.analyse_binary(
        frame.loc[engaged, "converted"],
        frame.loc[engaged, "arm"],
        metric="conversion among engaged users",
    )
    failures = [c.check for c in checks if c.status == "FAIL"]
    rec = recommend(
        primary,
        minimum_worthwhile_effect=design.minimum_worthwhile_effect,
        guardrails=guardrail_results,
        validity_failures=failures,
    )
    return ScenarioRun(
        label,
        plan,
        tuple(checks),
        by_segment,
        lost_profile,
        primary,
        stratified,
        revenue,
        guardrail_results,
        rec,
        selected,
        population.truth.itt_conversion,
        population.truth.itt_revenue,
    )


def analyse_scenario_case() -> tuple[ScenarioRun, ScenarioRun]:
    plan = plan_binary(
        0.10, mde_absolute=SCENARIO_DESIGN.minimum_worthwhile_effect, daily_eligible=2_500.0
    )
    clean = generate_experiment_population(replace(SCENARIO_CLEAN, n_units=plan.n_total + 1))
    defective = generate_experiment_population(
        replace(SCENARIO_DEFECTIVE, n_units=plan.n_total + 1)
    )
    return (
        _scenario_run("clean", clean, plan, 11),
        _scenario_run("defective", defective, plan, 11),
    )


# ---------------------------------------------------------------------------
# 3. Repeated-sampling studies
# ---------------------------------------------------------------------------


def _streams(seed: int) -> tuple[np.random.Generator, np.random.Generator]:
    """(assignment, analysis) generators independent of the population's own stream."""
    assignment, analysis = spawn_generators(seed, 2)
    return assignment, analysis


def _assign(population: ExperimentPopulation, seed: int, *, scheme: str = "complete") -> np.ndarray:
    rng = _streams(seed)[0]
    n = len(population.units)
    if scheme == "stratified":
        labels = stratified_randomisation(population.units["segment"], rng=rng)
    elif scheme == "cluster":
        labels = cluster_randomisation(population.units["cluster_id"], rng=rng)
    else:
        labels = complete_randomisation(n, rng=rng)
    return np.asarray(labels == "treatment", dtype=int)


def recovery_study(
    config: ExperimentScenarioConfig,
    *,
    n_replications: int,
    first_seed: int = 100_000,
) -> dict[str, ReplicateSummary]:
    """Valid design: conversion (absolute, relative) and revenue effects; stratified vs not."""
    rows: dict[str, tuple[list[float], list[float], list[bool], list[float]]] = {
        k: ([], [], [], [])
        for k in (
            "conversion: difference (Newcombe)",
            "conversion: relative lift (Katz)",
            "revenue: difference (Welch)",
            "conversion, stratified assignment: Welch-type SE",
            "conversion, stratified assignment: stratified estimator",
        )
    }

    def add(key: str, estimate: float, truth: float, lo: float, hi: float, se: float) -> None:
        rows[key][0].append(estimate)
        rows[key][1].append(truth)
        rows[key][2].append(lo <= truth <= hi)
        rows[key][3].append(se)

    for r in range(n_replications):
        population = generate_experiment_population(replace(config, seed=first_seed + r))
        truth = population.truth
        pot = population.potential
        relative_truth = float(pot["p1"].mean() / pot["p0"].mean() - 1)
        frame = realise_experiment(population, _assign(population, first_seed + r)).analysis_frame
        conv = an.analyse_binary(frame["converted"], frame["arm"])
        e = conv.effect
        add(
            "conversion: difference (Newcombe)",
            e.estimate,
            truth.ate_conversion,
            e.ci_lower,
            e.ci_upper,
            e.standard_error,
        )
        add(
            "conversion: relative lift (Katz)",
            conv.relative_lift,
            relative_truth,
            conv.relative_ci_lower,
            conv.relative_ci_upper,
            math.nan,
        )
        rev = an.analyse_continuous(frame["revenue"], frame["arm"]).effect
        add(
            "revenue: difference (Welch)",
            rev.estimate,
            truth.ate_revenue,
            rev.ci_lower,
            rev.ci_upper,
            rev.standard_error,
        )
        strat = realise_experiment(
            population, _assign(population, first_seed + r, scheme="stratified")
        ).analysis_frame
        plain = an.analyse_binary(strat["converted"], strat["arm"]).effect
        add(
            "conversion, stratified assignment: Welch-type SE",
            plain.estimate,
            truth.ate_conversion,
            plain.ci_lower,
            plain.ci_upper,
            plain.standard_error,
        )
        s = an.stratified_difference(strat["converted"], strat["arm"], strat["segment"])
        add(
            "conversion, stratified assignment: stratified estimator",
            s.estimate,
            truth.ate_conversion,
            s.ci_lower,
            s.ci_upper,
            s.standard_error,
        )
    return {
        k: summarise_replicates(k, est, tru, cov, se) for k, (est, tru, cov, se) in rows.items()
    }


def aa_study(
    *, n_units: int, n_replications: int, first_seed: int = 110_000
) -> list[RejectionRate]:
    """A/A tests (no effect anywhere): false-positive rates of the standard analyses."""
    config = ExperimentScenarioConfig(
        n_units=n_units, conversion_effect=(0.0, 0.0, 0.0), revenue_effect=0.0
    )
    z_rej, w_rej, boot_rej = [], [], []
    for r in range(n_replications):
        population = generate_experiment_population(replace(config, seed=first_seed + r))
        frame = realise_experiment(population, _assign(population, first_seed + r)).analysis_frame
        z_rej.append(an.analyse_binary(frame["converted"], frame["arm"]).significant)
        rev = an.analyse_continuous(
            frame["revenue"], frame["arm"], rng=_streams(first_seed + r)[1], n_bootstrap=999
        )
        w_rej.append(rev.significant)
        assert rev.bootstrap_ci is not None
        boot_rej.append(not rev.bootstrap_ci[0] <= 0.0 <= rev.bootstrap_ci[1])
    return [
        rejection_rate(f"conversion z-test ({n_units} units)", z_rej),
        rejection_rate(f"revenue Welch t-test ({n_units} units)", w_rej),
        rejection_rate(f"revenue BCa bootstrap CI excludes 0 ({n_units} units)", boot_rej),
    ]


@dataclass(frozen=True)
class PowerCheck:
    design: str
    n_per_arm: int
    target_power: float
    empirical: RejectionRate


def planning_validation_study(
    *, n_replications: int, first_seed: int = 120_000
) -> list[PowerCheck]:
    """Simulated power at the planned sample sizes (binary and continuous metrics)."""
    rng = seeded_generator(first_seed)
    checks: list[PowerCheck] = []
    for baseline, mde in ((0.05, 0.01), (0.10, 0.01), (0.10, 0.02), (0.30, 0.03)):
        plan = plan_binary(baseline, mde_absolute=mde)
        rejected = []
        for _ in range(n_replications):
            s_t = int(rng.binomial(plan.n_treatment, baseline + mde))
            s_c = int(rng.binomial(plan.n_control, baseline))
            y = np.concatenate(
                [
                    np.repeat([1.0, 0.0], [s_t, plan.n_treatment - s_t]),
                    np.repeat([1.0, 0.0], [s_c, plan.n_control - s_c]),
                ]
            )
            arm = np.repeat(["treatment", "control"], [plan.n_treatment, plan.n_control])
            rejected.append(an.analyse_binary(y, arm).significant)
        checks.append(
            PowerCheck(
                f"binary: baseline {baseline:g}, MDE +{mde:g}",
                plan.n_control,
                0.8,
                rejection_rate("empirical power", rejected),
            )
        )
    for sd, mde in ((10.0, 1.0), (10.0, 0.5)):
        plan = plan_continuous(50.0, sd, mde_absolute=mde)
        rejected = []
        for _ in range(n_replications):
            y_t = rng.normal(50.0 + mde, sd, plan.n_treatment)
            y_c = rng.normal(50.0, sd, plan.n_control)
            arm = np.repeat(["treatment", "control"], [plan.n_treatment, plan.n_control])
            rejected.append(an.analyse_continuous(np.concatenate([y_t, y_c]), arm).significant)
        checks.append(
            PowerCheck(
                f"continuous: SD {sd:g}, MDE +{mde:g}",
                plan.n_control,
                0.8,
                rejection_rate("empirical power", rejected),
            )
        )
    return checks


def srm_size_study(*, n_units: int, n_replications: int) -> list[RejectionRate]:
    """SRM false-alarm rate under correct hash-based assignment (new salt per replicate)."""
    at_001, at_05 = [], []
    ids = np.arange(n_units)
    for r in range(n_replications):
        labels = hash_assignment(ids, experiment="srm-size-study", salt=str(r))
        p = srm_from_labels(labels, HALF).p_value
        at_001.append(rejects(p, 0.001))
        at_05.append(rejects(p, 0.05))
    return [
        rejection_rate("SRM test p <= 0.001 (default threshold)", at_001),
        rejection_rate("SRM test p <= 0.05", at_05),
    ]


@dataclass(frozen=True)
class SRMCauseRow:
    mechanism: str
    loss_share: float
    detection: RejectionRate
    bias: ReplicateSummary


def srm_cause_study(
    *,
    loss_shares: Sequence[float],
    n_replications: int,
    n_units: int = 20_000,
    first_seed: int = 130_000,
) -> list[SRMCauseRow]:
    """Same SRM, different causes: random loss is benign, engagement-related loss biases."""
    rows = []
    for mechanism in ("random", "low_engagement", "inactive"):
        for share in loss_shares:
            detected, estimates, truths = [], [], []
            for r in range(n_replications):
                config = ExperimentScenarioConfig(
                    seed=first_seed + r,
                    n_units=n_units,
                    treatment_loss_share=share,
                    loss_mechanism=mechanism,
                )
                population = generate_experiment_population(config)
                frame = realise_experiment(
                    population, _assign(population, first_seed + r)
                ).analysis_frame
                detected.append(srm_from_labels(frame["arm"], HALF).detected)
                estimates.append(
                    an.analyse_binary(frame["converted"], frame["arm"]).effect.estimate
                )
                truths.append(population.truth.ate_conversion)
            rows.append(
                SRMCauseRow(
                    mechanism,
                    share,
                    rejection_rate(f"{mechanism} loss {share:.1%}", detected),
                    summarise_replicates(f"{mechanism} loss {share:.1%}", estimates, truths),
                )
            )
    return rows


def noncompliance_study(
    config: ExperimentScenarioConfig,
    *,
    n_replications: int,
    first_seed: int = 140_000,
) -> dict[str, ReplicateSummary]:
    rows: dict[str, tuple[list[float], list[float], list[bool], list[float]]] = {
        k: ([], [], [], [])
        for k in (
            "ITT (vs ITT truth)",
            "CACE / Wald (vs complier effect)",
            "as-treated (vs ATE)",
            "per-protocol (vs ATE)",
        )
    }
    for r in range(n_replications):
        population = generate_experiment_population(replace(config, seed=first_seed + r))
        truth = population.truth
        z = _assign(population, first_seed + r)
        frame = realise_experiment(population, z).analysis_frame
        result = co.noncompliance_analysis(
            frame["converted"], (frame["arm"] == "treatment").astype(float), frame["exposed"]
        )
        for key, estimate, target in (
            ("ITT (vs ITT truth)", result.itt, truth.itt_conversion),
            ("CACE / Wald (vs complier effect)", result.cace, truth.cace_conversion),
            ("as-treated (vs ATE)", result.as_treated, truth.ate_conversion),
            ("per-protocol (vs ATE)", result.per_protocol, truth.ate_conversion),
        ):
            rows[key][0].append(estimate.estimate)
            rows[key][1].append(target)
            rows[key][2].append(estimate.covers(target))
            rows[key][3].append(estimate.standard_error)
    return {k: summarise_replicates(k, e, t, c, s) for k, (e, t, c, s) in rows.items()}


@dataclass(frozen=True)
class AttritionStudy:
    complete_case: ReplicateSummary
    lee_contains_truth: RejectionRate
    lee_ci_contains_truth: RejectionRate
    lee_mean_width: float
    manski_contains_truth: RejectionRate
    manski_mean_width: float
    mean_response_control: float
    mean_response_treatment: float


def attrition_study(
    config: ExperimentScenarioConfig,
    *,
    n_replications: int,
    n_bootstrap: int = 199,
    first_seed: int = 150_000,
) -> AttritionStudy:
    cc_est, cc_truth, cc_cov, cc_se = [], [], [], []
    lee_in, lee_ci_in, lee_w, man_in, man_w, resp_c, resp_t = [], [], [], [], [], [], []
    for r in range(n_replications):
        population = generate_experiment_population(replace(config, seed=first_seed + r))
        truth = population.truth
        frame = realise_experiment(population, _assign(population, first_seed + r)).analysis_frame
        z = (frame["arm"] == "treatment").astype(float).to_numpy()
        y = frame["converted"].to_numpy(float)
        cc = co.complete_case_difference(y, z)
        cc_est.append(cc.estimate)
        cc_truth.append(truth.itt_conversion)
        cc_cov.append(cc.covers(truth.itt_conversion))
        cc_se.append(cc.standard_error)
        lee = co.lee_bounds(y, z, rng=_streams(first_seed + r)[1], n_bootstrap=n_bootstrap)
        target = truth.always_observed_itt_conversion
        lee_in.append(lee.lower <= target <= lee.upper)
        assert lee.ci_lower is not None and lee.ci_upper is not None
        lee_ci_in.append(lee.ci_lower <= target <= lee.ci_upper)
        lee_w.append(lee.upper - lee.lower)
        lo, hi = co.manski_bounds(y, z, lower=0.0, upper=1.0)
        man_in.append(lo <= truth.itt_conversion <= hi)
        man_w.append(hi - lo)
        summary = co.attrition_summary(y, z)
        resp_c.append(summary.response_control)
        resp_t.append(summary.response_treatment)
    return AttritionStudy(
        complete_case=summarise_replicates(
            "complete-case difference (vs ITT)", cc_est, cc_truth, cc_cov, cc_se
        ),
        lee_contains_truth=rejection_rate(
            "Lee point bounds contain always-observed effect", lee_in
        ),
        lee_ci_contains_truth=rejection_rate("Lee bootstrap 95% CI contains it", lee_ci_in),
        lee_mean_width=float(np.mean(lee_w)),
        manski_contains_truth=rejection_rate("Manski bounds contain ITT", man_in),
        manski_mean_width=float(np.mean(man_w)),
        mean_response_control=float(np.mean(resp_c)),
        mean_response_treatment=float(np.mean(resp_t)),
    )


def clustering_study(
    *,
    n_clusters: int,
    cluster_sd: float,
    n_replications: int,
    n_units: int = 6_000,
    first_seed: int = 160_000,
) -> list[RejectionRate]:
    """A/A cluster-randomised experiment: false-positive rates of three analyses."""
    config = ExperimentScenarioConfig(
        n_units=n_units,
        n_clusters=n_clusters,
        cluster_sd=cluster_sd,
        conversion_effect=(0.0, 0.0, 0.0),
    )
    naive, robust, means = [], [], []
    for r in range(n_replications):
        population = generate_experiment_population(replace(config, seed=first_seed + r))
        frame = realise_experiment(
            population, _assign(population, first_seed + r, scheme="cluster")
        ).analysis_frame
        result = an.clustered_difference(frame["converted"], frame["arm"], frame["cluster_id"])
        naive.append(not result.naive.covers(0.0))
        robust.append(not result.cluster_robust.covers(0.0))
        means.append(not result.cluster_means.covers(0.0))
    tag = f"{n_clusters} clusters, cluster SD {cluster_sd:g}"
    return [
        rejection_rate(f"naive unit-level Welch ({tag})", naive),
        rejection_rate(f"CR1 cluster-robust, t(G-1) ({tag})", robust),
        rejection_rate(f"Welch on cluster means ({tag})", means),
    ]


def post_selection_study(*, n_replications: int, first_seed: int = 170_000) -> ReplicateSummary:
    """Conditioning on a post-randomisation variable ('engaged'), scored against the ATE."""
    estimates, truths = [], []
    for r in range(n_replications):
        population = generate_experiment_population(ExperimentScenarioConfig(seed=first_seed + r))
        frame = realise_experiment(population, _assign(population, first_seed + r)).analysis_frame
        engaged = frame["engaged"] == 1
        estimates.append(
            an.analyse_binary(
                frame.loc[engaged, "converted"], frame.loc[engaged, "arm"]
            ).effect.estimate
        )
        truths.append(population.truth.ate_conversion)
    return summarise_replicates("difference among 'engaged' users (vs ATE)", estimates, truths)


@dataclass(frozen=True)
class MultipleTestingRow:
    procedure: str
    familywise_error: RejectionRate
    false_discovery_rate: float
    power: float


def multiple_testing_study(
    *,
    n_metrics: int = 10,
    n_effects: int = 3,
    effect_size: float = 0.1,
    n_per_arm: int = 1_000,
    n_replications: int,
    first_seed: int = 180_000,
) -> list[MultipleTestingRow]:
    """``n_metrics`` independent Welch tests; the first ``n_effects`` have a true effect."""
    rng = seeded_generator(first_seed)
    procedures = ("unadjusted", "bonferroni", "holm", "benjamini-hochberg")
    any_false: dict[str, list[bool]] = {p: [] for p in procedures}
    fdp: dict[str, list[float]] = {p: [] for p in procedures}
    hits: dict[str, list[float]] = {p: [] for p in procedures}
    truth = np.arange(n_metrics) < n_effects
    arm = np.repeat(["treatment", "control"], n_per_arm)
    for _ in range(n_replications):
        p_values = []
        for m in range(n_metrics):
            shift = effect_size if truth[m] else 0.0
            y = np.concatenate(
                [rng.normal(10.0 + shift, 1.0, n_per_arm), rng.normal(10.0, 1.0, n_per_arm)]
            )
            p_values.append(an.analyse_continuous(y, arm).p_value)
        for procedure in procedures:
            if procedure == "unadjusted":
                rejected = np.asarray(p_values) <= 0.05  # rule: p <= alpha
            else:
                rejected = adjust_p_values(p_values, method=procedure).rejected
            false = rejected & ~truth
            any_false[procedure].append(bool(false.any()))
            fdp[procedure].append(int(false.sum()) / max(1, int(rejected.sum())))
            hits[procedure].append(int((rejected & truth).sum()) / max(1, int(truth.sum())))
    return [
        MultipleTestingRow(
            p, rejection_rate(p, any_false[p]), float(np.mean(fdp[p])), float(np.mean(hits[p]))
        )
        for p in procedures
    ]


def decision_operating_characteristics(
    *,
    effects: Sequence[float],
    baseline: float = 0.10,
    delta: float = 0.01,
    n_replications: int,
    first_seed: int = 190_000,
) -> tuple[ExperimentPlan, pd.DataFrame]:
    """Decision frequencies at the sample size planned for ``delta`` (80% power)."""
    plan = plan_binary(baseline, mde_absolute=delta)
    rng = seeded_generator(first_seed)
    arm = np.repeat(["treatment", "control"], [plan.n_treatment, plan.n_control])
    rows = []
    for effect in effects:
        counts = {"LAUNCH": 0, "ITERATE": 0, "HOLD": 0, "STOP": 0}
        for _ in range(n_replications):
            s_t = int(rng.binomial(plan.n_treatment, baseline + effect))
            s_c = int(rng.binomial(plan.n_control, baseline))
            y = np.concatenate(
                [
                    np.repeat([1.0, 0.0], [s_t, plan.n_treatment - s_t]),
                    np.repeat([1.0, 0.0], [s_c, plan.n_control - s_c]),
                ]
            )
            counts[
                recommend(an.analyse_binary(y, arm), minimum_worthwhile_effect=delta).decision
            ] += 1
        rows.append({"true effect": effect, **{k: v / n_replications for k, v in counts.items()}})
    return plan, pd.DataFrame(rows)


__all__ = [
    "M2_DESIGN",
    "SCENARIO_DESIGN",
    "AttritionStudy",
    "M2ExperimentAnalysis",
    "M2ReplicationStudy",
    "MultipleTestingRow",
    "PowerCheck",
    "SRMCauseRow",
    "ScenarioRun",
    "aa_study",
    "analyse_m2_experiment",
    "analyse_scenario_case",
    "attrition_study",
    "clustering_study",
    "decision_operating_characteristics",
    "m2_experiment_frame",
    "m2_replication_study",
    "multiple_testing_study",
    "noncompliance_study",
    "planning_validation_study",
    "post_selection_study",
    "recovery_study",
    "srm_cause_study",
    "srm_size_study",
]
