"""Synthetic randomised-experiment populations with known potential outcomes (Milestone 6).

Additive to the M2 pipeline: nothing here reads or modifies an M2 table,
generator or contract. NexaMarket question: *does the new recommendation
algorithm raise 14-day conversion among eligible users?*

The generator separates three things a real experiment also separates:

1. **The population** (:func:`generate_experiment_population`): units, their
   observable pre-treatment covariates, and — evaluation-only — their latent
   engagement and *both* potential outcomes, compliance type, response
   (attrition) behaviour and data-loss exposure. Every random draw happens
   here, so the population fixes all potential outcomes.
2. **The assignment**: a 0/1 vector chosen by the caller (normally with
   :mod:`decision_intelligence.experimentation.assignment`). The data layer
   never assigns treatment itself.
3. **The realisation** (:func:`realise_experiment`): deterministic given the
   population and assignment — the logged tables an analyst would receive,
   including deliberately injected defects (post-randomisation data loss,
   duplicate log rows, cross-over exposure, attrition).

Ground truth (:class:`ExperimentScenarioTruth`) is computed from the
potential outcomes before any assignment and is never an input to analysis.
Effects are *expected* effects averaged over the generated units
(``mean(p1 - p0)``), the estimand repeated-sampling studies score against.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from decision_intelligence.common.seeding import seeded_generator
from decision_intelligence.common.validation import FloatArray

SEGMENTS = ("casual", "regular", "power")
ASSIGNMENT_TIME = pd.Timestamp("2025-03-03")
OBSERVATION_WINDOW_DAYS = 14

LossMechanism = Literal["random", "low_engagement", "inactive"]


def _expit(x: FloatArray) -> FloatArray:
    return np.asarray(1.0 / (1.0 + np.exp(-x)), dtype=np.float64)


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


@dataclass(frozen=True)
class ExperimentScenarioConfig:
    """Population and defect switches. Defaults: a clean, unit-randomised experiment."""

    seed: int = 20260601
    n_units: int = 20_000
    segment_shares: tuple[float, float, float] = (0.5, 0.35, 0.15)
    baseline_conversion: tuple[float, float, float] = (0.06, 0.12, 0.20)
    conversion_effect: tuple[float, float, float] = (0.01, 0.02, 0.04)
    """Absolute effect of *exposure* on conversion probability, by segment."""
    engagement_on_conversion: float = 0.6
    revenue_log_mean: float = 3.5
    revenue_log_sd: float = 1.0
    revenue_effect: float = 0.05
    """Multiplicative effect of exposure on spend among converters."""
    # Clustering (e.g. users within cities); 0 clusters = no cluster structure.
    n_clusters: int = 0
    cluster_sd: float = 0.0
    """SD of a cluster random intercept on the conversion log-odds."""
    # Noncompliance (exposure differs from assignment).
    never_taker_share: float = 0.0
    always_taker_share: float = 0.0
    compliance_engagement_link: float = 0.0
    """>0: engaged users are more likely to comply (makes as-treated analyses biased)."""
    # Attrition: outcome not observed. Monotone: treatment never reduces response.
    attrition_control: float = 0.0
    attrition_treatment: float = 0.0
    attrition_engagement_link: float = 0.0
    """>0: low-engagement users drop out more (attrition related to outcomes)."""
    # Post-randomisation data loss in the treatment arm (produces sample ratio mismatch).
    treatment_loss_share: float = 0.0
    loss_mechanism: LossMechanism = "random"
    # Logging defects.
    duplicate_share: float = 0.0
    """Share of units logged twice; half of the duplicates carry the opposite arm."""
    # Guardrails.
    latency_shift_ms: float = 0.0
    error_rate_control: float = 0.01
    error_rate_shift: float = 0.0

    def __post_init__(self) -> None:
        if self.n_units < 20:
            raise ValueError("n_units must be at least 20")
        if abs(sum(self.segment_shares) - 1.0) > 1e-9 or min(self.segment_shares) <= 0:
            raise ValueError("segment_shares must be positive and sum to 1")
        for p in self.baseline_conversion:
            if not 0.0 < p < 1.0:
                raise ValueError("baseline conversion rates must lie in (0, 1)")
        shares = (
            self.never_taker_share, self.always_taker_share, self.attrition_control,
            self.attrition_treatment, self.treatment_loss_share, self.duplicate_share,
        )
        if any(not 0.0 <= s < 1.0 for s in shares):
            raise ValueError("shares must lie in [0, 1)")
        if self.never_taker_share + self.always_taker_share >= 1.0:
            raise ValueError("there must be compliers")
        if self.attrition_treatment < self.attrition_control:
            raise ValueError(
                "attrition_treatment must be >= attrition_control (monotone response, "
                "required for Lee bounds)"
            )
        if self.n_clusters < 0 or (0 < self.n_clusters < 4):
            raise ValueError("n_clusters must be 0 or at least 4")


@dataclass(frozen=True)
class ExperimentScenarioTruth:
    """Evaluation-only estimands, computed from the potential outcomes."""

    ate_conversion: float
    """Average effect of *exposure* on conversion probability (all units)."""
    itt_conversion: float
    """Average effect of *assignment* on conversion probability (noncompliance included)."""
    cace_conversion: float
    """Average effect of exposure among compliers."""
    complier_share: float
    ate_revenue: float
    itt_revenue: float
    always_observed_itt_conversion: float
    """Effect of assignment among units whose outcome is observed in either arm (Lee's target)."""
    ate_by_segment: dict[str, float]
    latency_shift_ms: float
    error_rate_shift: float
    engaged_effect: float
    """Effect of exposure on the post-randomisation 'engaged' indicator (selection variable)."""


@dataclass(frozen=True)
class ExperimentPopulation:
    """Units with observable covariates and (evaluation-only) potential outcomes."""

    config: ExperimentScenarioConfig
    units: pd.DataFrame
    """Observable, pre-treatment: unit_id, segment, cluster_id, pre_sessions, pre_revenue."""
    potential: pd.DataFrame = field(repr=False)
    """Evaluation-only: latent engagement and every potential outcome and behaviour."""
    truth: ExperimentScenarioTruth

    @property
    def randomisation_units(self) -> np.ndarray:
        """Cluster ids when the population is clustered, else unit ids."""
        column = "cluster_id" if self.config.n_clusters else "unit_id"
        return self.units[column].to_numpy()


def generate_experiment_population(
    config: ExperimentScenarioConfig | None = None,
) -> ExperimentPopulation:
    config = config or ExperimentScenarioConfig()
    rng = seeded_generator(config.seed)
    n = config.n_units

    segment_index = rng.choice(3, size=n, p=np.asarray(config.segment_shares))
    segment = np.asarray(SEGMENTS)[segment_index]
    engagement = rng.standard_normal(n)
    if config.n_clusters:
        cluster_id = rng.integers(0, config.n_clusters, size=n)
        cluster_effect = rng.normal(0.0, config.cluster_sd, size=config.n_clusters)[cluster_id]
    else:
        cluster_id = np.zeros(n, dtype=np.int64)
        cluster_effect = np.zeros(n)

    # Observable pre-treatment covariates (noisy proxies of engagement).
    pre_sessions = rng.poisson(np.exp(1.2 + 0.5 * engagement + 0.2 * segment_index))
    pre_revenue = np.where(
        rng.random(n) < _expit(-1.0 + 0.6 * engagement),
        np.exp(rng.normal(3.0 + 0.3 * engagement, 0.8)),
        0.0,
    )

    # Conversion potential outcomes, coupled through one uniform per unit.
    base_logit = np.array([_logit(p) for p in config.baseline_conversion])[segment_index]
    p0 = _expit(base_logit + config.engagement_on_conversion * engagement + cluster_effect)
    p1 = np.minimum(p0 + np.asarray(config.conversion_effect)[segment_index], 0.99)
    u_conversion = rng.random(n)
    y0 = (u_conversion < p0).astype(float)
    y1 = (u_conversion < p1).astype(float)

    spend = np.exp(rng.normal(config.revenue_log_mean + 0.3 * engagement, config.revenue_log_sd))
    expected_spend = np.exp(
        config.revenue_log_mean + 0.3 * engagement + config.revenue_log_sd**2 / 2
    )
    rev0, rev1 = y0 * spend, y1 * spend * (1 + config.revenue_effect)
    erev0 = p0 * expected_spend
    erev1 = p1 * expected_spend * (1 + config.revenue_effect)

    # Compliance types by a latent score (engaged users comply more when the link is > 0).
    score = config.compliance_engagement_link * engagement + rng.logistic(size=n)
    order = np.argsort(np.argsort(score)) / n  # rank quantile in [0, 1)
    never = order < config.never_taker_share
    always = order >= 1.0 - config.always_taker_share
    d0 = always.astype(float)
    d1 = (~never).astype(float)

    # Monotone response (outcome observed) behaviour; treatment never increases response.
    def response_rate(rate: float) -> FloatArray:
        weight = 2 * _expit(-config.attrition_engagement_link * engagement)
        return np.asarray(np.clip(1.0 - rate * weight, 0.0, 1.0), dtype=np.float64)

    u_response = rng.random(n)
    r0 = (u_response < response_rate(config.attrition_control)).astype(float)
    r1 = (u_response < response_rate(config.attrition_treatment)).astype(float)
    # Same uniform and rate_treatment >= rate_control: r1 <= r0 for every unit (monotone).

    # Post-randomisation loss (applies only if assigned to treatment):
    # random (e.g. a logging outage), low_engagement (e.g. a client crash on low-end
    # devices) or inactive (a "bot filter" dropping treated users who did not convert).
    u_loss = rng.random(n)
    if config.loss_mechanism == "random":
        lost_if_treated = u_loss < config.treatment_loss_share
    elif config.loss_mechanism == "low_engagement":
        weight = 2 * _expit(-2.0 * engagement)
        lost_if_treated = u_loss < np.clip(config.treatment_loss_share * weight, 0, 1)
    else:
        y_if_assigned = np.where(d1 == 1, y1, y0)
        rate = min(1.0, config.treatment_loss_share / max(1e-9, 1 - float(y_if_assigned.mean())))
        lost_if_treated = (u_loss < rate) & (y_if_assigned == 0)

    # Post-randomisation engagement indicator (affected by exposure).
    u_engaged = rng.random(n)
    engaged0 = (u_engaged < _expit(-0.5 + 0.8 * engagement)).astype(float)
    engaged1 = (u_engaged < _expit(0.5 + 0.8 * engagement)).astype(float)

    latency_base = 180.0 * np.exp(0.25 * rng.standard_normal(n))
    u_error = rng.random(n)
    error0 = (u_error < config.error_rate_control).astype(float)
    error1 = (u_error < config.error_rate_control + config.error_rate_shift).astype(float)

    duplicated = rng.random(n) < config.duplicate_share
    duplicate_conflicting = duplicated & (rng.random(n) < 0.5)

    units = pd.DataFrame(
        {
            "unit_id": np.arange(1, n + 1),
            "segment": segment,
            "cluster_id": cluster_id,
            "pre_sessions": pre_sessions.astype(np.int64),
            "pre_revenue": np.round(pre_revenue, 2),
        }
    )
    potential = pd.DataFrame(
        {
            "engagement": engagement, "p0": p0, "p1": p1, "y0": y0, "y1": y1,
            "rev0": rev0, "rev1": rev1, "erev0": erev0, "erev1": erev1,
            "d0": d0, "d1": d1, "r0": r0, "r1": r1, "lost_if_treated": lost_if_treated,
            "engaged0": engaged0, "engaged1": engaged1, "latency_base": latency_base,
            "error0": error0, "error1": error1, "duplicated": duplicated,
            "duplicate_conflicting": duplicate_conflicting,
        }
    )

    # Expected outcomes under assignment z: exposure d(z) determines the outcome.
    p_z0 = np.where(d0 == 1, p1, p0)
    p_z1 = np.where(d1 == 1, p1, p0)
    erev_z0 = np.where(d0 == 1, erev1, erev0)
    erev_z1 = np.where(d1 == 1, erev1, erev0)
    compliers = (d1 == 1) & (d0 == 0)
    always_observed = (r0 == 1) & (r1 == 1)
    truth = ExperimentScenarioTruth(
        ate_conversion=float(np.mean(p1 - p0)),
        itt_conversion=float(np.mean(p_z1 - p_z0)),
        cace_conversion=float(np.mean((p1 - p0)[compliers])),
        complier_share=float(compliers.mean()),
        ate_revenue=float(np.mean(erev1 - erev0)),
        itt_revenue=float(np.mean(erev_z1 - erev_z0)),
        always_observed_itt_conversion=float(np.mean((p_z1 - p_z0)[always_observed])),
        ate_by_segment={
            s: float(np.mean((p1 - p0)[segment == s])) for s in SEGMENTS
        },
        latency_shift_ms=config.latency_shift_ms,
        error_rate_shift=config.error_rate_shift,
        engaged_effect=float(
            np.mean(_expit(0.5 + 0.8 * engagement) - _expit(-0.5 + 0.8 * engagement))
        ),
    )
    return ExperimentPopulation(config=config, units=units, potential=potential, truth=truth)


@dataclass(frozen=True)
class RealisedExperiment:
    """What an analyst receives after the experiment ran."""

    assignment_log: pd.DataFrame
    """Server-side assignment record: one row per assignment event (duplicates included)."""
    analysis_frame: pd.DataFrame
    """Telemetry joined to assignment: units lost after randomisation are absent;
    attrited outcomes are NaN; duplicate log rows appear as extra rows."""


def realise_experiment(
    population: ExperimentPopulation, assignment: np.ndarray
) -> RealisedExperiment:
    """Deterministic: observed tables given the population and a 0/1 assignment per unit."""
    z = np.asarray(assignment)
    units, pot = population.units, population.potential
    if z.shape != (len(units),) or not np.isin(z, (0, 1)).all():
        raise ValueError("assignment must be a 0/1 vector with one entry per unit")
    d = np.where(z == 1, pot["d1"], pot["d0"])
    converted = np.where(d == 1, pot["y1"], pot["y0"])
    revenue = np.where(d == 1, pot["rev1"], pot["rev0"])
    observed = np.where(z == 1, pot["r1"], pot["r0"]) == 1
    engaged = np.where(d == 1, pot["engaged1"], pot["engaged0"])
    latency = pot["latency_base"].to_numpy() + population.config.latency_shift_ms * d
    error = np.where(d == 1, pot["error1"], pot["error0"])
    arm = np.where(z == 1, "treatment", "control")

    assignment_log = pd.DataFrame(
        {
            "unit_id": units["unit_id"].to_numpy(),
            "cluster_id": units["cluster_id"].to_numpy(),
            "arm": arm,
            "assigned_at": ASSIGNMENT_TIME,
        }
    )
    frame = units.assign(
        arm=arm,
        exposed=d,
        engaged=engaged,
        converted=np.where(observed, converted, np.nan),
        revenue=np.where(observed, np.round(revenue, 2), np.nan),
        latency_ms=np.round(latency, 1),
        error=error,
        observation_end=ASSIGNMENT_TIME + pd.Timedelta(OBSERVATION_WINDOW_DAYS, unit="D"),
    )
    lost = pot["lost_if_treated"].to_numpy() & (z == 1)
    frame = frame.loc[~lost]

    duplicated = pot["duplicated"].to_numpy()
    conflicting = pot["duplicate_conflicting"].to_numpy()
    if duplicated.any():
        # Duplicate log rows are appended by repeating rows (not concatenating frames).
        flip = {"treatment": "control", "control": "treatment"}
        n_log = len(assignment_log)
        rows = np.concatenate([np.arange(n_log), np.flatnonzero(duplicated)])
        assignment_log = assignment_log.iloc[rows].reset_index(drop=True)
        extra = np.arange(n_log, len(assignment_log))
        arms = assignment_log["arm"].to_numpy(dtype=object)
        flip_log = extra[conflicting[duplicated]]
        arms[flip_log] = [flip[a] for a in arms[flip_log]]
        times = assignment_log["assigned_at"].to_numpy().copy()
        times[extra] = np.datetime64(ASSIGNMENT_TIME + pd.Timedelta(1, unit="h"), "ns")
        assignment_log = assignment_log.assign(arm=arms, assigned_at=times)

        n_frame = len(frame)
        keep = duplicated[~lost]
        rows = np.concatenate([np.arange(n_frame), np.flatnonzero(keep)])
        frame = frame.iloc[rows].reset_index(drop=True)
        extra = np.arange(n_frame, len(frame))
        arms = frame["arm"].to_numpy(dtype=object)
        flip_frame = extra[conflicting[~lost][keep]]
        arms[flip_frame] = [flip[a] for a in arms[flip_frame]]
        frame = frame.assign(arm=arms)
    return RealisedExperiment(
        assignment_log=assignment_log, analysis_frame=frame.reset_index(drop=True)
    )


__all__ = [
    "ASSIGNMENT_TIME",
    "ExperimentPopulation",
    "ExperimentScenarioConfig",
    "ExperimentScenarioTruth",
    "OBSERVATION_WINDOW_DAYS",
    "RealisedExperiment",
    "SEGMENTS",
    "generate_experiment_population",
    "realise_experiment",
]
