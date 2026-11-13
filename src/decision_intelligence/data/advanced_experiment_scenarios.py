"""Synthetic scenarios for advanced experimentation (Milestone 7), with known ground truth.

Additive: nothing here reads or modifies an M2 or M6 generator, table or
contract. Clustered experiments reuse the M6 generator
(:mod:`decision_intelligence.data.experiment_scenarios`, ``n_clusters``) unchanged.

Two NexaMarket scenarios:

1. **Pre-period metrics for variance reduction and sequential monitoring**
   (:func:`generate_cuped_population`). Each user has a latent spending
   propensity ``lambda_i``; 28-day spend *before* assignment (``pre_revenue``)
   and 14-day spend *after* it are independent noisy measurements of it, so the
   pre-period metric predicts the outcome (correlation about 0.6 by default)
   without being affected by treatment. New users have no pre-period history
   (``pre_revenue`` is missing). Users arrive over ``n_days`` days, which a
   sequential analysis uses as its accrual order. The population also
   carries two covariates that **must not** be used for adjustment, to
   demonstrate invalid analyses:

   * ``in_experiment_sessions`` — measured after assignment and raised by the
     treatment (post-randomisation covariate);
   * ``pre_revenue_leaky`` — a "pre-period" window that wrongly ends 7 days
     *after* assignment, so it contains part of the treatment effect.

   The treatment effect on revenue is heterogeneous and proportional to
   ``lambda_i`` (bigger spenders gain more), so it is correlated with the
   covariate — the situation where regression adjustment with interactions
   (Lin 2013) and CUPED's single slope can differ.

2. **A marketplace switchback** (:func:`generate_switchback_market`): an
   hourly series (daily seasonality, AR(1) noise) in which the platform-wide
   treatment state is switched between time periods. The treatment acts in
   the hour it is on (*direct* effect) and for a few hours after it is
   switched off (*carryover*, e.g. drivers repositioned by a pricing
   change). Potential outcomes depend on the whole assignment path, so the
   estimand is the *global* average treatment effect — all-on vs all-off per
   hour, ``direct + sum(carryover)``.

Ground truth is computed from the generating process before any assignment
and is never an input to analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from decision_intelligence.common.seeding import seeded_generator
from decision_intelligence.common.validation import FloatArray

ASSIGNMENT_TIME = pd.Timestamp("2025-06-02")
PRE_PERIOD_DAYS = 28
OUTCOME_WINDOW_DAYS = 14
LEAKY_WINDOW_OVERLAP_DAYS = 7


def _expit(x: FloatArray) -> FloatArray:
    return np.asarray(1.0 / (1.0 + np.exp(-x)), dtype=np.float64)


# ---------------------------------------------------------------------------
# 1. Pre-period metrics (CUPED, sequential monitoring, Bayesian analysis)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CupedScenarioConfig:
    """Population with a pre-period version of the outcome. Defaults: a clean experiment."""

    seed: int = 20260701
    n_units: int = 20_000
    n_days: int = 28
    """Accrual period: users arrive uniformly over ``n_days`` days."""
    propensity_log_mean: float = 2.5
    propensity_log_sd: float = 0.7
    """SD of log spending propensity: larger values strengthen the pre/post correlation."""
    measurement_shape: float = 4.0
    """Gamma shape of the period-specific noise around the propensity (larger = less noise)."""
    new_user_share: float = 0.1
    """Share of users without pre-period history (``pre_revenue`` missing)."""
    revenue_effect: float = 0.6
    """Average additive effect on 14-day revenue (currency units per user)."""
    baseline_conversion: float = 0.12
    conversion_effect: float = 0.01
    """Average additive effect on the 14-day conversion probability."""
    sessions_effect: float = 1.5
    """Effect of treatment on in-experiment sessions (the post-randomisation covariate)."""

    def __post_init__(self) -> None:
        if self.n_units < 20 or self.n_days < 1:
            raise ValueError("n_units must be >= 20 and n_days >= 1")
        if not 0.0 <= self.new_user_share < 1.0:
            raise ValueError("new_user_share must lie in [0, 1)")
        if self.propensity_log_sd < 0 or self.measurement_shape <= 0:
            raise ValueError("propensity_log_sd must be >= 0 and measurement_shape > 0")
        if not 0.0 < self.baseline_conversion < 1.0:
            raise ValueError("baseline_conversion must lie in (0, 1)")


@dataclass(frozen=True)
class CupedScenarioTruth:
    """Evaluation-only estimands (sample averages over the generated users)."""

    ate_revenue: float
    ate_conversion: float
    pre_post_correlation: float
    """Correlation of ``pre_revenue`` with untreated revenue among users with history."""
    sessions_effect: float


@dataclass(frozen=True)
class CupedPopulation:
    config: CupedScenarioConfig
    units: pd.DataFrame
    """Observable before assignment: unit_id, arrival_day, is_new_user, pre_revenue,
    pre_converted (NaN for new users)."""
    potential: pd.DataFrame = field(repr=False)
    """Evaluation-only: propensity and both potential outcomes of every metric."""
    truth: CupedScenarioTruth


def generate_cuped_population(config: CupedScenarioConfig | None = None) -> CupedPopulation:
    config = config or CupedScenarioConfig()
    rng = seeded_generator(config.seed)
    n = config.n_units
    k = config.measurement_shape

    log_lambda = config.propensity_log_mean + config.propensity_log_sd * rng.standard_normal(n)
    lam = np.exp(log_lambda)
    arrival_day = rng.integers(1, config.n_days + 1, size=n)
    is_new = rng.random(n) < config.new_user_share

    # Spend in a window = propensity x window length factor x independent Gamma(k, 1/k) noise.
    pre_revenue = 2.0 * lam * rng.gamma(k, 1.0 / k, size=n)  # 28-day pre-period
    y0 = lam * rng.gamma(k, 1.0 / k, size=n)  # 14-day outcome window
    tau = config.revenue_effect * lam / lam.mean()  # heterogeneous, mean exactly revenue_effect
    y1 = y0 + tau
    first_week_share = rng.beta(4.0, 4.0, size=n)  # share of 14-day spend in its first 7 days

    # Conversion: probability rises with propensity; pre-period conversion shares the latent.
    z_lambda = (log_lambda - config.propensity_log_mean) / max(config.propensity_log_sd, 1e-12)
    base = math.log(config.baseline_conversion / (1 - config.baseline_conversion))
    p0 = _expit(base + 1.2 * z_lambda)
    p1 = np.minimum(p0 + config.conversion_effect * p0 / p0.mean(), 0.99)
    u = rng.random(n)
    c0, c1 = (u < p0).astype(float), (u < p1).astype(float)
    pre_converted = (rng.random(n) < _expit(base + 1.2 * z_lambda)).astype(float)

    sessions0 = rng.poisson(2.0 + 0.4 * lam).astype(float)
    sessions1 = sessions0 + rng.poisson(config.sessions_effect, size=n)

    units = pd.DataFrame(
        {
            "unit_id": np.arange(1, n + 1),
            "arrival_day": arrival_day.astype(np.int64),
            "is_new_user": is_new,
            "pre_revenue": np.where(is_new, np.nan, np.round(pre_revenue, 2)),
            "pre_converted": np.where(is_new, np.nan, pre_converted),
        }
    )
    potential = pd.DataFrame(
        {
            "propensity": lam, "revenue0": y0, "revenue1": y1, "p0": p0, "p1": p1,
            "converted0": c0, "converted1": c1, "sessions0": sessions0, "sessions1": sessions1,
            "first_week_share": first_week_share, "pre_revenue_true": pre_revenue,
        }
    )
    has_history = ~is_new
    truth = CupedScenarioTruth(
        ate_revenue=float(np.mean(y1 - y0)),
        ate_conversion=float(np.mean(p1 - p0)),
        pre_post_correlation=float(np.corrcoef(pre_revenue[has_history], y0[has_history])[0, 1]),
        sessions_effect=float(np.mean(sessions1 - sessions0)),
    )
    return CupedPopulation(config=config, units=units, potential=potential, truth=truth)


def realise_cuped_experiment(population: CupedPopulation, assignment: np.ndarray) -> pd.DataFrame:
    """Deterministic: the analysis frame given a 0/1 assignment per user.

    Columns: the pre-treatment units columns, ``arm``, ``revenue``, ``converted``, and the two
    covariates that must not be used — ``in_experiment_sessions`` (post-randomisation) and
    ``pre_revenue_leaky`` (window ends ``LEAKY_WINDOW_OVERLAP_DAYS`` after assignment).
    """
    z = np.asarray(assignment)
    if z.shape != (len(population.units),) or not np.isin(z, (0, 1)).all():
        raise ValueError("assignment must be a 0/1 vector with one entry per unit")
    pot = population.potential
    revenue = np.where(z == 1, pot["revenue1"], pot["revenue0"])
    first_week = revenue * pot["first_week_share"].to_numpy()
    # Leaky window: last 21 pre-period days (3/4 of the 28-day spend) plus the first post week.
    leaky = 0.75 * pot["pre_revenue_true"].to_numpy() + first_week
    return population.units.assign(
        arm=np.where(z == 1, "treatment", "control"),
        revenue=np.round(revenue, 2),
        converted=np.where(z == 1, pot["converted1"], pot["converted0"]),
        in_experiment_sessions=np.where(z == 1, pot["sessions1"], pot["sessions0"]),
        pre_revenue_leaky=np.where(population.units["is_new_user"], np.nan, np.round(leaky, 2)),
    )


# ---------------------------------------------------------------------------
# 2. Marketplace switchback
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SwitchbackConfig:
    """Hourly marketplace series; the treatment state is switched per period."""

    seed: int = 20260702
    n_days: int = 14
    period_hours: int = 4
    base_level: float = 100.0
    """Mean completed orders per hour."""
    daily_amplitude: float = 30.0
    weekend_lift: float = 10.0
    ar_coefficient: float = 0.6
    noise_sd: float = 8.0
    """SD of the stationary AR(1) noise (orders per hour)."""
    direct_effect: float = 4.0
    carryover_effects: tuple[float, ...] = (3.0, 1.5)
    """Effect of the treatment having been on 1, 2, ... hours earlier (carryover by lag)."""

    def __post_init__(self) -> None:
        if self.period_hours < 1 or 24 % self.period_hours != 0:
            raise ValueError("period_hours must divide 24")
        if self.n_days < 2:
            raise ValueError("n_days must be at least 2")
        if not 0.0 <= self.ar_coefficient < 1.0:
            raise ValueError("ar_coefficient must lie in [0, 1)")
        if len(self.carryover_effects) >= self.period_hours:
            raise ValueError("carryover must be shorter than a period")

    @property
    def n_periods(self) -> int:
        return self.n_days * 24 // self.period_hours


@dataclass(frozen=True)
class SwitchbackTruth:
    global_effect: float
    """All-on vs all-off difference in expected orders per hour: direct + sum(carryover)."""
    direct_effect: float
    carryover_hours: int


@dataclass(frozen=True)
class SwitchbackMarket:
    config: SwitchbackConfig
    hours: pd.DataFrame
    """Observable calendar: hour index, day, hour_of_day, period."""
    baseline: FloatArray = field(repr=False)
    """Evaluation-only: untreated orders per hour (seasonality + AR(1) noise)."""
    truth: SwitchbackTruth


def generate_switchback_market(config: SwitchbackConfig | None = None) -> SwitchbackMarket:
    config = config or SwitchbackConfig()
    rng = seeded_generator(config.seed)
    n_hours = config.n_days * 24
    hour = np.arange(n_hours)
    hour_of_day = hour % 24
    day = hour // 24
    seasonal = (config.base_level
                + config.daily_amplitude * np.sin(2 * np.pi * (hour_of_day - 9) / 24)
                + config.weekend_lift * np.isin(day % 7, (5, 6)))
    innovation_sd = config.noise_sd * math.sqrt(1 - config.ar_coefficient**2)
    noise = np.empty(n_hours)
    noise[0] = config.noise_sd * rng.standard_normal()
    shocks = innovation_sd * rng.standard_normal(n_hours)
    for h in range(1, n_hours):
        noise[h] = config.ar_coefficient * noise[h - 1] + shocks[h]
    hours = pd.DataFrame(
        {"hour": hour, "day": day, "hour_of_day": hour_of_day,
         "period": hour // config.period_hours,
         "hour_in_period": hour % config.period_hours}
    )
    truth = SwitchbackTruth(
        global_effect=config.direct_effect + float(sum(config.carryover_effects)),
        direct_effect=config.direct_effect,
        carryover_hours=len(config.carryover_effects),
    )
    return SwitchbackMarket(config, hours, np.asarray(seasonal + noise), truth)


def realise_switchback(market: SwitchbackMarket, period_assignment: np.ndarray) -> pd.DataFrame:
    """Hourly frame given a 0/1 treatment state per period (deterministic)."""
    w_period = np.asarray(period_assignment)
    if w_period.shape != (market.config.n_periods,) or not np.isin(w_period, (0, 1)).all():
        raise ValueError("period_assignment must be a 0/1 vector with one entry per period")
    w = w_period[market.hours["period"].to_numpy()].astype(float)
    outcome = market.baseline + market.config.direct_effect * w
    for lag, effect in enumerate(market.config.carryover_effects, start=1):
        lagged = np.concatenate([np.zeros(lag), w[:-lag]])
        outcome = outcome + effect * lagged
    return market.hours.assign(
        arm=np.where(w == 1, "treatment", "control"),
        orders=np.round(outcome, 3),
    )


__all__ = [
    "ASSIGNMENT_TIME",
    "LEAKY_WINDOW_OVERLAP_DAYS",
    "OUTCOME_WINDOW_DAYS",
    "PRE_PERIOD_DAYS",
    "CupedPopulation",
    "CupedScenarioConfig",
    "CupedScenarioTruth",
    "SwitchbackConfig",
    "SwitchbackMarket",
    "SwitchbackTruth",
    "generate_cuped_population",
    "generate_switchback_market",
    "realise_cuped_experiment",
    "realise_switchback",
]
