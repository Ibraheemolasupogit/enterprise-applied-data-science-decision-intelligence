"""Loyalty-tier assignment at a score threshold (M5 regression-discontinuity data, additive).

Business story: customers whose quarterly engagement score reaches 60
(0–100 scale) are upgraded to a loyalty tier with extra benefits; the
outcome is next-quarter spend. Assignment is *sharp*: tier = 1{score >= 60}.
The cutoff sits above the median score, so the effect at the cutoff differs
from the population-average effect.

Data-generating process
-----------------------
* Latent engagement ``u ~ N(0, 1)``; true score
  ``x = clip(50 + 15 (0.5 u + sqrt(0.75) z), 0, 100)`` with ``z ~ N(0, 1)``,
  so the score density is smooth through the cutoff.
* Pre-determined covariate ``tenure_months = 12 + 0.1 x + 3 u + N(0, 2^2)``.
* ``Y(0) = 20 + 0.25 x + 4 sin(x / 8) + 2 u + N(0, noise_sd^2)`` (smooth,
  non-linear in x — a global polynomial fits it poorly).
* Effect ``tau(x) = tau_at_cutoff + tau_slope (x - cutoff)``: heterogeneous, so
  the effect *at the cutoff* differs from the population ATE and the ATT.

Deliberate failures (off by default):

* ``manipulation_share``: that share of highly engaged customers (``u > 0``)
  whose true score falls just below the cutoff (within
  ``manipulation_window``) get their *reported* score pushed just above it.
  Assignment uses the reported score; outcomes depend on the true score and
  ``u``. Units just above the cutoff are then systematically more engaged
  — continuity of potential outcomes fails.
* ``compound_jump``: another benefit switches on at the same threshold
  (e.g. a separate credit-limit rule), adding a jump the tier did not cause.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from decision_intelligence.common.seeding import spawn_generators


@dataclass(frozen=True)
class ThresholdConfig:
    seed: int = 20260503
    n_customers: int = 5000
    cutoff: float = 60.0
    tau_at_cutoff: float = 3.0
    tau_slope: float = 0.08
    noise_sd: float = 2.0
    manipulation_share: float = 0.0
    manipulation_window: float = 3.0
    compound_jump: float = 0.0

    def __post_init__(self) -> None:
        if self.n_customers < 100:
            raise ValueError("n_customers must be at least 100")
        if not 0 < self.cutoff < 100:
            raise ValueError("cutoff must lie inside the score range (0, 100)")
        if not 0 <= self.manipulation_share <= 1:
            raise ValueError("manipulation_share must be in [0, 1]")
        if self.manipulation_window <= 0 or self.noise_sd <= 0:
            raise ValueError("manipulation_window and noise_sd must be positive")


@dataclass(frozen=True)
class ThresholdGroundTruth:
    tau_at_cutoff: float
    population_ate: float
    att: float
    n_manipulated: int
    individual_effects: np.ndarray = field(repr=False)
    """Evaluation-only per-customer effect tau(x_true)."""


@dataclass(frozen=True)
class ThresholdData:
    config: ThresholdConfig
    frame: pd.DataFrame
    """Columns: customer_id, score (reported running variable), tier, tenure_months, spend."""
    ground_truth: ThresholdGroundTruth


def baseline_spend(score: np.ndarray) -> np.ndarray:
    """Smooth, non-linear untreated mean spend as a function of the true score."""
    return np.asarray(20 + 0.25 * score + 4 * np.sin(score / 8), dtype=np.float64)


def generate_threshold_data(config: ThresholdConfig | None = None) -> ThresholdData:
    config = config or ThresholdConfig()
    rng_score, rng_cov, rng_out, rng_manip = spawn_generators(config.seed, 4)
    n = config.n_customers
    u = rng_score.normal(size=n)
    true_score = np.clip(
        50 + 15 * (0.5 * u + math.sqrt(0.75) * rng_score.normal(size=n)), 0.0, 100.0
    )
    tenure = 12 + 0.1 * true_score + 3 * u + rng_cov.normal(0.0, 2.0, n)

    reported = true_score.copy()
    eligible = (
        (true_score < config.cutoff)
        & (true_score >= config.cutoff - config.manipulation_window)
        & (u > 0)
    )
    manipulated = eligible & (rng_manip.random(n) < config.manipulation_share)
    reported[manipulated] = config.cutoff + rng_manip.uniform(0.0, 0.5, int(manipulated.sum()))

    tier = (reported >= config.cutoff).astype(int)
    effect = config.tau_at_cutoff + config.tau_slope * (true_score - config.cutoff)
    spend = (
        baseline_spend(true_score)
        + 2 * u
        + tier * (effect + config.compound_jump)
        + rng_out.normal(0.0, config.noise_sd, n)
    )
    frame = pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "score": reported,
            "tier": tier,
            "tenure_months": tenure,
            "spend": spend,
        }
    )
    return ThresholdData(
        config=config,
        frame=frame,
        ground_truth=ThresholdGroundTruth(
            tau_at_cutoff=config.tau_at_cutoff,
            population_ate=float(effect.mean()),
            att=float(effect[tier == 1].mean()),
            n_manipulated=int(manipulated.sum()),
            individual_effects=effect,
        ),
    )
