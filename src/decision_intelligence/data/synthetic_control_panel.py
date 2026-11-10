"""One treated region and a donor pool, from a factor model (M5, additive).

Business story: NexaMarket launches a flagship pricing policy in one region
only; the other regions form the donor pool. The outcome is monthly revenue
per customer in each region.

Data-generating process (interactive fixed effects; Abadie et al. 2010):

    Y_jt(0) = 100 + delta_t + sum_f lambda_tf mu_jf + eps_jt,   eps ~ N(0, noise_sd^2)

Common shocks ``delta_t`` and latent factors ``lambda_t`` move all regions,
but each region loads differently (``mu_j``), so regions do *not* share
parallel trends — a plain difference-in-differences would be biased while a
weighted combination of donors that matches the treated unit's loadings is
not.

* ``treated_in_convex_hull=True``: the treated unit's loadings are an exact
  convex combination of ``n_true_donors`` donors' loadings (true weights in
  the ground truth), so a synthetic control can reproduce its untreated path.
* ``treated_in_convex_hull=False``: the treated unit's loadings lie beyond
  every donor's (``extrapolation`` × the donor maximum), so no convex
  combination fits — the design should be flagged by poor pre-period fit.
* ``contaminated_donors``: that many of the true donors also receive
  ``contamination_share`` of the effect after treatment (spillover to
  neighbouring regions), violating the no-interference assumption for donors.

Unit 0 is the treated unit. Periods ``1..n_pre`` are pre-treatment; the
policy starts at ``n_pre + 1`` with effect ``effect_size * min(e / ramp, 1)``
for post-period ``e = 1, 2, ...``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from decision_intelligence.common.seeding import spawn_generators


@dataclass(frozen=True)
class SyntheticControlConfig:
    seed: int = 20260502
    n_donors: int = 30
    n_pre: int = 30
    n_post: int = 12
    n_factors: int = 3
    noise_sd: float = 1.0
    effect_size: float = 5.0
    effect_ramp: int = 3
    treated_in_convex_hull: bool = True
    n_true_donors: int = 4
    extrapolation: float = 1.8
    contaminated_donors: int = 0
    contamination_share: float = 0.5

    def __post_init__(self) -> None:
        if self.n_donors < 2 or self.n_pre < 2 or self.n_post < 1:
            raise ValueError("need at least 2 donors, 2 pre-periods and 1 post-period")
        if not 1 <= self.n_true_donors <= self.n_donors:
            raise ValueError("n_true_donors must be between 1 and n_donors")
        if not 0 <= self.contaminated_donors <= self.n_true_donors:
            raise ValueError("contaminated_donors must be between 0 and n_true_donors")
        if self.effect_ramp < 1 or self.n_factors < 1:
            raise ValueError("effect_ramp and n_factors must be positive")


@dataclass(frozen=True)
class SyntheticControlGroundTruth:
    effect_by_period: dict[int, float]
    average_post_effect: float
    true_donor_weights: dict[int, float] = field(repr=False)
    """Weights reproducing the treated loadings (only if in the convex hull)."""
    contaminated_donor_ids: tuple[int, ...]


@dataclass(frozen=True)
class SyntheticControlPanel:
    config: SyntheticControlConfig
    panel: pd.DataFrame
    """Columns: unit_id (0 = treated), period, treated_unit, post, outcome."""
    ground_truth: SyntheticControlGroundTruth


def generate_synthetic_control_panel(
    config: SyntheticControlConfig | None = None,
) -> SyntheticControlPanel:
    config = config or SyntheticControlConfig()
    rng_load, rng_time, rng_noise, rng_pick = spawn_generators(config.seed, 4)
    n_periods = config.n_pre + config.n_post
    periods = np.arange(1, n_periods + 1)

    donor_loadings = rng_load.uniform(0.0, 2.0, (config.n_donors, config.n_factors))
    true_donors = np.sort(rng_pick.choice(config.n_donors, config.n_true_donors, replace=False))
    weights = np.zeros(config.n_donors)
    if config.treated_in_convex_hull:
        weights[true_donors] = rng_pick.dirichlet(np.ones(config.n_true_donors))
        treated_loading = weights @ donor_loadings
    else:
        treated_loading = config.extrapolation * donor_loadings.max(axis=0)

    common = np.cumsum(rng_time.normal(0.3, 1.0, n_periods))
    factors = np.cumsum(rng_time.normal(0.0, 1.0, (n_periods, config.n_factors)), axis=0)
    loadings = np.vstack([treated_loading, donor_loadings])
    untreated = 100.0 + common[None, :] + loadings @ factors.T
    untreated = untreated + rng_noise.normal(0.0, config.noise_sd, untreated.shape)

    post = periods > config.n_pre
    effect = np.where(
        post, config.effect_size * np.minimum((periods - config.n_pre) / config.effect_ramp, 1.0), 0
    )
    outcome = untreated.copy()
    outcome[0] += effect
    contaminated = true_donors[: config.contaminated_donors]
    for donor in contaminated:
        outcome[donor + 1] += config.contamination_share * effect

    n_units = config.n_donors + 1
    panel = pd.DataFrame(
        {
            "unit_id": np.repeat(np.arange(n_units), n_periods),
            "period": np.tile(periods, n_units),
            "treated_unit": np.repeat(np.arange(n_units) == 0, n_periods).astype(int),
            "post": np.tile(post, n_units).astype(int),
            "outcome": outcome.ravel(),
        }
    )
    effect_by_period = {int(t): float(e) for t, e in zip(periods[post], effect[post], strict=True)}
    return SyntheticControlPanel(
        config=config,
        panel=panel,
        ground_truth=SyntheticControlGroundTruth(
            effect_by_period=effect_by_period,
            average_post_effect=float(np.mean(effect[post])),
            true_donor_weights={int(d) + 1: float(weights[d]) for d in true_donors}
            if config.treated_in_convex_hull
            else {},
            contaminated_donor_ids=tuple(int(d) + 1 for d in contaminated),
        ),
    )
