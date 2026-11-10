"""Regional service-policy rollout: a balanced panel with staggered adoption (M5).

Additive extension of the synthetic foundation for difference-in-differences
and event studies. It is self-contained — it does not read or modify any M2
table, generator or contract — and follows the same conventions: one frozen
config, deterministic seeding through :mod:`decision_intelligence.common`,
and ground truth kept separate from the observed panel.

Business story: NexaMarket rolls a new delivery-service policy out to
regional markets in waves. The outcome is weekly revenue per active
customer in each market. Some markets never adopt.

Treatment-timing convention (used by every M5 estimator)
--------------------------------------------------------
* Periods are integers ``1..n_periods``.
* ``cohort`` (``g``) is the first period in which a unit is treated; ``0``
  means never treated within the panel.
* ``treated`` = ``1{g > 0 and period >= g}`` (absorbing: once on, always on).
* Event time ``e = period - g``; ``e = 0`` is the first treated period and
  ``e = -1`` the last untreated one.

Data-generating process
-----------------------
``Y_it(0) = 50 + alpha_i + lambda_t + delta * (t - 1) * 1{i ever treated} + s_it + u_it``

* ``alpha_i ~ N(0, unit_effect_sd^2)``; ``lambda_t = trend*t + season*sin(2 pi t / 12)``;
* ``u_it`` is AR(1) within unit (``rho``), the serial correlation that makes
  unclustered panel standard errors invalid;
* ``delta`` (``differential_trend``) > 0 breaks parallel trends: units that
  adopt were already on a steeper path;
* ``s_it`` is a spillover onto *never-treated* units (``spillover_share`` of
  the mean direct effect among currently treated units) — a SUTVA failure.

``Y_it(1) = Y_it(0) + tau_g(e)`` for ``e >= 0`` with the dynamic,
cohort-heterogeneous effect ``tau_g(e) = scale_g * base_effect * min(1 + e / ramp_periods, 2)``.
With ``anticipation_periods = k`` units already respond at
``e = -k..-1`` by ``anticipation_share * tau_g(0)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from decision_intelligence.common.seeding import spawn_generators

PANEL_COLUMNS: dict[str, str] = {
    "unit_id": "int",
    "period": "int",
    "cohort": "int",
    "treated": "int",
    "outcome": "float",
}
"""Observed-panel contract (column → type). Ground truth is returned separately."""


@dataclass(frozen=True)
class RolloutConfig:
    """Configuration of one rollout panel; identical configs give identical panels."""

    seed: int = 20260501
    n_units: int = 60
    n_periods: int = 24
    cohorts: tuple[tuple[int, float], ...] = ((9, 0.25), (13, 0.25), (17, 0.25))
    """(first treated period, share of units). The remaining share is never treated."""
    cohort_effect_scales: tuple[float, ...] = (1.5, 1.0, 0.5)
    """Effect multiplier for each cohort, in the order of ``cohorts`` (heterogeneity)."""
    base_effect: float = 2.0
    ramp_periods: int = 4
    unit_effect_sd: float = 3.0
    trend: float = 0.15
    seasonal_amplitude: float = 1.0
    ar1_rho: float = 0.5
    noise_sd: float = 1.0
    anticipation_periods: int = 0
    anticipation_share: float = 0.5
    differential_trend: float = 0.0
    spillover_share: float = 0.0

    def __post_init__(self) -> None:
        if self.n_units < 4 or self.n_periods < 4:
            raise ValueError("need at least 4 units and 4 periods")
        if len(self.cohort_effect_scales) != len(self.cohorts):
            raise ValueError("cohort_effect_scales must have one entry per cohort")
        shares = [share for _, share in self.cohorts]
        if any(s <= 0 for s in shares) or sum(shares) > 1 + 1e-12:
            raise ValueError("cohort shares must be positive and sum to at most 1")
        for first_period, _ in self.cohorts:
            if not 2 <= first_period <= self.n_periods:
                raise ValueError("cohort first-treated periods must lie in 2..n_periods")
        if len({g for g, _ in self.cohorts}) != len(self.cohorts):
            raise ValueError("cohort first-treated periods must be distinct")
        if not -1 < self.ar1_rho < 1:
            raise ValueError("ar1_rho must be in (-1, 1)")
        if self.ramp_periods < 1 or self.anticipation_periods < 0:
            raise ValueError("ramp_periods >= 1 and anticipation_periods >= 0 required")
        if not 0 <= self.spillover_share <= 1:
            raise ValueError("spillover_share must be in [0, 1]")


@dataclass(frozen=True)
class RolloutGroundTruth:
    """Evaluation-only truth: never an estimator input.

    ``cell_effects`` has one row per unit-period with the true direct effect
    (non-zero only for treated cells and anticipation cells) and the
    spillover received (non-zero only for never-treated units).
    ``att_by_cohort_period`` is the true ATT(g, t) = mean direct effect in
    cohort ``g`` at period ``t``; ``overall_att`` averages the true effect
    over all treated unit-periods (each treated cell weighted equally).
    """

    cell_effects: pd.DataFrame = field(repr=False)
    att_by_cohort_period: dict[tuple[int, int], float]
    overall_att: float
    event_time_att: dict[int, float]
    parallel_trends_hold: bool
    no_anticipation: bool
    no_spillovers: bool


@dataclass(frozen=True)
class RolloutPanel:
    config: RolloutConfig
    panel: pd.DataFrame
    ground_truth: RolloutGroundTruth


def effect_path(config: RolloutConfig, cohort_index: int, event_time: int) -> float:
    """``tau_g(e)`` including anticipation; 0 outside the effect window."""
    scale = config.cohort_effect_scales[cohort_index]
    if event_time >= 0:
        return scale * config.base_effect * min(1 + event_time / config.ramp_periods, 2.0)
    if config.anticipation_periods and event_time >= -config.anticipation_periods:
        return config.anticipation_share * scale * config.base_effect
    return 0.0


def generate_rollout_panel(config: RolloutConfig | None = None) -> RolloutPanel:
    """Generate one balanced rollout panel and its ground truth."""
    config = config or RolloutConfig()
    rng_assign, rng_units, rng_noise = spawn_generators(config.seed, 3)
    n, periods = config.n_units, np.arange(1, config.n_periods + 1)

    # Cohort assignment: exact counts by largest remainder, then a random permutation.
    shares = np.array([share for _, share in config.cohorts])
    ideal = n * shares
    counts = np.floor(ideal).astype(int)
    remainder = int(round(n * shares.sum())) - int(counts.sum())
    counts[np.argsort(-(ideal - counts), kind="stable")[:max(remainder, 0)]] += 1
    labels = np.concatenate(
        [np.full(c, g) for (g, _), c in zip(config.cohorts, counts, strict=True)]
        + [np.zeros(n - int(counts.sum()), dtype=int)]
    )
    cohort = rng_assign.permutation(labels).astype(int)
    cohort_index = {g: k for k, (g, _) in enumerate(config.cohorts)}

    alpha = rng_units.normal(0.0, config.unit_effect_sd, n)
    time_effect = config.trend * periods + config.seasonal_amplitude * np.sin(
        2 * math.pi * periods / 12
    )

    innovations = rng_noise.normal(0.0, config.noise_sd, (n, config.n_periods))
    noise = np.empty_like(innovations)
    noise[:, 0] = innovations[:, 0] / math.sqrt(1 - config.ar1_rho**2)
    for t in range(1, config.n_periods):
        noise[:, t] = config.ar1_rho * noise[:, t - 1] + innovations[:, t]

    ever = cohort > 0
    direct = np.zeros((n, config.n_periods))
    for i in np.flatnonzero(ever):
        k = cohort_index[int(cohort[i])]
        for t_index, period_value in enumerate(periods):
            direct[i, t_index] = effect_path(config, k, int(period_value - cohort[i]))
    treated = (ever[:, None]) & (periods[None, :] >= cohort[:, None])

    spill = np.zeros_like(direct)
    if config.spillover_share > 0:
        on = treated.sum(axis=0)
        mean_effect = np.divide(
            (direct * treated).sum(axis=0), on, out=np.zeros(config.n_periods), where=on > 0
        )
        spill[~ever, :] = config.spillover_share * mean_effect[None, :]

    y0 = (
        50.0
        + alpha[:, None]
        + time_effect[None, :]
        + config.differential_trend * (periods[None, :] - 1) * ever[:, None]
        + spill
        + noise
    )
    outcome = y0 + direct

    unit_ids = np.repeat(np.arange(1, n + 1), config.n_periods)
    panel = pd.DataFrame(
        {
            "unit_id": unit_ids,
            "period": np.tile(periods, n),
            "cohort": np.repeat(cohort, config.n_periods),
            "treated": treated.ravel().astype(int),
            "outcome": outcome.ravel(),
        }
    )
    cell_effects = pd.DataFrame(
        {
            "unit_id": unit_ids,
            "period": np.tile(periods, n),
            "true_direct_effect": direct.ravel(),
            "true_spillover": spill.ravel(),
        }
    )

    att: dict[tuple[int, int], float] = {}
    for g, _ in config.cohorts:
        members = cohort == g
        if not members.any():
            continue
        for t_index, period_value in enumerate(periods):
            if period_value >= g:
                att[(g, int(period_value))] = float(direct[members, t_index].mean())
    event_att: dict[int, float] = {}
    for e in range(-config.n_periods, config.n_periods):
        cells = [
            direct[i, int(cohort[i]) + e - 1]
            for i in np.flatnonzero(ever)
            if 1 <= cohort[i] + e <= config.n_periods
        ]
        if cells and (e >= 0 or any(c != 0 for c in cells)):
            event_att[e] = float(np.mean(cells))
    ground_truth = RolloutGroundTruth(
        cell_effects=cell_effects,
        att_by_cohort_period=att,
        overall_att=float(direct[treated].mean()) if treated.any() else 0.0,
        event_time_att=event_att,
        parallel_trends_hold=config.differential_trend == 0,
        no_anticipation=config.anticipation_periods == 0,
        no_spillovers=config.spillover_share == 0,
    )
    return RolloutPanel(config=config, panel=panel, ground_truth=ground_truth)


def validate_rollout_panel(panel: pd.DataFrame) -> list[str]:
    """Contract checks: columns, balance, absorbing treatment consistent with cohort."""
    problems: list[str] = []
    missing = set(PANEL_COLUMNS) - set(panel.columns)
    if missing:
        return [f"missing columns: {sorted(missing)}"]
    counts = panel.groupby("unit_id")["period"].nunique()
    if counts.nunique() != 1 or len(panel) != counts.sum():
        problems.append("panel is not balanced (each unit must appear once per period)")
    implied = ((panel["cohort"] > 0) & (panel["period"] >= panel["cohort"])).astype(int)
    if not implied.equals(panel["treated"].astype(int)):
        problems.append("treated is inconsistent with cohort (treatment must be absorbing)")
    if panel["outcome"].isna().any():
        problems.append("outcome contains missing values")
    return problems
