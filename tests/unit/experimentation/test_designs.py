"""ICC estimation, cluster planning and switchback analysis."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from decision_intelligence.data.advanced_experiment_scenarios import (
    SwitchbackConfig,
    generate_switchback_market,
    realise_switchback,
)
from decision_intelligence.experimentation import designs as de
from decision_intelligence.experimentation.planning import design_effect, plan_binary
from decision_intelligence.statistics.hypothesis_tests import permutation_test


def _clustered(seed: int, sizes: list[int], icc: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(len(sizes)), sizes)
    effects = rng.normal(0, np.sqrt(icc), len(sizes))[labels]
    return effects + rng.normal(0, np.sqrt(1 - icc), labels.size), labels


def test_icc_matches_the_anova_f_statistic_identity() -> None:
    """ICC = (F - 1) / (F + n0 - 1) with F from SciPy's one-way ANOVA (unbalanced n0)."""
    sizes = [30, 45, 12, 60, 25, 40, 33, 18]
    y, labels = _clustered(1, sizes, 0.2)
    f = stats.f_oneway(*(y[labels == g] for g in range(len(sizes)))).statistic
    n = sum(sizes)
    n0 = (n - sum(s * s for s in sizes) / n) / (len(sizes) - 1)
    assert de.estimate_icc(y, labels).icc == pytest.approx((f - 1) / (f + n0 - 1), rel=1e-10)


def test_icc_is_consistent_and_bootstrap_interval_contains_it() -> None:
    y, labels = _clustered(2, [100] * 200, 0.05)
    estimate = de.estimate_icc(y, labels, rng=np.random.default_rng(0), n_bootstrap=299)
    assert estimate.icc == pytest.approx(0.05, abs=0.02)
    assert estimate.ci_lower is not None and estimate.ci_upper is not None
    assert estimate.ci_lower < estimate.icc < estimate.ci_upper
    assert estimate.mean_cluster_size == 100 and estimate.cluster_size_cv == 0
    independent, labels2 = _clustered(3, [50] * 100, 0.0)
    assert abs(de.estimate_icc(independent, labels2).icc) < 0.01
    assert de.estimate_icc(independent, labels2).icc_for_planning >= 0


def test_icc_validation() -> None:
    with pytest.raises(ValueError):
        de.estimate_icc([1.0, 2.0], [0, 1])
    with pytest.raises(ValueError):
        de.estimate_icc([1.0, np.nan, 2.0], [0, 1, 2])


def test_cluster_power_reduces_to_individual_power_without_icc() -> None:
    plan = plan_binary(0.12, mde_absolute=0.02)
    # 100 clusters per arm of the M6 per-arm n / 100: ICC = 0 -> about the individual power.
    m = plan.n_treatment / 100
    power = de.cluster_power(0.12, 0.02, clusters_per_arm=100, mean_cluster_size=m, icc=0.0)
    assert power == pytest.approx(0.8, abs=0.01)


def test_cluster_plan_grows_with_the_icc_and_beats_the_naive_plan() -> None:
    small = de.plan_cluster_randomised(0.12, mde_absolute=0.02, mean_cluster_size=100, icc=0.005)
    large = de.plan_cluster_randomised(0.12, mde_absolute=0.02, mean_cluster_size=100, icc=0.02)
    assert large.clusters_per_arm > small.clusters_per_arm > small.naive_clusters_per_arm
    assert large.design_effect == pytest.approx(design_effect(100, 0.02))
    assert large.planned_power >= 0.8
    below = de.cluster_power(0.12, 0.02, clusters_per_arm=large.clusters_per_arm - 1,
                             mean_cluster_size=100, icc=0.02)
    assert below < 0.8  # the smallest sufficient number
    assert large.units_per_arm == large.clusters_per_arm * 100


def test_switchback_assignment_balances_blocks() -> None:
    blocks = np.tile(np.arange(6), 14)
    w = de.switchback_assignment(blocks.size, rng=np.random.default_rng(1), strata=blocks)
    for b in range(6):
        assert w[blocks == b].sum() == 7
    plain = de.switchback_assignment(84, rng=np.random.default_rng(2))
    assert plain.sum() == 42
    with pytest.raises(ValueError):
        de.switchback_assignment(10, rng=np.random.default_rng(0), strata=np.arange(9))


def test_unstratified_randomisation_p_value_is_the_m3_permutation_test() -> None:
    rng_data = np.random.default_rng(4)
    y = rng_data.normal(size=30)
    treated = np.arange(30) < 15
    ours = de.randomisation_p_value(y, treated, rng=np.random.default_rng(7), n_permutations=499)
    m3 = permutation_test(y[treated], y[~treated], n_permutations=499,
                          rng=np.random.default_rng(7))
    assert ours == m3.p_value


def test_stratified_randomisation_p_value_matches_full_enumeration() -> None:
    y = np.array([3.0, 1.0, 2.5, 0.5, 10.0, 8.0, 9.5, 7.0])
    strata = np.repeat([0, 1], 4)
    treated = np.array([True, False, True, False, True, False, True, False])
    observed = abs(de._difference(y, treated, strata))
    stats_all = []
    for a in itertools.combinations(range(4), 2):
        for b in itertools.combinations(range(4, 8), 2):
            z = np.zeros(8, dtype=bool)
            z[list(a) + list(b)] = True
            stats_all.append(abs(de._difference(y, z, strata)))
    exact = np.mean(np.asarray(stats_all) >= observed - 1e-12)
    mc = de.randomisation_p_value(y, treated, rng=np.random.default_rng(0), strata=strata,
                                  n_permutations=20_000)
    assert mc == pytest.approx(exact, abs=0.01)


def _noiseless_market() -> tuple[SwitchbackConfig, pd.DataFrame, np.ndarray]:
    config = SwitchbackConfig(noise_sd=1e-9, daily_amplitude=0.0, weekend_lift=0.0,
                              n_days=4, period_hours=4)
    market = generate_switchback_market(config)
    blocks = (np.arange(config.n_periods) * 4 % 24) // 4
    w = de.switchback_assignment(config.n_periods, rng=np.random.default_rng(3), strata=blocks)
    return config, realise_switchback(market, w), w


def test_burn_in_covering_the_carryover_recovers_the_global_effect_exactly() -> None:
    config, hourly, _ = _noiseless_market()
    truth = config.direct_effect + sum(config.carryover_effects)
    exact = de.analyse_switchback(hourly, period_hours=4, burn_in_hours=2,
                                  rng=np.random.default_rng(0), n_permutations=99)
    assert exact.effect.estimate == pytest.approx(truth, abs=1e-6)
    biased = de.analyse_switchback(hourly, period_hours=4, burn_in_hours=0,
                                   rng=np.random.default_rng(0), n_permutations=99)
    assert biased.effect.estimate < truth - 0.5


def test_period_means_validation() -> None:
    _, hourly, _ = _noiseless_market()
    with pytest.raises(ValueError, match="burn_in"):
        de.period_means(hourly, outcome="orders", burn_in_hours=4, period_hours=4)
    mixed = hourly.copy()
    mixed.loc[mixed.index[0], "arm"] = "treatment" if mixed["arm"].iloc[0] == "control" \
        else "control"
    with pytest.raises(ValueError, match="both arms"):
        de.period_means(mixed, outcome="orders", burn_in_hours=0, period_hours=4)
    with pytest.raises(ValueError, match="columns"):
        de.period_means(hourly.drop(columns="hour_of_day"), outcome="orders", burn_in_hours=0,
                        period_hours=4)


def test_icc_refuses_all_singleton_clusters() -> None:
    # Review fix: raised ZeroDivisionError (within-cluster mean square 0 / 0).
    with pytest.raises(ValueError, match="single unit"):
        de.estimate_icc([1.0, 2.0, 4.0, 3.0], [0, 1, 2, 3])


def test_stratified_randomisation_p_value_refuses_a_stratum_without_both_arms() -> None:
    # Review fix: the contrast was NaN in every permutation, so no permutation "exceeded" it
    # and the p-value was the minimum 1 / (m + 1) -- spurious significance.
    y = np.arange(12, dtype=float)
    treated = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1], dtype=bool)
    strata = np.repeat([0, 1, 2], 4)
    with pytest.raises(ValueError, match="stratum 2"):
        de.randomisation_p_value(y, treated, rng=np.random.default_rng(0), strata=strata,
                                 n_permutations=199)


def test_period_blocks_use_the_period_start_for_periods_crossing_midnight() -> None:
    # Review fix: the block came from the smallest kept hour of day, so a 5-hour period
    # starting at 20:00 (hours 20-23 and 0) was put in block 0 instead of block 4.
    hours = np.arange(48)
    hourly = pd.DataFrame({"period": hours // 5, "hour_in_period": hours % 5,
                           "hour_of_day": hours % 24,
                           "arm": np.where((hours // 5) % 2 == 0, "treatment", "control"),
                           "orders": 1.0})
    for burn_in in (0, 2):
        periods = de.period_means(hourly, outcome="orders", burn_in_hours=burn_in,
                                  period_hours=5)
        expected = (periods["period"] * 5 % 24) // 5
        assert periods["block"].tolist() == expected.tolist()
