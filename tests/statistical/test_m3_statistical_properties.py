"""Repeated-sampling properties of the M3 methods — the evidence that they are valid.

Reference comparisons (unit tests) show the code computes the intended
formula. These tests show the formula *behaves as claimed* under its
assumptions, and fails in the documented way when an assumption is
violated:

* type I error at the nominal level under H0 (and the documented failure
  of Student's pooled t under variance/size imbalance);
* confidence-interval coverage at the nominal level;
* analytic power agreeing with simulated power;
* clustered errors breaking naive SEs, and cluster-robust SEs fixing them.

Every simulation uses a fixed seed, so results are deterministic. The
consistency check allows 3.5 binomial standard errors around the target,
so a correct procedure fails only through a code change, never by chance.
"""

from __future__ import annotations

import math

import numpy as np

from decision_intelligence.common import bootstrap_ci, seeded_generator
from decision_intelligence.statistics import hypothesis_tests as ht
from decision_intelligence.statistics.estimators import t_interval_for_mean
from decision_intelligence.statistics.linear_regression import fit_ols
from decision_intelligence.statistics.power import power_two_proportions, power_two_sample_t
from decision_intelligence.statistics.simulation import (
    simulate_interval_coverage,
    simulate_rejection_rate,
)

_Z = 3.5


def _normal_pair(n_x: int, n_y: int, sd_x: float = 1.0, sd_y: float = 1.0, shift: float = 0.0):
    def generate(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        return rng.normal(shift, sd_x, n_x), rng.normal(0.0, sd_y, n_y)

    return generate


def test_welch_t_holds_type_one_error_under_unequal_variances() -> None:
    rate = simulate_rejection_rate(
        _normal_pair(10, 40, sd_x=3.0, sd_y=1.0),
        lambda x, y: ht.two_sample_t_test(x, y).p_value,
        alpha=0.05,
        n_replications=4000,
        rng=seeded_generator(101),
    )
    assert rate.is_consistent_with(0.05, z=_Z)


def test_student_t_type_one_error_is_inflated_when_small_group_has_large_variance() -> None:
    """The documented failure mode: Student's pooled test is not robust to this design."""
    rate = simulate_rejection_rate(
        _normal_pair(10, 40, sd_x=3.0, sd_y=1.0),
        lambda x, y: ht.two_sample_t_test(x, y, equal_var=True).p_value,
        alpha=0.05,
        n_replications=4000,
        rng=seeded_generator(101),
    )
    assert rate.rate > 0.15


def test_mann_whitney_and_permutation_hold_type_one_error() -> None:
    generate = _normal_pair(15, 15)
    mwu = simulate_rejection_rate(
        lambda rng: (rng.exponential(1.0, 15), rng.exponential(1.0, 15)),
        lambda x, y: ht.mann_whitney_u(x, y).p_value,
        alpha=0.05,
        n_replications=3000,
        rng=seeded_generator(102),
    )
    assert mwu.rate <= 0.05 + _Z * math.sqrt(0.05 * 0.95 / 3000)
    permutation_rng = seeded_generator(1031)  # dedicated stream for the permutations
    permutation = simulate_rejection_rate(
        generate,
        lambda x, y: ht.permutation_test(
            x, y, n_permutations=199, rng=permutation_rng
        ).p_value,
        alpha=0.05,
        n_replications=600,
        rng=seeded_generator(103),
    )
    assert permutation.rate <= 0.05 + _Z * math.sqrt(0.05 * 0.95 / 600)


def test_two_proportion_z_test_type_one_error() -> None:
    def generate(rng: np.random.Generator) -> tuple[np.ndarray, ...]:
        return (np.array([rng.binomial(500, 0.2), rng.binomial(500, 0.2)], dtype=float),)

    rate = simulate_rejection_rate(
        generate,
        lambda counts: ht.two_proportion_z_test(int(counts[0]), 500, int(counts[1]), 500).p_value,
        alpha=0.05,
        n_replications=5000,
        rng=seeded_generator(104),
    )
    assert rate.is_consistent_with(0.05, z=_Z)


def test_t_interval_coverage_normal_and_degradation_for_skewed_small_samples() -> None:
    normal = simulate_interval_coverage(
        lambda rng: (rng.normal(5.0, 2.0, 12),),
        t_interval_for_mean,
        5.0,
        n_replications=4000,
        rng=seeded_generator(105),
    )
    assert normal.is_consistent_with(0.95, z=_Z)
    # Lognormal(0, 1.5), n = 10: heavy right skew — the CLT has not kicked in.
    true_mean = math.exp(1.5**2 / 2)
    skewed = simulate_interval_coverage(
        lambda rng: (rng.lognormal(0.0, 1.5, 10),),
        t_interval_for_mean,
        true_mean,
        n_replications=4000,
        rng=seeded_generator(106),
    )
    assert skewed.rate < 0.88


def test_wilson_interval_covers_where_wald_fails() -> None:
    def wald(successes: float) -> tuple[float, float]:
        p = successes / 40
        half = 1.959964 * math.sqrt(p * (1 - p) / 40)
        return p - half, p + half

    def generate(rng: np.random.Generator) -> tuple[float]:
        return (float(rng.binomial(40, 0.05)),)

    wilson = simulate_interval_coverage(
        generate,
        lambda s: ht.wilson_interval(int(s), 40),
        0.05,
        n_replications=4000,
        rng=seeded_generator(107),
    )
    wald_coverage = simulate_interval_coverage(
        generate, wald, 0.05, n_replications=4000, rng=seeded_generator(107)
    )
    assert wilson.rate > 0.92
    assert wald_coverage.rate < 0.88


def test_bootstrap_bca_coverage_for_a_skewed_mean() -> None:
    resampling_rng = seeded_generator(1081)  # dedicated stream for the resampling

    def interval(x: np.ndarray) -> tuple[float, float]:
        result = bootstrap_ci(x, np.mean, rng=resampling_rng, n_resamples=500, method="bca")
        return result.ci_lower, result.ci_upper

    coverage = simulate_interval_coverage(
        lambda rng: (rng.exponential(2.0, 40),),
        interval,
        2.0,
        n_replications=400,
        rng=seeded_generator(108),
    )
    # BCa is second-order accurate but still slightly liberal at n = 40 for this skew.
    assert 0.88 <= coverage.rate <= 0.98


def test_analytic_t_power_agrees_with_simulation() -> None:
    analytic = power_two_sample_t(0.5, 40)
    simulated = simulate_rejection_rate(
        _normal_pair(40, 40, shift=0.5),
        lambda x, y: ht.two_sample_t_test(x, y, equal_var=True).p_value,
        alpha=0.05,
        n_replications=4000,
        rng=seeded_generator(109),
    )
    assert simulated.is_consistent_with(analytic, z=_Z)


def test_analytic_proportion_power_agrees_with_simulation() -> None:
    analytic = power_two_proportions(0.14, 0.10, 800)

    def generate(rng: np.random.Generator) -> tuple[np.ndarray, ...]:
        return (np.array([rng.binomial(800, 0.14), rng.binomial(800, 0.10)], dtype=float),)

    simulated = simulate_rejection_rate(
        generate,
        lambda c: ht.two_proportion_z_test(int(c[0]), 800, int(c[1]), 800).p_value,
        alpha=0.05,
        n_replications=4000,
        rng=seeded_generator(110),
    )
    assert simulated.is_consistent_with(analytic, z=_Z)


def test_cluster_robust_se_restores_coverage_under_within_cluster_correlation() -> None:
    """Regressor and error both have a cluster component (as with customer-level
    attributes and repeated transactions). Classical SEs under-cover badly;
    cluster-robust SEs are close to nominal with 60 clusters."""
    n_clusters, per_cluster, beta = 60, 10, 0.5

    def generate(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cluster = np.repeat(np.arange(n_clusters), per_cluster)
        x = rng.normal(size=n_clusters)[cluster] + 0.3 * rng.normal(size=cluster.size)
        error = rng.normal(size=n_clusters)[cluster] + 0.5 * rng.normal(size=cluster.size)
        return x, beta * x + error, cluster.astype(float)

    def classical(x: np.ndarray, y: np.ndarray, cluster: np.ndarray) -> tuple[float, float]:
        result = fit_ols(x, y)
        return float(result.ci_lower[1]), float(result.ci_upper[1])

    def clustered(x: np.ndarray, y: np.ndarray, cluster: np.ndarray) -> tuple[float, float]:
        result = fit_ols(x, y, covariance="cluster", clusters=cluster)
        return float(result.ci_lower[1]), float(result.ci_upper[1])

    naive = simulate_interval_coverage(
        generate, classical, beta, n_replications=1000, rng=seeded_generator(111)
    )
    robust = simulate_interval_coverage(
        generate, clustered, beta, n_replications=1000, rng=seeded_generator(111)
    )
    assert naive.rate < 0.80
    assert robust.rate > 0.92
