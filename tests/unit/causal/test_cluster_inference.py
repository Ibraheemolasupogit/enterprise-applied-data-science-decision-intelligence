"""Wild cluster bootstrap: exact agreement with a brute-force reference implementation."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from decision_intelligence.causal_inference.cluster_inference import (
    cluster_robust_t_test,
    wild_cluster_bootstrap,
)
from decision_intelligence.statistics.linear_regression import fit_ols


@pytest.fixture
def small_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    clusters = np.repeat(np.arange(6), 8)
    x = rng.normal(size=48) + rng.normal(size=6)[clusters]
    y = 0.3 * x + rng.normal(size=48) + rng.normal(size=6)[clusters]
    design = np.column_stack([np.ones(48), x, rng.normal(size=48)])
    return design, y, clusters


def _brute_force_p(design, y, clusters, column, beta0):
    """Independent implementation: refit every sign pattern with M3 fit_ols (CR1 SEs)."""
    observed = fit_ols(design, y, add_intercept=False, covariance="cluster", clusters=clusters)
    t_obs = (observed.coefficients[column] - beta0) / observed.standard_errors[column]
    others = np.delete(design, column, axis=1)
    restricted_y = y - beta0 * design[:, column]
    restricted = fit_ols(others, restricted_y, add_intercept=False)
    fitted = restricted.fitted_values + beta0 * design[:, column]
    residual = restricted.residuals
    t_values = []
    for signs in itertools.product([-1.0, 1.0], repeat=int(clusters.max()) + 1):
        y_star = fitted + residual * np.asarray(signs)[clusters]
        refit = fit_ols(design, y_star, add_intercept=False, covariance="cluster",
                        clusters=clusters)
        t_values.append((refit.coefficients[column] - beta0) / refit.standard_errors[column])
    return float(np.mean(np.abs(np.asarray(t_values)) >= abs(t_obs) - 1e-12))


@pytest.mark.parametrize("beta0", [0.0, 0.3, -0.5])
def test_enumerated_wild_bootstrap_matches_brute_force(small_problem, beta0: float) -> None:
    design, y, clusters = small_problem
    ours = wild_cluster_bootstrap(
        design, y, clusters, 1, rng=np.random.default_rng(0), beta0=beta0, n_draws=99,
        confidence_level=None,
    )
    assert ours.enumerated and ours.n_draws == 64
    assert ours.p_value == pytest.approx(_brute_force_p(design, y, clusters, 1, beta0))


def test_cr1_t_test_matches_m3_cluster_regression(small_problem) -> None:
    design, y, clusters = small_problem
    beta, se, p = cluster_robust_t_test(design, y, clusters, 1)
    reference = fit_ols(design, y, add_intercept=False, covariance="cluster", clusters=clusters)
    assert beta == pytest.approx(reference.coefficients[1])
    assert se == pytest.approx(reference.standard_errors[1])
    assert p == pytest.approx(reference.p_values[1])


def test_confidence_interval_inverts_the_test(small_problem) -> None:
    design, y, clusters = small_problem
    rng = np.random.default_rng(1)
    result = wild_cluster_bootstrap(design, y, clusters, 1, rng=rng, n_draws=999, weights="webb")
    assert result.ci_lower is not None and result.ci_upper is not None
    assert result.ci_lower < result.coefficient < result.ci_upper
    for bound in (result.ci_lower, result.ci_upper):
        at_bound = wild_cluster_bootstrap(
            design, y, clusters, 1, rng=np.random.default_rng(1), beta0=bound, n_draws=999,
            weights="webb", confidence_level=None,
        )
        assert at_bound.p_value == pytest.approx(0.05, abs=0.02)


def test_random_draws_are_reproducible_and_validated(small_problem) -> None:
    design, y, clusters = small_problem
    many = np.repeat(np.arange(12), 4)
    a = wild_cluster_bootstrap(design, y, many, 1, rng=np.random.default_rng(5), n_draws=499,
                               confidence_level=None)
    b = wild_cluster_bootstrap(design, y, many, 1, rng=np.random.default_rng(5), n_draws=499,
                               confidence_level=None)
    assert not a.enumerated and a.p_value == b.p_value
    with pytest.raises(ValueError, match="rank-deficient"):
        wild_cluster_bootstrap(np.column_stack([design, design[:, 1]]), y, clusters, 1,
                               rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="two clusters"):
        wild_cluster_bootstrap(design, y, np.zeros(48), 1, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="weights"):
        wild_cluster_bootstrap(design, y, clusters, 1, rng=np.random.default_rng(0),
                               weights="normal")  # type: ignore[arg-type]


def test_few_treated_clusters_are_flagged() -> None:
    """Regression test: the unreliable one-treated-cluster regime is reported explicitly."""
    from decision_intelligence.causal_inference import did
    from decision_intelligence.data.rollout_panel import RolloutConfig, generate_rollout_panel

    config = RolloutConfig(n_units=10, n_periods=12, cohorts=((7, 0.1),),
                           cohort_effect_scales=(1.0,), base_effect=0.0)
    design, y, clusters, column = did.twfe_design(generate_rollout_panel(config).panel)
    result = wild_cluster_bootstrap(design, y, clusters, column, rng=np.random.default_rng(0),
                                    n_draws=999, confidence_level=None)
    assert result.clusters_with_regressor == 1
    assert any("unreliable" in w for w in result.warnings)
    balanced = RolloutConfig(n_units=10, n_periods=12, cohorts=((7, 0.5),),
                             cohort_effect_scales=(1.0,))
    design, y, clusters, column = did.twfe_design(generate_rollout_panel(balanced).panel)
    ok = wild_cluster_bootstrap(design, y, clusters, column, rng=np.random.default_rng(0),
                                n_draws=999, confidence_level=None)
    assert ok.clusters_with_regressor == 5 and not ok.warnings
