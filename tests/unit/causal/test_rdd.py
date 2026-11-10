"""Regression discontinuity: reference agreement, bandwidth properties and validity checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from decision_intelligence.causal_inference import rdd
from decision_intelligence.data.threshold_assignment import ThresholdConfig, generate_threshold_data


@pytest.fixture(scope="module")
def default_data() -> tuple[np.ndarray, np.ndarray, float]:
    data = generate_threshold_data()
    return data.frame["score"].to_numpy(), data.frame["spend"].to_numpy(), data.config.cutoff


def _statsmodels_local(x, y, c, h, order):
    z = x - c
    w = np.clip(1 - np.abs(z / h), 0, None)
    k = w > 0
    d = (z >= 0).astype(float)
    columns = [np.ones(k.sum()), d[k]]
    for p in range(1, order + 1):
        columns += [z[k] ** p, d[k] * z[k] ** p]
    return sm.WLS(y[k], np.column_stack(columns), weights=w[k]).fit(cov_type="HC1")


@pytest.mark.parametrize("order", [1, 2])
def test_local_polynomial_matches_statsmodels_wls(default_data, order: int) -> None:
    x, y, c = default_data
    ours = rdd.local_polynomial_rdd(x, y, c, 7.0, order=order)
    reference = _statsmodels_local(x, y, c, 7.0, order)
    assert ours.estimate == pytest.approx(reference.params[1], rel=1e-10)
    assert ours.standard_error == pytest.approx(reference.bse[1], rel=1e-8)


def test_robust_bias_corrected_equals_local_quadratic(default_data) -> None:
    x, y, c = default_data
    robust = rdd.robust_bias_corrected_rdd(x, y, c, 7.0)
    quadratic = rdd.local_polynomial_rdd(x, y, c, 7.0, order=2)
    assert robust.estimate == quadratic.estimate and robust.ci_lower == quadratic.ci_lower
    assert "rho = 1" in robust.method
    conventional = rdd.local_linear_rdd(x, y, c, 7.0)
    assert robust.standard_error > conventional.standard_error


def test_ik_bandwidth_equivariance(default_data) -> None:
    """h scales with the running variable and is invariant to the outcome's scale and shift."""
    x, y, c = default_data
    h = rdd.ik_bandwidth(x, y, c)
    assert 1.0 < h < 30.0
    assert rdd.ik_bandwidth(2 * x, y, 2 * c) == pytest.approx(2 * h, rel=1e-8)
    assert rdd.ik_bandwidth(x, 3 * y + 7, c) == pytest.approx(h, rel=1e-8)
    assert rdd.ik_bandwidth(x + 10, y, c + 10) == pytest.approx(h, rel=1e-8)


def test_bandwidth_sensitivity_table(default_data) -> None:
    x, y, c = default_data
    table = rdd.bandwidth_sensitivity(x, y, c, [4.0, 8.0])
    assert list(table["bandwidth"]) == [4.0, 8.0]
    assert (table["robust_ci_upper"] - table["robust_ci_lower"]
            > table["conventional_ci_upper"] - table["conventional_ci_lower"]).all()


def test_density_tests_detect_manipulation_and_not_its_absence(default_data) -> None:
    x, _, c = default_data
    assert rdd.mccrary_density_test(x, c).p_value > 0.05
    assert rdd.binomial_density_test(x, c, 2.0).p_value > 0.05
    manipulated = generate_threshold_data(ThresholdConfig(manipulation_share=0.6))
    xm = manipulated.frame["score"].to_numpy()
    mccrary = rdd.mccrary_density_test(xm, c)
    assert mccrary.p_value < 0.001 and mccrary.log_density_difference is not None
    assert mccrary.log_density_difference > 0
    assert rdd.binomial_density_test(xm, c, 2.0).p_value < 0.001


def test_binomial_window_counts() -> None:
    x = np.array([9.0, 9.5, 9.9, 10.0, 10.2, 11.5, 3.0, 20.0, 9.1, 10.9, 12.0])
    result = rdd.binomial_density_test(x, 10.0, 1.0)
    assert "4 below, 3 above" in result.method


def test_covariate_continuity_and_placebo_cutoffs(default_data) -> None:
    x, y, c = default_data
    data = generate_threshold_data()
    covariates = rdd.covariate_continuity(x, data.frame[["tenure_months"]], c, 8.0)
    assert bool(covariates["covers_zero"].iloc[0])
    placebos = rdd.placebo_cutoffs(x, y, c, [c - 8, c + 8], 6.0)
    assert len(placebos) == 2
    with pytest.raises(ValueError):
        rdd.placebo_cutoffs(x, y, c, [c], 6.0)


def test_global_polynomial_and_donut_run(default_data) -> None:
    x, y, c = default_data
    estimates = {p: rdd.global_polynomial_rdd(x, y, c, p).estimate for p in (1, 2, 4)}
    assert max(estimates.values()) - min(estimates.values()) > 0.5  # order-sensitive
    donut = rdd.donut_rdd(x, y, c, 8.0, 1.0)
    assert "donut 1" in donut.method


def test_validation(default_data) -> None:
    x, y, c = default_data
    with pytest.raises(ValueError, match="bandwidth"):
        rdd.local_linear_rdd(x, y, c, 0.0)
    with pytest.raises(ValueError, match="too few"):
        rdd.local_linear_rdd(x, y, c, 0.01)
    with pytest.raises(ValueError, match="same length"):
        rdd.local_linear_rdd(x, y[:-1], c, 5.0)
    with pytest.raises(ValueError, match="order"):
        rdd.local_polynomial_rdd(x, y, c, 5.0, order=3)
    frame = pd.DataFrame({"x": np.linspace(0, 1, 20), "y": np.zeros(20)})
    with pytest.raises(ValueError):
        rdd.ik_bandwidth(frame["x"], frame["y"], 0.99)
