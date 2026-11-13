"""CUPED and regression adjustment against manual formulas and statsmodels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from decision_intelligence.experimentation import variance_reduction as vr


def _data(n: int = 4_000, seed: int = 1, effect: float = 0.5, hetero: float = 0.0
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(10, 2, n)
    arm = np.where(rng.random(n) < 0.5, "treatment", "control")
    z = (arm == "treatment").astype(float)
    y = 3 + 0.8 * x + z * (effect + hetero * (x - 10)) + rng.normal(0, 1.5, n)
    return y, arm, x


def test_cuped_equals_the_manual_formula() -> None:
    y, arm, x = _data()
    result = vr.cuped(y, arm, x)
    theta = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
    adjusted = y - theta * (x - x.mean())
    manual = adjusted[arm == "treatment"].mean() - adjusted[arm == "control"].mean()
    assert result.effect.estimate == pytest.approx(manual, abs=1e-10)
    assert result.coefficients["x0"] == pytest.approx(theta, rel=1e-10)
    assert result.effect.adjustment_set == ("x0",)


def test_lin_regression_adjustment_equals_statsmodels_hc2() -> None:
    y, arm, x = _data(hetero=0.3)
    result = vr.regression_adjustment(y, arm, pd.DataFrame({"pre": x}))
    z = (arm == "treatment").astype(float)
    xc = x - x.mean()
    design = sm.add_constant(np.column_stack([z, xc, z * xc]))
    fit = sm.OLS(y, design).fit(cov_type="HC2")
    assert result.effect.estimate == pytest.approx(fit.params[1], abs=1e-10)
    assert result.effect.standard_error == pytest.approx(fit.bse[1], rel=1e-8)


def test_variance_reduction_matches_one_minus_rho_squared() -> None:
    y, arm, x = _data(n=40_000, seed=2)
    result = vr.cuped(y, arm, x)
    rho = np.corrcoef(x, y)[0, 1]
    assert result.r_squared == pytest.approx(rho**2, abs=0.01)
    assert result.realised_variance_reduction == pytest.approx(rho**2, abs=0.02)
    assert result.equivalent_sample_size_factor == pytest.approx(
        1 / (1 - result.realised_variance_reduction))


def test_uninformative_covariate_changes_little() -> None:
    y, arm, _ = _data(seed=3)
    noise = np.random.default_rng(9).normal(size=y.size)
    result = vr.cuped(y, arm, noise)
    assert abs(result.realised_variance_reduction) < 0.01
    assert result.effect.estimate == pytest.approx(result.unadjusted.effect.estimate, abs=0.02)


def test_missing_covariates_are_imputed_with_an_indicator() -> None:
    y, arm, x = _data(seed=4)
    x_missing = x.copy()
    x_missing[:300] = np.nan
    matrix, names, n_missing = vr.prepare_covariates(pd.DataFrame({"pre": x_missing}))
    assert names == ["pre", "pre_missing"] and n_missing == 300
    assert np.all(matrix[:300, 0] == np.nanmean(x_missing)) and matrix[:300, 1].sum() == 300
    result = vr.cuped(y, arm, pd.DataFrame({"pre": x_missing}))
    assert result.n_missing_covariate == 300 and result.effect.n == y.size


def test_covariate_validation() -> None:
    y, arm, x = _data(n=200)
    with pytest.raises(ValueError, match="constant"):
        vr.cuped(y, arm, np.ones_like(x))
    with pytest.raises(ValueError, match="every unit"):
        vr.cuped(y, arm, np.full_like(x, np.nan))
    with pytest.raises(ValueError, match="infinite"):
        vr.cuped(y, arm, np.where(np.arange(x.size) == 0, np.inf, x))
    with pytest.raises(ValueError, match="one row per unit"):
        vr.cuped(y, arm, x[:-1])


def test_adjustment_covariate_timing_check() -> None:
    y, arm, x = _data(n=500)
    z = (arm == "treatment").astype(float)
    t0 = pd.Timestamp("2025-06-02")
    ok = vr.check_adjustment_covariate("pre", x, z, window_end=t0, assignment_time=t0)
    late = vr.check_adjustment_covariate("pre", x, z, window_end=t0 + pd.Timedelta(1, unit="D"),
                                         assignment_time=t0)
    assert ok.status == "PASS" and "pre-treatment" in ok.detail
    assert late.status == "FAIL" and "must not be used" in late.detail


def test_plan_with_covariate_scales_sample_size_by_one_minus_r2() -> None:
    plain, adjusted = vr.plan_with_covariate(15.0, 16.0, historical_r_squared=0.36,
                                             mde_absolute=0.5, daily_eligible=700)
    assert adjusted.n_total / plain.n_total == pytest.approx(0.64, abs=0.01)
    assert adjusted.days is not None and plain.days is not None and adjusted.days < plain.days
    with pytest.raises(ValueError):
        vr.plan_with_covariate(15.0, 16.0, historical_r_squared=1.0, mde_absolute=0.5)
