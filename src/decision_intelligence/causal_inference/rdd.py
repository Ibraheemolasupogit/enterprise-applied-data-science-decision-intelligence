"""Sharp regression discontinuity design (RDD).

Estimand
--------
``tau(c) = E[Y(1) - Y(0) | X = c]``: the average effect for units *at the
cutoff*. It is local. It equals the population ATE only if the effect does
not vary with the running variable — which the data cannot show — so it is
never reported as a population-wide effect.

Identification (sharp design: ``D = 1{X >= c}``)
------------------------------------------------
1. **Continuity**: ``E[Y(0) | X = x]`` and ``E[Y(1) | X = x]`` are continuous
   at ``c``. Violated by **manipulation/sorting** of the running variable
   (units choosing which side they land on) or by **compound treatments**
   that switch on at the same cutoff.
2. The cutoff is known and assignment is deterministic in ``X`` (sharp).

Testable implications (necessary, not sufficient):

* no discontinuity in the density of ``X`` at ``c`` (McCrary 2008;
  approximate binomial window test);
* no discontinuity in pre-determined covariates at ``c``;
* no "effects" at placebo cutoffs away from ``c``;
* stability across bandwidths and when units nearest the cutoff are
  dropped ("donut").

Estimation and inference
------------------------
Local polynomial regression on each side with a triangular kernel within
bandwidth ``h`` (weighted least squares through M3 ``fit_ols`` on
sqrt-weight-transformed data, HC1 standard errors). The default bandwidth
is Imbens & Kalyanaraman's (2012) MSE-optimal choice for local linear.

**At an MSE-optimal bandwidth the smoothing bias is of the same order as the
standard error**, so the *conventional* local-linear interval under-covers.
:func:`robust_bias_corrected_rdd` implements Calonico, Cattaneo &
Titiunik's (2014) robust bias-corrected (RBC) inference with pilot bandwidth
equal to the main bandwidth (``rho = h / b = 1``): in that case the
bias-corrected estimate and its robust variance are exactly those of the
local-*quadratic* fit at ``h`` (CCT 2014; verified against ``rdrobust`` during
development). RBC is the default inference in this module's diagnostics.
Variance: HC1 here vs ``rdrobust``'s nearest-neighbour estimator (≈1% apart in
the development check).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats

from decision_intelligence.common.validation import (
    FloatArray,
    as_float_array,
    require_probability,
)
from decision_intelligence.statistics.linear_regression import fit_ols

RDD_ASSUMPTIONS = (
    "continuity of both potential-outcome regressions at the cutoff",
    "no manipulation (sorting) of the running variable around the cutoff",
    "no other treatment switching on at the same cutoff",
    "estimand is local to the cutoff (not a population ATE)",
)
_IK_TRIANGULAR = 3.4375


@dataclass(frozen=True)
class RDDEstimate:
    estimand: str
    estimate: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    bandwidth: float
    n_left: int
    n_right: int
    method: str
    assumptions: tuple[str, ...] = RDD_ASSUMPTIONS
    warnings: tuple[str, ...] = ()

    def covers(self, value: float) -> bool:
        return self.ci_lower <= value <= self.ci_upper


def _inputs(running: ArrayLike, outcome: ArrayLike) -> tuple[FloatArray, FloatArray]:
    x = as_float_array(running, name="running", min_size=10)
    y = as_float_array(outcome, name="outcome", min_size=10)
    if x.shape != y.shape:
        raise ValueError("running variable and outcome must have the same length")
    return x, y


def triangular_kernel(u: FloatArray) -> FloatArray:
    return np.asarray(np.clip(1 - np.abs(u), 0, None), dtype=np.float64)


def local_polynomial_rdd(
    running: ArrayLike,
    outcome: ArrayLike,
    cutoff: float,
    bandwidth: float,
    *,
    order: int = 1,
    confidence_level: float = 0.95,
    min_per_side: int = 10,
) -> RDDEstimate:
    """Local polynomial estimate of the jump at ``cutoff`` (triangular kernel, separate fits).

    Regression within ``|x - c| < h`` with weights ``K((x - c)/h)``:
    ``y = a + tau D + sum_p [b_p (x - c)^p + d_p D (x - c)^p]``; ``tau`` is the
    estimate. ``order=1`` is the conventional local-linear estimator.
    """
    x, y = _inputs(running, outcome)
    confidence_level = require_probability(confidence_level, "confidence_level")
    if not bandwidth > 0 or not math.isfinite(bandwidth):
        raise ValueError("bandwidth must be positive and finite")
    if order not in (1, 2):
        raise ValueError("order must be 1 (local linear) or 2 (local quadratic)")
    centred = x - cutoff
    w = triangular_kernel(centred / bandwidth)
    keep = w > 0
    d = (centred >= 0).astype(float)
    n_left, n_right = int(np.sum(keep & (d == 0))), int(np.sum(keep & (d == 1)))
    if min(n_left, n_right) < max(min_per_side, 2 * (order + 1)):
        raise ValueError(
            f"too few observations inside the bandwidth (left {n_left}, right {n_right}; "
            f"need {min_per_side} per side)"
        )
    root = np.sqrt(w[keep])
    columns = {"intercept": np.ones(keep.sum()), "jump": d[keep]}
    for power in range(1, order + 1):
        columns[f"slope_{power}"] = centred[keep] ** power
        columns[f"slope_change_{power}"] = d[keep] * centred[keep] ** power
    design = pd.DataFrame({k: v * root for k, v in columns.items()})
    result = fit_ols(
        design, y[keep] * root, add_intercept=False, covariance="HC1",
        confidence_level=confidence_level,
    )
    k = result.feature_names.index("jump")
    warnings: list[str] = []
    if min(n_left, n_right) < 50:
        warnings.append("fewer than 50 observations on one side: estimate is imprecise")
    name = "local linear" if order == 1 else "local quadratic"
    return RDDEstimate(
        estimand="tau(c): average effect for units at the cutoff",
        estimate=float(result.coefficients[k]),
        standard_error=float(result.standard_errors[k]),
        ci_lower=float(result.ci_lower[k]),
        ci_upper=float(result.ci_upper[k]),
        confidence_level=confidence_level,
        bandwidth=float(bandwidth),
        n_left=n_left,
        n_right=n_right,
        method=f"{name}, triangular kernel, HC1 (conventional)",
        warnings=tuple(warnings),
    )


def local_linear_rdd(
    running: ArrayLike,
    outcome: ArrayLike,
    cutoff: float,
    bandwidth: float,
    *,
    confidence_level: float = 0.95,
    min_per_side: int = 10,
) -> RDDEstimate:
    """Conventional local-linear estimate (MSE-optimal point estimate; CI ignores bias)."""
    return local_polynomial_rdd(
        running, outcome, cutoff, bandwidth, order=1,
        confidence_level=confidence_level, min_per_side=min_per_side,
    )


def robust_bias_corrected_rdd(
    running: ArrayLike,
    outcome: ArrayLike,
    cutoff: float,
    bandwidth: float,
    *,
    confidence_level: float = 0.95,
    min_per_side: int = 10,
) -> RDDEstimate:
    """Robust bias-corrected estimate and CI (Calonico, Cattaneo & Titiunik 2014, ``rho = 1``).

    The local-linear estimate minus its estimated leading bias, with a
    variance that accounts for the bias estimation; with the pilot bandwidth
    equal to ``h`` this is exactly the local-quadratic fit at ``h``. Wider than
    the conventional interval, but with valid coverage at MSE-optimal ``h``.
    """
    estimate = local_polynomial_rdd(
        running, outcome, cutoff, bandwidth, order=2,
        confidence_level=confidence_level, min_per_side=min_per_side,
    )
    return replace(
        estimate,
        method="robust bias-corrected (CCT 2014, rho = 1 = local quadratic at h), HC1",
    )


def ik_bandwidth(running: ArrayLike, outcome: ArrayLike, cutoff: float) -> float:
    """Imbens & Kalyanaraman (2012) MSE-optimal bandwidth for local linear, triangular kernel.

    1. Pilot ``h1 = 1.84 S_x N^-1/5``: density ``f(c)`` and conditional
       variances on each side from observations within ``h1``.
    2. Third derivative from a global cubic (common slope terms, separate
       intercepts) between the medians of ``X`` on each side; pilot
       bandwidths ``h2 = 3.56 (sigma^2 / (f m3^2))^1/7 N_side^-1/7``; local
       quadratic fits on each side give second derivatives ``m''_±``.
    3. ``h = 3.4375 [ (sigma^2_- + sigma^2_+) / (f ((m''_+ - m''_-)^2 + r_+ + r_-)) ]^1/5 N^-1/5``
       with regularisation ``r_± = 2160 sigma^2_± / (N_2± h2±^4)``.
    """
    x, y = _inputs(running, outcome)
    n = x.size
    left_all, right_all = x < cutoff, x >= cutoff
    if left_all.sum() < 10 or right_all.sum() < 10:
        raise ValueError("need at least 10 observations on each side of the cutoff")
    h1 = 1.84 * float(np.std(x, ddof=1)) * n ** (-0.2)
    near_left = left_all & (x >= cutoff - h1)
    near_right = right_all & (x <= cutoff + h1)
    if near_left.sum() < 3 or near_right.sum() < 3:
        raise ValueError("too few observations near the cutoff for the IK pilot bandwidth")
    density = (near_left.sum() + near_right.sum()) / (2 * n * h1)
    var_left = float(np.var(y[near_left], ddof=1))
    var_right = float(np.var(y[near_right], ddof=1))

    median_left = float(np.median(x[left_all]))
    median_right = float(np.median(x[right_all]))
    middle = (x >= median_left) & (x <= median_right)
    c = x[middle] - cutoff
    cubic = fit_ols(
        np.column_stack([(c >= 0).astype(float), c, c**2, c**3]), y[middle]
    )
    m3 = 6 * float(cubic.coefficients[-1])
    m3_sq = max(m3**2, 1e-12)

    def second_derivative(
        side: np.ndarray, variance: float, n_side: int
    ) -> tuple[float, float, int]:
        h2 = 3.56 * (variance / (density * m3_sq)) ** (1 / 7) * n_side ** (-1 / 7)
        window = side & (np.abs(x - cutoff) <= h2)
        if window.sum() < 4:
            raise ValueError("too few observations for the IK second-derivative step")
        z = x[window] - cutoff
        quadratic = fit_ols(np.column_stack([z, z**2]), y[window])
        return 2 * float(quadratic.coefficients[-1]), h2, int(window.sum())

    m2_left, h2_left, n2_left = second_derivative(left_all, var_left, int(left_all.sum()))
    m2_right, h2_right, n2_right = second_derivative(right_all, var_right, int(right_all.sum()))
    r_left = 2160 * var_left / (n2_left * h2_left**4)
    r_right = 2160 * var_right / (n2_right * h2_right**4)
    denominator = density * ((m2_right - m2_left) ** 2 + r_left + r_right)
    return float(_IK_TRIANGULAR * ((var_left + var_right) / denominator) ** 0.2 * n ** (-0.2))


def bandwidth_sensitivity(
    running: ArrayLike, outcome: ArrayLike, cutoff: float, bandwidths: Sequence[float]
) -> pd.DataFrame:
    """Local-linear estimate and robust bias-corrected interval at each bandwidth."""
    rows = []
    for h in bandwidths:
        conventional = local_linear_rdd(running, outcome, cutoff, h)
        robust = robust_bias_corrected_rdd(running, outcome, cutoff, h)
        rows.append(
            {"bandwidth": h, "local_linear": conventional.estimate,
             "conventional_ci_lower": conventional.ci_lower,
             "conventional_ci_upper": conventional.ci_upper,
             "bias_corrected": robust.estimate, "robust_ci_lower": robust.ci_lower,
             "robust_ci_upper": robust.ci_upper,
             "n_left": conventional.n_left, "n_right": conventional.n_right}
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class DensityTest:
    method: str
    statistic: float
    p_value: float
    log_density_difference: float | None
    note: str


def mccrary_density_test(
    running: ArrayLike,
    cutoff: float,
    *,
    bin_width: float | None = None,
    bandwidth: float | None = None,
) -> DensityTest:
    """McCrary (2008) test of a discontinuity in the running variable's density at ``c``.

    Histogram with bins ending exactly at ``c`` (default width
    ``2 sd(X) N^-1/2``), local linear fits of normalised bin heights on each
    side (triangular kernel, bandwidth ``h``), ``theta = ln f_+ - ln f_-`` with
    ``se = sqrt( (24/5) (1/f_+ + 1/f_-) / (N h) )``. Default ``h`` is McCrary's
    rule of thumb from global quartic fits on each side. A rejection signals
    possible sorting; a non-rejection cannot rule out manipulation that does
    not change the density (e.g. two-way sorting).
    """
    x = as_float_array(running, name="running", min_size=50)
    n = x.size
    b = bin_width or 2 * float(np.std(x, ddof=1)) * n ** (-0.5)
    left_edge = cutoff - b * math.ceil((cutoff - float(np.min(x))) / b)
    right_edge = cutoff + b * math.ceil((float(np.max(x)) - cutoff) / b + 1e-12)
    edges = np.arange(left_edge, right_edge + b / 2, b)
    counts, _ = np.histogram(x, bins=edges)
    mids = (edges[:-1] + edges[1:]) / 2
    heights = counts / (n * b)
    left, right = mids < cutoff, mids > cutoff
    if left.sum() < 5 or right.sum() < 5:
        raise ValueError("too few histogram bins on one side of the cutoff")

    if bandwidth is None:
        widths = []
        for side in (left, right):
            z = mids[side] - cutoff
            quartic = fit_ols(np.column_stack([z, z**2, z**3, z**4]), heights[side])
            coef = quartic.coefficients
            curvature = 2 * coef[2] + 6 * coef[3] * z + 12 * coef[4] * z**2
            spread = float(np.ptp(mids[side])) or b
            sigma2 = quartic.residual_std_error**2
            roughness = max(float(np.sum(curvature**2)), 1e-12)
            widths.append(3.348 * (sigma2 * spread / roughness) ** 0.2)
        bandwidth = float(np.mean(widths))

    def boundary_height(side: np.ndarray) -> float:
        z = mids[side] - cutoff
        w = triangular_kernel(z / bandwidth)
        keep = w > 0
        if keep.sum() < 3:
            raise ValueError("too few bins inside the density-test bandwidth")
        root = np.sqrt(w[keep])
        fit = fit_ols(
            np.column_stack([np.ones(keep.sum()), z[keep]]) * root[:, None],
            heights[side][keep] * root,
            add_intercept=False,
        )
        return float(fit.coefficients[0])

    f_minus, f_plus = boundary_height(left), boundary_height(right)
    if f_minus <= 0 or f_plus <= 0:
        raise ValueError("estimated density at the cutoff is not positive; test undefined")
    theta = math.log(f_plus) - math.log(f_minus)
    se = math.sqrt((24 / 5) * (1 / f_plus + 1 / f_minus) / (n * bandwidth))
    z_stat = theta / se
    return DensityTest(
        method=f"McCrary (bin width {b:.3g}, bandwidth {bandwidth:.3g})",
        statistic=z_stat,
        p_value=float(2 * stats.norm.sf(abs(z_stat))),
        log_density_difference=theta,
        note="tests the density only; cannot detect sorting that leaves the density smooth",
    )


def binomial_density_test(running: ArrayLike, cutoff: float, window: float) -> DensityTest:
    """Counts in ``[c - w, c)`` vs ``[c, c + w)``: Binomial(n_w, 1/2) under a locally flat density.

    Approximate (in the spirit of Bugni & Canay 2021): valid as ``w`` shrinks;
    a sloped density biases it for wide windows. Exact two-sided binomial p.
    """
    x = as_float_array(running, name="running", min_size=10)
    if not window > 0:
        raise ValueError("window must be positive")
    below = int(np.sum((x >= cutoff - window) & (x < cutoff)))
    above = int(np.sum((x >= cutoff) & (x < cutoff + window)))
    if below + above == 0:
        raise ValueError("no observations inside the window")
    result = stats.binomtest(above, below + above, 0.5)
    return DensityTest(
        method=f"binomial window test (w = {window:g}; {below} below, {above} above)",
        statistic=above / (below + above),
        p_value=float(result.pvalue),
        log_density_difference=None,
        note="approximate: assumes a locally flat density within the window",
    )


def placebo_cutoffs(
    running: ArrayLike, outcome: ArrayLike, cutoff: float, placebos: Sequence[float],
    bandwidth: float,
) -> pd.DataFrame:
    """RDD at fake cutoffs (robust bias-corrected), using only data on the same side of the
    real cutoff. Each should show no jump; rejections indicate the method finds
    'effects' where none exist (mis-specification or unrelated discontinuities)."""
    x, y = _inputs(running, outcome)
    rows = []
    for fake in placebos:
        if fake == cutoff:
            raise ValueError("placebo cutoff equals the real cutoff")
        side = x < cutoff if fake < cutoff else x >= cutoff
        est = robust_bias_corrected_rdd(x[side], y[side], fake, bandwidth)
        rows.append({"placebo_cutoff": fake, "estimate": est.estimate,
                     "ci_lower": est.ci_lower, "ci_upper": est.ci_upper,
                     "covers_zero": est.covers(0.0)})
    return pd.DataFrame(rows)


def donut_rdd(
    running: ArrayLike, outcome: ArrayLike, cutoff: float, bandwidth: float, donut: float
) -> RDDEstimate:
    """Local linear RDD excluding ``|x - c| < donut`` (sensitivity to units nearest the cutoff,
    where sorting concentrates). Extrapolates over the hole; less precise by design."""
    x, y = _inputs(running, outcome)
    keep = np.abs(x - cutoff) >= donut
    estimate = robust_bias_corrected_rdd(x[keep], y[keep], cutoff, bandwidth)
    return replace(estimate, method=f"{estimate.method}; donut {donut:g}")


def covariate_continuity(
    running: ArrayLike, covariates: pd.DataFrame, cutoff: float, bandwidth: float
) -> pd.DataFrame:
    """Robust bias-corrected 'effects' on pre-determined covariates; under continuity ~0."""
    x = as_float_array(running, name="running", min_size=10)
    rows = []
    for column in covariates.columns:
        est = robust_bias_corrected_rdd(x, covariates[column].to_numpy(float), cutoff, bandwidth)
        rows.append({"covariate": column, "jump": est.estimate, "ci_lower": est.ci_lower,
                     "ci_upper": est.ci_upper, "covers_zero": est.covers(0.0)})
    return pd.DataFrame(rows)


def global_polynomial_rdd(
    running: ArrayLike, outcome: ArrayLike, cutoff: float, order: int,
    *, confidence_level: float = 0.95,
) -> RDDEstimate:
    """Jump from a separate global polynomial of ``order`` on each side (all data, HC1).

    Included as the *cautionary* comparison: high-order global polynomials
    let distant observations drive the boundary fit and are sensitive to
    the order (Gelman & Imbens 2019). Not recommended for estimation.
    """
    x, y = _inputs(running, outcome)
    if order < 1:
        raise ValueError("order must be at least 1")
    c = (x - cutoff) / float(np.std(x, ddof=1))
    d = (c >= 0).astype(float)
    columns = {"jump": d}
    for p in range(1, order + 1):
        columns[f"c{p}"] = c**p
        columns[f"dc{p}"] = d * c**p
    result = fit_ols(pd.DataFrame(columns), y, covariance="HC1", confidence_level=confidence_level)
    k = result.feature_names.index("jump")
    return RDDEstimate(
        estimand="tau(c) via global polynomial (not recommended)",
        estimate=float(result.coefficients[k]),
        standard_error=float(result.standard_errors[k]),
        ci_lower=float(result.ci_lower[k]),
        ci_upper=float(result.ci_upper[k]),
        confidence_level=confidence_level,
        bandwidth=float("inf"),
        n_left=int(np.sum(d == 0)),
        n_right=int(np.sum(d == 1)),
        method=f"global polynomial of order {order} on each side, HC1",
        warnings=("global polynomial: boundary estimate driven by distant observations",),
    )
