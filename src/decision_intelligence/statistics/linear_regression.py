"""Ordinary least squares with classical, heteroskedasticity-robust and cluster-robust inference.

Model and estimand
------------------
``y = X beta + u``. OLS estimates ``beta_hat = argmin ||y - X b||^2``,
computed here by a QR decomposition (numerically stable; never by forming
and inverting ``X'X``). ``beta_hat`` estimates the coefficients of the best
linear predictor of ``y`` given ``X``. Reading a coefficient as "the
effect of changing x_j" is a *causal* claim that needs an identification
argument (M4/M5); without one it is a conditional association.

Covariance estimators (``covariance=``)
--------------------------------------
* ``"classical"``: ``sigma^2 (X'X)^-1`` — assumes homoskedastic, independent
  errors.
* ``"HC0"``–``"HC3"``: White/MacKinnon sandwich estimators, valid under
  heteroskedasticity of unknown form with independent observations. HC0 is
  the original (biased down in small samples); HC1 scales by ``n/(n-k)``;
  HC2/HC3 rescale by leverage, with HC3 recommended for small/moderate ``n``
  (Long & Ervin 2000).
* ``"cluster"``: CR1 cluster-robust sandwich (Liang–Zeger with the
  Stata small-sample factor ``G/(G-1) * (n-1)/(n-k)``), valid when errors
  are correlated *within* clusters (repeated observations of a customer)
  but independent *between* them. Inference uses t with ``G - 1`` df.
  Unreliable with few clusters (roughly G < 30-50): the SEs are then
  biased down and a wild-cluster bootstrap is preferable.

Ignoring within-cluster dependence understates standard errors — often
severely — even though the point estimates are unaffected. The M3
evidence script demonstrates this on M2 transaction data.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import linalg, stats

from decision_intelligence.common.validation import FloatArray, require_probability
from decision_intelligence.statistics._design import prepare_design, prepare_response

CovarianceType = Literal["classical", "HC0", "HC1", "HC2", "HC3", "cluster"]
_COVARIANCE_TYPES: tuple[str, ...] = ("classical", "HC0", "HC1", "HC2", "HC3", "cluster")
_ZERO_RESIDUAL_RELATIVE_TOLERANCE = 1e-10


@dataclass(frozen=True)
class LinearRegressionResult:
    """Fitted OLS model with inference under the chosen covariance estimator."""

    feature_names: tuple[str, ...]
    coefficients: FloatArray
    standard_errors: FloatArray
    t_values: FloatArray
    p_values: FloatArray
    ci_lower: FloatArray
    ci_upper: FloatArray
    confidence_level: float
    covariance: FloatArray = field(repr=False)
    covariance_type: CovarianceType
    n_obs: int
    df_resid: float
    """Degrees of freedom used for t-based inference (``n - k``; ``G - 1`` for cluster)."""
    n_clusters: int | None
    r_squared: float
    adj_r_squared: float
    residual_std_error: float
    f_statistic: float | None
    f_p_value: float | None
    condition_number: float
    has_intercept: bool
    fitted_values: FloatArray = field(repr=False)
    residuals: FloatArray = field(repr=False)
    leverage: FloatArray = field(repr=False)
    design_matrix: FloatArray = field(repr=False)
    warnings: tuple[str, ...] = ()

    def summary_frame(self) -> pd.DataFrame:
        """Coefficient table indexed by feature name."""
        return pd.DataFrame(
            {
                "estimate": self.coefficients,
                "std_error": self.standard_errors,
                "t_value": self.t_values,
                "p_value": self.p_values,
                "ci_lower": self.ci_lower,
                "ci_upper": self.ci_upper,
            },
            index=pd.Index(self.feature_names, name="feature"),
        )

    def predict(self, X: ArrayLike) -> FloatArray:
        """Predictions for new rows given in the *original* feature columns (no intercept)."""
        matrix = np.asarray(X, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        if self.has_intercept:
            matrix = np.column_stack([np.ones(matrix.shape[0]), matrix])
        if matrix.shape[1] != self.coefficients.size:
            raise ValueError("X has the wrong number of columns for this model")
        return np.asarray(matrix @ self.coefficients, dtype=np.float64)


def fit_ols(
    X: ArrayLike | pd.DataFrame,
    y: ArrayLike,
    *,
    feature_names: Sequence[str] | None = None,
    add_intercept: bool = True,
    covariance: str = "classical",
    clusters: ArrayLike | None = None,
    confidence_level: float = 0.95,
) -> LinearRegressionResult:
    """Fit OLS and compute inference.

    Args:
        X: Feature matrix (rows = observations). Categorical variables must
            already be dummy-coded with a reference level dropped.
        y: Continuous response.
        covariance: One of ``classical``, ``HC0``–``HC3``, ``cluster``.
        clusters: Cluster labels, required for (and only for) ``cluster``.
    """
    if covariance not in _COVARIANCE_TYPES:
        raise ValueError(f"covariance must be one of {_COVARIANCE_TYPES}")
    covariance_type = cast(CovarianceType, covariance)
    confidence_level = require_probability(confidence_level, "confidence_level")
    design = prepare_design(X, feature_names, add_intercept)
    matrix = design.matrix
    n, k = matrix.shape
    response = prepare_response(y, n)

    q, r = np.linalg.qr(matrix)
    coefficients = linalg.solve_triangular(r, q.T @ response)
    r_inverse = linalg.solve_triangular(r, np.eye(k))
    xtx_inverse = r_inverse @ r_inverse.T
    fitted = matrix @ coefficients
    residuals = response - fitted
    leverage = np.sum(q**2, axis=1)
    rss = float(residuals @ residuals)
    df_model_resid = n - k
    sigma2 = rss / df_model_resid

    warnings: list[str] = []
    n_clusters: int | None = None
    df_inference = float(df_model_resid)
    if covariance_type == "classical":
        if clusters is not None:
            raise ValueError("clusters given but covariance is not 'cluster'")
        cov = sigma2 * xtx_inverse
    elif covariance_type == "cluster":
        if clusters is None:
            raise ValueError("covariance='cluster' requires cluster labels")
        cov, n_clusters = _cluster_covariance(matrix, residuals, xtx_inverse, clusters)
        df_inference = float(n_clusters - 1)
        if n_clusters < 30:
            warnings.append(
                f"only {n_clusters} clusters; cluster-robust SEs are biased down with few "
                "clusters — consider a wild-cluster bootstrap"
            )
    else:
        if clusters is not None:
            raise ValueError("clusters given but covariance is not 'cluster'")
        cov = _hc_covariance(matrix, residuals, xtx_inverse, leverage, covariance_type)

    # Residuals at rounding-error level (exact linear relationship or constant response):
    # every covariance estimator collapses to ~0 and would report spurious certainty
    # (p-values like 1e-120). Point estimates and predictions remain valid; inference
    # is undefined and reported as NaN.
    residuals_vanish = rss <= n * (_ZERO_RESIDUAL_RELATIVE_TOLERANCE * float(
        np.max(np.abs(response))
    )) ** 2
    if residuals_vanish:
        warnings.append(
            "residuals are numerically zero (exact linear fit or constant response): "
            "standard errors, t/p-values, confidence intervals and the F-test are undefined"
        )
        cov = np.full((k, k), math.nan)

    standard_errors = np.sqrt(np.diag(cov))
    t_values = coefficients / standard_errors
    p_values = 2 * stats.t.sf(np.abs(t_values), df_inference)
    t_critical = float(stats.t.ppf(0.5 + confidence_level / 2, df_inference))

    if design.has_intercept:
        tss = float(np.sum((response - response.mean()) ** 2))
    else:
        tss = float(response @ response)
    r_squared = 1 - rss / tss if tss > 0 else math.nan
    df_model = k - 1 if design.has_intercept else k
    adj_r_squared = (
        1 - (1 - r_squared) * (n - int(design.has_intercept)) / df_model_resid
        if tss > 0
        else math.nan
    )

    f_statistic: float | None = None
    f_p_value: float | None = None
    slopes = slice(1, k) if design.has_intercept else slice(0, k)
    if df_model > 0 and not residuals_vanish:
        b = coefficients[slopes]
        v = cov[slopes, slopes]
        if int(np.linalg.matrix_rank(v)) < df_model:
            # E.g. a cluster-robust covariance has rank <= G - 1, so a joint Wald test
            # of more than G - 1 restrictions is not identified.
            warnings.append(
                "joint F-test not computed: the coefficient covariance is singular "
                f"(rank < {df_model} restrictions), e.g. fewer clusters than slopes"
            )
        else:
            f_statistic = float(b @ np.linalg.solve(v, b)) / df_model
            f_p_value = float(stats.f.sf(f_statistic, df_model, df_inference))

    condition_number = float(np.linalg.cond(matrix))
    if condition_number > 1e8:
        warnings.append(
            f"condition number {condition_number:.2e}: near-collinearity; coefficients "
            "and SEs are numerically fragile"
        )

    return LinearRegressionResult(
        feature_names=design.feature_names,
        coefficients=coefficients,
        standard_errors=standard_errors,
        t_values=t_values,
        p_values=p_values,
        ci_lower=coefficients - t_critical * standard_errors,
        ci_upper=coefficients + t_critical * standard_errors,
        confidence_level=confidence_level,
        covariance=cov,
        covariance_type=covariance_type,
        n_obs=n,
        df_resid=df_inference,
        n_clusters=n_clusters,
        r_squared=r_squared,
        adj_r_squared=adj_r_squared,
        residual_std_error=math.sqrt(sigma2),
        f_statistic=f_statistic,
        f_p_value=f_p_value,
        condition_number=condition_number,
        has_intercept=design.has_intercept,
        fitted_values=fitted,
        residuals=residuals,
        leverage=leverage,
        design_matrix=matrix,
        warnings=tuple(warnings),
    )


def _hc_covariance(
    matrix: FloatArray,
    residuals: FloatArray,
    xtx_inverse: FloatArray,
    leverage: FloatArray,
    covariance_type: str,
) -> FloatArray:
    n, k = matrix.shape
    squared = residuals**2
    if covariance_type in ("HC2", "HC3"):
        if (leverage > 1 - 1e-10).any():
            raise ValueError(
                f"{covariance_type} undefined: an observation has leverage 1 (it alone "
                "determines a coefficient); use HC1 or remove/inspect that observation"
            )
        power = 1 if covariance_type == "HC2" else 2
        squared = squared / (1 - leverage) ** power
    meat = matrix.T @ (matrix * squared[:, None])
    cov = xtx_inverse @ meat @ xtx_inverse
    if covariance_type == "HC1":
        cov *= n / (n - k)
    return np.asarray(cov, dtype=np.float64)


def _cluster_covariance(
    matrix: FloatArray,
    residuals: FloatArray,
    xtx_inverse: FloatArray,
    clusters: ArrayLike,
) -> tuple[FloatArray, int]:
    labels = np.asarray(clusters)
    n, k = matrix.shape
    if labels.shape != (n,):
        raise ValueError("clusters must have one label per observation")
    if pd.isna(labels).any():
        raise ValueError("cluster labels must not contain missing values")
    _, codes = np.unique(labels, return_inverse=True)
    n_clusters = int(np.max(codes)) + 1
    if n_clusters < 2:
        raise ValueError("cluster-robust covariance needs at least two clusters")
    scores = matrix * residuals[:, None]
    cluster_scores = np.zeros((n_clusters, k))
    np.add.at(cluster_scores, codes, scores)
    meat = cluster_scores.T @ cluster_scores
    correction = n_clusters / (n_clusters - 1) * (n - 1) / (n - k)
    return np.asarray(correction * xtx_inverse @ meat @ xtx_inverse, dtype=np.float64), n_clusters


@dataclass(frozen=True)
class OLSDiagnostics:
    """Residual, influence and specification diagnostics for an OLS fit.

    Diagnostics describe the fitted model; none of them proves it correct.
    ``durbin_watson`` is only meaningful when rows are in a natural order
    (time); for cross-sectional data in arbitrary order it is noise.
    """

    studentized_residuals: FloatArray = field(repr=False)
    cooks_distance: FloatArray = field(repr=False)
    high_leverage_indices: tuple[int, ...]
    influential_indices: tuple[int, ...]
    breusch_pagan_statistic: float | None
    breusch_pagan_p_value: float | None
    jarque_bera_statistic: float
    jarque_bera_p_value: float
    durbin_watson: float
    vif: dict[str, float]


def ols_diagnostics(result: LinearRegressionResult) -> OLSDiagnostics:
    """Compute diagnostics for a fitted OLS model.

    * Internally studentised residuals ``u_i / (sigma sqrt(1 - h_i))``.
    * Cook's distance ``r_i^2 h_i / (k (1 - h_i))``; flagged above ``4/n``.
    * High leverage: ``h_i > 2k/n``.
    * Breusch–Pagan (Koenker's studentised LM form, robust to non-normal
      errors): ``n R^2`` from regressing ``u^2`` on the regressors,
      chi-square with ``k - 1`` df. Requires an intercept.
    * Jarque–Bera normality of residuals: ``n/6 (S^2 + (K-3)^2/4)``, chi-square(2).
      Normal errors matter for exact small-sample t inference, not for the
      consistency of OLS.
    * Variance inflation factors ``1 / (1 - R_j^2)`` for every non-intercept
      column (needs an intercept; infinite under exact collinearity).
    """
    matrix = result.design_matrix
    n, k = matrix.shape
    residuals = result.residuals
    leverage = result.leverage
    sigma = result.residual_std_error
    with np.errstate(divide="ignore", invalid="ignore"):
        studentized = residuals / (sigma * np.sqrt(1 - leverage))
        cooks = studentized**2 * leverage / (k * (1 - leverage))

    bp_statistic: float | None = None
    bp_p_value: float | None = None
    if result.has_intercept and k > 1:
        squared = residuals**2
        auxiliary = fit_ols(matrix[:, 1:], squared, add_intercept=True)
        bp_statistic = n * auxiliary.r_squared
        bp_p_value = float(stats.chi2.sf(bp_statistic, k - 1))

    centred = residuals - residuals.mean()
    m2 = float(np.mean(centred**2))
    skew = float(np.mean(centred**3)) / m2**1.5 if m2 > 0 else 0.0
    kurt = float(np.mean(centred**4)) / m2**2 if m2 > 0 else 3.0
    jb = n / 6 * (skew**2 + (kurt - 3) ** 2 / 4)

    vif: dict[str, float] = {}
    if result.has_intercept:
        for j in range(1, k):
            others = np.delete(matrix, j, axis=1)[:, 1:]
            if others.shape[1] == 0:
                vif[result.feature_names[j]] = 1.0
                continue
            r2 = fit_ols(others, matrix[:, j], add_intercept=True).r_squared
            vif[result.feature_names[j]] = math.inf if r2 >= 1 else 1 / (1 - r2)

    return OLSDiagnostics(
        studentized_residuals=studentized,
        cooks_distance=cooks,
        high_leverage_indices=tuple(int(i) for i in np.flatnonzero(leverage > 2 * k / n)),
        influential_indices=tuple(int(i) for i in np.flatnonzero(cooks > 4 / n)),
        breusch_pagan_statistic=bp_statistic,
        breusch_pagan_p_value=bp_p_value,
        jarque_bera_statistic=jb,
        jarque_bera_p_value=float(stats.chi2.sf(jb, 2)),
        durbin_watson=float(np.sum(np.diff(residuals) ** 2) / np.sum(residuals**2)),
        vif=vif,
    )
