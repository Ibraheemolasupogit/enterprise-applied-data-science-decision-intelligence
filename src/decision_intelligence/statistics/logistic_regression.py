"""Maximum-likelihood logistic regression and probabilistic-classification diagnostics.

Model
-----
``P(Y = 1 | x) = sigma(x'beta)`` with ``sigma(t) = 1 / (1 + e^-t)``. The
log-likelihood ``l(beta) = sum y_i eta_i - log(1 + e^eta_i)`` is concave, so
Newton–Raphson (equivalently IRLS) converges to the unique maximum *when
one exists*. Steps are halved whenever they fail to increase ``l``.

Each coefficient is a log-odds ratio: ``exp(beta_j)`` is the multiplicative
change in the odds of ``Y = 1`` associated with a one-unit difference in
``x_j`` holding the other regressors fixed. As with OLS that is an
association; a causal reading requires an identification argument.
Odds ratios are not risk ratios, and they diverge from them when the
outcome is common.

Failure modes handled explicitly
--------------------------------
* **Separation**: if a linear combination of regressors perfectly (or
  quasi-perfectly) predicts the outcome, the MLE does not exist — the
  likelihood keeps increasing as coefficients go to infinity. Detection is
  *exact*, not a heuristic on fitted probabilities: by Albert & Anderson
  (1984) the MLE fails to exist iff some ``beta != 0`` satisfies
  ``(2 y_i - 1) x_i' beta >= 0`` for every ``i``, which is checked before
  fitting with a linear programme (Konis 2007; see :func:`is_separated`).
  The fitter then raises :class:`SeparationError` instead of returning
  huge, meaningless coefficients with tiny "p-values". Remedies: penalised
  (Firth) likelihood, merging sparse categories, or exact methods.
* Non-binary or single-class outcomes, rank-deficient designs and
  non-convergence all raise.

Standard errors come from the inverse observed Fisher information
(``covariance="model"``) or the HC0 sandwich (``"robust"``), which remains
valid for the pseudo-true parameter under link misspecification.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import optimize, special, stats

from decision_intelligence.common.validation import (
    FloatArray,
    as_float_array,
    require_positive_int,
    require_probability,
)
from decision_intelligence.statistics._design import prepare_design, prepare_response

LogisticCovariance = Literal["model", "robust"]
_SEPARATION_TOLERANCE = 1e-7
"""Relative LP objective (per observation) above which the data are declared separated.

The LP below is scaled so its objective is 0 exactly when no separating
direction exists and O(1) per strictly separated observation otherwise;
HiGHS's default feasibility tolerance (1e-7 per constraint) can produce a
spurious objective of at most ~1e-7 per observation, which this threshold
absorbs.
"""


class SeparationError(ValueError):
    """The logistic MLE does not exist (complete or quasi-complete separation)."""


@dataclass(frozen=True)
class LogisticRegressionResult:
    """Fitted logistic regression with Wald inference."""

    feature_names: tuple[str, ...]
    coefficients: FloatArray
    standard_errors: FloatArray
    z_values: FloatArray
    p_values: FloatArray
    ci_lower: FloatArray
    ci_upper: FloatArray
    confidence_level: float
    covariance: FloatArray = field(repr=False)
    covariance_type: LogisticCovariance
    n_obs: int
    log_likelihood: float
    null_log_likelihood: float | None
    """Intercept-only log-likelihood (only defined when the model has an intercept)."""
    likelihood_ratio_statistic: float | None
    likelihood_ratio_p_value: float | None
    aic: float
    mcfadden_r_squared: float | None
    n_iterations: int
    has_intercept: bool
    fitted_probabilities: FloatArray = field(repr=False)

    @property
    def odds_ratios(self) -> FloatArray:
        return np.asarray(np.exp(self.coefficients), dtype=np.float64)

    def summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "estimate": self.coefficients,
                "std_error": self.standard_errors,
                "z_value": self.z_values,
                "p_value": self.p_values,
                "ci_lower": self.ci_lower,
                "ci_upper": self.ci_upper,
                "odds_ratio": self.odds_ratios,
            },
            index=pd.Index(self.feature_names, name="feature"),
        )

    def predict_probability(self, X: ArrayLike) -> FloatArray:
        """Predicted ``P(Y=1)`` for new rows in the original feature columns (no intercept)."""
        matrix = np.asarray(X, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        if self.has_intercept:
            matrix = np.column_stack([np.ones(matrix.shape[0]), matrix])
        if matrix.shape[1] != self.coefficients.size:
            raise ValueError("X has the wrong number of columns for this model")
        return np.asarray(special.expit(matrix @ self.coefficients), dtype=np.float64)


def is_separated(matrix: FloatArray, response: FloatArray) -> bool:
    """Exact check for complete or quasi-complete separation (Albert & Anderson 1984).

    With ``a_i = (2 y_i - 1) x_i``, the data are separated iff the linear
    programme ``max sum_i a_i' beta`` subject to ``a_i' beta >= 0`` for all ``i``
    and ``-1 <= beta_j <= 1`` has a strictly positive optimum: a feasible
    ``beta != 0`` classifies every observation weakly correctly, and (with a
    full-rank design) at least one strictly. ``beta = 0`` is always feasible
    with objective 0. Columns are rescaled to unit maximum magnitude first,
    which does not change whether a separating direction exists.
    """
    signed = matrix * (2.0 * response - 1.0)[:, None]
    scale = np.max(np.abs(signed), axis=0)
    scale[scale == 0] = 1.0
    signed = signed / scale
    n, k = signed.shape
    result = optimize.linprog(
        c=-signed.sum(axis=0),
        A_ub=-signed,
        b_ub=np.zeros(n),
        bounds=[(-1.0, 1.0)] * k,
        method="highs",
    )
    if result.status != 0:
        raise RuntimeError(f"separation check failed: linear programme status {result.status}")
    return bool(float(-result.fun) > _SEPARATION_TOLERANCE * n)


def _log_likelihood(y: FloatArray, eta: FloatArray) -> float:
    # sum y*eta - log(1 + e^eta), with logaddexp avoiding overflow for large |eta|.
    return float(np.sum(y * eta - np.logaddexp(0.0, eta)))


def fit_logistic(
    X: ArrayLike | pd.DataFrame,
    y: ArrayLike,
    *,
    feature_names: Sequence[str] | None = None,
    add_intercept: bool = True,
    covariance: str = "model",
    confidence_level: float = 0.95,
    max_iterations: int = 100,
    tolerance: float = 1e-10,
) -> LogisticRegressionResult:
    """Fit a logistic regression by damped Newton–Raphson.

    Raises:
        SeparationError: If the data are (quasi-)separated so no finite MLE exists.
        ValueError: For non-binary ``y``, a single observed class, a
            rank-deficient design, or failure to converge.
    """
    if covariance not in ("model", "robust"):
        raise ValueError("covariance must be 'model' or 'robust'")
    covariance_type: LogisticCovariance = "model" if covariance == "model" else "robust"
    confidence_level = require_probability(confidence_level, "confidence_level")
    max_iterations = require_positive_int(max_iterations, "max_iterations")
    design = prepare_design(X, feature_names, add_intercept)
    matrix = design.matrix
    n, k = matrix.shape
    response = prepare_response(y, n)
    if not np.isin(response, (0.0, 1.0)).all():
        raise ValueError("y must be binary (0/1 or boolean)")
    n_positive = float(response.sum())
    if n_positive in (0.0, float(n)):
        raise ValueError("y contains a single class; the logistic MLE does not exist")
    if is_separated(matrix, response):
        raise SeparationError(
            "complete or quasi-complete separation: a linear combination of the regressors "
            "classifies every observation (weakly) correctly, so the likelihood increases "
            "without bound and no finite MLE exists. Consider Firth's penalised likelihood, "
            "merging sparse categories, removing the separating regressor, or exact methods."
        )

    beta = np.zeros(k)
    eta = matrix @ beta
    log_lik = _log_likelihood(response, eta)
    converged = False
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        p = special.expit(eta)
        weights = p * (1 - p)
        gradient = matrix.T @ (response - p)
        hessian = matrix.T @ (matrix * weights[:, None])
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as error:  # pragma: no cover - excluded by is_separated
            raise ValueError(
                "information matrix became numerically singular during fitting"
            ) from error
        step_size = 1.0
        for _ in range(50):
            candidate = beta + step_size * step
            candidate_eta = matrix @ candidate
            candidate_log_lik = _log_likelihood(response, candidate_eta)
            if candidate_log_lik >= log_lik - 1e-12:
                break
            step_size /= 2
        improvement = candidate_log_lik - log_lik
        beta, eta, log_lik = candidate, candidate_eta, candidate_log_lik
        if np.max(np.abs(step_size * step)) < tolerance or abs(improvement) < tolerance * (
            abs(log_lik) + tolerance
        ):
            converged = True
            break

    if not converged:
        raise ValueError(f"logistic regression did not converge in {max_iterations} iterations")

    p = special.expit(eta)
    information = matrix.T @ (matrix * (p * (1 - p))[:, None])
    information_inverse = np.linalg.inv(information)
    if covariance_type == "model":
        cov = information_inverse
    else:
        scores = matrix * (response - p)[:, None]
        cov = information_inverse @ (scores.T @ scores) @ information_inverse

    standard_errors = np.sqrt(np.diag(cov))
    z_values = beta / standard_errors
    z_critical = float(stats.norm.ppf(0.5 + confidence_level / 2))

    null_log_lik: float | None = None
    lr_statistic: float | None = None
    lr_p_value: float | None = None
    mcfadden: float | None = None
    if design.has_intercept:
        rate = n_positive / n
        null_log_lik = n_positive * math.log(rate) + (n - n_positive) * math.log(1 - rate)
        if k > 1:
            lr_statistic = 2 * (log_lik - null_log_lik)
            lr_p_value = float(stats.chi2.sf(lr_statistic, k - 1))
        mcfadden = 1 - log_lik / null_log_lik

    return LogisticRegressionResult(
        feature_names=design.feature_names,
        coefficients=beta,
        standard_errors=standard_errors,
        z_values=z_values,
        p_values=2 * stats.norm.sf(np.abs(z_values)),
        ci_lower=beta - z_critical * standard_errors,
        ci_upper=beta + z_critical * standard_errors,
        confidence_level=confidence_level,
        covariance=cov,
        covariance_type=covariance_type,
        n_obs=n,
        log_likelihood=log_lik,
        null_log_likelihood=null_log_lik,
        likelihood_ratio_statistic=lr_statistic,
        likelihood_ratio_p_value=lr_p_value,
        aic=2 * k - 2 * log_lik,
        mcfadden_r_squared=mcfadden,
        n_iterations=iteration,
        has_intercept=design.has_intercept,
        fitted_probabilities=p,
    )


# ---------------------------------------------------------------------------
# Probabilistic-classification diagnostics
# ---------------------------------------------------------------------------


def roc_auc(y_true: ArrayLike, scores: ArrayLike) -> float:
    """Area under the ROC curve via the Mann–Whitney identity (ties count half).

    ``AUC = P(score of a random positive > score of a random negative)``. It
    measures ranking (discrimination) only: a model can have a high AUC and
    badly miscalibrated probabilities.
    """
    y = as_float_array(y_true, name="y_true", min_size=2)
    s = as_float_array(scores, name="scores", min_size=2)
    if y.shape != s.shape:
        raise ValueError("y_true and scores must have the same length")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("y_true must be binary")
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC undefined with a single class")
    ranks = stats.rankdata(s)
    u = float(ranks[y == 1].sum()) - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


@dataclass(frozen=True)
class ClassificationDiagnostics:
    """Discrimination and calibration of predicted probabilities."""

    n: int
    prevalence: float
    brier_score: float
    log_loss: float
    auc: float
    calibration_in_the_large: float
    """``mean(p) - mean(y)``: overall over/under-prediction."""
    calibration_slope: float
    """Slope of a logistic recalibration of ``y`` on ``logit(p)``; 1 is ideal, < 1 = overfit."""
    hosmer_lemeshow_statistic: float
    hosmer_lemeshow_p_value: float
    calibration_table: pd.DataFrame = field(repr=False)


def classification_diagnostics(
    y_true: ArrayLike, probabilities: ArrayLike, *, n_bins: int = 10
) -> ClassificationDiagnostics:
    """Evaluate predicted probabilities against binary outcomes.

    Should be computed on data *not* used to fit the model; in-sample
    calibration is optimistic (a fitted logistic model with an intercept
    has calibration-in-the-large exactly zero on its training data).

    The Hosmer–Lemeshow statistic groups by predicted-probability quantile
    and compares observed and expected counts (chi-square, ``n_bins - 2``
    df on development data). It is sensitive to the arbitrary binning and
    has low power at small ``n`` and excessive power at large ``n``; read
    the calibration table rather than relying on its p-value.
    """
    y = as_float_array(y_true, name="y_true", min_size=2)
    p = as_float_array(probabilities, name="probabilities", min_size=2)
    if y.shape != p.shape:
        raise ValueError("y_true and probabilities must have the same length")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("y_true must be binary")
    if ((p <= 0) | (p >= 1)).any():
        raise ValueError("probabilities must lie strictly inside (0, 1)")
    n_bins = require_positive_int(n_bins, "n_bins", minimum=3)

    eps = 1e-15
    clipped = np.clip(p, eps, 1 - eps)
    log_loss = float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)))

    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    frame = pd.DataFrame({"bin": bins, "y": y, "p": p})
    table = (
        frame.groupby("bin")
        .agg(n=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
        .reset_index(drop=True)
    )
    expected = table["n"] * table["mean_predicted"]
    observed = table["n"] * table["observed_rate"]
    denominator = expected * (1 - table["mean_predicted"])
    hl = float(np.sum((observed - expected) ** 2 / denominator))
    hl_df = max(len(table) - 2, 1)

    logit_p = np.log(clipped / (1 - clipped))
    try:
        slope = float(fit_logistic(logit_p, y, add_intercept=True).coefficients[1])
    except ValueError:
        slope = math.nan

    return ClassificationDiagnostics(
        n=y.size,
        prevalence=float(y.mean()),
        brier_score=float(np.mean((p - y) ** 2)),
        log_loss=log_loss,
        auc=roc_auc(y, p),
        calibration_in_the_large=float(p.mean() - y.mean()),
        calibration_slope=slope,
        hosmer_lemeshow_statistic=hl,
        hosmer_lemeshow_p_value=float(stats.chi2.sf(hl, hl_df)),
        calibration_table=table,
    )
