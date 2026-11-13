"""Variance reduction with pre-treatment covariates: CUPED and regression adjustment.

Randomisation already makes the difference in means unbiased; a covariate
measured *before* assignment cannot change that, but it can explain part of
the outcome's variance and so shrink the standard error. Two estimators:

* **CUPED** (Deng, Xu, Kohavi & Walker 2013) — :func:`cuped`. Replace the
  outcome by ``Y - (X - mean(X))' theta`` with ``theta`` the pooled
  least-squares slope of ``Y`` on ``X`` (all units, both arms), then compare
  arms with the M6 Welch analysis. Variance falls by the factor ``1 - R^2``
  (``1 - rho^2`` for one covariate) asymptotically.
* **Regression adjustment with treatment interactions** (Lin 2013) —
  :func:`regression_adjustment`. OLS of ``Y`` on ``Z``, centred ``X`` and
  ``Z x X`` with HC2 standard errors (M3 OLS). Never less efficient than the
  unadjusted estimator asymptotically, even with heterogeneous effects or
  unequal allocation, and its HC2 SE is consistent; CUPED's single slope can
  be slightly less efficient when effects vary with ``X``.

Both target the same estimand as the unadjusted analysis — the ITT average
effect — and are consistent (bias ``O(1/n)``), *provided the covariate is not
affected by treatment*. That proviso is the whole method:

* a covariate measured after assignment (``in-experiment sessions``) or a
  "pre-period" window that overlaps the experiment absorbs part of the
  treatment effect and **biases** the estimate (demonstrated in the M7
  evidence). :func:`check_adjustment_covariate` refuses a covariate whose
  measurement window ends after assignment and reports the arm difference as
  supporting (not conclusive) evidence;
* covariates and the adjustment method are pre-specified; choosing them after
  seeing the treatment comparison is a form of specification search.

Units without pre-period data (new users) are handled by mean imputation plus
a missingness indicator — valid because both are pre-treatment — rather than
by dropping them, which would change the estimand.

The variance reduction a covariate will deliver must be estimated from
historical (pre-experiment) data when *planning* (:func:`plan_with_covariate`);
the experiment itself only reports the reduction realised.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from decision_intelligence.causal_inference.estimators import CausalEstimate
from decision_intelligence.common.validation import FloatArray
from decision_intelligence.experimentation.analysis import (
    ITT_ESTIMAND,
    MetricResult,
    analyse_continuous,
    split_by_arm,
)
from decision_intelligence.experimentation.planning import ExperimentPlan, plan_continuous
from decision_intelligence.experimentation.validity import (
    CheckResult,
    check_post_randomisation_selection,
)
from decision_intelligence.statistics.linear_regression import fit_ols


def _covariate_matrix(covariates: ArrayLike | pd.DataFrame | Mapping[str, ArrayLike]
                      ) -> tuple[FloatArray, list[str]]:
    if isinstance(covariates, pd.DataFrame):
        frame = covariates
    elif isinstance(covariates, Mapping):
        frame = pd.DataFrame(dict(covariates))
    else:
        values = np.asarray(covariates, dtype=np.float64)
        values = values.reshape(-1, 1) if values.ndim == 1 else values
        frame = pd.DataFrame(values, columns=[f"x{i}" for i in range(values.shape[1])])
    return frame.to_numpy(dtype=np.float64), [str(c) for c in frame.columns]


def prepare_covariates(
    covariates: ArrayLike | pd.DataFrame | Mapping[str, ArrayLike],
) -> tuple[FloatArray, list[str], int]:
    """Mean-impute missing pre-treatment covariate values and add a missingness indicator
    per affected column (Lin-style handling; pre-treatment, so it cannot bias the ITT).

    Returns the design columns, their names and the number of units with any missing value.
    """
    x, names = _covariate_matrix(covariates)
    if np.isinf(x).any():
        raise ValueError("covariates contain infinite values")
    columns, labels = [], []
    missing_any = np.zeros(x.shape[0], dtype=bool)
    for j, name in enumerate(names):
        column = x[:, j]
        missing = np.isnan(column)
        if missing.all():
            raise ValueError(f"covariate {name!r} is missing for every unit")
        filled = np.where(missing, float(np.nanmean(column)), column)
        columns.append(filled)
        labels.append(name)
        if missing.any():
            columns.append(missing.astype(float))
            labels.append(f"{name}_missing")
            missing_any |= missing
    matrix = np.column_stack(columns)
    constant = np.ptp(matrix, axis=0) == 0
    if constant.any():
        names = [labels[i] for i in np.flatnonzero(constant)]
        raise ValueError(f"constant covariate columns: {names}")
    return np.asarray(matrix, dtype=np.float64), labels, int(missing_any.sum())


@dataclass(frozen=True)
class VarianceReductionResult:
    """An adjusted ITT estimate beside the unadjusted M6 estimate on the same data."""

    method: str
    effect: CausalEstimate
    unadjusted: MetricResult
    coefficients: dict[str, float]
    """CUPED: the pooled slopes ``theta``; Lin: the covariate main-effect coefficients."""
    r_squared: float
    """Share of outcome variance explained by the covariates (pooled; ``rho^2`` for one)."""
    n_missing_covariate: int

    @property
    def standard_error_ratio(self) -> float:
        """Adjusted / unadjusted SE on these data."""
        return self.effect.standard_error / self.unadjusted.effect.standard_error

    @property
    def realised_variance_reduction(self) -> float:
        """``1 - (SE_adjusted / SE_unadjusted)^2``; compare with ``r_squared`` (theory)."""
        return 1.0 - self.standard_error_ratio**2

    @property
    def equivalent_sample_size_factor(self) -> float:
        """The unadjusted analysis would need ``1 / (1 - reduction)`` times as many units."""
        return 1.0 / max(1e-12, 1.0 - self.realised_variance_reduction)


def cuped(
    outcome: ArrayLike,
    arm: ArrayLike,
    covariates: ArrayLike | pd.DataFrame | Mapping[str, ArrayLike],
    *,
    control: str = "control",
    treatment: str = "treatment",
    confidence_level: float = 0.95,
    metric: str = "revenue per user",
) -> VarianceReductionResult:
    """CUPED: Welch difference in means of ``Y - (X - mean X)' theta`` (pooled ``theta``)."""
    y = np.asarray(outcome, dtype=np.float64)
    labels = np.asarray(arm).astype(str)
    unadjusted = analyse_continuous(y, labels, metric=metric, control=control,
                                    treatment=treatment, confidence_level=confidence_level)
    x, names, n_missing = prepare_covariates(covariates)
    if x.shape[0] != y.size:
        raise ValueError("covariates must have one row per unit")
    centred = x - x.mean(axis=0)
    theta, *_ = np.linalg.lstsq(centred, y - y.mean(), rcond=None)
    adjusted = y - centred @ theta
    result = analyse_continuous(adjusted, labels, metric=f"{metric} (CUPED)", control=control,
                                treatment=treatment, confidence_level=confidence_level)
    r2 = 1.0 - float(np.var(adjusted)) / float(np.var(y))
    effect = CausalEstimate(
        estimand=ITT_ESTIMAND,
        estimator="CUPED: difference in means of the covariate-adjusted outcome",
        estimate=result.effect.estimate,
        standard_error=result.effect.standard_error,
        ci_lower=result.effect.ci_lower,
        ci_upper=result.effect.ci_upper,
        confidence_level=confidence_level,
        se_method=result.effect.se_method + "; theta treated as fixed (asymptotically exact)",
        n=result.effect.n,
        n_treated=result.effect.n_treated,
        adjustment_set=tuple(names),
        assumptions=("random assignment", "covariates measured before assignment "
                     "(unaffected by treatment)", "no interference (SUTVA)"),
        warnings=result.effect.warnings,
    )
    return VarianceReductionResult("CUPED (pooled theta)", effect, unadjusted,
                                   dict(zip(names, (float(t) for t in theta), strict=True)),
                                   r2, n_missing)


def regression_adjustment(
    outcome: ArrayLike,
    arm: ArrayLike,
    covariates: ArrayLike | pd.DataFrame | Mapping[str, ArrayLike],
    *,
    control: str = "control",
    treatment: str = "treatment",
    confidence_level: float = 0.95,
    metric: str = "revenue per user",
) -> VarianceReductionResult:
    """Lin (2013): OLS of ``Y`` on ``Z``, centred ``X`` and ``Z x X``; HC2 SE (M3 OLS)."""
    y = np.asarray(outcome, dtype=np.float64)
    labels = np.asarray(arm).astype(str)
    split_by_arm(y, labels, control=control, treatment=treatment)  # validation
    unadjusted = analyse_continuous(y, labels, metric=metric, control=control,
                                    treatment=treatment, confidence_level=confidence_level)
    x, names, n_missing = prepare_covariates(covariates)
    if x.shape[0] != y.size:
        raise ValueError("covariates must have one row per unit")
    z = (labels == treatment).astype(float)
    centred = x - x.mean(axis=0)
    design = np.column_stack([z, centred, z[:, None] * centred])
    feature_names = ["treatment", *names, *(f"treatment:{n}" for n in names)]
    fit = fit_ols(design, y, feature_names=feature_names, covariance="HC2",
                  confidence_level=confidence_level)
    index = 1  # coefficient 0 is the intercept
    pooled = fit_ols(centred, y, feature_names=names, covariance="classical")
    effect = CausalEstimate(
        estimand=ITT_ESTIMAND,
        estimator="regression adjustment with treatment interactions (Lin 2013)",
        estimate=float(fit.coefficients[index]),
        standard_error=float(fit.standard_errors[index]),
        ci_lower=float(fit.ci_lower[index]),
        ci_upper=float(fit.ci_upper[index]),
        confidence_level=confidence_level,
        se_method=f"HC2 sandwich, t({fit.df_resid:.0f})",
        n=y.size,
        n_treated=int(z.sum()),
        adjustment_set=tuple(names),
        assumptions=("random assignment", "covariates measured before assignment "
                     "(unaffected by treatment)", "no interference (SUTVA)"),
        warnings=fit.warnings,
    )
    coefficients = {n: float(fit.coefficients[2 + i]) for i, n in enumerate(names)}
    return VarianceReductionResult("regression adjustment (Lin 2013, HC2)", effect, unadjusted,
                                   coefficients, float(pooled.r_squared), n_missing)


def check_adjustment_covariate(
    name: str,
    values: ArrayLike,
    assigned: ArrayLike,
    *,
    window_end: pd.Timestamp,
    assignment_time: pd.Timestamp,
) -> CheckResult:
    """FAIL when the covariate's measurement window ends after assignment.

    Timing is the decisive check: a covariate measured (even partly) after assignment can
    absorb treatment effect. The arm difference is reported as evidence (M6
    :func:`~.validity.check_post_randomisation_selection`); a non-significant difference
    does not make a post-assignment covariate safe.
    """
    v = np.asarray(values, dtype=np.float64)
    observed = ~np.isnan(v)
    result = check_post_randomisation_selection(
        name, v[observed], np.asarray(assigned, dtype=float)[observed],
        measured_after_assignment=bool(window_end > assignment_time),
    )
    timing = (f"window ends {window_end:%Y-%m-%d}, assignment {assignment_time:%Y-%m-%d}: "
              + ("after assignment — must not be used for adjustment. "
                 if window_end > assignment_time else "pre-treatment. "))
    return CheckResult(f"adjustment covariate: {name}", result.status, timing + result.detail,
                       result.value)


def plan_with_covariate(
    baseline_mean: float,
    standard_deviation: float,
    *,
    historical_r_squared: float,
    mde_absolute: float,
    alpha: float = 0.05,
    power: float = 0.8,
    daily_eligible: float | None = None,
    metric: str = "revenue per user",
) -> tuple[ExperimentPlan, ExperimentPlan]:
    """Unadjusted and CUPED plans (M6 ``plan_continuous``): the adjusted residual SD is
    ``sd * sqrt(1 - R^2)`` with ``R^2`` estimated on *historical* data, never on the experiment.
    """
    if not 0.0 <= historical_r_squared < 1.0:
        raise ValueError("historical_r_squared must lie in [0, 1)")
    plain = plan_continuous(baseline_mean, standard_deviation, mde_absolute=mde_absolute,
                            alpha=alpha, power=power, daily_eligible=daily_eligible,
                            metric=metric)
    adjusted = plan_continuous(
        baseline_mean, standard_deviation * math.sqrt(1.0 - historical_r_squared),
        mde_absolute=mde_absolute, alpha=alpha, power=power, daily_eligible=daily_eligible,
        metric=f"{metric} (CUPED)",
    )
    return plain, adjusted


__all__ = [
    "VarianceReductionResult",
    "check_adjustment_covariate",
    "cuped",
    "plan_with_covariate",
    "prepare_covariates",
    "regression_adjustment",
]
