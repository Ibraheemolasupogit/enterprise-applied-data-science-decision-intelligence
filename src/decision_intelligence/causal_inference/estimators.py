"""Backdoor-adjustment estimators of the ATE and ATT.

All estimators target an estimand defined on potential outcomes
(:mod:`.estimands`) and are valid only under the identification
assumptions below, which no estimator can check from the data alone:

1. **Consistency** — the observed outcome equals the potential outcome under
   the treatment actually received (one well-defined version of treatment).
2. **No interference (SUTVA)** — a unit's outcome does not depend on other
   units' treatments.
3. **Conditional exchangeability** — ``Y(t) ⊥ T | X`` for the adjustment set
   ``X``; on a DAG, ``X`` satisfies the backdoor criterion (:mod:`.dag`).
4. **Positivity** — ``0 < P(T = 1 | X) < 1`` for every ``X`` in the target
   population (for the ATT only ``P(T = 1 | X) < 1`` is needed).

Under 1–4, ``E[Y(t)] = E_X[ E[Y | T = t, X] ]`` and the estimators below
are different routes to that functional:

* **Regression adjustment (g-computation)**: fit ``mu_t(x) = E[Y | T=t, X=x]``
  separately in each arm, average ``mu_1 - mu_0`` over the target
  population. Consistent if the outcome models are correctly specified.
* **Inverse probability weighting**: reweight each arm by ``1 / e(X)`` or
  ``1 / (1 - e(X))``, with ``e(X) = P(T=1 | X)``. Consistent if the
  propensity model is correct. Horvitz–Thompson (unnormalised) and Hájek
  (normalised weights; bounded by the outcome range, usually preferred).
* **Augmented IPW (doubly robust)**: outcome-model prediction plus an
  inverse-weighted residual correction; consistent if *either* model is
  correct, and semiparametrically efficient when both are.

Uncertainty
-----------
``se_method="influence_function"`` uses the empirical variance of the
estimator's influence function:

* **IPW with an estimated propensity**: the influence function includes the
  M-estimation correction for the logistic propensity fit,
  ``phi* = phi + H I^-1 s_i`` (``s_i`` the logistic score, ``I`` its
  information, ``H`` the mean derivative of ``phi`` in the propensity
  coefficients; Lunceford & Davidian 2004). Ignoring that step is
  conservative for the ATE but can be *anti*-conservative for the ATT; the
  corrected form is neither and matches statsmodels' joint-GMM standard
  errors (tested).
* **IPW with a known propensity** (supplied, not estimated): no correction
  is needed — the uncorrected form is exact.
* **AIPW**: nuisance models treated as fixed. Asymptotically exact when
  *both* models are correctly specified (each model's estimation error then
  has zero first-order effect); under misspecification of one model, prefer
  ``se_method="bootstrap"``.

``se_method="bootstrap"`` resamples customers and refits every model
(nonparametric bootstrap via :func:`decision_intelligence.common.bootstrap_ci`),
propagating nuisance-estimation uncertainty; regression adjustment always
uses it. Neither accounts for model *selection* done after looking at
outcomes.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats

from decision_intelligence.common.bootstrap import bootstrap_ci
from decision_intelligence.common.validation import (
    FloatArray,
    as_float_array,
    require_positive_int,
    require_probability,
)
from decision_intelligence.statistics.hypothesis_tests import two_sample_t_test
from decision_intelligence.statistics.linear_regression import fit_ols
from decision_intelligence.statistics.logistic_regression import SeparationError, fit_logistic

Estimand = Literal["ATE", "ATT"]
SEMethod = Literal["influence_function", "bootstrap"]
OutcomeModelKind = Literal["auto", "logistic", "linear"]

IDENTIFICATION_ASSUMPTIONS = (
    "consistency: one well-defined version of treatment",
    "no interference between units (SUTVA)",
    "conditional exchangeability given the adjustment set (backdoor criterion)",
    "positivity: every adjustment-set stratum has treated and untreated units",
)
_EXTREME_PROPENSITY = 0.01


class PositivityError(ValueError):
    """Treatment is (near-)deterministic given the covariates: positivity fails."""


@dataclass(frozen=True)
class CausalEstimate:
    """A point estimate with its estimand, uncertainty and conditions."""

    estimand: str
    estimator: str
    estimate: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    se_method: str
    n: int
    n_treated: int
    adjustment_set: tuple[str, ...]
    assumptions: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    def covers(self, value: float) -> bool:
        return self.ci_lower <= value <= self.ci_upper


@dataclass(frozen=True)
class CausalData:
    """Validated treatment, outcome and adjustment covariates for one analysis."""

    treatment: FloatArray
    outcome: FloatArray
    covariates: pd.DataFrame

    @classmethod
    def from_arrays(
        cls, treatment: ArrayLike, outcome: ArrayLike, covariates: pd.DataFrame | None = None
    ) -> CausalData:
        t = as_float_array(treatment, name="treatment", min_size=4)
        y = as_float_array(outcome, name="outcome", min_size=4)
        if t.shape != y.shape:
            raise ValueError("treatment and outcome must have the same length")
        if not np.isin(t, (0.0, 1.0)).all():
            raise ValueError("treatment must be binary (0/1)")
        n_treated = int(t.sum())
        if n_treated < 2 or t.size - n_treated < 2:
            raise ValueError("need at least two treated and two untreated units")
        frame = pd.DataFrame(index=range(t.size)) if covariates is None else covariates
        if len(frame) != t.size:
            raise ValueError("covariates must have one row per unit")
        values = frame.to_numpy(dtype=np.float64) if frame.shape[1] else np.empty((t.size, 0))
        if not np.isfinite(values).all():
            raise ValueError("covariates contain NaN or infinite values")
        return cls(
            treatment=t,
            outcome=y,
            covariates=pd.DataFrame(values, columns=[str(c) for c in frame.columns]),
        )

    @property
    def n(self) -> int:
        return int(self.treatment.size)

    @property
    def n_treated(self) -> int:
        return int(self.treatment.sum())

    @property
    def adjustment_set(self) -> tuple[str, ...]:
        return tuple(self.covariates.columns)

    def subset(self, indices: ArrayLike) -> CausalData:
        rows = np.asarray(indices, dtype=int)
        return CausalData(
            treatment=self.treatment[rows],
            outcome=self.outcome[rows],
            covariates=self.covariates.iloc[rows].reset_index(drop=True),
        )


# ---------------------------------------------------------------------------
# Nuisance models
# ---------------------------------------------------------------------------


def estimate_propensity(data: CausalData) -> FloatArray:
    """``e(X) = P(T = 1 | X)`` by main-effects logistic regression (M3 ``fit_logistic``).

    With no covariates the propensity is the marginal treated share (an
    intercept-only logistic model). Perfect prediction of treatment by the
    covariates is a *structural* positivity violation and raises
    :class:`PositivityError`.
    """
    if data.covariates.shape[1] == 0:
        return np.full(data.n, data.treatment.mean())
    try:
        model = fit_logistic(data.covariates, data.treatment)
    except SeparationError as error:
        raise PositivityError(
            "the covariates perfectly (or quasi-perfectly) predict treatment: some covariate "
            "region contains only treated or only untreated units, so positivity fails and "
            "the effect there is not identified without extrapolation"
        ) from error
    return model.fitted_probabilities


def _resolve_kind(outcome: FloatArray, kind: str) -> Literal["logistic", "linear"]:
    if kind == "auto":
        return "logistic" if np.isin(outcome, (0.0, 1.0)).all() else "linear"
    if kind == "logistic":
        if not np.isin(outcome, (0.0, 1.0)).all():
            raise ValueError("logistic outcome model requires a binary outcome")
        return "logistic"
    if kind == "linear":
        return "linear"
    raise ValueError("outcome_model must be 'auto', 'logistic' or 'linear'")


def fit_outcome_models(
    data: CausalData, kind: OutcomeModelKind = "auto"
) -> tuple[FloatArray, FloatArray]:
    """Arm-specific outcome regressions ``mu_0(X)``, ``mu_1(X)`` predicted for every unit.

    Fitting each arm separately allows the effect to vary with ``X`` instead
    of imposing a constant effect on the model's scale. Predicting an arm's
    model outside that arm's covariate range is extrapolation — another
    reason positivity matters even for regression adjustment.
    """
    resolved = _resolve_kind(data.outcome, kind)
    predictions: list[FloatArray] = []
    for arm in (0.0, 1.0):
        in_arm = data.treatment == arm
        y_arm = data.outcome[in_arm]
        if data.covariates.shape[1] == 0:
            predictions.append(np.full(data.n, y_arm.mean()))
            continue
        x_arm = data.covariates[in_arm]
        if resolved == "logistic":
            if y_arm.min() == y_arm.max():
                predictions.append(np.full(data.n, y_arm.mean()))
                continue
            try:
                model = fit_logistic(x_arm, y_arm)
            except SeparationError as error:
                raise ValueError(
                    f"outcome model for arm T={int(arm)} is not estimable (separation); "
                    "use outcome_model='linear' or coarsen the covariates"
                ) from error
            predictions.append(model.predict_probability(data.covariates.to_numpy()))
        else:
            linear = fit_ols(x_arm, y_arm)
            predictions.append(linear.predict(data.covariates.to_numpy()))
    return predictions[0], predictions[1]


# ---------------------------------------------------------------------------
# Point estimators (return the estimate and its influence function)
# ---------------------------------------------------------------------------

PointEstimator = Callable[[CausalData], tuple[float, FloatArray | None]]


def _propensity_warnings(scores: FloatArray, estimand: str) -> list[str]:
    warnings: list[str] = []
    upper = float(np.mean(scores > 1 - _EXTREME_PROPENSITY))
    lower = float(np.mean(scores < _EXTREME_PROPENSITY))
    if upper > 0:
        warnings.append(
            f"{upper:.1%} of units have propensity > {1 - _EXTREME_PROPENSITY}: near-violations "
            "of positivity; control weights are extreme"
        )
    if lower > 0 and estimand == "ATE":
        warnings.append(
            f"{lower:.1%} of units have propensity < {_EXTREME_PROPENSITY}: near-violations of "
            "positivity; treated weights are extreme"
        )
    return warnings


def _propensity_design(data: CausalData) -> FloatArray:
    """Design matrix of the logistic propensity model (intercept + covariates)."""
    return np.column_stack([np.ones(data.n), data.covariates.to_numpy(dtype=np.float64)])


def _propensity_estimation_correction(
    data: CausalData, e: FloatArray, derivative_weights: FloatArray
) -> FloatArray:
    """Per-unit term ``H I^-1 s_i`` accounting for estimating the logistic propensity.

    ``derivative_weights`` holds ``g_i`` with ``d phi_i / d beta = g_i x_i``
    (``x_i`` the propensity design row), so ``H = mean(g_i x_i)``; the
    logistic score is ``s_i = x_i (T_i - e_i)`` and its information
    ``I = mean(e_i (1 - e_i) x_i x_i')``.
    """
    design = _propensity_design(data)
    n = data.n
    gradient = (design * derivative_weights[:, None]).mean(axis=0)
    information = design.T @ (design * (e * (1 - e))[:, None]) / n
    scores = design * (data.treatment - e)[:, None]
    return np.asarray(scores @ np.linalg.solve(information, gradient), dtype=np.float64)


def _ipw_point(
    data: CausalData,
    estimand: Estimand,
    normalised: bool,
    propensity: FloatArray | None,
) -> tuple[float, FloatArray]:
    """IPW estimate and its influence function.

    With ``propensity=None`` the propensity is estimated and the influence
    function carries the estimation correction; a supplied (known)
    propensity needs none.
    """
    t, y = data.treatment, data.outcome
    estimated = propensity is None
    e = estimate_propensity(data) if propensity is None else propensity
    if estimand == "ATE":
        if normalised:
            m1 = float(np.sum(t * y / e) / np.sum(t / e))
            m0 = float(np.sum((1 - t) * y / (1 - e)) / np.sum((1 - t) / (1 - e)))
            tau = m1 - m0
            influence = t * (y - m1) / e - (1 - t) * (y - m0) / (1 - e)
            derivative = -t * (y - m1) * (1 - e) / e - (1 - t) * (y - m0) * e / (1 - e)
        else:
            tau = float(np.mean(t * y / e - (1 - t) * y / (1 - e)))
            influence = t * y / e - (1 - t) * y / (1 - e) - tau
            derivative = -t * y * (1 - e) / e - (1 - t) * y * e / (1 - e)
    else:
        odds = e / (1 - e)
        p = float(t.mean())
        mean_treated = float(y[t == 1].mean())
        m0 = float(np.sum((1 - t) * odds * y) / np.sum((1 - t) * odds))
        tau = mean_treated - m0
        influence = (t * (y - mean_treated) - (1 - t) * odds * (y - m0)) / p
        derivative = -(1 - t) * odds * (y - m0) / p
    if estimated:
        influence = influence + _propensity_estimation_correction(data, e, derivative)
    return tau, np.asarray(influence, dtype=np.float64)


def _aipw_point(
    data: CausalData,
    estimand: Estimand,
    kind: OutcomeModelKind,
    propensity: FloatArray | None,
) -> tuple[float, FloatArray]:
    t, y = data.treatment, data.outcome
    e = estimate_propensity(data) if propensity is None else propensity
    mu0, mu1 = fit_outcome_models(data, kind)
    if estimand == "ATE":
        scores = mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)
        tau = float(np.mean(scores))
        return tau, np.asarray(scores - tau, dtype=np.float64)
    p = float(t.mean())
    odds = e / (1 - e)
    tau = float(np.sum(t * (y - mu0) - (1 - t) * odds * (y - mu0)) / np.sum(t))
    influence = (t * (y - mu0 - tau) - (1 - t) * odds * (y - mu0)) / p
    return tau, np.asarray(influence, dtype=np.float64)


def _regression_point(data: CausalData, estimand: Estimand, kind: OutcomeModelKind) -> float:
    mu0, mu1 = fit_outcome_models(data, kind)
    if estimand == "ATE":
        return float(np.mean(mu1 - mu0))
    treated = data.treatment == 1
    return float(np.mean(data.outcome[treated] - mu0[treated]))


# ---------------------------------------------------------------------------
# Public estimators
# ---------------------------------------------------------------------------


def naive_difference(data: CausalData, *, confidence_level: float = 0.95) -> CausalEstimate:
    """``E[Y | T=1] - E[Y | T=0]`` with a Welch interval (M3).

    An *association*. It equals the ATE only if treatment is independent of
    the potential outcomes (e.g. randomised); otherwise it equals
    ATT + selection bias (:class:`.estimands.NaiveDecomposition`).
    """
    t, y = data.treatment, data.outcome
    test = two_sample_t_test(y[t == 1], y[t == 0], confidence_level=confidence_level)
    if test.ci_lower is None or test.ci_upper is None:  # pragma: no cover - always set
        raise RuntimeError("Welch test returned no interval")
    se = (test.ci_upper - test.ci_lower) / (
        2 * float(stats.t.ppf(0.5 + confidence_level / 2, test.df))
    )
    return CausalEstimate(
        estimand="association (difference in means; not causal without exchangeability)",
        estimator="naive difference in means",
        estimate=test.estimate,
        standard_error=se,
        ci_lower=test.ci_lower,
        ci_upper=test.ci_upper,
        confidence_level=confidence_level,
        se_method="Welch",
        n=data.n,
        n_treated=data.n_treated,
        adjustment_set=(),
        assumptions=("independent units",),
    )


def regression_adjustment(
    data: CausalData,
    *,
    estimand: Estimand = "ATE",
    outcome_model: OutcomeModelKind = "auto",
    rng: np.random.Generator,
    n_bootstrap: int = 500,
    confidence_level: float = 0.95,
) -> CausalEstimate:
    """G-computation (standardisation) with arm-specific outcome models; bootstrap CI.

    ATE: ``mean_i [mu_1(X_i) - mu_0(X_i)]``. ATT: ``mean_{T=1} [Y_i - mu_0(X_i)]``.
    """
    estimand = _require_estimand(estimand)
    estimate = _regression_point(data, estimand, outcome_model)

    def statistic(sample: CausalData, rows: FloatArray) -> float:
        return _regression_point(sample, estimand, outcome_model)

    se, lower, upper = _bootstrap(data, statistic, rng, n_bootstrap, confidence_level)
    return CausalEstimate(
        estimand=estimand,
        estimator=(
            "regression adjustment (g-computation, "
            f"{_resolve_kind(data.outcome, outcome_model)} outcome models)"
        ),
        estimate=estimate,
        standard_error=se,
        ci_lower=lower,
        ci_upper=upper,
        confidence_level=confidence_level,
        se_method=f"nonparametric bootstrap ({n_bootstrap}, percentile)",
        n=data.n,
        n_treated=data.n_treated,
        adjustment_set=data.adjustment_set,
        assumptions=(*IDENTIFICATION_ASSUMPTIONS, "outcome models correctly specified"),
    )


def inverse_probability_weighting(
    data: CausalData,
    *,
    estimand: Estimand = "ATE",
    normalised: bool = True,
    propensity: ArrayLike | None = None,
    se_method: SEMethod = "influence_function",
    rng: np.random.Generator | None = None,
    n_bootstrap: int = 500,
    confidence_level: float = 0.95,
) -> CausalEstimate:
    """IPW estimator of the ATE (weights ``1/e``, ``1/(1-e)``) or ATT (control weights ``e/(1-e)``).

    ``propensity`` supplies *known* assignment probabilities instead of
    estimating them. Valid only if treatment was actually randomised with
    exactly these probabilities (an experiment, or a stochastic targeting
    policy that logs them) — then identification holds even with the
    confounders unmeasured. A deterministic rule's score is not a
    propensity, and ordinary observational data contain no such
    probabilities. Supplying *estimated* probabilities here treats them as
    known, so the influence-function SE omits their estimation error.
    """
    estimand = _require_estimand(estimand)
    known = _validate_propensity(propensity, data.n)
    estimate, influence = _ipw_point(data, estimand, normalised, known)
    scores = estimate_propensity(data) if known is None else known

    def statistic(sample: CausalData, rows: FloatArray) -> float:
        given = None if known is None else known[rows.astype(int)]
        return _ipw_point(sample, estimand, normalised, given)[0]

    se, lower, upper, method = _uncertainty(
        data, estimate, influence, statistic, se_method, rng, n_bootstrap, confidence_level,
        "known propensity" if known is not None
        else "propensity estimation accounted for",
    )
    kind = "Hájek (normalised)" if normalised else "Horvitz–Thompson"
    source = "known propensity" if known is not None else "estimated propensity (logistic)"
    return CausalEstimate(
        estimand=estimand,
        estimator=f"IPW, {kind}, {source}",
        estimate=estimate,
        standard_error=se,
        ci_lower=lower,
        ci_upper=upper,
        confidence_level=confidence_level,
        se_method=method,
        n=data.n,
        n_treated=data.n_treated,
        adjustment_set=data.adjustment_set if known is None else ("<known propensity>",),
        assumptions=(
            *IDENTIFICATION_ASSUMPTIONS,
            "propensity model correctly specified" if known is None
            else "supplied propensities equal the true assignment probabilities",
        ),
        warnings=tuple(_propensity_warnings(scores, estimand)),
    )


def augmented_ipw(
    data: CausalData,
    *,
    estimand: Estimand = "ATE",
    outcome_model: OutcomeModelKind = "auto",
    propensity: ArrayLike | None = None,
    se_method: SEMethod = "influence_function",
    rng: np.random.Generator | None = None,
    n_bootstrap: int = 500,
    confidence_level: float = 0.95,
) -> CausalEstimate:
    """Doubly robust AIPW estimator of the ATE or ATT.

    ATE scores: ``mu1 - mu0 + T (Y - mu1)/e - (1-T)(Y - mu0)/(1-e)``.
    ATT: ``sum[T (Y - mu0) - (1-T) e/(1-e) (Y - mu0)] / n_1``.
    """
    estimand = _require_estimand(estimand)
    known = _validate_propensity(propensity, data.n)
    estimate, influence = _aipw_point(data, estimand, outcome_model, known)
    scores = estimate_propensity(data) if known is None else known

    def statistic(sample: CausalData, rows: FloatArray) -> float:
        given = None if known is None else known[rows.astype(int)]
        return _aipw_point(sample, estimand, outcome_model, given)[0]

    se, lower, upper, method = _uncertainty(
        data, estimate, influence, statistic, se_method, rng, n_bootstrap, confidence_level,
        "nuisance models treated as fixed; valid when both are correctly specified",
    )
    return CausalEstimate(
        estimand=estimand,
        estimator="AIPW (doubly robust)",
        estimate=estimate,
        standard_error=se,
        ci_lower=lower,
        ci_upper=upper,
        confidence_level=confidence_level,
        se_method=method,
        n=data.n,
        n_treated=data.n_treated,
        adjustment_set=data.adjustment_set,
        assumptions=(
            *IDENTIFICATION_ASSUMPTIONS,
            "outcome model OR propensity model correctly specified",
        ),
        warnings=tuple(_propensity_warnings(scores, estimand)),
    )


@dataclass(frozen=True)
class RiskRatioEstimate:
    """Marginal risks ``E[Y(1)]``, ``E[Y(0)]`` and their ratio (binary outcome, AIPW)."""

    risk_treated: float
    risk_untreated: float
    risk_ratio: float
    ci_lower: float
    ci_upper: float
    confidence_level: float


def aipw_risk_ratio(
    data: CausalData, *, confidence_level: float = 0.95, propensity: ArrayLike | None = None
) -> RiskRatioEstimate:
    """AIPW marginal risk ratio with a delta-method interval on the log scale."""
    if not np.isin(data.outcome, (0.0, 1.0)).all():
        raise ValueError("risk ratio requires a binary outcome")
    confidence_level = require_probability(confidence_level, "confidence_level")
    known = _validate_propensity(propensity, data.n)
    t, y = data.treatment, data.outcome
    e = estimate_propensity(data) if known is None else known
    mu0, mu1 = fit_outcome_models(data, "logistic")
    psi1 = mu1 + t * (y - mu1) / e
    psi0 = mu0 + (1 - t) * (y - mu0) / (1 - e)
    r1, r0 = float(psi1.mean()), float(psi0.mean())
    if r1 <= 0 or r0 <= 0:
        raise ValueError("estimated marginal risk is not positive; risk ratio undefined")
    log_influence = (psi1 - r1) / r1 - (psi0 - r0) / r0
    se = float(np.std(log_influence, ddof=1)) / math.sqrt(data.n)
    z = float(stats.norm.ppf(0.5 + confidence_level / 2))
    log_rr = math.log(r1 / r0)
    return RiskRatioEstimate(
        risk_treated=r1,
        risk_untreated=r0,
        risk_ratio=r1 / r0,
        ci_lower=math.exp(log_rr - z * se),
        ci_upper=math.exp(log_rr + z * se),
        confidence_level=confidence_level,
    )


def subgroup_effects(
    data: CausalData,
    groups: ArrayLike,
    *,
    estimand: Estimand = "ATE",
    outcome_model: OutcomeModelKind = "auto",
    confidence_level: float = 0.95,
) -> dict[str, CausalEstimate]:
    """Conditional effects (CATE for a discrete modifier) by AIPW within each subgroup.

    Each subgroup is a separate analysis with its own nuisance models; the
    adjustment set must remain sufficient *within* subgroups. Differences
    between subgroup effects are effect-measure modification on the chosen
    (here, additive) scale and may vanish or appear on another scale.
    """
    labels = np.asarray(groups).astype(str)
    if labels.shape != data.treatment.shape:
        raise ValueError("groups must align with the data")
    results: dict[str, CausalEstimate] = {}
    for label in np.unique(labels):
        sample = data.subset(np.flatnonzero(labels == label))
        estimate = augmented_ipw(
            sample, estimand=estimand, outcome_model=outcome_model,
            confidence_level=confidence_level,
        )
        results[label] = replace(estimate, estimand=f"{estimand} | group = {label}")
    return results


@dataclass(frozen=True)
class HeterogeneityTest:
    """Cochran-type test that subgroup effects are equal (on the estimated scale)."""

    pooled_effect: float
    q_statistic: float
    df: int
    p_value: float


def heterogeneity_test(estimates: Mapping[str, CausalEstimate]) -> HeterogeneityTest:
    """``Q = sum_g (tau_g - tau_bar)^2 / se_g^2`` with inverse-variance pooled ``tau_bar``.

    Chi-square with ``G - 1`` df under equal subgroup effects and independent
    subgroups (true for disjoint subgroups). Low power with few, small
    subgroups: a non-significant Q is not evidence of homogeneity, and
    subgroups examined after seeing the data invalidate the p-value.
    """
    if len(estimates) < 2:
        raise ValueError("need at least two subgroup estimates")
    values = np.array([e.estimate for e in estimates.values()])
    errors = np.array([e.standard_error for e in estimates.values()])
    if (errors <= 0).any() or not np.isfinite(errors).all():
        raise ValueError("subgroup standard errors must be positive and finite")
    weights = 1 / errors**2
    pooled = float(np.sum(weights * values) / np.sum(weights))
    q = float(np.sum(weights * (values - pooled) ** 2))
    df = len(estimates) - 1
    return HeterogeneityTest(
        pooled_effect=pooled, q_statistic=q, df=df, p_value=float(stats.chi2.sf(q, df))
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _require_estimand(estimand: str) -> Estimand:
    if estimand == "ATE":
        return "ATE"
    if estimand == "ATT":
        return "ATT"
    raise ValueError("estimand must be 'ATE' or 'ATT'")


def _validate_propensity(propensity: ArrayLike | None, n: int) -> FloatArray | None:
    if propensity is None:
        return None
    scores = as_float_array(propensity, name="propensity")
    if scores.size != n:
        raise ValueError("propensity must have one value per unit")
    if ((scores <= 0) | (scores >= 1)).any():
        raise PositivityError("supplied propensities must lie strictly inside (0, 1)")
    return scores


def _bootstrap(
    data: CausalData,
    statistic: Callable[[CausalData, FloatArray], float],
    rng: np.random.Generator | None,
    n_bootstrap: int,
    confidence_level: float,
) -> tuple[float, float, float]:
    """Resample units with replacement, refit every model, percentile interval.

    Built on the shared M3 primitive; ``statistic`` receives the resampled
    data and the resampled row indices (for aligning known propensities). A
    resample on which the estimator is undefined (an empty arm, separation)
    makes the bootstrap raise rather than silently drop that resample.
    """
    if rng is None:
        raise ValueError("bootstrap uncertainty requires an explicit rng")
    n_bootstrap = require_positive_int(n_bootstrap, "n_bootstrap", minimum=100)

    def on_rows(rows: FloatArray) -> float:
        sample = data.subset(rows)
        if sample.n_treated < 2 or sample.n - sample.n_treated < 2:
            return math.nan
        try:
            return statistic(sample, rows)
        except ValueError:
            return math.nan

    result = bootstrap_ci(
        np.arange(data.n, dtype=np.float64),
        on_rows,
        rng=rng,
        n_resamples=n_bootstrap,
        confidence_level=confidence_level,
        method="percentile",
    )
    return result.standard_error, result.ci_lower, result.ci_upper


def _uncertainty(
    data: CausalData,
    estimate: float,
    influence: FloatArray,
    statistic: Callable[[CausalData, FloatArray], float],
    se_method: str,
    rng: np.random.Generator | None,
    n_bootstrap: int,
    confidence_level: float,
    influence_label: str,
) -> tuple[float, float, float, str]:
    confidence_level = require_probability(confidence_level, "confidence_level")
    if se_method == "influence_function":
        se = float(np.std(influence, ddof=1)) / math.sqrt(data.n)
        z = float(stats.norm.ppf(0.5 + confidence_level / 2))
        return se, estimate - z * se, estimate + z * se, f"influence function ({influence_label})"
    if se_method == "bootstrap":
        se, lower, upper = _bootstrap(data, statistic, rng, n_bootstrap, confidence_level)
        return se, lower, upper, f"nonparametric bootstrap ({n_bootstrap}, percentile)"
    raise ValueError("se_method must be 'influence_function' or 'bootstrap'")
