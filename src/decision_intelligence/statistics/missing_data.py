"""Missing-data mechanisms, simple handling strategies and their consequences.

Rubin's taxonomy, for an outcome ``Y`` with missingness indicator ``R``:

* **MCAR** — ``P(R | Y, X) = P(R)``: missingness unrelated to anything.
  Complete-case analysis is unbiased (only less precise).
* **MAR** — ``P(R | Y, X) = P(R | X)`` for *observed* ``X``: missingness
  depends only on observed data. Complete-case means are biased when ``X``
  is related to ``Y``; methods that condition on ``X`` (regression
  imputation, inverse-probability weighting, multiple imputation) can
  remove the bias, *if* the model for ``X`` is right.
* **MNAR** — missingness depends on the unobserved value itself. No
  method using only the observed data identifies the full-data mean
  without untestable assumptions; sensitivity analysis is required.

The mechanism cannot be determined from the observed data alone: MAR vs
MNAR is untestable. It is an assumption, argued from how the data were
collected. Missing-outcome problems are a form of selection bias — the
analysed sample is selected by a process related to the outcome.

Strategies implemented (deliberately the simple ones, to show their
failure modes):

* complete-case (listwise deletion);
* mean imputation — preserves the mean under MCAR only, and always
  understates variance, so naive standard errors are too small;
* deterministic regression imputation — removes MAR bias in the mean
  but still understates variance (imputed points sit on the line);
* stochastic regression imputation — adds residual noise to restore the
  spread, but single imputation still treats imputed values as known, so
  naive SEs remain too small. Proper multiple imputation with Rubin's
  rules is out of M3 scope (see docs/methodology/statistical-foundations.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, special, stats

from decision_intelligence.common.validation import (
    FloatArray,
    as_float_array,
    require_positive_int,
    require_probability,
)
from decision_intelligence.statistics.linear_regression import fit_ols

Mechanism = Literal["MCAR", "MAR", "MNAR"]
Strategy = Literal[
    "complete_case", "mean_imputation", "regression_imputation", "stochastic_regression"
]
STRATEGIES: tuple[Strategy, ...] = (
    "complete_case",
    "mean_imputation",
    "regression_imputation",
    "stochastic_regression",
)


def missingness_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-column count and rate of missing values, most-missing first."""
    counts = frame.isna().sum()
    return (
        pd.DataFrame({"n_missing": counts, "missing_rate": counts / max(len(frame), 1)})
        .sort_values("missing_rate", ascending=False)
        .rename_axis("column")
    )


def _standardise(values: FloatArray) -> FloatArray:
    sd = float(values.std())
    if sd == 0:
        raise ValueError("missingness driver is constant; cannot induce MAR/MNAR from it")
    return (values - float(values.mean())) / sd


def missingness_mask(
    values: ArrayLike,
    mechanism: str,
    rate: float,
    rng: np.random.Generator,
    *,
    covariate: ArrayLike | None = None,
    strength: float = 1.5,
) -> NDArray[np.bool_]:
    """Boolean mask (True = missing) under a chosen mechanism with expected ``rate``.

    MAR: ``logit P(R=1) = a + strength * z(covariate)``. MNAR: the same
    with ``z(values)`` itself as the driver. The intercept ``a`` is solved
    so the *expected* missing rate equals ``rate`` exactly for this data.
    Positive ``strength`` makes high values (of the driver) more likely to
    be missing.
    """
    y = as_float_array(values, name="values", min_size=2)
    rate = require_probability(rate, "rate")
    if mechanism == "MCAR":
        return np.asarray(rng.random(y.size) < rate)
    if mechanism == "MAR":
        if covariate is None:
            raise ValueError("MAR requires an observed covariate driving missingness")
        driver = _standardise(as_float_array(covariate, name="covariate", min_size=2))
        if driver.size != y.size:
            raise ValueError("covariate must align with values")
    elif mechanism == "MNAR":
        driver = _standardise(y)
    else:
        raise ValueError("mechanism must be 'MCAR', 'MAR' or 'MNAR'")

    def expected_rate_gap(intercept: float) -> float:
        return float(np.mean(special.expit(intercept + strength * driver))) - rate

    intercept = float(optimize.brentq(expected_rate_gap, -50.0, 50.0, xtol=1e-12))
    probabilities = special.expit(intercept + strength * driver)
    return np.asarray(rng.random(y.size) < probabilities)


def impute(
    values: ArrayLike,
    missing: ArrayLike,
    strategy: str,
    rng: np.random.Generator,
    *,
    covariate: ArrayLike | None = None,
) -> FloatArray:
    """Return the analysis sample for ``strategy``.

    ``complete_case`` returns only the observed values; the imputation
    strategies return a full-length array with missing entries filled.
    Regression strategies regress the outcome on ``covariate`` among the
    observed cases (the covariate must be fully observed).
    """
    y = as_float_array(values, name="values", min_size=2, allow_nan=True)
    mask = np.asarray(missing, dtype=bool)
    if mask.shape != y.shape:
        raise ValueError("missing mask must align with values")
    observed = ~mask
    if observed.sum() < 3:
        raise ValueError("fewer than three observed values; nothing can be estimated")
    if np.isnan(y[observed]).any():
        raise ValueError("values marked observed contain NaN")

    if strategy == "complete_case":
        return np.asarray(y[observed], dtype=np.float64)
    filled = y.copy()
    if strategy == "mean_imputation":
        filled[mask] = y[observed].mean()
        return filled
    if strategy not in ("regression_imputation", "stochastic_regression"):
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    if covariate is None:
        raise ValueError("regression imputation requires a fully observed covariate")
    x = as_float_array(covariate, name="covariate", min_size=2)
    if x.shape != y.shape:
        raise ValueError("covariate must align with values")
    model = fit_ols(x[observed], y[observed])
    predicted = model.predict(x[mask])
    if strategy == "stochastic_regression":
        predicted = predicted + rng.normal(0.0, model.residual_std_error, predicted.size)
    filled[mask] = predicted
    return filled


@dataclass(frozen=True)
class MissingDataStrategyResult:
    """Repeated-sampling performance of one strategy for estimating the population mean.

    ``naive_se_ratio`` = mean reported SE / empirical SD of the estimates.
    Below 1 means the strategy's naive standard error (treating the
    analysis sample as fully observed i.i.d. data) is too small, and
    ``coverage`` falls below nominal accordingly.
    """

    mechanism: Mechanism
    strategy: Strategy
    bias: float
    bias_standard_error: float
    empirical_sd: float
    mean_naive_se: float
    naive_se_ratio: float
    coverage: float


def compare_missing_data_strategies(
    *,
    mechanism: str,
    missing_rate: float,
    sample_size: int,
    n_replications: int,
    rng: np.random.Generator,
    correlation: float = 0.6,
    strength: float = 1.5,
    confidence_level: float = 0.95,
) -> list[MissingDataStrategyResult]:
    """Simulate bias, SE validity and coverage of each strategy under one mechanism.

    Data: ``(X, Y)`` bivariate standard normal with correlation
    ``correlation``; target is ``E[Y] = 0``. ``Y`` is made missing under
    ``mechanism`` (MAR driven by ``X``, MNAR by ``Y`` itself); ``X`` is
    always observed. The reported SE for every strategy is the naive
    ``sd / sqrt(n_analysed)``.
    """
    if mechanism not in ("MCAR", "MAR", "MNAR"):
        raise ValueError("mechanism must be 'MCAR', 'MAR' or 'MNAR'")
    checked_mechanism: Mechanism = (
        "MCAR" if mechanism == "MCAR" else "MAR" if mechanism == "MAR" else "MNAR"
    )
    sample_size = require_positive_int(sample_size, "sample_size", minimum=10)
    n_replications = require_positive_int(n_replications, "n_replications", minimum=10)
    confidence_level = require_probability(confidence_level, "confidence_level")
    if not -1 < correlation < 1:
        raise ValueError("correlation must be in (-1, 1)")
    z = float(stats.norm.ppf(0.5 + confidence_level / 2))

    estimates = {s: np.empty(n_replications) for s in STRATEGIES}
    naive_ses = {s: np.empty(n_replications) for s in STRATEGIES}
    for r in range(n_replications):
        x = rng.standard_normal(sample_size)
        y = correlation * x + math.sqrt(1 - correlation**2) * rng.standard_normal(sample_size)
        mask = missingness_mask(
            y, checked_mechanism, missing_rate, rng, covariate=x, strength=strength
        )
        if (~mask).sum() < 3:
            mask[:3] = False
        for strategy in STRATEGIES:
            sample = impute(y, mask, strategy, rng, covariate=x)
            estimates[strategy][r] = float(sample.mean())
            naive_ses[strategy][r] = float(sample.std(ddof=1)) / math.sqrt(sample.size)

    results: list[MissingDataStrategyResult] = []
    for strategy in STRATEGIES:
        values, ses = estimates[strategy], naive_ses[strategy]
        empirical_sd = float(values.std(ddof=1))
        covered = (values - z * ses <= 0.0) & (0.0 <= values + z * ses)
        results.append(
            MissingDataStrategyResult(
                mechanism=checked_mechanism,
                strategy=strategy,
                bias=float(values.mean()),
                bias_standard_error=empirical_sd / math.sqrt(n_replications),
                empirical_sd=empirical_sd,
                mean_naive_se=float(ses.mean()),
                naive_se_ratio=float(ses.mean()) / empirical_sd,
                coverage=float(covered.mean()),
            )
        )
    return results
