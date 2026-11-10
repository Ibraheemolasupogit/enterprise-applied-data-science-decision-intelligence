"""Overlap, balance and weight diagnostics for backdoor adjustment.

These diagnostics check what *can* be checked from data. They cannot
establish exchangeability (unmeasured confounding is invisible to them);
they can reveal positivity problems, residual imbalance in the *measured*
covariates, and unstable weights.

* **Overlap**: the distribution of ``e(X)`` in each arm. Mass near 0 or 1
  means some covariate regions are (almost) never treated or never
  untreated, so effects there are extrapolated by the outcome model or
  carried by a handful of huge weights.
* **Balance**: the standardised mean difference
  ``SMD = (mean_1 - mean_0) / sqrt((var_1 + var_0) / 2)`` before and after
  weighting (denominator from the *unweighted* variances, so raw and
  weighted SMDs are comparable; Austin 2009). |SMD| < 0.1 is the usual
  heuristic for adequate balance; the variance ratio checks second moments.
* **Weights**: Kish effective sample size ``(Σw)² / Σw²`` per arm, the
  largest weight and the share of total weight carried by the top 1% of
  units. A small ESS means the estimate rests on few units even when ``n``
  is large.

Trimming units with extreme propensities (:func:`trim_to_overlap`) reduces
variance but **changes the estimand** to the effect in the trimmed
population; it must be reported as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from decision_intelligence.causal_inference.estimators import CausalData, Estimand
from decision_intelligence.common.validation import FloatArray, as_float_array

SMD_THRESHOLD = 0.1


@dataclass(frozen=True)
class OverlapSummary:
    """Propensity-score distribution by arm."""

    treated_quantiles: dict[str, float]
    control_quantiles: dict[str, float]
    share_below: float
    share_above: float
    bound: float
    common_support: tuple[float, float]
    share_outside_common_support: float

    @property
    def adequate(self) -> bool:
        return self.share_below == 0.0 and self.share_above == 0.0


def overlap_summary(
    propensity: ArrayLike, treatment: ArrayLike, *, bound: float = 0.05
) -> OverlapSummary:
    """Summarise propensity overlap; flags units with ``e < bound`` or ``e > 1 - bound``."""
    e = as_float_array(propensity, name="propensity")
    t = as_float_array(treatment, name="treatment")
    if e.shape != t.shape:
        raise ValueError("propensity and treatment must align")
    if not 0 < bound < 0.5:
        raise ValueError("bound must be in (0, 0.5)")
    levels = (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)
    names = ("min", "p01", "p05", "median", "p95", "p99", "max")

    def quantiles(values: FloatArray) -> dict[str, float]:
        return {n: float(q) for n, q in zip(names, np.quantile(values, levels), strict=True)}

    treated, control = e[t == 1], e[t == 0]
    support = (
        float(max(treated.min(), control.min())),
        float(min(treated.max(), control.max())),
    )
    outside = (e < support[0]) | (e > support[1])
    return OverlapSummary(
        treated_quantiles=quantiles(treated),
        control_quantiles=quantiles(control),
        share_below=float(np.mean(e < bound)),
        share_above=float(np.mean(e > 1 - bound)),
        bound=bound,
        common_support=support,
        share_outside_common_support=float(np.mean(outside)),
    )


def balancing_weights(
    propensity: ArrayLike, treatment: ArrayLike, estimand: Estimand = "ATE"
) -> FloatArray:
    """IPW weights: ATE ``T/e + (1-T)/(1-e)``; ATT ``T + (1-T) e/(1-e)``."""
    e = as_float_array(propensity, name="propensity")
    t = as_float_array(treatment, name="treatment")
    if ((e <= 0) | (e >= 1)).any():
        raise ValueError("propensities must lie strictly inside (0, 1)")
    if estimand == "ATE":
        return np.asarray(t / e + (1 - t) / (1 - e), dtype=np.float64)
    if estimand == "ATT":
        return np.asarray(t + (1 - t) * e / (1 - e), dtype=np.float64)
    raise ValueError("estimand must be 'ATE' or 'ATT'")


def balance_table(
    data: CausalData, weights: ArrayLike | None = None, *, covariates: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Raw and (optionally) weighted SMD and variance ratio for each covariate.

    ``covariates`` defaults to the adjustment set; pass other pre-treatment
    variables to check balance on variables *not* adjusted for.
    """
    frame = data.covariates if covariates is None else covariates
    t = data.treatment == 1
    w = None if weights is None else as_float_array(weights, name="weights")
    rows = []
    for column in frame.columns:
        x = frame[column].to_numpy(dtype=np.float64)
        x1, x0 = x[t], x[~t]
        pooled_sd = float(np.sqrt((x1.var(ddof=1) + x0.var(ddof=1)) / 2))
        row: dict[str, float | str] = {
            "covariate": str(column),
            "mean_treated": float(x1.mean()),
            "mean_control": float(x0.mean()),
            "smd_raw": _smd(float(x1.mean()), float(x0.mean()), pooled_sd),
            "variance_ratio_raw": _ratio(float(x1.var(ddof=1)), float(x0.var(ddof=1))),
        }
        if w is not None:
            w1, w0 = w[t], w[~t]
            m1, m0 = _weighted_mean(x1, w1), _weighted_mean(x0, w0)
            row["smd_weighted"] = _smd(m1, m0, pooled_sd)
            row["variance_ratio_weighted"] = _ratio(
                _weighted_variance(x1, w1), _weighted_variance(x0, w0)
            )
        rows.append(row)
    return pd.DataFrame(rows).set_index("covariate")


@dataclass(frozen=True)
class WeightDiagnostics:
    """Stability of inverse-probability weights."""

    max_weight: float
    max_weight_treated: float
    max_weight_control: float
    ess_treated: float
    ess_control: float
    n_treated: int
    n_control: int
    top_one_percent_weight_share: float
    n_weights_above_threshold: int
    threshold: float
    warnings: tuple[str, ...] = field(default=())


def weight_diagnostics(
    weights: ArrayLike, treatment: ArrayLike, *, threshold: float = 20.0
) -> WeightDiagnostics:
    """Kish ESS per arm, extreme weights and concentration of weight mass."""
    w = as_float_array(weights, name="weights")
    t = as_float_array(treatment, name="treatment") == 1
    if (w < 0).any():
        raise ValueError("weights must be non-negative")
    ess_t, ess_c = _kish_ess(w[t]), _kish_ess(w[~t])
    ordered = np.sort(w)[::-1]
    top = max(1, int(np.ceil(0.01 * w.size)))
    top_share = float(ordered[:top].sum() / w.sum())
    n_above = int(np.sum(w > threshold))
    warnings: list[str] = []
    for arm, ess, size in (("treated", ess_t, int(t.sum())), ("control", ess_c, int((~t).sum()))):
        if ess < 0.5 * size:
            warnings.append(
                f"{arm} effective sample size {ess:.0f} is under half of its {size} units: "
                "weights are unstable"
            )
    if n_above:
        warnings.append(f"{n_above} weight(s) exceed {threshold:g}")
    if top_share > 0.1:
        warnings.append(f"top 1% of units carry {top_share:.0%} of total weight")
    return WeightDiagnostics(
        max_weight=float(np.max(w)),
        max_weight_treated=float(w[t].max()),
        max_weight_control=float(w[~t].max()),
        ess_treated=ess_t,
        ess_control=ess_c,
        n_treated=int(t.sum()),
        n_control=int((~t).sum()),
        top_one_percent_weight_share=top_share,
        n_weights_above_threshold=n_above,
        threshold=threshold,
        warnings=tuple(warnings),
    )


def trim_to_overlap(propensity: ArrayLike, *, bound: float = 0.1) -> NDArray[np.bool_]:
    """Mask of units with ``bound <= e <= 1 - bound`` (Crump et al. 2009 style trimming).

    Estimating on the kept units identifies the effect **in the trimmed
    population**, not the original ATE — report it under that name.
    """
    e = as_float_array(propensity, name="propensity")
    if not 0 < bound < 0.5:
        raise ValueError("bound must be in (0, 0.5)")
    return np.asarray((e >= bound) & (e <= 1 - bound))


def _kish_ess(w: FloatArray) -> float:
    total = float(w.sum())
    squares = float(np.sum(w**2))
    return total**2 / squares if squares > 0 else 0.0


def _smd(m1: float, m0: float, pooled_sd: float) -> float:
    if pooled_sd == 0:
        return 0.0 if m1 == m0 else float("inf")
    return (m1 - m0) / pooled_sd


def _ratio(v1: float, v0: float) -> float:
    return v1 / v0 if v0 > 0 else float("inf")


def _weighted_mean(x: FloatArray, w: FloatArray) -> float:
    return float(np.sum(w * x) / np.sum(w))


def _weighted_variance(x: FloatArray, w: FloatArray) -> float:
    mean = _weighted_mean(x, w)
    return float(np.sum(w * (x - mean) ** 2) / np.sum(w))
