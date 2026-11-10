"""Sensitivity of adjusted estimates to unmeasured confounding.

No data analysis can rule out an unmeasured confounder. Sensitivity analysis
asks a quantitative question instead: *how strong* would such a confounder
have to be to change the conclusion? Two complementary tools:

* **E-value** (VanderWeele & Ding 2017): the minimum strength of association,
  on the risk-ratio scale, that an unmeasured confounder would need with
  *both* treatment and outcome (conditional on the measured covariates) to
  fully explain away an observed risk ratio ``RR``:
  ``E = RR + sqrt(RR (RR - 1))`` for ``RR >= 1`` (use ``1/RR`` below 1). The
  E-value for the confidence limit closest to 1 says how strong a
  confounder would need to be to make the interval include 1. It makes no
  assumption about the confounder's prevalence and is conservative.

* **Bias formula for a binary confounder on the risk-difference scale**
  (simple additive form, e.g. VanderWeele & Arah 2011): if an unmeasured
  binary ``U`` raises the outcome probability by ``gamma`` (same in both
  arms, no ``U × T`` interaction) and its prevalence differs between
  treated and untreated by ``delta``, the adjusted estimate is biased by
  ``gamma × delta``. :func:`risk_difference_bias_grid` tabulates corrected
  estimates and intervals over a grid of ``(gamma, delta)``.

Neither tool says how strong real unmeasured confounding *is*; that is a
substantive judgment, best anchored by benchmarking against the strength of
measured covariates.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class EValue:
    """E-values for a risk-ratio point estimate and its confidence limit nearest 1."""

    risk_ratio: float
    e_value_point: float
    e_value_ci: float


def e_value_from_rr(rr: float) -> float:
    """``RR + sqrt(RR (RR - 1))`` after mapping protective ratios to ``1/RR``."""
    if not rr > 0 or not math.isfinite(rr):
        raise ValueError("risk ratio must be positive and finite")
    ratio = rr if rr >= 1 else 1 / rr
    return ratio + math.sqrt(ratio * (ratio - 1))


def bounding_factor(rr_exposure_confounder: float, rr_confounder_outcome: float) -> float:
    """Ding & VanderWeele (2016) joint bounding factor ``RR_EU RR_UD / (RR_EU + RR_UD - 1)``.

    The largest factor by which a confounder with these two associations
    could inflate an observed risk ratio. The E-value ``E`` is exactly the
    value with ``bounding_factor(E, E) = RR``; a confounder with *unequal*
    associations can explain away ``RR`` even if one of them is below ``E``
    (only ``bounding_factor >= RR`` matters).
    """
    for value, name in (
        (rr_exposure_confounder, "rr_exposure_confounder"),
        (rr_confounder_outcome, "rr_confounder_outcome"),
    ):
        if not value >= 1 or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite risk ratio >= 1")
    return (
        rr_exposure_confounder
        * rr_confounder_outcome
        / (rr_exposure_confounder + rr_confounder_outcome - 1)
    )


def e_value(rr: float, ci_lower: float, ci_upper: float) -> EValue:
    """E-value for the estimate and for the confidence limit closest to the null."""
    if not 0 < ci_lower <= rr <= ci_upper:
        raise ValueError("need 0 < ci_lower <= rr <= ci_upper")
    if ci_lower <= 1 <= ci_upper:
        ci_value = 1.0
    else:
        ci_value = e_value_from_rr(ci_lower if rr > 1 else ci_upper)
    return EValue(risk_ratio=rr, e_value_point=e_value_from_rr(rr), e_value_ci=ci_value)


def risk_difference_bias_grid(
    estimate: float,
    ci_lower: float,
    ci_upper: float,
    *,
    outcome_effects: Sequence[float],
    prevalence_differences: Sequence[float],
) -> pd.DataFrame:
    """Bias-corrected risk differences under a hypothetical unmeasured binary confounder.

    ``outcome_effects`` (gamma): risk difference in the outcome per unit of
    ``U``; ``prevalence_differences`` (delta): ``P(U=1 | T=1) - P(U=1 | T=0)``.
    Positive ``gamma × delta`` means the observed estimate is *inflated* by
    the confounder. The interval is shifted, not widened: uncertainty about
    the confounder itself is not propagated.
    """
    if not ci_lower <= estimate <= ci_upper:
        raise ValueError("need ci_lower <= estimate <= ci_upper")
    rows = []
    for gamma in outcome_effects:
        if not -1 <= gamma <= 1:
            raise ValueError("outcome effects must lie in [-1, 1] on the risk scale")
        for delta in prevalence_differences:
            if not -1 <= delta <= 1:
                raise ValueError("prevalence differences must lie in [-1, 1]")
            bias = gamma * delta
            rows.append(
                {
                    "outcome_effect_gamma": gamma,
                    "prevalence_difference_delta": delta,
                    "bias": bias,
                    "corrected_estimate": estimate - bias,
                    "corrected_ci_lower": ci_lower - bias,
                    "corrected_ci_upper": ci_upper - bias,
                    "ci_excludes_zero": (ci_lower - bias) > 0 or (ci_upper - bias) < 0,
                }
            )
    return pd.DataFrame(rows)
