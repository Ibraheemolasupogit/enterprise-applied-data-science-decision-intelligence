"""Multiple-testing corrections (carried from M3; first needed by M6).

Testing ``m`` hypotheses each at level ``alpha`` makes at least one false
rejection likely when several nulls are true (``1 - (1 - alpha)^m`` for
independent tests). Two error rates can be controlled:

* **Family-wise error rate** (FWER): P(any false rejection) <= alpha.
  ``bonferroni`` (any dependence) and ``holm`` (step-down, uniformly more
  powerful than Bonferroni, any dependence; Holm 1979).
* **False discovery rate** (FDR): E[false rejections / rejections] <= alpha.
  ``benjamini-hochberg`` (independent or positively dependent tests;
  Benjamini & Hochberg 1995) and ``benjamini-yekutieli`` (any dependence;
  Benjamini & Yekutieli 2001).

Use FWER control for a small family that drives one decision (e.g. a
primary metric and its guardrails); FDR for screening many exploratory
metrics. Corrections are pre-specified with the family — choosing the
family after seeing the p-values defeats them.

**Boundary convention.** A hypothesis is rejected iff ``p <= alpha``
(:func:`rejects`). This is the definition of the p-value as the smallest
level at which the test rejects, the convention of the M3 simulation
helpers (``statistics.simulation.rejection_rate``) and of statsmodels
``multipletests``; every M6 test-based decision (significance, SRM, balance,
exclusions, multiplicity) uses it. For continuous test statistics
``p == alpha`` has probability zero; for discrete ones it is rare, and the
inclusive rule keeps a test's size at most ``alpha`` exactly when its
p-value is valid (``P(p <= u) <= u``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from decision_intelligence.common.validation import FloatArray, as_float_array, require_probability

CorrectionMethod = Literal["bonferroni", "holm", "benjamini-hochberg", "benjamini-yekutieli"]
_METHODS: tuple[str, ...] = ("bonferroni", "holm", "benjamini-hochberg", "benjamini-yekutieli")


@dataclass(frozen=True)
class MultipleTestingResult:
    method: CorrectionMethod
    controls: Literal["FWER", "FDR"]
    alpha: float
    p_values: FloatArray
    adjusted_p_values: FloatArray
    rejected: np.ndarray


def rejects(p_value: float, alpha: float) -> bool:
    """The package's rejection rule: ``p_value <= alpha`` (see the module docstring)."""
    return bool(p_value <= alpha)


def adjust_p_values(
    p_values: ArrayLike, *, method: str = "holm", alpha: float = 0.05
) -> MultipleTestingResult:
    """Adjusted p-values (reject ``H_i`` iff ``adjusted_i <= alpha``), in the input order."""
    p = as_float_array(p_values, name="p_values", min_size=1)
    if ((p < 0) | (p > 1)).any():
        raise ValueError("p-values must lie in [0, 1]")
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}")
    alpha = require_probability(alpha, "alpha")
    m = p.size
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    if method == "bonferroni":
        adjusted_sorted = np.minimum(sorted_p * m, 1.0)
    elif method == "holm":
        adjusted_sorted = np.minimum(np.maximum.accumulate(sorted_p * (m - np.arange(m))), 1.0)
    else:
        by = method == "benjamini-yekutieli"
        harmonic = float(np.sum(1.0 / np.arange(1, m + 1))) if by else 1.0
        raw = sorted_p * m * harmonic / np.arange(1, m + 1)
        adjusted_sorted = np.minimum(np.minimum.accumulate(raw[::-1])[::-1], 1.0)
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    return MultipleTestingResult(
        method=method,  # type: ignore[arg-type]
        controls="FWER" if method in ("bonferroni", "holm") else "FDR",
        alpha=alpha,
        p_values=p,
        adjusted_p_values=adjusted,
        rejected=adjusted <= alpha,
    )


__all__ = ["CorrectionMethod", "MultipleTestingResult", "adjust_p_values", "rejects"]
