"""Split conformal prediction intervals (added in M11 P6).

For a model fitted on one part of the data and absolute residual scores ``s_i = |y_i - m(x_i)|``
on a separate calibration part of size ``n``, the interval ``m(x) +/- q`` with ``q`` the
``ceil((n + 1) level)``-th smallest score covers a new ``y`` with probability at least
``level`` whenever the calibration points and the new point are exchangeable (Vovk, Gammerman
& Shafer 2005; Lei et al. 2018). With continuous scores the coverage is exactly
``ceil((n + 1) level) / (n + 1)``, so it is at most ``level + 1 / (n + 1)``. If
``ceil((n + 1) level) > n`` the interval is unbounded.

The guarantee is marginal and requires exchangeability; it does not hold across time when the
data-generating process drifts, nor conditionally on covariates.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from decision_intelligence.common.validation import FloatArray


def conformal_rank(n: int, level: float) -> int:
    """``ceil((n + 1) level)``: the rank of the calibration score used as the quantile."""
    if n < 1 or not 0 < level < 1:
        raise ValueError("need n >= 1 and 0 < level < 1")
    return int(math.ceil((n + 1) * level - 1e-12))


def conformal_quantile(scores: ArrayLike, level: float) -> float:
    """The split-conformal quantile of non-negative ``scores`` (``inf`` if the rank exceeds n)."""
    s = np.sort(np.asarray(scores, dtype=np.float64))
    if s.ndim != 1 or s.size == 0 or not np.all(np.isfinite(s)) or np.any(s < 0):
        raise ValueError("scores must be a non-empty 1-D array of finite non-negative values")
    k = conformal_rank(s.size, level)
    return math.inf if k > s.size else float(s[k - 1])


def conformal_interval(prediction: ArrayLike, residual_scores: ArrayLike, level: float
                       ) -> tuple[FloatArray, FloatArray]:
    """``prediction -/+ q`` with ``q`` from the calibration ``residual_scores``."""
    q = conformal_quantile(residual_scores, level)
    m = np.asarray(prediction, dtype=np.float64)
    return np.asarray(m - q, dtype=np.float64), np.asarray(m + q, dtype=np.float64)
