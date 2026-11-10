"""Input validation shared by every statistical routine.

Statistical code fails in two ways: loudly (an exception) or silently (a
plausible-looking number computed from invalid input — a NaN quietly
propagated, a probability of 1.3, a sample of one treated as if it had a
variance). Centralising validation here means every estimator in every
pillar rejects the same invalid inputs in the same way, with the same
messages, instead of each module guessing differently.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
Alternative = Literal["two-sided", "greater", "less"]

_ALTERNATIVES: tuple[str, ...] = ("two-sided", "greater", "less")


def as_float_array(
    values: ArrayLike,
    *,
    name: str = "values",
    min_size: int = 1,
    allow_nan: bool = False,
) -> FloatArray:
    """Return ``values`` as a 1-D float64 array, rejecting invalid input.

    Args:
        values: Anything ``numpy.asarray`` accepts (list, Series, array).
        name: Used in error messages so the caller knows which input failed.
        min_size: Minimum number of elements required (after NaN removal is
            *not* applied — NaNs count towards size, and are rejected unless
            ``allow_nan``).
        allow_nan: If ``False`` (the default), any NaN raises. Infinite
            values are always rejected: no estimator in this package has a
            meaningful result for them.

    Raises:
        ValueError: If the input is not 1-D, is too small, contains
            infinities, or contains NaN when ``allow_nan`` is ``False``.
    """
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric: {error}") from error
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if array.size < min_size:
        raise ValueError(f"{name} must contain at least {min_size} value(s), got {array.size}")
    if np.isinf(array).any():
        raise ValueError(f"{name} contains infinite values")
    if not allow_nan and np.isnan(array).any():
        raise ValueError(
            f"{name} contains NaN; remove or impute missing values explicitly before analysis"
        )
    return array


def as_float_matrix(values: ArrayLike, *, name: str = "X") -> FloatArray:
    """Return ``values`` as a finite 2-D float64 array (1-D input becomes one column)."""
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric: {error}") from error
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def require_probability(value: float, name: str, *, inclusive: bool = False) -> float:
    """Validate a probability-like scalar.

    Args:
        inclusive: If ``False`` (the default) the open interval (0, 1) is
            required — appropriate for significance levels, confidence
            levels and target power, where 0 and 1 are degenerate.
    """
    value = float(value)
    if math.isnan(value):
        raise ValueError(f"{name} must not be NaN")
    if inclusive:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
    elif not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be in (0, 1), got {value}")
    return value


def require_positive_int(value: int, name: str, *, minimum: int = 1) -> int:
    """Validate an integer count (sample size, number of replications, ...)."""
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def require_alternative(alternative: str) -> Alternative:
    """Validate a hypothesis-test alternative and narrow its type."""
    if alternative == "two-sided":
        return "two-sided"
    if alternative == "greater":
        return "greater"
    if alternative == "less":
        return "less"
    raise ValueError(f"alternative must be one of {_ALTERNATIVES}, got {alternative!r}")
