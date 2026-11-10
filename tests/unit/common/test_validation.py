"""Shared input validation: every invalid input class is rejected, valid input passes through."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from decision_intelligence.common.validation import (
    as_float_array,
    as_float_matrix,
    require_alternative,
    require_positive_int,
    require_probability,
)


def test_as_float_array_accepts_lists_series_and_bools() -> None:
    assert as_float_array([1, 2, 3]).dtype == np.float64
    assert as_float_array(pd.Series([1.5, 2.5])).tolist() == [1.5, 2.5]
    assert as_float_array([True, False]).tolist() == [1.0, 0.0]


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([[1, 2], [3, 4]], "one-dimensional"),
        ([], "at least 1"),
        ([1.0, math.inf], "infinite"),
        ([1.0, math.nan], "NaN"),
        (["a", "b"], "numeric"),
    ],
)
def test_as_float_array_rejects_invalid_input(values: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        as_float_array(values)  # type: ignore[arg-type]


def test_as_float_array_allows_nan_only_when_asked() -> None:
    assert np.isnan(as_float_array([1.0, math.nan], allow_nan=True)).sum() == 1
    with pytest.raises(ValueError, match="infinite"):
        as_float_array([math.inf], allow_nan=True)


def test_as_float_array_enforces_min_size() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        as_float_array([1.0, 2.0], min_size=3)


def test_as_float_matrix_promotes_vectors_and_rejects_non_finite() -> None:
    assert as_float_matrix([1, 2, 3]).shape == (3, 1)
    with pytest.raises(ValueError, match="NaN or infinite"):
        as_float_matrix([[1.0, math.nan]])
    with pytest.raises(ValueError, match="two-dimensional"):
        as_float_matrix(np.zeros((2, 2, 2)))


@pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.1, math.nan])
def test_require_probability_open_interval(value: float) -> None:
    with pytest.raises(ValueError):
        require_probability(value, "p")


def test_require_probability_inclusive_accepts_bounds() -> None:
    assert require_probability(0.0, "p", inclusive=True) == 0.0
    assert require_probability(1.0, "p", inclusive=True) == 1.0


@pytest.mark.parametrize("value", [0, -3, 2.5, True])
def test_require_positive_int_rejects(value: object) -> None:
    with pytest.raises(ValueError):
        require_positive_int(value, "n")  # type: ignore[arg-type]


def test_require_positive_int_accepts_numpy_integers() -> None:
    assert require_positive_int(np.int64(5), "n") == 5


def test_require_alternative() -> None:
    assert require_alternative("less") == "less"
    with pytest.raises(ValueError, match="alternative"):
        require_alternative("two_sided")
