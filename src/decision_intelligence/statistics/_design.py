"""Design-matrix preparation shared by the linear and logistic regression modules."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from decision_intelligence.common.validation import FloatArray, as_float_array, as_float_matrix

INTERCEPT_NAME = "intercept"


@dataclass(frozen=True)
class Design:
    matrix: FloatArray
    feature_names: tuple[str, ...]
    has_intercept: bool


def prepare_design(
    X: ArrayLike | pd.DataFrame,
    feature_names: Sequence[str] | None,
    add_intercept: bool,
) -> Design:
    """Validate ``X``, optionally prepend an intercept column, and check full column rank.

    Rank deficiency (perfect collinearity, a constant column alongside an
    intercept, more parameters than observations) means the coefficients
    are not identified: infinitely many coefficient vectors fit equally
    well. Returning one of them silently — as a pseudo-inverse would — is
    an error this function refuses to make.
    """
    if isinstance(X, pd.DataFrame):
        names = list(X.columns.astype(str)) if feature_names is None else list(feature_names)
        matrix = as_float_matrix(X.to_numpy(dtype=np.float64), name="X")
    else:
        matrix = as_float_matrix(X, name="X")
        names = (
            [f"x{i}" for i in range(matrix.shape[1])]
            if feature_names is None
            else list(feature_names)
        )
    if len(names) != matrix.shape[1]:
        raise ValueError("feature_names must have one entry per column of X")
    if add_intercept:
        if INTERCEPT_NAME in names:
            raise ValueError(f"X already has a column named {INTERCEPT_NAME!r}")
        matrix = np.column_stack([np.ones(matrix.shape[0]), matrix])
        names = [INTERCEPT_NAME, *names]
    n, k = matrix.shape
    if n <= k:
        raise ValueError(f"need more observations ({n}) than parameters ({k})")
    rank = int(np.linalg.matrix_rank(matrix))
    if rank < k:
        raise ValueError(
            f"design matrix is rank-deficient (rank {rank} < {k} columns): coefficients are "
            "not identified — remove perfectly collinear columns (e.g. one dummy per "
            "categorical variable when an intercept is present)"
        )
    return Design(matrix=matrix, feature_names=tuple(names), has_intercept=add_intercept)


def prepare_response(y: ArrayLike, n: int) -> FloatArray:
    response = as_float_array(y, name="y", min_size=1)
    if response.size != n:
        raise ValueError(f"y has {response.size} observations but X has {n} rows")
    return response
