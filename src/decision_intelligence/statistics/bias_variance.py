"""Bias–variance decomposition of prediction error, measured by simulation.

For squared-error loss at a fixed input ``x0``, with ``y = f(x0) + e``,
``E[e] = 0``, ``Var(e) = sigma^2``, and a model ``f_hat`` fitted to a random
training set independent of the test noise:

    E[(y - f_hat(x0))^2] = sigma^2 + (E[f_hat(x0)] - f(x0))^2 + Var(f_hat(x0))
                         = noise   + bias^2                    + variance

Expectations are over training sets *and* test noise. Averaging over a
grid of ``x0`` gives the integrated decomposition reported here. The
identity is exact in expectation; the simulated components satisfy it up
to Monte Carlo error, which the result reports.

The decomposition requires knowing ``f`` and ``sigma`` — it is a
simulation-study tool for understanding model complexity, not something
computable on real data. On real data the analogous evidence is the gap
between training error (optimistic, falls monotonically with complexity)
and held-out error (U-shaped).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.polynomial import Polynomial

from decision_intelligence.common.validation import FloatArray, require_positive_int

Fitter = Callable[[FloatArray, FloatArray], Callable[[FloatArray], FloatArray]]


def polynomial_fitter(degree: int) -> Fitter:
    """Least-squares polynomial of ``degree`` (domain-scaled for numerical stability)."""
    degree = require_positive_int(degree, "degree", minimum=0)

    def fit(x: FloatArray, y: FloatArray) -> Callable[[FloatArray], FloatArray]:
        if x.size <= degree:
            raise ValueError("training set too small for this polynomial degree")
        polynomial = Polynomial.fit(x, y, degree)

        def predict(x_new: FloatArray) -> FloatArray:
            return np.asarray(polynomial(x_new), dtype=np.float64)

        return predict

    return fit


def ridge_polynomial_fitter(degree: int, penalty: float) -> Fitter:
    """Polynomial features (standardised) with an L2 penalty on non-intercept coefficients.

    Increasing ``penalty`` trades variance for bias at a fixed degree —
    the continuous counterpart of choosing a lower degree.
    """
    degree = require_positive_int(degree, "degree", minimum=1)
    if penalty < 0:
        raise ValueError("penalty must be non-negative")

    def fit(x: FloatArray, y: FloatArray) -> Callable[[FloatArray], FloatArray]:
        centre, scale = float(x.mean()), float(x.std()) or 1.0

        def features(values: FloatArray) -> FloatArray:
            z = (values - centre) / scale
            return np.column_stack([z**p for p in range(1, degree + 1)])

        train = features(x)
        means, sds = train.mean(axis=0), train.std(axis=0)
        sds[sds == 0] = 1.0
        standardised = (train - means) / sds
        y_mean = float(y.mean())
        gram = standardised.T @ standardised + penalty * np.eye(degree)
        weights = np.linalg.solve(gram, standardised.T @ (y - y_mean))

        def predict(x_new: FloatArray) -> FloatArray:
            return np.asarray(((features(x_new) - means) / sds) @ weights + y_mean)

        return predict

    return fit


@dataclass(frozen=True)
class BiasVarianceRow:
    """Integrated decomposition for one model.

    ``decomposition_gap = expected_test_mse - (noise + bias_squared + variance)``
    should be ~0 up to Monte Carlo error.
    """

    model: str
    bias_squared: float
    variance: float
    noise: float
    expected_test_mse: float
    mean_training_mse: float
    decomposition_gap: float


def bias_variance_decomposition(
    true_function: Callable[[FloatArray], FloatArray],
    models: Mapping[str, Fitter],
    *,
    noise_sd: float,
    train_size: int,
    x_test: Sequence[float] | FloatArray,
    n_replications: int,
    rng: np.random.Generator,
    x_low: float = -1.0,
    x_high: float = 1.0,
) -> list[BiasVarianceRow]:
    """Simulate the integrated bias–variance decomposition for several models.

    Each replication draws a fresh training set (``x ~ U(x_low, x_high)``,
    ``y = f(x) + N(0, noise_sd^2)``); *the same* training sets are used for
    every model, so differences between models are not confounded with
    training-set luck. Expected test MSE is estimated directly from fresh
    noisy test responses at ``x_test`` — independently of the components —
    so the decomposition identity is a genuine check, not a tautology.
    """
    if not noise_sd >= 0:
        raise ValueError("noise_sd must be non-negative")
    train_size = require_positive_int(train_size, "train_size", minimum=2)
    n_replications = require_positive_int(n_replications, "n_replications", minimum=2)
    grid = np.asarray(x_test, dtype=np.float64)
    truth = np.asarray(true_function(grid), dtype=np.float64)

    predictions = {name: np.empty((n_replications, grid.size)) for name in models}
    test_squared_errors = {name: np.empty(n_replications) for name in models}
    training_mse = {name: np.empty(n_replications) for name in models}
    for r in range(n_replications):
        x_train = rng.uniform(x_low, x_high, train_size)
        y_train = true_function(x_train) + rng.normal(0.0, noise_sd, train_size)
        y_test = truth + rng.normal(0.0, noise_sd, grid.size)
        for name, fitter in models.items():
            predict = fitter(x_train, y_train)
            prediction = predict(grid)
            predictions[name][r] = prediction
            test_squared_errors[name][r] = float(np.mean((y_test - prediction) ** 2))
            training_mse[name][r] = float(np.mean((y_train - predict(x_train)) ** 2))

    rows: list[BiasVarianceRow] = []
    for name in models:
        mean_prediction = predictions[name].mean(axis=0)
        bias_squared = float(np.mean((mean_prediction - truth) ** 2))
        variance = float(np.mean(predictions[name].var(axis=0, ddof=1)))
        noise = noise_sd**2
        expected_mse = float(test_squared_errors[name].mean())
        rows.append(
            BiasVarianceRow(
                model=name,
                bias_squared=bias_squared,
                variance=variance,
                noise=noise,
                expected_test_mse=expected_mse,
                mean_training_mse=float(training_mse[name].mean()),
                decomposition_gap=expected_mse - (noise + bias_squared + variance),
            )
        )
    return rows
