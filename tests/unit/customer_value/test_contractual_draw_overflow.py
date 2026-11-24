"""Development regressions for the unbounded Hessian-draw prediction path."""

import math
from dataclasses import replace
from decimal import Decimal, localcontext

import numpy as np
import pytest

from decision_intelligence.customer_value import contractual as ct
from decision_intelligence.customer_value import predictive_cases as pc
from decision_intelligence.customer_value import predictive_evaluation as ev


def _legacy_paid(model, theta, ages, horizon):
    steps = np.arange(1, horizon + 1, dtype=np.float64)
    base = ct.log_survival(model, theta, ages)[0]
    ahead = ct.log_survival(model, theta, (ages[:, None] + steps).ravel())[0]
    return np.exp(ahead.reshape(-1, horizon) - base[:, None])


@pytest.mark.parametrize("model", ct.MODEL_IDS)
def test_fitted_range_probabilities_are_bitwise_unchanged(model):
    rng = np.random.default_rng(6412)
    ages = np.arange(0.0, 52.0, 3.0)
    for theta in rng.uniform(-ct.LOG_BOX, ct.LOG_BOX, (100, 2)):
        np.testing.assert_array_equal(
            ct.conditional_paid(model, theta, ages, 12), _legacy_paid(model, theta, ages, 12))


@pytest.mark.parametrize(("model", "theta"), [
    ("E1", (1000.0, 1000.5)), ("E1", (-1000.0, -1000.5)), ("E1", (18.0, 19.5)),
    ("E2", (-2.0, -900.0)), ("E2", (900.0, 0.0)), ("E2", (800.0, -803.0)),
    ("E2", (-40.0, 36.0)),
])
def test_outside_box_matches_high_precision_conditional_ratio(model, theta):
    with localcontext() as ctx:
        ctx.prec = 520
        p0, p1 = (Decimal(x).exp() for x in theta)
        expected = []
        for age in (0, 2, 40):
            row = []
            for step in (1, 2, 3):
                if model == "E1":
                    factor = math.prod((p1 + age + k) / (p0 + p1 + age + k)
                                       for k in range(step))
                else:
                    x = p0 * p1 * step / (1 + p0 * p1 * age)
                    hazard = ((1 + x).ln()) / p1
                    factor = (-hazard).exp() if hazard < 800 else Decimal(0)
                row.append(float(factor))
            expected.append(row)
    np.testing.assert_allclose(
        ct.conditional_paid(model, np.asarray(theta), np.array([0.0, 2.0, 40.0]), 3),
        expected, rtol=2e-13, atol=1e-15)


@pytest.mark.parametrize("model", ct.MODEL_IDS)
def test_extreme_finite_draws_give_valid_nondecreasing_survival_loss(model):
    largest = np.finfo(np.float64).max
    values = [-largest, -1000.0, -710.0, 0.0, 710.0, 1000.0, largest]
    with np.errstate(all="raise"):
        for a in values:
            for b in values:
                paid = ct.conditional_paid(model, np.array([a, b]), np.array([0., 1., 40.]), 12)
                assert np.all(np.isfinite(paid))
                assert np.all((paid >= 0) & (paid <= 1))
                assert np.all(np.diff(paid, axis=1) <= 0)
    with pytest.raises(OverflowError):
        _legacy_paid(model, np.array([1000.0, 0.0]), np.array([0.0, 2.0]), 12)


def test_bounded_r2_reproduces_overflow_and_preserves_draw_streams(small, monkeypatch):
    original_predict = ct.predict
    fixture = {m: original_predict(m, small, 30, pc.W, pc.POLICY) for m in ct.MODEL_IDS}
    states = {}

    def run(*, wide, legacy):
        def prediction(model, *_args):
            pred = fixture[model]
            factor = 2000.0 if wide else 0.001
            used = {
                key: (status, replace(
                    fit, log_params=np.log([0.7, 3.8] if model == "E1" else [0.08, 0.5]),
                    cov_log=np.eye(2) * factor ** 2
                ) if fit.finite and fit.boundary is None and fit.geometric_churn is None else fit)
                for key, (status, fit) in pred.fits.used.items()
            }
            return replace(pred, fits=replace(pred.fits, used=used))

        rng, noise = np.random.default_rng(6413), np.random.default_rng(6414)
        with monkeypatch.context() as scoped:
            scoped.setattr(ct, "predict", prediction)
            if legacy:
                scoped.setattr(ev, "conditional_paid", _legacy_paid)
            result = pc._r2_contractual_coverage(small, "development", 30, ("A",), 4, rng, noise)
        states[wide, legacy] = (rng.bit_generator.state, noise.bit_generator.state)
        return result

    with pytest.raises(OverflowError):
        run(wide=True, legacy=True)
    baseline = run(wide=False, legacy=True)
    assert run(wide=False, legacy=False) == baseline
    corrected = run(wide=True, legacy=False)
    assert {r["model"] for r in corrected if r["n"] > 0} >= {"E1", "E2"}
    assert run(wide=True, legacy=False) == corrected
    assert states[False, True] == states[False, False] == states[True, False]
