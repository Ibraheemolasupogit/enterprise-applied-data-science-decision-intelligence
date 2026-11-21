"""Unapproved A15 development: deterministic synthetic inputs only, no scenario generation."""

from __future__ import annotations

from decimal import Decimal, localcontext

import numpy as np
import pandas as pd
import pytest
from scipy import linalg, special

from decision_intelligence.customer_value import btyd
from decision_intelligence.customer_value import predictive_cases as pc
from decision_intelligence.customer_value import predictive_evaluation as ev
from decision_intelligence.customer_value.model_selection import FitDiagnostics


def _reference(a: float, b: float, k: float, rate: float, left: float, right: float) -> float:
    """90-digit integrated binomial series using exact Beta moments, not Jacobi quadrature.

    Valid here because right/rate < 1. Term m integrates p**(m-1); no singular Beta-density
    integration or floating-point endpoint subtraction. 600 terms is ample for these fixtures.
    """
    with localcontext() as ctx:
        ctx.prec = 90
        aa, bb, kk, rr, ll, hh = map(Decimal.from_float, (a, b, k, rate, left, right))
        lo, hi = ll / rr, hh / rr
        pl, ph, coefficient, moment = lo, hi, Decimal(1), Decimal(1)
        total = Decimal(0)
        for m in range(1, 601):
            coefficient *= (kk + m - 1) / m
            term = coefficient * (ph - pl) * moment
            total += term if m % 2 else -term
            if m > 100 and abs(term) < Decimal("1e-75"):
                return float(total)
            moment *= (aa + (m - 1)) / (aa + bb + (m - 1))
            pl, ph = pl * lo, ph * hi
        raise AssertionError("reference series did not converge")


def _moment(a: float, b: float, power: int) -> float:
    with localcontext() as ctx:
        ctx.prec = 90
        aa, bb = Decimal.from_float(a), Decimal.from_float(b)
        value = Decimal(1)
        for j in range(power):
            value *= (aa + j) / (aa + bb + j)
        return float(value)


SHAPES = [(2.0 ** -60, 2.0 ** -61), (1e-300, 2e-300),
          *[(a, 3.0) for a in (1e-6, 1e-12, 1e-20, 1e-100, 1e-300)],
          (0.8, 1e-20), (0.5, 0.5), (0.25, 0.75), (0.4, 0.6),
          (0.8, 3.5), (0.8, 503.5), (2.4, 60.0), (5.0, 1.0)]


@pytest.mark.parametrize("a,b", SHAPES)
def test_scalar_batch_boundary_weights_and_beta_moments(a: float, b: float) -> None:
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        scalar = btyd.beta_nodes(a, b)
        pn, w = pc.beta_nodes_batch(np.array([a]), np.array([b]))
    for nodes, weights in (scalar, (pn[0], w[0])):
        assert len(nodes) == btyd.E4_JACOBI_NODES == 64
        assert np.all(np.isfinite(nodes)) and np.all((nodes >= 0) & (nodes <= 1))
        assert np.all(np.isfinite(weights)) and np.all(weights >= 0)
        assert weights.sum() == pytest.approx(1.0, abs=2e-15)
        for power in range(1, 6):
            assert float(weights @ nodes ** power) == pytest.approx(
                _moment(a, b, power), rel=1e-11, abs=0.0)
    # Eigenvectors in nearly degenerate interior eigenspaces need not coincide: compare measures.
    for power in range(6):
        assert float(scalar[1] @ scalar[0] ** power) == pytest.approx(
            float(w[0] @ pn[0] ** power), rel=1e-11, abs=0.0)


def test_recurrence_singularity_neighbors_and_one_node_rule() -> None:
    a = np.array([0.25, 0.25, 0.25])
    b = np.array([np.nextafter(0.75, 0), 0.75, np.nextafter(0.75, 1)])
    diag, off = btyd.beta_jacobi(a, b, 64)
    assert np.all(np.isfinite(diag)) and np.all(np.isfinite(off))
    np.testing.assert_allclose(off[:, 0] ** 2, a * b / ((a + b) ** 2 * (a + b + 1)),
                               rtol=1e-14)
    pn, w = btyd.beta_nodes(2.0 ** -60, 2.0 ** -61, n=1)
    np.testing.assert_allclose(pn, [2 / 3], rtol=2e-16, atol=0)
    np.testing.assert_array_equal(w, [1.0])


def test_ordinary_scalar_batch_nodes_and_weights_agree() -> None:
    a, b = np.array([0.4, 1.3, 5.0]), np.array([2.0, 30.0, 1.0])
    pn, w = pc.beta_nodes_batch(a, b)
    for i in range(len(a)):
        sn, sw = btyd.beta_nodes(float(a[i]), float(b[i]))
        np.testing.assert_allclose(pn[i], sn, atol=1e-13, rtol=0)
        np.testing.assert_allclose(w[i], sw, atol=1e-13, rtol=0)


def test_continuous_zero_node_and_positive_subnormal_node() -> None:
    p = np.array([0.0, np.nextafter(0.0, 1.0), 1e-300, 1e-20, 0.2, 1.0])
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        got = btyd.purchase_window(p, np.asarray(0.8), np.asarray(112.0),
                                   np.asarray(8.0), np.asarray(60.0))
    np.testing.assert_allclose(got[:4], 0.8 * 52 / 112, rtol=1e-15)
    for i in (4, 5):
        with localcontext() as ctx:
            ctx.prec = 90
            pp = Decimal.from_float(float(p[i]))
            kk, rr = Decimal.from_float(0.8), Decimal(112)
            ref = (((1 + pp * 8 / rr).ln() * -kk).exp()
                   - ((1 + pp * 60 / rr).ln() * -kk).exp()) / pp
        assert got[i] == pytest.approx(float(ref), rel=2e-15)


@pytest.mark.parametrize("a,b", [(2.0 ** -60, 2.0 ** -61), (1e-300, 2e-300),
                                   (1e-20, 3.0), (0.8, 1e-6), (0.8, 1e-12),
                                   (0.8, 1e-20), (0.25, 0.75), (0.8, 3.5)])
def test_windows_against_independent_high_precision_reference(a: float, b: float) -> None:
    x = np.array([0.0, 1.0, 44.0, 45.0])
    tx, T = np.minimum(x * 1.5, 100), np.full(4, 104.0)
    params = np.array([[0.8, 12.0, a, b]])
    edges = np.linspace(8.0, 60.0, 13)
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        fast, alive = pc.e4_window_purchases(params, x, tx, T, 8.0, 60.0)
        slow = btyd.bgnbd_conditional(tuple(params[0]), x, tx, T, edges)
    for i, count in enumerate(x):
        k = 0.8 + count
        with localcontext() as ctx:
            ctx.prec = 90
            aa, bb = Decimal.from_float(a), Decimal.from_float(b)
            log_ratio = (Decimal(116) / Decimal.from_float(12 + float(tx[i]))).ln()
            odds = (aa / (bb + (int(count) - 1)) *
                    (Decimal.from_float(k) * log_ratio).exp()) if count else Decimal(0)
            pa = float(1 / (1 + odds))
        ref = pa * _reference(a, b + float(count), k, 116.0, 8.0, 60.0)
        assert alive[0, i] == pytest.approx(pa, rel=2e-14, abs=0.0)
        assert fast[0, i] == pytest.approx(ref, rel=2e-11, abs=0.0)
        assert slow.purchases[i].sum() == pytest.approx(ref, rel=2e-11, abs=0.0)
    np.testing.assert_allclose(fast[0], slow.purchases.sum(axis=1), rtol=2e-11, atol=0)
    assert pc.E4_ACCELERATED_MIN_A == 0.4


def _reference_alive(params: tuple[float, float, float, float], count: float,
                     recency: float, age: float) -> float:
    if count == 0:
        return 1.0
    with localcontext() as ctx:
        ctx.prec = 90
        r, alpha, a, b = map(Decimal.from_float, params)
        t, T = map(Decimal.from_float, (recency, age))
        rate_ratio = (alpha + T) / (alpha + t)
        odds = a / (b + (int(count) - 1)) * rate_ratio ** (r + int(count))
        return float(1 / (1 + odds))


@pytest.mark.parametrize("params", [(0.8, 1.0, 0.8, 1e-310), (446.0, 1.0, 1e-310, 3.0),
                                    (0.8, 1.0, 0.8, 1e-308), (0.8, 1.0, 0.8, 3.5)])
def test_extreme_weights_and_alive_odds_match_decimal(
        params: tuple[float, float, float, float]) -> None:
    r, alpha, a, b = params
    x = np.array([0.0, 1.0, 44.0, 45.0])
    tx, T = np.array([0.0, 1.0, 8.0, 8.0]), np.full(4, 9.0)
    edges = np.array([0.0, 0.125, 0.25])
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        fast, alive = pc.e4_window_purchases(np.array([params]), x, tx, T, edges[0], edges[-1])
        slow = btyd.bgnbd_conditional(params, x, tx, T, edges)
    pa = np.array([_reference_alive(params, float(c), float(t), 9.0)
                   for c, t in zip(x, tx, strict=True)])
    ref = np.array([p * _reference(a, b + float(c), r + float(c), alpha + 9.0, 0.0, 0.25)
                    for p, c in zip(pa, x, strict=True)])
    # Allow a single binary64 ulp at zero for correctly rounded subnormal probabilities/values.
    ulp = np.nextafter(0.0, 1.0)
    np.testing.assert_array_equal(alive[0], slow.p_alive_T)
    np.testing.assert_allclose(alive[0], pa, rtol=2e-12, atol=ulp)
    np.testing.assert_allclose(fast[0], ref, rtol=2e-11, atol=ulp)
    np.testing.assert_allclose(slow.purchases.sum(axis=1), ref, rtol=2e-11, atol=ulp)
    sim = ev.SimulatedValue(np.repeat(fast.T, 2, axis=1), np.zeros((4, 2)), pd.RangeIndex(4), ())
    rows = pc._coverage_rows("fixture", "E4", "parameter vs oracle",
                             *sim.interval(0.8, outcome=False), ref, np.array(["A"] * 4), x, 0.8)
    assert rows[0]["n"] == 4
    if b == 1e-310:
        assert _reference_alive(params, 1.0, 1.0, 9.0) > 0.0
        assert np.isinf(special.betaln(a, b))
        assert np.isfinite(special.betaln(a, b + 1.0))
    if r == 446.0:
        with np.errstate(over="ignore"):
            assert np.isinf(np.exp((r + 1.0) * np.log(5.0)))
        assert 0.01 < alive[0, 1] < 0.02


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("column", [0, 1, 2])
def test_e4_coverage_rejects_nonfinite_bounds_or_targets(bad: float, column: int) -> None:
    values = np.array([[1.0, 2.0], [3.0, 4.0], [2.0, 3.0]])
    segment, decile = np.array(["A", "B"]), np.array([0, 1])
    rows = pc._coverage_rows("fixture", "E4", "parameter vs oracle",
                             *values, segment, decile, 0.8)
    assert rows[0]["n"] == 2
    assert all(row["n"] == 1 for row in rows[1:])
    values[column, 1] = bad
    with pytest.raises(ValueError, match="requires finite interval bounds and targets"):
        pc._coverage_rows("fixture", "E4", "parameter vs oracle", *values, segment, decile, 0.8)


def _frozen_nodes(a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    """Unmodified checkpoint 8e12c12 rule, retained only as an ordinary-parameter comparator."""
    al, be = b - 1, a - 1
    j = np.arange(64, dtype=float)
    s = 2 * j + al + be
    diag = np.empty(64)
    diag[0] = (be - al) / (al + be + 2)
    diag[1:] = (be * be - al * al) / (s[1:] * (s[1:] + 2))
    j, s = j[1:], s[1:]
    off = np.sqrt(4 * j * (j + al) * (j + be) * (j + al + be)
                  / (s * s * (s + 1) * (s - 1)))
    y, v = linalg.eigh_tridiagonal(diag, off)
    return (1 + y) / 2, v[0] ** 2 / np.sum(v[0] ** 2)


def _frozen_purchases(params: tuple[float, ...], x: np.ndarray, tx: np.ndarray,
                      T: np.ndarray, edges: np.ndarray) -> np.ndarray:
    r, alpha, a, b = params
    out = np.empty((len(x), len(edges) - 1))
    for i, count in enumerate(x):
        pn, w = _frozen_nodes(a, b + count)
        A = (1 + pn[:, None] * edges / (alpha + T[i])) ** -(r + count)
        integral = np.einsum("j,jm->m", w / pn, A)
        odds = (a / (b + count - 1) * ((alpha + T[i]) / (alpha + tx[i])) **
                (r + count)) if count else 0
        out[i] = (integral[:-1] - integral[1:]) / (1 + odds)
    return out


@pytest.mark.parametrize("a", [0.39, 0.4, 0.8, 1.0, 2.4, 5.0])
def test_ordinary_parameters_match_frozen_tolerances(a: float) -> None:
    x = np.array([0.0, 1.0, 3.0, 44.0, 45.0])
    tx, T = np.minimum(x * 1.5, 100), np.full(len(x), 104.0)
    edges = np.linspace(8.7, 60.8, 13)
    for r, alpha, b in ((0.3, 5.0, 0.5), (0.8, 12.0, 3.0), (2.0, 30.0, 8.0)):
        params = (r, alpha, a, b)
        old = _frozen_purchases(params, x, tx, T, edges)
        new = btyd.bgnbd_conditional(params, x, tx, T, edges).purchases
        # P5 doubled-rule tolerance, and A12/A14 ordinary window equivalence bound.
        np.testing.assert_allclose(new, old, rtol=1e-10, atol=0)
        fast, _ = pc.e4_window_purchases(np.array([params]), x, tx, T,
                                        float(edges[0]), float(edges[-1]))
        np.testing.assert_allclose(fast[0], old.sum(axis=1), rtol=2e-11, atol=0)


def _prediction() -> btyd.E4Prediction:
    """Four fixed histories and known synthetic fits; no optimiser, generator or evidence file."""
    x = np.array([0.0, 1.0, 44.0, 45.0])
    data = btyd.PurchaseData(pd.DataFrame({
        "segment": "fixture", "channel": "fixture", "x": x,
        "t_x": np.minimum(x * 1.5, 100), "T": 104.0, "n": x + 1,
        "S": 20 * (x + 1), "L": (x + 1) * np.log(20.0),
    }), 30, 28, np.linspace(8.7, 60.8, 13), 0.6)

    def fit(component: str, params: tuple[float, ...]) -> btyd.ComponentFit:
        names = btyd.BGNBD_PARAMS if component == "bgnbd" else btyd.GG_PARAMS
        return btyd.ComponentFit(component, names, params, np.log(params),
                                 np.eye(len(params)) * 0.005, 0.0, 4, 90, "ok", (),
                                 FitDiagnostics(), ())

    bf, gf = fit("bgnbd", (0.8, 12.0, 0.8, 3.5)), fit("gg", (6.0, 4.0, 15.0))
    key = ("fixture", "fixture")
    fits = btyd.E4Fits({key: bf}, {key: gf}, bf, gf, {key: ("ok", bf, "ok", gf)})
    return btyd.predict_from(data, fits)


def _frozen_conditional(params: tuple[float, ...], x: np.ndarray, tx: np.ndarray,
                         T: np.ndarray, edges: np.ndarray) -> btyd.Conditional:
    r, alpha, a, b = params
    odds = np.where(x > 0, a / (b + np.where(x > 0, x, 1.0) - 1)
                    * ((alpha + T) / (alpha + tx)) ** (r + x), 0.0)
    pa = 1 / (1 + odds)
    end = np.empty(len(x))
    for i, count in enumerate(x):
        pn, w = _frozen_nodes(a, b + count)
        end[i] = pa[i] * (w @ (1 + pn * edges[0] / (alpha + T[i])) ** -(r + count))
    return btyd.Conditional(pa, end, _frozen_purchases(params, x, tx, T, edges))


def _frozen_window(params: np.ndarray, x: np.ndarray, tx: np.ndarray, T: np.ndarray,
                    e0: float, eh: float, max_x: int = 44) -> tuple[np.ndarray, np.ndarray]:
    out, alive = np.empty((len(params), len(x))), np.empty((len(params), len(x)))
    for d, (r, alpha, a, b) in enumerate(params):
        pn0, w0 = _frozen_nodes(a, b)
        for i, count in enumerate(x):
            if count > max_x or a < 0.4:
                pn, w = _frozen_nodes(a, b + count)
            else:
                pn = pn0
                w = w0 * (1 - pn) ** count * np.exp(special.betaln(a, b)
                                                    - special.betaln(a, b + count))
            k, rate = r + count, alpha + T[i]
            odds = (a / (b + count - 1) * (rate / (alpha + tx[i])) ** k) if count else 0
            alive[d, i] = 1 / (1 + odds)
            out[d, i] = alive[d, i] * (w / pn @ ((1 + pn * e0 / rate) ** -k
                                               - (1 + pn * eh / rate) ** -k))
    return out, alive


def test_point_predictions_and_p7_and_p8_streams_against_checkpoint(
        monkeypatch: pytest.MonkeyPatch) -> None:
    pred = _prediction()
    with monkeypatch.context() as patch:
        patch.setattr(btyd, "bgnbd_conditional", _frozen_conditional)
        old = btyd.predict_from(pred.data, pred.fits)
    columns = ["prediction", "prediction_discounted", "p_alive_T", "p_alive"]
    np.testing.assert_allclose(pred.frame[columns], old.frame[columns], rtol=2e-11, atol=0)
    # Ordinary synthetic point predictions format identically at P7's four decimal places.
    assert pred.frame[columns].round(4).equals(old.frame[columns].round(4))
    for simulator in (ev.e4_simulate, pc.e4_simulate_fast):
        new_rng, old_rng = np.random.default_rng(7003), np.random.default_rng(7003)
        new = simulator(pred, 20, new_rng)
        with monkeypatch.context() as patch:
            patch.setattr(ev, "bgnbd_conditional", _frozen_conditional)
            patch.setattr(pc, "e4_window_purchases", _frozen_window)
            old_sim = simulator(pred, 20, old_rng)
        np.testing.assert_allclose(new.expected, old_sim.expected, rtol=2e-11, atol=0)
        np.testing.assert_array_equal(new.outcome, old_sim.outcome)
        assert new_rng.bit_generator.state == old_rng.bit_generator.state
