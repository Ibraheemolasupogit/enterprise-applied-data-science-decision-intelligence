"""M11 P3, amendment A6: the narrow E2 ``v = 0`` boundary exception. Independent checks that
the substitute is the geometric limit and a genuine boundary maximum, that the exception is
narrow (every other status and fallback rule unchanged), and that disabling it reproduces the
pre-A6 rule (development seeds only)."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy import optimize

from decision_intelligence.customer_value import contractual as ct
from decision_intelligence.customer_value import predictive_protocol as pp
from decision_intelligence.customer_value.features import PromotionPolicy
from decision_intelligence.data import customer_economics_scenarios as m10g

AT_BOUND = "parameter at the log-scale bound"


def _geometric_counts(b: float, scale: float = 1000.0, max_window: int = 60
                      ) -> ct.DurationData:
    """Expected counts (no sampling noise) of geometric lifetimes, windows 1..max_window."""
    q = 1 - math.exp(-b)
    ts, es, cs = [], [], []
    for w in range(1, max_window + 1):
        for k in range(1, w):
            ts.append(k)
            es.append(True)
            cs.append(scale * q * (1 - q) ** (k - 1))
        ts.append(w)
        es.append(False)
        cs.append(scale * (1 - q) ** (w - 1))
    return ct.DurationData(np.array(ts, float), np.array(es), np.array(cs))


def _sample(b: float, v: float, n: int, seed: int) -> ct.DurationData:
    rng = np.random.default_rng(seed)
    z = rng.gamma(1 / v, v, n) if v > 0 else np.ones(n)
    life = rng.geometric(-np.expm1(-b * z))
    win = rng.integers(1, 49, n)
    return ct.DurationData.from_customers(np.minimum(life, win), life < win)


def _geo_loglik(q: float, data: ct.DurationData) -> float:
    """Geometric log-likelihood written directly (no shared code with the module)."""
    t, e, n = data.t, data.event, data.count
    return float(np.sum(n * np.where(e, math.log(q) + (t - 1) * math.log1p(-q),
                                     (t - 1) * math.log1p(-q))))


def test_geometric_mle_is_the_independent_maximiser() -> None:
    data = _sample(0.07, 0.0, 5000, 4901)
    b, ll, info = ct.geometric_mle(data)
    res = optimize.minimize_scalar(lambda q: -_geo_loglik(q, data), bounds=(1e-6, 0.5),
                                   method="bounded", options={"xatol": 1e-12})
    assert b == pytest.approx(-math.log1p(-res.x), rel=1e-7)
    assert ll == pytest.approx(-res.fun, rel=1e-12)
    assert ll == pytest.approx(ct.log_likelihood("E2", np.array([math.log(b), -math.inf]),
                                                 data)[0], rel=1e-12)
    h = 1e-4  # observed information of log b by a second difference of the direct likelihood

    def lu(u: float) -> float:
        return _geo_loglik(-math.expm1(-math.exp(u)), data)

    u = math.log(b)
    fd = -(lu(u + h) - 2 * lu(u) + lu(u - h)) / (h * h)
    assert info == pytest.approx(fd, rel=1e-5)
    no_events = ct.DurationData.from_customers(np.full(3, 4), np.zeros(3, bool))
    assert math.isnan(ct.geometric_mle(no_events)[0])  # no MLE without events


@pytest.mark.parametrize("v_true", [0.0, 0.5])
def test_v_score_at_zero_is_the_derivative_of_the_e2_likelihood(v_true: float) -> None:
    data = _sample(0.06, v_true, 4000, 4902)
    b = ct.geometric_mle(data)[0]
    base = ct.log_likelihood("E2", np.array([math.log(b), -math.inf]), data)[0]

    def slope(h: float) -> float:
        return (ct.log_likelihood("E2", np.log([b, h]), data)[0] - base) / h

    richardson = 2 * slope(1e-7) - slope(2e-7)
    assert ct.e2_v_score_at_zero(b, data) == pytest.approx(richardson, rel=1e-4, abs=1e-6)
    if v_true > 0:
        assert ct.e2_v_score_at_zero(b, data) > 0  # frailty: v = 0 is not a maximum


def test_survival_at_v_zero_is_exactly_geometric() -> None:
    ages = np.arange(0.0, 200.0)
    np.testing.assert_allclose(ct.survival("E2", (0.05, 0.0), ages), np.exp(-0.05 * ages),
                               rtol=1e-14)
    paid = ct.conditional_paid("E2", np.array([math.log(0.05), -math.inf]),
                               np.array([0.0, 7.0, 30.0]), 12)
    np.testing.assert_allclose(paid, np.exp(-0.05 * np.arange(1, 13))[None, :].repeat(3, 0),
                               rtol=1e-14)


def test_boundary_fit_is_the_verified_geometric_limit_and_a_profile_maximum() -> None:
    b_true = 0.06
    data = _geometric_counts(b_true)
    f = ct.fit("E2", data)
    assert f.status == "unstable" and AT_BOUND in f.reasons  # the frozen rule, unchanged
    g = ct.geometric_boundary(f, data)
    assert g is not None
    assert g.status == "ok" and g.boundary == pp.E2_GEOMETRIC_BOUNDARY.label
    assert g.params[1] == 0.0 and g.params[0] == pytest.approx(b_true, rel=1e-12)
    assert math.isnan(g.se[1]) and g.se[0] > 0
    assert g.log_likelihood >= f.log_likelihood - 1e-9 * abs(f.log_likelihood)
    # Convergence, independently: the profile log-likelihood over v (b re-optimised for each
    # fixed v by a 1-D search, not the module's optimiser) is maximised at v = 0.
    prof = []
    for v in (1e-4, 1e-3, 1e-2, 0.1, 0.5):
        res = optimize.minimize_scalar(
            lambda lb, v=v: -ct.log_likelihood("E2", np.array([lb, math.log(v)]), data)[0],
            bounds=(math.log(b_true) - 1, math.log(b_true) + 1), method="bounded",
            options={"xatol": 1e-12})
        prof.append(-float(res.fun))
    assert all(p <= g.log_likelihood + 1e-9 for p in prof)
    assert all(np.diff(prof) < 0)


def test_the_exception_is_narrow() -> None:
    data = _geometric_counts(0.06)
    f = ct.fit("E2", data)
    assert ct.geometric_boundary(f, data) is not None
    # (a) only the boundary reason (+ Hessian) qualifies
    for extra in ("too few events", "starting points disagree", "too few events per parameter"):
        assert ct.geometric_boundary(replace(f, reasons=(*f.reasons, extra)), data) is None
    assert ct.geometric_boundary(replace(f, status="failed"), data) is None
    assert ct.geometric_boundary(replace(f, status="ok", reasons=()), data) is None
    bad_conv = replace(f, diagnostics=replace(f.diagnostics, converged=False))
    assert ct.geometric_boundary(bad_conv, data) is None
    # (b) v at the UPPER bound, or b at a bound, does not qualify
    assert ct.geometric_boundary(replace(f, log_params=np.array([f.log_params[0], 12.0])),
                                 data) is None
    assert ct.geometric_boundary(replace(f, log_params=np.array([-11.0, -15.0])), data) is None
    # (d) the E2 b must be the geometric b
    off = replace(f, log_params=f.log_params + np.array([1e-3, 0.0]))
    assert ct.geometric_boundary(off, data) is None
    # (e) a better E2 likelihood than the geometric one is not a boundary limit
    assert ct.geometric_boundary(replace(f, log_likelihood=f.log_likelihood + 1.0), data) \
        is None
    # (f) frailty data: positive v score at v = 0, so v = 0 is not a maximum
    frail = _sample(0.06, 0.5, 20000, 4903)
    fake = replace(f, log_params=np.array([math.log(ct.geometric_mle(frail)[0]), -15.0]),
                   log_likelihood=ct.geometric_mle(frail)[1])
    assert ct.geometric_boundary(fake, frail) is None
    # (g) the one-parameter geometric fit must itself be ok (>= 30 events)
    tiny = _geometric_counts(0.06, scale=0.4, max_window=30)
    assert tiny.n_events < pp.STATUS_RULES.min_events
    assert ct.geometric_boundary(replace(f, n_events=tiny.n_events), tiny) is None
    # E1 is never covered
    e1 = ct.fit("E1", data)
    assert ct.geometric_boundary(e1, data) is None


def test_cells_use_the_geometric_limit_and_other_fallbacks_are_unchanged() -> None:
    rng = np.random.default_rng(4904)
    rows = []
    for s, c, n, v in (("a", "x", 6000, 0.0), ("a", "y", 6000, 0.0), ("b", "x", 6000, 0.6),
                       ("b", "y", 30, 0.0)):
        z = rng.gamma(1 / v, v, n) if v > 0 else np.ones(n)
        life = rng.geometric(-np.expm1(-0.07 * z))
        win = rng.integers(1, 40, n)
        rows.append(pd.DataFrame({"segment": s, "channel": c, "paid_months": np.minimum(life, win),
                                  "churned": life < win}))
    status = pd.concat(rows, ignore_index=True)
    after = ct.fit_cells("E2", status)
    before = ct.fit_cells("E2", status, boundary_exception=False)
    assert after.cells.keys() == before.cells.keys()
    for key, f in after.cells.items():  # the unrestricted fits are identical
        assert f.log_likelihood == before.cells[key].log_likelihood
        if f.status == "ok":
            assert after.used[key][0] == before.used[key][0] == "ok"
            assert after.used[key][1] is f and before.used[key][1] is before.cells[key]
    for key in after.boundary_cells:
        assert before.used[key][0] == "fallback"
        st, g = after.used[key]
        assert st == "ok" and g.boundary is not None and g.params[1] == 0.0
    assert after.used[("b", "y")][0] == "fallback"  # too few events: unchanged rule
    assert after.used[("b", "x")][0] == "ok" and after.used[("b", "x")][1].boundary is None


def test_disabling_the_exception_reproduces_the_pre_a6_predictions() -> None:
    """A homogeneous-churn M10 scenario: with A6 off the boundary cells fall back exactly as
    before; with A6 on they use the geometric limit and nothing else changes."""
    cfg = m10g.EconomicsScenarioConfig(seed=4905, frailty_variance=0.0)
    sc = m10g.generate_economics_scenario(cfg)
    pol = PromotionPolicy(cfg.promo_months, ("paid_search", "social"))
    off = ct.predict("E2", sc, 24, 12, pol, boundary_exception=False)
    on = ct.predict("E2", sc, 24, 12, pol)
    assert not off.fits.boundary_cells and on.fits.boundary_cells
    changed = set(on.fits.boundary_cells)
    for key, (st, f) in on.fits.used.items():
        if key not in changed:
            assert off.fits.used[key][0] == st
            np.testing.assert_array_equal(off.fits.used[key][1].log_params, f.log_params)
    keep = ~on.frame.set_index(["segment", "channel"]).index.isin(list(changed))
    pd.testing.assert_frame_equal(on.frame.loc[keep], off.frame.loc[keep])
    e1_on = ct.predict("E1", sc, 24, 12, pol)
    e1_off = ct.predict("E1", sc, 24, 12, pol, boundary_exception=False)
    pd.testing.assert_frame_equal(e1_on.frame, e1_off.frame)  # E1 ignores the A6 switch
    e1_pre_a7 = ct.predict("E1", sc, 24, 12, pol, e1_boundary_exception=False)
    assert not e1_pre_a7.fits.boundary_cells  # E1 boundary cells come from A7 only
