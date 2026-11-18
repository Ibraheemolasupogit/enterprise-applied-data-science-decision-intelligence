"""M11 P3, amendment A7: the narrow E1 geometric boundary exception. E1's geometric limit is
``alpha, beta -> infinity`` with ``mu = alpha / (alpha + beta)`` fixed, the closed boundary
``phi = 1 / (alpha + beta) = 0``. Independent checks (exact rational Beta moments, a separately
written geometric likelihood, an independent profile search) that the substitute is that
limit and a boundary maximum; narrowness; A6 and E2 untouched (development seeds only)."""

from __future__ import annotations

import math
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from scipy import optimize

from decision_intelligence.customer_value import contractual as ct
from decision_intelligence.customer_value import predictive_protocol as pp
from decision_intelligence.customer_value.features import PromotionPolicy
from decision_intelligence.data import customer_economics_scenarios as m10g

NOT_CONVERGED = ("not converged",)
AT_BOUND = "parameter at the log-scale bound"


def _geometric_counts(q: float, scale: float = 50.0, max_window: int = 60
                      ) -> ct.DurationData:
    """Expected counts of geometric lifetimes; the default is about 3,000 customers (a cell's
    size)."""
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


def _beta_moment(k: int, mu: Fraction, phi: Fraction) -> Fraction:
    """``E[theta^k]`` for ``theta ~ Beta(mu / phi, (1 - mu) / phi)``, exactly."""
    out = Fraction(1)
    for i in range(k):
        out *= (mu + i * phi) / (1 + i * phi)
    return out


def _expect(poly: list[int], mu: Fraction, phi: Fraction) -> Fraction:
    return sum((c * _beta_moment(k, mu, phi) for k, c in enumerate(poly)), Fraction(0))


def _poly(event: bool, t: int) -> list[int]:
    """Coefficients of ``(1 - theta)^(t - 1)`` (censored) or ``theta (1 - theta)^(t - 1)``."""
    base = [math.comb(t - 1, k) * (-1) ** k for k in range(t)]
    return [0, *base] if event else base


@pytest.mark.parametrize("t", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("event", [True, False])
def test_dispersion_score_matches_exact_beta_moments(t: int, event: bool) -> None:
    """``d log E[g(theta)] / d phi`` at ``phi = 0`` from exact rational arithmetic (error
    ``O(phi)`` with ``phi = 1e-15``), against the closed form used by the exception."""
    mu, phi = Fraction(7, 100), Fraction(1, 10**15)
    poly = _poly(event, t)
    e0, e1 = _expect(poly, mu, Fraction(0)), _expect(poly, mu, phi)
    exact = float((e1 - e0) / (phi * e0))
    data = ct.DurationData(np.array([float(t)]), np.array([event]), np.array([1.0]))
    got = ct.e1_dispersion_score_at_zero(float(mu), data)
    assert got == pytest.approx(exact, rel=1e-12, abs=1e-12)


def test_product_likelihood_is_exact_and_matches_the_fit_likelihood() -> None:
    rng = np.random.default_rng(4911)
    data = ct.DurationData.from_customers(rng.integers(1, 12, 300), rng.random(300) < 0.5)
    for la, lb in ((0.3, 1.5), (2.0, 4.0), (-1.0, 0.5)):
        theta = np.array([la, lb])
        assert ct.e1_log_likelihood_product(theta, data) == pytest.approx(
            ct.log_likelihood("E1", theta, data)[0], rel=1e-12)
    # On the ridge (alpha, beta ~ 1e5 .. 1e6): exact rational reference.
    a, b = 120_000, 1_900_000
    small = ct.DurationData(np.array([1.0, 3.0, 6.0, 9.0]), np.array([True, False, True, False]),
                            np.array([2.0, 5.0, 1.0, 3.0]))
    ref = 0.0
    for t, e, n in zip(small.t, small.event, small.count, strict=True):
        s = Fraction(1)
        for i in range(int(t) - 1):
            s *= Fraction(b + i, a + b + i)
        p = s * Fraction(a, a + b + int(t) - 1) if e else s
        ref += float(n) * math.log(p)
    got = ct.e1_log_likelihood_product(np.log([float(a), float(b)]), small)
    assert got == pytest.approx(ref, rel=1e-14)


def _geo_loglik(q: float, data: ct.DurationData) -> float:
    t, e, n = data.t, data.event, data.count
    return float(np.sum(n * np.where(e, math.log(q) + (t - 1) * math.log1p(-q),
                                     (t - 1) * math.log1p(-q))))


def test_e1_converges_to_the_geometric_likelihood_and_the_boundary_is_the_maximum() -> None:
    q = 0.06
    data = _geometric_counts(q)
    geo = _geo_loglik(q, data)
    for phi in (1e-6, 1e-8):
        ll = ct.e1_log_likelihood_product(np.log([q / phi, (1 - q) / phi]), data)
        assert ll == pytest.approx(geo + ct.e1_dispersion_score_at_zero(q, data) * phi,
                                   rel=1e-10)
    # Independent profile over phi: mu re-optimised by a bounded 1-D search for each fixed phi.
    prof = []
    for phi in (1e-4, 1e-3, 1e-2, 0.05, 0.2):
        res = optimize.minimize_scalar(
            lambda m, phi=phi: -ct.e1_log_likelihood_product(
                np.log([m / phi, (1 - m) / phi]), data),
            bounds=(0.03, 0.12), method="bounded", options={"xatol": 1e-12})
        prof.append(-float(res.fun))
    assert all(p < geo for p in prof)
    assert all(np.diff(prof) < 0)
    assert ct.e1_dispersion_score_at_zero(q, data) < 0


def test_boundary_fit_is_the_verified_geometric_limit() -> None:
    q = 0.06
    data = _geometric_counts(q)
    f = ct.fit("E1", data)
    assert f.status in ("unstable", "failed")  # the frozen rule, unchanged
    g = ct.e1_geometric_boundary(f, data)
    assert g is not None and g.status == "ok"
    assert g.boundary == pp.E1_GEOMETRIC_BOUNDARY.label
    assert g.geometric_churn == pytest.approx(q, rel=1e-12)
    assert g.params == (math.inf, math.inf) and g.finite
    assert g.log_likelihood == pytest.approx(_geo_loglik(q, data), rel=1e-12)
    # The E1 and E2 geometric limits are the same model.
    e2 = ct.geometric_boundary(ct.fit("E2", data), data)
    assert e2 is not None
    assert -math.expm1(-e2.params[0]) == pytest.approx(g.geometric_churn, rel=1e-12)


def test_the_e1_exception_is_narrow() -> None:
    data = _geometric_counts(0.06)
    f = ct.fit("E1", data)
    assert ct.e1_geometric_boundary(f, data) is not None
    ridge = f.log_params.copy()
    fl = replace(f, status="failed", reasons=NOT_CONVERGED)
    un = replace(f, status="unstable", reasons=(AT_BOUND,))
    assert ct.e1_geometric_boundary(fl, data) is not None
    assert ct.e1_geometric_boundary(un, data) is not None
    # (a) other reasons or statuses never qualify
    for bad in (replace(f, status="failed", reasons=("non-finite likelihood",)),
                replace(f, status="failed", reasons=("not converged", "non-finite likelihood")),
                replace(f, status="unstable", reasons=(AT_BOUND, "too few events")),
                replace(f, status="unstable", reasons=(AT_BOUND, "starting points disagree")),
                replace(f, status="unstable", reasons=("Hessian not positive definite",)),
                replace(f, status="ok", reasons=()), replace(f, status="undefined")):
        assert ct.e1_geometric_boundary(bad, data) is None
    # (b) diagnostics
    d = f.diagnostics
    for bad_d in (replace(d, finite_likelihood=False), replace(d, separated_without_penalty=True),
                  replace(d, start_log_likelihood_gap=1e-3)):
        assert ct.e1_geometric_boundary(replace(fl, diagnostics=bad_d), data) is None
    # (c) off the ridge: a degenerate lower boundary, or alpha + beta not beyond the bound
    for lp in ([-11.0, ridge[1]], [ridge[0] - 6.0, ridge[1] - 6.0]):
        assert ct.e1_geometric_boundary(replace(fl, log_params=np.array(lp)), data) is None
    # (d) mu disagreement
    shifted = ridge + np.array([0.01, 0.0])
    assert ct.e1_geometric_boundary(replace(fl, log_params=shifted), data) is None
    # (f) frailty data: positive phi score, the boundary is not a maximum
    rng = np.random.default_rng(4912)
    th = rng.beta(1.5, 20.0, 20000)
    life = rng.geometric(th)
    win = rng.integers(1, 49, 20000)
    frail = ct.DurationData.from_customers(np.minimum(life, win), life < win)
    mu = ct.geometric_mle(frail)
    churn = -math.expm1(-mu[0])
    fake = replace(fl, log_params=np.log([churn * 1e6, (1 - churn) * 1e6]),
                   log_likelihood=mu[1])
    assert ct.e1_dispersion_score_at_zero(churn, frail) > 0
    assert ct.e1_geometric_boundary(fake, frail) is None
    # (g) too few events for the one-parameter fit
    tiny = _geometric_counts(0.06, scale=0.4, max_window=30)
    assert ct.e1_geometric_boundary(fl, tiny) is None
    # very large cells: on the flat ridge the start gap is platform-dependent floating point
    # (3.7e-4 on macOS arm64, 6.1e-5 on Linux x86_64 CI), so only the rule is asserted: the
    # frozen 1e-4 start-agreement check decides, whichever side the gap falls.
    huge = _geometric_counts(0.06, scale=1000.0)
    big = ct.fit("E1", huge)
    agrees = big.diagnostics.start_log_likelihood_gap <= pp.STATUS_RULES.start_agreement_tolerance
    assert (ct.e1_geometric_boundary(big, huge) is not None) == agrees
    gap = replace(big.diagnostics, start_log_likelihood_gap=1e-3)
    assert ct.e1_geometric_boundary(replace(big, diagnostics=gap), huge) is None
    # never E2
    assert ct.e1_geometric_boundary(ct.fit("E2", data), data) is None


def test_a7_changes_only_e1_boundary_cells_and_leaves_a6_untouched() -> None:
    cfg = m10g.EconomicsScenarioConfig(seed=4913, frailty_variance=0.0)
    sc = m10g.generate_economics_scenario(cfg)
    pol = PromotionPolicy(cfg.promo_months, ("paid_search", "social"))
    off = ct.predict("E1", sc, 24, 12, pol, e1_boundary_exception=False)
    on = ct.predict("E1", sc, 24, 12, pol)
    assert not off.fits.boundary_cells and on.fits.boundary_cells
    changed = set(on.fits.boundary_cells)
    for key, (st, f) in on.fits.used.items():
        if key not in changed:
            assert off.fits.used[key][0] == st
            np.testing.assert_array_equal(off.fits.used[key][1].log_params, f.log_params)
        else:
            assert off.fits.used[key][0] == "fallback"
    keep = ~on.frame.set_index(["segment", "channel"]).index.isin(list(changed))
    pd.testing.assert_frame_equal(on.frame.loc[keep], off.frame.loc[keep])
    # E2 (A6) does not depend on the A7 switch.
    e2a = ct.predict("E2", sc, 24, 12, pol)
    e2b = ct.predict("E2", sc, 24, 12, pol, e1_boundary_exception=False)
    pd.testing.assert_frame_equal(e2a.frame, e2b.frame)
    # Where both models use their geometric limit, the customer predictions agree.
    both = changed & set(e2a.fits.boundary_cells)
    assert both
    sel = on.frame.set_index(["segment", "channel"]).index.isin(list(both))
    np.testing.assert_allclose(on.frame.loc[sel, "prediction"],
                               e2a.frame.loc[sel, "prediction"], rtol=1e-9)


def test_cohort_value_of_a_geometric_limit_converges() -> None:
    """A geometric tail always converges: E2 at v = 0 (A6, and E1's A7 limit passed as its E2
    equivalent) gives the geometric sum, never a division error or 'undefined'."""
    mu = 0.06
    b = -math.log1p(-mu)

    def flat(a: np.ndarray) -> np.ndarray:
        return np.ones(np.shape(a))

    got = ct.cohort_value("E2", (b, 0.0), flat, horizon=None, monthly_rate=0.0)
    assert got.status == "ok"
    assert got.value == pytest.approx(1 / mu, rel=1e-12)  # sum (1 - mu)^a
