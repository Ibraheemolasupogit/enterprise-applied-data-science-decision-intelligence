"""M11 P5: E4 (BG/NBD + Gamma-Gamma). Phase gate: likelihoods match quadrature, conditional
expectations match Monte Carlo (and the published closed form and the independent P2 oracle),
parameters are recovered; plus data, leakage, fallback and validation (development seeds)."""

from __future__ import annotations

import ast
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, special, stats

from decision_intelligence.customer_value import btyd
from decision_intelligence.customer_value import predictive_protocol as pp
from decision_intelligence.customer_value import purchase_oracle as po
from decision_intelligence.customer_value.features import noncontractual_features
from decision_intelligence.customer_value.panel import month_end
from decision_intelligence.data import predictive_ltv_scenarios as g
from decision_intelligence.data.customer_economics_scenarios import CHANNELS, SEGMENTS
from decision_intelligence.data.point_in_time import as_of

SMALL_ACQ = (20.0, 25.0, 30.0, 12.0)


@pytest.fixture(scope="module")
def sc() -> g.PurchasesScenario:
    return g.generate_purchases_scenario(g.PurchasesScenarioConfig(seed=4951))


def _raw_bgnbd(lam: float, p: float, x: int, t_x: float, T: float) -> float:
    alive = (1 - p) ** x * lam ** x * math.exp(-lam * T)
    dead = p * (1 - p) ** (x - 1) * lam ** x * math.exp(-lam * t_x) if x > 0 else 0.0
    return alive + dead


@pytest.mark.parametrize("params", [(0.75, 6.0, 0.8, 3.5), (1.3, 4.0, 2.4, 6.0)])
def test_bgnbd_likelihood_matches_two_dimensional_quadrature(params: tuple[float, ...]
                                                             ) -> None:
    r, alpha, a, b = params
    c_lam = r * math.log(alpha) - math.lgamma(r)
    c_p = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)

    def dens(lam: float, p: float) -> float:  # Gamma(r, alpha) x Beta(a, b), written out
        return math.exp(c_lam + (r - 1) * math.log(lam) - alpha * lam + c_p
                        + (a - 1) * math.log(p) + (b - 1) * math.log1p(-p))

    for x, t_x, T in ((0, 0.0, 30.0), (1, 12.0, 40.0), (4, 35.0, 60.0), (9, 50.0, 52.0)):
        ref, _ = integrate.dblquad(
            lambda p, lam, x=x, t_x=t_x, T=T: _raw_bgnbd(lam, p, x, t_x, T) * dens(lam, p),
            0, 40, 0, 1, epsabs=0, epsrel=1e-11)
        got = btyd.bgnbd_loglik(np.log(params), np.array([x], float), np.array([t_x]),
                                np.array([T]))[0][0]
        assert math.exp(got) == pytest.approx(ref, rel=1e-8)


def test_gamma_gamma_likelihood_matches_quadrature() -> None:
    p, q, gam = 6.0, 4.0, 15.0
    z = np.array([12.0, 30.5, 22.1, 9.4])
    lp = stats.gamma(q, scale=1 / gam).logpdf

    def integrand(nu: float) -> float:
        return math.exp(float(np.sum(stats.gamma(p, scale=1 / nu).logpdf(z))) + lp(nu)
                        + 30.0)

    ref = math.log(integrate.quad(integrand, 0, 5, epsabs=0, epsrel=1e-12, limit=400,
                                  points=[0.2, 0.3])[0]) - 30.0
    got = btyd.gg_loglik(np.log([p, q, gam]), np.array([4.0]), np.array([z.sum()]),
                         np.array([np.log(z).sum()]))[0][0]
    assert got == pytest.approx(ref, rel=1e-10)


def test_scores_match_finite_differences() -> None:
    rng = np.random.default_rng(4952)
    x = rng.integers(0, 12, 300).astype(float)
    T = rng.uniform(20, 120, 300)
    t_x = np.where(x > 0, T * rng.uniform(0.1, 1.0, 300), 0.0)
    n = x + 1
    S = n * rng.uniform(5, 40, 300)
    L = n * np.log(S / n) - rng.uniform(0, 0.5, 300)
    for fn, theta, args in ((btyd.bgnbd_loglik, np.log([0.7, 5.0, 0.9, 3.0]), (x, t_x, T)),
                            (btyd.gg_loglik, np.log([5.0, 3.0, 12.0]), (n, S, L))):
        g_an = fn(theta, *args)[1].sum(axis=0)
        for j in range(theta.size):
            e = np.zeros(theta.size)
            e[j] = 1e-6
            fd = (fn(theta + e, *args)[0].sum() - fn(theta - e, *args)[0].sum()) / 2e-6
            assert g_an[j] == pytest.approx(fd, rel=1e-6)


def test_beta_rule_reproduces_beta_moments_including_large_shapes() -> None:
    for a, b in ((0.8, 3.5), (0.8, 503.5), (2.4, 60.0), (5.0, 1.0)):
        pn, w = btyd.beta_nodes(a, b)
        assert np.all((pn > 0) & (pn < 1)) and w.sum() == pytest.approx(1.0, rel=1e-14)
        for k in range(1, 6):
            exact = math.prod((a + i) / (a + b + i) for i in range(k))
            assert float(np.sum(w * pn ** k)) == pytest.approx(exact, rel=1e-11)


def test_conditional_expectation_matches_the_published_closed_form() -> None:
    """Fader, Hardie & Lee (2005) eq. (10) holds for ``a > 1``."""
    r, alpha, a, b = 0.9, 5.0, 2.4, 6.0
    x = np.array([0.0, 2.0, 7.0])
    t_x = np.array([0.0, 20.0, 41.0])
    T = np.array([30.0, 45.0, 52.0])
    t = 26.0
    cond = btyd.bgnbd_conditional((r, alpha, a, b), x, t_x, T, np.array([0.0, t]))
    k = r + x
    hyp = special.hyp2f1(k, b + x, a + b + x - 1, t / (alpha + T + t))
    num = (a + b + x - 1) / (a - 1) * (1 - ((alpha + T) / (alpha + T + t)) ** k * hyp)
    den = 1 + (x > 0) * a / np.maximum(b + x - 1, 1e-12) * ((alpha + T) / (alpha + t_x)) ** k
    np.testing.assert_allclose(cond.purchases[:, 0], num / den, rtol=1e-11)
    np.testing.assert_allclose(cond.p_alive_T, 1 / den, rtol=1e-13)


def test_conditional_expectations_match_likelihood_weighted_monte_carlo() -> None:
    """Draw (lambda, p) from the prior, weight by the history's raw likelihood; independent of
    the quadrature. ``a < 1`` (the pinned dropout shape)."""
    r, alpha, a, b = 0.75, 6.0, 0.8, 3.5
    rng = np.random.default_rng(4953)
    draws = 1_000_000
    lam = rng.gamma(r, 1 / alpha, draws)
    p = rng.beta(a, b, draws)
    edges = np.array([6.0, 10.0, 30.0])
    for x, t_x, T in ((0, 0.0, 40.0), (3, 30.0, 45.0), (6, 20.0, 60.0)):
        alive = (1 - p) ** x * lam ** x * np.exp(-lam * T)
        dead = (p * (1 - p) ** (x - 1) * lam ** x * np.exp(-lam * t_x)) if x > 0 else 0.0
        w = alive + dead
        got = btyd.bgnbd_conditional((r, alpha, a, b), np.array([float(x)]), np.array([t_x]),
                                     np.array([T]), edges)
        targets = ((alive * (np.exp(-lam * p * edges[0]) - np.exp(-lam * p * edges[1])) / p,
                    got.purchases[0, 0]),
                   (alive * (np.exp(-lam * p * edges[1]) - np.exp(-lam * p * edges[2])) / p,
                    got.purchases[0, 1]),
                   (alive, got.p_alive_T[0]),
                   (alive * np.exp(-lam * p * edges[0]), got.p_alive_end[0]))
        wm = w.mean()
        for target, value in targets:
            est = target.mean() / wm
            se = float(np.std(target / wm - est * w / wm) / math.sqrt(draws))
            assert abs(value - est) <= 4.5 * se + 1e-12


def test_conditional_expectations_equal_the_p2_oracle_with_true_parameters(
        sc: g.PurchasesScenario) -> None:
    """Independent implementations: E4's formulas at the true cell parameters reproduce the
    oracle (which reads the config; this test may, the estimator may not)."""
    cfg = sc.config
    data = btyd.purchase_data(sc, 24, 12)
    oracle = po.purchases_oracle(sc, 24, 12).frame.reindex(data.frame.index)
    f = data.frame
    for s in SEGMENTS:
        for c in CHANNELS:
            sel = ((f["segment"] == s) & (f["channel"] == c)).to_numpy()
            si, ci = SEGMENTS.index(s), CHANNELS.index(c)
            cond = btyd.bgnbd_conditional(
                (cfg.r, cfg.alpha * cfg.segment_alpha_scale[si], cfg.dropout_a,
                 cfg.dropout_b[ci]), f["x"].to_numpy()[sel], f["t_x"].to_numpy()[sel],
                f["T"].to_numpy()[sel], data.edges)
            np.testing.assert_allclose(
                cond.purchases, oracle[[f"purchases_{j}" for j in range(1, 13)]]
                .to_numpy()[sel], rtol=1e-9, atol=1e-13)
            np.testing.assert_allclose(cond.p_alive_end, oracle["p_alive"].to_numpy()[sel],
                                       rtol=1e-12)
            spend = btyd.gg_expected_spend((cfg.gg_p, cfg.gg_q, cfg.gg_gamma[si]),
                                           f["n"].to_numpy()[sel], f["S"].to_numpy()[sel])
            np.testing.assert_allclose(spend, oracle["expected_spend"].to_numpy()[sel],
                                       rtol=1e-12)
    assert data.contribution_ratio == pytest.approx(cfg.gross_margin - cfg.fee_rate, abs=5e-4)


def test_doubled_quadrature_rule_changes_nothing() -> None:
    rng = np.random.default_rng(4954)
    x = rng.integers(0, 40, 200).astype(float)
    T = rng.uniform(5, 200, 200)
    t_x = np.where(x > 0, T * rng.uniform(0.05, 1.0, 200), 0.0)
    edges = np.linspace(4.0, 60.0, 13)
    a = btyd.bgnbd_conditional((0.75, 6.0, 0.8, 3.5), x, t_x, T, edges)
    b = btyd.bgnbd_conditional((0.75, 6.0, 0.8, 3.5), x, t_x, T, edges, nodes=128)
    np.testing.assert_allclose(a.purchases, b.purchases, rtol=1e-10, atol=1e-300)
    np.testing.assert_allclose(a.p_alive_end, b.p_alive_end, rtol=1e-12)


def _simulate_bgnbd(params: tuple[float, ...], n: int, seed: int
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r, alpha, a, b = params
    rng = np.random.default_rng(seed)
    lam, p = rng.gamma(r, 1 / alpha, n), rng.beta(a, b, n)
    T = rng.uniform(20.0, 160.0, n)
    t = np.zeros(n)
    x = np.zeros(n)
    t_x = np.zeros(n)
    alive = np.ones(n, dtype=bool)
    while alive.any():
        t[alive] += rng.exponential(1 / lam[alive])
        bought = alive & (t <= T)
        x[bought] += 1
        t_x[bought] = t[bought]
        alive = bought & (rng.random(n) >= p)
    return x, t_x, T


@pytest.mark.parametrize("params,seed", [((0.75, 6.0, 0.8, 3.5), 4955),
                                         ((1.5, 10.0, 2.0, 8.0), 4956)])
def test_bgnbd_parameters_are_recovered_within_3_se(params: tuple[float, ...], seed: int
                                                    ) -> None:
    x, t_x, T = _simulate_bgnbd(params, 15000, seed)
    fit = btyd.fit_bgnbd(x, t_x, T)
    assert fit.status == "ok"
    z = (fit.log_params - np.log(params)) / np.sqrt(np.diag(fit.cov_log))
    assert np.all(np.abs(z) <= 3.0)


def test_gamma_gamma_parameters_are_recovered_within_3_se() -> None:
    rng = np.random.default_rng(4957)
    p, q, gam = 6.0, 4.0, 15.0
    m = 8000
    nu = rng.gamma(q, 1 / gam, m)
    n = rng.integers(1, 8, m).astype(float)
    S, L = np.zeros(m), np.zeros(m)
    for i in range(m):
        z = rng.gamma(p, 1 / nu[i], int(n[i]))
        S[i], L[i] = z.sum(), np.log(z).sum()
    fit = btyd.fit_gamma_gamma(n, S, L)
    assert fit.status == "ok"
    z = (fit.log_params - np.log([p, q, gam])) / np.sqrt(np.diag(fit.cov_log))
    assert np.all(np.abs(z) <= 3.0)


def test_purchase_data_matches_the_features_and_ignores_post_cutoff_rows(
        sc: g.PurchasesScenario) -> None:
    data = btyd.purchase_data(sc, 24, 12)
    feats = noncontractual_features(sc, 24)
    pd.testing.assert_index_equal(data.frame.index, feats.index)
    np.testing.assert_array_equal(data.frame["x"].to_numpy(), feats["frequency"].to_numpy())
    np.testing.assert_allclose(data.frame["t_x"].to_numpy(), feats["recency"].to_numpy())
    np.testing.assert_allclose(data.frame["T"].to_numpy(), feats["age"].to_numpy())
    assert (data.frame["n"] >= 1).all()  # the acquisition purchase is always observed
    end = month_end(sc.start, 24)
    cut = replace(sc, purchases=as_of(sc.purchases, "recorded_at", end))
    tx = sc.purchases.copy()
    late = tx["recorded_at"] > end
    tx.loc[late, "revenue"] = 1e6
    scr = replace(sc, purchases=tx)
    a = btyd.predict(sc, 24, 12).frame
    pd.testing.assert_frame_equal(a, btyd.predict(cut, 24, 12).frame)
    pd.testing.assert_frame_equal(a, btyd.predict(scr, 24, 12).frame)


def test_cells_fall_back_to_the_pooled_fit_and_report_status() -> None:
    rng = np.random.default_rng(4958)
    rows = []
    for s, c, n in (("a", "x", 3000), ("a", "y", 3000), ("b", "x", 15)):
        x, t_x, T = _simulate_bgnbd((0.75, 6.0, 0.8, 3.5), n, int(rng.integers(1, 10**6)))
        nn = x + 1
        S = nn * rng.uniform(10, 30, n)
        rows.append(pd.DataFrame({"segment": s, "channel": c, "x": x, "t_x": t_x, "T": T,
                                  "n": nn, "S": S, "L": nn * np.log(S / nn)}))
    fits = btyd.fit_cells(pd.concat(rows, ignore_index=True))
    assert fits.pooled_bgnbd.status == "ok"
    bstat, used, _, _ = fits.used[("b", "x")]
    assert fits.bgnbd[("b", "x")].status != "ok"
    assert bstat == "fallback" and used is fits.pooled_bgnbd
    assert fits.used[("a", "x")][0] == "ok"
    empty = btyd.fit_bgnbd(np.zeros(5), np.zeros(5), np.full(5, 10.0))
    assert empty.status == "undefined"


def test_predictions_are_reproducible_and_order_invariant(sc: g.PurchasesScenario) -> None:
    a = btyd.predict(sc, 24, 12).frame
    shuffled = replace(sc, customers=sc.customers.sample(frac=1.0, random_state=1),
                       purchases=sc.purchases.sample(frac=1.0, random_state=2))
    b = btyd.predict(shuffled, 24, 12).frame
    # row order changes the floating-point summation order of the likelihood, so the optimum
    # is reproduced within the frozen convergence tolerance (score <= 1e-6 per customer)
    pd.testing.assert_frame_equal(a.sort_index(), b.sort_index(), rtol=1e-6)
    pd.testing.assert_frame_equal(a, btyd.predict(sc, 24, 12).frame)  # bit-identical repeat
    assert (a["p_alive"] <= a["p_alive_T"] + 1e-12).all()
    assert ((a["p_alive"] > 0) & (a["p_alive_T"] <= 1)).all()
    months = a[[f"purchases_{j}" for j in range(1, 13)]].to_numpy()
    np.testing.assert_allclose(months.sum(axis=1), a["expected_purchases"], rtol=1e-12)
    assert np.all(months >= 0)


def test_validation_uses_only_the_validation_landmarks_information(sc: g.PurchasesScenario
                                                                    ) -> None:
    cutoff, w = 30, 12
    lv = pp.validation_landmark(cutoff, w)
    assert lv is not None
    res = btyd.validate(sc, cutoff, w)
    assert res.landmark == lv and np.isfinite(res.candidate.mse)
    end = month_end(sc.start, lv)
    tx = sc.purchases.copy()
    tx.loc[tx["recorded_at"] > end, "revenue"] *= 3.0
    res2 = btyd.validate(replace(sc, purchases=tx), cutoff, w)
    pd.testing.assert_series_equal(res.frame["prediction"], res2.frame["prediction"])
    assert not res.frame["realised"].equals(res2.frame["realised"])


def test_estimation_code_never_reads_truth_or_oracles() -> None:
    tree = ast.parse(Path(btyd.__file__).read_text())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    modules = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not names & {"truth", "PurchasesTruth", "config", "account_closure_summary",
                        "oracle", "purchases_oracle"}
    assert not any(m.endswith("oracle") or m.startswith("decision_intelligence.data")
                   for m in modules)
