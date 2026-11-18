"""M11 P2: the non-contractual oracle (BG/NBD + Gamma-Gamma and its predeclared variants)
against independent references: published closed forms, brute-force numerical integration of
the raw likelihood, likelihood-weighted Monte Carlo, and quadrature convergence."""

from __future__ import annotations

import ast
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, special, stats

from decision_intelligence.customer_value import purchase_oracle as po
from decision_intelligence.customer_value.features import noncontractual_features
from decision_intelligence.customer_value.predictive_protocol import ORACLE_NUMERICS as ON
from decision_intelligence.data import predictive_ltv_scenarios as g

CFG = g.PurchasesScenarioConfig()
R, ALPHA, A, B = CFG.r, CFG.alpha, CFG.dropout_a, CFG.dropout_b[1]
X = np.array([0.0, 1.0, 3.0, 7.0, 20.0])
TX = np.array([0.0, 5.0, 30.0, 40.0, 90.0])
T = np.array([10.0, 50.0, 60.0, 45.0, 100.0])
EDGES = np.array([0.0, 4.3, 30.0, 120.0])
SMALL_ACQ = (20.0, 25.0, 30.0, 12.0)


def _full(v: float) -> np.ndarray:
    return np.full(X.size, v)


def test_quadrature_rules_reproduce_beta_and_gamma_moments() -> None:
    for a, b in ((0.8, 3.5), (0.8, 200.0), (2.0, 0.6)):
        p, w = po.beta_rule(a, b, 32)
        for j in range(1, 6):  # E[P^j] = prod (a + i) / (a + b + i)
            exact = math.prod((a + i) / (a + b + i) for i in range(j))
            assert np.sum(w * p ** j) == pytest.approx(exact, rel=1e-12)
    for shape in (0.75, 10.0, 400.0):
        y, w = po.log_gamma_rule(shape, 64)
        assert np.sum(w * y) == pytest.approx(shape, rel=1e-8)
        assert np.sum(w * y * y) == pytest.approx(shape * (shape + 1.0), rel=1e-8)
        assert np.sum(w * np.log(y)) == pytest.approx(special.digamma(shape), abs=1e-8)


def test_bgnbd_matches_the_published_closed_form() -> None:
    """Fader, Hardie and Lee (2005), eq. 10 and P(alive): valid for a > 1."""
    a = 2.4
    p_alive, _, n = po.bgnbd_conditional(X, TX, T, EDGES, r=R, alpha=_full(ALPHA), a=a,
                                         b=_full(B))
    k = R + X
    exact_alive = 1.0 / (1.0 + (X > 0) * a / (B + X - 1.0) * ((ALPHA + T) / (ALPHA + TX)) ** k)
    np.testing.assert_allclose(p_alive, exact_alive, rtol=1e-13)
    for j in range(1, EDGES.size):
        t = EDGES[j]
        f = special.hyp2f1(k, B + X, a + B + X - 1.0, t / (ALPHA + T + t))
        exact = ((a + B + X - 1.0) / (a - 1.0)
                 * (1.0 - ((ALPHA + T) / (ALPHA + T + t)) ** k * f)
                 / (1.0 + (X > 0) * a / (B + X - 1.0) * ((ALPHA + T) / (ALPHA + TX)) ** k))
        np.testing.assert_allclose(n[:, :j].sum(axis=1), exact, rtol=1e-12)


def _dblquad(f: object, lo2: float, hi2: float) -> float:
    return float(integrate.dblquad(f, lo2, hi2, 0.0, np.inf, epsabs=0.0, epsrel=1e-11)[0])


@pytest.mark.parametrize("i", [0, 2, 4])
def test_bgnbd_matches_brute_force_integration_at_the_pinned_parameters(i: int) -> None:
    """Raw likelihood over (lambda, p), a = 0.8 (no closed form): nothing shared with the
    oracle's algebra."""
    x, tx, t_obs = X[i], TX[i], T[i]
    _, at, n = po.bgnbd_conditional(X, TX, T, EDGES, r=R, alpha=_full(ALPHA), a=A,
                                    b=_full(B))

    def prior(lam: float, p: float) -> float:
        return float(lam ** (R - 1) * math.exp(-ALPHA * lam) * p ** (A - 1) * (1 - p) ** (B - 1))

    def alive(lam: float, p: float) -> float:
        return float((1 - p) ** x * lam ** x * math.exp(-lam * t_obs))

    def dead(lam: float, p: float) -> float:
        return float((x > 0) * p * (1 - p) ** max(x - 1, 0) * lam ** x * math.exp(-lam * tx))

    z = _dblquad(lambda lam, p: prior(lam, p) * (alive(lam, p) + dead(lam, p)), 0.0, 1.0)
    for j in range(EDGES.size - 1):
        e1, e2 = EDGES[j], EDGES[j + 1]
        num = _dblquad(lambda lam, p, e1=e1, e2=e2: prior(lam, p) * alive(lam, p)
                       * (math.exp(-lam * p * e1) - math.exp(-lam * p * e2)) / p, 0.0, 1.0)
        assert n[i, j] == pytest.approx(num / z, rel=1e-9)
    stay = _dblquad(lambda lam, p: prior(lam, p) * alive(lam, p)
                    * math.exp(-lam * p * EDGES[1]), 0.0, 1.0)
    assert at[i, 1] == pytest.approx(stay / z, rel=1e-9)


@pytest.mark.parametrize("i", [0, 2, 4])
def test_pareto_matches_brute_force_integration(i: int) -> None:
    """Raw Pareto/NBD likelihood over (lambda, mu) (Schmittlein, Morrison and Colombo)."""
    s, beta = CFG.pareto_s, CFG.pareto_beta
    x, tx, t_obs = X[i], TX[i], T[i]
    p_alive, _, n = po.pareto_conditional(X, TX, T, EDGES, r=R, alpha=_full(ALPHA), s=s,
                                          beta=beta)

    def prior(lam: float, mu: float) -> float:
        return float(lam ** (R - 1) * math.exp(-ALPHA * lam) * mu ** (s - 1) * math.exp(-beta * mu))

    def alive(lam: float, mu: float) -> float:
        return float(lam ** x * math.exp(-(lam + mu) * t_obs))

    def lik(lam: float, mu: float) -> float:
        return float(lam ** x * mu / (lam + mu) * math.exp(-(lam + mu) * tx)
                     + lam ** (x + 1) / (lam + mu) * math.exp(-(lam + mu) * t_obs))

    z = _dblquad(lambda lam, mu: prior(lam, mu) * lik(lam, mu), 0.0, np.inf)
    assert p_alive[i] == pytest.approx(
        _dblquad(lambda lam, mu: prior(lam, mu) * alive(lam, mu), 0.0, np.inf) / z, rel=1e-9)
    for j in range(EDGES.size - 1):
        e1, e2 = EDGES[j], EDGES[j + 1]
        num = _dblquad(lambda lam, mu, e1=e1, e2=e2: prior(lam, mu) * alive(lam, mu) * lam / mu
                       * (math.exp(-mu * e1) - math.exp(-mu * e2)), 0.0, np.inf)
        assert n[i, j] == pytest.approx(num / z, rel=1e-9)


def test_gamma_gamma_spend_matches_quadrature_and_the_published_form() -> None:
    p, q, gam = CFG.gg_p, CFG.gg_q, 15.0
    rng = np.random.default_rng(2)
    for n in (1, 4, 30):
        z = rng.gamma(p, 1.0 / 0.4, n)
        got = float(po.gamma_gamma_spend(np.array([n]), np.array([z.sum()]), p=p, q=q,
                                         gamma=np.array([gam]))[0])

        def log_post(nu: float, z: np.ndarray = z) -> float:
            return float(stats.gamma.logpdf(nu, q, scale=1 / gam)
                         + np.sum(stats.gamma.logpdf(z, p, scale=1 / nu)))

        grid = np.linspace(1e-4, 5.0, 20001)  # locate the posterior mode by brute force
        mode = float(grid[np.argmax([log_post(v) for v in grid])])
        top = log_post(mode)

        def post(nu: float, top: float = top) -> float:
            return math.exp(log_post(nu) - top)

        zn = (integrate.quad(post, 0, mode, epsrel=1e-13, limit=400)[0]
              + integrate.quad(post, mode, np.inf, epsrel=1e-13, limit=400)[0])
        mean = (integrate.quad(lambda nu: p / nu * post(nu), 0, mode, epsrel=1e-13,
                               limit=400)[0]
                + integrate.quad(lambda nu: p / nu * post(nu), mode, np.inf, epsrel=1e-13,
                                 limit=400)[0]) / zn
        assert got == pytest.approx(mean, rel=1e-8)
        weighted = ((q - 1) / (p * n + q - 1) * gam * p / (q - 1)
                    + p * n / (p * n + q - 1) * z.mean())  # Fader, Hardie and Lee (2005b)
        assert got == pytest.approx(weighted, rel=1e-13)


def _histories(m: int = 300) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1)
    x = rng.integers(0, 60, m).astype(float)
    t_obs = rng.uniform(1.0, 200.0, m)
    n = x + 1.0
    return {"x": x, "t_x": np.where(x > 0, t_obs * rng.uniform(0.3, 1.0, m), 0.0),
            "age": t_obs, "n_spend": n, "total": n * rng.uniform(5.0, 40.0, m),
            "alpha": rng.choice([7.5, 6.0, 4.5], m), "b": rng.choice(CFG.dropout_b, m),
            "gamma": rng.choice(CFG.gg_gamma, m)}


def _copula(h: dict[str, np.ndarray], rho: float, edges: np.ndarray,
            nodes: tuple[int, int, int] = po.COPULA_NODES) -> tuple[np.ndarray, ...]:
    return po.bgnbd_copula_conditional(
        h["x"], h["t_x"], h["age"], h["n_spend"], h["total"], edges, r=R, alpha=h["alpha"],
        a=A, b=h["b"], gg_p=CFG.gg_p, gg_q=CFG.gg_q, gamma=h["gamma"], rho=rho, nodes=nodes)


def test_copula_oracle_reduces_to_the_independent_oracle_at_zero_correlation() -> None:
    h = _histories()
    edges = np.linspace(0.0, 113.0, 25)
    p_alive, at, n = po.bgnbd_conditional(h["x"], h["t_x"], h["age"], edges, r=R,
                                          alpha=h["alpha"], a=A, b=h["b"])
    spend = po.gamma_gamma_spend(h["n_spend"], h["total"], p=CFG.gg_p, q=CFG.gg_q,
                                 gamma=h["gamma"])
    c_alive, c_at, c_n, c_v, c_s = _copula(h, 0.0, edges)
    np.testing.assert_allclose(c_alive, p_alive, rtol=1e-12)
    np.testing.assert_allclose(c_at, at, rtol=1e-7, atol=1e-14)
    np.testing.assert_allclose(c_n, n, rtol=1e-7, atol=1e-14)
    np.testing.assert_allclose(c_v, n * spend[:, None], rtol=1e-7, atol=1e-14)
    np.testing.assert_allclose(c_s, spend, rtol=1e-8)


def test_oracle_numerics_are_the_protocols() -> None:
    assert po.JACOBI_NODES == ON.bgnbd_jacobi_nodes == 64
    assert po.PARETO_NODES == ON.pareto_jacobi_nodes == 64
    assert po.COPULA_NODES == ON.copula_nodes == (64, 48, 32)
    assert ON.monte_carlo_draws == 1_000_000


def test_oracle_quadrature_converges() -> None:
    """Protocol check (A5): default rules vs doubled rules on realistic and extreme
    histories."""
    h = _histories()
    edges = np.linspace(0.0, 113.0, 25)
    base = _copula(h, -0.5, edges)
    doubled = (2 * ON.copula_nodes[0], 2 * ON.copula_nodes[1], 2 * ON.copula_nodes[2])
    fine = _copula(h, -0.5, edges, nodes=doubled)
    material = fine[3].sum(axis=1) > 1e-6  # ignore values that are numerically zero
    np.testing.assert_allclose(base[3].sum(axis=1)[material], fine[3].sum(axis=1)[material],
                               rtol=ON.copula_convergence_rtol)
    np.testing.assert_allclose(base[4], fine[4], rtol=ON.copula_spend_convergence_rtol)
    for coarse, dense in ((po.pareto_conditional(h["x"], h["t_x"], h["age"], edges, r=R,
                                                 alpha=h["alpha"], s=CFG.pareto_s,
                                                 beta=CFG.pareto_beta),
                           po.pareto_conditional(h["x"], h["t_x"], h["age"], edges, r=R,
                                                 alpha=h["alpha"], s=CFG.pareto_s,
                                                 beta=CFG.pareto_beta,
                                                 nodes=2 * ON.pareto_jacobi_nodes)),
                          (po.bgnbd_conditional(h["x"], h["t_x"], h["age"], edges, r=R,
                                                alpha=h["alpha"], a=A, b=h["b"]),
                           po.bgnbd_conditional(h["x"], h["t_x"], h["age"], edges, r=R,
                                                alpha=h["alpha"], a=A, b=h["b"],
                                                nodes=2 * ON.bgnbd_jacobi_nodes))):
        for c, d in zip(coarse, dense, strict=True):
            np.testing.assert_allclose(c, d, rtol=ON.convergence_rtol, atol=1e-300)


def test_copula_oracle_matches_likelihood_weighted_monte_carlo() -> None:
    """Draw (lambda, nu, p) from the copula prior exactly as the generator does and weight by
    the raw likelihood of the history (independent of the oracle's quadrature)."""
    rho, alpha, b, gam = -0.5, ALPHA, B, CFG.gg_gamma[1]
    rng = np.random.default_rng(7)
    draws = ON.monte_carlo_draws
    z1 = rng.standard_normal(draws)
    z2 = rho * z1 + math.sqrt(1 - rho * rho) * rng.standard_normal(draws)
    lam = stats.gamma.ppf(special.ndtr(z1), R, scale=1 / alpha)
    nu = stats.gamma.ppf(special.ndtr(z2), CFG.gg_q, scale=1 / gam)
    p = rng.beta(A, b, draws)
    edges = np.array([5.0, 9.0])
    for x, tx, t_obs, mean_spend in ((8, 50.0, 60.0, 40.0), (8, 50.0, 60.0, 10.0),
                                     (2, 20.0, 70.0, 30.0), (0, 0.0, 30.0, 25.0)):
        n = x + 1
        alive = (1 - p) ** x * lam ** x * np.exp(-lam * t_obs)
        dead = (x > 0) * p * (1 - p) ** max(x - 1, 0) * lam ** x * np.exp(-lam * tx)
        spend_lik = np.exp(CFG.gg_p * n * np.log(nu) - nu * n * mean_spend)
        w, wa = (alive + dead) * spend_lik, alive * spend_lik
        w_mean = w.mean()
        value = wa * (np.exp(-lam * p * edges[0]) - np.exp(-lam * p * edges[1])) / p * (
            CFG.gg_p / nu)
        got = po.bgnbd_copula_conditional(
            np.array([float(x)]), np.array([tx]), np.array([t_obs]), np.array([float(n)]),
            np.array([n * mean_spend]), edges, r=R, alpha=np.array([alpha]), a=A,
            b=np.array([b]), gg_p=CFG.gg_p, gg_q=CFG.gg_q, gamma=np.array([gam]), rho=rho)
        for target, oracle in ((value, got[3][0, 0]), (wa, got[0][0]),
                               (w * CFG.gg_p / nu, got[4][0])):
            est = target.mean() / w_mean
            se = float(np.std(target / w_mean - est * w / w_mean) / math.sqrt(draws))
            assert abs(oracle - est) <= ON.monte_carlo_z * se + 1e-12


def _truncate(sc: g.PurchasesScenario, cutoff: int, scramble: bool) -> g.PurchasesScenario:
    tx = sc.purchases
    late = tx["recorded_at"] > sc.month_end(cutoff)
    if not scramble:
        return replace(sc, purchases=tx.loc[~late].reset_index(drop=True))
    tx = tx.copy()
    tx.loc[late, "revenue"] = 1e6
    return replace(sc, purchases=tx)


@pytest.mark.parametrize("variant", [None, "pareto_nbd_dropout", "correlated_spend"])
def test_oracle_information_matches_features_and_ignores_post_cutoff_rows(
        variant: str | None) -> None:
    cfg = g.PurchasesScenarioConfig(seed=4706, n_months=30, acquisitions_per_month=SMALL_ACQ)
    sc = g.generate_purchases_scenario(g.purchases_variant(variant, cfg) if variant else cfg)
    cutoff, w = 20, 6
    res = po.purchases_oracle(sc, cutoff, w)
    feats = noncontractual_features(sc, cutoff)
    hist = res.history.frame
    assert hist.index.equals(feats.index) and res.frame.index.equals(feats.index)
    np.testing.assert_array_equal(hist["x"].to_numpy(), feats["frequency"].to_numpy())
    np.testing.assert_allclose(hist["t_x"].to_numpy(), feats["recency"].to_numpy())
    np.testing.assert_allclose(hist["age"].to_numpy(), feats["age"].to_numpy())
    assert (hist["n_spend"] == hist["x"] + 1).all()
    f = res.frame
    assert np.isfinite(f.to_numpy()).all() and (f["oracle"] >= 0).all()
    assert ((f["p_alive"] <= f["p_alive_observed"] + 1e-12)
            & (f["p_alive_observed"] <= 1.0)).all()
    assert (np.diff(res.history.edges) > 0).all() and res.history.edges[0] > 0
    for scramble in (False, True):
        pd.testing.assert_frame_equal(f, po.purchases_oracle(_truncate(sc, cutoff, scramble),
                                                             cutoff, w).frame)


def test_purchase_oracle_never_reads_the_latent_truth() -> None:
    tree = ast.parse(Path(po.__file__).read_text())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert not names & {"truth", "PurchasesTruth", "latent_expected_value", "purchase_rate",
                        "spend_rate", "death_time", "dropout_probability", "death_rate"}
