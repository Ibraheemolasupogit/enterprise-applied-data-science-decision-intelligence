"""Observable-information oracle for the Milestone 11 non-contractual scenario (P2).

**Evaluation reference only.** ``E[R | F_C]`` under the *true* generator of
:func:`~decision_intelligence.data.predictive_ltv_scenarios.generate_purchases_scenario` and
its *true parameters* (read from the scenario's config, never from its latent truth), using
the protocol's non-contractual information set at cutoff ``C``: the customer's segment,
channel and acquisition date, and every purchase (time and spend) in the completely recorded
months ``<= T_C = complete_through(C)`` (the same months the features use; all of them are
recorded by ``C``). Purchases partially recorded in months ``C - 1`` and ``C`` are excluded,
as they are from the features, so this is the ceiling for models with the protocol's
information. No estimator or feature module may import it (AST-tested).

Times are in weeks (``config.time_unit_days``) since the acquisition purchase. ``T`` is the
end of month ``T_C``; ``x`` the repeat purchases in ``(0, T]``, ``t_x`` the last of them
(0 if none); the spends are every observed purchase including the acquisition one. The
target window is months ``C+1 .. C+H``: ``e_0 < e_1 < ... < e_H`` are the month ends of
``C, C+1, ..., C+H`` measured in weeks after ``T`` (the same for every customer).

**BG/NBD + Gamma-Gamma (base) — exact up to one smooth 1-D quadrature.** Rates vary by cell
(``alpha_s`` by segment, ``b_ch`` by channel), all known. With ``k = r + x``:

* ``P(alive at T | F) = 1 / (1 + [x > 0] a / (b + x - 1) ((alpha + T) / (alpha + t_x))^k)``;
* given alive, ``lambda ~ Gamma(k, alpha + T)`` and ``p ~ Beta(a, b + x)`` independently,
  so with ``A(e) = (1 + p e / (alpha + T))^-k`` (``lambda`` integrated analytically)
  ``E[N(T + e_j, T + e_{j+1}) | F] = P(alive) E_p[(A(e_j) - A(e_{j+1})) / p]``, and
  ``P(alive at T + e | F) = P(alive) E_p[A(e)]``. ``E_p`` is Gauss-Jacobi on the exact
  ``Beta(a, b + x)`` weight (the integrand is analytic on ``[0, 1]``);
* spend is independent of purchasing, so ``E[value] = E[N] x E[spend | F] x (margin - fee)``
  with the conjugate ``E[spend | F] = gg_p (gamma_s + sum z) / (gg_p n + gg_q - 1)``
  (``n`` observed spends). The published hypergeometric closed forms need ``a > 1``; the
  pinned ``a = 0.8``, hence the quadrature (tested against the closed form when ``a > 1``).

**Pareto/NBD dropout (variant)** — lifetime ``tau ~ Exp(mu)``, ``mu ~ Gamma(s, beta)``. Given
alive at ``T``, ``lambda ~ Gamma(k, alpha + T)`` and ``mu ~ Gamma(s, beta + T)``, so the
window expectation is closed form; the death-before-``T`` term is a 1-D integral over the
death time, which a change of variable turns into a Beta expectation of an analytic
function (Gauss-Jacobi).

**Correlated spend (variant)** — a Gaussian copula couples ``lambda`` and ``nu``. The oracle
reweights the independent posterior (alive and dead mixture components for ``lambda``, the
conjugate ``nu`` posterior, ``p`` by Gauss-Jacobi) by the copula density, by tensor Gauss
quadrature (Gauss-Legendre in ``log lambda`` and ``log nu`` against the exact Gamma
densities, Golub-Welsch Gauss-Jacobi for ``p``). With zero correlation
it reduces to the base oracle (tested); its quadrature error is measured in the tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import linalg, special

from decision_intelligence.common.validation import FloatArray
from decision_intelligence.customer_value.panel import month_end, observe
from decision_intelligence.customer_value.predictive_protocol import ORACLE_NUMERICS
from decision_intelligence.data.customer_economics_scenarios import CHANNELS, SEGMENTS
from decision_intelligence.data.predictive_ltv_scenarios import PurchasesScenario

JACOBI_NODES = ORACLE_NUMERICS.bgnbd_jacobi_nodes
"""Gauss-Jacobi nodes for ``p`` (BG/NBD); pinned by the protocol (amendment A5)."""
PARETO_NODES = ORACLE_NUMERICS.pareto_jacobi_nodes
"""Gauss-Jacobi nodes for the Pareto/NBD death-time integral; pinned by the protocol."""
COPULA_NODES = ORACLE_NUMERICS.copula_nodes
"""Correlated-spend tensor rule: nodes for ``lambda`` (per mixture component), ``p``, ``nu``;
pinned by the protocol."""
_CHUNK = 256


# --- Gauss rules for exact Beta and Gamma weights (Golub-Welsch, normalised) ---------------
@lru_cache(maxsize=4096)
def beta_rule(a: float, b: float, n: int) -> tuple[FloatArray, FloatArray]:
    """Nodes and weights with ``sum w f(p) ~= E[f(P)]``, ``P ~ Beta(a, b)``."""
    al, be = b - 1.0, a - 1.0  # Jacobi weight (1 - y)^al (1 + y)^be on [-1, 1], p = (1+y)/2
    i = np.arange(n, dtype=np.float64)
    s = 2.0 * i + al + be
    with np.errstate(divide="ignore", invalid="ignore"):
        diag = np.where(s * (s + 2.0) != 0.0, (be * be - al * al) / (s * (s + 2.0)),
                        (be - al) / (al + be + 2.0))
    diag[0] = (be - al) / (al + be + 2.0)
    j = i[1:]
    sj = 2.0 * j + al + be
    off = np.sqrt(4.0 * j * (j + al) * (j + be) * (j + al + be)
                  / (sj * sj * (sj + 1.0) * (sj - 1.0)))
    y, vec = linalg.eigh_tridiagonal(diag, off)
    w = vec[0, :] ** 2
    return np.asarray((1.0 + y) / 2.0), np.asarray(w / w.sum())


@lru_cache(maxsize=4096)
def log_gamma_rule(shape: float, n: int, tail: float = 1e-15) -> tuple[FloatArray, FloatArray]:
    """Nodes ``y`` and weights with ``sum w f(y) ~= E[f(Y)]``, ``Y ~ Gamma(shape, 1)``: Gauss-
    Legendre in ``log y`` between the ``tail`` and ``1 - tail`` quantiles, weighted by the
    exact density of ``log Y``. Functions of ``log y`` that are smooth (such as a copula
    density through ``Phi^-1(F(lambda))``, which is not smooth in ``y`` at 0) integrate with
    exponential convergence."""
    lo = math.log(float(special.gammaincinv(shape, tail)))
    hi = math.log(float(special.gammainccinv(shape, tail)))
    t, w = np.polynomial.legendre.leggauss(n)
    v = lo + (hi - lo) * (t + 1.0) / 2.0
    log_density = shape * v - np.exp(v) - special.gammaln(shape)
    wts = w * np.exp(log_density)
    return np.asarray(np.exp(v)), np.asarray(wts / wts.sum())


def _groups(*columns: FloatArray) -> list[tuple[tuple[float, ...], NDArray[np.intp]]]:
    """Row indices of each distinct combination of ``columns`` (sorted)."""
    keys, inverse = np.unique(np.column_stack(columns), axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    return [(tuple(float(v) for v in keys[i]), np.flatnonzero(inverse == i))
            for i in range(keys.shape[0])]


def _log_odds_alive_bgnbd(x: FloatArray, t_x: FloatArray, age: FloatArray, r: float,
                          alpha: FloatArray, a: float, b: FloatArray) -> FloatArray:
    """``log P(dead by T | F) - log P(alive at T | F)`` (``-inf`` when ``x = 0``)."""
    out = np.full(x.shape, -np.inf)
    pos = x > 0
    k = r + x[pos]
    out[pos] = (np.log(a / (b[pos] + x[pos] - 1.0))
                + k * np.log((alpha[pos] + age[pos]) / (alpha[pos] + t_x[pos])))
    return out


# --- BG/NBD ------------------------------------------------------------------------------------
def bgnbd_conditional(x: FloatArray, t_x: FloatArray, age: FloatArray, edges: FloatArray, *,
                      r: float, alpha: FloatArray, a: float, b: FloatArray,
                      nodes: int = JACOBI_NODES
                      ) -> tuple[FloatArray, FloatArray, FloatArray]:
    """BG/NBD conditional quantities per customer.

    Returns ``P(alive at T)`` ``(n,)``, ``P(alive at T + e_j)`` ``(n, m+1)`` and the expected
    purchases in ``(T + e_j, T + e_{j+1}]`` ``(n, m)``, for ``edges`` ``e_0 < ... < e_m``."""
    x, t_x, age = (np.asarray(v, dtype=np.float64) for v in (x, t_x, age))
    alpha, b = np.asarray(alpha, dtype=np.float64), np.asarray(b, dtype=np.float64)
    e = np.asarray(edges, dtype=np.float64)
    p_alive = special.expit(-_log_odds_alive_bgnbd(x, t_x, age, r, alpha, a, b))
    alive_at = np.empty((x.size, e.size))
    purchases = np.empty((x.size, e.size - 1))
    for (xi, bi), rows in _groups(x, b):
        p, w = beta_rule(a, float(bi) + float(xi), nodes)
        k = r + float(xi)
        c = alpha[rows] + age[rows]
        log_a = -k * np.log1p(p[None, :, None] * e[None, None, :] / c[:, None, None])
        big_a = np.exp(log_a)  # (g, nodes, m+1)
        step = big_a[:, :, :-1] * -np.expm1(log_a[:, :, 1:] - log_a[:, :, :-1])
        alive_at[rows] = np.einsum("q,gqj->gj", w, big_a)
        purchases[rows] = np.einsum("q,gqj->gj", w / p, step)
    return (p_alive, alive_at * p_alive[:, None], purchases * p_alive[:, None])


# --- Pareto/NBD --------------------------------------------------------------------------------
def pareto_conditional(x: FloatArray, t_x: FloatArray, age: FloatArray, edges: FloatArray, *,
                       r: float, alpha: FloatArray, s: float, beta: float,
                       nodes: int = PARETO_NODES
                       ) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Pareto/NBD conditional quantities (same outputs as :func:`bgnbd_conditional`).

    ``P(dead by T | F) / P(alive at T | F) = s (beta + T)^s int_{t_x}^T ((alpha + T) /
    (alpha + u))^k (beta + u)^-(s+1) du``. With ``q = (alpha + t_x) / (alpha + u)`` and
    ``m = k + s`` this is ``s (beta + T)^s ((alpha + T) / (alpha + t_x))^k (alpha + t_x) / m
    x J``, ``J = int_{q_min}^1 m q^(m-1) g(q) dq``, ``g(q) = (alpha + t_x + (beta - alpha)
    q)^-(s+1)``, ``q_min = (alpha + t_x) / (alpha + T)``: ``J = E[g(Q)] - q_min^m E[g(q_min Q)]``
    with ``Q ~ Beta(m, 1)`` (Gauss-Jacobi on the exact weight; ``g`` is analytic on
    ``[0, 1]``)."""
    x, t_x, age = (np.asarray(v, dtype=np.float64) for v in (x, t_x, age))
    alpha = np.asarray(alpha, dtype=np.float64)
    e = np.asarray(edges, dtype=np.float64)
    k = r + x
    m = k + s
    at, bt = alpha + age, beta + age
    c0 = alpha + t_x
    q_min = c0 / at
    j_int = np.empty(x.size)
    for (xi,), rows in _groups(x):
        q, w = beta_rule(r + xi + s, 1.0, nodes)
        cr, dr = c0[rows][:, None], (beta - alpha[rows])[:, None]
        full = np.sum(w * (cr + dr * q) ** -(s + 1.0), axis=1)
        part = np.sum(w * (cr + dr * q_min[rows][:, None] * q) ** -(s + 1.0), axis=1)
        j_int[rows] = full - np.exp(m[rows] * np.log(q_min[rows])) * part
    with np.errstate(divide="ignore"):
        log_odds = (math.log(s) + s * np.log(bt) - k * np.log(q_min) + np.log(c0)
                    - np.log(m) + np.log(np.maximum(j_int, 0.0)))
    p_alive = special.expit(-log_odds)
    log_ratio = np.log(bt[:, None] / (bt[:, None] + e[None, :]))  # (n, m+1), <= 0
    alive_at = p_alive[:, None] * np.exp(s * log_ratio)
    if abs(s - 1.0) < 1e-12:
        diff = log_ratio[:, :-1] - log_ratio[:, 1:]
    else:
        g = np.expm1((s - 1.0) * log_ratio)
        diff = (g[:, :-1] - g[:, 1:]) / (s - 1.0)
    purchases = (p_alive * k / at * bt)[:, None] * diff
    return p_alive, alive_at, purchases


# --- Gamma-Gamma -------------------------------------------------------------------------------
def gamma_gamma_spend(n_spend: FloatArray, total: FloatArray, *, p: float, q: float,
                      gamma: FloatArray) -> FloatArray:
    """``E[spend per purchase | F] = p (gamma + sum z) / (p n + q - 1)`` (conjugate)."""
    return np.asarray(p * (np.asarray(gamma) + np.asarray(total))
                      / (p * np.asarray(n_spend) + q - 1.0), dtype=np.float64)


# --- correlated spend (Gaussian copula) ---------------------------------------------------------
def _probit_of_gamma(shape: float, rate: FloatArray, v: FloatArray) -> FloatArray:
    """``Phi^-1(F(v))`` for ``Gamma(shape, rate)``, accurate in both tails."""
    z = rate * v
    lower = special.gammainc(shape, z)
    upper = special.gammaincc(shape, z)
    return np.asarray(np.where(lower < 0.5, special.ndtri(lower), -special.ndtri(upper)))


def bgnbd_copula_conditional(x: FloatArray, t_x: FloatArray, age: FloatArray,
                             n_spend: FloatArray, total: FloatArray, edges: FloatArray, *,
                             r: float, alpha: FloatArray, a: float, b: FloatArray,
                             gg_p: float, gg_q: float, gamma: FloatArray, rho: float,
                             nodes: tuple[int, int, int] = COPULA_NODES
                             ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """BG/NBD + Gamma-Gamma with a Gaussian copula (correlation ``rho``) on ``(lambda, nu)``.

    Returns ``P(alive at T)``, ``P(alive at T + e_j)``, expected purchases per interval and the
    expected **spend** per interval ``E[N_j x gg_p / nu | F]``, and the posterior mean spend
    per purchase ``E[gg_p / nu | F]``."""
    x, t_x, age = (np.asarray(v, dtype=np.float64) for v in (x, t_x, age))
    n_spend, total = np.asarray(n_spend, dtype=np.float64), np.asarray(total, dtype=np.float64)
    alpha, b = np.asarray(alpha, dtype=np.float64), np.asarray(b, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    e = np.asarray(edges, dtype=np.float64)
    n_lam, n_p, n_nu = nodes
    cop_a, cop_b = rho / (1.0 - rho * rho), rho * rho / (2.0 * (1.0 - rho * rho))
    log_odds = _log_odds_alive_bgnbd(x, t_x, age, r, alpha, a, b)
    out_alive = np.empty(x.size)
    out_at = np.empty((x.size, e.size))
    out_n = np.empty((x.size, e.size - 1))
    out_v = np.empty((x.size, e.size - 1))
    out_s = np.empty(x.size)
    for (xi, bi, ni), all_rows in _groups(x, b, n_spend):
        k = r + float(xi)
        y_lam, w_lam = log_gamma_rule(k, n_lam)
        p, w_p = beta_rule(a, float(bi) + float(xi), n_p)
        y_nu, w_nu = log_gamma_rule(gg_q + gg_p * float(ni), n_nu)
        for start in range(0, all_rows.size, _CHUNK):
            rows = all_rows[start:start + _CHUNK]
            nu = y_nu[None, :] / (gamma[rows] + total[rows])[:, None]  # (g, n_nu)
            z2 = _probit_of_gamma(gg_q, gamma[rows][:, None], nu)
            spend = gg_p / nu

            def copula(lam: FloatArray, rows: NDArray[np.intp] = rows, z2: FloatArray = z2
                       ) -> FloatArray:
                """``log c`` at ``lam`` (g, L) x the nu nodes: (g, L, n_nu)."""
                z1 = _probit_of_gamma(r, alpha[rows][:, None], lam)
                return np.asarray(cop_a * z1[:, :, None] * z2[:, None, :]
                                  - cop_b * (z1[:, :, None] ** 2 + z2[:, None, :] ** 2))

            lam_alive = y_lam[None, :] / (alpha[rows] + age[rows])[:, None]
            lc_alive = copula(lam_alive)
            lam_dead = y_lam[None, :] / (alpha[rows] + t_x[rows])[:, None]
            # log mixture weights relative to the alive component (dead: log_odds; -inf if x=0)
            ld = log_odds[rows][:, None, None] + copula(lam_dead)
            shift = np.maximum(lc_alive.max(axis=(1, 2)), ld.max(axis=(1, 2)))
            wa = np.exp(lc_alive - shift[:, None, None]) * w_lam[None, :, None] * w_nu
            wd = np.exp(ld - shift[:, None, None]) * w_lam[None, :, None] * w_nu
            z_alive = wa.sum(axis=(1, 2))
            norm = z_alive + wd.sum(axis=(1, 2))
            # p integrals at each alive lambda node: (g, L, m+1) and (g, L, m)
            decay = np.exp(-lam_alive[:, :, None, None] * p[None, None, :, None]
                           * e[None, None, None, :])  # (g, L, n_p, m+1)
            at_lam = np.einsum("q,glqj->glj", w_p, decay)
            step = decay[..., :-1] * -np.expm1(-lam_alive[:, :, None, None]
                                               * p[None, None, :, None]
                                               * np.diff(e)[None, None, None, :])
            n_lam_j = np.einsum("q,glqj->glj", w_p / p, step)
            wa_lam = wa.sum(axis=2)  # (g, L)
            wa_spend = np.einsum("glv,gv->gl", wa, spend)
            out_alive[rows] = z_alive / norm
            out_at[rows] = np.einsum("gl,glj->gj", wa_lam, at_lam) / norm[:, None]
            out_n[rows] = np.einsum("gl,glj->gj", wa_lam, n_lam_j) / norm[:, None]
            out_v[rows] = np.einsum("gl,glj->gj", wa_spend, n_lam_j) / norm[:, None]
            out_s[rows] = ((wa + wd) * spend[:, None, :]).sum(axis=(1, 2)) / norm
    return out_alive, out_at, out_n, out_v, out_s


# --- scenario-level oracle -----------------------------------------------------------------------
@dataclass(frozen=True)
class PurchaseHistory:
    """The oracle's information set at ``cutoff`` (per customer, indexed by ``customer_id``):
    ``segment, channel, x, t_x, age`` (weeks), ``n_spend, total_spend``."""

    frame: pd.DataFrame
    cutoff: int
    through: int
    edges: FloatArray
    """``e_0 .. e_H``: month ends of ``C .. C+H`` in weeks after ``T`` (common to everyone)."""


def purchase_history(scenario: PurchasesScenario, cutoff: int, horizon: int
                     ) -> PurchaseHistory:
    cfg = scenario.config
    obs = observe(scenario, cutoff)
    through = obs.complete_through()
    cust = obs.customers.set_index("customer_id")
    cust = cust.loc[cust["acquisition_month"] <= through]
    tx = obs.transactions.drop_duplicates("transaction_id")
    tx = tx.loc[(tx["billing_month"] <= through) & tx["customer_id"].isin(cust.index)]
    unit = cfg.time_unit_days * 86400.0
    acq = cust["acquisition_date"]
    t = pd.to_timedelta(tx["billing_date"].to_numpy()
                        - acq.reindex(tx["customer_id"]).to_numpy()).total_seconds() / unit
    tx = tx.assign(t=np.asarray(t))
    rep = tx.loc[tx["t"] > 0].groupby("customer_id")["t"]
    by = tx.groupby("customer_id")["revenue"]
    t_end = month_end(scenario.start, through)
    frame = pd.DataFrame({
        "segment": cust["segment"], "channel": cust["channel"],
        "x": rep.size().reindex(cust.index, fill_value=0).astype(np.float64),
        "t_x": rep.max().reindex(cust.index, fill_value=0.0),
        "age": (t_end - acq).dt.total_seconds() / unit,
        "n_spend": by.size().reindex(cust.index, fill_value=0).astype(np.float64),
        "total_spend": by.sum().reindex(cust.index, fill_value=0.0),
    }, index=cust.index)
    edges = np.array([(month_end(scenario.start, cutoff + j) - t_end).total_seconds() / unit
                      for j in range(horizon + 1)])
    return PurchaseHistory(frame, cutoff, through, edges)


@dataclass(frozen=True)
class PurchaseOracleResult:
    frame: pd.DataFrame
    """Per customer: ``oracle``, ``expected_spend`` (per purchase), ``oracle_discounted``,
    ``expected_purchases``, ``p_alive_observed`` (alive at the end of ``T_C``), ``p_alive``
    (alive at the end of ``C``) and
    ``purchases_1 .. purchases_H`` (expected purchases in month ``C + j``)."""
    history: PurchaseHistory


def purchases_oracle(scenario: PurchasesScenario, cutoff: int, horizon: int, *,
                     monthly_rate: float = 0.0) -> PurchaseOracleResult:
    """Exact (quadrature) observable-information oracle for the purchases scenario and its
    variants, chosen from the scenario's own generating process."""
    cfg = scenario.config
    hist = purchase_history(scenario, cutoff, horizon)
    f = hist.frame
    seg = np.array([SEGMENTS.index(s) for s in f["segment"]], dtype=np.int64)
    ch = np.array([CHANNELS.index(c) for c in f["channel"]], dtype=np.int64)
    alpha = cfg.alpha * np.asarray(cfg.segment_alpha_scale)[seg]
    gamma = np.asarray(cfg.gg_gamma, dtype=np.float64)[seg]
    b = np.asarray(cfg.dropout_b, dtype=np.float64)[ch]
    x, t_x, age = f["x"].to_numpy(), f["t_x"].to_numpy(), f["age"].to_numpy()
    n_spend, total = f["n_spend"].to_numpy(), f["total_spend"].to_numpy()
    margin = cfg.gross_margin - cfg.fee_rate
    rho = cfg.spend_frequency_correlation
    if rho != 0.0:
        if cfg.dropout_process != "bgnbd":
            raise NotImplementedError("correlated spend is predeclared with BG/NBD only")
        p_obs, at, n_j, spend_j, spend = bgnbd_copula_conditional(
            x, t_x, age, n_spend, total, hist.edges, r=cfg.r, alpha=alpha, a=cfg.dropout_a,
            b=b, gg_p=cfg.gg_p, gg_q=cfg.gg_q, gamma=gamma, rho=rho)
        value_j = spend_j * margin
    else:
        if cfg.dropout_process == "bgnbd":
            p_obs, at, n_j = bgnbd_conditional(x, t_x, age, hist.edges, r=cfg.r, alpha=alpha,
                                               a=cfg.dropout_a, b=b)
        else:
            p_obs, at, n_j = pareto_conditional(x, t_x, age, hist.edges, r=cfg.r,
                                                alpha=alpha, s=cfg.pareto_s,
                                                beta=cfg.pareto_beta)
        spend = gamma_gamma_spend(n_spend, total, p=cfg.gg_p, q=cfg.gg_q, gamma=gamma)
        value_j = n_j * (spend * margin)[:, None]
    d = (1.0 + monthly_rate) ** -np.arange(1, horizon + 1)
    out = pd.DataFrame({"oracle": value_j.sum(axis=1),
                        "expected_spend": spend,
                        "oracle_discounted": (value_j * d).sum(axis=1),
                        "expected_purchases": n_j.sum(axis=1),
                        "p_alive_observed": p_obs, "p_alive": at[:, 0]}, index=f.index)
    out = out.join(pd.DataFrame(n_j, index=f.index,
                                columns=[f"purchases_{j}" for j in range(1, horizon + 1)]))
    return PurchaseOracleResult(out, hist)
