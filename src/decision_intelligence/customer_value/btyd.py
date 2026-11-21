"""Non-contractual LTV model for Milestone 11 (P5): E4, BG/NBD + Gamma-Gamma.

**Information** (Tier A, non-contractual; as the features and the P2 oracle): for every customer
acquired by ``T_C = complete_through(C)``, the purchases recorded by ``C`` in completely recorded
months ``<= T_C``. Times are in weeks (``NONCONTRACTUAL_TIME_UNIT_DAYS``) from the acquisition
purchase: ``x`` repeat purchases, the last at ``t_x`` (0 if none), observation length ``T`` to the
end of month ``T_C``; spends are every observed purchase, the acquisition purchase included
(``n`` spends, total ``S``, sum of logs ``L``). The target window is months ``C+1 .. C+H``: the
edges ``e_0 < ... < e_H`` are the month ends of ``C .. C+H`` in weeks after ``T``.

**BG/NBD** (Fader, Hardie & Lee 2005): while alive purchases are Poisson(``lambda``),
``lambda ~ Gamma(r, alpha)``; after each repeat purchase the customer drops out with probability
``p ~ Beta(a, b)``. Log-likelihood of ``(x, t_x, T)``:
``log[A1 + [x > 0] A2]`` with
``log A1 = lnB(a, b+x) - lnB(a, b) + lnG(r+x) - lnG(r) + r ln alpha - (r+x) ln(alpha+T)`` and
``log A2 = lnB(a+1, b+x-1) - lnB(a, b) + lnG(r+x) - lnG(r) + r ln alpha - (r+x) ln(alpha+t_x)``.

**Conditional expectations** (derived here, independently of the oracle). With ``k = r + x``:
``P(alive at T | F) = 1 / (1 + [x > 0] a / (b+x-1) ((alpha+T) / (alpha+t_x))^k)``; given
alive, ``lambda ~ Gamma(k, alpha+T)`` and ``p ~ Beta(a, b+x)`` independently. A customer alive at
``T`` with ``(lambda, p)`` makes ``E[N(T, T+e]] = (1 - exp(-lambda p e)) / p`` purchases and is
still alive at ``T+e`` with probability ``exp(-lambda p e)``; integrating ``lambda`` gives
``A(e) = (1 + p e / (alpha+T))^-k``, so
``E[N(T+e_j, T+e_{j+1}] | F] = P(alive) E_p[(A(e_j) - A(e_{j+1})) / p]`` and
``P(alive at T+e | F) = P(alive) E_p[A(e)]``. ``E_p`` is Gauss-Jacobi on the exact ``Beta(a, b+x)``
weight (Golub-Welsch, normalised; ``E4_JACOBI_NODES``); the published hypergeometric closed form
needs ``a > 1`` (it matches when ``a > 1``; tested).

**Gamma-Gamma** (spend): a purchase's spend is ``Gamma(p, nu)``, ``nu ~ Gamma(q, gamma)``,
independent of purchasing. The full likelihood of a customer's ``n`` spends is
``(p-1) L - n lnG(p) + lnG(pn+q) - lnG(q) + q ln gamma - (pn+q) ln(gamma+S)`` (every observed
spend contributes; the published mean-only form drops ``L``, which carries information on
``p``); ``E[spend | F] = p (gamma+S) / (pn+q-1)``.

**Value** of a purchase = its recorded contribution; E4 uses the business's contribution-to-
revenue ratio over completely recorded purchases at ``C`` (in the generator it is a constant
margin, recovered up to cent rounding). ``E[value] = sum_j E[N_j] E[spend] ratio`` (spend is
independent of purchasing under the model).

**Fitting** (``E4_FIT_UNIT``, amendment A2): separate BG/NBD ``(r, alpha, a, b)`` and Gamma-Gamma
``(p, q, gamma)`` fits in each segment x channel cell, by maximum likelihood on the log
parameters (analytic score; observed information by central differences of the score);
``n_starts`` deterministic starts, L-BFGS-B in a box ``|log| <=`` :data:`LOG_BOX`, then Newton
from the best start; fit status by :func:`~.model_selection.classify_fit` with the protocol's
tolerances (events: repeat purchases for BG/NBD, observed spends for Gamma-Gamma). A component
that is not ``ok`` uses the pooled fit of that component, labelled ``fallback`` (the only allowed
fallback). The pooled fit is also reported as the labelled misspecified reference
(``E4_POOLED_REFERENCE``), never a selection candidate.

Estimation code only: observable tables; never truth, oracles or generator parameters (AST
tested).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import linalg, optimize, special

from decision_intelligence.common.validation import FloatArray
from decision_intelligence.customer_value import predictive_protocol as pp
from decision_intelligence.customer_value.contractual import calibration_slope
from decision_intelligence.customer_value.economics import contribution
from decision_intelligence.customer_value.features import NONCONTRACTUAL_TIME_UNIT_DAYS
from decision_intelligence.customer_value.landmarks import training_stack, view_at
from decision_intelligence.customer_value.model_selection import (
    CandidateResult,
    FitDiagnostics,
    classify_fit,
)
from decision_intelligence.customer_value.panel import EconomicsTables, month_end, observe

LOG_BOX = 15.0
E4_JACOBI_NODES = 64
"""Gauss-Jacobi nodes for ``E_p`` (the value pinned for the oracle's identical integral in
``ORACLE_NUMERICS``; doubled-rule convergence is tested)."""
_FD_STEP = 1e-5
_MAX_NEWTON = 100
BGNBD_PARAMS = ("r", "alpha", "a", "b")
GG_PARAMS = ("p", "q", "gamma")


# --- data ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PurchaseData:
    """Per customer (index ``customer_id``): ``segment, channel, x, t_x, T, n, S, L``."""

    frame: pd.DataFrame
    cutoff: int
    through: int
    edges: FloatArray
    contribution_ratio: float
    """Recorded contribution / revenue over completely recorded purchases."""


def purchase_data(tables: EconomicsTables, cutoff: int, horizon: int) -> PurchaseData:
    """E4's sufficient statistics at ``cutoff`` from the tables as recorded by then."""
    obs = observe(tables, cutoff)
    through = obs.complete_through()
    cust = obs.customers.set_index("customer_id")
    cust = cust.loc[cust["acquisition_month"] <= through]
    tx = obs.transactions.drop_duplicates("transaction_id")
    tx = tx.loc[(tx["billing_month"] <= through) & tx["customer_id"].isin(cust.index)]
    unit = NONCONTRACTUAL_TIME_UNIT_DAYS * 86400.0
    acq = cust["acquisition_date"]
    t = pd.to_timedelta(tx["billing_date"].to_numpy()
                        - acq.reindex(tx["customer_id"]).to_numpy()).total_seconds() / unit
    rev = tx["revenue"].to_numpy(dtype=np.float64)
    tx = tx.assign(t=np.asarray(t), logz=np.log(np.maximum(rev, 1e-300)))
    rep = tx.loc[tx["t"] > 0].groupby("customer_id")["t"]
    by = tx.groupby("customer_id")
    t_end = month_end(tables.start, through)
    frame = pd.DataFrame({
        "segment": cust["segment"], "channel": cust["channel"],
        "x": rep.size().reindex(cust.index, fill_value=0).astype(np.float64),
        "t_x": rep.max().reindex(cust.index, fill_value=0.0),
        "T": (t_end - acq).dt.total_seconds() / unit,
        "n": by.size().reindex(cust.index, fill_value=0).astype(np.float64),
        "S": by["revenue"].sum().reindex(cust.index, fill_value=0.0),
        "L": by["logz"].sum().reindex(cust.index, fill_value=0.0),
    }, index=cust.index)
    edges = np.array([(month_end(tables.start, cutoff + j) - t_end).total_seconds() / unit
                      for j in range(horizon + 1)])
    total_rev = float(rev.sum())
    ratio = float(contribution(tx).sum()) / total_rev if total_rev > 0 else math.nan
    return PurchaseData(frame, cutoff, through, np.asarray(edges, dtype=np.float64), ratio)


# --- BG/NBD likelihood --------------------------------------------------------------------------
def bgnbd_loglik(theta: FloatArray, x: FloatArray, t_x: FloatArray, T: FloatArray
                 ) -> tuple[FloatArray, FloatArray]:
    """Per-customer log-likelihood and its gradient in ``log (r, alpha, a, b)``: ``(n, 4)``."""
    r, alpha, a, b = (math.exp(v) for v in theta)
    psi = special.digamma
    common = (special.gammaln(r + x) - special.gammaln(r) + r * math.log(alpha)
              - special.betaln(a, b))
    l1 = common + special.betaln(a, b + x) - (r + x) * np.log(alpha + T)
    has = x > 0
    xm = np.where(has, x, 1.0)
    l2 = np.where(has, common + special.betaln(a + 1, b + xm - 1)
                  - (r + x) * np.log(alpha + t_x), -np.inf)
    ll = np.logaddexp(l1, l2)
    w2 = np.where(has, np.exp(l2 - ll), 0.0)
    w1 = 1.0 - w2
    # d/d r
    dr_c = psi(r + x) - psi(r) + math.log(alpha)
    dr = dr_c - w1 * np.log(alpha + T) - w2 * np.log(alpha + t_x)
    # d/d alpha
    dal = r / alpha - (r + x) * (w1 / (alpha + T) + w2 / (alpha + t_x))
    # d/d a, d/d b of -betaln(a, b)
    dab_a = -(psi(a) - psi(a + b))
    dab_b = -(psi(b) - psi(a + b))
    d1a = psi(a) - psi(a + b + x)
    d1b = psi(b + x) - psi(a + b + x)
    d2a = psi(a + 1) - psi(a + b + xm)
    d2b = psi(b + xm - 1) - psi(a + b + xm)
    da = dab_a + w1 * d1a + w2 * d2a
    db = dab_b + w1 * d1b + w2 * d2b
    grad = np.column_stack([r * dr, alpha * dal, a * da, b * db])
    return np.asarray(ll, dtype=np.float64), np.asarray(grad, dtype=np.float64)


def gg_loglik(theta: FloatArray, n: FloatArray, S: FloatArray, L: FloatArray
              ) -> tuple[FloatArray, FloatArray]:
    """Per-customer Gamma-Gamma log-likelihood (all ``n`` spends) and its gradient in
    ``log (p, q, gamma)``: ``(n, 3)``; customers with ``n = 0`` contribute 0."""
    p, q, g = (math.exp(v) for v in theta)
    psi = special.digamma
    pn = p * n
    ll = ((p - 1.0) * L - n * special.gammaln(p) + special.gammaln(pn + q) - special.gammaln(q)
          + q * math.log(g) - (pn + q) * np.log(g + S))
    dp = L - n * psi(p) + n * psi(pn + q) - n * np.log(g + S)
    dq = psi(pn + q) - psi(q) + math.log(g) - np.log(g + S)
    dg = q / g - (pn + q) / (g + S)
    keep = n > 0
    ll = np.where(keep, ll, 0.0)
    grad = np.column_stack([p * dp, q * dq, g * dg]) * keep[:, None]
    return np.asarray(ll, dtype=np.float64), np.asarray(grad, dtype=np.float64)


# --- generic ML fit with the protocol's status rules ------------------------------------------
LogLik = Callable[[FloatArray], tuple[FloatArray, FloatArray]]


@dataclass(frozen=True)
class ComponentFit:
    component: str
    """``bgnbd`` or ``gg``."""
    names: tuple[str, ...]
    params: tuple[float, ...]
    log_params: FloatArray
    cov_log: FloatArray
    log_likelihood: float
    n_customers: int
    n_events: int
    status: str
    reasons: tuple[str, ...]
    diagnostics: FitDiagnostics
    start_log_likelihoods: tuple[float, ...]

    @property
    def finite(self) -> bool:
        return bool(np.all(np.isfinite(self.log_params)))

    @property
    def se(self) -> tuple[float, ...]:
        return tuple(float(v) for v in np.sqrt(np.diag(self.cov_log)) * np.asarray(self.params))


def _total(loglik: LogLik, theta: FloatArray) -> tuple[float, FloatArray]:
    ll, g = loglik(theta)
    return float(np.sum(ll)), np.asarray(g.sum(axis=0), dtype=np.float64)


def _information(loglik: LogLik, theta: FloatArray) -> FloatArray:
    k = theta.size
    h = np.zeros((k, k))
    for j in range(k):
        e = np.zeros(k)
        e[j] = _FD_STEP
        h[:, j] = (_total(loglik, theta + e)[1] - _total(loglik, theta - e)[1]) / (2 * _FD_STEP)
    return np.asarray(-(h + h.T) / 2, dtype=np.float64)


def _projected(theta: FloatArray, g: FloatArray) -> FloatArray:
    out = g.copy()
    out[(theta >= LOG_BOX - 1e-9) & (out > 0)] = 0.0
    out[(theta <= -LOG_BOX + 1e-9) & (out < 0)] = 0.0
    return out


def _optimise(loglik: LogLik, start: FloatArray, n: int
              ) -> tuple[FloatArray, float, float, float]:
    def neg(th: FloatArray) -> tuple[float, FloatArray]:
        ll, g = _total(loglik, th)
        if not math.isfinite(ll):
            return 1e300, np.zeros_like(th)
        return -ll / n, -g / n

    res = optimize.minimize(neg, start, jac=True, method="L-BFGS-B",
                            bounds=[(-LOG_BOX, LOG_BOX)] * start.size,
                            options={"maxiter": 3000, "ftol": 1e-15, "gtol": 1e-12})
    theta = np.clip(np.asarray(res.x, dtype=np.float64), -LOG_BOX, LOG_BOX)
    ll, g = _total(loglik, theta)
    rel = math.inf
    for _ in range(_MAX_NEWTON):
        if float(np.max(np.abs(_projected(theta, g)))) / n <= \
                pp.STATUS_RULES.gradient_tolerance * 1e-3:
            rel = 0.0
            break
        info = _information(loglik, theta)
        try:
            ok = bool(np.all(np.linalg.eigvalsh(info) > 0))
        except np.linalg.LinAlgError:
            ok = False
        if not ok:
            break
        step = np.linalg.solve(info, g)
        moved = False
        for _half in range(40):
            cand = np.clip(theta + step, -LOG_BOX, LOG_BOX)
            ll_c, g_c = _total(loglik, cand)
            if math.isfinite(ll_c) and ll_c >= ll - 1e-12 * max(1.0, abs(ll)):
                rel = abs(ll_c - ll) / max(1.0, abs(ll))
                theta, ll, g = cand, ll_c, g_c
                moved = True
                break
            step = step / 2
        if not moved or rel <= 1e-15:
            break
    if not math.isfinite(rel):
        rel = 0.0 if res.success else math.inf
    return theta, ll, float(np.max(np.abs(_projected(theta, g)))) / n, rel


def _fit(component: str, names: tuple[str, ...], loglik: LogLik, starts: list[FloatArray],
         n_customers: int, n_events: int) -> ComponentFit:
    rules = pp.STATUS_RULES
    k = len(names)
    if n_customers == 0 or n_events == 0:
        d = FitDiagnostics(has_data=False, n_events=n_events, n_parameters=k)
        status, reasons = classify_fit(d)
        return ComponentFit(component, names, tuple([math.nan] * k), np.full(k, np.nan),
                            np.full((k, k), np.nan), math.nan, n_customers, n_events, status,
                            reasons, d, ())
    runs = [_optimise(loglik, np.clip(s, -LOG_BOX, LOG_BOX), n_customers) for s in starts]
    lls = sorted((r[1] for r in runs), reverse=True)
    theta, ll, pg, rel = max(runs, key=lambda r: r[1])
    info = _information(loglik, theta)
    if np.all(np.isfinite(info)):
        eig = np.linalg.eigvalsh(info)
        pd_ok = bool(float(np.min(eig)) > 1e-12 * max(float(np.max(np.abs(eig))), 1e-300))
    else:
        pd_ok = False
    cov = np.linalg.inv(info) if pd_ok else np.full((k, k), np.nan)
    d = FitDiagnostics(
        has_data=True,
        converged=pg <= rules.gradient_tolerance and rel <= rules.relative_change_tolerance,
        finite_likelihood=math.isfinite(ll) and bool(np.all(np.isfinite(theta))),
        hessian_positive_definite=pd_ok, max_abs_log_parameter=float(np.max(np.abs(theta))),
        start_log_likelihood_gap=float(lls[0] - lls[1]) if len(lls) > 1 else 0.0,
        n_events=n_events, n_parameters=k)
    status, reasons = classify_fit(d)
    return ComponentFit(component, names, tuple(float(math.exp(v)) for v in theta), theta,
                        np.asarray(cov, dtype=np.float64), float(ll), n_customers, n_events,
                        status, reasons, d, tuple(float(r[1]) for r in runs))


def fit_bgnbd(x: FloatArray, t_x: FloatArray, T: FloatArray) -> ComponentFit:
    x, t_x, T = (np.asarray(v, dtype=np.float64) for v in (x, t_x, T))
    rate = float(x.sum() / max(T.sum(), 1e-12)) if x.size else 0.1
    rate = min(max(rate, 1e-3), 10.0)
    starts = [np.log([r0, r0 / rate, a0, b0]) for r0, a0, b0 in
              ((0.5, 1.0, 3.0), (1.0, 0.5, 2.0), (2.0, 2.0, 8.0), (0.25, 1.0, 1.0),
               (1.0, 4.0, 20.0))]
    return _fit("bgnbd", BGNBD_PARAMS, lambda th: bgnbd_loglik(th, x, t_x, T), starts,
                int(x.size), int(x.sum()))


def fit_gamma_gamma(n: FloatArray, S: FloatArray, L: FloatArray) -> ComponentFit:
    n, S, L = (np.asarray(v, dtype=np.float64) for v in (n, S, L))
    keep = n > 0
    n, S, L = n[keep], S[keep], L[keep]
    mean = float(S.sum() / max(n.sum(), 1.0)) if n.size else 1.0
    starts = []
    for p0, q0 in ((2.0, 3.0), (6.0, 4.0), (1.0, 2.0), (10.0, 10.0), (4.0, 1.5)):
        g0 = mean * (q0 - 1.0) / p0 if q0 > 1 else mean / p0
        starts.append(np.log([p0, q0, max(g0, 1e-3)]))
    return _fit("gg", GG_PARAMS, lambda th: gg_loglik(th, n, S, L), starts, int(n.size),
                int(n.sum()))


# --- Gauss-Jacobi for Beta weights (Golub-Welsch) ----------------------------------------------
def beta_jacobi(a: FloatArray, b: FloatArray, n: int) -> tuple[FloatArray, FloatArray]:
    """Jacobi matrix on [0, 1], directly in the positive Beta shapes (batched).

    Avoid forming a-1/b-1 and then recovering a+b by cancellation. The first off-diagonal
    analytically cancels the removable (a+b-1) factor; the remaining factors are positive.
    Factored ratios also avoid squaring tiny shapes. No shape or node count is changed.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    total = a + b
    if n < 1 or a.ndim != 1 or a.shape != b.shape or not np.all(
            np.isfinite(total) & (a > 0.0) & (b > 0.0)):
        raise ValueError("Beta quadrature needs positive finite shapes and sum, and n >= 1")
    diag = np.empty((a.size, n))
    diag[:, 0] = a / total
    j = np.arange(1, n, dtype=np.float64)[None, :]
    aa, bb, tt = a[:, None], b[:, None], total[:, None]
    diag[:, 1:] = ((j + aa) / (2 * j + tt) * ((j - 1) + tt) / ((2 * j - 1) + tt)
                   + j / ((2 * j - 1) + tt) * ((j - 1) + bb) / ((2 * j - 2) + tt))
    off = np.empty((a.size, n - 1))
    if n > 1:
        off[:, 0] = np.sqrt(a / total) * np.sqrt(b / total) / np.sqrt(total + 1)
        j = j[:, 1:]
        # sqrt each ratio before multiplying: preserve subnormal positive shapes.
        off[:, 1:] = (np.sqrt(j / ((2 * j - 2) + tt))
                      * np.sqrt(((j - 1) + aa) / ((2 * j - 2) + tt))
                      * np.sqrt(((j - 1) + bb) / ((2 * j - 1) + tt))
                      * np.sqrt(((j - 2) + tt) / ((2 * j - 3) + tt)))
    return diag, off


@lru_cache(maxsize=8192)
def beta_nodes(a: float, b: float, n: int = E4_JACOBI_NODES) -> tuple[FloatArray, FloatArray]:
    """Nodes on ``[0, 1]`` and weights with ``sum w f(p) ~= E[f(P)]``, ``P ~ Beta(a, b)``:
    the Jacobi weight ``(1-y)^(b-1) (1+y)^(a-1)`` on ``[-1, 1]``, ``p = (1+y) / 2``."""
    reflected = a > b
    lo, hi = (b, a) if reflected else (a, b)
    diag, off = beta_jacobi(np.array([lo]), np.array([hi]), n)
    # STEV retains small first-row eigenvector components (hence small boundary weights).
    pn, v = linalg.eigh_tridiagonal(diag[0], off[0], lapack_driver="stev")
    w = np.asarray(v[0, :], dtype=np.float64) ** 2
    # A symmetric eigensolver can place an endpoint a few ulps outside its support.
    pn, w = np.clip(pn, 0.0, 1.0), w / w.sum()
    if reflected:
        pn, w = 1.0 - pn[::-1], w[::-1]
    return np.asarray(pn, dtype=np.float64), np.asarray(w, dtype=np.float64)


def purchase_window(p: FloatArray, k: FloatArray, rate: FloatArray, left: FloatArray,
                    right: FloatArray) -> FloatArray:
    """(A(left)-A(right))/p, including its continuous value k*(right-left)/rate at p=0.

    Factor log1p(z)/z and (1-exp(-u))/u instead of dividing a cancelled difference by p.
    This also handles positive p so small that z underflows. Inputs follow NumPy broadcasting.
    """
    start_rate = rate + p * left
    z = p * (right - left) / start_rate
    logz = np.log1p(z)
    log_ratio = np.divide(logz, z, out=np.ones_like(z), where=z != 0.0)
    u = k * logz
    return np.asarray(np.exp(-k * np.log1p(p * left / rate)) * special.exprel(-u)
                      * log_ratio * k * ((right - left) / start_rate), dtype=np.float64)


# --- conditional expectations ---------------------------------------------------------------------
@dataclass(frozen=True)
class Conditional:
    p_alive_T: FloatArray
    p_alive_end: FloatArray
    """P(alive at the end of month ``C``) = at ``e_0``."""
    purchases: FloatArray
    """``(n, H)``: expected purchases in month ``C + j``."""


def bgnbd_alive(k: FloatArray, a: FloatArray | float, b: FloatArray | float,
                observed_rate: FloatArray, last_purchase_rate: FloatArray,
                x: FloatArray) -> FloatArray:
    """Compute BG/NBD P(alive at T) from log-odds, preserving representable subnormals."""
    denom = b + np.where(x > 0, x - 1.0, 0.0)
    # The mantissas have a safe ratio even when the natural-scale ratio overflows,
    # or when two nearly equal logarithms would lose precision for tiny shapes.
    a_mantissa, a_exponent = np.frexp(a)
    b_mantissa, b_exponent = np.frexp(denom)
    log_shape_ratio = (np.log(a_mantissa / b_mantissa)
                       + (a_exponent - b_exponent) * math.log(2.0))
    log_odds = (log_shape_ratio
                + k * (np.log(observed_rate) - np.log(last_purchase_rate)))
    return np.asarray(np.exp(-np.logaddexp(0.0, np.where(x > 0, log_odds, -np.inf))),
                      dtype=np.float64)


def bgnbd_conditional(params: tuple[float, ...], x: FloatArray, t_x: FloatArray, T: FloatArray,
                      edges: FloatArray, *, nodes: int = E4_JACOBI_NODES) -> Conditional:
    r, alpha, a, b = params
    x, t_x, T = (np.asarray(v, dtype=np.float64) for v in (x, t_x, T))
    k = r + x
    p_alive = bgnbd_alive(k, a, b, alpha + T, alpha + t_x, x)
    m = edges.size
    a_e = np.zeros((x.size, m))
    purchases = np.zeros((x.size, m - 1))
    for xv in np.unique(x):
        sel = x == xv
        pn, w = beta_nodes(float(a), float(b + xv), nodes)
        rate = (alpha + T[sel])[:, None, None]
        power = k[sel][:, None, None]
        p = pn[None, :, None]
        A = np.exp(-power * np.log1p(p * edges[None, None, :] / rate))
        a_e[sel] = np.einsum("j,cjm->cm", w, A)
        step = purchase_window(p, power, rate, edges[None, None, :-1], edges[None, None, 1:])
        purchases[sel] = p_alive[sel, None] * np.einsum("j,cjm->cm", w, step)
    return Conditional(np.asarray(p_alive, dtype=np.float64),
                       np.asarray(p_alive * a_e[:, 0], dtype=np.float64),
                       np.asarray(purchases, dtype=np.float64))


def gg_expected_spend(params: tuple[float, ...], n: FloatArray, S: FloatArray) -> FloatArray:
    p, q, g = params
    return np.asarray(p * (g + np.asarray(S)) / (p * np.asarray(n) + q - 1.0), dtype=np.float64)


# --- cells, fallback, prediction ---------------------------------------------------------------
@dataclass(frozen=True)
class E4Fits:
    bgnbd: dict[tuple[str, str], ComponentFit]
    gg: dict[tuple[str, str], ComponentFit]
    pooled_bgnbd: ComponentFit
    pooled_gg: ComponentFit
    used: dict[tuple[str, str], tuple[str, ComponentFit, str, ComponentFit]] = field(
        default_factory=dict)
    """Per cell: (BG/NBD status, fit used, Gamma-Gamma status, fit used)."""

    @property
    def ok_cell_fraction(self) -> float:
        """Cells whose two components are both ``ok`` or ``fallback``."""
        if not self.used:
            return math.nan
        good = ("ok", "fallback")
        return sum(s1 in good and s2 in good for s1, _, s2, _ in self.used.values()) / len(
            self.used)

    def status_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s1, _, s2, _ in self.used.values():
            for key in (f"bgnbd {s1}", f"gg {s2}"):
                out[key] = out.get(key, 0) + 1
        return out


def _resolve(own: ComponentFit, pooled: ComponentFit) -> tuple[str, ComponentFit]:
    if own.status == "ok":
        return "ok", own
    if pooled.status == "ok":
        return "fallback", pooled
    return own.status, own


def fit_cells(frame: pd.DataFrame) -> E4Fits:
    f = frame
    pooled_b = fit_bgnbd(f["x"].to_numpy(), f["t_x"].to_numpy(), f["T"].to_numpy())
    pooled_g = fit_gamma_gamma(f["n"].to_numpy(), f["S"].to_numpy(), f["L"].to_numpy())
    b_fits: dict[tuple[str, str], ComponentFit] = {}
    g_fits: dict[tuple[str, str], ComponentFit] = {}
    used: dict[tuple[str, str], tuple[str, ComponentFit, str, ComponentFit]] = {}
    for (s, c), cell in f.groupby(["segment", "channel"], sort=True):
        key = (str(s), str(c))
        bf = fit_bgnbd(cell["x"].to_numpy(), cell["t_x"].to_numpy(), cell["T"].to_numpy())
        gf = fit_gamma_gamma(cell["n"].to_numpy(), cell["S"].to_numpy(), cell["L"].to_numpy())
        b_fits[key], g_fits[key] = bf, gf
        used[key] = (*_resolve(bf, pooled_b), *_resolve(gf, pooled_g))
    return E4Fits(b_fits, g_fits, pooled_b, pooled_g, used)


@dataclass(frozen=True)
class E4Prediction:
    cutoff: int
    frame: pd.DataFrame
    """Per customer: ``segment, channel, bgnbd_status, gg_status, prediction,
    prediction_discounted, expected_purchases, expected_spend, p_alive_T, p_alive,
    purchases_1 .. purchases_H``."""
    fits: E4Fits
    data: PurchaseData


def _value(frame: pd.DataFrame, data: PurchaseData, bg: tuple[float, ...],
           gg: tuple[float, ...], rows: np.ndarray) -> tuple[Conditional, FloatArray]:
    sub = frame.iloc[rows]
    cond = bgnbd_conditional(bg, sub["x"].to_numpy(), sub["t_x"].to_numpy(),
                             sub["T"].to_numpy(), data.edges)
    spend = gg_expected_spend(gg, sub["n"].to_numpy(), sub["S"].to_numpy())
    return cond, spend


def predict_from(data: PurchaseData, fits: E4Fits, *, pooled: bool = False,
                 monthly_rate: float = pp.MONTHLY_DISCOUNT_RATE) -> E4Prediction:
    """Predictions with the per-cell fits (``pooled=True``: the pooled misspecified
    reference for every customer)."""
    f = data.frame
    h = data.edges.size - 1
    purchases = np.full((len(f), h), np.nan)
    pa_t, pa, spend = (np.full(len(f), np.nan) for _ in range(3))
    s1 = np.empty(len(f), dtype=object)
    s2 = np.empty(len(f), dtype=object)
    seg, ch = f["segment"].to_numpy(), f["channel"].to_numpy()
    reference = ("reference", fits.pooled_bgnbd, "reference", fits.pooled_gg)
    cells = ([(("pooled", "pooled"), reference, np.ones(len(f), dtype=bool))] if pooled
             else [(k, v, (seg == k[0]) & (ch == k[1])) for k, v in fits.used.items()])
    for _key, (st1, bf, st2, gf), sel in cells:
        rows = np.flatnonzero(sel)
        if rows.size == 0:
            continue
        s1[rows], s2[rows] = st1, st2
        if not (bf.finite and gf.finite):
            continue
        cond, sp = _value(f, data, bf.params, gf.params, rows)
        purchases[rows], pa_t[rows], pa[rows] = cond.purchases, cond.p_alive_T, \
            cond.p_alive_end
        spend[rows] = sp
    d = (1.0 + monthly_rate) ** -np.arange(1, h + 1)
    unit_value = spend * data.contribution_ratio
    out = pd.DataFrame({
        "segment": seg, "channel": ch, "bgnbd_status": s1, "gg_status": s2,
        "prediction": purchases.sum(axis=1) * unit_value,
        "prediction_discounted": (purchases * d).sum(axis=1) * unit_value,
        "expected_purchases": purchases.sum(axis=1), "expected_spend": spend,
        "p_alive_T": pa_t, "p_alive": pa}, index=f.index)
    out = out.join(pd.DataFrame(purchases, index=f.index,
                                columns=[f"purchases_{j}" for j in range(1, h + 1)]))
    return E4Prediction(data.cutoff, out, fits, data)


def predict(tables: EconomicsTables, cutoff: int, horizon: int, *, pooled: bool = False,
            monthly_rate: float = pp.MONTHLY_DISCOUNT_RATE) -> E4Prediction:
    """E4 fitted on the information recorded by ``cutoff``; months ``cutoff+1 ..
    cutoff+horizon``."""
    data = purchase_data(tables, cutoff, horizon)
    return predict_from(data, fit_cells(data.frame), pooled=pooled, monthly_rate=monthly_rate)


# --- temporal validation (selection inputs; never the holdout) ----------------------------------
@dataclass(frozen=True)
class E4Validation:
    cutoff: int
    landmark: int
    frame: pd.DataFrame
    candidate: CandidateResult
    fits: E4Fits


def validate(tables: EconomicsTables, cutoff: int, horizon: int) -> E4Validation:
    """Fit with the information of the validation landmark ``L_v`` only (tables as recorded by
    ``cutoff``, then observed at ``L_v``); score on ``L_v``'s completely recorded labels."""
    lv = pp.validation_landmark(cutoff, horizon)
    if lv is None:
        raise ValueError(f"no validation landmark at cutoff {cutoff} (horizon {horizon})")
    pred = predict(view_at(tables, cutoff), lv, horizon)
    stack = training_stack(tables, cutoff, horizon, "non_contractual", "A",
                           landmarks=[lv]).set_index("customer_id")
    if not stack.index.equals(pred.frame.index):
        raise RuntimeError("validation population differs from the prediction population")
    frame = pd.DataFrame({"prediction": pred.frame["prediction"],
                          "realised": stack["realised"]})
    p = frame["prediction"].to_numpy(np.float64)
    r = frame["realised"].to_numpy(np.float64)
    ok = np.isfinite(p)
    cand = CandidateResult(
        "E4", pred.fits.ok_cell_fraction,
        float(np.sum(p[ok])) / float(np.sum(r[ok])) - 1.0 if ok.any() else math.nan,
        calibration_slope(p[ok], r[ok]) if ok.any() else math.nan,
        float(np.mean((r - p) ** 2)) if ok.all() else math.nan)
    return E4Validation(cutoff, lv, frame, cand, pred.fits)
