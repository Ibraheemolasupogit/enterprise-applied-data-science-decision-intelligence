"""Milestone 11 P7: evaluation, uncertainty and selection reporting.

Metric primitives live in :mod:`decision_intelligence.statistics.predictive_metrics`; this
module applies them to the M11 models. Nothing here reads truth, oracles or generator
parameters (AST tested): comparisons with the oracle happen in the evidence script.

**Predictive count distributions.** Contractual (E1, E2, E3): paid months in the window are
contiguous, so ``P(N = k) = S(k) - S(k + 1)`` from the models' ``paid_j = S(j)``. E4: the exact
conditional distribution of window purchases. Given alive at ``T`` (probability
``P(alive at T)``), ``lambda ~ Gamma(r + x, alpha + T)`` and ``p ~ Beta(a, b + x)``
independently; alive at the window start ``e_0`` with probability ``c(p) = ((alpha + T) /
(alpha + T + p e_0))^(r + x)``, after which ``lambda ~ Gamma(r + x, alpha + T + p e_0)``; with
window length ``t = e_H - e_0`` the raw Poisson count is negative binomial ``NB(r + x, q)``,
``q = (alpha + T + p e_0) / (alpha + T + p e_0 + t)``, and the customer drops out after the
``K``-th purchase, ``K ~ Geometric(p)`` on ``{1, 2, ...}``, so ``N = min(NB, K)``:
``P(N = k | p) = NB(k) (1 - p)^(k - 1) + p (1 - p)^(k - 1) P(NB > k)`` (``k >= 1``),
``P(N = 0 | p) = NB(0)``. ``p`` is integrated by Gauss-Jacobi (E4's nodes).

**Parameter uncertainty** (Hessian-based, parametric simulation). E1/E2 and E4: per cell, log
parameters drawn from ``N(theta_hat, inverse observed information)``; a verified geometric
boundary fit (A6/A7) or a fit without a positive-definite information matrix is held fixed and
counted. E3 and E5: the CR1 customer-clustered sandwich (``CLUSTERED_INFERENCE.sandwich``) of
the penalised coefficients; E3's coefficients are drawn from ``N(beta_hat, V_CR1)``. Each draw
gives the customer's expected window value (parameter interval). **Predictive intervals**
(parameter + outcome uncertainty) add one simulated outcome per draw: contractual paid months
from the drawn ``S`` (pre-A11 construction: value = the contribution model's expected
contribution of each simulated paid month; with ``CONTRACTUAL_OUTCOME_NOISE``, amendment A11,
each simulated paid month also gets an independent residual drawn from the completely
recorded pre-cutoff months of the source group the contribution model used for its mean, from a
separate random stream so that the pre-A11 draws are unchanged), E4 purchases and spends from
the BG/NBD and Gamma-Gamma posteriors. Intervals are equal-tailed percentiles of the draws.

**Selection reporting** (``SELECTION``, A9, A10): candidates must be selection candidates of the
scenario's kind; a Tier A model enters a Tier B selection (``TIER_A_IN_TIER_B``); a model that
is not selection-eligible is excluded with its reason; a model with an undefined prediction
cannot qualify (``B1_DEFINITION``; applied to every model alike). Paired intervals are the
customer-clustered bootstrap of the validation MSE difference
(``CLUSTERED_INFERENCE.metric_bootstrap_resamples`` resamples, level
``SELECTION.paired_interval_level``). The frozen :func:`~.model_selection.select_model`
decides; the reference fallback is shown only within ``REFERENCE_FALLBACK_SCOPE``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import cast

import numpy as np
import pandas as pd
from scipy import special

from decision_intelligence.common.validation import FloatArray
from decision_intelligence.customer_value import predictive_protocol as pp
from decision_intelligence.customer_value.btyd import E4Prediction, beta_nodes, bgnbd_conditional
from decision_intelligence.customer_value.contractual import (
    ContractualPrediction,
    ContributionModel,
    conditional_paid,
)
from decision_intelligence.customer_value.features import contractual_features
from decision_intelligence.customer_value.landmarks import training_stack, view_at
from decision_intelligence.customer_value.model_selection import (
    CandidateResult,
    GateResult,
    gate,
    select_model,
)
from decision_intelligence.customer_value.panel import EconomicsTables, observe
from decision_intelligence.customer_value.precision import classify_coverage, coverage_mc_se
from decision_intelligence.customer_value.predictive import E5Model
from decision_intelligence.customer_value.survival import (
    E3Model,
    HazardFit,
    band_basis,
    person_months,
    tenure_band,
    window_paid,
)
from decision_intelligence.statistics import predictive_metrics as pm
from decision_intelligence.statistics.gamma_regression import GammaRegressionResult
from decision_intelligence.statistics.penalised_regression import expit


# --- predictive count distributions ---------------------------------------------------------------
def contractual_pmf(frame: pd.DataFrame, horizon: int = pp.PRIMARY_HORIZON) -> FloatArray:
    """``P(N = k)``, ``k = 0 .. H``, from a contractual prediction frame's ``paid_1 .. paid_H``."""
    return pm.paid_months_pmf(frame[[f"paid_{j}" for j in range(1, horizon + 1)]].to_numpy())


def e4_count_pmf(pred: E4Prediction, k_max: int) -> FloatArray:
    """Exact ``P(N = k)``, ``k = 0 .. k_max``, of window purchases under E4's cell fits; the mass
    above ``k_max`` is ``1 - row sum``."""
    data = pred.data
    f = data.frame
    e0, t = float(data.edges[0]), float(data.edges[-1] - data.edges[0])
    out = np.full((len(f), k_max + 1), np.nan)
    seg, ch = f["segment"].to_numpy(), f["channel"].to_numpy()
    ks = np.arange(k_max + 1, dtype=np.float64)
    for key, (_, bf, _, _) in pred.fits.used.items():
        sel = np.flatnonzero((seg == key[0]) & (ch == key[1]))
        if sel.size == 0 or not bf.finite:
            continue
        r, alpha, a, b = bf.params
        x = f["x"].to_numpy()[sel]
        T = f["T"].to_numpy()[sel]
        p_alive = bgnbd_conditional(bf.params, x, f["t_x"].to_numpy()[sel], T,
                                    data.edges).p_alive_T
        for xv in np.unique(x):
            for rows in np.array_split(np.flatnonzero(x == xv),
                                       max(1, int(np.sum(x == xv)) // 256)):
                k = r + xv
                beta = alpha + T[rows]
                nodes, w = beta_nodes(float(a), float(b + xv))
                bp = beta[:, None] + nodes[None, :] * e0  # (c, J)
                c_alive = (beta[:, None] / bp) ** k
                q = bp / (bp + t)
                log_nb = ((special.gammaln(ks + k) - special.gammaln(k)
                           - special.gammaln(ks + 1.0))[None, None, :]
                          + k * np.log(q)[:, :, None]
                          + ks[None, None, :] * np.log1p(-q)[:, :, None])
                nb = np.exp(log_nb)  # (c, J, K): P(NB = k), NB(r + x, q)
                nb_sf = np.clip(1.0 - np.cumsum(nb, axis=2), 0.0, None)  # P(NB > k)
                pn = nodes[None, :, None]
                surv = (1.0 - pn) ** np.maximum(ks - 1.0, 0.0)[None, None, :]
                cond = nb * surv + np.where(ks[None, None, :] >= 1, pn * surv * nb_sf, 0.0)
                mix = np.einsum("j,cj,cjk->ck", w, c_alive, cond)
                mix[:, 0] += np.einsum("j,cj->c", w, 1.0 - c_alive)
                pa = p_alive[rows][:, None]
                dist = pa * mix
                dist[:, 0] += 1.0 - pa[:, 0]
                out[sel[rows]] = dist
    return np.asarray(out, dtype=np.float64)


# --- parameter uncertainty and predictive intervals ---------------------------------------------
def _draw_log_params(log_params: FloatArray, cov: FloatArray, n: int,
                     rng: np.random.Generator) -> tuple[FloatArray, bool]:
    cov = np.asarray(cov, dtype=np.float64)
    if not np.all(np.isfinite(cov)) or not np.all(np.isfinite(log_params)):
        return np.repeat(np.asarray(log_params, np.float64)[None, :], n, axis=0), False
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        return np.repeat(np.asarray(log_params, np.float64)[None, :], n, axis=0), False
    z = rng.standard_normal((n, log_params.size))
    return np.asarray(log_params[None, :] + z @ chol.T, dtype=np.float64), True


@dataclass(frozen=True)
class SimulatedValue:
    expected: FloatArray
    """``(n_customers, n_draws)``: expected window value under each parameter draw."""
    outcome: FloatArray
    """``(n_customers, n_draws)``: one simulated window value per draw (parameter + outcome;
    for contractual models the pre-A11 construction, without contribution noise)."""
    index: pd.Index
    fixed_cells: tuple[tuple[str, str], ...]
    """Cells held at their point estimate (boundary fit or no positive-definite information)."""
    outcome_noise: FloatArray | None = None
    """Contractual only, A11: the same simulated paid months plus contribution residuals."""

    def interval(self, level: float, *, outcome: bool, noise: bool = False
                 ) -> tuple[FloatArray, FloatArray]:
        if noise and self.outcome_noise is None:
            raise ValueError("no A11 outcome noise was simulated")
        draws = (self.outcome_noise if noise else self.outcome) if outcome else self.expected
        assert draws is not None
        a = (1.0 - level) / 2.0
        lo, hi = np.quantile(draws, [a, 1.0 - a], axis=1)
        return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)


def _paid_outcome(paid: FloatArray, m: FloatArray, u: FloatArray) -> FloatArray:
    """Simulated value: ``N`` from ``paid`` by inversion with uniforms ``u``; value = sum of
    ``m`` over the first ``N`` months."""
    n_paid = np.sum(paid > u[:, None], axis=1)  # P(N >= j) = paid_j, contiguous
    cum = np.column_stack([np.zeros(m.shape[0]), np.cumsum(m, axis=1)])
    return np.asarray(cum[np.arange(m.shape[0]), n_paid], dtype=np.float64)


@dataclass(frozen=True)
class ResidualPool:
    """A11 (``CONTRACTUAL_OUTCOME_NOISE``): per ``(segment, channel, promo)`` row of the
    contribution model, the residuals (recorded contribution minus that row's mean) of the
    completely recorded months of the source group the row's mean came from."""

    pools: dict[tuple[str, str, bool], FloatArray]
    sources: dict[tuple[str, str, bool], str]
    model: ContributionModel

    def draw(self, segment: np.ndarray, channel: np.ndarray, ages: np.ndarray,
             rng: np.random.Generator) -> FloatArray:
        """Independent residuals, shape of ``ages`` (NaN where the model has no mean)."""
        policy = self.model.policy
        promo = (ages < policy.months) & np.isin(channel, policy.channels)[:, None]
        out = np.full(ages.shape, np.nan)
        for key, pool in self.pools.items():
            seg, ch, pr = key
            mask = ((segment == seg) & (channel == ch))[:, None] & (promo == pr)
            k = int(mask.sum())
            if k and pool.size:
                out[mask] = pool[rng.integers(0, pool.size, size=k)]
        return np.asarray(out, dtype=np.float64)


def contribution_residuals(tables: EconomicsTables, cutoff: int, model: ContributionModel
                           ) -> ResidualPool:
    """Residual pools from the months ``fit_contribution`` used (``billing_month <=
    complete_through(cutoff)``), grouped exactly as each row's ``source``."""
    obs = observe(tables, cutoff)
    tx = obs.transactions.loc[obs.transactions["billing_month"] <= obs.complete_through()]
    cust = obs.customers.set_index("customer_id")
    seg = cust["segment"].reindex(tx["customer_id"]).to_numpy()
    ch = cust["channel"].reindex(tx["customer_id"]).to_numpy()
    promo = (tx["age"].to_numpy() < model.policy.months) & np.isin(ch, model.policy.channels)
    c = tx["contribution"].to_numpy(np.float64)
    pools: dict[tuple[str, str, bool], FloatArray] = {}
    sources: dict[tuple[str, str, bool], str] = {}
    for key, row in model.table.iterrows():
        s, chn, pr = cast(tuple[str, str, bool], key)
        source = str(row["source"])
        if source == "cell":
            sel = (seg == s) & (ch == chn) & (promo == pr)
        elif source == "segment":
            sel = (seg == s) & (promo == pr)
        elif source == "pooled":
            sel = promo == pr
        else:
            sel = np.zeros(c.size, dtype=bool)
        pools[(s, chn, bool(pr))] = np.asarray(c[sel] - float(row["mean"]), dtype=np.float64)
        sources[(s, chn, bool(pr))] = source
    return ResidualPool(pools, sources, model)


def _paid_outcome_noise(paid: FloatArray, m: FloatArray, u: FloatArray, eps: FloatArray
                        ) -> FloatArray:
    """A11: as :func:`_paid_outcome` with ``m + eps`` for the simulated paid months only."""
    n_paid = np.sum(paid > u[:, None], axis=1)
    cum = np.column_stack([np.zeros(m.shape[0]), np.cumsum(m + np.nan_to_num(eps), axis=1)])
    return np.asarray(cum[np.arange(m.shape[0]), n_paid], dtype=np.float64)


def _check_noise(residuals: ResidualPool | None, noise_rng: np.random.Generator | None) -> bool:
    if (residuals is None) != (noise_rng is None):
        raise ValueError("A11 outcome noise needs both residuals and noise_rng")
    return residuals is not None


def simulate_contractual(pred: ContractualPrediction, n_draws: int, rng: np.random.Generator,
                         horizon: int = pp.PRIMARY_HORIZON, *,
                         residuals: ResidualPool | None = None,
                         noise_rng: np.random.Generator | None = None) -> SimulatedValue:
    """E1/E2: Hessian-based parameter draws per cell and simulated paid months. With
    ``residuals`` and ``noise_rng`` (A11) the same draws are also valued with contribution
    residuals; ``rng`` is consumed exactly as without them."""
    noisy = _check_noise(residuals, noise_rng)
    f = pred.frame
    seg, ch = f["segment"].to_numpy(), f["channel"].to_numpy()
    a0 = f["age"].to_numpy().astype(np.float64)
    ages = a0[:, None].astype(np.int64) + np.arange(1, horizon + 1)[None, :]
    m = pred.contribution.expected(seg, ch, ages)
    expected = np.full((len(f), n_draws), np.nan)
    outcome = np.full((len(f), n_draws), np.nan)
    outcome_noise = np.full((len(f), n_draws), np.nan) if noisy else None
    fixed: list[tuple[str, str]] = []
    j = np.arange(1, horizon + 1, dtype=np.float64)
    for key, (_, fit) in pred.fits.used.items():
        sel = np.flatnonzero((seg == key[0]) & (ch == key[1]))
        if sel.size == 0 or not fit.finite:
            continue
        if fit.boundary is not None or fit.geometric_churn is not None:
            draws, drawn = np.repeat(fit.log_params[None, :], n_draws, axis=0), False
        else:
            draws, drawn = _draw_log_params(fit.log_params, fit.cov_log, n_draws, rng)
        if not drawn:
            fixed.append(key)
        for d in range(n_draws):
            if fit.geometric_churn is not None:
                paid = np.repeat(((1.0 - fit.geometric_churn) ** j)[None, :], sel.size, axis=0)
            elif fit.boundary is not None:
                paid = pred.frame.iloc[sel][[f"paid_{k}" for k in range(1, horizon + 1)]
                                            ].to_numpy(np.float64)
            else:
                paid = conditional_paid(pred.model, draws[d], a0[sel], horizon)
            expected[sel, d] = np.sum(paid * m[sel], axis=1)
            u = rng.uniform(size=sel.size)
            outcome[sel, d] = _paid_outcome(paid, m[sel], u)
            if outcome_noise is not None and residuals is not None and noise_rng is not None:
                eps = residuals.draw(seg[sel], ch[sel], ages[sel], noise_rng)
                outcome_noise[sel, d] = _paid_outcome_noise(paid, m[sel], u, eps)
    return SimulatedValue(expected, outcome, f.index, tuple(fixed), outcome_noise)


NO_DROPOUT = np.iinfo(np.int64).max
"""Purchases until dropout when the dropout probability is 0: never within any window (the
value NumPy itself returns for every positive p below about 1e-300)."""


def dropout_purchase_limit(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Purchases until dropout, ``K ~ Geometric(p)`` on ``{1, 2, ...}``. A probability that is
    exactly 0 in floating point (a Beta(a, b + x) draw below the smallest positive double, as
    happens for very small ``a``) is its limit: no dropout (``E4_ZERO_DROPOUT_RULE``, A14). The
    geometric draw is made with such ``p`` replaced by 1, so every other draw and the random
    stream are unchanged. Non-finite or out-of-range ``p`` raises."""
    p = np.asarray(p, dtype=np.float64)
    if not np.all(np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("dropout probability non-finite or outside [0, 1]")
    zero = p == 0.0
    k = rng.geometric(np.where(zero, 1.0, p))
    return np.asarray(np.where(zero, NO_DROPOUT, k), dtype=np.int64)


def e4_simulate(pred: E4Prediction, n_draws: int, rng: np.random.Generator) -> SimulatedValue:
    """E4: Hessian-based draws of each cell's BG/NBD and Gamma-Gamma log parameters; outcomes
    from the posteriors (alive, ``lambda``, ``p``, purchases, spends)."""
    data = pred.data
    f = data.frame
    seg, ch = f["segment"].to_numpy(), f["channel"].to_numpy()
    e0, t = float(data.edges[0]), float(data.edges[-1] - data.edges[0])
    expected = np.full((len(f), n_draws), np.nan)
    outcome = np.full((len(f), n_draws), np.nan)
    fixed: list[tuple[str, str]] = []
    ratio = data.contribution_ratio
    for key, (_, bf, _, gf) in pred.fits.used.items():
        sel = np.flatnonzero((seg == key[0]) & (ch == key[1]))
        if sel.size == 0 or not (bf.finite and gf.finite):
            continue
        bd, ok1 = _draw_log_params(bf.log_params, bf.cov_log, n_draws, rng)
        gd, ok2 = _draw_log_params(gf.log_params, gf.cov_log, n_draws, rng)
        if not (ok1 and ok2):
            fixed.append(key)
        x, t_x, T = (f[c].to_numpy()[sel] for c in ("x", "t_x", "T"))
        n_sp, s_sp = f["n"].to_numpy()[sel], f["S"].to_numpy()[sel]
        for d in range(n_draws):
            r, alpha, a, b = (float(v) for v in np.exp(bd[d]))
            p_s, q_s, g_s = (float(v) for v in np.exp(gd[d]))
            cond = bgnbd_conditional((r, alpha, a, b), x, t_x, T, data.edges)
            spend = p_s * (g_s + s_sp) / (p_s * n_sp + q_s - 1.0)
            expected[sel, d] = cond.purchases.sum(axis=1) * spend * ratio
            alive = rng.uniform(size=sel.size) < cond.p_alive_T
            lam = rng.gamma(r + x, 1.0 / (alpha + T))
            p = rng.beta(a, b + x)
            n_gap = rng.poisson(lam * e0)
            k_drop = dropout_purchase_limit(p, rng)
            n_raw = rng.poisson(lam * t)
            count = np.where(alive, np.clip(k_drop - n_gap, 0, n_raw), 0)
            nu = rng.gamma(p_s * n_sp + q_s, 1.0 / (g_s + s_sp))
            total = np.where(count > 0, rng.gamma(np.maximum(count, 1) * p_s, 1.0 / nu), 0.0)
            outcome[sel, d] = total * ratio
    return SimulatedValue(expected, outcome, f.index, tuple(fixed))


# --- CR1 for the penalised models ----------------------------------------------------------------
@dataclass(frozen=True)
class ClusteredCovariance:
    columns: tuple[str, ...]
    coefficients: FloatArray
    model_se: FloatArray
    cr1_se: FloatArray
    cr1: FloatArray
    n_rows: int
    n_clusters: int


def _e3_rows(model: E3Model, tables: EconomicsTables
             ) -> tuple[FloatArray, FloatArray, np.ndarray]:
    stack = training_stack(tables, model.cutoff, model.horizon, "contractual", model.tier,
                           model.contribution.policy)
    status = observe(view_at(tables, model.cutoff), model.cutoff).status()
    n = window_paid(status, stack["customer_id"].to_numpy(), stack["landmark"].to_numpy(),
                    model.horizon)
    rows = person_months(n, stack["tenure_months"].to_numpy(), model.horizon)
    fr = stack.iloc[rows.row].reset_index(drop=True)
    fit = model.hazard
    group = fit.band_map[tenure_band(rows.age)]
    X = np.column_stack([band_basis(group, len(fit.bands)), fit.design.matrix(fr)])
    return (np.asarray(X, dtype=np.float64), rows.event.astype(np.float64),
            fr["customer_id"].to_numpy())


def e3_cr1(model: E3Model, tables: EconomicsTables) -> ClusteredCovariance:
    """CR1 sandwich of E3's penalised hazard coefficients, clustered by customer (the final fit's
    person-month rows rebuilt exactly)."""
    X, y, cid = _e3_rows(model, tables)
    res = model.hazard.result
    beta = np.asarray(res.coefficients, dtype=np.float64)
    scores = X * (y - expit(X @ beta))[:, None]
    v = pm.cr1_covariance(res.covariance, scores, cid)
    cols = tuple(model.hazard.bands) + tuple(model.hazard.design.columns)
    return ClusteredCovariance(cols, beta, np.sqrt(np.diag(res.covariance)),
                               np.sqrt(np.diag(v)), v, int(X.shape[0]), int(np.unique(cid).size))


def e5_cr1(model: E5Model, tables: EconomicsTables) -> tuple[ClusteredCovariance,
                                                              ClusteredCovariance]:
    """CR1 sandwiches of E5's two parts (the amount part conditional on the estimated shape)."""
    stack = training_stack(tables, model.cutoff, model.horizon, model.kind, model.tier,
                           model.policy)
    r = stack["realised"].to_numpy(np.float64)
    pos = r > 0
    out = []
    for part, rows, y in ((model.positive, stack, pos.astype(np.float64)),
                          (model.amount, stack.loc[pos], r[pos])):
        X = part.design.matrix(rows)
        beta = np.asarray(part.result.coefficients, dtype=np.float64)
        if part.part == "positive":
            scores = X * (y - expit(X @ beta))[:, None]
        else:
            k = float(cast(GammaRegressionResult, part.result).shape)
            scores = X * (k * (y * np.exp(-(X @ beta)) - 1.0))[:, None]
        cid = rows["customer_id"].to_numpy()
        v = pm.cr1_covariance(part.result.covariance, scores, cid)
        out.append(ClusteredCovariance(tuple(part.design.columns), beta,
                                       np.sqrt(np.diag(part.result.covariance)),
                                       np.sqrt(np.diag(v)), v, int(X.shape[0]),
                                       int(np.unique(cid).size)))
    return out[0], out[1]


def _paid_from(fit: HazardFit, feats: pd.DataFrame, horizon: int) -> FloatArray:
    n = len(feats)
    a0 = feats["tenure_months"].to_numpy()
    k = np.tile(np.arange(1, horizon + 1), n)
    rep = feats.iloc[np.repeat(np.arange(n), horizon)].reset_index(drop=True)
    h = fit.hazard(rep, np.repeat(a0, horizon) + k - 1).reshape(n, horizon)
    return np.asarray(np.cumprod(1.0 - h, axis=1), dtype=np.float64)


def simulate_e3(model: E3Model, tables: EconomicsTables, cov: FloatArray, n_draws: int,
                rng: np.random.Generator, *, residuals: ResidualPool | None = None,
                noise_rng: np.random.Generator | None = None) -> SimulatedValue:
    """E3: coefficients drawn from ``N(beta_hat, cov)`` (CR1 by default in the record); A11
    noise as in :func:`simulate_contractual`."""
    noisy = _check_noise(residuals, noise_rng)
    feats = contractual_features(tables, model.cutoff, model.tier, model.contribution.policy)
    seg, ch = feats["segment"].to_numpy(), feats["channel"].to_numpy()
    a0 = feats["tenure_months"].to_numpy().astype(np.int64)
    ages = a0[:, None] + np.arange(1, model.horizon + 1)
    m = model.contribution.expected(seg, ch, ages)
    beta = np.asarray(model.hazard.result.coefficients, dtype=np.float64)
    draws, drawn = _draw_log_params(beta, cov, n_draws, rng)
    expected = np.empty((len(feats), n_draws))
    outcome = np.empty((len(feats), n_draws))
    outcome_noise = np.empty((len(feats), n_draws)) if noisy else None
    for d in range(n_draws):
        fit = replace(model.hazard, result=replace(model.hazard.result, coefficients=draws[d]))
        paid = _paid_from(fit, feats, model.horizon)
        expected[:, d] = np.sum(paid * m, axis=1)
        u = rng.uniform(size=len(feats))
        outcome[:, d] = _paid_outcome(paid, m, u)
        if outcome_noise is not None and residuals is not None and noise_rng is not None:
            outcome_noise[:, d] = _paid_outcome_noise(
                paid, m, u, residuals.draw(seg, ch, ages, noise_rng))
    return SimulatedValue(expected, outcome, feats.index, () if drawn else (("all", "all"),),
                          outcome_noise)


# --- coverage ------------------------------------------------------------------------------------
def coverage_summary(lower: FloatArray, upper: FloatArray, y: FloatArray, level: float,
                     groups: Sequence[object] | np.ndarray | None = None) -> pd.DataFrame:
    """Coverage with its Monte Carlo SE, Clopper-Pearson interval and the protocol's label
    (``COVERAGE_REPORTING_RULE``: reported, never asserted), marginally or by group."""
    covered = (np.asarray(y) >= lower) & (np.asarray(y) <= upper)
    g = np.zeros(covered.size, dtype=np.int64) if groups is None else np.asarray(groups)
    rows = []
    for key in pd.unique(g):
        c = covered[g == key]
        k, n = int(c.sum()), int(c.size)
        lo, hi = pm.clopper_pearson(k, n)
        rows.append({"group": key, "n": n, "coverage": k / n,
                     "mc_se": coverage_mc_se(level, n), "cp_lower": lo, "cp_upper": hi,
                     "label": classify_coverage(k, n, level)})
    return pd.DataFrame(rows)


# --- selection reporting -------------------------------------------------------------------------
@dataclass(frozen=True)
class CandidateInput:
    model_id: str
    tier: str
    """The information tier the model used."""
    selection_eligible: bool
    eligibility: str
    candidate: CandidateResult | None
    frame: pd.DataFrame | None
    """Validation ``prediction`` and ``realised`` indexed by ``customer_id``."""


@dataclass(frozen=True)
class SelectionReport:
    kind: str
    tier: str
    status: str
    selected: str | None
    gates: tuple[GateResult, ...]
    excluded: tuple[tuple[str, str], ...]
    intervals: dict[tuple[str, str], tuple[float, float, float]]
    """``(a, b)`` -> (MSE_a - MSE_b, lower, upper) for qualified pairs."""
    reference_fallback: str | None
    reference_label: str | None
    reason: str


UNDEFINED_REASON = "not qualified: undefined predictions"


def selection_report(kind: str, tier: str, inputs: Sequence[CandidateInput],
                     rng: np.random.Generator, *,
                     resamples: int = pp.CLUSTERED_INFERENCE.metric_bootstrap_resamples
                     ) -> SelectionReport:
    """Apply the frozen selection rule to one scenario kind and information tier."""
    if kind not in ("contractual", "non_contractual") or tier not in ("A", "B"):
        raise ValueError("unknown kind or tier")
    excluded: list[tuple[str, str]] = []
    usable: list[CandidateInput] = []
    for c in inputs:
        spec = pp.model(c.model_id)
        if spec.complexity_rank is None or kind not in spec.kinds:
            raise ValueError(f"{c.model_id} is not a selection candidate for {kind}")
        if c.tier not in spec.tiers or not (c.tier == tier or (c.tier == "A" and tier == "B")):
            raise ValueError(f"{c.model_id} Tier {c.tier} cannot enter a Tier {tier} selection")
        if not c.selection_eligible or c.candidate is None or c.frame is None:
            excluded.append((c.model_id, c.eligibility))
            continue
        pred = c.frame["prediction"].to_numpy(np.float64)
        if not (np.all(np.isfinite(pred)) and math.isfinite(c.candidate.mse)):
            excluded.append((c.model_id, UNDEFINED_REASON))
            continue
        usable.append(c)
    gates = {c.model_id: gate(c.candidate) for c in usable if c.candidate is not None}
    qualified = [c for c in usable if gates[c.model_id].qualified]
    level = pp.SELECTION.paired_interval_level
    intervals: dict[tuple[str, str], tuple[float, float, float]] = {}
    for a in qualified:
        for b in qualified:
            if a.model_id == b.model_id:
                continue
            fa, fb = a.frame, b.frame
            assert fa is not None and fb is not None
            if not fa.index.equals(fb.index) or not np.allclose(
                    fa["realised"].to_numpy(np.float64), fb["realised"].to_numpy(np.float64)):
                raise RuntimeError("paired comparison needs the same customers and labels")
            r = fa["realised"].to_numpy(np.float64)
            la = (r - fa["prediction"].to_numpy(np.float64)) ** 2
            lb = (r - fb["prediction"].to_numpy(np.float64)) ** 2
            intervals[(a.model_id, b.model_id)] = pm.cluster_bootstrap_mean_difference(
                la, lb, fa.index.to_numpy(), resamples, level, rng)
    sel = select_model([c.candidate for c in usable if c.candidate is not None],
                       {k: (v[1], v[2]) for k, v in intervals.items()})
    ref, label = sel.reference_fallback, sel.reference_label
    if kind not in pp.model(pp.REFERENCE_FALLBACK_MODEL).kinds:  # REFERENCE_FALLBACK_SCOPE
        ref, label = None, None
    return SelectionReport(kind, tier, sel.status, sel.selected, sel.gates, tuple(excluded),
                           intervals, ref, label, sel.reason)
