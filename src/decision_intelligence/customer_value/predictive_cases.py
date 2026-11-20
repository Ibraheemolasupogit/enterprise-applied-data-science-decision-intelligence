"""Milestone 11 P8: per-draw computations and aggregation for the A12 design.

Evaluation code (it may read generator truth and oracles; estimation modules never do). Every
function takes an explicit seed or scenario, so a draw's result depends on its seed only
(``P8_EXECUTION.seeding``); ``scripts/generate_m11_evidence.py`` schedules draws and writes the
records.

**Studies** (``REPEATED_SAMPLING``, ``PC_DEFINITIONS``, ``ACCEPTANCE_GATE_4``):

* :func:`r1_draw` - ``ORACLE_GATE`` quantities on signals_contractual (cutoffs 36, 42) and the
  PC2 draw statistic on the m10_contractual heterogeneity grid.
* :func:`r2_draw` - interval coverage (E1, E2, E3, E4 and S2; single declared cutoff per
  setting).
* :func:`r3_draw` - ``ACCEPTANCE_GATE_4`` quantities (E2 on M10, E4 on purchases) at a scale.
* :func:`refit_draw` - the E2-only refit bootstrap (M10, C = 30).
* :func:`main_seed_analysis` - PC1, PC3, PC4, PC5 (``PC_TEST``) and the selection reports.

**Accelerated E4 simulation** (``E4_ACCELERATED_SIMULATION``): :func:`e4_simulate_fast` uses one
Gauss-Jacobi rule for Beta(a, b) per parameter draw, reweighted exactly by
``(1 - p)^x B(a, b) / B(a, b + x)``, the telescoped window total, and all draws at once, for
customers with ``x <= E4_ACCELERATED_MAX_X``; every other customer uses the original per-count
rule (``btyd.beta_nodes(a, b + x)``). Parameter draws and the outcome steps are those of P7's
``e4_simulate`` (unchanged), drawn in a different order.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from scipy import special, stats

from decision_intelligence.customer_value import btyd
from decision_intelligence.customer_value import contractual as ct
from decision_intelligence.customer_value import heuristic as he
from decision_intelligence.customer_value import landmarks as lm
from decision_intelligence.customer_value import oracle as orc
from decision_intelligence.customer_value import predictive as pr
from decision_intelligence.customer_value import predictive_baselines as pb
from decision_intelligence.customer_value import predictive_evaluation as ev
from decision_intelligence.customer_value import predictive_protocol as pp
from decision_intelligence.customer_value import purchase_oracle as po
from decision_intelligence.customer_value import survival as sv
from decision_intelligence.customer_value.features import PromotionPolicy
from decision_intelligence.customer_value.panel import observe
from decision_intelligence.customer_value.protocol import CHURN_WINDOW_MONTHS, HETEROGENEITY_GRID
from decision_intelligence.data import customer_economics_scenarios as m10g
from decision_intelligence.data import predictive_ltv_scenarios as gen
from decision_intelligence.data.customer_economics_scenarios import CHANNELS
from decision_intelligence.statistics import predictive_metrics as pm
from decision_intelligence.statistics.multiple_testing import adjust_p_values

W = pp.PRIMARY_HORIZON
RATE = pp.MONTHLY_DISCOUNT_RATE
PARAMETRIC_DRAWS = pp.CLUSTERED_INFERENCE.refit_bootstrap_resamples  # P7's parametric draws
LEVELS = pp.CALIBRATION.interval_levels
POLICY = PromotionPolicy(2, tuple(c for c, p in zip(CHANNELS, (False, True, True, False),
                                                    strict=True) if p))
"""The business's own promotion policy (P3-P7 convention)."""


def _rng(seed: int, stream: int) -> np.random.Generator:
    """Independent component streams of one draw (SeedSequence of ``[seed, stream]``)."""
    return np.random.default_rng([seed, stream])


def m10_scenario(seed: int, scale: float = 1.0, **changes: Any) -> m10g.EconomicsScenario:
    base = m10g.EconomicsScenarioConfig(seed=seed)
    cfg = replace(base, acquisitions_per_month=tuple(a * scale for a in
                                                     base.acquisitions_per_month), **changes)
    return m10g.generate_economics_scenario(cfg)


def signals_scenario(seed: int) -> gen.SignalsScenario:
    return gen.generate_signals_scenario(gen.SignalsScenarioConfig(seed=seed))


def purchases_scenario(seed: int, scale: float = 1.0) -> gen.PurchasesScenario:
    base = gen.PurchasesScenarioConfig(seed=seed)
    return gen.generate_purchases_scenario(replace(
        base, acquisitions_per_month=tuple(a * scale for a in base.acquisitions_per_month)))


# --- accelerated E4 simulation (E4_ACCELERATED_SIMULATION) ---------------------------------------
def beta_nodes_batch(a: np.ndarray, b: np.ndarray, n: int = btyd.E4_JACOBI_NODES
                     ) -> tuple[np.ndarray, np.ndarray]:
    """``btyd.beta_nodes`` for many ``(a, b)`` at once (same Jacobi matrix, batched solver)."""
    al, be = b - 1.0, a - 1.0
    i = np.arange(n, dtype=np.float64)
    s = 2.0 * i[None, :] + (al + be)[:, None]
    diag = np.empty((a.size, n))
    diag[:, 0] = (be - al) / (al + be + 2.0)
    diag[:, 1:] = ((be * be - al * al)[:, None]) / (s[:, 1:] * (s[:, 1:] + 2.0))
    j = i[1:][None, :]
    sj = 2.0 * j + (al + be)[:, None]
    off = np.sqrt(4.0 * j * (j + al[:, None]) * (j + be[:, None]) * (j + (al + be)[:, None])
                  / (sj * sj * (sj + 1.0) * (sj - 1.0)))
    mat = np.zeros((a.size, n, n))
    idx = np.arange(n)
    mat[:, idx, idx] = diag
    mat[:, idx[1:], idx[:-1]] = off
    mat[:, idx[:-1], idx[1:]] = off
    y, v = np.linalg.eigh(mat)
    w = v[:, 0, :] ** 2
    return (np.asarray((1.0 + y) / 2.0, dtype=np.float64),
            np.asarray(w / w.sum(axis=1, keepdims=True), dtype=np.float64))


def e4_window_purchases(params: np.ndarray, x: np.ndarray, t_x: np.ndarray, T: np.ndarray,
                        e0: float, eh: float, max_x: int = pp.E4_ACCELERATED_MAX_X
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Expected window purchases and P(alive at T), shape ``(D, n)``, for natural BG/NBD
    parameters ``params`` ``(D, 4)``; accelerated rule for ``x <= max_x``, the original
    per-count rule otherwise."""
    r, alpha, a, b = (params[:, k][:, None] for k in range(4))
    k = r + x[None, :]
    with np.errstate(divide="ignore", over="ignore"):
        odds = np.where(x[None, :] > 0,
                        a / (b + np.where(x > 0, x, 1.0)[None, :] - 1.0)
                        * np.exp(k * (np.log(alpha + T[None, :]) - np.log(alpha + t_x[None, :]))),
                        0.0)
    p_alive = 1.0 / (1.0 + odds)
    window = np.empty_like(p_alive)
    pn0, w0 = beta_nodes_batch(params[:, 2], params[:, 3])
    for xv in np.unique(x):
        cols = np.flatnonzero(x == xv)
        den = params[:, 1][:, None] + T[cols][None, :]
        kk = (params[:, 0] + xv)[:, None, None]
        if xv <= max_x:
            ratio = np.exp(special.betaln(params[:, 2], params[:, 3])
                           - special.betaln(params[:, 2], params[:, 3] + xv))
            pn, w = pn0, w0 * (1.0 - pn0) ** xv * ratio[:, None]
        else:  # the original per-count Gauss-Jacobi rule
            nodes = [btyd.beta_nodes(float(params[d, 2]), float(params[d, 3] + xv))
                     for d in range(params.shape[0])]
            pn = np.stack([q[0] for q in nodes])
            w = np.stack([q[1] for q in nodes])
        a0 = (1.0 + pn[:, None, :] * e0 / den[:, :, None]) ** (-kk)
        ah = (1.0 + pn[:, None, :] * eh / den[:, :, None]) ** (-kk)
        window[:, cols] = np.einsum("dj,dcj->dc", w / pn, a0 - ah)
    return p_alive * window, p_alive


def e4_simulate_fast(pred: btyd.E4Prediction, n_draws: int, rng: np.random.Generator, *,
                     param_draws: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] | None
                     = None, max_x: int = pp.E4_ACCELERATED_MAX_X) -> ev.SimulatedValue:
    """Statistically equivalent to P7's ``e4_simulate`` (A12); ``param_draws`` (log scale,
    per cell) are for equivalence tests only."""
    data = pred.data
    f = data.frame
    seg, ch = f["segment"].to_numpy(), f["channel"].to_numpy()
    e0, eh = float(data.edges[0]), float(data.edges[-1])
    t = eh - e0
    expected = np.full((len(f), n_draws), np.nan)
    outcome = np.full((len(f), n_draws), np.nan)
    fixed: list[tuple[str, str]] = []
    for key, (_, bf, _, gf) in pred.fits.used.items():
        sel = np.flatnonzero((seg == key[0]) & (ch == key[1]))
        if sel.size == 0 or not (bf.finite and gf.finite):
            continue
        if param_draws is not None:
            bd, gd = param_draws[key]
        else:
            bd, ok1 = ev._draw_log_params(bf.log_params, bf.cov_log, n_draws, rng)
            gd, ok2 = ev._draw_log_params(gf.log_params, gf.cov_log, n_draws, rng)
            if not (ok1 and ok2):
                fixed.append(key)
        bp, gp = np.exp(bd), np.exp(gd)
        x, t_x, T = (f[c].to_numpy()[sel] for c in ("x", "t_x", "T"))
        n_sp, s_sp = f["n"].to_numpy()[sel], f["S"].to_numpy()[sel]
        purchases, p_alive = e4_window_purchases(bp, x, t_x, T, e0, eh, max_x)
        p_s, q_s, g_s = gp[:, 0][:, None], gp[:, 1][:, None], gp[:, 2][:, None]
        spend = p_s * (g_s + s_sp[None, :]) / (p_s * n_sp[None, :] + q_s - 1.0)
        expected[sel] = (purchases * spend * data.contribution_ratio).T
        shape = p_alive.shape
        alive = rng.uniform(size=shape) < p_alive
        lam = rng.gamma(bp[:, 0][:, None] + x[None, :], 1.0 / (bp[:, 1][:, None] + T[None, :]))
        p = rng.beta(np.broadcast_to(bp[:, 2][:, None], shape), bp[:, 3][:, None] + x[None, :])
        n_gap = rng.poisson(lam * e0)
        k_drop = rng.geometric(p)
        n_raw = rng.poisson(lam * t)
        count = np.where(alive, np.clip(k_drop - n_gap, 0, n_raw), 0)
        nu = rng.gamma(p_s * n_sp[None, :] + q_s, 1.0 / (g_s + s_sp[None, :]))
        total = np.where(count > 0, rng.gamma(np.maximum(count, 1) * p_s, 1.0 / nu), 0.0)
        outcome[sel] = (total * data.contribution_ratio).T
    return ev.SimulatedValue(expected, outcome, f.index, tuple(fixed))


# --- PC_TEST -----------------------------------------------------------------------------------
@dataclass(frozen=True)
class PairedTest:
    estimate: float
    se: float
    z: float
    p_value: float
    ci_lower: float
    ci_upper: float
    rows: int
    customers: int
    excluded: int


def paired_test(pred_a: np.ndarray, pred_b: np.ndarray, realised: np.ndarray,
                customer: np.ndarray, rng: np.random.Generator,
                resamples: int = pp.CLUSTERED_INFERENCE.metric_bootstrap_resamples
                ) -> PairedTest:
    """``PC_TEST``: mean of ``D = (R - a)^2 - (R - b)^2`` over customer x cutoff rows, CR1 z
    (``G / (G - 1)`` correction), two-sided normal p-value, percentile cluster bootstrap."""
    ok = np.isfinite(pred_a) & np.isfinite(pred_b)
    a, b, r, c = pred_a[ok], pred_b[ok], realised[ok], customer[ok]
    d = (r - a) ** 2 - (r - b) ** 2
    _, codes = np.unique(c, return_inverse=True)
    g = int(np.max(codes)) + 1
    est = float(np.mean(d))
    sums = np.bincount(codes, weights=d - est, minlength=g)
    se = math.sqrt(g / (g - 1) * float(np.sum(sums ** 2))) / d.size
    z = est / se
    p = float(2.0 * stats.norm.sf(abs(z)))
    _, lo, hi = pm.cluster_bootstrap_mean_difference((r - a) ** 2, (r - b) ** 2, c, resamples,
                                                     0.95, rng)
    return PairedTest(est, se, z, p, lo, hi, int(d.size), g, int((~ok).sum()))


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values at ``FAMILY_ALPHA`` across the given comparisons."""
    keys = list(p_values)
    res = adjust_p_values([p_values[k] for k in keys], method="holm", alpha=pp.FAMILY_ALPHA)
    return {k: float(v) for k, v in zip(keys, res.adjusted_p_values, strict=True)}


# --- R1: ORACLE_GATE and the PC2 draw statistic -----------------------------------------------
def oracle_gate_rows(sc: gen.SignalsScenario) -> list[dict[str, Any]]:
    rows = []
    for cutoff in pp.scenario("signals_contractual").evaluation_cutoffs:
        h = lm.holdout_frame(sc, sc.config.n_months, cutoff, W, "contractual", "B", POLICY
                             ).set_index("customer_id")
        mu_a = orc.signals_oracle(sc, cutoff, W, "A").frame["oracle"].reindex(h.index).to_numpy()
        mu_b = orc.signals_oracle(sc, cutoff, W, "B").frame["oracle"].reindex(h.index).to_numpy()
        r = h["realised"].to_numpy(np.float64)
        rows.append({"cutoff": cutoff,
                     "D": float(np.mean((r - mu_a) ** 2 - (r - mu_b) ** 2)),
                     "G": float(np.mean((mu_b - mu_a) ** 2))})
    return rows


def _cell_cohort_value(model: str, fit: ct.Fit, cm: ct.ContributionModel, seg: str, ch: str
                       ) -> float:
    def contribution(ages: np.ndarray) -> np.ndarray:
        return np.asarray(cm.expected(np.array([seg]), np.array([ch]),
                                      np.asarray(ages)[None, :])[0], dtype=np.float64)

    if fit.geometric_churn is not None:  # E1 A7 limit as its geometric equivalent
        cv = ct.cohort_value("E2", (-math.log(1.0 - fit.geometric_churn), 0.0), contribution,
                             horizon=None, monthly_rate=RATE)
    elif not fit.finite:
        return math.nan
    else:
        cv = ct.cohort_value(model, fit.params, contribution, horizon=None, monthly_rate=RATE)
    return cv.value if cv.status == "ok" else math.nan


def channel_lifetime_values(sc: m10g.EconomicsScenario, cutoff: int
                            ) -> list[dict[str, Any]]:
    """Per channel at ``cutoff``: truth, E2, E1 and H4 expected discounted lifetime values per
    acquired customer (``PC_DEFINITIONS['PC2']``)."""
    obs = observe(sc, cutoff)
    st = obs.status()
    cm = ct.fit_contribution(sc, cutoff, POLICY)
    fits = {m: ct.fit_cells(m, st) for m in ("E1", "E2")}
    recent = st.loc[st["acquisition_month"] > obs.cutoff_month - CHURN_WINDOW_MONTHS]
    rows = []
    for chn in CHANNELS:
        counts = recent.loc[recent["channel"] == chn, "segment"].value_counts()
        row: dict[str, Any] = {"cutoff": cutoff, "channel": chn,
                               "truth": sc.truth.population_value(horizon=None,
                                                                  monthly_rate=RATE,
                                                                  channel=chn)}
        for m, cf in fits.items():
            if counts.sum() == 0:
                row[m] = math.nan
                continue
            vals = []
            for seg, n in counts.items():
                used = cf.used.get((str(seg), chn))
                vals.append(n * (_cell_cohort_value(m, used[1], cm, str(seg), chn)
                                 if used is not None else math.nan))
            row[m] = float(np.sum(vals) / counts.sum())
        h4 = he.h4(obs, RATE, None, channel=chn)
        row["H4"] = h4.value if h4.status != "undefined" else math.nan
        rows.append(row)
    return rows


def pc2_rows(seed: int) -> list[dict[str, Any]]:
    rows = []
    for v in HETEROGENEITY_GRID:
        sc = m10_scenario(seed, frailty_variance=v)
        for cutoff in pp.scenario("m10_contractual").evaluation_cutoffs:
            for row in channel_lifetime_values(sc, cutoff):
                rows.append({"frailty_variance": v, **row})
    return rows


def pc2_statistic(rows: Sequence[dict[str, Any]]) -> float:
    """Draw statistic S (NaN if any value is undefined)."""
    f = pd.DataFrame(rows)
    vals = f[["truth", "E2", "H4"]].to_numpy(np.float64)
    if not np.all(np.isfinite(vals)):
        return math.nan
    bias_e2 = f["E2"] / f["truth"] - 1.0
    bias_h4 = f["H4"] / f["truth"] - 1.0
    return float(np.mean(np.abs(bias_h4) - np.abs(bias_e2)))


def r1_draw(seed: int) -> dict[str, Any]:
    gate = oracle_gate_rows(signals_scenario(seed))
    rows = pc2_rows(seed)
    return {"study": "R1", "seed": seed, "oracle_gate": gate, "pc2": rows,
            "S": pc2_statistic(rows)}


# --- R2: interval coverage (single declared cutoff per setting) ---------------------------------
def _coverage_rows(setting: str, model: str, interval: str, lo: np.ndarray, hi: np.ndarray,
                   y: np.ndarray, segment: np.ndarray, decile: np.ndarray, level: float
                   ) -> list[dict[str, Any]]:
    ok = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(y)
    cov = (y >= lo) & (y <= hi) & ok
    out = [{"setting": setting, "model": model, "interval": interval, "level": level,
            "group": "all", "covered": int(cov.sum()), "n": int(ok.sum())}]
    for kind, g in (("segment", segment), ("decile", decile)):
        for key in pd.unique(g):
            sel = (g == key) & ok
            out.append({"setting": setting, "model": model, "interval": interval,
                        "level": level, "group": f"{kind}={key}",
                        "covered": int(cov[sel].sum()), "n": int(sel.sum())})
    return out


def r2_draw(seed: int, n_draws: int = PARAMETRIC_DRAWS) -> dict[str, Any]:
    rng, noise = _rng(seed, 1), _rng(seed, 2)
    rows: list[dict[str, Any]] = []
    m10 = m10_scenario(seed)
    sig = signals_scenario(seed)
    for name, sc, cutoff, tiers in (("m10_contractual", m10, 30, ("A",)),
                                    ("signals_contractual", sig, 42, ("A", "B"))):
        preds = {m: ct.predict(m, sc, cutoff, W, POLICY) for m in ("E1", "E2")}
        for tier in tiers:
            setting = f"{name} C={cutoff} Tier {tier}"
            h = lm.holdout_frame(sc, sc.config.n_months, cutoff, W, "contractual", tier, POLICY
                                 ).set_index("customer_id")
            oracle = (orc.signals_oracle(sc, cutoff, W, tier) if isinstance(
                sc, gen.SignalsScenario) else orc.m10_oracle(sc, cutoff, W)).frame
            real = h["realised"].to_numpy(np.float64)
            mu = oracle["oracle"].reindex(h.index).to_numpy(np.float64)
            seg = h["segment"].to_numpy()
            sims: dict[str, tuple[ev.SimulatedValue, np.ndarray]] = {}
            if tier == "A":
                for m in ("E1", "E2"):
                    sims[m] = (ev.simulate_contractual(
                        preds[m], n_draws, rng, residuals=ev.contribution_residuals(
                            sc, cutoff, preds[m].contribution), noise_rng=noise),
                        preds[m].frame["prediction"].to_numpy(np.float64))
            e3 = sv.fit_e3(sc, cutoff, W, tier, POLICY)
            cr = ev.e3_cr1(e3, sc)
            sims["E3"] = (ev.simulate_e3(e3, sc, cr.cr1, n_draws, rng,
                                         residuals=ev.contribution_residuals(sc, cutoff,
                                                                             e3.contribution),
                                         noise_rng=noise),
                          sv.predict_e3(e3, sc).frame["prediction"].to_numpy(np.float64))
            for m, (sim, point) in sims.items():
                if not sim.index.equals(h.index):
                    raise RuntimeError(f"{m}: population differs from the holdout")
                dec = pm.decile_groups(point)
                for level in LEVELS:
                    rows += _coverage_rows(setting, m, "parameter vs oracle",
                                           *sim.interval(level, outcome=False), mu, seg, dec,
                                           level)
                    rows += _coverage_rows(setting, m, "predictive vs realised",
                                           *sim.interval(level, outcome=True, noise=True), real,
                                           seg, dec, level)
            e5 = pr.fit_e5(sc, cutoff, W, "contractual", tier, POLICY)
            s2 = pr.fit_s2(sc, cutoff, W, "contractual", tier, POLICY, penalty=e5.penalty)
            fr = s2.frame.reindex(h.index)
            dec5 = pm.decile_groups(pr.predict_e5(e5, sc).frame["prediction"].reindex(h.index)
                                    .to_numpy(np.float64))
            for level in LEVELS:
                rows += _coverage_rows(setting, "E5 + S2", "conformal vs realised",
                                       fr[f"lower_{level:g}"].to_numpy(),
                                       fr[f"upper_{level:g}"].to_numpy(), real, seg, dec5,
                                       level)
    pur = purchases_scenario(seed)
    setting = "purchases_noncontractual C=30 Tier A"
    h = lm.holdout_frame(pur, pur.config.n_months, 30, W, "non_contractual", "A"
                         ).set_index("customer_id")
    e4 = btyd.predict(pur, 30, W)
    sim = e4_simulate_fast(e4, n_draws, rng)
    real = h["realised"].reindex(sim.index).to_numpy(np.float64)
    mu = po.purchases_oracle(pur, 30, W).frame["oracle"].reindex(sim.index).to_numpy(np.float64)
    seg = e4.frame["segment"].to_numpy()
    dec = pm.decile_groups(e4.frame["prediction"].to_numpy(np.float64))
    for level in LEVELS:
        rows += _coverage_rows(setting, "E4", "parameter vs oracle",
                               *sim.interval(level, outcome=False), mu, seg, dec, level)
        rows += _coverage_rows(setting, "E4", "predictive vs realised",
                               *sim.interval(level, outcome=True), real, seg, dec, level)
    e5 = pr.fit_e5(pur, 30, W, "non_contractual", "A")
    s2 = pr.fit_s2(pur, 30, W, "non_contractual", "A", penalty=e5.penalty)
    fr = s2.frame.reindex(h.index)
    dec5 = pm.decile_groups(pr.predict_e5(e5, pur).frame["prediction"].reindex(h.index)
                            .to_numpy(np.float64))
    for level in LEVELS:
        rows += _coverage_rows(setting, "E5 + S2", "conformal vs realised",
                               fr[f"lower_{level:g}"].to_numpy(),
                               fr[f"upper_{level:g}"].to_numpy(),
                               h["realised"].to_numpy(np.float64), h["segment"].to_numpy(), dec5,
                               level)
    return {"study": "R2", "seed": seed, "coverage": rows}


# --- R3: ACCEPTANCE_GATE_4 -----------------------------------------------------------------------
def r3_draw(seed: int, scale: float) -> dict[str, Any]:
    rows = []
    m10 = m10_scenario(seed, scale)
    for cutoff in pp.scenario("m10_contractual").evaluation_cutoffs:
        o = orc.m10_oracle(m10, cutoff, W).frame["oracle"]
        p = ct.predict("E2", m10, cutoff, W, POLICY).frame["prediction"].reindex(o.index)
        rows.append(_gate4_row("E2", cutoff, scale, p.to_numpy(np.float64), o.to_numpy()))
    pur = purchases_scenario(seed, scale)
    for cutoff in pp.scenario("purchases_noncontractual").evaluation_cutoffs:
        o = po.purchases_oracle(pur, cutoff, W).frame["oracle"]
        p = btyd.predict(pur, cutoff, W).frame["prediction"].reindex(o.index)
        rows.append(_gate4_row("E4", cutoff, scale, p.to_numpy(np.float64), o.to_numpy()))
    return {"study": "R3", "seed": seed, "scale": scale, "gate4": rows}


def _gate4_row(model: str, cutoff: int, scale: float, pred: np.ndarray, oracle: np.ndarray
               ) -> dict[str, Any]:
    ok = np.isfinite(pred) & np.isfinite(oracle)
    return {"model": model, "cutoff": cutoff, "scale": scale,
            "bias": float(np.sum(pred[ok]) / np.sum(oracle[ok]) - 1.0),
            "excess_rmse": float(np.sqrt(np.mean((pred[ok] - oracle[ok]) ** 2))),
            "undefined": int((~ok).sum())}


# --- REFIT: E2 refit bootstrap (M10, C = 30) -----------------------------------------------------
def e2_expected_values(pred: ct.ContractualPrediction, fits: ct.CellFits) -> np.ndarray:
    """Expected window value of ``pred``'s customers under ``fits`` (E2's prediction rule; the
    contribution model of ``pred`` held fixed)."""
    f = pred.frame
    seg, ch = f["segment"].to_numpy(), f["channel"].to_numpy()
    a0 = f["age"].to_numpy().astype(np.float64)
    paid = np.full((len(f), W), np.nan)
    for (s, c), (_, fit) in fits.used.items():
        sel = (seg == s) & (ch == c)
        if sel.any() and fit.finite:
            paid[sel] = ct.conditional_paid("E2", fit.log_params, a0[sel], W)
    ages = a0[:, None].astype(np.int64) + np.arange(1, W + 1)[None, :]
    return np.asarray(np.sum(paid * pred.contribution.expected(seg, ch, ages), axis=1),
                      dtype=np.float64)


def refit_draw(seed: int, refits: int, n_draws: int = PARAMETRIC_DRAWS) -> dict[str, Any]:
    sc = m10_scenario(seed)
    cutoff = 30
    pred = ct.predict("E2", sc, cutoff, W, POLICY)
    mu = orc.m10_oracle(sc, cutoff, W).frame["oracle"].reindex(pred.frame.index).to_numpy()
    st = observe(sc, cutoff).status()
    rng = _rng(seed, 3)
    boot = np.empty((len(pred.frame), refits))
    for b in range(refits):
        idx = rng.integers(0, len(st), len(st))  # customers resampled with replacement
        boot[:, b] = e2_expected_values(pred, ct.fit_cells("E2", st.iloc[idx]))
    hessian = ev.simulate_contractual(pred, n_draws, _rng(seed, 4))
    seg = pred.frame["segment"].to_numpy()
    dec = pm.decile_groups(pred.frame["prediction"].to_numpy(np.float64))
    rows: list[dict[str, Any]] = []
    for level in LEVELS:
        a = (1.0 - level) / 2.0
        lo, hi = np.quantile(boot, [a, 1.0 - a], axis=1)
        rows += _coverage_rows("m10_contractual C=30 Tier A", "E2", "refit bootstrap vs oracle",
                               lo, hi, mu, seg, dec, level)
        rows += _coverage_rows("m10_contractual C=30 Tier A", "E2", "Hessian vs oracle",
                               *hessian.interval(level, outcome=False), mu, seg, dec, level)
    return {"study": "REFIT", "seed": seed, "refits": refits, "coverage": rows}


# --- main-seed analysis (PC1, PC3, PC4, PC5 and selection) --------------------------------------
def _holdout(sc: Any, cutoff: int, kind: str, tier: str) -> pd.DataFrame:
    return lm.holdout_frame(sc, sc.config.n_months, cutoff, W, kind, tier,
                            POLICY if kind == "contractual" else None).set_index("customer_id")


def _contractual_candidates(sc: Any, cutoff: int, tier: str
                            ) -> tuple[list[ev.CandidateInput], dict[str, pd.Series]]:
    """Validation inputs and holdout predictions of the contractual candidates of ``tier``."""
    preds: dict[str, pd.Series] = {}
    inputs = []
    v1 = pb.validate_baseline("B1", sc, cutoff, policy=POLICY)
    inputs.append(ev.CandidateInput("B1", "A", v1.selection_eligible, v1.eligibility,
                                    v1.candidate, v1.frame))
    preds["B1"] = pb.predict_b1(sc, cutoff).frame["prediction"]
    for m in ("E1", "E2"):
        v = ct.validate(m, sc, cutoff, W, POLICY)
        inputs.append(ev.CandidateInput(m, "A", True, "eligible", v.candidate, v.frame))
        preds[m] = ct.predict(m, sc, cutoff, W, POLICY).frame["prediction"]
    e3 = sv.fit_e3(sc, cutoff, W, tier, POLICY)
    v3 = sv.validate_e3(sc, cutoff, W, tier, POLICY, penalty=e3.penalty)
    inputs.append(ev.CandidateInput("E3", tier, v3.selection_eligible, v3.eligibility,
                                    v3.candidate, v3.frame))
    preds["E3"] = sv.predict_e3(e3, sc).frame["prediction"]
    e5 = pr.fit_e5(sc, cutoff, W, "contractual", tier, POLICY)
    v5 = pr.validate_e5(sc, cutoff, W, "contractual", tier, POLICY, penalty=e5.penalty)
    inputs.append(ev.CandidateInput("E5", tier, v5.selection_eligible, v5.eligibility,
                                    v5.candidate, v5.frame))
    preds["E5"] = pr.predict_e5(e5, sc).frame["prediction"]
    return inputs, preds


def _stack(parts: Sequence[tuple[pd.Series, pd.Series, pd.DataFrame]]
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a = np.concatenate([p.reindex(h.index).to_numpy(np.float64) for p, _, h in parts])
    b = np.concatenate([q.reindex(h.index).to_numpy(np.float64) for _, q, h in parts])
    r = np.concatenate([h["realised"].to_numpy(np.float64) for _, _, h in parts])
    c = np.concatenate([h.index.to_numpy() for _, _, h in parts])
    return a, b, r, c


def pc3_comparator(selections: dict[int, ev.SelectionReport]) -> dict[int, str | None]:
    """PC3: the Tier A model selected at each cutoff, ``None`` when non-evaluable (never an
    unqualified model)."""
    return {c: (rep.selected if rep.status == "SELECTED" else None)
            for c, rep in selections.items()}


def main_seed_analysis(seed: int) -> dict[str, Any]:
    """PC1, PC3, PC4, PC5 (``PC_TEST``) and every selection report, on one seed."""
    boot = _rng(seed, 5)
    out: dict[str, Any] = {"seed": seed, "selection": [], "tests": {}}
    m10 = m10_scenario(seed)
    parts = []
    for cutoff in pp.scenario("m10_contractual").evaluation_cutoffs:
        inputs, preds = _contractual_candidates(m10, cutoff, "A")
        rep = ev.selection_report("contractual", "A", inputs, boot)
        out["selection"].append(("m10_contractual", cutoff, "A", rep))
        parts.append((preds["E2"], preds["B1"], _holdout(m10, cutoff, "contractual", "A")))
    out["tests"]["PC1"] = paired_test(*_stack(parts), boot)
    sig = signals_scenario(seed)
    tier_a: dict[int, ev.SelectionReport] = {}
    preds_by: dict[tuple[int, str], dict[str, pd.Series]] = {}
    for cutoff in pp.scenario("signals_contractual").evaluation_cutoffs:
        for tier in ("A", "B"):
            inputs, preds = _contractual_candidates(sig, cutoff, tier)
            rep = ev.selection_report("contractual", tier, inputs, boot)
            out["selection"].append(("signals_contractual", cutoff, tier, rep))
            preds_by[(cutoff, tier)] = preds
            if tier == "A":
                tier_a[cutoff] = rep
    comparator = pc3_comparator(tier_a)
    out["pc3_comparator"] = comparator
    pc3_parts = [(preds_by[(c, "B")]["E3"], preds_by[(c, "A")][m],
                  _holdout(sig, c, "contractual", "B"))
                 for c, m in comparator.items() if m is not None]
    out["tests"]["PC3"] = paired_test(*_stack(pc3_parts), boot) if pc3_parts else None
    pc5_parts = [(preds_by[(c, "B")]["E5"], preds_by[(c, "B")]["E3"],
                  _holdout(sig, c, "contractual", "B"))
                 for c in pp.scenario("signals_contractual").evaluation_cutoffs]
    out["tests"]["PC5"] = paired_test(*_stack(pc5_parts), boot)
    pur = purchases_scenario(seed)
    pc4_parts = []
    for cutoff in pp.scenario("purchases_noncontractual").evaluation_cutoffs:
        v2 = pb.validate_baseline("B2", pur, cutoff)
        v4 = btyd.validate(pur, cutoff, W)
        e5 = pr.fit_e5(pur, cutoff, W, "non_contractual", "A")
        v5 = pr.validate_e5(pur, cutoff, W, "non_contractual", "A", penalty=e5.penalty)
        rep = ev.selection_report("non_contractual", "A", [
            ev.CandidateInput("B2", "A", v2.selection_eligible, v2.eligibility, v2.candidate,
                              v2.frame),
            ev.CandidateInput("E4", "A", True, "eligible", v4.candidate, v4.frame),
            ev.CandidateInput("E5", "A", v5.selection_eligible, v5.eligibility, v5.candidate,
                              v5.frame)], boot)
        out["selection"].append(("purchases_noncontractual", cutoff, "A", rep))
        pc4_parts.append((btyd.predict(pur, cutoff, W).frame["prediction"],
                          pb.predict_b2(pur, cutoff).frame["prediction"],
                          _holdout(pur, cutoff, "non_contractual", "A")))
    out["tests"]["PC4"] = paired_test(*_stack(pc4_parts), boot)
    return out


# --- aggregation ---------------------------------------------------------------------------------
def t_interval(x: np.ndarray, level: float = 0.95) -> tuple[float, float, float, float]:
    """Mean, SE, and the two-sided t interval of ``x``."""
    x = np.asarray(x, dtype=np.float64)
    m, se = float(np.mean(x)), float(np.std(x, ddof=1) / math.sqrt(x.size))
    q = float(stats.t.ppf(0.5 + level / 2.0, x.size - 1))
    return m, se, m - q * se, m + q * se


def aggregate_oracle_gate(draws: Sequence[dict[str, Any]]) -> pd.DataFrame:
    f = pd.DataFrame([r for d in draws for r in d["oracle_gate"]])
    rows = []
    gate = pp.ORACLE_GATE
    for cutoff, g in f.groupby("cutoff"):
        diff = (g["D"] - g["G"]).to_numpy()
        m, se, _, _ = t_interval(diff)
        md, _, lo, hi = t_interval(g["D"].to_numpy(), gate.positive_interval_level)
        rows.append({"cutoff": cutoff, "draws": len(g), "mean(D - G)": m, "z": m / se,
                     "mean D": md, "D lower": lo, "D upper": hi,
                     "pass": bool(abs(m / se) < gate.consistency_z and lo > 0)})
    return pd.DataFrame(rows)


def aggregate_pc2(draws: Sequence[dict[str, Any]]) -> dict[str, float]:
    s = np.array([d["S"] for d in draws], dtype=np.float64)
    ok = s[np.isfinite(s)]
    m, se, lo, hi = t_interval(ok)
    p = float(2.0 * stats.t.sf(abs(m / se), ok.size - 1))
    return {"draws": float(ok.size), "excluded": float(s.size - ok.size), "mean S": m,
            "se": se, "ci_lower": lo, "ci_upper": hi, "t": m / se, "p_value": p}


def aggregate_gate4(draws: Sequence[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    f = pd.DataFrame([{**r, "seed": d["seed"]} for d in draws for r in d["gate4"]])
    bias_rows = []
    for (model, cutoff, scale), g in f.groupby(["model", "cutoff", "scale"]):
        m, se, _, _ = t_interval(g["bias"].to_numpy())
        bias_rows.append({"model": model, "cutoff": cutoff, "scale": scale, "draws": len(g),
                          "mean bias": m, "z": m / se,
                          "pass": bool(abs(m / se) < pp.ORACLE_GATE.consistency_z)})
    slope_rows = []
    for (model, cutoff), g in f.groupby(["model", "cutoff"]):
        slopes = []
        for _, d in g.groupby("seed"):
            if d["scale"].nunique() >= 3:
                slopes.append(float(np.polyfit(np.log(d["scale"]), np.log(d["excess_rmse"]),
                                               1)[0]))
        m, se, lo, hi = t_interval(np.array(slopes))
        slope_rows.append({"model": model, "cutoff": cutoff, "draws": len(slopes),
                           "mean slope": m, "lower": lo, "upper": hi, "pass": bool(hi < 0)})
    return pd.DataFrame(bias_rows), pd.DataFrame(slope_rows)


def classify_draw_coverage(lower: float, upper: float, level: float) -> str:
    """Draw-level rule (proposed A13): the 95% t interval of mean per-draw coverage across
    independent generator draws against the nominal level."""
    if not (math.isfinite(lower) and math.isfinite(upper)):
        return "insufficient draws"
    if upper < level:
        return "below nominal"
    if lower > level:
        return "above nominal"
    return "consistent with nominal"


def aggregate_coverage(draws: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Coverage across independent draws (the unit of replication: customers within a draw share
    parameter draws). Per key: mean of per-draw coverage, its 95% t interval across draws and the
    draw-level classification; the pooled customer proportion is descriptive only. Never used
    for recalibration (COVERAGE_REPORTING_RULE)."""
    f = pd.DataFrame([{**r, "seed": d["seed"]} for d in draws for r in d["coverage"]])
    keys = ["setting", "model", "interval", "level", "group"]
    rows = []
    for key, g in f.groupby(keys, sort=False):
        per = (g["covered"] / g["n"].where(g["n"] > 0)).dropna().to_numpy(np.float64)
        level = float(g["level"].iloc[0])
        if per.size >= 2:
            m, se, lo, hi = t_interval(per)
        else:
            m = float(per[0]) if per.size else math.nan
            se = lo = hi = math.nan
        rows.append({**dict(zip(keys, key, strict=True)), "draws": int(per.size),
                     "mean coverage": m, "draw SE": se, "95% lower": lo, "95% upper": hi,
                     "classification": classify_draw_coverage(lo, hi, level),
                     "pooled coverage (descriptive)": float(g["covered"].sum()
                                                            / max(g["n"].sum(), 1))})
    return pd.DataFrame(rows)


def m10_comparability(seed: int) -> dict[str, Any]:
    """The labelled M10 comparability replication: PC1's comparison and the selection on the
    m10_contractual scenario only (descriptive; never a Holm member)."""
    boot = _rng(seed, 5)
    m10 = m10_scenario(seed)
    parts, reports = [], []
    for cutoff in pp.scenario("m10_contractual").evaluation_cutoffs:
        inputs, preds = _contractual_candidates(m10, cutoff, "A")
        reports.append(("m10_contractual", cutoff, "A",
                        ev.selection_report("contractual", "A", inputs, boot)))
        parts.append((preds["E2"], preds["B1"], _holdout(m10, cutoff, "contractual", "A")))
    return {"seed": seed, "selection": reports, "tests": {"PC1": paired_test(*_stack(parts),
                                                                               boot)}}


@dataclass(frozen=True)
class Task:
    study: str
    """``R1``, ``R2``, ``R3``, ``REFIT``, ``MAIN`` or ``M10``."""
    seed: int
    scale: float = 1.0
    refits: int = 0


def run_task(task: Task) -> dict[str, Any]:
    """One unit of P8 work; the result depends on ``task`` only (picklable, for workers)."""
    import os
    import time
    t0 = time.perf_counter()
    if task.study == "R1":
        out = r1_draw(task.seed)
    elif task.study == "R2":
        out = r2_draw(task.seed)
    elif task.study == "R3":
        out = r3_draw(task.seed, task.scale)
    elif task.study == "REFIT":
        out = refit_draw(task.seed, task.refits)
    elif task.study == "MAIN":
        out = main_seed_analysis(task.seed)
    elif task.study == "M10":
        out = m10_comparability(task.seed)
    else:
        raise ValueError(f"unknown study {task.study!r}")
    out.update({"task": task, "seconds": time.perf_counter() - t0, "pid": os.getpid()})
    return out


def primary_p_values(tests: dict[str, PairedTest | None], pc2_p: float) -> dict[str, float]:
    """The five Holm-family p-values; a non-evaluable PC3 (``None``) enters with p = 1
    (``PC_DEFINITIONS['PC3']``)."""
    def p(key: str) -> float:
        t = tests[key]
        return 1.0 if t is None else t.p_value
    return {"PC1": p("PC1"), "PC2": pc2_p, "PC3": p("PC3"), "PC4": p("PC4"), "PC5": p("PC5")}
