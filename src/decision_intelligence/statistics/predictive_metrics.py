"""Predictive-evaluation metrics (added in M11 P7).

Owned here (statistical primitives live in :mod:`decision_intelligence.statistics`); pure
NumPy/SciPy on arrays, no customer-value objects. Every function has a brute-force reference
in the tests.

**Point predictions** (``pred``, outcome ``y``): relative bias ``sum(pred) / sum(y) - 1`` (the
protocol's calibration-in-the-large), MAE, RMSE; the calibration line ``y = a + b pred`` by OLS
with HC1 standard errors; decile reliability (mean prediction and mean outcome within
prediction deciles, rank-based, ties broken by position); Spearman; top-decile capture (M10's
definition: share of the total outcome held by the ``ceil(0.1 n)`` largest predictions);
normalised Gini (the Gini of ``y`` ordered by ``pred`` over the Gini ordered by ``y``).

**Discrete-time survival** (months). ``T`` = months survived; failure means month ``T + 1``
is not survived. Observed: ``time = T, event = True`` if the follow-up ``C`` exceeds ``T``,
otherwise ``time = C, event = False`` (survived at least ``C`` months). Predictions
``surv[:, t-1] = S(t) = P(T >= t)``, ``t = 1 .. H`` (``S(0) = 1``). The censoring survival
``g[u] = P(C > u)`` is the Kaplan-Meier estimate of ``C`` in which an observed failure at ``u``
is at risk of, but not, censored at ``u`` (its follow-up exceeded ``u``). Weights: an observed
failure at ``T`` gets ``1 / g[T]``; a subject observed alive at ``t`` (``time >= t``) gets
``1 / P(C >= t) = 1 / g[t - 1]``. Harrell's C: pairs with ``time_i < time_j`` and ``event_i``
(``j`` survived beyond ``i``'s failure), concordant when ``risk_i > risk_j``, risk ties 1/2;
Uno's C: the same pairs weighted by ``g[time_i]^-2``, ``time_i < tau``. IPCW Brier at ``t``
(Graf et al. 1999): ``mean[1{time < t, event} S(t)^2 / g[time] + 1{time >= t} (1 - S(t))^2 /
g[t - 1]]``, unbiased for ``E[(1{T >= t} - S(t))^2]`` under independent censoring; the
integrated Brier score averages ``t = 1 .. H``. D-calibration (Haider et al. 2020) with the
exact uniform mass of each subject's survival-probability interval in each of ``bins`` equal
bins (no randomisation): a failure at ``T`` is uniform on ``[S(T + 1), S(T)]``, a censored
subject on ``[0, S(C)]``; Pearson chi-square against equal counts.

**Count distributions** (``pmf[i, k] = P(N_i = k)``, ``k = 0 .. K``): log score
``-log pmf[y]``; CRPS ``sum_k (F(k) - 1{y <= k})^2``; the randomised PIT
``F(y - 1) + U (F(y) - F(y - 1))``.

**Inference.** Paired customer-clustered bootstrap of a mean loss difference (customers
resampled with replacement, all their rows carried: multinomial cluster weights, which is the
same resampling as drawing clusters); CR1 cluster-robust sandwich
``c H^-1 (sum_g s_g s_g') H^-1`` with ``c = G / (G - 1) (n - 1) / (n - p)`` (statsmodels'
correction); Clopper-Pearson coverage intervals.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats

from decision_intelligence.common.validation import FloatArray
from decision_intelligence.statistics.linear_regression import fit_ols

TOP_FRACTION = 0.10


def _pair(pred: ArrayLike, y: ArrayLike) -> tuple[FloatArray, FloatArray]:
    p = np.asarray(pred, dtype=np.float64)
    o = np.asarray(y, dtype=np.float64)
    if p.ndim != 1 or p.shape != o.shape or p.size == 0:
        raise ValueError("pred and y must be non-empty 1-D arrays of equal length")
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(o))):
        raise ValueError("pred and y must be finite")
    return p, o


# --- point predictions --------------------------------------------------------------------------
def relative_bias(pred: ArrayLike, y: ArrayLike) -> float:
    p, o = _pair(pred, y)
    return float(np.sum(p)) / float(np.sum(o)) - 1.0


def mae(pred: ArrayLike, y: ArrayLike) -> float:
    p, o = _pair(pred, y)
    return float(np.mean(np.abs(o - p)))


def rmse(pred: ArrayLike, y: ArrayLike) -> float:
    p, o = _pair(pred, y)
    return math.sqrt(float(np.mean((o - p) ** 2)))


def calibration_line(pred: ArrayLike, y: ArrayLike) -> dict[str, float]:
    """OLS of ``y`` on ``pred``: intercept, slope and their HC1 standard errors."""
    p, o = _pair(pred, y)
    res = fit_ols(p[:, None], o, feature_names=["pred"], covariance="HC1")
    coef = np.asarray(res.coefficients, dtype=np.float64)
    se = np.asarray(res.standard_errors, dtype=np.float64)
    return {"intercept": float(coef[0]), "slope": float(coef[1]),
            "intercept_se": float(se[0]), "slope_se": float(se[1])}


def decile_groups(pred: ArrayLike, bins: int = 10) -> np.ndarray:
    """Group 0 .. bins-1 by prediction rank (stable; equal-size groups up to one row)."""
    p = np.asarray(pred, dtype=np.float64)
    order = np.argsort(p, kind="stable")
    groups = np.empty(p.size, dtype=np.int64)
    groups[order] = (np.arange(p.size) * bins) // max(p.size, 1)
    return groups


def decile_reliability(pred: ArrayLike, y: ArrayLike, bins: int = 10) -> pd.DataFrame:
    p, o = _pair(pred, y)
    g = decile_groups(p, bins)
    frame = pd.DataFrame({"group": g, "pred": p, "y": o})
    out = frame.groupby("group").agg(n=("y", "size"), mean_pred=("pred", "mean"),
                                     mean_y=("y", "mean"))
    return out.reset_index()


def spearman(pred: ArrayLike, y: ArrayLike) -> float:
    p, o = _pair(pred, y)
    return float(stats.spearmanr(p, o).statistic)


def top_decile_capture(pred: ArrayLike, y: ArrayLike, fraction: float = TOP_FRACTION) -> float:
    p, o = _pair(pred, y)
    n_top = max(1, int(np.ceil(fraction * p.size)))
    top = np.argsort(-p, kind="stable")[:n_top]
    return float(np.sum(o[top])) / float(np.sum(o))


def _gini(order_key: FloatArray, y: FloatArray) -> float:
    order = np.argsort(-order_key, kind="stable")
    cum = np.cumsum(y[order]) / float(np.sum(y))
    n = y.size
    return float(np.sum(cum - np.arange(1, n + 1) / n)) / n


def normalised_gini(pred: ArrayLike, y: ArrayLike) -> float:
    p, o = _pair(pred, y)
    return _gini(p, o) / _gini(o, o)


# --- discrete-time survival ---------------------------------------------------------------------
def _survival_inputs(time: ArrayLike, event: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(time, dtype=np.float64)
    e = np.asarray(event, dtype=bool)
    if t.ndim != 1 or t.shape != e.shape or t.size == 0 or not np.all(np.isfinite(t)) \
            or np.any(t < 0) or not np.allclose(t, np.round(t)):
        raise ValueError("time must be non-negative integers aligned with event")
    return np.round(t).astype(np.int64), e


def censoring_km(time: ArrayLike, event: ArrayLike, horizon: int) -> FloatArray:
    """``g[u] = P(C > u)``, ``u = 0 .. horizon`` (see the module docstring)."""
    t, e = _survival_inputs(time, event)
    g = np.ones(horizon + 1)
    surv = 1.0
    for u in range(horizon + 1):
        at_risk = int(np.sum(t >= u))
        cens = int(np.sum((t == u) & ~e))
        if at_risk > 0 and cens > 0:
            surv *= 1.0 - cens / at_risk
        g[u] = surv
    return np.asarray(g, dtype=np.float64)


def _concordance(time: np.ndarray, event: np.ndarray, risk: FloatArray, weight: FloatArray,
                 tau: int | None) -> float:
    num = den = 0.0
    for s in np.unique(time[event]):
        if tau is not None and s >= tau:
            continue
        i = np.flatnonzero((time == s) & event)
        later = np.sort(risk[time > s])
        if later.size == 0:
            continue
        lo = np.searchsorted(later, risk[i], side="left")
        hi = np.searchsorted(later, risk[i], side="right")
        conc = lo + 0.5 * (hi - lo)  # j with risk_j < risk_i; ties 1/2
        num += float(np.sum(weight[i] * conc))
        den += float(np.sum(weight[i])) * later.size
    return num / den if den > 0 else math.nan


def harrell_c(time: ArrayLike, event: ArrayLike, risk: ArrayLike) -> float:
    t, e = _survival_inputs(time, event)
    return _concordance(t, e, np.asarray(risk, dtype=np.float64), np.ones(t.size), None)


def uno_c(time: ArrayLike, event: ArrayLike, risk: ArrayLike, tau: int) -> float:
    t, e = _survival_inputs(time, event)
    g = censoring_km(t, e, int(t.max()))
    gt = g[t]
    w = np.divide(1.0, gt ** 2, out=np.zeros(t.size), where=gt > 0)  # 0 only if censored
    return _concordance(t, e, np.asarray(risk, dtype=np.float64), w, tau)


def ipcw_brier(t_eval: int, time: ArrayLike, event: ArrayLike, surv_t: ArrayLike,
               g: FloatArray | None = None) -> float:
    """IPCW Brier score at ``t_eval >= 1`` (``g`` = :func:`censoring_km` by default)."""
    t, e = _survival_inputs(time, event)
    s = np.asarray(surv_t, dtype=np.float64)
    if s.shape != t.shape or t_eval < 1:
        raise ValueError("surv_t must align with time and t_eval >= 1")
    gg = censoring_km(t, e, max(int(t.max()), t_eval)) if g is None else g
    failed = (t < t_eval) & e
    alive = t >= t_eval
    term = np.zeros(t.size)
    term[failed] = s[failed] ** 2 / gg[t[failed]]
    term[alive] = (1.0 - s[alive]) ** 2 / gg[t_eval - 1]
    return float(np.mean(term))


def integrated_brier(time: ArrayLike, event: ArrayLike, surv: ArrayLike) -> float:
    t, e = _survival_inputs(time, event)
    sv = np.asarray(surv, dtype=np.float64)
    h = sv.shape[1]
    g = censoring_km(t, e, max(int(t.max()), h))
    return float(np.mean([ipcw_brier(k, t, e, sv[:, k - 1], g) for k in range(1, h + 1)]))


def d_calibration(time: ArrayLike, event: ArrayLike, surv: ArrayLike, bins: int = 10
                  ) -> dict[str, float]:
    t, e = _survival_inputs(time, event)
    sv = np.asarray(surv, dtype=np.float64)
    h = sv.shape[1]
    if np.any(t > h) or np.any(e & (t >= h)):
        raise ValueError("observations beyond the predicted horizon")
    full = np.column_stack([np.ones(t.size), sv])  # full[:, k] = S(k), k = 0 .. H
    idx = np.arange(t.size)
    upper = full[idx, t]
    lower = np.where(e, full[idx, np.minimum(t + 1, h)], 0.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    width = upper - lower
    point = width <= 1e-15
    mass = np.empty((t.size, bins))
    for b in range(bins):
        ov = np.clip(np.minimum(upper, edges[b + 1]) - np.maximum(lower, edges[b]), 0.0, None)
        last = b == bins - 1
        inside = (lower >= edges[b]) & ((lower < edges[b + 1]) | last)
        mass[:, b] = np.where(point, inside.astype(np.float64),
                              ov / np.where(point, 1.0, width))
    counts = mass.sum(axis=0)
    expected = t.size / bins
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    return {"chi2": chi2, "p_value": float(stats.chi2.sf(chi2, bins - 1)),
            "max_bin_share": float(np.max(counts)) / t.size,
            "min_bin_share": float(np.min(counts)) / t.size}


def paid_months_pmf(paid: ArrayLike) -> FloatArray:
    """Contractual window: ``paid[:, j-1] = P(month j paid) = S(j)`` with paid months
    contiguous, so ``P(N = k) = S(k) - S(k + 1)`` (``S(0) = 1``, ``S(H + 1) = 0``)."""
    p = np.asarray(paid, dtype=np.float64)
    full = np.column_stack([np.ones(p.shape[0]), p, np.zeros(p.shape[0])])
    return np.asarray(full[:, :-1] - full[:, 1:], dtype=np.float64)


# --- count distributions ------------------------------------------------------------------------
def _counts(pmf: ArrayLike, y: ArrayLike) -> tuple[FloatArray, np.ndarray]:
    p = np.asarray(pmf, dtype=np.float64)
    o = np.asarray(y)
    if p.ndim != 2 or o.shape != (p.shape[0],):
        raise ValueError("pmf must be (n, K + 1) and y length n")
    if np.any(o < 0) or np.any(o >= p.shape[1]) or not np.allclose(o, np.round(o)):
        raise ValueError("y must be integers in 0 .. K")
    return p, np.round(o).astype(np.int64)


def log_score(pmf: ArrayLike, y: ArrayLike) -> FloatArray:
    p, o = _counts(pmf, y)
    return np.asarray(-np.log(np.maximum(p[np.arange(o.size), o], 1e-300)), dtype=np.float64)


def crps_counts(pmf: ArrayLike, y: ArrayLike) -> FloatArray:
    p, o = _counts(pmf, y)
    cdf = np.cumsum(p, axis=1)
    step = (np.arange(p.shape[1])[None, :] >= o[:, None]).astype(np.float64)
    return np.asarray(np.sum((cdf - step) ** 2, axis=1), dtype=np.float64)


def randomised_pit(pmf: ArrayLike, y: ArrayLike, rng: np.random.Generator) -> FloatArray:
    p, o = _counts(pmf, y)
    cdf = np.cumsum(p, axis=1)
    hi = cdf[np.arange(o.size), o]
    lo = np.where(o > 0, cdf[np.arange(o.size), np.maximum(o - 1, 0)], 0.0)
    return np.asarray(lo + rng.uniform(size=o.size) * (hi - lo), dtype=np.float64)


# --- inference ----------------------------------------------------------------------------------
def cluster_bootstrap_mean_difference(loss_a: ArrayLike, loss_b: ArrayLike, cluster: ArrayLike,
                                      resamples: int, level: float, rng: np.random.Generator
                                      ) -> tuple[float, float, float]:
    """Point estimate and percentile interval of ``mean(loss_a - loss_b)`` over rows, with
    clusters (customers) resampled with replacement and all their rows carried."""
    a = np.asarray(loss_a, dtype=np.float64)
    b = np.asarray(loss_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("loss arrays must be aligned 1-D arrays")
    _, codes = np.unique(np.asarray(cluster), return_inverse=True)
    n_c = int(np.max(codes)) + 1
    d_sum = np.bincount(codes, weights=a - b, minlength=n_c)
    rows = np.bincount(codes, minlength=n_c).astype(np.float64)
    est = float(d_sum.sum() / rows.sum())
    w = rng.multinomial(n_c, np.full(n_c, 1.0 / n_c), size=resamples).astype(np.float64)
    boot = (w @ d_sum) / (w @ rows)
    alpha = 1.0 - level
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return est, float(lo), float(hi)


def cr1_covariance(bread: ArrayLike, row_scores: ArrayLike, cluster: ArrayLike) -> FloatArray:
    """CR1 sandwich ``c B M B`` with ``B`` = the inverse (penalised) Hessian, ``M`` = the sum
    over clusters of outer products of summed row scores, ``c = G/(G-1) (n-1)/(n-p)``."""
    B = np.asarray(bread, dtype=np.float64)
    s = np.asarray(row_scores, dtype=np.float64)
    n, p = s.shape
    _, codes = np.unique(np.asarray(cluster), return_inverse=True)
    g = int(np.max(codes)) + 1
    if g < 2 or n <= p:
        raise ValueError("CR1 needs at least two clusters and more rows than parameters")
    sums = np.zeros((g, p))
    np.add.at(sums, codes, s)
    c = g / (g - 1) * (n - 1) / (n - p)
    return np.asarray(c * B @ (sums.T @ sums) @ B, dtype=np.float64)


def clopper_pearson(covered: int, n: int, level: float = 0.95) -> tuple[float, float]:
    if not 0 <= covered <= n or n < 1:
        raise ValueError("covered must lie in [0, n] with n >= 1")
    a = 1.0 - level
    lo = 0.0 if covered == 0 else float(stats.beta.ppf(a / 2, covered, n - covered + 1))
    hi = 1.0 if covered == n else float(stats.beta.ppf(1 - a / 2, covered + 1, n - covered))
    return lo, hi
