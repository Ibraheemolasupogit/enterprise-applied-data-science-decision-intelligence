"""Milestone 10 case studies: the computations behind the M10 evidence report.

Estimators read observable views only (:func:`~decision_intelligence.customer_value.panel.
observe`); the ground truth (``scenario.truth``) is read here, after estimation, for comparison.
Three quantities are labelled throughout: **realised** (historical), **projected** (finite
horizon) and **expected lifetime** (infinite horizon) contribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from decision_intelligence.common.bootstrap import bootstrap_ci
from decision_intelligence.customer_value import backtest as bt
from decision_intelligence.customer_value import heuristic as he
from decision_intelligence.customer_value import historical as hi
from decision_intelligence.customer_value import protocol as pr
from decision_intelligence.customer_value.economics import (
    channel_cac,
    discount_factors,
    geometric_value,
    monthly_rate,
    payback_age,
)
from decision_intelligence.customer_value.panel import Observable, observe
from decision_intelligence.customer_value.retention import (
    constant_churn,
    kaplan_meier,
    paid_month_km,
)
from decision_intelligence.data import SyntheticDataConfig, build_synthetic_dataset
from decision_intelligence.data.customer_economics_scenarios import (
    CHANNELS,
    SEGMENTS,
    EconomicsScenario,
    EconomicsScenarioConfig,
    generate_economics_scenario,
)
from decision_intelligence.statistics.hypothesis_tests import wilson_interval

RATE = pr.MONTHLY_DISCOUNT_RATE
HEURISTICS = (("H1/H2 constant churn", he.h1_h2), ("H3 by cell", he.h3),
              ("H4 Kaplan–Meier", he.h4))


def scenario(seed: int, **changes: Any) -> EconomicsScenario:
    return generate_economics_scenario(replace(EconomicsScenarioConfig(seed=seed), **changes))


def final_view(sc: EconomicsScenario) -> Observable:
    """Everything recorded by the data end plus the recording lag (all payments recorded)."""
    return observe(sc, sc.config.n_months - 1, lag_days=pr.MAX_RECORDING_LAG_DAYS)


# ---------------------------------------------------------------------------
# Data audit and retention
# ---------------------------------------------------------------------------


def data_audit(sc: EconomicsScenario) -> dict[str, float]:
    obs = final_view(sc)
    st = obs.status()
    tx = sc.transactions
    late = (tx["recorded_at"] - tx["billing_date"]).dt.days > 15
    return {"customers": float(len(sc.customers)), "transactions": float(len(tx)),
            "churned (observed)": float(st["churned"].sum()),
            "active at data end (censored)": float((~st["churned"]).sum()),
            "late-recorded payments (> 15 days)": float(late.sum()),
            "duplicate transaction rows": float(len(tx) - tx["transaction_id"].nunique()),
            "total revenue": float(tx["revenue"].sum()),
            "total contribution": float(obs.transactions["contribution"].sum())}


def retention_table(sc: EconomicsScenario, ages: tuple[int, ...] = (6, 12, 24, 36)
                    ) -> pd.DataFrame:
    """KM paid-month survival with Greenwood (plain) intervals vs the population truth."""
    obs = final_view(sc)
    st = obs.status()
    rows = []
    groups = [("all", None, None)] + [(f"segment {s}", s, None) for s in SEGMENTS] + [
        (f"channel {c}", None, c) for c in CHANNELS]
    for label, seg, ch in groups:
        m = np.ones(len(st), dtype=bool)
        if seg is not None:
            m &= (st["segment"] == seg).to_numpy()
        if ch is not None:
            m &= (st["channel"] == ch).to_numpy()
        km = paid_month_km(st.loc[m, "paid_months"], st.loc[m, "churned"])
        lo, hi_ = km.interval()
        for a in ages:
            i = int(np.searchsorted(km.times, a, side="right")) - 1
            s = float(km.at([a])[0])
            truth = _population_survival(sc, a, seg, ch)
            rows.append({"group": label, "age": a, "km": s,
                         "lo": float(lo[i]) if i >= 0 else 1.0,
                         "hi": float(hi_[i]) if i >= 0 else 1.0, "truth": truth,
                         "covers": bool((lo[i] if i >= 0 else 1.0) <= truth
                                        <= (hi_[i] if i >= 0 else 1.0))})
    return pd.DataFrame(rows)


def _population_survival(sc: EconomicsScenario, age: int, seg: str | None, ch: str | None
                         ) -> float:
    cfg, t = sc.config, sc.truth
    num = den = 0.0
    for ci, c in enumerate(CHANNELS):
        if ch is not None and c != ch:
            continue
        for si, s in enumerate(SEGMENTS):
            if seg is not None and s != seg:
                continue
            w = cfg.acquisitions_per_month[ci] * cfg.segment_mix[ci][si]
            num += w * float(t.population_paid_probability(s, c, np.array([age]))[0])
            den += w
    return num / den


# ---------------------------------------------------------------------------
# Historical (realised) value
# ---------------------------------------------------------------------------


def historical_table(sc: EconomicsScenario, ages: tuple[int, ...] = (12, 24)) -> pd.DataFrame:
    """Realised cohort-matched value by channel vs truth, with the naive counterexamples."""
    obs = final_view(sc)
    rows = []
    for age in ages:
        cm = hi.cohort_matched(obs, age, RATE)
        na = hi.naive_all_customers(obs, age, RATE)
        ns = hi.naive_survivors(obs, age, RATE)
        for ch in CHANNELS:
            v = cm.loc[cm["channel"] == ch, "value"]
            truth = sc.truth.population_value(horizon=age, monthly_rate=RATE, channel=ch)
            se = float(v.std(ddof=1) / math.sqrt(v.size))
            rows.append({"age": age, "channel": ch, "customers": int(v.size),
                         "realised (cohort-matched)": float(v.mean()), "se": se,
                         "truth": truth, "z": (float(v.mean()) - truth) / se,
                         "naive: all customers": float(na.loc[na["channel"] == ch,
                                                              "value"].mean()),
                         "naive: survivors only": float(ns.loc[ns["channel"] == ch,
                                                                "value"].mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Heuristic projected and expected-lifetime value
# ---------------------------------------------------------------------------


def _revenue_view(obs: Observable) -> Observable:
    return replace(obs, transactions=obs.transactions.assign(
        contribution=obs.transactions["revenue"]))


def heuristic_table(sc: EconomicsScenario, *, by: str = "channel") -> pd.DataFrame:
    """H1–H4 projected (24, 36) and expected-lifetime (infinite) value vs truth."""
    obs = final_view(sc)
    keys = CHANNELS if by == "channel" else SEGMENTS
    rows = []
    for k in keys:
        seg = k if by == "segment" else None
        chn = k if by == "channel" else None
        for horizon in (*pr.HORIZONS, None):
            truth = sc.truth.population_value(horizon=horizon, monthly_rate=RATE, segment=seg,
                                              channel=chn)
            row: dict[str, Any] = {by: k, "horizon": "infinite" if horizon is None else horizon,
                                   "quantity": "expected lifetime" if horizon is None
                                   else "projected", "truth": truth}
            for name, fn in HEURISTICS:
                e = fn(obs, RATE, horizon, segment=seg, channel=chn)
                row[name] = e.value
                row[f"{name} bias"] = e.value / truth - 1
                row[f"{name} status"] = e.status
            rows.append(row)
        h0 = he.h0_legacy(obs, segment=seg, channel=chn)
        rows.append({by: k, "horizon": "legacy (12-month, undiscounted)",
                     "quantity": "historical + projected revenue", "truth": math.nan,
                     "H1/H2 constant churn": h0.value,
                     "H1/H2 constant churn status": h0.status})
    return pd.DataFrame(rows)


def revenue_vs_contribution(sc: EconomicsScenario) -> pd.DataFrame:
    """Segment ranking by H4 36-month value on revenue vs on contribution (designed-in)."""
    obs = final_view(sc)
    rev = _revenue_view(obs)
    rows = []
    for s in SEGMENTS:
        rows.append({"segment": s,
                     "revenue value (H4, 36 m)": he.h4(rev, RATE, 36, segment=s).value,
                     "contribution value (H4, 36 m)": he.h4(obs, RATE, 36, segment=s).value,
                     "contribution truth (36 m)": sc.truth.population_value(
                         horizon=36, monthly_rate=RATE, segment=s)})
    out = pd.DataFrame(rows)
    for col, label in (("revenue value (H4, 36 m)", "rank by revenue"),
                       ("contribution value (H4, 36 m)", "rank by contribution"),
                       ("contribution truth (36 m)", "rank by contribution truth")):
        out[label] = out[col].rank(ascending=False).astype(int)
    return out


def stability_flags(sc: EconomicsScenario) -> pd.DataFrame:
    """Status of H1 (infinite) in every segment x channel cell: how many are unstable."""
    obs = final_view(sc)
    rows = []
    for s in SEGMENTS:
        for c in CHANNELS:
            mask = ((obs.status()["segment"] == s) & (obs.status()["channel"] == c)).to_numpy()
            ch = constant_churn(obs, pr.CHURN_WINDOW_MONTHS, mask)
            e = he.h1_h2(obs, RATE, None, segment=s, channel=c)
            rows.append({"segment": s, "channel": c, "churn events": ch.events,
                         "trials": ch.trials, "H1 infinite": e.value, "status": e.status})
    return pd.DataFrame(rows)


def convergence_table(sc: EconomicsScenario) -> pd.DataFrame:
    """Finite-horizon sums approach the infinite-horizon closed form (heuristic and truth)."""
    obs = final_view(sc)
    inf_h = he.h1_h2(obs, RATE, None).value
    inf_t = sc.truth.population_value(horizon=None, monthly_rate=RATE)
    rows = []
    for T in (12, 24, 36, 60, 120, 240, 600):
        rows.append({"horizon": T, "H1/H2 value": he.h1_h2(obs, RATE, T).value,
                     "H1 infinite": inf_h,
                     "relative gap (H1)": he.h1_h2(obs, RATE, T).value / inf_h - 1,
                     "truth value": sc.truth.population_value(horizon=T, monthly_rate=RATE),
                     "truth infinite (summed to 1200)": inf_t})
    undefined = geometric_value(5.0, 1.0, 0.0, None)
    return pd.DataFrame(rows).assign(
        note=f"no churn and no discounting: {undefined.status} ({undefined.reason})")


# ---------------------------------------------------------------------------
# Unit economics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UnitEconomics:
    channels: pd.DataFrame
    cells: pd.DataFrame


def unit_economics(sc: EconomicsScenario, *, seed: int = 0) -> UnitEconomics:
    """Per channel: CAC, realised 24-month LTV, LTV/CAC with bootstrap CI and precision vs the
    target, payback (realised curve; else H4-projected) and truth; per segment x channel cell:
    LTV/CAC precision."""
    obs = final_view(sc)
    cm = hi.cohort_matched(obs, pr.PRIMARY_LTV_AGE, RATE)
    months = sorted(cm["acquisition_month"].unique())
    cac = {c.channel: c for c in channel_cac(obs.marketing, obs.overhead, months)}
    rng = np.random.default_rng(seed)
    rows = []
    st = obs.status()
    for ci, ch in enumerate(CHANNELS):
        v = cm.loc[cm["channel"] == ch, "value"].to_numpy(dtype=np.float64)
        b = bootstrap_ci(v, lambda x: float(np.mean(x)), rng=rng,
                         n_resamples=pr.BOOTSTRAP_RESAMPLES, method="percentile")
        c = cac[ch]
        half = (b.ci_upper - b.ci_lower) / 2 / b.estimate
        truth_ltv = sc.truth.population_value(horizon=pr.PRIMARY_LTV_AGE, monthly_rate=RATE,
                                              channel=ch)
        mask = (st["channel"] == ch).to_numpy()
        row: dict[str, Any] = {
            "channel": ch, "customers": int(v.size), "CAC media": c.media,
            "CAC fully loaded": c.fully_loaded, "LTV24 realised": b.estimate,
            "LTV/CAC (media)": b.estimate / c.media,
            "LTV/CAC 95% CI": f"[{b.ci_lower / c.media:.3f}, {b.ci_upper / c.media:.3f}]",
            "LTV/CAC fully loaded": b.estimate / c.fully_loaded,
            "relative half-width": half, "precision target met": half <= pr.PRECISION_TARGET,
            "truth LTV/CAC (true CPA)": truth_ltv / sc.config.cost_per_acquisition[ci]}
        for label, rate in (("discounted", RATE), ("undiscounted", 0.0)):
            curve = hi.cumulative_curve(obs, 36, rate, mask)  # longest complete window
            pb = payback_age(curve, c.media)
            row[f"payback {label} (media CAC)"] = (
                f"{pb + 1} months (realised)" if pb is not None
                else _projected_payback(obs, rate, c.media, mask))
        rows.append(row)
    cells = []
    for s in SEGMENTS:
        for ch in CHANNELS:
            v = cm.loc[(cm["channel"] == ch) & (cm["segment"] == s), "value"].to_numpy(
                dtype=np.float64)
            if v.size < 2:
                continue
            b = bootstrap_ci(v, lambda x: float(np.mean(x)), rng=rng,
                             n_resamples=pr.BOOTSTRAP_RESAMPLES, method="percentile")
            half = (b.ci_upper - b.ci_lower) / 2 / abs(b.estimate)
            cells.append({"segment": s, "channel": ch, "customers": int(v.size),
                          "LTV/CAC (media)": b.estimate / cac[ch].media,
                          "relative half-width": half,
                          "precision target met": half <= pr.PRECISION_TARGET})
    return UnitEconomics(pd.DataFrame(rows), pd.DataFrame(cells))


def _projected_payback(obs: Observable, rate: float, cac: float, mask: np.ndarray) -> str:
    curve = he.empirical_curve(obs, mask)
    if curve is None:
        return "undefined"
    ages = np.arange(240)
    surv = np.where(ages <= curve.last_age, curve.survival[np.minimum(ages, curve.last_age)],
                    curve.survival[curve.last_age]
                    * curve.tail_continuation ** (ages - curve.last_age))
    contrib = curve.contribution[np.minimum(ages, curve.last_age)]
    cum = np.cumsum(contrib * surv * discount_factors(ages.size, rate))
    pb = payback_age(cum, cac)
    return "not reached within 240 months (projected)" if pb is None else \
        f"{pb + 1} months (H4-projected)"


def segment_economics(sc: EconomicsScenario) -> pd.DataFrame:
    obs = final_view(sc)
    st = obs.status()
    cm = hi.cohort_matched(obs, pr.PRIMARY_LTV_AGE, RATE)
    rows = []
    for s in SEGMENTS:
        m = (st["segment"] == s).to_numpy()
        km = paid_month_km(st.loc[m, "paid_months"], st.loc[m, "churned"])
        tx = obs.transactions.loc[obs.transactions["customer_id"].isin(st.index[m])]
        rows.append({"segment": s, "customers": int(m.sum()),
                     "revenue per paid month": float(tx["revenue"].mean()),
                     "contribution per paid month": float(tx["contribution"].mean()),
                     "contribution margin": float(tx["contribution"].sum() / tx["revenue"].sum()),
                     "retention 12 m (KM)": float(km.at([12])[0]),
                     "retention 24 m (KM)": float(km.at([24])[0]),
                     "LTV24 realised": float(cm.loc[cm["segment"] == s, "value"].mean()),
                     "H4 expected lifetime": he.h4(obs, RATE, None, segment=s).value,
                     "truth expected lifetime": sc.truth.population_value(
                         horizon=None, monthly_rate=RATE, segment=s)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Uncertainty: delta method vs bootstrap for H1; coverage in repeated sampling
# ---------------------------------------------------------------------------


def h1_uncertainty(sc: EconomicsScenario, *, n_boot: int = 1000, seed: int = 3) -> pd.DataFrame:
    """H1 (infinite) per channel: delta-method SE (churn and run-rate) vs customer bootstrap."""
    obs = final_view(sc)
    st = obs.status()
    c_ = obs.cutoff_month
    lo, hi_m = c_ - pr.CHURN_WINDOW_MONTHS, c_ - 1
    through = obs.complete_through()
    tx = obs.transactions
    win = tx.loc[(tx["billing_month"] <= through)
                 & (tx["billing_month"] > through - pr.CHURN_WINDOW_MONTHS)]
    csum = win.groupby("customer_id")["contribution"].sum().reindex(st.index, fill_value=0.0)
    ccnt = win.groupby("customer_id").size().reindex(st.index, fill_value=0)
    acq = st["acquisition_month"].to_numpy()
    last = acq + st["paid_months"].to_numpy() - 1
    churned = st["churned"].to_numpy()
    first = np.maximum(acq, lo)
    last_trial = np.where(churned, last, np.minimum(last, hi_m))
    trials = np.maximum(0, np.minimum(last_trial, hi_m) - first + 1)
    events = (churned & (last >= lo) & (last <= hi_m)).astype(float)
    d = 1.0 / (1.0 + RATE)
    rng = np.random.default_rng(seed)
    rows = []
    for ch in CHANNELS:
        m = (st["channel"] == ch).to_numpy()
        ev, tr = events[m], trials[m].astype(float)
        cs, cn = csum.to_numpy()[m], ccnt.to_numpy()[m].astype(float)
        c = float(np.sum(ev)) / float(np.sum(tr))
        mbar = float(np.sum(cs)) / float(np.sum(cn))

        def h1(cc: float, mm: float) -> float:
            return float(mm / (1.0 - (1.0 - cc) * d))

        est = h1(c, mbar)
        se_c = math.sqrt(c * (1 - c) / float(np.sum(tr)))
        resid = cs - mbar * cn  # ratio-estimator linearisation for the run rate
        se_m = math.sqrt(len(cs) / (len(cs) - 1) * float(np.sum(resid ** 2))) / float(
            np.sum(cn))
        g_c = -mbar * d / (1.0 - (1.0 - c) * d) ** 2
        g_m = 1.0 / (1.0 - (1.0 - c) * d)
        delta = math.sqrt((g_c * se_c) ** 2 + (g_m * se_m) ** 2)
        n = int(m.sum())
        boots = []
        for _ in range(n_boot):
            i = rng.integers(0, n, n)
            boots.append(h1(float(np.sum(ev[i])) / float(np.sum(tr[i])),
                            float(np.sum(cs[i])) / float(np.sum(cn[i]))))
        rows.append({"channel": ch, "H1 infinite": est, "delta-method SE": delta,
                     "bootstrap SE": float(np.std(boots, ddof=1)),
                     "ratio": delta / float(np.std(boots, ddof=1))})
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class Coverage:
    rows: pd.DataFrame
    n_replications: int


def coverage_study(n_replications: int = pr.COVERAGE_REPLICATIONS,
                   seed_start: int = pr.COVERAGE_SEED_START, n_resamples: int = 999
                   ) -> Coverage:
    """Repeated generator draws: does the 95% percentile bootstrap interval of the 24-month
    realised LTV per channel cover the population truth? And of LTV/CAC (media) vs the truth
    with the true CPA (the interval ignores spend noise)?"""
    rec = []
    for r in range(n_replications):
        sc = scenario(seed_start + r)
        obs = final_view(sc)
        cm = hi.cohort_matched(obs, pr.PRIMARY_LTV_AGE, RATE)
        months = sorted(cm["acquisition_month"].unique())
        cac = {c.channel: c for c in channel_cac(obs.marketing, obs.overhead, months)}
        rng = np.random.default_rng(r)
        for ci, ch in enumerate(CHANNELS):
            v = cm.loc[cm["channel"] == ch, "value"].to_numpy(dtype=np.float64)
            b = bootstrap_ci(v, lambda x: float(np.mean(x)), rng=rng, n_resamples=n_resamples,
                             method="percentile")
            truth = sc.truth.population_value(horizon=pr.PRIMARY_LTV_AGE, monthly_rate=RATE,
                                              channel=ch)
            t_ratio = truth / sc.config.cost_per_acquisition[ci]
            rec.append({"channel": ch, "estimate": b.estimate, "truth": truth,
                        "ltv_covered": b.ci_lower <= truth <= b.ci_upper,
                        "ratio_covered": b.ci_lower / cac[ch].media <= t_ratio
                        <= b.ci_upper / cac[ch].media,
                        "half": (b.ci_upper - b.ci_lower) / 2 / b.estimate})
    df = pd.DataFrame(rec)
    out = []
    for key, g in df.groupby("channel", sort=False):
        n = len(g)
        row: dict[str, Any] = {"channel": str(key), "replications": n,
                               "bias": float((g["estimate"] - g["truth"]).mean()),
                               "mc_se": float(g["estimate"].std(ddof=1) / math.sqrt(n)),
                               "mean relative half-width": float(g["half"].mean())}
        for col, label in (("ltv_covered", "LTV24"), ("ratio_covered", "LTV/CAC")):
            k = int(g[col].sum())
            lo, hi_ = wilson_interval(k, n)
            row[f"{label} coverage"] = k / n
            row[f"{label} Wilson"] = f"[{lo:.3f}, {hi_:.3f}]"
            row[f"{label} binomial p"] = float(stats.binomtest(k, n, 0.95).pvalue)
        out.append(row)
    return Coverage(pd.DataFrame(out), n_replications)


# ---------------------------------------------------------------------------
# Heterogeneity grid, backtests, sensitivity, M2 compatibility
# ---------------------------------------------------------------------------


def heterogeneity_grid(seed: int, grid: tuple[float, ...] = pr.HETEROGENEITY_GRID
                       ) -> pd.DataFrame:
    """Bias of H1/H3/H4 vs truth (36-month projected and expected lifetime), all customers,
    as the churn heterogeneity grows (same seed)."""
    rows = []
    for v in grid:
        sc = scenario(seed, frailty_variance=v)
        obs = final_view(sc)
        for horizon in (36, None):
            truth = sc.truth.population_value(horizon=horizon, monthly_rate=RATE)
            row: dict[str, Any] = {"frailty variance": v,
                                   "horizon": "infinite" if horizon is None else horizon,
                                   "truth": truth}
            for name, fn in HEURISTICS:
                e = fn(obs, RATE, horizon)
                row[f"{name} bias"] = e.value / truth - 1
            rows.append(row)
    return pd.DataFrame(rows)


def customer_backtests(sc: EconomicsScenario) -> tuple[pd.DataFrame, list[str]]:
    t = sc.truth
    rows, excluded = [], []
    for c in pr.BACKTEST_CUTOFFS:
        try:
            def expected(proj: pd.DataFrame) -> np.ndarray:
                idx = np.searchsorted(t.customer_ids, proj.index.to_numpy())
                a0 = proj["paid_age"].to_numpy()
                out = np.empty(idx.size)
                w = pr.BACKTEST_WINDOW_MONTHS
                for k, (i, a) in enumerate(zip(idx, a0, strict=True)):
                    ages = np.arange(a + 1, a + w + 1)
                    out[k] = float(np.sum(t.contribution_by_age(t.segment[i], t.channel[i], ages)
                                          * np.exp(-t.hazard[i] * np.arange(1, w + 1))))
                return out

            res = bt.customer_backtest(sc, sc.config.n_months, c, pr.BACKTEST_WINDOW_MONTHS,
                                       expected)
            rows.append(res.summary.assign(cutoff=c))
        except bt.IncompleteWindow as exc:
            excluded.append(f"cutoff {c}: {exc}")
    return pd.concat(rows, ignore_index=True), excluded


def cohort_backtests(sc: EconomicsScenario) -> tuple[pd.DataFrame, list[str]]:
    rows, excluded = [], []
    for c in pr.BACKTEST_CUTOFFS:
        try:
            res = bt.cohort_backtest(sc, sc.config.n_months, c, pr.BACKTEST_WINDOW_MONTHS, RATE,
                                     CHANNELS)
            truth = [sc.truth.population_value(horizon=pr.BACKTEST_WINDOW_MONTHS,
                                               monthly_rate=RATE, channel=ch)
                     for ch in res["channel"]]
            rows.append(res.assign(cutoff=c, truth=truth))
        except bt.IncompleteWindow as exc:
            excluded.append(f"cutoff {c}: {exc}")
    return pd.concat(rows, ignore_index=True), excluded


def sensitivity(sc: EconomicsScenario) -> pd.DataFrame:
    """H4 36-month and expected-lifetime value per segment under discount-rate, gross-margin
    and cost-to-serve changes (applied to the recorded components; deterministic)."""
    obs = final_view(sc)
    rows = []
    base_tx = obs.transactions
    for annual in pr.SENSITIVITY_DISCOUNT_RATES:
        for gm in pr.SENSITIVITY_MARGIN_SHIFTS:
            for cost in pr.SENSITIVITY_COST_SHIFTS:
                if sum(x != y for x, y in ((annual, pr.ANNUAL_DISCOUNT_RATE), (gm, 0.0),
                                           (cost, 0.0))) > 1:
                    continue  # one factor at a time
                gross = (base_tx["revenue"] - base_tx["cogs"]) * (1 + gm)
                contrib = gross - base_tx["payment_fee"] - base_tx["cost_to_serve"] * (1 + cost)
                view = replace(obs, transactions=base_tx.assign(contribution=contrib))
                r = monthly_rate(annual)
                row: dict[str, Any] = {"annual discount": annual, "gross margin shift": gm,
                                       "cost-to-serve shift": cost}
                for s in SEGMENTS:
                    row[f"{s} 36 m"] = he.h4(view, r, 36, segment=s).value
                    e = he.h4(view, r, None, segment=s)
                    row[f"{s} lifetime"] = e.value if e.status != "undefined" else math.nan
                rows.append(row)
    return pd.DataFrame(rows)


def m2_compatibility(config: SyntheticDataConfig | None = None) -> pd.DataFrame:
    """The M2 subscription tables (365 days): KM retention, constant churn and heuristic values
    on revenue and on gross margin (M2 has no cost-to-serve)."""
    data = build_synthetic_dataset(config)
    t = data.tables
    ret = t["customer_retention"]
    churned = ret["churned"].to_numpy(dtype=bool)
    days = ret["tenure_days"].to_numpy(dtype=np.float64)
    # Continuous tenure: a churn in month m is an event at ceil(days/30); an active customer is
    # known to have survived floor(days/30) whole months (censored there).
    months = np.where(churned, np.maximum(1, np.ceil(days / 30.0)), np.floor(days / 30.0))
    km = kaplan_meier(months, churned)
    tx = t["transactions"].merge(t["products"][["product_id", "margin_rate"]], on="product_id")
    c = float(np.sum(churned)) / float(np.sum(np.maximum(months, 1.0)))
    rev = float(tx["amount"].mean())
    gm = float((tx["amount"] * tx["margin_rate"]).mean())
    rows = [{"quantity": "customers", "value": float(len(ret))},
            {"quantity": "max observed tenure (months)", "value": float(months.max())},
            {"quantity": "KM retention at 6 months", "value": float(km.at([6])[0])},
            {"quantity": "KM retention at 12 months", "value": float(km.at([12])[0])},
            {"quantity": "constant monthly churn (events / customer-months)", "value": c},
            {"quantity": "H1 expected lifetime, revenue", "value":
             geometric_value(rev, 1 - c, RATE, None).value},
            {"quantity": "H1 expected lifetime, gross margin", "value":
             geometric_value(gm, 1 - c, RATE, None).value},
            {"quantity": "H2 36-month, gross margin", "value":
             geometric_value(gm, 1 - c, RATE, 36).value}]
    return pd.DataFrame(rows)
