"""Milestone 11 P3 record: E1 (sBG) and E2 (exponential-Gamma) against the P2 oracles.

Writes ``docs/evidence/m11-p3-contractual-models.md``. **Development seeds only** (``<=
predictive_protocol.DEVELOPMENT_SEED_MAX``); the pilot, main and fresh seeds are not used.
Not evidence of final model performance (that is P8, on the main seed): it records fit status,
temporal-validation gates, holdout comparisons with the exact oracles, parameter recovery and
runtime at full scenario scale.

    python scripts/report_m11_p3_contractual.py
"""

from __future__ import annotations

import math
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import ParamSpec, TypeVar

import numpy as np
import pandas as pd
import scipy

from decision_intelligence.customer_value import contractual as ct
from decision_intelligence.customer_value import landmarks as lm
from decision_intelligence.customer_value import oracle as orc
from decision_intelligence.customer_value import predictive_protocol as pp
from decision_intelligence.customer_value import protocol as m10p
from decision_intelligence.customer_value.features import PromotionPolicy
from decision_intelligence.customer_value.model_selection import gate
from decision_intelligence.data import customer_economics_scenarios as m10g
from decision_intelligence.data import predictive_ltv_scenarios as gen
from decision_intelligence.data.customer_economics_scenarios import CHANNELS

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "docs" / "evidence" / "m11-p3-contractual-models.md"
SEED_M10 = 5101
SEED_SIGNALS = 5102
REPEAT_SEEDS = tuple(range(5111, 5121))
SCALE_SEEDS = tuple(range(5131, 5136))
SCALES = (0.25, 1.0, 4.0, 16.0)
A6_SEEDS = tuple(range(5141, 5151))
BIAS_SEEDS = tuple(range(5161, 5173))
W = pp.PRIMARY_HORIZON
TIMINGS: dict[str, float] = {}
Scenario = m10g.EconomicsScenario | gen.SignalsScenario
Config = m10g.EconomicsScenarioConfig | gen.SignalsScenarioConfig
P = ParamSpec("P")
T = TypeVar("T")


def table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def f(x: float, digits: int = 4) -> str:
    return "nan" if not math.isfinite(x) else f"{x:.{digits}g}"


def policy(cfg: Config) -> PromotionPolicy:
    return PromotionPolicy(cfg.promo_months, tuple(
        c for c, p in zip(CHANNELS, cfg.promo_channels, strict=True) if p))


def slope_se(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    x = pred - pred.mean()
    sxx = float(np.sum(x * x))
    slope = float(np.sum(x * (target - target.mean())) / sxx)
    e = target - target.mean() - slope * x
    return slope, math.sqrt(float(np.sum(x * x * e * e))) / sxx


def git_state() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, check=True,
                              capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, check=True,
                               capture_output=True, text=True).stdout.strip().splitlines()
        return f"{head} + {len(dirty)} uncommitted or untracked paths" if dirty else head
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def timed(label: str, fn: Callable[P, T], *args: P.args, **kw: P.kwargs) -> T:
    t = time.perf_counter()
    out = fn(*args, **kw)
    TIMINGS[label] = TIMINGS.get(label, 0.0) + time.perf_counter() - t
    return out


def evaluate(name: str, sc: Scenario, cutoff: int, oracle: pd.DataFrame
             ) -> list[list[object]]:
    pol = policy(sc.config)
    hold = lm.holdout_frame(sc, sc.config.n_months, cutoff, W, "contractual", "A", pol)
    r = hold.set_index("customer_id")["realised"].reindex(oracle.index).to_numpy(np.float64)
    mu = oracle["oracle"].to_numpy(np.float64)
    mse_o = float(np.mean((r - mu) ** 2))
    rows: list[list[object]] = []
    for model in ct.MODEL_IDS:
        pred = timed(f"fit + predict {model}", ct.predict, model, sc, cutoff, W, pol)
        val = timed(f"temporal validation {model}", ct.validate, model, sc, cutoff, W, pol)
        p = pred.frame["prediction"].reindex(oracle.index).to_numpy(np.float64)
        mse = float(np.mean((r - p) ** 2))
        excess = float(np.mean((p - mu) ** 2))
        slope, se = slope_se(p, r)
        g = gate(val.candidate)
        counts = pred.fits.status_counts()
        nb = len(pred.fits.boundary_cells)
        rows.append([name, cutoff, model,
                     ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
                     + (f" (v=0 boundary {nb})" if nb else ""),
                     pred.fits.pooled.status,
                     f"L={val.landmark}: CITL {val.candidate.calibration_in_the_large:+.3f}, "
                     f"slope {val.candidate.calibration_slope:.3f} -> "
                     f"{'qualified' if g.qualified else 'not qualified'}",
                     len(p), f(float(np.mean(p - mu)), 3), f(math.sqrt(excess), 3),
                     f(excess / mse_o, 3), f(mse), f(mse_o),
                     f(float(np.sum(p)) / float(np.sum(r)) - 1, 3),
                     f"{slope:.3f} ({se:.3f})"])
    return rows


HEADER = ["scenario", "C", "model", "cells (status)", "pooled", "validation (never holdout)",
          "n", "mean(pred - oracle)", "RMSE(pred - oracle)", "excess / MSE_oracle",
          "realised MSE", "oracle MSE", "holdout CITL", "holdout slope (HC SE)"]


def main() -> int:
    start = time.perf_counter()
    lines: list[str] = []
    w = lines.append

    # 1. published sBG example
    w("\n## 1. Published sBG example (Fader & Hardie 2007, Table 1, years 1-7)\n")
    rows: list[list[object]] = []
    for label, surv, ref in (("High End", [869, 743, 653, 593, 551, 517, 491],
                              "0.668 / 3.806 / -1,611.2"),
                             ("Regular", [631, 468, 382, 326, 289, 262, 241],
                              "0.704 / 1.182 / not reported")):
        s = [1000, *surv]
        churn = [s[i] - s[i + 1] for i in range(7)]
        paid = np.concatenate([np.full(c, i + 1) for i, c in enumerate(churn)]
                              + [np.full(s[7], 8)])
        ev = np.concatenate([np.ones(sum(churn), bool), np.zeros(s[7], bool)])
        fit = ct.fit("E1", ct.DurationData.from_customers(paid, ev))
        rows.append([label, f"{fit.params[0]:.4f} / {fit.params[1]:.4f} / "
                     f"{fit.log_likelihood:,.2f}", ref, fit.status])
    w(table(["segment", "alpha / beta / LL (this code)", "paper", "status"], rows))

    # 2. scenarios vs oracles
    w("\n## 2. Holdout comparison with the P2 oracles (Tier A)\n")
    w("Population: active at `C` with at least one complete month (the oracles' and "
      "features' set). `excess` = `mean((pred - oracle)^2)`; since `E[(R - m)^2] = "
      "E[(R - oracle)^2] + E[(oracle - m)^2]`, it is the model's loss above the attainable "
      "floor. Validation gates use the validation landmark only (`SELECTION`); the holdout "
      "is scored here for the record, not for selection.\n")
    rows = []
    for v in m10g_grid():
        mcfg = m10g.EconomicsScenarioConfig(seed=SEED_M10, frailty_variance=v)
        msc = timed("generate M10 scenario", m10g.generate_economics_scenario, mcfg)
        for cutoff in pp.scenario("m10_contractual").evaluation_cutoffs:
            o = timed("oracle", orc.m10_oracle, msc, cutoff, W).frame
            rows += evaluate(f"m10 v={v}", msc, cutoff, o)
    sig_cfg = gen.SignalsScenarioConfig(seed=SEED_SIGNALS)
    for label, scfg in (("signals (PH)", sig_cfg),
                        ("signals (NPH)", gen.signals_variant("non_proportional_hazards",
                                                              sig_cfg))):
        ssc = timed("generate signals scenario", gen.generate_signals_scenario, scfg)
        for cutoff in pp.scenario("signals_contractual").evaluation_cutoffs:
            o = timed("oracle", orc.signals_oracle, ssc, cutoff, W, "A").frame
            rows += evaluate(label, ssc, cutoff, o)
    w(table(HEADER, rows))

    # 3. parameter recovery (M10, v = 0.5, both cutoffs)
    w("\n## 3. E2 parameter recovery by cell (M10 generator, v = 0.5; true parameters from "
      "the generator)\n")
    sc = m10g.generate_economics_scenario(m10g.EconomicsScenarioConfig(seed=SEED_M10))
    rows = []
    for cutoff in (24, 30):
        fits = ct.predict("E2", sc, cutoff, W, policy(sc.config)).fits
        for (seg, ch), fit in fits.cells.items():
            b = sc.truth.base_hazard(seg, ch)
            if fit.status == "ok":
                z = (fit.log_params - np.log([b, 0.5])) / np.sqrt(np.diag(fit.cov_log))
                zs = f"{z[0]:+.2f} / {z[1]:+.2f}"
            else:
                zs = "n/a"
            rows.append([cutoff, f"{seg} x {ch}", fit.n_events, fit.status,
                         f"{fit.params[0]:.4f} / {fit.params[1]:.3f}",
                         f"{fit.se[0]:.4f} / {fit.se[1]:.3f}" if fit.status == "ok" else "n/a",
                         f"{b:.4f} / 0.5", zs])
    w(table(["C", "cell", "events", "status", "b / v", "SE", "true", "z (log scale)"], rows))

    # 4. repeated draws and sample size (correct specification)
    w("\n## 4. E2 on the M10 generator across independent draws (C = 24, v = 0.5)\n")
    diffs, ex = [], []
    for seed in REPEAT_SEEDS:
        s2 = m10g.generate_economics_scenario(m10g.EconomicsScenarioConfig(seed=seed))
        o = orc.m10_oracle(s2, 24, W).frame
        p = ct.predict("E2", s2, 24, W, policy(s2.config)).frame["prediction"].reindex(o.index)
        diffs.append(float((p - o["oracle"]).mean()))
        ex.append(float(np.sqrt(((p - o["oracle"]) ** 2).mean())))
    d = np.asarray(diffs)
    w(table(["draws", "mean of mean(E2 - oracle)", "SD across draws", "z",
             "mean RMSE(E2 - oracle)"],
            [[d.size, f(float(d.mean()), 3), f(float(d.std(ddof=1)), 3),
              f(float(d.mean() / (d.std(ddof=1) / math.sqrt(d.size))), 3),
              f(float(np.mean(ex)), 3)]]))
    w("\nSample size (acquisition rates scaled; same parameters): the oracle excess should "
      "shrink roughly as `1/sqrt(n)`.\n")
    base = m10g.EconomicsScenarioConfig(seed=0)
    rows = []
    for k in SCALES:
        rm = []
        for seed in SCALE_SEEDS:
            cfg = replace(base, seed=seed, acquisitions_per_month=tuple(
                a * k for a in base.acquisitions_per_month))
            s2 = m10g.generate_economics_scenario(cfg)
            o = orc.m10_oracle(s2, 24, W).frame
            p = ct.predict("E2", s2, 24, W, policy(cfg)).frame["prediction"].reindex(o.index)
            rm.append((len(o), float(np.sqrt(((p - o["oracle"]) ** 2).mean()))))
        n_mean = float(np.mean([a for a, _ in rm]))
        rows.append([k, f"{n_mean:,.0f}", f(float(np.mean([b for _, b in rm])), 3),
                     f(float(np.std([b for _, b in rm], ddof=1)), 2)])
    w(table(["acquisition scale", "mean n (population)", "mean RMSE(E2 - oracle)",
             "SD"], rows))

    t6 = time.perf_counter()
    amendment_section(w, "5", "A6", "E2", "boundary_exception")
    TIMINGS["amendment A6 before/after (10 draws)"] = time.perf_counter() - t6
    t7 = time.perf_counter()
    amendment_section(w, "6", "A7", "E1", "e1_boundary_exception")
    TIMINGS["amendment A7 before/after (10 draws)"] = time.perf_counter() - t7
    t8 = time.perf_counter()
    boundary_bias_section(w)
    TIMINGS["v = 0 boundary bias vs sample size"] = time.perf_counter() - t8
    TIMINGS["total"] = time.perf_counter() - start
    w("\n## 8. Runtime (indicative, this machine)\n")
    w(table(["step", "seconds (cumulative)"], [[k, f"{v:.2f}"] for k, v in TIMINGS.items()]))

    head = [
        "# Milestone 11 P3 — E1 (sBG) and E2 (exponential-Gamma): fit, validation, oracle "
        "comparison",
        "",
        "Generated by `python scripts/report_m11_p3_contractual.py`. **Development seeds "
        "only** (pilot, main and fresh seeds untouched). Not final performance evidence.",
        "",
        table(["item", "value"], [
            ["protocol", f"{pp.PROTOCOL_ID} ({pp.PROTOCOL_STAGE})"],
            ["protocol digest / code digest",
             f"`{pp.protocol_digest()[:16]}…` / `{pp.protocol_code_digest()[:16]}…`"],
            ["git", git_state()],
            ["seeds", f"M10 {SEED_M10}; signals {SEED_SIGNALS}; repeated "
                      f"{REPEAT_SEEDS[0]}-{REPEAT_SEEDS[-1]}; scale "
                      f"{SCALE_SEEDS[0]}-{SCALE_SEEDS[-1]}"],
            ["python / numpy / scipy / pandas",
             f"{platform.python_version()} / {np.__version__} / {scipy.__version__} / "
             f"{pd.__version__}"],
        ]),
    ]
    OUTPUT.write_text("\n".join(head + lines) + "\n")
    print(f"wrote {OUTPUT} in {TIMINGS['total']:.1f} s")
    return 0


def amendment_section(w: Callable[[str], None], section: str, amendment: str, model: str,
                      flag: str) -> None:
    """Before/after on the PC2 development grid for one boundary amendment: ``model`` with
    the amendment's switch off (the prior rule) and on, same fits otherwise."""
    w(f"\n## {section}. Amendment {amendment} ({model} geometric boundary exception): "
      "before / after\n")
    w(f"M10 generator, heterogeneity grid, cutoffs 24 and 30, {len(A6_SEEDS)} independent "
      f"development draws ({A6_SEEDS[0]}-{A6_SEEDS[-1]}). *Before* = the rule before "
      f"{amendment}; *after* = with {amendment} (and every earlier amendment). Means over "
      "draws; counts are totals over draws (12 cells per fit). `qualified` = "
      "validation-landmark gates passed (`SELECTION`), never the holdout.\n")
    recs = []
    for seed in A6_SEEDS:
        for v in m10g_grid():
            sc = m10g.generate_economics_scenario(
                m10g.EconomicsScenarioConfig(seed=seed, frailty_variance=v))
            pol = policy(sc.config)
            for cutoff in pp.scenario("m10_contractual").evaluation_cutoffs:
                o = orc.m10_oracle(sc, cutoff, W).frame
                mu = o["oracle"].to_numpy(np.float64)
                r = lm.holdout_frame(sc, sc.config.n_months, cutoff, W, "contractual", "A",
                                     pol).set_index("customer_id")["realised"].reindex(
                                         o.index).to_numpy(np.float64)
                mse_o = float(np.mean((r - mu) ** 2))
                for exc in (False, True):
                    kw = {flag: exc}
                    pr = ct.predict(model, sc, cutoff, W, pol, **kw)
                    p = pr.frame["prediction"].reindex(o.index).to_numpy(np.float64)
                    va = ct.validate(model, sc, cutoff, W, pol, **kw)
                    recs.append({
                        "v": v, "C": cutoff, "after": exc, "seed": seed,
                        "excess": float(np.mean((p - mu) ** 2)) / mse_o,
                        "slope": slope_se(p, r)[0],
                        "citl": float(np.sum(p)) / float(np.sum(r)) - 1.0,
                        "fallback": pr.fits.status_counts().get("fallback", 0),
                        "boundary": len(pr.fits.boundary_cells),
                        "qualified": int(gate(va.candidate).qualified),
                        "vslope": va.candidate.calibration_slope})
    d = pd.DataFrame(recs)
    rows: list[list[object]] = []
    for key, g in d.groupby(["v", "C"], sort=True):
        b, a = g.loc[~g["after"]], g.loc[g["after"]]
        rows.append([*key,
                     f"{b['excess'].mean():.4f} -> {a['excess'].mean():.4f}",
                     f"{b['excess'].max():.4f} -> {a['excess'].max():.4f}",
                     f"{b['slope'].mean():.3f} -> {a['slope'].mean():.3f}",
                     f"{b['citl'].mean():+.4f} -> {a['citl'].mean():+.4f}",
                     f"{int(b['fallback'].sum())} -> {int(a['fallback'].sum())}",
                     int(a["boundary"].sum()),
                     f"{b['vslope'].mean():.3f} -> {a['vslope'].mean():.3f}",
                     f"{int(b['qualified'].sum())} -> {int(a['qualified'].sum())}"])
    w(table(["v", "C", "excess / MSE_oracle (mean)", "excess (max draw)",
             "holdout slope", "holdout CITL", "fallback cells", "v=0 boundary cells",
             "validation slope", "qualified draws"], rows))
    piv = d.pivot_table(index=["seed", "v", "C"], columns="after", values="excess")
    diff = (piv[True] - piv[False]).to_numpy()
    changed = diff[diff != 0]
    pos = d.loc[d["v"] > 0]
    piv_pos = pos.pivot_table(index=["seed", "v", "C"], columns="after", values="excess")
    dpos = (piv_pos[True] - piv_pos[False]).to_numpy()
    w(f"\nPaired change in excess / MSE_oracle (after - before) over the {diff.size} fits: "
      f"{changed.size} changed; v > 0: {int(np.sum(dpos != 0))} of {dpos.size} changed, "
      f"largest increase {float(np.max(dpos)):+.5f}, largest decrease "
      f"{float(np.min(dpos)):+.5f}, mean {float(np.mean(dpos)):+.5f}.")


def boundary_bias_section(w: Callable[[str], None]) -> None:
    """Aggregate bias against the oracle at v = 0 and v = 0.5, before and after A6/A7, at two
    sample sizes (C = 24)."""
    w("\n## 7. Boundary bias at v = 0 versus sample size\n")
    w(f"Mean over {len(BIAS_SEEDS)} draws ({BIAS_SEEDS[0]}-{BIAS_SEEDS[-1]}) of "
      "`mean(pred - oracle)` per customer (SE across draws; % of the oracle mean), C = 24. "
      "At v = 0 the dispersion parameter is on the boundary, its MLE is truncated at 0 and "
      "positively biased, so a small upward bias is expected; it should shrink as "
      "`n^-1/2`.\n")
    base = m10g.EconomicsScenarioConfig(seed=0)
    variants: tuple[tuple[str, str, dict[str, bool]], ...] = (
        ("E2 (A6)", "E2", {}), ("E2 before A6", "E2", {"boundary_exception": False}),
        ("E1 (A7)", "E1", {}), ("E1 before A7", "E1", {"e1_boundary_exception": False}))
    rows: list[list[object]] = []
    for v in (0.0, 0.5):
        for k in (1.0, 4.0):
            res: dict[str, list[float]] = {lab: [] for lab, _, _ in variants}
            om = []
            for seed in BIAS_SEEDS:
                cfg = replace(base, seed=seed, frailty_variance=v, acquisitions_per_month=tuple(
                    a * k for a in base.acquisitions_per_month))
                sc = m10g.generate_economics_scenario(cfg)
                o = orc.m10_oracle(sc, 24, W).frame
                om.append(float(o["oracle"].mean()))
                for lab, model, kw in variants:
                    p = ct.predict(model, sc, 24, W, policy(cfg), **kw).frame["prediction"]
                    res[lab].append(float((p.reindex(o.index) - o["oracle"]).mean()))
            m = float(np.mean(om))
            rows.append([v, k] + [
                f"{np.mean(x):+.3f} ({np.std(x, ddof=1) / math.sqrt(len(x)):.3f}; "
                f"{100 * np.mean(x) / m:+.2f}%)" for x in res.values()])
    w(table(["v", "acquisition scale"] + [lab for lab, _, _ in variants], rows))


def m10g_grid() -> tuple[float, ...]:
    return tuple(m10p.HETEROGENEITY_GRID)


if __name__ == "__main__":
    sys.exit(main())
