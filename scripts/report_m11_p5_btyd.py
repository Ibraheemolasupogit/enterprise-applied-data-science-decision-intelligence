"""Milestone 11 P5 record: E4 (BG/NBD + Gamma-Gamma, per cell) against the P2 oracle.

Writes ``docs/evidence/m11-p5-btyd-models.md``. **Development seeds only**; pilot, main and fresh
seeds are not used. Not evidence of final model performance (P8): it records fit status,
validation-gate inputs, holdout comparisons with the oracle for the base scenario and the
predeclared misspecification variants (S4), the pooled misspecified reference
(``E4_POOLED_REFERENCE``), parameter recovery, repeated draws, sample size and runtime.

    python scripts/report_m11_p5_btyd.py
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

from decision_intelligence.customer_value import btyd
from decision_intelligence.customer_value import landmarks as lm
from decision_intelligence.customer_value import predictive_protocol as pp
from decision_intelligence.customer_value import purchase_oracle as po
from decision_intelligence.customer_value.model_selection import gate
from decision_intelligence.data import predictive_ltv_scenarios as gen
from decision_intelligence.data.customer_economics_scenarios import CHANNELS, SEGMENTS

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "docs" / "evidence" / "m11-p5-btyd-models.md"
SEED = 5401
REPEAT_SEEDS = tuple(range(5411, 5421))
SCALE_SEEDS = tuple(range(5431, 5436))
SCALES = (0.25, 1.0, 4.0)
GATE_SEEDS = tuple(range(5441, 5447))
W = pp.PRIMARY_HORIZON
TIMINGS: dict[str, float] = {}
P = ParamSpec("P")
T = TypeVar("T")


def table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def f(x: float, digits: int = 4) -> str:
    return "nan" if not math.isfinite(x) else f"{x:.{digits}g}"


def slope_se(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    x = pred - pred.mean()
    sxx = float(np.sum(x * x))
    slope = float(np.sum(x * (target - target.mean())) / sxx)
    e = target - target.mean() - slope * x
    return slope, math.sqrt(float(np.sum(x * x * e * e))) / sxx


def timed(label: str, fn: Callable[P, T], *args: P.args, **kw: P.kwargs) -> T:
    t = time.perf_counter()
    out = fn(*args, **kw)
    TIMINGS[label] = TIMINGS.get(label, 0.0) + time.perf_counter() - t
    return out


def git_state() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, check=True,
                              capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, check=True,
                               capture_output=True, text=True).stdout.strip().splitlines()
        return f"{head} + {len(dirty)} uncommitted or untracked paths" if dirty else head
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def scenarios() -> list[tuple[str, gen.PurchasesScenario]]:
    base = gen.PurchasesScenarioConfig(seed=SEED)
    out = [("base (BG/NBD + Gamma-Gamma)", gen.generate_purchases_scenario(base))]
    for v in gen.PURCHASES_VARIANTS:
        out.append((f"variant: {v}", gen.generate_purchases_scenario(
            gen.purchases_variant(v, base))))
    return out


def main() -> int:
    start = time.perf_counter()
    lines: list[str] = []
    w = lines.append
    rows: list[list[object]] = []
    meta: list[list[object]] = []
    for name, sc in scenarios():
        for cutoff in pp.scenario("purchases_noncontractual").evaluation_cutoffs:
            o = timed("oracle", po.purchases_oracle, sc, cutoff, W).frame
            hold = lm.holdout_frame(sc, sc.config.n_months, cutoff, W, "non_contractual", "A")
            r = hold.set_index("customer_id")["realised"].reindex(o.index).to_numpy(np.float64)
            mu = o["oracle"].to_numpy(np.float64)
            mse_o = float(np.mean((r - mu) ** 2))
            data = timed("data", btyd.purchase_data, sc, cutoff, W)
            fits = timed("E4 fit (12 cells + pooled)", btyd.fit_cells, data.frame)
            val = timed("E4 validation", btyd.validate, sc, cutoff, W)
            for label, pooled in (("E4 (per cell)", False),
                                  ("pooled reference (misspecified)", True)):
                p = timed("E4 predict", btyd.predict_from, data, fits, pooled=pooled).frame
                pv = p["prediction"].reindex(o.index).to_numpy(np.float64)
                excess = float(np.mean((pv - mu) ** 2))
                slope, se = slope_se(pv, r)
                pa = p["p_alive"].reindex(o.index).to_numpy(np.float64)
                rows.append([name, cutoff, label, len(pv), f(float(np.mean(pv - mu)), 3),
                             f(math.sqrt(excess), 3), f(excess / mse_o, 3),
                             f(float(np.mean(pa - o["p_alive"].to_numpy())), 3),
                             f(float(np.mean((r - pv) ** 2))), f(mse_o),
                             f(float(np.sum(pv)) / float(np.sum(r)) - 1.0, 3),
                             f"{slope:.3f} ({se:.3f})"])
            cand = val.candidate
            meta.append([name, cutoff,
                         ", ".join(f"{k} {v}" for k, v in sorted(fits.status_counts().items())),
                         f"{fits.pooled_bgnbd.status} / {fits.pooled_gg.status}",
                         f"{data.contribution_ratio:.5f}",
                         f"L={val.landmark}: CITL {cand.calibration_in_the_large:+.3f}, slope "
                         f"{cand.calibration_slope:.3f} -> "
                         f"{'qualified' if gate(cand).qualified else 'not qualified'}"])
    w("\n## 1. Fit status and validation inputs\n")
    w("Status per component (BG/NBD, Gamma-Gamma) over the 12 cells; `fallback` = pooled fit "
      "of that component (the only allowed fallback). Validation: the fit with the "
      "information of the validation landmark only, scored on its completely recorded labels "
      "(`SELECTION` gates); the holdout is never used.\n")
    w(table(["scenario", "C", "cells (status)", "pooled BG/NBD / GG", "contribution ratio",
             "validation"], meta))
    w("\n## 2. Holdout comparison with the P2 oracle\n")
    w("`excess` = `mean((pred - oracle)^2)`, the loss above the attainable floor. The pooled "
      "fit is the labelled misspecified reference (`E4_POOLED_REFERENCE`), not a selection "
      "candidate. The two variants are the predeclared misspecifications (S4); E4 is "
      "misspecified there by design.\n")
    w(table(["scenario", "C", "model", "n", "mean(pred - oracle)", "RMSE(pred - oracle)",
             "excess / MSE_oracle", "mean(P(alive) - oracle)", "realised MSE", "oracle MSE",
             "holdout CITL", "holdout slope (HC SE)"], rows))
    t0 = time.perf_counter()
    recovery_section(w)
    TIMINGS["recovery"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    repeat_section(w)
    TIMINGS["repeated draws and sample size"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    gate_power_section(w)
    TIMINGS["validation-gate check (oracle vs E4)"] = time.perf_counter() - t0
    TIMINGS["total"] = time.perf_counter() - start
    w("\n## 6. Runtime (indicative, this machine)\n")
    w(table(["step", "seconds (cumulative)"], [[k, f"{v:.2f}"] for k, v in TIMINGS.items()]))
    head = [
        "# Milestone 11 P5 — E4 (BG/NBD + Gamma-Gamma): fit, validation, oracle comparison",
        "",
        "Generated by `python scripts/report_m11_p5_btyd.py`. **Development seeds only** "
        "(pilot, main and fresh seeds untouched). Not final performance evidence.",
        "",
        table(["item", "value"], [
            ["protocol", f"{pp.PROTOCOL_ID} ({pp.PROTOCOL_STAGE})"],
            ["protocol digest / code digest",
             f"`{pp.protocol_digest()[:16]}…` / `{pp.protocol_code_digest()[:16]}…`"],
            ["git", git_state()],
            ["seeds", f"{SEED}; repeated {REPEAT_SEEDS[0]}-{REPEAT_SEEDS[-1]}; scale "
                      f"{SCALE_SEEDS[0]}-{SCALE_SEEDS[-1]}"],
            ["python / numpy / scipy / pandas",
             f"{platform.python_version()} / {np.__version__} / {scipy.__version__} / "
             f"{pd.__version__}"],
        ]),
    ]
    OUTPUT.write_text("\n".join(head + lines) + "\n")
    print(f"wrote {OUTPUT} in {TIMINGS['total']:.1f} s")
    return 0


def gate_power_section(w: Callable[[str], None]) -> None:
    """The frozen selection gates applied to the exact oracle at the validation landmark: how
    often does the attainable ceiling itself pass?"""
    lo, hi = pp.SELECTION.slope_bounds
    tol = pp.SELECTION.calibration_in_the_large_tolerance
    w("\n## 5. The frozen validation gates applied to the oracle\n")
    w(f"Base scenario, {len(GATE_SEEDS)} development draws ({GATE_SEEDS[0]}-{GATE_SEEDS[-1]}). "
      "At each validation landmark `L_v` the P2 oracle (information recorded by `L_v`, true "
      "parameters) and E4 (fitted with the information of `L_v`) are scored on `L_v`'s "
      f"labels with the frozen gates: slope in [{lo}, {hi}], |CITL| <= {tol}. The slope's HC "
      "SE shows the precision of the gate itself.\n")
    rows: list[list[object]] = []
    passes = {"oracle": 0, "E4": 0}
    for seed in GATE_SEEDS:
        sc = gen.generate_purchases_scenario(gen.PurchasesScenarioConfig(seed=seed))
        for cutoff in pp.scenario("purchases_noncontractual").evaluation_cutoffs:
            lv = pp.validation_landmark(cutoff, W)
            assert lv is not None
            o = po.purchases_oracle(sc, lv, W).frame
            lab = lm.training_stack(sc, cutoff, W, "non_contractual", "A",
                                    landmarks=[lv]).set_index("customer_id")["realised"]
            r = lab.reindex(o.index).to_numpy(np.float64)
            e4 = btyd.validate(sc, cutoff, W).frame["prediction"].reindex(o.index)
            cells = []
            for name, pv in (("oracle", o["oracle"].to_numpy(np.float64)),
                             ("E4", e4.to_numpy(np.float64))):
                slope, se = slope_se(pv, r)
                citl = float(np.sum(pv)) / float(np.sum(r)) - 1.0
                ok = lo <= slope <= hi and abs(citl) <= tol
                passes[name] += int(ok)
                cells.append(f"{slope:.3f} ({se:.3f}), {citl:+.3f} -> "
                             f"{'pass' if ok else 'FAIL'}")
            rows.append([seed, cutoff, lv, len(o), *cells])
    w(table(["seed", "C", "L_v", "n", "oracle: slope (SE), CITL", "E4: slope (SE), CITL"],
            rows))
    n = len(rows)
    w(f"\nGate passed: oracle {passes['oracle']} of {n}; E4 {passes['E4']} of {n}.")


def recovery_section(w: Callable[[str], None]) -> None:
    w("\n## 3. Parameter recovery by cell (base scenario, C = 30; true parameters from the "
      "generator)\n")
    sc = gen.generate_purchases_scenario(gen.PurchasesScenarioConfig(seed=SEED))
    cfg = sc.config
    fits = btyd.fit_cells(btyd.purchase_data(sc, 30, W).frame)
    rows: list[list[object]] = []
    for (s, c), bf in fits.bgnbd.items():
        si, ci = SEGMENTS.index(s), CHANNELS.index(c)
        gf = fits.gg[(s, c)]
        out = []
        for fit, true in ((bf, (cfg.r, cfg.alpha * cfg.segment_alpha_scale[si],
                                cfg.dropout_a, cfg.dropout_b[ci])),
                          (gf, (cfg.gg_p, cfg.gg_q, cfg.gg_gamma[si]))):
            z = ((fit.log_params - np.log(true)) / np.sqrt(np.diag(fit.cov_log))
                 if fit.status == "ok" else np.full(len(true), np.nan))
            out.append((fit, z))
        (bfit, bz), (gfit, gz) = out
        rows.append([f"{s} x {c}", bfit.n_events, bfit.status,
                     " / ".join(f"{v:.3g}" for v in bfit.params),
                     " / ".join(f"{v:+.2f}" for v in bz), gfit.status,
                     " / ".join(f"{v:.3g}" for v in gfit.params),
                     " / ".join(f"{v:+.2f}" for v in gz)])
    w(table(["cell", "repeat purchases", "BG/NBD", "r / alpha / a / b", "z (log scale)",
             "GG", "p / q / gamma", "z (log scale)"], rows))


def repeat_section(w: Callable[[str], None]) -> None:
    w("\n## 4. Repeated draws and sample size (base scenario, C = 24)\n")
    rows: list[list[object]] = []
    for label, pooled in (("E4 (per cell)", False), ("pooled reference", True)):
        diffs, ex = [], []
        for seed in REPEAT_SEEDS:
            sc = gen.generate_purchases_scenario(gen.PurchasesScenarioConfig(seed=seed))
            o = po.purchases_oracle(sc, 24, W).frame
            p = btyd.predict(sc, 24, W, pooled=pooled).frame["prediction"].reindex(o.index)
            diffs.append(float((p - o["oracle"]).mean()))
            ex.append(math.sqrt(float(((p - o["oracle"]) ** 2).mean())))
        d = np.asarray(diffs)
        rows.append([label, d.size, f(float(d.mean()), 3), f(float(d.std(ddof=1)), 3),
                     f(float(d.mean() / (d.std(ddof=1) / math.sqrt(d.size))), 3),
                     f(float(np.mean(ex)), 3)])
    w(table(["model", "draws", "mean of mean(pred - oracle)", "SD", "z",
             "mean RMSE(pred - oracle)"], rows))
    w("\nSample size (acquisition rates scaled; same parameters): RMSE(pred - oracle), mean "
      f"over {len(SCALE_SEEDS)} draws.\n")
    base = gen.PurchasesScenarioConfig(seed=0)
    srows: list[list[object]] = []
    for k in SCALES:
        cell: list[float] = []
        pool: list[float] = []
        ns: list[int] = []
        for seed in SCALE_SEEDS:
            cfg = replace(base, seed=seed, acquisitions_per_month=tuple(
                a * k for a in base.acquisitions_per_month))
            sc = gen.generate_purchases_scenario(cfg)
            o = po.purchases_oracle(sc, 24, W).frame
            data = btyd.purchase_data(sc, 24, W)
            fits = btyd.fit_cells(data.frame)
            for store, pooled in ((cell, False), (pool, True)):
                p = btyd.predict_from(data, fits, pooled=pooled).frame["prediction"]
                store.append(math.sqrt(float(((p.reindex(o.index) - o["oracle"]) ** 2)
                                             .mean())))
            ns.append(len(o))
        srows.append([k, f"{np.mean(ns):,.0f}", f(float(np.mean(cell)), 3),
                      f(float(np.mean(pool)), 3)])
    w(table(["acquisition scale", "mean n", "E4 (per cell)", "pooled reference"], srows))


if __name__ == "__main__":
    sys.exit(main())
