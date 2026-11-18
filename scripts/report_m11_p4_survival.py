"""Milestone 11 P4 record: E3 (landmark discrete-time hazard) and S1 (Cox) against the oracles.

Writes ``docs/evidence/m11-p4-survival-models.md``. **Development seeds only** (``<=
predictive_protocol.DEVELOPMENT_SEED_MAX``); pilot, main and fresh seeds are not used. Not
evidence of final model performance (P8): it records fit status, the temporal penalty choice,
tenure bands, dropped columns, selection eligibility and validation-gate inputs, holdout
comparisons with the P2 oracle of the same information tier, the amendment A8 before/after
on the full development grid, a descriptive preview of the PC3 contrast, and runtime. E3 is
the protocol model (with A8); S1 uses the full registry.

    python scripts/report_m11_p4_survival.py
"""

from __future__ import annotations

import math
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
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
from decision_intelligence.customer_value import survival as sv
from decision_intelligence.customer_value.features import PromotionPolicy
from decision_intelligence.customer_value.model_selection import gate
from decision_intelligence.data import customer_economics_scenarios as m10g
from decision_intelligence.data import predictive_ltv_scenarios as gen
from decision_intelligence.data.customer_economics_scenarios import CHANNELS

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "docs" / "evidence" / "m11-p4-survival-models.md"
SEED_M10 = 5301
SEED_SIGNALS = 5302
PREVIEW_SEEDS = tuple(range(5311, 5316))
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


def oracle(sc: Scenario, cutoff: int, horizon: int, tier: str) -> pd.DataFrame:
    if isinstance(sc, gen.SignalsScenario):
        return orc.signals_oracle(sc, cutoff, horizon, tier).frame
    return orc.m10_oracle(sc, cutoff, horizon).frame


def evaluate(name: str, sc: Scenario, cutoff: int, horizon: int, tier: str
             ) -> tuple[list[list[object]], list[list[object]]]:
    pol = policy(sc.config)
    o = timed("oracles", oracle, sc, cutoff, horizon, tier)
    hold = lm.holdout_frame(sc, sc.config.n_months, cutoff, horizon, "contractual", tier, pol)
    r = hold.set_index("customer_id")["realised"].reindex(o.index).to_numpy(np.float64)
    mu = o["oracle"].to_numpy(np.float64)
    mse_o = float(np.mean((r - mu) ** 2))
    e3 = timed("E3 fit (incl. tuning)", sv.fit_e3, sc, cutoff, horizon, tier, pol)
    val = timed("E3 validation", sv.validate_e3, sc, cutoff, horizon, tier, pol,
                penalty=e3.penalty)
    preds = {"E3": timed("E3 predict", sv.predict_e3, e3, sc).frame}
    s1 = timed("S1 fit", sv.fit_s1, sc, cutoff, horizon, tier, pol)
    preds["S1"] = timed("S1 predict", sv.predict_s1, s1, sc).frame
    if tier == "A" and horizon == W:
        preds["E2 (P3)"] = ct.predict("E2", sc, cutoff, horizon, pol).frame
    rows: list[list[object]] = []
    for model, frame in preds.items():
        p = frame["prediction"].reindex(o.index).to_numpy(np.float64)
        excess = float(np.mean((p - mu) ** 2))
        slope, se = slope_se(p, r)
        status = (e3.hazard.status if model == "E3" else s1.status if model == "S1"
                  else ", ".join(f"{k} {v}" for k, v in
                                 sorted(ct.predict("E2", sc, cutoff, horizon, pol)
                                        .fits.status_counts().items())))
        rows.append([name, cutoff, horizon, tier, model, status, len(p),
                     f(float(np.mean(p - mu)), 3), f(math.sqrt(excess), 3),
                     f(excess / mse_o, 3), f(float(np.mean((r - p) ** 2))), f(mse_o),
                     f(float(np.sum(p)) / float(np.sum(r)) - 1.0, 3),
                     f"{slope:.3f} ({se:.3f})"])
    cand = val.candidate
    gate_txt = ("not selection-eligible (no validation fit)" if cand is None else
                f"CITL {cand.calibration_in_the_large:+.3f}, slope "
                f"{cand.calibration_slope:.3f} -> "
                f"{'qualified' if gate(cand).qualified else 'not qualified'}")
    tuning = ("; ".join(f"{t.penalty:g}: {t.validation_mse:.1f}" for t in e3.tuning)
              or "—")
    meta = [[name, cutoff, horizon, tier, f"{e3.penalty:g}", e3.penalty_source, tuning,
             ", ".join(e3.hazard.bands), ", ".join(e3.hazard.design.duplicates) or "—",
             ", ".join(e3.hazard.design.standardiser.dropped) or "—",
             f"{e3.hazard.n_person_months:,} / {e3.hazard.n_events:,}",
             f"L={val.landmark}: {gate_txt}"]]
    return rows, meta


def main() -> int:
    start = time.perf_counter()
    lines: list[str] = []
    w = lines.append
    rows: list[list[object]] = []
    meta: list[list[object]] = []
    for v in m10p.HETEROGENEITY_GRID:
        cfg = m10g.EconomicsScenarioConfig(seed=SEED_M10, frailty_variance=v)
        msc = timed("generate", m10g.generate_economics_scenario, cfg)
        for cutoff in pp.scenario("m10_contractual").evaluation_cutoffs:
            r_, m_ = evaluate(f"m10 v={v}", msc, cutoff, W, "A")
            rows += r_
            meta += m_
    base = gen.SignalsScenarioConfig(seed=SEED_SIGNALS)
    for label, scfg in (("signals (PH)", base),
                        ("signals (NPH)", gen.signals_variant("non_proportional_hazards",
                                                              base))):
        ssc = timed("generate", gen.generate_signals_scenario, scfg)
        spec = pp.scenario("signals_contractual")
        pairs = [(c, W) for c in spec.evaluation_cutoffs] + [
            (c, h) for h, c in spec.secondary_horizon]
        for cutoff, horizon in pairs:
            for tier in spec.tiers:
                r_, m_ = evaluate(label, ssc, cutoff, horizon, tier)
                rows += r_
                meta += m_
    w("\n## 1. Fit, penalty and validation inputs (E3, protocol model with A8)\n")
    w("Penalty: temporal tuning on the validation landmark `L_v` from the tuning landmarks "
      "(tables as recorded by `L_v`; squared error of window value; ties to the larger "
      "penalty), else `DEFAULT_PENALTY`. Bands: merged tenure bands (events < "
      f"{pp.TENURE_BAND_MIN_EVENTS} join their lower neighbour). Validation inputs feed the "
      "frozen selection gates (`SELECTION`); the holdout is not used.\n")
    w(table(["scenario", "C", "W", "tier", "penalty", "source", "tuning (lambda: val MSE)",
             "tenure bands", "duplicate columns dropped", "SD-0 columns dropped",
             "person-months / events", "validation"], meta))
    w("\n## 2. Holdout comparison with the P2 oracle of the same tier\n")
    w("`excess` = `mean((pred - oracle)^2)`, the loss above the attainable floor for that "
      "tier. E2 (P3, Tier A) is shown for context.\n")
    w(table(["scenario", "C", "W", "tier", "model", "status", "n", "mean(pred - oracle)",
             "RMSE(pred - oracle)", "excess / MSE_oracle", "realised MSE", "oracle MSE",
             "holdout CITL", "holdout slope (HC SE)"], rows))
    t0 = time.perf_counter()
    preview_section(w)
    TIMINGS["PC3 preview (5 draws)"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    a8_section(w)
    TIMINGS["A8 before/after"] = time.perf_counter() - t0
    TIMINGS["total"] = time.perf_counter() - start
    w("\n## 5. Runtime (indicative, this machine)\n")
    w(table(["step", "seconds (cumulative)"], [[k, f"{v:.2f}"] for k, v in TIMINGS.items()]))
    head = [
        "# Milestone 11 P4 — E3 (landmark hazard, A8) and S1 (Cox): fit, tuning, oracle comparison",
        "",
        "Generated by `python scripts/report_m11_p4_survival.py`. **Development seeds only** "
        "(pilot, main and fresh seeds untouched). Not final performance evidence; the PC3 "
        "preview is descriptive and is not the pre-registered test.",
        "",
        table(["item", "value"], [
            ["protocol", f"{pp.PROTOCOL_ID} ({pp.PROTOCOL_STAGE})"],
            ["protocol digest / code digest",
             f"`{pp.protocol_digest()[:16]}…` / `{pp.protocol_code_digest()[:16]}…`"],
            ["git", git_state()],
            ["seeds", f"M10 {SEED_M10}; signals {SEED_SIGNALS}; preview "
                      f"{PREVIEW_SEEDS[0]}-{PREVIEW_SEEDS[-1]}"],
            ["python / numpy / scipy / pandas",
             f"{platform.python_version()} / {np.__version__} / {scipy.__version__} / "
             f"{pd.__version__}"],
        ]),
    ]
    OUTPUT.write_text("\n".join(head + lines) + "\n")
    print(f"wrote {OUTPUT} in {TIMINGS['total']:.1f} s")
    return 0


def a8_section(w: Callable[[str], None]) -> None:
    """Amendment A8 before/after on the full development grid: E3 with every registry feature
    (pre-A8, ``exclude_features=()``) and with ``E3_EXCLUDED_COVARIATES`` removed (A8)."""
    w("\n## 4. Amendment A8 (E3 covariates): before / after\n")
    w("*Before* = E3 with every registry feature (pre-A8); *after* = A8, without "
      f"`{', '.join(pp.E3_EXCLUDED_COVARIATES)}` (tenure enters through the tenure-band "
      "basis). Same data, seeds and penalty rule. Holdout calibration is reported for the "
      "record (never used for selection). **Flag**: `REGRESSION` if RMSE(pred - oracle) rises "
      "by more than 5%, a validation gate goes from qualified to not qualified, or the "
      "holdout |CITL| crosses the 5% selection tolerance upwards.\n")
    rows: list[list[object]] = []
    scenarios: dict[str, Scenario] = {}
    settings: list[tuple[str, int, int, str]] = []
    for v in m10p.HETEROGENEITY_GRID:
        scenarios[f"m10 v={v}"] = m10g.generate_economics_scenario(
            m10g.EconomicsScenarioConfig(seed=SEED_M10, frailty_variance=v))
        settings += [(f"m10 v={v}", c, W, "A") for c in (24, 30)]
    base = gen.SignalsScenarioConfig(seed=SEED_SIGNALS)
    for label, scfg in (("signals (PH)", base),
                        ("signals (NPH)", gen.signals_variant("non_proportional_hazards",
                                                              base))):
        scenarios[label] = gen.generate_signals_scenario(scfg)
        settings += [(label, c, h, tier) for c, h in ((36, W), (42, W), (33, 24))
                     for tier in ("A", "B")]
    tol = pp.SELECTION.calibration_in_the_large_tolerance
    n_flags = 0
    for name, cutoff, horizon, tier in settings:
        sc = scenarios[name]
        pol = policy(sc.config)
        o = oracle(sc, cutoff, horizon, tier)
        mu = o["oracle"].to_numpy(np.float64)
        hold = lm.holdout_frame(sc, sc.config.n_months, cutoff, horizon, "contractual", tier,
                                pol)
        r = hold.set_index("customer_id")["realised"].reindex(o.index).to_numpy(np.float64)
        out = []
        for exclude in ((), pp.E3_EXCLUDED_COVARIATES):
            m = sv.fit_e3(sc, cutoff, horizon, tier, pol, exclude_features=exclude)
            p = sv.predict_e3(m, sc).frame["prediction"].reindex(o.index).to_numpy(np.float64)
            val = sv.validate_e3(sc, cutoff, horizon, tier, pol, penalty=m.penalty,
                                 exclude_features=exclude)
            q = None if val.candidate is None else gate(val.candidate).qualified
            g = ("not eligible" if val.candidate is None else
                 f"{val.candidate.calibration_in_the_large:+.3f} "
                 f"{'qualified' if q else 'NOT qualified'}")
            out.append((float(np.mean(p - mu)), math.sqrt(float(np.mean((p - mu) ** 2))),
                        float(np.sum(p)) / float(np.sum(r)) - 1.0, slope_se(p, r)[0], g, q))
        (b0, r0, c0, s0, g0, q0), (b1, r1, c1, s1, g1, q1) = out
        flag = (r1 > 1.05 * r0 or (q0 is True and q1 is False)
                or (abs(c0) <= tol < abs(c1)))
        n_flags += int(flag)
        rows.append([name, cutoff, horizon, tier, f"{b0:+.2f} -> {b1:+.2f}",
                     f"{r0:.2f} -> {r1:.2f}", f"{c0:+.3f} -> {c1:+.3f}",
                     f"{s0:.3f} -> {s1:.3f}", f"{g0} -> {g1}",
                     "REGRESSION" if flag else ""])
    w(table(["scenario", "C", "W", "tier", "mean(pred - oracle)", "RMSE(pred - oracle)",
             "holdout CITL", "holdout slope", "validation CITL / gate", "flag"], rows))
    w(f"\nFlagged settings: {n_flags} of {len(settings)}.")


def preview_section(w: Callable[[str], None]) -> None:
    """Descriptive PC3 preview: E3 Tier B vs the Tier A models, realised MSE at C = 36 over
    independent development draws, with the oracle gap for reference."""
    w("\n## 3. Descriptive preview of the PC3 contrast (not the pre-registered test)\n")
    w(f"Signals scenario, C = 36, W = 12, {len(PREVIEW_SEEDS)} development draws "
      f"({PREVIEW_SEEDS[0]}-{PREVIEW_SEEDS[-1]}). `D` = realised MSE(model) - realised "
      "MSE(E3 Tier B); positive favours E3 Tier B. `oracle D` = MSE(oracle A) - MSE(oracle B), "
      "the population ceiling of the Tier B gain (P2).\n")
    recs: dict[str, list[float]] = {k: [] for k in ("E3 A", "E2 A", "E1 A", "S1 A", "oracle")}
    capture: list[float] = []
    for seed in PREVIEW_SEEDS:
        cfg = gen.SignalsScenarioConfig(seed=seed)
        sc = gen.generate_signals_scenario(cfg)
        pol = policy(cfg)
        oa = orc.signals_oracle(sc, 36, W, "A").frame
        ob = orc.signals_oracle(sc, 36, W, "B").frame
        hold = lm.holdout_frame(sc, cfg.n_months, 36, W, "contractual", "B", pol)
        r = hold.set_index("customer_id")["realised"].reindex(ob.index).to_numpy(np.float64)

        def mse(frame: pd.DataFrame, col: str = "prediction", ob: pd.DataFrame = ob,
                r: np.ndarray = r) -> float:
            p = frame[col].reindex(ob.index).to_numpy(np.float64)
            return float(np.mean((r - p) ** 2))

        b = mse(sv.predict_e3(sv.fit_e3(sc, 36, W, "B", pol), sc).frame)
        recs["E3 A"].append(mse(sv.predict_e3(sv.fit_e3(sc, 36, W, "A", pol), sc).frame) - b)
        recs["E2 A"].append(mse(ct.predict("E2", sc, 36, W, pol).frame) - b)
        recs["E1 A"].append(mse(ct.predict("E1", sc, 36, W, pol).frame) - b)
        recs["S1 A"].append(mse(sv.predict_s1(sv.fit_s1(sc, 36, W, "A", pol), sc).frame) - b)
        od = mse(oa, "oracle") - mse(ob, "oracle")
        recs["oracle"].append(od)
        capture.append(recs["E3 A"][-1] / od)
    rows = [[k, f"{np.mean(v):+.2f}", f"{np.std(v, ddof=1) / math.sqrt(len(v)):.2f}",
             ", ".join(f"{x:+.1f}" for x in v)] for k, v in recs.items()]
    w(table(["contrast (minus E3 Tier B)", "mean D", "SE", "per draw"], rows))
    w(f"\nShare of the oracle Tier B gain realised by E3 (E3 A - E3 B over oracle A - oracle "
      f"B): mean {np.mean(capture):.2f} (per draw "
      f"{', '.join(f'{c:.2f}' for c in capture)}).")


if __name__ == "__main__":
    sys.exit(main())
