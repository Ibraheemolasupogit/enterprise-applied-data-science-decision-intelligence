"""Milestone 11 P6 record: E5 (two-part penalised GLM) and S2 (split conformal intervals).

Writes ``docs/evidence/m11-p6-predictive-models.md``. **Development seeds only**; pilot, main
and fresh seeds are not used. Not evidence of final model performance (P8): it records E5's fit
status, penalty, design, validation inputs, holdout comparisons with the P2 oracle of the same
tier (E3 / E4 alongside), S2's interval half-widths and holdout coverage (reported, never
asserted: ``COVERAGE_REPORTING_RULE``), a descriptive PC5 preview, an extrapolation diagnostic
for the owner's decision, and runtime.

    python scripts/report_m11_p6_predictive.py
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

from decision_intelligence.customer_value import btyd
from decision_intelligence.customer_value import landmarks as lm
from decision_intelligence.customer_value import oracle as orc
from decision_intelligence.customer_value import precision as prec
from decision_intelligence.customer_value import predictive as pr
from decision_intelligence.customer_value import predictive_protocol as pp
from decision_intelligence.customer_value import protocol as m10p
from decision_intelligence.customer_value import purchase_oracle as po
from decision_intelligence.customer_value import survival as sv
from decision_intelligence.customer_value.features import PromotionPolicy
from decision_intelligence.customer_value.model_selection import gate
from decision_intelligence.data import customer_economics_scenarios as m10g
from decision_intelligence.data import predictive_ltv_scenarios as gen
from decision_intelligence.data.customer_economics_scenarios import CHANNELS

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "docs" / "evidence" / "m11-p6-predictive-models.md"
SEED_M10 = 5601
SEED_SIGNALS = 5602
SEED_PURCHASES = 5603
PREVIEW_SEEDS = tuple(range(5611, 5616))
W = pp.PRIMARY_HORIZON
TIMINGS: dict[str, float] = {}
P = ParamSpec("P")
T = TypeVar("T")
Scenario = m10g.EconomicsScenario | gen.SignalsScenario | gen.PurchasesScenario


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


POL = PromotionPolicy(2, tuple(c for c, p in zip(CHANNELS, (False, True, True, False),
                                                 strict=True) if p))


def oracle_frame(sc: Scenario, cutoff: int, horizon: int, tier: str) -> pd.DataFrame:
    if isinstance(sc, gen.PurchasesScenario):
        return po.purchases_oracle(sc, cutoff, horizon).frame
    if isinstance(sc, gen.SignalsScenario):
        return orc.signals_oracle(sc, cutoff, horizon, tier).frame
    return orc.m10_oracle(sc, cutoff, horizon).frame


def settings() -> list[tuple[str, Scenario, str, int, int, str]]:
    out: list[tuple[str, Scenario, str, int, int, str]] = []
    for v in m10p.HETEROGENEITY_GRID:
        sc: Scenario = m10g.generate_economics_scenario(
            m10g.EconomicsScenarioConfig(seed=SEED_M10, frailty_variance=v))
        out += [(f"m10 v={v}", sc, "contractual", c, W, "A") for c in (24, 30)]
    base = gen.SignalsScenarioConfig(seed=SEED_SIGNALS)
    for label, cfg in (("signals (PH)", base),
                       ("signals (NPH)", gen.signals_variant("non_proportional_hazards",
                                                             base))):
        sc = gen.generate_signals_scenario(cfg)
        out += [(label, sc, "contractual", c, h, tier) for c, h in ((36, W), (42, W), (33, 24))
                for tier in ("A", "B")]
    pbase = gen.PurchasesScenarioConfig(seed=SEED_PURCHASES)
    for label, pcfg in [("purchases (base)", pbase)] + [
            (f"purchases ({v})", gen.purchases_variant(v, pbase))
            for v in gen.PURCHASES_VARIANTS]:
        sc = gen.generate_purchases_scenario(pcfg)
        out += [(label, sc, "non_contractual", c, W, "A") for c in (24, 30)]
    return out


def main() -> int:
    start = time.perf_counter()
    lines: list[str] = []
    w = lines.append
    meta: list[list[object]] = []
    rows: list[list[object]] = []
    cover: list[list[object]] = []
    for name, sc, kind, cutoff, horizon, tier in settings():
        pol = POL if kind == "contractual" else None
        o = timed("oracles", oracle_frame, sc, cutoff, horizon, tier)
        mu = o["oracle"].to_numpy(np.float64)
        hold = lm.holdout_frame(sc, sc.config.n_months, cutoff, horizon, kind, tier, pol)
        r = hold.set_index("customer_id")["realised"].reindex(o.index).to_numpy(np.float64)
        mse_o = float(np.mean((r - mu) ** 2))
        e5 = timed("E5 fit (incl. tuning)", pr.fit_e5, sc, cutoff, horizon, kind, tier, pol)
        val = timed("E5 validation", pr.validate_e5, sc, cutoff, horizon, kind, tier, pol,
                    penalty=e5.penalty)
        preds: dict[str, np.ndarray] = {
            "E5": timed("E5 predict", pr.predict_e5, e5, sc).frame["prediction"]
            .reindex(o.index).to_numpy(np.float64)}
        if kind == "contractual" and not isinstance(sc, gen.PurchasesScenario):
            e3 = sv.fit_e3(sc, cutoff, horizon, tier, POL)
            preds["E3 (P4, A8)"] = sv.predict_e3(e3, sc).frame["prediction"].reindex(
                o.index).to_numpy(np.float64)
        elif isinstance(sc, gen.PurchasesScenario):
            preds["E4 (P5)"] = btyd.predict(sc, cutoff, horizon).frame["prediction"].reindex(
                o.index).to_numpy(np.float64)
        for model, pv in preds.items():
            excess = float(np.mean((pv - mu) ** 2))
            slope, se = slope_se(pv, r)
            rows.append([name, cutoff, horizon, tier, model, f(float(np.mean(pv - mu)), 3),
                         f(math.sqrt(excess), 3), f(excess / mse_o, 3),
                         f(float(np.max(pv)), 4), f(float(np.max(mu)), 4),
                         f(float(np.sum(pv)) / float(np.sum(r)) - 1.0, 3),
                         f"{slope:.3f} ({se:.3f})"])
        cand = val.candidate
        gate_txt = ("not selection-eligible (no validation fit)" if cand is None else
                    f"CITL {cand.calibration_in_the_large:+.3f}, slope "
                    f"{cand.calibration_slope:.3f} -> "
                    f"{'qualified' if gate(cand).qualified else 'not qualified'}")
        tuning = "; ".join(f"{t.penalty:g}: {t.validation_mse:.1f}" for t in e5.tuning) or "—"
        meta.append([name, cutoff, horizon, tier, f"{e5.penalty:g}", e5.penalty_source,
                     tuning, f"{e5.positive.status} / {e5.amount.status}",
                     ", ".join(e5.positive.design.duplicates) or "—",
                     ", ".join(e5.positive.design.standardiser.dropped) or "—",
                     f"{e5.negative_windows} / {e5.negative_mass:+.4f}", gate_txt])
        if horizon == W:
            s2 = timed("S2", pr.fit_s2, sc, cutoff, horizon, kind, tier, pol,
                       penalty=e5.penalty)
            fr = s2.frame.reindex(o.index)
            cells: list[object] = [name, cutoff, tier, s2.n_calibration]
            for level in pp.CALIBRATION.interval_levels:
                inside = (r >= fr[f"lower_{level}"].to_numpy()) & (
                    r <= fr[f"upper_{level}"].to_numpy())
                k, n = int(inside.sum()), int(inside.size)
                cells += [f(s2.quantiles[level], 4), f"{k / n:.3f}",
                          prec.classify_coverage(k, n, level)]
            cover.append(cells)
    w("\n## 1. E5 fit, penalty, design and validation inputs\n")
    w("One penalty for both parts (temporal tuning on the validation landmark by squared error "
      "of window value, ties to the larger penalty; else `DEFAULT_PENALTY`). Status: "
      "positive part / amount part. Negative windows: count in the training stack / mean of "
      "`min(R, 0)` (the mass the two-part mean omits).\n")
    w(table(["scenario", "C", "W", "tier", "penalty", "source", "tuning (lambda: val MSE)",
             "status", "duplicates dropped", "SD-0 dropped", "negative windows / mass",
             "validation"], meta))
    w("\n## 2. Holdout comparison with the P2 oracle of the same tier\n")
    w("`excess` = `mean((pred - oracle)^2)`. E3 (P4, with A8) and E4 (P5) are shown for "
      "context. `max pred` vs `max oracle` shows extrapolation.\n")
    w(table(["scenario", "C", "W", "tier", "model", "mean(pred - oracle)",
             "RMSE(pred - oracle)", "excess / MSE_oracle", "max pred", "max oracle",
             "holdout CITL", "holdout slope (HC SE)"], rows))
    w("\n## 3. S2 split conformal intervals: holdout coverage (reported, not asserted)\n")
    w("Half-width `q` per level from one latest landmark per calibration customer; coverage of "
      "the realised window value over the holdout customers at `C`. Labels from "
      "`precision.classify_coverage` (exact binomial, as `COVERAGE_REPORTING_RULE` "
      "prescribes) treat customers as independent, which they are not (they share one `q`); "
      "the guarantee needs exchangeability, which landmark windows before `C` and the window "
      "after `C` do not have.\n")
    w(table(["scenario", "C", "tier", "calibration customers", "q (80%)", "coverage (80%)",
             "label (80%)", "q (95%)", "coverage (95%)", "label (95%)"], cover))
    t0 = time.perf_counter()
    preview_section(w)
    TIMINGS["PC5 preview (5 draws)"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    diagnostic_section(w)
    TIMINGS["extrapolation diagnostic"] = time.perf_counter() - t0
    TIMINGS["total"] = time.perf_counter() - start
    w("\n## 6. Runtime (indicative, this machine)\n")
    w(table(["step", "seconds (cumulative)"], [[k, f"{v:.2f}"] for k, v in TIMINGS.items()]))
    head = [
        "# Milestone 11 P6 — E5 (two-part penalised GLM) and S2 (split conformal)",
        "",
        "Generated by `python scripts/report_m11_p6_predictive.py`. **Development seeds only** "
        "(pilot, main and fresh seeds untouched). Not final performance evidence; the PC5 "
        "preview is descriptive, not the pre-registered test.",
        "",
        table(["item", "value"], [
            ["protocol", f"{pp.PROTOCOL_ID} ({pp.PROTOCOL_STAGE})"],
            ["protocol digest / code digest",
             f"`{pp.protocol_digest()[:16]}…` / `{pp.protocol_code_digest()[:16]}…`"],
            ["git", git_state()],
            ["seeds", f"M10 {SEED_M10}; signals {SEED_SIGNALS}; purchases {SEED_PURCHASES}; "
                      f"preview {PREVIEW_SEEDS[0]}-{PREVIEW_SEEDS[-1]}"],
            ["python / numpy / scipy / pandas",
             f"{platform.python_version()} / {np.__version__} / {scipy.__version__} / "
             f"{pd.__version__}"],
        ]),
    ]
    OUTPUT.write_text("\n".join(head + lines) + "\n")
    print(f"wrote {OUTPUT} in {TIMINGS['total']:.1f} s")
    return 0


def preview_section(w: Callable[[str], None]) -> None:
    w("\n## 4. Descriptive preview of the PC5 contrast (not the pre-registered test)\n")
    w(f"Signals scenario, Tier B, C = 36, W = 12, {len(PREVIEW_SEEDS)} development draws "
      f"({PREVIEW_SEEDS[0]}-{PREVIEW_SEEDS[-1]}). `D` = realised MSE(E5) - realised MSE(E3); "
      "negative favours E5.\n")
    rows: list[list[object]] = []
    ds: list[float] = []
    for seed in PREVIEW_SEEDS:
        sc = gen.generate_signals_scenario(gen.SignalsScenarioConfig(seed=seed))
        o = orc.signals_oracle(sc, 36, W, "B").frame
        r = lm.holdout_frame(sc, sc.config.n_months, 36, W, "contractual", "B", POL).set_index(
            "customer_id")["realised"].reindex(o.index).to_numpy(np.float64)
        p5 = pr.predict_e5(pr.fit_e5(sc, 36, W, "contractual", "B", POL), sc).frame[
            "prediction"].reindex(o.index).to_numpy(np.float64)
        p3 = sv.predict_e3(sv.fit_e3(sc, 36, W, "B", POL), sc).frame["prediction"].reindex(
            o.index).to_numpy(np.float64)
        d = float(np.mean((r - p5) ** 2) - np.mean((r - p3) ** 2))
        ds.append(d)
        rows.append([seed, f"{d:+.2f}"])
    arr = np.asarray(ds)
    w(table(["seed", "D"], rows))
    w(f"\nMean D {arr.mean():+.2f} (SE {arr.std(ddof=1) / math.sqrt(arr.size):.2f}).")


def diagnostic_section(w: Callable[[str], None]) -> None:
    """Not protocol: E5 with features clipped to the training landmarks' range at prediction,
    and (contractual) without A8's three covariates, for the owner's decision."""
    w("\n## 5. Extrapolation diagnostic (NOT the protocol model; for an owner decision)\n")
    w("E5 (penalty 1) as specified, with each numeric feature at `C` clipped to its range on "
      "the training landmarks, and (contractual) without "
      f"`{', '.join(pp.E3_EXCLUDED_COVARIATES)}` (A8's E3 exclusion, which does not apply to "
      "E5).\n")
    cases: list[tuple[str, Scenario, str, int, str]] = [
        ("signals (PH), Tier B", gen.generate_signals_scenario(
            gen.SignalsScenarioConfig(seed=SEED_SIGNALS)), "contractual", 36, "B"),
        ("m10 v=1.0, Tier A", m10g.generate_economics_scenario(
            m10g.EconomicsScenarioConfig(seed=SEED_M10, frailty_variance=1.0)),
         "contractual", 24, "A"),
        ("purchases (base)", gen.generate_purchases_scenario(
            gen.PurchasesScenarioConfig(seed=SEED_PURCHASES)), "non_contractual", 24, "A"),
        ("purchases (base)", gen.generate_purchases_scenario(
            gen.PurchasesScenarioConfig(seed=SEED_PURCHASES)), "non_contractual", 30, "A"),
    ]
    rows: list[list[object]] = []
    for name, sc, kind, cutoff, tier in cases:
        pol = POL if kind == "contractual" else None
        mu = oracle_frame(sc, cutoff, W, tier)["oracle"]
        stack = lm.training_stack(sc, cutoff, W, kind, tier, pol)
        feats = pr._features(sc, cutoff, kind, tier, pol)
        variants: list[tuple[str, Sequence[str], bool]] = [("as specified", (), False),
                                                          ("clipped to training range", (),
                                                           True)]
        if kind == "contractual":
            variants.append(("without A8's covariates", pp.E3_EXCLUDED_COVARIATES, False))
        for label, exclude, clip in variants:
            p1, p2 = pr.fit_parts(stack, kind, tier, 1.0, exclude)
            x = feats.copy()
            if clip:
                for c in x.columns:
                    if c not in pr.CATEGORICAL:
                        x[c] = x[c].clip(stack[c].min(), stack[c].max())
            p, m = pr._predict_parts(p1, p2, x)
            d = p * m - mu.reindex(feats.index).to_numpy(np.float64)
            rows.append([name, cutoff, label, f"{d.mean():+.2f}",
                         f"{math.sqrt(float(np.mean(d ** 2))):.2f}", f"{np.max(p * m):.1f}"])
    w(table(["setting", "C", "E5 variant", "mean(pred - oracle)", "RMSE(pred - oracle)",
             "max pred"], rows))


if __name__ == "__main__":
    sys.exit(main())
