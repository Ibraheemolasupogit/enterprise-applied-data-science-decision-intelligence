"""Milestone 11 P7 record: baselines, evaluation, uncertainty and selection reporting.

Writes ``docs/evidence/m11-p7-evaluation.md``. **Development seeds only**; pilot, main and fresh
seeds are not used. Not evidence of final model performance (P8): it records, on one
development draw per scenario, every P7 metric for every model, the selection reports
(``SELECTION`` with A9 and A10), parameter and predictive interval coverage (reported, never
asserted: ``COVERAGE_REPORTING_RULE``), CR1 versus model-based standard errors, the S3 IPCW
demonstration and runtime.

    python scripts/report_m11_p7_evaluation.py
"""

from __future__ import annotations

import math
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import ParamSpec, TypeVar, cast

import numpy as np
import pandas as pd
import scipy
from scipy import stats

from decision_intelligence.customer_value import btyd
from decision_intelligence.customer_value import contractual as ct
from decision_intelligence.customer_value import landmarks as lm
from decision_intelligence.customer_value import oracle as orc
from decision_intelligence.customer_value import predictive as pr
from decision_intelligence.customer_value import predictive_baselines as pb
from decision_intelligence.customer_value import predictive_evaluation as ev
from decision_intelligence.customer_value import predictive_protocol as pp
from decision_intelligence.customer_value import purchase_oracle as po
from decision_intelligence.customer_value import survival as sv
from decision_intelligence.customer_value.features import PromotionPolicy
from decision_intelligence.data import customer_economics_scenarios as m10g
from decision_intelligence.data import predictive_ltv_scenarios as gen
from decision_intelligence.data.customer_economics_scenarios import CHANNELS
from decision_intelligence.statistics import predictive_metrics as pm

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "docs" / "evidence" / "m11-p7-evaluation.md"
SEED_M10 = 5701
SEED_PURCHASES = 5702
SEED_SIGNALS = 5703
SEED_BOOTSTRAP = 5704
SEED_SIMULATION = 5705
SEED_CENSORING = 5706
SEED_PIT = 5707
SEED_NOISE = 5708  # A11 residual draws (a separate stream)
CONFIRMATION_SEEDS = (5711, 5712, 5713)  # A11 confirmation on new development draws
DRAWS = pp.CLUSTERED_INFERENCE.refit_bootstrap_resamples  # 200 parametric draws
W = pp.PRIMARY_HORIZON
LEVELS = pp.CALIBRATION.interval_levels
POL = PromotionPolicy(2, tuple(c for c, p in zip(CHANNELS, (False, True, True, False),
                                                 strict=True) if p))
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


def point_row(name: str, pred: np.ndarray, real: np.ndarray, oracle: np.ndarray) -> list[str]:
    ok = np.isfinite(pred)
    if not ok.all():
        return [name, f"{int((~ok).sum())} undefined"] + ["-"] * 8
    line = pm.calibration_line(pred, real)
    return [name, f(pm.relative_bias(pred, real), 3), f(pm.mae(pred, real)),
            f(pm.rmse(pred, real)), f(pm.rmse(pred, oracle)),
            f"{line['slope']:.3f} ({line['slope_se']:.3f})", f(pm.spearman(pred, real), 3),
            f(pm.top_decile_capture(pred, real), 3), f(pm.normalised_gini(pred, real), 3),
            f(pm.relative_bias(pred, oracle), 3)]


POINT_HEADER = ["model", "rel. bias", "MAE", "RMSE", "RMSE vs oracle", "slope (HC1 SE)",
                "Spearman", "top-decile capture", "norm. Gini", "rel. bias vs oracle"]


def coverage_cells(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, level: float,
                   seg: np.ndarray, dec: np.ndarray) -> list[str]:
    marg = ev.coverage_summary(lo, hi, y, level).iloc[0]
    by_seg = ev.coverage_summary(lo, hi, y, level, seg)
    by_dec = ev.coverage_summary(lo, hi, y, level, dec)
    below = int((by_seg["label"] == "below nominal").sum() + (by_dec["label"] == "below nominal"
                                                               ).sum())
    return [f"{marg['coverage']:.3f} ({marg['label']})",
            f"{by_seg['coverage'].min():.3f}-{by_seg['coverage'].max():.3f}",
            f"{by_dec['coverage'].min():.3f}-{by_dec['coverage'].max():.3f}", str(below)]


def survival_row(name: str, frame: pd.DataFrame, paid_in_window: np.ndarray,
                 rng: np.random.Generator) -> list[str]:
    surv = frame[[f"paid_{j}" for j in range(1, W + 1)]].to_numpy(np.float64)
    n = paid_in_window.astype(np.int64)
    event = n < W
    risk = 1.0 - surv[:, -1]
    dcal = pm.d_calibration(n, event, surv)
    pmf = ev.contractual_pmf(frame)
    pit = pm.randomised_pit(pmf, n, rng)
    dec = pm.decile_groups(risk)
    km_gap = max(abs(float(np.mean(n[dec == g] >= W)) - float(np.mean(surv[dec == g, -1])))
                 for g in range(10))
    return [name, f(pm.harrell_c(n, event, risk), 3), f(pm.uno_c(n, event, risk, W), 3),
            f(pm.integrated_brier(n, event, surv), 4), f(dcal["p_value"], 2),
            f(km_gap, 3), f(float(np.mean(pm.log_score(pmf, n))), 4),
            f(float(np.mean(pm.crps_counts(pmf, n))), 4),
            f(float(stats.kstest(pit, "uniform").pvalue), 2)]


SURV_HEADER = ["model", "Harrell C", "Uno C", "integrated Brier", "D-calibration p",
               "max |KM - S(12)| over risk deciles", "log score", "CRPS (paid months)",
               "PIT KS p"]


def selection_line(label: str, rep: ev.SelectionReport) -> list[str]:
    gates = "; ".join(f"{g.model_id}: {'Q' if g.qualified else 'x'}" for g in rep.gates)
    excl = "; ".join(f"{m} ({why})" for m, why in rep.excluded) or "-"
    ref = f"{rep.reference_fallback} ({rep.reference_label})" if rep.reference_fallback else "-"
    return [label, rep.status, rep.selected or "-", gates or "-", excl, ref]


def a11_row(label: str, m: str, level: float, sim: ev.SimulatedValue, real: np.ndarray,
            seg: np.ndarray, dec: np.ndarray) -> list[str]:
    """Pre-A11 and A11 predictive coverage from the same paid-month draws."""
    plo, phi = sim.interval(level, outcome=True)
    nlo, nhi = sim.interval(level, outcome=True, noise=True)
    pre = ev.coverage_summary(plo, phi, real, level).iloc[0]
    return ([label, m, f"{level:.2f}", f"{pre['coverage']:.3f}",
             f"{float(np.mean(phi - plo)):.1f}", f"{float(np.mean(nhi - nlo)):.1f}"]
            + coverage_cells(nlo, nhi, real, level, seg, dec))


def contractual_setting(name: str, sc: Scenario, cutoff: int, tiers: Sequence[str],
                        out: dict[str, list[list[str]]], rng: np.random.Generator,
                        noise_rng: np.random.Generator) -> None:
    n_months = sc.config.n_months
    b1 = timed("B1", pb.predict_b1, sc, cutoff)
    preds_a = {m: timed(m, ct.predict, m, sc, cutoff, W, POL) for m in ("E1", "E2")}
    for tier in tiers:
        label = f"{name}, C = {cutoff}, Tier {tier}"
        hold = lm.holdout_frame(sc, n_months, cutoff, W, "contractual", tier, POL
                                ).set_index("customer_id")
        real = hold["realised"].to_numpy(np.float64)
        ofr = (orc.signals_oracle(sc, cutoff, W, tier).frame
               if isinstance(sc, gen.SignalsScenario)
               else orc.m10_oracle(cast(m10g.EconomicsScenario, sc), cutoff, W).frame)
        oracle = ofr["oracle"].reindex(hold.index).to_numpy()
        e3 = timed("E3 fit", sv.fit_e3, sc, cutoff, W, tier, POL)
        e3p = sv.predict_e3(e3, sc).frame
        e5 = timed("E5 fit", pr.fit_e5, sc, cutoff, W, "contractual", tier, POL)
        e5p = pr.predict_e5(e5, sc).frame
        frames = {"B1": b1.frame, "E1": preds_a["E1"].frame, "E2": preds_a["E2"].frame,
                  "E3": e3p, "E5": e5p}
        for m, fr in frames.items():
            if not fr.index.equals(hold.index):
                raise RuntimeError(f"{m}: population differs from the holdout")
            out["point"].append([label] + point_row(m, fr["prediction"].to_numpy(np.float64),
                                                    real, oracle)[0:])
        n_paid = hold["paid_in_window"].to_numpy()
        # Tier A models (B1, E1, E2) are identical in Tier B rows: their distributional and
        # interval results are reported once, in the Tier A rows, against the Tier A oracle.
        own = ("E1", "E2", "E3") if tier == "A" else ("E3",)
        for m in own:
            out["survival"].append([label] + survival_row(m, frames[m], n_paid, rng))
        # uncertainty: parameter (vs oracle) and predictive (vs realised) intervals
        seg = hold["segment"].to_numpy()
        sims = {m: timed(f"{m} simulation", ev.simulate_contractual, preds_a[m], DRAWS, rng,
                         residuals=ev.contribution_residuals(sc, cutoff, preds_a[m].contribution),
                         noise_rng=noise_rng)
                for m in own if m != "E3"}
        cr1 = timed("CR1", ev.e3_cr1, e3, sc)
        sims["E3"] = timed("E3 simulation", ev.simulate_e3, e3, sc, cr1.cr1, DRAWS, rng,
                           residuals=ev.contribution_residuals(sc, cutoff, e3.contribution),
                           noise_rng=noise_rng)
        for m, sim in sims.items():
            dec = pm.decile_groups(frames[m]["prediction"].to_numpy(np.float64))
            for level in LEVELS:
                lo, hi = sim.interval(level, outcome=False)
                plo, phi = sim.interval(level, outcome=True)
                out["coverage"].append([label, m, f"{level:.2f}", "parameter vs oracle"]
                                       + coverage_cells(lo, hi, oracle, level, seg, dec))
                out["coverage"].append([label, m, f"{level:.2f}", "predictive vs realised"]
                                       + coverage_cells(plo, phi, real, level, seg, dec))
                out["coverage_a11"].append(a11_row(label, m, level, sim, real, seg, dec))
        s2 = timed("S2", pr.fit_s2, sc, cutoff, W, "contractual", tier, POL)
        dec5 = pm.decile_groups(e5p["prediction"].to_numpy(np.float64))
        for level in LEVELS:
            key = f"{level:g}"
            fr2 = s2.frame.reindex(hold.index)
            out["coverage"].append([label, "E5 + S2", f"{level:.2f}", "conformal vs realised"]
                                   + coverage_cells(fr2[f"lower_{key}"].to_numpy(),
                                                    fr2[f"upper_{key}"].to_numpy(), real, level,
                                                    seg, dec5))
        a, b = ev.e5_cr1(e5, sc)
        for part, c in (("E3 hazard", cr1), ("E5 positive", a), ("E5 amount", b)):
            ratio = c.cr1_se / c.model_se
            out["cr1"].append([label, part, str(c.n_rows), str(c.n_clusters),
                               f(float(np.min(ratio)), 3), f(float(np.median(ratio)), 3),
                               f(float(np.max(ratio)), 3)])
        # selection (validation landmark only)
        inputs = []
        v1 = pb.validate_baseline("B1", sc, cutoff, policy=POL)
        inputs.append(ev.CandidateInput("B1", "A", v1.selection_eligible, v1.eligibility,
                                        v1.candidate, v1.frame))
        for m in ("E1", "E2"):
            v = ct.validate(m, sc, cutoff, W, POL)
            inputs.append(ev.CandidateInput(m, "A", True, "eligible", v.candidate, v.frame))
        v3 = sv.validate_e3(sc, cutoff, W, tier, POL, penalty=e3.penalty)
        inputs.append(ev.CandidateInput("E3", tier, v3.selection_eligible, v3.eligibility,
                                        v3.candidate, v3.frame))
        v5 = pr.validate_e5(sc, cutoff, W, "contractual", tier, POL, penalty=e5.penalty)
        inputs.append(ev.CandidateInput("E5", tier, v5.selection_eligible, v5.eligibility,
                                        v5.candidate, v5.frame))
        rep = timed("selection", ev.selection_report, "contractual", tier, inputs,
                    np.random.default_rng(SEED_BOOTSTRAP))
        out["selection"].append(selection_line(label, rep))
        for inp in inputs:
            if inp.candidate is not None:
                cand = inp.candidate
                out["gates"].append([label, inp.model_id, f(cand.ok_cell_fraction, 3),
                                     f(cand.calibration_in_the_large, 3),
                                     f(cand.calibration_slope, 3), f(cand.mse)])


def noncontractual_setting(sc: gen.PurchasesScenario, cutoff: int,
                           out: dict[str, list[list[str]]], rng: np.random.Generator) -> None:
    label = f"purchases, C = {cutoff}, Tier A"
    hold = lm.holdout_frame(sc, sc.config.n_months, cutoff, W, "non_contractual", "A"
                            ).set_index("customer_id")
    real = hold["realised"].to_numpy(np.float64)
    ofr = po.purchases_oracle(sc, cutoff, W).frame.reindex(hold.index)
    oracle = ofr["oracle"].to_numpy()
    b2 = timed("B2", pb.predict_b2, sc, cutoff)
    e4 = timed("E4", btyd.predict, sc, cutoff, W)
    e5 = timed("E5 fit", pr.fit_e5, sc, cutoff, W, "non_contractual", "A")
    frames = {"B2": b2.frame, "E4": e4.frame, "E5": pr.predict_e5(e5, sc).frame}
    for m, fr in frames.items():
        if not fr.index.equals(hold.index):
            raise RuntimeError(f"{m}: population differs from the holdout")
        out["point"].append([label] + point_row(m, fr["prediction"].to_numpy(np.float64), real,
                                                oracle))
    counts = hold["paid_in_window"].to_numpy().astype(np.int64)
    k_max = int(max(300, int(np.max(counts)) + 50))
    pmf = timed("E4 count pmf", ev.e4_count_pmf, e4, k_max)
    tail = float(np.max(1.0 - pmf.sum(axis=1)))
    pit = pm.randomised_pit(pmf, counts, np.random.default_rng(SEED_PIT))
    pa = e4.frame["p_alive_T"].to_numpy()
    truth = sc.truth
    alive = pd.Series(truth.alive_at(e4.data.through), index=truth.customer_ids
                      ).reindex(hold.index).to_numpy(np.float64)
    rel = pm.decile_reliability(pa, alive)
    out["count"].append([label, "E4", f(float(np.mean(pm.log_score(pmf, counts))), 4),
                         f(float(np.mean(pm.crps_counts(pmf, counts))), 4),
                         f(float(stats.kstest(pit, "uniform").pvalue), 2), f(tail, 2),
                         f(float(np.max(np.abs(rel["mean_pred"] - rel["mean_y"]))), 3),
                         f(pm.relative_bias(pa, alive), 3)])
    seg = hold["segment"].to_numpy()
    sim = timed("E4 simulation", ev.e4_simulate, e4, DRAWS, rng)
    dec = pm.decile_groups(e4.frame["prediction"].to_numpy(np.float64))
    for level in LEVELS:
        lo, hi = sim.interval(level, outcome=False)
        plo, phi = sim.interval(level, outcome=True)
        out["coverage"].append([label, "E4", f"{level:.2f}", "parameter vs oracle"]
                               + coverage_cells(lo, hi, oracle, level, seg, dec))
        out["coverage"].append([label, "E4", f"{level:.2f}", "predictive vs realised"]
                               + coverage_cells(plo, phi, real, level, seg, dec))
    s2 = timed("S2", pr.fit_s2, sc, cutoff, W, "non_contractual", "A")
    dec5 = pm.decile_groups(frames["E5"]["prediction"].to_numpy(np.float64))
    for level in LEVELS:
        key = f"{level:g}"
        fr2 = s2.frame.reindex(hold.index)
        out["coverage"].append([label, "E5 + S2", f"{level:.2f}", "conformal vs realised"]
                               + coverage_cells(fr2[f"lower_{key}"].to_numpy(),
                                                fr2[f"upper_{key}"].to_numpy(), real, level,
                                                seg, dec5))
    a, b = ev.e5_cr1(e5, sc)
    for part, c in (("E5 positive", a), ("E5 amount", b)):
        ratio = c.cr1_se / c.model_se
        out["cr1"].append([label, part, str(c.n_rows), str(c.n_clusters),
                           f(float(np.min(ratio)), 3), f(float(np.median(ratio)), 3),
                           f(float(np.max(ratio)), 3)])
    inputs = []
    v2 = pb.validate_baseline("B2", sc, cutoff)
    inputs.append(ev.CandidateInput("B2", "A", v2.selection_eligible, v2.eligibility,
                                    v2.candidate, v2.frame))
    v4 = btyd.validate(sc, cutoff, W)
    inputs.append(ev.CandidateInput("E4", "A", True, "eligible", v4.candidate, v4.frame))
    v5 = pr.validate_e5(sc, cutoff, W, "non_contractual", "A", penalty=e5.penalty)
    inputs.append(ev.CandidateInput("E5", "A", v5.selection_eligible, v5.eligibility,
                                    v5.candidate, v5.frame))
    rep = timed("selection", ev.selection_report, "non_contractual", "A", inputs,
                np.random.default_rng(SEED_BOOTSTRAP))
    out["selection"].append(selection_line(label, rep))
    for inp in inputs:
        if inp.candidate is not None:
            cand = inp.candidate
            out["gates"].append([label, inp.model_id, f(cand.ok_cell_fraction, 3),
                                 f(cand.calibration_in_the_large, 3),
                                 f(cand.calibration_slope, 3), f(cand.mse)])


def a11_confirmation(out: dict[str, list[list[str]]]) -> None:
    """A11 on new development draws (5711-5713): pre-A11 and A11 predictive coverage only."""
    plan: list[tuple[str, Scenario, int, tuple[str, ...]]] = [
        ("M10", m10g.generate_economics_scenario(
            m10g.EconomicsScenarioConfig(seed=CONFIRMATION_SEEDS[0])), 30, ("A",)),
        ("signals", gen.generate_signals_scenario(
            gen.SignalsScenarioConfig(seed=CONFIRMATION_SEEDS[1])), 36, ("A", "B")),
        ("signals", gen.generate_signals_scenario(
            gen.SignalsScenarioConfig(seed=CONFIRMATION_SEEDS[2])), 42, ("A", "B"))]
    for (name, sc, cutoff, tiers), seed in zip(plan, CONFIRMATION_SEEDS, strict=True):
        rng = np.random.default_rng([SEED_SIMULATION, seed])
        noise_rng = np.random.default_rng([SEED_NOISE, seed])
        preds_a = {m: ct.predict(m, sc, cutoff, W, POL) for m in ("E1", "E2")}
        for tier in tiers:
            label = f"{name} (seed {seed}), C = {cutoff}, Tier {tier}"
            hold = lm.holdout_frame(sc, sc.config.n_months, cutoff, W, "contractual", tier, POL
                                    ).set_index("customer_id")
            real = hold["realised"].to_numpy(np.float64)
            seg = hold["segment"].to_numpy()
            sims: dict[str, ev.SimulatedValue] = {}
            preds: dict[str, np.ndarray] = {}
            if tier == "A":
                for m in ("E1", "E2"):
                    sims[m] = timed("A11 confirmation", ev.simulate_contractual, preds_a[m],
                                    DRAWS, rng, residuals=ev.contribution_residuals(
                                        sc, cutoff, preds_a[m].contribution),
                                    noise_rng=noise_rng)
                    preds[m] = preds_a[m].frame["prediction"].to_numpy(np.float64)
            e3 = sv.fit_e3(sc, cutoff, W, tier, POL)
            cov = ev.e3_cr1(e3, sc).cr1
            sims["E3"] = timed("A11 confirmation", ev.simulate_e3, e3, sc, cov, DRAWS, rng,
                               residuals=ev.contribution_residuals(sc, cutoff, e3.contribution),
                               noise_rng=noise_rng)
            preds["E3"] = sv.predict_e3(e3, sc).frame["prediction"].to_numpy(np.float64)
            for m, sim in sims.items():
                if not sim.index.equals(hold.index):
                    raise RuntimeError(f"{m}: population differs from the holdout")
                dec = pm.decile_groups(preds[m])
                for level in LEVELS:
                    out["confirm"].append(a11_row(label, m, level, sim, real, seg, dec))


def ipcw_demonstration(sc: gen.SignalsScenario) -> list[list[str]]:
    """S3: E2's holdout survival at C = 36 with synthetic independent censoring of the window."""
    rng = np.random.default_rng(SEED_CENSORING)
    pred = ct.predict("E2", sc, 36, W, POL)
    hold = lm.holdout_frame(sc, sc.config.n_months, 36, W, "contractual", "A", POL
                            ).set_index("customer_id")
    surv = pred.frame[[f"paid_{j}" for j in range(1, W + 1)]].to_numpy(np.float64)
    n = hold["paid_in_window"].to_numpy().astype(np.int64)
    event = n < W
    complete = pm.integrated_brier(n, event, surv)
    rows = []
    for share in (0.2, 0.5):
        c = np.where(rng.uniform(size=n.size) < share, rng.integers(0, W, n.size), W)
        ev_c = event & (n < c)
        t_c = np.where(ev_c, n, np.minimum(n, c))
        ipcw = pm.integrated_brier(t_c, ev_c, surv)
        naive = []
        for t in range(1, W + 1):
            known = (t_c >= t) | ev_c
            naive.append(float(np.mean(((t_c[known] >= t).astype(float)
                                        - surv[known, t - 1]) ** 2)))
        rows.append([f"{share:.0%} of customers censored uniformly in the window",
                     f(complete, 5), f(ipcw, 5), f(float(np.mean(naive)), 5),
                     f(int((~ev_c & (t_c < W)).sum()), 5)])
    return rows


def main() -> None:
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED_SIMULATION)
    out: dict[str, list[list[str]]] = {k: [] for k in ("point", "survival", "count",
                                                      "coverage", "cr1", "selection", "gates",
                                                      "coverage_a11", "confirm")}
    noise_rng = np.random.default_rng(SEED_NOISE)
    m10 = m10g.generate_economics_scenario(m10g.EconomicsScenarioConfig(seed=SEED_M10))
    sig = gen.generate_signals_scenario(gen.SignalsScenarioConfig(seed=SEED_SIGNALS))
    pur = gen.generate_purchases_scenario(gen.PurchasesScenarioConfig(seed=SEED_PURCHASES))
    contractual_setting("M10", m10, 30, ("A",), out, rng, noise_rng)
    for cutoff in (36, 42):
        contractual_setting("signals", sig, cutoff, ("A", "B"), out, rng, noise_rng)
    for cutoff in (24, 30):
        noncontractual_setting(pur, cutoff, out, rng)
    demo = ipcw_demonstration(sig)
    a11_confirmation(out)
    total = time.perf_counter() - t0

    lines: list[str] = []
    w = lines.append
    w("# Milestone 11 P7 record — evaluation, uncertainty and selection reporting")
    w("")
    w("Generated by `python scripts/report_m11_p7_evaluation.py`. **Development seeds only** "
      "(pilot, main and fresh seeds not used); one draw per scenario, so every number here is "
      "descriptive. Coverage is reported, never asserted (`COVERAGE_REPORTING_RULE`). "
      "Findings: [m11-p7-findings.md](m11-p7-findings.md).")
    w("")
    w(table(["item", "value"], [
        ["protocol stage", pp.PROTOCOL_STAGE],
        ["protocol manifest digest", f"`{pp.protocol_digest()}`"],
        ["protocol code digest", f"`{pp.protocol_code_digest()}`"],
        ["seeds", f"M10 {SEED_M10}; purchases {SEED_PURCHASES}; signals {SEED_SIGNALS}; "
                  f"bootstrap {SEED_BOOTSTRAP}; simulation {SEED_SIMULATION}; censoring "
                  f"{SEED_CENSORING}; PIT {SEED_PIT}; A11 noise {SEED_NOISE}; A11 confirmation "
                  f"{CONFIRMATION_SEEDS[0]}-{CONFIRMATION_SEEDS[-1]}"],
        ["parametric draws per customer", str(DRAWS)],
        ["paired bootstrap resamples", str(pp.CLUSTERED_INFERENCE.metric_bootstrap_resamples)],
        ["git", git_state()],
        ["python / numpy / scipy / pandas", f"{sys.version.split()[0]} / {np.__version__} / "
                                            f"{scipy.__version__} / {pd.__version__}"],
        ["platform", platform.platform()],
    ]))
    w("")
    w("## 1. Point accuracy, discrimination and calibration (holdout, W = 12)")
    w("")
    w("Relative bias = sum(pred) / sum(realised) - 1; slope = OLS of realised on prediction "
      "(HC1 SE); oracle = the P2 observable-information oracle of the same tier. B1 and B2 "
      "follow amendment A10; in Tier B rows, B1, E1 and E2 are the Tier A models "
      "(`TIER_A_IN_TIER_B`).")
    w("")
    w(table(["setting"] + POINT_HEADER, out["point"]))
    w("")
    w("## 2. Survival and predictive distributions (contractual)")
    w("")
    w("Time = paid months in the window, event = churn within it (no censoring in the holdout "
      "except at W). Risk = 1 - S(12). `KM` = observed share surviving the window within each "
      "risk decile (the Kaplan-Meier estimate, as nothing is censored before W). Tier A models "
      "appear in Tier A rows only.")
    w("")
    w(table(["setting"] + SURV_HEADER, out["survival"]))
    w("")
    w("## 3. Count distribution and P(alive) (E4, non-contractual)")
    w("")
    w("Exact window-purchase distribution (`e4_count_pmf`); P(alive at T) reliability against "
      "the generator's latent state (evaluation only).")
    w("")
    w(table(["setting", "model", "log score", "CRPS (purchases)", "PIT KS p",
             "max truncated mass", "max decile |P(alive) - alive|", "P(alive) rel. bias"],
            out["count"]))
    w("")
    w("## 4. Interval coverage (reported, not asserted)")
    w("")
    w("Parameter intervals (expected value under Hessian or CR1 parameter draws) are compared "
      "with the oracle of the model's own tier (Tier A models only in Tier A rows); "
      "predictive intervals (parameter + outcome) and S2 with the realised "
      "value. Groups: segment and prediction decile; `below` = groups labelled below nominal "
      "by the exact binomial test.")
    w("")
    w(table(["setting", "model", "level", "interval", "marginal", "by segment (min-max)",
             "by decile (min-max)", "groups below"], out["coverage"]))
    w("")
    w("## 4a. Amendment A11: contractual predictive intervals with contribution noise")
    w("")
    w("`CONTRACTUAL_OUTCOME_NOISE` (A11): the **same** simulated paid months as section 4, each "
      "valued with an independent residual from the completely recorded pre-cutoff months of "
      "the contribution model's source group (separate random stream, seed "
      f"{SEED_NOISE}; section 4's pre-A11 rows are unchanged). Nothing is tuned or rescaled. "
      "Reported, not asserted.")
    w("")
    a11_header = ["setting", "model", "level", "pre-A11 marginal", "pre-A11 mean width",
                  "A11 mean width", "A11 marginal", "A11 by segment (min-max)",
                  "A11 by decile (min-max)", "A11 groups below"]
    w(table(a11_header, out["coverage_a11"]))
    w("")
    w("## 4b. A11 confirmation on new development draws "
      f"({CONFIRMATION_SEEDS[0]}-{CONFIRMATION_SEEDS[-1]})")
    w("")
    w(f"M10 seed {CONFIRMATION_SEEDS[0]} (C = 30), signals seeds {CONFIRMATION_SEEDS[1]} "
      f"(C = 36) and {CONFIRMATION_SEEDS[2]} (C = 42); E1/E2 in Tier A rows, E3 in each tier; "
      "both constructions from the same draws.")
    w("")
    w(table(a11_header, out["confirm"]))
    w("")
    w("## 5. Customer-clustered (CR1) vs model-based standard errors")
    w("")
    w(table(["setting", "part", "rows", "customers", "min ratio", "median ratio", "max ratio"],
            out["cr1"]))
    w("")
    w("## 6. Selection (validation landmark only)")
    w("")
    w("Frozen `select_model` with A9 eligibility and A10 rules; `Q` = qualified. The reference "
      "fallback appears only for contractual scenarios (`REFERENCE_FALLBACK_SCOPE`).")
    w("")
    w(table(["setting", "status", "selected", "gates", "excluded", "reference fallback"],
            out["selection"]))
    w("")
    w(table(["setting", "model", "ok-cell fraction", "CITL", "slope", "validation MSE"],
            out["gates"]))
    w("")
    w("## 7. S3: IPCW censored-evaluation demonstration (E2, signals, C = 36, Tier A)")
    w("")
    w("Synthetic independent censoring of the complete holdout window; the IPCW integrated "
      "Brier score should recover the complete-data value; the naive score drops customers "
      "censored before t.")
    w("")
    w(table(["censoring", "complete-data IBS", "IPCW IBS", "naive IBS", "censored before W"],
            demo))
    w("")
    w("## 8. Runtime")
    w("")
    w(table(["step", "seconds"], [[k, f"{v:.1f}"] for k, v in sorted(TIMINGS.items())]
            + [["total", f"{total:.1f}"]]))
    w("")
    OUTPUT.write_text("\n".join(lines))
    print(f"wrote {OUTPUT} in {total:.0f} s")


if __name__ == "__main__":
    main()
