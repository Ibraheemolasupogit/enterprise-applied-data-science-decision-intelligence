"""M11 P0: pilot precision and computational-feasibility plan.

Writes ``docs/evidence/m11-p0-precision-plan.md``. Uses only the M11 pilot seed (via
``predictive_protocol.authorise_seed("pilot")``) and only observable tables: no model is fitted,
no ground truth is read, and neither the main nor the fresh seed is touched. Generators for the
two new M11 scenarios do not exist yet (P1); their planning uses the M10 generator at the new
scenarios' horizon as a size proxy and is marked as such.

Usage:
    python scripts/plan_m11_precision.py              # writes the plan record
    python scripts/plan_m11_precision.py --output OUT.md
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import resource
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from decision_intelligence.customer_value import precision as pr11
from decision_intelligence.customer_value import predictive_protocol as pp
from decision_intelligence.customer_value.panel import observe
from decision_intelligence.data.customer_economics_scenarios import (
    EconomicsScenario,
    EconomicsScenarioConfig,
    generate_economics_scenario,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "docs" / "evidence" / "m11-p0-precision-plan.md"
BENCH_SEED = 1  # development range: synthetic matrices for timing only
TIMINGS: dict[str, float] = {}


def table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    def fmt(v: object) -> str:
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)

    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


@contextmanager
def timed(label: str) -> Iterator[None]:
    t = time.perf_counter()
    try:
        yield
    finally:
        TIMINGS[label] = TIMINGS.get(label, 0.0) + time.perf_counter() - t


def peak_rss_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1.0 if sys.platform == "darwin" else 1024.0  # bytes on macOS, KiB on Linux
    return rss * scale / 1e9


def common_set(sc: EconomicsScenario, cutoff: int, horizon: int) -> pd.DataFrame:
    """M10's existing-customer comparison set at ``cutoff`` with realised contribution over
    ``cutoff+1 .. cutoff+horizon`` read after the recording lag (observable tables only)."""
    obs = observe(sc, cutoff)
    st = obs.status()
    through = obs.complete_through()
    active = st.loc[~st["churned"] & (st["acquisition_month"] <= through)]
    later = observe(sc, cutoff + horizon, lag_days=pp.RECORDING_LAG_DAYS)
    tx = later.transactions
    win = tx.loc[(tx["billing_month"] > cutoff) & (tx["billing_month"] <= cutoff + horizon)]
    realised = win.groupby("customer_id")["contribution"].sum().reindex(active.index,
                                                                         fill_value=0.0)
    return active.assign(realised=realised, tenure=cutoff - active["acquisition_month"])


def sigma_proxy(frame: pd.DataFrame) -> float:
    """Pooled within-cell SD of realised value, cells = segment x channel x tenure band."""
    edges = list(pp.PRECISION_PROXY_TENURE_EDGES) + [10**6]
    band = pd.cut(frame["tenure"], edges, right=False, labels=False)
    num, den = 0.0, 0
    for _, g in frame.assign(band=band).groupby(["segment", "channel", "band"]):
        if len(g) >= 2:
            num += float(g["realised"].var(ddof=1)) * (len(g) - 1)
            den += len(g) - 1
    return float(np.sqrt(num / den))


def active_at(sc: EconomicsScenario, landmark: int) -> int:
    st = observe(sc, landmark).status()
    return int((~st["churned"]).sum())


def newton_iteration_seconds(rows: int, cols: int, repeats: int = 3) -> float:
    rng = np.random.default_rng(BENCH_SEED)
    x = rng.standard_normal((rows, cols))
    w = rng.uniform(0.01, 0.25, rows)
    g = rng.standard_normal(rows)
    t = time.perf_counter()
    for _ in range(repeats):
        h = x.T @ (w[:, None] * x) + np.eye(cols)
        np.linalg.solve(h, x.T @ g)
    return (time.perf_counter() - t) / repeats


def simulation_seconds(n_customers: int, draws: int, horizon: int, chunk: int = 250) -> float:
    rng = np.random.default_rng(BENCH_SEED)
    hazard = rng.uniform(0.01, 0.2, n_customers)
    value = rng.uniform(1, 20, n_customers)
    t = time.perf_counter()
    total = 0.0
    for s in range(0, n_customers, chunk):
        h = hazard[s:s + chunk]
        alive = np.cumprod(rng.random((draws, h.size, horizon)) > h[None, :, None], axis=2)
        total += float((alive.sum(axis=2) * value[s:s + chunk]).sum())
    assert np.isfinite(total)
    return time.perf_counter() - t


def bootstrap_seconds(n_customers: int, resamples: int) -> float:
    rng = np.random.default_rng(BENCH_SEED)
    d = rng.standard_normal(n_customers)
    t = time.perf_counter()
    for _ in range(resamples // 100):
        idx = rng.integers(0, n_customers, (100, n_customers))
        d[idx].mean(axis=1)
    return time.perf_counter() - t


def git_state() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts",
                                "tests", "docs", "pyproject.toml"],
                               cwd=REPO_ROOT, capture_output=True, text=True,
                               check=True).stdout.strip()
        n = len(dirty.splitlines()) if dirty else 0
        return f"{head}{f' + {n} uncommitted or untracked paths' if n else ''}"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    problems = pp.validate_protocol()
    if problems:
        raise SystemExit("protocol inconsistent: " + "; ".join(problems))
    seed = pp.authorise_seed("pilot")
    start = time.perf_counter()
    alpha, power = pp.PLANNING_ALPHA, pp.PLANNING_POWER
    k = pr11.critical_sum(alpha, power)
    lines: list[str] = []
    w = lines.append

    # --- 1. M10 scenario: comparison-set sizes and noise floor -------------------------------
    m10 = pp.scenario("m10_contractual")
    with timed("generate M10 scenario (48 months)"):
        sc = generate_economics_scenario(replace(EconomicsScenarioConfig(), seed=seed))
    rows = []
    frames = {}
    with timed("comparison sets and realised windows"):
        for c in m10.evaluation_cutoffs:
            f = common_set(sc, c, pp.PRIMARY_HORIZON)
            frames[c] = f
            r = f["realised"].to_numpy()
            sig = sigma_proxy(f)
            rel = pr11.minimum_detectable_relative_excess(len(f), alpha, power)
            rows.append([c, len(f), float(np.mean(r == 0)), float(np.mean(r)),
                         float(np.std(r, ddof=1)), sig, float(np.quantile(r, 0.5)),
                         float(np.quantile(r, 0.9)), float(np.quantile(r, 0.99)), rel,
                         sig * float(np.sqrt(rel)),
                         "met" if rel <= pp.TARGET_RELATIVE_MDE else "NOT met"])
    w("## 1. M10 scenario: comparison sets and the noise floor (pilot seed)\n")
    w("**Scope:** final for `m10_contractual`; **PROVISIONAL** for the two new M11 scenarios, "
      "whose generators do not exist yet (P1 repeats this plan on them).\n")
    w("Existing customers active at the cutoff with at least one completely recorded month "
      "(M10 rule); realised = undiscounted contribution over the next 12 months, read after the "
      "45-day lag. `sigma proxy` = pooled within-cell SD (segment x channel x tenure band), a "
      "conservative stand-in for sqrt(E[Var(R | F_C)]). Relative MDE = 4k^2/N with "
      f"alpha = {alpha:.4g} (Holm's strictest step for 5 primary comparisons), power = "
      f"{power}, k = {k:.4f}; target <= {pp.TARGET_RELATIVE_MDE}.\n")
    w(table(["cutoff", "N", "share R = 0", "mean R", "SD R", "sigma proxy", "median", "p90",
             "p99", "relative MDE", "detectable excess RMSE", "target"], rows))

    ids = [set(frames[c].index) for c in m10.evaluation_cutoffs]
    pooled = pd.Series(np.concatenate([frames[c].index.to_numpy()
                                       for c in m10.evaluation_cutoffs]))
    sizes = pooled.value_counts().to_numpy()
    n_rows, n_clusters = int(sizes.sum()), int(sizes.size)
    overlap = len(ids[0] & ids[1]) if len(ids) == 2 else 0
    w("\n### Pooled cutoffs and customer clustering (amendment 2)\n")
    w("Kish design effect with the actual cluster sizes, `n_eff = n / (1 + rho (sum m_g^2 / n "
      "- 1))`; `rho` is the within-customer correlation of the paired loss difference across "
      "cutoffs (unknown; positive here because the 12-month windows of cutoffs 24 and 30 "
      "overlap by 6 months), so `rho = 1` is the conservative planning value. Clustering "
      "*raises* the detectable difference relative to treating rows as independent (`rho = "
      "0`); the pooled value is still below the single-cutoff value because pooling adds "
      "distinct customers and partly independent second rows.\n")
    w(table(["rows (customer x cutoff)", "distinct customers", "in both cutoffs",
             "n_eff rho=0", "n_eff rho=0.5", "n_eff rho=1",
             "relative MDE rho=0", "relative MDE rho=1"],
            [[n_rows, n_clusters, overlap,
              pr11.effective_n(sizes, 0.0), pr11.effective_n(sizes, 0.5),
              pr11.effective_n(sizes, 1.0),
              pr11.minimum_detectable_relative_excess(pr11.effective_n(sizes, 0.0), alpha,
                                                      power),
              pr11.minimum_detectable_relative_excess(pr11.effective_n(sizes, 1.0), alpha,
                                                      power)]]))
    sig24 = sigma_proxy(frames[m10.evaluation_cutoffs[0]])
    n24 = len(frames[m10.evaluation_cutoffs[0]])
    w("\n### General case: SE of a paired MSE difference, first cutoff\n")
    w("`SE = 2 tau sigma / sqrt(N)` where `tau` is the RMS difference between two models' "
      "predictions; the difference is detectable when it exceeds `k SE`.\n")
    w(table(["tau / sigma", "tau", "SE(mean D)", "k x SE", "tau^2 (near-oracle E[D])",
             "detectable"],
            [[r, r * sig24, pr11.paired_mse_difference_se(r * sig24, sig24, n24),
              k * pr11.paired_mse_difference_se(r * sig24, sig24, n24), (r * sig24) ** 2,
              (r * sig24) ** 2 >= k * pr11.paired_mse_difference_se(r * sig24, sig24, n24)]
             for r in (0.05, 0.1, 0.25, 0.5)]))

    # --- 2. Repeated-sampling precision ----------------------------------------------------------
    w("\n## 2. Repeated-sampling precision\n")
    w("Reporting criteria, not acceptance tests (`COVERAGE_REPORTING_RULE`): an observed "
      "coverage is labelled consistent with, below or above nominal by the exact two-sided "
      "binomial test; the region below is the 'consistent' range.\n")
    w(table(["study", "nominal", "replications", "Monte Carlo SE",
             "exact 95% 'consistent' region"],
            [[study, lvl, reps, pr11.coverage_mc_se(lvl, reps),
              "[{:.3f}, {:.3f}]".format(*pr11.coverage_acceptance_region(lvl, reps))]
             for study, reps in (("coverage", pp.COVERAGE_REPLICATIONS),
                                 ("refit-bootstrap", pp.REFIT_STUDY_REPLICATIONS))
             for lvl in pp.CALIBRATION.interval_levels]))

    # --- 3. Feasibility --------------------------------------------------------------------------
    sig_sc = pp.scenario("signals_contractual")
    with timed("generate 60-month size proxy (M10 generator)"):
        sc60 = generate_economics_scenario(replace(EconomicsScenarioConfig(), seed=seed,
                                                   n_months=sig_sc.n_months))
    stack = {}
    with timed("landmark risk-set sizes"):
        for c in sig_sc.evaluation_cutoffs:
            stack[c] = sum(active_at(sc60, L)
                           for L in pp.training_landmarks(c, pp.PRIMARY_HORIZON))
    with timed("comparison set at the signals cutoffs (proxy)"):
        n36 = len(common_set(sc60, sig_sc.evaluation_cutoffs[0], pp.PRIMARY_HORIZON))
    big = max(stack.values())
    cols = 25
    with timed("benchmark: penalised Newton iteration"):
        it = newton_iteration_seconds(big, cols)
    with timed("benchmark: predictive simulation"):
        sim = simulation_seconds(n36, 1000, pp.PRIMARY_HORIZON)
    with timed("benchmark: customer bootstrap"):
        boot = bootstrap_seconds(n36, pp.CLUSTERED_INFERENCE.metric_bootstrap_resamples)
    gen48 = TIMINGS["generate M10 scenario (48 months)"]
    gen60 = TIMINGS["generate 60-month size proxy (M10 generator)"]
    iters = 25
    fits_e3 = sum((len(pp.PENALTY_GRID) + 1) if pp.temporal_tuning_feasible(c, h) else 1
                  for s in pp.SCENARIOS if s.kind == "contractual"
                  for c, h in [(c, pp.PRIMARY_HORIZON) for c in s.evaluation_cutoffs]
                  + [(c, h) for h, c in s.secondary_horizon]) * 2  # x tiers (upper bound)
    main_est = (gen48 * 2 + gen60 * 1.5 + fits_e3 * iters * it * 2  # E3 and E5
                + sim * 4 * 7 + boot * 5 * 7)
    per_draw = gen48 * 0.5 + 0.5 * fits_e3 * iters * it / 4 + sim / 2
    rep_est = pp.COVERAGE_REPLICATIONS * per_draw
    refit_est = pp.REFIT_STUDY_REPLICATIONS * (
        gen48 * 0.5 + pp.CLUSTERED_INFERENCE.refit_bootstrap_resamples * iters * it / 4)
    w("\n## 3. Computational feasibility (indicative; this machine; PROVISIONAL for the new "
      "scenarios)\n")
    w(table(["item", "value"], [
        ["M10 scenario customers / transactions", f"{len(sc.customers)} / {len(sc.transactions)}"],
        ["60-month proxy customers / transactions",
         f"{len(sc60.customers)} / {len(sc60.transactions)}"],
        ["comparison set at C = 36 (60-month proxy)", n36],
        ["landmark stack rows, C = 36 / 42 (Tier A, W = 12)",
         " / ".join(str(stack[c]) for c in sig_sc.evaluation_cutoffs)],
        [f"penalised Newton iteration ({big} x {cols})", f"{it:.3f} s"],
        [f"predictive simulation (1000 draws x {n36} x 12)", f"{sim:.2f} s"],
        [f"customer bootstrap ({pp.CLUSTERED_INFERENCE.metric_bootstrap_resamples} x {n36})",
         f"{boot:.2f} s"],
        ["extrapolated main-seed run (all scenarios, essential models)",
         f"~{main_est / 60:.1f} min"],
        [f"extrapolated {pp.COVERAGE_REPLICATIONS} coverage draws (half scale)",
         f"~{rep_est / 60:.1f} min"],
        [f"extrapolated {pp.REFIT_STUDY_REPLICATIONS}-draw refit-bootstrap study "
         f"({pp.CLUSTERED_INFERENCE.refit_bootstrap_resamples} refits per draw, one E3 fit)",
         f"~{refit_est / 60:.1f} min"],
        ["peak memory of this planning run", f"{peak_rss_gb():.2f} GB"],
    ]))
    w("\nExtrapolations assume 25 Newton iterations per fit, the penalty grid plus a refit "
      "where temporal tuning is feasible, two tiers, E3 and E5, four distributional models and "
      "five bootstrapped metrics per cutoff; they exclude oracle filtering and maximum-"
      "likelihood fits (P2/P3 measure those). The 60-month proxy is the M10 generator, not "
      "the P1 signals generator.")

    # --- header and runtime ------------------------------------------------------------------
    text = (REPO_ROOT / "src/decision_intelligence/customer_value/predictive_protocol.py"
            ).read_bytes()
    head = [
        "# Milestone 11 P0 — pilot precision and feasibility plan",
        "",
        "Generated by `python scripts/plan_m11_precision.py`. **Planning record, not evidence "
        "of model performance**: pilot seed only; no model fitted; no ground truth read; main "
        "and fresh seeds untouched (refused by `authorise_seed` while protocol items are "
        "pending).",
        "",
        f"**Scope.** {pp.PILOT_FINDINGS_SCOPE}",
        "",
        table(["item", "value"], [
            ["protocol", f"{pp.PROTOCOL_ID} ({pp.PROTOCOL_STAGE})"],
            ["protocol digest (manifest SHA-256)", f"`{pp.protocol_digest()}`"],
            ["protocol file SHA-256", f"`{hashlib.sha256(text).hexdigest()}`"],
            ["git", git_state()],
            ["seed", f"pilot ({seed})"],
            ["pending before main", len(pp.PENDING_BEFORE_MAIN)],
            ["python / numpy / scipy / pandas",
             f"{platform.python_version()} / {np.__version__} / {scipy.__version__} / "
             f"{pd.__version__}"],
        ]),
        "",
    ]
    TIMINGS["total"] = time.perf_counter() - start
    lines.append("\n## Runtime\n")
    lines.append(table(["step", "seconds (indicative)"],
                       [[k_, f"{v:.2f}"] for k_, v in TIMINGS.items()]))
    args.output.write_text("\n".join(head + lines) + "\n")
    print(f"wrote {args.output} in {TIMINGS['total']:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
