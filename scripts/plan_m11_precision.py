"""M11 pilot precision and computational-feasibility plans (P0 and P1).

``--stage p0`` (default) writes ``docs/evidence/m11-p0-precision-plan.md``; ``--stage p1``
writes ``docs/evidence/m11-p1-scenario-plan.md`` for the P1 generators; ``--stage p2`` writes
``docs/evidence/m11-p2-oracle-plan.md`` (oracle calibration, the Tier B advantage and the
non-contractual oracle's calibration and quadrature convergence, using the P2 oracles:
evaluation references with the true generator parameters, never models).
Uses only the M11 pilot seed (via ``predictive_protocol.authorise_seed("pilot")``) and
development seeds; no model is fitted; the latent truth tables are never read (P2 uses the
oracles' true parameters only); neither the main nor the fresh seed is touched.
The P0 stage plans the new scenarios with the M10 generator as a size proxy (marked
PROVISIONAL); the P1 stage replaces that proxy with the P1 generators.

Usage:
    python scripts/plan_m11_precision.py              # writes the P0 plan record
    python scripts/plan_m11_precision.py --stage p1   # writes the P1 scenario plan record
    python scripts/plan_m11_precision.py --stage p2   # writes the P2 oracle plan record
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

from decision_intelligence.common.bootstrap import bootstrap_ci
from decision_intelligence.customer_value import landmarks as lm
from decision_intelligence.customer_value import oracle as orc
from decision_intelligence.customer_value import precision as pr11
from decision_intelligence.customer_value import predictive_protocol as pp
from decision_intelligence.customer_value import purchase_oracle as por
from decision_intelligence.customer_value.features import PromotionPolicy
from decision_intelligence.customer_value.panel import EconomicsTables, observe
from decision_intelligence.data import predictive_ltv_scenarios as gen
from decision_intelligence.data.customer_economics_scenarios import (
    CHANNELS,
    SEGMENTS,
    EconomicsScenarioConfig,
    generate_economics_scenario,
)
from decision_intelligence.data.point_in_time import as_of

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "docs" / "evidence" / "m11-p0-precision-plan.md"
P1_OUTPUT = REPO_ROOT / "docs" / "evidence" / "m11-p1-scenario-plan.md"
P2_OUTPUT = REPO_ROOT / "docs" / "evidence" / "m11-p2-oracle-plan.md"
P2_DEV_SEEDS = tuple(range(5101, 5111))  # development range: population-level replication
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


def common_set(sc: EconomicsTables, cutoff: int, horizon: int) -> pd.DataFrame:
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


def active_at(sc: EconomicsTables, landmark: int) -> int:
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


def usage_tercile(sc: gen.SignalsScenario, frame: pd.DataFrame, cutoff: int) -> pd.Series:
    """Tier B planning cell: tercile of mean usage over the last 3 usable signal months
    (recorded by the cutoff, months <= C - SIGNAL_COMPLETE_OFFSET)."""
    seen = as_of(sc.signals, "recorded_at", sc.month_end(cutoff))
    seen = seen.loc[(seen["month"] <= cutoff - pp.SIGNAL_COMPLETE_OFFSET)
                    & (seen["month"] > cutoff - pp.SIGNAL_COMPLETE_OFFSET - 3)]
    usage = seen.groupby("customer_id")["usage"].mean().reindex(frame.index)
    return pd.Series(pd.qcut(usage.rank(method="first"), 3, labels=False), index=frame.index)


def frequency_tercile(sc: gen.PurchasesScenario, frame: pd.DataFrame, cutoff: int) -> pd.Series:
    """Richer Tier A planning cell: tercile of repeat purchases in completely recorded months."""
    obs = observe(sc, cutoff)
    tx = obs.transactions.loc[obs.transactions["billing_month"] <= obs.complete_through()]
    repeats = (tx.groupby("customer_id").size() - 1).reindex(frame.index, fill_value=0)
    return pd.Series(pd.qcut(repeats.rank(method="first"), 3, labels=False), index=frame.index)


def sigma_cells(frame: pd.DataFrame, extra: pd.Series | None = None) -> float:
    edges = list(pp.PRECISION_PROXY_TENURE_EDGES) + [10**6]
    cells = frame.assign(band=pd.cut(frame["tenure"], edges, right=False, labels=False))
    keys = ["segment", "channel", "band"]
    if extra is not None:
        cells = cells.assign(extra=extra)
        keys.append("extra")
    num, den = 0.0, 0
    for _, g in cells.groupby(keys):
        if len(g) >= 2:
            num += float(g["realised"].var(ddof=1)) * (len(g) - 1)
            den += len(g) - 1
    return float(np.sqrt(num / den))


def precision_rows(sets: dict[tuple[int, int], pd.DataFrame],
                   extras: dict[tuple[int, int], pd.Series], alpha: float, power: float
                   ) -> list[list[object]]:
    rows: list[list[object]] = []
    for (c, h), f in sets.items():
        r = f["realised"].to_numpy()
        sa = sigma_cells(f)
        sb = sigma_cells(f, extras[(c, h)])
        rel = pr11.minimum_detectable_relative_excess(len(f), alpha, power)
        rows.append([c, h, len(f), float(np.mean(r == 0)), float(np.mean(r)),
                     float(np.std(r, ddof=1)), sa, sb, 1 - (sb / sa) ** 2, rel,
                     sa * float(np.sqrt(rel)),
                     "met" if rel <= pp.TARGET_RELATIVE_MDE else "NOT met"])
    return rows


def pooled_row(sets: dict[tuple[int, int], pd.DataFrame], cutoffs: Sequence[int],
               alpha: float, power: float) -> list[object]:
    ids = pd.Series(np.concatenate([sets[(c, pp.PRIMARY_HORIZON)].index.to_numpy()
                                    for c in cutoffs]))
    sizes = ids.value_counts().to_numpy()
    return [int(sizes.sum()), int(sizes.size), int((sizes > 1).sum()),
            pr11.effective_n(sizes, 1.0),
            pr11.minimum_detectable_relative_excess(pr11.effective_n(sizes, 1.0), alpha,
                                                    power)]


def plan_p1(seed: int, output: Path) -> int:
    start = time.perf_counter()
    alpha, power = pp.PLANNING_ALPHA, pp.PLANNING_POWER
    lines: list[str] = []
    w = lines.append
    prec_headers = ["cutoff", "horizon", "N", "share R = 0", "mean R", "SD R",
                    "sigma (Tier A cells)", "sigma (+ planning signal)",
                    "variance share of planning signal", "relative MDE",
                    "detectable excess RMSE", "target"]
    pooled_headers = ["rows", "distinct customers", "in 2 cutoffs", "n_eff rho=1",
                      "relative MDE rho=1"]

    # --- 1. Contractual subscriptions with behavioural signals ------------------------------
    s_spec = pp.scenario("signals_contractual")
    s_cfg = replace(gen.SignalsScenarioConfig(), seed=seed)
    with timed("generate signals scenario"):
        sig = gen.generate_signals_scenario(s_cfg)
    pairs = [(c, pp.PRIMARY_HORIZON) for c in s_spec.evaluation_cutoffs]
    pairs += [(c, h) for h, c in s_spec.secondary_horizon]
    with timed("signals: comparison sets"):
        s_sets = {(c, h): common_set(sig, c, h) for c, h in pairs}
        s_extra = {(c, h): usage_tercile(sig, f, c) for (c, h), f in s_sets.items()}
    with timed("signals: landmark risk-set sizes"):
        stack = {c: sum(active_at(sig, L) for L in pp.training_landmarks(c, pp.PRIMARY_HORIZON))
                 for c in s_spec.evaluation_cutoffs}
    w("## 1. `signals_contractual` (pilot seed)\n")
    w(table(["item", "value"], [
        ["customers / transactions / signal rows",
         f"{len(sig.customers)} / {len(sig.transactions)} / {len(sig.signals)}"],
        ["recorded cancellations / closure-summary (trap) rows",
         f"{len(sig.cancellations)} / {len(sig.account_closure_summary)}"],
        ["mean observed paid months", f"{len(sig.transactions) / len(sig.customers):.2f}"],
        ["landmark stack rows, C = 36 / 42 (W = 12)",
         " / ".join(str(stack[c]) for c in s_spec.evaluation_cutoffs)],
    ]))
    w("\nComparison set: M10's rule (active at C, >= 1 complete month). `sigma (Tier A cells)` "
      "as in P0. `sigma (+ planning signal)` adds a tercile of mean usage over the last 3 usable "
      "signal months (record-time, months <= C - 1): a crude, model-free indication of how much "
      "Tier B information there is for PC3 (planning only, not a model result).\n")
    w(table(prec_headers, precision_rows(s_sets, s_extra, alpha, power)))
    w("\nPooled cutoffs 36 and 42 (Kish, rho = 1):\n")
    w(table(pooled_headers, [pooled_row(s_sets, s_spec.evaluation_cutoffs, alpha, power)]))

    # --- 2. Non-contractual purchasing ---------------------------------------------------------
    p_spec = pp.scenario("purchases_noncontractual")
    p_cfg = replace(gen.PurchasesScenarioConfig(), seed=seed)
    with timed("generate purchases scenario"):
        pur = gen.generate_purchases_scenario(p_cfg)
    with timed("purchases: comparison sets"):
        p_sets = {(c, pp.PRIMARY_HORIZON): common_set(pur, c, pp.PRIMARY_HORIZON)
                  for c in p_spec.evaluation_cutoffs}
        p_extra = {k_: frequency_tercile(pur, f, k_[0]) for k_, f in p_sets.items()}
    reps = pur.purchases.groupby("customer_id").size() - 1
    w("\n## 2. `purchases_noncontractual` (pilot seed)\n")
    w(table(["item", "value"], [
        ["customers / purchases", f"{len(pur.customers)} / {len(pur.purchases)}"],
        ["mean repeat purchases per customer (all data)", f"{reps.mean():.2f}"],
        ["share with no repeat purchase (all data)", f"{(reps == 0).mean():.3f}"],
    ]))
    w("\nPopulation: every customer acquired by `complete_through(C)` (dropout is unobservable, "
      "so nobody is excluded as churned). `sigma (+ planning signal)` adds a tercile of repeat "
      "purchases in completely recorded months.\n")
    w(table(prec_headers, precision_rows(p_sets, p_extra, alpha, power)))
    w("\nPooled cutoffs 24 and 30 (Kish, rho = 1):\n")
    w(table(pooled_headers, [pooled_row(p_sets, p_spec.evaluation_cutoffs, alpha, power)]))

    # --- 3. Predeclared misspecification variants -----------------------------------------------
    rows = []
    variant_cfgs: list[tuple[str, object]] = [
        (v, gen.signals_variant(v, s_cfg)) for v in gen.SIGNALS_VARIANTS]
    variant_cfgs += [(v, gen.purchases_variant(v, p_cfg)) for v in gen.PURCHASES_VARIANTS]
    for name, cfg in variant_cfgs:
        t = time.perf_counter()
        if isinstance(cfg, gen.SignalsScenarioConfig):
            sc_v: EconomicsTables = gen.generate_signals_scenario(cfg)
            c0 = s_spec.evaluation_cutoffs[0]
        else:
            assert isinstance(cfg, gen.PurchasesScenarioConfig)
            sc_v = gen.generate_purchases_scenario(cfg)
            c0 = p_spec.evaluation_cutoffs[0]
        elapsed = time.perf_counter() - t
        f = common_set(sc_v, c0, pp.PRIMARY_HORIZON)
        rows.append([name, f"{elapsed:.2f}", len(sc_v.customers), len(sc_v.transactions), c0,
                     len(f), float(np.mean(f["realised"] == 0)), float(f["realised"].mean()),
                     sigma_cells(f),
                     pr11.minimum_detectable_relative_excess(len(f), alpha, power)])
    w("\n## 3. Predeclared misspecification variants (pilot seed)\n")
    w(table(["variant", "generation s", "customers", "transactions", "cutoff", "N",
             "share R = 0", "mean R", "sigma (Tier A cells)", "relative MDE"], rows))
    w("\n`correlated_spend` changes only spend (its purchase streams equal the base "
      "scenario's by construction), so differences against the base isolate the spend "
      "assumption.")

    # --- 4. Repeated-sampling feasibility (generation component) -------------------------------
    half_s = replace(s_cfg, acquisitions_per_month=tuple(a / 2 for a in
                                                         s_cfg.acquisitions_per_month))
    half_p = replace(p_cfg, acquisitions_per_month=tuple(a / 2 for a in
                                                         p_cfg.acquisitions_per_month))
    with timed("generate signals scenario, half scale"):
        gen.generate_signals_scenario(half_s)
    with timed("generate purchases scenario, half scale"):
        gen.generate_purchases_scenario(half_p)
    n_rep = pp.COVERAGE_REPLICATIONS
    gen_full = TIMINGS["generate signals scenario"] + TIMINGS["generate purchases scenario"]
    gen_half = (TIMINGS["generate signals scenario, half scale"]
                + TIMINGS["generate purchases scenario, half scale"])
    w("\n## 4. Repeated-sampling feasibility: data-generation component\n")
    w(table(["item", "value"], [
        ["one draw of both new scenarios, full / half scale", f"{gen_full:.2f} / {gen_half:.2f} s"],
        [f"{n_rep} draws, generation only, full / half scale",
         f"~{n_rep * gen_full / 60:.1f} / ~{n_rep * gen_half / 60:.1f} min"],
        ["peak memory of this planning run", f"{peak_rss_gb():.2f} GB"],
    ]))
    w("\nGeneration is not the constraint for 1,000 draws. Whether 1,000 draws fit the "
      "45-minute evidence ceiling depends on the per-draw cost of the oracle filter (P2) and the "
      "model fits (P3-P6), which do not exist yet; the protocol's repeated-sampling preflight "
      "(pending item) is therefore completed only when those costs are measured.")

    # --- header -----------------------------------------------------------------------------------
    digests = [[name, f"`{gen.config_digest(cfg)}`"] for name, cfg in
               [("signals_contractual", s_cfg), ("purchases_noncontractual", p_cfg)]
               + variant_cfgs]
    head = [
        "# Milestone 11 P1 — pilot precision and runtime for the new scenarios",
        "",
        "Generated by `python scripts/plan_m11_precision.py --stage p1`. **Planning record, not "
        "evidence of model performance**: pilot seed only; observable tables only (no latent "
        "truth read, no model fitted); main and fresh seeds untouched. Supersedes the "
        "PROVISIONAL proxy rows of the P0 plan for `signals_contractual` and "
        "`purchases_noncontractual`, at the P1 generator parameters below.",
        "",
        table(["item", "value"], [
            ["protocol", f"{pp.PROTOCOL_ID} ({pp.PROTOCOL_STAGE})"],
            ["protocol digest / code digest",
             f"`{pp.protocol_digest()[:16]}…` / `{pp.protocol_code_digest()[:16]}…`"],
            ["git", git_state()],
            ["seed", f"pilot ({seed})"],
            ["alpha / power / k", f"{alpha:.4g} / {power} / "
                                  f"{pr11.critical_sum(alpha, power):.4f}"],
            ["python / numpy / scipy / pandas",
             f"{platform.python_version()} / {np.__version__} / {scipy.__version__} / "
             f"{pd.__version__}"],
        ]),
        "",
        "Generator parameter digests (`predictive_ltv_scenarios.config_digest`; seed excluded):",
        "",
        table(["configuration", "SHA-256"], digests),
        "",
    ]
    TIMINGS["total"] = time.perf_counter() - start
    lines.append("\n## Runtime\n")
    lines.append(table(["step", "seconds (indicative)"],
                       [[k_, f"{v:.2f}"] for k_, v in TIMINGS.items()]))
    output.write_text("\n".join(head + lines) + "\n")
    print(f"wrote {output} in {TIMINGS['total']:.1f} s")
    return 0


def _policy(promo_months: int, promo_channels: Sequence[bool]) -> PromotionPolicy:
    return PromotionPolicy(promo_months, tuple(c for c, p in zip(CHANNELS, promo_channels,
                                                               strict=True) if p))


def oracle_pair(sc: gen.SignalsScenario, cutoff: int, horizon: int) -> pd.DataFrame:
    """Tier A and Tier B oracles with the realised holdout value (comparison population)."""
    a = orc.signals_oracle(sc, cutoff, horizon, "A").frame["oracle"]
    b = orc.signals_oracle(sc, cutoff, horizon, "B").frame["oracle"].reindex(a.index)
    pol = _policy(sc.config.promo_months, sc.config.promo_channels)
    hf = lm.holdout_frame(sc, sc.config.n_months, cutoff, horizon, "contractual", "A", pol)
    r = hf.set_index("customer_id")["realised"].reindex(a.index)
    return pd.DataFrame({"mu_a": a, "mu_b": b, "r": r})


def slope(mu: np.ndarray, r: np.ndarray) -> float:
    x = mu - mu.mean()
    return float(np.sum(x * (r - r.mean())) / np.sum(x * x))


def slope_se(mu: np.ndarray, r: np.ndarray) -> float:
    """Heteroscedasticity-robust (HC0) standard error of :func:`slope`."""
    x = mu - mu.mean()
    e = r - r.mean() - slope(mu, r) * x
    return float(np.sqrt(np.sum(x * x * e * e)) / np.sum(x * x))


def advantage_row(label: str, f: pd.DataFrame, alpha: float, power: float,
                  rng: np.random.Generator) -> list[object]:
    r, a, b = f["r"].to_numpy(), f["mu_a"].to_numpy(), f["mu_b"].to_numpy()
    d = (r - a) ** 2 - (r - b) ** 2
    ci = bootstrap_ci(d, np.mean, rng=rng, method="percentile",
                      n_resamples=pp.CLUSTERED_INFERENCE.metric_bootstrap_resamples)
    gap = float(np.mean((b - a) ** 2))
    resid = d - (b - a) ** 2
    z = float(resid.mean() / (resid.std(ddof=1) / np.sqrt(resid.size)))
    mse_a, mse_b = float(np.mean((r - a) ** 2)), float(np.mean((r - b) ** 2))
    mde = pr11.minimum_detectable_relative_excess(len(f), alpha, power)
    return [label, len(f), mse_a, mse_b, float(d.mean()), f"[{ci.ci_lower:.2f}, {ci.ci_upper:.2f}]",
            gap, gap / mse_b, mde, gap / mse_b / mde, z, slope(a, r), slope(b, r)]


def pooled_advantage(frames: Sequence[pd.DataFrame], rng: np.random.Generator
                     ) -> tuple[float, float, float, int, int]:
    """Pooled cutoffs: mean paired difference with a customer-cluster bootstrap interval."""
    rows = pd.concat([f.assign(d=(f["r"] - f["mu_a"]) ** 2 - (f["r"] - f["mu_b"]) ** 2)[["d"]]
                      for f in frames])
    per = rows.groupby(level=0)["d"].agg(["sum", "count"])
    sums = per["sum"].to_numpy(dtype=np.float64)
    counts = per["count"].to_numpy(dtype=np.float64)

    def ratio(idx: np.ndarray) -> float:
        j = idx.astype(np.int64)
        return float(np.sum(sums[j])) / float(np.sum(counts[j]))

    ci = bootstrap_ci(np.arange(sums.size, dtype=float), ratio, rng=rng, method="percentile",
                      n_resamples=pp.CLUSTERED_INFERENCE.metric_bootstrap_resamples)
    total = float(np.sum(sums)) / float(np.sum(counts))
    return total, ci.ci_lower, ci.ci_upper, int(np.sum(counts)), int(sums.size)


def purchases_section(seed: int) -> list[str]:
    """Non-contractual oracle on the pilot seed: calibration against realised value (no latent
    truth read) and quadrature convergence (default vs doubled rules)."""
    out: list[str] = []
    spec = pp.scenario("purchases_noncontractual")
    rows, conv = [], []
    for name in (None, *gen.PURCHASES_VARIANTS):
        cfg = replace(gen.PurchasesScenarioConfig(), seed=seed)
        cfg = gen.purchases_variant(name, cfg) if name else cfg
        label = name or "bgnbd (base)"
        sc = gen.generate_purchases_scenario(cfg)
        for c in spec.evaluation_cutoffs:
            with timed(f"purchases oracle: {label}, C={c}"):
                res = por.purchases_oracle(sc, c, pp.PRIMARY_HORIZON)
            f = res.frame
            hf = lm.holdout_frame(sc, cfg.n_months, c, pp.PRIMARY_HORIZON, "non_contractual",
                                  "A").set_index("customer_id")
            r = hf["realised"].reindex(f.index).to_numpy()
            mu = f["oracle"].to_numpy()
            rows.append([label, c, len(f), float(np.mean((r - mu) ** 2)), float(np.mean(r - mu)),
                         float(np.mean(r - mu) / (np.std(r - mu, ddof=1) / np.sqrt(len(f)))),
                         f"{slope(mu, r):.3f} ({slope_se(mu, r):.3f})",
                         float(f["p_alive"].mean()),
                         f"{TIMINGS[f'purchases oracle: {label}, C={c}']:.2f}"])
        h = por.purchase_history(sc, spec.evaluation_cutoffs[0], pp.PRIMARY_HORIZON)
        hf_ = h.frame
        seg = np.array([SEGMENTS.index(v) for v in hf_["segment"]])
        ch = np.array([CHANNELS.index(v) for v in hf_["channel"]])
        args = (hf_["x"].to_numpy(), hf_["t_x"].to_numpy(), hf_["age"].to_numpy())
        alpha = cfg.alpha * np.asarray(cfg.segment_alpha_scale)[seg]
        b = np.asarray(cfg.dropout_b)[ch]
        if cfg.spend_frequency_correlation != 0.0:
            spend = (hf_["n_spend"].to_numpy(), hf_["total_spend"].to_numpy())
            gam = np.asarray(cfg.gg_gamma, dtype=float)[seg]
            n_l, n_p, n_v = por.COPULA_NODES
            lo, hi = (por.bgnbd_copula_conditional(
                *args, *spend, h.edges, r=cfg.r, alpha=alpha, a=cfg.dropout_a, b=b,
                gg_p=cfg.gg_p, gg_q=cfg.gg_q, gamma=gam, rho=cfg.spend_frequency_correlation,
                nodes=(f * n_l, f * n_p, f * n_v))[3].sum(axis=1) for f in (1, 2))
            rule = f"tensor {por.COPULA_NODES} (lambda, p, nu)"
        elif cfg.dropout_process == "bgnbd":
            lo = por.bgnbd_conditional(*args, h.edges, r=cfg.r, alpha=alpha, a=cfg.dropout_a,
                                       b=b)[2].sum(axis=1)
            hi = por.bgnbd_conditional(*args, h.edges, r=cfg.r, alpha=alpha, a=cfg.dropout_a,
                                       b=b, nodes=2 * por.JACOBI_NODES)[2].sum(axis=1)
            rule = f"Gauss-Jacobi {por.JACOBI_NODES} (p)"
        else:
            lo = por.pareto_conditional(*args, h.edges, r=cfg.r, alpha=alpha, s=cfg.pareto_s,
                                        beta=cfg.pareto_beta)[2].sum(axis=1)
            hi = por.pareto_conditional(*args, h.edges, r=cfg.r, alpha=alpha, s=cfg.pareto_s,
                                        beta=cfg.pareto_beta,
                                        nodes=2 * por.PARETO_NODES)[2].sum(axis=1)
            rule = f"Gauss-Jacobi {por.PARETO_NODES} (death time)"
        material = hi > 1e-9
        conv.append([label, rule, int(material.sum()),
                     f"{float(np.max(np.abs(lo - hi)[material] / hi[material])):.1e}"])
    out.append("\n## 4. `purchases_noncontractual`: observable-information oracle\n")
    out.append("BG/NBD + Gamma-Gamma (base) and each predeclared variant with its own oracle "
               "(true process and parameters; complete-month information at `C`). Realised "
               "value is the M10 contribution of purchases in months `C+1 .. C+12`; the "
               "z-statistic tests `mean(R - oracle) = 0`.\n")
    out.append(table(["scenario", "cutoff", "N", "oracle MSE (noise floor)", "mean residual",
                      "z", "calibration slope (HC SE)", "mean P(alive at C)",
                      "oracle seconds"], rows))
    out.append("\nQuadrature convergence (pilot, first cutoff): maximum relative change in "
               "expected purchases (copula: expected spend) when every rule is doubled, over "
               "customers with a non-negligible prediction.\n")
    out.append(table(["scenario", "default rule", "customers", "max relative change"], conv))
    return out


def plan_p2(seed: int, output: Path) -> int:
    start = time.perf_counter()
    alpha, power = pp.PLANNING_ALPHA, pp.PLANNING_POWER
    rng = np.random.default_rng(BENCH_SEED)
    lines: list[str] = []
    w = lines.append
    headers = ["case", "N", "MSE Tier A oracle", "MSE Tier B oracle", "D = MSE_A - MSE_B",
               "D 95% CI (customer bootstrap)", "E[(mu_B - mu_A)^2]",
               "relative Tier B advantage (/ MSE_B)", "relative MDE", "advantage / MDE",
               "gate z (D vs gap)", "calibration slope A", "calibration slope B"]
    s_spec = pp.scenario("signals_contractual")
    pairs = [(c, pp.PRIMARY_HORIZON) for c in s_spec.evaluation_cutoffs]
    pairs += [(c, h) for h, c in s_spec.secondary_horizon]

    # --- 1. Pilot seed ------------------------------------------------------------------------
    with timed("generate signals scenario (pilot)"):
        sig = gen.generate_signals_scenario(replace(gen.SignalsScenarioConfig(), seed=seed))
    frames: dict[tuple[int, int], pd.DataFrame] = {}
    with timed("oracles A and B + holdout labels (pilot, 3 cutoffs)"):
        for c, h in pairs:
            frames[(c, h)] = oracle_pair(sig, c, h)
    rows = [advantage_row(f"pilot C={c} W={h}", frames[(c, h)], alpha, power, rng)
            for c, h in pairs]
    with timed("NPH variant oracles"):
        nph = gen.generate_signals_scenario(gen.signals_variant(
            "non_proportional_hazards", replace(gen.SignalsScenarioConfig(), seed=seed)))
        c0 = s_spec.evaluation_cutoffs[0]
        rows.append(advantage_row(f"pilot NPH C={c0} W={pp.PRIMARY_HORIZON}",
                                  oracle_pair(nph, c0, pp.PRIMARY_HORIZON), alpha, power, rng))
    pooled = pooled_advantage([frames[(c, pp.PRIMARY_HORIZON)]
                               for c in s_spec.evaluation_cutoffs], rng)
    w("## 1. Tier B advantage and oracle calibration (`signals_contractual`)\n")
    w("`D` is the paired difference in realised squared error between the exact Tier A and "
      "Tier B oracles; in population expectation `D = E[(mu_B - mu_A)^2] >= 0` (amendment 1). "
      "The gate column is the z-statistic of `mean(D_i - (mu_B - mu_A)_i^2)`, whose expectation "
      "is 0; a single seed's sign of `D` is reported, not required. `relative Tier B "
      "advantage` = `E[(mu_B - mu_A)^2] / MSE_B` is the excess error of the Tier A oracle "
      "relative to the Tier B noise floor, directly comparable with the relative MDE "
      "(`4k^2/N`, alpha = Holm's strictest step). Calibration slopes regress realised value on "
      "each oracle (1 for a calibrated predictor).\n")
    w(table(headers, rows))
    w(f"\nPooled cutoffs {s_spec.evaluation_cutoffs} (customer-cluster bootstrap, "
      f"{pooled[3]} rows, {pooled[4]} customers): D = {pooled[0]:.2f}, 95% CI "
      f"[{pooled[1]:.2f}, {pooled[2]:.2f}].")

    # --- 2. Population level over development seeds -----------------------------------------------
    dev_rows = []
    with timed(f"population level: {len(P2_DEV_SEEDS)} development seeds"):
        for s in P2_DEV_SEEDS:
            sc = gen.generate_signals_scenario(replace(gen.SignalsScenarioConfig(), seed=s))
            for c in s_spec.evaluation_cutoffs:
                f = oracle_pair(sc, c, pp.PRIMARY_HORIZON)
                r, a, b = f["r"].to_numpy(), f["mu_a"].to_numpy(), f["mu_b"].to_numpy()
                gap = float(np.mean((b - a) ** 2))
                dev_rows.append((s, c, len(f), float(np.mean((r - a) ** 2 - (r - b) ** 2)), gap,
                                 gap / float(np.mean((r - b) ** 2)),
                                 pr11.minimum_detectable_relative_excess(len(f), alpha,
                                                                         power)))
    dev = pd.DataFrame(dev_rows, columns=["seed", "cutoff", "N", "D", "gap", "advantage",
                                          "mde"])
    summary = dev.groupby("cutoff").agg(
        draws=("seed", "size"), mean_N=("N", "mean"), mean_D=("D", "mean"),
        sd_D=("D", "std"), mean_gap=("gap", "mean"), mean_advantage=("advantage", "mean"),
        sd_advantage=("advantage", "std"), min_advantage=("advantage", "min"),
        mean_mde=("mde", "mean"))
    w("\n## 2. Population-level Tier B advantage (independent development draws)\n")
    w(f"Development seeds {P2_DEV_SEEDS[0]}-{P2_DEV_SEEDS[-1]} at the pinned parameters "
      "(`E[(mu_B - mu_A)^2]` is a population quantity; its across-draw mean estimates it).\n")
    w(table(["cutoff", "draws", "mean N", "mean D", "SD of D", "mean E[(mu_B - mu_A)^2]",
             "mean relative advantage", "SD", "min", "mean relative MDE"],
            [[c, int(r_.draws), float(r_.mean_N), float(r_.mean_D), float(r_.sd_D),
              float(r_.mean_gap), float(r_.mean_advantage), float(r_.sd_advantage),
              float(r_.min_advantage), float(r_.mean_mde)]
             for c, r_ in summary.iterrows()]))

    # --- 3. M10 scenario Tier A oracle ----------------------------------------------------------
    m10_rows = []
    with timed("M10 scenario oracle (pilot)"):
        m10sc = generate_economics_scenario(replace(EconomicsScenarioConfig(), seed=seed))
        pol = _policy(m10sc.config.promo_months, m10sc.config.promo_channels)
        for c in pp.scenario("m10_contractual").evaluation_cutoffs:
            mu = orc.m10_oracle(m10sc, c, pp.PRIMARY_HORIZON).frame["oracle"]
            hf = lm.holdout_frame(m10sc, m10sc.config.n_months, c, pp.PRIMARY_HORIZON,
                                  "contractual", "A", pol).set_index("customer_id")
            r = hf["realised"].reindex(mu.index).to_numpy()
            m10_rows.append([c, len(mu), float(np.mean((r - mu.to_numpy()) ** 2)),
                             float(np.mean(r - mu.to_numpy())), slope(mu.to_numpy(), r)])
    w("\n## 3. `m10_contractual`: exact Tier A oracle (closed-form Gamma posterior)\n")
    w(table(["cutoff", "N", "oracle MSE (noise floor for PC1)", "mean residual",
             "calibration slope"], m10_rows))

    lines.extend(purchases_section(seed))

    # --- 5. Runtime of the P2 machinery -----------------------------------------------------------
    pol_s = _policy(sig.config.promo_months, sig.config.promo_channels)
    c_last = s_spec.evaluation_cutoffs[-1]
    with timed(f"training stack Tier B, C={c_last} (pilot)"):
        stack = lm.training_stack(sig, c_last, pp.PRIMARY_HORIZON, "contractual", "B", pol_s)
    w("\n## 5. Runtime\n")
    w(table(["item", "value"], [
        [f"training stack rows (C={c_last}, Tier B)",
         f"{len(stack)} rows, {stack['customer_id'].nunique()} customers"],
        ["peak memory", f"{peak_rss_gb():.2f} GB"]]))

    head = [
        "# Milestone 11 P2 — oracle calibration and the Tier B advantage",
        "",
        "Generated by `python scripts/plan_m11_precision.py --stage p2`. **Planning record, "
        "not evidence of model performance**: no model is fitted. The oracles are evaluation "
        "references that use the generator's true parameters and only record-time "
        "observable data (P2). Pilot and development seeds only; main and fresh seeds "
        "untouched; generator parameters unchanged (pinned digests).",
        "",
        table(["item", "value"], [
            ["protocol", f"{pp.PROTOCOL_ID} ({pp.PROTOCOL_STAGE})"],
            ["protocol digest / code digest",
             f"`{pp.protocol_digest()[:16]}…` / `{pp.protocol_code_digest()[:16]}…`"],
            ["git", git_state()],
            ["seeds", f"pilot ({seed}); development {P2_DEV_SEEDS[0]}-{P2_DEV_SEEDS[-1]}"],
            ["python / numpy / scipy / pandas",
             f"{platform.python_version()} / {np.__version__} / {scipy.__version__} / "
             f"{pd.__version__}"],
        ]),
        "",
    ]
    TIMINGS["total"] = time.perf_counter() - start
    lines.append("\n## Step timings\n")
    lines.append(table(["step", "seconds (indicative)"],
                       [[k_, f"{v:.2f}"] for k_, v in TIMINGS.items()]))
    output.write_text("\n".join(head + lines) + "\n")
    print(f"wrote {output} in {TIMINGS['total']:.1f} s")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--stage", choices=("p0", "p1", "p2"), default="p0")
    args = parser.parse_args(argv)
    problems = pp.validate_protocol()
    if problems:
        raise SystemExit("protocol inconsistent: " + "; ".join(problems))
    seed = pp.authorise_seed("pilot")
    if args.stage == "p1":
        return plan_p1(seed, args.output or P1_OUTPUT)
    if args.stage == "p2":
        return plan_p2(seed, args.output or P2_OUTPUT)
    args.output = args.output or OUTPUT
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
