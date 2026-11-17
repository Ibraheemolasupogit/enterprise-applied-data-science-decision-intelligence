"""Generate the Milestone 10 (LTV foundations and customer economics) evidence report.

Writes ``docs/evidence/m10-ltv-foundations-results.md``. Seeded; every number reproduces exactly
with the recorded versions (the runtime table excepted). The fresh seed is evaluated only in the
last section, once, after the protocol and code are frozen.

Usage:
    python scripts/generate_m10_evidence.py
    python scripts/generate_m10_evidence.py --dry-run OUT.md   # skip the fresh seed
"""

from __future__ import annotations

import math
import platform
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy

from decision_intelligence.common.bootstrap import bootstrap_ci
from decision_intelligence.customer_value import cases as c
from decision_intelligence.customer_value import historical as hi
from decision_intelligence.customer_value import protocol as pr
from decision_intelligence.customer_value.economics import channel_cac
from decision_intelligence.data.customer_economics_scenarios import CHANNELS

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "docs" / "evidence" / "m10-ltv-foundations-results.md"
_CACHE: dict[str, Any] = {}


def table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    def fmt(value: object) -> str:
        if isinstance(value, bool | np.bool_):
            return "yes" if value else "no"
        if isinstance(value, float | np.floating):
            value = float(value)
            if math.isnan(value):
                return "—"
            if math.isinf(value):
                return "inf"
            if value.is_integer() and abs(value) >= 1e4:
                return str(int(value))
            return f"{value:.4g}" if abs(value) >= 1e-3 or value == 0 else f"{value:.2e}"
        return str(value)

    def cell(value: object) -> str:
        return fmt(value).replace("|", "\\|")

    lines = ["| " + " | ".join(cell(h) for h in headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def frame_table(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> str:
    cols = list(columns) if columns is not None else list(frame.columns)
    return table(cols, frame[cols].itertuples(index=False, name=None))


def main_sc() -> Any:
    if "main" not in _CACHE:
        _CACHE["main"] = c.scenario(pr.MAIN_SEED)
    return _CACHE["main"]


def protocol_section() -> str:
    rows = [["seeds", f"main {pr.MAIN_SEED}; pilot {pr.PILOT_SEED} (precision planning only); "
                      f"fresh {pr.FRESH_SEED} (§13 only); coverage {pr.COVERAGE_SEED_START}+"],
            ["periods and discounting", f"monthly; age 0 = acquisition month, undiscounted; "
                                        f"{pr.ANNUAL_DISCOUNT_RATE:.0%} a year = "
                                        f"{pr.MONTHLY_DISCOUNT_RATE:.6f} a month"],
            ["horizons", f"{pr.HORIZONS} months projected; infinite = expected lifetime "
                         "(conditional on convergence)"],
            ["primary estimand", f"{pr.PRIMARY_LTV_AGE}-month realised discounted contribution "
                                 "per acquired customer, complete cohorts only; LTV/CAC per "
                                 "channel (media CAC)"],
            ["precision target", f"95% interval half-width ≤ {pr.PRECISION_TARGET:.0%} of "
                                 "LTV/CAC per channel (a target; reported, design not changed)"],
            ["heuristic windows", f"churn and run rate over {pr.CHURN_WINDOW_MONTHS} months; "
                                  f"'unstable' below {pr.MIN_CHURN_EVENTS} churn events; KM used "
                                  f"while ≥ {pr.MIN_AT_RISK} at risk, tail over "
                                  f"{pr.KM_TAIL_MONTHS} months"],
            ["backtests", f"{pr.BACKTEST_WINDOW_MONTHS}-month windows at cutoffs "
                          f"{pr.BACKTEST_CUTOFFS}; read after a {pr.MAX_RECORDING_LAG_DAYS}-day "
                          "recording lag"],
            ["grids", f"frailty variance {pr.HETEROGENEITY_GRID}; discount rates "
                      f"{pr.SENSITIVITY_DISCOUNT_RATES}; margin {pr.SENSITIVITY_MARGIN_SHIFTS}; "
                      f"cost-to-serve {pr.SENSITIVITY_COST_SHIFTS}"],
            ["repeated sampling", f"coverage {pr.COVERAGE_REPLICATIONS} generator draws; "
                                  f"bootstrap {pr.BOOTSTRAP_RESAMPLES}"]]
    return (
        "Fixed in `customer_value/protocol.py` before any M10 estimate was computed.\n\n"
        + table(["item", "frozen value"], rows)
        + "\n\n**Disclosures.**\n\n"
        "1. *Protocol error found before evaluation:* the backtest cutoff 36 cannot have a "
        "complete 12-month window (it needs month 48; the data end at month 47). The protocol was "
        "not edited; the code refuses incomplete windows and §10 reports the cutoff as "
        "excluded.\n"
        "2. *Plan change:* the M3 Weibull survival function is not a reference for this generator "
        "(geometric churn with Gamma heterogeneity); the truth is checked against the Gamma "
        "Laplace transform (numerical integration) and Monte Carlo instead.\n"
        "3. *Designed-in mechanisms (not discoveries):* the premium segment has the highest price "
        "but a high cost-to-serve; the cheapest channel per acquisition (social) brings "
        "high-churn, low-value customers.\n"
        "4. *Corrections before any main-seed result* (development seeds and tests): the customer "
        "backtest compared projections and realised values on different customer sets (fixed: "
        "one comparison set); the survivors-only counterexample mixed survivorship and "
        "truncation (redefined as complete-window survivors); Kaplan–Meier computed 0 × ∞ when "
        "survival reached 0 (restructured).\n"
        "5. *Defect found from main-seed dry-run output:* Kaplan–Meier censored customers still "
        "active at the cutoff at their number of paid months t instead of t - 1 (the churn "
        "decision after month t is not yet observed), biasing survival upwards (the dry run's "
        "Greenwood intervals contained the truth in 72% of cells). Fixed with a regression test "
        "and every result regenerated; it is a correctness fix, not a choice among alternatives, "
        "and no other setting was changed after the main-seed output was seen.\n"
    )


def precision_section() -> str:
    sc = c.scenario(pr.PILOT_SEED)
    obs = c.final_view(sc)
    cm = hi.cohort_matched(obs, pr.PRIMARY_LTV_AGE, c.RATE)
    cac = {x.channel: x for x in channel_cac(obs.marketing, obs.overhead,
                                             sorted(cm["acquisition_month"].unique()))}
    rng = np.random.default_rng(1)
    rows = []
    for ch in CHANNELS:
        v = cm.loc[cm["channel"] == ch, "value"].to_numpy(dtype=np.float64)
        b = bootstrap_ci(v, lambda x: float(np.mean(x)), rng=rng,
                         n_resamples=pr.BOOTSTRAP_RESAMPLES, method="percentile")
        half = (b.ci_upper - b.ci_lower) / 2 / b.estimate
        rows.append([ch, v.size, b.estimate / cac[ch].media, half, half <= pr.PRECISION_TARGET])
    met = sum(r[4] for r in rows)
    return (
        f"Run on the **pilot seed** ({pr.PILOT_SEED}) before any main-seed result was examined, "
        "at the frozen design (acquisition volumes, horizon, cohorts with complete 24-month "
        "windows). Planned 95% bootstrap half-width of LTV/CAC per channel:\n\n"
        + table(["channel", "customers", "LTV/CAC (media)", "relative half-width",
                 f"≤ {pr.PRECISION_TARGET:.0%}"], rows)
        + f"\n\nReading: the target is planned to be met in {met} of {len(rows)} channels. The "
        "achieved precision on the main seed is in §7, including smaller segment × channel "
        "cells, where it may fail.\n"
    )


def audit_section() -> str:
    a = c.data_audit(main_sc())
    return table(["quantity", "value"], [[k, v] for k, v in a.items()])


def retention_section() -> str:
    r = c.retention_table(main_sc())
    cov = r["covers"].mean()
    return (
        "Kaplan–Meier P(paid at age a) with Greenwood 95% intervals (plain), all customers at the "
        "data end, against the population truth (Gamma-frailty survival, arrivals-weighted).\n\n"
        + frame_table(r)
        + f"\n\nReading: the interval contains the population truth in {cov:.0%} of the "
        f"{len(r)} group-age cells (a single realisation; the repeated-sampling check is in the "
        "tests). Estimates at different ages share data, so these are not independent trials.\n"
    )


def historical_section() -> str:
    h = c.historical_table(main_sc())
    below = int((h["naive: all customers"] < h["truth"]).sum())
    above = int((h["naive: survivors only"] > h["truth"]).sum())
    return (
        "**Realised** discounted contribution per acquired customer over the first 12 and 24 "
        "months — complete cohorts only, churned customers included as zeros — vs the truth; "
        "and the two naive counterexamples.\n\n" + frame_table(h)
        + f"\n\nReading: |z| ≤ 2 in {int((h['z'].abs() <= 2).sum())} of {len(h)} cells. The "
        f"all-customers mean is below the truth in {below} of {len(h)} cells (truncation); the "
        f"survivors-only mean is above it in {above} of {len(h)} (survivorship).\n"
    )


def heuristic_section() -> str:
    sc = main_sc()
    ch = c.heuristic_table(sc, by="channel")
    seg = c.heuristic_table(sc, by="segment")
    rc = c.revenue_vs_contribution(sc)
    st = c.stability_flags(sc)
    conv = c.convergence_table(sc)
    cols = ["horizon", "quantity", "truth", "H1/H2 constant churn", "H1/H2 constant churn bias",
            "H3 by cell", "H3 by cell bias", "H4 Kaplan–Meier", "H4 Kaplan–Meier bias"]
    proj = pd.concat([ch.assign(group=ch["channel"]), seg.assign(group=seg["segment"])])
    proj = proj[proj["quantity"] != "historical + projected revenue"]
    bias_cols = ["H1/H2 constant churn bias", "H3 by cell bias", "H4 Kaplan–Meier bias"]
    summary = proj.groupby("quantity")[bias_cols].agg(lambda s: float(np.mean(np.abs(s))))
    legacy = ch[ch["quantity"] == "historical + projected revenue"]
    statuses = pd.concat([ch, seg])[[f"{n} status" for n, _ in c.HEURISTICS]].stack()
    reversal = bool((rc["rank by revenue"] != rc["rank by contribution"]).any())
    n_unstable = int((st["status"] == "unstable").sum())
    return (
        "**Projected** (24, 36 months) and **expected-lifetime** (infinite horizon) discounted "
        "contribution per new customer, estimated at the data end, vs the truth.\n\n"
        "By channel:\n\n" + frame_table(ch[ch["quantity"] != "historical + projected revenue"],
                                         ["channel", *cols])
        + "\n\nBy segment:\n\n" + frame_table(seg[seg["quantity"] != "historical + projected "
                                                                       "revenue"],
                                              ["segment", *cols])
        + "\n\nMean absolute relative bias:\n\n" + frame_table(summary.reset_index())
        + "\n\nEstimate statuses across both tables: "
        + ", ".join(f"{k} {v}" for k, v in statuses.value_counts().items())
        + ".\n\n**Legacy baseline (H0)** — historical revenue + 12-month churn-adjusted revenue, "
        "undiscounted, per customer (revenue, not contribution, and not comparable with the "
        "truth above):\n\n" + frame_table(legacy, ["channel", "H1/H2 constant churn"])
        + "\n\n**Revenue vs contribution** (designed-in): segment ranking by H4 36-month value.\n\n"
        + frame_table(rc)
        + f"\n\nReading: the revenue ranking {'differs from' if reversal else 'matches'} the "
        "contribution ranking.\n\n**Stability**: H1 expected-lifetime value per segment × channel "
        f"cell; {n_unstable} of {len(st)} cells are flagged unstable (fewer than "
        f"{pr.MIN_CHURN_EVENTS} churn events in the window).\n\n" + frame_table(st)
        + "\n\n**Infinite-horizon convergence** (all customers): finite-horizon values approach "
        "the infinite-horizon value; with neither churn nor discounting the infinite horizon is "
        "undefined and is refused.\n\n"
        + frame_table(conv, ["horizon", "H1/H2 value", "H1 infinite", "relative gap (H1)",
                             "truth value", "truth infinite (summed to 1200)"])
        + f"\n\nRefusal check: {conv['note'].iloc[0]}.\n"
    )


def unit_section() -> str:
    ue = c.unit_economics(main_sc())
    ch, cells = ue.channels, ue.cells
    cheapest = ch.loc[ch["CAC media"].idxmin(), "channel"]
    best = ch.loc[ch["LTV/CAC (media)"].idxmax(), "channel"]
    return (
        "Per channel, cohorts with complete 24-month windows: CAC (media; fully loaded adds "
        "onboarding and allocated overhead), realised 24-month LTV, LTV/CAC with its 95% "
        "bootstrap interval (customers resampled; spend treated as known), precision vs the "
        "target, payback and the truth (true CPA).\n\n" + frame_table(ch)
        + f"\n\nReading: precision target met in {int(ch['precision target met'].sum())} of "
        f"{len(ch)} channels. Cheapest channel per acquisition: {cheapest}; best LTV/CAC: {best}."
        "\n\nSegment × channel cells:\n\n" + frame_table(cells)
        + f"\n\nReading: the ±{pr.PRECISION_TARGET:.0%} target is **not met** in "
        f"{int((~cells['precision target met']).sum())} of {len(cells)} cells (small cells).\n"
    )


def segment_section() -> str:
    return frame_table(c.segment_economics(main_sc()))


def uncertainty_section() -> str:
    h1 = c.h1_uncertainty(main_sc())
    cov = c.coverage_study()
    rows = cov.rows
    fails = rows.loc[rows["LTV/CAC binomial p"] < 0.05, "channel"].tolist()
    fails_ltv = rows.loc[rows["LTV24 binomial p"] < 0.05, "channel"].tolist()
    return (
        "H1 expected-lifetime value per channel: delta-method SE (churn and run-rate) vs a "
        "customer bootstrap (1,000 resamples).\n\n" + frame_table(h1)
        + f"\n\n**Coverage in repeated sampling** ({cov.n_replications} generator draws): the 95% "
        "percentile bootstrap interval of 24-month realised LTV per channel vs the population "
        "truth, and the same interval divided by the estimated CAC vs the true LTV/CPA (the "
        "interval ignores the spend noise in CAC).\n\n" + frame_table(rows)
        + "\n\nReading: LTV24 coverage significantly below 95% (exact binomial p < 0.05) for: "
        + (", ".join(fails_ltv) if fails_ltv else "none")
        + "; LTV/CAC coverage significantly below 95% for: "
        + (", ".join(fails) if fails else "none") + ".\n"
    )


def heterogeneity_section() -> str:
    g = c.heterogeneity_grid(pr.MAIN_SEED)
    return (
        "Same seed, churn heterogeneity varied (all customers; projected 36-month and "
        "expected-lifetime value; relative bias vs the truth).\n\n" + frame_table(g)
        + "\n\nReading: as heterogeneity grows the population hazard falls with tenure, which a "
        "constant churn rate cannot represent; compare the infinite-horizon rows.\n"
    )


def backtest_section() -> str:
    sc = main_sc()
    cust, ex1 = c.customer_backtests(sc)
    coh, ex2 = c.cohort_backtests(sc)
    return (
        "**Existing customers** (baseline only — predictive LTV is M11): 12-month undiscounted "
        "contribution of customers active at the cutoff; projections vs realised and vs the "
        "truth's expectation; one comparison set per cutoff.\n\n" + frame_table(cust)
        + "\n\n**New-customer cohorts**: projected 12-month discounted value per acquired "
        "customer at the cutoff vs the realised value of later cohorts with complete windows, "
        "and the truth.\n\n" + frame_table(coh)
        + "\n\nExcluded cutoffs (complete-window rule): " + "; ".join(sorted(set(ex1 + ex2)))
        + ".\n"
    )


def sensitivity_section() -> str:
    return (
        "H4 projected (36 months) and expected-lifetime value per segment, one factor at a time "
        "around the base case (10%, no margin or cost change). The undiscounted infinite "
        "horizon is reported only where it converges.\n\n" + frame_table(c.sensitivity(main_sc()))
    )


def m2_section() -> str:
    return (
        "The M2 subscription tables (365 days; read-only). M2 has no cost-to-serve or "
        "acquisition spend, so contribution here is gross margin only; tenure is observed for "
        "about one year at most, so longer retention and payback cannot be observed.\n\n"
        + frame_table(c.m2_compatibility()) + "\n"
    )


def fresh_section() -> str:
    sc = c.scenario(pr.FRESH_SEED)
    ue = c.unit_economics(sc)
    h = c.heuristic_table(sc)
    hist = c.historical_table(sc, ages=(24,))
    cust, _ = c.customer_backtests(sc)
    cols = ["channel", "horizon", "truth", "H1/H2 constant churn bias", "H3 by cell bias",
            "H4 Kaplan–Meier bias"]
    return (
        f"The frozen pipeline on seed {pr.FRESH_SEED}, evaluated once.\n\n"
        + frame_table(ue.channels, ["channel", "customers", "CAC media", "LTV24 realised",
                                    "LTV/CAC (media)", "LTV/CAC 95% CI", "relative half-width",
                                    "precision target met", "truth LTV/CAC (true CPA)",
                                    "payback discounted (media CAC)"])
        + "\n\n" + frame_table(hist, ["channel", "realised (cohort-matched)", "truth", "z"])
        + "\n\n" + frame_table(h[h["quantity"] != "historical + projected revenue"], cols)
        + "\n\n" + frame_table(cust, ["cutoff", "method", "bias_vs_realised", "bias_vs_expected",
                                      "spearman_vs_realised", "top_decile_capture"]) + "\n"
    )


def main() -> None:
    sections: list[tuple[str, Callable[[], str]]] = [
        ("1. Frozen protocol and disclosures", protocol_section),
        ("2. Precision plan (pilot seed, before main results)", precision_section),
        ("3. Data audit", audit_section),
        ("4. Retention: Kaplan–Meier vs truth", retention_section),
        ("5. Realised (historical) value", historical_section),
        ("6. Heuristic projected and expected-lifetime value", heuristic_section),
        ("7. Unit economics: CAC, LTV/CAC, payback, precision", unit_section),
        ("8. Segment economics", segment_section),
        ("9. Uncertainty and repeated-sampling coverage", uncertainty_section),
        ("10. Heterogeneity and constant-churn bias", heterogeneity_section),
        ("11. Backtests with complete windows", backtest_section),
        ("12. Sensitivity and M2 compatibility",
         lambda: sensitivity_section() + "\n\n" + m2_section()),
        ("13. Fresh-seed replication", fresh_section),
    ]
    dry_run = "--dry-run" in sys.argv
    output = Path(sys.argv[sys.argv.index("--dry-run") + 1]) if dry_run else OUTPUT
    rendered, runtimes = [], []
    for title, build in sections:
        if dry_run and title.startswith("13."):
            rendered.append(f"## {title}\n\nNot run (dry run: the fresh seed stays untouched).\n")
            continue
        start = time.perf_counter()
        rendered.append(f"## {title}\n\n{build()}\n")
        runtimes.append((title, time.perf_counter() - start))
        print(f"{title}: {runtimes[-1][1]:.0f} s", flush=True)
    environment = table(["component", "version"],
                        [["python", platform.python_version()], ["numpy", np.__version__],
                         ["scipy", scipy.__version__], ["pandas", pd.__version__]])
    header = (
        "# Milestone 10 evidence — numerical results\n\n"
        "Generated by `python scripts/generate_m10_evidence.py`; do not edit by hand. Seeded; "
        "every number reproduces exactly with the versions below (the runtime table excepted). "
        "Narrative and limitations: [m10-ltv-foundations.md](m10-ltv-foundations.md); "
        "methodology: [../methodology/customer-economics.md](../methodology/customer-economics.md)"
        ".\n\nThree quantities are kept apart: **realised** (historical) contribution, "
        "**projected** contribution (finite horizon) and **expected lifetime** contribution "
        "(infinite horizon). All values are discounted at 10% a year unless stated, in fictional "
        "currency units, on synthetic data; the truth is read only after estimation.\n"
    )
    output.write_text(header + "\n" + "\n".join(rendered) + "\n## Environment\n\n" + environment
                      + "\n\n## Runtime\n\n"
                      + table(["section", "runtime (s, indicative)"],
                              [[t, s] for t, s in runtimes]) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
