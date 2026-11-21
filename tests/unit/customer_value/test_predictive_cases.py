"""M11 P8 (A12): fast E4 path and fallback, PC_TEST, Holm, PC2 conventions, PC3 rule,
aggregators, seed isolation, determinism and failure behaviour (development seeds only)."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests

from decision_intelligence.customer_value import btyd
from decision_intelligence.customer_value import contractual as ct
from decision_intelligence.customer_value import heuristic as he
from decision_intelligence.customer_value import predictive_cases as pc
from decision_intelligence.customer_value import predictive_evaluation as ev
from decision_intelligence.customer_value import predictive_protocol as pp
from decision_intelligence.customer_value.panel import observe
from decision_intelligence.customer_value.protocol import CHURN_WINDOW_MONTHS
from decision_intelligence.data import customer_economics_scenarios as m10g

REPO = Path(__file__).resolve().parents[3]


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "generate_m11_evidence", REPO / "scripts" / "generate_m11_evidence.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["generate_m11_evidence"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def e4() -> btyd.E4Prediction:
    return btyd.predict(pc.purchases_scenario(6151, 0.5), 24, pp.PRIMARY_HORIZON)


# --- E4_ACCELERATED_SIMULATION ------------------------------------------------------------------
def test_accelerated_and_fallback_rules_match_the_original_per_count_rule() -> None:
    rng = np.random.default_rng(6152)
    params = np.exp(np.column_stack([rng.normal(np.log(0.6), 0.2, 6),
                                     rng.normal(np.log(8.0), 0.2, 6),
                                     rng.normal(np.log(0.7), 0.3, 6),
                                     rng.normal(np.log(2.5), 0.3, 6)]))
    x = np.array([0, 1, 3, 10, 30, 44, 45, 60, 90], dtype=float)  # beyond 44: fallback
    t_x = np.minimum(x * 1.5, 100.0)
    T = np.full(x.size, 104.0)
    edges = np.linspace(8.7, 60.8, 13)
    got, p_alive = pc.e4_window_purchases(params, x, t_x, T, float(edges[0]), float(edges[-1]))
    for d in range(params.shape[0]):
        ref = btyd.bgnbd_conditional(tuple(params[d]), x, t_x, T, edges)
        np.testing.assert_allclose(got[d], ref.purchases.sum(axis=1), rtol=1e-9)
        np.testing.assert_allclose(p_alive[d], ref.p_alive_T, rtol=1e-12)
    # forcing the fallback everywhere uses the original nodes; the only difference is the
    # telescoped (two-edge) sum, so agreement is to rounding (measured <= 3e-12 relative)
    fb, _ = pc.e4_window_purchases(params, x, t_x, T, float(edges[0]), float(edges[-1]),
                                   max_x=-1)
    for d in range(params.shape[0]):
        ref = btyd.bgnbd_conditional(tuple(params[d]), x, t_x, T, edges)
        np.testing.assert_allclose(fb[d], ref.purchases.sum(axis=1), rtol=1e-10)
    a, b = np.array([0.4, 1.3]), np.array([2.0, 30.0])
    pn, w = pc.beta_nodes_batch(a, b)
    for i in range(2):
        ref_n, ref_w = btyd.beta_nodes(float(a[i]), float(b[i]))
        np.testing.assert_allclose(pn[i], ref_n, atol=1e-13)
        np.testing.assert_allclose(w[i], ref_w, atol=1e-13)


def test_fast_e4_simulation_is_equivalent_to_p7(e4: btyd.E4Prediction) -> None:
    prng = np.random.default_rng(6153)
    draws = {k: (ev._draw_log_params(bf.log_params, bf.cov_log, 6, prng)[0],
                 ev._draw_log_params(gf.log_params, gf.cov_log, 6, prng)[0])
             for k, (_, bf, _, gf) in e4.fits.used.items()}
    fast = pc.e4_simulate_fast(e4, 6, np.random.default_rng(6154), param_draws=draws)
    f, edges = e4.data.frame, e4.data.edges
    for key, (bd, gd) in draws.items():
        sel = ((f["segment"] == key[0]) & (f["channel"] == key[1])).to_numpy()
        for d in range(6):
            cond = btyd.bgnbd_conditional(tuple(np.exp(bd[d])), f["x"].to_numpy()[sel],
                                          f["t_x"].to_numpy()[sel], f["T"].to_numpy()[sel],
                                          edges)
            spend = btyd.gg_expected_spend(tuple(np.exp(gd[d])), f["n"].to_numpy()[sel],
                                           f["S"].to_numpy()[sel])
            np.testing.assert_allclose(
                fast.expected[sel, d],
                cond.purchases.sum(axis=1) * spend * e4.data.contribution_ratio, rtol=1e-9)
    old = ev.e4_simulate(e4, 30, np.random.default_rng(6155))
    new = pc.e4_simulate_fast(e4, 30, np.random.default_rng(6156))
    diff = old.outcome.mean() - new.outcome.mean()
    se = math.sqrt(old.outcome.var() / old.outcome.size + new.outcome.var() / new.outcome.size)
    assert abs(diff) < 4.5 * se  # same outcome distribution (different random streams)


# --- PC_TEST, Holm, PC3 --------------------------------------------------------------------------
def test_paired_test_matches_a_brute_force_cr1_z() -> None:
    rng = np.random.default_rng(6157)
    cid = np.repeat(np.arange(300), 2)
    r = rng.gamma(2.0, 10.0, cid.size)
    a = r + rng.normal(0, 4, cid.size)
    b = r + rng.normal(0, 5, cid.size)
    a[7] = np.nan
    t = pc.paired_test(a, b, r, cid, np.random.default_rng(6158), resamples=300)
    ok = np.isfinite(a)
    d = (r[ok] - a[ok]) ** 2 - (r[ok] - b[ok]) ** 2
    sums = pd.Series(d - d.mean()).groupby(cid[ok]).sum().to_numpy()
    g = sums.size
    se = math.sqrt(g / (g - 1) * np.sum(sums ** 2)) / d.size
    assert t.estimate == pytest.approx(d.mean()) and t.se == pytest.approx(se)
    assert t.p_value == pytest.approx(2 * stats.norm.sf(abs(d.mean() / se)))
    assert (t.excluded, t.rows, t.customers) == (1, cid.size - 1, 300)
    assert t.ci_lower < t.estimate < t.ci_upper


def test_holm_and_the_five_p_values_with_a_non_evaluable_pc3() -> None:
    p = {"PC1": 0.001, "PC2": 0.04, "PC3": 0.2, "PC4": 0.012, "PC5": 0.03}
    ref = multipletests(list(p.values()), alpha=0.05, method="holm")[1]
    assert list(pc.holm(p).values()) == pytest.approx(list(ref))
    t = pc.PairedTest(1.0, 0.5, 2.0, 0.0455, 0.1, 1.9, 10, 5, 0)
    vals = pc.primary_p_values({"PC1": t, "PC3": None, "PC4": t, "PC5": t}, 0.2)
    assert vals == {"PC1": 0.0455, "PC2": 0.2, "PC3": 1.0, "PC4": 0.0455, "PC5": 0.0455}
    assert len(pc.holm(vals)) == 5  # the family stays five


def test_pc3_comparator_never_promotes_an_unqualified_model() -> None:
    def rep(status: str, selected: str | None) -> ev.SelectionReport:
        return ev.SelectionReport("contractual", "A", status, selected, (), (), {}, None, None,
                                  "")
    got = pc.pc3_comparator({36: rep("SELECTED", "E1"), 42: rep(pp.NO_QUALIFIED_MODEL, None)})
    assert got == {36: "E1", 42: None}
    assert pc.pc3_comparator({36: rep(pp.NO_QUALIFIED_MODEL, None),
                              42: rep(pp.NO_QUALIFIED_MODEL, None)}) == {36: None, 42: None}


# --- PC2 conventions -----------------------------------------------------------------------------
def test_pc2_values_follow_the_pinned_p3_and_m10_conventions() -> None:
    sc = pc.m10_scenario(6159, 0.5, frailty_variance=0.25)
    rows = pc.channel_lifetime_values(sc, 24)
    obs = observe(sc, 24)
    st = obs.status()
    cm = ct.fit_contribution(sc, 24, pc.POLICY)
    fits = ct.fit_cells("E2", st)
    recent = st.loc[st["acquisition_month"] > 24 - CHURN_WINDOW_MONTHS]
    for row in rows:
        ch = row["channel"]
        assert row["truth"] == sc.truth.population_value(horizon=None,
                                                         monthly_rate=pp.MONTHLY_DISCOUNT_RATE,
                                                         channel=ch)
        num = den = 0.0
        for seg, n in recent.loc[recent["channel"] == ch, "segment"].value_counts().items():
            fit = fits.used[(str(seg), ch)][1]

            def contribution(ages: np.ndarray, s: str = str(seg), c: str = ch) -> np.ndarray:
                return np.asarray(cm.expected(np.array([s]), np.array([c]), ages[None, :])[0],
                                  dtype=np.float64)
            v = ct.cohort_value("E2", fit.params, contribution, horizon=None,
                                monthly_rate=pp.MONTHLY_DISCOUNT_RATE).value
            num, den = num + n * v, den + n
        assert row["E2"] == pytest.approx(num / den, rel=1e-12)
        assert row["H4"] == pytest.approx(he.h4(obs, pp.MONTHLY_DISCOUNT_RATE, None,
                                                channel=ch).value)
    s = pc.pc2_statistic(rows)
    f = pd.DataFrame(rows)
    assert s == pytest.approx(float(np.mean(np.abs(f["H4"] / f["truth"] - 1)
                                            - np.abs(f["E2"] / f["truth"] - 1))))
    rows[0] = {**rows[0], "H4": math.nan}
    assert math.isnan(pc.pc2_statistic(rows))  # undefined -> the draw is excluded


# --- aggregators ---------------------------------------------------------------------------------
def test_aggregators_match_brute_force() -> None:
    rng = np.random.default_rng(6160)
    draws: list[dict[str, Any]] = []
    for i in range(40):
        g = rng.uniform(1.0, 2.0)
        draws.append({"seed": i, "S": float(rng.normal(0.05, 0.02)),
                      "oracle_gate": [{"cutoff": 36, "D": g + rng.normal(0, 0.3), "G": g}],
                      "gate4": [{"model": "E2", "cutoff": 24, "scale": s,
                                 "bias": float(rng.normal(0, 0.01)),
                                 "excess_rmse": float(3.0 * s ** -0.5 * rng.uniform(0.9, 1.1))}
                                for s in (0.5, 1.0, 2.0)]})
    og = pc.aggregate_oracle_gate(draws).iloc[0]
    diff = np.array([d["oracle_gate"][0]["D"] - d["oracle_gate"][0]["G"] for d in draws])
    assert og["z"] == pytest.approx(diff.mean() / (diff.std(ddof=1) / math.sqrt(40)))
    assert bool(og["pass"]) == (abs(og["z"]) < 3 and og["D lower"] > 0)
    bias, slope = pc.aggregate_gate4(draws)
    slopes = [np.polyfit(np.log([0.5, 1, 2]), np.log([r["excess_rmse"] for r in d["gate4"]]),
                         1)[0] for d in draws]
    assert slope.iloc[0]["mean slope"] == pytest.approx(np.mean(slopes))
    assert bool(slope.iloc[0]["pass"]) and slope.iloc[0]["upper"] < 0
    assert len(bias) == 3
    q = pc.aggregate_pc2(draws)
    s = np.array([d["S"] for d in draws])
    assert q["p_value"] == pytest.approx(stats.ttest_1samp(s, 0.0).pvalue)


# --- the script: seeds, refusal, determinism, failure --------------------------------------------
def test_dry_run_uses_development_seeds_only_and_final_is_refused(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pp, "PENDING_BEFORE_MAIN", ("test: final runs are never executed",))
    gen = _script()
    plan = gen.dry_run_plan()
    seeds = {plan.main, *[s for v in plan.draws.values() for s in v]}
    assert max(seeds) <= pp.DEVELOPMENT_SEED_MAX and not seeds & gen.reserved_seeds()
    assert {k: len(v) for k, v in plan.draws.items()} == {"R1": 10, "R2": 10, "R3": 10,
                                                         "REFIT": 10}
    assert plan.refits == 20 and not plan.fresh_step and plan.m10_comparability is None
    with pytest.raises(ValueError, match="not development seeds"):
        gen.check_development_seeds([pp.COVERAGE_SEED_START + 5])
    with pytest.raises(ValueError, match="not development seeds"):
        gen.check_development_seeds([pp.REFIT_STUDY_SEED_START])
    run_dir = tmp_path / "final"
    assert gen.main(["--mode", "final", "--run-dir", str(run_dir),
                     "--registry", str(tmp_path / "registry")]) == 2  # pending item
    assert not run_dir.exists() and pp.PENDING_BEFORE_MAIN


def test_results_do_not_depend_on_the_number_of_workers() -> None:
    gen = _script()
    tasks = [pc.Task("R3", 6161, 0.5), pc.Task("R3", 6162, 0.5)]
    serial = gen.run_tasks(tasks, 1)
    parallel = gen.run_tasks(tasks, 2)
    for a, b in zip(serial, parallel, strict=True):
        assert a["gate4"] == b["gate4"] and a["task"] == b["task"]


def test_a_failing_draw_aborts_without_a_record(tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    gen = _script()

    def boom(tasks: object, workers: int, fn: object = None) -> list[dict[str, Any]]:
        raise RuntimeError("draw failed")

    monkeypatch.setattr(gen, "run_tasks", boom)
    run_dir = tmp_path / "dry"
    assert gen.main(["--mode", "dry-run", "--run-dir", str(run_dir),
                     "--registry", str(tmp_path / "registry")]) == 1
    assert not (run_dir / "record.md").exists()
    assert gen.verify_run(run_dir) != []
    with pytest.raises(RuntimeError,
                       match=r"^task R9 seed 6163 scale 1 refits 0: ValueError$") as exc:
        pc.run_task(pc.Task("R9", 6163))
    assert isinstance(exc.value.__cause__, ValueError)


def test_scale_changes_only_acquisitions() -> None:
    base = m10g.EconomicsScenarioConfig(seed=6164)
    sc = pc.m10_scenario(6164, 2.0)
    assert sc.config.acquisitions_per_month == tuple(2 * a for a in base.acquisitions_per_month)
    assert sc.config.frailty_variance == base.frailty_variance and sc.config.seed == 6164
