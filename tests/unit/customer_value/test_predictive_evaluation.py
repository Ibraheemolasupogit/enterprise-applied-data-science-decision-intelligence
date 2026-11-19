"""M11 P7: B1 and B2 (amendment A10), predictive distributions, parameter uncertainty, CR1 and
selection reporting (development seeds only)."""

from __future__ import annotations

import ast
import math
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from decision_intelligence.customer_value import btyd
from decision_intelligence.customer_value import contractual as ct
from decision_intelligence.customer_value import predictive as pr
from decision_intelligence.customer_value import predictive_baselines as pb
from decision_intelligence.customer_value import predictive_evaluation as ev
from decision_intelligence.customer_value import predictive_protocol as pp
from decision_intelligence.customer_value import survival as sv
from decision_intelligence.customer_value.backtest import customer_projections
from decision_intelligence.customer_value.features import PromotionPolicy
from decision_intelligence.customer_value.landmarks import training_stack
from decision_intelligence.customer_value.model_selection import CandidateResult, select_model
from decision_intelligence.customer_value.panel import month_end, observe
from decision_intelligence.customer_value.protocol import MIN_AT_RISK, MIN_CHURN_EVENTS
from decision_intelligence.customer_value.retention import at_risk_curve
from decision_intelligence.data import customer_economics_scenarios as m10g
from decision_intelligence.data import predictive_ltv_scenarios as g
from decision_intelligence.statistics import predictive_metrics as pm
from decision_intelligence.statistics.gamma_regression import GammaRegressionResult

POL = PromotionPolicy(2, ("paid_search", "social"))


@pytest.fixture(scope="module")
def m10() -> m10g.EconomicsScenario:
    return m10g.generate_economics_scenario(m10g.EconomicsScenarioConfig(seed=5821))


@pytest.fixture(scope="module")
def small_m10() -> m10g.EconomicsScenario:
    return m10g.generate_economics_scenario(m10g.EconomicsScenarioConfig(
        seed=5822, n_months=30, acquisitions_per_month=(6.0, 30.0, 8.0, 3.0)))


@pytest.fixture(scope="module")
def pur() -> g.PurchasesScenario:
    return g.generate_purchases_scenario(g.PurchasesScenarioConfig(seed=5823))


@pytest.fixture(scope="module")
def sig() -> g.SignalsScenario:
    return g.generate_signals_scenario(g.SignalsScenarioConfig(
        seed=5824, n_months=50, acquisitions_per_month=(20.0, 25.0, 30.0, 12.0)))


# --- B1 -------------------------------------------------------------------------------------------
def test_b1_is_m10_kaplan_meier_by_cell_on_the_m11_population(m10: m10g.EconomicsScenario
                                                                ) -> None:
    b1 = pb.predict_b1(m10, 30)
    ref = customer_projections(observe(m10, 30), pp.PRIMARY_HORIZON)["Kaplan–Meier by cell"]
    pd.testing.assert_series_equal(b1.frame["prediction"], ref.reindex(b1.frame.index),
                                   check_names=False)
    stack = training_stack(m10, 44, 12, "contractual", "A", POL, landmarks=[30])
    assert b1.frame.index.equals(stack.set_index("customer_id").index)
    assert b1.undefined_customers == 0 and set(b1.cell_status.values()) == {"ok"}
    assert pb.B1_METHOD in pp.B1_DEFINITION


def test_b1_status_rules_use_m10_thresholds_without_fallback(small_m10: m10g.EconomicsScenario
                                                             ) -> None:
    obs = observe(small_m10, 24)
    st = obs.status()
    b1 = pb.predict_b1(small_m10, 24)
    seen = set()
    for (seg, ch), status in b1.cell_status.items():
        cell = st.loc[(st["segment"] == seg) & (st["channel"] == ch)]
        at_risk = at_risk_curve(cell["paid_months"].to_numpy(), int(cell["paid_months"].max()))
        usable = int(np.sum(at_risk >= MIN_AT_RISK))
        events = int(cell["churned"].sum())
        expected = ("undefined" if usable < 2 else
                    "unstable" if events < MIN_CHURN_EVENTS else "ok")
        assert status == expected, (seg, ch)
        seen.add(status)
        rows = b1.frame.loc[(b1.frame["segment"] == seg) & (b1.frame["channel"] == ch)]
        if status == "undefined":  # no fallback (A10 Option A): no prediction at all
            assert rows["prediction"].isna().all() and (rows["status"] == "undefined").all()
    assert seen == {"undefined", "unstable", "ok"}
    assert b1.undefined_customers > 0
    assert b1.ok_cell_fraction == pytest.approx(
        sum(s == "ok" for s in b1.cell_status.values()) / len(b1.cell_status))


def test_b1_validation_uses_only_information_recorded_by_the_validation_landmark(
        m10: m10g.EconomicsScenario) -> None:
    v = pb.validate_baseline("B1", m10, 36, policy=POL)
    lv = pp.validation_landmark(36, 12)
    assert lv is not None
    assert v.landmark == lv and v.selection_eligible and v.candidate is not None
    tx = m10.transactions.copy()
    late = (tx["recorded_at"] > month_end(m10.start, lv)).to_numpy()
    tx.loc[late, "revenue"] *= 3.0
    v2 = pb.validate_baseline("B1", replace(m10, transactions=tx), 36, policy=POL)
    assert v.frame is not None and v2.frame is not None
    pd.testing.assert_series_equal(v.frame["prediction"], v2.frame["prediction"])
    assert not v.frame["realised"].equals(v2.frame["realised"])
    assert pp.validation_landmark(16, 12) is None
    none = pb.validate_baseline("B1", m10, 16, policy=POL)
    assert not none.selection_eligible and none.candidate is None
    assert none.eligibility == pb.NOT_ELIGIBLE


# --- B2 -------------------------------------------------------------------------------------------
def test_b2_matches_a_brute_force_count_of_the_definition(pur: g.PurchasesScenario) -> None:
    cutoff = 30
    b2 = pb.predict_b2(pur, cutoff)
    obs = observe(pur, cutoff)
    through = obs.complete_through()
    tx = obs.transactions.drop_duplicates("transaction_id")
    tx = tx.loc[tx["billing_month"] <= through]
    data = btyd.purchase_data(pur, cutoff, 12)
    assert b2.frame.index.equals(data.frame.index)  # E4's population
    acq = obs.customers.set_index("customer_id")["acquisition_month"]
    for cid in b2.frame.index[::97]:
        own = tx.loc[tx["customer_id"] == cid]
        lo = max(int(acq[cid]), through - 12 + 1)
        f = int(((own["billing_month"] >= lo) & (own["billing_month"] <= through)).sum())
        s = float(own["revenue"].mean())  # acquisition purchase included
        assert int((own["billing_month"] == acq[cid]).sum()) >= 1
        assert b2.frame.loc[cid, "prediction"] == pytest.approx(f * s * data.contribution_ratio)
    assert (b2.frame["status"] == "ok").all() and b2.undefined_customers == 0


def test_b2_ignores_everything_recorded_after_the_cutoff(pur: g.PurchasesScenario) -> None:
    cutoff = 30
    a = pb.predict_b2(pur, cutoff).frame
    p = pur.purchases.copy()
    later = p["recorded_at"] > month_end(pur.start, cutoff)
    p.loc[later, "revenue"] *= 5.0
    b = pb.predict_b2(replace(pur, purchases=p), cutoff).frame
    pd.testing.assert_frame_equal(a, b)
    v = pb.validate_baseline("B2", pur, 30)
    assert v.landmark == pp.validation_landmark(30, 12) and v.candidate is not None


# --- predictive distributions and simulation ---------------------------------------------------
def test_contractual_pmf_is_a_distribution_with_the_model_mean(m10: m10g.EconomicsScenario
                                                                ) -> None:
    pred = ct.predict("E2", m10, 30, 12, POL)
    pmf = ev.contractual_pmf(pred.frame)
    assert np.allclose(pmf.sum(axis=1), 1.0) and (pmf >= -1e-12).all()
    paid = pred.frame[[f"paid_{j}" for j in range(1, 13)]].to_numpy()
    assert np.allclose(pmf @ np.arange(13), paid.sum(axis=1))


def test_e4_count_pmf_is_exact(pur: g.PurchasesScenario) -> None:
    pred = btyd.predict(pur, 24, 12)
    pmf = ev.e4_count_pmf(pred, 300)
    assert np.allclose(pmf.sum(axis=1), 1.0, atol=1e-8)
    assert np.allclose(pmf @ np.arange(301), pred.frame["expected_purchases"], atol=1e-6)
    # an independent event-by-event simulation of the BG/NBD posterior for a few customers
    rng = np.random.default_rng(5825)
    data = pred.data
    e0, eh = float(data.edges[0]), float(data.edges[-1])
    for i in (0, 7, int(np.argmax(data.frame["x"].to_numpy()))):
        row = data.frame.iloc[i]
        _, bf, _, _ = pred.fits.used[(str(row["segment"]), str(row["channel"]))]
        r, alpha, a, b = bf.params
        pa = btyd.bgnbd_conditional(bf.params, np.array([row["x"]]), np.array([row["t_x"]]),
                                    np.array([row["T"]]), data.edges).p_alive_T[0]
        n_sim = 20000
        counts = np.zeros(n_sim, dtype=np.int64)
        for s in range(n_sim):
            if rng.uniform() >= pa:
                continue
            lam = rng.gamma(r + row["x"], 1.0 / (alpha + row["T"]))
            p = rng.beta(a, b + row["x"])
            clock = 0.0
            while True:
                clock += rng.exponential(1.0 / lam)
                if clock > eh:
                    break
                if clock > e0:
                    counts[s] += 1
                if rng.uniform() < p:
                    break
        emp = np.bincount(counts, minlength=301)[:301] / n_sim
        se = np.sqrt(pmf[i] * (1 - pmf[i]) / n_sim)
        assert np.all(np.abs(emp - pmf[i]) <= 4.5 * se + 1e-4), i


def test_simulated_outcomes_are_centred_on_the_drawn_expectations(
        m10: m10g.EconomicsScenario, pur: g.PurchasesScenario) -> None:
    rng = np.random.default_rng(5826)
    for sim in (ev.simulate_contractual(ct.predict("E2", m10, 30, 12, POL), 40, rng),
                ev.e4_simulate(btyd.predict(pur, 24, 12), 20, rng)):
        d = (sim.outcome - sim.expected).mean(axis=1)
        se = (sim.outcome - sim.expected).std() / math.sqrt(sim.outcome.size)
        assert abs(float(d.mean())) < 4.5 * se
        lo, hi = sim.interval(0.80, outcome=True)
        plo, phi = sim.interval(0.80, outcome=False)
        assert np.all(lo <= hi) and np.mean(hi - lo) > np.mean(phi - plo)


# --- A11: contractual outcome noise ---------------------------------------------------------------
def test_a11_residual_pools_match_the_contribution_model_grouping_and_are_centred(
        small_m10: m10g.EconomicsScenario) -> None:
    cutoff = 24
    cm = ct.fit_contribution(small_m10, cutoff, POL)
    pool = ev.contribution_residuals(small_m10, cutoff, cm)
    obs = observe(small_m10, cutoff)
    tx = obs.transactions.loc[obs.transactions["billing_month"] <= obs.complete_through()]
    cust = obs.customers.set_index("customer_id")
    seg = cust["segment"].reindex(tx["customer_id"]).to_numpy()
    ch = cust["channel"].reindex(tx["customer_id"]).to_numpy()
    promo = (tx["age"].to_numpy() < POL.months) & np.isin(ch, POL.channels)
    c = tx["contribution"].to_numpy(float)
    assert set(pool.sources.values()) >= {"cell", "segment"}  # fallbacks exercised
    for key, row in cm.table.iterrows():
        s_, c_, p_ = cast(tuple[str, str, bool], key)
        k = (s_, c_, bool(p_))
        assert pool.sources[k] == row["source"]
        sel = {"cell": (seg == s_) & (ch == c_) & (promo == p_),
               "segment": (seg == s_) & (promo == p_),
               "pooled": promo == p_}[str(row["source"])]
        np.testing.assert_array_equal(pool.pools[k], c[sel] - row["mean"])
        assert abs(float(np.mean(pool.pools[k]))) < 1e-9  # centred on the model's own mean


def test_a11_residual_pools_use_only_completely_recorded_pre_cutoff_months(
        m10: m10g.EconomicsScenario) -> None:
    cutoff = 30
    cm = ct.fit_contribution(m10, cutoff, POL)
    a = ev.contribution_residuals(m10, cutoff, cm)
    through = observe(m10, cutoff).complete_through()
    tx = m10.transactions.copy()
    late = ((tx["billing_month"] > through)
            | (tx["recorded_at"] > month_end(m10.start, cutoff))).to_numpy()
    tx.loc[late, "cost_to_serve"] += 50.0
    b = ev.contribution_residuals(replace(m10, transactions=tx), cutoff, cm)
    for k in a.pools:
        np.testing.assert_array_equal(a.pools[k], b.pools[k])


def test_a11_outcome_mean_and_variance_follow_the_paid_months_and_residuals() -> None:
    rng = np.random.default_rng(5833)
    paid = np.tile(np.array([0.95, 0.9, 0.8, 0.7, 0.6, 0.5]), (40000, 1))
    m = np.tile(np.array([5.0, 5.0, 6.0, 6.0, 7.0, 7.0]), (40000, 1))
    pool = rng.normal(0.0, 2.0, 5000)
    pool -= pool.mean()
    u = rng.uniform(size=40000)
    eps = pool[rng.integers(0, pool.size, size=m.shape)]
    v = ev._paid_outcome_noise(paid, m, u, eps)
    v0 = ev._paid_outcome(paid, m, u)
    pmf = pm.paid_months_pmf(paid[:1])[0]
    cum = np.r_[0.0, np.cumsum(m[0])]
    mean_n = float(pmf @ np.arange(7))
    var_theory = float(pmf @ cum ** 2 - (pmf @ cum) ** 2) + mean_n * float(np.var(pool))
    assert abs(float(v.mean()) - float(pmf @ cum)) < 4 * math.sqrt(var_theory / v.size)
    assert float(np.var(v)) == pytest.approx(var_theory, rel=0.03)
    assert float(np.var(v0)) == pytest.approx(float(pmf @ cum ** 2 - (pmf @ cum) ** 2),
                                              rel=0.03)
    unpaid = np.zeros_like(paid)
    assert np.all(ev._paid_outcome_noise(unpaid, m, u, eps) == 0.0)  # noise only on paid months


def test_a11_keeps_the_pre_a11_draws_bit_identical_and_is_reproducible(
        m10: m10g.EconomicsScenario, sig: g.SignalsScenario) -> None:
    pred = ct.predict("E2", m10, 30, 12, POL)
    pool = ev.contribution_residuals(m10, 30, pred.contribution)
    old = ev.simulate_contractual(pred, 15, np.random.default_rng(5834))
    new = ev.simulate_contractual(pred, 15, np.random.default_rng(5834), residuals=pool,
                                  noise_rng=np.random.default_rng(5835))
    again = ev.simulate_contractual(pred, 15, np.random.default_rng(5834), residuals=pool,
                                    noise_rng=np.random.default_rng(5835))
    other = ev.simulate_contractual(pred, 15, np.random.default_rng(5834), residuals=pool,
                                    noise_rng=np.random.default_rng(5836))
    assert old.outcome_noise is None and new.outcome_noise is not None
    np.testing.assert_array_equal(old.expected, new.expected)
    np.testing.assert_array_equal(old.outcome, new.outcome)
    np.testing.assert_array_equal(new.outcome_noise, again.outcome_noise)
    assert other.outcome_noise is not None and not np.array_equal(new.outcome_noise,
                                                                  other.outcome_noise)
    for lvl in (0.80, 0.95):
        np.testing.assert_array_equal(old.interval(lvl, outcome=True)[0],
                                      new.interval(lvl, outcome=True)[0])
    lo, hi = new.interval(0.95, outcome=True, noise=True)
    lo0, hi0 = new.interval(0.95, outcome=True)
    assert float(np.mean(hi - lo)) > float(np.mean(hi0 - lo0))
    with pytest.raises(ValueError, match="no A11"):
        old.interval(0.8, outcome=True, noise=True)
    with pytest.raises(ValueError, match="both residuals and noise_rng"):
        ev.simulate_contractual(pred, 2, np.random.default_rng(1), residuals=pool)
    model = sv.fit_e3(sig, 40, 12, "A", POL, penalty=1.0)
    cov = model.hazard.result.covariance
    pool3 = ev.contribution_residuals(sig, 40, model.contribution)
    e_old = ev.simulate_e3(model, sig, cov, 5, np.random.default_rng(5837))
    e_new = ev.simulate_e3(model, sig, cov, 5, np.random.default_rng(5837), residuals=pool3,
                           noise_rng=np.random.default_rng(5838))
    np.testing.assert_array_equal(e_old.expected, e_new.expected)
    np.testing.assert_array_equal(e_old.outcome, e_new.outcome)
    assert e_new.outcome_noise is not None and np.all(np.isfinite(e_new.outcome_noise))


# --- CR1 ------------------------------------------------------------------------------------------
def test_e3_cr1_rebuilds_the_final_fit_exactly(sig: g.SignalsScenario) -> None:
    model = sv.fit_e3(sig, 40, 12, "B", POL, penalty=1.0)
    X, y, cid = ev._e3_rows(model, sig)
    beta = model.hazard.result.coefficients
    nb = len(model.hazard.bands)
    mask = np.r_[np.zeros(nb), np.ones(X.shape[1] - nb)]
    p = 1 / (1 + np.exp(-(X @ beta)))
    grad = X.T @ (y - p) - 1.0 * mask * beta  # penalised score at the optimum
    assert np.max(np.abs(grad)) / X.shape[0] < 1e-6
    hess = (X * (p * (1 - p))[:, None]).T @ X + np.diag(mask)
    np.testing.assert_allclose(np.linalg.inv(hess), model.hazard.result.covariance, rtol=1e-6,
                               atol=1e-12)
    c = ev.e3_cr1(model, sig)
    assert c.n_rows == model.hazard.n_person_months and c.n_clusters == np.unique(cid).size
    assert np.all(np.isfinite(c.cr1_se)) and np.all(c.cr1_se > 0)


def test_e5_cr1_rebuilds_both_parts_exactly(sig: g.SignalsScenario) -> None:
    model = pr.fit_e5(sig, 40, 12, "contractual", "A", POL, penalty=1.0)
    stack = training_stack(sig, 40, 12, "contractual", "A", POL)
    r = stack["realised"].to_numpy(float)
    pos = r > 0
    X1 = model.positive.design.matrix(stack)
    b1 = model.positive.result.coefficients
    m1 = np.r_[0.0, np.ones(X1.shape[1] - 1)]
    g1 = X1.T @ (pos - 1 / (1 + np.exp(-(X1 @ b1)))) - m1 * b1
    assert np.max(np.abs(g1)) / X1.shape[0] < 1e-6
    X2 = model.amount.design.matrix(stack.loc[pos])
    b2 = model.amount.result.coefficients
    k = float(cast(GammaRegressionResult, model.amount.result).shape)
    m2 = np.r_[0.0, np.ones(X2.shape[1] - 1)]
    g2 = X2.T @ (k * (r[pos] * np.exp(-(X2 @ b2)) - 1)) - m2 * b2
    assert np.max(np.abs(g2)) / X2.shape[0] < 1e-6
    a, b = ev.e5_cr1(model, sig)
    assert a.n_rows == len(stack) and b.n_rows == int(pos.sum())
    assert np.all(np.isfinite(a.cr1_se)) and np.all(np.isfinite(b.cr1_se))


# --- selection reporting ------------------------------------------------------------------------
def _input(model: str, tier: str, pred: np.ndarray, r: np.ndarray, *, ok: float = 1.0,
           eligible: bool = True) -> ev.CandidateInput:
    idx = pd.Index(np.arange(r.size), name="customer_id")
    frame = pd.DataFrame({"prediction": pred, "realised": r}, index=idx)
    fin = np.isfinite(pred)
    slope = float(np.polyfit(pred[fin], r[fin], 1)[0])
    cand = CandidateResult(model, ok, float(np.sum(pred[fin])) / float(np.sum(r[fin])) - 1, slope,
                           float(np.mean((r - pred) ** 2)) if fin.all() else math.nan)
    return ev.CandidateInput(model, tier, eligible, "eligible" if eligible else
                             "not selection-eligible", cand if eligible else None,
                             frame if eligible else None)


@pytest.fixture(scope="module")
def outcomes() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(5827)
    mu = rng.gamma(2.0, 10.0, 3000)
    return mu, mu + rng.normal(0, 5, 3000)


def test_selection_report_applies_the_frozen_rule_and_reference_scope(
        outcomes: tuple[np.ndarray, np.ndarray]) -> None:
    mu, r = outcomes
    good = _input("E2", "A", mu, r)
    better = _input("E3", "B", mu + 0.01 * (r - mu), r)
    rep = ev.selection_report("contractual", "B", [_input("B1", "A", 0.5 * mu, r), good, better],
                              np.random.default_rng(5828), resamples=300)
    ref = select_model([c.candidate for c in (_input("B1", "A", 0.5 * mu, r), good, better)
                        if c.candidate is not None],
                       {k: (v[1], v[2]) for k, v in rep.intervals.items()})
    assert (rep.status, rep.selected) == (ref.status, ref.selected)
    assert set(rep.intervals) == {("E2", "E3"), ("E3", "E2")}
    # no qualified model: E2 shown only for contractual scenarios (REFERENCE_FALLBACK_SCOPE)
    bad_c = ev.selection_report("contractual", "A", [_input("E1", "A", 0.5 * mu, r)],
                                np.random.default_rng(5829))
    assert bad_c.status == pp.NO_QUALIFIED_MODEL and bad_c.selected is None
    assert (bad_c.reference_fallback, bad_c.reference_label) == (
        "E2", pp.REFERENCE_FALLBACK_LABEL)
    bad_n = ev.selection_report("non_contractual", "A", [_input("E4", "A", 0.5 * mu, r),
                                                         _input("B2", "A", 2.0 * mu, r)],
                                np.random.default_rng(5830))
    assert bad_n.status == pp.NO_QUALIFIED_MODEL
    assert bad_n.reference_fallback is None and bad_n.reference_label is None


def test_selection_report_eligibility_tiers_and_undefined_predictions(
        outcomes: tuple[np.ndarray, np.ndarray]) -> None:
    mu, r = outcomes
    undefined = mu.copy()
    undefined[3] = np.nan
    rep = ev.selection_report(
        "contractual", "A",
        [_input("B1", "A", undefined, r), _input("E3", "A", mu, r, eligible=False),
         _input("E2", "A", mu, r)], np.random.default_rng(5831))
    assert dict(rep.excluded) == {"B1": ev.UNDEFINED_REASON, "E3": "not selection-eligible"}
    assert rep.selected == "E2"
    with pytest.raises(ValueError, match="cannot enter a Tier A"):
        ev.selection_report("contractual", "A", [_input("E3", "B", mu, r)],
                            np.random.default_rng(1))
    with pytest.raises(ValueError, match="not a selection candidate"):
        ev.selection_report("contractual", "A", [_input("E4", "A", mu, r)],
                            np.random.default_rng(1))
    with pytest.raises(ValueError, match="not a selection candidate"):
        ev.selection_report("non_contractual", "A", [_input("E2", "A", mu, r)],
                            np.random.default_rng(1))


def test_coverage_summary_reports_and_never_asserts() -> None:
    rng = np.random.default_rng(5832)
    y = rng.normal(size=1000)
    out = ev.coverage_summary(np.full(1000, -1.2816), np.full(1000, 1.2816), y, 0.80,
                              groups=np.repeat(["a", "b"], 500))
    assert list(out["group"]) == ["a", "b"] and (out["n"] == 500).all()
    assert set(out["label"]) <= {"consistent with nominal", "below nominal", "above nominal"}
    assert ((out["cp_lower"] <= out["coverage"]) & (out["coverage"] <= out["cp_upper"])).all()


@pytest.mark.parametrize("module", ["customer_value/predictive_baselines.py",
                                    "customer_value/predictive_evaluation.py",
                                    "statistics/predictive_metrics.py"])
def test_p7_code_never_reads_truth_or_oracles(module: str) -> None:
    root = Path(pb.__file__).resolve().parents[1]
    tree = ast.parse((root / module).read_text())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    modules = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not names & {"truth", "SignalsTruth", "EconomicsTruth", "PurchasesTruth", "config",
                        "account_closure_summary", "oracle", "m10_oracle", "signals_oracle",
                        "purchases_oracle"}
    assert not any(m.endswith("oracle") or m.startswith("decision_intelligence.data")
                   for m in modules)
