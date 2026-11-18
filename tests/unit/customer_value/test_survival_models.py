"""M11 P4: E3 (discrete-time landmark hazard) and S1 (Cox, Efron ties) — the phase gates
(tenure-only E3 = M10's Kaplan-Meier hazards exactly; S1 = statsmodels PHReg Efron to 1e-6),
landmark outcomes, the penalty convention, temporal tuning, leakage and status rules
(development seeds only)."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from statsmodels.duration.hazard_regression import PHReg

from decision_intelligence.customer_value import predictive_protocol as pp
from decision_intelligence.customer_value import survival as sv
from decision_intelligence.customer_value.features import PromotionPolicy
from decision_intelligence.customer_value.panel import month_end, observe
from decision_intelligence.customer_value.retention import paid_month_km
from decision_intelligence.data import customer_economics_scenarios as m10g
from decision_intelligence.data import predictive_ltv_scenarios as g
from decision_intelligence.data.point_in_time import as_of

POL = PromotionPolicy(2, ("paid_search", "social"))
SMALL_ACQ = (20.0, 25.0, 30.0, 12.0)


@pytest.fixture(scope="module")
def m10() -> m10g.EconomicsScenario:
    return m10g.generate_economics_scenario(m10g.EconomicsScenarioConfig(seed=4931))


@pytest.fixture(scope="module")
def sig() -> g.SignalsScenario:
    return g.generate_signals_scenario(g.SignalsScenarioConfig(
        seed=4932, n_months=50, acquisitions_per_month=SMALL_ACQ))


def _km_person_months(m10: m10g.EconomicsScenario, cutoff: int
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    st = observe(m10, cutoff).status()
    t, e = st["paid_months"].to_numpy(), st["churned"].to_numpy()
    dur = np.where(e, t, t - 1)  # M10: events at t, active censored at t - 1
    rows = np.repeat(np.arange(dur.size), dur)
    j = np.arange(rows.size) - np.repeat(np.cumsum(dur) - dur, dur) + 1
    return t, e, j - 1, e[rows] & (j == dur[rows])


@pytest.mark.parametrize("cutoff", [24, 30])
def test_tenure_only_e3_reproduces_m10_kaplan_meier_hazards(m10: m10g.EconomicsScenario,
                                                            cutoff: int) -> None:
    """P4 gate: E3's tenure basis and solver, with one band per age and no covariates, give
    exactly the discrete hazards d_j / n_j of M10's paid_month_km."""
    t, e, age, event = _km_person_months(m10, cutoff)
    km = paid_month_km(t, e)
    jmax = int(np.max(km.times))
    np.testing.assert_array_equal(km.times, np.arange(1, jmax + 1))  # an event at every age
    keep = age < jmax
    h = sv.tenure_hazards(age[keep], event[keep], list(range(jmax)),
                          gradient_tolerance=1e-13)
    np.testing.assert_allclose(h, km.events / km.at_risk, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(np.cumprod(1 - h), km.survival, rtol=1e-10)
    # protocol bands: the band hazard is the pooled events / at-risk of its ages
    edges = pp.TENURE_BAND_EDGES
    hb = sv.tenure_hazards(age[keep], event[keep], edges, gradient_tolerance=1e-13)
    band = sv.tenure_band(np.arange(jmax), edges)
    for b in range(len(edges)):
        sel = band == b
        if km.at_risk[sel].sum() > 0:
            assert hb[b] == pytest.approx(km.events[sel].sum() / km.at_risk[sel].sum(),
                                          rel=1e-10)
        else:
            assert np.isnan(hb[b])  # a band no customer reaches has no estimate


def test_cox_matches_statsmodels_phreg_efron() -> None:
    """P4 gate: coefficients, log-likelihood and SEs to 1e-6; Breslow baseline equal
    (statsmodels reports H0 just before each time; ours includes the time)."""
    rng = np.random.default_rng(4933)
    n = 1500
    Z = rng.normal(size=(n, 4))
    t = np.minimum(rng.geometric(1 / (1 + np.exp(-(-2.0 + Z @ [0.4, -0.3, 0.0, 0.2])))), 12)
    e = (t < 12) | (rng.random(n) < 0.0)
    cox = sv.fit_cox_efron(t.astype(float), e, Z)
    ref = PHReg(t.astype(float), Z, status=e.astype(int), ties="efron").fit()
    np.testing.assert_allclose(cox.coefficients, ref.params, rtol=1e-6, atol=1e-8)
    assert cox.log_likelihood == pytest.approx(ref.llf, rel=1e-10)
    np.testing.assert_allclose(np.sqrt(np.diag(cox.covariance)), ref.bse, rtol=1e-6)
    times, h0 = ref.baseline_cumulative_hazard[0][:2]
    np.testing.assert_array_equal(cox.times, times)
    np.testing.assert_allclose(cox.baseline_cumulative_hazard[:-1], h0[1:], rtol=1e-6)
    assert cox.converged and cox.hessian_positive_definite
    assert len(np.unique(t[e])) <= 12  # heavily tied, as in monthly data


def test_window_outcomes_equal_the_p2_labels_and_person_months_are_correct(
        sig: g.SignalsScenario) -> None:
    cutoff, w = 40, 12
    stack, n = sv._landmark_data(sig, cutoff, w, "A", POL, None)
    np.testing.assert_array_equal(n, stack["paid_in_window"].to_numpy())
    pm = sv.person_months(np.array([0, 2, 3]), np.array([5, 9, 2]), 3)
    np.testing.assert_array_equal(pm.row, [0, 1, 1, 1, 2, 2, 2])
    np.testing.assert_array_equal(pm.k, [1, 1, 2, 3, 1, 2, 3])
    np.testing.assert_array_equal(pm.age, [5, 9, 10, 11, 2, 3, 4])
    np.testing.assert_array_equal(pm.event, [True, False, False, True, False, False, False])


def test_tenure_band_merging_rule() -> None:
    np.testing.assert_array_equal(sv.merge_bands(np.array([50, 50, 50])), [0, 1, 2])
    np.testing.assert_array_equal(sv.merge_bands(np.array([50, 3, 50, 2])), [0, 0, 1, 1])
    np.testing.assert_array_equal(sv.merge_bands(np.array([0, 0, 40, 40])), [0, 0, 0, 1])
    np.testing.assert_array_equal(sv.merge_bands(np.array([4, 4])), [0, 0])
    m = sv.merge_bands(np.array([0, 0, 40, 40, 40, 40, 40]))
    assert sv.band_labels(m) == ("[0, 3)", "[3, 6)", "[6, 12)", "[12, 24)", "[24, inf)")


def test_design_follows_the_penalty_convention(sig: g.SignalsScenario) -> None:
    stack, _ = sv._landmark_data(sig, 40, 12, "B", POL, None)
    d = sv.build_design(stack, "B")
    assert "paid_months" in d.duplicates  # = tenure_months + 1 for active customers
    for var in ("segment", "channel"):
        kept, ref = d.levels[var]
        assert ref == stack[var].value_counts().idxmax() and ref not in kept
    z = d.matrix(stack)
    np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(z.std(axis=0), 1.0, rtol=1e-10)
    assert set(pp.features("contractual", "B")) and len(d.columns) == z.shape[1]


def test_e3_uses_the_fixed_penalty_when_temporal_tuning_is_infeasible(
        m10: m10g.EconomicsScenario) -> None:
    assert not pp.temporal_tuning_feasible(24, 12)
    model = sv.fit_e3(m10, 24, 12, "A", POL)
    assert model.penalty == pp.DEFAULT_PENALTY
    assert model.penalty_source == sv.FIXED_PENALTY_LABEL and model.tuning == ()
    assert model.hazard.status == "ok"
    val = sv.validate_e3(m10, 24, 12, "A", POL)
    assert val.candidate is None and val.landmark == 10  # no landmark completes before L_v
    assert not val.selection_eligible and val.eligibility == sv.NOT_ELIGIBLE  # A8


def test_temporal_tuning_uses_only_information_recorded_by_the_validation_landmark(
        sig: g.SignalsScenario) -> None:
    cutoff, w = 40, 12
    lv = pp.validation_landmark(cutoff, w)
    assert lv is not None and pp.temporal_tuning_feasible(cutoff, w)
    model = sv.fit_e3(sig, cutoff, w, "A", POL)
    assert model.penalty_source == sv.TUNED_LABEL
    assert [r.penalty for r in model.tuning] == list(pp.PENALTY_GRID)
    assert model.penalty == sv.choose_penalty(model.tuning)
    val = sv.validate_e3(sig, cutoff, w, "A", POL, penalty=model.penalty)
    assert val.selection_eligible and val.candidate is not None
    frame, _, _ = sv._fit_at_validation(sig, cutoff, w, "A", POL, 1.0)
    # rows recorded after L_v (which carry the validation labels) never reach the tuning fit
    end = month_end(sig.start, lv)
    tx = sig.transactions.copy()
    tx.loc[tx["recorded_at"] > end, "revenue"] *= 3.0
    frame2, _, _ = sv._fit_at_validation(replace(sig, transactions=tx), cutoff, w, "A", POL,
                                         1.0)
    pd.testing.assert_series_equal(frame["prediction"], frame2["prediction"])
    assert not frame["realised"].equals(frame2["realised"])


@pytest.mark.parametrize("scramble", [False, True])
def test_e3_and_s1_ignore_everything_recorded_after_the_cutoff(sig: g.SignalsScenario,
                                                               scramble: bool) -> None:
    cutoff, w = 40, 12
    end = month_end(sig.start, cutoff)
    if scramble:
        rng = np.random.default_rng(5)
        tx, s, cn = sig.transactions.copy(), sig.signals.copy(), sig.cancellations.copy()
        tx.loc[tx["recorded_at"] > end, "revenue"] = rng.uniform(
            0, 500, int((tx["recorded_at"] > end).sum()))
        s.loc[s["recorded_at"] > end, "usage"] = rng.integers(
            0, 100, int((s["recorded_at"] > end).sum()))
        cn.loc[cn["recorded_at"] > end, "last_paid_month"] = 0
        other = replace(sig, transactions=tx, signals=s, cancellations=cn)
    else:
        other = replace(sig, transactions=as_of(sig.transactions, "recorded_at", end),
                        cancellations=as_of(sig.cancellations, "recorded_at", end),
                        signals=as_of(sig.signals, "recorded_at", end))
    a = sv.predict_e3(sv.fit_e3(sig, cutoff, w, "B", POL, penalty=1.0), sig).frame
    b = sv.predict_e3(sv.fit_e3(other, cutoff, w, "B", POL, penalty=1.0), other).frame
    pd.testing.assert_frame_equal(a, b)
    c = sv.predict_s1(sv.fit_s1(sig, cutoff, w, "A", POL), sig).frame
    d = sv.predict_s1(sv.fit_s1(other, cutoff, w, "A", POL), other).frame
    pd.testing.assert_frame_equal(c, d)


def test_predictions_are_probabilities_reproducible_and_order_invariant(
        sig: g.SignalsScenario) -> None:
    m = sv.fit_e3(sig, 40, 12, "B", POL, penalty=10.0)
    a = sv.predict_e3(m, sig).frame
    paid = a[[f"paid_{k}" for k in range(1, 13)]].to_numpy()
    assert np.all((paid > 0) & (paid < 1)) and np.all(np.diff(paid, axis=1) <= 0)
    shuffled = replace(sig, customers=sig.customers.sample(frac=1.0, random_state=3),
                       transactions=sig.transactions.sample(frac=1.0, random_state=4))
    b = sv.predict_e3(sv.fit_e3(shuffled, 40, 12, "B", POL, penalty=10.0), shuffled).frame
    pd.testing.assert_frame_equal(a.sort_index(), b.sort_index(), rtol=1e-8)


def test_status_rules_flag_too_few_events() -> None:
    feats = pd.DataFrame({"segment": ["standard"] * 40, "channel": ["organic"] * 40,
                          "tenure_months": np.arange(40) % 10 + 3,
                          "acquisition_cohort": np.arange(40) % 7,
                          "promo_months_remaining": 0, "paid_months": np.arange(40) % 10 + 4,
                          "mean_contribution": np.linspace(5, 9, 40),
                          "last_contribution": np.linspace(5, 9, 40)})
    n = np.full(40, 12)
    n[:6] = 3  # six churn events
    fit = sv.fit_hazard(feats, n, 12, "A", 1.0)
    assert fit.n_events == 6 and fit.status == "unstable"
    assert "too few events" in fit.reasons
    assert fit.bands == ("[0, inf)",)  # every band merged when events are this sparse


@pytest.mark.parametrize("module", ["customer_value/survival.py",
                                    "statistics/penalised_regression.py"])
def test_estimation_code_never_reads_truth_or_oracles(module: str) -> None:
    root = Path(sv.__file__).resolve().parents[1]
    tree = ast.parse((root / module).read_text())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    modules = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not names & {"truth", "SignalsTruth", "EconomicsTruth", "config",
                        "account_closure_summary", "oracle", "m10_oracle", "signals_oracle"}
    assert not any(m.endswith("oracle") or m.startswith("decision_intelligence.data")
                   for m in modules)


def test_a8_e3_excludes_three_covariates_and_s1_keeps_the_registry(sig: g.SignalsScenario
                                                                   ) -> None:
    e3 = sv.fit_e3(sig, 40, 12, "B", POL, penalty=1.0)
    used = set(e3.hazard.design.raw_columns)
    assert not used & set(pp.E3_EXCLUDED_COVARIATES)
    assert {"mean_contribution", "usage_mean_3m"} <= used
    assert e3.hazard.bands[0] == "[0, 3)"  # tenure enters through the band basis only
    pre = sv.fit_e3(sig, 40, 12, "B", POL, penalty=1.0, exclude_features=())
    assert {"acquisition_cohort", "tenure_months"} <= set(pre.hazard.design.raw_columns)
    assert "paid_months" in pre.hazard.design.duplicates
    s1 = sv.fit_s1(sig, 40, 12, "A", POL)
    assert {"acquisition_cohort", "tenure_months"} <= set(s1.design.columns)  # unchanged


def test_tuning_choice_is_minimum_mse_with_ties_to_the_larger_penalty() -> None:
    rec = sv.TuningRecord
    assert sv.choose_penalty([rec(0.01, 5.0, "ok"), rec(1.0, 4.0, "ok"),
                              rec(100.0, 4.5, "ok")]) == 1.0
    assert sv.choose_penalty([rec(0.1, 4.0, "ok"), rec(10.0, 4.0, "ok"),
                              rec(1.0, 4.0, "ok")]) == 10.0
    assert sv.choose_penalty([rec(0.1, 1.0, "unstable"), rec(10.0, 4.0, "ok")]) == 10.0
    assert sv.choose_penalty([rec(0.1, float("nan"), "ok")]) is None


def test_duplicate_removal_is_deterministic_and_registry_ordered(sig: g.SignalsScenario
                                                                 ) -> None:
    stack, _ = sv._landmark_data(sig, 40, 12, "A", POL, None)
    a = sv.build_design(stack, "A")
    b = sv.build_design(stack[stack.columns[::-1]].sample(frac=1.0, random_state=9), "A")
    assert a.columns == b.columns and a.duplicates == b.duplicates == ("paid_months",)
    assert "tenure_months" in a.columns  # the earlier column in registry order is kept
