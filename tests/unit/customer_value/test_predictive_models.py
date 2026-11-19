"""M11 P6: E5 (two-part penalised GLM) and S2 (split conformal) — exact identities, the design
rules, temporal tuning, eligibility, leakage and the S2 split (development seeds only)."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from decision_intelligence.customer_value import landmarks as lm
from decision_intelligence.customer_value import predictive as pr
from decision_intelligence.customer_value import predictive_protocol as pp
from decision_intelligence.customer_value.features import PromotionPolicy
from decision_intelligence.customer_value.panel import month_end
from decision_intelligence.data import customer_economics_scenarios as m10g
from decision_intelligence.data import predictive_ltv_scenarios as g
from decision_intelligence.data.point_in_time import as_of

POL = PromotionPolicy(2, ("paid_search", "social"))
SMALL_ACQ = (20.0, 25.0, 30.0, 12.0)


@pytest.fixture(scope="module")
def sig() -> g.SignalsScenario:
    return g.generate_signals_scenario(g.SignalsScenarioConfig(
        seed=4981, n_months=50, acquisitions_per_month=SMALL_ACQ))


@pytest.fixture(scope="module")
def pur() -> g.PurchasesScenario:
    return g.generate_purchases_scenario(g.PurchasesScenarioConfig(
        seed=4982, acquisitions_per_month=SMALL_ACQ))


def _constant_frame(r: np.ndarray) -> pd.DataFrame:
    n = r.size
    return pd.DataFrame({"segment": "standard", "channel": "organic", "tenure_months": 5,
                         "acquisition_cohort": 2, "promo_months_remaining": 0,
                         "paid_months": 6, "mean_contribution": 7.0,
                         "last_contribution": 7.0, "realised": r},
                        index=np.arange(n))


def test_two_part_identity_with_no_informative_feature() -> None:
    """Every feature constant: both parts are intercept-only and the prediction is exactly
    P(R > 0) x mean(R | R > 0) (penalty irrelevant: the intercept is unpenalised)."""
    rng = np.random.default_rng(4983)
    r = np.where(rng.random(500) < 0.3, 0.0, rng.gamma(2.0, 25.0, 500))
    r[:3] = -2.0  # negative windows count as "not positive"
    frame = _constant_frame(r)
    p1, p2 = pr.fit_parts(frame, "contractual", "A", penalty=10.0)
    assert p1.design.columns == () and p2.design.columns == ()
    p, m = pr._predict_parts(p1, p2, frame.iloc[:1])
    assert p[0] == pytest.approx(float((r > 0).mean()), rel=1e-9)
    assert m[0] == pytest.approx(float(r[r > 0].mean()), rel=1e-9)


def test_both_parts_at_penalty_zero_equal_statsmodels(sig: g.SignalsScenario) -> None:
    stack = lm.training_stack(sig, 40, 12, "contractual", "A", POL)
    p1, p2 = pr.fit_parts(stack, "contractual", "A", penalty=0.0)
    r = stack["realised"].to_numpy()
    X1 = p1.design.matrix(stack)
    glm1 = sm.GLM((r > 0).astype(float), X1, family=sm.families.Binomial()).fit(tol=1e-14)
    np.testing.assert_allclose(p1.result.coefficients, glm1.params, rtol=1e-6, atol=1e-8)
    pos = stack.loc[r > 0]
    X2 = p2.design.matrix(pos)
    glm2 = sm.GLM(r[r > 0], X2, family=sm.families.Gamma(sm.families.links.Log())).fit(
        tol=1e-14)
    np.testing.assert_allclose(p2.result.coefficients, glm2.params, rtol=1e-6, atol=1e-8)


def test_design_follows_the_convention_and_e5_keeps_the_full_registry(
        sig: g.SignalsScenario) -> None:
    m = pr.fit_e5(sig, 40, 12, "contractual", "B", POL, penalty=1.0)
    used = set(m.positive.design.raw_columns)
    assert {"acquisition_cohort", "tenure_months", "usage_mean_3m"} <= used  # A8 is E3-only
    assert m.positive.design.duplicates == ("paid_months",)
    assert "promo_months_remaining" in m.positive.design.standardiser.dropped  # SD 0
    assert m.status == "ok" and m.negative_mass <= 0.0


def test_fixed_penalty_and_not_selection_eligible_without_tuning_landmarks() -> None:
    sc = m10g.generate_economics_scenario(m10g.EconomicsScenarioConfig(
        seed=4984, acquisitions_per_month=SMALL_ACQ))
    m = pr.fit_e5(sc, 24, 12, "contractual", "A", POL)
    assert m.penalty == pp.DEFAULT_PENALTY and m.penalty_source == pr.FIXED_PENALTY_LABEL
    val = pr.validate_e5(sc, 24, 12, "contractual", "A", POL)
    assert val.candidate is None and not val.selection_eligible
    assert val.eligibility == pr.NOT_ELIGIBLE


def test_tuning_uses_only_information_recorded_by_the_validation_landmark(
        sig: g.SignalsScenario) -> None:
    cutoff, w = 40, 12
    lv = pp.validation_landmark(cutoff, w)
    assert lv is not None and pp.temporal_tuning_feasible(cutoff, w)
    m = pr.fit_e5(sig, cutoff, w, "contractual", "A", POL)
    assert m.penalty_source == pr.TUNED_LABEL
    assert [t.penalty for t in m.tuning] == list(pp.PENALTY_GRID)
    val = pr.validate_e5(sig, cutoff, w, "contractual", "A", POL, penalty=m.penalty)
    assert val.selection_eligible and val.candidate is not None
    end = month_end(sig.start, lv)
    tx = sig.transactions.copy()
    tx.loc[tx["recorded_at"] > end, "revenue"] *= 3.0
    val2 = pr.validate_e5(replace(sig, transactions=tx), cutoff, w, "contractual", "A", POL,
                          penalty=m.penalty)
    assert val.frame is not None and val2.frame is not None
    pd.testing.assert_series_equal(val.frame["prediction"], val2.frame["prediction"])
    assert not val.frame["realised"].equals(val2.frame["realised"])


@pytest.mark.parametrize("kind", ["contractual", "non_contractual"])
def test_e5_and_s2_ignore_everything_recorded_after_the_cutoff(
        sig: g.SignalsScenario, pur: g.PurchasesScenario, kind: str) -> None:
    if kind == "contractual":
        sc: g.SignalsScenario | g.PurchasesScenario = sig
        cutoff, tier, pol = 40, "B", POL
        end = month_end(sig.start, cutoff)
        tx, s = sig.transactions.copy(), sig.signals.copy()
        tx.loc[tx["recorded_at"] > end, "revenue"] = 1e4
        s.loc[s["recorded_at"] > end, "usage"] = 999
        other: g.SignalsScenario | g.PurchasesScenario = replace(
            sig, transactions=tx, signals=s,
            cancellations=as_of(sig.cancellations, "recorded_at", end))
    else:
        sc, cutoff, tier, pol = pur, 30, "A", None
        end = month_end(pur.start, cutoff)
        tx = pur.purchases.copy()
        tx.loc[tx["recorded_at"] > end, "revenue"] = 1e4
        other = replace(pur, purchases=tx)
    a = pr.predict_e5(pr.fit_e5(sc, cutoff, 12, kind, tier, pol, penalty=1.0), sc).frame
    b = pr.predict_e5(pr.fit_e5(other, cutoff, 12, kind, tier, pol, penalty=1.0),
                      other).frame
    pd.testing.assert_frame_equal(a, b)
    sa = pr.fit_s2(sc, cutoff, 12, kind, tier, pol, penalty=1.0)
    sb = pr.fit_s2(other, cutoff, 12, kind, tier, pol, penalty=1.0)
    pd.testing.assert_frame_equal(sa.frame, sb.frame)


def test_s2_split_is_deterministic_by_customer_with_one_latest_landmark(
        sig: g.SignalsScenario) -> None:
    ids = np.arange(1, 20001)
    cal = pr.calibration_customers(ids)
    assert abs(float(cal.mean()) - pp.CLUSTERED_INFERENCE.conformal_calibration_fraction) < 0.02
    np.testing.assert_array_equal(cal, pr.calibration_customers(ids[::-1])[::-1])
    s2 = pr.fit_s2(sig, 40, 12, "contractual", "A", POL, penalty=1.0)
    stack = lm.training_stack(sig, 40, 12, "contractual", "A", POL)
    is_cal = pr.calibration_customers(stack["customer_id"].to_numpy())
    assert s2.n_calibration == stack.loc[is_cal, "customer_id"].nunique()  # one per customer
    assert s2.n_training_customers == stack.loc[~is_cal, "customer_id"].nunique()
    assert set(s2.half_model.landmarks) <= set(pp.training_landmarks(40, 12))
    lo80, hi80 = s2.frame["lower_0.8"], s2.frame["upper_0.8"]
    lo95, hi95 = s2.frame["lower_0.95"], s2.frame["upper_0.95"]
    assert (lo95 <= lo80).all() and (hi95 >= hi80).all()
    np.testing.assert_allclose(hi80 - s2.frame["centre"], s2.quantiles[0.8])


def test_predictions_are_reproducible_and_order_invariant(pur: g.PurchasesScenario) -> None:
    a = pr.predict_e5(pr.fit_e5(pur, 30, 12, "non_contractual", "A", penalty=10.0), pur).frame
    shuffled = replace(pur, customers=pur.customers.sample(frac=1.0, random_state=1),
                       purchases=pur.purchases.sample(frac=1.0, random_state=2))
    b = pr.predict_e5(pr.fit_e5(shuffled, 30, 12, "non_contractual", "A", penalty=10.0),
                      shuffled).frame
    pd.testing.assert_frame_equal(a.sort_index(), b.sort_index(), rtol=1e-6)
    assert ((a["p_positive"] > 0) & (a["p_positive"] < 1) & (a["mean_positive"] > 0)).all()


@pytest.mark.parametrize("module", ["customer_value/predictive.py",
                                    "statistics/gamma_regression.py", "statistics/conformal.py"])
def test_estimation_code_never_reads_truth_or_oracles(module: str) -> None:
    root = Path(pr.__file__).resolve().parents[1]
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


def test_e5_labels_follow_amendment_a9_verbatim() -> None:
    assert f"labelled '{pr.NO_OK_FIT_LABEL}'" in pp.E5_TUNING_LOSS
    assert "'not selection-eligible'" in pp.E5_SELECTION_ELIGIBILITY
    assert pr.NOT_ELIGIBLE.startswith("not selection-eligible")
    assert "ties to the larger penalty" in pp.E5_TUNING_LOSS
