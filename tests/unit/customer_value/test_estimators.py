"""Point-in-time view, historical and heuristic estimators, backtests (Milestone 10)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from decision_intelligence.customer_value import backtest as bt
from decision_intelligence.customer_value import heuristic as he
from decision_intelligence.customer_value import historical as hi
from decision_intelligence.customer_value import protocol as pr
from decision_intelligence.customer_value.economics import discount_factors, geometric_value
from decision_intelligence.customer_value.panel import month_end, observe
from decision_intelligence.customer_value.retention import (
    cohort_retention_table,
    constant_churn,
)
from decision_intelligence.data.customer_economics_scenarios import (
    EconomicsScenario,
    generate_economics_scenario,
)

from .conftest import SMALL

RATE = pr.MONTHLY_DISCOUNT_RATE


def _perturb_after(sc: EconomicsScenario, cutoff: pd.Timestamp) -> EconomicsScenario:
    """Rewrite every record recorded after the cutoff."""
    tx = sc.transactions.copy()
    late = tx["recorded_at"] > cutoff
    tx.loc[late, ["revenue", "cost_to_serve"]] = 999.0
    canc = sc.cancellations.copy()
    canc.loc[canc["recorded_at"] > cutoff, "last_paid_month"] = 0
    return replace(sc, transactions=tx, cancellations=canc)


def test_observable_view_ignores_everything_recorded_after_the_cutoff(
    small: EconomicsScenario,
) -> None:
    a = observe(small, 30)
    b = observe(_perturb_after(small, month_end(small.start, 30)), 30)
    pd.testing.assert_frame_equal(a.transactions, b.transactions)
    pd.testing.assert_frame_equal(a.status(), b.status())
    assert (a.transactions["recorded_at"] <= a.cutoff).all()
    assert a.complete_through() == 28  # 45-day recording lag


def test_late_payments_are_missing_until_recorded_and_duplicates_dropped() -> None:
    sc = generate_economics_scenario(replace(SMALL, duplicate_rate=0.05))
    assert not sc.transactions["transaction_id"].is_unique
    obs = observe(sc, 30)
    assert obs.transactions["transaction_id"].is_unique
    billed_30 = sc.transactions.drop_duplicates("transaction_id")
    billed_30 = billed_30.loc[billed_30["billing_month"] == 30]
    recorded = obs.transactions.loc[obs.transactions["billing_month"] == 30]
    assert len(recorded) < len(billed_30)  # some month-30 payments are recorded later
    later = observe(sc, 32)
    assert (later.transactions["billing_month"] == 30).sum() == len(billed_30)


def test_status_matches_the_generator_for_observed_churn(small: EconomicsScenario) -> None:
    obs = observe(small, 40)
    st = obs.status()
    tx = small.transactions.drop_duplicates("transaction_id")
    real_last = tx.groupby("customer_id")["billing_month"].max()
    churned = st.index[st["churned"]]
    assert np.array_equal(st.loc[churned, "acquisition_month"] + st.loc[churned, "paid_months"]
                          - 1, real_last.loc[churned])
    active = st.index[~st["churned"]]
    assert (real_last.loc[active] >= 40).all()  # every active customer really paid month 40


def test_cohort_table_reports_only_completely_reached_ages(small: EconomicsScenario) -> None:
    obs = observe(small, 30)
    table = cohort_retention_table(obs, 12)
    assert np.isnan(table.loc[25, 6]) and not np.isnan(table.loc[25, 5])
    assert (table[0].dropna() == 1.0).all()


def test_constant_churn_counts_events_and_trials_by_hand(small: EconomicsScenario) -> None:
    obs = observe(small, 30)
    st = obs.status()
    ev = tr = 0
    for _, row in st.iterrows():
        last = row["acquisition_month"] + row["paid_months"] - 1
        for m in range(max(row["acquisition_month"], 18), 30):
            if m > last:
                break
            tr += 1
            ev += int(row["churned"] and m == last)
    est = constant_churn(obs, 12)
    assert (est.events, est.trials) == (ev, tr)


def test_cohort_matched_value_equals_a_per_customer_sum(small: EconomicsScenario) -> None:
    obs = observe(small, 47, lag_days=pr.MAX_RECORDING_LAG_DAYS)
    cm = hi.cohort_matched(obs, 12, RATE)
    assert cm["acquisition_month"].max() + 11 <= obs.complete_through()
    tx = obs.transactions
    d = discount_factors(12, RATE)
    for cid in cm["customer_id"].iloc[::97]:
        t = tx.loc[(tx["customer_id"] == cid) & (tx["age"] < 12)]
        ref = float(np.sum(t["contribution"].to_numpy() * d[t["age"].to_numpy()]))
        assert cm.loc[cm["customer_id"] == cid, "value"].item() == pytest.approx(ref)
    with pytest.raises(ValueError, match="complete"):
        hi.cohort_matched(observe(small, 10), 24, RATE)


def test_naive_versions_select_the_documented_customers(small: EconomicsScenario) -> None:
    obs = observe(small, 47, lag_days=pr.MAX_RECORDING_LAG_DAYS)
    matched = hi.cohort_matched(obs, 24, RATE)
    survivors = hi.naive_survivors(obs, 24, RATE)
    assert set(survivors["customer_id"]) <= set(matched["customer_id"])
    paid = obs.status()["paid_months"]
    assert (paid.loc[survivors["customer_id"]] >= 24).all()
    assert len(hi.naive_all_customers(obs, 24, RATE)) == len(obs.customers)


def test_h1_is_the_geometric_closed_form_of_its_estimated_inputs(small: EconomicsScenario
                                                                 ) -> None:
    obs = observe(small, 47, lag_days=pr.MAX_RECORDING_LAG_DAYS)
    mask = np.ones(len(obs.status()), dtype=bool)
    c = constant_churn(obs, pr.CHURN_WINDOW_MONTHS).churn
    m = he.run_rate(obs, mask)
    for horizon in (24, 36, None):
        assert he.h1_h2(obs, RATE, horizon).value == pytest.approx(
            geometric_value(m, 1 - c, RATE, horizon).value, rel=1e-12)
    assert he.h1_h2(obs, RATE, 2000).value == pytest.approx(he.h1_h2(obs, RATE, None).value,
                                                           rel=1e-9)


def test_h4_finite_horizon_converges_to_its_infinite_value(small: EconomicsScenario) -> None:
    obs = observe(small, 47, lag_days=pr.MAX_RECORDING_LAG_DAYS)
    inf = he.h4(obs, RATE, None).value
    vals = [he.h4(obs, RATE, t).value for t in (12, 36, 120, 600, 3000)]
    assert all(b >= a for a, b in zip(vals, vals[1:], strict=False))
    assert vals[-1] == pytest.approx(inf, rel=1e-8)


def test_heuristics_flag_unstable_and_undefined_estimates(small: EconomicsScenario) -> None:
    obs = observe(small, 47, lag_days=pr.MAX_RECORDING_LAG_DAYS)
    e = he.h1_h2(obs, RATE, None, segment="premium", channel="social")
    st = obs.status()
    mask = ((st["segment"] == "premium") & (st["channel"] == "social")).to_numpy()
    if constant_churn(obs, pr.CHURN_WINDOW_MONTHS, mask).events < pr.MIN_CHURN_EVENTS:
        assert e.status == "unstable" and "churn events" in e.reason
    assert he.h1_h2(obs, RATE, None, segment="nonexistent").status == "undefined"
    assert he.h4(obs, RATE, None, segment="nonexistent").status == "undefined"


def test_h0_legacy_matches_a_hand_computation(small: EconomicsScenario) -> None:
    obs = observe(small, 47, lag_days=pr.MAX_RECORDING_LAG_DAYS)
    st = obs.status().loc[obs.status()["channel"] == "referral"]
    c = constant_churn(obs, pr.CHURN_WINDOW_MONTHS,
                       (obs.status()["channel"] == "referral").to_numpy()).churn
    tx = obs.transactions
    vals = []
    for cid, row in st.iterrows():
        t = tx.loc[tx["customer_id"] == cid].sort_values("billing_month")
        proj = 0.0 if row["churned"] or t.empty else t["revenue"].iloc[-1] * sum(
            (1 - c) ** k for k in range(1, 13))
        vals.append(t["revenue"].sum() + proj)
    assert he.h0_legacy(obs, channel="referral").value == pytest.approx(np.mean(vals))


def test_backtest_refuses_incomplete_windows_and_uses_one_comparison_set(
    small: EconomicsScenario,
) -> None:
    with pytest.raises(bt.IncompleteWindow):
        bt.check_window(48, 36, 12)
    bt.check_window(48, 35, 12)
    res = bt.customer_backtest(small, 48, 24, 12)
    s = res.summary
    assert s["customers"].nunique() == 1 and s["realised_total"].nunique() == 1
    sub = res.frame.dropna(subset=list(bt.CUSTOMER_METHODS))
    assert s["realised_total"].iloc[0] == pytest.approx(sub["realised"].sum())
    tx = small.transactions.drop_duplicates("transaction_id")
    win = tx.loc[(tx["billing_month"] > 24) & (tx["billing_month"] <= 36)]
    ref = (win["revenue"] - win["cogs"] - win["payment_fee"] - win["cost_to_serve"]).groupby(
        win["customer_id"]).sum().reindex(res.frame.index, fill_value=0.0)
    assert np.allclose(res.frame["realised"], ref)


def test_backtest_projection_ignores_post_cutoff_data(small: EconomicsScenario) -> None:
    a = bt.customer_projections(observe(small, 24), 12)
    b = bt.customer_projections(observe(_perturb_after(small, month_end(small.start, 24)), 24), 12)
    pd.testing.assert_frame_equal(a, b)
