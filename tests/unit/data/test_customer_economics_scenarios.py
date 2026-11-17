"""The M10 customer-economics generator: determinism, schema, truth (Milestone 10)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from decision_intelligence.data.customer_economics_scenarios import (
    CHANNELS,
    SEGMENTS,
    EconomicsScenarioConfig,
    generate_economics_scenario,
)

CFG = replace(EconomicsScenarioConfig(seed=515), acquisitions_per_month=(20.0, 25.0, 30.0, 10.0))


def test_same_seed_reproduces_and_tables_are_consistent() -> None:
    a, b = generate_economics_scenario(CFG), generate_economics_scenario(CFG)
    pd.testing.assert_frame_equal(a.transactions, b.transactions)
    pd.testing.assert_frame_equal(a.customers, b.customers)
    assert a.transactions["transaction_id"].is_unique
    assert (a.transactions["recorded_at"] >= a.transactions["billing_date"]).all()
    assert (a.transactions["billing_month"] <= CFG.n_months - 1).all()
    mk = a.marketing.groupby("channel")["new_customers"].sum()
    assert mk.sum() == len(a.customers)
    assert set(a.customers["segment"]) <= set(SEGMENTS)
    assert set(a.customers["channel"]) == set(CHANNELS)


def test_paid_months_are_contiguous_from_acquisition() -> None:
    sc = generate_economics_scenario(CFG)
    tx = sc.transactions.merge(sc.customers[["customer_id", "acquisition_month"]])
    g = tx.groupby("customer_id")
    assert (g["billing_month"].min() == g["acquisition_month"].first()).all()
    assert (g["billing_month"].max() - g["billing_month"].min() + 1 == g.size()).all()
    canc = sc.cancellations.set_index("customer_id")["last_paid_month"]
    assert (g["billing_month"].max().loc[canc.index] == canc).all()


def test_truth_matches_monte_carlo_of_the_generator() -> None:
    zs = []
    for seed in range(600, 606):
        sc = generate_economics_scenario(replace(CFG, seed=seed))
        t = sc.truth
        r = 0.008
        tx = sc.transactions.merge(sc.customers[["customer_id", "acquisition_month"]])
        tx = tx.assign(age=tx["billing_month"] - tx["acquisition_month"])
        keep = sc.customers["acquisition_month"] <= CFG.n_months - 12
        w = tx.loc[(tx["age"] < 12) & (tx["acquisition_month"] <= CFG.n_months - 12)]
        v = ((w["revenue"] - w["cogs"] - w["payment_fee"] - w["cost_to_serve"])
             * (1 + r) ** -w["age"]).groupby(w["customer_id"]).sum()
        v = v.reindex(sc.customers.loc[keep, "customer_id"], fill_value=0.0)
        truth = t.population_value(horizon=12, monthly_rate=r)
        zs.append((v.mean() - truth) / (v.std() / np.sqrt(v.size)))
    assert abs(np.mean(zs)) < 4 / np.sqrt(len(zs))


def test_customer_truth_is_consistent_with_population_truth() -> None:
    sc = generate_economics_scenario(replace(CFG, frailty_variance=0.0))
    t = sc.truth
    v = t.customer_value(horizon=24, monthly_rate=0.008)
    for seg in SEGMENTS:
        for ch in CHANNELS:
            sel = (t.segment == seg) & (t.channel == ch)
            if sel.any():  # no heterogeneity: every customer equals the population value
                ref = t.population_value(horizon=24, monthly_rate=0.008, segment=seg, channel=ch)
                assert np.allclose(v[sel], ref)


@pytest.mark.parametrize("changes", [{"n_months": 1}, {"frailty_variance": -1.0},
                                     {"segment_mix": ((1.0, 0.0, 0.0),) * 3},
                                     {"monthly_churn": (0.9, 0.9, 0.9)},
                                     {"late_payment_rate": 1.0}])
def test_invalid_configurations_are_refused(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        replace(CFG, **changes)
