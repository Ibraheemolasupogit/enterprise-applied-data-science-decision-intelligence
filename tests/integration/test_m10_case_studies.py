"""Every M10 case study end to end on a reduced scenario, plus reproducibility."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from decision_intelligence.customer_value import cases as c
from decision_intelligence.data.customer_economics_scenarios import (
    CHANNELS,
    SEGMENTS,
    EconomicsScenario,
)

SMALL = {"acquisitions_per_month": (25.0, 30.0, 35.0, 15.0)}


@pytest.fixture(scope="module")
def sc() -> EconomicsScenario:
    return c.scenario(8800, **SMALL)


def test_audit_retention_and_historical(sc: EconomicsScenario) -> None:
    audit = c.data_audit(sc)
    assert audit["customers"] == len(sc.customers)
    assert audit["late-recorded payments (> 15 days)"] > 0
    ret = c.retention_table(sc)
    assert set(ret["group"]) >= {"all", *(f"segment {s}" for s in SEGMENTS)}
    assert ((ret["lo"] <= ret["km"]) & (ret["km"] <= ret["hi"])).all()
    hist = c.historical_table(sc)
    assert {"naive: all customers", "naive: survivors only"} <= set(hist.columns)


def test_heuristics_unit_and_segment_economics(sc: EconomicsScenario) -> None:
    h = c.heuristic_table(sc)
    assert set(h["channel"]) == set(CHANNELS)
    assert {"projected", "expected lifetime"} <= set(h["quantity"])
    ue = c.unit_economics(sc)
    assert set(ue.channels["channel"]) == set(CHANNELS)
    assert ue.channels["precision target met"].dtype == bool
    assert len(c.segment_economics(sc)) == len(SEGMENTS)
    rc = c.revenue_vs_contribution(sc)
    assert set(rc.columns) >= {"rank by revenue", "rank by contribution"}
    assert set(c.stability_flags(sc)["status"]) <= {"ok", "unstable", "undefined"}


def test_convergence_uncertainty_backtests_sensitivity_m2(sc: EconomicsScenario) -> None:
    conv = c.convergence_table(sc)
    assert abs(conv["relative gap (H1)"].iloc[-1]) < 1e-6
    unc = c.h1_uncertainty(sc, n_boot=100)
    assert (unc["delta-method SE"] > 0).all()
    cust, excluded = c.customer_backtests(sc)
    assert excluded and not cust.empty
    coh, _ = c.cohort_backtests(sc)
    assert {"realised", "truth", "H4 Kaplan–Meier"} <= set(coh.columns)
    sens = c.sensitivity(sc)
    assert len(sens) == 1 + 3 + 2 + 2  # base + one-factor-at-a-time variations
    assert len(c.m2_compatibility()) == 8


def test_case_studies_are_reproducible(sc: EconomicsScenario) -> None:
    again = c.scenario(8800, **SMALL)
    pd.testing.assert_frame_equal(c.heuristic_table(sc), c.heuristic_table(again))
    pd.testing.assert_frame_equal(c.unit_economics(sc).channels, c.unit_economics(again).channels)


def test_invalid_cutoff_is_excluded_consistently_from_every_backtest(sc: EconomicsScenario
                                                                      ) -> None:
    """Protocol cutoff 36 cannot have a complete 12-month window (data end at month 47): both
    backtests refuse it, report it, and contain only the valid cutoffs (M10 audit)."""
    from decision_intelligence.customer_value import protocol as pr

    assert 36 in pr.BACKTEST_CUTOFFS  # the frozen protocol is not edited
    cust, cust_excl = c.customer_backtests(sc)
    coh, coh_excl = c.cohort_backtests(sc)
    for frame, excluded in ((cust, cust_excl), (coh, coh_excl)):
        assert sorted(frame["cutoff"].unique()) == [24, 30]
        assert len(excluded) == 1 and excluded[0].startswith("cutoff 36:")


def test_precision_flags_and_ltv_cac_interval_construction(sc: EconomicsScenario) -> None:
    """The precision flag is the pre-set rule on the reported half-width, for channels and
    segment x channel cells; the LTV/CAC interval is the LTV interval divided by the realised
    media CAC, i.e. spend is treated as known (the documented cause of under-coverage)."""
    from decision_intelligence.customer_value import protocol as pr

    ue = c.unit_economics(sc)
    for frame in (ue.channels, ue.cells):
        assert (frame["precision target met"]
                == (frame["relative half-width"] <= pr.PRECISION_TARGET)).all()
    ch = ue.channels
    lo_hi = ch["LTV/CAC 95% CI"].str.strip("[]").str.split(", ", expand=True).astype(float)
    rel = (lo_hi[1] - lo_hi[0]) / 2 / ch["LTV/CAC (media)"]
    assert np.allclose(rel, ch["relative half-width"], atol=2e-3)  # CI printed to 3 dp
    assert np.allclose(ch["LTV/CAC (media)"] * ch["CAC media"], ch["LTV24 realised"])
