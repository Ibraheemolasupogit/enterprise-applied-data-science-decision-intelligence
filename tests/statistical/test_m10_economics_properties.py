"""M10 statistical properties on non-reserved seeds."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from decision_intelligence.customer_value import cases as c
from decision_intelligence.customer_value import historical as hi
from decision_intelligence.customer_value import protocol as pr
from decision_intelligence.customer_value.retention import paid_month_km


def test_bootstrap_interval_for_24_month_ltv_covers_the_truth() -> None:
    cov = c.coverage_study(40, seed_start=7000, n_resamples=499)
    for _, row in cov.rows.iterrows():
        assert row["LTV24 coverage"] >= 0.85, row["channel"]
        assert abs(row["bias"]) < 4 * row["mc_se"] + 1e-9


def test_greenwood_interval_covers_the_population_retention() -> None:
    hits = []
    for seed in range(7100, 7160):
        sc = c.scenario(seed, acquisitions_per_month=(20.0, 25.0, 30.0, 10.0))
        st = c.final_view(sc).status()
        km = paid_month_km(st["paid_months"], st["churned"])
        lo, hi_ = km.interval()
        i = int(np.searchsorted(km.times, 12, side="right")) - 1
        truth = c._population_survival(sc, 12, None, None)
        hits.append(lo[i] <= truth <= hi_[i])
    assert stats.binomtest(int(np.sum(hits)), len(hits), 0.95).pvalue > 0.001


def test_cohort_matched_realised_value_is_unbiased() -> None:
    zs = []
    for seed in range(7200, 7210):
        sc = c.scenario(seed)
        cm = hi.cohort_matched(c.final_view(sc), 24, c.RATE)
        truth = sc.truth.population_value(horizon=24, monthly_rate=c.RATE)
        v = cm["value"]
        zs.append((v.mean() - truth) / (v.std(ddof=1) / math.sqrt(v.size)))
    assert abs(np.mean(zs)) < 4 / math.sqrt(len(zs))


def test_constant_churn_heuristics_underestimate_lifetime_value_under_heterogeneity() -> None:
    grid = c.heterogeneity_grid(7300, (0.0, 1.0))
    inf = grid[grid["horizon"] == "infinite"].set_index("frailty variance")
    for col in ("H1/H2 constant churn bias", "H3 by cell bias"):
        assert inf.loc[1.0, col] < -0.10
        assert inf.loc[1.0, col] < inf.loc[0.0, col] - 0.15


def test_realised_backtest_windows_agree_with_the_truth() -> None:
    sc = c.scenario(7400)
    table, excluded = c.customer_backtests(sc)
    assert any("cutoff 36" in e for e in excluded)
    for _, row in table.drop_duplicates("cutoff").iterrows():
        assert abs(row["realised_total"] / row["expected_total"] - 1) < 0.03
    assert set(table["cutoff"]) == {k for k in pr.BACKTEST_CUTOFFS if k + 12 <= 47}


def test_naive_historical_values_are_biased_against_the_truth() -> None:
    """Pooled over channels and seeds: truncation biases the all-customers mean down and
    survivorship biases the survivors-only mean up (empirical, on these data)."""
    down, up = [], []
    for seed in range(7500, 7505):
        table = c.historical_table(c.scenario(seed), ages=(24,))
        down.append((table["naive: all customers"] - table["truth"]).mean())
        up.append((table["naive: survivors only"] - table["truth"]).mean())
    assert max(down) < 0 < min(up)
