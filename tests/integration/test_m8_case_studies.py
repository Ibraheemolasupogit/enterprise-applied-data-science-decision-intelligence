"""Each M8 case study end to end on a reduced scenario, plus reproducibility."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from decision_intelligence.recommendations import cases as c

SMALL = {"n_users": 500, "n_items": 120}


@pytest.fixture(scope="module")
def prepared() -> c.Prepared:
    return c.prepare(c.default_config(**SMALL))


@pytest.fixture(scope="module")
def bench(prepared: c.Prepared) -> c.Benchmark:
    return c.benchmark(prepared)


def test_data_audit(prepared: c.Prepared) -> None:
    audit = c.data_audit(prepared)
    assert audit.counts["users"] == 500 and audit.counts["items"] == 120
    assert 0 < audit.counts["click_rate"] < 1
    assert audit.counts["top10pct_items_impression_share"] > audit.counts[
        "top10pct_items_attractiveness_share"]  # popular items are over-exposed
    assert len(audit.position) == 10


def test_benchmark_tunes_every_family_on_validation_and_reports_three_views(
    bench: c.Benchmark,
) -> None:
    families = set(c.candidate_grid())
    assert set(bench.models) == families
    assert bench.tuning.groupby("family")["selected"].sum().eq(1).all()
    for col in ("ndcg", "ndcg_lo", "ndcg_hi"):
        assert bench.logged[col].notna().all()
    assert (bench.logged["ndcg_lo"] <= bench.logged["ndcg"]).all()
    assert set(bench.logged_comparisons["model"]) == families - {c.REFERENCE}
    assert (bench.logged_comparisons["adjusted_p_value"]
            >= bench.logged_comparisons["p_value"]).all()
    assert {"logged_ndcg", "oracle_ndcg", "online_clicks", "ips_clicks"} <= set(bench.views)
    assert len(bench.beyond) == len(families)


def test_contrast_reports_all_views(prepared: c.Prepared, bench: c.Benchmark) -> None:
    pair = c.contrast(prepared, bench, "item-kNN", "global popularity")
    assert pair.logged["ci_lower"] <= pair.logged["difference"] <= pair.logged["ci_upper"]
    assert pair.online[1] <= pair.online[0] <= pair.online[2]


def test_leakage_study(prepared: c.Prepared) -> None:
    study = c.leakage_study(prepared)
    assert len(study.random_vs_chronological) == 3
    assert (study.perturbation["max_abs_score_change"] == 0).all()


def test_fairness_study(prepared: c.Prepared, bench: c.Benchmark) -> None:
    study = c.fairness_study(prepared, bench, ("global popularity", "item-kNN"))
    shares = study.category_exposure.groupby("model")[["exposure_share", "merit_share"]].sum()
    assert np.allclose(shares.to_numpy(), 1.0)
    assert (study.heterogeneity["holm_adjusted_p"] >= study.heterogeneity["kruskal_wallis_p"]
            ).all()


def test_cold_start_study(prepared: c.Prepared) -> None:
    study = c.cold_start_study(prepared)
    users = study.users.set_index("model")
    assert study.n_cold_users > 0 and study.n_new_items > 0
    assert users.loc["item-kNN, no fallback", "cold_users_with_constant_scores"] == 1.0
    assert users.loc["item-kNN → segment popularity", "cold_users_with_constant_scores"] == 0.0
    items = study.items.set_index("model")
    assert items.loc["item-kNN", "scores_constant_on_new_items"]
    assert not items.loc["content-based", "scores_constant_on_new_items"]


def test_off_policy_diagnostics(prepared: c.Prepared, bench: c.Benchmark) -> None:
    sweep = c.clipping_sweep(prepared, bench, "item-kNN")
    assert sweep["ess"].is_monotonic_increasing
    sens = c.examination_sensitivity(prepared, bench, "item-kNN")
    assert len(sens) == 3


def test_explicit_and_m2_cases(prepared: c.Prepared) -> None:
    ratings = c.explicit_ratings_case(prepared, n_random_pairs=2000)
    assert len(ratings) == 2 and (ratings["rmse_bias_model"] > 0).all()
    m2 = c.m2_compatibility_case()
    assert (m2["eligible_items"] <= 25).all() and len(m2) == 5


def test_case_studies_are_reproducible(bench: c.Benchmark) -> None:
    again = c.benchmark(c.prepare(c.default_config(**SMALL)))
    pd.testing.assert_frame_equal(bench.views, again.views)
    pd.testing.assert_frame_equal(bench.logged, again.logged)
    pd.testing.assert_frame_equal(bench.tuning, again.tuning)


def test_logging_regime_sweep_reports_every_pre_specified_cell() -> None:
    sweep = c.logging_regime_sweep((0.7, 0.0), (1.5, 2.5), **SMALL)
    assert len(sweep.summary) == 4 and len(sweep.per_model) == 4 * 6
    assert sweep.summary["tau_logged_vs_online"].between(-1, 1).all()
    reversed_ = np.sign(sweep.summary["content_minus_pop_logged"]) != np.sign(
        sweep.summary["content_minus_pop_online"])
    assert (reversed_ == sweep.summary["logged_sign_reversed"]).all()
    assert c.REGIME_CORRELATIONS == (0.7, 0.35, 0.0) and c.REGIME_WEIGHTS == (1.5, 2.0, 2.5)


def test_fresh_seed_replication_runs_the_frozen_protocol() -> None:
    rep = c.fresh_seed_replication(**SMALL)
    assert rep.seed == c.FRESH_SEED
    assert set(rep.views) == {"default", "hype"}
    k = rep.contrasts
    assert ((k.logged_lo <= k.logged) & (k.logged <= k.logged_hi)).all()
    assert ((k.online_lo <= k.online) & (k.online <= k.online_hi)).all()


def test_ips_repeated_sampling_reports_coverage_diagnostics() -> None:
    rep = c.ips_repeated_sampling(c.default_config(n_users=300, n_items=80), 20)
    row = rep.rows.iloc[0]
    assert row["coverage_wilson_lo"] <= row["coverage"] <= row["coverage_wilson_hi"]
    assert 0 <= row["coverage_binomial_p"] <= 1
    misses = row["misses_estimate_too_low"] + row["misses_estimate_too_high"]
    assert misses == round((1 - row["coverage"]) * 20)


def test_examination_uncertainty_is_reported(prepared: c.Prepared, bench: c.Benchmark) -> None:
    table = c.examination_uncertainty(prepared, bench, ("item-kNN",), n_bootstrap=20)
    assert (table["sd_from_examination"] > 0).all()
    assert table["share_of_total_variance"].between(0, 1).all()
