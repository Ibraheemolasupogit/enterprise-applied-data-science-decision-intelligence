"""M9 statistical properties: off-policy calibration, A/A error rate, A/B coverage and the
advanced models' quality on a reduced scenario (never the fresh seed)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from decision_intelligence.recommendations import advanced_cases as ac
from decision_intelligence.recommendations import baselines as bl
from decision_intelligence.recommendations import cases as m8
from decision_intelligence.recommendations import evaluation as ev
from decision_intelligence.recommendations import online
from decision_intelligence.recommendations import oracle as orc
from decision_intelligence.recommendations import protocol as pr
from decision_intelligence.recommendations.factorisation import BPR, EASE, ImplicitALS
from decision_intelligence.recommendations.interactions import InteractionData

SMALL = {"n_users": 400, "n_items": 100}
ABSetup = tuple[m8.Prepared, InteractionData, np.ndarray, dict[str, np.ndarray]]


@pytest.fixture(scope="module")
def calibration() -> ac.OPECalibration:
    return ac.ope_calibration(150, 199, **SMALL)


def test_ips_and_dr_are_unbiased_even_with_a_wrong_click_model(
    calibration: ac.OPECalibration,
) -> None:
    rows = calibration.rows
    for est in ("IPS", "DR (fitted click model)", "DR (constant click model)"):
        for _, r in rows[rows["estimator"] == est].iterrows():
            assert abs(r["bias"]) < 4 * r["mc_se"], (est, r["policy"])


def test_snips_bias_is_small(calibration: ac.OPECalibration) -> None:
    for _, r in calibration.rows[calibration.rows["estimator"] == "SNIPS"].iterrows():
        assert abs(r["bias"]) < 0.05 * r["exact"]


def test_interval_coverage_is_near_nominal(calibration: ac.OPECalibration) -> None:
    for _, r in calibration.rows.iterrows():
        for iv in ("normal", "percentile", "studentised"):
            assert r[f"{iv}_coverage"] >= 0.85, (r["estimator"], iv)


@pytest.fixture(scope="module")
def ab_setup() -> ABSetup:
    p = m8.prepare(m8.default_config(n_users=900, n_items=200))
    h = ac.history_at(p.scenario.logs, pr.AB_DESIGN.start_day)
    users = online.experiment_users(p.scenario, h.cutoff)
    lists = {"control": online.policy_lists(bl.GlobalPopularity().fit(h), h, users, pr.K),
             "single_model": online.policy_lists(
                 bl.Fallback(bl.ItemKNN(50), bl.SegmentPopularity(100.0)).fit(h), h, users,
                 pr.K)}
    lists["hybrid"] = lists["single_model"]
    return p, h, users, lists


def test_aa_false_positive_rate_is_nominal(ab_setup: ABSetup) -> None:
    p, h, users, lists = ab_setup
    aa = online.aa_study(p.scenario, h, lists["control"], users, pr.AB_DESIGN, 200)
    k = int((aa.rows["p_value"] <= 0.05).sum())
    assert stats.binomtest(k, 200, 0.05).pvalue > 0.001


def test_ab_intervals_cover_the_exact_simulated_effect(ab_setup: ABSetup) -> None:
    p, h, users, lists = ab_setup
    cov = online.coverage_study(p.scenario, h, lists, users, pr.AB_DESIGN, "single_model", 150)
    assert cov.rows["covered_95"].mean() >= 0.88
    assert cov.rows["covered_bonferroni"].mean() >= cov.rows["covered_95"].mean()


def test_factorisation_models_beat_popularity_on_the_oracle() -> None:
    p = m8.prepare(m8.default_config(n_users=1000, n_items=200))
    truth = p.scenario.truth

    def oracle_ndcg(model: bl.Recommender) -> float:
        recs = ev.recommend(model.fit(p.history), p.test, pr.K)
        return float(orc.oracle_metrics(recs, p.test, truth, p.history)["oracle_ndcg"].mean())

    pop = oracle_ndcg(bl.GlobalPopularity())
    for model in (ImplicitALS(16, 0.1, 5.0, pr.ALS_ITERATIONS), BPR(32, 0.01), EASE(200.0)):
        assert oracle_ndcg(bl.Fallback(model, bl.SegmentPopularity(100.0))) > pop + 0.05


def test_content_strategies_beat_constant_scores_on_new_items() -> None:
    run = ac.full_run(m8.default_config(n_users=800, n_items=200), with_ope=False)
    items = ac.cold_start(run.prepared, run.selection, run.views, ("item-kNN",)).items
    items = items.set_index("model")
    const = items.loc["global popularity", "oracle_ndcg_new_items"]
    assert items.loc["global popularity", "constant_on_new_items"]
    assert items.loc["content-based", "oracle_ndcg_new_items"] > const
    assert items.loc["content → ALS factors", "oracle_ndcg_new_items"] > const
    assert np.isfinite(items["oracle_ndcg_new_items"]).all()
