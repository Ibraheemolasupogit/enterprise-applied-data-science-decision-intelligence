"""M8 statistical properties: IPS in repeated sampling, exposure bias, leakage, positivity,
missing-not-at-random ratings and stability across generator seeds."""

from __future__ import annotations

import numpy as np
import pytest

from decision_intelligence.recommendations import baselines as bl
from decision_intelligence.recommendations import cases as c
from decision_intelligence.recommendations import evaluation as ev
from decision_intelligence.recommendations import off_policy as op
from decision_intelligence.recommendations import oracle as orc
from decision_intelligence.statistics.hypothesis_tests import paired_t_test


@pytest.fixture(scope="module")
def default() -> c.Prepared:
    return c.prepare(c.default_config())


@pytest.fixture(scope="module")
def hype() -> c.Prepared:
    return c.prepare(c.hype_config())


@pytest.fixture(scope="module")
def repeated() -> c.IPSRepeated:
    return c.ips_repeated_sampling(c.default_config(n_users=600, n_items=150), 150)


def test_ips_is_unbiased_with_the_true_examination(repeated: c.IPSRepeated) -> None:
    for _, row in repeated.rows.iterrows():
        assert abs(row["true_examination_bias"]) < 4 * row["true_examination_mc_se"]


def test_ips_with_estimated_examination_has_small_bias_and_near_nominal_coverage(
    repeated: c.IPSRepeated,
) -> None:
    for _, row in repeated.rows.iterrows():
        assert abs(row["bias"]) < 0.05 * row["exact_value"]
        assert 0.88 <= row["coverage"] <= 1.0


def test_clipping_biases_ips_downwards(repeated: c.IPSRepeated) -> None:
    for _, row in repeated.rows.iterrows():
        assert row["clipped_bias"] < -4 * row["mc_se"]


def test_ignoring_position_bias_underestimates_a_relevance_sorted_list(
    repeated: c.IPSRepeated,
) -> None:
    row = repeated.rows.set_index("policy").loc["item-kNN (k=50)"]
    assert row["position_ignored_bias"] < -4 * row["position_ignored_mc_se"]


def test_relative_examination_is_recovered_from_position_exchangeable_logs(
    default: c.Prepared,
) -> None:
    truth = default.scenario.truth.examination / default.scenario.truth.examination[0]
    assert np.allclose(default.examination.relative, truth, rtol=0.04)


def _views(p: c.Prepared, model: bl.Recommender) -> tuple[np.ndarray, orc.OnlineValue,
                                                          op.IPSResult]:
    recs = ev.recommend(model.fit(p.history), p.test, c.K)
    logged = ev.logged_metrics(recs, p.test)["ndcg"].to_numpy()
    online = orc.simulated_online_value(p.scenario.logs, p.history, p.scenario.truth,
                                        p.test.users, recs, p.split.validation_end,
                                        p.split.test_end)
    ips = op.ips_policy_value(p.scenario.logs, p.history, p.test.users, recs,
                              p.split.validation_end, p.split.test_end, p.examination.relative)
    return logged, online, ips


def test_exposure_bias_reverses_a_logged_comparison_and_ips_points_the_right_way(
    hype: c.Prepared,
) -> None:
    content = _views(hype, bl.Fallback(bl.ContentBased(), bl.SegmentPopularity(100.0)))
    pop = _views(hype, bl.GlobalPopularity())
    logged = paired_t_test(content[0], pop[0])
    assert logged.estimate < 0 and logged.p_value < 1e-6  # logged: popularity looks better
    online = op.clustered_mean(content[1].per_session - pop[1].per_session,
                               content[1].session_users)
    ips = op.clustered_mean(content[2].per_session - pop[2].per_session,
                            content[2].session_users)
    assert online[2] > 0  # simulated online value: content-based is better
    # IPS (same logs) has the right sign and its interval covers the true difference; with
    # few supported slots for content-based lists it is too noisy to be decisive.
    assert ips[0] > 0 and ips[2] <= online[0] <= ips[3]


def test_without_strong_hype_logged_oracle_and_online_views_agree(default: c.Prepared) -> None:
    knn = _views(default, bl.Fallback(bl.ItemKNN(50), bl.SegmentPopularity(100.0)))
    pop = _views(default, bl.GlobalPopularity())
    logged = paired_t_test(knn[0], pop[0])
    assert logged.ci_lower is not None and logged.ci_lower > 0
    online = op.clustered_mean(knn[1].per_session - pop[1].per_session, knn[1].session_users)
    assert online[2] > 0


def test_random_split_leakage_inflates_logged_metrics(default: c.Prepared) -> None:
    study = c.leakage_study(default)
    row = study.random_vs_chronological.set_index("model").loc["item-kNN (k=50)"]
    assert row["inflation"] > 0 and row["p_value"] < 0.01
    assert (study.perturbation["max_abs_score_change"] == 0).all()


def test_ips_targets_only_the_supported_value_when_positivity_fails() -> None:
    rows = c.positivity_study().rows.set_index("policy")
    content = rows.loc["content-based"]
    assert content["unsupported_share"] > 0.3
    assert content["ips_lo"] <= content["exact_supported_value"] <= content["ips_hi"]
    assert content["ips_hi"] < content["exact_value"]
    pop = rows.loc["global popularity"]
    assert pop["unsupported_share"] == 0.0
    assert pop["ips_lo"] <= pop["exact_value"] <= pop["ips_hi"]


def test_observed_rating_error_understates_error_on_random_pairs(default: c.Prepared) -> None:
    table = c.explicit_ratings_case(default).set_index("evaluation")
    observed, random_pairs = table.iloc[0], table.iloc[1]
    assert observed["mean_actual"] > random_pairs["mean_actual"] + 1.0
    assert random_pairs["rmse_bias_model"] > 2 * observed["rmse_bias_model"]
    assert random_pairs["mean_predicted"] > random_pairs["mean_actual"] + 1.0


def test_collaborative_baselines_beat_popularity_across_seeds() -> None:
    table = c.seed_stability((1, 2, 3)).pivot(index="seed", columns="model")
    for view in ("logged_ndcg", "oracle_ndcg", "online_clicks"):
        assert (table[view]["item-kNN (k=50)"] > table[view]["global popularity"]).all()
        assert (table[view]["global popularity"] > 0).all()
