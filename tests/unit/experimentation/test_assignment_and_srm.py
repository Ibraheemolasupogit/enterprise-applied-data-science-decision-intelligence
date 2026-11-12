"""Random assignment and sample-ratio-mismatch tests."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from decision_intelligence.experimentation import assignment as A
from decision_intelligence.experimentation.srm import srm_breakdown, srm_from_labels, srm_test

HALF = {"control": 0.5, "treatment": 0.5}


def test_hash_is_sha256_of_the_documented_key() -> None:
    digest = hashlib.sha256(b"exp|salt|42").digest()
    expected = int.from_bytes(digest[:8], "big") / 2**64
    assert A.hash_unit(42, experiment="exp", salt="salt") == expected


def test_hash_assignment_is_deterministic_order_free_and_stable() -> None:
    ids = np.arange(5000)
    labels = A.hash_assignment(ids, experiment="exp")
    shuffled = np.random.default_rng(0).permutation(ids)
    relabelled = dict(zip(shuffled.tolist(), A.hash_assignment(shuffled, experiment="exp").tolist(),
                          strict=True))
    assert all(relabelled[i] == labels[i] for i in range(5000))
    grown = A.hash_assignment(np.arange(8000), experiment="exp")
    assert np.array_equal(grown[:5000], labels)  # new units do not move existing ones


def test_hash_assignment_ratios_and_independence_across_experiments() -> None:
    ids = np.arange(40_000)
    weighted = A.hash_assignment(
        ids, experiment="a", arms=("c", "t1", "t2"), weights=(0.5, 0.3, 0.2)
    )
    counts = pd.Series(weighted).value_counts()
    observed = [counts["c"], counts["t1"], counts["t2"]]
    assert stats.chisquare(observed, np.array([0.5, 0.3, 0.2]) * 40_000).pvalue > 0.001
    other = A.hash_assignment(ids, experiment="b")
    first = A.hash_assignment(ids, experiment="a")
    table = pd.crosstab(first, other).to_numpy()
    assert stats.chi2_contingency(table).pvalue > 0.001  # experiments are independent
    salted = A.hash_assignment(ids, experiment="a", salt="v2")
    assert 0.45 < np.mean(salted == first) < 0.55


def test_complete_randomisation_has_exact_expected_counts_with_remainders() -> None:
    weights = (0.5, 0.3, 0.2)
    totals = np.zeros(3)
    reps = 4000
    for r in range(reps):
        labels = A.complete_randomisation(7, rng=np.random.default_rng(r), arms=("a", "b", "c"),
                                          weights=weights)
        totals += [np.sum(labels == k) for k in ("a", "b", "c")]
    expected = 7 * np.array(weights)
    # each count is floor or ceil of its target, and the mean matches n * w
    np.testing.assert_allclose(totals / reps, expected, atol=0.03)
    labels = A.complete_randomisation(10, rng=np.random.default_rng(0))
    assert np.sum(labels == "treatment") == 5


def test_each_unit_has_the_designed_marginal_probability() -> None:
    hits = np.zeros(5)
    reps = 6000
    for r in range(reps):
        hits += A.complete_randomisation(5, rng=np.random.default_rng(r)) == "treatment"
    assert np.all(np.abs(hits / reps - 0.5) < 4 * math.sqrt(0.25 / reps))


def test_stratified_randomisation_balances_every_stratum() -> None:
    strata = np.repeat(["a", "b", "c"], [101, 50, 7])
    labels = A.stratified_randomisation(strata, rng=np.random.default_rng(1))
    for s in ("a", "b", "c"):
        n_t = np.sum(labels[strata == s] == "treatment")
        size = np.sum(strata == s)
        assert n_t in (size // 2, size - size // 2)
    again = A.stratified_randomisation(strata, rng=np.random.default_rng(1))
    assert np.array_equal(labels, again)


def test_cluster_randomisation_assigns_whole_clusters() -> None:
    clusters = np.repeat(np.arange(12), 5)
    labels = A.cluster_randomisation(clusters, rng=np.random.default_rng(2))
    frame = pd.DataFrame({"c": clusters, "arm": labels})
    assert (frame.groupby("c")["arm"].nunique() == 1).all()
    assert (frame.groupby("c")["arm"].first() == "treatment").sum() == 6
    with pytest.raises(ValueError):
        A.cluster_randomisation(np.repeat([0, 1, 2], 3), rng=np.random.default_rng(0))


@pytest.mark.parametrize("weights", [(0.5,), (0.6, 0.6), (-0.5, 1.5)])
def test_weight_validation(weights: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        A.hash_assignment([1, 2], experiment="e", weights=weights)


def test_srm_chi_square_matches_scipy_and_exact_matches_binomial_sum() -> None:
    result = srm_test({"control": 5120, "treatment": 4880}, HALF)
    reference = stats.chisquare([5120, 4880], [5000, 5000])
    assert result.statistic == pytest.approx(reference.statistic)
    assert result.p_value == pytest.approx(reference.pvalue)
    exact = srm_test({"control": 30, "treatment": 18}, HALF, method="exact-binomial")
    pmf = np.array([math.comb(48, k) for k in range(49)]) / 2**48
    expected = pmf[pmf <= pmf[30] * (1 + 1e-7)].sum()
    assert exact.p_value == pytest.approx(expected)
    three = srm_test({"a": 500, "b": 300, "c": 230}, {"a": 0.5, "b": 0.3, "c": 0.2})
    ref3 = stats.chisquare([500, 300, 230], np.array([0.5, 0.3, 0.2]) * 1030)
    assert three.p_value == pytest.approx(ref3.pvalue) and three.df == 2


def test_srm_detection_and_wording() -> None:
    detected = srm_test({"control": 5300, "treatment": 4700}, HALF)
    assert detected.detected and "NOT identified" in detected.interpretation()
    fine = srm_test({"control": 5010, "treatment": 4990}, HALF)
    assert not fine.detected and "does not prove" in fine.interpretation()
    small = srm_test({"control": 3, "treatment": 1}, HALF)
    assert small.warnings
    with pytest.raises(ValueError):
        srm_test({"control": 1, "other": 2}, HALF)
    with pytest.raises(ValueError):
        srm_test({"control": 1, "treatment": 2}, {"control": 0.5, "treatment": 0.6})


def test_srm_breakdown_localises_the_imbalance() -> None:
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({
        "segment": np.repeat(["web", "ios", "android"], 4000),
        "arm": rng.choice(["control", "treatment"], 12_000),
    })
    android_treated = frame.index[(frame["segment"] == "android") & (frame["arm"] == "treatment")]
    frame = frame.drop(android_treated[:400])  # a client bug drops treated Android users
    assert srm_from_labels(frame["arm"], HALF).detected
    table = srm_breakdown(frame, arm_column="arm", by="segment", expected_shares=HALF)
    flagged = table.loc[table["flagged"], "segment"].tolist()
    assert flagged == ["android"]


def test_m6_interfaces_share_the_inclusive_boundary() -> None:
    """MetricResult.significant and SRMResult.detected use the same rule as the corrections:
    p == alpha rejects; the next representable p above alpha does not."""
    from dataclasses import replace

    from decision_intelligence.experimentation.analysis import analyse_continuous

    rng = np.random.default_rng(3)
    arm = np.repeat(["control", "treatment"], 50)
    result = analyse_continuous(rng.normal(size=100), arm)
    alpha = result.alpha  # 1 - confidence_level
    assert replace(result, p_value=alpha).significant
    assert not replace(result, p_value=float(np.nextafter(alpha, 1.0))).significant

    srm = srm_test({"control": 500, "treatment": 500}, HALF)
    assert replace(srm, p_value=srm.alpha).detected
    assert not replace(srm, p_value=float(np.nextafter(srm.alpha, 1.0))).detected
