"""Noncompliance, attrition bounds, validity checks and decision rules."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from decision_intelligence.experimentation import compliance as co
from decision_intelligence.experimentation import validity as va
from decision_intelligence.experimentation.analysis import analyse_binary, analyse_continuous
from decision_intelligence.experimentation.decision import (
    assess_guardrail,
    business_impact,
    recommend,
)
from decision_intelligence.experimentation.design import ExperimentDesign, MetricSpec


@pytest.fixture
def iv_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(9)
    n = 4000
    z = (rng.random(n) < 0.5).astype(float)
    complier = rng.random(n) < 0.6
    d = np.where(complier, z, 0.0)
    y = 1.0 + 2.0 * d + rng.normal(size=n)
    return y, z, d


def test_wald_equals_two_stage_least_squares(iv_data) -> None:
    y, z, d = iv_data
    result = co.noncompliance_analysis(y, z, d)
    # Independent 2SLS: X = [1, d], instruments Z = [1, z].
    x = np.column_stack([np.ones_like(d), d])
    zz = np.column_stack([np.ones_like(z), z])
    beta = np.linalg.solve(zz.T @ x, zz.T @ y)
    assert result.cace.estimate == pytest.approx(beta[1], rel=1e-12)
    u = y - x @ beta
    bread = np.linalg.inv(zz.T @ x)
    hc0 = bread @ (zz.T * u**2) @ zz @ bread.T
    n1, n0 = z.sum(), (1 - z).sum()
    correction = math.sqrt(
        (np.var(u[z == 1], ddof=1) / n1 + np.var(u[z == 0], ddof=1) / n0)
        / (np.var(u[z == 1]) / n1 + np.var(u[z == 0]) / n0)
    )
    assert result.cace.standard_error == pytest.approx(math.sqrt(hc0[1, 1]) * correction, rel=1e-9)
    assert result.first_stage.estimate == pytest.approx(0.6, abs=0.03)
    assert result.itt.estimate == pytest.approx(result.cace.estimate * result.first_stage.estimate)


def test_weak_first_stage_is_flagged_and_unidentified_case_refused() -> None:
    rng = np.random.default_rng(10)
    z = (rng.random(400) < 0.5).astype(float)
    d = np.zeros(400)
    d[np.flatnonzero(z == 1)[:5]] = 1.0  # 5 of ~200 assigned units comply: F ~ 5
    y = rng.normal(size=400)
    assert any("weak" in w for w in co.noncompliance_analysis(y, z, d).cace.warnings)
    with pytest.raises(ValueError, match="not identified"):
        co.noncompliance_analysis(y, z, np.zeros(400))


def test_trimmed_mean_and_lee_bounds_by_hand() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert co._trimmed_mean(values, 0.25, drop="top") == pytest.approx(2.0)
    assert co._trimmed_mean(values, 0.25, drop="bottom") == pytest.approx(3.0)
    assert co._trimmed_mean(values, 0.125, drop="top") == pytest.approx((1 + 2 + 3 + 0.5 * 4) / 3.5)
    # Control: 8 units, all observed, mean 0.5. Treatment: 8 units, 6 observed (q_t = 0.75).
    y_c = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    y_t = np.array([1, 1, 1, 0, 0, 0, np.nan, np.nan])
    y = np.concatenate([y_t, y_c])
    z = np.repeat([1.0, 0.0], 8)
    bounds = co.lee_bounds(y, z)
    # Control responds more: trim p = (1 - 0.75) / 1 = 0.25 of control (2 of 8).
    assert bounds.trimmed_arm == "control" and bounds.trimming_share == pytest.approx(0.25)
    assert bounds.lower == pytest.approx(0.5 - 4 / 6)  # drop two zeros -> control mean 4/6
    assert bounds.upper == pytest.approx(0.5 - 2 / 6)  # drop two ones -> control mean 2/6
    lo, hi = co.manski_bounds(y, z, lower=0.0, upper=1.0)
    assert (lo, hi) == pytest.approx((0.375 - 0.5, 0.625 - 0.5))
    with pytest.raises(ValueError):
        co.manski_bounds(y, z, lower=0.0, upper=0.5)


def test_lee_bootstrap_interval_contains_the_bounds() -> None:
    rng = np.random.default_rng(11)
    y = rng.normal(size=600)
    z = np.repeat([1.0, 0.0], 300)
    y[(z == 1) & (rng.random(600) < 0.2)] = np.nan
    bounds = co.lee_bounds(y, z, rng=np.random.default_rng(1), n_bootstrap=299)
    assert bounds.ci_lower is not None and bounds.ci_upper is not None
    assert bounds.ci_lower < bounds.lower <= bounds.upper < bounds.ci_upper
    summary = co.attrition_summary(y, z)
    assert summary.response_control == 1.0 and summary.differential < -0.1
    assert summary.differential_p_value < 0.001


def test_duplicate_and_first_assignment_checks() -> None:
    log = pd.DataFrame(
        {
            "unit_id": [1, 2, 2, 3, 3],
            "arm": ["control", "treatment", "treatment", "control", "treatment"],
            "assigned_at": pd.to_datetime(["2025-01-01"] * 3 + ["2025-01-01", "2025-01-02"]),
        }
    )
    assert va.check_duplicates(log).status == "FAIL"
    assert va.check_duplicates(log.iloc[:3]).status == "WARN"
    assert va.check_duplicates(log.iloc[[0, 1, 3]]).status == "PASS"
    first = va.first_assignment(log)
    assert first.set_index("unit_id")["arm"].to_dict() == {
        1: "control",
        2: "treatment",
        3: "control",
    }


def test_leakage_balance_exclusion_integrity_and_selection_checks() -> None:
    rng = np.random.default_rng(12)
    z = np.repeat([1.0, 0.0], 1000)
    exposed = z.copy()
    exposed[1000:1050] = 1.0  # 5% of control exposed
    assert va.check_assignment_leakage(z, exposed).status == "WARN"
    assert va.check_assignment_leakage(z, z).status == "PASS"

    covariates = pd.DataFrame({"segment": rng.choice(["a", "b"], 2000), "x": rng.normal(size=2000)})
    ok, table = va.check_pre_treatment_balance(covariates, z)
    assert ok.status == "PASS" and set(table.index) == {"segment_b", "x"}
    broken = covariates.assign(x=covariates["x"] + 0.5 * z)  # assignment depends on x
    assert va.check_pre_treatment_balance(broken, z)[0].status == "FAIL"

    excluded = np.zeros(2000, dtype=bool)
    assert va.check_exclusions(excluded, z, rule_timing={"test accounts": "pre"}).status == "PASS"
    assert va.check_exclusions(excluded, z, rule_timing={"bounced": "post"}).status == "FAIL"
    lopsided = excluded.copy()
    lopsided[:100] = True
    assert va.check_exclusions(lopsided, z, rule_timing={"x": "pre"}).status == "FAIL"

    assert va.check_metric_integrity([0, 1, 1], metric="m", kind="binary").status == "PASS"
    assert va.check_metric_integrity([0, 2, 1], metric="m", kind="binary").status == "FAIL"
    assert (
        va.check_metric_integrity([1.0, -3.0], metric="m", kind="continuous", lower=0.0).status
        == "FAIL"
    )
    assert va.check_metric_integrity([1.0, np.nan], metric="m", kind="continuous").status == "WARN"

    engaged = (rng.random(2000) < np.where(z == 1, 0.6, 0.4)).astype(float)
    post = va.check_post_randomisation_selection(
        "engaged", engaged, z, measured_after_assignment=True
    )
    assert post.status == "FAIL" and post.value is not None and post.value < 1e-6
    pre = va.check_post_randomisation_selection(
        "segment_b",
        (covariates["segment"] == "b").astype(float),
        z,
        measured_after_assignment=False,
    )
    assert pre.status == "PASS"
    assert va.overall_status([ok, post]) == "FAIL" and va.overall_status([ok]) == "PASS"


def _binary_result(s_t: int, s_c: int, n: int = 10_000):
    y = np.concatenate(
        [np.repeat([1.0, 0.0], [s_t, n - s_t]), np.repeat([1.0, 0.0], [s_c, n - s_c])]
    )
    return analyse_binary(y, np.repeat(["treatment", "control"], n))


@pytest.mark.parametrize(
    ("s_t", "s_c", "n", "expected"),
    [
        (1300, 1000, 10_000, "LAUNCH"),  # +3pp: CI excludes 0 and estimate >= 1pp
        (4320, 4000, 40_000, "ITERATE"),  # +0.8pp: CI excludes 0, estimate below 1pp
        (1050, 1000, 10_000, "HOLD"),  # CI contains both 0 and 1pp: inconclusive
        (4000, 4000, 40_000, "STOP"),  # CI upper below 1pp: worthwhile effect ruled out
        (900, 1000, 10_000, "STOP"),  # harmful
    ],
)
def test_decision_rule_branches(s_t: int, s_c: int, n: int, expected: str) -> None:
    result = _binary_result(s_t, s_c, n=n)
    assert recommend(result, minimum_worthwhile_effect=0.01).decision == expected


def test_validity_failure_and_guardrails_override_the_primary() -> None:
    strong = _binary_result(1300, 1000)
    held = recommend(strong, minimum_worthwhile_effect=0.01, validity_failures=["SRM"])
    assert held.decision == "HOLD" and "not interpretable" in " ".join(held.reasons)
    rng = np.random.default_rng(13)
    arm = np.repeat(["treatment", "control"], 5000)
    latency = rng.normal(200, 20, 10_000) + np.where(arm == "treatment", 15.0, 0.0)
    spec = MetricSpec("latency_ms", "continuous", "page latency", direction="decrease", margin=10)
    harm = assess_guardrail(analyse_continuous(latency, arm), spec)
    assert harm.status == "FAIL" and harm.deterioration == pytest.approx(15, abs=1.5)
    assert recommend(strong, minimum_worthwhile_effect=0.01, guardrails=[harm]).decision == "STOP"
    fine = assess_guardrail(
        analyse_continuous(latency - np.where(arm == "treatment", 15, 0), arm), spec
    )
    assert fine.status == "PASS"
    unclear = assess_guardrail(
        analyse_continuous(latency - np.where(arm == "treatment", 5, 0), arm), spec
    )
    assert unclear.status == "INCONCLUSIVE"
    assert (
        recommend(strong, minimum_worthwhile_effect=0.01, guardrails=[unclear]).decision
        == "ITERATE"
    )


def test_business_impact_and_design_validation() -> None:
    impact = business_impact(
        0.02, (0.01, 0.03), units_per_year=1e6, value_per_unit_effect=40, annual_cost=300_000
    )
    assert impact.annual_value == pytest.approx(500_000)
    assert impact.annual_value_ci == pytest.approx((100_000, 900_000))
    assert impact.break_even_effect == pytest.approx(0.0075)
    primary = MetricSpec("conversion", "binary", "converted in 14 days")
    with pytest.raises(ValueError, match="clustered analysis"):
        ExperimentDesign("e", "h", "p", "user", "city", "user", "ITT", primary, 0.01)
    with pytest.raises(ValueError, match="margin"):
        ExperimentDesign(
            "e",
            "h",
            "p",
            "user",
            "user",
            "user",
            "ITT",
            primary,
            0.01,
            guardrails=(MetricSpec("latency", "continuous", "ms", direction="decrease"),),
        )
    design = ExperimentDesign(
        "e", "h", "p", "user", "city", "user", "ITT", primary, 0.01, clustered_analysis=True
    )
    assert dict(design.summary_rows())["Randomisation unit"] == "city"


def _binary_from_counts(s_t: int, n_t: int, s_c: int, n_c: int) -> tuple[np.ndarray, np.ndarray]:
    y = np.concatenate([np.repeat([1.0, 0.0], [s_t, n_t - s_t]),
                        np.repeat([1.0, 0.0], [s_c, n_c - s_c])])
    arm = np.repeat(["treatment", "control"], [n_t, n_c])
    return y, arm


@pytest.mark.parametrize(
    ("counts", "significant", "excludes_zero"),
    [
        ((25, 100, 14, 100), True, False),  # z-test p = 0.0496; Newcombe [-0.0004, 0.218]
        ((168, 200, 36, 50), False, True),  # z-test p = 0.0502; Newcombe [0.0002, 0.264]
    ],
)
def test_test_interval_disagreement_at_the_boundary_holds(
    counts: tuple[int, int, int, int], significant: bool, excludes_zero: bool
) -> None:
    """Regression: the pooled z-test and the Newcombe interval are not dual. At the boundary
    they are reported as computed (no forced agreement), flagged, and the decision is HOLD.
    The second case was LAUNCH before the policy (CI lower > 0 and estimate >= delta)."""
    result = analyse_binary(*_binary_from_counts(*counts))
    assert result.significant is significant
    assert result.interval_excludes_zero is excludes_zero
    assert not result.test_and_interval_agree
    assert any("boundary disagreement" in w for w in result.warnings)
    assert any("boundary disagreement" in w for w in result.effect.warnings)
    decision = recommend(result, minimum_worthwhile_effect=0.01)
    assert decision.decision == "HOLD"
    assert "boundary evidence" in decision.reasons[0]
    # A STOP (worthwhile benefit ruled out) still takes precedence: it involves no test.
    assert recommend(result, minimum_worthwhile_effect=0.5).decision == "STOP"


def test_agreeing_test_and_interval_carry_no_boundary_warning() -> None:
    result = analyse_binary(*_binary_from_counts(40, 100, 14, 100))
    assert result.significant and result.interval_excludes_zero and result.test_and_interval_agree
    assert not any("boundary" in w for w in result.warnings)
    assert recommend(result, minimum_worthwhile_effect=0.01).decision == "LAUNCH"


def test_lee_bounds_with_an_unobserved_arm_are_undefined() -> None:
    y = np.array([1.0, 0.0, 1.0, np.nan, np.nan, np.nan])
    z = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    with pytest.raises(co.LeeBoundsUndefinedError, match="treatment arm has no observed"):
        co.lee_bounds(y, z)


def test_lee_bootstrap_is_refused_when_resamples_can_be_undefined() -> None:
    """Sparse response in a small arm: an empty-arm resample is likely, so the bootstrap is
    refused up front (independently of the seed); the point bounds remain available."""
    y = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, np.nan, np.nan, np.nan])
    z = np.array([0.0] * 6 + [1.0] * 4)
    point = co.lee_bounds(y, z)
    assert point.ci_lower is None and math.isfinite(point.lower) and math.isfinite(point.upper)
    for seed in range(3):
        with pytest.raises(co.LeeBoundsUndefinedError, match="bootstrap refused"):
            co.lee_bounds(y, z, rng=np.random.default_rng(seed), n_bootstrap=199)


def test_lee_bootstrap_never_discards_an_undefined_draw(monkeypatch: pytest.MonkeyPatch) -> None:
    """If an undefined resample occurs anyway, the bootstrap fails explicitly rather than
    dropping the draw or returning an interval (tolerance lifted to reach the draw loop)."""
    monkeypatch.setattr(co, "LEE_BOOTSTRAP_FAILURE_TOLERANCE", 1.0)
    y = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, np.nan, np.nan, np.nan])
    z = np.array([0.0] * 6 + [1.0] * 4)
    with pytest.raises(co.LeeBoundsUndefinedError, match="undefined draws are not discarded"):
        co.lee_bounds(y, z, rng=np.random.default_rng(0), n_bootstrap=199)
