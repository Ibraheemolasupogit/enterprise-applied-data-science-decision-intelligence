"""Power and sample size versus statsmodels and closed-form textbook values."""

from __future__ import annotations

import math

import pytest
from scipy import stats
from statsmodels.stats.power import TTestIndPower, TTestPower

from decision_intelligence.statistics import power as pw


@pytest.mark.parametrize(
    ("effect", "n_x", "n_y", "alternative", "sm_alternative"),
    [
        (0.5, 30, None, "two-sided", "two-sided"),
        (0.4, 40, 80, "greater", "larger"),
        (-0.3, 50, 50, "less", "smaller"),
        (0.2, 10, 15, "two-sided", "two-sided"),
    ],
)
def test_two_sample_t_power_matches_statsmodels(
    effect: float, n_x: int, n_y: int | None, alternative: str, sm_alternative: str
) -> None:
    ours = pw.power_two_sample_t(effect, n_x, n_y, alternative=alternative)
    ratio = 1.0 if n_y is None else n_y / n_x
    reference = TTestIndPower().power(
        effect, n_x, 0.05, ratio=ratio, alternative=sm_alternative
    )
    assert ours == pytest.approx(reference, abs=1e-10)


def test_one_sample_power_matches_statsmodels() -> None:
    assert pw.power_one_sample_t(0.3, 50) == pytest.approx(TTestPower().power(0.3, 50, 0.05))


def test_zero_effect_gives_alpha() -> None:
    assert pw.power_two_sample_t(0.0, 40) == pytest.approx(0.05)
    assert pw.power_one_sample_t(0.0, 40, alpha=0.01) == pytest.approx(0.01)


def test_classic_sample_size_d_half_is_64_per_group() -> None:
    assert pw.sample_size_two_sample_t(0.5) == (64, 64)
    n_continuous = TTestIndPower().solve_power(0.5, power=0.8, alpha=0.05)
    assert math.ceil(n_continuous) == 64


def test_sample_size_is_minimal_and_attains_target() -> None:
    for ratio in (1.0, 2.0, 0.5):
        n_x, n_y = pw.sample_size_two_sample_t(0.35, ratio=ratio, power=0.9)
        assert pw.power_two_sample_t(0.35, n_x, n_y) >= 0.9
        previous_y = max(2, math.ceil(ratio * (n_x - 1)))
        assert pw.power_two_sample_t(0.35, n_x - 1, previous_y) < 0.9
    assert pw.sample_size_one_sample_t(0.5) == math.ceil(
        TTestPower().solve_power(0.5, power=0.8, alpha=0.05)
    )


def test_minimum_detectable_effect_inverts_power() -> None:
    mde = pw.minimum_detectable_effect_two_sample_t(64)
    assert mde == pytest.approx(
        TTestIndPower().solve_power(nobs1=64, power=0.8, alpha=0.05), abs=1e-6
    )
    assert pw.power_two_sample_t(mde, 64) == pytest.approx(0.8, abs=1e-8)
    with pytest.raises(ValueError, match="exceed alpha"):
        pw.minimum_detectable_effect_two_sample_t(64, power=0.04)


def test_two_proportion_sample_size_matches_closed_form() -> None:
    """n = (z_a sqrt(2 pbar qbar) + z_b sqrt(p1 q1 + p2 q2))^2 / (p1 - p2)^2 per group."""
    p1, p2 = 0.12, 0.10
    z_a, z_b = stats.norm.ppf(0.975), stats.norm.ppf(0.8)
    pbar = (p1 + p2) / 2
    closed = (
        z_a * math.sqrt(2 * pbar * (1 - pbar)) + z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2 / (p1 - p2) ** 2
    n_x, n_y = pw.sample_size_two_proportions(p1, p2)
    assert n_x == n_y
    assert abs(n_x - closed) <= 1.0  # integer search vs continuous formula (opposite tail adds ~0)
    assert pw.power_two_proportions(p1, p2, n_x) >= 0.8


def test_power_increases_with_sample_size_and_effect() -> None:
    assert pw.power_two_proportions(0.15, 0.10, 500) > pw.power_two_proportions(0.15, 0.10, 200)
    assert pw.power_two_proportions(0.20, 0.10, 200) > pw.power_two_proportions(0.15, 0.10, 200)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: pw.sample_size_two_sample_t(0.0), "non-zero"),
        (lambda: pw.sample_size_two_sample_t(-0.3, alternative="greater"), "contradicts"),
        (lambda: pw.sample_size_two_proportions(0.1, 0.1), "equals"),
        (lambda: pw.power_two_sample_t(0.5, 1), ">= 2"),
        (lambda: pw.power_two_sample_t(math.inf, 10), "finite"),
        (lambda: pw.sample_size_two_sample_t(1e-6), "100 million"),
    ],
)
def test_invalid_designs_raise(call: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()  # type: ignore[operator]
