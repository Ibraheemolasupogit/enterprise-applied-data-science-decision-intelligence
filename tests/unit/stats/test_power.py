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


def test_t_power_is_finite_for_large_noncentrality() -> None:
    """Regression test (found in M6): SciPy's nct.cdf returns NaN deep in the lower tail
    (e.g. df = 98, ncp ~ 9-20), which made power NaN and the MDE solver fail at
    experiment-scale sample sizes."""
    from statsmodels.stats.power import TTestIndPower

    from decision_intelligence.statistics.power import (
        minimum_detectable_effect_two_sample_t,
        power_two_sample_t,
    )

    for n, d in ((50, 2.0), (50, 3.0), (2000, 0.5), (50, -2.0)):
        value = power_two_sample_t(d, n)
        assert math.isfinite(value) and value == pytest.approx(1.0, abs=1e-9)
    for n in (1000, 2000, 6000):
        mde = minimum_detectable_effect_two_sample_t(n, n)
        reference = TTestIndPower().solve_power(nobs1=n, power=0.8, alpha=0.05, ratio=1.0)
        assert mde == pytest.approx(reference, rel=1e-4)  # statsmodels solver tolerance


def _nan(*_args: object, **_kwargs: object) -> float:
    return math.nan


def test_t_power_nan_tail_fallback_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """The NaN fallback (M6 fix) replaces a non-finite tail by 0 only when the tail is provably
    below 1e-12 (``P(T < -c) <= P(T < 0) = Phi(-ncp)``), so power changes by at most 1e-12.
    NaN is injected so the branch is exercised on every SciPy version."""
    ncp = 2.0 * math.sqrt(50 * 50 / 100)  # d = 2, 50 per arm -> ncp = 10
    critical = float(stats.t.ppf(0.975, 98))
    dominant = float(stats.nct.sf(critical, 98, ncp))  # P(T > c); the other tail < Phi(-10)
    assert math.isfinite(dominant) and stats.norm.cdf(-ncp) < 1e-12
    with monkeypatch.context() as patch:
        patch.setattr(stats.nct, "cdf", _nan)  # lower tail fails
        value = pw.power_two_sample_t(2.0, 50)
        assert math.isfinite(value) and abs(value - dominant) <= 1e-12
    with monkeypatch.context() as patch:
        patch.setattr(stats.nct, "sf", _nan)  # upper tail fails (negative effect, mirror image)
        value = pw.power_two_sample_t(-2.0, 50)
        assert math.isfinite(value) and abs(value - dominant) <= 1e-12


def test_t_power_nan_tail_is_raised_when_not_negligible(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-finite tail that could be material is an error, never silently 0 or NaN: the fix
    does not weaken the M3 contract (finite, accurate power or an exception)."""
    with monkeypatch.context() as patch:
        patch.setattr(stats.nct, "cdf", _nan)
        with pytest.raises(FloatingPointError, match="lower tail"):
            pw.power_two_sample_t(0.1, 50)  # ncp = 0.5: Phi(-0.5) = 0.31
    with monkeypatch.context() as patch:
        patch.setattr(stats.nct, "sf", _nan)
        with pytest.raises(FloatingPointError, match="upper tail"):
            pw.power_two_sample_t(0.1, 50, alternative="greater")
