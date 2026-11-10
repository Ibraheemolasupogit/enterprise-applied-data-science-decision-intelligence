"""Effect sizes versus reference formulas/implementations, and numerical robustness."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.contingency_tables import Table2x2

from decision_intelligence.common import seeded_generator
from decision_intelligence.statistics import effect_sizes as es


@pytest.fixture
def groups() -> tuple[np.ndarray, np.ndarray]:
    rng = seeded_generator(31)
    return rng.normal(0.5, 1.0, 20), rng.normal(0.0, 1.0, 25)


def test_cohens_d_and_noncentral_interval_invert_the_cdf(
    groups: tuple[np.ndarray, np.ndarray],
) -> None:
    x, y = groups
    d = es.cohens_d(x, y)
    pooled = math.sqrt(((19) * x.var(ddof=1) + 24 * y.var(ddof=1)) / 43)
    assert d.estimate == pytest.approx((x.mean() - y.mean()) / pooled)
    assert d.ci_lower is not None and d.ci_upper is not None
    scale = math.sqrt(1 / 20 + 1 / 25)
    t_obs = d.estimate / scale
    # The defining property of the inverted interval:
    assert stats.nct.cdf(t_obs, 43, d.ci_lower / scale) == pytest.approx(0.975, abs=1e-8)
    assert stats.nct.cdf(t_obs, 43, d.ci_upper / scale) == pytest.approx(0.025, abs=1e-8)
    assert "noncentral-t inversion" in d.note


def test_hedges_correction_exact_and_approximation() -> None:
    assert es.hedges_correction(43) == pytest.approx(1 - 3 / (4 * 43 - 1), abs=1e-4)
    assert es.hedges_correction(10_000) == pytest.approx(1.0, abs=1e-4)
    with pytest.raises(ValueError):
        es.hedges_correction(1)


def test_hedges_g_shrinks_d(groups: tuple[np.ndarray, np.ndarray]) -> None:
    d, g = es.cohens_d(*groups), es.hedges_g(*groups)
    assert abs(g.estimate) < abs(d.estimate)
    assert g.estimate == pytest.approx(es.hedges_correction(43) * d.estimate)


def test_huge_standardised_difference_falls_back_gracefully() -> None:
    """For |t| ~ 100 SciPy's noncentral t CDF returns NaN in the tail; the interval must
    still be produced (by the documented normal approximation) rather than crash."""
    rng = seeded_generator(32)
    x, y = rng.normal(10.0, 1.0, 2000), rng.normal(0.0, 1.0, 2000)
    d = es.cohens_d(x, y)
    assert d.ci_lower is not None and d.ci_upper is not None
    assert d.ci_lower < d.estimate < d.ci_upper
    assert math.isfinite(d.ci_lower) and math.isfinite(d.ci_upper)


def test_cohens_dz_and_validation() -> None:
    before = seeded_generator(33).normal(10, 2, 15)
    after = before + seeded_generator(34).normal(1, 1, 15)
    dz = es.cohens_dz(after, before)
    diff = after - before
    assert dz.estimate == pytest.approx(diff.mean() / diff.std(ddof=1))
    with pytest.raises(ValueError, match="constant"):
        es.cohens_dz([2.0, 3.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="zero pooled"):
        es.cohens_d([1.0, 1.0], [2.0, 2.0])


def test_cliffs_delta_matches_pairwise_definition(groups: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = groups
    pairwise = np.sign(x[:, None] - y[None, :]).mean()
    assert es.cliffs_delta(x, y).estimate == pytest.approx(pairwise)
    assert es.cliffs_delta([5.0, 6.0], [1.0, 2.0]).estimate == 1.0


def test_binary_effect_sizes_match_statsmodels_table() -> None:
    result = es.binary_effect_sizes(30, 100, 18, 110)
    table = Table2x2(np.array([[30, 70], [18, 92]]))
    assert result.odds_ratio.estimate == pytest.approx(table.oddsratio)
    assert (result.odds_ratio.ci_lower, result.odds_ratio.ci_upper) == pytest.approx(
        table.oddsratio_confint()
    )
    assert result.risk_ratio.estimate == pytest.approx(table.riskratio)
    assert (result.risk_ratio.ci_lower, result.risk_ratio.ci_upper) == pytest.approx(
        table.riskratio_confint()
    )
    assert result.risk_difference.estimate == pytest.approx(0.30 - 18 / 110)
    assert result.cohens_h.estimate == pytest.approx(
        2 * math.asin(math.sqrt(0.3)) - 2 * math.asin(math.sqrt(18 / 110))
    )
    assert not result.zero_cell_correction_applied


def test_zero_cell_uses_haldane_correction() -> None:
    result = es.binary_effect_sizes(0, 50, 5, 50)
    assert result.zero_cell_correction_applied
    assert math.isfinite(result.odds_ratio.estimate) and result.odds_ratio.estimate > 0
    assert "Haldane" in result.odds_ratio.note
    with pytest.raises(ValueError, match="cannot exceed"):
        es.binary_effect_sizes(51, 50, 5, 50)


@pytest.mark.parametrize(("t_obs", "df"), [(20.0, 2), (40.0, 1), (-25.0, 2), (60.0, 2)])
def test_small_df_large_t_uses_exact_inversion(t_obs: float, df: int) -> None:
    """Regression test: SciPy's nct.cdf returns NaN inside these brackets; the
    survival-function identity keeps the inversion exact instead of crashing."""
    lower, upper, note = es._noncentrality_interval(t_obs, df, 0.95)
    assert note == "CI by noncentral-t inversion"
    assert stats.nct.sf(-t_obs, df, -lower) == pytest.approx(0.975, abs=1e-8)
    assert stats.nct.sf(-t_obs, df, -upper) == pytest.approx(0.025, abs=1e-8)


def test_fallback_is_labelled_and_caveated_when_cdf_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(es.stats.nct, "cdf", lambda *args: math.nan)
    monkeypatch.setattr(es.stats.nct, "sf", lambda *args: math.nan)
    lower, upper, note = es._noncentrality_interval(5.0, 10, 0.95)
    assert "normal approximation" in note and "small df" in note
    assert lower < 5.0 < upper
    _, _, large_df_note = es._noncentrality_interval(5.0, 500, 0.95)
    assert "small df" not in large_df_note
