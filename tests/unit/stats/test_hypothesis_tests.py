"""Hypothesis tests versus SciPy/statsmodels reference implementations, plus edge cases."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.proportion import confint_proportions_2indep, proportion_confint

from decision_intelligence.common import seeded_generator
from decision_intelligence.statistics import hypothesis_tests as ht

ALTERNATIVES = ("two-sided", "greater", "less")


@pytest.fixture
def samples() -> tuple[np.ndarray, np.ndarray]:
    rng = seeded_generator(21)
    return rng.normal(0.0, 1.0, 30), rng.normal(0.6, 2.0, 24)


# --- t-tests ---------------------------------------------------------------


@pytest.mark.parametrize("alternative", ALTERNATIVES)
@pytest.mark.parametrize("equal_var", [False, True])
def test_two_sample_t_matches_scipy(
    samples: tuple[np.ndarray, np.ndarray], alternative: str, equal_var: bool
) -> None:
    x, y = samples
    ours = ht.two_sample_t_test(x, y, equal_var=equal_var, alternative=alternative)
    reference = stats.ttest_ind(x, y, equal_var=equal_var, alternative=alternative)
    interval = reference.confidence_interval()
    assert ours.statistic == pytest.approx(reference.statistic)
    assert ours.p_value == pytest.approx(reference.pvalue)
    assert ours.df == pytest.approx(reference.df)
    assert ours.ci_lower == pytest.approx(interval.low)
    assert ours.ci_upper == pytest.approx(interval.high)


@pytest.mark.parametrize("alternative", ALTERNATIVES)
def test_one_sample_and_paired_t_match_scipy(
    samples: tuple[np.ndarray, np.ndarray], alternative: str
) -> None:
    x, y = samples
    one = ht.one_sample_t_test(x, popmean=0.3, alternative=alternative)
    assert one.p_value == pytest.approx(stats.ttest_1samp(x, 0.3, alternative=alternative).pvalue)
    paired = ht.paired_t_test(x[:24], y, alternative=alternative)
    reference = stats.ttest_rel(x[:24], y, alternative=alternative)
    assert paired.p_value == pytest.approx(reference.pvalue)
    assert paired.ci_lower == pytest.approx(reference.confidence_interval().low)


def test_t_tests_reject_degenerate_input() -> None:
    with pytest.raises(ValueError, match="constant"):
        ht.one_sample_t_test([2.0, 2.0, 2.0])
    with pytest.raises(ValueError, match="both samples are constant"):
        ht.two_sample_t_test([1.0, 1.0], [2.0, 2.0])
    with pytest.raises(ValueError, match="same length"):
        ht.paired_t_test([1.0, 2.0, 3.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="at least 2"):
        ht.two_sample_t_test([1.0], [1.0, 2.0])


def test_welch_handles_one_constant_group() -> None:
    result = ht.two_sample_t_test([5.0, 5.0, 5.0], [1.0, 2.0, 3.0, 4.0])
    assert result.p_value < 0.05


def test_results_carry_assumptions(samples: tuple[np.ndarray, np.ndarray]) -> None:
    result = ht.two_sample_t_test(*samples)
    assert result.method.startswith("Welch")
    assert any("independent" in a for a in result.assumptions)


# --- proportions and tables -----------------------------------------------


@pytest.mark.parametrize(("successes", "n"), [(0, 20), (7, 40), (40, 40), (1, 1000)])
def test_wilson_interval_matches_statsmodels(successes: int, n: int) -> None:
    ours = ht.wilson_interval(successes, n)
    reference = proportion_confint(successes, n, method="wilson")
    assert ours == pytest.approx(reference)


def test_two_proportion_test_matches_pooled_z_and_newcombe() -> None:
    result = ht.two_proportion_z_test(30, 100, 18, 110)
    pooled = 48 / 210
    z = (0.30 - 18 / 110) / math.sqrt(pooled * (1 - pooled) * (1 / 100 + 1 / 110))
    assert result.statistic == pytest.approx(z)
    assert result.p_value == pytest.approx(2 * stats.norm.sf(abs(z)))
    low, high = confint_proportions_2indep(30, 100, 18, 110, method="newcomb")
    assert (result.ci_lower, result.ci_upper) == pytest.approx((low, high))


def test_two_proportion_test_edge_cases() -> None:
    degenerate = ht.two_proportion_z_test(0, 10, 0, 12)
    assert degenerate.p_value == 1.0 and degenerate.warnings
    sparse = ht.two_proportion_z_test(1, 10, 3, 12)
    assert any("Fisher" in w or "fisher" in w for w in sparse.warnings)
    one_sided = ht.two_proportion_z_test(30, 100, 18, 110, alternative="greater")
    assert one_sided.ci_upper == math.inf
    with pytest.raises(ValueError, match="cannot exceed"):
        ht.two_proportion_z_test(11, 10, 3, 12)


def test_chi_square_matches_scipy_and_warns_on_sparse_tables() -> None:
    table = [[12, 5, 9], [7, 14, 3]]
    ours = ht.chi_square_independence(table)
    reference = stats.chi2_contingency(table, correction=False)
    assert ours.statistic == pytest.approx(reference.statistic)
    assert ours.p_value == pytest.approx(reference.pvalue)
    np.testing.assert_allclose(ht.expected_frequencies(table), reference.expected_freq)
    assert 0 <= ours.estimate <= 1
    assert ht.chi_square_independence([[1, 2], [3, 0]]).warnings
    with pytest.raises(ValueError, match="all-zero"):
        ht.chi_square_independence([[0, 0], [1, 2]])
    with pytest.raises(ValueError, match="integer"):
        ht.chi_square_independence([[1.5, 2], [3, 4]])


def test_fisher_exact_wraps_scipy() -> None:
    result = ht.fisher_exact_2x2([[8, 2], [1, 5]])
    assert result.p_value == pytest.approx(stats.fisher_exact([[8, 2], [1, 5]]).pvalue)
    with pytest.raises(ValueError, match="2x2"):
        ht.fisher_exact_2x2([[1, 2, 3], [4, 5, 6]])


# --- rank tests ------------------------------------------------------------


@pytest.mark.parametrize("alternative", ALTERNATIVES)
def test_mann_whitney_exact_matches_scipy(
    samples: tuple[np.ndarray, np.ndarray], alternative: str
) -> None:
    x, y = samples
    ours = ht.mann_whitney_u(x, y, alternative=alternative)
    reference = stats.mannwhitneyu(x, y, alternative=alternative, method="exact")
    assert "exact" in ours.method
    assert ours.statistic == pytest.approx(reference.statistic)
    assert ours.p_value == pytest.approx(reference.pvalue, abs=1e-12)


@pytest.mark.parametrize("alternative", ALTERNATIVES)
def test_mann_whitney_asymptotic_with_ties_matches_scipy(
    samples: tuple[np.ndarray, np.ndarray], alternative: str
) -> None:
    x, y = (np.round(s) for s in samples)
    ours = ht.mann_whitney_u(x, y, alternative=alternative)
    reference = stats.mannwhitneyu(x, y, alternative=alternative, method="asymptotic")
    assert "normal approximation" in ours.method and ours.warnings
    assert ours.p_value == pytest.approx(reference.pvalue)


def test_mann_whitney_exact_requires_no_ties_and_all_tied_raises() -> None:
    with pytest.raises(ValueError, match="exact"):
        ht.mann_whitney_u([1.0, 1.0, 2.0], [3.0, 4.0], method="exact")
    with pytest.raises(ValueError, match="all observations are tied"):
        ht.mann_whitney_u([1.0, 1.0], [1.0, 1.0])


def test_mann_whitney_estimand_is_probability_of_superiority() -> None:
    result = ht.mann_whitney_u([10.0, 11.0, 12.0], [1.0, 2.0])
    assert result.estimate == 1.0


@pytest.mark.parametrize("alternative", ALTERNATIVES)
def test_wilcoxon_signed_rank_matches_scipy(alternative: str) -> None:
    differences = seeded_generator(22).normal(0.4, 1.0, 20)
    exact = ht.wilcoxon_signed_rank(differences, alternative=alternative)
    reference = stats.wilcoxon(differences, alternative=alternative, method="exact")
    assert exact.p_value == pytest.approx(reference.pvalue, abs=1e-12)
    tied = np.round(differences, 1)
    approx = ht.wilcoxon_signed_rank(tied, alternative=alternative, method="asymptotic")
    reference = stats.wilcoxon(tied, alternative=alternative, method="approx")
    assert approx.p_value == pytest.approx(reference.pvalue)


def test_wilcoxon_drops_zero_differences_and_rejects_all_zero() -> None:
    result = ht.wilcoxon_signed_rank([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0])
    assert any("zero difference" in w for w in result.warnings)
    with pytest.raises(ValueError, match="all differences are zero"):
        ht.wilcoxon_signed_rank([1.0, 2.0], [1.0, 2.0])


def test_exact_null_distributions_are_symmetric_and_sum_correctly() -> None:
    counts = ht._mann_whitney_exact_counts(6, 9)
    assert sum(counts) == math.comb(15, 6)
    assert counts == counts[::-1]
    signed = ht._signed_rank_exact_counts(10)
    assert sum(signed) == 2**10
    assert signed == signed[::-1]


def test_exact_counts_stay_exact_for_large_groups() -> None:
    """C(100, 50) ~ 1e29 overflows float64 integer precision; Python ints keep it exact."""
    assert sum(ht._mann_whitney_exact_counts(50, 50)) == math.comb(100, 50)


# --- permutation and k-sample tests ---------------------------------------


@pytest.mark.parametrize("alternative", ALTERNATIVES)
def test_exact_permutation_test_matches_scipy(
    samples: tuple[np.ndarray, np.ndarray], alternative: str
) -> None:
    x, y = samples[0][:7], samples[1][:6]
    ours = ht.permutation_test(x, y, alternative=alternative, rng=seeded_generator(1))
    reference = stats.permutation_test(
        (x, y),
        lambda a, b, axis=-1: a.mean(axis) - b.mean(axis),
        permutation_type="independent",
        alternative=alternative,
        n_resamples=np.inf,
    )
    assert "exact" in ours.method
    assert ours.p_value == pytest.approx(reference.pvalue)


def test_monte_carlo_permutation_p_value_is_never_zero() -> None:
    x = np.arange(30.0) + 100
    y = np.arange(30.0)
    result = ht.permutation_test(x, y, n_permutations=999, rng=seeded_generator(2))
    assert "Monte Carlo" in result.method
    assert result.p_value == pytest.approx(2 / 1000)


def test_k_sample_tests_match_scipy() -> None:
    rng = seeded_generator(23)
    groups = [rng.normal(loc, 1.0 + loc, 12) for loc in (0.0, 0.5, 1.0)]
    assert ht.one_way_anova(*groups).p_value == pytest.approx(stats.f_oneway(*groups).pvalue)
    assert ht.kruskal_wallis(*groups).p_value == pytest.approx(stats.kruskal(*groups).pvalue)
    assert ht.brown_forsythe_levene(*groups).p_value == pytest.approx(
        stats.levene(*groups, center="median").pvalue
    )
    with pytest.raises(ValueError, match="at least two groups"):
        ht.one_way_anova(groups[0])
    with pytest.raises(ValueError, match="zero within-group"):
        ht.one_way_anova([1.0, 1.0], [2.0, 2.0])


def test_shapiro_wilk_wrapper_and_limits() -> None:
    data = seeded_generator(24).normal(size=50)
    assert ht.shapiro_wilk(data).p_value == pytest.approx(stats.shapiro(data).pvalue)
    with pytest.raises(ValueError, match="constant"):
        ht.shapiro_wilk([1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="5000"):
        ht.shapiro_wilk(np.arange(5001.0))
