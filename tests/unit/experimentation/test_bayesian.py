"""Bayesian A/B analysis: conjugacy, closed forms, quadrature vs Monte Carlo, decisions."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats
from scipy.special import betaln

from decision_intelligence.experimentation import bayesian as ba


def _miller(post: ba.BetaBinomialPosterior) -> float:
    """Closed form P(p_t > p_c) for integer treatment alpha (Miller 2015)."""
    at, bt = post.treatment_alpha, post.treatment_beta
    ac, bc = post.control_alpha, post.control_beta
    return sum(math.exp(betaln(ac + i, bc + bt) - math.log(bt + i) - betaln(1 + i, bt)
                        - betaln(ac, bc)) for i in range(int(at)))


def test_conjugate_update_and_priors() -> None:
    prior = ba.BetaPrior.from_mean(0.1, 200)
    assert (prior.alpha, prior.beta) == pytest.approx((20.0, 180.0))
    post = ba.beta_binomial_posterior(30, 100, 20, 100, prior_treatment=prior,
                                      prior_control=prior)
    assert (post.treatment_alpha, post.treatment_beta) == (50.0, 250.0)
    assert (post.control_alpha, post.control_beta) == (40.0, 260.0)
    assert post.mean == pytest.approx(50 / 300 - 40 / 300)
    with pytest.raises(ValueError):
        ba.BetaPrior(0.0, 1.0)
    with pytest.raises(ValueError):
        ba.beta_binomial_posterior(5, 3, 1, 3)


@pytest.mark.parametrize("data", [(12, 100, 8, 100), (1240, 10_000, 1180, 10_000),
                                  (3, 40, 9, 40), (15_200, 100_000, 15_000, 100_000)])
def test_probability_better_matches_closed_form(data: tuple[int, int, int, int]) -> None:
    post = ba.beta_binomial_posterior(*data)
    assert ba.probability_above(post, 0.0) == pytest.approx(_miller(post), abs=1e-8)


def test_beta_summaries_match_monte_carlo() -> None:
    post = ba.beta_binomial_posterior(1240, 10_000, 1180, 10_000)
    rng = np.random.default_rng(0)
    draws = (rng.beta(post.treatment_alpha, post.treatment_beta, 1_000_000)
             - rng.beta(post.control_alpha, post.control_beta, 1_000_000))
    lo, hi = ba.credible_interval(post)
    q_lo, q_hi = np.quantile(draws, [0.025, 0.975])
    assert lo == pytest.approx(q_lo, abs=1e-4) and hi == pytest.approx(q_hi, abs=1e-4)
    assert ba.probability_above(post, 0.01) == pytest.approx((draws > 0.01).mean(), abs=2e-3)
    assert post.shortfall(0.0) == pytest.approx(np.maximum(-draws, 0).mean(), rel=0.02)
    assert post.excess(0.005) == pytest.approx(np.maximum(draws - 0.005, 0).mean(), rel=0.02)


def test_excess_minus_shortfall_identity() -> None:
    post = ba.beta_binomial_posterior(40, 300, 30, 300)
    for t in (-0.05, 0.0, 0.02, 0.1):
        assert post.excess(t) - post.shortfall(t) == pytest.approx(post.mean - t, abs=1e-9)


def test_normal_posterior_matches_grid_integration() -> None:
    prior = ba.NormalPrior(0.2, 0.5)
    post = ba.normal_effect_posterior(1.1, 0.4, prior)
    grid = np.linspace(-4, 6, 200_001)
    density = stats.norm.pdf(grid, 0.2, 0.5) * stats.norm.pdf(1.1, grid, 0.4)
    density /= np.trapezoid(density, grid)
    mean = np.trapezoid(grid * density, grid)
    sd = math.sqrt(np.trapezoid((grid - mean) ** 2 * density, grid))
    assert post.mean == pytest.approx(mean, abs=1e-8) and post.sd == pytest.approx(sd, abs=1e-8)
    assert post.shrinkage == pytest.approx(0.16 / 0.41)
    expected_shortfall = np.trapezoid(np.maximum(0.5 - grid, 0) * density, grid)
    assert post.shortfall(0.5) == pytest.approx(expected_shortfall, abs=1e-7)


def test_flat_prior_numerically_matches_frequentist_but_is_a_different_statement() -> None:
    estimate, se = 0.57, 0.22
    post = ba.normal_effect_posterior(estimate, se, ba.NormalPrior(0.0, 1e6))
    lo, hi = ba.credible_interval(post)
    assert (lo, hi) == pytest.approx((estimate - 1.959964 * se, estimate + 1.959964 * se),
                                     abs=1e-5)
    one_sided_p = stats.norm.sf(estimate / se)
    assert 1 - ba.probability_above(post, 0.0) == pytest.approx(one_sided_p, rel=1e-6)
    sceptical = ba.normal_effect_posterior(estimate, se, ba.NormalPrior(0.0, 0.3))
    assert 1 - ba.probability_above(sceptical, 0.0) > 2 * one_sided_p  # not a p-value


def test_expected_loss_decision_branches() -> None:
    kw = {"units_per_year": 1_000_000, "value_per_unit_effect": 0.3, "annual_cost": 100_000,
          "loss_tolerance": 10_000}
    break_even = 100_000 / 300_000
    launch = ba.expected_loss_decision(ba.normal_effect_posterior(1.0, 0.1, ba.NormalPrior(
        0, 100)), **kw)
    stop = ba.expected_loss_decision(ba.normal_effect_posterior(0.0, 0.05, ba.NormalPrior(
        0, 100)), **kw)
    hold = ba.expected_loss_decision(ba.normal_effect_posterior(0.4, 0.3, ba.NormalPrior(
        0, 100)), **kw)
    assert launch.decision == "LAUNCH" and stop.decision == "STOP" and hold.decision == "HOLD"
    assert launch.break_even_effect == pytest.approx(break_even)
    post = ba.normal_effect_posterior(0.4, 0.3, ba.NormalPrior(0, 100))
    assert hold.expected_annual_value == pytest.approx(300_000 * post.mean - 100_000)
    assert hold.expected_loss_launch == pytest.approx(300_000 * post.shortfall(break_even))
    assert hold.probability_value_positive == pytest.approx(
        ba.probability_above(post, break_even))
    with pytest.raises(ValueError):
        ba.expected_loss_decision(post, units_per_year=0, value_per_unit_effect=1,
                                  annual_cost=0, loss_tolerance=1)


def test_normal_model_refuses_non_finite_inputs() -> None:
    # Review fix: a NaN estimate produced a NaN posterior and a silent HOLD decision.
    with pytest.raises(ValueError, match="finite"):
        ba.normal_effect_posterior(float("nan"), 1.0, ba.NormalPrior(0.0, 1.0))
    with pytest.raises(ValueError, match="finite"):
        ba.NormalPrior(float("inf"), 1.0)
    with pytest.raises(ValueError, match="finite"):
        ba.NormalPrior(0.0, float("inf"))
