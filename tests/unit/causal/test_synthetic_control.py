"""Synthetic control: optimiser correctness, diagnostics and placebo arithmetic."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from decision_intelligence.causal_inference import synthetic_control as sc
from decision_intelligence.data.synthetic_control_panel import (
    SyntheticControlConfig,
    generate_synthetic_control_panel,
)


def test_exact_convex_combination_is_recovered() -> None:
    rng = np.random.default_rng(1)
    donors = rng.normal(size=(40, 5))
    true_w = np.array([0.5, 0.0, 0.3, 0.2, 0.0])
    w = sc.simplex_least_squares(donors, donors @ true_w)
    np.testing.assert_allclose(w, true_w, atol=1e-6)


def test_matches_brute_force_grid_on_the_simplex() -> None:
    rng = np.random.default_rng(2)
    donors = rng.normal(size=(20, 3))
    target = rng.normal(size=20)
    w = sc.simplex_least_squares(donors, target)
    best = min(
        (np.array([a, b, 1 - a - b]) for a, b in itertools.product(np.linspace(0, 1, 201), repeat=2)
         if a + b <= 1 + 1e-12),
        key=lambda v: float(np.sum((donors @ v - target) ** 2)),
    )
    assert np.sum((donors @ w - target) ** 2) <= np.sum((donors @ best - target) ** 2) + 1e-9
    assert np.allclose(w, best, atol=0.01)


def test_optimality_certificate_rejects_non_optimal_weights() -> None:
    rng = np.random.default_rng(3)
    donors = rng.normal(size=(15, 4))
    target = donors @ np.array([0.7, 0.3, 0.0, 0.0]) + 0.1 * rng.normal(size=15)
    w = sc.simplex_least_squares(donors, target)
    assert sc.is_optimal(donors, target, w)
    assert not sc.is_optimal(donors, target, np.full(4, 0.25))
    with pytest.raises(ValueError):
        sc.simplex_least_squares(donors, target[:-1])


def test_certificate_detects_small_perturbations_on_the_canonical_design() -> None:
    """Regression test: the former KKT check accepted weights shifted by 1% between donors."""
    data = generate_synthetic_control_panel()
    wide = data.panel.pivot(index="period", columns="unit_id", values="outcome")
    pre = wide.index <= data.config.n_pre
    donors, target = wide.drop(columns=0).to_numpy()[pre], wide[0].to_numpy()[pre]
    w = sc.simplex_least_squares(donors, target)
    assert sc.is_optimal(donors, target, w)
    first, second = np.argsort(-w)[:2]
    for eps in (1e-2, 1e-4):
        shifted = w.copy()
        shifted[first] -= eps
        shifted[second] += eps
        assert not sc.is_optimal(donors, target, shifted), eps


def test_refinement_reaches_the_optimum_where_slsqp_stops_early() -> None:
    """Seed 5: SLSQP alone stops at a relative duality gap ~1e-4; the certified solution's
    objective is no worse than any other feasible point found."""
    from decision_intelligence.data.synthetic_control_panel import SyntheticControlConfig as Cfg

    data = generate_synthetic_control_panel(Cfg(seed=5))
    wide = data.panel.pivot(index="period", columns="unit_id", values="outcome")
    pre = wide.index <= data.config.n_pre
    donors, target = wide.drop(columns=0).to_numpy()[pre], wide[0].to_numpy()[pre]
    w = sc.simplex_least_squares(donors, target)
    assert sc.is_optimal(donors, target, w)
    scale = float(np.linalg.norm(target))
    assert sc.optimality_gap(donors / scale, target / scale, w) < 1e-9


def test_fit_on_valid_design_and_counterfactual_identity() -> None:
    data = generate_synthetic_control_panel()
    fit = sc.fit_synthetic_control(data.panel, 0, data.config.n_pre + 1)
    assert sum(fit.weights.values()) == pytest.approx(1.0)
    assert not fit.warnings and fit.relative_pre_fit < 0.5
    wide = data.panel.pivot(index="period", columns="unit_id", values="outcome")
    rebuilt = wide[list(fit.weights)].to_numpy() @ np.array(list(fit.weights.values()))
    np.testing.assert_allclose(rebuilt, fit.synthetic_path, atol=1e-6)
    assert fit.average_post_gap == pytest.approx(np.mean(list(fit.post_gaps.values())))


def test_outside_convex_hull_is_flagged() -> None:
    data = generate_synthetic_control_panel(SyntheticControlConfig(treated_in_convex_hull=False))
    fit = sc.fit_synthetic_control(data.panel, 0, data.config.n_pre + 1)
    assert fit.share_pre_outside_donor_range == 1.0
    assert any("poor pre-treatment fit" in w for w in fit.warnings)
    assert any("outside the donor range" in w for w in fit.warnings)


def test_placebo_arithmetic_and_exclusion() -> None:
    data = generate_synthetic_control_panel(SyntheticControlConfig(n_donors=8, n_pre=20))
    start = data.config.n_pre + 1
    placebo = sc.placebo_in_space(data.panel, 0, start)
    assert placebo.n_compared == 9 and placebo.minimum_attainable_p == pytest.approx(1 / 9)
    ratios = np.array([placebo.treated_ratio, *placebo.placebo_ratios.values()])
    assert placebo.rank == int(np.sum(ratios >= placebo.treated_ratio))
    assert placebo.permutation_p_value == pytest.approx(placebo.rank / 9)
    strict = sc.placebo_in_space(data.panel, 0, start, exclusion_multiple=1.0)
    assert strict.n_compared + len(strict.excluded_for_poor_fit) == 9


def test_placebo_exclusion_uses_mspe_multiples() -> None:
    """Regression test: exclusion follows Abadie et al. (2010) on MSPE, not RMSPE."""
    data = generate_synthetic_control_panel()
    start = data.config.n_pre + 1
    treated_mspe = sc.fit_synthetic_control(data.panel, 0, start).pre_rmspe ** 2
    donors = [u for u in data.panel["unit_id"].unique() if u != 0]
    placebo_mspe = {
        int(u): sc.fit_synthetic_control(
            data.panel, int(u), start, donors=[int(d) for d in donors if d != u]
        ).pre_rmspe ** 2
        for u in donors
    }
    result = sc.placebo_in_space(data.panel, 0, start, exclusion_multiple=2.0)
    expected = {u for u, m in placebo_mspe.items() if m > 2.0 * treated_mspe}
    assert set(result.excluded_for_poor_fit) == expected and expected


def test_in_time_placebo_and_leave_one_out() -> None:
    data = generate_synthetic_control_panel()
    start = data.config.n_pre + 1
    fake = sc.placebo_in_time(data.panel, 0, start, start - 10)
    assert max(fake.periods) == start - 1
    assert abs(fake.average_post_gap) < 2.0
    with pytest.raises(ValueError):
        sc.placebo_in_time(data.panel, 0, start, start)
    loo = sc.leave_one_out(data.panel, 0, start)
    assert set(loo) == set(sc.fit_synthetic_control(data.panel, 0, start).weights)


def test_input_validation() -> None:
    data = generate_synthetic_control_panel()
    with pytest.raises(ValueError, match="treated unit"):
        sc.fit_synthetic_control(data.panel, 999, 31)
    with pytest.raises(ValueError, match="balanced"):
        sc.fit_synthetic_control(data.panel.iloc[1:], 0, 31)
    with pytest.raises(ValueError, match="pre-treatment"):
        sc.fit_synthetic_control(data.panel, 0, 2)
    tiny = pd.DataFrame({"unit_id": [0, 0, 1, 1], "period": [1, 2, 1, 2], "outcome": [1.0] * 4})
    with pytest.raises(ValueError, match="two donors"):
        sc.fit_synthetic_control(tiny, 0, 2)
