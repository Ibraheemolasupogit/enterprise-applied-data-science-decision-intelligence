"""Unit tests for the brute-force-verified optimisation reference problem.

The optimum is re-derived here with an independent enumeration, rather
than by calling the generator's own brute-force helper, so this test can
actually catch a bug in that helper rather than merely echoing it.
"""

from __future__ import annotations

from itertools import combinations

import pytest

from decision_intelligence.common import seeded_generator
from decision_intelligence.data.optimisation_reference import (
    N_CAMPAIGNS,
    generate_optimisation_reference,
)


def _independent_best_value(costs: list[float], values: list[float], budget: float) -> float:
    best = 0.0
    for size in range(len(costs) + 1):
        for subset in combinations(range(len(costs)), size):
            if sum(costs[i] for i in subset) <= budget:
                best = max(best, sum(values[i] for i in subset))
    return best


def test_table_and_ground_truth_are_consistent() -> None:
    table, ground_truth = generate_optimisation_reference(seeded_generator(1))
    assert len(table) == N_CAMPAIGNS
    assert ground_truth.budget > 0
    assert ground_truth.optimal_objective_value > 0


def test_optimal_selection_respects_the_budget() -> None:
    _, ground_truth = generate_optimisation_reference(seeded_generator(1))
    cost_by_name = dict(zip(ground_truth.campaign_names, ground_truth.costs, strict=False))
    selected_cost = sum(cost_by_name[name] for name in ground_truth.optimal_selection)
    assert selected_cost <= ground_truth.budget


def test_optimal_value_matches_independent_enumeration() -> None:
    _, ground_truth = generate_optimisation_reference(seeded_generator(1))
    independent_best = _independent_best_value(
        ground_truth.costs, ground_truth.values, ground_truth.budget
    )
    assert ground_truth.optimal_objective_value == pytest.approx(independent_best)


def test_same_seed_is_deterministic() -> None:
    _, first = generate_optimisation_reference(seeded_generator(42))
    _, second = generate_optimisation_reference(seeded_generator(42))
    assert first == second
