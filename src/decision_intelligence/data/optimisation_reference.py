"""A small, brute-force-verified budget-allocation reference problem.

Deliberately standalone from the customer universe: Pillar 4 (M12/M13)
needs a fixture whose optimal objective value is known independently of
any solver, so it can assert that Gurobi — and the HiGHS/SciPy fallback —
actually reach the proven optimum rather than merely "a" feasible solution.
Eight binary campaign-selection decisions keep exhaustive enumeration
(2**8 = 256 subsets) trivial to run inside the generator itself.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from decision_intelligence.data.ground_truth import OptimisationReferenceGroundTruth

N_CAMPAIGNS = 8
_BUDGET_FRACTION_OF_TOTAL_COST = 0.55


def _brute_force_optimum(
    names: list[str], costs: np.ndarray, values: np.ndarray, budget: float
) -> tuple[list[str], float]:
    best_value = 0.0
    best_selection: list[str] = []
    n = len(names)
    for size in range(n + 1):
        for subset in combinations(range(n), size):
            subset_cost = costs[list(subset)].sum()
            if subset_cost > budget:
                continue
            subset_value = float(values[list(subset)].sum())
            if subset_value > best_value:
                best_value = subset_value
                best_selection = [names[i] for i in subset]
    return best_selection, float(best_value)


def generate_optimisation_reference(
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, OptimisationReferenceGroundTruth]:
    """Generate the campaign table and its brute-force-verified optimum."""
    names = [f"ref_campaign_{i + 1}" for i in range(N_CAMPAIGNS)]
    costs = np.round(rng.uniform(1_000, 10_000, size=N_CAMPAIGNS), 2)
    # Value per unit cost varies so the optimum is not simply "cheapest first"
    # or "priciest first" — a real trade-off exists.
    value_per_cost = rng.uniform(0.8, 2.2, size=N_CAMPAIGNS)
    values = np.round(costs * value_per_cost, 2)
    budget = round(float(costs.sum()) * _BUDGET_FRACTION_OF_TOTAL_COST, 2)

    optimal_selection, optimal_value = _brute_force_optimum(names, costs, values, budget)

    table = pd.DataFrame({"campaign_name": names, "cost": costs, "value": values})
    ground_truth = OptimisationReferenceGroundTruth(
        campaign_names=names,
        costs=costs.tolist(),
        values=values.tolist(),
        budget=budget,
        optimal_selection=optimal_selection,
        optimal_objective_value=optimal_value,
    )
    return table, ground_truth
