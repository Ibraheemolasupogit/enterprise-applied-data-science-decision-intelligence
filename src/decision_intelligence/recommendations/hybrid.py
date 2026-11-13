"""Score-blend hybrid recommender (Milestone 9).

Each component's scores are converted, per user, to percentile ranks over the user's allowed
items (launched by the cutoff, not clicked before it): ``rank / n_allowed`` in (0, 1], with
average ranks for ties. The hybrid score is a convex combination of the components' percentile
ranks. Rank-normalising makes components with different score scales comparable; the weights
are chosen on the validation window from a simplex grid (protocol ``BLEND_STEP``).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence

import numpy as np
from scipy import stats

from decision_intelligence.common.validation import FloatArray
from decision_intelligence.recommendations.baselines import Recommender
from decision_intelligence.recommendations.candidates import allowed_mask
from decision_intelligence.recommendations.interactions import IntArray, InteractionData


def percentile_ranks(scores: FloatArray, allowed: np.ndarray) -> FloatArray:
    """Per row: average rank of each allowed score / number allowed; 0 for disallowed."""
    out = np.zeros_like(scores, dtype=np.float64)
    for r in range(scores.shape[0]):
        a = allowed[r]
        n = int(a.sum())
        if n:
            out[r, a] = stats.rankdata(scores[r, a], method="average") / n
    return out


def simplex_grid(n_components: int, step: float) -> list[tuple[float, ...]]:
    """All weight vectors on the simplex with the given step (e.g. 0.25 -> multiples of 1/4)."""
    m = round(1.0 / step)
    if abs(m * step - 1.0) > 1e-9:
        raise ValueError("1 / step must be an integer")
    return [tuple(c / m for c in combo)
            for combo in itertools.product(range(m + 1), repeat=n_components)
            if sum(combo) == m]


class BlendRecommender(Recommender):
    def __init__(self, components: Mapping[str, Callable[[], Recommender]],
                 weights: Sequence[float]) -> None:
        super().__init__()
        if len(weights) != len(components) or abs(sum(weights) - 1.0) > 1e-9 or min(weights) < 0:
            raise ValueError("weights must be non-negative, sum to 1, one per component")
        self.factories = dict(components)
        self.weights = tuple(float(w) for w in weights)
        self.name = "blend (" + ", ".join(
            f"{n} {w:g}" for n, w in zip(components, self.weights, strict=True)) + ")"
        self.fitted: dict[str, Recommender] = {}

    def _fit(self, history: InteractionData) -> None:
        self.fitted = {n: f().fit(history) for n, f in self.factories.items()}

    def component_ranks(self, users: IntArray) -> list[FloatArray]:
        allowed = allowed_mask(self.history, users)
        return [percentile_ranks(m.score(users), allowed) for m in self.fitted.values()]

    def score(self, users: IntArray) -> FloatArray:
        ranks = self.component_ranks(users)
        return np.asarray(sum(w * r for w, r in zip(self.weights, ranks, strict=True)),
                          dtype=np.float64)
