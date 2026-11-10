"""Potential outcomes and causal estimands.

For a binary treatment ``T`` each unit has two potential outcomes, ``Y(1)``
and ``Y(0)``; only ``Y(T)`` is ever observed (the fundamental problem of
causal inference). Estimands are defined on the potential outcomes, not on
the data:

* ATE  = ``E[Y(1) - Y(0)]`` — effect of treating everyone vs no one;
* ATT  = ``E[Y(1) - Y(0) | T = 1]`` — effect among those actually treated;
* ATC  = ``E[Y(1) - Y(0) | T = 0]`` — effect among the untreated;
* CATE = ``E[Y(1) - Y(0) | X = x]`` — effect within a covariate stratum.

The naive (associational) contrast decomposes *exactly* as

    E[Y | T=1] - E[Y | T=0] = ATT + ( E[Y(0) | T=1] - E[Y(0) | T=0] )
                                     ^^^^^^^^^^^^^ selection bias ^^^^

so it equals a causal effect only when untreated outcomes would have been
the same in both groups — guaranteed by randomisation, not otherwise.

:class:`PotentialOutcomes` holds *both* potential outcomes for every unit.
That is only possible in simulation; it is how this package checks
estimators against the truth. Nothing in an estimator may read it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from decision_intelligence.common.validation import FloatArray, as_float_array


@dataclass(frozen=True)
class NaiveDecomposition:
    """``naive = att + selection_bias`` (an algebraic identity in any sample)."""

    naive_difference: float
    att: float
    selection_bias: float


class PotentialOutcomes:
    """Both potential outcomes and the realised treatment for a simulated population.

    ``y0``/``y1`` may be realised outcomes or *expected* outcomes (e.g. each
    unit's retention probability with and without treatment); estimands
    computed from expected outcomes have no outcome noise.
    """

    def __init__(self, y0: ArrayLike, y1: ArrayLike, treatment: ArrayLike) -> None:
        self.y0 = as_float_array(y0, name="y0")
        self.y1 = as_float_array(y1, name="y1")
        self.treatment = as_float_array(treatment, name="treatment")
        if not (self.y0.shape == self.y1.shape == self.treatment.shape):
            raise ValueError("y0, y1 and treatment must have the same length")
        if not np.isin(self.treatment, (0.0, 1.0)).all():
            raise ValueError("treatment must be binary")
        if float(np.min(self.treatment)) == float(np.max(self.treatment)):
            raise ValueError("both treated and untreated units are required")

    @property
    def observed(self) -> FloatArray:
        """``Y = T Y(1) + (1 - T) Y(0)`` (consistency)."""
        return np.asarray(
            self.treatment * self.y1 + (1 - self.treatment) * self.y0, dtype=np.float64
        )

    @property
    def individual_effects(self) -> FloatArray:
        return np.asarray(self.y1 - self.y0, dtype=np.float64)

    def ate(self) -> float:
        return float(np.mean(self.individual_effects))

    def att(self) -> float:
        return float(np.mean(self.individual_effects[self.treatment == 1]))

    def atc(self) -> float:
        return float(np.mean(self.individual_effects[self.treatment == 0]))

    def cate(self, groups: ArrayLike) -> dict[str, float]:
        """Average effect within each level of ``groups`` (a CATE for a discrete X)."""
        labels = np.asarray(groups).astype(str)
        if labels.shape != self.y0.shape:
            raise ValueError("groups must align with the outcomes")
        effects = self.individual_effects
        return {label: float(effects[labels == label].mean()) for label in np.unique(labels)}

    def naive_decomposition(self) -> NaiveDecomposition:
        treated = self.treatment == 1
        observed = self.observed
        naive = float(observed[treated].mean() - observed[~treated].mean())
        selection_bias = float(self.y0[treated].mean() - self.y0[~treated].mean())
        return NaiveDecomposition(
            naive_difference=naive, att=self.att(), selection_bias=selection_bias
        )
