"""Pre-specified experiment design: units, estimand, metrics and decision thresholds.

An experiment's analysis is only as credible as what was fixed *before*
outcomes were seen. :class:`ExperimentDesign` records it explicitly and
refuses inconsistent specifications — most importantly an analysis unit
finer than the randomisation unit without a clustered analysis, which
understates uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

MetricKind = Literal["binary", "continuous"]
Direction = Literal["increase", "decrease"]


@dataclass(frozen=True)
class MetricSpec:
    """A metric's definition and, for guardrails, its tolerance.

    ``direction`` is the *desirable* direction. For a guardrail,
    ``margin`` is the largest tolerated deterioration in absolute units
    (non-inferiority margin): latency may rise by at most ``margin`` ms.
    """

    name: str
    kind: MetricKind
    definition: str
    direction: Direction = "increase"
    margin: float | None = None

    def __post_init__(self) -> None:
        if self.margin is not None and self.margin < 0:
            raise ValueError("a guardrail margin is a tolerated deterioration and must be >= 0")


@dataclass(frozen=True)
class ExperimentDesign:
    """Everything an analysis plan must fix before the data are seen."""

    name: str
    hypothesis: str
    population: str
    experimental_unit: str
    """The entity the treatment is applied to (e.g. a customer)."""
    randomisation_unit: str
    """The entity assignment is drawn for (equal to, or coarser than, the experimental unit)."""
    analysis_unit: str
    estimand: str
    primary: MetricSpec
    minimum_worthwhile_effect: float
    """Smallest primary effect (absolute) worth acting on — the business threshold."""
    secondary: tuple[MetricSpec, ...] = ()
    guardrails: tuple[MetricSpec, ...] = ()
    alpha: float = 0.05
    power: float = 0.8
    treatment_share: float = 0.5
    srm_alpha: float = 0.001
    clustered_analysis: bool = False
    exclusion_rules: tuple[str, ...] = ()
    """Pre-treatment exclusion rules only; post-treatment exclusions are not allowed."""
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not 0 < self.alpha < 1 or not 0 < self.power < 1 or not 0 < self.treatment_share < 1:
            raise ValueError("alpha, power and treatment_share must lie in (0, 1)")
        if self.minimum_worthwhile_effect <= 0:
            raise ValueError("minimum_worthwhile_effect must be positive")
        if self.randomisation_unit != self.analysis_unit and not self.clustered_analysis:
            raise ValueError(
                "the analysis unit differs from the randomisation unit: outcomes of units "
                "randomised together are correlated, so a clustered analysis is required "
                "(set clustered_analysis=True and analyse with cluster-robust inference)"
            )
        for guardrail in self.guardrails:
            if guardrail.margin is None:
                raise ValueError(f"guardrail {guardrail.name!r} needs a non-inferiority margin")

    def summary_rows(self) -> list[tuple[str, str]]:
        rows = [
            ("Hypothesis", self.hypothesis),
            ("Population", self.population),
            ("Experimental unit", self.experimental_unit),
            ("Randomisation unit", self.randomisation_unit),
            ("Analysis unit", self.analysis_unit),
            ("Estimand", self.estimand),
            ("Primary metric", f"{self.primary.name}: {self.primary.definition}"),
            ("Minimum worthwhile effect", f"{self.minimum_worthwhile_effect:g} (absolute)"),
            ("Secondary metrics", "; ".join(m.name for m in self.secondary) or "none"),
            (
                "Guardrails (margin)",
                "; ".join(f"{g.name} ({g.margin:g})" for g in self.guardrails) or "none",
            ),
            ("Alpha / power / treatment share",
             f"{self.alpha:g} / {self.power:g} / {self.treatment_share:g}"),
            ("SRM threshold", f"p < {self.srm_alpha:g}"),
            ("Exclusion rules", "; ".join(self.exclusion_rules) or "none"),
        ]
        return rows


__all__ = ["Direction", "ExperimentDesign", "MetricKind", "MetricSpec"]
