"""Ground truth deliberately encoded into the synthetic data.

Every value here is a *design-time* fact about how the data generator works
("the true effect of this treatment is 0.08"), not an estimate. Later
pillars validate their estimators by checking how closely they recover
these known values — see docs/methodology/synthetic-data.md Section
"Ground-truth design".
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentGroundTruth:
    """Ground truth for the unconfounded, randomised recommendation experiment."""

    experiment_name: str
    assignment_probability: float
    true_absolute_effect: float
    primary_metric: str = "converted"


@dataclass(frozen=True)
class MarketingGroundTruth:
    """Ground truth for the confounded observational retention-offer campaign.

    Assignment is correlated with ``latent_value`` (see customers.py), which
    also independently affects the outcome. A naive treated-vs-control
    comparison is therefore expected to be biased away from
    ``true_absolute_effect`` — recovering the true effect requires the
    adjustment methods introduced in Pillar 1 (M4/M5).
    """

    campaign_name: str
    true_absolute_effect: float
    confounding_covariate: str
    primary_metric: str = "retained_90d"


@dataclass(frozen=True)
class RetentionGroundTruth:
    """Per-segment Weibull hazard parameters used to draw time-to-churn.

    Hazard rate at time t for shape k and scale lambda follows the standard
    Weibull hazard form; customers who receive the true retention-campaign
    effect have their hazard scaled down by ``(1 - marketing_true_effect)``
    multiplicatively on top of their segment baseline.
    """

    segment_shape: dict[str, float]
    segment_scale: dict[str, float]


@dataclass(frozen=True)
class OptimisationReferenceGroundTruth:
    """A small, brute-force-verified budget-allocation reference problem.

    Bounded to few enough campaigns that the optimal selection is verified
    by exhaustive enumeration in the generator itself (see
    optimisation_reference.py), independent of any solver — this is the
    known-optimum fixture M12/M13 will check Gurobi and the HiGHS/SciPy
    fallback against.
    """

    campaign_names: list[str]
    costs: list[float]
    values: list[float]
    budget: float
    optimal_selection: list[str]
    optimal_objective_value: float


@dataclass(frozen=True)
class GroundTruth:
    """Bundle of every known effect, probability and optimum in one build."""

    experiment: ExperimentGroundTruth
    marketing: MarketingGroundTruth
    retention: RetentionGroundTruth
    optimisation_reference: OptimisationReferenceGroundTruth
