"""Orchestrates one full, deterministic synthetic data build.

Every stage below draws from its own generator, spawned once from
``config.seed`` via :func:`decision_intelligence.common.spawn_generators` —
so inserting, removing or reordering an unrelated stage never changes
another stage's random draws. Rebuilding with an identical config
reproduces byte-identical tables; see tests/integration/test_pipeline.py.
"""

from __future__ import annotations

from decision_intelligence.common import spawn_generators
from decision_intelligence.data import customers as customers_module
from decision_intelligence.data import experiments as experiments_module
from decision_intelligence.data import interactions as interactions_module
from decision_intelligence.data import marketing as marketing_module
from decision_intelligence.data import optimisation_reference as optimisation_reference_module
from decision_intelligence.data import products as products_module
from decision_intelligence.data import quality as quality_module
from decision_intelligence.data import retention as retention_module
from decision_intelligence.data import subscriptions as subscriptions_module
from decision_intelligence.data import transactions as transactions_module
from decision_intelligence.data.config import SyntheticDataConfig
from decision_intelligence.data.dataset import SyntheticDataset
from decision_intelligence.data.ground_truth import GroundTruth, MarketingGroundTruth
from decision_intelligence.data.marketing import CONFOUNDING_COVARIATE

_STAGES = (
    "customers",
    "products",
    "marketing_treatments",
    "retention",
    "subscriptions",
    "transactions",
    "experiment",
    "latent_affinities",
    "interactions",
    "optimisation_reference",
    "quality",
)


def build_synthetic_dataset(config: SyntheticDataConfig | None = None) -> SyntheticDataset:
    """Build one full, deterministic synthetic NexaMarket dataset.

    Args:
        config: Build configuration. Defaults to
            :class:`SyntheticDataConfig`'s defaults (bounded for local/CI
            use) when omitted.
    """
    config = config or SyntheticDataConfig()
    rngs = dict(zip(_STAGES, spawn_generators(config.seed, len(_STAGES)), strict=False))

    customers = customers_module.generate_customers(config, rngs["customers"])
    products = products_module.generate_products(config, rngs["products"])

    marketing_campaigns = marketing_module.generate_marketing_campaigns(config)
    customer_treatments = marketing_module.generate_customer_treatments(
        config, customers, rngs["marketing_treatments"]
    )

    customer_retention, retention_ground_truth = retention_module.generate_customer_retention(
        config, customers, customer_treatments, rngs["retention"]
    )

    subscriptions = subscriptions_module.generate_subscriptions(
        config, customers, products, customer_retention, rngs["subscriptions"]
    )

    transactions = transactions_module.generate_transactions(
        config, customers, subscriptions, rngs["transactions"]
    )
    transactions = transactions_module.attach_margin(transactions, products)

    experiment_assignments, experiment_outcomes, experiment_ground_truth = (
        experiments_module.generate_experiment(config, customers, rngs["experiment"])
    )

    latent_affinities = interactions_module.generate_latent_affinities(
        config, customers, rngs["latent_affinities"]
    )
    sessions, recommendation_exposures, events = interactions_module.generate_interactions(
        config, customers, products, subscriptions, latent_affinities, rngs["interactions"]
    )

    optimisation_reference_campaigns, optimisation_ground_truth = (
        optimisation_reference_module.generate_optimisation_reference(
            rngs["optimisation_reference"]
        )
    )

    quality_rng = rngs["quality"]
    customers = quality_module.inject_missingness(
        customers, "household_income_proxy", config.quality.missing_income_rate, quality_rng
    )
    customers = quality_module.inject_missingness(
        customers, "region", config.quality.missing_region_rate, quality_rng
    )
    events = quality_module.inject_duplicate_events(
        events, config.quality.duplicate_event_rate, "event_id", quality_rng
    )
    events = quality_module.inject_late_arrivals(
        events,
        "event_timestamp",
        "recorded_at",
        config.quality.late_arrival_rate,
        config.quality.late_arrival_max_days,
        quality_rng,
    )

    tables = {
        "customers": customers,
        "products": products,
        "marketing_campaigns": marketing_campaigns,
        "customer_treatments": customer_treatments,
        "customer_retention": customer_retention,
        "subscriptions": subscriptions,
        "transactions": transactions,
        "experiment_assignments": experiment_assignments,
        "experiment_outcomes": experiment_outcomes,
        "sessions": sessions,
        "recommendation_exposures": recommendation_exposures,
        "events": events,
        "optimisation_reference_campaigns": optimisation_reference_campaigns,
    }

    marketing_ground_truth = MarketingGroundTruth(
        campaign_name=config.marketing_campaign_name,
        true_absolute_effect=config.marketing_true_effect,
        confounding_covariate=CONFOUNDING_COVARIATE,
    )
    ground_truth = GroundTruth(
        experiment=experiment_ground_truth,
        marketing=marketing_ground_truth,
        retention=retention_ground_truth,
        optimisation_reference=optimisation_ground_truth,
    )

    return SyntheticDataset(
        config=config,
        tables=tables,
        latent_affinities=latent_affinities,
        ground_truth=ground_truth,
    )
