"""Unit tests for confounded marketing-campaign assignment."""

from __future__ import annotations

from decision_intelligence.common import seeded_generator
from decision_intelligence.data.customers import generate_customers
from decision_intelligence.data.marketing import (
    CAMPAIGN_ID,
    generate_customer_treatments,
    generate_marketing_campaigns,
)


def test_campaign_table_has_one_row(small_config) -> None:
    campaigns = generate_marketing_campaigns(small_config)
    assert len(campaigns) == 1
    assert campaigns.loc[0, "campaign_id"] == CAMPAIGN_ID


def test_assignment_propensity_is_a_probability(small_config) -> None:
    customers = generate_customers(small_config, seeded_generator(1))
    treatments = generate_customer_treatments(small_config, customers, seeded_generator(2))
    assert treatments["assignment_propensity"].between(0.0, 1.0).all()
    assert set(treatments["customer_id"]) == set(customers["customer_id"])


def test_treatment_is_confounded_on_latent_value(small_config) -> None:
    """Treated customers must have systematically higher latent_value — the
    whole point of this table being a confounded, non-randomised design."""
    customers = generate_customers(small_config, seeded_generator(1))
    treatments = generate_customer_treatments(small_config, customers, seeded_generator(2))
    merged = customers.merge(treatments, on="customer_id")
    treated_mean = merged.loc[merged["treated"], "latent_value"].mean()
    untreated_mean = merged.loc[~merged["treated"], "latent_value"].mean()
    assert treated_mean > untreated_mean
