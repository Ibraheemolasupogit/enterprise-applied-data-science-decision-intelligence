"""Unit tests for sessions, recommendation exposures and engagement events."""

from __future__ import annotations

from decision_intelligence.common import seeded_generator
from decision_intelligence.data.customers import generate_customers
from decision_intelligence.data.interactions import (
    generate_interactions,
    generate_latent_affinities,
)
from decision_intelligence.data.marketing import generate_customer_treatments
from decision_intelligence.data.products import CATEGORIES, generate_products
from decision_intelligence.data.retention import generate_customer_retention
from decision_intelligence.data.subscriptions import generate_subscriptions


def _build(small_config):
    customers = generate_customers(small_config, seeded_generator(1))
    products = generate_products(small_config, seeded_generator(2))
    treatments = generate_customer_treatments(small_config, customers, seeded_generator(3))
    retention, _ = generate_customer_retention(
        small_config, customers, treatments, seeded_generator(4)
    )
    subscriptions = generate_subscriptions(
        small_config, customers, products, retention, seeded_generator(5)
    )
    affinities = generate_latent_affinities(small_config, customers, seeded_generator(6))
    sessions, exposures, events = generate_interactions(
        small_config, customers, products, subscriptions, affinities, seeded_generator(7)
    )
    return customers, products, affinities, sessions, exposures, events


def test_latent_affinities_sum_to_one_per_customer(small_config) -> None:
    customers = generate_customers(small_config, seeded_generator(1))
    affinities = generate_latent_affinities(small_config, customers, seeded_generator(6))
    totals = affinities[list(CATEGORIES)].sum(axis=1)
    assert (totals - 1.0).abs().max() < 1e-9


def test_exposures_reference_valid_sessions_and_products(small_config) -> None:
    _, products, _, sessions, exposures, _ = _build(small_config)
    assert set(exposures["session_id"]) <= set(sessions["session_id"])
    assert set(exposures["product_id"]) <= set(products["product_id"])
    assert set(exposures["exposure_mechanism"]) <= {"personalized", "popularity"}


def test_events_reference_valid_exposures(small_config) -> None:
    _, _, _, _, exposures, events = _build(small_config)
    assert set(events["exposure_id"]) <= set(exposures["exposure_id"])
    assert set(events["event_type"]) <= {"click", "purchase"}


def test_popularity_exposures_are_more_concentrated_than_personalized(small_config) -> None:
    """The exposure-bias mechanism: popularity-driven exposures should be
    concentrated on fewer products than affinity-driven ones, regardless of
    individual preference."""
    _, _, _, _, exposures, _ = _build(small_config)

    def herfindahl(product_ids) -> float:
        shares = product_ids.value_counts(normalize=True)
        return float((shares**2).sum())

    personalized = exposures.loc[exposures["exposure_mechanism"] == "personalized", "product_id"]
    popularity = exposures.loc[exposures["exposure_mechanism"] == "popularity", "product_id"]
    assert herfindahl(popularity) > herfindahl(personalized)
