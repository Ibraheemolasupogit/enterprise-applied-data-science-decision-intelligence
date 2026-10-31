"""Unit tests for the randomised recommendation-feature experiment."""

from __future__ import annotations

from decision_intelligence.common import seeded_generator
from decision_intelligence.data.customers import generate_customers
from decision_intelligence.data.experiments import generate_experiment


def test_assignment_and_outcome_tables_are_aligned(small_config) -> None:
    customers = generate_customers(small_config, seeded_generator(1))
    assignments, outcomes, ground_truth = generate_experiment(
        small_config, customers, seeded_generator(2)
    )
    assert set(assignments["customer_id"]) == set(outcomes["customer_id"])
    assert set(assignments["variant"]) <= {"treatment", "control"}
    assert ground_truth.experiment_name == small_config.experiment_name
    assert ground_truth.assignment_probability == small_config.experiment_assignment_probability


def test_eligible_population_is_a_subset_of_customers(small_config) -> None:
    customers = generate_customers(small_config, seeded_generator(1))
    assignments, _, _ = generate_experiment(small_config, customers, seeded_generator(2))
    assert set(assignments["customer_id"]) <= set(customers["customer_id"])


def test_observation_date_is_after_assignment_date(small_config) -> None:
    customers = generate_customers(small_config, seeded_generator(1))
    assignments, outcomes, _ = generate_experiment(small_config, customers, seeded_generator(2))
    merged = assignments.merge(outcomes, on=["customer_id", "experiment_name"])
    assert (merged["observation_date"] > merged["assignment_date"]).all()
