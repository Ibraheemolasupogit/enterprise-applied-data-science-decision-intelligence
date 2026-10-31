"""Statistical-property tests: does the encoded ground truth behave as designed?

These do not test "is the estimate exactly right" (no estimator ever is)
but the qualitative behaviour the Master Implementation Brief requires the
synthetic data to support:

* an unconfounded experiment's naive estimate should land near its true
  effect;
* a confounded observational campaign's naive estimate should NOT —
  demonstrating exactly the failure mode Pillar 1 (M4/M5) exists to fix;
* the optimisation reference's known optimum is feasible and internally
  consistent;
* popularity-driven exposure is measurably more concentrated than
  affinity-driven exposure.

A fixed seed makes every assertion below deterministic — there is no
run-to-run flakiness — but tolerances are kept generous (not hand-fitted to
one seed's exact decimals) so a small, legitimate change to a generator
does not spuriously break these tests.
"""

from __future__ import annotations

from decision_intelligence.data import build_synthetic_dataset
from decision_intelligence.data.config import SyntheticDataConfig

_CONFIG = SyntheticDataConfig(seed=7, n_customers=4000, n_products=20, horizon_days=330)


def _naive_experiment_effect(dataset) -> float:
    merged = dataset.tables["experiment_assignments"].merge(
        dataset.tables["experiment_outcomes"], on=["customer_id", "experiment_name"]
    )
    rate_treatment = merged.loc[merged["variant"] == "treatment", "converted"].mean()
    rate_control = merged.loc[merged["variant"] == "control", "converted"].mean()
    return float(rate_treatment - rate_control)


def _naive_marketing_effect(dataset) -> float:
    merged = dataset.tables["customer_treatments"].merge(
        dataset.tables["customer_retention"], on="customer_id"
    )
    rate_treated = merged.loc[merged["treated"], "retained_90d"].mean()
    rate_untreated = merged.loc[~merged["treated"], "retained_90d"].mean()
    return float(rate_treated - rate_untreated)


def test_unconfounded_experiment_naive_estimate_is_close_to_true_effect() -> None:
    dataset = build_synthetic_dataset(_CONFIG)
    naive_effect = _naive_experiment_effect(dataset)
    true_effect = dataset.ground_truth.experiment.true_absolute_effect
    assert abs(naive_effect - true_effect) < 0.04


def test_confounded_marketing_naive_estimate_is_biased() -> None:
    """The defining demonstration from Master Brief Section 6.5: a naive
    observational comparison should disagree with the true effect by more
    than sampling noise could plausibly explain."""
    dataset = build_synthetic_dataset(_CONFIG)
    naive_effect = _naive_marketing_effect(dataset)
    true_effect = dataset.ground_truth.marketing.true_absolute_effect
    assert abs(naive_effect - true_effect) > 0.05
    # Higher-value customers are both more likely to be treated and more
    # likely to retain regardless of treatment, so the naive estimate is
    # expected to overstate the true effect, not merely differ from it.
    assert naive_effect > true_effect


def test_experiment_assignment_rate_matches_configured_probability() -> None:
    dataset = build_synthetic_dataset(_CONFIG)
    assignments = dataset.tables["experiment_assignments"]
    observed_rate = (assignments["variant"] == "treatment").mean()
    assert abs(observed_rate - _CONFIG.experiment_assignment_probability) < 0.03


def test_optimisation_reference_optimum_is_feasible_and_better_than_greedy_by_value() -> None:
    dataset = build_synthetic_dataset(_CONFIG)
    ground_truth = dataset.ground_truth.optimisation_reference
    cost_by_name = dict(zip(ground_truth.campaign_names, ground_truth.costs, strict=False))
    value_by_name = dict(zip(ground_truth.campaign_names, ground_truth.values, strict=False))

    selected_cost = sum(cost_by_name[name] for name in ground_truth.optimal_selection)
    assert selected_cost <= ground_truth.budget

    # A naive greedy-by-value baseline must never beat the proven optimum.
    ranked = sorted(ground_truth.campaign_names, key=lambda n: value_by_name[n], reverse=True)
    greedy_cost = 0.0
    greedy_value = 0.0
    for name in ranked:
        if greedy_cost + cost_by_name[name] <= ground_truth.budget:
            greedy_cost += cost_by_name[name]
            greedy_value += value_by_name[name]
    assert greedy_value <= ground_truth.optimal_objective_value


def test_popularity_exposure_is_more_concentrated_than_personalized_exposure() -> None:
    dataset = build_synthetic_dataset(_CONFIG)
    exposures = dataset.tables["recommendation_exposures"]

    def herfindahl(product_ids) -> float:
        shares = product_ids.value_counts(normalize=True)
        return float((shares**2).sum())

    personalized = herfindahl(
        exposures.loc[exposures["exposure_mechanism"] == "personalized", "product_id"]
    )
    popularity = herfindahl(
        exposures.loc[exposures["exposure_mechanism"] == "popularity", "product_id"]
    )
    assert popularity > personalized * 1.5


def test_data_quality_rates_are_within_expected_bands() -> None:
    dataset = build_synthetic_dataset(_CONFIG)
    customers = dataset.tables["customers"]
    events = dataset.tables["events"]

    income_missing = customers["household_income_proxy"].isna().mean()
    region_missing = customers["region"].isna().mean()
    assert abs(income_missing - _CONFIG.quality.missing_income_rate) < 0.03
    assert abs(region_missing - _CONFIG.quality.missing_region_rate) < 0.02

    duplicate_business_keys = events.duplicated(
        subset=["exposure_id", "event_type", "event_timestamp"]
    ).mean()
    assert abs(duplicate_business_keys - _CONFIG.quality.duplicate_event_rate) < 0.015

    late = (events["recorded_at"] != events["event_timestamp"]).mean()
    assert abs(late - _CONFIG.quality.late_arrival_rate) < 0.015
