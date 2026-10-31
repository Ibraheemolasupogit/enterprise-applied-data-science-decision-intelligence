"""Integration tests for the full synthetic-data build pipeline.

Unlike tests/unit/data, these exercise every generator together through
:func:`build_synthetic_dataset` and check the dataset as a whole: full
schema/referential/temporal validation, and build-to-build reproducibility.
"""

from __future__ import annotations

from decision_intelligence.data import build_synthetic_dataset, validate_dataset
from decision_intelligence.data.config import SyntheticDataConfig
from decision_intelligence.data.schemas import TABLE_SCHEMAS

_TEST_CONFIG = SyntheticDataConfig(seed=2026, n_customers=500, n_products=15, horizon_days=240)


def test_every_declared_table_is_produced() -> None:
    dataset = build_synthetic_dataset(_TEST_CONFIG)
    assert set(TABLE_SCHEMAS) <= set(dataset.tables)
    for name, df in dataset.tables.items():
        assert len(df) > 0, f"{name} table is unexpectedly empty"


def test_full_dataset_satisfies_every_contract() -> None:
    dataset = build_synthetic_dataset(_TEST_CONFIG)
    violations = validate_dataset(dataset.tables)
    assert violations == []


def test_rebuilding_with_the_same_config_is_byte_identical() -> None:
    first = build_synthetic_dataset(_TEST_CONFIG)
    second = build_synthetic_dataset(_TEST_CONFIG)
    for name in first.tables:
        assert first.tables[name].equals(second.tables[name]), name
    assert first.latent_affinities.equals(second.latent_affinities)
    assert first.ground_truth == second.ground_truth


def test_different_seed_changes_the_dataset() -> None:
    other_config = SyntheticDataConfig(
        seed=_TEST_CONFIG.seed + 1,
        n_customers=_TEST_CONFIG.n_customers,
        n_products=_TEST_CONFIG.n_products,
        horizon_days=_TEST_CONFIG.horizon_days,
    )
    first = build_synthetic_dataset(_TEST_CONFIG)
    second = build_synthetic_dataset(other_config)
    assert not first.tables["customers"]["latent_value"].equals(
        second.tables["customers"]["latent_value"]
    )


def test_default_bounded_config_builds_and_validates() -> None:
    """The out-of-the-box default must stay bounded enough for local/CI use
    and must pass every contract — this is what a reviewer gets by running
    `build_synthetic_dataset()` with no arguments."""
    dataset = build_synthetic_dataset()
    assert dataset.config.n_customers <= 5_000
    assert validate_dataset(dataset.tables) == []
