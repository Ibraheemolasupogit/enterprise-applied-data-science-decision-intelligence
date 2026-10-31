"""Materialize a small, committed sample of the synthetic dataset.

This script exists so a reviewer can inspect real generated data without
running any code: `data/samples/*.csv` and `data/schemas/*.json` are
produced by running this script and are small (a few hundred customers'
worth) specifically so they stay lightweight in the repository. The
bounded-but-larger default build used by tests and CI
(`SyntheticDataConfig()`, 2000 customers) is not what gets committed here.

Usage:
    python scripts/generate_synthetic_dataset.py
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from decision_intelligence.data import build_synthetic_dataset, validate_dataset
from decision_intelligence.data.config import SyntheticDataConfig
from decision_intelligence.data.schemas import TABLE_SCHEMAS

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = REPO_ROOT / "data" / "samples"
SCHEMAS_DIR = REPO_ROOT / "data" / "schemas"

_SAMPLE_CONFIG = SyntheticDataConfig(seed=20260101, n_customers=60, n_products=10, horizon_days=180)


def main() -> None:
    dataset = build_synthetic_dataset(_SAMPLE_CONFIG)
    violations = validate_dataset(dataset.tables)
    if violations:
        raise SystemExit(f"Refusing to write an invalid sample dataset: {violations}")

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    SCHEMAS_DIR.mkdir(parents=True, exist_ok=True)

    for name, df in dataset.tables.items():
        df.to_csv(SAMPLES_DIR / f"{name}.csv", index=False)
    dataset.latent_affinities.to_csv(
        SAMPLES_DIR / "latent_affinities_ground_truth_only.csv", index=False
    )

    for name, schema in TABLE_SCHEMAS.items():
        payload = {
            "name": schema.name,
            "columns": schema.columns,
            "primary_key": list(schema.primary_key),
            "foreign_keys": {k: list(v) for k, v in schema.foreign_keys.items()},
            "not_null": list(schema.not_null),
        }
        (SCHEMAS_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")

    ground_truth_path = SAMPLES_DIR / "ground_truth.json"
    ground_truth_path.write_text(
        json.dumps(dataclasses.asdict(dataset.ground_truth), indent=2) + "\n"
    )

    print(f"Wrote {len(dataset.tables)} sample tables to {SAMPLES_DIR}")
    print(f"Wrote {len(TABLE_SCHEMAS)} schema contracts to {SCHEMAS_DIR}")
    print(f"Wrote ground truth to {ground_truth_path}")


if __name__ == "__main__":
    main()
