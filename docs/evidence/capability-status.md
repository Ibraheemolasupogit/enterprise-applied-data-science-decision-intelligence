# Capability status registry

This registry is the single source of truth for what is actually implemented versus merely planned.
The README and any other document must not claim a status more advanced than what is recorded here.

## Status definitions

| Status           | Meaning                                                        |
| ----------------- | ----------------------------------------------------------------- |
| PLANNED          | Defined but not built                                             |
| DESIGN ONLY      | Architecture or mathematical specification exists                 |
| IMPLEMENTED      | Code exists                                                       |
| TESTED           | Required validation passed                                        |
| SIMULATED        | Demonstrated using controlled synthetic conditions                |
| BENCHMARKED      | Compared against defined alternatives                             |
| DEMONSTRATED     | Reproducible user-facing example exists                           |
| PRODUCTION-READY | Only when substantial operational evidence supports the claim     |

`PRODUCTION-READY` must never be used for this repository's synthetic, local-first demonstrations.

## Pillar status — as of Milestone 2

| Pillar                                   | Status  | Notes                                              |
| ------------------------------------------ | ------- | ---------------------------------------------------- |
| 1. Statistics & Causal Inference          | PLANNED | Scheduled M3–M5; synthetic data foundation ready     |
| 2. A/B Testing & Experimentation          | PLANNED | Scheduled M6–M7; synthetic data foundation ready     |
| 3. Recommender Systems                    | PLANNED | Scheduled M8–M9; synthetic data foundation ready     |
| 4. Mathematical Optimisation / Gurobi     | PLANNED | Scheduled M12–M13; Gurobi licensing not yet verified |
| 5. Customer Lifetime Value                | PLANNED | Scheduled M10–M11; synthetic data foundation ready   |
| 6. Product Sense & Decision Science       | PLANNED | Scheduled M14–M15                                    |
| Forecasting (supporting capability)       | PLANNED | Scheduled M16; not a seventh pillar                  |

No pillar's own analytical methods, case studies or dashboard exist yet — only the data foundation
they will consume (below). Any statement to the contrary in the README or elsewhere is an error and
should be corrected to match this table.

## Foundation status — Milestone 1

| Item                                   | Status      | Evidence                                                  |
| ----------------------------------------- | ----------- | -------------------------------------------------------------- |
| Repository scaffold and package layout   | IMPLEMENTED | `src/decision_intelligence/`, `pyproject.toml`                 |
| Deterministic seeding utility             | TESTED      | `src/decision_intelligence/common/seeding.py`, smoke tests      |
| Architecture documentation                | DESIGN ONLY | `docs/architecture/overview.md`                                 |
| Milestone roadmap                         | DESIGN ONLY | `docs/milestones/roadmap.md`                                    |
| CI (lint, type-check, test on 3.11/3.12) | IMPLEMENTED | `.github/workflows/ci.yml` (runs on push to a hosted remote)    |

## Foundation status — Milestone 2

| Item                                            | Status      | Evidence                                                                    |
| -------------------------------------------------- | ----------- | -------------------------------------------------------------------------------- |
| Synthetic entity schemas and data contracts        | TESTED      | `src/decision_intelligence/data/schemas.py`, `tests/unit/data/test_schemas.py`    |
| Reproducible generators (13 tables, seeded)        | TESTED      | `src/decision_intelligence/data/*.py`, `tests/unit/data/`                        |
| Full-pipeline referential/temporal validation      | TESTED      | `src/decision_intelligence/data/validation.py`, `tests/integration/test_pipeline.py` |
| Deliberately encoded ground truth (4 mechanisms)   | TESTED      | `src/decision_intelligence/data/ground_truth.py`, `tests/statistical/test_ground_truth_recovery.py` |
| Confounding, exposure bias, data-quality failures  | TESTED      | `src/decision_intelligence/data/marketing.py`, `interactions.py`, `quality.py`    |
| Shared point-in-time utility                       | TESTED      | `src/decision_intelligence/data/point_in_time.py`, `tests/unit/data/test_point_in_time.py` |
| Bounded, committed sample dataset                  | DEMONSTRATED | `data/samples/`, `data/schemas/`, `scripts/generate_synthetic_dataset.py`        |
| Methodology documentation                          | DESIGN ONLY | `docs/methodology/synthetic-data.md`                                             |

No pillar's own analytical methods, case studies or dashboard exist yet.
