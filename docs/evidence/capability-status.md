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

## Pillar status — as of Milestone 3

| Pillar                                   | Status  | Notes                                              |
| ------------------------------------------ | ------- | ---------------------------------------------------- |
| 1. Statistics & Causal Inference          | Statistical foundations: TESTED, SIMULATED (M3). Causal inference: PLANNED (M4–M5) | Per-capability detail in the Milestone 3 table below |
| 2. A/B Testing & Experimentation          | PLANNED | Scheduled M6–M7; synthetic data foundation ready     |
| 3. Recommender Systems                    | PLANNED | Scheduled M8–M9; synthetic data foundation ready     |
| 4. Mathematical Optimisation / Gurobi     | PLANNED | Scheduled M12–M13; Gurobi licensing not yet verified |
| 5. Customer Lifetime Value                | PLANNED | Scheduled M10–M11; synthetic data foundation ready   |
| 6. Product Sense & Decision Science       | PLANNED | Scheduled M14–M15                                    |
| Forecasting (supporting capability)       | PLANNED | Scheduled M16; not a seventh pillar                  |

Only the statistical-foundations part of Pillar 1 exists (Milestone 3). No causal-inference method,
no other pillar's analytical methods, no case study and no dashboard exist yet. Any statement to the
contrary in the README or elsewhere is an error and should be corrected to match this table.

Pillar 1 spans three milestones, so its row states a status per sub-scope rather than one label.

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

## Pillar 1 status — Milestone 3 (statistical foundations)

Methodology: [../methodology/statistical-foundations.md](../methodology/statistical-foundations.md).
Evidence: [m3-statistical-foundations.md](m3-statistical-foundations.md) and the generated
[m3-statistical-foundations-results.md](m3-statistical-foundations-results.md).

| Capability                                              | Status                 | Evidence                                                                                   |
| -------------------------------------------------------- | ---------------------- | -------------------------------------------------------------------------------------------- |
| Shared input validation                                 | TESTED                 | `common/validation.py`, `tests/unit/common/test_validation.py`                              |
| Bootstrap CIs (percentile, basic, BCa; 1- and 2-sample)  | TESTED, SIMULATED      | `common/bootstrap.py`, `tests/unit/common/test_bootstrap.py`; coverage in results §7         |
| Distributions and sampling (inverse transform, rejection, SRS, stratified) | TESTED | `statistics/distributions.py`, `tests/unit/stats/test_distributions.py`                    |
| Descriptive statistics and stable moments               | TESTED                 | `statistics/descriptive.py`, `tests/unit/stats/test_descriptive.py`                          |
| Estimator properties, sampling distributions, CLT       | TESTED, SIMULATED      | `statistics/estimators.py`; results §9                                                       |
| Parametric and non-parametric hypothesis tests          | TESTED, SIMULATED      | `statistics/hypothesis_tests.py`; type I error in results §6                                  |
| Effect sizes with intervals                             | TESTED                 | `statistics/effect_sizes.py`, `tests/unit/stats/test_effect_sizes.py`                        |
| Power, sample size, minimum detectable effect           | TESTED, SIMULATED      | `statistics/power.py`; analytic vs simulated power in results §8                              |
| OLS with classical / HC0–HC3 / cluster-robust SEs, diagnostics | TESTED, SIMULATED | `statistics/linear_regression.py`; dependence coverage in results §12                      |
| Logistic regression, separation detection, calibration  | TESTED, BENCHMARKED    | `statistics/logistic_regression.py`; benchmarked only against a base-rate baseline on held-out customers (results §3) |
| Bias–variance decomposition                             | TESTED, SIMULATED      | `statistics/bias_variance.py`; results §10                                                   |
| Missing-data mechanisms and simple strategies           | TESTED, SIMULATED      | `statistics/missing_data.py`; results §5 and §11                                             |
| Monte Carlo engine (rates, coverage, MC error)          | TESTED                 | `statistics/simulation.py`, `tests/unit/stats/test_estimators_and_simulation.py`             |
| Applied analyses on M2 data                             | DEMONSTRATED           | `statistics/applied_analyses.py`, `scripts/generate_m3_evidence.py`, `tests/integration/test_applied_analyses.py` |
| Causal inference (potential outcomes, DAGs, adjustment, observational designs) | PLANNED | M4–M5                                                                          |

All M3 results are on synthetic data and simulations; none is `PRODUCTION-READY`.
