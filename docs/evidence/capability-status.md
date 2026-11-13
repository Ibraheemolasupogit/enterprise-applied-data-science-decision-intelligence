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

## Pillar status — as of Milestone 8

| Pillar                                   | Status  | Notes                                              |
| ------------------------------------------ | ------- | ---------------------------------------------------- |
| 1. Statistics & Causal Inference          | Statistical foundations: TESTED, SIMULATED (M3). Causal foundations and backdoor adjustment: TESTED, SIMULATED (M4). Observational designs (DiD, event study, synthetic control, sharp RDD): TESTED, SIMULATED (M5). ITS, fuzzy RDD, matching: PLANNED | Per-capability detail in the Milestone 3, 4 and 5 tables below |
| 2. A/B Testing & Experimentation          | Experimentation fundamentals (assignment, SRM, planning, ITT analysis, noncompliance/attrition, validity checks, decision rules): TESTED, SIMULATED (M6). Advanced experimentation (CUPED/regression adjustment, group-sequential and always-valid inference, Bayesian analysis, cluster-randomised planning, switchback): TESTED, SIMULATED, BENCHMARKED (M7). Geo, interference, bandits, heterogeneous and long-term effects: PLANNED | Per-capability detail in the Milestone 6 and 7 tables below |
| 3. Recommender Systems                    | Recommendation baselines (popularity, segment popularity, item/user kNN, content-based, PureSVD, explicit-ratings bias model), chronological evaluation, ranking/exposure/fairness metrics, IPS and simulated online value: IMPLEMENTED, TESTED, SIMULATED, BENCHMARKED (M8, awaiting review). Advanced retrieval/ranking, cold-start strategies, operational and online evaluation: PLANNED (M9) | Per-capability detail in the Milestone 8 table below |
| 4. Mathematical Optimisation / Gurobi     | PLANNED | Scheduled M12–M13; Gurobi licensing not yet verified |
| 5. Customer Lifetime Value                | PLANNED | Scheduled M10–M11; synthetic data foundation ready   |
| 6. Product Sense & Decision Science       | PLANNED | Scheduled M14–M15                                    |
| Forecasting (supporting capability)       | PLANNED | Scheduled M16; not a seventh pillar                  |

Pillar 1's statistical foundations (M3), causal foundations with backdoor adjustment (M4) and
observational designs — difference-in-differences, event studies, synthetic control and sharp
regression discontinuity (M5) — exist, as do Pillar 2's experimentation fundamentals (M6) and
advanced experimentation (M7), and Pillar 3's recommendation baselines (M8). Interrupted time
series, fuzzy RDD and propensity-score matching, geo/interference experiments and bandits, advanced
recommenders (M9), every other pillar's analytical methods, case studies and the dashboard do not
exist yet. Any statement to the
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
| Causal inference | See Milestone 4 table | M4 implemented; observational designs M5 |

All M3 results are on synthetic data and simulations; none is `PRODUCTION-READY`.

## Pillar 1 status — Milestone 4 (causal foundations, DAGs and adjustment)

Methodology: [../methodology/causal-foundations.md](../methodology/causal-foundations.md).
Evidence: [m4-causal-foundations.md](m4-causal-foundations.md) and the generated
[m4-causal-foundations-results.md](m4-causal-foundations-results.md).

| Capability | Status | Evidence |
| --- | --- | --- |
| Potential outcomes and estimands (ATE, ATT, ATC, CATE; naive = ATT + selection bias) | TESTED | `causal_inference/estimands.py`, `tests/unit/causal/test_estimators.py` |
| Causal DAGs: d-separation, paths, backdoor criterion, adjustment sets, roles | TESTED | `causal_inference/dag.py`; networkx-verified (`tests/unit/causal/test_dag.py`) |
| Regression adjustment, IPW (HT, Hájek), AIPW for ATE and ATT | TESTED, SIMULATED, BENCHMARKED | `causal_inference/estimators.py`; estimates and SEs checked against statsmodels `TreatmentEffect`; 200 M2 draws vs known truth and vs the naive baseline (results §7) |
| Subgroup (conditional) effects and heterogeneity test | TESTED, SIMULATED | `subgroup_effects`, `heterogeneity_test`; results §5 |
| Overlap, balance, weight and propensity diagnostics; trimming | TESTED, SIMULATED | `causal_inference/diagnostics.py`; results §3, §9 |
| Sensitivity: E-value, Ding–VanderWeele bounding factor, bias-formula grid | TESTED | `causal_inference/sensitivity.py`; results §6 |
| Negative-control outcome and DAG falsification test | DEMONSTRATED | `causal_inference/campaign.py`; results §3–4, §7 |
| Failure cases (unmeasured confounding, post-treatment adjustment/selection, M-bias, mediator, positivity, misspecification, interference) | SIMULATED | `causal_inference/scenarios.py`, `campaign.py`; `tests/statistical/test_m4_causal_properties.py` |
| M2 retention-offer case study | DEMONSTRATED | `causal_inference/campaign.py`, `scripts/generate_m4_evidence.py`, `tests/integration/test_campaign_analysis.py` |
| Propensity-score matching | PLANNED | Deferred: needs Abadie–Imbens variance (roadmap open item) |
| Observational designs (DiD, event study, synthetic control, sharp RDD) | See Milestone 5 table | M5 |

All M4 results are on synthetic data and simulations; none is `PRODUCTION-READY`. Known synthetic
ground truth validates the code and the identification logic, not real-world external validity.

## Pillar 1 status — Milestone 5 (observational causal inference)

Methodology: [../methodology/observational-causal-inference.md](../methodology/observational-causal-inference.md).
Evidence: [m5-observational-causal-inference.md](m5-observational-causal-inference.md) and the
generated [m5-observational-causal-inference-results.md](m5-observational-causal-inference-results.md).

| Capability | Status | Evidence |
| --- | --- | --- |
| Additive time-indexed and threshold synthetic datasets with known effects | TESTED | `data/rollout_panel.py`, `data/synthetic_control_panel.py`, `data/threshold_assignment.py`; `tests/unit/data/test_m5_generators.py` |
| Two-period and panel DiD; static TWFE with exact negative-weight diagnostic | TESTED, SIMULATED, BENCHMARKED | `causal_inference/did.py`; statsmodels reference; results §1–2 |
| Staggered adoption (Callaway–Sant'Anna) and event studies with pre-trend tests and nominal 95% sup-t simultaneous bands (observed joint coverage 0.920 — undercoverage) | TESTED, SIMULATED | `did.staggered_did`, `event_study_twfe`; independent reference implementation; results §1–2b |
| Few-cluster inference (CR1, wild cluster restricted bootstrap; unreliable with ≤ 2 treated clusters, warned) | TESTED, SIMULATED | `causal_inference/cluster_inference.py`; brute-force reference; results §3 |
| Synthetic control with placebo and sensitivity analyses | TESTED, SIMULATED, BENCHMARKED | `causal_inference/synthetic_control.py`; vs naive DiD; results §4–5 |
| Sharp RDD with robust bias-corrected inference and validity checks | TESTED, SIMULATED | `causal_inference/rdd.py`; statsmodels reference; results §6–7 |
| Deliberately invalid designs for each method | SIMULATED | `causal_inference/observational_cases.py`; `tests/statistical/test_m5_observational_properties.py` |
| M5 case studies (rollout, regional policy, loyalty tier) | DEMONSTRATED | `observational_cases.py`, `scripts/generate_m5_evidence.py`, `tests/integration/test_m5_case_studies.py` |
| Interrupted time series, fuzzy RDD, synthetic DiD, conformal SC inference, honest-DiD bounds | PLANNED | Not scheduled; see roadmap |

All M5 results are on synthetic data and simulations; none is `PRODUCTION-READY`.

## Pillar 2 status — Milestone 6 (experimentation fundamentals)

Methodology: [../methodology/experimentation-fundamentals.md](../methodology/experimentation-fundamentals.md).
Evidence: [m6-experimentation-fundamentals.md](m6-experimentation-fundamentals.md) and the
generated [m6-experimentation-fundamentals-results.md](m6-experimentation-fundamentals-results.md).

| Capability | Status | Evidence |
| --- | --- | --- |
| Pre-specified design (units, estimand, metrics, worthwhile effect, guardrail margins) | TESTED | `experimentation/design.py` |
| Deterministic hash, complete, stratified and cluster randomisation | TESTED, SIMULATED | `experimentation/assignment.py`; `tests/unit/experimentation/test_assignment_and_srm.py` |
| Sample ratio mismatch detection and localisation (not diagnosis) | TESTED, SIMULATED | `experimentation/srm.py`; SciPy reference; results §3, §5 |
| Planning: sample size, MDE, duration, design effect | TESTED, SIMULATED | `experimentation/planning.py` (M3 power); closed-form and statsmodels references; results §6 |
| ITT analysis of binary and continuous metrics; stratified, clustered and A/B/n analyses | TESTED, SIMULATED, BENCHMARKED | `experimentation/analysis.py`; statsmodels/SciPy references; results §4, §8 |
| Noncompliance (ITT, CACE) and attrition (Lee, Manski bounds) | TESTED, SIMULATED, BENCHMARKED | `experimentation/compliance.py`; 2SLS reference; results §7 |
| Validity checks (duplicates, leakage, balance, exclusions, metric integrity, post-randomisation selection) | TESTED, SIMULATED | `experimentation/validity.py`; results §1, §3, §10 |
| Multiple-testing corrections (Bonferroni, Holm, BH, BY) | TESTED, SIMULATED | `statistics/multiple_testing.py`; statsmodels reference; results §9 |
| Decision rules (LAUNCH/HOLD/ITERATE/STOP) with guardrails and business value | TESTED, SIMULATED | `experimentation/decision.py`; operating characteristics in results §9 |
| Additive experiment scenario generator with known potential outcomes | TESTED | `data/experiment_scenarios.py`; `tests/unit/data/test_experiment_scenarios.py` |
| M2 experiment and scenario case studies | DEMONSTRATED | `experimentation/cases.py`, `scripts/generate_m6_evidence.py`, `tests/integration/test_m6_case_studies.py` |
| CUPED/covariate adjustment, sequential and Bayesian testing, switchback | see Milestone 7 table | M7 |
| Geo experiments, interference, bandits | PLANNED | Not scheduled |

All M6 results are on synthetic data and simulations; none is `PRODUCTION-READY`.

## Pillar 2 status — Milestone 7 (advanced experimentation)

Methodology: [../methodology/advanced-experimentation.md](../methodology/advanced-experimentation.md).
Evidence: [m7-advanced-experimentation.md](m7-advanced-experimentation.md) and the generated
[m7-advanced-experimentation-results.md](m7-advanced-experimentation-results.md).

| Capability | Status | Evidence |
| --- | --- | --- |
| CUPED and Lin regression adjustment; covariate timing check; planning with historical R² | TESTED, SIMULATED, BENCHMARKED | `experimentation/variance_reduction.py`; statsmodels HC2 reference; vs M6 unadjusted; results §1–2 |
| Group-sequential boundaries (Lan–DeMets spending, Pocock, O'Brien–Fleming), operating characteristics, monitoring, repeated CIs | TESTED, SIMULATED, BENCHMARKED | `experimentation/sequential.py`; published constants, SciPy multivariate normal, simulation; vs fixed horizon and naive peeking; results §3–5 |
| Always-valid inference (mixture SPRT, always-valid p-values, confidence sequences) | TESTED, SIMULATED, BENCHMARKED | `experimentation/sequential.py`; quadrature reference; results §4–5 |
| Bayesian A/B analysis (Beta-binomial, normal-normal), credible intervals, expected-loss decisions | TESTED, SIMULATED, BENCHMARKED | `experimentation/bayesian.py`; closed form and Monte Carlo references; vs M6 rule; results §6–7 |
| ICC estimation and cluster-randomised planning | TESTED, SIMULATED, BENCHMARKED | `experimentation/designs.py`; ANOVA F identity; simulated CR1 power vs naive plan; results §8 |
| Switchback design and analysis (burn-in, blocking, randomisation inference) | TESTED, SIMULATED, BENCHMARKED | `experimentation/designs.py`; enumeration and M3 permutation references; vs naive hourly analysis; results §9 |
| Additive scenario generators (pre-period metrics, switchback market) with known truth | TESTED | `data/advanced_experiment_scenarios.py`; `tests/unit/data/test_advanced_experiment_scenarios.py` |
| M7 case studies (CUPED, sequential, Bayesian, cluster, switchback) | DEMONSTRATED | `experimentation/advanced_cases.py`, `scripts/generate_m7_evidence.py`, `tests/integration/test_m7_case_studies.py` |
| Geo experiments, interference/network designs, heterogeneous effects, bandits, long-term effects, futility boundaries, bias-adjusted estimation after stopping | PLANNED | Not scheduled |

All M7 results are on synthetic data and simulations; none is `PRODUCTION-READY`.

## Pillar 3 status — Milestone 8 (recommendation baselines) — IMPLEMENTED, awaiting review

Methodology: [../methodology/recommender-baselines.md](../methodology/recommender-baselines.md).
Evidence: [m8-recommendation-baselines.md](m8-recommendation-baselines.md) and the generated
[m8-recommendation-baselines-results.md](m8-recommendation-baselines-results.md).

| Capability | Status | Evidence |
| --- | --- | --- |
| Synthetic recommendation scenario (item-level preferences, position bias, logged propensities, cold start, MNAR ratings) with separate ground truth | TESTED | `data/recommendation_scenarios.py`; `tests/unit/data/test_recommendation_scenarios.py` |
| Point-in-time interaction data and chronological train/validation/test protocol; leakage perturbation tests | TESTED, SIMULATED | `recommendations/interactions.py`, `recommendations/splits.py`; results §2 |
| Baselines: random, global and segment popularity, item-kNN, user-kNN, content-based, PureSVD; cold-start fallback | TESTED, SIMULATED, BENCHMARKED | `recommendations/baselines.py`; brute-force and closed-form references; results §3–5, §10 |
| Explicit-ratings bias model | TESTED, SIMULATED | `recommendations/explicit.py`; least-squares reference; results §12 |
| Ranking, coverage, novelty, diversity, personalisation and exposure metrics | TESTED | `recommendations/metrics.py`; hand-worked and brute-force references; results §4, §8 |
| Bootstrap intervals, Holm-adjusted paired comparisons, validation-window tuning | TESTED, SIMULATED | `recommendations/evaluation.py` (M3 bootstrap and paired t, M6 Holm); results §3–5 |
| Synthetic-oracle metrics and simulated online value (evaluation only) | TESTED, SIMULATED | `recommendations/oracle.py`; loop references; results §5–7 |
| Position-aware IPS with examination estimated from logs, clipping, ESS and positivity diagnostics | TESTED, SIMULATED | `recommendations/off_policy.py`; loop reference; repeated sampling; results §11 |
| Exposure fairness (category) and segment/region performance | TESTED, SIMULATED | `recommendations/cases.py`; results §9 |
| M8 case studies | DEMONSTRATED | `recommendations/cases.py`, `scripts/generate_m8_evidence.py`, `tests/integration/test_m8_case_studies.py` |
| Implicit ALS, BPR, two-tower/embedding retrieval, learning-to-rank, session-based and hybrid models; cold-start strategy comparison; operational evaluation; online A/B evaluation | PLANNED | M9 |

All M8 results are on synthetic data and simulations; none is `PRODUCTION-READY`. Status is
**IMPLEMENTED — awaiting scientific assurance review**, not DONE.
