# Milestone roadmap

Status: agreed planning roadmap, updated through Milestone 10. This is a sequence of controlled
engineering steps, not permission to implement multiple milestones in one pass. Each milestone
requires explicit approval before implementation begins (see Milestone execution methodology below).

## Phase A — Foundation

| Milestone | Scope                                           | Acceptance evidence                                          | Status |
| --------- | ------------------------------------------------ | -------------------------------------------------------------- | ------ |
| M1        | Repository foundation, charter and architecture | README, standards, CI, baseline tests                        | DONE |
| M2        | Synthetic data contracts and generation          | Reproducible datasets, schemas, quality tests                | DONE — see [../evidence/capability-status.md](../evidence/capability-status.md) and [../methodology/synthetic-data.md](../methodology/synthetic-data.md) |
| M3        | Statistical foundations                          | Tested estimators, simulations, methodological documentation | DONE — see [../evidence/m3-statistical-foundations.md](../evidence/m3-statistical-foundations.md) and [../methodology/statistical-foundations.md](../methodology/statistical-foundations.md) |

## Phase B — Causal inference and experimentation

| Milestone | Scope                                     | Acceptance evidence                                       | Status |
| --------- | ------------------------------------------- | ------------------------------------------------------------ | ------ |
| M4        | Causal foundations, DAGs and adjustment    | Identification examples, known-effect recovery            | DONE — see [../evidence/m4-causal-foundations.md](../evidence/m4-causal-foundations.md) and [../methodology/causal-foundations.md](../methodology/causal-foundations.md) |
| M5        | Observational causal inference             | DiD, event study, synthetic control, RDD and diagnostics  | DONE — see [../evidence/m5-observational-causal-inference.md](../evidence/m5-observational-causal-inference.md) and [../methodology/observational-causal-inference.md](../methodology/observational-causal-inference.md) |
| M6        | Experimentation fundamentals               | Randomisation, power, SRM, ITT, hypothesis tests           | DONE — see [../evidence/m6-experimentation-fundamentals.md](../evidence/m6-experimentation-fundamentals.md) and [../methodology/experimentation-fundamentals.md](../methodology/experimentation-fundamentals.md) |
| M7        | Advanced experimentation                   | CUPED, sequential/Bayesian methods and selected designs    | DONE — approved, committed (`5e11c6d`; NumPy typing fix `31b93a7`), CI passing on Python 3.11 and 3.12 from `31b93a7`; see [../evidence/m7-advanced-experimentation.md](../evidence/m7-advanced-experimentation.md) and [../methodology/advanced-experimentation.md](../methodology/advanced-experimentation.md) |

## Phase C — Personalisation and customer economics

| Milestone | Scope                                 | Acceptance evidence                                        | Status |
| --------- | --------------------------------------- | ------------------------------------------------------------ | ------ |
| M8        | Recommendation baselines                | Relevance, coverage and leakage-controlled benchmarks      | DONE — approved after scientific assurance review, committed (`2d6b8e7`), CI passing on Python 3.11 and 3.12; see [../evidence/m8-recommendation-baselines.md](../evidence/m8-recommendation-baselines.md) and [../methodology/recommender-baselines.md](../methodology/recommender-baselines.md) |
| M9        | Advanced recommendation and ranking     | Model comparisons, cold-start and operational evaluation   | DONE — approved, committed (`9f738df`), CI passing on Python 3.11 and 3.12; see [../evidence/m9-advanced-recommendation.md](../evidence/m9-advanced-recommendation.md) and [../methodology/advanced-recommendation.md](../methodology/advanced-recommendation.md) |
| M10       | LTV foundations                         | Historical/heuristic LTV and customer economics            | DONE — approved, committed (`4459d89`), CI passing on Python 3.11 and 3.12; see [../evidence/m10-ltv-foundations.md](../evidence/m10-ltv-foundations.md) and [../methodology/customer-economics.md](../methodology/customer-economics.md) |
| M11       | Predictive LTV                          | Temporal validation, survival/probabilistic models, calibrated uncertainty | PLANNED — incremental (causal) LTV and uplift DEFERRED to a future milestone; not part of M11 |

## Phase D — Optimisation and decision science

| Milestone | Scope                                   | Acceptance evidence                                            |
| --------- | ------------------------------------------ | ------------------------------------------------------------------ |
| M12       | Optimisation foundations                  | LP/MILP formulations, solver tests, baselines                      |
| M13       | Gurobi business applications              | Allocation/scheduling cases and sensitivity analysis               |
| M14       | Product analytics and customer choice     | Product metrics, pricing, conjoint and preference models           |
| M15       | Decision theory and simulation            | Bayesian updates, expected utility, Monte Carlo, risk analysis     |

## Phase E — Integration and portfolio completion

| Milestone | Scope                                        | Acceptance evidence                                        |
| --------- | ------------------------------------------------ | -------------------------------------------------------------- |
| M16       | Supporting forecasting and uncertainty           | Forecast comparisons, intervals, backtesting                   |
| M17       | Integrated NexaMarket case study                 | End-to-end evidence-to-decision workflow                       |
| M18       | Interactive demonstration and reports            | Usable dashboard, decision memos and visual evidence            |
| M19       | Engineering hardening                            | CI, tests, performance, security and reproducibility            |
| M20       | Portfolio evidence and final documentation       | Technical index, limitations, demonstrations, final README      |

## Sequencing rule

This plan can be refined after each milestone's assessment. If a later capability (e.g. causal LTV)
needs infrastructure from an earlier pillar that is not yet built, that prerequisite must be
completed first — milestone numbering must never be preserved at the expense of validity.

## Open items carried forward

- **Gurobi licensing** — not yet confirmed. M12/M13 must not assume a specific licence tier until this
  is resolved; HiGHS/SciPy is the CI-safe default regardless of outcome.
- **Shared-primitive interfaces** — seeding (M1), point-in-time utilities (M2), input validation and
  the bootstrap/confidence-interval helper (M3, `decision_intelligence.common`) are implemented and
  tested (see [../architecture/overview.md](../architecture/overview.md)).
- **Carried from M3 to later milestones** — proper multiple imputation / IPW for missingness and MNAR sensitivity analysis, exact binomial power, and
  penalised (Firth) logistic regression. (The wild cluster bootstrap was delivered in M5.) See the limitations in
  [../methodology/statistical-foundations.md](../methodology/statistical-foundations.md).
- **Carried from M4** — propensity-score matching (Brief §6.4) was deliberately not shipped in M4
  and was not in the agreed M5 scope: honest inference needs the Abadie–Imbens variance (the
  bootstrap is invalid for matching). It still needs an explicit home. Machine-learning nuisance
  models with cross-fitting (double machine learning) and richer heterogeneous-effect methods remain
  unscheduled. See [../methodology/causal-foundations.md](../methodology/causal-foundations.md).
- **Carried from M5** — interrupted time series (Brief §6.4) was not in the agreed M5 scope and is not
  implemented; neither are fuzzy RDD, regression kink designs, synthetic DiD or augmented synthetic
  control, conformal inference for synthetic control, or honest-DiD sensitivity bounds for
  parallel-trends violations. M5's datasets are additive generators in `data/` (no M2 contract
  changed). See [../methodology/observational-causal-inference.md](../methodology/observational-causal-inference.md).
- **Delivered in M6** — multiple-testing corrections (carried from M3) now live in
  `statistics/multiple_testing.py`; an M3 defect (NaN noncentral-t power at large noncentrality
  under SciPy 1.18) was fixed with a regression test.
- **Carried from M6 to M7** — CUPED and pre-specified covariate adjustment, sequential testing and
  peeking, Bayesian experimentation, cluster-randomised power with estimated ICCs, switchback and geo
  designs, interference, heterogeneous effects, winner's-curse correction for A/B/n, and exact
  binomial planning. See [../methodology/experimentation-fundamentals.md](../methodology/experimentation-fundamentals.md).
- **Delivered in M7** — CUPED and pre-specified covariate adjustment, sequential testing and peeking
  (group-sequential and always-valid), Bayesian experimentation, cluster-randomised planning with
  estimated ICCs and switchback designs. No M2–M6 implementation, contract or evidence was changed.
- **Carried from M7** — geo experiments, interference/network designs, heterogeneous effects,
  bandits, long-term/surrogate effects, futility boundaries and bias-adjusted estimation after
  group-sequential stopping (not scheduled), plus winner's-curse correction for A/B/n (from M6) and
  exact binomial planning (from M3). See
  [../methodology/advanced-experimentation.md](../methodology/advanced-experimentation.md).
- **Delivered in M8** — recommendation baselines (popularity, segment popularity, item/user kNN,
  content-based, PureSVD, explicit-ratings bias model), a chronological leakage-controlled
  protocol, ranking and beyond-accuracy metrics, exposure/segment fairness diagnostics,
  position-aware IPS with positivity and clipping diagnostics, and simulated online value, on an
  additive synthetic scenario (`data/recommendation_scenarios.py`). No M1–M7 implementation,
  contract or evidence was changed.
- **Carried from M8 to M9** — implicit ALS, BPR, embedding/two-tower retrieval, learning-to-rank,
  session-based and hybrid models, comparison of cold-start strategies, operational evaluation at
  scale, self-normalised and doubly robust off-policy estimators, and online A/B evaluation of a
  recommender with the M6/M7 machinery. See
  [../methodology/recommender-baselines.md](../methodology/recommender-baselines.md).
- **Delivered in M9** — implicit ALS, BPR (uniform and exposure-aware negatives), EASE, a shallow
  two-tower model with exact and approximate retrieval, two-stage pointwise and LambdaRank-style
  ranking (naive and propensity-weighted), a score-blend hybrid, cold-start strategies, SNIPS and
  doubly robust off-policy estimation with studentised cluster-bootstrap intervals, operational
  benchmarks with preflight memory checks, and a SIMULATED user-randomised A/B evaluation analysed
  with the M6/M7 components — under a protocol frozen in `recommendations/protocol.py`. Additive:
  no M1–M8 implementation, contract or evidence was changed.
- **Carried from M9** (deferred, not scheduled) — LambdaMART / gradient-boosted ranking, deep
  neural ranking, session-based and context-aware models, long-term retention, interference and
  feedback-loop simulation. See
  [../methodology/advanced-recommendation.md](../methodology/advanced-recommendation.md).
- **Delivered in M10** — contribution-based customer economics on an additive subscription
  scenario (`data/customer_economics_scenarios.py`): point-in-time panel, discrete-time
  Kaplan–Meier with Greenwood intervals, cohort retention, realised (cohort-matched) LTV,
  heuristic projected and expected-lifetime LTV (H0–H4) with explicit undefined/unstable states,
  CAC, payback, LTV/CAC with precision against a pre-set target, segment economics, sensitivity,
  complete-window backtests and repeated-sampling coverage — under a protocol frozen in
  `customer_value/protocol.py`. Additive: no M1–M9 implementation, contract or evidence changed.
- **Carried from M10 to M11** — probabilistic (BG/NBD, Gamma-Gamma) and survival-model LTV,
  predictive ML LTV with customer-level intervals, non-contractual purchasing, and incremental
  (causal) LTV with treatment targeting. See
  [../methodology/customer-economics.md](../methodology/customer-economics.md). *Scope decision
  at M11 planning:* incremental (causal) LTV with treatment targeting is DEFERRED — future
  milestone; not part of M11.

## Milestone execution methodology

```
Inspect current repository state
              v
Confirm milestone requirements
              v
Identify existing reusable components
              v
Define exact files and acceptance criteria
              v
Implement only the approved scope
              v
Run automated tests and validation
              v
Review scientific correctness
              v
Inspect changes and dependencies
              v
Document results and limitations
              v
Report evidence (commit/push performed manually by the project owner)
              v
STOP
```
