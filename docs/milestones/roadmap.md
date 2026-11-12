# Milestone roadmap

Status: agreed planning roadmap, updated through Milestone 6. This is a sequence of controlled
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
| M6        | Experimentation fundamentals               | Randomisation, power, SRM, ITT, hypothesis tests           | IMPLEMENTED — awaiting review; see [../evidence/m6-experimentation-fundamentals.md](../evidence/m6-experimentation-fundamentals.md) and [../methodology/experimentation-fundamentals.md](../methodology/experimentation-fundamentals.md) |
| M7        | Advanced experimentation                   | CUPED, sequential/Bayesian methods and selected designs    | Not started |

## Phase C — Personalisation and customer economics

| Milestone | Scope                                 | Acceptance evidence                                        |
| --------- | --------------------------------------- | ------------------------------------------------------------ |
| M8        | Recommendation baselines                | Relevance, coverage and leakage-controlled benchmarks      |
| M9        | Advanced recommendation and ranking     | Model comparisons, cold-start and operational evaluation   |
| M10       | LTV foundations                         | Historical/heuristic LTV and customer economics            |
| M11       | Predictive and incremental LTV          | Temporal validation, survival/probabilistic models, uplift |

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
