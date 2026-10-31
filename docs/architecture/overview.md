# Architecture overview

Status: architecture-of-record as agreed at Milestone 1. No pillar code exists yet — see
[../evidence/capability-status.md](../evidence/capability-status.md).

## Purpose

This repository demonstrates transferable statistical, causal, experimentation, personalisation,
optimisation and decision-science methods — generalisable across industries — as distinct from the
domain-specific platform repositories elsewhere in the portfolio. It is Python-first, locally
reproducible and cloud-neutral.

## Six permanent pillars

1. Statistics & Causal Inference
2. A/B Testing & Experimentation
3. Recommender Systems
4. Mathematical Optimisation / Gurobi
5. Customer Lifetime Value
6. Product Sense & Decision Science

Time-series forecasting is a supporting capability used by pillars 1, 3 and 5 wherever forecasting is
required; it is not a seventh pillar and must not gain its own standalone portfolio section.

## Layered architecture

```
                  SYNTHETIC DATA SOURCES
       Customers | Transactions | Interactions
       Experiments | Campaigns | Product Data
       Prices | Demand | Operational Capacity
                            |
                            v
                  DATA FOUNDATION
       Validation | Schemas | Point-in-Time
       Feature Engineering | Data Contracts
                            |
           +----------------+-----------------+
           v                v                 v
     STATISTICS       EXPERIMENTATION     RECOMMENDERS
     & CAUSAL ML      & A/B TESTING       & RANKING
           |                |                 |
           +----------------+-----------------+
                            v
                 CUSTOMER ECONOMICS
              LTV | Retention | Incrementality
                            |
                            v
                  DECISION SCIENCE
            Expected Utility | Risk | Scenarios
                            |
                            v
                   OPTIMISATION
                 Gurobi LP / MILP
                            |
                            v
                  DECISION OUTPUTS
          Reports | Dashboards | Recommendations
                            |
                            v
                 VALIDATION & MONITORING
```

Cross-cutting throughout: reproducibility, data quality, statistical validity, testing, experiment
tracking, model evaluation, version control, documentation, responsible analytics, CI/CD, performance,
security, cost awareness, decision auditability.

This is a logical architecture. It does not require cloud deployment; local reproducibility is the
baseline for every pillar.

## Two usage modes

Every pillar must be independently useful on its own synthetic data and its own tests. The layered
diagram above describes one *optional* composition — the NexaMarket integrated scenario — not a
mandatory execution order. Do not force a method into the integrated flow if it is better demonstrated
standalone.

- **Pillar-level mode:** each pillar's module, tests and docs stand alone and are reviewable in
  isolation.
- **Integrated mode:** `case_studies/nexamarket/` composes selected pillar outputs into one end-to-end
  decision workflow (introduced at M17, not before).

## Shared-utility ownership

To avoid duplicated statistical primitives across pillars (explicitly prohibited — see
[../milestones/roadmap.md](../milestones/roadmap.md)), ownership is fixed as follows:

| Concern                                   | Owning module                         | Consumers                                   |
| ------------------------------------------ | -------------------------------------- | -------------------------------------------- |
| Deterministic seeding                      | `decision_intelligence.common.seeding` | all pillars                                  |
| Bootstrap / confidence-interval helpers    | `decision_intelligence.common` (future)| statistics, experimentation, decision_science|
| Point-in-time / temporal-leakage utilities | `decision_intelligence.data` (future)  | forecasting, recommendations, customer_value |

A pillar module must depend on the owning module above. It must not reimplement an owned primitive
locally, even approximately.

## Decoupled pillar interfaces

Pillars that are frequently combined in the integrated scenario must still communicate through plain,
documented data interfaces rather than direct imports of one another's internals:

- `optimisation` accepts plain value/cost/constraint arrays (e.g. from `recommendations` or
  `customer_value`). It must never import `recommendations` or `customer_value` directly.
- `decision_science` consumes CLV and causal-effect outputs as plain data (point estimates plus
  uncertainty), not by importing `customer_value` or `causal_inference` internals.
- `forecasting` is a supporting capability consumed by `statistics`, `recommendations` and
  `customer_value` through its public forecast-output interface, not by those pillars reimplementing
  forecasting logic.

This keeps each pillar removable and independently testable, and keeps the integrated scenario a
composition rather than a hidden dependency graph.

## Technology position

Python-first; local reproducibility is the baseline. Cloud integration is optional and only
introduced if it materially improves a specific demonstration — this repository is not tied to Azure,
AWS or GCP. Mathematical optimisation targets the Gurobi Python API where a verified licence permits
it, with HiGHS/SciPy as the CI-safe open-source fallback (Gurobi licensing is an open item — see
[../evidence/capability-status.md](../evidence/capability-status.md)).
