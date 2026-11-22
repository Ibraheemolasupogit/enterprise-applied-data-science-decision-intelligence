# Enterprise Applied Data Science & Decision Intelligence Platform

A Python platform for testing statistical methods and using their results to make decisions
under uncertainty. It uses reproducible synthetic data to study causal effects, evaluate
experiments and recommenders, and estimate customer economics. The workflows run locally
and are cloud-neutral.

**M1–M10 are complete. M11 predictive customer lifetime value (LTV) is under development,
and its confirmatory study remains incomplete.** The completed analyses use synthetic data
and simulations, so their findings apply to the documented scenarios and assumptions.

## Six permanent pillars

| Pillar | Status and scope |
| --- | --- |
| **1. Statistics & Causal Inference** | **Implemented and tested (M3–M5).** Estimation, inference and statistical simulation, causal adjustment, and observational designs. |
| **2. A/B Testing & Experimentation** | **Implemented and tested (M6–M7).** Randomised design, power and validity checks, treatment-effect analysis, variance reduction, sequential monitoring and Bayesian analysis. |
| **3. Recommender Systems** | **Implemented and benchmarked (M8–M9).** Baselines, retrieval and ranking, cold-start strategies, and chronological, off-policy and simulated online evaluation. |
| **4. Mathematical Optimisation / Gurobi** | **Planned (M12–M13).** Linear and mixed-integer optimisation for allocation and scheduling. Gurobi use depends on licence verification. |
| **5. Customer Lifetime Value** | **Foundations implemented and tested (M10).** Contribution-based LTV, retention, acquisition cost and payback. **Predictive LTV is under development (M11).** Incremental LTV is deferred. |
| **6. Product Sense & Decision Science** | **Planned (M14–M15).** Product metrics, pricing and customer choice, expected utility and decision-making under risk. |

Time-series forecasting is planned as a supporting capability (M16). The integrated
NexaMarket case study is planned for M17. See the
[capability status registry](docs/evidence/capability-status.md) for the approved scope
through M10 and the [milestone roadmap](docs/milestones/roadmap.md) for sequencing.

## Completed work (M1–M10)

### Foundation (M1–M2)

M1 established the Python package, architecture, deterministic seeding and CI. M2
introduced a seeded, 13-table synthetic dataset with schema, referential integrity
and point-in-time validation. It includes deliberately encoded treatment effects,
confounding and exposure bias. Later studies add dedicated scenarios with known
outcomes or model-based reference values. [Synthetic data methodology](docs/methodology/synthetic-data.md)

### Statistics and causal inference (M3–M5)

- **Statistics.** Sampling, estimation, bootstrap intervals, hypothesis tests, effect
  sizes, power and regression. Repeated-sampling studies measure bias, type I error,
  power and interval coverage.
- **Causal inference.** Causal graphs and backdoor adjustment, including regression,
  weighting and doubly robust estimators, with diagnostics for overlap and sensitivity.
  Observational designs cover staggered difference-in-differences, event studies,
  synthetic control and sharp regression discontinuity.
- **Evidence.** In the synthetic retention campaign, a naive comparison estimated a
  19–23 percentage point lift against a true 8-point effect. A doubly robust
  estimator was close to the truth on average across 200 simulations with the
  required confounder observed. [Causal adjustment evidence](docs/evidence/m4-causal-foundations.md)

### Experimentation (M6–M7)

- **Design and analysis.** Pre-specified estimands and guardrails, random assignment,
  power and sample size, sample ratio mismatch checks, intention-to-treat analysis,
  and methods for noncompliance and attrition.
- **Advanced methods.** CUPED and regression adjustment, group-sequential and
  always-valid inference, Bayesian expected-loss decisions, cluster-randomised
  planning and switchback experiments.
- **Evidence.** A synthetic experiment estimated a 3.1 percentage point lift with a
  95% confidence interval of −0.7 to +6.9 points. The pre-specified decision was
  **HOLD**, and repeated simulation showed limited power to detect the true 4-point
  lift. [Experimentation evidence](docs/evidence/m6-experimentation-fundamentals.md)

### Recommender systems (M8–M9)

- **Models.** Popularity, nearest-neighbour and content-based baselines, plus
  factorisation, two-tower retrieval, two-stage ranking and hybrid models.
- **Evaluation.** Chronological train, validation and test splits, ranking and exposure
  metrics, off-policy estimators, indicative resource benchmarks, and simulated
  randomised A/B comparisons.
- **Evidence.** Gains over tuned item-based nearest neighbours varied across
  scenarios. On a fresh synthetic seed, that baseline led on both oracle ranking
  quality and simulated online value. [Ranking benchmark](docs/evidence/m9-advanced-recommendation.md)

### Customer economics (M10)

- **Measurement.** Realised and projected value on a contribution basis, cohort
  retention with Kaplan–Meier estimates, customer acquisition cost (CAC), payback
  and LTV/CAC. Backtests use point-in-time panels and complete outcome windows.
- **Evidence.** In repeated simulations, nominal 95% LTV/CAC intervals covered the
  underlying ratio only 79–85% of the time. The analysis traced the shortfall to
  omitted uncertainty in acquisition spend. [Customer economics evidence](docs/evidence/m10-ltv-foundations.md)

## Predictive LTV (M11) — under development

M11 studies finite-horizon customer value in contractual and non-contractual settings.
Development covers survival, probabilistic purchase and predictive regression models,
with point-in-time validation and checks for calibration and interval coverage.
[Development-stage evaluation](docs/evidence/m11-p7-findings.md)

The confirmatory study remains incomplete following governed resource-limit failures.
Earlier final attempts also failed during repeated sampling.
[First final attempt](docs/evidence/m11-p8-final-attempt-1.md) ·
[Retry record](docs/evidence/m11-p8-a14-handover.md)

Final execution follows a pre-specified protocol with reserved seeds and monitored
memory and runtime limits. The recorded dry run is preliminary evidence, and
confirmatory model comparisons and acceptance conclusions remain pending.
[Dry-run record](docs/evidence/m11-p8-dry-run/record.md) ·
[Resource monitoring decision](docs/milestones/m11-p8-resource-monitoring-approval.md)

## Scientific practice

- **Synthetic truth and reproducibility.** Seeded generators make it possible to
  test recovery of known effects and probe failure modes such as confounding,
  censoring and biased exposure. Scripts regenerate datasets and evidence.
- **Benchmarking.** Methods are checked against reference libraries, independent
  calculations or known synthetic truth where applicable. Model evaluations
  compare against named baselines with chronological splits where time matters.
- **Uncertainty and decisions.** Reports include intervals, power, calibration or
  sensitivity checks as appropriate. Experiment decisions use pre-specified
  thresholds and guardrails, with inconclusive and negative results recorded.
- **Governance.** Study protocols document estimands, comparisons and acceptance
  criteria. Amendments and execution records preserve the history of changes,
  resource limits and failed attempts.

## Architecture

The Python package separates synthetic data generation and validation from statistical,
causal, experimentation, recommendation and customer-value modules. Shared utilities
provide seeding, resampling, validation and point-in-time data handling. Analytical
modules can be exercised independently. The architecture documents how future pillars
will combine their outputs. [Architecture overview](docs/architecture/overview.md)

| Path | Contents |
| --- | --- |
| `src/decision_intelligence/` | Python package, data generators and analytical methods |
| `data/samples/` and `data/schemas/` | Committed sample data and table contracts |
| `scripts/` | Dataset and evidence generation |
| `tests/` | Unit, integration and statistical-property tests |

## Installation

From the repository root, use Python 3.11 or 3.12, the versions covered by CI.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]" --config-settings editable_mode=compat
```

Regenerate the sample data and schemas with the seeded generator. It writes to
`data/samples/` and `data/schemas/`.

```bash
python scripts/generate_synthetic_dataset.py
```

## Testing

Run the same lint, type and test commands configured in the [CI workflow](.github/workflows/ci.yml).

```bash
ruff check .
mypy src
pytest
```

The suite covers estimator correctness, end-to-end workflows and statistical
properties such as ground-truth recovery, coverage and power.

## Documentation

Each evidence report describes the scenario, benchmarks, findings and limitations
and points to the scripts used to generate its results.

| Topic | Methodology | Evidence |
| --- | --- | --- |
| Statistical foundations | [Methods](docs/methodology/statistical-foundations.md) | [M3](docs/evidence/m3-statistical-foundations.md) |
| Causal adjustment | [Methods](docs/methodology/causal-foundations.md) | [M4](docs/evidence/m4-causal-foundations.md) |
| Observational designs | [Methods](docs/methodology/observational-causal-inference.md) | [M5](docs/evidence/m5-observational-causal-inference.md) |
| Experiment design and analysis | [Methods](docs/methodology/experimentation-fundamentals.md) | [M6](docs/evidence/m6-experimentation-fundamentals.md) |
| Advanced experimentation | [Methods](docs/methodology/advanced-experimentation.md) | [M7](docs/evidence/m7-advanced-experimentation.md) |
| Recommendation baselines | [Methods](docs/methodology/recommender-baselines.md) | [M8](docs/evidence/m8-recommendation-baselines.md) |
| Advanced retrieval and ranking | [Methods](docs/methodology/advanced-recommendation.md) | [M9](docs/evidence/m9-advanced-recommendation.md) |
| LTV foundations and customer economics | [Methods](docs/methodology/customer-economics.md) | [M10](docs/evidence/m10-ltv-foundations.md) |

## Licence

[MIT](LICENSE)
