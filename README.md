# Enterprise Applied Data Science & Decision Intelligence Platform

A Python-first, locally reproducible, cloud-neutral platform demonstrating transferable statistical,
causal, experimentation, personalisation, optimisation and decision-science methods — the evidence
behind business decisions, not an industry-specific application.

> From evidence to prediction, from prediction to causal understanding, and from causal understanding
> to optimised business decisions.

**Status: through Milestone 4 (awaiting review). The synthetic data foundation (M2), Pillar 1's
statistical foundations (M3) and its causal foundations with backdoor adjustment (M4) are implemented
and tested; observational causal designs (M5) and every other pillar's methods are not yet built.** See
[docs/evidence/capability-status.md](docs/evidence/capability-status.md) for the authoritative,
per-pillar status table — treat any claim elsewhere that contradicts it as an error.

## Six permanent pillars

| # | Pillar                                | Central question                                                                  |
| - | ---------------------------------------- | -------------------------------------------------------------------------------------- |
| 1 | Statistics & Causal Inference           | What would have happened if the intervention had not occurred?                        |
| 2 | A/B Testing & Experimentation            | Did the intervention cause an improvement, and is the evidence strong enough to act on?|
| 3 | Recommender Systems                      | Which items should be presented to which users, and does that create real value?       |
| 4 | Mathematical Optimisation / Gurobi      | Given limited resources, what allocation or schedule best achieves the objective?      |
| 5 | Customer Lifetime Value                  | What is a customer economically worth over time, and how should that guide investment? |
| 6 | Product Sense & Decision Science         | What should the organisation do, given the evidence and the cost of being wrong?        |

Time-series forecasting is a supporting capability used by several pillars; it is **not** a seventh
pillar.

Full pillar scope, architecture and sequencing: [docs/architecture/overview.md](docs/architecture/overview.md)
and [docs/milestones/roadmap.md](docs/milestones/roadmap.md).

## Integrated scenario

A fictional digital marketplace/subscription platform (provisional internal name: *NexaMarket*) is
the flagship integrated case study, composing selected pillar outputs into one evidence-to-decision
workflow. It is introduced at Milestone 17, alongside independent cross-industry case studies that
demonstrate methods that do not naturally fit that scenario. Neither exists yet.

## Development

```bash
pip install -e ".[dev]" --config-settings editable_mode=compat
ruff check .
mypy src
pytest
```

The `editable_mode=compat` setting works around a known interaction where hatchling's editable-install
`.pth` file can be marked OS-hidden (observed on macOS) and silently skipped by CPython's `site.py`;
`tests/conftest.py` also inserts `src/` directly so test collection never depends on that install path
working. Requires Python 3.11 or 3.12.

## Synthetic data foundation

Every pillar and the NexaMarket flagship build on one deterministic synthetic dataset — 13 tables
covering customers, transactions, retention, a confounded observational campaign, an unconfounded
experiment, recommendation exposures and a brute-force-verified optimisation reference problem, with
ground truth deliberately encoded throughout. Full design, assumptions and limitations:
[docs/methodology/synthetic-data.md](docs/methodology/synthetic-data.md).

```bash
python scripts/generate_synthetic_dataset.py   # writes a small sample to data/samples/ and data/schemas/
```

## Statistical foundations (Milestone 3)

Reusable, tested statistical primitives under `src/decision_intelligence/statistics/`, with the
shared bootstrap and input-validation helpers in `common/`:

- probability distributions and sampling (inverse-transform, rejection, simple random and stratified
  sampling);
- descriptive statistics with stated conventions and numerically stable moments;
- estimator bias/variance/MSE, sampling distributions and Central Limit Theorem diagnostics;
- bootstrap confidence intervals (percentile, basic, BCa);
- parametric and non-parametric hypothesis tests (t, proportions, χ², exact/asymptotic rank tests,
  permutation, ANOVA/Kruskal–Wallis), each returning its estimand, interval and assumptions;
- effect sizes with intervals; power, sample size and minimum detectable effect;
- OLS with classical, heteroskedasticity-robust and cluster-robust inference plus diagnostics;
  logistic regression with separation detection and calibration diagnostics;
- bias–variance, missing-data (MCAR/MAR/MNAR) and Monte Carlo simulation studies.

Every method is checked against SciPy or statsmodels as a reference implementation and against its
repeated-sampling properties (type I error, coverage, power). The methods are applied to real
questions on the M2 data — e.g. the randomised experiment's estimated lift is +3.1 points (95% CI
−0.7 to +6.9) against a true +4.0, an experiment shown to have had only ≈53% power. Group
comparisons and coefficients on observational data are reported as associations; causal adjustment
starts in M4.

```bash
python scripts/generate_m3_evidence.py   # regenerates docs/evidence/m3-statistical-foundations-results.md
```

Methodology: [docs/methodology/statistical-foundations.md](docs/methodology/statistical-foundations.md).
Evidence: [docs/evidence/m3-statistical-foundations.md](docs/evidence/m3-statistical-foundations.md).

## Causal foundations (Milestone 4)

`src/decision_intelligence/causal_inference/` asks what retention would have been without the M2
retention offer, a campaign deliberately targeted at customers who would have retained anyway:

- potential outcomes and estimands (ATE, ATT, conditional effects) and the exact decomposition of a
  naive comparison into effect plus selection bias;
- causal DAGs with d-separation, the backdoor criterion and confounder/mediator/collider roles (the
  M2 generator's DAG says `latent_value` must be adjusted for, and the proxies are not enough);
- regression adjustment, inverse probability weighting and doubly robust AIPW, with overlap, balance
  and weight diagnostics;
- negative controls, a DAG falsification test, E-values and bias-formula sensitivity analysis;
- deliberate failures: unmeasured confounding, post-treatment adjustment and selection, collider
  (M-bias) and mediator adjustment, positivity violations, model misspecification and interference.

The true effect is +8 points; the naive comparison says about +19–23. Valid adjustment is unbiased
across 200 independent M2 draws (AIPW bias −0.001, 95% CI coverage 0.93), while adjusting only for
observed proxies still overstates the effect (+0.028) — and on the canonical seed most valid
intervals narrowly miss the truth while invalid ones cover it, which the evidence reports rather
than hides.

```bash
python scripts/generate_m4_evidence.py   # regenerates docs/evidence/m4-causal-foundations-results.md
```

Methodology: [docs/methodology/causal-foundations.md](docs/methodology/causal-foundations.md).
Evidence: [docs/evidence/m4-causal-foundations.md](docs/evidence/m4-causal-foundations.md).

## Repository layout

```
src/decision_intelligence/   Python package: common/ (shared primitives), data/ (synthetic foundation),
                              statistics/ (Pillar 1 statistical foundations),
                              causal_inference/ (Pillar 1 causal foundations)
data/samples/                 Small, committed sample dataset (regenerate with scripts/)
data/schemas/                 Table contracts as JSON, generated by the same script
tests/                        Unit, integration and statistical-property tests
docs/architecture/            Architecture-of-record
docs/methodology/              Per-capability methodology write-ups (synthetic data, statistics, causal foundations)
docs/milestones/               Agreed roadmap
docs/evidence/                 Capability-status registry and milestone evidence
scripts/                       Reproducible evidence-generation scripts
.github/workflows/             CI (lint, type-check, test on Python 3.11 and 3.12)
```

The remaining pillar packages (`experimentation/`, `recommendations/`,
`optimisation/`, `customer_value/`, `product_analytics/`, `decision_science/`, `forecasting/`) and
`case_studies/` are intentionally not scaffolded yet — each is created by the milestone that first
needs it, to avoid placeholder directories with no content.

## Relationship to other repositories

This repository is one of a small set of substantial portfolio repositories. Others demonstrate
industry-specific end-to-end platforms (healthcare, aviation, banking/fintech, manufacturing, energy,
product growth); this one demonstrates the cross-industry scientific and mathematical methods those
platforms draw on. It reuses validated *patterns* from
[`azure-product-growth-intelligence-platform`](https://github.com/Ibraheemolasupogit/azure-product-growth-intelligence-platform)
(e.g. experimentation and recommendation pipeline shape, evidence-doc conventions) but is not a
dependency of, or a duplicate of, that repository.

## License

MIT — see [LICENSE](LICENSE).
