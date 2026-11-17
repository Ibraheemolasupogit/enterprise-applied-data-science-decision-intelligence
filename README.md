# Enterprise Applied Data Science & Decision Intelligence Platform

A Python-first, locally reproducible, cloud-neutral platform demonstrating transferable statistical,
causal, experimentation, personalisation, optimisation and decision-science methods — the evidence
behind business decisions, not an industry-specific application.

> From evidence to prediction, from prediction to causal understanding, and from causal understanding
> to optimised business decisions.

**Status: through Milestone 10 (approved; M11 planned). The synthetic data foundation (M2), Pillar 1 —
statistical foundations (M3), causal foundations with backdoor adjustment (M4) and observational
causal designs (M5) — Pillar 2's experimentation fundamentals (M6) and advanced experimentation
(M7), Pillar 3's recommendation baselines (M8) and advanced recommendation and ranking (M9), and
Pillar 5's LTV foundations and customer economics (M10) are implemented and tested; every other
pillar's methods are not yet built.** See
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

## Observational causal inference (Milestone 5)

Designs that identify effects without measuring every confounder, each with its own untestable
assumption, explicit diagnostics and a deliberately broken counterpart:

- **Difference-in-differences and event studies** for a staggered regional policy rollout:
  Callaway–Sant'Anna group-time ATTs, an exact decomposition showing why two-way fixed effects goes
  wrong under staggered adoption (negative weights), pre-trend tests that are honest about their
  power, and a wild cluster bootstrap for few clusters;
- **Synthetic control** for a single treated region, with fit diagnostics, placebo ranks reported as
  ranks, and leave-one-out sensitivity;
- **Sharp regression discontinuity** for a loyalty tier at a score threshold, with robust
  bias-corrected inference, bandwidth sensitivity, manipulation and covariate tests — and the
  estimand stated as the effect *at the cutoff*, not population-wide.

The datasets are new, additive generators (`data/rollout_panel.py`, `data/synthetic_control_panel.py`,
`data/threshold_assignment.py`); no M2 contract changed. Over 150 simulated rollouts the
Callaway–Sant'Anna estimate is unbiased (coverage 0.97) while the standard two-way fixed-effects
coefficient understates the effect by about a fifth (coverage 0.01).

```bash
python scripts/generate_m5_evidence.py   # regenerates docs/evidence/m5-observational-causal-inference-results.md
```

Methodology: [docs/methodology/observational-causal-inference.md](docs/methodology/observational-causal-inference.md).
Evidence: [docs/evidence/m5-observational-causal-inference.md](docs/evidence/m5-observational-causal-inference.md).

## Experimentation fundamentals (Milestone 6)

`src/decision_intelligence/experimentation/` asks whether a randomised change caused an improvement
and whether the evidence is strong enough to act on — and checks that the experiment that ran is the
one that was designed:

- a pre-specified design that fixes the experimental, randomisation and analysis units, the estimand,
  the primary metric, guardrail margins and the minimum worthwhile effect;
- deterministic hash-based assignment and seeded complete, stratified and cluster randomisation;
- sample ratio mismatch detection — explicitly separated from diagnosing its cause;
- power, sample size, minimum detectable effect and design effect (built on M3);
- intention-to-treat analysis of binary and continuous metrics with absolute and relative effects,
  stratified, clustered and A/B/n analyses, and multiple-testing corrections;
- noncompliance (complier effects) and attrition (Lee and Manski bounds), with the naive as-treated,
  per-protocol and complete-case comparisons shown failing;
- validity checks and pre-specified LAUNCH / HOLD / ITERATE / STOP rules with guardrails.

The M2 experiment's lift is +3.1 points (95% CI −0.7 to +6.9): the rule says **HOLD**, not "no
effect" — the true lift is +4.0, and re-running the experiment 1,000 times shows the design detects
it only about half of the time. A larger synthetic experiment launches when clean and is held when
post-randomisation data loss produces a sample ratio mismatch.

```bash
python scripts/generate_m6_evidence.py   # regenerates docs/evidence/m6-experimentation-fundamentals-results.md
```

Methodology: [docs/methodology/experimentation-fundamentals.md](docs/methodology/experimentation-fundamentals.md).
Evidence: [docs/evidence/m6-experimentation-fundamentals.md](docs/evidence/m6-experimentation-fundamentals.md).

## Advanced experimentation (Milestone 7)

M7 extends the M6 fixed-horizon analysis in directions that answer *different* questions, each
compared with the M6 baseline on the same data:

- **CUPED and regression adjustment** (Lin 2013) with pre-treatment covariates — same estimand,
  smaller standard error; covariates measured after assignment are refused by a timing check;
- **sequential monitoring** — Lan–DeMets alpha-spending and classical group-sequential boundaries
  (reproducing published constants), repeated confidence intervals, and always-valid inference
  (mixture SPRT, confidence sequences); naive peeking is shown inflating false positives;
- **Bayesian A/B analysis** with explicit priors, credible intervals (distinguished from confidence
  intervals), posterior probabilities (not p-values) and expected-loss decisions in money;
- **selected designs** — cluster-randomised planning from an estimated ICC, and switchback
  experiments with carryover, burn-in, time-of-day blocking and randomisation inference.

On the synthetic revenue experiment CUPED cuts the variance by 35%, so the same power needs 66% of
the users (75 instead of 112 days); an O'Brien–Fleming design stops on day 28 of 42; ten naive peeks
raise the false-positive rate from 0.050 to 0.197; ignoring a city-level ICC of 0.013 would have
halved the number of cities and left the rollout with 48% power.

```bash
python scripts/generate_m7_evidence.py   # regenerates docs/evidence/m7-advanced-experimentation-results.md
```

Methodology: [docs/methodology/advanced-experimentation.md](docs/methodology/advanced-experimentation.md).
Evidence: [docs/evidence/m7-advanced-experimentation.md](docs/evidence/m7-advanced-experimentation.md).

## Recommendation baselines (Milestone 8)

M8 starts Pillar 3 with the benchmarks later recommenders must beat — random, global and
segment popularity, item- and user-kNN, content-based filtering, PureSVD matrix factorisation and
a bias model for explicit ratings — and an evaluation protocol designed not to fool itself:

- a dedicated synthetic scenario with item-level preferences, position bias, an exposure-biased
  logging policy with logged propensities, cold-start users and items, and ratings missing not at
  random; ground truth is kept out of every training, tuning and selection interface;
- a strict chronological train / validation / test protocol over the full eligible catalogue,
  with deterministic tie-breaking and a perturbation test showing every model ignores post-cutoff
  data;
- three views kept apart — logged feedback, a synthetic oracle, and simulated online value — plus
  position-aware IPS with positivity, effective-sample-size and clipping diagnostics;
- coverage, novelty, diversity, exposure concentration, category exposure and segment-level
  results, with bootstrap intervals and Holm-adjusted paired comparisons.

Collaborative baselines beat popularity in every view under the default logs (oracle NDCG@10 0.59
vs 0.43). Logged NDCG ranks content-based filtering below global popularity while its simulated
online value is higher in 5 of 9 pre-specified logging regimes, in five further seeds and on a
never-inspected seed — offline ranking gains need not be online gains. When the logging policy
never shows an item, IPS can only estimate the supported part of a list's value, and its normal
intervals under-cover (about 0.92–0.93 at a nominal 0.95).

```bash
python scripts/generate_m8_evidence.py   # regenerates docs/evidence/m8-recommendation-baselines-results.md
```

Methodology: [docs/methodology/recommender-baselines.md](docs/methodology/recommender-baselines.md).
Evidence: [docs/evidence/m8-recommendation-baselines.md](docs/evidence/m8-recommendation-baselines.md).

## Advanced recommendation and ranking (Milestone 9)

M9 compares implicit ALS, BPR (uniform and exposure-aware negatives), EASE, a shallow two-tower
model with exact and approximate retrieval, two-stage pointwise and LambdaRank-style ranking
(naive and propensity-weighted) and a score-blend hybrid with the tuned M8 baselines, under an
evaluation protocol frozen in code before any model was evaluated. It adds cold-start strategies,
SNIPS and doubly robust off-policy estimation with studentised bootstrap intervals, operational
benchmarks with preflight memory checks, and a **SIMULATED** user-randomised A/B test analysed
with the M6/M7 components.

The results are mostly modest or negative, and are reported as such: no advanced model reliably
beats the tuned M8 item-kNN (on a never-inspected seed item-kNN is best on oracle and simulated
online value); exposure-aware negatives and the two-tower model underperform; propensity
weighting changes little; off-policy estimators cannot resolve gaps of 1–2%; the studentised
bootstrap raises but does not restore 95% coverage; and both simulated A/B comparisons end in
HOLD — uninformatively, because the frozen A/B design had only about 7% power at its own
minimum worthwhile effect (about 17,600 users per arm would be needed for 80%).

```bash
python scripts/generate_m9_evidence.py   # regenerates docs/evidence/m9-advanced-recommendation-results.md
```

Methodology: [docs/methodology/advanced-recommendation.md](docs/methodology/advanced-recommendation.md).
Evidence: [docs/evidence/m9-advanced-recommendation.md](docs/evidence/m9-advanced-recommendation.md).

## LTV foundations and customer economics (Milestone 10)

M10 starts Pillar 5 on a contribution basis — revenue minus cost of goods, payment fees and
cost-to-serve — and keeps **realised**, **projected** and **expected lifetime** contribution
apart. On an additive synthetic subscription business with known truth it provides a
point-in-time customer panel, discrete-time Kaplan–Meier retention with Greenwood intervals,
cohort-matched realised LTV, heuristic LTV (from a legacy revenue baseline to a Kaplan–Meier
projection) with explicit undefined and unstable states, CAC, payback, LTV/CAC with a precision
target set before the main results, segment economics, sensitivity analysis and backtests that
refuse incomplete windows.

The Kaplan–Meier projection is accurate over 24–36 months (mean absolute bias about 1%) while
constant-churn heuristics over-project by 7–19% and every heuristic misjudges expected lifetime
value when customers differ in churn; LTV/CAC intervals under-cover the true LTV/CPA (0.79–0.85)
because they treat acquisition spend as known; the cheapest acquisition channel is not the most
valuable (a designed-in mechanism). Nothing here is incremental value — that is Milestone 11.

```bash
python scripts/generate_m10_evidence.py   # regenerates docs/evidence/m10-ltv-foundations-results.md
```

Methodology: [docs/methodology/customer-economics.md](docs/methodology/customer-economics.md).
Evidence: [docs/evidence/m10-ltv-foundations.md](docs/evidence/m10-ltv-foundations.md).

## Repository layout

```
src/decision_intelligence/   Python package: common/ (shared primitives), data/ (synthetic foundation),
                              statistics/ (Pillar 1 statistical foundations),
                              causal_inference/ (Pillar 1 causal and observational methods),
                              experimentation/ (Pillar 2 experimentation: M6 fundamentals, M7 advanced),
                              recommendations/ (Pillar 3: M8 baselines, M9 advanced ranking),
                              customer_value/ (Pillar 5: M10 LTV foundations)
data/samples/                 Small, committed sample dataset (regenerate with scripts/)
data/schemas/                 Table contracts as JSON, generated by the same script
tests/                        Unit, integration and statistical-property tests
docs/architecture/            Architecture-of-record
docs/methodology/              Per-capability methodology write-ups (synthetic data, statistics, causal, observational, experimentation, advanced experimentation, recommender baselines, advanced recommendation, customer economics)
docs/milestones/               Agreed roadmap
docs/evidence/                 Capability-status registry and milestone evidence
scripts/                       Reproducible evidence-generation scripts
.github/workflows/             CI (lint, type-check, test on Python 3.11 and 3.12)
```

The remaining pillar packages (
`optimisation/`, `product_analytics/`, `decision_science/`, `forecasting/`) and
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
