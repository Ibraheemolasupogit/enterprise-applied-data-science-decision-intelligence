# Milestone 7 evidence — advanced experimentation

Status: **DONE** — approved and committed (`5e11c6d`; NumPy typing fix `31b93a7`); CI run
43407936369 on `31b93a7` passed on Python 3.11 and 3.12 (the run on `5e11c6d` failed before the
fix). Status line updated 2026-11-17; the evidence below is unchanged. At submission it read:
"implemented and validated locally; awaiting scientific review. Not committed or pushed."
Statuses: [capability-status.md](capability-status.md). Methodology:
[../methodology/advanced-experimentation.md](../methodology/advanced-experimentation.md).
Generated numbers: [m7-advanced-experimentation-results.md](m7-advanced-experimentation-results.md)
(`python scripts/generate_m7_evidence.py`; byte-reproducible apart from runtimes).

## Acceptance criteria

| Requirement | Evidence |
| --- | --- |
| CUPED with pre-treatment covariates; unbiasedness and variance-reduction diagnostics | `variance_reduction.py`; results §1–2 (bias ± MC SE, coverage, realised reduction vs R²) |
| Valid sequential methods: stopping boundaries, error control, repeated-peeking simulations | `sequential.py`; results §3 (published values), §4 (4,000 replicates per effect), §5 (case) |
| Bayesian A/B testing: explicit priors, posteriors, credible intervals, decision uncertainty | `bayesian.py`; results §6–7 |
| Selected advanced designs justified by the Brief (§7.4) | cluster-randomised planning and switchback (`designs.py`); results §8–9 |
| Additive scenarios with known truth, valid and invalid analyses | `data/advanced_experiment_scenarios.py`; invalid covariates, naive peeking, hourly switchback analysis, ICC-blind plans |
| Comparison with M6 fixed-horizon baselines under equivalent assumptions | every study runs the M6 analysis on the same simulated data; results §10 |
| Repeated-sampling type I error, power and coverage | results §2, §4, §7, §8, §9 |
| Business decisions with costs, benefits, uncertainty and policies | results §1 (days saved), §5 (early stop), §6 (expected loss in £), §8 (underpowered rollout avoided) |
| Independent references, unit and statistical-property tests | see below |
| Reproducible evidence; README, methodology, roadmap, architecture, registry updated | this file; generator run twice with identical output (runtimes excepted) |

## Scientific distinctions enforced in code and reports

- **Credible vs confidence intervals**: Bayesian functions return `credible_interval`; frequentist
  results keep the M6 `CausalEstimate` confidence interval. Results tables label each.
- **Posterior probabilities are not p-values**: `probability_above` is documented as such; the flat-prior
  numerical coincidence with the one-sided p-value is shown and explained (results §6), and the
  sceptical-prior counterexample is tested.
- **Repeated monitoring**: naive peeking is quantified (0.197 at 10 looks); valid designs and the
  mSPRT control α; Bayesian rules applied at every look inflate frequentist error (0.046 → 0.153).
- **Pre-treatment leakage / post-randomisation selection**: `check_adjustment_covariate` FAILs
  covariates measured after assignment; both planted invalid covariates are shown biasing estimates.
- **Randomisation vs analysis units**: cluster planning counts clusters, not users; switchback
  analysis uses periods, and hour-level analyses are shown wrong in both directions.

## Headline results

| Finding | Number |
| --- | --- |
| CUPED case: estimate (truth 0.600) | CUPED 0.581 [0.227, 0.934] vs unadjusted 0.394 [−0.045, 0.834]; variance −35.3% |
| CUPED repeated sampling (400) | bias −0.011 ± 0.015; coverage 0.953; power 0.487 vs 0.340; A/A 0.048; reduction 0.339 = R² 0.339 |
| Invalid covariates | in-experiment sessions bias −2.79 ± 0.01 (coverage 0); leaky window −0.114 ± 0.013 |
| CUPED planning | 66% of the users; 75 vs 112 days; M6 rule HOLD → LAUNCH on the same data |
| Boundaries vs published | largest difference 6.3e-4 (rounding); recursion vs SciPy MVN within Genz error (~1e-7; tested ≤ 1e-6) |
| Peeking, 10 looks (size) | naive 0.197; O'Brien–Fleming 0.051; Pocock 0.049; mSPRT 0.015; fixed 0.050 |
| Power at the design effect / expected sample | fixed 0.803 / 1; OBF 0.789 / 0.77; Pocock 0.705 / 0.69; mSPRT 0.527 / 0.78 |
| Estimates when rejecting (truth 0.015) | fixed horizon 0.0169 (significance filter); OBF 0.0191; Pocock 0.0231; mSPRT 0.0245 |
| Sequential case | stops on day 28 of 42 (Z 3.37 ≥ 2.96); repeated CI [0.106, 1.636] covers 0.600 |
| Bayesian conversion | P(better) 0.96, P(> break-even) 0.025 → STOP (M6 STOP) |
| Bayesian revenue | vague LAUNCH; sceptical HOLD; optimistic LAUNCH (M6 LAUNCH) |
| Bayesian rule at fixed null | 0.046 once; 0.153 at 10 looks |
| Bayesian calibration | 0.949 once; 0.952 optional stopping; wrong prior: launched effects overstated 3× |
| Cluster planning | ICC 0.0126 [0.0079, 0.0176]; 127 vs 57 cities per arm; CR1 power 0.802 (planned 0.803) vs 0.482 |
| Switchback (400 markets) | bias 0 h −1.50 ± 0.08; 2 h +0.04 ± 0.09 (coverage 0.955); A/A randomisation test 0.040; hourly Welch 0.297 / 0.003 |

## Independent reference comparisons (tests and results §3)

| Component | Reference | Agreement |
| --- | --- | --- |
| Pocock and O'Brien–Fleming constants, K = 2–5, 10 | Jennison & Turnbull (2000) Table 2.3 | ≤ 6e-4 (published to 3 dp) |
| Lan–DeMets O'Brien–Fleming, five looks | gsDesign / rpact / ldbounds | ≤ 2e-4 |
| Repeated 5% tests, K = 2–20 | Armitage, McPherson & Rowe (1969) | ≤ 6e-4 |
| Maximum-sample inflation R | Jennison & Turnbull Tables 2.2/2.4 | ≤ 1.5e-3 |
| Crossing probabilities (size, power) | SciPy multivariate normal (Genz); simulation | ≤ 1e-6; within 4 MC SE |
| Spent alpha | the spending function at unequal looks | ≤ 1e-9 |
| mSPRT log-likelihood ratio; confidence sequence | quadrature of the mixture; test inversion | ≤ 1e-7; exact |
| CUPED | manual formula | ≤ 1e-10 |
| Lin regression adjustment | statsmodels OLS HC2 | estimate ≤ 1e-10; SE ≤ 1e-8 relative |
| P(p_t > p_c), Beta posteriors | closed form (Miller 2015) | ≤ 1e-8 |
| Credible interval, tail probability, expected loss | 1M-draw Monte Carlo | within MC error |
| Normal-normal posterior and expected shortfall | grid integration of prior × likelihood | ≤ 1e-7 |
| ICC (ANOVA) | SciPy one-way ANOVA F identity | ≤ 1e-10 relative |
| Stratified randomisation p-value | full enumeration | within 0.01 (MC) |
| Unblocked randomisation p-value | M3 permutation test | identical |

## Tests added (86; suite 495 → 581)

| File | Tests | Covers |
| --- | --- | --- |
| `tests/unit/experimentation/test_sequential.py` | 32 | spending functions, published constants, exact spending, MVN and simulation references, peeking, inflation, monitoring, validation, mSPRT |
| `tests/unit/experimentation/test_variance_reduction.py` | 8 | manual CUPED, statsmodels HC2, ρ², uninformative covariate, missing covariates, validation, timing check, planning |
| `tests/unit/experimentation/test_bayesian.py` | 11 | conjugacy, closed form, Monte Carlo, identity, grid integration, flat-prior coincidence vs sceptical counterexample, decisions |
| `tests/unit/experimentation/test_designs.py` | 13 | ICC identity and consistency, cluster power and plans, blocked assignment, randomisation p-values, exact recovery with burn-in, validation |
| `tests/unit/data/test_advanced_experiment_scenarios.py` | 8 | determinism, truth invariants, new users, planted defects, carryover, validation |
| `tests/statistical/test_m7_advanced_properties.py` | 8 | CUPED bias/coverage/size/power, invalid adjustments, peeking, Bayesian peeking and calibration, switchback, cluster power |
| `tests/integration/test_m7_case_studies.py` | 6 | each case end to end, published references, reproducibility |

(Counts are pytest items, including parametrised cases.)

## Changes outside new M7 files

- `experimentation/__init__.py` — package docstring lists the M7 modules.
- README, architecture overview, capability registry, roadmap (M6 marked DONE, M7 IMPLEMENTED —
  awaiting review) and the M6 methodology's scope note (now points to M7).
- No M2–M6 implementation, contract, test or evidence file was changed; no dependency added.

## Problems found and corrected during implementation

1. **Lan–DeMets O'Brien–Fleming convention.** The first implementation applied the two-sided
   spending function to the total α (first of five boundaries 4.38). Published software
   (gsDesign, rpact, ldbounds) applies the one-sided function per side at α/2 (4.877). Switched to
   the software convention and documented both.
2. **Quadrature on narrow posteriors.** With thousands of users the Beta posteriors are ~0.003 wide
   on [0, 1]; integration is restricted to the control posterior's effective support. Verified
   against the closed form from 40 to 100,000 users per arm.
3. **Switchback analysis.** A fully interacted regression on time-of-day blocks failed when a block
   had a single period in one arm (leverage 1). Replaced by blocking in the *design* (stratified
   randomisation) and the M6 stratified estimator, with a within-block randomisation test.
4. **A first switchback configuration** had carryover too small relative to time-of-day noise to
   demonstrate burn-in bias; carryover was set comparable to the direct effect (stated).
5. **A Bayesian rate of 0.028** (400 replicates) for a rule expected near 0.05 was traced to chance
   in that draw (P ≈ 0.02; the one-sided z-test on the same draws gave the same 0.0275); the study
   uses 1,000 replicates (0.046).

## Validation (local)

`ruff check .` passed; strict `mypy src` passed (71 files); `pytest`: 581 passed (Python 3.12.8,
NumPy 2.5.4, SciPy 1.18.1, pandas 2.3.3, statsmodels 0.15.0, mypy 2.4.1). The M7 evidence generator
was run twice with identical output apart from runtimes (about 3 minutes). Python 3.11 is exercised
by CI only. The M7 tests (86) also pass at the declared dependency floor (SciPy 1.13.1, NumPy
2.0.2, pandas 2.2.3, statsmodels 0.14.4; Python 3.12), checked in a scratch environment after the
review fix below.

## Scientific assurance review (pre-approval)

An independent review re-derived the methods, cross-checked them against references and probed
edge cases. Additional independent checks: Lan–DeMets Pocock-type boundaries (2.438, 2.427, 2.410,
2.397, 2.386) match ldbounds/gsDesign; size at unequal looks (0.15, 0.40, 0.55, 1) 0.0498 by
400,000-path simulation vs 0.0500 exact; simultaneous repeated-CI coverage 0.950; the mSPRT
crossing rate over 2,000 looks 0.027 ≤ 0.05 (Ville); Beta posteriors match the closed form or
2M-draw Monte Carlo for 0/10 v 0/10, 1/1 v 0/1, 3/40 v 500/100,000 and Jeffreys priors;
CUPED/Lin coverage 0.94–0.96 with 80/20 allocation, heterogeneous effects and 15% missing
covariates (n = 200 and 2,000); Lin with missing indicators equals statsmodels HC2 to 1e-15;
cluster plans are the smallest cluster count reaching the target (brute force over 36 inputs).

Defects found and corrected (each with a regression test that fails on the original code):

1. `monitor` reported the terminal decision "do not reject H0" before the maximum information was
   reached; it now reports `continue` until the final look.
2. `monitor` accepted non-finite estimates (NaN never crosses a boundary) and returned a decision;
   it now refuses them.
3. `randomisation_p_value` with a stratum lacking one arm returned the minimum p-value
   `1 / (m + 1)` (every contrast NaN, none "exceeded") — spurious significance; it now refuses.
4. `period_means` labelled a period crossing midnight by its smallest kept hour (5-hour period
   from 20:00 → block 0 instead of 4); blocks now come from the period's start hour. Periods that
   divide 24, as in all M7 evidence, were unaffected.
5. `estimate_icc` raised `ZeroDivisionError` when every cluster had one unit; it now raises a
   `ValueError` explaining that within-cluster variance is undefined.
6. `normal_effect_posterior` / `NormalPrior` accepted NaN or infinite inputs, propagating to a
   silent HOLD; they now refuse them.
7. The MVN reference check passed `rng=` to `scipy.stats.multivariate_normal.cdf`, which SciPy
   1.13 (the declared minimum) rejects: 4 M7 tests failed there. The argument is removed; the Genz
   integration error (~1e-7) is below the reported precision.
8. Documentation: recursion-vs-MVN agreement was stated as 1e-8 (it is ~1e-7, the Genz error);
   the exaggeration of estimates among rejecting experiments was attributed to early stopping alone
   although the fixed-horizon test shows it too (0.0169 vs truth 0.015); the mSPRT check interval
   was misstated as 10% (it is 1%) in the methodology.

No reported number changed: the evidence regenerates identically apart from the corrected sentence
in §4 and runtimes.

## Limitations

See the methodology's "Known limitations". Most material: synthetic validation only; normal
approximations behind sequential and normal-normal Bayesian analyses; no futility boundaries or
bias-adjusted estimates after early stopping; delayed outcomes handled only by complete windows;
priors and economics are illustrative assumptions; switchback assumes carryover shorter than the
burn-in; geo, interference, bandits, heterogeneous and long-term effects not implemented.
