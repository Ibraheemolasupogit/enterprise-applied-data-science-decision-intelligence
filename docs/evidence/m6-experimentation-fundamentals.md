# Milestone 6 evidence — experimentation fundamentals

Status: implemented and validated locally; **awaiting scientific review**. Not committed or pushed.
Statuses: [capability-status.md](capability-status.md). Methodology:
[../methodology/experimentation-fundamentals.md](../methodology/experimentation-fundamentals.md).
Generated numbers: [m6-experimentation-fundamentals-results.md](m6-experimentation-fundamentals-results.md)
(`python scripts/generate_m6_evidence.py`; byte-reproducible apart from runtimes).

## Acceptance criteria

| Requirement | Evidence |
| --- | --- |
| Deterministic assignment, unit-level allocation, reproducible seeding, stratified randomisation | `assignment.py` (hash, complete, stratified, cluster); `tests/unit/experimentation/test_assignment_and_srm.py` |
| SRM detection with goodness-of-fit tests and expected ratios; detection vs diagnosis | `srm.py`; results §3, §5 |
| Planning: baseline, effect size, power, MDE, sample size (reusing M3) | `planning.py` wraps `statistics.power`; results §1, §6 |
| A/B analysis for binary and continuous outcomes with intervals and tests | `analysis.py`; results §1, §3, §4 |
| ITT; assignment vs treatment received; noncompliance and attrition risks | `compliance.py`; results §7 |
| Validity: duplicates, leakage, pre-treatment imbalance, exclusions, metric integrity, post-randomisation selection | `validity.py`; results §1, §3, §8, §10 |
| M2 randomised experiment plus additive synthetic scenarios with known truth | `cases.analyse_m2_experiment`, `data/experiment_scenarios.py`; results §1–3 |
| Business decisions, practical significance, uncertainty | `decision.py`; results §1, §3, §9 |
| Failures of naive analysis | as-treated, per-protocol, complete-case, per-user analysis of cluster randomisation, post-randomisation selection, unadjusted multiple testing, "non-significant = no effect"; results §2, §7–9 |
| Independent references and repeated-sampling validation with Monte Carlo uncertainty | see below; every rate with a Wilson interval, every bias with a Monte Carlo SE |
| Reuse of M3/M4/M5 | M3 tests, intervals, power, bootstrap, OLS; M4 `CausalEstimate`, `CausalData`, `balance_table`, `ReplicateSummary`; M5 `cluster_robust_t_test`, `summarise_replicates`, `rejection_rate` |

## Explicit definitions (M2 experiment)

Experimental unit, randomisation unit and analysis unit: the customer. Estimand: ITT — the
difference in 14-day conversion probability between assignment to the recommendation feature and
to control, over customers eligible on the assignment date. Primary metric: 14-day conversion.
Minimum worthwhile effect: +0.02, the break-even effect under stated, illustrative economics.
Assignment is preserved: every analysis groups units by the arm they were randomised to.

## Headline results

| Finding | Number |
| --- | --- |
| M2: validity checks | duplicates PASS; SRM PASS (p = 0.929); balance WARN (tenure SMD +0.166, joint p = 0.012 — chance, reported) |
| M2: primary ITT | +0.031 [−0.007, +0.069], p = 0.109; relative +29% [−6%, +77%] |
| M2: secondary (true effects 0) | revenue +0.096 [−0.31, +0.50]; sessions +0.0002; Holm p = 1 |
| M2: design | MDE 0.055 at the realised 566/569; 1,035/arm needed for +0.04, 3,841 for +0.02 |
| M2: decision | **HOLD** — inconclusive, not evidence of no effect; annual value −£40,613 to +£73,941 |
| M2 re-run 1,000 times (truth +0.04) | coverage 0.947; detected 0.523 (planned 0.507); significant estimates average 0.056 (type M) |
| M2 re-run 1,000 times (truth 0) | coverage 0.946; false positives 0.054; LAUNCH 0.029 |
| Scenario, clean run | LAUNCH: +0.0206 [0.0131, 0.0280] vs truth 0.0179; guardrails non-inferior |
| Scenario, defective run | HOLD: SRM (p = 1.3e-5) and conflicting duplicates; lost units were less engaged beforehand |
| Coverage, 300 populations | conversion 0.940; relative lift 0.943; revenue 0.937; stratified 0.950 |
| A/A false positives at 20,000 users | 0.053 / 0.060 / 0.067 (z / Welch / BCa) |
| Small skewed samples (100 per arm) | Welch 0.022 (conservative), BCa bootstrap 0.144 (anti-conservative) |
| Simulated power at planned n | 0.786–0.797 vs 0.80 target |
| SRM false-alarm rate (p ≤ 0.001) | 0.003 |
| 5% loss: detection / bias, random vs outcome-related | 0.963 / −0.0001 vs 0.977 / +0.0069 |
| 2% outcome-related loss | detected 0.000; bias +0.0026 (≈14% of the effect) |
| Noncompliance: ITT / CACE coverage | 0.953 / 0.953 |
| As-treated / per-protocol vs ATE 0.018 | 0.064 / 0.047 (coverage 0) |
| Attrition: complete-case bias; bounds | +0.0060 ± 0.0004 (coverage 0.805); Lee width 0.140, Manski 0.350, both contain targets |
| City-randomised A/A, naive per-user analysis | 0.346 (40 cities), 0.596 (10 cities); CR1 0.042 / 0.072; cluster means 0.036 / 0.042 |
| Analysing only "engaged" users | bias −0.0094 ± 0.0004 (half the effect) |
| 10 metrics, unadjusted / Holm / BH | FWER 0.293 / 0.038 / 0.073; power 0.598 / 0.275 / 0.326 |
| Decision rule, LAUNCH at effect −0.005 / 0 / δ / 2δ | 0.000 / 0.003 / 0.510 / 0.998 |

## Independent reference comparisons (tests)

| Component | Reference | Agreement |
| --- | --- | --- |
| Hash assignment | SHA-256 of the documented key | exact |
| SRM χ² | `scipy.stats.chisquare` (2 and 3 arms) | exact |
| SRM exact | direct binomial-probability sum | exact |
| Binary plan | Fleiss closed form | within 1 unit per arm |
| Continuous plan / MDE | statsmodels `TTestIndPower` | within 1 unit / 1e-4 relative |
| Two-proportion test and Newcombe interval | statsmodels `proportions_ztest`, `confint_proportions_2indep(method="newcomb")` | ≤ 1e-10 / 1e-12 |
| Welch test | `scipy.stats.ttest_ind(equal_var=False)` | ≤ 1e-10 |
| Relative-lift delta-method SE | simulated sampling SD of the log ratio | within 6% |
| Stratified estimator | manual computation | exact |
| CR1 cluster SE | statsmodels OLS `cov_type="cluster"` | ≤ 1e-8 |
| CACE | two-stage least squares (point); 2SLS HC0 sandwich (SE) | exact; exact up to `n/(n−1)` |
| Lee and Manski bounds | hand-computed examples | exact |
| Multiple-testing corrections | statsmodels `multipletests` (4 methods × 3 seeds) | ≤ 1e-12 |
| Noncentral-t power at large ncp (M3 fix) | statsmodels `TTestIndPower`; NaN tails injected | finite; 1e-4 relative; fallback error ≤ 1e-12, otherwise raises |

## Tests added (95; suite 400 → 495)

| File | Tests | Covers |
| --- | --- | --- |
| `tests/unit/stats/test_multiple_testing.py` | 15 | statsmodels identity, ordering/ties, validation, inclusive rejection at p = α |
| `tests/unit/stats/test_power.py` (3 added) | 3 | regression: finite power and MDE at large noncentrality; injected-NaN fallback bounded by 1e-12 (both tails); non-negligible NaN tail raises |
| `tests/unit/data/test_experiment_scenarios.py` | 14 | determinism, truth invariants (ITT = complier share × CACE), loss/attrition/duplicate switches, validation |
| `tests/unit/experimentation/test_assignment_and_srm.py` | 14 | hashing, order/stability, ratios, independence, exact probabilities, strata, clusters, SRM references, SRM localisation, shared p = α boundary |
| `tests/unit/experimentation/test_planning_and_analysis.py` | 13 | planning references, options, MDE inversion, design effect, binary/continuous references, refusal of missing outcomes, stratified, clustered, A/B/n |
| `tests/unit/experimentation/test_compliance_validity_decision.py` | 19 | 2SLS identity, weak instruments, Lee/Manski by hand, bootstrap bounds, every validity check, every decision branch, guardrails, business impact, design validation, z-test/Newcombe boundary disagreement (both directions) → HOLD, Lee undefined/refused/never-discarded bootstrap |
| `tests/statistical/test_m6_experimentation_properties.py` | 12 | coverage, A/A size, small-sample distortion, planned power, SRM size and cause, noncompliance, attrition, clustering, selection, multiple testing, decision operating characteristics, M2 repeated sampling |
| `tests/integration/test_m6_case_studies.py` | 5 | M2 case (HOLD, truth covered, M3 consistency), scenario clean/defective, reproducibility, M2 build unaffected |

## Problems found and corrected during implementation

1. **M3 defect: NaN power at large noncentrality.** SciPy 1.18's `nct.cdf` returns NaN deep in the
   lower tail, so `power_two_sample_t` returned NaN (e.g. d = 2, 50 per arm) and the continuous MDE
   solver failed at ≥ ~1,000 per arm. Fixed in `statistics/power.py` with a bounded fallback
   (replace by 0 only if `Φ(−ncp) < 1e-12`, else raise); regression test; M3 tests and evidence
   unchanged.
2. **Shared random streams.** The first draft seeded the population and its assignment from the same
   generator, which can correlate assignment with unit characteristics. Assignment and analysis now
   use streams spawned from the replicate seed (`common.seeding.spawn_generators`).
3. **A first SRM mechanism was too weak to show damage** (low-engagement loss biased the estimate by
   only +0.0017 at 5% loss). An outcome-related "inactive-user filter" mechanism was added; all three
   are reported.
4. **The M2 null LAUNCH rate is not zero (0.029).** In an underpowered design, the rare upward false
   positives clear the 2-point threshold because their estimates are exaggerated; reported, and the
   test asserts ≤ α rather than zero.
5. **A pandas/NumPy 2.5 deprecation** in concatenating datetime frames was avoided by repeating rows
   instead of concatenating in the generator.

## Pre-commit hardening (after review)

1. **Test/interval boundary.** The pooled z-test and the Newcombe interval are not dual and can
   disagree near p = 0.05 in either direction (25/100 vs 14/100: p = 0.0496, interval contains 0;
   168/200 vs 36/50: p = 0.0502, interval excludes 0 — formerly LAUNCH at δ = 0.01). Both are kept
   as computed; `MetricResult` now exposes `significant` (test), `interval_excludes_zero` and
   `test_and_interval_agree`; disagreement adds a warning, and `recommend` returns HOLD (a non-zero
   effect is established only when both agree). Methodology §4 and §8.
2. **p = α.** All M6 test decisions now use one rule, reject iff p ≤ α
   (`statistics.multiple_testing.rejects`), matching the M3 simulation helpers and statsmodels;
   previously `MetricResult.significant`, SRM detection/breakdown, the balance and exclusion checks
   and the simulation tallies used p < α while the corrections used ≤.
3. **Lee bootstrap.** An arm with no observed outcomes (in the data or in a resample) now raises
   `LeeBoundsUndefinedError` instead of producing NaN warnings or a `ZeroDivisionError`. The
   bootstrap is refused up front when the probability that any draw empties an arm exceeds 1e-6;
   undefined draws are never discarded and no interval is fabricated. Methodology §5.
4. **M3 power fix** — two further tests inject NaN tails so both the bounded fallback and the
   error branch are exercised on any SciPy version.

## Changes outside new M6 files

- `src/decision_intelligence/statistics/power.py` — the NaN-tail fix (demonstrated defect).
- `src/decision_intelligence/statistics/__init__.py` — docstring lists `multiple_testing`.
- `tests/unit/stats/test_power.py` — three regression tests for the fix (one against statsmodels,
  two with injected NaN tails).
- README, architecture overview, capability registry and roadmap — M6 status and links.
- No M2, M4 or M5 implementation, contract, test or evidence file was changed.

## Dependencies

None added.

## Validation (local)

`ruff check .` passed; strict `mypy src` passed under NumPy 2.4.6 with mypy 2.4.1 (CI-pinned) and
under NumPy 2.5.4; `pytest`: 485 passed in both environments (Python 3.12.8; Python 3.11 is
exercised by CI and mypy's 3.11 target). The M6 evidence generator was run twice with identical
output apart from runtimes; the M3 evidence regenerates identically to the committed version after
the power fix (runtimes excepted).

After the pre-commit hardening (Python 3.12.8, NumPy 2.5.4, SciPy 1.18.1, statsmodels 0.15.0,
mypy 2.4.1): `ruff check .` and strict `mypy src` pass; `pytest`: 495 passed. The M6 results were
regenerated twice (identical apart from runtimes); against the pre-hardening results every number is
unchanged — only the threshold wording (p ≤ α) differs, i.e. no simulated decision fell on a
test/interval boundary. M3, M4 and M5 evidence regenerate identically to the committed versions
(runtimes excepted); no M3–M5 module imports the files changed by the hardening.

## Limitations and risks

See the methodology's "Known limitations" and results §10. Most material: synthetic validation only;
normal-approximation planning; no winner's-curse correction in A/B/n; CACE and Lee bounds rely on
untestable assumptions; SRM and balance checks have limited power; decision thresholds and
economics are illustrative; CUPED, covariate adjustment, sequential, Bayesian and interference
methods are M7.
