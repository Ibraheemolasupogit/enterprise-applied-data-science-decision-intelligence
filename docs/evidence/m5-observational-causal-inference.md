# Milestone 5 evidence — observational causal inference

Status: implemented, assurance-reviewed and validated locally; **awaiting review**. Not committed or
pushed. Statuses: [capability-status.md](capability-status.md). Methodology:
[../methodology/observational-causal-inference.md](../methodology/observational-causal-inference.md).
Generated numbers: [m5-observational-causal-inference-results.md](m5-observational-causal-inference-results.md)
(`python scripts/generate_m5_evidence.py`; byte-reproducible apart from runtimes). External
development cross-checks: `python scripts/crosscheck_m5_external.py` (needs the optional packages).

## Acceptance criteria

| Requirement | Evidence |
| --- | --- |
| DiD: two-period and panel designs | `did.two_period_did`, `twfe_did`, `staggered_did` (Callaway–Sant'Anna); results §1–2 |
| Event studies, pre-treatment diagnostics, explicit timing | `staggered_did` event-time ATTs with pointwise intervals and nominal 95% sup-t bands (observed joint coverage 0.920; under-cover), `event_study_twfe`, windowed finite-cluster pre-trend test; results §1, §2b |
| Synthetic control with donor selection, fit diagnostics, placebo/sensitivity | `synthetic_control.py`: certified optimal weights, pre-fit and hull diagnostics, in-space and in-time placebos, leave-one-out; results §4–5 |
| Sharp RDD: local estimation, bandwidth sensitivity, manipulation and cutoff checks | `rdd.py`; results §6–7 |
| Additive time-indexed and threshold datasets with known effects | `data/rollout_panel.py`, `data/synthetic_control_panel.py`, `data/threshold_assignment.py`; M2 rebuild byte-identical (tested) |
| Identification assumptions and falsification tests per method | Methodology tables and the detectable/undetectable classification |
| Recovery and deliberately invalid designs | 7 DiD, 4 synthetic-control and 3 RDD designs in repeated sampling |
| Reuse of M3/M4 | `fit_ols`, `two_sample_t_test`, `bootstrap_ci`, `wilson_interval`, M4 `CausalEstimate` and `ReplicateSummary` |
| Robust uncertainty with limitations | CR1 + wild cluster bootstrap with regime warnings; CS influence functions; finite-cluster pre-trend F test; simultaneous event bands; RBC RDD intervals; placebo ranks with calibration caveat |

## Corrections from the pre-commit scientific assurance review

1. **Event-study multiplicity.** Pointwise 95% intervals over the event window covered the whole
   true path jointly in only 0.492 [0.444, 0.541] of panels. Added nominal 95% sup-t simultaneous
   bands (Montiel Olea & Plagborg-Møller 2019). Their observed joint coverage is 0.920 (Monte Carlo
   95% interval [0.889, 0.943], 400 panels): the design exhibits **undercoverage**, plausibly
   concentrated at the long horizons (one 15-market cohort; not separately verified). The bands
   improve on pointwise intervals but are not fully calibrated; nominal coverage is not guaranteed.
2. **Pre-trend test calibration was overstated.** The reported 8% came from 150 panels (Wilson
   [0.046, 0.135]). With 1,000 panels per row the windowed F test's size is 0.054 [0.042, 0.070] at
   60 units, 0.049 at 40, 0.045 at 120 — but **0.102 [0.085, 0.122] at 20 units** (5 per group).
   `staggered_did` now warns in that regime; the calibration table is generated (results §2b).
3. **Wild bootstrap regimes not surfaced.** With one treated cluster of ten the WCR test (almost)
   never rejects (0.000–0.004) while CR1 rejects 47.6%. Results now report the number of clusters
   carrying the tested regressor and warn at three or fewer; the one-treated-cluster row is in
   results §3.
4. **Synthetic-control optimality check was not discriminating.** The KKT tolerance accepted weights
   shifted by 1% between donors, and plain SLSQP sometimes stopped at a relative duality gap of
   ~1e-4. Replaced by an exact active-set refinement and a Frank–Wolfe duality-gap certificate that
   rejects 1e-4 perturbations (regression tests). Effect on the canonical design: none beyond
   rounding.
5. **Placebo exclusion used the wrong criterion.** It compared RMSPE multiples while citing
   Abadie et al. (2010), who use MSPE (5× RMSPE = 25× MSPE). Now MSPE (regression test).
6. **TWFE comparisons labelled as bias.** TWFE targets a different weighted average from the overall
   ATT; Monte Carlo rows now read "deviation from the overall ATT", and the decomposition's
   assumptions (balanced panel, `Y = α + λ + τD + ε`; weights depend only on adoption timing) and
   the meaning of "22%" (count share of treated cells; weight mass −0.047) are stated.
7. **Hull diagnostic overstated.** The per-period donor-range check is necessary, not sufficient,
   for convex-hull membership; wording corrected.
8. **Untraceable development figures** in the documents (37%, 30%, "coverage 1.00 in a pilot")
   replaced by generated results: all-leads χ² size 0.339 at 60 units (0.986 at 20); IF vs unit
   bootstrap coverage 0.990 vs 1.000 with reported SEs 0.260 vs 0.343 (empirical 0.247).

Verified without change: CS ATT(g, t), comparison groups and aggregation (independent pandas
re-implementation, now a test); influence-function SEs (match Welch per cell; 2.5% above a
within-cohort stratified bootstrap, the expected √(15/14) small-sample factor — development check);
TWFE decomposition; event-study timing and references; WCR null imposition, weights, studentisation
and finite-sample factor (brute force and `wildboottest`); RDD estimand, IK steps, CCT ρ = 1 and
effective sample sizes (`rdrobust` with equivalent settings); McCrary implementation and size;
canonical RDD seed (unchanged).

## Headline results

| Finding | Number |
| --- | --- |
| Staggered rollout, true overall ATT | 4.028 |
| Callaway–Sant'Anna, canonical panel | 4.304 [3.780, 4.829] |
| Static TWFE, canonical panel | 3.164 [2.652, 3.676]; implied estimand 3.157 |
| CS over 150 panels | bias −0.019 ± 0.021, coverage 0.967 [0.924, 0.986] |
| Static TWFE over 150 panels | deviation from overall ATT −0.878 ± 0.013, coverage 0.013 |
| Pre-trend test size (1,000 panels): 20 / 40 / 60 / 120 units | 0.102 / 0.049 / 0.054 / 0.045 |
| Dynamic TWFE false pre-trend rate (parallel trends true) | 0.853 |
| Event path joint coverage: pointwise 95% / nominal 95% sup-t band | 0.492 / 0.920 [0.889, 0.943] — undercoverage |
| Pre-test power against a violation that biases CS by +0.54 | 0.207 |
| Few clusters, null: 1 treated of 10 — CR1 / WCR | 0.476 / 0.004 |
| Few clusters, null: 2 treated of 10 — CR1 / WCR | 0.158 / 0.012 |
| Synthetic control, valid design | bias +0.149 ± 0.091, RMSE 0.72 (naive DiD 2.67) |
| SC flagged: outside hull / contaminated donors | 100% / 0% |
| Placebo rank 1: no effect / effect | 1.7% / 98.3% (nominal 3.2%) |
| RDD τ(c); population ATE; ATT | 3.000; 2.198; 3.705 |
| RDD robust coverage (300 datasets) | 0.930 [0.895, 0.954] |
| Placebo-cutoff false discoveries: conventional / robust | 9.3% / 4.0% |
| IK bandwidth vs Monte Carlo oracle | IK 8.53 (SD 0.97); oracle MSE falls to the top of the grid (14) |
| RDD manipulation / compound treatment bias | +0.781 (detected 100%) / +1.96 (chance-level detection) |
| Canonical RDD draw | robust 1.740 [0.927, 2.554] — ≈3 SE below truth; `rdrobust` agrees; seed unchanged |

## Reference comparisons

| Component | Reference | Agreement |
| --- | --- | --- |
| 2×2 DiD | TWFE on the same two periods; Welch on unit-level changes | exact |
| TWFE estimate and clustered SE | statsmodels OLS with unit/period dummies, cluster by unit | ≤ 1e-10 / ≤ 1e-8 |
| TWFE weights | `β = Σ w τ` in noiseless panels | exact (≤ 1e-8) |
| CS ATT(g, t), both comparison groups, overall aggregate | independent pandas implementation on noisy data | ≤ 1e-12 (test) |
| CS ATT(g, t) | true effects in noiseless panels | ≤ 1e-8 |
| CS event study (common timing) | saturated dynamic TWFE | ≤ 1e-9 |
| CS cell SE | Welch SE | exact |
| sup-t critical value | 1 coefficient → 1.96; independent → Šidák; perfectly correlated → 1.96 | ≤ 0.01 (test) |
| Wild cluster bootstrap p-value | brute-force refit of all sign patterns with M3 `fit_ols` | exact (test) |
| Wild cluster bootstrap p-value | `wildboottest` 0.3.2, WCR Rademacher, full enumeration | identical (0.2051; development) |
| Synthetic-control weights | known convex combination; brute-force grid; duality-gap certificate | recovered; grid not beaten; 1e-4 perturbations rejected |
| RDD estimates and SEs | statsmodels WLS (HC1) | ≤ 1e-10 / ≤ 1e-8 (tests) |
| RDD with equivalent settings | `rdrobust` 2.1.1, h = b, triangular, p = 1, `vce="hc1"` | estimates to 10 decimals; N_h identical; SEs within 0.01% (development) |
| IK bandwidth | equivariance; Monte Carlo oracle; PyPI `rdd` | exact (tests); conservative vs oracle (results §7); `rdd` differs by an ad-hoc floor (development) |

## Tests added (77)

| File | Tests | Covers |
| --- | --- | --- |
| `tests/unit/data/test_m5_generators.py` | 17 | determinism, contracts, ground-truth invariants, failure switches, validation |
| `tests/unit/causal/test_did.py` | 18 | identities, statsmodels TWFE, weights, noiseless CS, **independent CS implementation** (2), CS = dynamic TWFE, Welch SE, anticipation, pre-trend references, **sup-t identities**, **simultaneous band**, **small-group warning**, bootstrap option, warnings |
| `tests/unit/causal/test_cluster_inference.py` | 7 | brute-force equality (3 nulls), CR1 vs M3, CI inversion, validation, **one-treated-cluster warning** |
| `tests/unit/causal/test_synthetic_control.py` | 11 | exact recovery, brute-force grid, **certificate**, **1e-4 perturbation rejected**, **SLSQP early-stop case**, diagnostics, hull flag, placebo arithmetic, **MSPE exclusion**, in-time/LOO, validation |
| `tests/unit/causal/test_rdd.py` | 10 | statsmodels WLS, RBC = local quadratic, IK equivariance, sensitivity, density tests, covariates/placebos, global polynomial, validation |
| `tests/statistical/test_m5_observational_properties.py` | 9 | CS recovery, TWFE deviation, pre-test size, spurious TWFE pre-trends, invalid DiD designs, anticipation, few clusters, synthetic control, RDD designs, placebo cutoffs, **simultaneous band coverage** |
| `tests/integration/test_m5_case_studies.py` | 5 | canonical analyses, local RDD estimand, reproducibility, M2 build unaffected |

Bold = added in the assurance review (10 tests). Full suite: 400 passed (323 pre-existing unchanged
+ 77 M5).

## Problems found and corrected during development (before the review)

1. CS pre-trend test over-sized when testing all leads with a χ² reference (results §2b quantifies
   it) — replaced by the windowed finite-cluster F test.
2. CS unit-bootstrap intervals conservative under fixed cohort sizes (results §2b) — influence
   functions became the default.
3. Conventional RDD intervals unreliable where the outcome curves (results §7) — robust
   bias-corrected inference became the default.
4. The first RDD design could not show "local ≠ population" (cutoff at the score mean) — cutoff moved
   to 60 before any evidence was generated.
5. A ±3 donut was uninformative alone — ±0.5 and ±3 donuts now reported with intervals.
6. Unit fixed effects nested in unit clusters make the dummies' clustered covariance degenerate; the
   resulting NumPy warning from M3 `fit_ols` is suppressed at M5's call site only (M3 unchanged).

No M2, M3 or M4 scientific implementation, contract, test, script or evidence file was modified.
The only edit to an existing code file is the package docstring of `causal_inference/__init__.py`;
the M3 and M4 evidence regenerates identically to the committed versions.

## Dependencies

None added (runtime or dev). `rdrobust`, `wildboottest` and `rdd` are used only by the optional
development cross-check script (heavy or unstable dependency chains: matplotlib/plotnine, numba).

## Validation (local)

`ruff check .` passed; `mypy src` (strict) passed under NumPy 2.4.6 with mypy 2.4.1 (CI-pinned) and
under NumPy 2.5.4; `pytest`: 400 passed in both environments (Python 3.12.8; Python 3.11 is exercised
by CI and mypy's 3.11 target). The evidence generator was run twice with identical output apart from
runtimes.

## Limitations and risks

See the methodology's "Known limitations" and the detectable/undetectable table. Most material:

- The canonical RDD seed is an unlucky draw (≈3 SE); reported, not replaced.
- CS inference conditions on cohort composition and uses no covariates; the pre-trend test is
  over-sized with ~5 units per group; the nominal 95% sup-t bands under-cover jointly (observed 0.920, Monte Carlo 95% interval [0.889, 0.943]), particularly at long horizons; nominal coverage is not guaranteed.
- No sensitivity bounds for parallel-trends violations; spillovers under common timing, donor
  contamination and compound treatments at a cutoff are undetectable from the data.
- Synthetic control has no model-based interval; placebo ranks are descriptive unless assignment was
  random.
- No cluster-based inference is reliable with one or two treated clusters.
- Interrupted time series, fuzzy RDD and propensity-score matching remain unimplemented.
