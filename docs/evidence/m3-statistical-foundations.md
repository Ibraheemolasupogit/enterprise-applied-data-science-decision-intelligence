# Milestone 3 evidence — statistical foundations

Status: implemented and validated locally; **awaiting review**. Not committed or pushed (commits are
made manually by the project owner). Capability statuses:
[capability-status.md](capability-status.md). Methodology:
[../methodology/statistical-foundations.md](../methodology/statistical-foundations.md). Generated
numerical results: [m3-statistical-foundations-results.md](m3-statistical-foundations-results.md).

## Acceptance criteria (Master Brief §24: "tested estimators, simulations, methodological documentation")

| Criterion | Evidence |
| --- | --- |
| Probability distributions, sampling, reproducible simulation | `statistics/distributions.py`, `statistics/simulation.py`; KS-tested samplers; every stochastic function takes an explicit generator; the evidence report reproduces byte-for-byte (runtimes excepted) |
| Descriptive statistics, estimators, sampling distributions, CLT | `statistics/descriptive.py`, `statistics/estimators.py`; results §9 |
| Bootstrap CIs, hypothesis tests, effect sizes, power/sample size | `common/bootstrap.py`, `statistics/hypothesis_tests.py`, `effect_sizes.py`, `power.py`; results §1, §6–§8 |
| Parametric and non-parametric tests with explicit assumptions | Every test returns `assumptions` and data-specific `warnings`; Welch vs Student failure demonstrated (results §6) |
| Linear and logistic regression with diagnostics | `statistics/linear_regression.py`, `logistic_regression.py`; results §3, §4, §12 |
| Bias–variance, missing data, Monte Carlo | `statistics/bias_variance.py`, `missing_data.py`, `simulation.py`; results §5, §10, §11 |
| Reusable primitives in `statistics/` and `common/` where justified | `common/` gains only cross-pillar primitives (validation; the bootstrap helper the architecture already assigned to it) |
| M2 data used selectively for real questions | `statistics/applied_analyses.py`: experiment lift, revenue distribution, retention model vs baseline, clustered transactions, income missingness (results §1–§5) |
| Mathematical documentation, assumptions, failure cases, baselines, interpretation | Methodology doc (§17 twelve items per method family) |
| Unit, statistical-property and reference tests | 196 new tests (below) |
| No unsupported causal claims; no leakage | Only the randomised experiment is read causally; leakage controls asserted by tests (below) |
| Python 3.11/3.12, minimal dependencies | No 3.12-only syntax; mypy targets 3.11; one new runtime dependency (SciPy) |

## Reference comparisons

Each implementation is compared against a trusted reference in the test suite. The agreement column is the maximum discrepancy observed in a validation run; the tests assert the comparison with `pytest.approx`'s default relative tolerance (1e-6) or the tighter tolerance shown in the test.

| Implementation | Reference | Observed agreement |
| --- | --- | --- |
| Welch / Student / one-sample / paired t (statistic, p, df, CI; all alternatives) | `scipy.stats.ttest_*` | ≤ 1e-12 relative |
| Mann–Whitney U exact and asymptotic (ties) | `scipy.stats.mannwhitneyu` | ≤ 1e-12 |
| Wilcoxon signed-rank exact and asymptotic (ties) | `scipy.stats.wilcoxon` | ≤ 1e-12 |
| Exact permutation test | `scipy.stats.permutation_test` | exact |
| χ² independence, expected counts | `scipy.stats.chi2_contingency(correction=False)` | ≤ 1e-12 |
| ANOVA, Kruskal–Wallis, Brown–Forsythe | `scipy.stats.f_oneway`, `kruskal`, `levene(center="median")` | ≤ 1e-12 |
| Wilson interval; Newcombe interval | statsmodels `proportion_confint`, `confint_proportions_2indep` | ≤ 1e-12 |
| Odds ratio / risk ratio and log intervals | statsmodels `Table2x2` | ≤ 1e-12 |
| Bootstrap percentile / basic / BCa (one sample) | `scipy.stats.bootstrap` (same seed) | ≤ 1e-10 relative |
| Bootstrap BCa (two sample) | `scipy.stats.bootstrap` | within Monte Carlo error (different resampling streams) |
| t power (one-/two-sample, unequal n, one-sided); MDE | statsmodels `TTestIndPower`, `TTestPower` | ≤ 1e-10 |
| Proportion sample size | closed-form normal-approximation formula | within 1 unit (integer search) |
| OLS coefficients; classical, HC0–HC3, CR1 SEs; R², adj. R², F, condition number | statsmodels `OLS` | ≤ 1e-10 relative |
| Breusch–Pagan, Durbin–Watson, Jarque–Bera, Cook's D, studentised residuals, VIF | statsmodels diagnostics | ≤ 1e-8 relative |
| Logistic coefficients, SEs, log-likelihood, AIC, McFadden R², LR p; HC0 SEs | statsmodels `Logit` | ≤ 1e-8 (HC0 ≤ 1e-7) |
| Skewness, kurtosis, SEM, MAD, IQR, trimmed mean | SciPy | floating-point precision |
| Samplers (exponential, Weibull, Beta via rejection) | SciPy distributions (KS test) | p > 0.01 at fixed seeds |

## Headline numerical results

Full tables, with Monte Carlo standard errors: [m3-statistical-foundations-results.md](m3-statistical-foundations-results.md).

| Finding | Number |
| --- | --- |
| M2 experiment lift (true +0.040) | +0.031, 95% CI −0.007 to +0.069, p = 0.109; CI covers truth |
| Design power of that experiment for +0.04 | 0.53 (≈1,080/arm needed for 0.80) |
| Student pooled t type I error, n=10 (σ=3) vs n=40 (σ=1) | **0.256** vs Welch 0.052 (nominal 0.05) |
| t-interval coverage, log-normal(σ=1.5): n=10 / n=200 | 0.718 / 0.883 (nominal 0.95) |
| Wald vs Wilson coverage, Bin(40, 0.05) | 0.870 vs 0.950 |
| Bootstrap at n=20 (exponential): percentile / BCa / t | 0.896 / 0.906 / 0.913 — bootstrap does not beat t here |
| Analytic vs simulated power, d=0.5, n=64 | 0.8015 vs 0.801 ± 0.006 |
| Slope CI coverage with clustered errors: classical / HC3 / cluster | 0.537 / 0.534 / 0.939 |
| M2 transactions: cluster SE ÷ classical SE | 1.9–2.3× |
| Retention model vs base rate (held-out Brier / log loss / AUC) | 0.217 vs 0.241 / 0.621 vs 0.675 / 0.683 vs 0.5 |
| Missing data, MAR, 40% missing: complete-case bias / coverage | −0.256 / 0.17; regression imputation ≈0 bias but 0.80 coverage |
| M2 income (MCAR): complete-case error vs known full-data mean | 0.25 SE |

## Tests added

| File | Tests | Covers |
| --- | --- | --- |
| `tests/unit/common/test_validation.py` | 21 | every invalid-input class |
| `tests/unit/common/test_bootstrap.py` | 15 | SciPy agreement, reproducibility, degenerate/undefined cases |
| `tests/unit/stats/test_descriptive.py` | 9 | SciPy agreement, 1e9-offset stability, Welford merge, NaN semantics |
| `tests/unit/stats/test_distributions.py` | 10 | KS-validated samplers, envelope violation, stratified unbiasedness/efficiency |
| `tests/unit/stats/test_estimators_and_simulation.py` | 12 | estimator bias/MSE theory, CLT decay, exact intervals, MC engine |
| `tests/unit/stats/test_hypothesis_tests.py` | 40 | all tests × alternatives vs SciPy/statsmodels, exact-count integrity, edge cases |
| `tests/unit/stats/test_effect_sizes.py` | 13 | noncentral-t inversion property (incl. small df, large t), Hedges J, labelled fallback, zero cells |
| `tests/unit/stats/test_power.py` | 17 | statsmodels agreement, minimality of sample sizes, invalid designs |
| `tests/unit/stats/test_linear_regression.py` | 18 | all covariance types and diagnostics vs statsmodels, rank deficiency, leverage 1 |
| `tests/unit/stats/test_logistic_regression.py` | 14 | statsmodels agreement, exact separation criterion, rare-event non-separation, AUC, calibration |
| `tests/unit/stats/test_bias_variance_and_missing_data.py` | 9 | U-shaped error, decomposition identity, mechanisms, strategy bias pattern |
| `tests/statistical/test_m3_statistical_properties.py` | 10 | type I error, coverage, analytic vs simulated power, cluster-robust coverage |
| `tests/integration/test_applied_analyses.py` | 8 | M2 analyses: consistency, known-truth recovery, no leakage, reproducibility |
| **Total new** | **196** | (64 pre-existing M1/M2 tests unchanged; 260 in total) |

Statistical-property tests use fixed seeds and a 3.5-standard-error tolerance: deterministic, and a
correct implementation cannot fail by chance.

## Leakage and causal-claim controls (asserted by tests)

- `latent_value`, campaign treatment/propensity and post-outcome columns are absent from the
  retention features; the model is evaluated on held-out customers; the baseline uses training
  prevalence only.
- Ground truth is read only after estimation (experiment CI coverage; income twin-build check).
- Observational comparisons (segments, regression coefficients) are labelled associational in code,
  docs and the generated report.

## Defects found and corrected during M3 validation

Recorded so a reviewer can see which checks found real problems:

1. **Separation detection.** An initial |η| > 30 rule missed quasi-complete separation (Newton
   "converged" at |η| ≈ 24, SE ≈ 9×10⁴); lowering it to |η| > 18 then *falsely rejected* a valid
   rare-event model (pre-commit review). Replaced by the exact Albert–Anderson criterion solved as a
   linear programme; stress-tested on 2,977 random designs (1,073 flagged, every one with a valid
   separating certificate; every unflagged one converged to a finite MLE).
2. **Noncentral-t CI inversion crashed** on M2 revenue (|t| ≈ 25) because SciPy's `nct.cdf` returns
   NaN far in the tail; the first fix still crashed for small df (e.g. df = 2, t = 20: NaN *inside*
   the bracket — found in pre-commit review). The CDF now falls back to the exact identity
   `F(t; ν, δ) = S(−t; ν, −δ)`, and only if both routes fail is a labelled normal approximation used
   (with a small-df caveat).
3. Four initial test expectations were wrong and were corrected *after* checking the theory, not
   loosened to pass: Breusch–Pagan cannot detect variance in |x| (fixture changed to monotone
   heteroskedasticity); variance is not monotone in polynomial degree under random design; a
   `max+1` statistic does not make BCa undefined (ties); an exact float comparison of a quantile.
4. Generated-report interpretations were checked against the numbers and three statements were
   corrected (bootstrap vs t ranking at n=20; slow recovery of t coverage for log-normal data; the
   cause of zero revenue). A fourth — that `latent_value` "drives" segment — reversed the
   generator's actual direction (segment is drawn first) and was corrected in pre-commit review.
5. **Two-sample BCa acceleration** pooled all jackknife values around one grand mean instead of
   centring each sample on its own jackknife mean with `(nⱼ−1)/nⱼ` influence scaling; for unequal
   sizes and non-linear statistics this changed `a` (0.0506 vs 0.0475 in a test case). One-sample
   intervals were unaffected (the factors cancel). Fixed; tested against an independent statement of
   the formula.
6. **OLS inference with vanishing residuals** (exact fit or constant response) reported spurious
   certainty (p ≈ 1e-120; p = 0.03 for a slope on a constant response). Now NaN with a warning.
7. **Joint Wald F with a singular covariance** (fewer clusters than slopes) returned a huge negative
   F. Now skipped with a warning.

## Dependencies

- **Added runtime:** `scipy>=1.13,<2` — t/F/χ²/noncentral-t distributions, root-finding, special
  functions. 1.13 is the first SciPy release supporting NumPy 2, matching the existing
  `numpy>=2.0,<3` pin. Reimplementing these distributions would be less reliable, not more.
- **Added dev-only:** `statsmodels>=0.14.2,<1` — reference oracle for tests; never imported by
  `src/`.
- **mypy:** `ignore_missing_imports` override for `scipy`/`statsmodels` (no inline types); every
  SciPy return value is converted to `float`/`ndarray` at the call site, so strict mode still
  type-checks all repository code.

## Validation run (local)

Clean virtual environment, `pip install -e ".[dev]" --config-settings editable_mode=compat`, Python
3.12.8, NumPy 2.5.4, SciPy 1.18.1, pandas 2.3.3, statsmodels 0.15.0:
`ruff check .` passed; `mypy src` (strict) passed; `pytest` 260 passed. Python 3.11 was not
available locally; 3.11 compatibility rests on CI's 3.11 job and mypy's `python_version = "3.11"`
target.

## Limitations and outstanding risks

See the methodology doc's "Known limitations". The material risks for review:

- **Python 3.11 not run locally** — first confirmation will be CI.
- **Separated data are rejected, not fitted** (no Firth penalisation); separation within the LP's
  numerical tolerance cannot be distinguished from exact separation.
- **Normal approximations** remain for tied/large rank tests and for proportion power.
- **Single imputation only**; uncertainty from imputation is not propagated (documented, and shown
  to under-cover).
- **Tolerances in statistical-property tests** are deterministic at fixed seeds; a future dependency
  upgrade that changes NumPy's random streams could shift simulated values (not their validity) and
  require re-baselining the generated report.
- All evidence is synthetic or simulated; none is production validation.
