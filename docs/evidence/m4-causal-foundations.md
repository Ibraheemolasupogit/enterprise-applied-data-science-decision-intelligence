# Milestone 4 evidence — causal foundations, DAGs and adjustment

Status: implemented and validated locally; **awaiting review**. Not committed or pushed. Statuses:
[capability-status.md](capability-status.md). Methodology:
[../methodology/causal-foundations.md](../methodology/causal-foundations.md). Generated numbers:
[m4-causal-foundations-results.md](m4-causal-foundations-results.md)
(`python scripts/generate_m4_evidence.py`, ~1 minute, byte-reproducible apart from runtimes).

## Acceptance criteria (roadmap M4: "identification examples, known-effect recovery")

| Requirement | Evidence |
| --- | --- |
| Estimands ATE, ATT, conditional effects; association vs causation | `estimands.py` (exact naive = ATT + selection bias); `subgroup_effects`, `heterogeneity_test`; naive estimates labelled as associations |
| DAGs, paths, confounders, mediators, colliders, backdoor logic | `dag.py`; networkx-verified d-separation; M2 campaign DAG with roles and minimal adjustment sets (results §1) |
| Consistency, exchangeability, positivity, SUTVA | Methodology table; positivity and interference failures simulated (results §9) |
| Outcome regression, IPW, AIPW | `estimators.py`, ATE and ATT; exact standardisation identities in tests |
| Overlap, balance, extreme-weight, specification diagnostics | `diagnostics.py`; results §3 and positivity study |
| Naive vs adjusted vs known truth on M2 | Results §2 (canonical seed), §7 (200 M2 draws), §8 (n = 20,000) |
| Sensitivity, negative/failure cases | E-value, bias grid, negative-control outcome, DAG falsification test, eight failure cases |
| Reuse of M3 primitives | `fit_logistic`, `fit_ols`, `two_sample_t_test`, `bootstrap_ci`, validation helpers; no reimplementation |
| Deterministic, reference and statistical-property tests | 63 new tests (below) |

## Headline results

| Finding | Number |
| --- | --- |
| True effect (design, evaluation only) | +0.080 |
| Naive contrast, canonical seed / mean of 200 draws | 0.226 / 0.190 |
| Regime A AIPW over 200 M2 draws | bias −0.0012 ± 0.0016, coverage 0.930 [Wilson 0.886, 0.958], exact binomial p vs 0.95 = 0.19 |
| Regime C (logged propensity) IPW over 200 draws | bias −0.0004 ± 0.0017, coverage 0.930 [0.886, 0.958] |
| Regime B (proxies, confounder unmeasured) | bias +0.028 ± 0.0015, coverage 0.78; E-value 1.90; omitted confounder's bounding factor 1.17–1.85 (2–10 bins) |
| Invalid: adjust for post-treatment billing count | bias −0.044, coverage 0.20 |
| Invalid: select on billed customers | bias −0.025, coverage 0.85 |
| Canonical seed, regime A AIPW | 0.130 [0.082, 0.177] — a ~2-SE draw; most valid CIs narrowly miss 0.08 |
| n = 20,000 draw: regime A / regime B | 0.087 [0.073, 0.102] / 0.116 [0.102, 0.129] |
| DAG falsification test (`treated ⊥ proxies | latent_value`) | χ² = 5.80, 9 df, p = 0.76 |
| Negative control (log income) over 200 draws: A / B | +0.0006 (coverage of 0: 0.96) / +0.048 (0.20) |
| M-bias: adjusting for a collider | error −1.18 on a true effect of 1.0 |
| Positivity s = 6: min-arm ESS share / AIPW RMSE | 2.8% / 9.7; kept-population effect recovered by trimming (error +0.010) |
| Double robustness (Kang–Schafer): one model wrong / both wrong | RMSE 1.67 or 0.24 / 26.6 |

## Reference comparisons

| Component | Reference | Agreement |
| --- | --- | --- |
| d-separation | networkx `is_d_separator` | identical on >1,000 random queries over 60 random DAGs |
| d-separation (internal) | explicit path enumeration vs moralisation | identical on every query |
| Backdoor criterion | networkx d-separation in the backdoor graph + descendant rule | identical on 60 random DAGs, all subsets ≤ 2 |
| Minimal adjustment sets | Pearl's textbook z1–z3 example | `{z1, z3}`, `{z2, z3}`; `{z3}` alone invalid |
| g-computation, Hájek IPW, AIPW (ATE and ATT) | hand-computed stratified standardisation | exact (≤ 1e-9) with saturated models |
| Regression adjustment (ATE, ATT), IPW (ATE, ATT), AIPW (ATE) | statsmodels `TreatmentEffect` (`ra`, `ipw`, `aipw`) | RA ≤ 1e-10; IPW/AIPW ≤ 5e-4 (tested; ≤ 1e-12 with good overlap) |
| IPW and AIPW ATE standard errors | statsmodels `TreatmentEffect` joint-GMM SEs (account for propensity estimation) | within 3% (tested; ≤ 1% observed) |
| IPW ATT standard error | Monte Carlo calibration (`test_att_ipw_standard_error_is_calibrated_with_estimated_propensity`: 200 replications, strong confounding + effect heterogeneity) | asserted: mean SE / empirical SD in (0.85, 1.2), coverage ≥ 0.91 |
| E-value | `bounding_factor(E, E) = RR` (Ding & VanderWeele identity) | exact |
| HT IPW | direct formula | exact |
| AIPW, no covariates | difference in means, unpooled SE | exact estimate; SE within 1% |
| AIPW risk ratio | standardised risk ratio | exact |
| Balance after saturated IPW | theory (exact balance) | weighted SMD = 0 (≤ 1e-10), ATE and ATT weights |
| E-value | closed-form values (RR = 2 → 3.414; protective symmetry) | exact |
| Billing count used in replicates | pipeline `transactions` table | identical per customer |

## Tests added (63)

| File | Tests | Covers |
| --- | --- | --- |
| `tests/unit/causal/test_dag.py` | 16 | networkx agreement, path rule, cycles, backdoor on classic and random DAGs, roles, campaign DAG and every strategy's validity |
| `tests/unit/causal/test_estimators.py` | 16 | statsmodels `TreatmentEffect` agreement (estimates and SEs), known-propensity influence function, standardisation identities, HT formula, no-covariate reduction, known propensity, bootstrap reproducibility, positivity error, warnings, invalid inputs, risk ratio, subgroups, potential-outcome identity |
| `tests/unit/causal/test_diagnostics_and_sensitivity.py` | 14 | bounding factor and E-value identity, SMD formula, exact weighted balance, ESS and weight warnings, empty common support, trimming, E-values, bias grid, scenario reproducibility and truths |
| `tests/statistical/test_m4_causal_properties.py` | 9 | ATT IPW SE calibration, coverage and unbiasedness (ATE, ATT), M-bias, mediator, positivity/trimming, double robustness, interference, M2 Monte Carlo (valid unbiased; invalid biased) |
| `tests/integration/test_campaign_analysis.py` | 8 | confounder benchmark definition, exact billing rule, replicate generator, validity flags, ground truth attached last, diagnostics on M2, sensitivity outputs, reproducibility |

Full suite: 323 passed (260 pre-existing unchanged + 63 new).

## Defects and corrections found during M4 validation

1. Interpretations drafted before the numbers existed were checked against the generated report and
   five were corrected: the claim that *every* valid interval missed 0.08 (one covers); regime B's
   negative-control bias described as "small" (+0.048, ≈3 SE); E-value benchmark described as
   "close" (that correction was itself flawed — see the assurance-review section below); coverage "at or above 95%" (0.93–0.955); and an apparent +0.017 bias in the confounded
   scenario, shown by larger independent runs to be a ~3-SE chance draw — the report now uses 500
   replications and prints Monte Carlo SEs.
2. A double-robustness test tolerance (fixed 0.3) ignored the high Monte Carlo variance of the
   "outcome-wrong" combination; replaced by the Monte-Carlo-SE criterion used elsewhere.
3. The replicate study first approximated billing counts from tenure (99.2% agreement); replaced by
   an exact re-implementation of the M2 billing rule, verified against the pipeline per customer.

No M2 or M3 method or contract was modified.

## Corrections from the pre-commit scientific assurance review

1. **IPW standard errors ignored propensity estimation.** Corrected by the M-estimation term
   `H I⁻¹ sᵢ` for the logistic propensity fit (ATE and ATT, HT and Hájek); known propensities keep
   the uncorrected form, which is exact for them. Effect: ATE intervals were ≈1.6× too wide in the reference design (the regression test fails on the old code);
   the ATT had the theoretical risk of anti-conservative intervals (not observed in the designs
   tested). Now within 3% of statsmodels' joint-GMM SEs for the ATE (asserted) and calibrated by
   Monte Carlo for the ATT (asserted). Point estimates unchanged. Under poor overlap statsmodels'
   own joint-GMM IPW estimate can drift from the exact two-step value; the comparison tests
   therefore use designs with adequate overlap.
2. **E-value benchmark used the wrong quantity and reached an unsupported conclusion.** It compared
   `P(T | U high) / P(T | U low)` (not the prevalence ratio RR_EU) with the E-value, on a median
   split, and concluded the real confounder "could explain part but not all" of the regime-B
   estimate. Replaced by the Ding & VanderWeele bounding factor with RR_EU and RR_UD at 2, 4 and 10
   categories: 1.17 / 1.43 / 1.85 against an observed RR of 1.29 — at finer resolution the actual
   omitted confounder *could*, in the worst case, explain the entire estimate. The corrected reading:
   the E-value correctly flags regime B as not robust; the real bias is smaller than the bound.
3. **DAG edge `price_sensitivity → billing_transactions` was not in the generator** for the billing
   *count* (price sensitivity affects amounts). Removed; no adjustment-set verdict changed; a test
   pins the node's parents to the generator.
4. **Positivity trimming was scored against the wrong population.** The "trimmed truth" used units
   kept by the *true* propensity while the analysis trims on the *estimated* one; now scored against
   the true effect of the units actually kept.
5. **Coverage claims lacked an uncertainty statement.** The Monte Carlo table now reports Wilson
   intervals and exact binomial p-values against 0.95: valid strategies' 0.93–0.94 coverage is not
   distinguishable from nominal (smallest p = 0.19), and 200 replications cannot exclude coverage a
   few points low.
6. **Regime C wording** now states explicitly that a known propensity requires actual randomisation
   with logged probabilities — unavailable in ordinary observational data.
7. **Hand-written figures had drifted from the regenerated report** (positivity and double-robustness
   numbers from a 200-replication run). All cited figures re-synchronised; generated prose now
   computes its numbers instead of hard-coding them.

Verified without change: d-separation and backdoor logic (networkx); RA/IPW/AIPW estimates
(statsmodels); E-value formula; negative-control conclusions; invalid post-treatment adjustment,
selection, M-bias, mediator and interference results; canonical-seed results (seed unchanged).

## Dependencies

- Runtime: none added (NumPy, pandas, SciPy as before).
- Dev-only: `networkx>=3.3,<4` — reference oracle for d-separation in tests; never imported by `src/`,
  so no mypy override was added for it.

## Validation (local)

`ruff check .` passed. `mypy src` (strict) passed under NumPy 2.5.4 and under the CI-pinned NumPy
2.4.6 + mypy 2.4.1. `pytest`: 323 passed in both environments (Python 3.12.8; 3.11 compatibility rests
on CI and mypy's 3.11 target). Evidence generation run twice: identical apart from runtimes.

Intermittent `RuntimeWarning`s during tests all originate in the M2 generator
(`data/retention.py:83`, NaN timedeltas passed to `pd.to_timedelta` under NumPy 2.4–2.5); they
predate M4, do not affect results (M2's byte-identical rebuild test passes) and were left untouched
because M2 is out of scope.

## Limitations and risks

See the methodology's "Known limitations". Most material for review:

- The canonical M2 seed is an unlucky draw for the valid estimators; the evidence says so plainly and
  relies on repeated sampling. A reviewer looking only at section 2 could misread it.
- The M2 campaign has a constant effect and a time-invariant exposure; richer heterogeneity and
  dated interventions are simulated, not drawn from M2.
- Parametric nuisance models only; no cross-fitting or ML nuisance models.
- Propensity-score matching deferred (needs Abadie–Imbens variance).
- All evidence is synthetic; none establishes real-world validity.
