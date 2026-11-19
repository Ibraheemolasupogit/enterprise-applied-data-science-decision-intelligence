# Milestone 11 P7 — baselines, evaluation, uncertainty and selection reporting: findings

Status: **implemented; A10 and A11 applied; awaiting owner review — decisions listed below**.
Numbers: [m11-p7-evaluation.md](m11-p7-evaluation.md)
(`python scripts/report_m11_p7_evaluation.py`; development seeds 5701-5708 and 5711-5713 only; one
draw per scenario, so every number is descriptive). The protocol changes are **A10**
(owner-approved: `B1_DEFINITION`, `B2_DEFINITION`, `REFERENCE_FALLBACK_SCOPE`,
`TIER_A_IN_TIER_B`, `E3_NO_OK_FIT_RULE`) and **A11** (owner-approved after finding 5:
`CONTRACTUAL_OUTCOME_NOISE`); stage `P7-amended`, manifest `0a1ea3bd…08a8`, code `244acc01…ada0`
([m11-protocol-amendments.md](m11-protocol-amendments.md)). Selection thresholds, tolerances,
targets, comparisons and seeds are unchanged; pilot, main and fresh seeds were not used. P0–P6
model code and evidence are untouched. P8 is not started.

## Scope (the approved phase table)

P7 = evaluation, uncertainty, IPCW, calibration and status reporting. Gate: IPCW unbiasedness
test; metrics match brute-force references; status rules tested. Plus, from the P5/P6
dispositions: B1/B2 and selection reporting.

| Module | Content |
|---|---|
| `statistics/predictive_metrics.py` | Relative bias, MAE, RMSE; calibration line (OLS, HC1); decile reliability; Spearman; top-decile capture (M10's definition); normalised Gini; discrete-time censoring KM; Harrell and Uno C; IPCW Brier and integrated Brier (Graf); exact-mass D-calibration (Haider); log score, CRPS and randomised PIT for counts; paired customer-clustered bootstrap; CR1 sandwich; Clopper-Pearson |
| `customer_value/predictive_baselines.py` | B1 (M10 *Kaplan–Meier by cell*, unchanged, M10 status thresholds, no fallback) and B2 (A10 RFM heuristic); validation at `L_v` as E1/E2 and E4 |
| `customer_value/predictive_evaluation.py` | Contractual paid-month distributions; the **exact** E4 window-purchase distribution; Hessian-based parametric simulation (E1/E2, E4) and CR1-based simulation (E3) giving parameter and predictive intervals; CR1 for E3 and both E5 parts; coverage summaries (marginal, by segment, by decile); selection reporting |

## Correctness gates (tests) — all pass

| Gate | Result |
|---|---|
| **IPCW unbiasedness** (300 replications, n = 800, heterogeneous discrete hazards, imperfect prediction, independent censoring) | IPCW Brier at every t = 1..12 and the integrated score agree with the exact complete-data expectation (\|z\| < 4); the naive complete-case score is biased (\|z\| > 10) |
| Metrics vs brute force | Every metric against loops or independent formulas: calibration line = statsmodels OLS HC1 (1e-8); Harrell/Uno C = O(n²) pair loops with risk ties; censoring KM and IPCW Brier = explicit loops; CRPS = sum form and `E\|X-y\| - E\|X-X'\|/2`; CR1 = statsmodels GLM cluster covariance (1e-6); Clopper-Pearson = statsmodels |
| D-calibration and PIT | True model: D-calibration p > 1e-3 with bin shares 0.08-0.12; misspecified (`S³`) p < 1e-6; randomised PIT of the true Poisson KS p > 1e-3 |
| **E4 count distribution exact** | sums to 1 (1e-8) and its mean equals E4's expected purchases (1e-6); matches an independent event-by-event simulation of the BG/NBD posterior within 4.5 MC SE |
| **B1 status rules** | every cell's status equals M10's thresholds recomputed independently; `ok`, `unstable` and `undefined` all exercised; undefined cells have no prediction (no fallback) |
| B1 = M10 | equals `customer_projections`' *Kaplan–Meier by cell* exactly; population = the contractual `training_stack` population |
| B2 = definition | brute-force count of purchases (acquisition included) x mean revenue x E4's ratio; population = E4's |
| Leakage | B1 validation ignores rows recorded after `L_v`; B2 ignores rows recorded after `C` |
| CR1 rebuild exact | E3's person-month design and E5's two designs are rebuilt exactly (penalised score ≈ 0 at the fit; E3's inverse Hessian reproduced to 1e-6) |
| Selection reporting | equals the frozen `select_model`; E2 reference shown for contractual and **never** for non-contractual; ineligible and undefined models excluded with reasons; Tier B models refused in Tier A selections; models of the wrong kind refused |
| Protocol (A10) | exact wording of all five constants; stage; A1–A10; gates, tolerances, targets, comparisons unchanged; main and fresh seeds refused; digests pinned; E3's fallback label equals `E3_NO_OK_FIT_RULE` (forced no-ok tuning) |
| **A11 residual pools** | grouping equals the contribution model's source for every row (cell and segment fallbacks exercised) and each pool is exactly the residuals of that group's months; every pool centred at 0 (1e-9); pools ignore every month after `complete_through(C)` (perturbation test) |
| **A11 variance** | simulated value variance = Var(sum of means over N paid months) + E[N] σ² (within 3%); mean unbiased (4 SE); no noise on unpaid months |
| **A11 backward compatibility and reproducibility** | with noise on, the expected values, pre-A11 outcomes and pre-A11 intervals are bit-identical to a run with noise off (E2 and E3); same noise seed → identical, different → different; noise needs both inputs |
| **Pre-A11 record preserved** | record §§1-7 regenerated byte-identically after A11 (only the header digests, seed list and timings changed) |
| No truth or oracle access | AST test on the three new modules |

## Statistical findings (development seeds; record §§1-8)

1. **Contractual accuracy.** E1, E2, E3 and E5 are close (holdout RMSE 26.2-27.1); against the
   oracle, Tier A RMSE 1.2-1.4 (E1/E2), 1.4-2.1 (E3), 2.0-2.4 (E5). **B1 is far worse**: RMSE
   vs oracle 18.8-20.3, relative bias -10% to -11%, calibration slope 0.47-0.49 — the run rate
   x Kaplan–Meier projection ignores promotion roll-off and individual history. In Tier B, E3
   has the lowest holdout RMSE (26.4 / 26.6) and the best discrimination (Spearman 0.55 / 0.54).
2. **Non-contractual accuracy.** E4: relative bias +0.1% / +1.4%, RMSE vs oracle 4.4 / 2.8.
   **B2 over-predicts by +52-53%** (no churn) with slope 0.50-0.53. **E5 remains unusable**
   (P6 finding): at C = 24 relative bias +1,440% and RMSE vs oracle 22,270.
3. **Distributions.** Contractual C-index 0.56-0.61 (Harrell = Uno: nothing is censored before
   W); D-calibration p 0.15-0.99 for every model; worst risk-decile gap between observed survival
   and S(12) 0.04-0.08. E4 PIT KS p 0.88 / 0.24; P(alive) worst decile gap 0.02 / 0.06 against the
   latent state.
4. **Selection.** Contractual: SELECTED in every setting (M10 C = 30: E2; signals Tier A: E1;
   signals Tier B: E3). E5 is a candidate under A9 and qualifies at signals C = 36 Tier A and
   C = 42, but fails the calibration gate at C = 36 Tier B (CITL +7.1%); it is not
   selection-eligible at M10 C = 30. **B1 never qualifies** (CITL -12% to -18%, slope 0.40-0.46).
   Non-contractual: C = 24 **NO QUALIFIED MODEL** (E4 CITL +22%, slope 0.56; B2 fails) with **no
   reference model shown** (A10); C = 30 E4 selected. E5 is not selection-eligible in the
   purchases scenario at C = 24 and 30 (A9).
5. **Interval coverage (reported, not asserted).**
   - **Contractual predictive intervals are far below nominal**: 80% intervals cover 0.58-0.62,
     95% intervals 0.68-0.70 (E1, E2, E3). Cause: the outcome simulation draws paid months but
     values each at the contribution model's *expected* contribution, so month-to-month cost
     noise in the realised value is not represented; when the paid-month distribution is
     concentrated (most customers survive the window) the interval nearly collapses. Not changed
     post hoc (`COVERAGE_REPORTING_RULE`); these pre-A11 results are kept (record §4).
   - **After A11** (record §4a, the same paid-month draws plus pre-cutoff contribution
     residuals; nothing tuned or rescaled): 80% intervals cover **0.786-0.797** (8 of 11 rows
     labelled consistent with nominal, 3 below, at 0.786-0.787), 95% intervals **0.960-0.970**
     (all labelled above nominal). Mean widths grow about 10% (80%) and 13% (95%).
     **Confirmation on new development draws 5711-5713** (record §4b): pre-A11 0.578-0.639 /
     0.679-0.721; A11 **0.787-0.810** / **0.959-0.970** (80%: 10 of 11 consistent with
     nominal, 1 below at 0.787; 95%: all above nominal). The remaining 95% over-coverage is
     reported, not corrected.
   - **Parameter intervals vs the oracle**: E1/E2 0.70-0.75 (80%) and 0.88-0.92 (95%), below
     nominal (contribution-model uncertainty is not propagated; boundary cells are held fixed);
     E3 0.32-0.55 (80%), dominated by E3's bias against the oracle rather than its variance.
   - **E4**: parameter intervals 0.78-0.85 / 0.96-0.98, predictive 0.91-0.92 / 0.97 (above
     nominal: conservative).
   - **S2** (half-sample centring retained): above nominal everywhere (0.86-0.95 / 0.96-0.99),
     as in P6.
6. **Customer clustering matters.** CR1 / model-based SE: E3 hazard median 2.2-2.7 (max 3.2); E5
   amount part 1.6-1.8 (contractual) and 2.5-2.9 (non-contractual); E5 positive part about 1
   (contractual) and 2.2-2.5 (non-contractual). Model-based SEs of the penalised models are
   materially optimistic.
7. **S3 IPCW demonstration** (E2, signals C = 36, synthetic independent censoring): complete-data
   integrated Brier 0.1823; IPCW 0.1828 (20% censored) and 0.1810 (50%); naive 0.1861 and 0.1905.

## Conventions chosen in P7 (for review)

1. **Tier B selection**: Tier-A-only models (B1, E1, E2) enter with their Tier A predictions; E3
   and E5 enter with Tier B (`select_model` requires unique ids).
2. **Undefined predictions**: A10 states the rule for B1; `selection_report` applies it to every
   model (a model with any non-finite validation prediction cannot qualify).
3. **Parametric draws**: 200 per customer (`CLUSTERED_INFERENCE.refit_bootstrap_resamples`
   reused; the protocol names no parametric-draw count); equal-tailed percentile intervals.
4. **Calibration line SE**: HC1 (the protocol says "OLS, HC").
5. **Counts**: log score, CRPS and PIT are on paid months (contractual) and purchases (E4); a
   value-level CRPS would need a contribution distribution, which no model has.
6. **Paired bootstrap**: multinomial customer weights (identical to resampling customers),
   2,000 resamples, percentile interval at 95%.

## Decisions requiring owner approval

1. ~~Contractual predictive intervals~~ — **resolved by A11** (finding 5). Remaining 95%
   over-coverage (about 0.96-0.97) is reported, not corrected; confirm.
2. **Parameter-uncertainty scope**: survival parameters only (contribution model fixed), 200
   draws — confirm or extend.
3. **Conventions 1-2 above** (Tier B entry; undefined rule for all models) — confirm.
4. Still open from P6: non-contractual E5 (unusable as specified; kept as a candidate under A9 as
   instructed) and contractual E5 extrapolation. S2 half-sample centring is retained (owner).

## Limitations and remaining blockers

1. **Repeated-sampling coverage is not run**: P7 provides the per-draw coverage summaries; the
   repeated-sampling study across generator draws is P8's and needs the pending preflight
   (`PENDING_BEFORE_MAIN`), so main and fresh seeds remain refused.
2. One development draw per scenario: coverage labels and selection outcomes are descriptive.
3. E3 parameter draws use the CR1 covariance with the band basis included; E5 parameter
   uncertainty is reported as CR1 SEs only (its intervals are S2's).
4. Customer resampling still lives in `landmarks.py` (P2 code, unchanged); the new clustered
   bootstrap is in `statistics/predictive_metrics.py`.
5. **A11 scope**: residuals are pooled across customers within the source group and drawn
   independently per month (the pre-cutoff within-customer correlation is about 0.01 and the
   lag-1 correlation about 0.00); uncertainty in the contribution *means* is still not
   propagated (decision 2); E4 is unaffected (it already simulates spend).
6. **Runtime** (preflight input): record 124 s including the A11 confirmation (14 s); E4 parametric simulation is the largest cost
   (≈ 19 s per cutoff at 200 draws), E3 fit ≈ 2-3 s, selection ≈ 1.3 s per setting; new P7 tests
   ≈ 10 s.
