# Milestone 11 P3 — E1 (sBG) and E2 (exponential-Gamma): findings

Status: **closed at P3 with amendments A6 and A7, awaiting owner review**. Numbers:
[m11-p3-contractual-models.md](m11-p3-contractual-models.md)
(`python scripts/report_m11_p3_contractual.py`; development seeds only). The only protocol
changes are the owner-approved narrow geometric boundary exceptions **A6** (E2, `v = 0`) and
**A7** (E1, `alpha + beta -> infinity`) (stage `P3-amended`, manifest `33329a84…a5cc`, code
`2e69fb2d…23b5`; [m11-protocol-amendments.md](m11-protocol-amendments.md)). Generators and signal parameters
are unchanged; the pilot, main and fresh seeds were not used. P0–P2 code and evidence are
untouched. P4–P6 (E3, S1, E4, E5, S2) are not started.

## Scope (the approved phase table)

P3 = **E1** (shifted-beta-geometric) and **E2** (exponential-Gamma frailty, discrete Lomax),
contractual, Tier A, per segment x channel cell with the pooled fit as the only fallback.
Gate: likelihoods match brute force; parameters recovered within 3 SE; the published sBG
example matches (values checked against the paper first); E2 reduces to the geometric model as
`v -> 0`.

| Module | Content |
|---|---|
| `customer_value/contractual.py` | E1/E2 log-survival, event likelihood and analytic score; ML fit (5 deterministic starts, L-BFGS-B in a log-parameter box, Newton polish, observed information by central differences of the score); fit status via `model_selection.classify_fit`; per-cell fits with the pooled fallback; cell x promotion-state contribution model; customer conditional value and `paid_1..paid_H`; cohort (new-customer) value with divergence flag; temporal validation on the validation landmark |

**Information and leakage.** Fits use M10's `observe` at the cutoff (durations: churned = event
at `t` paid months, active = censored at `t - 1`, M10's discrete-time rule). Contribution uses
completely recorded months only. The prediction population is the features' / oracles'
population. **No hyperparameters**: E1/E2 have no penalty (the protocol's penalty grid applies
to E3/E5), so there is nothing to tune; temporal validation refits each model with the
information of `L_v = validation_landmark(C, W)` only and scores its completely recorded labels.

## Correctness gates (tests)

| Gate | Result |
|---|---|
| Likelihood vs brute-force integration over the latent churn parameter (quantile form; E1 and E2, events and censoring, `t` up to 40) | equal to 1e-9 |
| Analytic score vs finite differences | 1e-6 relative |
| **Published sBG example** (Fader & Hardie 2007, J. Interactive Marketing 21(1), Table 1, years 1-7; checked against the paper) | High End 0.6681 / 3.8061, LL -1,611.16 (paper 0.668 / 3.806 / -1,611.2); Regular 0.7041 / 1.1820 (paper 0.704 / 1.182). The body text's "0.688" is a typo; Appendix B gives 0.668 |
| **E2 -> geometric** as `v -> 0` | log-likelihood within `10 v` relative; survival to 1e-10; geometric expected counts give `v` at the boundary (status `unstable`, KKT-converged) with `b` exact to 1e-6 |
| E2 survival = the M10 generator's `(1 + v b a)^(-1/v)`; E2 conditional value with true parameters and true contribution = P2 oracle | 1e-12 |
| **Parameter recovery** (30,000 simulated customers, 2 settings per model) | every log parameter within 3 SE; all 5 starts agree to 1e-6 |
| Recovery by cell on the M10 generator (`ok` cells, both cutoffs) | joint Wald chi-square <= 0.999 quantile, every \|z\| <= 3.5; in the record all 24 cell fits are `ok` with \|z\| <= 1.6 |
| Status rules | no data -> `undefined`; < 30 events -> `unstable`; no events -> never `ok`; unstable cells use the `ok` pooled fit labelled `fallback`; extreme parameters and durations give finite likelihoods |
| Reproducibility | identical output on repeat; invariant to row order (1e-9) |
| Leakage | deleting or scrambling every row recorded after `C` leaves predictions unchanged; validation predictions equal a fit on the tables as of `L_v`, and changing rows recorded after `L_v` changes labels but not predictions |
| No aggregate bias (correct specification, M10, 8 draws in the test; 10 in the record) | mean of `mean(E2 - oracle)` = -0.03 (z = -0.17) |
| Calibration against realised value (M10, C = 24, 30) | mean residual and slope within 4 SE |
| Estimation code never reads truth, oracles, generator config or the trap | AST test |

## Comparison with the P2 oracles (record, development seeds)

Record seeds (M10 5101, signals 5102); E2 with A6. Before-A6 values in brackets where A6
changed the result.

| Setting | E2 vs oracle | Excess / oracle MSE | Status |
|---|---|---|---|
| M10, v = 0.5 (correctly specified), C = 24 / 30 | mean diff +0.18 / +0.10; RMSE 1.31 / 0.75 (oracle mean about 54) | 0.26% / 0.08% | 12/12 cells ok |
| M10, v = 1.0 | RMSE 1.17 / 0.61 | 0.23% / 0.06% | 12/12 ok |
| M10, v = 0.25 | RMSE 1.97 / 1.24 [2.18 / 2.45] | 0.53% / 0.21% [0.65% / 0.83%] | 12/12 ok, 1 / 2 at the v = 0 boundary [1 / 2 fallback] |
| M10, **v = 0** (homogeneous churn) | RMSE 1.46 / 0.93 [5.18 / 4.78]; holdout slope 0.995 / 1.008 [1.179 / 1.190] | **0.29% / 0.11%** [3.6% / 3.0%] | 12/12 ok, 6 / 5 at the boundary [6 / 5 fallback]; validation qualified at both cutoffs [not at C = 30] |
| Signals (PH), Tier A (E2 misspecified), C = 36 / 42 | RMSE 1.37 / 1.38 | 0.26% | 12/12 ok |
| Signals (NPH), Tier A | RMSE 1.66 / 1.56 [2.0 / 2.3] | 0.36% / 0.31% [0.53% / 0.70%] | 12/12 ok, 3 at the boundary [3 fallback] |

Sample size (M10, v = 0.5, 5 draws per scale): RMSE(E2 - oracle) 3.15, 1.37, 0.97, 0.38 at
0.25x, 1x, 4x, 16x acquisition — about `1/sqrt(n)` (64x the customers, 1/8 the excess). The
excess is mostly survival-parameter estimation; the contribution model's part is smaller
(about 0.1-0.4) and also shrinks.

## Scientific findings

1. **E1 and E2 give nearly identical predictions** (equal to 3-4 significant figures). Not a
   bug: for small monthly churn, sBG's `S(a) = B(alpha, beta + a) / B(alpha, beta)` is
   `(1 + a / beta)^(-alpha) (1 + O(1/beta))`, the E2 form with `v = 1/alpha`,
   `b = alpha / beta`. They separate only when per-period churn is large (the published
   annual example). On monthly data the fitted models are statistically indistinguishable,
   and with A6 and A7 both use the same geometric limit at the boundary (customer predictions
   equal to 1e-9 where both apply).
2. **Status rule at v = 0 — resolved by amendment A6.** Under homogeneous churn the correctly
   specified MLE is `v = 0`, on the boundary, so `|log v| > 10` marked the cell `unstable` and
   the protocol substituted the pooled fit, which pools cells with different hazards and so
   shows apparent frailty: miscalibrated (slope 1.16-1.19, excess 3-6%). A6 (owner-approved)
   uses the verified geometric limit instead; see *Amendment A6* below.
3. **`v` is weakly identified in small cells.** With about 100 events a cell's SE of `v` is
   about 0.5, so its MLE can reach the boundary even when the true `v` is 0.25-1 (1-3 cells
   per fit). Before A6 these cells fell back to the pooled fit; with A6 they use their own
   geometric limit, which is better on average (below). Boundary and fallback counts are
   always reported.
4. **Tier A leaves little to gain on the signals scenario.** Misspecified E1/E2 are within
   0.3% (PH) and 0.5-0.7% (NPH) of the Tier A oracle's MSE; Tier A information is survival,
   segment and channel only, so the room for any Tier A model is small. Tier B (P2: 2.3-3x the
   MDE) is where E3 must show its value.
5. **Cost.** Fit + predict is about 0.1 s per model per cutoff at full scale (12 cell fits +
   pooled); temporal validation the same. Negligible for the repeated-sampling preflight.

## Amendment A6: the narrow E2 `v = 0` boundary exception

**Rule** (`predictive_protocol.E2_GEOMETRIC_BOUNDARY`; applied in `contractual.geometric_boundary`
and `fit_cells`). An E2 fit, cell or pooled, is replaced by its `v = 0` limit, the geometric
model, and counted `ok` (labelled `v=0 boundary (geometric)`) only if **all** hold:
`unstable` with reasons exactly *parameter at the log-scale bound* (optionally *Hessian not
positive definite*); `log v < -10` and `|log b| <= 10`; converged with a finite likelihood;
the closed-form geometric MLE `b = -log(1 - events / decisions)` within 1e-4 of the E2 `b`
(log scale); geometric log-likelihood not lower (relative 1e-9); the one-sided score for `v` at
`(b, v = 0)` not positive (<= 1e-6 per customer: `v = 0` is a boundary maximum); and the
one-parameter fit `ok` under `STATUS_RULES` (>= 30 events, >= 10 per parameter). Every other
case keeps the frozen rules: E1 is never covered; `v` at the upper bound, `b` at a bound, any
other reason (too few events, start disagreement, non-convergence) or any failed check keep the
original status and the pooled fallback. `classify_fit`, `STATUS_RULES` and `FIT_STATUSES` are
unchanged.

**Independent verification** (`tests/unit/customer_value/test_contractual_boundary.py`):

| Check | Result |
|---|---|
| Geometric MLE vs a 1-D bounded maximisation of a separately written geometric likelihood | `b` to 1e-7, log-likelihood to 1e-12; information of `log b` vs a second difference to 1e-5 |
| E2 likelihood at `v = 0` (new exact branch) = geometric likelihood | 1e-12; survival `exp(-b a)` to 1e-14 |
| One-sided `v` score vs Richardson-extrapolated difference quotient of the E2 likelihood | 1e-4 relative; positive under frailty (`v = 0.5`) |
| Convergence: profile log-likelihood over `v` (b re-optimised by an independent 1-D search) on geometric expected counts | maximised at `v = 0`, strictly decreasing over `v` = 1e-4 .. 0.5 |
| Narrowness | each excluded case returns no substitute: extra reasons, `failed`, `ok`, non-convergence, `v` at the upper bound, `b` at a bound, `b` disagreement, a higher E2 likelihood, positive `v` score (frailty data), < 30 events, E1 |
| Pre-A6 reproduction | `boundary_exception=False` reproduces the pre-A6 predictions exactly (48 seed x v x cutoff x model cases); E1 is identical with or without A6 |
| Calibration at `v = 0` (statistical) | mean residual and slope within 4 SE at C = 24, 30; relative aggregate bias against the oracle within 2.5% and oracle excess below half the pre-A6 value (6 draws). **Correction:** the first A6 report stated "no aggregate bias against the oracle" at `v = 0`. That gate was mis-specified: at `v = 0` the dispersion MLE is truncated at the boundary and positively biased, so a small upward bias (+1.0% at this scale, +0.6% at 4x, shrinking as `n^-1/2`; record §7) is expected. The A6 test passed on its seeds; on other seeds (4852-4857) it fails. It is replaced by the bound above |

**Before / after on the PC2 development grid** (10 independent draws, 5141-5150; record §5):

| v | C | excess / oracle MSE | holdout slope | fallback cells | boundary cells | validation-qualified draws |
|---|---|---|---|---|---|---|
| 0 | 24 | 4.35% -> 0.26% | 1.171 -> 0.994 | 65 -> 0 | 65 | 9 -> 10 |
| 0 | 30 | 3.79% -> 0.20% | 1.155 -> 0.990 | 71 -> 0 | 71 | 6 -> 10 |
| 0.25 | 24 / 30 | 0.97% / 0.90% -> 0.39% / 0.37% | 0.994 / 1.008 -> 0.985 / 0.999 | 20 / 13 -> 0 | 20 / 13 | 9 / 9 -> 10 / 10 |
| 0.5 | 24 / 30 | 0.51% / 0.40% -> 0.39% / 0.31% | 1.002 / 1.007 -> 0.999 / 1.008 | 9 / 4 -> 0 | 9 / 4 | 10 / 10 -> 10 / 10 |
| 1.0 | 24 / 30 | 0.34% / 0.17% -> 0.35% / 0.17% | 0.998 / 0.999 -> 0.998 / 0.999 | 3 / 1 -> 0 | 3 / 1 | 10 / 10 -> 10 / 10 |

At `v > 0`, 32 of 60 fits changed: mean change in excess -0.22 percentage points, largest
increase +0.09 points (one draw), largest decrease -2.25 points. The correction removes the
`v = 0` miscalibration and does not degrade `v > 0`: where a small cell's `v` reaches the
boundary, its own geometric fit is a better substitute than the pooled fit.

## Amendment A7: the narrow E1 geometric boundary exception

**Geometric limit of E1.** With `theta ~ Beta(alpha, beta)`, write `mu = alpha / (alpha +
beta)` and `phi = 1 / (alpha + beta)`. Then `Var(theta) = mu (1 - mu) phi / (1 + phi)` and
higher central moments are `O(phi^2)`, so as `phi -> 0` (`alpha, beta -> infinity`, `mu` fixed)
`Beta` collapses to `mu` and `S(a) = E[(1 - theta)^a] -> (1 - mu)^a`: the geometric model with
monthly churn `mu` (the same model as E2's `v = 0` limit, `b = -log(1 - mu)`). The other E1
boundaries (`alpha` or `beta -> 0`, or one of them alone `-> infinity`) are degenerate (all
mass at 0 or 1, or no churn) and never qualify. The one-sided score at `phi = 0` is
`d log E[g(theta)] / d phi = g''(mu) mu (1 - mu) / (2 g(mu))`: `a (a - 1) mu / (2 (1 - mu))`
for a censored `S(a)` and `-(t - 1) + (t - 1)(t - 2) mu / (2 (1 - mu))` for an event at `t`.

**Why `not converged` is admitted.** Under homogeneous churn the supremum is at `phi = 0`,
i.e. infinite `alpha, beta`; the iterates drift along a numerically flat ridge
(`log alpha` about 11-13, `log beta` 13.4-15) where no finite point can meet the gradient
tolerance, so most such fits are `failed: not converged` (32 of 47 on a first inventory).
Convergence is instead established exactly at the closed boundary: the closed-form `mu` has
zero score and the one-sided `phi` score is checked. Only that single failure reason is
admitted, and only on the ridge; every other failure reason disqualifies. *(A projected-Newton
variant of the optimiser was tried and withdrawn: it shifted statuses in both directions without
resolving the ridge, and P3 fits must stay bit-identical.)*

**Rule** (`predictive_protocol.E1_GEOMETRIC_BOUNDARY`; `contractual.e1_geometric_boundary`):
status `unstable` (bound reason, optionally Hessian) or `failed` (*not converged* only); finite
likelihood, no separation, start agreement within the frozen 1e-4; on the ridge
(`log(alpha + beta) > 10`, `min(log alpha, log beta) >= -10`); `mu = events / decisions` within
1e-4 (log) of the fit's `mu`; geometric log-likelihood not lower than the fit's (exact
finite-product evaluation, relative 1e-9); `phi` score at `phi = 0` <= 1e-6 per customer; the
one-parameter fit `ok`. The fit is then `ok` with `alpha = beta = infinity` and churn `mu`,
labelled and counted. A6 and all other rules unchanged.

**Independent verification** (`tests/unit/customer_value/test_contractual_e1_boundary.py`):

| Check | Result |
|---|---|
| `phi` score vs exact rational Beta moments (`E[theta^k] = prod (mu + i phi) / (1 + i phi)`, `phi = 1e-15`), events and censoring, `t` = 1..8 | 1e-12 |
| Exact finite-product likelihood vs the fit's `betaln` likelihood (moderate parameters) and vs exact rationals on the ridge (`alpha, beta` = 1.2e5, 1.9e6) | 1e-12; 1e-14 |
| E1 likelihood at `phi` = 1e-6, 1e-8 vs geometric + score x `phi` | 1e-10 |
| Boundary optimality: profile over `phi` (`mu` re-optimised by an independent 1-D search) | below the geometric limit and strictly decreasing over `phi` = 1e-4 .. 0.2; `phi` score negative |
| Substitute | churn = `q` (1e-12); log-likelihood = a separately written geometric likelihood (1e-12); equals E2's A6 limit |
| Narrowness (each returns no substitute) | other failure reasons, other unstable reasons, `ok`/`undefined`, non-finite likelihood, separation, start disagreement, a lower bound, off the ridge, `mu` disagreement, positive `phi` score (frailty data), too few events, very large cells with ridge start disagreement, E2 fits |
| A6 and E2 preserved | with A7 off: all 88 (scenario x seed x v x cutoff x model) predictions and every cell status identical to the A6 state; with A7 on: 24 E1 cases change, no E2 case; E2 does not depend on the A7 switch |
| Calibration at `v = 0` (statistical) | as for A6: slope and mean residual within 4 SE; relative bias within 2.5%; oracle excess below half the pre-A7 value |

**Before / after on the PC2 development grid** (10 draws, 5141-5150; record §6):

| v | C | excess / oracle MSE | holdout slope | fallback cells | boundary cells | validation-qualified draws |
|---|---|---|---|---|---|---|
| 0 | 24 | 4.43% -> 0.34% | 1.172 -> 0.994 | 67 -> 2 | 65 | 9 -> 10 |
| 0 | 30 | 3.80% -> 0.20% | 1.155 -> 0.990 | 72 -> 1 | 71 | 6 -> 10 |
| 0.25 | 24 / 30 | 0.97% / 0.90% -> 0.39% / 0.37% | 0.994 / 1.008 -> 0.985 / 0.999 | 20 / 13 -> 0 | 20 / 13 | 9 / 9 -> 10 / 10 |
| 0.5 | 24 / 30 | 0.51% / 0.40% -> 0.39% / 0.31% | 1.002 / 1.007 -> 0.999 / 1.008 | 9 / 4 -> 0 | 9 / 4 | 9 / 10 -> 9 / 10 |
| 1.0 | 24 / 30 | 0.34% / 0.17% -> 0.35% / 0.17% | unchanged | 3 / 1 -> 0 | 3 / 1 | 10 / 10 -> 10 / 10 |

At `v > 0`, 32 of 60 fits changed: mean -0.22 percentage points of excess, largest increase
+0.09 points (one draw), largest decrease -2.25 points. Signals (NPH), E1: excess 0.53% /
0.70% -> 0.36% / 0.31%. No material regression anywhere; E2 results are byte-identical to the
A6 record.

## Limitations and open items

1. **Distributional outputs.** `paid_1..paid_H` give the full predictive distribution of paid
   months given the point estimates; parameter uncertainty (Hessian / parametric simulation),
   intervals, CRPS and PIT are P7.
2. **Model selection is not run.** `validate` produces `CandidateResult`s and the gates
   (recorded), but selection needs B1, E3 and paired customer-bootstrap intervals (P4, P7).
3. **Cohort value** (`cohort_value`, with the divergence flag; it takes natural parameters, so
   an E1 A7 limit is passed as E2's geometric equivalent `(b, 0)` with `b = -log(1 - mu)`; a
   latent A6 defect — the divergence check divided by `v`, failing at `v = 0` for an
   undiscounted infinite horizon — was fixed during A7: a geometric tail is `ok`) is
   implemented and matches the
   M10 truth with true parameters; PC2's grid evaluation against H4 is P7/P8.
4. **Contribution model assumes contribution is independent of churn**, true in both
   contractual generators (cost noise is independent of the latent state); sparse
   cell x promotion states borrow from segment, then pooled, means (reported).
5. **Convergence convention** (an implementation choice within the protocol's tolerances):
   projected score per customer `<= 1e-6` and a final relative log-likelihood change
   `<= 1e-9`; a boundary optimum with zero projected score counts as converged (then
   `unstable` by the boundary rule). The observed information is a central difference of the
   analytic score (step 1e-5).
6. **No left truncation.** Every customer is observed from acquisition, so the likelihood needs
   no delayed-entry conditioning; data with left truncation would.
7. **E1 at `v = 0` — resolved by A7.** Before A7, E1 cells fell back to the pooled fit under
   homogeneous churn (slope 1.16-1.19); A7 uses E1's own geometric limit (below).
8. **Boundary bias at `v = 0`** (A6 and A7). With the dispersion on its boundary, its MLE is
   truncated at 0, so predictions are biased upward by about +1.0% at full scale and +0.6% at
   4x the customers (record §7): a consistent `O(n^-1/2)` bias, unlike the pre-amendment
   pooled substitute, whose error did not shrink (+0.1% at 1x with three times the variance,
   -1.6% at 4x). Excess MSE against the oracle falls from about 4% to 0.2-0.3% of the oracle
   MSE. Not corrected (no bias correction is in the protocol).
9. **A7 conservatism.** Of the 139 non-`ok` E1 cells on the 10-draw `v = 0` grid, 136 use the
   geometric limit and 3 keep the pooled fallback. In a separate 5-draw inventory the two
   exclusions were a cell with a positive `phi` score (a genuine interior optimum just short
   of the ridge, correctly excluded) and a 67-event cell whose non-converged `mu` missed the
   1e-4 agreement check by 2%. Very large cells (about 60,000 customers) can also fail the frozen start-agreement
   rule (1e-4 in total log-likelihood) along the flat ridge and keep the fallback (tested);
   no development-grid cell did.
10. **E1 likelihood precision on the ridge.** The fitted E1 likelihood uses `betaln`
    differences, which lose about 1e-9 relative precision at `alpha, beta` near e^12-e^15; A7
    compares likelihoods with the exact finite-product form instead. The fit itself is
    unchanged (bit-identical with A7 off).
11. The protocol's one pending item (repeated-sampling preflight) stays open until the P4-P6
   fit costs exist; the main and fresh seeds remain refused.
