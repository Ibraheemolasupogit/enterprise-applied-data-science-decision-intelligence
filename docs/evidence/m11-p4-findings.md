# Milestone 11 P4 — E3 (landmark discrete-time hazard) and S1 (Cox): findings

Status: **implemented with amendment A8, awaiting owner review**. Numbers:
[m11-p4-survival-models.md](m11-p4-survival-models.md)
(`python scripts/report_m11_p4_survival.py`; development seeds only). The only protocol change
is **A8** (stage `P4-amended`, manifest `a86df2f1…a5ba`, code `b9675dcf…b2c6`;
[m11-protocol-amendments.md](m11-protocol-amendments.md)). Generators, signal parameters, the
feature registry, tolerances and seeds are unchanged; pilot, main and fresh seeds were not used.
P0–P3 code and evidence are untouched (historical digests kept in the amendment record). P5–P8
are not started.

## Scope (the approved phase table)

P4 = **E3** (discrete-time landmark hazard x contribution by age; essential; Tier A and B) and
**S1** (Cox proportional hazards, landmark covariates, Efron ties, Breslow baseline;
secondary). Gate: tenure-only E3 reproduces M10's `paid_month_km` hazards exactly; S1 matches
statsmodels `PHReg` (Efron) to 1e-6.

**P2 omission, closed here.** The approved P2 row also listed
`statistics/penalised_regression.py` (gate: at penalty 0 it matches M3's `fit_logistic` and
statsmodels' GLM; finite-difference gradient checks). It was not built in P2; E3 needs it, so it
is added in P4 with that gate.

| Module | Content |
|---|---|
| `statistics/penalised_regression.py` | L2-penalised logistic regression under `PENALTY_CONVENTION` (total log-likelihood, penalty mask, frequency weights, damped Newton, inverse penalised Hessian); exact separation check (M3's `is_separated` on the unpenalised block); `Standardiser` (weighted, `ddof = 0`, SD-0 columns dropped); overflow-free `expit` |
| `customer_value/survival.py` | Window outcomes and person-months; tenure bands and the merging rule; the penalty-convention design with deterministic duplicate removal; E3 (A8 covariates), temporal penalty tuning (`choose_penalty`), prediction, validation inputs and selection eligibility; S1 (Efron partial likelihood, Breslow baseline), prediction |

## Amendment A8 and the confirmed conventions

1. **E3 covariates (A8).** E3 uses the tier's registry features except `acquisition_cohort`,
   `tenure_months` and `paid_months` (`E3_EXCLUDED_COVARIATES`); tenure enters only through the
   unpenalised tenure-band basis on the decision age. The registry is unchanged; **S1 is
   unchanged** and keeps all registry features.
2. **Tuning loss** (`E3_TUNING_LOSS`): squared error of window (12-month) value at the validation
   landmark; ties to the larger penalty (`survival.choose_penalty`, tested).
3. **Selection eligibility** (`E3_SELECTION_ELIGIBILITY`): with no tuning landmark whose labels
   are complete before the validation landmark, there is no validation fit and E3 is marked *not
   selection-eligible* at that cutoff (M10 C = 24 and 30; W = 24 at C = 33). No censored
   validation windows.
4. **Duplicate columns** (`DESIGN_DUPLICATE_RULE`): after standardisation on the rows of the
   fit, a column identical to an earlier one in design order (registry order, then one-hot
   indicators) is dropped and reported (`paid_months` in pre-A8 E3 and in S1); independent of row
   and column order (tested).

Other specification choices: the window outcome comes from the cancellation status as recorded
by `C` (identical to P2's `paid_in_window`, tested); decision `k` is taken at tenure
`a_L + k - 1`; tenure bands `[0, 1)` and `[1, 2)` are always empty on landmark rows (a complete
month is required, so tenure >= 2) and merge into `[2, 3)`, reported as `[0, 3)`;
`promo_months_remaining` has SD 0 on landmark rows (the policy runs 2 months) and is dropped by
the protocol's SD-0 rule; for `lambda > 0` the objective is strictly convex, so one Newton start
suffices.

## Correctness gates (tests) — all pass

| Gate | Result |
|---|---|
| **Tenure-only E3 = M10 `paid_month_km`** (one band per age, E3's own basis and solver; C = 24 and 30) | hazards equal to 1e-10 (measured 2.8e-17); survival equal; with the protocol bands, band hazard = pooled events / at risk |
| **S1 = statsmodels `PHReg` (Efron)**, heavily tied monthly times | coefficients 1e-6 (measured 4e-17), log-likelihood 1e-10, SEs 1e-6; Breslow baseline equal (statsmodels reports `H0` just before each time) |
| Penalised regression at penalty 0 vs M3 `fit_logistic` and statsmodels GLM | coefficients 1e-9 (measured 0 and 9e-16); SEs 1e-7 |
| Gradient and Hessian vs finite differences; stationarity; mask; weights; separation | all hold (separation detected exactly) |
| **E3 recovers its own generating parameters** (60,000 landmark rows) | every \|z\| <= 3.5; joint Wald chi-square below the 0.999 quantile |
| A8 | E3's design excludes the three covariates; the pre-A8 path still contains them; S1 keeps them |
| Tuning rule, eligibility, duplicate rule | minimum MSE, ties to the larger penalty; ineligibility labelled; deterministic |
| Labels, leakage, tuning information, status, reproducibility, no truth access | as before: all pass |

## Development-grid results (record; E3 with A8)

**Validation gates** (validation landmark only; never the holdout): every *eligible* setting
qualifies — signals PH and NPH at C = 36 and 42, Tier A and B (8 of 8; validation
calibration-in-the-large -1.7% to 0.0%, slopes 1.00-1.05). *Not selection-eligible*: M10 C = 24
and 30 (all four `v`), W = 24 at C = 33 (12 settings). **No gate failures.**

**Oracle comparison (holdout; excess = loss above the oracle of the same tier, as % of the
oracle MSE).**

| Setting | E3 (A8) | S1 (unchanged) | E2 (P3) |
|---|---|---|---|
| Signals PH, Tier A, C = 36 / 42 | 0.11% / 0.14% | 0.23% / 0.38% | 0.24% / 0.21% |
| Signals PH, Tier B, C = 36 / 42 | 1.2% / 1.2% | 1.3% / 1.5% | — |
| Signals NPH, Tier A, C = 36 / 42 | 0.08% / 0.08% | 0.10% / 0.11% | 0.21% / 0.26% |
| Signals NPH, Tier B, C = 36 / 42 | 1.1% / 1.1% | 1.1% / 1.1% | — |
| Signals W = 24 at C = 33, Tier A (PH / NPH) | 0.35% / 0.86% | **11.6% / 26.6%** | — |
| M10 v = 0 / 0.25 / 0.5 / 1, C = 24 | 0.22 / 0.26 / 0.50 / 0.43% | 0.21 / 0.35 / 2.7 / 4.6% | 0.34 / 0.58 / 0.29 / 0.31% |
| M10 v = 0 / 0.25 / 0.5 / 1, C = 30 | **1.31** / 0.30 / 0.23 / **0.77%** | 0.20 / 0.80 / 1.7 / 0.59% | 0.12 / 0.19 / 0.13 / 0.31% |

**Holdout calibration (E3, A8).** Calibration-in-the-large within ±2.4% in every signals setting
(W = 24 included: +0.8% to +1.2%); in M10 within ±2.9% except v = 0, C = 30 (-3.7%); slopes
0.92-1.07.

**PC3 preview** (descriptive, 5 draws, C = 36; not the pre-registered test): realised MSE minus
E3 Tier B — E3 A +13.8 (SE 1.2), E2 A +14.5 (1.4), S1 A +17.5 (0.7); oracle A minus oracle B +21.0
(1.7). E3 realises about 66% of the oracle's Tier B gain.

## A8: before and after (record §4)

| Effect | Before (registry covariates) | After (A8) |
|---|---|---|
| W = 24 at C = 33 (PH / NPH, Tier A) | mean(pred - oracle) -18.3 / -24.9; holdout CITL -21% / -34% | -0.01 / -0.02; +0.9% / +0.9% |
| M10 v = 0.5 / 1, C = 24 | -1.9 / -4.2 | +0.03 / -0.43 |
| Validation-gate failures (eligible settings) | 5 (all NPH, PH C = 42 Tier B) | **0** |
| **Regression flags** (RMSE +5%, gate lost, or CITL crossing 5%) | — | **1 of 20**: M10 v = 1, C = 30, RMSE vs oracle 1.32 -> 2.06, holdout CITL -2.1% -> -2.9% (within tolerance; E3 not selection-eligible there) |
| Unflagged worsening | — | M10 v = 0, C = 30: RMSE 3.01 -> 3.14 (+4.3%), CITL -3.2% -> -3.7% |

**Cause of the regression (diagnosed, not changed).** With strong heterogeneity the hazard
declines within the wide frozen tenure band `[12, 24)`; the removed linear tenure term had partly
absorbed that decline. A diagnostic with one band per age (not the protocol) gives RMSE 1.06 at
M10 v = 1, C = 30 (better than pre-A8 1.32) and 1.92 at v = 0, C = 30, but 2.14 at v = 1,
C = 24 (worse than A8's 1.60): a bias-variance trade-off, not a uniform improvement.
`TENURE_BAND_EDGES` is frozen and was not changed.

The flagged regression is not a gate failure and is in a setting where E3 is not
selection-eligible; per the stop rule ("material new failures"), A8 introduced **no new gate
failures**.

## Limitations

1. **S1's extrapolation limitation (unchanged by design).** S1 keeps `acquisition_cohort`,
   `tenure_months` (and duplicate `paid_months`) as linear covariates frozen at the landmark and
   has no tenure basis, so with few landmarks it extrapolates them: W = 24 at C = 33, mean(pred
   - oracle) -14.3 (PH) and -22.3 (NPH), holdout CITL -17% and -30%; M10 v = 1, C = 24, -3.3
   (CITL -5.6%). S1 is secondary (no acceptance weight) and is reported as is.
2. **Coarse tenure bands under strong heterogeneity** (above): E3 can be biased where hazards
   change within a band (M10 C = 30: v = 1 -1.0, v = 0 -1.6). A finer or different basis would be
   a separate amendment.
3. **E3 is not selection-eligible** at M10 C = 24 and 30 and at W = 24 (no validation fit
   exists); its holdout results there are reported, not selectable.
4. **Frozen covariates** (protocol design): signals at `C` are used for the whole window; no
   model of covariate evolution.
5. **Model-based covariances are optimistic**: customers repeat across landmarks; clustered
   (CR1) inference is P7's.
6. **Feature conventions** carried from P2: `usage_change_3m = 0` with an empty previous window
   and `months_since_payment_failure` capped at 24 enter as numeric columns without indicators.
7. **Penalty tuning is nearly flat** (validation MSE changes in the fourth significant figure);
   it often selects a grid edge (0.01 or 100).
8. **Selection is not run** (needs B1 and paired customer-bootstrap intervals; P7).
9. **Runtime** (preflight input): E3 fit including temporal tuning about 1.2-1.5 s per tier and
   cutoff at full scale (0.4 s with the fixed penalty); S1 about 0.2 s; predictions under 0.1 s.
