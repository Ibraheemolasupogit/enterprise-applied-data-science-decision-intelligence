# Milestone 11 P6 — E5 (two-part penalised GLM) and S2 (split conformal): findings

Status: **implemented; A9 applied; awaiting owner review — three decisions remain open below**.
Numbers:
[m11-p6-predictive-models.md](m11-p6-predictive-models.md)
(`python scripts/report_m11_p6_predictive.py`; development seeds only). The only protocol change
is **A9** (owner-approved at the P6 final review: `E5_TUNING_LOSS`, `E5_SELECTION_ELIGIBILITY`;
stage `P6-amended`, manifest `69773941…28a3`, code `a1af7b47…a9ed`;
[m11-protocol-amendments.md](m11-protocol-amendments.md)). A9 changes no P6 number: the
implementation already followed its wording. Generators, the feature registry, selection gates,
tolerances and seeds are unchanged; pilot, main and fresh seeds were not used. P0–P5 code and
evidence are untouched. P7–P8 are not started.

## Scope (the approved phase table)

P6 = **E5** (two-part penalised GLM, the "predictive ML"; essential; contractual Tier A/B and
non-contractual Tier A) and **S2** (split conformal intervals for E5; secondary). Gate: the GLM
matches statsmodels; conformal achieves finite-sample coverage under exchangeable data.

| Module | Content |
|---|---|
| `statistics/gamma_regression.py` | L2-penalised Gamma GLM, log link, shape estimated jointly (the objective is the total log-likelihood, as `PENALTY_CONVENTION` requires) |
| `statistics/conformal.py` | Split-conformal rank, quantile and interval |
| `customer_value/predictive.py` | E5: part 1 `P(R > 0)` (penalised logistic, P4's module), part 2 `E[R \| R > 0]` (penalised Gamma, log link), prediction = product; design per `PENALTY_CONVENTION` and `DESIGN_DUPLICATE_RULE`; temporal tuning; validation and eligibility. S2: deterministic customer split, one latest landmark per calibration customer, intervals at 80% and 95% |

## Specification choices (for review)

1. **Gamma family with a log link** for part 2 (the plan: "a log-link or Gamma GLM").
2. **"Positive"** means `R > 0`. Contractual windows can be slightly negative (a month whose cost to
   serve exceeds its revenue): 8-85 windows per training stack, mean `min(R, 0)` -0.001 to
   -0.009 per row; they count as "not positive", so the two-part mean omits that mass (reported).
3. **One penalty for both parts**, chosen by squared error of window value at the validation
   landmark, ties to the larger penalty, `DEFAULT_PENALTY` (labelled) if no tuning fit has both
   parts `ok` or tuning is infeasible — **now authorised by A9 (`E5_TUNING_LOSS`)**. (Before A9
   these rules were stated for E3 only and were not independently authorised for E5.)
4. **Selection eligibility**: no validation fit where no tuning landmark exists, so E5 is *not
   selection-eligible* there — **now authorised by A9 (`E5_SELECTION_ELIGIBILITY`)**.
5. **Events** for the status rules: the minority class (part 1); positive windows (part 2).
6. **S2**: customers split by a hash of `customer_id` (no random seed; 50.8% to calibration in a
   20,000-id check); the calibration landmark is each calibration customer's **latest** (the plan:
   "temporally ordered"); intervals are centred on the **half-sample** E5 fit that the guarantee
   requires, not on the full-data E5 prediction.
7. E5 uses its own design builder (same rules as E3's) so that P4's committed `survival.py` is not
   modified; `exclude` is a diagnostic hook only (default: the full registry).

## Correctness gates (tests) — all pass

| Gate | Result |
|---|---|
| **Gamma GLM = statsmodels `GLM(Gamma(log))`** at penalty 0 | coefficients 1e-9 (measured 1.3e-15); deviance 1e-10 |
| **E5's two parts = statsmodels** (Binomial and Gamma-log GLMs on E5's own design, penalty 0, signals stack) | 1e-6 |
| Gamma shape | the first-order condition holds to 1e-12; an independent 1-D search agrees to 1e-6 |
| Gradient and Hessian (beta and log shape) vs finite differences | 1e-6 |
| Penalised stationarity, unpenalised intercept, intercept-only = mean, weights = replicated rows | to 1e-7 / 1e-9 |
| **Two-part identity** (every feature constant) | prediction = P(R > 0) x mean(R \| R > 0) exactly (1e-9), with negative windows counted as not positive |
| **Conformal: exact finite-sample coverage under exchangeability** (20,000 replications, n = 49, heavy-tailed noise, imperfect fixed predictor) | coverage = ceil((n+1) level)/(n+1) within 4 MC SE, at 80% and 95% |
| **S2 pipeline, end to end, on exchangeable customers** (E5 fit, S2's hash split, conformal quantile; 300 replications) | coverage = E[ceil((n+1) level)/(n+1)] within 4 SE, at 80% and 95% |
| **E5 recovers its own generating parameters** (40,000 rows; both parts; shape) | every \|z\| <= 3.5; shape within 5% |
| Leakage (contractual Tier B and non-contractual) | rows recorded after `C` change neither E5 nor S2 |
| Temporal tuning | the validation fit ignores rows recorded after `L_v`; fixed penalty and *not selection-eligible* where tuning is infeasible |
| S2 split | deterministic and order-invariant; one latest landmark per calibration customer; 95% interval contains the 80% one |
| Reproducibility; no truth, oracle or config access | row-order invariant (1e-6); AST test |

## Statistical performance (record; development seeds)

**Contractual.** All fits `ok`. E5 is less accurate than E3 (with A8) in every setting:

| Setting | E5 excess / oracle MSE | E3 excess / oracle MSE | E5 bias vs oracle |
|---|---|---|---|
| Signals, C = 36 / 42, Tier A | 1.1% / 0.6% | 0.26% / 0.15% | +1.8 / +0.8 |
| Signals, C = 36 / 42, Tier B | 2.6% / 1.8% | 1.3% / 1.2% | +2.2 / +0.9 |
| Signals NPH, C = 36 / 42, Tier B | 1.1% / 1.0% | 1.1% / 1.1% | +0.6 / +0.3 (comparable) |
| Signals W = 24 at C = 33, Tier A (PH / NPH) | 1.8% / 26% | 0.42% / 1.1% | +5.0 / +22.1 |
| M10, C = 24 (v = 0 / 0.25 / 0.5 / 1) | 0.8 / 2.8 / 3.5 / 15% | 0.2 / 0.6 / 1.2 / 0.4% | +1.9 / +2.7 / -2.8 / +6.4 |

Holdout calibration-in-the-large -4.9% to +11.2% (W = 24: +8.5% to +35%). **PC5 preview** (descriptive,
5 draws, signals Tier B, C = 36): realised MSE(E5) - MSE(E3) = +12.3 (SE 2.2): E3 is better.

**Validation gates** (validation landmark only): eligible in 8 signals settings, qualified in 6
(PH C = 42: calibration-in-the-large +5.1% / +6.3%); not selection-eligible in the 18 others
(M10, W = 24, and every non-contractual setting at C = 24 and 30).

**Non-contractual: E5 as specified is numerically unusable.** Fits are `ok`, but the log-link
amount part, fitted on short training histories (at C = 24: up to 39 weeks of age and 31 repeat
purchases), extrapolates exponentially at `C` (up to 100 weeks and 64 purchases): the largest
prediction is 16,413 against an oracle maximum of 1,718 (base, C = 24) and up to 2.4 million in
the variants; RMSE against the oracle 91-419 (base) and up to 30,100 (variants), against 2.3-17.8
for E4. Clipping each feature to its training range halves the error but does not fix it
(RMSE 187 / 40 at C = 24 / 30).

**Extrapolation diagnostic (contractual; not the protocol model).** Clipping to the training
range reduces E5's bias (M10 v = 1, C = 24: +6.4 -> +1.2; RMSE 9.4 -> 3.0); removing A8's three
covariates over-corrects (-2.6). Neither is applied.

**S2 holdout coverage** (reported, not asserted): over-coverage in nearly every setting — 80%
intervals cover 0.82-0.97, 95% intervals 0.95-0.997. The calibration windows (latest landmark
before `C`, with its churn and purchase mix) and the window after `C` are not exchangeable: the
customers present at `C` have longer tenures and lower churn. The finite-sample guarantee holds
only under exchangeability (verified by the gate), so this is the expected, disclosed
non-exchangeability, here conservative.

## Decisions requiring owner approval

1. **Non-contractual E5.** As specified (registry features entering a log-link GLM linearly) E5
   is unusable on the purchases scenario. Options: report as specified (E5 will lose PC-style
   comparisons by orders of magnitude and never qualify); or amend E5's non-contractual design (for
   example clipping to the training range, or a transformation such as `log1p` of counts, ages and
   spend), with before/after evidence.
2. **Contractual E5 extrapolation.** Same structural cause as A8 for E3, but A8 explicitly excludes
   E5. Options: keep E5 as specified (report the bias), or amend (clipping performed best in the
   diagnostic).
3. ~~Tuning loss and selection eligibility for E5~~ — **resolved by A9** (P6 final review).
4. **S2 centring** (choice 6): intervals around the half-sample fit (valid under exchangeability)
   rather than the full-data E5 prediction; confirm.

## Unresolved items carried forward (not P6 blockers)

- **E3's no-ok-fit penalty fallback is unauthorised** (P4 code): A9's equivalent clause is E5's
  only. Recorded as an open item for review before P8 in
  [m11-protocol-amendments.md](m11-protocol-amendments.md); no code changed.
- B2's definition (amendment needed before P7) and the non-contractual reference fallback
  (P7 reporting) — P5 dispositions, recorded there too.

## Limitations

1. E5 is a conditional mean with conformal intervals, not a full predictive distribution (the
   protocol: interval coverage only, no CRPS).
2. Frozen covariates, as for E3; customers repeat across landmarks, so model-based covariances
   are optimistic (clustered inference is P7's).
3. S2's guarantee is marginal and needs exchangeability; under drift its coverage at `C` can be
   above (observed here) or below nominal.
4. The omitted negative window mass biases contractual E5 upwards by at most about 0.009 per
   customer.
5. **Runtime** (preflight input): E5 fit including temporal tuning 0.1-0.8 s per setting (0.35 s
   on average), validation 0.04 s, prediction 0.02 s, S2 0.2 s per cutoff at full scale.
