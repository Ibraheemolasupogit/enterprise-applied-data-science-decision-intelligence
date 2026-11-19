# Milestone 11 P5 — E4 (BG/NBD + Gamma-Gamma): findings

Status: **implemented; owner disposition recorded below; awaiting P5 closure**. Numbers:
[m11-p5-btyd-models.md](m11-p5-btyd-models.md) (`python scripts/report_m11_p5_btyd.py`;
development seeds only). **No protocol change** (stage `P4-amended`, manifest
`a86df2f1…a5ba`, code `b9675dcf…b2c6`, amendments A1–A8); generators, the feature registry,
tolerances and seeds are unchanged; pilot, main and fresh seeds were not used. P0–P4 code and
evidence are untouched. P6–P8 are not started.

## Scope (the approved phase table)

P5 = **E4**: BG/NBD purchases + Gamma-Gamma spend with P(alive), fitted per segment x channel
cell (`E4_FIT_UNIT`, amendment A2), the pooled fit the only fallback and a labelled
misspecified reference (`E4_POOLED_REFERENCE`). Gate: likelihoods match quadrature; conditional
expectations match Monte Carlo; parameters are recovered.

`customer_value/btyd.py` (the module named in the plan): E4's sufficient statistics from the
tables as recorded by `C` (completely recorded months only, as the features and the oracle);
BG/NBD and Gamma-Gamma log-likelihoods with analytic scores; ML fits on log parameters
(5 deterministic starts, L-BFGS-B, Newton, the frozen status rules); per-cell fits with
component-wise pooled fallback; conditional window expectations and P(alive) (its own
Golub-Welsch Gauss-Jacobi rule on the exact Beta posterior; the P2 rule forbids importing the
oracle); value; temporal validation at the validation landmark.

## Specification choices (within the protocol; for review)

1. **Gamma-Gamma uses every observed spend**, the acquisition purchase included (the full
   individual-spend likelihood; the published mean-only form drops the log-spend term, which
   carries information on `p`). Every customer therefore enters the spend fit.
2. **Value per purchase** = the business's recorded contribution-to-revenue ratio over completely
   recorded purchases (0.37000 in every scenario: the generator's margin less fee, up to cent
   rounding); E4 never reads the margin parameters.
3. **Component-wise fallback**: BG/NBD and Gamma-Gamma each fall back to their own pooled fit.
4. **Events** for the status rules: repeat purchases (BG/NBD), observed spends (Gamma-Gamma).
5. **Quadrature**: 64 Gauss-Jacobi nodes (the value pinned for the oracle's identical integral);
   doubling changes nothing (tested).

## Gates (tests) — all pass

| Gate | Result |
|---|---|
| **BG/NBD likelihood vs 2-D quadrature** of the raw likelihood over (lambda, p), `a < 1` and `a > 1`, `x = 0 .. 9` | 1e-8 |
| **Gamma-Gamma likelihood vs 1-D quadrature** over `nu` of the product of Gamma spend densities | 1e-10 |
| **Conditional expectations vs Monte Carlo** (10^6 likelihood-weighted prior draws, `a = 0.8`): window purchases in two windows, P(alive at T), P(alive at the end of C) | within 4.5 MC SE |
| Conditional expectations vs the published Fader-Hardie-Lee (2005) closed form (`a > 1`) | 1e-11 (P(alive) 1e-13) |
| Conditional expectations with the **true parameters vs the independent P2 oracle** (all 12 cells) | purchases 1e-9 (measured 3.5e-12), P(alive) 1e-12, spend 1e-12 |
| Doubled quadrature rule | 1e-10 |
| Scores vs finite differences; Beta rule vs exact Beta moments (shapes up to 503) | 1e-6; 1e-11 |
| **Parameter recovery** (direct simulation, 15,000 / 8,000 customers) | every log parameter within 3 SE |
| **Per-cell recovery on the purchases generator** (C = 30) | joint Wald chi-square below the 0.999 quantile; every \|z\| <= 3.5 (record: max 2.7) |
| Calibration against realised value (C = 24, 30) | mean residual and slope within 4 SE |
| **No aggregate bias against the oracle** (correct specification; 6 draws in the test, 10 in the record) | z = 0.32 (record) |
| Data | `x, t_x, T` equal the registry features; the acquisition purchase always observed |
| Leakage | deleting or scrambling rows recorded after `C` leaves predictions unchanged; validation ignores rows recorded after `L_v` |
| Fallback, status, reproducibility, no truth access | pooled fallback for a sparse cell; `undefined` without events; bit-identical repeat, row-order invariant within the convergence tolerance (1e-6); AST test |

## Results (record; development seeds)

**Fit status.** Base and correlated-spend: 12/12 cells `ok` for both components; Pareto variant:
one BG/NBD cell `fallback` at C = 24. Fit time 0.5 s for 12 cells + pooled.

**Holdout comparison with the oracle** (excess = loss above the oracle, as % of the oracle MSE;
oracle mean value about 19 per customer):

| Scenario | E4 per cell, C = 24 / 30 | Pooled reference | Holdout CITL (E4) |
|---|---|---|---|
| Base (correctly specified per cell) | 1.5% / 0.9% (RMSE 5.3 / 4.3) | 0.8% / 0.7% | -3.7% / -5.1% |
| Pareto/NBD dropout (S4) | 2.1% / 1.8%; P(alive) +0.25 | 1.5% / 1.5% | +7.9% / +6.5% |
| Correlated spend (S4) | 2.5% / 1.5% | 1.4% / 1.3% | -6.5% / -8.1% |

**Repeated draws** (base, C = 24, 10 draws): mean(E4 - oracle) +0.03 (SD 0.31, z = 0.32).

**Sample size** (RMSE vs oracle, 5 draws per scale):

| Acquisition scale (mean n) | E4 per cell | Pooled reference |
|---|---|---|
| 0.25 (1,763) | 12.5 | 5.1 |
| 1 (7,130) | 4.6 | 3.7 |
| 4 (28,555) | 2.2 | 3.6 |

## Scientific findings

1. **E4 is correct where it is correctly specified**: exact agreement with the oracle at the
   true parameters, recovery within 3 SE, no aggregate bias across draws.
2. **At the protocol's scale, per-cell estimation noise outweighs the pooling bias.** The pooled
   misspecified reference has the lower excess at 1x (3.7 vs 4.6) and far lower at 0.25x; per
   cell wins only at 4x (2.2 vs 3.6). With about 600 customers per cell the 7-parameter cell fits
   are noisy, which also shows as over-dispersed predictions (validation slopes below 1). A2's
   per-cell unit is correct in expectation but not the lowest-error choice at this size.
3. **The misspecifications are visible.** Pareto/NBD dropout: BG/NBD over-states P(alive) by
   0.25 and value by 7-8%. Correlated spend: value under-stated by 6-8%.
4. **The frozen validation gates have little power for this outcome.** With heavy-tailed spend and
   2,700-4,700 customers at the validation landmark, the slope's HC SE is 0.07-0.37 against a
   gate of [0.85, 1.15]. The exact oracle itself fails the full gate (slope and CITL) in 4 of 12
   and 8 of 12 evaluations in two sets of development draws (see *Oracle gate failure rates*).
   In the record's main run E4 fails the gate in all six scenario x cutoff settings, base
   included.

## Owner disposition (P5 review)

1. **Non-contractual validation gate unchanged.** Where no candidate qualifies, the result is
   reported as `NO QUALIFIED MODEL`; contractual E2 is **not** substituted.
   `model_selection.select_model` (frozen, code-digest covered) always attaches
   `REFERENCE_FALLBACK_MODEL = "E2"` with `REFERENCE_FALLBACK_LABEL` to a no-qualified result;
   for the non-contractual scenario that reference entry is not applicable and must not be
   reported or used (P7 reporting rule).
2. **A2 per-cell fitting preserved.** The pooled E4 is recorded only as a descriptive
   bias-variance comparison (finding 2); it is neither a candidate nor a fallback beyond A2's
   component-wise pooled fallback.
3. **B2 assigned to P7** (not implemented). The approved plan names B2 (RFM heuristics:
   historical average spend x expected purchases, last period carried forward) among the
   models (§4) and uses it in PC4, but its phase table assigns it nowhere. B1 (the M10
   baselines) is likewise only wired into the comparisons later: the plan's P7 runs evaluation
   and model selection, where the baselines are candidates (complexity rank 0), and P8 runs the
   main-seed comparisons. P7 is the earliest existing phase that needs B2 and precedes PC4 (P8).
   **Gap to close before implementing B2:** the protocol pins only B2's id and name
   (`ModelSpec("B2", "RFM heuristics", ...)`); its formula exists only in the approved plan's
   prose, so its exact definition must be fixed in the protocol (an amendment) before P7 uses it.

## Oracle gate failure rates and the selection limitation

The frozen gate (validation landmark; slope in [0.85, 1.15] and |CITL| <= 0.05), applied to the
exact P2 oracle (true parameters, information recorded by `L_v`) and to E4, base scenario,
cutoffs 24 and 30:

| Set | Seeds (development) | Evaluations | Oracle passes | E4 passes | Source |
|---|---|---|---|---|---|
| A (record) | 5441-5446 | 12 | 8 (fails 4) | 6 (fails 6) | record §5; `GATE_SEEDS` in `scripts/report_m11_p5_btyd.py` |
| B (diagnostic) | 5401-5406 | 12 | 4 (fails 8); slope alone 5 | 2 (fails 10); slope alone 4 | ad-hoc scratch diagnostic during P5, not in the repository; numbers from its output |

Across both sets the oracle fails 12 of 24 evaluations (50%) and E4 16 of 24 (67%). The
oracle's slope SE ranges 0.065-0.37. **Unresolved limitation:** with the frozen gate,
non-contractual selection will often return `NO QUALIFIED MODEL` even for a correctly specified
model, and no non-contractual reference model exists; PC4 itself (a paired MSE comparison) is
unaffected, but selection-based reporting for this scenario will be uninformative.

**Seed provenance.** Both sets are **development draws** (seeds <= `DEVELOPMENT_SEED_MAX` =
9,999), not the protocol's reserved fresh seed (`FRESH_SEED` = 20261103), which was never
requested or used; `authorise_seed` still refuses it. Earlier wording ("fresh draws") meant
additional independent development draws. Seed 5401 in set B is also the record's main
development seed (`SEED`); set A's seeds are used nowhere else. The earlier summary's "5 of 12"
and "4 of 12" for set B were slope-only counts; the full-gate counts are those above.

## Verification of the specification choices against P2

| Choice | P2 / generator definition | Verdict |
|---|---|---|
| 1. Gamma-Gamma uses every observed spend, the acquisition purchase included | Generator: the acquisition purchase is a purchase at `t = 0` whose spend is drawn from the same `Gamma(p, nu)` as repeat purchases (`spend = gamma(gg_p, 1/nu[row])` over all rows). P2 oracle: "the spends are every observed purchase including the acquisition one"; `n_spend` counts all purchases. (The registry feature `mean_spend` is repeat-only, but it is a regression feature, not the oracle's information.) | **Consistent** with P2's information set |
| Repeat counts `x`, `t_x`, `T` | P2: repeat purchases are those at `t > 0` in completely recorded months, weeks since the acquisition purchase; E4 uses the same rule and equals the registry `frequency`, `recency`, `age` exactly (tested). A repeat purchase within 0.5 s of acquisition would round to `t = 0` in the stored timestamp; the same convention applies to P2. | **Consistent** |
| 2. Contribution conversion | Generator: contribution = revenue - cogs - fee = spend x (0.40 - 0.03), each column rounded to cents; P2's oracle uses the unrounded `spend x 0.37`; labels use the rounded contribution. E4 uses the recorded aggregate ratio (0.37000, within about 1e-5 of 0.37) times expected spend from recorded (rounded) revenue. | **Consistent**; differences are cent rounding only |
| 3. Component-wise pooled fallback | `E4_FIT_UNIT`: "cell fit status follows STATUS_RULES, with the pooled fit as the only allowed fallback" — granularity not specified by P2 or the protocol | **Not P2-defined**; compatible with A2, a P5 convention |
| 4. Events for the status rules | `STATUS_RULES.min_events` is M10's churn-event count; dropout is never observed in the non-contractual scenario, and neither P2 nor the protocol defines events for E4 | **Not P2-defined**; a P5 convention (repeat purchases; observed spends) |
| 5. 64 Gauss-Jacobi nodes | `ORACLE_NUMERICS.bgnbd_jacobi_nodes = 64` for the identical integral in the oracle; E4 defines its own constant (no oracle import) | **Consistent**; doubled-rule test passes |

## Limitations

1. **Distributional outputs**: E4 gives P(alive), monthly expected purchases and expected spend;
   the full predictive distribution of window counts and value (log score, CRPS, PIT) and
   parameter uncertainty are P7's.
2. **Heavy-tailed realised value**: holdout slopes have SEs 0.08-0.32; holdout CITL is within
   ±5% only in the base scenario at C = 24.
3. **Complete months only** (as the features and the oracle): purchases recorded in months
   `C - 1` and `C` are not used.
4. **Model-based covariances** from the observed information; no customer-clustered inference
   (P7).
5. **Runtime** (preflight input): data 0.01 s, fit 0.5 s (12 cells + pooled), prediction 0.1 s,
   validation 0.4 s per cutoff at full scale.
