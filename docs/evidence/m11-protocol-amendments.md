# Milestone 11 — protocol amendments before main-seed evaluation

The M11 protocol (`src/decision_intelligence/customer_value/predictive_protocol.py`) was
committed at P0 (`82e3ebd`). Every later change is listed here and in the protocol's own
`PROTOCOL_AMENDMENTS`. **All were made before any main-seed evaluation**: the main and fresh
seeds have never been authorised (`authorise_seed` refuses them while `PENDING_BEFORE_MAIN` is
non-empty), and no M11 model exists yet.

| Id | Stage | Change | Reason |
|---|---|---|---|
| A1 | P1 | Pinned the generator parameters of `signals_contractual`, `purchases_noncontractual` and the three predeclared variants as seed-independent SHA-256 digests (`SCENARIO_PARAMETER_DIGESTS`); scenario status `pending P1` → `fixed at P1` | P1 generators implemented (`data/predictive_ltv_scenarios.py`); a test recomputes every digest from the generator defaults |
| A2 | P1 | E4 (BG/NBD + Gamma–Gamma) is fitted **per segment × channel cell** (`E4_FIT_UNIT`); a pooled fit is reported only as a labelled misspecified reference, not a selection candidate (`E4_POOLED_REFERENCE`) | Purchase rate and spend vary by segment and dropout by channel, so the generator is BG/NBD + Gamma–Gamma only within a cell; pooled, frequency and spend are correlated (development seed: within-population Spearman −0.056) |
| A3 | P1 | Behavioural-signal parameters kept unchanged (`SIGNAL_PARAMETERS_DECISION`) | Owner decision after the P1 pilot showed a weak crude Tier B signal; P2's exact oracles quantify the available Tier B gain; the data are not tuned toward a PC3 result |
| A4 | P1 | Closed the pending items for generator parameters and the new-scenario pilot; kept the repeated-sampling preflight (relabelled "P2–P6, deferred from P1") and the P2 oracle item open; `PILOT_FINDINGS_SCOPE` now points to the P1 scenario plan; `PROTOCOL_STAGE` `P0` → `P1-amended` | P1 pilot completed ([m11-p1-scenario-plan.md](m11-p1-scenario-plan.md)); the preflight needs oracle and model-fit costs that exist only after P2–P6 |
| A5 | P2 | Pinned the non-contractual oracle numerics (`ORACLE_NUMERICS`): Gauss–Jacobi 64 nodes (BG/NBD) and 64 (Pareto/NBD death-time integral), correlated-spend tensor rule (64, 48, 32) for (λ, p, ν); every oracle rechecked with all node counts doubled (relative change ≤ 1e-10 for BG/NBD and Pareto/NBD; ≤ 1e-5 for copula values and ≤ 1e-8 for copula spend); the copula oracle checked against 1,000,000 independent likelihood-weighted Monte Carlo draws from the generator's prior (within 4.5 MC SE). Closed pending item "P2: oracle Monte Carlo validation sizes and quadrature settings"; `PROTOCOL_STAGE` `P1-amended` → `P2-amended`. `purchase_oracle` reads its node counts from the protocol | Owner approval of the settings measured in the P2 assurance review ([m11-p2-oracle-findings.md](m11-p2-oracle-findings.md), [m11-p2-oracle-plan.md](m11-p2-oracle-plan.md)): doubled-rule changes on the pilot about 1e-15 (BG/NBD), 4e-15 (Pareto/NBD), 3.3e-8 (copula) |
| A6 | P3 | Narrow E2 `v = 0` boundary exception (`E2_GEOMETRIC_BOUNDARY`): an E2 fit (cell or pooled) that is `unstable` **only** because `log v` fell below the bound (optionally with the consequent non-positive-definite Hessian), converged, with `b` inside the bound, is replaced by its geometric limit (`v = 0`) and counted as `ok` — provided the closed-form geometric `b` agrees with the E2 `b` (1e-4, log scale), the geometric log-likelihood is not lower (relative 1e-9), the one-sided score for `v` at `v = 0` is not positive (≤ 1e-6 per customer), and the one-parameter fit is `ok` under `STATUS_RULES`. Labelled and counted. All other status and fallback rules unchanged (E1, upper bounds, other reasons, failed checks). `PROTOCOL_STAGE` `P2-amended` → `P3-amended` | Owner approval after P3 finding 2 ([m11-p3-findings.md](m11-p3-findings.md)): under homogeneous churn the correct E2 MLE is `v = 0`, on the boundary; the frozen rule substituted the pooled fit, which was miscalibrated (holdout slope 1.16–1.19). Before/after on 10 development draws: [m11-p3-contractual-models.md](m11-p3-contractual-models.md) §5 |
| A7 | P3 | Narrow E1 geometric boundary exception (`E1_GEOMETRIC_BOUNDARY`), separate from A6. E1's geometric limit is `alpha, beta → ∞` with `mu = alpha / (alpha + beta)` fixed (the closed boundary `phi = 1 / (alpha + beta) = 0`; `S(a) → (1 − mu)^a`). An E1 fit (cell or pooled) is replaced by this limit and counted `ok` only if: it is `unstable` with the bound reason (optionally the Hessian reason) **or** `failed` with the single reason *not converged* (the iterates drift along the flat ridge to `phi = 0`); finite likelihood, no separation, start agreement within the frozen 1e-4; on the ridge (`log(alpha + beta) > 10`, no lower bound); closed-form `mu = events / decisions` within 1e-4 (log) of the fit's `mu`; geometric log-likelihood not lower than the fit's, evaluated with the exact finite-product form (relative 1e-9); one-sided `phi` score at `phi = 0` not positive (≤ 1e-6 per customer); the one-parameter fit `ok`. A6 and all other rules unchanged; `PROTOCOL_STAGE` stays `P3-amended` | Owner approval after P3 limitation 7 ([m11-p3-findings.md](m11-p3-findings.md)): under homogeneous churn E1 cells still fell back to the miscalibrated pooled fit (holdout slope 1.16–1.17). Before/after on 10 development draws: [m11-p3-contractual-models.md](m11-p3-contractual-models.md) §6 |
| A8 | P4 | **E3 covariates**: E3 no longer uses `acquisition_cohort`, `tenure_months`, `paid_months` as linear covariates (`E3_EXCLUDED_COVARIATES`); tenure enters E3 only through its unpenalised tenure-band basis on the decision age. The feature registry, S1 and every other model are unchanged. **Clarifications** (owner decisions on points the protocol left open): `E3_TUNING_LOSS` — penalty chosen by squared error of window (12-month) value at the validation landmark, ties to the larger penalty; `E3_SELECTION_ELIGIBILITY` — with no tuning landmark completed before the validation landmark there is no validation fit and E3 is *not selection-eligible* there (no censored validation windows); `DESIGN_DUPLICATE_RULE` — exact duplicate design columns after standardisation are removed deterministically (earlier column in registry order kept) without altering the registry. `PROTOCOL_STAGE` `P3-amended` → `P4-amended` | P4 diagnosis ([m11-p4-findings.md](m11-p4-findings.md)): frozen at the landmark and entered linearly, these covariates were extrapolated far beyond the training landmarks' range (W = 24 at C = 33: training cohorts 0–5, prediction 0–31). **Original (pre-A8) results:** W = 24 mean(pred − oracle) −18.3 to −24.9, holdout CITL −21% to −34%; M10 v = 1, C = 24: −4.2; five validation-gate failures. **After A8** (record §4): W = 24 bias ≈ 0 (CITL ≤ +1.2%); all eligible validation gates pass (5 → 0 failures). **Trade-off:** one flagged regression — M10 v = 1, C = 30, RMSE vs oracle 1.32 → 2.06 (holdout CITL −2.1% → −2.9%, within tolerance; E3 not selection-eligible there) — and M10 v = 0, C = 30, 3.01 → 3.14 (below the 5% flag); cause: within-band hazard decline under heterogeneity that the coarse frozen tenure bands cannot follow |
| A9 | P6 | **E5 tuning and eligibility** (prospective; no change to E5's design, S2, `SELECTION` gates or any tolerance). `E5_TUNING_LOSS`: one penalty shared by both parts, chosen under `TUNING_RULE` by squared error of the window (12-month) value `P(R > 0) x E[R \| R > 0]` at the validation landmark; ties to the larger penalty; if no tuning fit has both parts `ok`, `DEFAULT_PENALTY`, labelled 'fixed penalty: no tuning fit was ok'. `E5_SELECTION_ELIGIBILITY`: E5 is a selection candidate only if a validation fit exists (at least one tuning landmark with labels complete before the validation landmark); otherwise 'not selection-eligible'; no censored validation windows. `PROTOCOL_STAGE` `P4-amended` → `P6-amended` | P6 final review ([m11-p6-findings.md](m11-p6-findings.md)): `TUNING_RULE` names no loss and A8's clarifications are scoped to E3, so E5's penalty choice, tie-break and eligibility were not independently authorised. The P6 implementation already followed this wording, so no P6 number changes |

## Integrity

| Item | After A1–A4 (P1) | After A5 (P2) | After A6 (P3) | After A7 (P3) | After A8 (P4) | After A9 (P6, current) |
|---|---|---|---|---|---|---|
| Protocol stage | `P1-amended` | `P2-amended` | `P3-amended` | `P3-amended` | `P4-amended` | `P6-amended` |
| Protocol manifest digest | `85eb0b4cc7d15c705dce2e23cb4ac97aa0631fd95efe8b7b402da467497379c1` | `b81cdbbc0f29c22cadc6f7bc9383e3d4ee8165ed71d3d64f19f61f41b6ef2175` | `aef1b81a62d7189c90675be87d0bff42ab4da04b793c1788f4c80c97b9064ec9` | `33329a8483c6c1e8fc586dc94d99fa9d203edef590bf1a158c9a8239330ba5cc` | `a86df2f157893f0cf8250b4b30c8f10509a947498f238be770b5ba9f65eea5ba` | `6977394140d65b7058d35385213f6a7f110f5acb7f65875c48f25c0b4d2b28a3` |
| Protocol code digest (protocol, selection, precision) | `eefebd8067285e70a07ad5ad0bf270cdf3434c9968e5aa32967d56e49d8bed9b` | `f5ae4a9f645d6f88f0fd2716842a2f5075e6f5bc9718efc1d5d071b0379fe0db` | `e68d4457a1e078faa213fa93ba1cc97692946024486ad07f43499f75f7bb963f` | `2e69fb2d7647e1fc18f53124945edbd76e3223052d1a1849bae188b9539c23b5` | `b9675dcfa62a64b7ac2e3cb5e234c5f8845fdcb3adc03ec0539a70101e03b2c6` | `a1af7b4717f3e57749a8b160da33e45333beee8bc7e53e7b66f7ea957365a9ed` |
| Pending before main | 2 (repeated-sampling preflight; P2 oracle validation sizes) | 1 (repeated-sampling preflight) | 1 | 1 | 1 | 1 |

The current digests are pinned in `tests/unit/customer_value/test_predictive_protocol.py`. The
main and fresh seeds remain refused while the preflight item is pending. A6–A9 changed only
`predictive_protocol.py` among the code-digest files (`model_selection.classify_fit` and
`STATUS_RULES` are unchanged; the exceptions are applied in `customer_value/contractual.py`, the
E3 rules in `customer_value/survival.py`, the E5 rules in `customer_value/predictive.py`).

## Open items for review before P8 (not amendments)

1. **E3's no-ok-fit penalty fallback is unauthorised.** `survival.py` (committed in P4) falls back
   to `DEFAULT_PENALTY`, labelled 'fixed penalty: no tuning fit was ok', when no tuning fit is
   `ok`. `TUNING_RULE` and `E3_TUNING_LOSS` do not provide this fallback (A9's equivalent clause
   is E5's only). It never triggered in any development record. Options: extend
   `E3_TUNING_LOSS` by amendment, or accept it as a documented implementation convention.
   Unresolved; no code changed.
2. **B2 definition** (P5 disposition): the protocol pins only B2's id and name; its formula must be
   fixed by amendment before P7 implements it.
3. **Reference fallback for non-contractual selection** (P5 disposition): `select_model` attaches
   E2 to any no-qualified result; P7 reporting must not show it for the non-contractual scenario.

## Unchanged

Seeds and seed blocks, cutoffs, landmarks, horizons, information tiers and features, targets,
primary comparisons, selection and calibration rules, status rules (apart from the narrow A6
and A7 exceptions), the feature registry (A8 changes only which registry features E3 uses), the selection gates (A9 adds no gate), penalty conventions, clustered
inference, the oracle gate, coverage reporting, runtime budgets, and the M10 economics. The
committed P0 plan record ([m11-p0-precision-plan.md](m11-p0-precision-plan.md)) is kept as
generated; its header shows the P0 stage and digests that were current at the time. Likewise
the P1 scenario plan ([m11-p1-scenario-plan.md](m11-p1-scenario-plan.md)) keeps its P1
header (`P1-amended`, `85eb0b4c…` / `eefebd80…`), and the committed P2 records
([m11-p2-oracle-plan.md](m11-p2-oracle-plan.md), [m11-p2-oracle-findings.md](m11-p2-oracle-findings.md))
keep their P2 header (`P2-amended`, `b81cdbbc…` / `f5ae4a9f…`), and the committed P3 records keep
the digests current when they were generated. A5–A9 changed no other protocol value: no seed,
seed block, cutoff, horizon, feature registry entry, model, target, comparison, selection rule,
status threshold, tolerance or budget.
