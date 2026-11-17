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

## Integrity

| Item | Value after the amendments |
|---|---|
| Protocol manifest digest | `85eb0b4cc7d15c705dce2e23cb4ac97aa0631fd95efe8b7b402da467497379c1` |
| Protocol code digest (protocol, selection, precision) | `eefebd8067285e70a07ad5ad0bf270cdf3434c9968e5aa32967d56e49d8bed9b` |
| Pending before main | 2 (repeated-sampling preflight; P2 oracle validation sizes) |

Both digests are pinned in `tests/unit/customer_value/test_predictive_protocol.py`.

## Unchanged

Seeds and seed blocks, cutoffs, landmarks, horizons, information tiers and features, targets,
primary comparisons, selection, calibration and status rules, penalty conventions, clustered
inference, the oracle gate, coverage reporting, runtime budgets, and the M10 economics. The
committed P0 plan record ([m11-p0-precision-plan.md](m11-p0-precision-plan.md)) is kept as
generated; its header shows the P0 stage and digests that were current at the time.
