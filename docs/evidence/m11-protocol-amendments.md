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

## Integrity

| Item | After A1–A4 (P1, historical) | After A5 (P2, current) |
|---|---|---|
| Protocol stage | `P1-amended` | `P2-amended` |
| Protocol manifest digest | `85eb0b4cc7d15c705dce2e23cb4ac97aa0631fd95efe8b7b402da467497379c1` | `b81cdbbc0f29c22cadc6f7bc9383e3d4ee8165ed71d3d64f19f61f41b6ef2175` |
| Protocol code digest (protocol, selection, precision) | `eefebd8067285e70a07ad5ad0bf270cdf3434c9968e5aa32967d56e49d8bed9b` | `f5ae4a9f645d6f88f0fd2716842a2f5075e6f5bc9718efc1d5d071b0379fe0db` |
| Pending before main | 2 (repeated-sampling preflight; P2 oracle validation sizes) | 1 (repeated-sampling preflight) |

The current digests are pinned in `tests/unit/customer_value/test_predictive_protocol.py`. The
main and fresh seeds remain refused while the preflight item is pending.

## Unchanged

Seeds and seed blocks, cutoffs, landmarks, horizons, information tiers and features, targets,
primary comparisons, selection, calibration and status rules, penalty conventions, clustered
inference, the oracle gate, coverage reporting, runtime budgets, and the M10 economics. The
committed P0 plan record ([m11-p0-precision-plan.md](m11-p0-precision-plan.md)) is kept as
generated; its header shows the P0 stage and digests that were current at the time. Likewise
the P1 scenario plan ([m11-p1-scenario-plan.md](m11-p1-scenario-plan.md)) keeps its P1
header (`P1-amended`, `85eb0b4c…` / `eefebd80…`). A5 changed no other protocol value: no seed,
seed block, cutoff, horizon, feature, model, target, comparison, rule or budget.
