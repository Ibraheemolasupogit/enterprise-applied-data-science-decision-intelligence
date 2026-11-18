# Milestone 11 P2 — oracles, point-in-time features and landmarks: findings

Status: **closed at P2 (amendment A5), awaiting owner review**. Numbers: [m11-p2-oracle-plan.md](m11-p2-oracle-plan.md)
(`python scripts/plan_m11_precision.py --stage p2`; pilot and development seeds only). The only
protocol change is amendment **A5** (owner-approved at P2 closure): the non-contractual oracle
numerics are pinned in `ORACLE_NUMERICS` and the P2 pending item is closed; stage
`P2-amended`, manifest digest `b81cdbbc…2175`, code digest `f5ae4a9f…e0db`
([m11-protocol-amendments.md](m11-protocol-amendments.md)). Generator parameters are the
pinned ones; the main and fresh seeds were never authorised. No predictive model exists yet.

## What P2 adds

| Module | Content |
|---|---|
| `customer_value/oracle.py` | Exact observable-information oracles (evaluation references, true parameters): a generic finite class x state forward filter for `signals_contractual` (Tier A: survival only; Tier B: + signal emissions), and the closed-form Gamma-posterior Tier A oracle for `m10_contractual` |
| `customer_value/purchase_oracle.py` | Observable-information oracle for `purchases_noncontractual` and each predeclared variant, chosen from the scenario's own process: BG/NBD + Gamma-Gamma (exact in `lambda`; Gauss-Jacobi in `p`), Pareto/NBD dropout (closed-form window; Gauss-Jacobi death-time integral), correlated spend (copula-reweighted posterior, tensor Gauss quadrature). Complete-month information at `C` (the features' months) |
| `customer_value/features.py` | The protocol feature registry, computed from data recorded by the cutoff (transactions and cancellations through M10's `observe`, signals through `as_of` on `recorded_at`, months `<= C - 1`) |
| `customer_value/landmarks.py` | Training stacks computed only from the tables as recorded at the evaluation cutoff; holdout frames (evaluation only, M10 complete-window rule); customer-level resampling for repeated landmark rows |

## Oracle correctness (tests)

| Check | Result |
|---|---|
| Filter vs brute-force enumeration of every class and state path (random parameters, ages 0-4, with and without non-proportional hazards) | equal to 1e-10 (posterior and survival) |
| Forward survival vs the P1 truth recursion (itself checked against Monte Carlo) | equal to 1e-12 |
| Filter with 80 Gauss-Laguerre classes vs M10's closed-form Gamma oracle | equal to 1e-8 |
| M10 oracle vs M10's population survival ratio | equal to 1e-12 |
| Tier A = Tier B with zero emission log-likelihood | equal to 1e-12 |
| Calibration against realised value (development seeds; PH and NPH; M10) | mean residual and OLS slope within 4 SE of 0 and 1 |
| Posterior class and state probabilities vs the latent truth | within 4.5 SE |
| Oracle gate (amendment 1): `mean(D_i - (mu_B - mu_A)_i^2)` and `mean(mu_B - mu_A)` | consistent with 0 (no per-seed sign of D asserted) |

Pilot calibration slopes: 0.987-1.007 (signals, both tiers, three cutoffs, NPH variant) and
1.001 / 1.021 (M10, cutoffs 24 / 30).

## Non-contractual oracle (added in the assurance review)

Coverage: every scenario of the protocol now has an exact (closed-form or converged-quadrature)
observable-information oracle.

| Scenario | Oracle | Status |
|---|---|---|
| `m10_contractual` | closed-form Gamma posterior (Tier A) | P2 |
| `signals_contractual` (+ NPH variant) | exact forward filter (Tier A, Tier B) | P2 |
| `purchases_noncontractual` (BG/NBD + Gamma-Gamma) | `lambda` analytic, `p` Gauss-Jacobi on the exact Beta weight; conjugate spend | assurance review |
| `pareto_nbd_dropout` variant | window closed form; death-time integral as a Beta expectation (Gauss-Jacobi) | assurance review |
| `correlated_spend` variant | independent posterior reweighted by the Gaussian-copula density; Gauss-Legendre in `log lambda`, `log nu`, Gauss-Jacobi in `p` | assurance review |

The published BG/NBD hypergeometric closed form needs `a > 1`; the pinned `a = 0.8`, hence the
1-D quadrature. Information set: segment, channel, acquisition date and every purchase (time
and spend, acquisition purchase included) in completely recorded months `<= C - 2`; the
partially recorded months `C - 1`, `C` are excluded exactly as the features exclude them.

| Independent check (tests) | Result |
|---|---|
| BG/NBD vs Fader-Hardie-Lee (2005) closed form and P(alive), `a = 2.4` | equal to 1e-12 |
| BG/NBD (`a = 0.8`) and Pareto/NBD vs brute-force 2-D integration of the raw likelihood | equal to 1e-9 (quadrature tolerance of the reference) |
| Gamma-Gamma posterior spend vs 1-D integration and the published weighted-average form | 1e-8 / 1e-13 |
| Copula oracle at zero correlation vs the independent oracle | 1e-7 (value), 1e-12 (P(alive)) |
| Copula oracle (rho = -0.5) vs likelihood-weighted Monte Carlo from the generator's prior (10^6 independent draws; protocol check, A5) | within 4.5 MC SE (value, P(alive), spend) |
| Quadrature: default vs doubled rules (protocol check, A5) | BG/NBD, Pareto 1e-10 (measured ~1e-14); copula 1e-5 value, 1e-8 spend (pilot: 3.3e-8) |
| Gauss rules vs exact Beta and Gamma moments | 1e-12 (Beta), 1e-8 (log-Gamma) |
| History vs the features (`frequency`, `recency`, `age`); deleting or scrambling post-cutoff rows | identical; unchanged |
| Each variant vs **its own** latent truth (`E[latent | F_C] = oracle`): mean, slope, P(alive) reliability, spend | within 4 SE / 4.5 SE |
| Power: the BG/NBD oracle on Pareto/NBD data | P(alive) z > 8 (detected) |
| Copula spend adjustment vs the true spend (slope 1 vs 0) | slope within 4 SE of 1, > 5 SE from 0 |

Pilot calibration against realised value (cutoffs 24 / 30, W = 12): mean residual z between
-0.6 and 0.7; calibration slopes 0.88-1.09, each within 1.3 HC SE of 1 (heavy-tailed spend).
Runtime per cutoff: 0.03-0.10 s (base, Pareto), about 3 s (copula).

## Leakage and repeated customers (tests)

- Deleting, or scrambling, every row recorded after the cutoff leaves features (both tiers),
  training stacks and oracles unchanged.
- Training labels from the cutoff view equal the fully observed labels (complete-month rule);
  landmarks outside `training_landmarks` and empty landmark sets are refused.
- The backdated account-closure summary: a record-time join finds nothing for customers active
  at a landmark; an event-time join finds rows, every one of which churns inside its label
  window, with mean realised value below half the stack mean — the leak the trap was planted to
  demonstrate. No feature module reads the table (AST-tested).
- Stack rows repeat customers across landmarks; `resample_customers` keeps each customer's
  rows together and the Kish effective size is at most the number of customers.

## Tier B advantage (population level)

`relative advantage = E[(mu_B - mu_A)^2] / MSE_B`, compared with the relative MDE `4k^2/N`.

| Source | Relative advantage | Relative MDE | Ratio |
|---|---|---|---|
| Pilot, C = 36 / 42 (W = 12) | 0.0348 / 0.0343 | 0.0121 / 0.0114 | 2.9 / 3.0 |
| Pilot, C = 33 (W = 24) | 0.0319 | 0.0126 | 2.5 |
| Pilot, NPH variant, C = 36 | 0.0310 | 0.0134 | 2.3 |
| 10 development draws, C = 36 / 42 | 0.0331 / 0.0329 (SD 0.0014; min 0.031) | 0.0123 / 0.0116 | 2.7 / 2.8 |

Pooled pilot cutoffs 36 + 42 (customer-cluster bootstrap): D = 21.9, 95% CI [14.2, 29.2].
**PC3 is adequately powered at the pinned parameters**; the P1 concern (a crude usage tercile
explained 1-2% of variance) understated the information the full signal history carries.
The behavioural-signal parameters were not changed.

## Limitations and open items

1. **Closed (A5).** The P2 pending item (oracle Monte Carlo validation sizes and quadrature
   settings) is closed with the owner-approved values: BG/NBD Gauss-Jacobi 64 nodes;
   Pareto/NBD Gauss-Jacobi 64; copula (64, 48, 32); doubled-rule convergence checks; a
   1,000,000-draw independent Monte Carlo check. `purchase_oracle` and its tests read these
   from `ORACLE_NUMERICS`. One item remains in `PENDING_BEFORE_MAIN` (below), so the main and
   fresh seeds stay refused.
2. **Repeated-sampling preflight** (pending item) still needs the P3-P6 model-fit costs. P2
   costs are small: both oracles and holdout labels for three pilot cutoffs take about 0.3 s.
3. **Oracle scope.** The contractual oracles condition on survival, segment, channel, tenure
   and (Tier B) signals; contribution history is correctly uninformative because cost noise is independent
   of the latent state. They do not use the planted trap, which no information tier contains.
4. **Customer resampling** lives in `landmarks.py`; under the repository's shared-utility rule
   it should move to `common.bootstrap` when P7 implements clustered intervals.
5. `usage_change_3m` is set to 0 when the previous three-month window is empty, and
   `months_since_payment_failure` is capped at 24 (also the value with no failure); P4/P6 must
   treat these as documented conventions.
6. The non-contractual oracle uses complete months only (`<= C - 2`), as the features do;
   purchases already recorded in months `C - 1` and `C` would add information but would require
   modelling the recording lag. Its realised-value checks are noisy (heavy-tailed spend), so
   correctness rests on the latent-truth and numerical checks, not on pilot slopes.
7. Pre-existing: an intermittent NumPy "invalid value encountered in multiply" warning in an M4
   test; the M11 tests run clean with deprecation warnings treated as errors.
