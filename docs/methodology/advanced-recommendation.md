# Advanced recommendation and ranking (Milestone 9)

Pillar 3's second milestone compares a small, justified set of advanced recommenders with the M8
baselines on the M8 chronological protocol, evaluates cold start and operational cost, and
connects offline evaluation to a **simulated** user-randomised experiment analysed with the
M6/M7 machinery. Code: `src/decision_intelligence/recommendations/` (M9 modules listed below) and
`src/decision_intelligence/data/recommendation_experiments.py`. Evidence:
[../evidence/m9-advanced-recommendation.md](../evidence/m9-advanced-recommendation.md).

## 1. Frozen protocol

Every evaluation choice — seeds, windows, K, metrics, hyperparameter grids, selection rules, the
A/B design (primary metric, guardrails, multiplicity correction, minimum worthwhile effect) and
the repeated-sampling sizes — is fixed in `recommendations/protocol.py`, written before any M9
model was evaluated. Model selection uses mean logged NDCG@10 on the validation window only. The
fresh seed (20260915) is evaluated once, by the final evidence run; tests never use it. Changes
made after the protocol was written are listed in the evidence (all prompted by development-seed
diagnostics, none by a main-seed validation or test result).

## 2. Methods and why they were chosen

| Method | Module | Justification |
| --- | --- | --- |
| Implicit ALS (Hu, Koren & Volinsky 2008) | `factorisation.py` | The standard model for implicit click counts: confidence-weighted least squares over all pairs, exact per-user solves |
| BPR (Rendle et al. 2009), uniform and exposure-aware negatives | `factorisation.py` | Optimises a pairwise ranking objective; exposure-aware negatives (shown, not clicked) use the logging data to sample negatives the user demonstrably saw |
| EASE (Steck 2019) | `factorisation.py` | Closed-form linear item model — a strong, cheap benchmark that guards against over-crediting complex models |
| Shallow two-tower retrieval | `two_tower.py`, `retrieval.py` | Feature towers score users without history and items without clicks; exact search and a k-means inverted-file index measure the recall–latency trade-off |
| Pointwise and LambdaRank-style learning-to-rank | `candidates.py`, `features.py`, `ranking.py` | Two-stage ranking over the union of generators' candidates with point-in-time features |
| Propensity-weighted ranking | `ranking.py` | Position-bias correction of click labels (Joachims, Swaminathan & Schnabel 2017) |
| Score-blend hybrid | `hybrid.py` | Rank-normalised convex combination of collaborative, content and popularity scores |
| Cold-start strategies | `cold_start.py` | History buckets; content → ALS-factor regression for new items |

**Deferred** (approved): LambdaMART / gradient-boosted ranking and deep neural ranking (new
dependencies, little expected gain on these data), session-based and context-aware models (the
generator has no within-session dynamics or context, so claims would be unsupported), and
long-term retention (the generator has no churn process).

## 3. Mathematics and implementation

- **ALS** minimises `Σ_{u,i} c_ui (p_ui − x_u·y_i)² + reg(|X|² + |Y|²)`, `p = 1{r > 0}`,
  `c = 1 + α r`. Each half-step is the exact ridge solution per row using
  `YᵀC_uY = YᵀY + Yᵀ(C_u − I)Y`; the objective never increases. Users without clicks get zero
  factors (served by the segment-popularity fallback).
- **BPR** maximises `Σ ln σ(x_uij) − (reg/2)|θ|²` by seeded mini-batch stochastic gradient ascent;
  each step equals `learning_rate ×` the analytic gradient (checked against finite differences).
- **EASE**: `B = I − P diag(1/diag P)`, `P = (XᵀX + λI)⁻¹`, `diag(B) = 0` — the solution of
  `min |X − XB|² + λ|B|²` subject to a zero diagonal (checked column by column).
- **Two-tower**: user features (segment, region, mean content of clicked items, log history
  length) and item features (content, log popularity) through linear or one-hidden-layer towers;
  full-softmax cross-entropy over launched items; Adam; hand-derived gradients checked by finite
  differences.
- **Two-stage ranking**: candidates = union of the top 50 allowed items from ALS, item-kNN, EASE,
  content-based and segment popularity; features at the candidate cutoff (generator scores, item
  popularity and trend, item age, user category affinity, price, margin, history length).
  Pointwise: L2 logistic regression (L-BFGS; equals the M3 maximum-likelihood fit at `L2 = 0`).
  LambdaRank-style: pairwise logistic loss weighted by the |ΔNDCG| of swapping the pair (checked
  against brute-force swaps), full-batch gradient descent; training NDCG is monitored because the
  weighted loss is not a fixed objective.
- **Propensity weights**: a click at position `p` is weighted `1/θ̂_p` (relative examination
  estimated from logs at or before the label window's start). Pointwise: positives only — a
  heuristic that does not make the logistic loss unbiased. LambdaRank: the pair weight is the
  clicked item's `1/θ̂_p`, then all pair weights are rescaled by one constant (scale only).
  Weighting corrects position bias among *shown* items; candidates never shown remain labelled 0.
- **Ranker windows**: generators and features at day 70, labels (70, 84], validated with
  generators at 84 on (84, 98]; the final ranker is retrained on labels (84, 98] with generators
  at 84 and applied with generators at 98. No ranker sees labels from the window it is scored on.

## 4. Off-policy estimators

All share the M8 estimand `V(π)` (expected clicks per logged test session under the fixed target
list, position-based model, units with `θ₁ = 1`).

| Estimator | Unbiased when | Notes |
| --- | --- | --- |
| IPS (M8) | (1) position-based model, (2) known propensities, independent slots, (3) positivity, (4) known relative examination | Estimates only the supported part of `V` when positivity fails |
| SNIPS | (1)–(4) give consistency, not unbiasedness | Ratio estimator with control variate `D_s = (1/L) Σ_j ρ_j / m_s`, `E[D_s] = 1`, `m_s` = number of showable list items; bias O(1/n) |
| DR | (1), (2), (4), for the supported part — for **any** click model | Direct method + IPS-weighted residual; relies on the click model alone for unsupported items; low variance only if the click model is good |

The DR click model `q(u,i) = expit(c₀ + c₁ z_ui + c₂ log(1 + pop_i))` is fitted **out of sample**
(ALS score at day 84, labels in (84, 98], applied with ALS refitted at day 98). Intervals: the
cluster-robust (user) linearised standard error with a normal interval; the percentile cluster
bootstrap; and the **studentised** cluster bootstrap (bootstrap-t), intended to correct the skew
that made M8's normal IPS interval under-cover. All treat the estimated examination as known (M8
showed its contribution to the variance is under 1%).

## 5. Simulated online evaluation (M6/M7 machinery)

`data/recommendation_experiments.py` simulates a 14-day period from day 120 in which each user
sees their arm's fixed top-10 list in every session (M8 click and purchase model). Sessions are
drawn independently of the arm, so every user's exact expected outcome under every arm is known
and the sample average treatment effect (SATE) is the target. `recommendations/online.py`
analyses one experiment with M6 hash assignment, the SRM check, Welch ITT analysis
(`analyse_continuous`), guardrail non-inferiority and the LAUNCH/ITERATE/HOLD/STOP rule, and M7
CUPED with pre-period clicks; two primary comparisons use Bonferroni (α/2 each, with matching
intervals) and Holm p-values are reported. Policies are fixed before outcomes are simulated by
the frozen rule. A/A experiments check the false-positive rate; re-randomisation with sessions
fixed checks interval coverage of the exact SATE. **All online results are SIMULATED**: they
confirm consistency with the generator's click model, not real user behaviour. Assumed absent:
interference between users, feedback loops, effects on visit frequency, novelty and long-term
effects.

## 6. Operational evaluation

Sparse seeded click matrices at 3k × 600, 10k × 2,000 and 30k × 5,000 (Zipf popularity, about 20
items per user). Each workload's largest dense allocations are estimated first; workloads above a
1.5 GB budget are **skipped** and reported, never attempted (the dense-input M8 models at the
largest tier). Reported: fit time, peak traced memory, model-state size, single-request latency
(p50, p95), batch throughput, and exact vs approximate retrieval (recall of the exact top-10 and
latency). Timings depend on the machine and its load.

## 7. Leakage and ground-truth controls

- Every model, feature, candidate set, exposure-aware sampler and the two-stage pipeline is
  checked for exact invariance to rewriting all data after its cutoff.
- An AST test fails if any training, tuning or selection module references the ground-truth type.
- Ground truth is used only by the evaluation layer (oracle metrics, simulated online value and
  the "true click model" DR diagnostic).

## 8. Limitations

- Synthetic data from a latent-factor generator, which favours latent-factor models; the
  generator's click model is also the simulated online environment, so simulated A/B results
  cannot reveal model–environment mismatch.
- Shallow NumPy towers are not production two-tower systems; the k-means index is a simple
  heuristic for inner-product search.
- Propensity weighting corrects only position bias; exposure (selection) bias in ranker labels
  remains. The pointwise IPW variant is a heuristic.
- Off-policy intervals ignore examination-estimation error; their empirical coverage is reported,
  not assumed — and it is below 95% for every estimator, even with the studentised bootstrap.
- The frozen A/B design was fixed without a power calculation and has about 7% power at its
  minimum worthwhile effect (post-hoc audit); its HOLD decisions are uninformative. A future
  design should be sized with the M6 planning tools before it is frozen.
- In the simulated A/B test the hybrid's ranker is the final ranker (labels 84–98) applied with
  generators refitted at day 120; the protocol did not specify retraining for the A/B period.
- The main seed's test window was inspected in M8; the fresh seed is the clean holdout.
- Timings are machine-dependent; scale tiers use synthetic sparse data without preference truth.
- Deferred: LambdaMART, deep neural ranking, session-based and context-aware models, long-term
  retention, interference and feedback-loop simulation.
