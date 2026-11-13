# Milestone 9 evidence — advanced recommendation and ranking

Status: **IMPLEMENTED — awaiting scientific assurance review**. Not committed or pushed.
Statuses: [capability-status.md](capability-status.md). Methodology:
[../methodology/advanced-recommendation.md](../methodology/advanced-recommendation.md).
Generated numbers: [m9-advanced-recommendation-results.md](m9-advanced-recommendation-results.md)
(`python scripts/generate_m9_evidence.py`; reproducible apart from §11 timings and the runtime
table). All online results are **SIMULATED**.

## Acceptance criteria

| Requirement | Evidence |
| --- | --- |
| Implicit ALS, BPR, EASE | `factorisation.py`; results §2, §4 |
| Shallow two-tower retrieval, exact and approximate search | `two_tower.py`, `retrieval.py`; results §4, §7, §11 |
| Pointwise and LambdaRank-style learning-to-rank; propensity weighting; hybrid | `candidates.py`, `features.py`, `ranking.py`, `hybrid.py`; results §2–§6 |
| Candidate recall reported separately from ranking quality | results §2 (validation), §3 (test) |
| Cold-start strategies for users and items | `cold_start.py`; results §7 |
| SNIPS, DR, studentised cluster bootstrap; assumptions defined; coverage investigated | `off_policy_advanced.py`; methodology §4; results §5, §8 |
| Operational benchmarks with preflight memory checks | `operational.py`; results §11 |
| SIMULATED user-randomised A/B with M6/M7 components; pre-specified primary metric, guardrails, multiplicity; policy selection separate from outcomes | `recommendation_experiments.py`, `online.py`, `protocol.AB_DESIGN`; results §9 |
| Frozen protocol; fresh seed untouched until the end | `protocol.py`; results §1 (disclosed changes), §12 (fresh seed, run once) |
| Leakage and ground-truth separation | perturbation tests for every model, feature, sampler and the two-stage pipeline; AST test over M9 training/selection modules |
| Reproducibility; NumPy 2.4.6 and 2.5.x; strict mypy; Python 3.11/3.12 | see Validation |

## Headline results (main seed, test window, K = 10)

| Finding | Number |
| --- | --- |
| Selected on validation | best M9 single-stage EASE (λ = 50); best ranker pointwise IPW (L2 0.1); blend (ALS 0.25, item-kNN 0.75); hybrid = two-stage pointwise IPW |
| Logged NDCG@10 vs item-kNN (0.247) | EASE +0.011 [0.006, 0.016] (Holm p 4e-4); rankers +0.007 to +0.008 (Holm p 0.03–0.09); blend +0.003 (n.s.); ALS −0.023, BPR −0.033, exposure-aware BPR −0.094, two-tower −0.067 |
| Oracle NDCG@10 vs item-kNN (0.593) | blend +0.014 (best, 0.607); rankers +0.006 to +0.008; EASE +0.007; ALS −0.012; BPR −0.053; two-tower −0.078 |
| SIMULATED online clicks/session | blend 1.257 (best) vs item-kNN 1.227; EASE 1.236; rankers 1.233–1.241 |
| Candidate recall (test) | union of 5 generators × 50: 82.4% of logged test clicks, 80.8% of oracle top-10 |
| Naive vs propensity-weighted rankers | differences of ≤ 0.001 logged NDCG and ≤ 0.003 oracle NDCG — no material effect |
| Cold-start users (0 clicks, n = 330) | fallback (segment popularity) 0.594; two-tower user tower 0.514 — the feature tower is **worse** than the fallback |
| New items (n = 21) | content → ALS factors 0.623, content-based 0.607, two-tower item tower 0.511, random 0.344, popularity / ALS 0.279 (constant scores) |
| Seeds 1–5 (fixed configurations) | blend beats item-kNN on oracle NDCG in 5/5; EASE 0/5, ALS 0/5, BPR 0/5 — the main-seed EASE advantage does not replicate |
| Logging-regime grid (9) | τ(logged, SIMULATED online) 0.36–0.86; best logged always EASE, best online always blend |
| Off-policy views (main seed) | τ with online: IPS 0.75, SNIPS 0.75, DR 0.74, logged 0.82, oracle 0.94; mean absolute error IPS 0.040, SNIPS 0.038, DR 0.032 |
| OPE calibration (1,000 replicates) | normal-interval coverage 0.897–0.940 (significantly below 0.95 in 7/8 cells); studentised bootstrap 0.927–0.938 (higher in 6/8 cells, still significantly below 0.95 in 6/8); bias within 2 MC SE everywhere, including DR with a constant click model |
| SIMULATED A/B (3,000 users; SRM p 0.08) | EASE vs item-kNN +0.19 clicks/user [−0.12, 0.50] (exact SATE +0.057); hybrid +0.085 [−0.22, 0.39] (SATE +0.048); both **HOLD**; CUPED variance reduction < 1% |
| A/A and coverage | A/A false-positive rate 0.040 [0.023, 0.069]; 95% coverage of the exact SATE 0.950, Bonferroni intervals 0.973 |
| Fresh seed 20260915 (run once) | item-kNN is best on oracle and SIMULATED online (1.101); EASE 1.082, blend 1.091, rankers 1.086–1.095; logged best = LambdaRank IPW; A/B: both HOLD, EASE's exact SATE −0.039 (worse than control) while its estimate was +0.18 [−0.08, 0.45] |
| Operational (indicative) | 30k × 5k: ALS fit 21 s, BPR 18 s, EASE 1 s (807 MB peak); dense M8 item-kNN (1.84 GB planned) and PureSVD (2.44 GB) **skipped** by the preflight check; k-means index (32 clusters, 8 probes) recall 0.84 scoring 170 of 5,000 items |

## Negative and null findings (reported, not hidden)

1. **No advanced model reliably beats the tuned M8 item-kNN.** On the main seed EASE wins on
   logged NDCG and the blend on oracle and simulated online value, but across five further seeds
   only the blend beats item-kNN on oracle NDCG, and on the fresh seed item-kNN is best on both
   oracle and simulated online value. Gains, where present, are about 1–2% of NDCG.
2. **Exposure-aware BPR is clearly worse** than uniform BPR (logged −0.094 vs item-kNN). A likely
   explanation (not tested separately): items the logging policy showed are disproportionately
   popular and relevant (exposure bias), so treating shown-but-unclicked items as negatives
   penalises good items.
3. **The two-tower model is the weakest learned model**, and its user tower serves zero-history
   users worse than segment popularity (0.514 vs 0.594). Its item tower beats random but not
   content-based scoring for new items.
4. **Propensity weighting had no material effect** on the rankers. It corrects only position bias
   among shown items; the dominant bias here is exposure (which items were shown at all).
5. **Off-policy estimators did not predict the A/B direction.** For both treatments IPS, SNIPS and
   DR predicted a negative per-session gap on the test window while the exact test-window gap and
   the A/B SATE were positive; the gaps (≈ 0.01 clicks/session) are far smaller than the
   estimators' uncertainty (the 95% intervals for each model's value are ≈ 0.25 wide).
6. **The studentised bootstrap only partly fixes M8's under-coverage**: coverage rises from
   0.897–0.940 to 0.927–0.938 but remains significantly below 0.95 in 6 of 8 cells; **none of the
   off-policy intervals attains nominal 95% coverage** reliably. In the single main-seed
   realisation DR with the fitted click model had the smallest mean absolute error (0.032 vs IPS
   0.040) but intervals only about 2% narrower (0.246 vs 0.251).
7. **The frozen A/B design was severely underpowered — a design defect, not only a small
   effect.** Post-hoc audit (main seed; standard deviation of clicks per user 2.93, standard
   error of each difference ≈ 0.135–0.138): power ≈ 0.07 at the pre-specified minimum worthwhile
   effect (0.10) and ≈ 0.03–0.04 at the true SATEs (0.048–0.057); about 17,600 users per arm would
   be needed for 80% power at the MWE, against about 1,000. The protocol fixed the A/B design
   without a power calculation (the M6 planning tools were available). Both HOLD decisions are
   therefore **uninformative**, not evidence of no effect; the A/A and coverage studies still
   validate the analysis machinery. CUPED with 14-day pre-period clicks reduced variance by under
   1%.

## Scientific assurance audit (pre-approval)

**Coverage of the acceptance criteria.** The 68 implementation tests were mapped to each
criterion. Gaps found and closed with 3 targeted tests (`tests/unit/recommendations/test_m9_audit.py`)
and one assertion: (a) end-to-end post-cutoff invariance of the multi-window two-stage pipeline,
the blend and the content → ALS-factor model; (b) SNIPS and DR against explicit loops when
positivity fails (sessions with some or no showable list items); (c) a guard that no test
evaluates the M9 fresh seed; (d) the A/B arms equal the validation-selected policies. No defect
was found by these tests.

**A/B hybrid training cutoff.** The frozen protocol specifies ranker windows only for the
test-window evaluation (ranker trained on labels (84, 98] with features at 84, applied at 98) and,
for the A/B test, only the start day and the selection rule. The implementation refits the
single-stage models and the hybrid's candidate generators on all logs up to day 120 but reuses the
final ranker without retraining, so the ranker is applied 36 days after its training-feature
cutoff (14 days in the test-window pattern). This is a choice the protocol left unspecified, not a
breach of a written rule; it may handicap the hybrid arm. It is documented here and not changed.

**Pre-declared vs exploratory.**

| Pre-declared (frozen protocol, or the M8 review's pre-specified grid) | Exploratory / descriptive |
| --- | --- |
| seeds; windows; K; grids; selection rules; A/B arms, primary metric, guardrails, Bonferroni, MWE; repeated-sampling sizes; ANN grid; scale tiers; memory budget; 3 × 3 logging-regime grid | candidate recall tables; cold-start bucket and new-item comparisons; LambdaRank training traces; DR "true click model" diagnostic; offline / off-policy vs A/B prediction table; comparison with M8 coverage; causal explanations of negative findings (hypotheses); the post-hoc A/B power analysis (this audit) |

Disclosed post-protocol changes are listed in results §1. One further change after the main-seed
dry run: a sentence in the cold-start reading (results §7) was corrected to say the two-stage
ranker differs for zero-history users; no number changed (dry-run and final §1–10 identical
apart from that sentence).

**Status corrections.** The M8 roadmap row and the M8 heading and footer in the capability
registry were updated from "awaiting review" to DONE (approved, committed `2d6b8e7`, CI run
43421279560 successful). Not changed (outside this audit's scope): the M7 roadmap row still reads
"awaiting review" although M7's CI-fix commit passed (run 43407936369), and the committed M8
evidence document's status line still reads "awaiting approval".

## Defects found and corrected during implementation (before any main-seed result)

All were found on a development seed or by reference tests, and are disclosed in results §1.

1. **DR click model fitted in sample** — the ALS score was fitted on the same clicks as the click
   model, so DR collapsed to IPS. Now fitted out of sample (score at day 84, labels (84, 98],
   applied with ALS at 98); regression test checks correlation and calibration with the truth on
   unseen items.
2. **LambdaRank with propensity weights oscillated** at the frozen step size (weights up to ~6).
   Pair weights are now rescaled by one constant (scale only; tested to leave the solution
   unchanged); training NDCG replaced the non-fixed weighted loss as the monitor.
3. **Pointwise ranker stopped early** (L-BFGS default tolerances): agreement with the M3
   logistic regression was 2e-4; tightened tolerances give 1e-6 and a convergence flag.

## Independent reference comparisons (tests)

| Component | Reference | Agreement |
| --- | --- | --- |
| ALS half-step; fold-in | dense weighted ridge solve per user | ≤ 1e-10 |
| ALS objective | full dense objective per iteration | non-increasing |
| BPR gradient; SGD step | central finite differences; `lr × gradient` incl. repeated rows | ≤ 1e-6; ≤ 1e-14 |
| EASE | column-by-column ridge regression excluding the column | ≤ 1e-10 |
| Two-tower gradients (linear, hidden layer) | central finite differences | ≤ 1e-6 |
| Exact retrieval; k-means index with every cluster probed | brute-force sort; exact search | identical |
| |ΔNDCG| | brute-force swap of each pair | ≤ 1e-12 |
| LambdaRank gradient | finite differences (fixed |ΔNDCG|) | rel ≤ 1e-4 |
| Pointwise ranker at L2 = 0 | M3 `fit_logistic` (Newton–Raphson) | ≤ 1e-6 |
| IPS | M8 `ips_policy_value` | ≤ 1e-12 relative |
| SNIPS, DR | explicit loops over sessions and slots | ≤ 1e-10 relative |
| Cluster-robust SE | closed form | exact |
| Simulated online period | per-session loop over the truth | ≤ 1e-12 |
| Ridge map (content → factors) | augmented least squares | ≤ 1e-10 |

## Tests

| File | Tests |
| --- | --- |
| `tests/unit/recommendations/test_factorisation.py` | 13 |
| `tests/unit/recommendations/test_two_tower_retrieval.py` | 6 |
| `tests/unit/recommendations/test_ranking.py` | 10 |
| `tests/unit/recommendations/test_off_policy_advanced.py` | 6 |
| `tests/unit/recommendations/test_hybrid_online_operational.py` | 20 |
| `tests/statistical/test_m9_recommender_properties.py` | 7 |
| `tests/integration/test_m9_case_studies.py` | 6 |
| `tests/unit/recommendations/test_m9_audit.py` (assurance audit) | 3 |

71 new tests (68 from implementation plus 3 from the audit; the plan estimated about 130 — the
reference checks above are covered, with fewer parametrised cases than planned).

## Validation

| Check | Result |
| --- | --- |
| `ruff check .` | passed |
| strict `mypy src` (and `--python-version 3.11`) — NumPy 2.5.4 / SciPy 1.18.1 and NumPy 2.4.6 / SciPy 1.17.1 | no issues, 96 files |
| `pytest`, both dependency sets | 768 passed (700 before M9 + 68); remaining warnings come only from an existing M4 statistical test |
| Evidence reproducibility | sections 1–10 regenerated without re-running the fresh seed: byte-identical to the final run on NumPy 2.5.4 and on NumPy 2.4.6; §11 deterministic columns (status, planned memory) identical; the fresh seed was evaluated once |
| M1–M8 preservation | tracked source, tests, data, scripts, configuration, M3–M8 evidence and methodology unchanged; the recommendations package docstring gained an M9 section |

## Deviations from the approved plan

- Fewer tests than estimated (68 vs ~130).
- The two-tower model is not in the scale-tier benchmark (its feature builder uses the dense
  history matrix); its fit time is reported at the main scale only.
- The A/B hybrid uses the final ranker trained on labels (84, 98]; it was not retrained on later
  data before the simulated period (protocol windows kept).
- Post-protocol changes are listed in results §1; none was prompted by a validation or test
  result.

## Limitations

See methodology §8. In particular: the generator's click model is also the simulated online
environment; the main seed's test window was inspected in M8 (the fresh seed is the clean holdout);
propensity weighting addresses position bias only; off-policy intervals under-cover even with the
studentised bootstrap; timings are machine-dependent; interference, feedback loops, novelty and
long-term effects are not simulated.
