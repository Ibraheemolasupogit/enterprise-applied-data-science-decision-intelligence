# Milestone 8 evidence — recommendation baselines

Status: **IMPLEMENTED — scientific assurance review completed (see the last section); awaiting
approval**. Not committed or pushed.
Statuses: [capability-status.md](capability-status.md). Methodology:
[../methodology/recommender-baselines.md](../methodology/recommender-baselines.md).
Generated numbers: [m8-recommendation-baselines-results.md](m8-recommendation-baselines-results.md)
(`python scripts/generate_m8_evidence.py`; byte-reproducible apart from the §15 timings and the
runtime table).

## Acceptance criteria

| Requirement | Evidence |
| --- | --- |
| Synthetic recommendation data with known preferences; M2 unchanged | `data/recommendation_scenarios.py` (additive); results §1; M2 tables used read-only in §14 |
| Ground truth separate from training, tuning and selection | truth returned separately; only `oracle.py` reads it; AST test over the other modules; `tune` accepts logged data only |
| Strict chronological train / validation / test with leakage perturbation tests | `splits.py`; every model's scores identical after rewriting all post-cutoff data (tests, results §2); random-holdout counterexample on the same held-out clicks |
| Baselines of Brief §8.3 incl. PureSVD and an explicit-ratings bias model | `baselines.py`, `explicit.py`; results §3–5, §12 |
| Full eligible catalogue, documented metric conventions, deterministic ties | `evaluation.top_k`, `metrics.py`; methodology §2, §4 |
| Ranking, coverage, novelty, diversity, exposure and fairness metrics | `metrics.py`; results §4, §8, §9 |
| Logged vs synthetic-oracle vs simulated online value kept distinct | `evaluation.py` / `oracle.py`; results §5–7 |
| IPS with defined estimand, propensities, position-dependent observation, positivity, ESS, clipping, unsupported recommendations | `off_policy.py`; methodology §6; results §11 |
| Offline-vs-online counterexample | results §6–7 (hype logging) |
| Cold-start users and items; fallback behaviour | `baselines.Fallback`; results §10; tests |
| Confidence intervals, paired comparisons, multiple-testing correction | M3 bootstrap and paired t, M6 Holm; results §4, §5, §9 |
| Independent references for critical methods | see below |
| NumPy 2.4.6 and 2.5.x, strict mypy, Python 3.11/3.12 | see Validation |

## Headline results (default logging unless stated; K = 10)

| Finding | Number |
| --- | --- |
| Logs | 3,000 users, 600 items, 53,083 sessions, 530,830 impressions, CTR 0.109 |
| Exposure bias in the logs | top 10% of items: 63.1% of impressions vs 23.1% of true attractiveness (hype: 77.1% vs 13.0%) |
| Position bias recovered from logs | largest error in relative examination 0.0045 |
| Logged NDCG@10 (test) | user-kNN 0.250 [0.238, 0.260]; item-kNN 0.247; PureSVD (k=8) 0.239; segment pop. 0.198; global pop. 0.164; content 0.072; random 0.011 |
| vs global popularity (Holm) | all six differences significant; content-based −0.092 [−0.102, −0.083] |
| Oracle NDCG@10 | user-kNN 0.600; item-kNN 0.593; PureSVD 0.578; segment pop. 0.523; global pop. 0.434; content 0.365; random 0.153 |
| Default logging, main seed: view agreement | Kendall τ with simulated online clicks: logged 1.0, oracle 1.0, IPS 0.90 (seed-specific: see the seed rows below) |
| Hype logging (regime chosen after inspecting results): view agreement | logged 0.52, oracle 0.90, IPS 1.0; logged winner user-kNN, online winner PureSVD |
| Counterexample (hype), content − global popularity | logged NDCG −0.103 [−0.115, −0.092]; oracle +0.129; simulated online clicks/session +0.300 [0.283, 0.317]; IPS +0.364 [0.026, 0.702] |
| Pre-specified regime grid (9 regimes, fixed models) | logged NDCG reverses the content-vs-popularity sign in 5/9; τ(logged, online) 0.33–1.00; τ(IPS, online) ≥ 0.87; IPS gap has the wrong sign in 1/9 |
| Fresh seed 20261114 (frozen protocol) | collaborative > popularity in all three views; content − popularity: default logged −0.052 [−0.060, −0.043] vs online +0.145 [0.129, 0.162], IPS +0.081 [−0.081, 0.244]; hype logged −0.061 vs online +0.285, IPS +0.225 [−0.006, 0.457] |
| Leakage (random holdout, same held-out clicks) | NDCG inflation: item-kNN +0.016 (p 1.6e-7), popularity +0.005 (p 1e-12), PureSVD +0.006 (p 0.09); post-cutoff perturbation changes no score (max 0) |
| IPS repeated sampling (1,000) | bias ≤ 1.2 MC SE (true examination), ≤ 1.8 MC SE (estimated); coverage of the nominal 95% interval: popularity 0.921 [0.903, 0.936] (p 8e-5), item-kNN 0.930 [0.912, 0.944] (p 0.006); misses one-sided (75:4, 65:5); mean SE = SD of estimates (0.052 / 0.053; 0.063 / 0.064); clipping at 50: −0.15 / −0.23; ignoring position: +1.0% / −5.3% of the value |
| Examination-estimation uncertainty | ≤ 0.6% of the IPS variance (session bootstrap of the training logs) |
| Positivity failure (logging limited to 150 items) | content-based: 74% of examination mass unsupported; IPS 0.259 [0.213, 0.305] ≈ supported value 0.250, not the full value 0.741 |
| Cold start | 330 cold test users: item-kNN without fallback 0.177 (item-id order) → 0.594 with segment-popularity fallback; 21 new items: content 0.607 vs popularity / item-kNN 0.279 (constant scores) vs random 0.344 |
| Beyond accuracy | coverage: popularity 0.043, item-kNN 0.325, content 0.959; exposure Gini 0.98 / 0.96 / 0.72 |
| Fairness | every model over-exposes `streaming_basic` (ratio 1.24–1.77); segment gaps significant for 5/5 models, region gaps for 0/5 after Holm (region is null by construction) |
| Explicit ratings (MNAR) | observed held-out RMSE 0.73 vs 1.88 on random pairs; mean rating 4.33 observed vs 1.98 on random pairs |
| Seed stability (5 seeds, default logging) | item-kNN oracle NDCG 0.550 ± 0.017 vs global pop. 0.321 ± 0.025 (item-kNN ahead in 5/5); content − popularity negative on logged NDCG and positive on simulated online clicks in 5/5 seeds |

## Independent reference comparisons (tests)

| Component | Reference | Agreement |
| --- | --- | --- |
| Precision, recall, AP, RR, NDCG | hand-worked examples; AP by definition over all 120 ordered 3-subsets; NDCG ideal vs brute-force best permutation | ≤ 1e-12 |
| Expected precision of a random ranking | exact enumeration: `|R|/n` | ≤ 1e-12 |
| Gini, diversity, personalisation | mean-absolute-difference formula; pairwise loops | ≤ 1e-12 |
| Global / time-decayed / segment popularity | pandas groupby; closed-form decay and shrinkage | exact / ≤ 1e-15 |
| Item-kNN, user-kNN, content-based | brute-force cosine loops with shrinkage and deterministic top-k | ≤ 1e-12 |
| PureSVD | Eckart–Young residual `Σ_{j>k} σ_j²`; `U_k Σ_k V_kᵀ`; sign invariance | rel ≤ 1e-10 |
| Bias rating model | direct augmented least-squares solve | ≤ 1e-8 |
| IPS estimator | explicit per-slot loop (clipped and unclipped) | rel ≤ 1e-10 |
| Simulated online value | per-session loop over the truth | ≤ 1e-12 |
| Logged propensities | recomputed logging distribution; slot frequencies over 300 re-draws | ≤ 1e-12; within 4 binomial SE |
| Expected rating | 200,000-draw Monte Carlo | ≤ 0.01 |
| IPS unbiasedness | 150–200 re-draws of the logging with the estimand fixed | within 4 MC SE (tests); results §11 |

## Tests

| File | Tests | Covers |
| --- | --- | --- |
| `tests/unit/recommendations/test_metrics.py` | 20 | hand-worked and brute-force metric references, conventions, validation |
| `tests/unit/recommendations/test_baselines.py` | 16 | each baseline vs brute force or closed form, fallback, ties, determinism, validation |
| `tests/unit/recommendations/test_splits.py` | 16 | split ordering, post-cutoff invariance for every model and the rating model, relevance/eligibility/seen rules, de-duplication, random-holdout leakage |
| `tests/unit/recommendations/test_evaluation.py` | 22 | truth separation (AST), tuning, paired/Holm, bootstrap, IPS loop reference, clipping, refusals, oracle loop reference, bias model vs least squares (incl. the chain-coupled regression case) |
| `tests/unit/data/test_recommendation_scenarios.py` | 21 | determinism, logging seed, schemas, time constraints, exact propensities, slot frequencies, support, truth, MNAR, defects, validation |
| `tests/statistical/test_m8_recommender_properties.py` | 11 | IPS bias/coverage/clipping/position, examination recovery, exposure-bias reversal, leakage, positivity, MNAR, seed stability |
| `tests/integration/test_m8_case_studies.py` | 13 | every case end to end on a reduced scenario (incl. the regime grid, fresh-seed replication and coverage diagnostics); reproducibility |

119 new tests (counts are pytest items, including parametrised cases).

## Validation

| Check | Result |
| --- | --- |
| `ruff check .` | passed |
| Strict `mypy src` — NumPy 2.5.4 / SciPy 1.18.1 (Python 3.12 job's versions) | no issues, 82 files |
| Strict `mypy --python-version 3.11 src` — NumPy 2.4.6 / SciPy 1.17.1 (Python 3.11 job's versions) | no issues, 82 files |
| `pytest`, both dependency sets | 700 passed (581 before M8 + 119) |
| Evidence regenerated twice | identical apart from §15 timings and the runtime table |
| Evidence under NumPy 2.4.6 / SciPy 1.17.1 vs 2.5.4 / 1.18.1 | sections 1–14, 16, 17 identical |

## Deviations from the approved plan, and analysis-history disclosure

- **The hype regime was not pre-declared.** It was chosen after model results had been compared
  in four exploratory regimes, as one that showed a logged-vs-online reversal. Earlier versions of
  this document and the results called it "pre-declared"; that was wrong and is corrected. The
  assurance review added a 3 × 3 regime grid fixed before its results were seen (results §16) and
  a replication on a never-inspected seed (§17); both show the reversal is common, not a single
  selected case.
- **The main seed's test window was used in exploration.** PureSVD (16 and 64 factors) and other
  fixed configurations were scored on it before the tuning grid was written; the 8-factor option
  was then added and is what validation selected. The fresh-seed replication (§17) is the clean
  holdout.
- The default regime's click intercept and rating parameters were calibrated on descriptive log
  statistics (click-through rate, rating distribution) before any model was evaluated.
- IPS uses examination estimated from the logs (the realistic case); the true examination is also
  reported. IPS intervals are not always decisive, and the tests assert only what the evidence
  supports.
- The random-holdout counterexample scores both protocols on the *same* held-out clicks; its effect
  is modest and not significant for PureSVD.

## Limitations and unresolved risks

See methodology §9. In particular: synthetic latent-factor data favour latent-factor models;
independent-slot logging makes propensities exact and examination identifiable in a way real slate
logging does not; simulated online value is not an experiment; the normal IPS interval
under-covers (0.92–0.93 in 1,000 replications) because the estimator is right-skewed, and is wide
where few logged slots support a list; fairness merit uses synthetic truth; bootstrap intervals
hold fitted models fixed; the main seed's test window is not a pristine holdout.

## Scientific assurance review (pre-approval)

An independent review re-derived the critical computations, audited the protocol and the
analysis history, and investigated IPS coverage.

**Independent checks that passed (no defect).**

| Area | Check | Agreement |
| --- | --- | --- |
| Ranking metrics | pure-Python reimplementation, 3,000 random cases incl. empty relevance sets, graded gains with ties, K > |R|; IDCG vs brute force over permutations; `top_k` vs a (−score, index) sort with ties | ≤ 3.3e-16; all rankings identical |
| Item-kNN, user-kNN | dictionary-of-sets cosine with shrinkage and deterministic top-k, on scenario data | ≤ 1.3e-15 |
| PureSVD | ARPACK `scipy.sparse.linalg.svds` projector (singular-value gap 0.25) | ≤ 4.6e-15 |
| Segment popularity | pandas group shares with shrinkage | exact |
| Category exposure / merit shares; Kruskal–Wallis | pandas recomputation; `scipy.stats.kruskal` | ≤ 1.9e-15; exact |
| Simulated margin per session; clustered contrast intervals | per-session loop; independent CR formula | ≤ 2.6e-13; ≤ 1.1e-10 |
| Leakage | chronological cutoffs, launch-date eligibility, seen-item exclusion, validation-only tuning, examination estimated before the test window, post-cutoff perturbation of every model | no leakage found |
| IPS | logged propensities exact; independent slots; estimand, positivity diagnostics, clipping and ESS as documented | as specified |

**Confirmed defects and corrections.**

1. **Explicit bias model did not converge, silently.** Block coordinate descent stopped at its
   500-iteration cap without meeting its tolerance on the main scenario (bias errors 2e-8 — too
   small to change any reported number) and, on weakly regularised chain-coupled ratings, with
   errors of 5e-3. Replaced by an exact sparse solve of the ridge normal equations; regression test
   `test_bias_model_is_exact_on_weakly_regularised_chain_coupled_ratings` fails on the old code
   and passes now. Reported RMSEs unchanged at the reported precision.
2. **False "pre-declared" claim** for the hype regime, and undisclosed exploratory use of the main
   seed's test window — corrected in the code docstring, the evidence generator, this document,
   the methodology and the README; regime grid (§16) and fresh-seed replication (§17) added.
3. **"Under default logging the views agree" was seed-specific.** True for the main seed only;
   across seeds 1–5 and the fresh seed, logged NDCG puts content-based below popularity while
   simulated online value puts it above. Readings corrected and computed from the data.
4. **IPS coverage overstated as benign.** With 200 replications the reported 0.925–0.930 was not
   distinguishable from 0.95 (MC SE 0.015). With 1,000 replications coverage is 0.921 and 0.930,
   significantly below 0.95 (exact binomial p 8e-5 and 0.006); an independent 1,000-replication
   run on other logging seeds gave 0.932 and 0.950. Diagnosis: the clustered SE is calibrated
   (mean SE ≈ SD of estimates), clustering and examination estimation contribute little (i.i.d.
   SE gives the same coverage; ≤ 0.6% of variance), and the misses are one-sided with a 0.7–0.8
   estimate–SE correlation — the skewness of an importance-weighted mean, a limitation of the
   normal interval rather than a coding defect. Documented; not "corrected" by an ad hoc interval.
5. **Smaller overstatements**: "cold-start users" are users with no training clicks (most sign up
   after the cutoff, some never clicked) — methodology wording corrected; ignoring position bias
   is detectable for popularity (+1.0% of the value, 2.4 MC SE), not "negligible"; the README's
   "offline ranking gains are not online gains" is now "need not be".

**Recommendation of the review:** see the completion report; the remaining limitations are listed
above and in the methodology.
