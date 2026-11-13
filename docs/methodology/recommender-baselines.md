# Recommendation baselines (Milestone 8)

Pillar 3's first milestone: the benchmarks every later recommender must beat, and an evaluation
protocol that does not fool itself. Code: `src/decision_intelligence/recommendations/` and
`src/decision_intelligence/data/recommendation_scenarios.py`. Evidence:
[../evidence/m8-recommendation-baselines.md](../evidence/m8-recommendation-baselines.md).

The central question (Brief §8.1): *which items should be presented to which users, and how do we
establish that the recommendations create value?* M8 answers the first half with baselines and the
second half only as far as offline data allow — and shows where offline data stop being enough.

## 1. Synthetic data with known preferences

The M2 catalogue (25 products, category-level affinity) cannot distinguish item-level models, so
M8 adds an independent generator (no M2 table or contract is changed; the six M2 categories are
reused). Full specification in the module docstring; in short:

| Component | Model | Observable? |
| --- | --- | --- |
| Users | segment, region, sign-up time; latent taste `u_u = m_segment + e_u`, drifting linearly over the horizon | segment, region, sign-up |
| Items | category, 12 binary tags, price, margin, launch time; latent `v_i` partly from tags and category, partly idiosyncratic; quality `b_i` | attributes only |
| True attractiveness | `a_ui(t) = expit(α + b_i + u_u(t)·v_i)` | no |
| Position bias | examination `θ_k = k^-0.8`; click with probability `θ_k a_ui(t)` (position-based model) | no (ratios estimable) |
| Purchase | after a click, `expit(q0 + q1 s_ui − q2 price_z)` | yes |
| Logging policy | each of 10 slots filled independently from `μ(i|u,t) = (1−ε) softmax(β_pop pop_i + β_pers û_u·v_i) + ε/|S_t|` over launched (and supported) items | propensity of the shown item is logged |
| Explicit ratings | 1–5, left on clicked items with probability rising in `s_ui` (missing not at random) | yes |

The logging policy's popularity score `pop_i` is correlated with quality but is not quality; its
correlation and weight define the regimes studied: **default** (correlation 0.7, weight 1.5;
fixed before any model was evaluated) and **hype** (correlation 0, weight 2.5) — the latter
**chosen after inspecting model results** in four exploratory regimes, so it is an illustration,
not a pre-specified test. The assurance review added a 3 × 3 grid (correlation 0.7 / 0.35 / 0 ×
weight 1.5 / 2.0 / 2.5) fixed before its results were seen, and a replication of the frozen
protocol on a never-inspected seed (20261114). Independent slots make propensities exact and
logging position-exchangeable, at the cost that a slate may repeat an item.

Ground truth (`RecommendationTruth`) is returned separately from the logs (`RecommendationLogs`).
Only `recommendations/oracle.py` reads it; a test parses the training, tuning and selection
modules and fails if any of them references the truth type.

## 2. Evaluation protocol

1. **Chronological split** at two global cutoffs: train ≤ day 84, validation days 84–98, test days
   98–120. No per-user leave-last-out (it lets a model train on interactions that happen after
   other users' test interactions).
2. **Point in time**: every training table is truncated with
   `data.point_in_time.as_of`; duplicated impression rows are dropped first. A perturbation test
   rewrites every impression, click and rating after the cutoff and requires every model's scores
   to be *identical*.
3. **Tuning**: each family's grid is fitted on train and scored on validation by logged NDCG@10;
   the winner is refitted on train + validation and scored once on test. Ground truth never enters
   tuning (the `tune` signature accepts only logged data).
4. **Eligible catalogue**: every item launched by the cutoff; models rank the **full** eligible
   catalogue (no sampled negatives — sampled metrics can reorder models). Items the user clicked
   before the cutoff are excluded from both the ranking and the relevant set (a discovery task).
5. **Deterministic ranking**: scores are sorted with a stable sort; ties go to the lower item id.
   `top_k` refuses rows with fewer than K allowed items or non-finite allowed scores.
6. **Invalid protocol, for contrast**: a random split of the test-window clicks (half held out,
   half added to training) scored on exactly the same held-out clicks as the chronological model.

## 3. Baselines (Brief §8.3)

| Model | Score | Hyperparameter grid |
| --- | --- | --- |
| Random | seeded uniform | — |
| Global popularity | click count, optionally exponentially decayed | half-life none / 7 / 28 days |
| Segment popularity | `(n_gi + λ s_i)/(n_g + λ)` — segment share shrunk to the global share | λ = 10 / 100 / 1000 clicks |
| Item-item kNN | `Σ_j x_uj S_ji`, shrunk cosine (shrink 10), top-k neighbours per item | k = 20 / 50 / 100 |
| User-user kNN | `Σ_v S_uv x_vi` over the top-k most similar users | k = 25 / 50 / 100 |
| Content-based | cosine between item content vector (tags + category) and the user's profile | — |
| PureSVD (matrix factorisation) | `x_u V_k V_kᵀ`, truncated SVD of the click matrix | k = 8 / 16 / 32 / 64 |
| Bias rating model (explicit) | `μ + b_u + b_i`, ridge-penalised biases | λ = 5 |

Personalised models fall back to segment popularity (λ = 100) for users with no training clicks.
Implicit ALS, BPR, two-tower retrieval, learning-to-rank and session models are M9.

## 4. Metrics and conventions

Ranking metrics (users with ≥ 1 relevant item; K = 10): Precision@K (denominator K), Recall@K
(denominator |R|), HitRate@K, MAP@K with denominator min(|R|, K), MRR@K (0 if no hit in the top K),
NDCG@K with linear gains and ideal DCG over the *allowed* items. Beyond accuracy: catalogue
coverage, exposure Gini / normalised entropy / top-10% share (position-weighted 1/log2(k+1),
over all eligible items including zeros), novelty (add-one-smoothed self-information), intra-list
diversity (1 − mean pairwise cosine of content vectors) and personalisation (1 − mean pairwise
Jaccard). Explicit ratings: RMSE. Conventions are fixed in `metrics.py`; libraries differ on the
MAP denominator, NDCG gain and tie handling.

## 5. Three views that must not be confused

| View | Question | Available in practice? |
| --- | --- | --- |
| Logged feedback | Does the list contain items the user clicked in the test window? | yes |
| Synthetic oracle | Does the list contain the items the user truly finds most attractive? (graded NDCG with gains `a_ui` at the test-window midpoint) | no — synthetic only |
| Simulated online value | Expected clicks / purchases / margin per test session had the list been shown, under the generator's click model | no — a simulation of an online test, not one |

Logged metrics are biased by exposure: an item can only be clicked if the logging policy showed it,
so a model that agrees with the logging policy is rewarded. The evidence shows this reversing a
model comparison (content-based vs global popularity) in 5 of 9 pre-specified logging regimes, in
every one of five further seeds under default logging, and on the fresh seed (results §7, §13,
§16, §17); on the main seed under default logging the views happen to agree. Other reasons offline
ranking gains need not become online gains — discussed, not modelled here — include the list
being shown in a different context, novelty effects, and business value depending on purchases
and margin rather than clicks. Online confirmation requires an experiment (M6/M7 machinery; M9).

## 6. Off-policy (IPS) evaluation

**Estimand.** `V(π)` = expected reward per logged test session had the target list `π_u` (fixed at
the cutoff) been shown in that session — the simulated online value, conditional on the realised
sessions.

**Estimator.** For logged slot j (item `i_j`, position `p_j`, propensity `μ_j`, reward `y_j`) and
the item's target position `k(i_j)`: `w_j = (θ_k/θ_p)/μ_j` if the item is in the list, else 0;
`V̂_s = (1/L) Σ_j y_j w_j`; `V̂ = mean_s V̂_s`. Standard errors are clustered by user (a user's
sessions share one list); intervals are normal.

**Assumptions** (unbiasedness holds only under all four): (1) position-based examination with no
slate interactions; (2) known propensities and independently filled slots; (3) positivity — every
target item could have been logged in the session; (4) known *relative* examination, estimated
here by `CTR_k/CTR_1`, which is valid only because logging is position-exchangeable.

**Diagnostics reported.** Effective sample size `(Σw)²/Σw²`; the largest weight; the share of
target examination mass on unsupported items (where IPS estimates only the supported part of `V`
— a downward bias that more data do not remove); clipping at M (lower variance, biased downwards
for non-negative rewards, and reported as biased); sensitivity to estimated, true and ignored
examination; repeated-sampling bias and interval coverage with everything but the logging
randomness fixed.

## 7. Fairness, segments and cold start

- **Item side**: position-weighted exposure share by category ÷ the category's share of users'
  oracle top-10 lists (merit defined by synthetic truth).
- **User side**: oracle and logged NDCG by segment and region; Kruskal–Wallis across groups with
  Holm across the table. Segments differ in taste by construction; region does not affect
  preference, so region differences are noise — a built-in false-positive check. Gaps are prompts
  for review, not proof of unfair treatment. No income-related attribute is generated or used.
- **Cold start**: users with no training clicks (most sign up after the cutoff; some signed up
  earlier but never clicked) and items launched during the test
  window. Collaborative models give cold users constant scores (item-id order); the fallback
  replaces them with segment popularity. New items can be ranked only from content.

## 8. Uncertainty and comparisons

Per-user metrics are averaged with percentile bootstrap intervals over users (M3
`bootstrap_ci`, 2,000 resamples). Models are compared with the reference (global popularity) by
paired t-tests on per-user differences (M3) with Holm adjustment across the family (M6). These
intervals hold fitted models fixed; variation from the data-generating draw is reported separately
across five generator seeds. Online and IPS differences between two lists use per-session
differences with user-clustered standard errors.

## 9. Limitations

- Synthetic data only. The generator is a latent-factor model, which favours latent-factor and
  neighbourhood methods; external validity is not established.
- The click model is position-based with independent slots; real logging policies fill slates
  jointly, so exact propensities would need the slate-level probability and the examination ratio
  would need randomised positions or intervention harvesting.
- Simulated online value is a model output, not an online test.
- The normal IPS interval under-covers: 0.921 and 0.930 in 1,000 replications (exact binomial p
  8e-5 and 0.006 against 0.95). The clustered SE is calibrated and examination estimation adds
  ≤ 0.6% to the variance; the cause is the right skew of the importance-weighted mean (misses are
  almost all on the low side, estimate and SE correlate 0.7–0.8). The interval treats the
  estimated examination as known. A skew-aware interval (e.g. a studentised cluster bootstrap) is
  not implemented.
- The main seed's test window was used in exploratory runs before the tuning grid was written (the
  PureSVD 8-factor option was added afterwards); the fresh-seed replication is the clean holdout.
- Bootstrap intervals ignore training variability; the fallback shrinkage is fixed, not tuned;
  hyperparameter grids are small.
- Exposure-fairness merit is defined by synthetic truth, which a real system does not have.
- Scale is small (3,000 × 600, dense matrices); item-kNN and user-kNN memory grows quadratically.
  Operational evaluation, implicit ALS, BPR, learning-to-rank, session-based and hybrid models,
  cold-start strategy comparison and online A/B evaluation are M9.
