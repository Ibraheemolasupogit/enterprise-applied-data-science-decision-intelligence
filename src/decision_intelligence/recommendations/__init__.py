"""Pillar 3: recommender systems — Milestone 8 (recommendation baselines).

* :mod:`.interactions` — implicit-feedback matrices from logged data as of a cutoff
  (point-in-time via :func:`decision_intelligence.data.point_in_time.as_of`),
  de-duplication and item content features.
* :mod:`.splits` — the chronological train / validation / test protocol, the eligible
  catalogue, seen-item exclusion, and the (invalid) random-holdout counterexample.
* :mod:`.baselines` — random, global and segment popularity, item- and user-kNN,
  content-based and PureSVD recommenders, and the cold-start fallback.
* :mod:`.explicit` — a regularised bias model for explicit ratings.
* :mod:`.metrics` — ranking, coverage, exposure, novelty, diversity and personalisation
  metrics with documented conventions.
* :mod:`.evaluation` — deterministic top-K, logged-feedback evaluation, bootstrap
  intervals, Holm-adjusted paired comparisons and tuning on a validation window.
* :mod:`.off_policy` — position-based IPS estimates of a list's value with examination
  estimated from the logs, clipping, positivity and effective-sample-size diagnostics.
* :mod:`.oracle` — synthetic-oracle metrics and simulated online value (evaluation only;
  the one module that reads ground truth).
* :mod:`.cases` — the case studies behind the M8 evidence.

Milestone 9 (advanced recommendation and ranking), additive:

* :mod:`.protocol` — the frozen evaluation protocol (seeds, windows, grids, A/B design).
* :mod:`.factorisation` — implicit ALS, BPR and EASE; :mod:`.two_tower` and :mod:`.retrieval` —
  a shallow two-tower model with exact and approximate search.
* :mod:`.candidates`, :mod:`.features`, :mod:`.ranking` — two-stage learning-to-rank (pointwise,
  LambdaRank-style, propensity-weighted); :mod:`.hybrid` — score blend; :mod:`.cold_start`.
* :mod:`.off_policy_advanced` — SNIPS, doubly robust estimation, bootstrap intervals.
* :mod:`.operational` — latency, throughput, size and memory benchmarks with preflight checks.
* :mod:`.online` — SIMULATED user-randomised A/B evaluation with the M6/M7 components.
* :mod:`.advanced_cases` — the case studies behind the M9 evidence.

Training, tuning and model selection accept logged data only; ground truth enters only through
the evaluation layer (:mod:`.oracle`, and in M9 :mod:`.online` and :mod:`.advanced_cases`), after
the fact.
"""
