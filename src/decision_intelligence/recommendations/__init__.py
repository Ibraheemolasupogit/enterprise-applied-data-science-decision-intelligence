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

Training, tuning and model selection accept logged data only; ground truth enters only
through :mod:`.oracle`, after the fact.
"""
