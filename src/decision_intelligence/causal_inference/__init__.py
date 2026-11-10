"""Pillar 1 (part two): causal foundations, DAGs and backdoor adjustment — Milestone 4.

Modules:

* :mod:`.estimands` — potential outcomes; ATE, ATT, ATC, CATE; the exact
  decomposition naive difference = ATT + selection bias.
* :mod:`.dag` — causal DAGs: d-separation, path blocking, backdoor
  criterion, valid adjustment sets, confounder/mediator/collider roles.
* :mod:`.estimators` — naive contrast, regression adjustment
  (g-computation), IPW (Horvitz–Thompson, Hájek) and doubly robust AIPW for
  the ATE and ATT; subgroup (conditional) effects; AIPW risk ratio.
* :mod:`.diagnostics` — overlap, covariate balance, weight stability, trimming.
* :mod:`.sensitivity` — E-values and a bias grid for unmeasured confounding.
* :mod:`.scenarios` — simulations with known truth, including failures
  (mediator and collider adjustment, positivity, misspecification, interference).
* :mod:`.campaign` — the M2 retention-offer case study and its repeated-sampling study.

Milestone 5 adds observational designs that exploit time or thresholds:

* :mod:`.did` — two-period and panel difference-in-differences, staggered
  adoption (Callaway & Sant'Anna), TWFE with an exact negative-weight
  diagnostic, event studies and windowed finite-cluster pre-trend tests.
* :mod:`.cluster_inference` — wild cluster restricted bootstrap for few clusters.
* :mod:`.synthetic_control` — synthetic control with fit diagnostics, placebo
  ranks, in-time placebos and leave-one-out sensitivity.
* :mod:`.rdd` — sharp regression discontinuity with robust bias-corrected
  inference and density, covariate, placebo-cutoff and donut checks.
* :mod:`.observational_cases` — the M5 case studies and repeated-sampling studies.

Statistical primitives (logistic/linear regression, bootstrap, validation)
come from M3 (``decision_intelligence.statistics`` and ``.common``).
Interrupted time series, fuzzy RDD and propensity-score matching are not
implemented.
"""
