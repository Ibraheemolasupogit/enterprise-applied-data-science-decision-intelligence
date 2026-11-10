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

Statistical primitives (logistic/linear regression, bootstrap, validation)
come from M3 (``decision_intelligence.statistics`` and ``.common``).
Observational designs that exploit time or thresholds (difference-in-
differences, event studies, synthetic control, regression discontinuity)
are Milestone 5 and deliberately absent.
"""
