"""Pillar 5: customer lifetime value — Milestone 10 (LTV foundations and customer economics).

* :mod:`.protocol` — the frozen evaluation protocol.
* :mod:`.economics` — period conventions, discounting, contribution, CAC, payback, LTV/CAC.
* :mod:`.panel` — the point-in-time customer-month panel and cohorts.
* :mod:`.retention` — cohort retention tables, Kaplan–Meier with Greenwood intervals.
* :mod:`.historical` — realised (historical) contribution with censoring-aware cohort rules.
* :mod:`.heuristic` — heuristic projected and expected-lifetime contribution (H0–H4).
* :mod:`.backtest` — temporal holdouts with complete observation windows.
* :mod:`.cases` — the case studies behind the M10 evidence.

Three quantities are kept apart throughout: **realised** contribution (observed, historical),
**projected** contribution (a heuristic forecast over a stated horizon) and **expected
lifetime** contribution (a model's expectation to an infinite or long horizon). Estimators read
observable tables only; synthetic ground truth is read only by evaluation code in
:mod:`.cases`. Predictive, probabilistic and causal (incremental) LTV are Milestone 11.
"""
