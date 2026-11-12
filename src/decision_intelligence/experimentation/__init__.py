"""Pillar 2: experimentation fundamentals — Milestone 6.

Modules:

* :mod:`.design` — the pre-specified design: experimental, randomisation and
  analysis units, estimand, primary/secondary/guardrail metrics, the
  minimum worthwhile effect and exclusion rules.
* :mod:`.assignment` — deterministic hash-based assignment; complete,
  stratified and cluster randomisation with exact allocation probabilities.
* :mod:`.srm` — sample ratio mismatch detection (detection, not diagnosis)
  and localisation by pre-treatment segment.
* :mod:`.planning` — sample size, minimum detectable effect, duration and
  design effect (wrapping the M3 power functions).
* :mod:`.analysis` — ITT analysis of binary and continuous metrics with
  absolute and relative effects; stratified, clustered and A/B/n analyses.
* :mod:`.compliance` — assignment vs exposure (ITT, CACE, and the biased
  as-treated/per-protocol comparisons) and attrition (Manski and Lee bounds).
* :mod:`.validity` — duplicate, SRM, leakage, balance, exclusion, metric
  integrity and post-randomisation selection checks.
* :mod:`.decision` — pre-specified LAUNCH / HOLD / ITERATE / STOP rules with
  guardrail non-inferiority and business-value translation.
* :mod:`.cases` — the M2 experiment and scenario case studies, and the
  repeated-sampling studies behind the M6 evidence.

Statistical primitives come from M3 (``decision_intelligence.statistics``,
including the new :mod:`~decision_intelligence.statistics.multiple_testing`),
causal estimate containers and cluster-robust inference from M4/M5
(``decision_intelligence.causal_inference``). CUPED and covariate
adjustment, sequential and Bayesian testing, switchback/geo designs and
interference are M7 scope and not implemented.
"""
