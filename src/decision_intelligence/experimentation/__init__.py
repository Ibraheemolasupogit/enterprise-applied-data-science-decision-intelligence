"""Pillar 2: A/B testing and experimentation — Milestones 6 and 7.

Fundamentals (M6):

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

Advanced experimentation (M7):

* :mod:`.variance_reduction` — CUPED and regression adjustment with treatment
  interactions (Lin 2013), covariate timing checks and planning with a
  historical R².
* :mod:`.sequential` — Lan–DeMets alpha-spending and classical
  group-sequential boundaries (exact recursive integration), operating
  characteristics, monitoring with repeated confidence intervals, and
  always-valid inference (mixture SPRT, confidence sequences).
* :mod:`.bayesian` — Beta-binomial and normal-normal posteriors with explicit
  priors, credible intervals, posterior probabilities and expected-loss
  decisions in money.
* :mod:`.designs` — ICC estimation and cluster-randomised planning; switchback
  assignment and analysis with burn-in and randomisation inference.
* :mod:`.advanced_cases` — the M7 case studies and repeated-sampling studies,
  each compared with the M6 fixed-horizon baseline.

Statistical primitives come from M3 (``decision_intelligence.statistics``,
including the new :mod:`~decision_intelligence.statistics.multiple_testing`),
causal estimate containers and cluster-robust inference from M4/M5
(``decision_intelligence.causal_inference``). Geo experiments, interference
modelling, heterogeneous-effect analysis, multi-armed bandits and long-term
effect estimation are not implemented.
"""
