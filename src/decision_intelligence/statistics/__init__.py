"""Pillar 1 (part one): statistical foundations — Milestone 3.

Modules:

* :mod:`.distributions` — inverse-transform and rejection sampling,
  empirical CDF/KS distance, simple random and stratified sampling.
* :mod:`.descriptive` — summaries with explicit conventions, stable moments.
* :mod:`.estimators` — sampling distributions, bias/variance/MSE, CLT
  diagnostics, exact exponential-rate and t intervals.
* :mod:`.simulation` — reproducible Monte Carlo with simulation error,
  rejection-rate (type I error / power) and coverage studies.
* :mod:`.hypothesis_tests` — t, proportion, chi-square, exact and
  asymptotic rank tests, permutation, ANOVA/Kruskal–Wallis, diagnostics.
* :mod:`.effect_sizes` — standardised and binary effect sizes with CIs.
* :mod:`.power` — power, sample size and minimum detectable effect.
* :mod:`.linear_regression` — OLS with classical/HC/cluster-robust SEs
  and diagnostics.
* :mod:`.logistic_regression` — MLE logistic regression with separation
  detection; discrimination and calibration diagnostics.
* :mod:`.bias_variance` — simulated bias–variance decomposition.
* :mod:`.missing_data` — MCAR/MAR/MNAR mechanisms and simple strategies.
* :mod:`.applied_analyses` — the above applied to the M2 synthetic data.

Bootstrap intervals live in :mod:`decision_intelligence.common.bootstrap`
(the shared owner); input validation in
:mod:`decision_intelligence.common.validation`.

Causal identification (M4), observational causal methods (M5) and
experimentation infrastructure (M6+) are deliberately not here.
"""
