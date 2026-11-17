"""Milestone 11 evaluation protocol — predictive customer lifetime value (P0).

Everything that could be tuned to results is fixed here before any M11 model is fitted: seeds,
scenarios, cutoffs, landmarks, horizons, information tiers and features, models and their
roles, targets, the primary comparisons, model-selection and calibration rules, fit-status
rules, customer-clustered inference, repeated-sampling sizes and runtime budgets. The approved
design and its three scientific amendments are encoded as follows:

1. **Tier B vs Tier A oracle** — the inequality ``MSE_A >= MSE_B`` is required in population
   expectation only; finite-sample differences are assessed with uncertainty
   (:data:`ORACLE_GATE`). A single seed with the opposite sign is reported, not a failure.
2. **Repeated customer landmarks** — the unit of resampling and inference is the customer
   (:data:`CLUSTERED_INFERENCE`); penalty-tuning landmarks have label windows that end before
   the validation landmark (:func:`tuning_landmarks`).
3. **No qualified model** — if no candidate passes the selection gates the result is
   :data:`NO_QUALIFIED_MODEL`; E2 may appear only as :data:`REFERENCE_FALLBACK_LABEL`, never as
   a selection (:mod:`.model_selection`).

**Stage.** This is the P0 protocol. Items that later phases must fix before any main-seed
evaluation (the repeated-sampling preflight and the oracle validation sizes) are listed in
:data:`PENDING_BEFORE_MAIN`; :func:`authorise_seed` refuses the main and fresh seeds while any
item is pending. Every change made after P0 and before any main-seed evaluation is recorded in
:data:`PROTOCOL_AMENDMENTS`.

**Integrity.** :func:`protocol_digest` hashes every protocol value; :func:`protocol_code_digest`
hashes the exact bytes of this module and of the decision logic in :mod:`.model_selection` and
:mod:`.precision`. Tests pin both, so any change is a visible, reviewable diff that must be
disclosed in the M11 evidence. Generator parameters are pinned by
:data:`SCENARIO_PARAMETER_DIGESTS` (checked against the generators by a test). P0 pilot
findings are final only for the M10 scenario (:data:`PILOT_FINDINGS_SCOPE`).

**Economics are M10's** (fair comparison): contribution, the 10% annual discount rate, the
45-day recording lag, the stability thresholds and the heterogeneity grid are imported from
:mod:`.protocol`, not redefined.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

from decision_intelligence.customer_value import protocol as m10

PROTOCOL_ID = "M11-predictive-ltv"
PROTOCOL_STAGE = "P1-amended"

# --- seeds -------------------------------------------------------------------------------
MAIN_SEED = 20261101
PILOT_SEED = 20261102  # precision and feasibility planning only
FRESH_SEED = 20261103  # one final evaluation only, last, after everything else is frozen
COVERAGE_SEED_START = 810_000  # no integer literal in this block exists elsewhere (tested)
COVERAGE_REPLICATIONS = 1000
"""Repeated-sampling draws for studies whose per-draw cost is one fit per model plus analytic
or simulation intervals (owner decision: 1,000 where computationally feasible; a P1 preflight
confirms it, otherwise a reduced count >= MIN_COVERAGE_REPLICATIONS is declared before main)."""
MIN_COVERAGE_REPLICATIONS = 500
REFIT_STUDY_SEED_START = 820_000
REFIT_STUDY_REPLICATIONS = 200
"""Separate, smaller studies whose every draw needs refit-bootstrap intervals
(``CLUSTERED_INFERENCE.refit_bootstrap_resamples`` refits per draw); reported separately."""
M10_COMPARABILITY_SEED = m10.MAIN_SEED
"""M10's main seed, already examined in M10: used only for the labelled M10 comparability
replication ("comparability, not fresh evaluation")."""
DEVELOPMENT_SEED_MAX = 9_999
"""Tests and development scenarios use seeds in ``[0, DEVELOPMENT_SEED_MAX]`` only."""
OTHER_PROTOCOL_SEEDS = (m10.MAIN_SEED, m10.PILOT_SEED, m10.FRESH_SEED,
                        20260801, 20260915, 20261114)
"""Seeds reserved by earlier protocols (M10; M8/M9 main, M9 fresh, M8 fresh) — M11 must not
reuse them (checked against the recommendation protocol by a test, without importing it here:
pillars do not import one another)."""
OTHER_COVERAGE_BLOCKS = ((m10.COVERAGE_SEED_START,
                          m10.COVERAGE_SEED_START + m10.COVERAGE_REPLICATIONS),)
_PROTOCOL_TEST = "tests/unit/customer_value/test_predictive_protocol.py"
SEED_ALLOWED_FILES = {
    "MAIN_SEED": ("scripts/generate_m11_evidence.py", _PROTOCOL_TEST),
    "FRESH_SEED": ("scripts/generate_m11_evidence.py", _PROTOCOL_TEST),
    "PILOT_SEED": ("scripts/plan_m11_precision.py", "scripts/generate_m11_evidence.py",
                   _PROTOCOL_TEST),
}
"""The only files (besides this module) that may reference each reserved seed by name or by
its name as a string. The protocol's own test checks the seed gate's return values and never
generates data with them."""
AUTHORISE_ALLOWED_FILES = {
    "pilot": ("scripts/plan_m11_precision.py", "scripts/generate_m11_evidence.py",
              _PROTOCOL_TEST),
    "main": ("scripts/generate_m11_evidence.py", _PROTOCOL_TEST),
    "fresh": ("scripts/generate_m11_evidence.py", _PROTOCOL_TEST),
}
"""The only files that may call :func:`authorise_seed` for each purpose; calls must pass the
purpose as a literal string."""
PROTOCOL_CODE_FILES = ("predictive_protocol.py", "model_selection.py", "precision.py")
"""Files whose exact bytes are pinned by :func:`protocol_code_digest` (the manifest digest
covers values; this covers the decision logic)."""

# --- time, economics, recording ----------------------------------------------------------
ANNUAL_DISCOUNT_RATE = m10.ANNUAL_DISCOUNT_RATE
MONTHLY_DISCOUNT_RATE = m10.MONTHLY_DISCOUNT_RATE
INFINITE_SUM_MONTHS = m10.INFINITE_SUM_MONTHS
RECORDING_LAG_DAYS = m10.MAX_RECORDING_LAG_DAYS
COMPLETE_MONTH_OFFSET = 2
"""``complete_through(C) = C - 2`` under the 45-day lag (checked against M10's
``Observable.complete_through`` on the real calendar by a test)."""
SIGNAL_RECORDING_LAG_DAYS = 10
SIGNAL_COMPLETE_OFFSET = 1
"""Behavioural signals for month ``m`` are usable at cutoff ``C`` only if ``m <= C - 1``."""
HAZARD_COVARIATE_LAG_MONTHS = 1
"""A churn decision at the end of month ``a`` may use signals of months ``<= a - 1`` only."""

# --- horizons, cutoffs, landmarks ----------------------------------------------------------
PRIMARY_HORIZON = 12
COHORT_HORIZONS = m10.HORIZONS  # (24, 36) months, discounted from acquisition
LANDMARK_START = 3
LANDMARK_STEP = 1
MIN_TUNING_LANDMARKS = 3
PENALTY_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
"""L2 penalty ``lambda`` on the scale defined by :data:`PENALTY_CONVENTION`."""
DEFAULT_PENALTY = 1.0
TUNING_RULE = ("temporal: choose the penalty on the validation landmark, fitting only tuning "
               "landmarks whose label windows end before it; if fewer than "
               "MIN_TUNING_LANDMARKS exist, use the fixed, predeclared DEFAULT_PENALTY with no "
               "data-driven tuning (owner decision; labelled 'fixed penalty: temporal tuning "
               "infeasible')")
TENURE_BAND_EDGES = (0, 1, 2, 3, 6, 12, 24)
"""Piecewise-constant tenure basis of the discrete-time hazard (last band open-ended)."""
TENURE_BAND_MIN_EVENTS = 10
"""A tenure band with fewer training events is merged into its lower neighbour (repeatedly),
before fitting; the merged bands are reported. The basis is unpenalised, so an empty band
would otherwise have no finite estimate."""


@dataclass(frozen=True)
class PenaltyConvention:
    objective: str
    standardisation: str
    unpenalised: str
    categorical: str
    interpretation: str
    applies_to: tuple[str, ...]


PENALTY_CONVENTION = PenaltyConvention(
    objective=("minimise -sum_i loglik_i(beta) + (lambda / 2) * ||beta_penalised||^2: the "
               "TOTAL (not mean) log-likelihood over training rows, so lambda does not "
               "depend on n"),
    standardisation=("every penalised column, indicators included, is centred at its mean and "
                     "divided by its population SD (ddof 0) over the rows of the fit being made "
                     "(tuning rows when tuning, all training rows for the final fit); "
                     "validation and holdout rows use those statistics unchanged; a column "
                     "with SD 0 is dropped and reported"),
    unpenalised="intercept and the tenure-band basis",
    categorical="one-hot, reference level = the most frequent training category",
    interpretation=("lambda = 1 equals independent N(0, 1) priors on per-SD effects (MAP "
                    "estimate): weak shrinkage whose weight relative to the data falls as "
                    "training rows grow; its main role is a finite estimate under "
                    "separation"),
    applies_to=("E3", "E5 (both parts)"))


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    kind: str
    """``contractual`` or ``non_contractual``."""
    generator: str
    n_months: int
    start: str
    nominal_acquisitions_per_month: float
    evaluation_cutoffs: tuple[int, ...]
    secondary_horizon: tuple[tuple[int, int], ...]
    """``(horizon, cutoff)`` pairs evaluated in addition to the primary horizon."""
    tiers: tuple[str, ...]
    variants: tuple[str, ...]
    parameters_status: str


SCENARIOS = (
    ScenarioSpec("m10_contractual", "contractual",
                 "data.customer_economics_scenarios (M10 generator, unchanged)", 48,
                 "2022-01-01", 310.0, (24, 30), (), ("A",),
                 tuple(f"frailty_variance={v}" for v in m10.HETEROGENEITY_GRID),
                 "fixed (M10 defaults)"),
    ScenarioSpec("signals_contractual", "contractual",
                 "data.predictive_ltv_scenarios (P1): latent risk class x 3-state engagement "
                 "HMM; usage, support contacts, payment failures; involuntary churn; planted "
                 "account-closure leakage trap", 60, "2022-01-01", 310.0, (36, 42),
                 ((24, 33),), ("A", "B"), ("non_proportional_hazards",),
                 "fixed at P1: SignalsScenarioConfig defaults (SCENARIO_PARAMETER_DIGESTS)"),
    ScenarioSpec("purchases_noncontractual", "non_contractual",
                 "data.predictive_ltv_scenarios (P1): BG/NBD purchasing with Gamma-Gamma spend",
                 48, "2022-01-01", 310.0, (24, 30), (), ("A",),
                 ("pareto_nbd_dropout", "correlated_spend"),
                 "fixed at P1: PurchasesScenarioConfig defaults (SCENARIO_PARAMETER_DIGESTS)"),
)
SCENARIO_PARAMETER_DIGESTS = {
    "signals_contractual": "fa53fee9e0bbc04504aadfedaff84639358a221a2cdad178db421c6c986fc792",
    "non_proportional_hazards":
        "14a2b5b58bd7af02de49f25d75559e1f15d793210c2404227376d34b68926935",
    "purchases_noncontractual":
        "70de724902d7bc1410a8668db6fe1871132d10c9d149d93a07c95dec63fcdaee",
    "pareto_nbd_dropout": "aa60b4637a33eba597323a0a20e822e4021804b5d16e1e9de24d29ed8c119ee0",
    "correlated_spend": "728fc783f3882aae6187c2e7236a11fafab0d2ed57625b82601e96cce0cec3a2",
}
"""``data.predictive_ltv_scenarios.config_digest`` (all parameters except the seed) of the
default configuration of each new scenario and of each predeclared variant applied to it.
Every M11 run must use exactly these parameters; only the seed varies."""
SIGNAL_PARAMETERS_DECISION = (
    "Behavioural-signal parameters are kept as generated at P1 (owner decision). The P1 pilot "
    "showed a weak crude Tier B signal; P2's exact Tier A and Tier B oracles quantify the "
    "available gain; the signals are not re-tuned toward any expected PC3 result.")


def scenario(name: str) -> ScenarioSpec:
    for s in SCENARIOS:
        if s.name == name:
            return s
    raise KeyError(name)


def training_landmarks(cutoff: int, horizon: int) -> tuple[int, ...]:
    """Landmarks ``L`` whose labels ``L+1 .. L+horizon`` are completely recorded by ``cutoff``:
    ``L + horizon <= cutoff - COMPLETE_MONTH_OFFSET``."""
    last = cutoff - COMPLETE_MONTH_OFFSET - horizon
    return tuple(range(LANDMARK_START, last + 1, LANDMARK_STEP))


def validation_landmark(cutoff: int, horizon: int) -> int | None:
    """The latest training landmark (used to choose the penalty), or None if there is none."""
    t = training_landmarks(cutoff, horizon)
    return t[-1] if t else None


def tuning_landmarks(cutoff: int, horizon: int) -> tuple[int, ...]:
    """Landmarks fitted while tuning: their labels must be complete before the validation
    landmark, ``L + horizon <= L_v - COMPLETE_MONTH_OFFSET`` (amendment 2)."""
    lv = validation_landmark(cutoff, horizon)
    if lv is None:
        return ()
    return tuple(L for L in training_landmarks(cutoff, horizon)
                 if L + horizon <= lv - COMPLETE_MONTH_OFFSET)


def temporal_tuning_feasible(cutoff: int, horizon: int) -> bool:
    return len(tuning_landmarks(cutoff, horizon)) >= MIN_TUNING_LANDMARKS


# --- information tiers and point-in-time features ------------------------------------------
STATIC = "static"
RECORD_TIME = "recorded_at"
FORBIDDEN_SOURCES = ("account_closure_summary",)
"""Planted leakage trap: indexed by the final usage month but recorded at closure. No feature
may read it; the event-time join counterexample uses it deliberately."""


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: str
    tier: str
    source: str
    time_column: str
    """``recorded_at`` (point-in-time via ``data.point_in_time.as_of``) or ``static`` (known at
    acquisition)."""
    usable_through_offset: int
    """Data for month ``m`` is used at cutoff ``C`` only if ``m <= C - offset``."""
    definition: str


FEATURES = (
    FeatureSpec("segment", "contractual", "A", "customers", STATIC, 0, "plan segment"),
    FeatureSpec("channel", "contractual", "A", "customers", STATIC, 0, "acquisition channel"),
    FeatureSpec("tenure_months", "contractual", "A", "customers", STATIC, 0,
                "C - acquisition month"),
    FeatureSpec("acquisition_cohort", "contractual", "A", "customers", STATIC, 0,
                "acquisition calendar month"),
    FeatureSpec("promo_months_remaining", "contractual", "A", "customers", STATIC, 0,
                "months of the (known) promotion policy still to run at C"),
    FeatureSpec("paid_months", "contractual", "A", "cancellations", RECORD_TIME, 0,
                "paid months at C (M10 status rule)"),
    FeatureSpec("mean_contribution", "contractual", "A", "transactions", RECORD_TIME,
                COMPLETE_MONTH_OFFSET, "mean contribution per completely recorded month"),
    FeatureSpec("last_contribution", "contractual", "A", "transactions", RECORD_TIME,
                COMPLETE_MONTH_OFFSET, "contribution of the last completely recorded month"),
    FeatureSpec("usage_mean_3m", "contractual", "B", "signals", RECORD_TIME,
                SIGNAL_COMPLETE_OFFSET, "mean usage over the last 3 usable months"),
    FeatureSpec("usage_change_3m", "contractual", "B", "signals", RECORD_TIME,
                SIGNAL_COMPLETE_OFFSET, "log(1+usage, last 3) - log(1+usage, previous 3)"),
    FeatureSpec("support_contacts_3m", "contractual", "B", "signals", RECORD_TIME,
                SIGNAL_COMPLETE_OFFSET, "support contacts over the last 3 usable months"),
    FeatureSpec("payment_failures_6m", "contractual", "B", "signals", RECORD_TIME,
                SIGNAL_COMPLETE_OFFSET, "failed payment attempts over the last 6 usable months"),
    FeatureSpec("months_since_payment_failure", "contractual", "B", "signals", RECORD_TIME,
                SIGNAL_COMPLETE_OFFSET, "months since the last usable failed payment (capped)"),
    FeatureSpec("segment", "non_contractual", "A", "customers", STATIC, 0, "segment"),
    FeatureSpec("channel", "non_contractual", "A", "customers", STATIC, 0, "channel"),
    FeatureSpec("frequency", "non_contractual", "A", "purchases", RECORD_TIME,
                COMPLETE_MONTH_OFFSET, "repeat purchases x"),
    FeatureSpec("recency", "non_contractual", "A", "purchases", RECORD_TIME,
                COMPLETE_MONTH_OFFSET, "time of last purchase t_x"),
    FeatureSpec("age", "non_contractual", "A", "customers", STATIC, 0,
                "observation length T to the end of the last complete month"),
    FeatureSpec("mean_spend", "non_contractual", "A", "purchases", RECORD_TIME,
                COMPLETE_MONTH_OFFSET, "mean spend of repeat purchases"),
    FeatureSpec("purchases_6m", "non_contractual", "A", "purchases", RECORD_TIME,
                COMPLETE_MONTH_OFFSET, "purchases in the last 6 complete months"),
)


def features(kind: str, tier: str) -> tuple[FeatureSpec, ...]:
    """Features available to an information tier (Tier B includes Tier A)."""
    allowed = {"A": ("A",), "B": ("A", "B")}[tier]
    return tuple(f for f in FEATURES if f.kind == kind and f.tier in allowed)


# --- models, targets, comparisons ------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    role: str
    """``baseline``, ``essential``, ``secondary`` or ``reference``."""
    output: str
    """``point``, ``distribution``, ``survival``, ``mean_interval`` or ``reference``."""
    kinds: tuple[str, ...]
    tiers: tuple[str, ...]
    complexity_rank: int | None
    """Parsimony order for selection (lower is simpler); None = not a selection candidate."""


M10_CUSTOMER_METHODS = ("run rate (no churn)", "historical average (no churn)",
                        "constant churn, pooled", "constant churn by cell",
                        "Kaplan–Meier by cell")
"""Identical to ``customer_value.backtest.CUSTOMER_METHODS`` (checked by a test)."""
M10_COHORT_HEURISTICS = ("H1", "H2", "H3", "H4")

MODELS = (
    ModelSpec("B1", "M10 baselines, unchanged", "baseline", "point", ("contractual",), ("A",), 0),
    ModelSpec("B2", "RFM heuristics", "baseline", "point", ("non_contractual",), ("A",), 0),
    ModelSpec("E1", "shifted-beta-geometric", "essential", "distribution", ("contractual",),
              ("A",), 1),
    ModelSpec("E2", "exponential-Gamma frailty (discrete Lomax)", "essential", "distribution",
              ("contractual",), ("A",), 1),
    ModelSpec("E3", "discrete-time landmark hazard x contribution by age", "essential",
              "survival", ("contractual",), ("A", "B"), 2),
    ModelSpec("E4", "BG/NBD + Gamma-Gamma, fitted per segment x channel cell", "essential",
              "distribution", ("non_contractual",), ("A",), 1),
    ModelSpec("E5", "two-part penalised GLM", "essential", "mean_interval",
              ("contractual", "non_contractual"), ("A", "B"), 3),
    ModelSpec("S1", "Cox proportional hazards (landmark covariates, Efron ties)", "secondary",
              "survival", ("contractual",), ("A", "B"), None),
    ModelSpec("S2", "split conformal intervals for E5", "secondary", "mean_interval",
              ("contractual", "non_contractual"), ("A", "B"), None),
    ModelSpec("S3", "IPCW censored-evaluation demonstration", "secondary", "survival",
              ("contractual",), ("A",), None),
    ModelSpec("ORACLE", "observable-information oracle (per tier)", "reference", "reference",
              ("contractual", "non_contractual"), ("A", "B"), None),
    ModelSpec("LATENT", "latent-truth expectation (decomposition only)", "reference",
              "reference", ("contractual", "non_contractual"), ("A", "B"), None),
)
SECONDARY_STUDIES = ("S4: misspecification variants (Pareto/NBD dropout, correlated spend)",
                     "S5: non-proportional-hazards switch")
E4_FIT_UNIT = ("segment x channel cell: separate BG/NBD (r, alpha, a, b) and Gamma-Gamma (p, q, "
               "gamma) fits in each of the 12 cells, the unit within which the purchases "
               "generator is BG/NBD + Gamma-Gamma (purchase rate and spend vary by segment, "
               "dropout by channel); cell fit status follows STATUS_RULES, with the pooled fit "
               "as the only allowed fallback")
E4_POOLED_REFERENCE = ("a single pooled fit is reported as a labelled MISSPECIFIED REFERENCE "
                       "(pooled across segments, frequency and spend are correlated); it is "
                       "not a selection candidate")


def model(model_id: str) -> ModelSpec:
    for m in MODELS:
        if m.id == model_id:
            return m
    raise KeyError(model_id)


@dataclass(frozen=True)
class TargetSpec:
    id: str
    population: str
    definition: str
    primary: bool


TARGETS = (
    TargetSpec("existing_undiscounted_12", "active at C with >= 1 complete month (M10 rule)",
               "sum of contribution over billing months C+1..C+12 (identical to M10)", True),
    TargetSpec("existing_discounted_12", "active at C with >= 1 complete month (M10 rule)",
               "sum over k = 1..12 of d^k x contribution of month C+k", False),
    TargetSpec("cohort_discounted_24", "acquired customers, complete windows (M10 rule)",
               "M10 cohort-matched discounted value, ages 0..23", False),
    TargetSpec("cohort_discounted_36", "acquired customers, complete windows (M10 rule)",
               "M10 cohort-matched discounted value, ages 0..35", False),
    TargetSpec("cohort_expected_lifetime", "acquisition cohorts (truth comparison only)",
               "expected discounted lifetime value; 'undefined' if the sum diverges", False),
)
PRIMARY_LOSS = "squared_error"
REPORTED_METRICS = (
    "realised RMSE", "excess RMSE vs oracle", "skill vs baseline", "MAE (median-oriented)",
    "calibration-in-the-large", "calibration slope and intercept (OLS, HC)",
    "decile reliability", "Spearman", "top-decile capture", "normalised Gini",
    "Harrell C", "Uno C (IPCW)", "IPCW Brier and integrated Brier", "D-calibration",
    "log score (counts)", "CRPS", "randomised PIT", "interval coverage")


@dataclass(frozen=True)
class ComparisonSpec:
    id: str
    scenario: str
    tier: str
    target: str
    model_a: str
    model_b: str
    statistic: str


PRIMARY_COMPARISONS = (
    ComparisonSpec("PC1", "m10_contractual", "A", "existing_undiscounted_12", "E2", "B1",
                   "paired MSE difference vs 'Kaplan–Meier by cell'"),
    ComparisonSpec("PC2", "m10_contractual", "A", "cohort_expected_lifetime", "E2", "B1",
                   "absolute relative bias vs H4 across the heterogeneity grid (E1 reported "
                   "alongside)"),
    ComparisonSpec("PC3", "signals_contractual", "B", "existing_undiscounted_12", "E3", "E3",
                   "paired MSE difference, E3 Tier B vs the best Tier A model"),
    ComparisonSpec("PC4", "purchases_noncontractual", "A", "existing_undiscounted_12", "E4",
                   "B2", "paired MSE difference"),
    ComparisonSpec("PC5", "signals_contractual", "B", "existing_undiscounted_12", "E5", "E3",
                   "paired MSE difference"),
)
FAMILY_ALPHA = 0.05  # Holm across PRIMARY_COMPARISONS; everything else is descriptive


# --- model selection, fit status, calibration reporting ---------------------------------------
NO_QUALIFIED_MODEL = "NO QUALIFIED MODEL"
REFERENCE_FALLBACK_MODEL = "E2"
REFERENCE_FALLBACK_LABEL = "REFERENCE FALLBACK — not a selection"


@dataclass(frozen=True)
class SelectionRule:
    assessed_on: str
    min_ok_cell_fraction: float
    calibration_in_the_large_tolerance: float
    slope_bounds: tuple[float, float]
    paired_interval_level: float
    rule: str


SELECTION = SelectionRule(
    assessed_on="validation landmark of each evaluation cutoff (never the holdout)",
    min_ok_cell_fraction=0.90,
    calibration_in_the_large_tolerance=0.05,
    slope_bounds=(0.85, 1.15),
    paired_interval_level=0.95,
    rule=("qualified = status ok/fallback in >= min_ok_cell_fraction of cells, "
          "|sum(pred)/sum(realised) - 1| <= tolerance, slope within bounds; selected = the "
          "simplest qualified model that no qualified model beats significantly (paired "
          "customer-bootstrap interval of the MSE difference entirely below 0); ties in "
          "complexity -> lower MSE; none qualified -> NO QUALIFIED MODEL"))


@dataclass(frozen=True)
class StatusRules:
    gradient_tolerance: float
    relative_change_tolerance: float
    log_parameter_bound: float
    n_starts: int
    start_agreement_tolerance: float
    min_events: int
    min_events_per_parameter: float
    min_at_risk: int
    fallback: str


STATUS_RULES = StatusRules(
    gradient_tolerance=1e-6, relative_change_tolerance=1e-9, log_parameter_bound=10.0,
    n_starts=5, start_agreement_tolerance=1e-4, min_events=m10.MIN_CHURN_EVENTS,
    min_events_per_parameter=10.0, min_at_risk=m10.MIN_AT_RISK,
    fallback="pooled fit of the same model, labelled 'fallback' and counted; nothing else")
FIT_STATUSES = ("ok", "fallback", "unstable", "failed", "undefined")


@dataclass(frozen=True)
class CalibrationReporting:
    reliability_bins: int
    d_calibration_bins: int
    interval_levels: tuple[float, ...]
    survival_horizon: int
    coverage_test: str
    recalibration: str


CALIBRATION = CalibrationReporting(
    reliability_bins=10, d_calibration_bins=10, interval_levels=(0.80, 0.95),
    survival_horizon=PRIMARY_HORIZON,
    coverage_test="Clopper-Pearson interval and exact binomial test; Holm within each family",
    recalibration="none in M11")
COVERAGE_REPORTING_RULE = (
    "Coverage and calibration levels are scientific reporting criteria, not acceptance tests: "
    "each is reported with its Monte Carlo SE, Clopper-Pearson interval and the label from "
    "precision.classify_coverage ('consistent with nominal' / 'below nominal' / 'above "
    "nominal'). No automated test asserts a coverage or calibration level; a result below "
    "nominal is reported as a finding and never triggers a post-hoc change of method, seed, "
    "scale or replication count. Distinct from correctness gates (ORACLE_GATE, parameter "
    "recovery in correctly specified settings), which test identities that hold by theory and "
    "indicate a bug when violated.")


# --- customer-clustered inference (amendment 2) and the oracle gate (amendment 1) ------------
@dataclass(frozen=True)
class ClusteredInference:
    unit: str
    resampling: str
    sandwich: str
    metric_bootstrap_resamples: int
    refit_bootstrap_resamples: int
    conformal_split: str
    conformal_calibration_fraction: float


CLUSTERED_INFERENCE = ClusteredInference(
    unit="customer_id",
    resampling="resample customers with replacement, carrying all their landmark and cutoff "
               "rows; never person-months or landmark rows",
    sandwich="CR1 (customer clusters) for penalised-model standard errors",
    metric_bootstrap_resamples=m10.BOOTSTRAP_RESAMPLES,
    refit_bootstrap_resamples=200,
    conformal_split="split by customer; one landmark per customer in the calibration set",
    conformal_calibration_fraction=0.5)


@dataclass(frozen=True)
class OracleGate:
    statement: str
    consistency_z: float
    positive_interval_level: float


ORACLE_GATE = OracleGate(
    statement=("population: MSE_A - MSE_B = E[(mu_B - mu_A)^2] >= 0. Per seed: report D = "
               "mean[(R - mu_A)^2 - (R - mu_B)^2] with a customer-clustered paired bootstrap "
               "interval (a negative D on one seed is reported, not a failure). Across "
               "repeated draws: mean(D - mean[(mu_B - mu_A)^2]) consistent with 0 (|z| < "
               "consistency_z) and the mean of D has a positive_interval_level interval above "
               "0"),
    consistency_z=3.0, positive_interval_level=0.95)


# --- precision planning and budgets ----------------------------------------------------------
PLANNING_ALPHA = FAMILY_ALPHA / len(PRIMARY_COMPARISONS)  # Holm's most stringent step
PLANNING_POWER = 0.80
TARGET_RELATIVE_MDE = 0.02
"""Plan requirement: a model whose excess MSE is 2% of the noise floor ``sigma^2`` must be
detectable against a near-oracle model, per evaluation cutoff, with the conservative
(customer-level) effective sample size."""
PRECISION_PROXY_TENURE_EDGES = (0, 3, 6, 12, 24)
"""Cells for the pilot ``sigma^2`` proxy: segment x channel x tenure band (a conservative upper
bound on E[Var(R | F_C)], since the cell is F_C-measurable)."""


@dataclass(frozen=True)
class RuntimeBudgets:
    new_tests_target_s: float
    new_tests_ceiling_s: float
    single_test_ceiling_s: float
    test_memory_gb: float
    evidence_target_min: float
    evidence_ceiling_min: float
    evidence_memory_gb: float


BUDGETS = RuntimeBudgets(new_tests_target_s=90.0, new_tests_ceiling_s=150.0,
                         single_test_ceiling_s=10.0, test_memory_gb=1.5,
                         evidence_target_min=20.0, evidence_ceiling_min=45.0,
                         evidence_memory_gb=4.0)

# --- stage gate ---------------------------------------------------------------------------------
PENDING_BEFORE_MAIN = (
    "P2-P6 (deferred from P1): preflight confirmation that COVERAGE_REPLICATIONS (1,000) fits "
    "the budget, with the repeated-sampling scenario scale; otherwise a declared count >= "
    "MIN_COVERAGE_REPLICATIONS (needs the measured oracle and model-fit costs)",
    "P2: oracle Monte Carlo validation sizes and quadrature settings",
)
PILOT_FINDINGS_SCOPE = (
    "P0 pilot findings use the M10 generator only. They are final for m10_contractual and were "
    "PROVISIONAL for signals_contractual and purchases_noncontractual (60-month runs of the M10 "
    "generator were a size proxy); for those two scenarios they are superseded by the P1 "
    "scenario plan (docs/evidence/m11-p1-scenario-plan.md) at the pinned parameters.")


@dataclass(frozen=True)
class Amendment:
    id: str
    stage: str
    date: str
    before_main_seed: bool
    change: str
    reason: str


PROTOCOL_AMENDMENTS = (
    Amendment("A1", "P1", "2026-11-17", True,
              "pinned the generator parameters of signals_contractual, purchases_noncontractual "
              "and the three predeclared variants (SCENARIO_PARAMETER_DIGESTS); scenario "
              "parameters_status 'pending P1' -> 'fixed at P1'",
              "P1 generators implemented; closes pending item 'P1: numerical generator "
              "parameters'"),
    Amendment("A2", "P1", "2026-11-17", True,
              "E4 is fitted per segment x channel cell (E4_FIT_UNIT); a pooled fit is reported "
              "only as a misspecified reference (E4_POOLED_REFERENCE)",
              "P1 finding: rates vary by segment and channel, so the generator is BG/NBD + "
              "Gamma-Gamma only within a cell; pooled, frequency and spend are correlated "
              "(pilot-independent development-seed check, Spearman -0.056)"),
    Amendment("A3", "P1", "2026-11-17", True,
              "behavioural-signal parameters kept unchanged (SIGNAL_PARAMETERS_DECISION)",
              "owner decision after the P1 pilot showed a weak crude Tier B signal; avoids "
              "tuning the data toward PC3"),
    Amendment("A4", "P1", "2026-11-17", True,
              "closed pending items 'P1: generator parameters' and 'P1: pilot of the new "
              "scenarios'; kept the repeated-sampling preflight (relabelled 'P2-P6, deferred "
              "from P1') and the P2 oracle item open; PILOT_FINDINGS_SCOPE points to the P1 "
              "scenario plan; PROTOCOL_STAGE 'P0' -> 'P1-amended'",
              "P1 pilot completed (docs/evidence/m11-p1-scenario-plan.md); the preflight needs "
              "oracle and model-fit costs that do not exist before P2-P6"),
)
"""Every protocol change after the P0 commit (82e3ebd), all made before any main-seed
evaluation; the main seed has never been authorised."""


class ProtocolNotReady(RuntimeError):
    """A reserved seed was requested before the protocol allows it."""


def authorise_seed(purpose: str, *, confirm_single_final_run: bool = False) -> int:
    """The only sanctioned way to obtain a reserved seed.

    ``pilot`` is always allowed (planning). ``main`` requires no pending protocol items.
    ``fresh`` additionally requires ``confirm_single_final_run=True``.
    """
    if purpose == "pilot":
        return PILOT_SEED
    if purpose not in ("main", "fresh"):
        raise ValueError("purpose must be 'pilot', 'main' or 'fresh'")
    if PENDING_BEFORE_MAIN:
        raise ProtocolNotReady(f"{len(PENDING_BEFORE_MAIN)} protocol items are pending; "
                               f"the {purpose} seed is refused")
    if purpose == "main":
        return MAIN_SEED
    if not confirm_single_final_run:
        raise ProtocolNotReady("the fresh seed is evaluated once, last; confirm explicitly")
    return FRESH_SEED


# --- integrity ------------------------------------------------------------------------------------
def _plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in sorted(value.items())}
    if isinstance(value, tuple | list):
        return [_plain(v) for v in value]
    return value


def protocol_manifest() -> dict[str, Any]:
    """Every protocol value (upper-case module constants), as plain JSON-compatible data."""
    out: dict[str, Any] = {}
    for name, value in sorted(globals().items()):
        if name.isupper() and not name.startswith("_"):
            out[name] = _plain(value)
    return out


def protocol_digest() -> str:
    """SHA-256 of the canonical JSON manifest (insensitive to formatting and comments)."""
    blob = json.dumps(protocol_manifest(), sort_keys=True, ensure_ascii=True,
                      separators=(",", ":"))
    return hashlib.sha256(blob.encode("ascii")).hexdigest()


def protocol_code_digest() -> str:
    """SHA-256 over the exact bytes (LF line endings) of :data:`PROTOCOL_CODE_FILES`."""
    here = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for name in PROTOCOL_CODE_FILES:
        h.update(name.encode("ascii") + b"\0")
        h.update((here / name).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def validate_protocol() -> list[str]:
    """Internal-consistency checks; returns the problems found (empty when consistent)."""
    problems: list[str] = []
    reserved = (MAIN_SEED, PILOT_SEED, FRESH_SEED)
    if len(set(reserved)) != 3:
        problems.append("reserved seeds are not distinct")
    for s in reserved:
        if s in OTHER_PROTOCOL_SEEDS:
            problems.append(f"seed {s} is reserved by an earlier protocol")
        if s <= DEVELOPMENT_SEED_MAX:
            problems.append(f"seed {s} lies in the development range")
        if COVERAGE_SEED_START <= s < COVERAGE_SEED_START + COVERAGE_REPLICATIONS:
            problems.append(f"seed {s} lies in the coverage block")
    blocks = [(COVERAGE_SEED_START, COVERAGE_SEED_START + COVERAGE_REPLICATIONS),
              (REFIT_STUDY_SEED_START, REFIT_STUDY_SEED_START + REFIT_STUDY_REPLICATIONS)]
    if blocks[0][0] < blocks[1][1] and blocks[1][0] < blocks[0][1]:
        problems.append("coverage and refit-study blocks overlap")
    for lo, hi in blocks:
        if lo <= DEVELOPMENT_SEED_MAX:
            problems.append("a seed block overlaps the development range")
        for a, b in OTHER_COVERAGE_BLOCKS:
            if lo < b and a < hi:
                problems.append("a seed block overlaps an earlier protocol's block")
        for s in reserved:
            if lo <= s < hi:
                problems.append(f"seed {s} lies in a replication block")
    if not MIN_COVERAGE_REPLICATIONS <= COVERAGE_REPLICATIONS:
        problems.append("COVERAGE_REPLICATIONS below the declared minimum")
    if set(AUTHORISE_ALLOWED_FILES) != {"pilot", "main", "fresh"}:
        problems.append("AUTHORISE_ALLOWED_FILES must cover pilot, main and fresh")
    for sc in SCENARIOS:
        if sc.kind not in ("contractual", "non_contractual"):
            problems.append(f"{sc.name}: unknown kind")
        for c in sc.evaluation_cutoffs:
            if c + PRIMARY_HORIZON > sc.n_months - 1:
                problems.append(f"{sc.name}: cutoff {c} has no complete {PRIMARY_HORIZON}-month "
                                "window")
            if not training_landmarks(c, PRIMARY_HORIZON):
                problems.append(f"{sc.name}: cutoff {c} has no training landmark")
        for h, c in sc.secondary_horizon:
            if c + h > sc.n_months - 1:
                problems.append(f"{sc.name}: cutoff {c} has no complete {h}-month window")
        for t in sc.tiers:
            if t not in ("A", "B"):
                problems.append(f"{sc.name}: unknown tier {t}")
            if not features(sc.kind, t):
                problems.append(f"{sc.name}: tier {t} has no features")
    keys = [(f.kind, f.name) for f in FEATURES]
    if len(keys) != len(set(keys)):
        problems.append("duplicate feature definitions")
    for f in FEATURES:
        if f.source in FORBIDDEN_SOURCES:
            problems.append(f"feature {f.name} reads a forbidden source")
        if f.time_column not in (STATIC, RECORD_TIME):
            problems.append(f"feature {f.name}: time column must be recorded_at or static")
        if f.time_column == STATIC and f.source != "customers":
            problems.append(f"feature {f.name}: only customer attributes may be static")
        if f.source == "signals" and f.usable_through_offset < SIGNAL_COMPLETE_OFFSET:
            problems.append(f"feature {f.name}: signals used before their recording lag")
        if f.source in ("transactions", "purchases") and \
                f.usable_through_offset < COMPLETE_MONTH_OFFSET:
            problems.append(f"feature {f.name}: incomplete months used")
    ids = [m.id for m in MODELS]
    if len(ids) != len(set(ids)):
        problems.append("duplicate model ids")
    for m in MODELS:
        selectable = m.role in ("baseline", "essential")
        if selectable != (m.complexity_rank is not None):
            problems.append(f"model {m.id}: only baselines and essential models are candidates")
    if model(REFERENCE_FALLBACK_MODEL).role != "essential":
        problems.append("the reference fallback must be an essential model")
    names = {s.name for s in SCENARIOS}
    targets = {t.id for t in TARGETS}
    for pc in PRIMARY_COMPARISONS:
        if pc.scenario not in names or pc.target not in targets:
            problems.append(f"{pc.id}: unknown scenario or target")
        elif pc.tier not in scenario(pc.scenario).tiers:
            problems.append(f"{pc.id}: tier not available in the scenario")
        for mid in (pc.model_a, pc.model_b):
            if mid not in ids:
                problems.append(f"{pc.id}: unknown model {mid}")
    if sum(t.primary for t in TARGETS) != 1:
        problems.append("exactly one primary target is required")
    lo_s, hi_s = SELECTION.slope_bounds
    if not (0 < lo_s < 1 < hi_s) or not 0 < SELECTION.calibration_in_the_large_tolerance < 1:
        problems.append("invalid selection bounds")
    if not 0 < SELECTION.min_ok_cell_fraction <= 1:
        problems.append("invalid minimum ok-cell fraction")
    pinned = {"signals_contractual", "purchases_noncontractual"}
    pinned |= {v for name in pinned for v in scenario(name).variants}
    if set(SCENARIO_PARAMETER_DIGESTS) != pinned:
        problems.append("SCENARIO_PARAMETER_DIGESTS must cover the new scenarios and variants")
    if any(len(d) != 64 for d in SCENARIO_PARAMETER_DIGESTS.values()):
        problems.append("parameter digests must be SHA-256 hex strings")
    if any("pending" in sc.parameters_status for sc in SCENARIOS):
        problems.append("a scenario's parameters are still pending")
    ids_a = [a.id for a in PROTOCOL_AMENDMENTS]
    if len(ids_a) != len(set(ids_a)) or not all(a.before_main_seed for a in PROTOCOL_AMENDMENTS):
        problems.append("amendments must have unique ids and precede the main seed")
    if DEFAULT_PENALTY not in PENALTY_GRID:
        problems.append("the default penalty must be on the grid")
    if not 0 < PLANNING_ALPHA < FAMILY_ALPHA or not 0.5 < PLANNING_POWER < 1:
        problems.append("invalid planning error rates")
    return problems
