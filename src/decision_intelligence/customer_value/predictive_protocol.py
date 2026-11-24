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

**Stage.** This is the P0 protocol as amended through P8 (:data:`PROTOCOL_STAGE`; A12, drafted
at the P8 preflight, fixes the P8 design). Items that must be closed before any main-seed
evaluation are listed in :data:`PENDING_BEFORE_MAIN`. A13 closed the P8 dry-run item.
A16 configured the candidate attempt under an execution hold. A17 records the owner's
confirmation of continued outcome blindness. A18 replaces the static launch hold with
verification of separate owner approval for the exact release and designated host. The
owner issues approval after green CI and a passing immediate host preflight.
:func:`authorise_seed` refuses the main and fresh seeds while any item or outcome-blindness
confirmation is pending. Amendments and their timing are recorded in
:data:`PROTOCOL_AMENDMENTS`.

**Integrity.** :func:`protocol_digest` hashes every protocol value; :func:`protocol_code_digest`
hashes the exact bytes of this module and of the decision logic in :mod:`.model_selection` and
:mod:`.precision` and :mod:`.launch_approval`. Tests pin both, so any change is a visible,
reviewable diff that must be
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
from datetime import date
from pathlib import Path
from typing import Any

from decision_intelligence.customer_value import p8_candidate
from decision_intelligence.customer_value import protocol as m10
from decision_intelligence.customer_value.launch_approval import LaunchApprovalError

PROTOCOL_ID = "M11-predictive-ltv"
PROTOCOL_STAGE = "P8-amended"

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
PROTOCOL_CODE_FILES = (
    "predictive_protocol.py", "model_selection.py", "precision.py", "launch_approval.py",
)
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
E3_EXCLUDED_COVARIATES = ("acquisition_cohort", "tenure_months", "paid_months")
"""Amendment A8: registry features E3 does not use as linear covariates (the registry itself is
unchanged; S1, E5 and every other model are unaffected). Frozen at the landmark and entered
linearly, they were extrapolated far outside the training landmarks' range of cohorts and
tenures; tenure enters E3 only through its unpenalised tenure-band basis on the decision age."""
E3_TUNING_LOSS = ("squared error of the window (12-month at the primary horizon) value at the "
                  "validation landmark; ties to the larger penalty")
"""Amendment A8 (clarifies TUNING_RULE, which did not name the loss)."""
E3_SELECTION_ELIGIBILITY = (
    "E3 is a selection candidate at an evaluation cutoff only if a validation fit exists: at "
    "least one tuning landmark whose labels are completely recorded before the validation "
    "landmark. Otherwise E3 is 'not selection-eligible' there; no censored validation windows")
"""Amendment A8 (owner decision): no relaxation of the complete-label rule."""
DESIGN_DUPLICATE_RULE = (
    "after standardisation on the rows of the fit, a column identical (max |difference| <= "
    "1e-10) to an earlier column in design order (registry order, then one-hot indicators by "
    "level) is dropped and reported, like an SD-0 column; the feature registry is unchanged")
"""Amendment A8 (owner decision): deterministic removal of exact duplicate design columns (for
example ``paid_months = tenure_months + 1`` for active customers)."""
E5_TUNING_LOSS = (
    "E5: one penalty shared by both parts, chosen under TUNING_RULE by squared error of the "
    "window (12-month at the primary horizon) value P(R > 0) x E[R | R > 0] at the "
    "validation landmark; ties to the larger penalty; if no tuning fit has both parts 'ok', "
    "DEFAULT_PENALTY, labelled 'fixed penalty: no tuning fit was ok'")
"""Amendment A9 (owner decision): E5's tuning loss, shared penalty, tie-break and no-ok-fit
fallback (TUNING_RULE names no loss; E3_TUNING_LOSS is E3's only)."""
E5_SELECTION_ELIGIBILITY = (
    "E5 is a selection candidate at an evaluation cutoff only if a validation fit exists: at "
    "least one tuning landmark whose labels are completely recorded before the validation "
    "landmark. Otherwise E5 is 'not selection-eligible' there; no censored validation windows")
"""Amendment A9 (owner decision): E5's selection eligibility (E3_SELECTION_ELIGIBILITY is
E3's only)."""
E3_NO_OK_FIT_RULE = (
    "E3: if temporal tuning is feasible but no tuning fit is 'ok', DEFAULT_PENALTY, labelled "
    "'fixed penalty: no tuning fit was ok'; the resulting fit is still subject to the SELECTION "
    "status gate")
"""Amendment A10 (owner decision): authorises E3's existing no-ok-fit fallback (P4 code,
unchanged); E3_TUNING_LOSS keeps its A8 wording."""


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
REFERENCE_FALLBACK_SCOPE = (
    "the reference fallback applies only to scenarios whose kind is among "
    "model(REFERENCE_FALLBACK_MODEL).kinds (contractual); for a non-contractual scenario a "
    "NO QUALIFIED MODEL result is reported with no reference model")
"""Amendment A10 (owner decision, P5 disposition): E2 is never shown for a non-contractual
scenario."""
TIER_A_IN_TIER_B = (
    "a Tier A model is a selection candidate in a Tier B selection of a scenario whose kind it "
    "supports: it uses only Tier A information, which Tier B includes")
"""Amendment A10 (owner decision)."""
B1_DEFINITION = (
    "B1 at the customer level (existing_undiscounted_12; the selection candidate and PC1's "
    "comparator) is M10's 'Kaplan–Meier by cell' (backtest.customer_projections, unchanged) "
    "with W = PRIMARY_HORIZON: run rate x sum_{t=1..W} S(a0 + t) / S(a0), where the run rate "
    "is the customer's mean contribution over its last up to 3 completely recorded months "
    "(months <= complete_through(C)), a0 = C - acquisition month, and S is the paid-month "
    "Kaplan–Meier curve of the customer's segment x channel cell at C, truncated where fewer "
    "than MIN_AT_RISK customers remain at risk and extended by M10's geometric tail over the "
    "last KM_TAIL_MONTHS. Population: the contractual training_stack population (active at C, "
    "acquired by complete_through(C)). Cell status (M10's thresholds): 'undefined' if the cell "
    "has no usable curve (fewer than 2 ages with MIN_AT_RISK at risk); 'unstable' if fewer "
    "than MIN_CHURN_EVENTS churn events; otherwise 'ok'. No fallback (M10 unchanged): a cell "
    "that is not 'ok' counts against min_ok_cell_fraction, and a customer without a finite "
    "prediction is 'undefined' and prevents B1 from qualifying at that cutoff. Nothing is "
    "tuned. Selection inputs as E1/E2: computed from the tables as recorded by C, observed at "
    "the validation landmark L_v, scored on L_v's completely recorded labels; "
    "selection-eligible wherever a validation landmark exists. In PC2 B1 is H4 (cohort "
    "level), never a selection candidate. M10's other customer methods (run rate, historical "
    "average, constant churn pooled and by cell) and H1-H3 are reported descriptively only")
"""Amendment A10 (owner decision): B1's single selection candidate."""
B2_DEFINITION = (
    "B2: for every customer acquired by T_C = complete_through(C) (E4's population), prediction "
    "of existing_undiscounted_12 = f x s x k, where f = the number of the customer's purchases "
    "(the acquisition purchase included) in the complete months max(acquisition month, "
    "T_C - PRIMARY_HORIZON + 1) .. T_C, carried forward unchanged to months C+1 .. C+12 (no "
    "churn, no annualisation); s = the customer's mean revenue per purchase over every "
    "completely recorded purchase (months <= T_C, the acquisition purchase included); k = the "
    "business's recorded contribution-to-revenue ratio over completely recorded purchases at C "
    "(E4's conversion). Nothing is fitted or tuned; status 'ok' where the prediction is finite, "
    "otherwise 'undefined'. Selection inputs: the same rule with the tables as recorded by the "
    "validation landmark L_v, scored on L_v's completely recorded labels (as E4); B2 is "
    "selection-eligible wherever a validation landmark exists. No other target")
"""Amendment A10 (owner decision): the RFM heuristic named in the approved plan."""
CONTRACTUAL_OUTCOME_NOISE = (
    "Contractual predictive intervals (E1, E2, E3): each simulated paid month's contribution is "
    "the ContributionModel expected contribution plus a residual drawn with replacement, "
    "independently per month, from the residuals (recorded contribution minus that mean) of "
    "completely recorded months (billing_month <= complete_through(C)) in the source group "
    "ContributionModel used for that mean (cell x promotion state, else segment x promotion "
    "state, else promotion state). Nothing is estimated from validation or holdout outcomes and "
    "no interval is rescaled (CALIBRATION.recalibration 'none in M11' stands). Intervals "
    "without the residual are kept and reported as the pre-A11 construction")
"""Amendment A11 (owner decision): completes the plan's 'parameter + outcome uncertainty' for
the contractual models (contribution noise was omitted)."""


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
class BoundaryException:
    model: str
    parameter: str
    label: str
    statement: str
    max_score_per_customer: float
    log_likelihood_tolerance: float
    log_rate_tolerance: float


E2_GEOMETRIC_BOUNDARY = BoundaryException(
    model="E2", parameter="v", label="v=0 boundary (geometric)",
    statement=(
        "An E2 fit (cell or pooled) is replaced by its v = 0 limit, the geometric model, only "
        "if ALL hold: (a) its status is 'unstable' and its reasons are exactly 'parameter at "
        "the log-scale bound', optionally with 'Hessian not positive definite'; (b) log v < "
        "-log_parameter_bound and |log b| <= log_parameter_bound; (c) it is converged with a "
        "finite likelihood; (d) the geometric MLE b_geo = -log(1 - events / decisions) agrees "
        "with the E2 b within log_rate_tolerance (log scale); (e) the geometric "
        "log-likelihood is >= the E2 log-likelihood - log_likelihood_tolerance x "
        "max(1, |loglik|); (f) the one-sided score for v at (b_geo, v = 0) per customer is <= "
        "max_score_per_customer (v = 0 is a boundary maximum); (g) the one-parameter "
        "geometric fit is 'ok' under STATUS_RULES. The fit is then 'ok' with v = 0, labelled "
        "and counted. Every other status and fallback rule is unchanged: E1 boundaries, v at "
        "the upper bound, b at a bound, any other reason and any failed check keep the "
        "original status and the pooled fallback."),
    max_score_per_customer=STATUS_RULES.gradient_tolerance,
    log_likelihood_tolerance=STATUS_RULES.relative_change_tolerance,
    log_rate_tolerance=1e-4)
"""Amendment A6 (P3): the correctly specified homogeneous-churn case (v = 0) lies on E2's
parameter boundary; without this exception the frozen boundary rule replaced a correct cell
fit with a miscalibrated pooled fit (P3 finding 2)."""
E1_GEOMETRIC_BOUNDARY = BoundaryException(
    model="E1", parameter="phi = 1 / (alpha + beta)",
    label="alpha + beta -> infinity boundary (geometric)",
    statement=(
        "E1's geometric limit is alpha, beta -> infinity with mu = alpha / (alpha + beta) "
        "fixed: Beta(alpha, beta) collapses to mu and S(a) -> (1 - mu)^a; in (mu, phi = "
        "1 / (alpha + beta)) it is the closed boundary phi = 0. An E1 fit (cell or pooled) is "
        "replaced by this limit only if ALL hold: (a) its status is 'unstable' with reasons "
        "'parameter at the log-scale bound', optionally with 'Hessian not positive definite', "
        "or 'failed' with the single reason 'not converged' (the iterates drift along the "
        "flat ridge towards phi = 0, where no finite (log alpha, log beta) can meet the "
        "gradient tolerance); (b) a finite likelihood, no separation and start-point "
        "agreement within STATUS_RULES.start_agreement_tolerance; (c) it lies on the "
        "geometric ridge: log(alpha + beta) > log_parameter_bound and min(log alpha, "
        "log beta) >= -log_parameter_bound (degenerate lower boundaries never qualify); (d) "
        "the geometric MLE mu_geo = events / decisions agrees with the fit's mu within "
        "log_rate_tolerance (log scale); (e) the geometric log-likelihood is >= the E1 "
        "log-likelihood at the fit, evaluated with the exact finite-product form (well "
        "conditioned on the ridge), - log_likelihood_tolerance x max(1, |loglik|); (f) the "
        "one-sided score for phi at (mu_geo, phi = 0) per customer is <= "
        "max_score_per_customer (phi = 0 is a boundary maximum; convergence is then exact: "
        "the mu score is 0 at mu_geo); (g) the "
        "one-parameter geometric fit is 'ok' under STATUS_RULES. The fit is then 'ok' with "
        "alpha = beta = infinity and churn mu_geo, labelled and counted. A6 (E2) and every "
        "other status and fallback rule are unchanged."),
    max_score_per_customer=STATUS_RULES.gradient_tolerance,
    log_likelihood_tolerance=STATUS_RULES.relative_change_tolerance,
    log_rate_tolerance=1e-4)
"""Amendment A7 (P3): the E1 counterpart of A6 (P3 limitation 7): under homogeneous churn E1
cells ran to the geometric ridge and fell back to the miscalibrated pooled fit."""


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


@dataclass(frozen=True)
class OracleNumerics:
    bgnbd_jacobi_nodes: int
    pareto_jacobi_nodes: int
    copula_nodes: tuple[int, int, int]
    convergence_check: str
    convergence_rtol: float
    copula_convergence_rtol: float
    copula_spend_convergence_rtol: float
    monte_carlo_draws: int
    monte_carlo_check: str
    monte_carlo_z: float


ORACLE_NUMERICS = OracleNumerics(
    bgnbd_jacobi_nodes=64, pareto_jacobi_nodes=64, copula_nodes=(64, 48, 32),
    convergence_check=("every oracle is recomputed with every node count doubled on realistic "
                       "and extreme histories; the relative change must not exceed "
                       "convergence_rtol (BG/NBD, Pareto/NBD: all outputs) or "
                       "copula_convergence_rtol (correlated spend: material values) and "
                       "copula_spend_convergence_rtol (its posterior spend)"),
    convergence_rtol=1e-10, copula_convergence_rtol=1e-5, copula_spend_convergence_rtol=1e-8,
    monte_carlo_draws=1_000_000,
    monte_carlo_check=("the correlated-spend oracle is compared with a likelihood-weighted "
                       "Monte Carlo estimate from monte_carlo_draws independent draws of the "
                       "generator's copula prior (no quadrature shared with the oracle); value, "
                       "P(alive) and spend must lie within monte_carlo_z Monte Carlo SE"),
    monte_carlo_z=4.5)
"""Non-contractual oracle numerics (amendment A5; P2-measured: doubled-rule changes about
1e-14 for BG/NBD and Pareto/NBD and 3.3e-8 for the copula on the pilot). The contractual
oracles are exact finite sums and need no settings."""


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
                         evidence_target_min=20.0, evidence_ceiling_min=180.0,
                         evidence_memory_gb=4.0)
"""Amendment A12 raised ``evidence_ceiling_min`` from 45 to 180 (measured P8 design ~150 min on
P8_EXECUTION.workers workers); every other budget is unchanged."""
BUDGET_SCOPE = (
    "BUDGETS.evidence_target_min applies to one main-seed evidence run and to each phase's "
    "development record; BUDGETS.evidence_ceiling_min bounds the complete P8 execution (every "
    "REPEATED_SAMPLING study, the main-seed run, the M10 comparability replication and the "
    "fresh seed) on P8_EXECUTION, within BUDGETS.evidence_memory_gb; the dry run reports both "
    "against its measured and extrapolated full-size runtimes")
"""Amendment A12 (owner clarification)."""
P8_DRY_RUN = (
    "development seeds only (never a reserved seed or block): 10 draws for each of R1, R2 and "
    "REFIT, 10 per scale for R3, 20 refits per REFIT draw, and the main-seed analysis on one "
    "development seed; same code path and P8_EXECUTION as the final run; its record reports "
    "runtime, peak memory, extrapolated full-size runtime and every statistic, labelled "
    "'dry run - not evidence'")
"""Amendment A12 (owner decision): the dry run that the pending item requires."""
COVERAGE_DRAW_LEVEL_RULE = (
    "For REPEATED_SAMPLING customer-level intervals (R2 and REFIT) the unit of replication is "
    "the generator draw: for each setting, model, interval, level and group, the mean of "
    "per-draw coverage, its two-sided 95% t interval across draws, and the label 'below "
    "nominal' (interval entirely below the nominal level), 'above nominal' (entirely above) "
    "or 'consistent with nominal'; the pooled customer proportion is descriptive only. "
    "precision.classify_coverage remains the rule for one interval per replication. Coverage "
    "is reported, never asserted, and never triggers recalibration or a change of method, "
    "seed, scale or replication count (COVERAGE_REPORTING_RULE)")
"""Amendment A13 (owner decision)."""
P8_SINGLE_ATTEMPT = (
    "The final P8 run (scripts/generate_m11_evidence.py --mode final with --expect-manifest and "
    "--expect-code equal to the digests pinned after A13) is attempted once in the repository: "
    "after protocol verification, and only if no final ledger exists anywhere under "
    "docs/evidence, it takes the fixed reservation 'final' in the attempt registry "
    "docs/evidence/m11-p8-attempts (an atomically created, immutable entry recorded in a "
    "hash-chained attempts.log written under an exclusive lock; the registry is validated "
    "before every reservation and any alteration, deletion or unlogged entry refuses every "
    "further attempt); only then is the run directory's ledger.json created and any reserved "
    "seed used. Phases run in the order main seed, REPEATED_SAMPLING, M10 comparability, fresh "
    "seed; the fresh seed is authorised and recorded only at that last phase. Raw results are "
    "kept outside Git with their SHA-256 in the ledger; the evidence is record.md, summary.json "
    "and CSV tables, written only when every phase is complete. A failed or interrupted run is "
    "recorded and is never resumed, repeated or moved to replacement seeds, in any directory, "
    "without a new amendment")
"""Amendment A13 (owner decision)."""
P8_APPROVED_DRY_RUN = (
    "docs/evidence/m11-p8-dry-run (reservation dry-run-a3cef63df948e378), generated under "
    "manifest e40970d2d725966e57f8c648b897528a932bc09a8a8f98c60c91392c792cd6eb and code "
    "b4a1554bd3a4fff2e418e8f359f85c64a66ed080764db7869b92841ee1587b4b: record.md sha256 "
    "57a645b1f159e9c64e6ceee7fc5f8b641f2692c2d6722abc9ad75a9dcdf406a4; ledger.json "
    "f1977321a727cc2e87b9594debccb537a526154d64e1e6925292d03c2269c5ae; summary.json "
    "472ded23d1eb6583c7f45c2c296143745691a6c1210cb2f19f0f6efdaec84816; 10 draws per study, "
    "wall 3.0 min, peak 2.31 GB, extrapolated full size 141 min (ceiling 180)")
"""Amendment A13 (owner approval of the P8 dry-run record)."""
P8_FAILED_ATTEMPTS = (
    ("m11-p8-final/ledger.json",
     "cd7c8391347df2f9a2cf802161f35a0cb6a9a889b29c8fce3825a22ebe5902a9"),
    ("m11-p8-final-a14/ledger.json",
     "74a0cdd66175ab19f5ecf0dddef08114230713897d5f968d6295df390b678008"),
)
"""Preserved failed final ledgers (path relative to docs/evidence, SHA-256).

Recognition of a failed ledger does not authorise resumption or a new attempt.
"""
P8_FAILED_ATTEMPT_RECORD = (
    "First final attempt (reservation 'final', registry entry sha256 e5dc98f0402381f5c7308327"
    "8ca2cf140e6cf76fb77a3b6e5c5641f724c6ad04; ledger m11-p8-final/ledger.json), started "
    "2026-11-20T01:04:11Z on commit 981aa10 under manifest 9f92d0eb… and code a8b23ce6…: the "
    "main phase completed (raw/main.pkl sha256 d514d59e4b8e813f23855c482f2f952818ebf202446bfef7"
    "1a341b072cd23b11, never opened); the repeated-sampling phase failed at 03:23:18Z with "
    "ValueError 'p <= 0, p > 1 or p contains NaNs' from Generator.geometric in E4's R2 outcome "
    "simulation (purchases_noncontractual, C = 30, a seed in 810000-810499; not recorded): "
    "Beta(a, b + x) dropout draws underflow to exactly 0 when a weakly identified cell's "
    "Hessian parameter draw gives a below about 0.02. No repeated-sampling result was stored; "
    "the fresh seed was not consumed (docs/evidence/m11-p8-final-attempt-1.md)")
"""Amendment A14."""
E4_ZERO_DROPOUT_RULE = (
    "In E4 outcome simulation (P7 e4_simulate and P8 e4_simulate_fast) a dropout probability "
    "that is exactly 0 in floating point is its limit, no dropout within the window (purchases "
    "until dropout = the largest int64, the value NumPy returns for every positive p below about "
    "1e-300); the geometric draw is made with p replaced by 1, so every other draw and the "
    "random stream are unchanged. A non-finite or out-of-range p, or a non-finite or "
    "non-positive BG/NBD parameter draw, raises an error naming the task's study and seed")
"""Amendment A14."""
E4_ACCELERATED_MIN_A = 0.4
E4_ACCELERATED_SMALL_A = (
    "Parameter draws with a < E4_ACCELERATED_MIN_A use the original per-count Gauss-Jacobi "
    "nodes with the telescoped integrand (A(e_0) - A(e_H)) / p, finite at p = 0; for a >= 0.4 "
    "the accelerated rule is within 1.3e-11 of the per-count rule (the A12 bound 2e-11 holds). "
    "Validated at the A14 review on development seeds: 2-4 times more accurate than the frozen "
    "per-edge differencing against an exact reference for a <= 0.01; on four draws with the "
    "weakest cells (35,000+ customers) the fallback changed expected 12-month values by at most "
    "1.2e-9 relative and interval endpoints by at most 4.7e-9, left outcome draws bit-identical "
    "and changed no customer's coverage; only R2's E4 interval coverage (reported, never "
    "asserted) uses it, never a point prediction, primary comparison, selection or gate")
"""Amendment A14."""
P8_FINAL_ATTEMPT_KEY = "final-a20"
"""The A20 successor. Execution requires separate signed owner approval."""
P8_RETRY_RULE = (
    "Exactly one new final attempt: reservation final-a14 in docs/evidence/"
    "m11-p8-attempts and run directory docs/evidence/m11-p8-final-a14, on the same seed plan as "
    "the failed attempt (main, M10 comparability, coverage and refit blocks; the fresh seed "
    "still unused and last), allowed only if the only final ledgers under docs/evidence are "
    "those in P8_FAILED_ATTEMPTS with unchanged SHA-256. The failed attempt's ledger, raw "
    "results and reservation are kept unchanged as evidence and its main-seed results are never "
    "inspected. P8_SINGLE_ATTEMPT otherwise stands: no further attempt without a new amendment")
"""Historical A14 rule (owner approval, 2026-11-20). Its single retry was consumed."""

# --- P8: confirmatory tests, repeated sampling, execution (amendment A12) -----------------------
PC_TEST = (
    "PC1, PC3, PC4 and PC5: one two-sided test each of H0: E[D] = 0 on the main seed, D = "
    "(R - prediction of model_a)^2 - (R - prediction of model_b)^2, pooled over the comparison's "
    "evaluation cutoffs (customer x cutoff rows); estimate mean(D); p-value from the CR1 "
    "customer-clustered z (normal reference); a 95% customer-cluster percentile bootstrap "
    "interval (CLUSTERED_INFERENCE.metric_bootstrap_resamples) is reported; a customer without "
    "a finite prediction from either model is excluded and counted; a negative estimate favours "
    "model_a. PC2 follows PC_DEFINITIONS. Holm at FAMILY_ALPHA across the five p-values PC1-PC5")
PC_DEFINITIONS = {
    "PC1": ("E2 vs B1 (B1_DEFINITION) on m10_contractual, Tier A, existing_undiscounted_12, "
            "cutoffs 24 and 30, on the customers both models predict (M10's common-support "
            "rule); excluded customers are counted and reported"),
    "PC2": ("inference across R1's independent generator draws. On each draw, for every "
            "frailty_variance in the m10_contractual heterogeneity grid (one M10-generator "
            "scenario per value with the draw's seed, every other parameter at its M10 "
            "default), every channel and cutoffs 24 and 30, three expected discounted lifetime "
            "values per acquired customer (MONTHLY_DISCOUNT_RATE, summed to "
            "INFINITE_SUM_MONTHS): truth = the generator's population_value for the channel "
            "(its configured segment mix; M10's truth convention); E2 = the channel's segment x "
            "channel cells, each the P3 cohort_value with the cell's fit as used by E2's "
            "predictions (pooled fallback and A6 boundary included) and the ContributionModel's "
            "expected contribution by age, weighted by the channel's acquisitions in each "
            "segment over the CHURN_WINDOW_MONTHS before C (M10's H3 cell weighting); H4 = "
            "M10's H4 for the channel at C. Relative bias = value / truth - 1; draw statistic "
            "S = mean over grid values, channels and cutoffs of (|bias of H4| - |bias of E2|); "
            "a draw with any undefined value is excluded and counted; H0: E[S] = 0, two-sided "
            "one-sample t test across the R1 draws; positive favours E2; E1, built like E2 (A7 "
            "limit as its geometric equivalent), is reported alongside and not tested"),
    "PC3": ("E3 Tier B vs, at each of cutoffs 36 and 42 of signals_contractual, the Tier A model "
            "selected by the frozen SELECTION rule (Tier A candidates only); a cutoff whose Tier "
            "A selection is NO QUALIFIED MODEL is non-evaluable and its rows are excluded, never "
            "replaced by an unqualified model; PC_TEST pooled over the evaluable cutoffs; if "
            "neither cutoff is evaluable, PC3 enters Holm with p = 1, reported 'non-evaluable'"),
    "PC4": ("E4 vs B2 (B2_DEFINITION) on purchases_noncontractual, Tier A, "
            "existing_undiscounted_12, cutoffs 24 and 30, E4's population"),
    "PC5": ("E5 Tier B vs E3 Tier B on signals_contractual, existing_undiscounted_12, cutoffs "
            "36 and 42; E5 as specified (A9)"),
}
"""Amendment A12: the exact form of each frozen primary comparison (PRIMARY_COMPARISONS is
unchanged)."""


@dataclass(frozen=True)
class RepeatedStudy:
    id: str
    seed_start: int
    replications: int
    scales: tuple[float, ...]
    """Multipliers of the scenario's acquisitions_per_month (1.0 = the frozen scenario)."""
    content: str
    role: str
    """``confirmatory`` (an acceptance criterion) or ``reported`` (COVERAGE_REPORTING_RULE)."""


R3_REPLICATIONS = 200
REPEATED_SAMPLING = (
    RepeatedStudy("R1", COVERAGE_SEED_START, COVERAGE_REPLICATIONS, (1.0,),
                  "ORACLE_GATE on signals_contractual (Tier A vs Tier B oracles, cutoffs 36 "
                  "and 42) and the PC2 draw statistic", "confirmatory"),
    RepeatedStudy("R2", COVERAGE_SEED_START, MIN_COVERAGE_REPLICATIONS, (1.0,),
                  "interval coverage at CALIBRATION.interval_levels, by segment and prediction "
                  "decile as well as marginally: E1 and E2 on m10_contractual C = 30 and "
                  "signals_contractual C = 42 Tier A; E3 on m10_contractual C = 30 Tier A and "
                  "signals_contractual C = 42 Tiers A and B; E4 on purchases_noncontractual "
                  "C = 30; S2 with E5 on the same four settings (excluded for runtime: "
                  "m10_contractual C = 24, signals_contractual C = 36, purchases_noncontractual "
                  "C = 24). Parameter intervals (Hessian "
                  "draws for E1, E2 and E4, CR1 draws for E3; survival and purchase-process "
                  "parameters only: the contribution model and E4's contribution ratio are held "
                  "fixed) against the same-tier oracle; predictive intervals (contractual with "
                  "CONTRACTUAL_OUTCOME_NOISE) and S2 against the realised value; no "
                  "recalibration", "reported"),
    RepeatedStudy("R3", COVERAGE_SEED_START, R3_REPLICATIONS, (0.5, 1.0, 2.0),
                  "ACCEPTANCE_GATE_4: E2 on m10_contractual and E4 on purchases_noncontractual "
                  "(their correctly specified settings) at both evaluation cutoffs, with the "
                  "oracle, at every scale", "confirmatory"),
    RepeatedStudy("REFIT", REFIT_STUDY_SEED_START, REFIT_STUDY_REPLICATIONS, (1.0,),
                  "E2 on m10_contractual, Tier A, cutoff 30 only (cutoff 24 excluded for "
                  "runtime; E3 not included, E3_REFIT_DISCLOSURE): per draw "
                  "CLUSTERED_INFERENCE.refit_bootstrap_resamples customer resamples "
                  "(CLUSTERED_INFERENCE.resampling), E2 cell fits refitted on each, percentile "
                  "intervals of each customer's expected window value at "
                  "CALIBRATION.interval_levels (the contribution model held fixed); coverage "
                  "against the oracle, beside the Hessian parameter intervals", "reported"),
)
"""Amendment A12: the repeated-sampling studies (P8)."""
SHARED_SEEDS_DISCLOSURE = (
    "R1, R2 and R3 use the same coverage-block draws (seed COVERAGE_SEED_START + r): R2 the "
    "first MIN_COVERAGE_REPLICATIONS of R1's draws, R3 the first R3_REPLICATIONS at each scale "
    "(the scale changes only acquisitions_per_month). Draws are independent within each study; "
    "the studies are dependent on one another and are never pooled or combined. Within a draw "
    "every scenario's generator receives the same integer seed; analyses of different scenarios "
    "are never combined within a draw")
E3_REFIT_DISCLOSURE = (
    "No refit-bootstrap validation of E3 is performed (owner decision, P8 preflight): E3's "
    "parameter intervals rest on the CR1 sandwich, conditional on the chosen penalty, and are "
    "reported as not validated by a refit bootstrap")
ACCEPTANCE_GATE_4 = (
    "Correctness gate (approved plan, acceptance 4), on R3 for E2 on m10_contractual and E4 on "
    "purchases_noncontractual: (i) at every scale and cutoff, the mean over draws of the "
    "aggregate relative bias against the oracle, sum(pred) / sum(oracle) - 1, is consistent "
    "with 0 (|z| < ORACLE_GATE.consistency_z); (ii) per draw, the least-squares slope of "
    "log(oracle-excess RMSE) on log(scale) over the three scales; the mean slope across draws "
    "has a two-sided 95% t interval entirely below 0. A failure indicates a bug")
CONFIRMATORY_TESTS = (
    "PC1-PC5 (PC_TEST, PC_DEFINITIONS), Holm at FAMILY_ALPHA",
    "ORACLE_GATE on R1",
    "ACCEPTANCE_GATE_4 on R3",
)
EXPLORATORY_DIAGNOSTICS = (
    "E3 Tier B vs E3 Tier A", "R2 coverage and the REFIT study (COVERAGE_REPORTING_RULE)",
    "selection frequencies across draws", "P(alive) reliability",
    "pre-A11 vs A11 interval comparison", "CR1 vs model-based standard errors",
    "E5 extrapolation diagnostics", "E4_ACCELERATED_SIMULATION equivalence check",
    "M10 comparability replication (M10_COMPARABILITY_SEED)",
)
"""Amendment A12: reported, never acceptance criteria or Holm members."""
E4_ACCELERATED_MAX_X = 44
E4_ACCELERATED_SIMULATION = (
    "P8 repeated-sampling E4 parameter and predictive draws may use the accelerated "
    "computation: one Gauss-Jacobi rule for Beta(a, b) per parameter draw, reweighted exactly by "
    "(1 - p)^x B(a, b) / B(a, b + x), the window's expected purchases from its first and last "
    "edges (the monthly terms telescope), and all parameter draws at once; only for customers "
    "with x <= E4_ACCELERATED_MAX_X repeat purchases (the range validated at the P8 preflight: "
    "relative difference <= 2e-11 from the per-count rule); every other customer uses the "
    "original per-count Gauss-Jacobi rule unless the accuracy is independently established and "
    "recorded. Point predictions and P7's e4_simulate (and so P7's record) are unchanged")


@dataclass(frozen=True)
class P8Execution:
    workers: int
    threads_per_worker: int
    seeding: str
    equivalent_optimisations: tuple[str, ...]


P8_EXECUTION = P8Execution(
    workers=3, threads_per_worker=1,
    seeding="each draw is generated and simulated from its own seed only, so results do not "
            "depend on scheduling or the number of workers",
    equivalent_optimisations=(
        "O1 parallel draws", "O2 the E3/E5 tuning fit at the chosen penalty is the validation "
        "input", "O3 one paired-bootstrap weight matrix per selection report; (b, a) mirrors "
        "(a, b)", "O4 E4_ACCELERATED_SIMULATION", "O5 one pass per seed feeds every study",
        "O6 repeated draws skip quantities no repeated-sampling study reports"))
"""Amendment A12; memory is bounded by BUDGETS.evidence_memory_gb (measured 3.1 GB)."""

# --- stage gate ---------------------------------------------------------------------------------
PENDING_BEFORE_MAIN: tuple[str, ...] = ()
"""A18 requires independent owner approval before reserved-seed access and reservation."""
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
    """True: made before any main-seed computation."""
    change: str
    reason: str
    before_outcome_inspection: bool | None = True
    """True: made before any main-seed, fresh-seed, coverage-block or refit-block outcome was
    inspected (implied by ``before_main_seed``; recorded separately since A14).
    None denoted a pending confirmation in the held A16 configuration."""


AMENDMENTS_AFTER_MAIN_SEED_COMPUTATION = ("A14", "A15", "A16", "A17", "A18", "A19", "A20")
"""A14 followed the first main computation. A15 through A18 followed both."""


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
    Amendment("A5", "P2", "2026-11-18", True,
              "pinned the non-contractual oracle numerics (ORACLE_NUMERICS): Gauss-Jacobi 64 "
              "nodes (BG/NBD) and 64 (Pareto/NBD), copula tensor rule (64, 48, 32), doubled-rule "
              "convergence checks and a 1,000,000-draw independent Monte Carlo check; closed "
              "pending item 'P2: oracle Monte Carlo validation sizes and quadrature settings'; "
              "PROTOCOL_STAGE 'P1-amended' -> 'P2-amended'",
              "owner approval of the settings measured in the P2 assurance review "
              "(docs/evidence/m11-p2-oracle-findings.md, m11-p2-oracle-plan.md)"),
    Amendment("A6", "P3", "2026-11-18", True,
              "narrow E2 v = 0 boundary exception (E2_GEOMETRIC_BOUNDARY): an E2 fit unstable "
              "only because v reached its lower log-scale bound, and verified to be the "
              "geometric limit, is used with v = 0 as 'ok' instead of the pooled fallback; "
              "all other fit-status and fallback rules unchanged; PROTOCOL_STAGE "
              "'P2-amended' -> 'P3-amended'",
              "owner approval after P3 finding 2 (docs/evidence/m11-p3-findings.md): under "
              "homogeneous churn the correct MLE is v = 0, and the pooled substitute was "
              "miscalibrated (holdout slope 1.18-1.19)"),
    Amendment("A7", "P3", "2026-11-18", True,
              "narrow E1 geometric boundary exception (E1_GEOMETRIC_BOUNDARY): an E1 fit "
              "unstable at the upper log-scale bound, or not converged, only because its "
              "iterates ran onto the alpha + beta -> infinity ridge, and verified to be at the "
              "geometric limit phi = 0 (closed-form mu, non-positive one-sided phi score), is "
              "used with alpha = beta = infinity as 'ok' instead of the pooled fallback; A6 and "
              "all other fit-status and fallback rules unchanged; PROTOCOL_STAGE stays "
              "'P3-amended'",
              "owner approval after P3 limitation 7 (docs/evidence/m11-p3-findings.md): under "
              "homogeneous churn E1 cells still fell back to the miscalibrated pooled fit "
              "(holdout slope 1.18-1.19)"),
    Amendment("A8", "P4", "2026-11-18", True,
              "E3 excludes acquisition_cohort, tenure_months and paid_months from its linear "
              "covariates (E3_EXCLUDED_COVARIATES), keeping its tenure-band basis; S1 and the "
              "feature registry unchanged. Clarifications: E3_TUNING_LOSS (window-value MSE, "
              "ties to the larger penalty), E3_SELECTION_ELIGIBILITY (no validation fit -> not "
              "selection-eligible; no censored validation), DESIGN_DUPLICATE_RULE "
              "(deterministic removal of exact duplicate design columns); PROTOCOL_STAGE "
              "'P3-amended' -> 'P4-amended'",
              "owner decision after the P4 diagnosis (docs/evidence/m11-p4-findings.md): the "
              "frozen linear covariates were extrapolated beyond the landmarks' range (W = 24 "
              "at C = 33: mean(pred - oracle) -18 to -25; five validation-gate failures)"),
    Amendment("A9", "P6", "2026-11-19", True,
              "E5 tuning loss, shared penalty, tie-break and no-ok-fit fallback "
              "(E5_TUNING_LOSS); E5 selection eligibility (E5_SELECTION_ELIGIBILITY); no change "
              "to E5's design, S2, SELECTION gates or any tolerance; PROTOCOL_STAGE "
              "'P4-amended' -> 'P6-amended'",
              "P6 final review: TUNING_RULE names no loss and the A8 clarifications are scoped "
              "to E3, so E5's penalty choice and eligibility were not independently "
              "authorised (docs/evidence/m11-p6-findings.md)"),
    Amendment("A10", "P7", "2026-11-19", True,
              "B1 definition (B1_DEFINITION: M10 Kaplan–Meier by cell, unchanged, no "
              "fallback); B2 definition (B2_DEFINITION); reference-fallback scope "
              "(REFERENCE_FALLBACK_SCOPE: contractual only); Tier A models in Tier B "
              "selections (TIER_A_IN_TIER_B); E3 no-ok-fit fallback (E3_NO_OK_FIT_RULE); no "
              "change to any model's behaviour, SELECTION gates, tolerance, target, comparison "
              "or seed; PROTOCOL_STAGE 'P6-amended' -> 'P7-amended'",
              "P5 dispositions (B2 unspecified; E2 not applicable to non-contractual), the P6 "
              "review (E3 fallback unauthorised) and the P7 preflight (B1 is nine M10 methods "
              "under one id; Tier B candidacy unstated); prospective, before any B1 or B2 "
              "selection result exists"),
    Amendment("A11", "P7", "2026-11-19", True,
              "contractual predictive-interval outcome noise (CONTRACTUAL_OUTCOME_NOISE); no "
              "change to any point prediction, parameter interval, SELECTION gate, tolerance, "
              "target, comparison or seed; PROTOCOL_STAGE stays 'P7-amended'",
              "P7 finding 5: the plan's 'parameter + outcome uncertainty' omitted contribution "
              "noise (80%/95% predictive intervals covered ~0.60/~0.70). Pre-cutoff residuals: "
              "SD 2.1 against a mean of 5.3 per month, within-customer correlation about 0.01, "
              "lag-1 about 0.00 (development seeds; months <= complete_through(C) only). "
              "Prospective; pre-A11 results kept"),
    Amendment("A12", "P8", "2026-11-19", True,
              "P8 design: PC_TEST and PC_DEFINITIONS (exact form of PC1-PC5; PC3 comparator = "
              "the frozen Tier A selection, non-evaluable cutoffs excluded, p = 1 if none); "
              "REPEATED_SAMPLING (R1 1,000 draws, R2 500, R3 200 per scale at 0.5/1/2, E2-only "
              "REFIT study 200 x 200) with SHARED_SEEDS_DISCLOSURE; E3_REFIT_DISCLOSURE; "
              "ACCEPTANCE_GATE_4; CONFIRMATORY_TESTS vs EXPLORATORY_DIAGNOSTICS; "
              "E4_ACCELERATED_SIMULATION (x <= E4_ACCELERATED_MAX_X); P8_EXECUTION (3 workers, "
              "1 thread each); BUDGETS.evidence_ceiling_min 45 -> 180 (BUDGET_SCOPE); "
              "P8_DRY_RUN; PC2 pinned to the P3 cohort_value, M10 population_value, H3 cell "
              "weighting and H4 conventions; R2 and REFIT excluded cutoffs recorded. The "
              "repeated-sampling "
              "preflight item is closed and replaced by the P8 dry-run item, so the main and "
              "fresh seeds stay refused. PRIMARY_COMPARISONS, FAMILY_ALPHA, SELECTION, "
              "STATUS_RULES, ORACLE_GATE, tolerances, targets, models and seeds unchanged; "
              "PROTOCOL_STAGE 'P7-amended' -> 'P8-amended'",
              "P8 preflight (measured, development seeds): one full-scope draw 60 s; reduced "
              "draw 29 s on one worker, 10.9 s effective on 3 workers (3.1 GB); P8 design about "
              "150 min. Owner decisions at the P8 preflight review; prospective, before any "
              "main-seed, fresh-seed, coverage-block or refit-block evaluation"),
    Amendment("A13", "P8", "2026-11-20", True,
              "closes the P8 dry-run item (PENDING_BEFORE_MAIN empty: the main and fresh seeds "
              "become available through authorise_seed); COVERAGE_DRAW_LEVEL_RULE; "
              "P8_SINGLE_ATTEMPT; P8_APPROVED_DRY_RUN; no change to any comparison, gate, "
              "tolerance, study size, model or seed; PROTOCOL_STAGE stays 'P8-amended'",
              "owner approval of the P8 dry-run record and of the single-attempt safeguards "
              "(attempt registry) tested at the P8 execution-readiness review"),
    Amendment("A14", "P8", "2026-11-20", False,
              "records the failed first final attempt (P8_FAILED_ATTEMPT_RECORD, "
              "P8_FAILED_ATTEMPTS); E4_ZERO_DROPOUT_RULE; E4_ACCELERATED_MIN_A and "
              "E4_ACCELERATED_SMALL_A; P8_FINAL_ATTEMPT_KEY and P8_RETRY_RULE (one new attempt "
              "on the same seeds); no change to any estimand, comparison, gate, tolerance, "
              "study size, model or seed; PROTOCOL_STAGE stays 'P8-amended'",
              "the floating-point underflow defect found in the first final attempt (an "
              "implementation defect, not a model or parameter issue); same seeds rather than "
              "replacements, because dropping the draws that triggered the failure would bias "
              "coverage; validation: P7 record sections and every P8 dry-run table reproduce "
              "byte-identically with the correction, and the small-a fallback changes no "
              "coverage outcome",
              before_outcome_inspection=True),
    Amendment("A15", "P8", "2026-11-21", False,
              "Correct E4 Beta quadrature and its continuous window integrand, accelerated "
              "Beta reweighting and BG/NBD alive probabilities. Refuse nonfinite E4 R2 coverage "
              "inputs and retain outcome-free failure trace locations. Preserve all frozen "
              "statistical specifications and A14 protections. Both final reservations "
              "remain consumed. Final execution is held pending separate owner authorisation.",
              "Synthetic fixtures reproduce the numerical defects. The historical final-a14 "
              "crash cause remains unconfirmed. Both failed attempts completed main before A15. "
              "The owner confirmed outcome blindness across both attempts. The first main "
              "pickle was previously read as bytes for SHA-256 verification, without "
              "deserialisation or outcome inspection. Both ledgers and reservations "
              "are preserved.",
              before_outcome_inspection=True),
    Amendment("A16", "P8", "2026-11-21", False,
              "Configure the candidate reservation key final-a15 in docs/evidence/"
              "m11-p8-attempts and its output directory docs/evidence/m11-p8-final-a15/. "
              "Retain the execution hold. Preserve the A15 corrections, frozen statistical "
              "specifications, resource limits and both permanently consumed reservations "
              "(final and final-a14).",
              "The owner approved configuration only. Reservation creation, reserved-seed "
              "generation and dispatch require separate owner execution authorisation. "
              "Confirmation of continued outcome blindness was pending at A16 and recorded "
              "under A17.",
              before_outcome_inspection=True),
    Amendment("A17", "P8", "2026-11-22", False,
              "Retain a fail-closed launch hold for the sole proposed final-a15 attempt in "
              "docs/evidence/m11-p8-final-a15/ with registry docs/evidence/m11-p8-attempts. "
              "Preserve the frozen study, A15 execution safeguards, resource limits and "
              "both pinned failures. The final and final-a14 reservations are permanently "
              "consumed. The A14 retry is exhausted.",
              "The owner confirmed the protected statistical outcomes from both failed "
              "attempts remain uninspected. The owner accepted the unconfirmed A14 failure "
              "cause, residual A15 numerical precision risks, possible transient RSS peaks "
              "between samples and the possibility of exceeding the 180-minute ceiling. "
              "Approval covers hold-release configuration only. Reservation creation, "
              "reserved-seed data generation and dispatch require green CI on the exact "
              "release commit, a passing final host preflight and separate explicit owner "
              "authorisation for a single launch.",
              before_outcome_inspection=True),
    Amendment("A18", "P8", "2026-11-22", False,
              "Replace the static A17 launch hold with a fail-closed check for detached "
              "owner approval of one final-a15 launch. Bind approval to the exact release "
              "commit, green CI run, protocol digests, runner and supervisor hashes, "
              "designated Mac and canonical registry and output paths. Preserve the "
              "frozen study, A15 safeguards and both consumed attempts.",
              "The owner approved release preparation only, with previously confirmed "
              "outcome blindness. A signed launch approval requires separate explicit "
              "owner authorisation after green CI and a passing immediate host preflight.",
              before_outcome_inspection=True),
    Amendment("A19", "P8", "2026-11-24", False,
              'Configure final-a19 for one governed confirmatory dispatch. Preserve the '
              'frozen estimands, scenario and seed assignments, RNG sequences, model and '
              'inference methods, simulation counts, acceptance thresholds, output '
              'semantics and fresh-last ordering. Keep isolated fits, three workers for '
              'R1/R3/REFIT and a fresh two-worker pool for each R2 pair. Retain the '
              '4,000,000,000-byte aggregate sampled RSS ceiling and 10,800-second deadline. '
              'Require owner-pinned private history, immutable source and runtime, durable '
              'consumption before dispatch, and a protected terminal receipt followed by a '
              'within-budget certification decision. Keep outcomes private. The final, '
              'final-a14 and final-a15 identities remain permanently consumed. Failure or '
              'interruption consumes final-a19 permanently.',
              'The owner confirmed continued outcome blindness after prior main-seed '
              'computation. Full-count development qualification supports this execution '
              'configuration. A19 adoption alone grants no launch authority. A separate '
              'signed owner approval is required after green CI on the exact amended '
              'release and a passing immediate host preflight.',
              before_outcome_inspection=True),
    Amendment("A20", "P8", "2026-11-24", False,
              'Configure final-a20 for at most one independently authorised confirmatory '
              'dispatch. Keep final, final-a14, final-a15 and final-a19 permanently '
              'consumed. Preserve the protected A19 startup-failure record and bind '
              'approval to its journal chain tip. Require the dedicated worker to verify '
              'the immutable checkout, including the Git index, the signed release and the '
              'private history under native supervision before consumption. Preserve all '
              'frozen estimands, numerical and inference methods, seeds, RNG sequences, '
              'simulation counts and scales, acceptance thresholds, reporting semantics and '
              'fresh-last ordering. Keep the qualified hybrid schedule and the '
              '4,000,000,000-byte aggregate sampled RSS and 10,800-second limits.',
              'The final-a19 attempt on release 493fd0b was consumed by a child failure '
              'with exit code 2 after 1.657 seconds. The dedicated worker could not read '
              'the root-owned Git index and stopped before an execution ledger was created. '
              'The index permissions were repaired and worker Git status and complete '
              'signed-release verification passed. The owner confirmed continued outcome '
              'blindness. A20 adoption grants no launch authority. A separate '
              'attempt-specific signed approval requires green CI on the exact amended '
              'release and a passing immediate host preflight.',
              before_outcome_inspection=True),
)
"""Protocol changes since P0 (82e3ebd), with their relation to computation and inspection."""


class ProtocolNotReady(RuntimeError):
    """A reserved seed was requested before the protocol allows it."""


def require_successor_amendment() -> None:
    """Refuse a successor until the owner adopts A20 and records outcome blindness."""
    if P8_FINAL_ATTEMPT_KEY in p8_candidate.CONSUMED_IDENTITIES:
        raise ProtocolNotReady(f"attempt {P8_FINAL_ATTEMPT_KEY} is permanently consumed")
    amendment = PROTOCOL_AMENDMENTS[-1]
    if (P8_FINAL_ATTEMPT_KEY != p8_candidate.A20_ATTEMPT
            or (amendment.id, amendment.stage) != ("A20", "P8")
            or amendment.before_main_seed is not False
            or amendment.before_outcome_inspection is not True
            or AMENDMENTS_AFTER_MAIN_SEED_COMPUTATION[-1] != "A20"):
        raise ProtocolNotReady("the A20 successor amendment has not been adopted")
    try:
        adopted = date.fromisoformat(amendment.date)
    except ValueError as exc:
        raise ProtocolNotReady("the A20 owner decision date is invalid") from exc
    if adopted.isoformat() != amendment.date or adopted < date.fromisoformat(
            PROTOCOL_AMENDMENTS[-2].date):
        raise ProtocolNotReady("the A20 owner decision date is invalid")


def require_launch_approval(*, require_clean: bool = True) -> None:
    """Verify approval against the release containing this protocol."""
    require_successor_amendment()
    try:
        repo = Path(__file__).resolve().parents[3]
        approval_sha256 = p8_candidate.require_owner_approval(
            repo, attempt=P8_FINAL_ATTEMPT_KEY,
            manifest=protocol_digest(), code=protocol_code_digest(), require_clean=require_clean)
        p8_candidate.require_active_claim(repo, attempt=P8_FINAL_ATTEMPT_KEY,
                                          manifest=protocol_digest(), code=protocol_code_digest(),
                                          approval_sha256=approval_sha256)
    except LaunchApprovalError as exc:
        raise ProtocolNotReady(str(exc)) from exc


def authorise_seed(purpose: str, *, confirm_single_final_run: bool = False) -> int:
    """The only sanctioned way to obtain a reserved seed.

    ``pilot`` is always allowed (planning). ``main`` requires no pending protocol items and
    confirmed outcome-blindness timing for every amendment and verified owner launch approval.
    ``fresh`` additionally requires ``confirm_single_final_run=True``.
    """
    if purpose == "pilot":
        return PILOT_SEED
    if purpose not in ("main", "fresh"):
        raise ValueError("purpose must be 'pilot', 'main' or 'fresh'")
    if PENDING_BEFORE_MAIN:
        raise ProtocolNotReady(f"{len(PENDING_BEFORE_MAIN)} protocol items are pending; "
                               f"the {purpose} seed is refused")
    if any(a.before_outcome_inspection is not True for a in PROTOCOL_AMENDMENTS):
        raise ProtocolNotReady("outcome-blindness confirmation is required")
    if purpose == "fresh" and not confirm_single_final_run:
        raise ProtocolNotReady("the fresh seed is evaluated once, last; confirm explicitly")
    # All governed writes are private. The source tree remains clean through fresh.
    require_launch_approval()
    return MAIN_SEED if purpose == "main" else FRESH_SEED


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
    if model(REFERENCE_FALLBACK_MODEL).kinds != ("contractual",):
        problems.append("REFERENCE_FALLBACK_SCOPE needs a contractual-only reference model")
    if set(PC_DEFINITIONS) != {pc.id for pc in PRIMARY_COMPARISONS}:
        problems.append("PC_DEFINITIONS must define exactly the primary comparisons")
    studies = {st.id: st for st in REPEATED_SAMPLING}
    if len(studies) != len(REPEATED_SAMPLING) or set(studies) != {"R1", "R2", "R3", "REFIT"}:
        problems.append("REPEATED_SAMPLING must define R1, R2, R3 and REFIT once each")
    else:
        cov = range(COVERAGE_SEED_START, COVERAGE_SEED_START + COVERAGE_REPLICATIONS)
        refit = range(REFIT_STUDY_SEED_START, REFIT_STUDY_SEED_START + REFIT_STUDY_REPLICATIONS)
        for st in REPEATED_SAMPLING:
            block = refit if st.id == "REFIT" else cov
            if st.replications < 1 or st.seed_start not in block or \
                    st.seed_start + st.replications - 1 not in block:
                problems.append(f"{st.id}: seeds outside its reserved block")
            if st.role not in ("confirmatory", "reported") or not st.scales:
                problems.append(f"{st.id}: invalid role or scales")
        if studies["R1"].replications != COVERAGE_REPLICATIONS:
            problems.append("R1 must use COVERAGE_REPLICATIONS")
        if not MIN_COVERAGE_REPLICATIONS <= studies["R2"].replications <= COVERAGE_REPLICATIONS:
            problems.append("R2 replications must lie in [MIN_COVERAGE_REPLICATIONS, "
                            "COVERAGE_REPLICATIONS]")
        if studies["REFIT"].replications != REFIT_STUDY_REPLICATIONS:
            problems.append("REFIT must use REFIT_STUDY_REPLICATIONS")
        if len(studies["R3"].scales) < 3 or 1.0 not in studies["R3"].scales:
            problems.append("R3 needs at least three scales including 1.0")
    if P8_EXECUTION.workers < 1 or P8_EXECUTION.threads_per_worker < 1:
        problems.append("invalid P8 execution")
    if BUDGETS.evidence_ceiling_min < BUDGETS.evidence_target_min:
        problems.append("the evidence ceiling must not be below the target")
    if E4_ACCELERATED_MAX_X < 0:
        problems.append("invalid E4 accelerated range")
    if not 0 < E4_ACCELERATED_MIN_A:
        problems.append("invalid E4 accelerated minimum a")
    if P8_FINAL_ATTEMPT_KEY == "final" or any(
            len(sha) != 64 for _, sha in P8_FAILED_ATTEMPTS):
        problems.append("A14 retry needs a new attempt key and SHA-256 failed-ledger hashes")
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
    if len(ids_a) != len(set(ids_a)):
        problems.append("amendments must have unique ids")
    after = tuple(a.id for a in PROTOCOL_AMENDMENTS if not a.before_main_seed)
    if after != AMENDMENTS_AFTER_MAIN_SEED_COMPUTATION:
        problems.append("only the amendments in AMENDMENTS_AFTER_MAIN_SEED_COMPUTATION may "
                        "follow a main-seed computation; every other one must precede it")
    if any(a.before_outcome_inspection is not True
           and not (a.id == "A16" and a.before_outcome_inspection is None and PENDING_BEFORE_MAIN)
           for a in PROTOCOL_AMENDMENTS):
        problems.append("amendments require confirmed outcome blindness before execution")
    flags = [a.before_main_seed for a in PROTOCOL_AMENDMENTS]
    if any(not f and any(flags[i + 1:]) for i, f in enumerate(flags)):
        problems.append("an amendment cannot precede the main seed after one that followed it")
    contractual_a = {f.name for f in features("contractual", "A")}
    if not set(E3_EXCLUDED_COVARIATES) <= contractual_a:
        problems.append("E3_EXCLUDED_COVARIATES must name contractual Tier A registry features")
    on = ORACLE_NUMERICS
    if min(on.bgnbd_jacobi_nodes, on.pareto_jacobi_nodes, *on.copula_nodes) < 1 or \
            len(on.copula_nodes) != 3:
        problems.append("oracle quadrature node counts must be positive")
    if not 0 < on.convergence_rtol <= on.copula_spend_convergence_rtol \
            <= on.copula_convergence_rtol < 1:
        problems.append("invalid oracle convergence tolerances")
    if on.monte_carlo_draws < 1 or on.monte_carlo_z <= 0:
        problems.append("invalid oracle Monte Carlo check")
    if DEFAULT_PENALTY not in PENALTY_GRID:
        problems.append("the default penalty must be on the grid")
    if not 0 < PLANNING_ALPHA < FAMILY_ALPHA or not 0.5 < PLANNING_POWER < 1:
        problems.append("invalid planning error rates")
    return problems
