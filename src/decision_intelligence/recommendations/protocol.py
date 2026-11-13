"""Milestone 9 evaluation protocol — frozen before any M9 model was evaluated.

Every choice that could be tuned to the results is fixed here: seeds, windows, K, metrics,
hyperparameter grids, selection rules, the simulated A/B design (primary metric, guardrails,
multiplicity correction, minimum worthwhile effect) and the repeated-sampling sizes. Any later
change must be disclosed in the M9 evidence. The fresh seed is used only by the final evidence
run, after this file and the code are frozen; tests never use it.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- seeds -------------------------------------------------------------------------------
MAIN_SEED = 20260801  # the M8 default scenario (its test window was inspected in M8; disclosed)
STABILITY_SEEDS = (1, 2, 3, 4, 5)
FRESH_SEED = 20260915  # untouched until the final evidence run

# --- ranking -----------------------------------------------------------------------------
K = 10
SELECTION_METRIC = "ndcg"  # mean logged NDCG@10 on the validation window (as M8)
CANDIDATES_PER_GENERATOR = 50
REFERENCE_MODEL = "item-kNN"  # paired comparisons (Holm) are against the tuned M8 item-kNN
# Personalised single-stage models (M8 and M9) fall back to segment popularity (lambda = 100)
# for users with no training clicks; the two-tower model scores them from user features.
# Ranker variants: {pointwise, LambdaRank} x {naive labels, propensity-weighted}; within each of
# the four cells the L2 penalty is chosen on validation; the two-stage hybrid uses the cell
# and penalty with the best validation logged NDCG@10. All four cells are reported on test.

# --- chronological windows (days since start) --------------------------------------------
# Single-stage models: fit <= 84, validate (84, 98], refit <= 98, test (98, 120] (as M8).
# Two-stage ranker: tune = generators/features at 70, labels (70, 84], validate on (84, 98]
# with generators/features at 84; final = ranker retrained on labels (84, 98] with
# generators/features at 84, applied with generators/features at 98 to the test window.
RANKER_TRAIN_START = 70.0
TRAIN_DAYS = 84.0
VALIDATION_DAYS = 14.0
TEST_END_DAYS = 120.0

# --- hyperparameter grids (frozen) --------------------------------------------------------
ALS_GRID = tuple((f, reg, alpha) for f in (16, 32, 64) for reg in (0.1, 1.0)
                 for alpha in (5.0, 20.0))
ALS_ITERATIONS = 15
BPR_GRID = tuple((f, reg) for f in (16, 32, 64) for reg in (0.001, 0.01))
BPR_EPOCHS = 30
BPR_LEARNING_RATE = 0.05
EASE_GRID = (50.0, 200.0, 500.0, 2000.0)
TWO_TOWER_GRID = tuple((dim, hidden) for dim in (16, 32) for hidden in (0, 32))
TWO_TOWER_EPOCHS = 20
TWO_TOWER_LEARNING_RATE = 0.01
POINTWISE_L2_GRID = (0.1, 1.0, 10.0)
LAMBDARANK_L2_GRID = (0.01, 0.1)
LAMBDARANK_EPOCHS = 30
LAMBDARANK_LEARNING_RATE = 0.05
BLEND_STEP = 0.25  # weights on a simplex grid over (ALS, item-kNN, content, segment popularity)
ANN_CLUSTERS = (8, 16, 32)
ANN_PROBES = (1, 2, 4, 8)

# --- repeated sampling --------------------------------------------------------------------
OPE_REPLICATIONS = 1000
BOOTSTRAP_RESAMPLES = 999
OPE_SCENARIO = {"n_users": 600, "n_items": 150}
AA_REPLICATIONS = 300
AB_COVERAGE_REPLICATIONS = 300

# --- operational benchmark ----------------------------------------------------------------
SCALE_TIERS = ((3000, 600), (10_000, 2000), (30_000, 5000))
MEMORY_BUDGET_BYTES = 1_500_000_000  # any single planned allocation above this is skipped
LATENCY_REQUESTS = 200


@dataclass(frozen=True)
class ABDesign:
    """The simulated user-randomised A/B test, pre-specified."""

    experiment: str = "m9_recommender_ab"
    arms: tuple[str, ...] = ("control", "single_model", "hybrid")
    start_day: float = 120.0
    duration_days: float = 14.0
    pre_period_days: float = 14.0
    primary_metric: str = "clicks per user"
    secondary_metrics: tuple[str, ...] = ("purchases per user", "margin per user")
    guardrails: tuple[tuple[str, float], ...] = (("purchases per user", 0.10),
                                                ("margin per user", 0.75))
    """Non-inferiority margins (absolute deterioration per user over the test period)."""
    minimum_worthwhile_effect: float = 0.10
    """Clicks per user over the period."""
    alpha: float = 0.05
    multiplicity: str = "bonferroni"
    """Two primary comparisons (each treatment vs control): each at alpha / 2, with
    matching (1 - alpha / 2) intervals so tests and intervals stay dual."""
    srm_alpha: float = 0.001
    simulation_seed: int = 9119
    selection_rule: str = (
        "control = M8 item-kNN (M8 grid); single_model = the M9 single-stage family with the "
        "highest validation logged NDCG@10; hybrid = the better of the score blend and the "
        "two-stage ranker on validation logged NDCG@10. Chosen before any A/B outcome is "
        "simulated; A/B outcomes are never used for selection.")


AB_DESIGN = ABDesign()
