"""Milestone 10 evaluation protocol — frozen before any M10 estimate was computed.

Seeds, horizons, discounting, estimands, backtest cutoffs, stability rules, the heterogeneity
grid, the precision target and the repeated-sampling sizes are fixed here. The pilot seed is
used only for the LTV/CAC precision calculation, which is run before any main-seed result is
examined; the fresh seed only by the final evidence run; tests use neither. Any later change
must be disclosed in the M10 evidence.
"""

from __future__ import annotations

# --- seeds -------------------------------------------------------------------------------
MAIN_SEED = 20261001
PILOT_SEED = 20261002  # precision planning only
FRESH_SEED = 20261003  # one final evaluation only
COVERAGE_SEED_START = 30_000
COVERAGE_REPLICATIONS = 200

# --- time and money ----------------------------------------------------------------------
N_MONTHS = 48  # calendar months 0..47; the data end is the end of month 47
ANNUAL_DISCOUNT_RATE = 0.10
MONTHLY_DISCOUNT_RATE = (1.0 + ANNUAL_DISCOUNT_RATE) ** (1.0 / 12.0) - 1.0
HORIZONS = (24, 36)  # finite projection horizons, months; infinite horizon is conditional
INFINITE_SUM_MONTHS = 1200  # truncation for numerically summed "infinite" truths

# --- estimands ---------------------------------------------------------------------------
PRIMARY_LTV_AGE = 24
"""Primary LTV for LTV/CAC: realised discounted contribution over a customer's first 24 months,
averaged over every acquired customer (churned customers contribute zeros), using only cohorts
whose 24-month window is completely observed."""
PRECISION_TARGET = 0.15
"""Target relative half-width of the 95% interval of LTV/CAC per channel (a target, reported
whether met or not; the design is not changed after main results are seen)."""
BOOTSTRAP_RESAMPLES = 2000

# --- heuristics ---------------------------------------------------------------------------
CHURN_WINDOW_MONTHS = 12  # churn and run-rate estimated from the trailing 12 months
MIN_CHURN_EVENTS = 30  # fewer churn events -> heuristic flagged "unstable"
KM_TAIL_MONTHS = 12  # KM tail extrapolation: hazard over the last 12 observed tenure months
MIN_AT_RISK = 50  # KM curve truncated where fewer customers remain at risk

# --- backtests ----------------------------------------------------------------------------
BACKTEST_WINDOW_MONTHS = 12
BACKTEST_CUTOFFS = (24, 30, 36)  # month-end cutoffs; each window must end by the data end
MAX_RECORDING_LAG_DAYS = 45  # realised windows are read only after this lag has elapsed
TOP_FRACTION = 0.10

# --- grids ---------------------------------------------------------------------------------
HETEROGENEITY_GRID = (0.0, 0.25, 0.5, 1.0)  # variance of the Gamma churn frailty
SENSITIVITY_DISCOUNT_RATES = (0.0, 0.05, 0.10, 0.15)
SENSITIVITY_MARGIN_SHIFTS = (-0.10, 0.0, 0.10)  # relative change in gross margin
SENSITIVITY_COST_SHIFTS = (-0.20, 0.0, 0.20)  # relative change in cost-to-serve
