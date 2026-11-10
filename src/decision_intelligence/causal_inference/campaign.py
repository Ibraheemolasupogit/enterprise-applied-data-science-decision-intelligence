"""Causal analysis of the M2 retention-offer campaign (confounded, known effect).

Question: *what is the effect of the retention offer on 90-day retention?*

The M2 generator (data/marketing.py, data/retention.py) assigns the offer
with probability ``sigmoid(-2 + 4 · latent_value)`` and raises each
customer's 90-day retention probability by exactly
``config.marketing_true_effect`` (0.08) — no clipping binds, so the
individual effect on the probability scale is constant: ATE = ATT = every
subgroup CATE = 0.08. ``latent_value`` also raises baseline retention, so
it confounds the naive comparison.

Three analyst information regimes are compared, because what is
identifiable depends on what is measured:

* **A — confounder measured**: adjust for ``latent_value`` (+ segment). The
  DAG says this satisfies the backdoor criterion.
* **B — confounder unmeasured**: the M2 documentation calls ``latent_value``
  a hidden driver. With only observed proxies (segment, price sensitivity,
  channel, region) the backdoor path ``treated <- latent_value -> churn``
  stays open: the effect is *not identified*, and the analysis says so.
* **C — assignment mechanism known**: M2's targeting *randomised* each
  customer with a probability it logged in ``customer_treatments``.
  Weighting by those probabilities identifies the effect without measuring
  the confounder, because the logged probability is the true propensity (a
  balancing score). This is a design feature, not something observational
  data provide: it requires that treatment was actually drawn at random with
  the recorded probabilities (a stochastic targeting policy or an
  experiment). A logged model *score* used by a deterministic rule is not a
  propensity, and ordinary observational data record no assignment
  probabilities at all — there, regime C is unavailable.

Ground truth (``dataset.ground_truth``) is read only after estimation, to
evaluate recovery. Post-treatment variables (billing activity, tenure) are
used only in the explicitly labelled *invalid* adjustments.

Modelling limitation inherited from M2: the campaign is a time-invariant
customer attribute — the generator applies the effect to every treated
customer's retention regardless of signup date relative to the campaign
start. The estimand is therefore "effect of being a targeted customer",
not of an offer received at a particular time.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy import stats

from decision_intelligence.causal_inference.dag import AdjustmentAssessment, CausalDAG
from decision_intelligence.causal_inference.diagnostics import (
    OverlapSummary,
    WeightDiagnostics,
    balance_table,
    balancing_weights,
    overlap_summary,
    weight_diagnostics,
)
from decision_intelligence.causal_inference.estimators import (
    CausalData,
    CausalEstimate,
    HeterogeneityTest,
    RiskRatioEstimate,
    aipw_risk_ratio,
    augmented_ipw,
    estimate_propensity,
    heterogeneity_test,
    inverse_probability_weighting,
    naive_difference,
    regression_adjustment,
    subgroup_effects,
)
from decision_intelligence.causal_inference.sensitivity import (
    EValue,
    bounding_factor,
    e_value,
    risk_difference_bias_grid,
)
from decision_intelligence.common.seeding import spawn_generators
from decision_intelligence.common.validation import FloatArray
from decision_intelligence.data import customers as customers_module
from decision_intelligence.data import marketing as marketing_module
from decision_intelligence.data import retention as retention_module
from decision_intelligence.data.config import SyntheticDataConfig
from decision_intelligence.data.customers import horizon_end
from decision_intelligence.data.dataset import SyntheticDataset
from decision_intelligence.statistics.hypothesis_tests import wilson_interval
from decision_intelligence.statistics.logistic_regression import fit_logistic

TREATMENT = "treated"
OUTCOME = "retained_90d"

#: Analysis-frame columns → DAG node they measure.
COLUMN_TO_NODE: Mapping[str, str] = {
    "latent_value": "latent_value",
    "segment_regular": "segment",
    "segment_power": "segment",
    "price_sensitivity": "price_sensitivity",
    "channel_paid_search": "acquisition_channel",
    "channel_social": "acquisition_channel",
    "channel_referral": "acquisition_channel",
    "region_north": "region",
    "region_south": "region",
    "region_west": "region",
    "log_income": "household_income",
    "billing_transactions": "billing_transactions",
}

SEGMENT = ("segment_regular", "segment_power")
PROXIES = (
    *SEGMENT,
    "price_sensitivity",
    "channel_paid_search",
    "channel_social",
    "channel_referral",
    "region_north",
    "region_south",
    "region_west",
)
REGIME_A = ("latent_value", *SEGMENT)

#: Named adjustment strategies compared on the M2 data (valid and invalid).
ADJUSTMENT_STRATEGIES: Mapping[str, tuple[str, ...]] = {
    "none (naive)": (),
    "segment only": SEGMENT,
    "regime B: observed proxies": PROXIES,
    "regime A: latent_value": ("latent_value",),
    "regime A: latent_value + segment": REGIME_A,
    "INVALID: regime A + post-treatment billing count": (*REGIME_A, "billing_transactions"),
}


def campaign_dag() -> CausalDAG:
    """The DAG implied by the M2 generator code (customers, marketing, retention,
    subscriptions, transactions, experiments modules).

    ``billing_transactions`` is the *count* of billing periods, which the
    transactions generator derives from signup and churn dates only (price
    sensitivity changes billed amounts, not the count) — see
    :func:`billing_periods`, verified against the pipeline per customer.
    """
    return CausalDAG(
        [
            ("segment", "latent_value"),
            ("segment", "price_sensitivity"),
            ("latent_value", "price_sensitivity"),
            ("latent_value", "household_income"),
            ("latent_value", TREATMENT),
            ("segment", "churn_time"),
            ("latent_value", "churn_time"),
            (TREATMENT, "churn_time"),
            ("churn_time", OUTCOME),
            ("churn_time", "billing_transactions"),
            ("signup_date", "billing_transactions"),
            ("segment", "experiment_conversion"),
            ("latent_value", "experiment_conversion"),
        ],
        nodes=("acquisition_channel", "region"),
    )


def campaign_frame(dataset: SyntheticDataset) -> pd.DataFrame:
    """One row per customer: treatment, outcome, pre-treatment covariates and,
    clearly separated, post-treatment and logged-propensity columns.

    ``region`` is missing for ~2% of customers (M2 injects it completely at
    random); those customers fall into the reference level. Region has no
    edge in the generator DAG, so this coding cannot bias any estimate here.
    ``log_income`` keeps its missing values (used only as a negative-control
    outcome, on complete cases).
    """
    tables = dataset.tables
    frame = (
        tables["customers"]
        .merge(tables["customer_treatments"], on="customer_id")
        .merge(tables["customer_retention"], on="customer_id")
    )
    billing = tables["transactions"].groupby("customer_id").size().rename("billing_transactions")
    frame = frame.merge(billing, on="customer_id", how="left").fillna({"billing_transactions": 0})
    out = pd.DataFrame(
        {
            "customer_id": frame["customer_id"],
            TREATMENT: frame["treated"].astype(float),
            OUTCOME: frame["retained_90d"].astype(float),
            "latent_value": frame["latent_value"].astype(float),
            "price_sensitivity": frame["price_sensitivity"].astype(float),
            "log_income": np.log(frame["household_income_proxy"].astype(float)),
            # post-treatment (descendants of the offer) — invalid adjustment variables
            "billing_transactions": frame["billing_transactions"].astype(float),
            # logged assignment probability (regime C)
            "logged_propensity": frame["assignment_propensity"].astype(float),
        }
    )
    for level in ("regular", "power"):
        out[f"segment_{level}"] = (frame["segment"] == level).astype(float)
    for level in ("paid_search", "social", "referral"):
        out[f"channel_{level}"] = (frame["acquisition_channel"] == level).astype(float)
    for level in ("north", "south", "west"):
        out[f"region_{level}"] = (frame["region"] == level).astype(float)
    out["segment"] = frame["segment"].astype(str)
    return out


def causal_data(frame: pd.DataFrame, columns: Sequence[str], outcome: str = OUTCOME) -> CausalData:
    return CausalData.from_arrays(frame[TREATMENT], frame[outcome], frame[list(columns)])


def assess_strategy(columns: Sequence[str]) -> AdjustmentAssessment:
    """Backdoor assessment of a column set, mapped onto DAG nodes."""
    nodes = sorted({COLUMN_TO_NODE[c] for c in columns})
    return campaign_dag().assess_adjustment_set(TREATMENT, OUTCOME, nodes)


# ---------------------------------------------------------------------------
# Full analysis of one dataset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategyResult:
    strategy: str
    dag_valid: bool
    dag_explanation: str
    estimates: tuple[CausalEstimate, ...]


@dataclass(frozen=True)
class FalsificationTest:
    """Likelihood-ratio test of a conditional independence the DAG implies."""

    claim: str
    lr_statistic: float
    df: int
    p_value: float


@dataclass(frozen=True)
class CampaignAnalysis:
    n_customers: int
    treated_share: float
    strategies: tuple[StrategyResult, ...]
    att_regime_a: CausalEstimate
    known_propensity_ipw: CausalEstimate
    overlap: OverlapSummary
    weights: WeightDiagnostics
    balance: pd.DataFrame
    estimated_vs_logged_propensity_max_abs_diff: float
    falsification: FalsificationTest
    negative_control: tuple[CausalEstimate, ...]
    selection_on_billing: CausalEstimate
    selection_share_kept: float
    subgroup_effects: dict[str, CausalEstimate]
    subgroup_heterogeneity: HeterogeneityTest
    regime_b_risk_ratio: RiskRatioEstimate
    regime_b_e_value: EValue
    confounder_benchmarks: tuple[ConfounderBenchmark, ...]
    bias_grid: pd.DataFrame
    true_effect: float = field(default=math.nan)


def analyse_campaign(
    dataset: SyntheticDataset, *, rng: np.random.Generator, n_bootstrap: int = 500
) -> CampaignAnalysis:
    """Run every M4 analysis on one M2 build. Ground truth is attached last."""
    frame = campaign_frame(dataset)
    strategies: list[StrategyResult] = []
    for name, columns in ADJUSTMENT_STRATEGIES.items():
        data = causal_data(frame, columns)
        assessment = assess_strategy(columns)
        if not columns:
            estimates: tuple[CausalEstimate, ...] = (naive_difference(data),)
        else:
            estimates = (
                regression_adjustment(data, rng=rng, n_bootstrap=n_bootstrap),
                inverse_probability_weighting(data),
                augmented_ipw(data),
            )
        strategies.append(
            StrategyResult(name, assessment.valid, assessment.explanation, estimates)
        )

    data_a = causal_data(frame, REGIME_A)
    propensity_a = estimate_propensity(data_a)
    weights_a = balancing_weights(propensity_a, data_a.treatment)
    balance_frame = frame[["latent_value", *PROXIES]].reset_index(drop=True)
    balance = balance_table(data_a, weights_a, covariates=balance_frame)

    # DAG-implied independence: treated ⊥ (segment, price sensitivity, channel, region) | latent.
    falsification = _lr_independence_test(
        frame, base=("latent_value",), extra=PROXIES,
        claim="treated ⊥ segment, price_sensitivity, channel, region | latent_value",
    )

    # Negative-control outcome: log income is pre-treatment (cannot be affected by the offer)
    # but shares the confounder latent_value.
    complete = frame[frame["log_income"].notna()].reset_index(drop=True)
    negative_control = tuple(
        _relabel(estimate, f"negative control (log income), {label}")
        for label, estimate in (
            ("naive", naive_difference(causal_data(complete, (), "log_income"))),
            ("regime A", augmented_ipw(causal_data(complete, REGIME_A, "log_income"))),
            ("regime B", augmented_ipw(causal_data(complete, PROXIES, "log_income"))),
        )
    )

    # Selection on a post-treatment variable: keep only customers billed at least once.
    billed = frame[frame["billing_transactions"] > 0].reset_index(drop=True)
    selection = _relabel(
        augmented_ipw(causal_data(billed, REGIME_A)),
        "INVALID: regime A among customers billed at least once (post-treatment selection)",
    )

    subgroups = subgroup_effects(
        causal_data(frame, ("latent_value",)), frame["segment"].to_numpy()
    )
    data_b = causal_data(frame, PROXIES)
    rr_b = aipw_risk_ratio(data_b)
    regime_b = next(s for s in strategies if s.strategy.startswith("regime B")).estimates[-1]

    analysis = CampaignAnalysis(
        n_customers=len(frame),
        treated_share=float(frame[TREATMENT].mean()),
        strategies=tuple(strategies),
        att_regime_a=augmented_ipw(data_a, estimand="ATT"),
        known_propensity_ipw=inverse_probability_weighting(
            causal_data(frame, ()), propensity=frame["logged_propensity"].to_numpy()
        ),
        overlap=overlap_summary(propensity_a, data_a.treatment),
        weights=weight_diagnostics(weights_a, data_a.treatment),
        balance=balance,
        estimated_vs_logged_propensity_max_abs_diff=float(
            np.max(np.abs(propensity_a - frame["logged_propensity"].to_numpy()))
        ),
        falsification=falsification,
        negative_control=negative_control,
        selection_on_billing=selection,
        selection_share_kept=len(billed) / len(frame),
        subgroup_effects=subgroups,
        subgroup_heterogeneity=heterogeneity_test(subgroups),
        regime_b_risk_ratio=rr_b,
        regime_b_e_value=e_value(rr_b.risk_ratio, rr_b.ci_lower, rr_b.ci_upper),
        confounder_benchmarks=tuple(
            confounder_benchmark(frame, rr_b.risk_ratio, n_categories=k) for k in (2, 4, 10)
        ),
        bias_grid=risk_difference_bias_grid(
            regime_b.estimate,
            regime_b.ci_lower,
            regime_b.ci_upper,
            outcome_effects=(0.05, 0.1, 0.2, 0.3),
            prevalence_differences=(0.1, 0.2, 0.3, 0.4),
        ),
    )
    # Evaluation only: attach the known design effect after all estimation is complete.
    return replace(analysis, true_effect=dataset.ground_truth.marketing.true_absolute_effect)


def _relabel(estimate: CausalEstimate, label: str) -> CausalEstimate:
    return replace(estimate, estimand=f"{estimate.estimand} — {label}")


def _lr_independence_test(
    frame: pd.DataFrame, *, base: Sequence[str], extra: Sequence[str], claim: str
) -> FalsificationTest:
    restricted = fit_logistic(frame[list(base)], frame[TREATMENT])
    full = fit_logistic(frame[[*base, *extra]], frame[TREATMENT])
    statistic = 2 * (full.log_likelihood - restricted.log_likelihood)
    return FalsificationTest(
        claim=claim,
        lr_statistic=statistic,
        df=len(extra),
        p_value=float(stats.chi2.sf(statistic, len(extra))),
    )


@dataclass(frozen=True)
class ConfounderBenchmark:
    """Strength of the actual omitted confounder (``latent_value``) on the E-value scale.

    ``latent_value`` is categorised into ``n_categories`` quantile bins ``u``:

    * ``rr_exposure_confounder`` = max_u P(U=u | T=1) / P(U=u | T=0)  (RR_EU);
    * ``rr_confounder_outcome`` = max over arms of max_u P(Y=1 | U=u, T) / min_u
      P(Y=1 | U=u, T)  (RR_UD);
    * ``bounding_factor`` = RR_EU RR_UD / (RR_EU + RR_UD - 1) (Ding & VanderWeele 2016):
      the most this confounder could inflate the risk ratio;
    * ``risk_ratio_lower_bound`` = observed RR / bounding factor.

    Possible only because the data are synthetic. These are *marginal*
    associations (the E-value refers to associations conditional on the
    measured covariates), coarse bins understate a continuous confounder's
    strength, and with many bins the max/min ratios are inflated by
    sampling noise — so the benchmark is indicative, not exact.
    """

    n_categories: int
    rr_exposure_confounder: float
    rr_confounder_outcome: float
    bounding_factor: float
    risk_ratio_lower_bound: float


def confounder_benchmark(
    frame: pd.DataFrame, observed_rr: float, *, n_categories: int
) -> ConfounderBenchmark:
    bins = pd.qcut(frame["latent_value"], n_categories, labels=False).to_numpy()
    treated = frame[TREATMENT].to_numpy() == 1
    levels = np.unique(bins)
    prevalence_treated = np.array([np.mean(bins[treated] == u) for u in levels])
    prevalence_control = np.array([np.mean(bins[~treated] == u) for u in levels])
    rr_eu = float(np.max(prevalence_treated / prevalence_control))
    outcome = frame[OUTCOME].to_numpy()
    rr_ud = 1.0
    for arm in (True, False):
        risks = np.array([outcome[(treated == arm) & (bins == u)].mean() for u in levels])
        rr_ud = max(rr_ud, float(np.max(risks) / np.min(risks)))
    factor = bounding_factor(rr_eu, rr_ud)
    return ConfounderBenchmark(
        n_categories=n_categories,
        rr_exposure_confounder=rr_eu,
        rr_confounder_outcome=rr_ud,
        bounding_factor=factor,
        risk_ratio_lower_bound=observed_rr / factor,
    )


# ---------------------------------------------------------------------------
# Repeated sampling over independent M2 replicates
# ---------------------------------------------------------------------------


def simulate_campaign_replicate(config: SyntheticDataConfig, seed: int) -> pd.DataFrame:
    """One fresh draw of the M2 customer → offer → retention process.

    Calls the M2 stage generators directly (the same code the pipeline
    runs) with independent spawned generators; only the tables the
    campaign analysis needs are produced, so many replicates are cheap.
    """
    rng_customers, rng_offer, rng_retention = spawn_generators(seed, 3)
    customers = customers_module.generate_customers(config, rng_customers)
    treatments = marketing_module.generate_customer_treatments(config, customers, rng_offer)
    retention, _ = retention_module.generate_customer_retention(
        config, customers, treatments, rng_retention
    )
    frame = customers.merge(treatments, on="customer_id").merge(retention, on="customer_id")
    out = pd.DataFrame(
        {
            TREATMENT: frame["treated"].astype(float),
            OUTCOME: frame["retained_90d"].astype(float),
            "latent_value": frame["latent_value"].astype(float),
            "price_sensitivity": frame["price_sensitivity"].astype(float),
            "log_income": np.log(frame["household_income_proxy"].astype(float)),
            "billing_transactions": billing_periods(frame, config),
            "logged_propensity": frame["assignment_propensity"].astype(float),
        }
    )
    for level in ("regular", "power"):
        out[f"segment_{level}"] = (frame["segment"] == level).astype(float)
    for level in ("paid_search", "social", "referral"):
        out[f"channel_{level}"] = (frame["acquisition_channel"] == level).astype(float)
    for level in ("north", "south", "west"):
        out[f"region_{level}"] = (frame["region"] == level).astype(float)
    return out


def billing_periods(frame: pd.DataFrame, config: SyntheticDataConfig) -> FloatArray:
    """Number of 30-day billing periods per customer, by the rule in data/transactions.py.

    A subscription bills once per completed 30-day period between signup and
    churn (or the data horizon). Reproduces the pipeline's per-customer
    transaction counts exactly (checked in tests) without generating the
    transactions table. A *post-treatment* quantity: it depends on churn.
    """
    end = horizon_end(config)
    churn_date = pd.to_datetime(frame["churn_date"])
    active_end = churn_date.where(frame["churned"].astype(bool), end)
    days = (active_end - pd.to_datetime(frame["signup_date"])).dt.days
    return np.asarray(np.maximum(days // 30, 0), dtype=np.float64)


def _replicate_estimates(
    frame: pd.DataFrame, effect: float
) -> dict[str, tuple[CausalEstimate, float]]:
    """Estimator name -> (estimate, the true value of *its* estimand)."""
    data_a = causal_data(frame, REGIME_A)
    data_b = causal_data(frame, PROXIES)
    propensity_a = estimate_propensity(data_a)
    propensity_b = estimate_propensity(data_b)
    billed = frame[frame["billing_transactions"] > 0].reset_index(drop=True)
    return {
        "naive": (naive_difference(data_a), effect),
        "IPW (Hájek), regime A": (inverse_probability_weighting(data_a), effect),
        "AIPW, regime A": (augmented_ipw(data_a), effect),
        "AIPW ATT, regime A": (augmented_ipw(data_a, estimand="ATT"), effect),
        "AIPW, regime B (proxies)": (augmented_ipw(data_b), effect),
        "IPW, regime C (logged propensity)": (
            inverse_probability_weighting(
                causal_data(frame, ()), propensity=frame["logged_propensity"].to_numpy()
            ),
            effect,
        ),
        "AIPW, propensity correct / outcome model wrong": (
            augmented_ipw(data_b, propensity=propensity_a),
            effect,
        ),
        "AIPW, propensity wrong / outcome model correct": (
            augmented_ipw(data_a, propensity=propensity_b),
            effect,
        ),
        "INVALID: AIPW regime A + post-treatment billing count": (
            augmented_ipw(causal_data(frame, (*REGIME_A, "billing_transactions"))),
            effect,
        ),
        "INVALID: AIPW regime A among billed customers": (
            augmented_ipw(causal_data(billed, REGIME_A)),
            effect,
        ),
        "negative control (log income), AIPW regime A": (
            augmented_ipw(causal_data(frame, REGIME_A, "log_income")),
            0.0,
        ),
        "negative control (log income), AIPW regime B": (
            augmented_ipw(causal_data(frame, PROXIES, "log_income")),
            0.0,
        ),
    }


@dataclass(frozen=True)
class ReplicateSummary:
    estimator: str
    truth: float
    mean_estimate: float
    bias: float
    bias_mc_se: float
    empirical_sd: float
    mean_reported_se: float
    rmse: float
    coverage: float
    coverage_ci_lower: float
    coverage_ci_upper: float
    coverage_p_value_vs_nominal: float
    """Exact two-sided binomial test of coverage = 0.95."""


def campaign_monte_carlo(
    config: SyntheticDataConfig, *, n_replications: int, first_seed: int = 10_000
) -> list[ReplicateSummary]:
    """Bias, variability and CI coverage of each estimator across independent M2 draws.

    Each estimator is scored against the truth of *its own* estimand: the
    design effect ``config.marketing_true_effect`` (ATE = ATT = 0.08 by
    construction) for effect estimators, 0 for the negative-control outcome.
    The invalid strategies are scored against 0.08 because that is the
    question they purport to answer. Regression adjustment is omitted only
    because its bootstrap interval is too costly to repeat here.
    """
    if n_replications < 2:
        raise ValueError("n_replications must be at least 2")
    collected: dict[str, list[CausalEstimate]] = {}
    truths: dict[str, float] = {}
    for replicate in range(n_replications):
        frame = simulate_campaign_replicate(config, first_seed + replicate)
        replicate_results = _replicate_estimates(frame, config.marketing_true_effect)
        for name, (estimate, truth) in replicate_results.items():
            collected.setdefault(name, []).append(estimate)
            truths[name] = truth
    summaries: list[ReplicateSummary] = []
    for name, results in collected.items():
        truth = truths[name]
        values = np.array([e.estimate for e in results])
        sd = float(values.std(ddof=1))
        hits = int(sum(e.covers(truth) for e in results))
        low, high = wilson_interval(hits, len(results))
        summaries.append(
            ReplicateSummary(
                estimator=name,
                truth=truth,
                mean_estimate=float(values.mean()),
                bias=float(values.mean() - truth),
                bias_mc_se=sd / math.sqrt(values.size),
                empirical_sd=sd,
                mean_reported_se=float(np.mean([e.standard_error for e in results])),
                rmse=float(np.sqrt(np.mean((values - truth) ** 2))),
                coverage=hits / len(results),
                coverage_ci_lower=low,
                coverage_ci_upper=high,
                coverage_p_value_vs_nominal=float(
                    stats.binomtest(hits, len(results), 0.95).pvalue
                ),
            )
        )
    return summaries
