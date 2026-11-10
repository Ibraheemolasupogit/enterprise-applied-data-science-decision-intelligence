"""Statistical-foundations methods applied to real analytical questions on the M2 data.

Each function answers one question a NexaMarket analyst could plausibly be
asked, using only columns a real pipeline would have. Two guard rails:

* **No leakage of evaluation-only knowledge.** ``latent_value`` is the
  generator's hidden driver (see data/customers.py) and is never used as a
  feature here. ``dataset.ground_truth`` is read *only after* estimation,
  to report whether an estimate recovered the known design value.
* **No unsupported causal claims.** Only the randomised experiment
  supports a causal reading of a group difference. Every other comparison
  is labelled associational; confounding adjustment is M4/M5 scope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from decision_intelligence.common.bootstrap import (
    BootstrapResult,
    bootstrap_ci,
    bootstrap_two_sample_ci,
)
from decision_intelligence.common.validation import FloatArray
from decision_intelligence.data.config import SyntheticDataConfig
from decision_intelligence.data.dataset import SyntheticDataset
from decision_intelligence.data.pipeline import build_synthetic_dataset
from decision_intelligence.statistics.descriptive import DescriptiveSummary, describe
from decision_intelligence.statistics.effect_sizes import (
    BinaryEffectSizes,
    EffectSize,
    binary_effect_sizes,
    cliffs_delta,
    hedges_g,
)
from decision_intelligence.statistics.estimators import t_interval_for_mean
from decision_intelligence.statistics.hypothesis_tests import (
    HypothesisTestResult,
    mann_whitney_u,
    two_proportion_z_test,
    two_sample_t_test,
    wilson_interval,
)
from decision_intelligence.statistics.linear_regression import fit_ols
from decision_intelligence.statistics.logistic_regression import (
    ClassificationDiagnostics,
    LogisticRegressionResult,
    classification_diagnostics,
    fit_logistic,
)
from decision_intelligence.statistics.missing_data import impute, missingness_mask
from decision_intelligence.statistics.power import (
    power_two_proportions,
    sample_size_two_proportions,
)

# ---------------------------------------------------------------------------
# 1. Randomised experiment: did the recommendation feature raise conversion?
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentConversionAnalysis:
    """Two-arm conversion comparison for the randomised recommendation experiment.

    ``design_power`` is the power the realised arm sizes give to detect the
    pre-specified ``minimum_effect`` — a design quantity, not post-hoc
    power. ``true_effect`` and ``ci_covers_true_effect`` come from the
    evaluation-only ground truth and are reported, never used.
    """

    n_control: int
    n_treatment: int
    control_rate: float
    treatment_rate: float
    control_rate_ci: tuple[float, float]
    treatment_rate_ci: tuple[float, float]
    test: HypothesisTestResult
    effect_sizes: BinaryEffectSizes
    bootstrap_difference: BootstrapResult
    minimum_effect: float
    design_power: float
    required_n_per_arm: int
    true_effect: float
    ci_covers_true_effect: bool


def experiment_conversion_analysis(
    dataset: SyntheticDataset,
    *,
    rng: np.random.Generator,
    minimum_effect: float = 0.04,
    alpha: float = 0.05,
) -> ExperimentConversionAnalysis:
    """Analyse treatment vs control conversion (causal reading justified by randomisation)."""
    merged = dataset.tables["experiment_assignments"].merge(
        dataset.tables["experiment_outcomes"], on=["customer_id", "experiment_name"]
    )
    treatment = merged.loc[merged["variant"] == "treatment", "converted"].to_numpy(dtype=float)
    control = merged.loc[merged["variant"] == "control", "converted"].to_numpy(dtype=float)
    s_t, n_t = int(treatment.sum()), treatment.size
    s_c, n_c = int(control.sum()), control.size

    test = two_proportion_z_test(s_t, n_t, s_c, n_c, confidence_level=1 - alpha)
    bootstrap = bootstrap_two_sample_ci(
        treatment, control, _mean_difference, rng=rng, method="percentile", n_resamples=2000
    )
    control_rate = s_c / n_c
    true_effect = dataset.ground_truth.experiment.true_absolute_effect
    if test.ci_lower is None or test.ci_upper is None:  # pragma: no cover - always set
        raise RuntimeError("two-proportion test returned no confidence interval")
    return ExperimentConversionAnalysis(
        n_control=n_c,
        n_treatment=n_t,
        control_rate=control_rate,
        treatment_rate=s_t / n_t,
        control_rate_ci=wilson_interval(s_c, n_c),
        treatment_rate_ci=wilson_interval(s_t, n_t),
        test=test,
        effect_sizes=binary_effect_sizes(s_t, n_t, s_c, n_c),
        bootstrap_difference=bootstrap,
        minimum_effect=minimum_effect,
        design_power=power_two_proportions(
            control_rate + minimum_effect, control_rate, n_t, n_c, alpha=alpha
        ),
        required_n_per_arm=sample_size_two_proportions(
            control_rate + minimum_effect, control_rate, power=0.8, alpha=alpha
        )[0],
        true_effect=true_effect,
        ci_covers_true_effect=test.ci_lower <= true_effect <= test.ci_upper,
    )


def _mean_difference(a: FloatArray, b: FloatArray) -> float:
    return float(a.mean() - b.mean())


# ---------------------------------------------------------------------------
# 2. Revenue distribution: how skewed is customer revenue, and do segments differ?
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RevenueAnalysis:
    """Skewed-outcome inference: t vs bootstrap intervals, parametric vs rank tests.

    The segment comparison is associational: segments differ in many ways
    at once (price plan, tenure, latent value).
    """

    revenue_summary: DescriptiveSummary
    mean_t_interval: tuple[float, float]
    mean_bootstrap: BootstrapResult
    median_bootstrap: BootstrapResult
    power_vs_casual_welch: HypothesisTestResult
    power_vs_casual_mann_whitney: HypothesisTestResult
    power_vs_casual_hedges_g: EffectSize
    power_vs_casual_cliffs_delta: EffectSize


def revenue_analysis(dataset: SyntheticDataset, *, rng: np.random.Generator) -> RevenueAnalysis:
    """Per-customer total billed revenue over the data horizon.

    Customers with no transactions (signed up within the final billing
    period) are included with zero revenue — dropping them would bias the
    mean upwards.
    """
    customers = dataset.tables["customers"][["customer_id", "segment"]]
    totals = (
        dataset.tables["transactions"].groupby("customer_id")["amount"].sum().rename("revenue")
    )
    revenue = customers.merge(totals, on="customer_id", how="left").fillna({"revenue": 0.0})
    values = revenue["revenue"].to_numpy(dtype=float)
    power = revenue.loc[revenue["segment"] == "power", "revenue"].to_numpy(dtype=float)
    casual = revenue.loc[revenue["segment"] == "casual", "revenue"].to_numpy(dtype=float)
    return RevenueAnalysis(
        revenue_summary=describe(values),
        mean_t_interval=t_interval_for_mean(values),
        mean_bootstrap=bootstrap_ci(values, _mean, rng=rng, method="bca"),
        median_bootstrap=bootstrap_ci(values, _median, rng=rng, method="percentile"),
        power_vs_casual_welch=two_sample_t_test(power, casual),
        power_vs_casual_mann_whitney=mann_whitney_u(power, casual),
        power_vs_casual_hedges_g=hedges_g(power, casual),
        power_vs_casual_cliffs_delta=cliffs_delta(power, casual),
    )


def _mean(values: FloatArray) -> float:
    return float(np.mean(values))


def _median(values: FloatArray) -> float:
    return float(np.median(values))


# ---------------------------------------------------------------------------
# 3. Retention: which observable customer attributes are associated with
#    90-day retention, and is a simple model better than the base rate?
# ---------------------------------------------------------------------------

RETENTION_FEATURES = (
    "segment_regular",
    "segment_power",
    "price_sensitivity",
    "channel_paid_search",
    "channel_social",
    "channel_referral",
)


@dataclass(frozen=True)
class RetentionModelAnalysis:
    """Logistic model vs base-rate baseline, evaluated on a held-out customer split."""

    n_train: int
    n_test: int
    model: LogisticRegressionResult
    model_test_diagnostics: ClassificationDiagnostics
    baseline_test_brier: float
    baseline_test_log_loss: float


def retention_feature_frame(dataset: SyntheticDataset) -> pd.DataFrame:
    """Observable, pre-outcome customer attributes with dummy coding.

    Reference levels: segment ``casual``, channel ``organic``. Excludes
    ``latent_value`` (hidden), ``household_income_proxy`` (has injected
    missingness) and campaign treatment (a confounded post-signup
    exposure whose effect is M4/M5 scope).
    """
    customers = dataset.tables["customers"]
    retention = dataset.tables["customer_retention"][["customer_id", "retained_90d"]]
    frame = customers.merge(retention, on="customer_id")
    return pd.DataFrame(
        {
            "customer_id": frame["customer_id"],
            "segment_regular": (frame["segment"] == "regular").astype(float),
            "segment_power": (frame["segment"] == "power").astype(float),
            "price_sensitivity": frame["price_sensitivity"].astype(float),
            "channel_paid_search": (frame["acquisition_channel"] == "paid_search").astype(float),
            "channel_social": (frame["acquisition_channel"] == "social").astype(float),
            "channel_referral": (frame["acquisition_channel"] == "referral").astype(float),
            "retained_90d": frame["retained_90d"].astype(float),
        }
    )


def retention_model_analysis(
    dataset: SyntheticDataset, *, rng: np.random.Generator, test_fraction: float = 0.3
) -> RetentionModelAnalysis:
    """Fit on a random customer split, evaluate on held-out customers only.

    The split is by customer (one row per customer here), so no customer
    contributes to both fitting and evaluation. The baseline predicts the
    *training* prevalence for everyone — it uses no test-set information.
    """
    frame = retention_feature_frame(dataset)
    n = len(frame)
    test_mask = np.zeros(n, dtype=bool)
    test_mask[rng.choice(n, size=int(round(test_fraction * n)), replace=False)] = True
    train, test = frame[~test_mask], frame[test_mask]
    features = list(RETENTION_FEATURES)

    model = fit_logistic(train[features], train["retained_90d"].to_numpy())
    test_probabilities = model.predict_probability(test[features].to_numpy())
    y_test = test["retained_90d"].to_numpy()
    prevalence = float(train["retained_90d"].mean())
    return RetentionModelAnalysis(
        n_train=len(train),
        n_test=len(test),
        model=model,
        model_test_diagnostics=classification_diagnostics(y_test, test_probabilities),
        baseline_test_brier=float(np.mean((prevalence - y_test) ** 2)),
        baseline_test_log_loss=float(
            -np.mean(y_test * math.log(prevalence) + (1 - y_test) * math.log(1 - prevalence))
        ),
    )


# ---------------------------------------------------------------------------
# 4. Dependence: what happens to standard errors when transactions are
#    treated as independent although they are repeated per customer?
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClusteredTransactionAnalysis:
    """Same OLS point estimates, three covariance estimators."""

    n_transactions: int
    n_customers: int
    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    classical_se: tuple[float, ...]
    hc3_se: tuple[float, ...]
    cluster_se: tuple[float, ...]


def clustered_transaction_analysis(dataset: SyntheticDataset) -> ClusteredTransactionAnalysis:
    """Transaction amount on customer segment and price sensitivity.

    Every transaction of a customer shares that customer's plan price, so
    residuals are strongly correlated within customer. Classical and HC
    standard errors assume independent rows and are therefore too small;
    the customer-clustered SE is the appropriate one. Coefficients are
    associations between customer attributes and billed amounts.
    """
    customers = dataset.tables["customers"][["customer_id", "segment", "price_sensitivity"]]
    frame = dataset.tables["transactions"][["customer_id", "amount"]].merge(
        customers, on="customer_id"
    )
    X = pd.DataFrame(
        {
            "segment_regular": (frame["segment"] == "regular").astype(float),
            "segment_power": (frame["segment"] == "power").astype(float),
            "price_sensitivity": frame["price_sensitivity"].astype(float),
        }
    )
    y = frame["amount"].to_numpy(dtype=float)
    classical = fit_ols(X, y)
    hc3 = fit_ols(X, y, covariance="HC3")
    clustered = fit_ols(X, y, covariance="cluster", clusters=frame["customer_id"].to_numpy())
    return ClusteredTransactionAnalysis(
        n_transactions=len(frame),
        n_customers=int(frame["customer_id"].nunique()),
        feature_names=classical.feature_names,
        coefficients=tuple(float(v) for v in classical.coefficients),
        classical_se=tuple(float(v) for v in classical.standard_errors),
        hc3_se=tuple(float(v) for v in hc3.standard_errors),
        cluster_se=tuple(float(v) for v in clustered.standard_errors),
    )


# ---------------------------------------------------------------------------
# 5. Missing data: is the mean income estimate trustworthy given missingness?
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IncomeMissingnessAnalysis:
    """Known-truth check of missing-data handling on ``household_income_proxy``.

    ``full_mean_log_income`` comes from a twin build with the same seed but
    no injected missingness — available only because the data are
    synthetic. All quantities are on the log-income scale.
    """

    m2_missing_rate: float
    full_mean_log_income: float
    m2_complete_case_mean: float
    m2_complete_case_se: float
    mar_missing_rate: float
    mar_complete_case_mean: float
    mar_regression_imputation_mean: float
    covariate_correlation: float


def income_missingness_analysis(
    config: SyntheticDataConfig, *, rng: np.random.Generator, mar_rate: float = 0.3
) -> IncomeMissingnessAnalysis:
    """Compare complete-case and regression-imputation means with the known full-data mean.

    1. The M2 pipeline injects MCAR missingness, so the complete-case mean
       should agree with the full-data mean within sampling error.
    2. A MAR mechanism driven by the observed ``price_sensitivity`` (which
       is associated with income through segment) is then imposed on the
       complete column: the complete-case mean becomes biased and
       regression imputation on ``price_sensitivity`` largely repairs it.
    """
    observed_build = build_synthetic_dataset(config)
    clean_quality = replace(config.quality, missing_income_rate=0.0, missing_region_rate=0.0)
    full_build = build_synthetic_dataset(replace(config, quality=clean_quality))
    _check_same_population(observed_build, full_build)

    observed_income = np.log(
        observed_build.tables["customers"]["household_income_proxy"].to_numpy(dtype=float)
    )
    full_income = np.log(
        full_build.tables["customers"]["household_income_proxy"].to_numpy(dtype=float)
    )
    sensitivity = full_build.tables["customers"]["price_sensitivity"].to_numpy(dtype=float)
    observed = ~np.isnan(observed_income)

    mask = missingness_mask(
        full_income, "MAR", mar_rate, rng, covariate=-sensitivity, strength=1.5
    )
    complete_case = impute(full_income, mask, "complete_case", rng)
    regression = impute(full_income, mask, "regression_imputation", rng, covariate=sensitivity)
    return IncomeMissingnessAnalysis(
        m2_missing_rate=float(1 - observed.mean()),
        full_mean_log_income=float(full_income.mean()),
        m2_complete_case_mean=float(observed_income[observed].mean()),
        m2_complete_case_se=float(observed_income[observed].std(ddof=1))
        / math.sqrt(int(observed.sum())),
        mar_missing_rate=float(mask.mean()),
        mar_complete_case_mean=float(complete_case.mean()),
        mar_regression_imputation_mean=float(regression.mean()),
        covariate_correlation=float(np.corrcoef(full_income, sensitivity)[0, 1]),
    )


def _check_same_population(a: SyntheticDataset, b: SyntheticDataset) -> None:
    columns = ["customer_id", "segment", "price_sensitivity"]
    if not a.tables["customers"][columns].equals(b.tables["customers"][columns]):
        raise RuntimeError("twin builds diverged; missingness comparison would be invalid")

