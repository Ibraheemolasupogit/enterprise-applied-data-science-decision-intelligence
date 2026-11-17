"""Economic definitions and retention estimators against independent references (M10)."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

from decision_intelligence.customer_value import protocol as pr
from decision_intelligence.customer_value.economics import (
    Estimate,
    channel_cac,
    curve_value,
    discount_factors,
    geometric_value,
    monthly_rate,
    payback_age,
)
from decision_intelligence.customer_value.protocol import FRESH_SEED as RESERVED_FRESH
from decision_intelligence.customer_value.protocol import MAIN_SEED as RESERVED_MAIN
from decision_intelligence.customer_value.protocol import PILOT_SEED as RESERVED_PILOT
from decision_intelligence.customer_value.retention import kaplan_meier

PACKAGE = Path(pr.__file__).resolve().parent
TESTS = Path(__file__).resolve().parents[2]


def test_monthly_rate_and_discount_factors() -> None:
    r = monthly_rate(0.10)
    assert (1 + r) ** 12 == pytest.approx(1.10, rel=1e-14)
    assert r == pytest.approx(pr.MONTHLY_DISCOUNT_RATE)
    d = discount_factors(5, r)
    assert d[0] == 1.0 and np.allclose(d, [(1 + r) ** -a for a in range(5)])
    with pytest.raises(ValueError):
        monthly_rate(-1.0)


@pytest.mark.parametrize("q,r", [(0.95, 0.008), (0.7, 0.0), (1.0, 0.01), (0.0, 0.01)])
def test_geometric_value_matches_brute_force(q: float, r: float) -> None:
    for horizon in (1, 12, 36):
        brute = sum(5.0 * q ** a * (1 + r) ** -a for a in range(horizon))
        assert geometric_value(5.0, q, r, horizon).value == pytest.approx(brute, rel=1e-12)
    inf = geometric_value(5.0, q, r, None)
    long_sum = sum(5.0 * q ** a * (1 + r) ** -a for a in range(20_000))
    assert inf.status == "ok" and inf.value == pytest.approx(long_sum, rel=1e-9)


def test_infinite_horizon_is_undefined_without_churn_or_discounting() -> None:
    e = geometric_value(5.0, 1.0, 0.0, None)
    assert e.status == "undefined" and math.isinf(e.value)
    assert geometric_value(5.0, 1.0, 0.0, 10).value == pytest.approx(50.0)
    assert geometric_value(float("nan"), 0.9, 0.01, None).status == "undefined"
    with pytest.raises(ValueError):
        Estimate(1.0, "maybe")


def test_curve_value_and_payback_against_brute_force() -> None:
    m = np.array([-3.0, 2.0, 5.0, 5.0, 5.0])
    p = np.array([1.0, 0.9, 0.8, 0.7, 0.6])
    r = 0.01
    assert curve_value(m, p, r) == pytest.approx(sum(m[a] * p[a] / 1.01 ** a for a in range(5)))
    cum = np.cumsum(m * p)
    assert payback_age(cum, 5.0) == next(a for a in range(5) if cum[a] >= 5.0)
    assert payback_age(cum, 1e9) is None


def test_channel_cac_matches_a_pandas_recomputation() -> None:
    mk = pd.DataFrame({"month": [0, 0, 1, 1], "channel": ["a", "b", "a", "b"],
                       "new_customers": [10, 30, 20, 0], "media_spend": [100.0, 600.0, 220.0, 50.0],
                       "onboarding_cost": [60.0, 180.0, 120.0, 0.0]})
    oh = pd.DataFrame({"month": [0, 1], "marketing_overhead": [400.0, 300.0]})
    cac = {c.channel: c for c in channel_cac(mk, oh, [0, 1])}
    assert cac["a"].media == pytest.approx(320 / 30)
    # overhead: month 0 split 10/40 to a; month 1 all to a
    assert cac["a"].fully_loaded == pytest.approx((320 + 180 + 100 + 300) / 30)
    assert cac["b"].media == pytest.approx(650 / 30)


def test_kaplan_meier_reproduces_the_published_6mp_example() -> None:
    # Freireich et al. (1963), 6-MP arm; KM and Greenwood SE as tabulated by Kleinbaum & Klein.
    t = [6, 6, 6, 6, 7, 9, 10, 10, 11, 13, 16, 17, 19, 20, 22, 23, 25, 32, 32, 34, 35]
    e = [1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    km = kaplan_meier(t, e)
    assert km.times.tolist() == [6, 7, 10, 13, 16, 22, 23]
    assert np.round(km.survival, 3).tolist() == [0.857, 0.807, 0.753, 0.690, 0.627, 0.538, 0.448]
    assert np.round(km.se, 3).tolist() == [0.076, 0.087, 0.096, 0.107, 0.114, 0.128, 0.135]


def test_kaplan_meier_without_censoring_is_the_empirical_survival() -> None:
    t = np.random.default_rng(1).integers(1, 20, 300)
    km = kaplan_meier(t, np.ones_like(t, dtype=bool))
    for a in (0, 3, 7, 15, 25):
        assert km.at([a])[0] == pytest.approx(np.mean(t > a), abs=1e-12)
    lo, hi = km.interval(kind="log-log")
    assert np.all(lo <= km.survival + 1e-12) and np.all(km.survival <= hi + 1e-12)
    with pytest.raises(ValueError):
        kaplan_meier([-1, 2], [1, 1])


def test_population_survival_is_the_gamma_laplace_transform() -> None:
    b, v = 0.07, 0.5
    for a in (1, 6, 24):
        def integrand(z: float, a: int = a) -> float:
            return math.exp(-b * z * a) * float(stats.gamma.pdf(z, 1 / v, scale=v))

        val, _ = integrate.quad(integrand, 0, np.inf)
        assert (1 + v * b * a) ** (-1 / v) == pytest.approx(val, rel=1e-9)


@pytest.mark.parametrize("module", ["economics", "panel", "retention", "historical",
                                    "heuristic", "backtest", "protocol"])
def test_estimators_never_read_ground_truth(module: str) -> None:
    tree = ast.parse((PACKAGE / f"{module}.py").read_text())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert not names & {"truth", "EconomicsTruth", "EconomicsScenario"}


def test_no_test_uses_the_reserved_seeds() -> None:
    reserved = {RESERVED_MAIN, RESERVED_PILOT, RESERVED_FRESH}
    offenders = []
    for path in TESTS.rglob("test_*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and node.value in reserved:
                offenders.append(str(path))
            if isinstance(node, ast.Attribute) and node.attr in {
                    "MAIN_SEED", "PILOT_SEED", "FRESH_SEED"} and isinstance(
                    node.value, ast.Name) and node.value.id == "pr":
                offenders.append(str(path))
    assert offenders == []


def test_paid_month_km_censors_active_customers_at_t_minus_one() -> None:
    """Regression (M10): an active customer with t paid months has only survived t - 1 churn
    decisions. Censoring at t overstated survival."""
    from decision_intelligence.customer_value.retention import paid_month_km

    rng = np.random.default_rng(5)
    lam = 0.08
    n = 20_000
    paid = 1 + np.floor(rng.exponential(1 / lam, n)).astype(int)
    window = rng.integers(1, 30, n)  # paid months observable before the cutoff
    churned = paid < window  # churn observed only if it happened before the last month
    observed = np.where(churned, paid, window)
    km = paid_month_km(observed, churned)
    for a in (3, 12, 20):
        se = float(km.se[np.searchsorted(km.times, a, side="right") - 1])
        assert abs(km.at([a])[0] - np.exp(-lam * a)) < 4 * se
    naive = kaplan_meier(observed, churned)  # censored at t: biased upwards
    assert naive.at([20])[0] > np.exp(-lam * 20) + 4 * se


def test_paid_month_km_matches_a_hand_life_table_and_statsmodels() -> None:
    """Independent references for the discrete-time censoring rule (M10 audit)."""
    from statsmodels.duration.survfunc import SurvfuncRight

    from decision_intelligence.customer_value.retention import paid_month_km

    # Hand life table: paid months [1, 2, 2, 3, 3], churned [y, y, n, n, y]. Actives are known
    # to have more than t - 1 paid months: durations [1, 2, 1, 2, 3]. At risk 5, 3, 1; one
    # event each -> S = 0.8, 0.8 * 2/3, 0. Greenwood SE: S * sqrt(sum d / (n (n - d))).
    km = paid_month_km([1, 2, 2, 3, 3], [True, True, False, False, True])
    assert km.times.tolist() == [1.0, 2.0, 3.0]
    assert km.at_risk.tolist() == [5.0, 3.0, 1.0]
    assert km.survival == pytest.approx([0.8, 0.8 * 2 / 3, 0.0], abs=1e-12)
    assert km.se[:2] == pytest.approx(
        [0.8 * np.sqrt(1 / 20), 0.8 * 2 / 3 * np.sqrt(1 / 20 + 1 / 6)], abs=1e-12)
    # statsmodels' Kaplan–Meier (independent implementation) on the shifted durations.
    rng = np.random.default_rng(11)
    paid = rng.integers(1, 25, 400)
    churned = rng.random(400) < 0.6
    km = paid_month_km(paid, churned)
    ref = SurvfuncRight(np.where(churned, paid, paid - 1).astype(float), churned.astype(int))
    assert km.times == pytest.approx(ref.surv_times, abs=0)
    assert km.survival == pytest.approx(ref.surv_prob, abs=1e-12)
    # Greenwood agrees wherever it is defined; where S reaches 0 statsmodels returns NaN and
    # this implementation returns 0 (documented convention).
    defined = np.isfinite(ref.surv_prob_se)
    assert km.se[defined] == pytest.approx(ref.surv_prob_se[defined], abs=1e-12)
    assert np.all(km.se[~defined] == 0) and np.all(km.survival[~defined] == 0)
