"""M11 P3: E1 (shifted-beta-geometric) and E2 (exponential-Gamma) — likelihood identities, the
published sBG example, the geometric limit, parameter recovery, fit status and fallback,
reproducibility, leakage and temporal validation (development seeds only)."""

from __future__ import annotations

import ast
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

from decision_intelligence.customer_value import contractual as ct
from decision_intelligence.customer_value import oracle as orc
from decision_intelligence.customer_value import predictive_protocol as pp
from decision_intelligence.customer_value.features import PromotionPolicy
from decision_intelligence.customer_value.landmarks import view_at
from decision_intelligence.customer_value.panel import month_end, observe
from decision_intelligence.data import customer_economics_scenarios as m10g
from decision_intelligence.data.customer_economics_scenarios import CHANNELS, SEGMENTS
from decision_intelligence.data.point_in_time import as_of

SMALL_ACQ = (20.0, 25.0, 30.0, 12.0)


def _policy(cfg: m10g.EconomicsScenarioConfig) -> PromotionPolicy:
    return PromotionPolicy(cfg.promo_months, tuple(
        c for c, p in zip(CHANNELS, cfg.promo_channels, strict=True) if p))


@pytest.fixture(scope="module")
def m10() -> m10g.EconomicsScenario:
    return m10g.generate_economics_scenario(m10g.EconomicsScenarioConfig(seed=4801))


def _published(survivors: list[int]) -> ct.DurationData:
    """Fader & Hardie (2007), Table 1, first 7 years of 1,000 customers: churners in year ``t``
    have ``t`` paid periods; the survivors at year 7 are censored (``T > 7``)."""
    s = [1000, *survivors]
    churn = [s[i] - s[i + 1] for i in range(7)]
    paid = np.concatenate([np.full(c, i + 1) for i, c in enumerate(churn)] + [np.full(s[7], 8)])
    event = np.concatenate([np.ones(sum(churn), bool), np.zeros(s[7], bool)])
    return ct.DurationData.from_customers(paid, event)


def test_published_sbg_example_is_reproduced() -> None:
    """Fader & Hardie (2007, J. Interactive Marketing 21(1)): High End alpha = 0.668,
    beta = 3.806, LL = -1,611.2 (Appendix B; the body text's '0.688' is a typo); Regular
    alpha = 0.704, beta = 1.182 (values checked against the paper)."""
    high = ct.fit("E1", _published([869, 743, 653, 593, 551, 517, 491]))
    assert high.status == "ok"
    assert high.params == pytest.approx((0.668, 3.806), abs=5e-4)
    assert high.log_likelihood == pytest.approx(-1611.2, abs=0.05)
    regular = ct.fit("E1", _published([631, 468, 382, 326, 289, 262, 241]))
    assert regular.params == pytest.approx((0.704, 1.182), abs=5e-4)


def _brute(model: str, params: tuple[float, float], t: int, event: bool) -> float:
    """Mixture likelihood by direct integration over the latent churn parameter, written as
    an integral over its quantile ``u`` (bounded integrand on [0, 1])."""
    if model == "E1":
        ppf = stats.beta(*params).ppf

        def h(u: float) -> float:
            th = float(ppf(u))
            return (th if event else 1.0) * (1 - th) ** (t - 1)
    else:
        bb, v = params
        gppf = stats.gamma(1 / v, scale=v).ppf

        def h(u: float) -> float:
            z = float(gppf(u))
            keep = math.exp(-bb * z * (t - 1))
            return keep * -math.expm1(-bb * z) if event else keep
    return float(integrate.quad(h, 0, 1, epsabs=0, epsrel=1e-12, limit=500)[0])


@pytest.mark.parametrize("model,params", [("E1", (0.7, 3.8)), ("E1", (2.5, 30.0)),
                                          ("E2", (0.08, 0.5)), ("E2", (0.03, 1.7))])
def test_likelihood_matches_brute_force_integration(model: str, params: tuple[float, float]
                                                    ) -> None:
    theta = np.log(params)
    for t in (1, 2, 5, 13, 40):
        for event in (True, False):
            if not event and t == 1:
                continue  # S(0) = 1: no information
            fn = ct.log_event if event else ct.log_survival
            arg = np.array([t if event else t - 1], dtype=float)
            got = float(np.exp(fn(model, theta, arg)[0][0]))
            assert got == pytest.approx(_brute(model, params, t, event), rel=1e-9)


@pytest.mark.parametrize("model", ct.MODEL_IDS)
def test_score_matches_finite_differences(model: str) -> None:
    rng = np.random.default_rng(4802)
    data = ct.DurationData.from_customers(rng.integers(1, 30, 400), rng.random(400) < 0.6)
    theta = np.log([0.9, 4.0] if model == "E1" else [0.07, 0.6])
    _, grad = ct.log_likelihood(model, theta, data)
    for k in range(2):
        e = np.zeros(2)
        e[k] = 1e-6
        fd = (ct.log_likelihood(model, theta + e, data)[0]
              - ct.log_likelihood(model, theta - e, data)[0]) / 2e-6
        assert grad[k] == pytest.approx(fd, rel=1e-6)


def test_e2_reduces_to_the_geometric_model() -> None:
    rng = np.random.default_rng(4803)
    b = 0.06
    t = rng.integers(1, 40, 600)
    e = rng.random(600) < 0.5
    data = ct.DurationData.from_customers(t, e)
    q = 1 - math.exp(-b)  # geometric monthly churn
    geo = float(np.sum(np.where(e, np.log(q) + (t - 1) * np.log1p(-q), (t - 1) * np.log1p(-q))))
    for v in (1e-6, 1e-9, 1e-12):
        ll = ct.log_likelihood("E2", np.log([b, v]), data)[0]
        assert ll == pytest.approx(geo, rel=10 * v + 1e-12)
    ages = np.arange(60.0)
    np.testing.assert_allclose(ct.survival("E2", (b, 1e-12), ages), np.exp(-b * ages),
                               rtol=1e-10)
    # Geometric data (expected counts, no sampling noise): the MLE of v is the boundary v = 0,
    # which the protocol flags as unstable; b is still estimated exactly.
    ts, es, cs = [], [], []
    for w in range(1, 61):
        for k in range(1, w):
            ts.append(k)
            es.append(True)
            cs.append(1000 * q * (1 - q) ** (k - 1))
        ts.append(w)
        es.append(False)
        cs.append(1000 * (1 - q) ** (w - 1))
    f = ct.fit("E2", ct.DurationData(np.array(ts, float), np.array(es), np.array(cs)))
    assert f.status == "unstable" and "parameter at the log-scale bound" in f.reasons
    assert f.diagnostics.converged  # a boundary optimum (projected score 0), not a failure
    assert f.params[0] == pytest.approx(b, rel=1e-6)
    # Sampled geometric data: v-hat may be interior and small; predictions are geometric.
    lifetimes = rng.geometric(q, 20000)
    window = rng.integers(1, 61, 20000)
    g = ct.fit("E2", ct.DurationData.from_customers(np.minimum(lifetimes, window),
                                                    lifetimes < window))
    np.testing.assert_allclose(ct.survival("E2", g.params, ages[:13]),
                               np.exp(-b * ages[:13]), rtol=0.05)


def test_e2_survival_and_conditional_value_are_the_m10_generators(m10: m10g.EconomicsScenario
                                                                  ) -> None:
    """With the true cell parameters and the true contribution, E2's conditional value is the
    P2 oracle exactly (the generator is E2 within a cell)."""
    truth, cfg = m10.truth, m10.config
    ages = np.arange(80)
    for s in SEGMENTS:
        for c in CHANNELS:
            b = truth.base_hazard(s, c)
            np.testing.assert_allclose(
                ct.survival("E2", (b, cfg.frailty_variance), ages.astype(np.float64)),
                truth.population_paid_probability(s, c, ages), rtol=1e-12)
    cutoff, w = 24, 12
    oracle = orc.m10_oracle(m10, cutoff, w).frame
    status = observe(m10, cutoff).status().reindex(oracle.index)
    seg, ch = status["segment"].to_numpy(), status["channel"].to_numpy()
    paid = np.empty((len(oracle), w))
    for s in SEGMENTS:
        for c in CHANNELS:
            sel = (seg == s) & (ch == c)
            theta = np.log([truth.base_hazard(s, c), cfg.frailty_variance])
            paid[sel] = ct.conditional_paid("E2", theta, oracle["age"].to_numpy()[sel], w)
    ages_ahead = oracle["age"].to_numpy()[:, None] + np.arange(1, w + 1)[None, :]
    m = orc.contribution_by_age(cfg, seg, ch, ages_ahead)
    np.testing.assert_allclose((paid * m).sum(axis=1), oracle["oracle"].to_numpy(), rtol=1e-12)


def _simulate(model: str, params: tuple[float, float], n: int, seed: int) -> ct.DurationData:
    rng = np.random.default_rng(seed)
    if model == "E1":
        churn = rng.beta(params[0], params[1], n)
    else:
        churn = -np.expm1(-params[0] * rng.gamma(1 / params[1], params[1], n))
    lifetimes = rng.geometric(np.clip(churn, 1e-300, 1.0))
    window = rng.integers(1, 49, n)  # paid months reachable by the cutoff
    return ct.DurationData.from_customers(np.minimum(lifetimes, window), lifetimes < window)


@pytest.mark.parametrize("model,params,seed", [
    ("E1", (0.8, 9.0), 4811), ("E1", (2.0, 30.0), 4812),
    ("E2", (0.07, 0.5), 4813), ("E2", (0.12, 1.2), 4814)])
def test_parameters_are_recovered_within_3_se(model: str, params: tuple[float, float],
                                              seed: int) -> None:
    f = ct.fit(model, _simulate(model, params, 30000, seed))
    assert f.status == "ok"
    z = (np.asarray(f.log_params) - np.log(params)) / np.sqrt(np.diag(f.cov_log))
    assert np.all(np.abs(z) <= 3.0)
    assert max(f.start_log_likelihoods) - min(f.start_log_likelihoods) < 1e-6  # starts agree


def test_fit_status_rules_and_failure_handling() -> None:
    empty = ct.fit("E1", ct.DurationData.from_customers(np.ones(5, int), np.zeros(5, bool)))
    assert empty.status == "undefined" and empty.reasons == ("no data",)
    assert not empty.finite
    few = ct.fit("E2", _simulate("E2", (0.07, 0.5), 40, 4815))
    assert few.n_events < pp.STATUS_RULES.min_events
    assert few.status == "unstable" and "too few events" in few.reasons
    none = ct.fit("E1", ct.DurationData.from_customers(np.full(50, 12), np.zeros(50, bool)))
    assert none.status in ("unstable", "failed")  # no events: boundary, never "ok"
    with pytest.raises(ValueError):
        ct.DurationData.from_customers(np.array([0, 2]), np.array([True, False]))
    with pytest.raises(ValueError):
        ct.fit("E9", ct.DurationData.from_customers(np.ones(1, int), np.ones(1, bool)))
    for model in ct.MODEL_IDS:  # extreme durations and parameters stay finite
        big = ct.DurationData.from_customers(np.array([1, 500, 5000]), np.array([1, 1, 0], bool))
        for theta in ([-12.0, 12.0], [12.0, -12.0], [0.0, 0.0]):
            ll, g = ct.log_likelihood(model, np.array(theta), big)
            assert math.isfinite(ll) and np.all(np.isfinite(g))


def test_unstable_cells_fall_back_to_the_pooled_fit() -> None:
    rng = np.random.default_rng(4816)
    rows = []
    for s, c, n in (("a", "x", 4000), ("a", "y", 4000), ("b", "x", 40)):
        churn = -np.expm1(-0.08 * rng.gamma(2.0, 0.5, n))
        life = rng.geometric(churn)
        win = rng.integers(1, 40, n)
        rows.append(pd.DataFrame({"segment": s, "channel": c,
                                  "paid_months": np.minimum(life, win),
                                  "churned": life < win}))
    fits = ct.fit_cells("E2", pd.concat(rows, ignore_index=True))
    assert fits.pooled.status == "ok"
    assert fits.cells[("b", "x")].status == "unstable"
    status, used = fits.used[("b", "x")]
    assert status == "fallback" and used is fits.pooled
    assert fits.used[("a", "x")][0] == "ok"
    assert fits.status_counts()["fallback"] == 1 + (("b", "y") in fits.used)  # empty cell
    assert set(fits.used) == set(fits.cells)


def test_fits_are_deterministic_and_row_order_invariant(m10: m10g.EconomicsScenario) -> None:
    pol = _policy(m10.config)
    a = ct.predict("E2", m10, 24, 12, pol)
    b = ct.predict("E2", m10, 24, 12, pol)
    pd.testing.assert_frame_equal(a.frame, b.frame)
    shuffled = replace(m10, customers=m10.customers.sample(frac=1.0, random_state=1),
                       transactions=m10.transactions.sample(frac=1.0, random_state=2))
    c = ct.predict("E2", shuffled, 24, 12, pol)
    pd.testing.assert_frame_equal(a.frame.sort_index(), c.frame.sort_index(), rtol=1e-9)


@pytest.mark.parametrize("scramble", [False, True])
def test_predictions_ignore_everything_recorded_after_the_cutoff(
        m10: m10g.EconomicsScenario, scramble: bool) -> None:
    cutoff = 24
    end = month_end(m10.start, cutoff)
    if scramble:
        tx = m10.transactions.copy()
        late = tx["recorded_at"] > end
        tx.loc[late, "revenue"] = np.random.default_rng(3).uniform(0, 500, int(late.sum()))
        cn = m10.cancellations.copy()
        cn.loc[cn["recorded_at"] > end, "last_paid_month"] = 0
        other = replace(m10, transactions=tx, cancellations=cn)
    else:
        other = replace(m10, transactions=as_of(m10.transactions, "recorded_at", end),
                        cancellations=as_of(m10.cancellations, "recorded_at", end))
    pol = _policy(m10.config)
    for model in ct.MODEL_IDS:
        pd.testing.assert_frame_equal(ct.predict(model, m10, cutoff, 12, pol).frame,
                                      ct.predict(model, other, cutoff, 12, pol).frame)


def test_temporal_validation_uses_only_the_validation_landmarks_information(
        m10: m10g.EconomicsScenario) -> None:
    cutoff, w = 30, 12
    pol = _policy(m10.config)
    lv = pp.validation_landmark(cutoff, w)
    assert lv == cutoff - pp.COMPLETE_MONTH_OFFSET - w
    res = ct.validate("E2", m10, cutoff, w, pol)
    assert res.landmark == lv
    direct = ct.predict("E2", view_at(m10, lv), lv, w, pol)
    pd.testing.assert_series_equal(res.frame["prediction"], direct.frame["prediction"],
                                   check_names=False)
    # Rows recorded after L_v but by the cutoff (they carry the labels) never reach the fit.
    end = month_end(m10.start, lv)
    tx = m10.transactions.copy()
    late = (tx["recorded_at"] > end)
    tx.loc[late, "revenue"] *= 3.0
    res2 = ct.validate("E2", replace(m10, transactions=tx), cutoff, w, pol)
    pd.testing.assert_series_equal(res.frame["prediction"], res2.frame["prediction"])
    assert not res.frame["realised"].equals(res2.frame["realised"])
    assert res.candidate.ok_cell_fraction == res.fits.ok_cell_fraction
    with pytest.raises(ValueError):
        ct.validate("E2", m10, 14, w, pol)  # no landmark has complete labels


def test_contribution_model_estimates_the_expected_paid_month_contribution(
        m10: m10g.EconomicsScenario) -> None:
    pol = _policy(m10.config)
    cm = ct.fit_contribution(m10, 24, pol)
    assert (cm.table["source"] == "cell").all()
    obs = observe(m10, 24)
    tx = obs.transactions.loc[obs.transactions["billing_month"] <= obs.complete_through()]
    cust = obs.customers.set_index("customer_id")
    seg = cust["segment"].reindex(tx["customer_id"]).to_numpy()
    ch = cust["channel"].reindex(tx["customer_id"]).to_numpy()
    promo = (tx["age"].to_numpy() < pol.months) & np.isin(ch, pol.channels)
    contrib = tx["contribution"].to_numpy()
    for key, mean, n in zip(cm.table.index.tolist(), cm.table["mean"].tolist(),
                            cm.table["n"].tolist(), strict=True):
        s, c, pr = key
        sel = (seg == s) & (ch == c) & (promo == pr)
        assert int(sel.sum()) == n
        true = float(m10.truth.contribution_by_age(s, c, np.array([0 if pr else 30]))[0])
        se = float(np.std(contrib[sel], ddof=1)) / math.sqrt(n)
        assert abs(mean - true) <= 4 * se
    small = m10g.generate_economics_scenario(m10g.EconomicsScenarioConfig(
        seed=4817, acquisitions_per_month=(2.0, 2.0, 2.0, 1.0), n_months=12))
    sm = ct.fit_contribution(small, 6, pol)
    assert set(sm.table["source"]) - {"cell"}  # sparse cells borrow from wider pools
    assert sm.table["mean"].notna().all()


def test_cohort_value_matches_the_m10_truth_and_flags_divergence(m10: m10g.EconomicsScenario
                                                                  ) -> None:
    truth, cfg = m10.truth, m10.config
    for s, c in (("standard", "organic"), ("premium", "paid_search")):
        params = (truth.base_hazard(s, c), cfg.frailty_variance)
        def contribution(a: np.ndarray, s: str = s, c: str = c) -> np.ndarray:
            return truth.contribution_by_age(s, c, a)

        got = ct.cohort_value("E2", params, contribution, horizon=36,
                              monthly_rate=pp.MONTHLY_DISCOUNT_RATE)
        want = truth.population_value(horizon=36, monthly_rate=pp.MONTHLY_DISCOUNT_RATE,
                                      segment=s, channel=c)
        assert got.status == "ok" and got.value == pytest.approx(want, rel=1e-12)
    def flat(a: np.ndarray) -> np.ndarray:
        return np.ones(np.shape(a))

    assert ct.cohort_value("E1", (0.9, 5.0), flat, horizon=None, monthly_rate=0.0).status \
        == "undefined"
    assert ct.cohort_value("E2", (0.1, 1.5), flat, horizon=None, monthly_rate=0.0).status \
        == "undefined"
    assert ct.cohort_value("E2", (0.1, 0.5), flat, horizon=None, monthly_rate=0.0).status == "ok"


def test_estimation_code_never_reads_truth_oracles_or_generator_parameters() -> None:
    tree = ast.parse(Path(ct.__file__).read_text())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    modules = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not names & {"truth", "EconomicsTruth", "SignalsTruth", "config",
                        "account_closure_summary", "oracle", "m10_oracle", "signals_oracle"}
    assert not any(m.endswith("oracle") or m.startswith("decision_intelligence.data")
                   for m in modules)
