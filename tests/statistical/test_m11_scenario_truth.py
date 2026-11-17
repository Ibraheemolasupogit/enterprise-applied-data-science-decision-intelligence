"""M11 P1: the generators' latent truth against independent references (Monte Carlo
simulators written here, closed forms, and M10's truth), on development seeds only."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from decision_intelligence.customer_value.economics import contribution
from decision_intelligence.data import customer_economics_scenarios as m10g
from decision_intelligence.data import predictive_ltv_scenarios as g
from decision_intelligence.data.customer_economics_scenarios import CHANNELS, SEGMENTS


@pytest.fixture(scope="module")
def sig() -> g.SignalsScenario:
    return g.generate_signals_scenario(g.SignalsScenarioConfig(seed=4401))


@pytest.fixture(scope="module")
def pur() -> g.PurchasesScenario:
    return g.generate_purchases_scenario(g.PurchasesScenarioConfig(seed=4402))


def _simulate_chain(cfg: g.SignalsScenarioConfig, k: int, s: int, age: int, seg: str, ch: str,
                    horizon: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Independent Monte Carlo of the class-k chain from (state s, paid month age)."""
    b = -math.log(1 - cfg.monthly_churn[SEGMENTS.index(seg)]
                  * cfg.churn_multiplier[CHANNELS.index(ch)])
    state = np.full(n, s)
    alive = np.ones(n, dtype=bool)
    paid = np.zeros(horizon)
    for j in range(horizon):
        a = age + j
        kappa = cfg.class_hazard_multiplier[k]
        if cfg.non_proportional:
            kappa = kappa ** math.exp(-a / cfg.nph_decay_months)
        h = 1 - np.exp(-b * kappa * np.asarray(cfg.state_hazard_multiplier)[state])
        churn = (rng.random(n) < h) | (rng.random(n) < np.asarray(cfg.involuntary_hazard)[state])
        alive &= ~churn
        cum = np.cumsum(np.asarray(cfg.transition[k])[state], axis=1)
        state = (rng.random(n)[:, None] > cum).sum(axis=1).clip(max=2)
        paid[j] = alive.mean()
    return paid


@pytest.mark.parametrize("nph", [False, True])
@pytest.mark.parametrize("k,s,age,seg,ch", [(0, 0, 0, "standard", "organic"),
                                            (2, 2, 5, "value", "social"),
                                            (1, 1, 14, "premium", "paid_search")])
def test_paid_probabilities_match_an_independent_simulation(
        nph: bool, k: int, s: int, age: int, seg: str, ch: str) -> None:
    cfg = g.SignalsScenarioConfig(seed=4403, n_months=3, non_proportional=nph,
                                  acquisitions_per_month=(1.0, 1.0, 1.0, 1.0))
    truth = g.generate_signals_scenario(cfg).truth
    n = 60_000
    mc = _simulate_chain(cfg, k, s, age, seg, ch, 12, n, np.random.default_rng(7))
    exact = truth.paid_probabilities(k, s, age, seg, ch, 12)
    se = np.sqrt(exact * (1 - exact) / n)
    assert np.all(np.abs(mc - exact) <= 4 * se + 1e-12)


def test_degenerate_chain_reduces_to_m10_geometric_survival() -> None:
    cfg = g.SignalsScenarioConfig(seed=4404, n_months=3, acquisitions_per_month=(1.0,) * 4,
                                  class_hazard_multiplier=(1.0, 1.0, 1.0),
                                  state_hazard_multiplier=(1.0, 1.0, 1.0),
                                  involuntary_hazard=(0.0, 0.0, 0.0))
    truth = g.generate_signals_scenario(cfg).truth
    m10 = m10g.generate_economics_scenario(m10g.EconomicsScenarioConfig(
        seed=4405, n_months=3, acquisitions_per_month=(1.0,) * 4, frailty_variance=0.0)).truth
    ages = np.arange(1, 25)
    for seg in SEGMENTS:
        for ch in CHANNELS:
            for k in range(3):
                for s in range(3):
                    np.testing.assert_allclose(
                        truth.paid_probabilities(k, s, 0, seg, ch, 24),
                        m10.population_paid_probability(seg, ch, ages), rtol=1e-12)
            np.testing.assert_allclose(truth.contribution_by_age(seg, ch, ages),
                                       m10.contribution_by_age(seg, ch, ages), rtol=1e-12)


def test_generated_chain_matches_its_transition_and_emission_parameters(
        sig: g.SignalsScenario) -> None:
    cfg, tr = sig.config, sig.truth
    st = tr.states.sort_values(["customer_id", "age"]).reset_index(drop=True)
    nxt = st.groupby("customer_id")["state"].shift(-1)
    pairs = st.assign(next=nxt).dropna(subset=["next"])
    pairs = pairs.assign(k=tr.latent_class[np.searchsorted(tr.customer_ids,
                                                           pairs["customer_id"])])
    for k in range(3):
        for s in range(3):
            sub = pairs.loc[(pairs["k"] == k) & (pairs["state"] == s), "next"].to_numpy()
            if sub.size < 200:
                continue
            for s2 in range(3):
                p = cfg.transition[k][s][s2]
                assert abs(np.mean(sub == s2) - p) <= 4 * math.sqrt(p * (1 - p) / sub.size) + 1e-9
    emitted = sig.signals.sort_values(["customer_id", "month"]).reset_index(drop=True)
    for s in range(3):
        rows = (st["state"] == s).to_numpy()
        fail = emitted.loc[rows, "payment_failures"].to_numpy()
        p = cfg.payment_failure_probability[s]
        assert abs(fail.mean() - p) <= 4 * math.sqrt(p * (1 - p) / rows.sum())
        sup = emitted.loc[rows, "support_contacts"].to_numpy()
        lam = cfg.support_mean[s]
        assert abs(sup.mean() - lam) <= 4 * math.sqrt(lam / rows.sum())


def test_generated_continuation_matches_the_truth_hazard(sig: g.SignalsScenario) -> None:
    cfg, tr = sig.config, sig.truth
    st = tr.states.copy()
    idx = np.searchsorted(tr.customer_ids, st["customer_id"])
    st["k"], st["seg"], st["ch"] = tr.latent_class[idx], tr.segment[idx], tr.channel[idx]
    st = st.loc[st["month"] < cfg.n_months - 1]  # a decision was drawn
    paid = pd.MultiIndex.from_frame(tr.states[["customer_id", "age"]])
    st["continued"] = pd.MultiIndex.from_arrays(
        [st["customer_id"].to_numpy(), st["age"].to_numpy() + 1]).isin(paid)
    expected = np.array([tr.continuation(int(k), int(a), str(sg), str(c))[int(s)]
                         for k, a, sg, c, s in st[["k", "age", "seg", "ch", "state"]]
                         .itertuples(index=False)])
    for k in range(3):
        for s in range(3):
            m = ((st["k"] == k) & (st["state"] == s)).to_numpy()
            obs, ex = st["continued"].to_numpy()[m], expected[m]
            se = math.sqrt(float(np.sum(ex * (1 - ex)))) / m.sum()
            assert abs(obs.mean() - ex.mean()) <= 4 * se


def test_realised_value_is_unbiased_for_the_latent_expectation(sig: g.SignalsScenario) -> None:
    cutoff, w = 36, 12
    lat = sig.truth.latent_expected_value(cutoff, w).set_index("customer_id")["expected"]
    tx = sig.transactions
    win = tx.loc[(tx["billing_month"] > cutoff) & (tx["billing_month"] <= cutoff + w)]
    realised = (win.assign(c=contribution(win)).groupby("customer_id")["c"].sum()
                .reindex(lat.index, fill_value=0.0))
    resid = realised - lat
    assert abs(resid.mean()) <= 4 * resid.std(ddof=1) / math.sqrt(len(resid))
    nph = g.generate_signals_scenario(g.signals_variant("non_proportional_hazards",
                                                        replace(sig.config, seed=4406)))
    lat2 = nph.truth.latent_expected_value(cutoff, w).set_index("customer_id")["expected"]
    t2 = nph.transactions
    w2 = t2.loc[(t2["billing_month"] > cutoff) & (t2["billing_month"] <= cutoff + w)]
    r2 = (w2.assign(c=contribution(w2)).groupby("customer_id")["c"].sum()
          .reindex(lat2.index, fill_value=0.0)) - lat2
    assert abs(r2.mean()) <= 4 * r2.std(ddof=1) / math.sqrt(len(r2))


def _simulate_purchases(process: str, lam: float, p_or_mu: float, weeks: float, n: int,
                        rng: np.random.Generator) -> np.ndarray:
    counts = np.zeros(n)
    t = np.zeros(n)
    life = rng.exponential(1 / p_or_mu, n) if process == "pareto_nbd" else np.full(n, np.inf)
    active = np.ones(n, dtype=bool)
    while active.any():
        t[active] += rng.exponential(1 / lam, active.sum())
        ok = active & (t <= weeks) & (t <= life)
        counts[ok] += 1
        if process == "bgnbd":
            dies = ok & (rng.random(n) < p_or_mu)
            ok &= ~dies
        active = ok
    return counts


@pytest.mark.parametrize("process,lam,param", [("bgnbd", 0.3, 0.15), ("bgnbd", 1.2, 0.4),
                                               ("pareto_nbd", 0.3, 0.02),
                                               ("pareto_nbd", 0.8, 0.1)])
def test_expected_purchases_match_an_independent_simulation(process: str, lam: float,
                                                            param: float) -> None:
    weeks, n = 52.0, 60_000
    mc = _simulate_purchases(process, lam, param, weeks, n, np.random.default_rng(9))
    p = np.array([param]) if process == "bgnbd" else np.array([np.nan])
    mu = np.array([param]) if process == "pareto_nbd" else np.array([np.nan])
    exact = float(g._expected_purchases(process, np.array([lam]), p, mu, weeks)[0])
    assert abs(mc.mean() - exact) <= 4 * mc.std(ddof=1) / math.sqrt(n)


def test_no_repeat_purchase_probability_is_the_gamma_laplace_transform(
        pur: g.PurchasesScenario) -> None:
    cfg, weeks = pur.config, 26.0
    unit_s = cfg.time_unit_days * 86400.0
    c = pur.customers
    end = pur.month_end(cfg.n_months - 1)
    eligible = c.loc[(end - c["acquisition_date"]).dt.total_seconds() / unit_s >= weeks]
    p = pur.purchases.merge(eligible, on="customer_id")
    since = (p["billing_date"] - p["acquisition_date"]).dt.total_seconds() / unit_s
    repeat_by = since.between(1e-9, weeks)
    has_repeat = p.loc[repeat_by, "customer_id"].unique()
    for si, seg in enumerate(SEGMENTS):
        e = eligible.loc[eligible["segment"] == seg, "customer_id"]
        obs = 1.0 - float(np.isin(e, has_repeat).mean())
        a_s = cfg.alpha * cfg.segment_alpha_scale[si]
        exact = (a_s / (a_s + weeks)) ** cfg.r
        assert abs(obs - exact) <= 4 * math.sqrt(exact * (1 - exact) / len(e))


def test_purchase_value_is_unbiased_for_the_latent_expectation(pur: g.PurchasesScenario) -> None:
    for cfg in (pur.config, g.purchases_variant("pareto_nbd_dropout",
                                                replace(pur.config, seed=4407)),
                g.purchases_variant("correlated_spend", replace(pur.config, seed=4408))):
        sc = pur if cfg is pur.config else g.generate_purchases_scenario(cfg)
        lat = sc.truth.latent_expected_value(30, 12).set_index("customer_id")["expected"]
        p = sc.purchases
        win = p.loc[(p["billing_month"] > 30) & (p["billing_month"] <= 42)]
        realised = (win.assign(c=contribution(win)).groupby("customer_id")["c"].sum()
                    .reindex(lat.index, fill_value=0.0))
        resid = realised - lat
        assert abs(resid.mean()) <= 4 * resid.std(ddof=1) / math.sqrt(len(resid))


def test_latent_distributions_and_copula(pur: g.PurchasesScenario) -> None:
    cfg, tr = pur.config, pur.truth
    for si, seg in enumerate(SEGMENTS):
        m = tr.segment == seg
        a_s, g_s = cfg.alpha * cfg.segment_alpha_scale[si], cfg.gg_gamma[si]
        assert stats.kstest(tr.purchase_rate[m], "gamma", (cfg.r, 0, 1 / a_s)).pvalue > 1e-3
        assert stats.kstest(tr.spend_rate[m], "gamma", (cfg.gg_q, 0, 1 / g_s)).pvalue > 1e-3
    for ci, ch in enumerate(CHANNELS):
        m = tr.channel == ch
        assert stats.kstest(tr.dropout_probability[m], "beta",
                            (cfg.dropout_a, cfg.dropout_b[ci])).pvalue > 1e-3
    # Gamma-Gamma independence holds within a segment (both rates vary by segment, so pooled
    # across segments they are correlated: a designed-in reason to fit E4 by cell).
    cor = g.generate_purchases_scenario(g.purchases_variant("correlated_spend",
                                                            replace(cfg, seed=4409))).truth
    target = 6 / math.pi * math.asin(-0.5 / 2)  # Spearman of a Gaussian copula, rho = -0.5
    for seg in SEGMENTS:
        m, mc = tr.segment == seg, cor.segment == seg
        tol = 4 / math.sqrt(m.sum() - 1)
        assert abs(stats.spearmanr(tr.purchase_rate[m], tr.spend_rate[m]).statistic) < tol
        assert stats.spearmanr(cor.purchase_rate[mc], cor.spend_rate[mc]).statistic == \
            pytest.approx(target, abs=4 / math.sqrt(mc.sum() - 1))
    pareto = g.generate_purchases_scenario(g.purchases_variant("pareto_nbd_dropout",
                                                               replace(cfg, seed=4410))).truth
    assert stats.kstest(pareto.death_rate, "gamma",
                        (cfg.pareto_s, 0, 1 / cfg.pareto_beta)).pvalue > 1e-3
    x = pareto.death_time * pareto.death_rate  # Exp(1)
    assert stats.kstest(x, "expon").pvalue > 1e-3
    spend = pur.purchases["revenue"].to_numpy()
    nu = tr.spend_rate[np.searchsorted(tr.customer_ids, pur.purchases["customer_id"])]
    ratio = spend * nu / cfg.gg_p  # mean 1 under Gamma(p, rate nu)
    assert abs(ratio.mean() - 1) <= 4 * ratio.std(ddof=1) / math.sqrt(ratio.size) + 1e-3
