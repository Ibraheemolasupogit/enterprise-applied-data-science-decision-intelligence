"""M11 P2: exact-oracle identities, point-in-time features, leakage-safe landmarks."""

from __future__ import annotations

import ast
import itertools
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import special

from decision_intelligence.customer_value import landmarks as lm
from decision_intelligence.customer_value import oracle as o
from decision_intelligence.customer_value import precision as pr11
from decision_intelligence.customer_value import predictive_protocol as pp
from decision_intelligence.customer_value.features import (
    PromotionPolicy,
    contractual_features,
    feature_names,
    noncontractual_features,
)
from decision_intelligence.data import customer_economics_scenarios as m10g
from decision_intelligence.data import predictive_ltv_scenarios as g
from decision_intelligence.data.customer_economics_scenarios import CHANNELS, SEGMENTS
from decision_intelligence.data.point_in_time import as_of

PACKAGE = Path(o.__file__).resolve().parent
SMALL_ACQ = (20.0, 25.0, 30.0, 12.0)


def _policy(cfg: g.SignalsScenarioConfig) -> PromotionPolicy:
    return PromotionPolicy(cfg.promo_months,
                           tuple(c for c, p in zip(CHANNELS, cfg.promo_channels, strict=True)
                                 if p))


@pytest.fixture(scope="module")
def sig() -> g.SignalsScenario:
    return g.generate_signals_scenario(g.SignalsScenarioConfig(
        seed=4701, n_months=36, acquisitions_per_month=SMALL_ACQ))


def _random_model(rng: np.random.Generator, nph: bool) -> o.ChainModel:
    k = s = 3
    return o.ChainModel(rng.dirichlet(np.ones(k)), rng.uniform(0.3, 3.0, k),
                        rng.dirichlet(np.ones(s), size=k), rng.dirichlet(np.ones(s), size=(k, s)),
                        rng.uniform(0.5, 3.0, s), rng.uniform(0.0, 0.05, s),
                        12.0 if nph else None)


def _brute_force(model: o.ChainModel, base: float, a0: int, le: np.ndarray, horizon: int
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate every class and state path (no recursion shared with the filter)."""
    k_n, s_n = model.initial_state.shape

    def cont(k: int, a: int, s: int) -> float:
        kap = float(model.class_multiplier[k])
        if model.nph_decay is not None:
            kap = kap ** math.exp(-a / model.nph_decay)
        return math.exp(-base * kap * float(model.state_multiplier[s])) * (
            1 - float(model.involuntary[s]))

    post = np.zeros((k_n, s_n))
    for k in range(k_n):
        for path in itertools.product(range(s_n), repeat=a0 + 1):
            w = model.class_probabilities[k] * model.initial_state[k, path[0]]
            w *= math.exp(le[0, path[0]])
            for a in range(1, a0 + 1):
                w *= cont(k, a - 1, path[a - 1]) * model.transition[k, path[a - 1], path[a]]
                w *= math.exp(le[a, path[a]])
            post[k, path[-1]] += w
    post /= post.sum()
    paid = np.zeros(horizon)
    for j in range(1, horizon + 1):  # P(month a0 + j paid): every path of length j
        for k in range(k_n):
            for s0 in range(s_n):
                for future in itertools.product(range(s_n), repeat=j):
                    w, prev = post[k, s0], s0
                    for step, s in enumerate(future):
                        w *= cont(k, a0 + step, prev) * model.transition[k, prev, s]
                        prev = s
                    paid[j - 1] += w
    return post, paid


@pytest.mark.parametrize("nph", [False, True])
def test_filter_matches_brute_force_enumeration(nph: bool) -> None:
    rng = np.random.default_rng(31)
    model = _random_model(rng, nph)
    a0 = np.array([0, 1, 3, 4])
    base = rng.uniform(0.02, 0.3, a0.size)
    le = rng.normal(0, 1.5, (a0.size, 5, 3))
    for i in range(a0.size):
        le[i, a0[i]:, :] = 0.0  # the cutoff month's signals are never observed
    post = o.filter_posterior(model, base, a0, le)
    paid = o.forward_paid(model, base, a0, post, 3)
    for i in range(a0.size):
        exact_post, exact_paid = _brute_force(model, float(base[i]), int(a0[i]), le[i], 3)
        np.testing.assert_allclose(post[i], exact_post, rtol=1e-10, atol=1e-14)
        np.testing.assert_allclose(paid[i], exact_paid, rtol=1e-10)


@pytest.mark.parametrize("nph", [False, True])
def test_forward_survival_matches_the_p1_truth_recursion(nph: bool) -> None:
    cfg = g.SignalsScenarioConfig(seed=4702, n_months=3, non_proportional=nph,
                                  acquisitions_per_month=(1.0,) * 4)
    sc = g.generate_signals_scenario(cfg)
    model = o.chain_from_signals(sc)
    for k, s, a0, seg, ch in [(0, 0, 0, "standard", "organic"), (2, 2, 7, "value", "social"),
                              (1, 1, 20, "premium", "paid_search")]:
        post = np.zeros((1, 3, 3))
        post[0, k, s] = 1.0
        base = o.base_hazard(cfg, np.array([seg]), np.array([ch]))
        np.testing.assert_allclose(
            o.forward_paid(model, base, np.array([a0]), post, 12)[0],
            sc.truth.paid_probabilities(k, s, a0, seg, ch, 12), rtol=1e-12)
        np.testing.assert_allclose(
            o.contribution_by_age(cfg, np.array([seg]), np.array([ch]),
                                  np.arange(a0 + 1, a0 + 13)[None, :])[0],
            sc.truth.contribution_by_age(seg, ch, np.arange(a0 + 1, a0 + 13)), rtol=1e-12)


def test_gamma_quadrature_filter_reproduces_the_m10_closed_form() -> None:
    """The finite-class filter with Gauss-Laguerre classes approximates M10's Gamma frailty;
    it converges to the closed-form oracle, which equals M10's population survival ratio."""
    v, b = 0.5, 0.09
    nodes, weights = special.roots_genlaguerre(80, 1 / v - 1)
    weights = weights / weights.sum()
    model = o.ChainModel(weights, v * nodes, np.ones((nodes.size, 1)),
                         np.ones((nodes.size, 1, 1)), np.ones(1), np.zeros(1))
    a0 = np.array([0, 5, 12, 30])
    base = np.full(a0.size, b)
    paid = o.forward_paid(model, base, a0, o.filter_posterior(model, base, a0), 12)
    j = np.arange(1, 13)[None, :]
    closed = ((1 + v * b * a0[:, None]) / (1 + v * b * (a0[:, None] + j))) ** (1 / v)
    np.testing.assert_allclose(paid, closed, rtol=1e-8)
    cfg = m10g.EconomicsScenarioConfig(seed=4703, n_months=40, frailty_variance=v,
                                       acquisitions_per_month=(10.0,) * 4)
    sc = m10g.generate_economics_scenario(cfg)
    res = o.m10_oracle(sc, 30, 12).frame
    st = sc.customers.set_index("customer_id").loc[res.index]
    for cid in res.index[:25]:
        seg, ch = st.loc[cid, "segment"], st.loc[cid, "channel"]
        a = int(res.loc[cid, "age"])
        surv = sc.truth.population_paid_probability(seg, ch, np.arange(a, a + 13))
        np.testing.assert_allclose(res.loc[cid, [f"paid_{i}" for i in range(1, 13)]],
                                   surv[1:] / surv[0], rtol=1e-12)
        np.testing.assert_allclose(
            o.contribution_by_age(cfg, np.array([seg]), np.array([ch]),
                                  np.arange(a + 1, a + 13)[None, :])[0],
            sc.truth.contribution_by_age(seg, ch, np.arange(a + 1, a + 13)), rtol=1e-12)


def test_tier_a_is_tier_b_without_emissions(sig: g.SignalsScenario) -> None:
    pop = o._population(sig, 24, "comparison")
    model = o.chain_from_signals(sig)
    base = o.base_hazard(sig.config, pop["segment"].to_numpy(), pop["channel"].to_numpy())
    a0 = (24 - pop["acquisition_month"]).to_numpy()
    zero = np.zeros((len(pop), int(a0.max()) + 1, 3))
    np.testing.assert_allclose(o.filter_posterior(model, base, a0, zero),
                               o.filter_posterior(model, base, a0, None), rtol=1e-12)
    a = o.signals_oracle(sig, 24, 12, "A").frame
    assert len(a) == len(pop) and np.isfinite(a["oracle"]).all()


def test_features_follow_the_registry_and_recompute_by_hand(sig: g.SignalsScenario) -> None:
    cutoff = 24
    pol = _policy(sig.config)
    fb = contractual_features(sig, cutoff, "B", pol)
    assert list(fb.columns) == feature_names("contractual", "B")
    assert list(contractual_features(sig, cutoff, "A", pol).columns) == feature_names(
        "contractual", "A")
    end = sig.month_end(cutoff)
    raw = sig.signals.loc[(sig.signals["recorded_at"] <= end)
                          & (sig.signals["month"].between(cutoff - 3, cutoff - 1))]
    by_hand = raw.groupby("customer_id")["usage"].mean()
    pd.testing.assert_series_equal(fb["usage_mean_3m"], by_hand.reindex(fb.index),
                                   check_names=False)
    cust = sig.customers.set_index("customer_id").loc[fb.index]
    tenure = cutoff - cust["acquisition_month"]
    promo = np.where(cust["channel"].isin(pol.channels),
                     np.maximum(0, pol.months - tenure - 1), 0)
    np.testing.assert_array_equal(fb["promo_months_remaining"].to_numpy(), promo)
    assert (fb["paid_months"] == tenure + 1).all()
    assert fb.notna().all().all()


def _truncate(sc: g.SignalsScenario, cutoff: int, scramble: bool) -> g.SignalsScenario:
    """Remove (or scramble) every row recorded after ``cutoff``."""
    end = sc.month_end(cutoff)
    if not scramble:
        return replace(sc, transactions=as_of(sc.transactions, "recorded_at", end),
                       cancellations=as_of(sc.cancellations, "recorded_at", end),
                       signals=as_of(sc.signals, "recorded_at", end),
                       account_closure_summary=as_of(sc.account_closure_summary,
                                                     "recorded_at", end))
    rng = np.random.default_rng(5)
    tx, s = sc.transactions.copy(), sc.signals.copy()
    late_t, late_s = tx["recorded_at"] > end, s["recorded_at"] > end
    tx.loc[late_t, "revenue"] = rng.uniform(0, 500, int(late_t.sum()))
    s.loc[late_s, "usage"] = rng.integers(0, 100, int(late_s.sum()))
    return replace(sc, transactions=tx, signals=s)


@pytest.mark.parametrize("scramble", [False, True])
def test_post_cutoff_data_cannot_change_features_stacks_or_oracles(
        sig: g.SignalsScenario, scramble: bool) -> None:
    cutoff, w = 30, 12
    pol = _policy(sig.config)
    other = _truncate(sig, cutoff, scramble)
    for tier in ("A", "B"):
        pd.testing.assert_frame_equal(contractual_features(sig, cutoff, tier, pol),
                                      contractual_features(other, cutoff, tier, pol))
        pd.testing.assert_frame_equal(o.signals_oracle(sig, cutoff, w, tier).frame,
                                      o.signals_oracle(other, cutoff, w, tier).frame)
    pd.testing.assert_frame_equal(lm.training_stack(sig, cutoff, w, "contractual", "B", pol),
                                  lm.training_stack(other, cutoff, w, "contractual", "B", pol))


def test_training_labels_are_complete_and_landmarks_are_enforced(sig: g.SignalsScenario) -> None:
    cutoff, w = 34, 12  # temporal tuning is feasible here (it is not at 30)
    pol = _policy(sig.config)
    stack = lm.training_stack(sig, cutoff, w, "contractual", "A", pol)
    assert set(stack["landmark"]) == set(pp.training_landmarks(cutoff, w))
    assert (stack["landmark"] + w <= cutoff - pp.COMPLETE_MONTH_OFFSET).all()
    for landmark in sorted(set(stack["landmark"])):
        rows = stack.loc[stack["landmark"] == landmark]
        full = lm.holdout_frame(sig, sig.config.n_months, int(landmark), w, "contractual",
                                "A", pol).set_index("customer_id")
        mine = rows.set_index("customer_id")
        pd.testing.assert_series_equal(mine["realised"], full.loc[mine.index, "realised"])
        assert (mine["paid_in_window"] <= w).all()
    with pytest.raises(ValueError):
        lm.training_stack(sig, cutoff, w, "contractual", "A", pol, landmarks=[cutoff - w - 1])
    with pytest.raises(ValueError):
        lm.training_stack(sig, 30, w, "contractual", "A", pol, landmarks=pp.tuning_landmarks(30, w))
    tune = lm.training_stack(sig, cutoff, w, "contractual", "A", pol,
                             landmarks=pp.tuning_landmarks(cutoff, w))
    lv = pp.validation_landmark(cutoff, w)
    assert lv is not None and (tune["landmark"] + w <= lv - pp.COMPLETE_MONTH_OFFSET).all()


def test_backdated_closure_summary_leaks_only_through_an_event_time_join(
        sig: g.SignalsScenario) -> None:
    cutoff, w = 30, 12
    stack = lm.training_stack(sig, cutoff, w, "contractual", "B", _policy(sig.config))
    trap = sig.account_closure_summary
    assert not {"final_usage_mean", "exit_type", "summary_month"} & set(stack.columns)
    event_hits, record_hits = [], []
    for landmark in sorted(set(stack["landmark"])):
        rows = stack.loc[stack["landmark"] == landmark]
        ids = rows["customer_id"]
        event = trap.loc[trap["summary_month"] <= landmark, "customer_id"]
        record = as_of(trap, "recorded_at", sig.month_end(int(landmark)))["customer_id"]
        event_hits.append(rows.loc[ids.isin(event)])
        record_hits.append(rows.loc[ids.isin(record)])
    leaked = pd.concat(event_hits)
    assert pd.concat(record_hits).empty  # a record-time join sees nothing for active customers
    assert len(leaked) > 0
    # Every leaked row churns inside its label window: the event-time join reveals the future.
    assert (leaked["paid_in_window"] < w).all()
    assert leaked["realised"].mean() < 0.5 * stack["realised"].mean()


def test_repeated_customers_are_resampled_as_clusters(sig: g.SignalsScenario) -> None:
    stack = lm.training_stack(sig, 30, 12, "contractual", "A", _policy(sig.config))
    sizes = lm.cluster_sizes(stack)
    assert sizes.sum() == len(stack) and sizes.max() > 1
    assert pr11.effective_n(sizes, 1.0) <= stack["customer_id"].nunique()
    boot = lm.resample_customers(stack, np.random.default_rng(3))
    assert boot["cluster"].nunique() == stack["customer_id"].nunique()
    original = stack.groupby("customer_id")["landmark"].apply(frozenset)
    for _, rows in boot.groupby("cluster"):
        assert rows["customer_id"].nunique() == 1
        assert frozenset(rows["landmark"]) == original[rows["customer_id"].iloc[0]]


def test_m10_and_noncontractual_features() -> None:
    m10 = m10g.generate_economics_scenario(m10g.EconomicsScenarioConfig(
        seed=4704, n_months=30, acquisitions_per_month=SMALL_ACQ))
    pol = PromotionPolicy(2, ("paid_search", "social"))
    stack = lm.training_stack(m10, 26, 12, "contractual", "A", pol)
    assert list(stack.columns[2:10]) == feature_names("contractual", "A")
    with pytest.raises(ValueError):
        contractual_features(m10, 20, "B", pol)
    pur = g.generate_purchases_scenario(g.PurchasesScenarioConfig(
        seed=4705, n_months=30, acquisitions_per_month=SMALL_ACQ))
    f = noncontractual_features(pur, 20)
    assert list(f.columns) == feature_names("non_contractual", "A")
    through = 20 - pp.COMPLETE_MONTH_OFFSET
    p = pur.purchases.loc[(pur.purchases["recorded_at"] <= pur.month_end(20))
                          & (pur.purchases["billing_month"] <= through)]
    by_hand = (p.groupby("customer_id").size() - 1).reindex(f.index, fill_value=0)
    pd.testing.assert_series_equal(f["frequency"], by_hand, check_names=False)
    assert (f["recency"] <= f["age"] + 1e-9).all() and (f["frequency"] >= 0).all()
    nstack = lm.training_stack(pur, 26, 12, "non_contractual", "A")
    assert nstack["customer_id"].duplicated().any()


@pytest.mark.parametrize("module", ["features", "landmarks"])
def test_feature_and_landmark_code_never_reads_truth_or_generator_parameters(module: str
                                                                             ) -> None:
    tree = ast.parse((PACKAGE / f"{module}.py").read_text())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not names & {"truth", "SignalsTruth", "PurchasesTruth", "EconomicsTruth", "config",
                        "account_closure_summary", "oracle"}
    assert not any(m and m.endswith("oracle") for m in modules)
    assert SEGMENTS  # registry modules are scenario-agnostic
