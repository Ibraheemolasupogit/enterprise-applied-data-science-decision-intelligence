"""M11 P1 generators: contracts, determinism, record times, leakage trap, protocol alignment."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from decision_intelligence.customer_value import predictive_protocol as pp
from decision_intelligence.customer_value.panel import observe
from decision_intelligence.data import predictive_ltv_scenarios as g
from decision_intelligence.data.point_in_time import as_of

REPO = Path(__file__).resolve().parents[3]
SMALL_ACQ = (15.0, 20.0, 25.0, 10.0)
LATENT_COLUMNS = {"state", "latent_class", "hazard", "alive", "death_time", "purchase_rate",
                  "dropout_probability", "spend_rate", "exit_type"}


@pytest.fixture(scope="module")
def sig() -> g.SignalsScenario:
    return g.generate_signals_scenario(g.SignalsScenarioConfig(
        seed=4301, n_months=24, acquisitions_per_month=SMALL_ACQ))


@pytest.fixture(scope="module")
def pur() -> g.PurchasesScenario:
    return g.generate_purchases_scenario(g.PurchasesScenarioConfig(
        seed=4302, n_months=24, acquisitions_per_month=SMALL_ACQ))


def _frames(sc: g.SignalsScenario | g.PurchasesScenario) -> list[pd.DataFrame]:
    if isinstance(sc, g.SignalsScenario):
        return [sc.customers, sc.transactions, sc.cancellations, sc.signals,
                sc.account_closure_summary, sc.marketing]
    return [sc.customers, sc.purchases, sc.cancellations, sc.marketing]


def _assert_reproducible(a: list[pd.DataFrame], b: list[pd.DataFrame],
                         c: list[pd.DataFrame]) -> None:
    for x, y in zip(a, b, strict=True):
        pd.testing.assert_frame_equal(x, y)
    assert not a[1].equals(c[1])


def test_same_seed_reproduces_and_seeds_differ() -> None:
    small = g.SignalsScenarioConfig(seed=4303, n_months=12, acquisitions_per_month=SMALL_ACQ)
    for cfg in [small] + [g.signals_variant(v, small) for v in g.SIGNALS_VARIANTS]:
        _assert_reproducible(_frames(g.generate_signals_scenario(cfg)),
                             _frames(g.generate_signals_scenario(cfg)),
                             _frames(g.generate_signals_scenario(replace(cfg, seed=4305))))
    small_p = g.PurchasesScenarioConfig(seed=4304, n_months=12,
                                        acquisitions_per_month=SMALL_ACQ)
    for cfg_p in [small_p] + [g.purchases_variant(v, small_p) for v in g.PURCHASES_VARIANTS]:
        _assert_reproducible(_frames(g.generate_purchases_scenario(cfg_p)),
                             _frames(g.generate_purchases_scenario(cfg_p)),
                             _frames(g.generate_purchases_scenario(replace(cfg_p, seed=4306))))


def test_invalid_configurations_are_rejected() -> None:
    with pytest.raises(ValueError):
        g.SignalsScenarioConfig(class_probabilities=(0.5, 0.5, 0.5))
    with pytest.raises(ValueError):
        g.SignalsScenarioConfig(transition=(((0.5, 0.5, 0.5),) * 3,) * 3)
    with pytest.raises(ValueError):
        g.SignalsScenarioConfig(involuntary_hazard=(0.0, 0.0, 1.0))
    with pytest.raises(ValueError):
        g.PurchasesScenarioConfig(gg_q=1.0)
    with pytest.raises(ValueError):
        g.PurchasesScenarioConfig(dropout_process="weibull")
    with pytest.raises(ValueError):
        g.PurchasesScenarioConfig(spend_frequency_correlation=1.0)
    with pytest.raises(KeyError):
        g.signals_variant("unknown", g.SignalsScenarioConfig())
    with pytest.raises(KeyError):
        g.purchases_variant("unknown", g.PurchasesScenarioConfig())


def test_observable_tables_carry_no_latent_truth(sig: g.SignalsScenario,
                                                 pur: g.PurchasesScenario) -> None:
    for frame in (sig.customers, sig.transactions, sig.cancellations, sig.signals,
                  pur.customers, pur.purchases, pur.cancellations):
        assert not set(frame.columns) & LATENT_COLUMNS
    # Only the forbidden trap table reveals anything about the exit.
    assert "exit_type" in sig.account_closure_summary.columns


def test_signals_paid_months_are_contiguous_and_consistent(sig: g.SignalsScenario) -> None:
    tx, signals = sig.transactions, sig.signals
    assert len(tx) == len(signals) == len(sig.truth.states)
    by = tx.merge(sig.customers, on="customer_id").assign(
        age=lambda f: f["billing_month"] - f["acquisition_month"])
    agg = by.groupby("customer_id")["age"].agg(["min", "max", "count"])
    assert (agg["min"] == 0).all() and (agg["count"] == agg["max"] + 1).all()
    assert tx["billing_month"].max() <= sig.config.n_months - 1
    pd.testing.assert_series_equal(
        tx.sort_values(["customer_id", "billing_month"])["billing_month"].reset_index(
            drop=True),
        signals.sort_values(["customer_id", "month"])["month"].reset_index(drop=True),
        check_names=False)
    last = tx.groupby("customer_id")["billing_month"].max()
    cancels = sig.cancellations.set_index("customer_id")["last_paid_month"]
    assert (last.reindex(cancels.index) == cancels).all()
    churned = sig.truth.exit_type != ""
    assert set(sig.cancellations["customer_id"]) == set(sig.truth.customer_ids[churned])
    assert (cancels < sig.config.n_months - 1).all()


def test_record_times_respect_the_protocol_lags(sig: g.SignalsScenario,
                                                pur: g.PurchasesScenario) -> None:
    for tx in (sig.transactions, pur.purchases):
        lag = (tx["recorded_at"] - tx["billing_date"]).dt.days
        assert lag.min() >= 0 and lag.max() <= 40 < pp.RECORDING_LAG_DAYS
    s = sig.signals
    month_end = pd.DatetimeIndex([sig.month_end(m) for m in s["month"]])
    delay = (pd.DatetimeIndex(s["recorded_at"]) - month_end).total_seconds() / 86400
    assert delay.min() >= 0 and delay.max() <= sig.config.signal_lag_days
    assert sig.config.signal_lag_days == pp.SIGNAL_RECORDING_LAG_DAYS
    next_end = pd.DatetimeIndex([sig.month_end(m + pp.SIGNAL_COMPLETE_OFFSET)
                                 for m in s["month"]])
    assert (s["recorded_at"].to_numpy() <= next_end).all()  # usable at C iff m <= C - 1
    c = sig.cancellations
    start_next = pd.DatetimeIndex([sig.month_end(m) for m in c["last_paid_month"]])
    after = (pd.DatetimeIndex(c["recorded_at"]) - start_next).total_seconds() / 86400
    assert after.min() > 0 and after.max() < 4


def test_purchase_table_conventions(pur: g.PurchasesScenario) -> None:
    p = pur.purchases.merge(pur.customers, on="customer_id")
    first = p.groupby("customer_id").first()
    assert (first["billing_date"] == first["acquisition_date"]).all()
    months = ((p["billing_date"].dt.year - pur.start.year) * 12
              + p["billing_date"].dt.month - pur.start.month)
    assert (months == p["billing_month"]).all()
    assert p["billing_date"].max() <= pur.month_end(pur.config.n_months - 1)
    assert pur.cancellations.empty and pur.transactions is pur.purchases
    assert (p["cost_to_serve"] == 0).all()


def test_leakage_trap_leaks_only_through_an_event_time_join(sig: g.SignalsScenario) -> None:
    trap = sig.account_closure_summary
    leaked_any = False
    for cutoff in (8, 12, 16):
        end = sig.month_end(cutoff)
        cancelled_by_c = set(as_of(sig.cancellations, "recorded_at", end)["customer_id"])
        active = set(sig.customers.loc[sig.customers["acquisition_month"] <= cutoff,
                                       "customer_id"]) - cancelled_by_c
        event_join = set(trap.loc[trap["summary_month"] <= cutoff, "customer_id"])
        record_join = set(as_of(trap, "recorded_at", end)["customer_id"])
        leaked_any |= bool(event_join & active)
        assert not record_join & active
    assert leaked_any
    assert pp.FORBIDDEN_SOURCES == ("account_closure_summary",)
    assert {f.source for f in pp.FEATURES} <= {"customers", "transactions", "cancellations",
                                                "signals", "purchases"}


def test_generators_match_the_protocol_scenarios() -> None:
    s, p = g.SignalsScenarioConfig(), g.PurchasesScenarioConfig()
    for cfg, name, variants in ((s, "signals_contractual", g.SIGNALS_VARIANTS),
                                (p, "purchases_noncontractual", g.PURCHASES_VARIANTS)):
        spec = pp.scenario(name)
        assert cfg.n_months == spec.n_months and cfg.start == spec.start
        assert sum(cfg.acquisitions_per_month) == spec.nominal_acquisitions_per_month
        assert variants == spec.variants
        assert cfg.seed <= pp.DEVELOPMENT_SEED_MAX  # defaults are development seeds
    assert g.config_digest(s) == g.config_digest(replace(s, seed=7))  # parameters, not seed
    assert g.config_digest(s) != g.config_digest(g.signals_variant(g.SIGNALS_VARIANTS[0], s))


def test_m10_machinery_reads_the_new_tables(sig: g.SignalsScenario,
                                            pur: g.PurchasesScenario) -> None:
    obs = observe(sig, 12)
    st = obs.status()
    assert len(st) == int((sig.customers["acquisition_month"] <= 12).sum())
    assert st["churned"].sum() == len(as_of(sig.cancellations, "recorded_at",
                                            sig.month_end(12)).query("last_paid_month < 12"))
    po = observe(pur, 12)
    assert np.isfinite(po.transactions["contribution"]).all()
    dup = g.generate_signals_scenario(replace(sig.config, duplicate_rate=0.05))
    assert dup.transactions["transaction_id"].duplicated().any()
    assert not observe(dup, 12).transactions["transaction_id"].duplicated().any()


def test_planning_script_never_reads_ground_truth() -> None:
    tree = ast.parse((REPO / "scripts/plan_m11_precision.py").read_text())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert not names & {"truth", "SignalsTruth", "PurchasesTruth", "EconomicsTruth",
                        "latent_expected_value"}


def test_pinned_parameter_digests_match_the_generators() -> None:
    """Protocol amendment A1: every M11 run uses exactly the pinned generator parameters."""
    s, p = g.SignalsScenarioConfig(), g.PurchasesScenarioConfig()
    computed = {"signals_contractual": g.config_digest(s),
                "purchases_noncontractual": g.config_digest(p)}
    computed |= {v: g.config_digest(g.signals_variant(v, s)) for v in g.SIGNALS_VARIANTS}
    computed |= {v: g.config_digest(g.purchases_variant(v, p)) for v in g.PURCHASES_VARIANTS}
    assert computed == pp.SCENARIO_PARAMETER_DIGESTS
