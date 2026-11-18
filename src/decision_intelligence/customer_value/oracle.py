"""Observable-information oracles for Milestone 11 contractual scenarios (P2).

**Evaluation reference only.** An oracle is the Bayes-optimal prediction ``E[R | F_C]`` under
the *true* generator and its *true parameters*, using exactly the information of one tier at
cutoff ``C``: only data recorded by ``C`` (M10's :func:`~.panel.observe` for status,
``recorded_at`` for signals). It reads the generator's parameters, so no estimator or feature
module may import it (AST-tested). It is the attainable ceiling for models with that
information; ``oracle - latent truth`` is irreducible.

**Signals scenario — exact forward filter.** Latent risk class ``k`` (fixed) and engagement
state ``S_a`` per paid month. For an active customer with current paid age ``a0 = C - acq``
(month ``C`` is paid: a churn after month ``C - 1`` would have been recorded by ``C``),

    alpha_0(k, s)  = pi_k init_k(s) e_0(s)
    alpha_a(k, s') = sum_s alpha_{a-1}(k, s) c_k(a-1, s) P_k(s, s') e_a(s'),   a = 1 .. a0

with continuation ``c_k(a, s) = exp(-b kappa_k(a) psi_s)(1 - eta_s)`` (the survival of every
churn decision before ``a0``) and emission likelihood ``e_a(s)`` = Poisson(usage) x
Poisson(support contacts) x Bernoulli(payment failure) for the signal months usable at ``C``
(ages ``0 .. a0 - 1``) in **Tier B**, and ``1`` in **Tier A** (survival, segment and channel
only). The posterior over ``(k, S_a0)`` is propagated forward with survival to give
``P(month a0 + j paid | F_C)`` and the expected contribution of months ``C+1 .. C+H``. The
filter is generic in the numbers of classes and states (:class:`ChainModel`).

**M10 scenario — closed form (Tier A).** Hazard ``b Z`` with ``Z ~ Gamma(1/v, v)``; an active
customer has survived ``a0`` decisions, so ``Z | F_C ~ Gamma(1/v, rate 1/v + b a0)`` and
``P(month a0 + j paid | F_C) = ((1 + v b a0) / (1 + v b (a0 + j)))^(1/v)`` (``exp(-b j)`` if
``v = 0``). Contribution history carries no information about ``Z`` (cost noise is
independent), so this is the exact Tier A oracle.

Expected contribution of a paid month at age ``a`` is M10's: price x (1 - promotion) x
(gross margin - fee rate) - cost-to-serve (cost noise has mean 1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from decision_intelligence.common.validation import FloatArray
from decision_intelligence.customer_value.features import usable_signals
from decision_intelligence.customer_value.panel import observe
from decision_intelligence.data.customer_economics_scenarios import (
    CHANNELS,
    SEGMENTS,
    EconomicsScenario,
)
from decision_intelligence.data.predictive_ltv_scenarios import SignalsScenario


@dataclass(frozen=True)
class ChainModel:
    """Finite latent classes x states with survival (generic K classes, S states)."""

    class_probabilities: FloatArray  # (K,)
    class_multiplier: FloatArray  # (K,)
    initial_state: FloatArray  # (K, S)
    transition: FloatArray  # (K, S, S)
    state_multiplier: FloatArray  # (S,)
    involuntary: FloatArray  # (S,)
    nph_decay: float | None = None

    def kappa(self, ages: FloatArray) -> FloatArray:
        """``kappa_k(a)`` for each age: shape ``(n, K)``."""
        a = np.asarray(ages, dtype=np.float64)[:, None]
        if self.nph_decay is None:
            return np.broadcast_to(self.class_multiplier, (a.shape[0],
                                                           self.class_multiplier.size)).copy()
        return np.asarray(self.class_multiplier[None, :] ** np.exp(-a / self.nph_decay),
                          dtype=np.float64)

    def continuation(self, base: FloatArray, ages: FloatArray) -> FloatArray:
        """``c(i, k, s)`` for each customer's base hazard and age: shape ``(n, K, S)``."""
        lam = base[:, None, None] * self.kappa(ages)[:, :, None] * self.state_multiplier
        return np.asarray(np.exp(-lam) * (1.0 - self.involuntary), dtype=np.float64)


def chain_from_signals(scenario: SignalsScenario) -> ChainModel:
    cfg = scenario.config
    return ChainModel(np.asarray(cfg.class_probabilities, dtype=np.float64),
                      np.asarray(cfg.class_hazard_multiplier, dtype=np.float64),
                      np.asarray(cfg.initial_state, dtype=np.float64),
                      np.asarray(cfg.transition, dtype=np.float64),
                      np.asarray(cfg.state_hazard_multiplier, dtype=np.float64),
                      np.asarray(cfg.involuntary_hazard, dtype=np.float64),
                      cfg.nph_decay_months if cfg.non_proportional else None)


def filter_posterior(model: ChainModel, base: FloatArray, current_age: np.ndarray,
                     log_emission: FloatArray | None = None) -> FloatArray:
    """Posterior ``P(k, S_a0 | survival to a0, emissions)``, shape ``(n, K, S)``.

    ``log_emission[i, a, s]`` is the log-likelihood of customer ``i``'s observed signals at age
    ``a`` given state ``s`` (0 where nothing is observed); ``None`` means Tier A."""
    n = base.size
    age = np.asarray(current_age, dtype=np.int64)
    k, s = model.initial_state.shape
    alpha = np.broadcast_to(model.class_probabilities[:, None] * model.initial_state,
                            (n, k, s)).copy()

    def emit(a: int, rows: np.ndarray) -> FloatArray:
        if log_emission is None:
            return np.ones((rows.size, 1, s))
        le = log_emission[rows, a, :]
        return np.exp(le - le.max(axis=1, keepdims=True))[:, None, :]

    every = np.arange(n)
    alpha *= emit(0, every)
    alpha /= alpha.sum(axis=(1, 2), keepdims=True)
    for a in range(1, int(age.max(initial=0)) + 1):
        rows = np.flatnonzero(age >= a)
        cont = model.continuation(base[rows], np.full(rows.size, a - 1.0))
        nxt = np.einsum("iks,kst->ikt", alpha[rows] * cont, model.transition) * emit(a, rows)
        alpha[rows] = nxt / nxt.sum(axis=(1, 2), keepdims=True)
    return alpha


def forward_paid(model: ChainModel, base: FloatArray, current_age: np.ndarray,
                 posterior: FloatArray, horizon: int) -> FloatArray:
    """``P(month a0 + j paid | F)``, ``j = 1 .. horizon``: shape ``(n, horizon)``."""
    w = posterior.copy()
    age = np.asarray(current_age, dtype=np.float64)
    out = np.empty((base.size, horizon))
    for j in range(1, horizon + 1):
        w = np.einsum("iks,kst->ikt", w * model.continuation(base, age + j - 1),
                      model.transition)
        out[:, j - 1] = w.sum(axis=(1, 2))
    return out


def contribution_by_age(config: Any, segment: np.ndarray, channel: np.ndarray,
                        ages: np.ndarray) -> FloatArray:
    """Expected contribution of a paid month: ``segment``, ``channel`` shape ``(n,)``,
    ``ages`` shape ``(n, H)``; M10's definition."""
    s = np.array([SEGMENTS.index(x) for x in segment])
    ch = np.array([CHANNELS.index(x) for x in channel])
    promo = (ages < config.promo_months) & np.asarray(config.promo_channels)[ch][:, None]
    revenue = np.asarray(config.price)[s][:, None] * (1.0 - config.promo_discount * promo)
    return np.asarray(revenue * (config.gross_margin - config.fee_rate)
                      - np.asarray(config.cost_to_serve)[s][:, None], dtype=np.float64)


def base_hazard(config: Any, segment: np.ndarray, channel: np.ndarray) -> FloatArray:
    s = np.array([SEGMENTS.index(x) for x in segment])
    ch = np.array([CHANNELS.index(x) for x in channel])
    return np.asarray(-np.log(1.0 - np.asarray(config.monthly_churn)[s]
                              * np.asarray(config.churn_multiplier)[ch]), dtype=np.float64)


def _population(tables: Any, cutoff: int, population: str) -> pd.DataFrame:
    obs = observe(tables, cutoff)
    st = obs.status()
    st = st.loc[~st["churned"]]
    if population == "comparison":
        st = st.loc[st["acquisition_month"] <= obs.complete_through()]
    elif population != "active":
        raise ValueError("population must be 'comparison' or 'active'")
    return st


def _value_frame(config: Any, pop: pd.DataFrame, cutoff: int, paid: FloatArray,
                 monthly_rate: float) -> pd.DataFrame:
    horizon = paid.shape[1]
    a0 = (cutoff - pop["acquisition_month"]).to_numpy()
    ages = a0[:, None] + np.arange(1, horizon + 1)[None, :]
    m = contribution_by_age(config, pop["segment"].to_numpy(), pop["channel"].to_numpy(), ages)
    d = (1.0 + monthly_rate) ** -np.arange(1, horizon + 1)
    return pd.DataFrame({"age": a0, "oracle": (paid * m).sum(axis=1),
                         "oracle_discounted": (paid * m * d).sum(axis=1)}, index=pop.index)


def emission_loglik(scenario: SignalsScenario, cutoff: int, pop: pd.DataFrame) -> FloatArray:
    """``log e_a(s)`` from the signals usable at ``cutoff``: shape ``(n, max age + 1, S)``."""
    cfg = scenario.config
    a0 = (cutoff - pop["acquisition_month"]).to_numpy()
    out = np.zeros((len(pop), int(a0.max(initial=0)) + 1, len(cfg.usage_mean)))
    sig = usable_signals(scenario, cutoff)
    pos = pd.Index(pop.index).get_indexer(pd.Index(sig["customer_id"]))
    sig, pos = sig.loc[pos >= 0], pos[pos >= 0]
    age = (sig["month"].to_numpy() - pop["acquisition_month"].to_numpy()[pos])
    seg = np.array([SEGMENTS.index(x) for x in pop["segment"].to_numpy()[pos]])
    scale = np.asarray(cfg.segment_usage_scale)[seg]
    usage, support = sig["usage"].to_numpy(), sig["support_contacts"].to_numpy()
    fail = sig["payment_failures"].to_numpy()
    for st in range(len(cfg.usage_mean)):
        phi = cfg.payment_failure_probability[st]
        out[pos, age, st] = (stats.poisson.logpmf(usage, cfg.usage_mean[st] * scale)
                             + stats.poisson.logpmf(support, cfg.support_mean[st])
                             + np.where(fail > 0, math.log(phi), math.log1p(-phi)))
    return out


@dataclass(frozen=True)
class OracleResult:
    frame: pd.DataFrame
    """Per customer: ``age, oracle, oracle_discounted`` and ``paid_1 .. paid_H``."""
    posterior: FloatArray | None
    """``(n, K, S)`` posterior over (class, state at the cutoff); None for closed forms."""


def signals_oracle(scenario: SignalsScenario, cutoff: int, horizon: int, tier: str, *,
                   population: str = "comparison", monthly_rate: float = 0.0
                   ) -> OracleResult:
    """Exact Tier A or Tier B oracle for the signals scenario at ``cutoff``."""
    if tier not in ("A", "B"):
        raise ValueError("tier must be 'A' or 'B'")
    pop = _population(scenario, cutoff, population)
    model = chain_from_signals(scenario)
    cfg = scenario.config
    base = base_hazard(cfg, pop["segment"].to_numpy(), pop["channel"].to_numpy())
    a0 = (cutoff - pop["acquisition_month"]).to_numpy()
    le = emission_loglik(scenario, cutoff, pop) if tier == "B" else None
    post = filter_posterior(model, base, a0, le)
    paid = forward_paid(model, base, a0, post, horizon)
    frame = _value_frame(cfg, pop, cutoff, paid, monthly_rate)
    frame = frame.join(pd.DataFrame(paid, index=pop.index,
                                    columns=[f"paid_{j}" for j in range(1, horizon + 1)]))
    return OracleResult(frame, post)


def m10_oracle(scenario: EconomicsScenario, cutoff: int, horizon: int, *,
               population: str = "comparison", monthly_rate: float = 0.0) -> OracleResult:
    """Exact Tier A oracle for the M10 scenario (closed-form Gamma posterior)."""
    pop = _population(scenario, cutoff, population)
    cfg = scenario.config
    b = base_hazard(cfg, pop["segment"].to_numpy(), pop["channel"].to_numpy())
    a0 = (cutoff - pop["acquisition_month"]).to_numpy().astype(np.float64)
    j = np.arange(1, horizon + 1)[None, :]
    v = cfg.frailty_variance
    if v == 0:
        paid = np.exp(-b[:, None] * j)
    else:
        paid = ((1 + v * b[:, None] * a0[:, None])
                / (1 + v * b[:, None] * (a0[:, None] + j))) ** (1 / v)
    paid = np.asarray(paid, dtype=np.float64)
    frame = _value_frame(cfg, pop, cutoff, paid, monthly_rate)
    frame = frame.join(pd.DataFrame(paid, index=pop.index,
                                    columns=[f"paid_{i}" for i in range(1, horizon + 1)]))
    return OracleResult(frame, None)
