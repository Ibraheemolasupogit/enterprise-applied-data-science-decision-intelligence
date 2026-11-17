"""Synthetic scenarios with known latent truth for Milestone 11 (predictive LTV).

Additive: nothing in M2–M10 is changed. Economics (segments, channels, prices, cost-to-serve,
promotions, payment fees, acquisition volumes and costs, payment-recording lags) are M10's
defaults, imported from :mod:`.customer_economics_scenarios`, so M11 and M10 values are
comparable.

**Contractual subscriptions with behavioural signals** (:func:`generate_signals_scenario`).
Each customer has a latent risk class ``k`` (fixed) and a latent engagement state ``S_a`` in
each paid month ``a`` (0 = acquisition month), following a class-specific 3-state Markov chain
(engaged, drifting, disengaged). In every paid month the state emits observable signals,
independently given ``S_a``: usage ``~ Poisson(usage_mean[S_a] x segment scale)``, support
contacts ``~ Poisson(support_mean[S_a])`` and a recovered payment failure
``~ Bernoulli(payment_failure_probability[S_a])``. At the end of month ``a`` the customer
continues with probability

    exp(-b_{s,ch} kappa_k(a) psi[S_a]) x (1 - eta[S_a])

(voluntary churn, then involuntary churn after an unrecoverable payment failure), where
``b_{s,ch} = -ln(1 - churn_s x multiplier_ch)`` is M10's base hazard and ``kappa_k(a) =
kappa_k`` (proportional hazards) or ``kappa_k ** exp(-a / tau)`` (the predeclared
non-proportional-hazards variant: the class effect fades with tenure). The acquisition month
is always paid. Because classes and states are finite, the observable-information oracle
(P2) is an exact forward filter.

Record times: payments as M10 (0-5 days, ``late_payment_rate`` 20-40 days); signals for a
month recorded 0-``signal_lag_days`` days after its end; cancellations on the first days of
the month after the last paid month. **Planted leakage trap** (``account_closure_summary``):
on cancellation a summary of the final ``closure_window_months`` months is written, indexed by
the *first* month of that window (an event-time index) but recorded at closure. A join on the
event-time index at a cutoff includes customers who churn after the cutoff; a record-time
(``recorded_at``) join never does. No feature may read it (protocol ``FORBIDDEN_SOURCES``).

**Non-contractual purchasing** (:func:`generate_purchases_scenario`). Time in weeks from the
acquisition (first) purchase. Rates vary by cell (``alpha_s`` and ``gamma_s`` by segment, the
dropout Beta by channel), so the process is BG/NBD + Gamma-Gamma *within* a segment x channel
cell; pooled across cells, frequency and spend are correlated through the segment mix.
BG/NBD: while alive a customer makes repeat purchases at rate
``lambda ~ Gamma(r, rate alpha_s)``; after each repeat purchase they drop out with probability
``p ~ Beta(a, b_ch)``. Spend per purchase ``~ Gamma(gg_p, rate nu)``, ``nu ~ Gamma(gg_q, rate
gamma_s)`` (Gamma-Gamma). Contribution = spend x (gross margin - fee rate). Purchases are
recorded 0-2 days later (``late_payment_rate`` 20-40 days).

**Predeclared misspecification variants** (:data:`SIGNALS_VARIANTS`,
:data:`PURCHASES_VARIANTS`; names match the protocol): ``non_proportional_hazards``;
``pareto_nbd_dropout`` (exponential lifetimes ``~ Exp(mu)``, ``mu ~ Gamma(s, rate beta)``,
independent of purchasing); ``correlated_spend`` (a Gaussian copula couples ``log lambda`` and
``nu`` with correlation ``spend_frequency_correlation``, keeping both Gamma marginals; with a
negative value frequent buyers spend more, breaking Gamma-Gamma independence).

Ground truth (:class:`SignalsTruth`, :class:`PurchasesTruth`) is returned separately and is
read only by evaluation code: latent classes and state paths, rates, dropout parameters and
the **latent-truth expectation** of future contribution (a decomposition reference, never a
model target).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import special, stats

from decision_intelligence.common.seeding import spawn_generators
from decision_intelligence.common.validation import FloatArray
from decision_intelligence.data.customer_economics_scenarios import (
    CHANNELS,
    SEGMENTS,
    EconomicsScenarioConfig,
)

IntArray = NDArray[np.int64]
_M10 = EconomicsScenarioConfig()
STATES = ("engaged", "drifting", "disengaged")
CLASSES = ("low_risk", "mid_risk", "high_risk")
SIGNALS_VARIANTS = ("non_proportional_hazards",)
PURCHASES_VARIANTS = ("pareto_nbd_dropout", "correlated_spend")
Triple = tuple[float, float, float]


def config_digest(config: Any) -> str:
    """SHA-256 of a generator configuration's parameters (canonical JSON of every field except
    ``seed``), so one digest identifies the parameters for every seed."""
    fields_ = {k: v for k, v in asdict(config).items() if k != "seed"}
    blob = json.dumps(fields_, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def _month_starts(start: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([start + pd.DateOffset(months=m) for m in range(n)])


def _month_end(start: pd.Timestamp, month: int) -> pd.Timestamp:
    return start + pd.DateOffset(months=int(month) + 1) - pd.to_timedelta(1, unit="s")


def _check_probabilities(values: tuple[float, ...], name: str) -> None:
    if any(v < 0 for v in values) or abs(sum(values) - 1.0) > 1e-9:
        raise ValueError(f"{name} must be non-negative and sum to 1")


def _acquire(rng_acq: np.random.Generator, rng_spend: np.random.Generator, n_months: int,
             acquisitions: tuple[float, ...], mix: tuple[tuple[float, float, float], ...],
             cpa: tuple[float, ...], spend_sd: float, onboarding: float, start: pd.Timestamp
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monthly Poisson acquisitions by channel (M10's process)."""
    rows, spend_rows = [], []
    for m in range(n_months):
        for ci, ch in enumerate(CHANNELS):
            n = int(rng_acq.poisson(acquisitions[ci]))
            seg = rng_acq.choice(len(SEGMENTS), size=n, p=mix[ci])
            day = rng_acq.integers(0, 28, size=n)
            rows.append(pd.DataFrame({"acquisition_month": m, "channel": ch,
                                      "segment_index": seg, "day": day}))
            noise = rng_spend.lognormal(-spend_sd ** 2 / 2, spend_sd)
            spend_rows.append({"month": m, "channel": ch, "new_customers": n,
                               "media_spend": round(cpa[ci] * n * noise, 2),
                               "onboarding_cost": onboarding * n})
    cust = pd.concat(rows, ignore_index=True)
    cust.insert(0, "customer_id", np.arange(1, len(cust) + 1, dtype=np.int64))
    cust["segment"] = np.array(SEGMENTS)[cust["segment_index"].to_numpy()]
    starts = _month_starts(start, n_months)
    cust["acquisition_date"] = (starts[cust["acquisition_month"].to_numpy()]
                                + pd.to_timedelta(cust["day"].to_numpy(), unit="D"))
    return cust, pd.DataFrame(spend_rows)


def _record_lags(rng: np.random.Generator, n: int, late_rate: float, normal_max: int = 5
                 ) -> IntArray:
    """Recording delay in days: ``0..normal_max``, or 20-40 for a ``late_rate`` share (always
    within the protocol's 45-day complete-month lag)."""
    late = rng.random(n) < late_rate
    return np.asarray(np.where(late, rng.integers(20, 41, n),
                               rng.integers(0, normal_max + 1, n)), dtype=np.int64)


def _with_duplicates(tx: pd.DataFrame, rng: np.random.Generator, rate: float) -> pd.DataFrame:
    if rate <= 0:
        return tx
    dup = tx.loc[rng.random(len(tx)) < rate]
    return pd.concat([tx, dup]).sort_values("transaction_id", kind="stable").reset_index(
        drop=True)


# ==========================================================================================
# Contractual subscriptions with behavioural signals
# ==========================================================================================
@dataclass(frozen=True)
class SignalsScenarioConfig:
    seed: int = 1101
    n_months: int = 60
    start: str = "2022-01-01"
    # --- M10 economics (defaults imported, not redefined) ---
    acquisitions_per_month: tuple[float, ...] = _M10.acquisitions_per_month
    cost_per_acquisition: tuple[float, ...] = _M10.cost_per_acquisition
    segment_mix: tuple[tuple[float, float, float], ...] = _M10.segment_mix
    churn_multiplier: tuple[float, ...] = _M10.churn_multiplier
    promo_channels: tuple[bool, ...] = _M10.promo_channels
    promo_months: int = _M10.promo_months
    promo_discount: float = _M10.promo_discount
    price: tuple[float, ...] = _M10.price
    cost_to_serve: tuple[float, ...] = _M10.cost_to_serve
    monthly_churn: tuple[float, ...] = _M10.monthly_churn
    gross_margin: float = _M10.gross_margin
    fee_rate: float = _M10.fee_rate
    cost_noise_sd: float = _M10.cost_noise_sd
    spend_noise_sd: float = _M10.spend_noise_sd
    onboarding_cost: float = _M10.onboarding_cost
    monthly_overhead: float = _M10.monthly_overhead
    late_payment_rate: float = _M10.late_payment_rate
    duplicate_rate: float = 0.0
    # --- latent risk classes and engagement states ---
    class_probabilities: Triple = (0.50, 0.35, 0.15)
    class_hazard_multiplier: Triple = (0.5, 1.0, 2.5)
    initial_state: tuple[Triple, Triple, Triple] = (
        (0.80, 0.15, 0.05), (0.65, 0.25, 0.10), (0.50, 0.30, 0.20))
    transition: tuple[tuple[Triple, Triple, Triple], ...] = (
        ((0.92, 0.07, 0.01), (0.30, 0.60, 0.10), (0.10, 0.30, 0.60)),
        ((0.85, 0.12, 0.03), (0.20, 0.65, 0.15), (0.05, 0.25, 0.70)),
        ((0.70, 0.22, 0.08), (0.10, 0.60, 0.30), (0.03, 0.17, 0.80)))
    state_hazard_multiplier: Triple = (0.6, 1.3, 3.0)
    involuntary_hazard: Triple = (0.005, 0.01, 0.04)
    # --- emissions (observable signals) ---
    usage_mean: Triple = (12.0, 6.0, 1.5)
    segment_usage_scale: Triple = (0.8, 1.0, 1.3)
    support_mean: Triple = (0.1, 0.4, 0.8)
    payment_failure_probability: Triple = (0.02, 0.05, 0.12)
    signal_lag_days: int = 10
    closure_window_months: int = 3
    # --- predeclared variant switch ---
    non_proportional: bool = False
    nph_decay_months: float = 12.0

    def __post_init__(self) -> None:
        k = len(CHANNELS)
        if not (len(self.acquisitions_per_month) == len(self.cost_per_acquisition)
                == len(self.segment_mix) == len(self.churn_multiplier)
                == len(self.promo_channels) == k):
            raise ValueError("every channel parameter needs one value per channel")
        if self.n_months < 3:
            raise ValueError("need n_months >= 3")
        _check_probabilities(self.class_probabilities, "class_probabilities")
        for row in self.initial_state:
            _check_probabilities(row, "initial_state rows")
        if len(self.initial_state) != len(CLASSES) or len(self.transition) != len(CLASSES):
            raise ValueError("one initial distribution and one transition matrix per class")
        for matrix in self.transition:
            if len(matrix) != len(STATES):
                raise ValueError("transition matrices must be 3 x 3")
            for row in matrix:
                _check_probabilities(row, "transition rows")
        for c in self.monthly_churn:
            for mult in self.churn_multiplier:
                if not 0 < c * mult < 1:
                    raise ValueError("churn probabilities must lie in (0, 1)")
        if min(self.class_hazard_multiplier) <= 0 or min(self.state_hazard_multiplier) <= 0:
            raise ValueError("hazard multipliers must be positive")
        for name in ("involuntary_hazard", "payment_failure_probability"):
            if any(not 0 <= v < 1 for v in getattr(self, name)):
                raise ValueError(f"{name} must lie in [0, 1)")
        if min(self.usage_mean) < 0 or min(self.support_mean) < 0:
            raise ValueError("emission means must be non-negative")
        if self.signal_lag_days < 0 or self.closure_window_months < 1:
            raise ValueError("invalid signal lag or closure window")
        if self.nph_decay_months <= 0:
            raise ValueError("nph_decay_months must be positive")
        if not 0 <= self.late_payment_rate < 1 or not 0 <= self.duplicate_rate < 1:
            raise ValueError("rates must lie in [0, 1)")


def signals_variant(name: str, base: SignalsScenarioConfig) -> SignalsScenarioConfig:
    """A predeclared misspecification variant of ``base``."""
    from dataclasses import replace

    if name == "non_proportional_hazards":
        return replace(base, non_proportional=True)
    raise KeyError(name)


@dataclass(frozen=True)
class SignalsTruth:
    """Evaluation-only truth (never an estimator input)."""

    config: SignalsScenarioConfig
    customer_ids: IntArray
    segment: NDArray[np.str_]
    channel: NDArray[np.str_]
    acquisition_month: IntArray
    latent_class: IntArray
    states: pd.DataFrame
    """``customer_id, age, month, state`` for every paid month."""
    exit_type: NDArray[np.str_]
    """``voluntary``, ``involuntary`` or ``""`` (no churn observed in the data)."""

    def base_hazard(self, segment: str, channel: str) -> float:
        cfg = self.config
        c = cfg.monthly_churn[SEGMENTS.index(segment)] * cfg.churn_multiplier[
            CHANNELS.index(channel)]
        return -math.log(1.0 - c)

    def class_multiplier(self, latent_class: int, ages: NDArray[Any]) -> FloatArray:
        cfg = self.config
        kappa = cfg.class_hazard_multiplier[latent_class]
        a = np.asarray(ages, dtype=np.float64)
        if not cfg.non_proportional:
            return np.full(a.shape, kappa)
        return np.asarray(kappa ** np.exp(-a / cfg.nph_decay_months), dtype=np.float64)

    def continuation(self, latent_class: int, age: int, segment: str, channel: str
                     ) -> FloatArray:
        """P(month ``age + 1`` paid | month ``age`` paid, state), one value per state."""
        cfg = self.config
        b = self.base_hazard(segment, channel)
        kappa = float(self.class_multiplier(latent_class, np.array([age]))[0])
        psi = np.asarray(cfg.state_hazard_multiplier)
        eta = np.asarray(cfg.involuntary_hazard)
        return np.asarray(np.exp(-b * kappa * psi) * (1.0 - eta), dtype=np.float64)

    def contribution_by_age(self, segment: str, channel: str, ages: NDArray[Any]
                            ) -> FloatArray:
        """Expected contribution of a paid month at each age (M10's definition)."""
        cfg = self.config
        s, ch = SEGMENTS.index(segment), CHANNELS.index(channel)
        promo = (np.asarray(ages) < cfg.promo_months) & cfg.promo_channels[ch]
        revenue = cfg.price[s] * (1.0 - cfg.promo_discount * promo)
        return np.asarray(revenue * (cfg.gross_margin - cfg.fee_rate) - cfg.cost_to_serve[s],
                          dtype=np.float64)

    def paid_probabilities(self, latent_class: int, state: int, age: int, segment: str,
                           channel: str, horizon: int) -> FloatArray:
        """``P(month age + j paid | month age paid in state)``, ``j = 1 .. horizon``: exact
        propagation of the (class-specific) state chain with survival."""
        matrix = np.asarray(self.config.transition[latent_class])
        w = np.zeros(len(STATES))
        w[state] = 1.0
        out = np.empty(horizon)
        for j in range(1, horizon + 1):
            w = (w * self.continuation(latent_class, age + j - 1, segment, channel)) @ matrix
            out[j - 1] = w.sum()
        return out

    def latent_expected_value(self, cutoff_month: int, horizon: int,
                              monthly_rate: float = 0.0) -> pd.DataFrame:
        """For customers who paid calendar month ``cutoff_month``: expected contribution of
        calendar months ``cutoff+1 .. cutoff+horizon`` given their latent class and their state
        in the cutoff month (month ``cutoff + j`` discounted by ``d^j`` when ``monthly_rate``
        is given). A decomposition reference, not a model target."""
        st = self.states.loc[self.states["month"] == cutoff_month]
        idx = pd.Index(self.customer_ids).get_indexer(pd.Index(st["customer_id"]))
        frame = pd.DataFrame({"customer_id": st["customer_id"].to_numpy(),
                              "latent_class": self.latent_class[idx],
                              "state": st["state"].to_numpy(), "age": st["age"].to_numpy(),
                              "segment": self.segment[idx], "channel": self.channel[idx]})
        d = (1.0 + monthly_rate) ** -np.arange(1, horizon + 1)
        expected = np.empty(len(frame))
        for key, g in frame.groupby(["latent_class", "state", "age", "segment", "channel"]):
            k, s, a, seg, ch = cast(tuple[Any, ...], key)
            p = self.paid_probabilities(int(k), int(s), int(a), str(seg), str(ch), horizon)
            m = self.contribution_by_age(str(seg), str(ch), int(a) + np.arange(1, horizon + 1))
            expected[frame.index.get_indexer(g.index)] = float(np.sum(p * m * d))
        return frame.assign(expected=expected)


@dataclass(frozen=True)
class SignalsScenario:
    config: SignalsScenarioConfig
    start: pd.Timestamp
    customers: pd.DataFrame
    """``customer_id, acquisition_month, acquisition_date, channel, segment``."""
    transactions: pd.DataFrame
    """M10 columns: ``transaction_id, customer_id, billing_month, billing_date, recorded_at,
    revenue, cogs, payment_fee, cost_to_serve``."""
    cancellations: pd.DataFrame
    """``customer_id, last_paid_month, recorded_at``."""
    signals: pd.DataFrame
    """``customer_id, month, usage, support_contacts, payment_failures, recorded_at``."""
    account_closure_summary: pd.DataFrame
    """LEAKAGE TRAP: ``customer_id, summary_month, final_usage_mean, exit_type,
    recorded_at``."""
    marketing: pd.DataFrame
    overhead: pd.DataFrame
    truth: SignalsTruth = field(repr=False)

    def month_end(self, month: int) -> pd.Timestamp:
        return _month_end(self.start, month)


def _draw_categorical(rng: np.random.Generator, probs: FloatArray) -> IntArray:
    """One draw per row of a (n x K) probability matrix."""
    u = np.asarray(rng.random(int(probs.shape[0])), dtype=np.float64)
    return np.asarray(np.minimum((u[:, None] > np.cumsum(probs, axis=1)).sum(axis=1),
                                 probs.shape[1] - 1), dtype=np.int64)


def generate_signals_scenario(config: SignalsScenarioConfig | None = None) -> SignalsScenario:
    cfg = config or SignalsScenarioConfig()
    rng_acq, rng_latent, rng_emit, rng_cost, rng_rec, rng_spend = spawn_generators(cfg.seed, 6)
    start = pd.Timestamp(cfg.start)
    cust, marketing = _acquire(rng_acq, rng_spend, cfg.n_months, cfg.acquisitions_per_month,
                               cfg.segment_mix, cfg.cost_per_acquisition, cfg.spend_noise_sd,
                               cfg.onboarding_cost, start)
    n = len(cust)
    seg_i = cust["segment_index"].to_numpy()
    ch_i = np.array([CHANNELS.index(c) for c in cust["channel"]])
    acq = cust["acquisition_month"].to_numpy().astype(np.int64)
    max_age = cfg.n_months - 1 - acq
    base = -np.log(1.0 - np.array(cfg.monthly_churn)[seg_i]
                   * np.array(cfg.churn_multiplier)[ch_i])
    k = rng_latent.choice(len(CLASSES), size=n, p=cfg.class_probabilities).astype(np.int64)
    state = _draw_categorical(rng_latent, np.asarray(cfg.initial_state)[k])
    transition = np.asarray(cfg.transition)
    kappa = np.asarray(cfg.class_hazard_multiplier)
    psi = np.asarray(cfg.state_hazard_multiplier)
    eta = np.asarray(cfg.involuntary_hazard)
    alive = np.ones(n, dtype=bool)
    last_age = max_age.copy()
    exit_type = np.full(n, "", dtype=object)
    rec_row, rec_age, rec_state = [], [], []
    rec_usage, rec_support, rec_fail = [], [], []
    usage_scale = np.asarray(cfg.segment_usage_scale)[seg_i]
    for a in range(cfg.n_months):
        idx = np.flatnonzero(alive & (a <= max_age))
        if idx.size == 0:
            break
        s = state[idx]
        rec_row.append(idx)
        rec_age.append(np.full(idx.size, a))
        rec_state.append(s)
        rec_usage.append(rng_emit.poisson(np.asarray(cfg.usage_mean)[s] * usage_scale[idx]))
        rec_support.append(rng_emit.poisson(np.asarray(cfg.support_mean)[s]))
        rec_fail.append((rng_emit.random(idx.size)
                         < np.asarray(cfg.payment_failure_probability)[s]).astype(np.int64))
        dec = idx[a < max_age[idx]]  # a churn decision whose next month lies in the data
        if dec.size == 0:
            continue
        kap = kappa[k[dec]]
        if cfg.non_proportional:
            kap = kap ** math.exp(-a / cfg.nph_decay_months)
        sd = state[dec]
        voluntary = rng_latent.random(dec.size) < 1.0 - np.exp(-base[dec] * kap * psi[sd])
        involuntary = ~voluntary & (rng_latent.random(dec.size) < eta[sd])
        churn = voluntary | involuntary
        gone = dec[churn]
        alive[gone] = False
        last_age[gone] = a
        exit_type[dec[voluntary]] = "voluntary"
        exit_type[dec[involuntary]] = "involuntary"
        stay = dec[~churn]
        state[stay] = _draw_categorical(rng_latent, transition[k[stay], state[stay]])
    row = np.concatenate(rec_row)
    age = np.concatenate(rec_age).astype(np.int64)
    st = np.concatenate(rec_state).astype(np.int64)
    order = np.lexsort((age, row))
    row, age, st = row[order], age[order], st[order]
    usage = np.concatenate(rec_usage)[order]
    support = np.concatenate(rec_support)[order]
    fail = np.concatenate(rec_fail)[order]
    month = acq[row] + age
    ids = cust["customer_id"].to_numpy()
    starts = _month_starts(start, cfg.n_months + 2)
    day = cust["day"].to_numpy()
    # Transactions (M10 conventions)
    promo = (age < cfg.promo_months) & np.array(cfg.promo_channels)[ch_i[row]]
    revenue = np.array(cfg.price)[seg_i[row]] * (1.0 - cfg.promo_discount * promo)
    cost = np.array(cfg.cost_to_serve)[seg_i[row]] * rng_cost.lognormal(
        -cfg.cost_noise_sd ** 2 / 2, cfg.cost_noise_sd, size=row.size)
    billing_date = starts[month] + pd.to_timedelta(day[row], unit="D")
    tx = pd.DataFrame({
        "transaction_id": np.arange(1, row.size + 1, dtype=np.int64),
        "customer_id": ids[row],
        "billing_month": month.astype(np.int64),
        "billing_date": billing_date,
        "recorded_at": billing_date + pd.to_timedelta(
            _record_lags(rng_rec, row.size, cfg.late_payment_rate), unit="D"),
        "revenue": np.round(revenue, 2),
        "cogs": np.round(revenue * (1.0 - cfg.gross_margin), 2),
        "payment_fee": np.round(revenue * cfg.fee_rate, 2),
        "cost_to_serve": np.round(cost, 2)})
    tx = _with_duplicates(tx, rng_rec, cfg.duplicate_rate)
    # Signals: recorded 0..signal_lag_days days after the end of their month
    month_end = starts[month + 1] - pd.to_timedelta(1, unit="s")
    signals = pd.DataFrame({
        "customer_id": ids[row], "month": month.astype(np.int64),
        "usage": usage.astype(np.int64), "support_contacts": support.astype(np.int64),
        "payment_failures": fail.astype(np.int64),
        "recorded_at": month_end + pd.to_timedelta(
            rng_rec.integers(0, cfg.signal_lag_days + 1, row.size), unit="D")})
    # Cancellations (M10 conventions)
    churned = exit_type != ""
    last_paid = acq + last_age
    cancel_rec = starts[last_paid[churned] + 1] + pd.to_timedelta(
        rng_rec.integers(0, 3, int(churned.sum())), unit="D")
    cancellations = pd.DataFrame({"customer_id": ids[churned],
                                  "last_paid_month": last_paid[churned].astype(np.int64),
                                  "recorded_at": cancel_rec})
    # Leakage trap: event-time indexed by the window's first month, recorded at closure
    w = cfg.closure_window_months
    churn_rows = np.flatnonzero(churned)
    in_window = churned[row] & (age > last_age[row] - w)
    final_usage = pd.Series(usage[in_window]).groupby(row[in_window]).mean()
    closure = pd.DataFrame({
        "customer_id": ids[churn_rows],
        "summary_month": np.maximum(acq[churn_rows],
                                    last_paid[churn_rows] - (w - 1)).astype(np.int64),
        "final_usage_mean": final_usage.reindex(churn_rows).to_numpy(),
        "exit_type": exit_type[churn_rows].astype(str),
        "recorded_at": cancel_rec + pd.to_timedelta(1, unit="D")})
    truth = SignalsTruth(
        cfg, ids.astype(np.int64), cust["segment"].to_numpy(dtype=str),
        cust["channel"].to_numpy(dtype=str), acq, k,
        pd.DataFrame({"customer_id": ids[row], "age": age, "month": month.astype(np.int64),
                      "state": st}),
        exit_type.astype(str))
    customers = cust[["customer_id", "acquisition_month", "acquisition_date", "channel",
                      "segment"]].copy()
    overhead = pd.DataFrame({"month": np.arange(cfg.n_months),
                             "marketing_overhead": cfg.monthly_overhead})
    return SignalsScenario(cfg, start, customers, tx, cancellations, signals, closure,
                           marketing, overhead, truth)


# ==========================================================================================
# Non-contractual purchasing (BG/NBD + Gamma-Gamma)
# ==========================================================================================
@dataclass(frozen=True)
class PurchasesScenarioConfig:
    seed: int = 1102
    n_months: int = 48
    start: str = "2022-01-01"
    acquisitions_per_month: tuple[float, ...] = _M10.acquisitions_per_month
    cost_per_acquisition: tuple[float, ...] = _M10.cost_per_acquisition
    segment_mix: tuple[tuple[float, float, float], ...] = _M10.segment_mix
    spend_noise_sd: float = _M10.spend_noise_sd
    onboarding_cost: float = _M10.onboarding_cost
    monthly_overhead: float = _M10.monthly_overhead
    late_payment_rate: float = _M10.late_payment_rate
    duplicate_rate: float = 0.0
    time_unit_days: float = 7.0
    # BG/NBD purchasing (rates per week)
    r: float = 0.75
    alpha: float = 6.0
    segment_alpha_scale: Triple = (1.25, 1.0, 0.75)
    dropout_a: float = 0.8
    dropout_b: tuple[float, ...] = (4.0, 3.5, 2.0, 5.0)
    # Pareto/NBD dropout (variant)
    dropout_process: str = "bgnbd"
    pareto_s: float = 1.2
    pareto_beta: float = 60.0
    # Gamma-Gamma spend
    gg_p: float = 6.0
    gg_q: float = 4.0
    gg_gamma: Triple = (10.0, 15.0, 25.0)
    spend_frequency_correlation: float = 0.0
    gross_margin: float = 0.40
    fee_rate: float = 0.03

    def __post_init__(self) -> None:
        if len(self.acquisitions_per_month) != len(CHANNELS) or len(self.dropout_b) != len(
                CHANNELS) or len(self.segment_mix) != len(CHANNELS):
            raise ValueError("every channel parameter needs one value per channel")
        if self.n_months < 3 or self.time_unit_days <= 0:
            raise ValueError("need n_months >= 3 and a positive time unit")
        positive = (self.r, self.alpha, self.dropout_a, self.pareto_s, self.pareto_beta,
                    self.gg_p, *self.segment_alpha_scale, *self.dropout_b, *self.gg_gamma)
        if min(positive) <= 0:
            raise ValueError("distribution parameters must be positive")
        if self.gg_q <= 1:
            raise ValueError("gg_q must exceed 1 (finite mean spend)")
        if self.dropout_process not in ("bgnbd", "pareto_nbd"):
            raise ValueError("dropout_process must be 'bgnbd' or 'pareto_nbd'")
        if not -1 < self.spend_frequency_correlation < 1:
            raise ValueError("spend_frequency_correlation must lie in (-1, 1)")
        if not 0 < self.fee_rate < self.gross_margin <= 1:
            raise ValueError("need 0 < fee_rate < gross_margin <= 1")
        if not 0 <= self.late_payment_rate < 1 or not 0 <= self.duplicate_rate < 1:
            raise ValueError("rates must lie in [0, 1)")


def purchases_variant(name: str, base: PurchasesScenarioConfig) -> PurchasesScenarioConfig:
    """A predeclared misspecification variant of ``base``."""
    from dataclasses import replace

    if name == "pareto_nbd_dropout":
        return replace(base, dropout_process="pareto_nbd")
    if name == "correlated_spend":
        return replace(base, spend_frequency_correlation=-0.5)
    raise KeyError(name)


@dataclass(frozen=True)
class PurchasesTruth:
    """Evaluation-only truth (never an estimator input). Times are in weeks
    (``config.time_unit_days``) since the acquisition purchase."""

    config: PurchasesScenarioConfig
    customer_ids: IntArray
    segment: NDArray[np.str_]
    channel: NDArray[np.str_]
    acquisition_month: IntArray
    acquisition_date: pd.DatetimeIndex
    purchase_rate: FloatArray
    """``lambda`` per week."""
    dropout_probability: FloatArray
    """BG/NBD ``p`` (NaN under Pareto/NBD dropout)."""
    death_rate: FloatArray
    """Pareto/NBD ``mu`` (NaN under BG/NBD)."""
    spend_rate: FloatArray
    """Gamma-Gamma ``nu`` (mean spend ``gg_p / nu``)."""
    death_time: FloatArray
    """Time of dropout; ``inf`` if a BG/NBD customer had not dropped out by the data end."""

    def weeks_to(self, timestamp: pd.Timestamp) -> FloatArray:
        seconds = (timestamp - self.acquisition_date).total_seconds()
        return np.asarray(seconds, dtype=np.float64) / (self.config.time_unit_days * 86400.0)

    def alive_at(self, month: int) -> NDArray[np.bool_]:
        """Acquired by ``month`` and not dropped out by its end."""
        t = self.weeks_to(_month_end(pd.Timestamp(self.config.start), month))
        return np.asarray((self.acquisition_month <= month) & (self.death_time > t))

    def expected_spend(self) -> FloatArray:
        return np.asarray(self.config.gg_p / self.spend_rate, dtype=np.float64)

    def expected_purchases_alive(self, weeks: float) -> FloatArray:
        """``E[purchases in the next `weeks` | alive, latent rates]``: BG/NBD
        ``(1 - exp(-lambda p t)) / p``; Pareto/NBD ``lambda / mu (1 - exp(-mu t))``."""
        return _expected_purchases(self.config.dropout_process, self.purchase_rate,
                                   self.dropout_probability, self.death_rate, weeks)

    def latent_expected_value(self, cutoff_month: int, horizon: int,
                              monthly_rate: float = 0.0) -> pd.DataFrame:
        """Per customer acquired by ``cutoff_month``: expected contribution of purchases in
        calendar months ``cutoff+1 .. cutoff+horizon`` given the latent rates and the latent
        alive state at the cutoff (zero if dropped out); month ``cutoff + j`` is discounted by
        ``d^j`` when ``monthly_rate`` is given. A decomposition reference only."""
        cfg = self.config
        start = pd.Timestamp(cfg.start)
        t0 = _month_end(start, cutoff_month)
        unit_s = cfg.time_unit_days * 86400.0
        acq = self.acquisition_month <= cutoff_month
        lam, p, mu = (self.purchase_rate[acq], self.dropout_probability[acq],
                      self.death_rate[acq])
        cum = np.zeros(int(acq.sum()))
        value = np.zeros_like(cum)
        for j in range(1, horizon + 1):
            weeks = (_month_end(start, cutoff_month + j) - t0).total_seconds() / unit_s
            new_cum = _expected_purchases(cfg.dropout_process, lam, p, mu, weeks)
            value += (new_cum - cum) * (1.0 + monthly_rate) ** -j
            cum = new_cum
        alive = self.alive_at(cutoff_month)[acq]
        per_purchase = self.expected_spend()[acq] * (cfg.gross_margin - cfg.fee_rate)
        return pd.DataFrame({"customer_id": self.customer_ids[acq], "alive": alive,
                             "expected": np.where(alive, value * per_purchase, 0.0)})


def _expected_purchases(process: str, lam: FloatArray, p: FloatArray, mu: FloatArray,
                        weeks: float) -> FloatArray:
    if process == "bgnbd":
        return np.asarray(-np.expm1(-lam * p * weeks) / p, dtype=np.float64)
    return np.asarray(lam / mu * -np.expm1(-mu * weeks), dtype=np.float64)


@dataclass(frozen=True)
class PurchasesScenario:
    config: PurchasesScenarioConfig
    start: pd.Timestamp
    customers: pd.DataFrame
    purchases: pd.DataFrame
    """``transaction_id, customer_id, billing_month, billing_date, recorded_at, revenue,
    cogs, payment_fee, cost_to_serve`` (revenue = spend; the first row per customer is the
    acquisition purchase). Same columns as M10, so M10's contribution applies."""
    cancellations: pd.DataFrame
    """Always empty: dropout is never observed in non-contractual data."""
    marketing: pd.DataFrame
    overhead: pd.DataFrame
    truth: PurchasesTruth = field(repr=False)

    @property
    def transactions(self) -> pd.DataFrame:
        return self.purchases

    def month_end(self, month: int) -> pd.Timestamp:
        return _month_end(self.start, month)


def generate_purchases_scenario(config: PurchasesScenarioConfig | None = None
                                ) -> PurchasesScenario:
    cfg = config or PurchasesScenarioConfig()
    rng_acq, rng_latent, rng_buy, rng_spend_amt, rng_rec, rng_spend = spawn_generators(
        cfg.seed, 6)
    start = pd.Timestamp(cfg.start)
    cust, marketing = _acquire(rng_acq, rng_spend, cfg.n_months, cfg.acquisitions_per_month,
                               cfg.segment_mix, cfg.cost_per_acquisition, cfg.spend_noise_sd,
                               cfg.onboarding_cost, start)
    n = len(cust)
    seg_i = cust["segment_index"].to_numpy()
    ch_i = np.array([CHANNELS.index(c) for c in cust["channel"]])
    # Latent rates: Gaussian copula on (lambda, nu); rho = 0 gives independence.
    rho = cfg.spend_frequency_correlation
    z1 = rng_latent.standard_normal(n)
    z2 = rho * z1 + math.sqrt(1.0 - rho * rho) * rng_latent.standard_normal(n)
    alpha_s = cfg.alpha * np.asarray(cfg.segment_alpha_scale)[seg_i]
    gamma_s = np.asarray(cfg.gg_gamma)[seg_i]
    lam = stats.gamma.ppf(special.ndtr(z1), cfg.r, scale=1.0 / alpha_s)
    nu = stats.gamma.ppf(special.ndtr(z2), cfg.gg_q, scale=1.0 / gamma_s)
    lam = np.maximum(np.asarray(lam, dtype=np.float64), 1e-12)
    nu = np.asarray(nu, dtype=np.float64)
    if cfg.dropout_process == "bgnbd":
        p = rng_latent.beta(cfg.dropout_a, np.asarray(cfg.dropout_b)[ch_i])
        mu = np.full(n, np.nan)
        tau = np.full(n, np.inf)
    else:
        p = np.full(n, np.nan)
        mu = rng_latent.gamma(cfg.pareto_s, 1.0 / cfg.pareto_beta, n)
        tau = rng_latent.exponential(1.0 / mu)
    acq_date = pd.DatetimeIndex(cust["acquisition_date"])
    data_end = _month_end(start, cfg.n_months - 1)
    horizon = np.asarray((data_end - acq_date).total_seconds(), dtype=np.float64) / (
        cfg.time_unit_days * 86400.0)
    # Repeat purchases, simulated purchase by purchase for the customers still active.
    death = tau.copy()  # BG/NBD: set at the dropout purchase; inf while still alive
    t = np.zeros(n)
    active = np.arange(n)
    times_row, times_t = [np.arange(n)], [np.zeros(n)]  # acquisition purchase at t = 0
    while active.size:
        t[active] += rng_buy.exponential(1.0 / lam[active])
        stop = (t[active] > horizon[active]) | (t[active] > tau[active])
        bought = active[~stop]
        times_row.append(bought)
        times_t.append(t[bought].copy())
        if cfg.dropout_process == "bgnbd" and bought.size:
            dies = rng_buy.random(bought.size) < p[bought]
            death[bought[dies]] = t[bought[dies]]
            bought = bought[~dies]
        active = bought
    row = np.concatenate(times_row)
    tt = np.concatenate(times_t)
    order = np.lexsort((tt, row))
    row, tt = row[order], tt[order]
    spend = rng_spend_amt.gamma(cfg.gg_p, 1.0 / nu[row])
    when = acq_date[row] + pd.to_timedelta(tt * cfg.time_unit_days * 86400.0, unit="s").round(
        "s")
    month = ((when.year - start.year) * 12 + (when.month - start.month)).to_numpy()
    purchases = pd.DataFrame({
        "transaction_id": np.arange(1, row.size + 1, dtype=np.int64),
        "customer_id": cust["customer_id"].to_numpy()[row],
        "billing_month": month.astype(np.int64),
        "billing_date": when,
        "recorded_at": when + pd.to_timedelta(
            _record_lags(rng_rec, row.size, cfg.late_payment_rate, normal_max=2), unit="D"),
        "revenue": np.round(spend, 2),
        "cogs": np.round(spend * (1.0 - cfg.gross_margin), 2),
        "payment_fee": np.round(spend * cfg.fee_rate, 2),
        "cost_to_serve": 0.0})
    purchases = _with_duplicates(purchases, rng_rec, cfg.duplicate_rate)
    truth = PurchasesTruth(
        cfg, cust["customer_id"].to_numpy(dtype=np.int64), cust["segment"].to_numpy(dtype=str),
        cust["channel"].to_numpy(dtype=str),
        cust["acquisition_month"].to_numpy().astype(np.int64), acq_date, lam, p, mu, nu,
        np.asarray(death, dtype=np.float64))
    customers = cust[["customer_id", "acquisition_month", "acquisition_date", "channel",
                      "segment"]].copy()
    cancellations = pd.DataFrame({"customer_id": pd.Series(dtype=np.int64),
                                  "last_paid_month": pd.Series(dtype=np.int64),
                                  "recorded_at": pd.Series(dtype="datetime64[ns]")})
    overhead = pd.DataFrame({"month": np.arange(cfg.n_months),
                             "marketing_overhead": cfg.monthly_overhead})
    return PurchasesScenario(cfg, start, customers, purchases, cancellations, marketing,
                             overhead, truth)
