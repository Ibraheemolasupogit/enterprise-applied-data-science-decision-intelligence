"""Synthetic subscription customer economics with known truth (Milestone 10).

Additive: nothing in M2–M9 is read or changed. A contractual monthly subscription business:

* **Acquisition.** Each calendar month ``m`` (0..n_months-1) and channel ``ch``, ``Poisson``
  new customers arrive on a uniform day of the month. Each channel has a true cost per
  acquisition (``media spend = CPA x arrivals x noise``, noise mean 1) and its own segment mix.
* **Plans and contribution.** Segment ``s`` fixes price, cost-to-serve and churn. In the first
  ``promo_months`` paid months, paid-media customers receive a price discount. A paid month
  ``a`` (0 = acquisition month) has revenue ``price x (1 - promo_a)``, cost of goods
  ``revenue x (1 - gross_margin)``, a payment fee ``fee_rate x revenue`` and a cost-to-serve
  ``cost_s x noise`` (lognormal, mean 1). Contribution = revenue - cost of goods - fee -
  cost-to-serve.
* **Churn.** Customer ``i`` has a constant monthly hazard ``lambda_i = b_{s,ch} Z_i`` with
  ``Z_i ~ Gamma(1/v, v)`` (mean 1, variance ``v``; ``v = 0`` means no heterogeneity) and
  ``b = -ln(1 - churn_s x multiplier_ch)``. The acquisition month is always paid; month
  ``a`` is paid with probability ``exp(-lambda_i a)``. Each customer's hazard is constant,
  but the *population* hazard falls with tenure when ``v > 0`` (the high-hazard customers
  leave first) — the mechanism behind the known bias of constant-churn heuristics.
* **Recording.** Payments are recorded 0-5 days after billing; ``late_payment_rate`` of them
  20-40 days after (crossing month ends). Cancellations are recorded on the first day after
  the last paid month (+0-2 days). Optional planted defect: duplicated transaction rows.

**Designed-in mechanisms (disclosed, not discoveries):** the premium segment has the highest
price but a high cost-to-serve, so ranking by revenue and by contribution differ; the cheapest
channel per acquisition (social) brings high-churn, low-value customers.

Ground truth (:class:`EconomicsTruth`) is returned separately from the observable tables and is
read only by evaluation code: per-customer hazards and analytic population expectations
(``E[exp(-b Z a)] = (1 + v b a)^(-1/v)``, the Gamma Laplace transform).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from decision_intelligence.common.seeding import spawn_generators
from decision_intelligence.common.validation import FloatArray

IntArray = NDArray[np.int64]
SEGMENTS = ("value", "standard", "premium")
CHANNELS = ("organic", "paid_search", "social", "referral")


@dataclass(frozen=True)
class EconomicsScenarioConfig:
    seed: int = 20261001
    n_months: int = 48
    start: str = "2022-01-01"
    acquisitions_per_month: tuple[float, ...] = (70.0, 90.0, 110.0, 40.0)
    cost_per_acquisition: tuple[float, ...] = (35.0, 70.0, 25.0, 45.0)
    segment_mix: tuple[tuple[float, float, float], ...] = (
        (0.35, 0.45, 0.20), (0.30, 0.45, 0.25), (0.65, 0.30, 0.05), (0.20, 0.50, 0.30))
    churn_multiplier: tuple[float, ...] = (1.0, 1.1, 1.6, 0.8)
    promo_channels: tuple[bool, ...] = (False, True, True, False)
    promo_months: int = 2
    promo_discount: float = 0.5
    price: tuple[float, ...] = (9.0, 16.0, 32.0)
    cost_to_serve: tuple[float, ...] = (3.0, 4.0, 17.0)
    monthly_churn: tuple[float, ...] = (0.09, 0.05, 0.045)
    gross_margin: float = 0.75
    fee_rate: float = 0.03
    cost_noise_sd: float = 0.25
    spend_noise_sd: float = 0.10
    frailty_variance: float = 0.5
    onboarding_cost: float = 6.0
    monthly_overhead: float = 3000.0
    late_payment_rate: float = 0.05
    duplicate_rate: float = 0.0

    def __post_init__(self) -> None:
        k = len(CHANNELS)
        if not (len(self.acquisitions_per_month) == len(self.cost_per_acquisition)
                == len(self.segment_mix) == len(self.churn_multiplier)
                == len(self.promo_channels) == k):
            raise ValueError("every channel parameter needs one value per channel")
        if not (len(self.price) == len(self.cost_to_serve) == len(self.monthly_churn)
                == len(SEGMENTS)):
            raise ValueError("every segment parameter needs one value per segment")
        if self.n_months < 2 or self.frailty_variance < 0:
            raise ValueError("need n_months >= 2 and frailty_variance >= 0")
        if any(abs(sum(m) - 1.0) > 1e-9 for m in self.segment_mix):
            raise ValueError("segment mixes must sum to 1")
        for c in self.monthly_churn:
            for mult in self.churn_multiplier:
                if not 0 < c * mult < 1:
                    raise ValueError("churn probabilities must lie in (0, 1)")
        if not 0 <= self.promo_discount < 1 or not 0 < self.gross_margin <= 1:
            raise ValueError("invalid promo discount or gross margin")
        if not 0 <= self.late_payment_rate < 1 or not 0 <= self.duplicate_rate < 1:
            raise ValueError("rates must lie in [0, 1)")


@dataclass(frozen=True)
class EconomicsTruth:
    """Evaluation-only truth: hazards and expected contribution (never an estimator input)."""

    config: EconomicsScenarioConfig
    customer_ids: IntArray
    segment: NDArray[np.str_]
    channel: NDArray[np.str_]
    acquisition_month: IntArray
    hazard: FloatArray
    """``lambda_i`` per customer."""

    def base_hazard(self, segment: str, channel: str) -> float:
        cfg = self.config
        c = cfg.monthly_churn[SEGMENTS.index(segment)] * cfg.churn_multiplier[
            CHANNELS.index(channel)]
        return -math.log(1.0 - c)

    def contribution_by_age(self, segment: str, channel: str, ages: IntArray) -> FloatArray:
        """Expected contribution of a *paid* month at each age (0 = acquisition month)."""
        cfg = self.config
        s, ch = SEGMENTS.index(segment), CHANNELS.index(channel)
        promo = (np.asarray(ages) < cfg.promo_months) & cfg.promo_channels[ch]
        revenue = cfg.price[s] * (1.0 - cfg.promo_discount * promo)
        return np.asarray(revenue * (cfg.gross_margin - cfg.fee_rate) - cfg.cost_to_serve[s],
                          dtype=np.float64)

    def population_paid_probability(self, segment: str, channel: str, ages: IntArray
                                    ) -> FloatArray:
        """``P(month a paid)`` averaged over the frailty: ``(1 + v b a)^(-1/v)``."""
        b = self.base_hazard(segment, channel)
        v = self.config.frailty_variance
        a = np.asarray(ages, dtype=np.float64)
        if v == 0:
            return np.asarray(np.exp(-b * a), dtype=np.float64)
        return np.asarray((1.0 + v * b * a) ** (-1.0 / v), dtype=np.float64)

    def population_value(self, *, horizon: int | None, monthly_rate: float,
                         segment: str | None = None, channel: str | None = None,
                         sum_months: int = 1200) -> float:
        """Expected discounted contribution per acquired customer from acquisition, over
        ``horizon`` months (``None``: summed to ``sum_months``), for a segment and/or channel
        (channels' segment mixes; arrivals-weighted when ``channel`` is None)."""
        cfg = self.config
        n = sum_months if horizon is None else horizon
        ages = np.arange(n)
        d = (1.0 + monthly_rate) ** -ages
        chans = [channel] if channel is not None else list(CHANNELS)
        total, weight = 0.0, 0.0
        for ch in chans:
            ci = CHANNELS.index(ch)
            w_ch = cfg.acquisitions_per_month[ci]
            for si, seg in enumerate(SEGMENTS):
                if segment is not None and seg != segment:
                    continue
                w = w_ch * cfg.segment_mix[ci][si]
                v = float(np.sum(self.contribution_by_age(seg, ch, ages)
                                 * self.population_paid_probability(seg, ch, ages) * d))
                total += w * v
                weight += w
        return total / weight

    def customer_value(self, *, horizon: int, monthly_rate: float,
                       from_age: int = 0, conditional_on_paid: bool = False) -> FloatArray:
        """Per customer: expected discounted contribution of ages ``from_age .. from_age +
        horizon - 1``, discounted to ``from_age``; optionally conditional on month
        ``from_age - 1`` having been paid (survival measured from there)."""
        ages = np.arange(from_age, from_age + horizon)
        out = np.empty(self.customer_ids.size)
        for seg in SEGMENTS:
            for ch in CHANNELS:
                sel = (self.segment == seg) & (self.channel == ch)
                if not sel.any():
                    continue
                c = self.contribution_by_age(seg, ch, ages)
                start = from_age - 1 if conditional_on_paid else 0
                surv = np.exp(-np.outer(self.hazard[sel], ages - start))
                d = (1.0 + monthly_rate) ** -(ages - from_age)
                out[sel] = surv @ (c * d)
        return out


@dataclass(frozen=True)
class EconomicsScenario:
    config: EconomicsScenarioConfig
    start: pd.Timestamp
    customers: pd.DataFrame
    """``customer_id, acquisition_month, acquisition_date, channel, segment``."""
    transactions: pd.DataFrame
    """``transaction_id, customer_id, billing_month, billing_date, recorded_at, revenue, cogs,
    payment_fee, cost_to_serve``."""
    cancellations: pd.DataFrame
    """``customer_id, last_paid_month, recorded_at``."""
    marketing: pd.DataFrame
    """``month, channel, new_customers, media_spend, onboarding_cost``."""
    overhead: pd.DataFrame
    """``month, marketing_overhead``."""
    truth: EconomicsTruth = field(repr=False)

    def month_end(self, month: int) -> pd.Timestamp:
        """The last instant of calendar month ``month`` (a cutoff)."""
        return self.start + pd.DateOffset(months=int(month) + 1) - pd.to_timedelta(1, unit="s")


def generate_economics_scenario(config: EconomicsScenarioConfig | None = None
                                ) -> EconomicsScenario:
    cfg = config or EconomicsScenarioConfig()
    rng_acq, rng_churn, rng_cost, rng_rec, rng_spend = spawn_generators(cfg.seed, 5)
    start = pd.Timestamp(cfg.start)
    month_starts = pd.DatetimeIndex([start + pd.DateOffset(months=m)
                                     for m in range(cfg.n_months + 2)])
    # Acquisition
    rows = []
    spend_rows = []
    for m in range(cfg.n_months):
        for ci, ch in enumerate(CHANNELS):
            n = int(rng_acq.poisson(cfg.acquisitions_per_month[ci]))
            seg = rng_acq.choice(len(SEGMENTS), size=n, p=cfg.segment_mix[ci])
            day = rng_acq.integers(0, 28, size=n)
            rows.append(pd.DataFrame({"acquisition_month": m, "channel": ch,
                                      "segment_index": seg, "day": day}))
            noise = rng_spend.lognormal(-cfg.spend_noise_sd ** 2 / 2, cfg.spend_noise_sd)
            spend_rows.append({"month": m, "channel": ch, "new_customers": n,
                               "media_spend": round(cfg.cost_per_acquisition[ci] * n * noise, 2),
                               "onboarding_cost": cfg.onboarding_cost * n})
    cust = pd.concat(rows, ignore_index=True)
    cust.insert(0, "customer_id", np.arange(1, len(cust) + 1, dtype=np.int64))
    cust["segment"] = np.array(SEGMENTS)[cust["segment_index"].to_numpy()]
    cust["acquisition_date"] = (month_starts[cust["acquisition_month"].to_numpy()]
                                + pd.to_timedelta(cust["day"].to_numpy(), unit="D"))
    # Churn
    v = cfg.frailty_variance
    n_c = len(cust)
    z = rng_churn.gamma(1.0 / v, v, size=n_c) if v > 0 else np.ones(n_c)
    seg_i = cust["segment_index"].to_numpy()
    ch_i = np.array([CHANNELS.index(c) for c in cust["channel"]])
    c_prob = np.array(cfg.monthly_churn)[seg_i] * np.array(cfg.churn_multiplier)[ch_i]
    hazard = -np.log(1.0 - c_prob) * z
    t = rng_churn.exponential(1.0 / np.maximum(hazard, 1e-300))
    paid_months = 1 + np.floor(t).astype(np.int64)
    acq = cust["acquisition_month"].to_numpy()
    last_paid = acq + paid_months - 1
    observable_months = np.minimum(paid_months, cfg.n_months - acq)
    # Transactions
    row = np.repeat(np.arange(n_c), observable_months)
    age = np.concatenate([np.arange(k) for k in observable_months])
    billing_month = acq[row] + age
    promo = (age < cfg.promo_months) & np.array(cfg.promo_channels)[ch_i[row]]
    revenue = np.array(cfg.price)[seg_i[row]] * (1.0 - cfg.promo_discount * promo)
    cost = np.array(cfg.cost_to_serve)[seg_i[row]] * rng_cost.lognormal(
        -cfg.cost_noise_sd ** 2 / 2, cfg.cost_noise_sd, size=row.size)
    billing_date = month_starts[billing_month] + pd.to_timedelta(
        cust["day"].to_numpy()[row], unit="D")
    late = rng_rec.random(row.size) < cfg.late_payment_rate
    lag = np.where(late, rng_rec.integers(20, 41, row.size), rng_rec.integers(0, 6, row.size))
    tx = pd.DataFrame({
        "transaction_id": np.arange(1, row.size + 1, dtype=np.int64),
        "customer_id": cust["customer_id"].to_numpy()[row],
        "billing_month": billing_month.astype(np.int64),
        "billing_date": billing_date,
        "recorded_at": billing_date + pd.to_timedelta(lag, unit="D"),
        "revenue": np.round(revenue, 2),
        "cogs": np.round(revenue * (1.0 - cfg.gross_margin), 2),
        "payment_fee": np.round(revenue * cfg.fee_rate, 2),
        "cost_to_serve": np.round(cost, 2)})
    if cfg.duplicate_rate > 0:
        dup = tx.loc[rng_rec.random(len(tx)) < cfg.duplicate_rate]
        tx = pd.concat([tx, dup]).sort_values("transaction_id", kind="stable").reset_index(
            drop=True)
    churned = last_paid < cfg.n_months - 1
    cancel_rec = month_starts[np.minimum(last_paid[churned] + 1, cfg.n_months + 1)] + \
        pd.to_timedelta(rng_rec.integers(0, 3, int(churned.sum())), unit="D")
    cancellations = pd.DataFrame({"customer_id": cust["customer_id"].to_numpy()[churned],
                                  "last_paid_month": last_paid[churned].astype(np.int64),
                                  "recorded_at": cancel_rec})
    truth = EconomicsTruth(cfg, cust["customer_id"].to_numpy(dtype=np.int64),
                           cust["segment"].to_numpy(dtype=str),
                           cust["channel"].to_numpy(dtype=str), acq.astype(np.int64), hazard)
    customers = cust[["customer_id", "acquisition_month", "acquisition_date", "channel",
                      "segment"]].copy()
    marketing = pd.DataFrame(spend_rows)
    overhead = pd.DataFrame({"month": np.arange(cfg.n_months),
                             "marketing_overhead": cfg.monthly_overhead})
    return EconomicsScenario(cfg, start, customers, tx, cancellations, marketing, overhead,
                             truth)
