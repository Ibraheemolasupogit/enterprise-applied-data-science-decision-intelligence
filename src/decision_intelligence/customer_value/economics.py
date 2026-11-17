"""Economic definitions: periods, discounting, contribution, CAC, payback, LTV/CAC (M10).

**Period convention.** Monthly periods; a customer's acquisition month is age 0 and every paid
month is billed at its start. A cash flow at age ``a`` is discounted by ``d^a`` with
``d = 1 / (1 + r_m)`` and ``r_m = (1 + R)^(1/12) - 1`` for an annual rate ``R`` — age 0 is not
discounted. A horizon of ``T`` months covers ages ``0 .. T-1``.

**Contribution** of a paid month = revenue - cost of goods - payment fee - cost-to-serve. Revenue
is not economic value; the two are always reported separately.

**Geometric value.** With a constant probability ``q`` of each further month being paid and
per-month contribution ``m``, the discounted value over ``T`` months is
``m (1 - s^T) / (1 - s)``, ``s = q d``; the infinite-horizon value ``m / (1 - s)`` exists only
if ``s < 1``. :func:`geometric_value` returns an :class:`Estimate` whose ``status`` is
``"undefined"`` when it does not exist.

**CAC** = acquisition cost / new customers over the same cohort months: *media* (media spend
only) or *fully loaded* (media + onboarding + marketing overhead allocated by new-customer
share). **Payback** = first age at which cumulative (discounted or undiscounted) contribution
per acquired customer reaches CAC; ``None`` if not within the curve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from decision_intelligence.common.validation import FloatArray

STATUS = ("ok", "unstable", "undefined")


@dataclass(frozen=True)
class Estimate:
    value: float
    status: str
    reason: str = ""

    def __post_init__(self) -> None:
        if self.status not in STATUS:
            raise ValueError(f"status must be one of {STATUS}")


def monthly_rate(annual: float) -> float:
    if annual <= -1:
        raise ValueError("annual rate must exceed -100%")
    return float((1.0 + annual) ** (1.0 / 12.0) - 1.0)


def discount_factors(n: int, rate: float) -> FloatArray:
    """``d^a`` for ages ``0 .. n-1``."""
    if n < 0 or rate <= -1:
        raise ValueError("n must be non-negative and rate > -1")
    return np.asarray((1.0 + rate) ** -np.arange(n), dtype=np.float64)


def contribution(transactions: pd.DataFrame) -> pd.Series:
    return (transactions["revenue"] - transactions["cogs"] - transactions["payment_fee"]
            - transactions["cost_to_serve"])


def geometric_value(m: float, q: float, rate: float, horizon: int | None) -> Estimate:
    """Discounted value of ``m`` per paid month with continuation probability ``q``."""
    if not np.isfinite(m) or not 0 <= q <= 1 or rate <= -1:
        return Estimate(float("nan"), "undefined", "invalid inputs")
    s = q / (1.0 + rate)
    if horizon is not None:
        if horizon < 0:
            raise ValueError("horizon must be non-negative")
        value = m * horizon if s == 1 else m * (1.0 - s ** horizon) / (1.0 - s)
        return Estimate(float(value), "ok")
    if s >= 1:
        return Estimate(float("inf"), "undefined",
                        "infinite horizon diverges: no churn and no discounting")
    return Estimate(float(m / (1.0 - s)), "ok")


def curve_value(contribution_by_age: ArrayLike, paid_probability: ArrayLike, rate: float
                ) -> float:
    """``sum_a m_a P(paid a) d^a`` over the given ages (starting at 0)."""
    m = np.asarray(contribution_by_age, dtype=np.float64)
    p = np.asarray(paid_probability, dtype=np.float64)
    if m.shape != p.shape:
        raise ValueError("contribution and probability curves must align")
    return float(np.sum(m * p * discount_factors(m.size, rate)))


def payback_age(cumulative_value: ArrayLike, cac: float) -> int | None:
    """First age whose cumulative value per acquired customer reaches ``cac``."""
    cum = np.asarray(cumulative_value, dtype=np.float64)
    hit = np.flatnonzero(cum >= cac)
    return int(hit[0]) if hit.size else None


@dataclass(frozen=True)
class CAC:
    channel: str
    new_customers: int
    media: float
    fully_loaded: float


def channel_cac(marketing: pd.DataFrame, overhead: pd.DataFrame, months: ArrayLike
                ) -> list[CAC]:
    """CAC per channel over acquisition ``months`` (overhead allocated by new-customer share
    within each month)."""
    mk = marketing.loc[marketing["month"].isin(np.asarray(months))]
    oh = overhead.set_index("month")["marketing_overhead"]
    per_month_total = mk.groupby("month")["new_customers"].sum()
    share = mk["new_customers"] / mk["month"].map(per_month_total).replace(0, np.nan)
    mk = mk.assign(overhead=(share * mk["month"].map(oh)).fillna(0.0))
    out = []
    for ch, g in mk.groupby("channel", sort=False):
        n = int(g["new_customers"].sum())
        if n == 0:
            continue
        out.append(CAC(str(ch), n, float(g["media_spend"].sum() / n),
                       float((g["media_spend"] + g["onboarding_cost"] + g["overhead"]).sum()
                             / n)))
    return out
