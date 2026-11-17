"""Retention: cohort tables, Kaplan–Meier with Greenwood intervals, constant churn (M10).

**Kaplan–Meier** (Kaplan & Meier 1958) on durations = paid months, events = observed churn,
censoring = still active at the cutoff: ``S(t) = prod_{t_j <= t} (1 - d_j / n_j)`` over distinct
event times, ``n_j`` at risk just before ``t_j`` (censored at ``t`` = at risk at ``t``).
**Greenwood** variance ``S(t)^2 sum d_j / (n_j (n_j - d_j))``; the plain interval
``S +/- z SE`` (clipped to [0, 1]) and the log-log interval are both provided. Valid under
non-informative censoring (censoring here is administrative: the cutoff).

For a customer, *month ``a`` paid* is *more than ``a`` paid months*, so the paid-month curve is
``P(paid at age a) = S(a)``, with ``S(0) = 1``. **Discrete-time censoring**: a customer still
active at the cutoff with ``t`` paid months is only known to have more than ``t - 1`` (the churn
decision after month ``t`` is not yet observed), so :func:`paid_month_km` censors them at
``t - 1``; churned customers are events at ``t``. (Censoring at ``t`` would assert survival past
``t`` and bias ``S`` upwards — found and fixed during M10; see the evidence.)

**Cohort retention table**: the share of each acquisition cohort paid at each age, reported only
for ages the cohort has completely reached by the cutoff.

**Constant monthly churn** over a trailing window: each paid month in the window whose outcome is
known (it is not the cutoff month) is a trial; an event is a recorded churn at its end;
``c = events / trials``. Its binomial standard error is reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats

from decision_intelligence.common.validation import FloatArray
from decision_intelligence.customer_value.panel import Observable


@dataclass(frozen=True)
class KaplanMeier:
    times: FloatArray
    """Distinct event times."""
    at_risk: FloatArray
    events: FloatArray
    survival: FloatArray
    se: FloatArray

    def at(self, t: ArrayLike) -> FloatArray:
        """``S(t)`` (right-continuous step function; 1 before the first event)."""
        tt = np.asarray(t, dtype=np.float64)
        idx = np.searchsorted(self.times, tt, side="right") - 1
        return np.asarray(np.where(idx >= 0, self.survival[np.maximum(idx, 0)], 1.0),
                          dtype=np.float64)

    def interval(self, level: float = 0.95, kind: str = "plain") -> tuple[FloatArray, FloatArray]:
        z = float(stats.norm.ppf(0.5 + level / 2))
        s = self.survival
        if kind == "plain":
            return np.clip(s - z * self.se, 0, 1), np.clip(s + z * self.se, 0, 1)
        if kind == "log-log":
            with np.errstate(divide="ignore", invalid="ignore"):
                v = (self.se / (s * np.log(s))) ** 2
                lo = s ** np.exp(z * np.sqrt(v))
                hi = s ** np.exp(-z * np.sqrt(v))
            return np.nan_to_num(lo, nan=0.0), np.nan_to_num(hi, nan=1.0)
        raise ValueError("kind must be 'plain' or 'log-log'")


def kaplan_meier(durations: ArrayLike, events: ArrayLike) -> KaplanMeier:
    t = np.asarray(durations, dtype=np.float64)
    e = np.asarray(events, dtype=bool)
    if t.shape != e.shape or t.size == 0 or np.any(t < 0) or not np.all(np.isfinite(t)):
        raise ValueError("durations must be finite, non-negative and match events")
    times = np.unique(t[e])
    at_risk = np.array([np.sum(t >= tj) for tj in times], dtype=np.float64)
    d = np.array([np.sum((t == tj) & e) for tj in times], dtype=np.float64)
    surv = np.asarray(np.cumprod(1.0 - d / at_risk), dtype=np.float64)
    # Greenwood: defined while S > 0 (at risk exceeds events); once every remaining customer
    # has the event, S = 0 and its standard error is 0.
    se = np.zeros_like(surv)
    alive = at_risk > d
    reached_zero = np.cumsum(~alive) > 0
    ok = ~reached_zero
    terms = np.zeros_like(surv)
    terms[alive] = d[alive] / (at_risk[alive] * (at_risk[alive] - d[alive]))
    se[ok] = surv[ok] * np.sqrt(np.cumsum(terms)[ok])
    return KaplanMeier(times, at_risk, d, surv, np.asarray(se, dtype=np.float64))


def paid_month_km(paid_months: ArrayLike, churned: ArrayLike) -> KaplanMeier:
    """KM of the number of paid months: churned = event at ``t``; active = censored at
    ``t - 1``."""
    t = np.asarray(paid_months, dtype=np.float64)
    e = np.asarray(churned, dtype=bool)
    return kaplan_meier(np.where(e, t, t - 1.0), e)


def paid_curve(km: KaplanMeier, n_ages: int) -> FloatArray:
    """``P(paid at age a)`` for ``a = 0 .. n_ages-1``."""
    return km.at(np.arange(n_ages))


def at_risk_curve(durations: ArrayLike, n_ages: int) -> FloatArray:
    t = np.asarray(durations, dtype=np.float64)
    return np.asarray([np.sum(t > a) for a in range(n_ages)], dtype=np.float64)


def cohort_retention_table(obs: Observable, max_age: int) -> pd.DataFrame:
    """Rows: acquisition cohort; columns: age; share paid; NaN where not fully reached."""
    st = obs.status()
    out = {}
    acq = st["acquisition_month"].to_numpy(dtype=np.int64)
    paid = st["paid_months"].to_numpy(dtype=np.int64)
    for cohort in np.unique(acq):
        g = paid[acq == cohort]
        reached = obs.cutoff_month - int(cohort)  # last age reached by the cutoff
        out[int(cohort)] = [float(np.mean(g > a)) if a <= reached else np.nan
                            for a in range(max_age)]
    return pd.DataFrame.from_dict(out, orient="index", columns=list(range(max_age)))


@dataclass(frozen=True)
class ChurnEstimate:
    churn: float
    events: int
    trials: int
    se: float


def constant_churn(obs: Observable, window_months: int, mask: ArrayLike | None = None
                   ) -> ChurnEstimate:
    """Monthly churn from trials in billing months ``(C - window, C - 1]``."""
    st = obs.status()
    if mask is not None:
        st = st.loc[np.asarray(mask, dtype=bool)]
    c = obs.cutoff_month
    lo, hi = c - window_months, c - 1
    acq = st["acquisition_month"].to_numpy()
    last = acq + st["paid_months"].to_numpy() - 1
    first_trial = np.maximum(acq, lo)
    last_trial = np.where(st["churned"].to_numpy(), last, np.minimum(last, hi))
    trials = int(np.sum(np.maximum(0, np.minimum(last_trial, hi) - first_trial + 1)))
    events = int(np.sum(st["churned"].to_numpy() & (last >= lo) & (last <= hi)))
    if trials == 0:
        return ChurnEstimate(float("nan"), 0, 0, float("nan"))
    p = events / trials
    return ChurnEstimate(p, events, trials, float(np.sqrt(p * (1 - p) / trials)))
