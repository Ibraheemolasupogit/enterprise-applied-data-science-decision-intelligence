"""Probabilistic contractual LTV models for Milestone 11 (P3): E1 and E2.

**Durations** (M10's discrete-time convention, :func:`~.retention.paid_month_km`). ``S(a) =
P(T > a) = P(month at age a is paid)``, ``S(0) = 1``, where ``T`` is the number of paid
months. A customer whose cancellation is recorded by the cutoff with ``t`` paid months is an
event, ``P(T = t) = S(t - 1) - S(t)``; an active customer with ``t`` paid months is censored,
``P(T > t - 1) = S(t - 1)`` (the churn decision after month ``t`` is not yet observed).

* **E1, shifted-beta-geometric** (Fader & Hardie 2007): each customer churns after every paid
  month with probability ``theta ~ Beta(alpha, beta)``, so
  ``S(a) = B(alpha, beta + a) / B(alpha, beta)`` and
  ``P(T = t) = B(alpha + 1, beta + t - 1) / B(alpha, beta)``.
* **E2, exponential-Gamma frailty (discrete Lomax)**: hazard ``b Z`` per month with
  ``Z ~ Gamma(1/v, v)`` (mean 1, variance ``v``), so ``S(a) = (1 + v b a)^(-1/v)``; as
  ``v -> 0`` it is the geometric model ``S(a) = exp(-b a)``. This is the M10 generator's own
  form within a segment x channel cell (correctly specified there) and is misspecified on the
  signals scenario (finite latent classes x hidden engagement states).

Both are fitted by maximum likelihood on the log parameters (``log alpha, log beta`` /
``log b, log v``), separately in each segment x channel cell and pooled, from M10's
:func:`~.panel.observe` at the cutoff (everything recorded by ``C``; cancellations of months
``< C`` are always recorded by ``C``). The likelihood uses the analytic score; the observed
information is a central difference of the score.

**Optimisation and fit status** (``predictive_protocol.STATUS_RULES``, classified by
:func:`~.model_selection.classify_fit`): ``n_starts`` deterministic starting points (the
crude geometric churn rate with five dispersions); L-BFGS-B in a box ``|log parameter| <=``
:data:`LOG_BOX` (wider than the protocol's instability bound, so a fit running to the boundary
is visible), then Newton steps from the best start. *Converged* means the projected score per
customer ``||grad mean log-lik||_inf <= gradient_tolerance`` and a last relative change of the
log-likelihood ``<= relative_change_tolerance``. Starting-point disagreement is the
log-likelihood gap between the best two starts. Events are churns recorded in the fit.

**Fallback** (``STATUS_RULES.fallback``): a cell whose fit is not ``ok`` uses the pooled fit of
the same model, labelled ``fallback``, if the pooled fit is ``ok``; otherwise the cell keeps
its own status (its estimates are used if finite, labelled with that status). Nothing is
dropped or refitted with other settings.

**E2 ``v = 0`` boundary exception** (amendment A6, ``predictive_protocol.E2_GEOMETRIC_BOUNDARY``).
Under homogeneous churn the correctly specified E2 MLE is ``v = 0``, on the boundary, which
the frozen rule marks ``unstable``. An E2 fit whose *only* problem is ``log v`` below the
bound is replaced by the geometric model (E2 at ``v = 0``) when that is verified to be its
limit: the closed-form geometric MLE ``b = -log(1 - events / decisions)`` agrees with the E2
``b``, its log-likelihood is not lower, the one-sided score for ``v`` at ``v = 0`` is not
positive (a boundary maximum), and the one-parameter fit is ``ok``. It is then ``ok``,
labelled (:attr:`Fit.boundary`) and counted (:attr:`CellFits.boundary_cells`). Every other
case follows the rules above unchanged.

**E1 geometric boundary exception** (amendment A7, ``predictive_protocol.E1_GEOMETRIC_BOUNDARY``).
E1's geometric limit is ``alpha, beta -> infinity`` with ``mu = alpha / (alpha + beta)`` fixed
(``Beta`` collapses to ``mu``; ``S(a) -> (1 - mu)^a``), the closed boundary ``phi = 1 / (alpha
+ beta) = 0``. Under homogeneous churn the iterates drift along that flat ridge, so the raw fit
is ``unstable`` (bound) or ``failed`` (not converged). It is replaced by the limit only when the
fit is on the ridge (``log(alpha + beta) > 10``, no lower bound), the closed-form ``mu`` agrees,
the geometric likelihood is not lower, the one-sided ``phi`` score at ``phi = 0``
(:func:`e1_dispersion_score_at_zero`) is not positive and the one-parameter fit is ``ok``; any
other failure reason or failed check keeps the frozen rules.

**Customer value.** For a customer active at ``C`` at paid age ``a0 = C - acquisition``
(month ``C`` paid), ``P(month a0 + j paid | F_C) = S(a0 + j) / S(a0)`` — the same conditioning
as the P2 oracles — times the expected contribution of a paid month at that age.
The window count of paid months has ``P(N >= j) = paid_j``, so the ``paid_*`` columns give its
full predictive distribution (parameter uncertainty is P7's).

**Expected contribution of a paid month** (:class:`ContributionModel`): the mean recorded
contribution of completely recorded months (``billing_month <= complete_through(C)``) in each
segment x channel x promotion-state cell; the promotion state of an age follows the business's
own :class:`~.features.PromotionPolicy` (Tier A information). Contribution carries no
information about churn in either contractual generator (cost noise is independent), so a
cell mean is the efficient Tier A estimate. A cell with fewer than ``STATUS_RULES.min_events``
recorded months uses the segment x promotion-state mean, then the promotion-state mean
(labelled in :attr:`ContributionModel.table`).

**No hyperparameters.** E1 and E2 have no penalty or tuning parameter (the protocol's penalty
grid applies to E3 and E5). Temporal validation (:func:`validate`) refits each model with the
information of the protocol's validation landmark ``L_v`` only and scores it on the
landmark's labels, which are completely recorded by ``C``; the holdout is never used.

Estimation code only: it reads observable tables and never a generator's parameters, latent
truth or oracle (AST-tested).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, special

from decision_intelligence.common.validation import FloatArray
from decision_intelligence.customer_value import predictive_protocol as pp
from decision_intelligence.customer_value.features import PromotionPolicy, contractual_features
from decision_intelligence.customer_value.landmarks import training_stack, view_at
from decision_intelligence.customer_value.model_selection import (
    CandidateResult,
    FitDiagnostics,
    classify_fit,
)
from decision_intelligence.customer_value.panel import EconomicsTables, observe

MODEL_IDS = ("E1", "E2")
PARAMETER_NAMES = {"E1": ("alpha", "beta"), "E2": ("b", "v")}
LOG_BOX = 15.0
"""Optimiser box for the log parameters (the protocol's instability bound is 10)."""
START_DISPERSIONS = (0.3, 1.0, 3.0, 10.0, 30.0)
"""E1 starts: ``alpha + beta``; E2 starts: ``1 / v``. Deterministic (no random starts)."""
_FD_STEP = 1e-5
_MAX_NEWTON = 100


# --- duration data ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DurationData:
    """Aggregated durations: distinct ``(paid_months, event)`` pairs with their counts."""

    t: FloatArray
    event: np.ndarray
    count: FloatArray

    @classmethod
    def from_customers(cls, paid_months: np.ndarray, churned: np.ndarray) -> DurationData:
        t = np.asarray(paid_months, dtype=np.int64)
        e = np.asarray(churned, dtype=bool)
        if t.shape != e.shape or np.any(t < 1):
            raise ValueError("paid months must be >= 1 and match the churn flags")
        if t.size == 0:
            return cls(np.zeros(0), np.zeros(0, dtype=bool), np.zeros(0))
        keys, counts = np.unique(np.column_stack([t, e.astype(np.int64)]), axis=0,
                                 return_counts=True)
        return cls(keys[:, 0].astype(np.float64), keys[:, 1].astype(bool),
                   counts.astype(np.float64))

    @property
    def n_customers(self) -> int:
        return int(self.count.sum())

    @property
    def n_events(self) -> int:
        return int(self.count[self.event].sum())

    @property
    def n_decisions(self) -> int:
        """Observed churn decisions (survived or not): ``t`` per event, ``t - 1`` censored."""
        return int(np.sum(self.count * np.where(self.event, self.t, self.t - 1)))


# --- E1 / E2 log-survival, log-event probability and their scores (log parameters) ---------
def _check(model: str) -> None:
    if model not in MODEL_IDS:
        raise ValueError(f"model must be one of {MODEL_IDS}")


def _g_ratio(x: FloatArray) -> FloatArray:
    """``log1p(x) - x / (1 + x)`` without cancellation for small ``x``."""
    small = x < 1e-4
    xs = np.where(small, x, 0.0)
    series = xs * xs / 2 - 2 * xs ** 3 / 3 + 3 * xs ** 4 / 4
    return np.asarray(np.where(small, series, np.log1p(x) - x / (1 + x)), dtype=np.float64)


def log_survival(model: str, theta: FloatArray, a: FloatArray) -> tuple[FloatArray, FloatArray]:
    """``log S(a)`` and its gradient with respect to the log parameters, shape ``(n, 2)``."""
    _check(model)
    a = np.asarray(a, dtype=np.float64)
    p0, p1 = math.exp(theta[0]), math.exp(theta[1])
    if model == "E1":
        alpha, beta = p0, p1
        val = special.betaln(alpha, beta + a) - special.betaln(alpha, beta)
        dig = special.digamma(alpha + beta + a)
        g0 = alpha * (special.digamma(alpha + beta) - dig)
        g1 = beta * (special.digamma(beta + a) - dig - special.digamma(beta)
                     + special.digamma(alpha + beta))
    else:
        b, v = p0, p1
        if v == 0.0:  # the geometric limit (amendment A6); the log-v score is not defined
            val, g0, g1 = -b * a, -b * a, np.zeros_like(a)
        else:
            x = v * b * a
            val = -np.log1p(x) / v
            g0 = -b * a / (1 + x)
            g1 = _g_ratio(x) / v
    return np.asarray(val, dtype=np.float64), np.column_stack([g0, g1])


def log_event(model: str, theta: FloatArray, t: FloatArray) -> tuple[FloatArray, FloatArray]:
    """``log P(T = t) = log(S(t - 1) - S(t))`` and its gradient (log parameters)."""
    _check(model)
    t = np.asarray(t, dtype=np.float64)
    if model == "E1":
        alpha, beta = math.exp(theta[0]), math.exp(theta[1])
        val = special.betaln(alpha + 1, beta + t - 1) - special.betaln(alpha, beta)
        dig = special.digamma(alpha + beta + t)
        g0 = alpha * (special.digamma(alpha + 1) - dig - special.digamma(alpha)
                      + special.digamma(alpha + beta))
        g1 = beta * (special.digamma(beta + t - 1) - dig - special.digamma(beta)
                     + special.digamma(alpha + beta))
        return np.asarray(val, dtype=np.float64), np.column_stack([g0, g1])
    u, gu = log_survival(model, theta, t - 1)
    w, gw = log_survival(model, theta, t)
    d = w - u
    one_minus_q = -np.expm1(d)
    q = np.exp(d)
    val = u + np.log(one_minus_q)
    grad = (gu - q[:, None] * gw) / one_minus_q[:, None]
    return np.asarray(val, dtype=np.float64), np.asarray(grad, dtype=np.float64)


def log_likelihood(model: str, theta: FloatArray, data: DurationData) -> tuple[float, FloatArray]:
    """Total log-likelihood and score (log parameters)."""
    ev, cens = data.event, ~data.event
    total, grad = 0.0, np.zeros(2)
    if ev.any():
        v, g = log_event(model, theta, data.t[ev])
        total += float(np.sum(data.count[ev] * v))
        grad += (data.count[ev][:, None] * g).sum(axis=0)
    if cens.any():
        v, g = log_survival(model, theta, data.t[cens] - 1)
        total += float(np.sum(data.count[cens] * v))
        grad += (data.count[cens][:, None] * g).sum(axis=0)
    return total, grad


def observed_information(model: str, theta: FloatArray, data: DurationData) -> FloatArray:
    """``-d^2 loglik / d theta^2`` by central differences of the analytic score."""
    h = np.zeros((2, 2))
    for k in range(2):
        e = np.zeros(2)
        e[k] = _FD_STEP
        h[:, k] = (log_likelihood(model, theta + e, data)[1]
                   - log_likelihood(model, theta - e, data)[1]) / (2 * _FD_STEP)
    return np.asarray(-(h + h.T) / 2, dtype=np.float64)


def survival(model: str, params: tuple[float, float], ages: FloatArray) -> FloatArray:
    """``S(a)`` at the natural parameters (E2 ``v = 0``: the geometric limit)."""
    with np.errstate(divide="ignore"):
        theta = np.log(np.asarray(params, dtype=np.float64))
    return np.asarray(np.exp(log_survival(model, theta, np.asarray(ages, np.float64))[0]),
                      dtype=np.float64)


# --- fitting ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class Fit:
    model: str
    params: tuple[float, float]
    """Natural parameters (``alpha, beta`` or ``b, v``); NaN when undefined."""
    log_params: FloatArray
    cov_log: FloatArray
    """Inverse observed information of the log parameters (NaN if not positive definite)."""
    log_likelihood: float
    n_customers: int
    n_events: int
    status: str
    reasons: tuple[str, ...]
    diagnostics: FitDiagnostics
    start_log_likelihoods: tuple[float, ...]
    boundary: str | None = None
    """The protocol label when this is a verified geometric boundary limit: E2 ``v = 0``
    (amendment A6; ``v`` fixed, no SE) or E1 ``alpha + beta -> infinity`` (amendment A7)."""
    geometric_churn: float | None = None
    """E1 geometric limit only (A7): the monthly churn probability ``mu``; ``alpha`` and
    ``beta`` are then infinite and ``S(a) = (1 - mu)^a``."""

    @property
    def se(self) -> tuple[float, float]:
        """Delta-method standard errors of the natural parameters (NaN where undefined)."""
        d = np.sqrt(np.diag(self.cov_log)) * np.asarray(self.params)
        return float(d[0]), float(d[1])

    @property
    def finite(self) -> bool:
        """Usable estimates: finite natural parameters (``v = 0`` included) or the E1
        geometric limit."""
        return self.geometric_churn is not None or bool(np.all(np.isfinite(self.params)))


def _starts(model: str, data: DurationData) -> list[FloatArray]:
    p = data.n_events / max(data.n_decisions, 1)
    p = min(max(p, 1e-3), 0.5)
    out = []
    for k in START_DISPERSIONS:
        if model == "E1":
            out.append(np.log([p * k, (1 - p) * k]))
        else:
            out.append(np.log([-math.log1p(-p), 1.0 / k]))
    return [np.clip(s, -LOG_BOX, LOG_BOX) for s in out]


def _projected(theta: FloatArray, grad: FloatArray) -> FloatArray:
    g = grad.copy()
    g[(theta >= LOG_BOX - 1e-9) & (g > 0)] = 0.0
    g[(theta <= -LOG_BOX + 1e-9) & (g < 0)] = 0.0
    return g


def _optimise(model: str, data: DurationData, start: FloatArray
              ) -> tuple[FloatArray, float, float, float]:
    """One start: L-BFGS-B then Newton. Returns ``theta, loglik, projected score per customer,
    last relative change``."""
    n = data.n_customers

    def neg(th: FloatArray) -> tuple[float, FloatArray]:
        ll, g = log_likelihood(model, th, data)
        if not math.isfinite(ll):
            return 1e300, np.zeros(2)
        return -ll / n, -g / n

    res = optimize.minimize(neg, start, jac=True, method="L-BFGS-B",
                            bounds=[(-LOG_BOX, LOG_BOX)] * 2,
                            options={"maxiter": 2000, "ftol": 1e-15, "gtol": 1e-12})
    theta = np.clip(np.asarray(res.x, dtype=np.float64), -LOG_BOX, LOG_BOX)
    ll, g = log_likelihood(model, theta, data)
    rel = math.inf
    for _ in range(_MAX_NEWTON):
        if np.max(np.abs(_projected(theta, g))) / n <= pp.STATUS_RULES.gradient_tolerance * 1e-3:
            rel = 0.0
            break
        info = observed_information(model, theta, data)
        try:
            ok = bool(np.all(np.linalg.eigvalsh(info) > 0))
        except np.linalg.LinAlgError:
            ok = False
        if not ok:
            break
        step = np.linalg.solve(info, g)
        improved = False
        for _half in range(40):
            cand = np.clip(theta + step, -LOG_BOX, LOG_BOX)
            ll_c, g_c = log_likelihood(model, cand, data)
            if math.isfinite(ll_c) and ll_c >= ll - 1e-12 * max(1.0, abs(ll)):
                rel = abs(ll_c - ll) / max(1.0, abs(ll))
                theta, ll, g = cand, ll_c, g_c
                improved = True
                break
            step = step / 2
        if not improved or rel <= 1e-15:
            break
    if not math.isfinite(rel):
        rel = 0.0 if res.success else math.inf
    pg = float(np.max(np.abs(_projected(theta, g)))) / n
    return theta, ll, pg, rel


def fit(model: str, data: DurationData) -> Fit:
    """Maximum-likelihood fit with the protocol's fit-status rules."""
    _check(model)
    rules = pp.STATUS_RULES
    nan2 = np.full(2, np.nan)
    if data.n_decisions == 0:
        d = FitDiagnostics(has_data=False, n_events=data.n_events, n_parameters=2)
        status, reasons = classify_fit(d)
        return Fit(model, (math.nan, math.nan), nan2, np.full((2, 2), np.nan), math.nan,
                   data.n_customers, data.n_events, status, reasons, d, ())
    runs = [_optimise(model, data, s) for s in _starts(model, data)]
    lls = sorted((r[1] for r in runs), reverse=True)
    best = max(runs, key=lambda r: r[1])
    theta, ll, pg, rel = best
    info = observed_information(model, theta, data)
    eig = np.linalg.eigvalsh(info) if np.all(np.isfinite(info)) else np.array([-1.0])
    pd_ok = bool(float(np.min(eig)) > 1e-12 * max(float(np.max(np.abs(eig))), 1e-300))
    cov = np.linalg.inv(info) if pd_ok else np.full((2, 2), np.nan)
    finite = math.isfinite(ll) and bool(np.all(np.isfinite(theta)))
    d = FitDiagnostics(
        has_data=True,
        converged=pg <= rules.gradient_tolerance and rel <= rules.relative_change_tolerance,
        finite_likelihood=finite, hessian_positive_definite=pd_ok,
        max_abs_log_parameter=float(np.max(np.abs(theta))),
        start_log_likelihood_gap=float(lls[0] - lls[1]) if len(lls) > 1 else 0.0,
        n_events=data.n_events, n_parameters=2)
    status, reasons = classify_fit(d)
    params = (float(math.exp(theta[0])), float(math.exp(theta[1])))
    return Fit(model, params, theta, np.asarray(cov, dtype=np.float64), float(ll),
               data.n_customers, data.n_events, status, reasons, d,
               tuple(float(r[1]) for r in runs))


# --- the E2 v = 0 boundary exception (amendment A6) ------------------------------------------
def geometric_mle(data: DurationData) -> tuple[float, float, float]:
    """Closed-form geometric fit (E2 with ``v = 0``): ``b = -log(1 - events / decisions)``.
    Returns ``b``, its log-likelihood and the observed information of ``log b`` (NaN if the
    MLE does not exist: no events, or every decision a churn)."""
    e = float(np.sum(data.count[data.event]))
    d = float(np.sum(data.count * np.where(data.event, data.t, data.t - 1)))
    if e <= 0 or e >= d:
        return math.nan, math.nan, math.nan
    p = e / d
    b = -math.log1p(-p)
    ll = e * math.log(p) + (d - e) * math.log1p(-p)
    info_log_b = b * b * e * math.exp(-b) / (-math.expm1(-b)) ** 2
    return b, ll, info_log_b


def e2_v_score_at_zero(b: float, data: DurationData) -> float:
    """``d loglik / d v`` of E2 at ``(b, v = 0)`` (natural ``v``; one-sided at the boundary):
    ``d log S(a) / dv = (b a)^2 / 2`` at ``v = 0``."""
    def s(a: FloatArray) -> FloatArray:
        return np.asarray((b * a) ** 2 / 2.0, dtype=np.float64)

    t, ev, n = data.t, data.event, data.count
    q = math.exp(-b)
    cens = float(np.sum(n[~ev] * s(t[~ev] - 1)))
    te = t[ev]
    events = float(np.sum(n[ev] * (s(te - 1) - q * s(te)) / (1.0 - q)))
    return cens + events


def geometric_boundary(f: Fit, data: DurationData) -> Fit | None:
    """The verified ``v = 0`` limit of an E2 fit, or None when any condition of
    ``predictive_protocol.E2_GEOMETRIC_BOUNDARY`` fails (then every other rule applies)."""
    rule, bound = pp.E2_GEOMETRIC_BOUNDARY, pp.STATUS_RULES.log_parameter_bound
    at_bound = "parameter at the log-scale bound"
    if (f.model != rule.model or f.status != "unstable" or at_bound not in f.reasons
            or not set(f.reasons) <= {at_bound, "Hessian not positive definite"}
            or not f.diagnostics.converged or not f.diagnostics.finite_likelihood):
        return None
    log_b, log_v = float(f.log_params[0]), float(f.log_params[1])
    if not (log_v < -bound and abs(log_b) <= bound):
        return None
    b, ll, info = geometric_mle(data)
    if not math.isfinite(b) or abs(math.log(b) - log_b) > rule.log_rate_tolerance:
        return None
    if ll < f.log_likelihood - rule.log_likelihood_tolerance * max(1.0, abs(f.log_likelihood)):
        return None
    if e2_v_score_at_zero(b, data) / data.n_customers > rule.max_score_per_customer:
        return None
    d = FitDiagnostics(has_data=True, converged=True, finite_likelihood=math.isfinite(ll),
                       hessian_positive_definite=info > 0,
                       max_abs_log_parameter=abs(math.log(b)), start_log_likelihood_gap=0.0,
                       n_events=data.n_events, n_parameters=1)
    status, reasons = classify_fit(d)
    if status != "ok":
        return None
    cov = np.full((2, 2), np.nan)
    cov[0, 0] = 1.0 / info
    return Fit(f.model, (b, 0.0), np.array([math.log(b), -math.inf]), cov, ll,
               data.n_customers, data.n_events, status, reasons, d, f.start_log_likelihoods,
               boundary=rule.label)


# --- the E1 geometric boundary exception (amendment A7) ---------------------------------------
def e1_dispersion_score_at_zero(mu: float, data: DurationData) -> float:
    """``d loglik / d phi`` of E1 at ``phi = 1 / (alpha + beta) = 0`` with mean churn ``mu``
    (one-sided at the boundary). ``Var(theta) = mu (1 - mu) phi / (1 + phi)`` and higher central
    moments are ``O(phi^2)``, so ``d log E[g(theta)] / d phi = g''(mu) mu (1 - mu) / (2 g(mu))``:
    censored at ``a = t - 1``: ``a (a - 1) mu / (2 (1 - mu))``; event at ``t``:
    ``-(t - 1) + (t - 1)(t - 2) mu / (2 (1 - mu))``."""
    t, ev, n = data.t, data.event, data.count
    k = mu / (2.0 * (1.0 - mu))
    a = t[~ev] - 1
    cens = float(np.sum(n[~ev] * a * (a - 1) * k))
    te = t[ev]
    events = float(np.sum(n[ev] * (-(te - 1) + (te - 1) * (te - 2) * k)))
    return cens + events


def e1_log_likelihood_product(log_params: FloatArray, data: DurationData) -> float:
    """E1 log-likelihood from the exact finite products (integer durations):
    ``log S(a) = sum_{i < a} log1p(-alpha / (alpha + beta + i))`` and
    ``log P(T = t) = log(alpha / (alpha + beta + t - 1)) + log S(t - 1)``. Well conditioned at
    very large ``alpha, beta`` (the ``betaln`` differences lose about 1e-9 relative there), so
    A7 compares likelihoods with it; the fit itself is unchanged."""
    alpha, beta = math.exp(float(log_params[0])), math.exp(float(log_params[1]))
    t = data.t.astype(np.int64)
    i = np.arange(int(np.max(t, initial=1)) + 1, dtype=np.float64)
    cs = np.concatenate([[0.0], np.cumsum(np.log1p(-alpha / (alpha + beta + i)))])
    ev = data.event
    log_s = cs[t - 1]
    log_p = np.log(alpha / (alpha + beta + t - 1)) + log_s
    return float(np.sum(data.count * np.where(ev, log_p, log_s)))


def e1_geometric_boundary(f: Fit, data: DurationData) -> Fit | None:
    """The verified ``alpha + beta -> infinity`` (geometric) limit of an E1 fit, or None when
    any condition of ``predictive_protocol.E1_GEOMETRIC_BOUNDARY`` fails."""
    rule, rules = pp.E1_GEOMETRIC_BOUNDARY, pp.STATUS_RULES
    bound = rules.log_parameter_bound
    at_bound, hess = "parameter at the log-scale bound", "Hessian not positive definite"
    d0 = f.diagnostics
    if f.model != rule.model:
        return None
    if f.status == "unstable":
        if at_bound not in f.reasons or not set(f.reasons) <= {at_bound, hess}:
            return None
    elif f.status != "failed" or f.reasons != ("not converged",):
        return None
    if (not d0.finite_likelihood or d0.separated_without_penalty
            or d0.start_log_likelihood_gap > rules.start_agreement_tolerance
            or not np.all(np.isfinite(f.log_params))):
        return None
    log_a, log_b = float(f.log_params[0]), float(f.log_params[1])
    log_kappa = float(np.logaddexp(log_a, log_b))
    if min(log_a, log_b) < -bound or log_kappa <= bound:
        return None  # not on the geometric ridge (e.g. a degenerate lower boundary)
    b, ll, info = geometric_mle(data)
    if not math.isfinite(b):
        return None
    mu = -math.expm1(-b)
    mu_fit = math.exp(log_a - log_kappa)
    if abs(math.log(mu_fit) - math.log(mu)) > rule.log_rate_tolerance:
        return None
    ll_fit = e1_log_likelihood_product(f.log_params, data)
    if ll < ll_fit - rule.log_likelihood_tolerance * max(1.0, abs(ll_fit)):
        return None
    if e1_dispersion_score_at_zero(mu, data) / data.n_customers > rule.max_score_per_customer:
        return None
    d = FitDiagnostics(has_data=True, converged=True, finite_likelihood=math.isfinite(ll),
                       hessian_positive_definite=info > 0,
                       max_abs_log_parameter=abs(math.log(b)), start_log_likelihood_gap=0.0,
                       n_events=data.n_events, n_parameters=1)
    status, reasons = classify_fit(d)
    if status != "ok":
        return None
    return Fit(f.model, (math.inf, math.inf), np.array([math.inf, math.inf]),
               np.full((2, 2), np.nan), ll, data.n_customers, data.n_events, status, reasons,
               d, f.start_log_likelihoods, boundary=rule.label, geometric_churn=mu)


# --- expected contribution of a paid month -----------------------------------------------------
@dataclass(frozen=True)
class ContributionModel:
    table: pd.DataFrame
    """Index ``(segment, channel, promo)``; columns ``mean, n, source``
    (``cell`` / ``segment`` / ``pooled``)."""
    policy: PromotionPolicy

    def expected(self, segment: np.ndarray, channel: np.ndarray, ages: np.ndarray
                 ) -> FloatArray:
        """``segment``, ``channel`` shape ``(n,)``; ``ages`` shape ``(n, H)``."""
        promoted = np.isin(channel, self.policy.channels)
        promo = (ages < self.policy.months) & promoted[:, None]
        out = np.full(ages.shape, np.nan)
        for key, mean in zip(self.table.index.tolist(), self.table["mean"].tolist(),
                             strict=True):
            seg, ch, pr = key
            sel = (segment == seg) & (channel == ch)
            if sel.any():
                out[sel[:, None] & (promo == bool(pr))] = float(mean)
        return out


def _group_means(d: pd.DataFrame, cols: list[str]
                 ) -> dict[tuple[object, ...], tuple[float, int]]:
    agg = d.groupby(cols)["c"].agg(["mean", "size"])
    keys = [k if isinstance(k, tuple) else (k,) for k in agg.index.tolist()]
    return {tuple(bool(x) if isinstance(x, np.bool_) else x for x in k): (float(m), int(n))
            for k, m, n in zip(keys, agg["mean"].tolist(), agg["size"].tolist(), strict=True)}


def fit_contribution(tables: EconomicsTables, cutoff: int, policy: PromotionPolicy
                     ) -> ContributionModel:
    """Cell x promotion-state mean contribution from completely recorded months at ``cutoff``."""
    obs = observe(tables, cutoff)
    tx = obs.transactions.loc[obs.transactions["billing_month"] <= obs.complete_through()]
    cust = obs.customers.set_index("customer_id")
    seg = cust["segment"].reindex(tx["customer_id"]).to_numpy()
    ch = cust["channel"].reindex(tx["customer_id"]).to_numpy()
    promo = (tx["age"].to_numpy() < policy.months) & np.isin(ch, policy.channels)
    d = pd.DataFrame({"segment": seg, "channel": ch, "promo": promo,
                      "c": tx["contribution"].to_numpy()})
    need = pp.STATUS_RULES.min_events
    cell = _group_means(d, ["segment", "channel", "promo"])
    by_seg = _group_means(d, ["segment", "promo"])
    by_promo = _group_means(d, ["promo"])
    rows: list[tuple[str, str, bool, float, int, str]] = []
    for s in sorted(cust["segment"].unique()):
        for c in sorted(cust["channel"].unique()):
            states = (False, True) if c in policy.channels else (False,)
            for pr in states:
                for source, hit in (("cell", cell.get((s, c, pr))),
                                    ("segment", by_seg.get((s, pr))),
                                    ("pooled", by_promo.get((pr,)))):
                    if hit is not None and (hit[1] >= need or source == "pooled"):
                        rows.append((s, c, pr, hit[0], hit[1], source))
                        break
                else:
                    rows.append((s, c, pr, math.nan, 0, "none"))
    table = pd.DataFrame(rows, columns=["segment", "channel", "promo", "mean", "n", "source"])
    return ContributionModel(table.set_index(["segment", "channel", "promo"]), policy)


# --- cells, fallback and prediction ----------------------------------------------------------
@dataclass(frozen=True)
class CellFits:
    model: str
    cells: dict[tuple[str, str], Fit]
    """The unrestricted ML fit of every cell, with its own status."""
    pooled: Fit
    used: dict[tuple[str, str], tuple[str, Fit]] = field(default_factory=dict)
    """Per cell: (reported status, the fit whose parameters are used)."""
    pooled_used: Fit | None = None
    """The pooled fit used as fallback (its A6 geometric limit when that applies)."""

    @property
    def boundary_cells(self) -> tuple[tuple[str, str], ...]:
        """Cells using the verified ``v = 0`` geometric limit (amendment A6)."""
        return tuple(k for k, (s, f) in self.used.items()
                     if s == "ok" and f.boundary is not None)

    @property
    def ok_cell_fraction(self) -> float:
        if not self.used:
            return math.nan
        return sum(s in ("ok", "fallback") for s, _ in self.used.values()) / len(self.used)

    def status_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s, _ in self.used.values():
            out[s] = out.get(s, 0) + 1
        return out


def _usable(f: Fit, data: DurationData, boundary_exception: bool,
            e1_boundary_exception: bool) -> Fit | None:
    """``f`` if ``ok``; else its verified geometric limit (E2: A6, E1: A7) when enabled;
    else None."""
    if f.status == "ok":
        return f
    if f.model == "E2" and boundary_exception:
        return geometric_boundary(f, data)
    if f.model == "E1" and e1_boundary_exception:
        return e1_geometric_boundary(f, data)
    return None


def fit_cells(model: str, status: pd.DataFrame, *, boundary_exception: bool = True,
              e1_boundary_exception: bool = True) -> CellFits:
    """Per segment x channel fits, the pooled fit and the protocol fallback.

    ``boundary_exception=False`` reproduces the E2 rule before amendment A6, and
    ``e1_boundary_exception=False`` the E1 rule before amendment A7; they exist only for
    the before/after record and is never the protocol's path."""
    paid = status["paid_months"].to_numpy()
    churned = status["churned"].to_numpy(dtype=bool)
    pooled_data = DurationData.from_customers(paid, churned)
    pooled = fit(model, pooled_data)
    pooled_used = _usable(pooled, pooled_data, boundary_exception, e1_boundary_exception)
    cells: dict[tuple[str, str], Fit] = {}
    used: dict[tuple[str, str], tuple[str, Fit]] = {}
    seg, ch = status["segment"].to_numpy(), status["channel"].to_numpy()
    for s in sorted(set(seg)):
        for c in sorted(set(ch)):
            sel = (seg == s) & (ch == c)
            data = DurationData.from_customers(paid[sel], churned[sel])
            f = fit(model, data)
            cells[(s, c)] = f
            own = _usable(f, data, boundary_exception, e1_boundary_exception)
            if own is not None:
                used[(s, c)] = ("ok", own)
            elif pooled_used is not None:
                used[(s, c)] = ("fallback", pooled_used)
            else:
                used[(s, c)] = (f.status, f)
    return CellFits(model, cells, pooled, used, pooled_used)


def conditional_paid(model: str, log_params: FloatArray, age: FloatArray, horizon: int
                     ) -> FloatArray:
    """``P(month age + j paid | month age paid) = S(age + j) / S(age)``, ``j = 1 .. horizon``;
    shape ``(n, horizon)``."""
    a0 = np.asarray(age, dtype=np.float64)
    j = np.arange(1, horizon + 1, dtype=np.float64)
    base = log_survival(model, log_params, a0)[0]
    ahead = log_survival(model, log_params, (a0[:, None] + j[None, :]).ravel()
                         )[0].reshape(-1, horizon)
    return np.asarray(np.exp(ahead - base[:, None]), dtype=np.float64)


@dataclass(frozen=True)
class ContractualPrediction:
    model: str
    cutoff: int
    frame: pd.DataFrame
    """Per customer (index ``customer_id``): ``segment, channel, age, status, prediction,
    prediction_discounted, paid_1 .. paid_H``."""
    fits: CellFits
    contribution: ContributionModel


def predict(model: str, tables: EconomicsTables, cutoff: int, horizon: int,
            policy: PromotionPolicy, *, monthly_rate: float = pp.MONTHLY_DISCOUNT_RATE,
            boundary_exception: bool = True, e1_boundary_exception: bool = True
            ) -> ContractualPrediction:
    """E1 or E2 fitted with the information recorded by ``cutoff``; predictions over months
    ``cutoff+1 .. cutoff+horizon`` for the contractual population (active at ``cutoff`` with
    at least one completely recorded month — the features' and oracles' population)."""
    _check(model)
    obs = observe(tables, cutoff)
    fits = fit_cells(model, obs.status(), boundary_exception=boundary_exception,
                     e1_boundary_exception=e1_boundary_exception)
    contrib = fit_contribution(tables, cutoff, policy)
    pop = contractual_features(tables, cutoff, "A", policy)
    seg, ch = pop["segment"].to_numpy(), pop["channel"].to_numpy()
    a0 = pop["tenure_months"].to_numpy().astype(np.float64)
    j = np.arange(1, horizon + 1, dtype=np.float64)
    paid = np.full((len(pop), horizon), np.nan)
    status = np.empty(len(pop), dtype=object)
    for (s, c), (st, f) in fits.used.items():
        sel = (seg == s) & (ch == c)
        if not sel.any():
            continue
        status[sel] = st
        if f.geometric_churn is not None:  # E1 geometric limit (A7)
            paid[sel] = (1.0 - f.geometric_churn) ** j[None, :]
        elif f.finite:
            paid[sel] = conditional_paid(model, f.log_params, a0[sel], horizon)
    ages = a0[:, None].astype(np.int64) + np.arange(1, horizon + 1)[None, :]
    m = contrib.expected(seg, ch, ages)
    d = (1.0 + monthly_rate) ** -j
    frame = pd.DataFrame({"segment": seg, "channel": ch, "age": a0.astype(np.int64),
                          "status": status, "prediction": (paid * m).sum(axis=1),
                          "prediction_discounted": (paid * m * d).sum(axis=1)},
                         index=pop.index)
    frame = frame.join(pd.DataFrame(paid, index=pop.index,
                                    columns=[f"paid_{k}" for k in range(1, horizon + 1)]))
    return ContractualPrediction(model, cutoff, frame, fits, contrib)


# --- new customers: cohort projection --------------------------------------------------------
@dataclass(frozen=True)
class CohortValue:
    value: float
    status: str
    """``ok`` or ``undefined`` (an undiscounted infinite sum that diverges)."""


def cohort_value(model: str, params: tuple[float, float], contribution: Callable[
        [np.ndarray], FloatArray], *, horizon: int | None, monthly_rate: float,
        sum_months: int = pp.INFINITE_SUM_MONTHS) -> CohortValue:
    """Expected discounted contribution per acquired customer over ages ``0 .. horizon - 1``
    (``None``: the infinite sum, truncated at ``sum_months`` like M10). The undiscounted
    infinite sum diverges when ``S(a)`` decays no faster than ``1 / a``: E1 ``alpha <= 1``,
    E2 ``1 / v <= 1``."""
    _check(model)
    if horizon is None and monthly_rate == 0.0:
        tail = params[0] if model == "E1" else (1.0 / params[1] if params[1] > 0 else math.inf)
        if tail <= 1.0:
            return CohortValue(math.nan, "undefined")
    n = sum_months if horizon is None else horizon
    ages = np.arange(n)
    s = survival(model, params, ages.astype(np.float64))
    d = (1.0 + monthly_rate) ** -ages.astype(np.float64)
    return CohortValue(float(np.sum(s * contribution(ages) * d)), "ok")


# --- temporal validation (selection inputs; never the holdout) ---------------------------------
@dataclass(frozen=True)
class ValidationResult:
    model: str
    cutoff: int
    landmark: int
    frame: pd.DataFrame
    """``prediction, realised`` per customer at the validation landmark."""
    candidate: CandidateResult
    fits: CellFits


def calibration_slope(pred: np.ndarray, realised: np.ndarray) -> float:
    """OLS slope of ``realised`` on ``pred``."""
    x = pred - pred.mean()
    sxx = float(np.sum(x * x))
    return float(np.sum(x * (realised - realised.mean())) / sxx) if sxx > 0 else math.nan


def validate(model: str, tables: EconomicsTables, cutoff: int, horizon: int,
             policy: PromotionPolicy, *, boundary_exception: bool = True,
             e1_boundary_exception: bool = True) -> ValidationResult:
    """Fit with the information of the validation landmark ``L_v`` only (tables as recorded by
    ``cutoff``, then observed at ``L_v``) and score on its labels ``L_v+1 .. L_v+horizon``,
    which are completely recorded by ``cutoff``."""
    lv = pp.validation_landmark(cutoff, horizon)
    if lv is None:
        raise ValueError(f"no validation landmark at cutoff {cutoff} (horizon {horizon})")
    view = view_at(tables, cutoff)
    pred = predict(model, view, lv, horizon, policy, boundary_exception=boundary_exception,
                   e1_boundary_exception=e1_boundary_exception)
    stack = training_stack(tables, cutoff, horizon, "contractual", "A", policy,
                           landmarks=[lv]).set_index("customer_id")
    if not stack.index.equals(pred.frame.index):
        raise RuntimeError("validation population differs from the prediction population")
    frame = pd.DataFrame({"prediction": pred.frame["prediction"],
                          "realised": stack["realised"]})
    p = frame["prediction"].to_numpy(dtype=np.float64)
    r = frame["realised"].to_numpy(dtype=np.float64)
    finite = np.isfinite(p)
    citl = (float(np.sum(p[finite])) / float(np.sum(r[finite])) - 1.0) if finite.any() \
        else math.nan
    cand = CandidateResult(model, pred.fits.ok_cell_fraction, citl,
                           calibration_slope(p[finite], r[finite]) if finite.any() else math.nan,
                           float(np.mean((r[finite] - p[finite]) ** 2)) if finite.all()
                           else math.nan)
    return ValidationResult(model, cutoff, lv, frame, cand, pred.fits)
