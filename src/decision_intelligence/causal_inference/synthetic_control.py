"""Synthetic control for one treated unit (Abadie, Diamond & Hainmueller 2010).

Estimand: the treated unit's effect path ``tau_t = Y_1t(1) - Y_1t(0)`` for each
post-treatment period, and its average over the post period.

Method: choose donor weights ``w >= 0``, ``sum w = 1`` minimising the
pre-treatment fit ``||y_1,pre - Y_0,pre w||^2`` (outcome lags as predictors,
identity predictor weighting); the synthetic control ``Y_0 w`` is the
counterfactual and the gap ``y_1 - Y_0 w`` the estimated effect.

Identification and assumptions
------------------------------
1. A convex combination of donors reproduces the treated unit's untreated
   path — plausible under a factor model when the treated unit's loadings
   lie in the donors' convex hull *and* the pre-period is long; checked
   (never proven) by pre-period fit.
2. Donors are unaffected by the treatment (no spillovers / contamination)
   and experience no idiosyncratic shocks of their own after treatment.
3. No anticipation before the stated treatment start.

Diagnostics: pre-period RMSPE relative to the outcome's scale, the share of
pre-periods where the treated unit lies outside the donors' range in that
period (a convex combination cannot leave it, so this is a *necessary*
condition for lying in the convex hull — not a sufficient one; the
pre-period fit is the operative check), weight concentration, leave-one-out
donor sensitivity, and in-time placebos.

Inference: there is no sampling-based standard error. In-space placebos
re-run the method with each donor as the "treated" unit and rank the
treated unit's post/pre RMSPE ratio. ``rank / (J + 1)`` is an exact
permutation p-value **only** if the treated unit was chosen at random from
the J + 1 units; otherwise it is a descriptive measure of how unusual the
treated unit's gap is. Its smallest attainable value is ``1 / (J + 1)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize

from decision_intelligence.common.validation import FloatArray

_GAP_ABSOLUTE = 1e-9
_GAP_RELATIVE = 1e-8


def simplex_least_squares(predictors: FloatArray, target: FloatArray) -> FloatArray:
    """``argmin ||target - predictors @ w||^2`` subject to ``w >= 0, sum(w) = 1``.

    1. SLSQP from a uniform start;
    2. exact active-set refinement: solve the equality-constrained least
       squares on the current support, drop donors whose weight turns
       negative, add the inactive donor that most violates dual feasibility,
       until the KKT conditions hold (removes SLSQP's early stopping);
    3. **certification** by the Frank–Wolfe duality gap
       ``g'w - min_j g_j`` (``g`` the gradient), which bounds the objective's
       distance from the optimum for this convex problem. Raises if the gap
       exceeds ``1e-9 + 1e-8 f(w)`` on the problem scaled to ``||target|| = 1``.

    When donors are collinear in the pre-period the optimal *weights* are not
    unique; any optimum is returned and only the fitted path is identified.
    """
    a = np.asarray(predictors, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    if a.ndim != 2 or b.shape != (a.shape[0],):
        raise ValueError("predictors must be (periods × donors) and target (periods,)")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("synthetic-control inputs must be finite")
    j = a.shape[1]
    scale = float(np.linalg.norm(b)) or 1.0
    a_s, b_s = a / scale, b / scale

    def objective(w: FloatArray) -> float:
        r = a_s @ w - b_s
        return float(r @ r)

    def gradient(w: FloatArray) -> FloatArray:
        return np.asarray(2 * a_s.T @ (a_s @ w - b_s), dtype=np.float64)

    result = optimize.minimize(
        objective,
        np.full(j, 1.0 / j),
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * j,
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0),
                      "jac": lambda w: np.ones_like(w)}],
        options={"ftol": 1e-14, "maxiter": 1000},
    )
    start = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, None)
    start /= start.sum()
    refined = _active_set_refine(a_s, b_s, start)
    w = refined if objective(refined) <= objective(start) else start
    if optimality_gap(a_s, b_s, w) > _GAP_ABSOLUTE + _GAP_RELATIVE * objective(w):
        raise RuntimeError("synthetic-control weight optimisation did not reach the optimum")
    return w


def _active_set_refine(a: FloatArray, b: FloatArray, w: FloatArray) -> FloatArray:
    """Primal active-set iterations for simplex-constrained least squares."""
    support: list[int] = [int(i) for i in np.flatnonzero(w > 1e-10)] or [int(np.argmax(w))]
    current = w.copy()
    for _ in range(4 * a.shape[1] + 10):
        k = len(support)
        sub = a[:, support]
        system = np.zeros((k + 1, k + 1))
        system[:k, :k] = 2 * sub.T @ sub
        system[:k, k] = 1.0
        system[k, :k] = 1.0
        rhs = np.concatenate([2 * sub.T @ b, [1.0]])
        solution = np.linalg.lstsq(system, rhs, rcond=None)[0][:k]
        if (solution < -1e-12).any():
            # Step from the current feasible point towards the solution until a weight hits 0.
            direction = np.zeros_like(current)
            direction[support] = solution
            direction -= current
            negative = direction < 0
            steps = np.where(negative, -current / np.where(negative, direction, 1.0), np.inf)
            step = float(min(1.0, np.min(steps[support])))
            current = np.clip(current + step * direction, 0.0, None)
            current /= current.sum()
            support = [i for i in support if current[i] > 1e-12] or [int(np.argmax(current))]
            continue
        current = np.zeros_like(current)
        current[support] = solution
        current = np.clip(current, 0.0, None)
        current /= current.sum()
        g = 2 * a.T @ (a @ current - b)
        nu = float(np.mean(g[support]))
        inactive = [i for i in range(a.shape[1]) if i not in support]
        if not inactive:
            break
        worst = min(inactive, key=lambda i: g[i])
        if g[worst] >= nu - 1e-12 * max(1.0, abs(nu)):
            break
        support.append(int(worst))
    return current


def optimality_gap(a: FloatArray, b: FloatArray, w: FloatArray) -> float:
    """Frank–Wolfe duality gap ``g'w - min_j g_j``: an upper bound on ``f(w) - f*``."""
    g = 2 * a.T @ (a @ w - b)
    return float(g @ w - np.min(g))


def is_optimal(a: FloatArray, b: FloatArray, w: FloatArray) -> bool:
    """Feasible and certified optimal (on the problem scaled to ``||b|| = 1``)."""
    scale = float(np.linalg.norm(b)) or 1.0
    a_s, b_s = a / scale, b / scale
    feasible = bool((w >= 0).all() and abs(float(w.sum()) - 1) < 1e-9)
    residual = a_s @ w - b_s
    return feasible and optimality_gap(a_s, b_s, w) <= (
        _GAP_ABSOLUTE + _GAP_RELATIVE * float(residual @ residual)
    )


@dataclass(frozen=True)
class SyntheticControlFit:
    treated_unit: int
    treatment_start: int
    donors: tuple[int, ...]
    weights: dict[int, float]
    periods: np.ndarray = field(repr=False)
    treated_path: FloatArray = field(repr=False)
    synthetic_path: FloatArray = field(repr=False)
    pre_rmspe: float
    post_rmspe: float
    rmspe_ratio: float
    average_post_gap: float
    relative_pre_fit: float
    """Pre-RMSPE divided by the SD of the treated unit's pre-period outcome."""
    share_pre_outside_donor_range: float
    effective_number_of_donors: float
    """``1 / sum(w^2)``: how many donors the counterfactual effectively rests on."""
    warnings: tuple[str, ...] = ()

    @property
    def gaps(self) -> FloatArray:
        return np.asarray(self.treated_path - self.synthetic_path, dtype=np.float64)

    @property
    def post_gaps(self) -> dict[int, float]:
        post = self.periods >= self.treatment_start
        return {int(t): float(g) for t, g in zip(self.periods[post], self.gaps[post], strict=True)}


def _wide(frame: pd.DataFrame) -> pd.DataFrame:
    for column in ("unit_id", "period", "outcome"):
        if column not in frame.columns:
            raise ValueError(f"panel is missing column {column!r}")
    wide = frame.pivot(index="period", columns="unit_id", values="outcome")
    if wide.isna().any().any():
        raise ValueError("synthetic control needs a balanced panel")
    return wide


def fit_synthetic_control(
    frame: pd.DataFrame,
    treated_unit: int,
    treatment_start: int,
    *,
    donors: list[int] | None = None,
    poor_fit_threshold: float = 0.5,
) -> SyntheticControlFit:
    """Fit weights on periods before ``treatment_start`` and build the counterfactual path."""
    wide = _wide(frame)
    unit_ids = [int(u) for u in wide.columns]
    if treated_unit not in unit_ids:
        raise ValueError("treated unit not in panel")
    pool = [u for u in (donors or unit_ids) if u != treated_unit]
    if len(pool) < 2:
        raise ValueError("need at least two donors")
    periods = wide.index.to_numpy(int)
    pre = periods < treatment_start
    if pre.sum() < 2 or (~pre).sum() < 1:
        raise ValueError("need at least two pre-treatment periods and one post-treatment period")
    y1 = wide[treated_unit].to_numpy(float)
    y0 = wide[pool].to_numpy(float)
    w = simplex_least_squares(y0[pre], y1[pre])
    synthetic = y0 @ w
    gaps = y1 - synthetic
    pre_rmspe = float(np.sqrt(np.mean(gaps[pre] ** 2)))
    post_rmspe = float(np.sqrt(np.mean(gaps[~pre] ** 2)))
    pre_sd = float(np.std(y1[pre], ddof=1)) or 1.0
    outside = (y1[pre] > y0[pre].max(axis=1)) | (y1[pre] < y0[pre].min(axis=1))
    warnings: list[str] = []
    relative = pre_rmspe / pre_sd
    if relative > poor_fit_threshold:
        warnings.append(
            f"poor pre-treatment fit (RMSPE = {relative:.2f} × pre-period SD): the synthetic "
            "control does not reproduce the treated unit's path; post-period gaps are not "
            "credible effect estimates"
        )
    if outside.mean() > 0.2:
        warnings.append(
            f"treated outcome lies outside the donor range in {outside.mean():.0%} of pre-periods: "
            "no convex combination can match it (consider a different donor pool or method)"
        )
    if pre.sum() < 10:
        warnings.append("short pre-period: good fit may reflect over-fitting noise")
    return SyntheticControlFit(
        treated_unit=treated_unit,
        treatment_start=treatment_start,
        donors=tuple(int(u) for u in pool),
        weights={int(u): float(v) for u, v in zip(pool, w, strict=True) if v > 1e-6},
        periods=periods,
        treated_path=y1,
        synthetic_path=synthetic,
        pre_rmspe=pre_rmspe,
        post_rmspe=post_rmspe,
        rmspe_ratio=post_rmspe / pre_rmspe if pre_rmspe > 0 else float("inf"),
        average_post_gap=float(np.mean(gaps[~pre])),
        relative_pre_fit=relative,
        share_pre_outside_donor_range=float(outside.mean()),
        effective_number_of_donors=float(1.0 / np.sum(w**2)),
        warnings=tuple(warnings),
    )


@dataclass(frozen=True)
class PlaceboInference:
    """In-space placebo ranking of the post/pre RMSPE ratio.

    ``rank`` is the treated unit's position (1 = most extreme) among the
    treated unit and the retained placebos. ``permutation_p_value`` =
    rank / (units compared) is an exact p-value only under random choice of
    the treated unit; read it as a descriptive rank otherwise.
    """

    treated_ratio: float
    placebo_ratios: dict[int, float]
    rank: int
    n_compared: int
    permutation_p_value: float
    minimum_attainable_p: float
    excluded_for_poor_fit: tuple[int, ...]
    exclusion_multiple: float | None
    """Placebos with pre-period MSPE above this multiple of the treated unit's were dropped."""


def placebo_in_space(
    frame: pd.DataFrame,
    treated_unit: int,
    treatment_start: int,
    *,
    exclusion_multiple: float | None = None,
) -> PlaceboInference:
    """Re-run synthetic control with each donor as the treated unit (true treated unit excluded).

    ``exclusion_multiple``: drop placebos whose pre-period **MSPE** (mean
    squared prediction error) exceeds that multiple of the treated unit's, as
    in Abadie et al. (2010), who used 20, 5 and 2; ``None`` keeps all.
    Results are sensitive to this choice — report it.
    """
    wide = _wide(frame)
    actual = fit_synthetic_control(frame, treated_unit, treatment_start)
    donors = [int(u) for u in wide.columns if int(u) != treated_unit]
    ratios: dict[int, float] = {}
    excluded: list[int] = []
    for placebo in donors:
        pool = [u for u in donors if u != placebo]
        fit = fit_synthetic_control(frame, placebo, treatment_start, donors=pool)
        poor = exclusion_multiple is not None and (
            fit.pre_rmspe**2 > exclusion_multiple * actual.pre_rmspe**2
        )
        if poor:
            excluded.append(int(placebo))
            continue
        ratios[int(placebo)] = fit.rmspe_ratio
    all_ratios = np.array([actual.rmspe_ratio, *ratios.values()])
    rank = int(np.sum(all_ratios >= actual.rmspe_ratio))
    return PlaceboInference(
        treated_ratio=actual.rmspe_ratio,
        placebo_ratios=ratios,
        rank=rank,
        n_compared=int(all_ratios.size),
        permutation_p_value=rank / all_ratios.size,
        minimum_attainable_p=1 / all_ratios.size,
        excluded_for_poor_fit=tuple(excluded),
        exclusion_multiple=exclusion_multiple,
    )


def placebo_in_time(
    frame: pd.DataFrame, treated_unit: int, treatment_start: int, fake_start: int
) -> SyntheticControlFit:
    """Pretend treatment began at ``fake_start`` (< real start), using only pre-treatment data.

    The 'effect' before the real start should be close to zero; a large gap
    suggests the method manufactures effects (or anticipation).
    """
    if fake_start >= treatment_start:
        raise ValueError("fake_start must precede the real treatment start")
    truncated = frame[frame["period"] < treatment_start]
    return fit_synthetic_control(truncated, treated_unit, fake_start)


def leave_one_out(
    frame: pd.DataFrame, treated_unit: int, treatment_start: int
) -> dict[int, float]:
    """Average post-period gap when each positively weighted donor is removed in turn."""
    base = fit_synthetic_control(frame, treated_unit, treatment_start)
    estimates: dict[int, float] = {}
    for donor in base.weights:
        pool = [u for u in base.donors if u != donor]
        estimates[donor] = fit_synthetic_control(
            frame, treated_unit, treatment_start, donors=pool
        ).average_post_gap
    return estimates
