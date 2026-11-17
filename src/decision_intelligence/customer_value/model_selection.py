"""Model-selection and fit-status rules of the Milestone 11 protocol (P0).

Pure decision logic over results that later phases compute; no model is fitted here. Thresholds
come from :mod:`.predictive_protocol` (``SELECTION``, ``STATUS_RULES``).

**Selection** (assessed on validation landmarks, never the holdout):

1. A candidate is *qualified* if its status is ``ok`` or ``fallback`` in at least
   ``min_ok_cell_fraction`` of cells, its calibration-in-the-large
   ``|sum(pred) / sum(realised) - 1|`` is within tolerance, and its calibration slope lies in
   ``slope_bounds``.
2. The selection is the *simplest qualified model that no qualified model beats
   significantly*: model ``m`` beats ``s`` if the paired customer-bootstrap interval of
   ``MSE_m - MSE_s`` lies entirely below 0. Ties in complexity go to the lower MSE.
3. If no candidate qualifies, the result is :data:`~.predictive_protocol.NO_QUALIFIED_MODEL`;
   the reference fallback (E2) is reported with
   :data:`~.predictive_protocol.REFERENCE_FALLBACK_LABEL` and is **never** the selection.

**Fit status**: ``failed`` (no convergence, non-finite likelihood, separation without a
penalty), ``unstable`` (Hessian not positive definite, a parameter at the log-scale bound,
disagreement between starting points, too few events, too few events per parameter),
``undefined`` (no data, or a divergent infinite-horizon sum), otherwise ``ok``. A fit is never
silently replaced: the only allowed substitute is the pre-declared pooled ``fallback``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from decision_intelligence.customer_value import predictive_protocol as pp


@dataclass(frozen=True)
class CandidateResult:
    model_id: str
    ok_cell_fraction: float
    calibration_in_the_large: float
    """``sum(pred) / sum(realised) - 1``."""
    calibration_slope: float
    mse: float


@dataclass(frozen=True)
class GateResult:
    model_id: str
    status_gate: bool
    calibration_gate: bool
    slope_gate: bool

    @property
    def qualified(self) -> bool:
        return self.status_gate and self.calibration_gate and self.slope_gate


@dataclass(frozen=True)
class Selection:
    status: str
    """``SELECTED`` or ``NO QUALIFIED MODEL``."""
    selected: str | None
    gates: tuple[GateResult, ...]
    reference_fallback: str | None
    reference_label: str | None
    reason: str


def gate(candidate: CandidateResult) -> GateResult:
    rule = pp.SELECTION
    lo, hi = rule.slope_bounds

    def finite(x: float) -> bool:
        return math.isfinite(x)

    return GateResult(
        candidate.model_id,
        status_gate=candidate.ok_cell_fraction >= rule.min_ok_cell_fraction,
        calibration_gate=finite(candidate.calibration_in_the_large)
        and abs(candidate.calibration_in_the_large) <= rule.calibration_in_the_large_tolerance,
        slope_gate=finite(candidate.calibration_slope) and lo <= candidate.calibration_slope <= hi)


def select_model(candidates: Sequence[CandidateResult],
                 paired_intervals: Mapping[tuple[str, str], tuple[float, float]]) -> Selection:
    """Apply the frozen selection rule.

    ``paired_intervals[(a, b)]`` is the paired customer-bootstrap interval of ``MSE_a - MSE_b``
    (at ``SELECTION.paired_interval_level``) for qualified candidates ``a != b``; a missing
    pair that the rule needs raises ``KeyError``.
    """
    ids = [c.model_id for c in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate candidates")
    ranks: dict[str, int] = {}
    for c in candidates:
        rank = pp.model(c.model_id).complexity_rank
        if rank is None:
            raise ValueError(f"{c.model_id} is not a selection candidate")
        ranks[c.model_id] = rank
    gates = tuple(gate(c) for c in candidates)
    qualified = [c for c, g in zip(candidates, gates, strict=True) if g.qualified]
    if not qualified:
        return Selection(pp.NO_QUALIFIED_MODEL, None, gates, pp.REFERENCE_FALLBACK_MODEL,
                         pp.REFERENCE_FALLBACK_LABEL, "no candidate passed the gates")

    def beaten(s: CandidateResult) -> bool:
        for m in qualified:
            if m.model_id == s.model_id:
                continue
            _, upper = paired_intervals[(m.model_id, s.model_id)]
            if upper < 0:
                return True
        return False

    order = sorted(qualified, key=lambda c: (ranks[c.model_id], c.mse))
    for s in order:
        if not beaten(s):
            return Selection("SELECTED", s.model_id, gates, None, None,
                             "simplest qualified model not significantly beaten")
    # Every qualified model is significantly beaten by another (possible only with
    # inconsistent intervals); fall back to the lowest MSE and say so.
    best = min(qualified, key=lambda c: c.mse)
    return Selection("SELECTED", best.model_id, gates, None, None,
                     "every qualified model beaten (inconsistent intervals): lowest MSE")


@dataclass(frozen=True)
class FitDiagnostics:
    has_data: bool = True
    converged: bool = True
    finite_likelihood: bool = True
    separated_without_penalty: bool = False
    hessian_positive_definite: bool = True
    max_abs_log_parameter: float = 0.0
    start_log_likelihood_gap: float = 0.0
    n_events: int = 10**9
    n_parameters: int = 1
    divergent_infinite_sum: bool = False


def classify_fit(d: FitDiagnostics) -> tuple[str, tuple[str, ...]]:
    """The fit status and every reason that applies (the first status in severity order wins:
    undefined, failed, unstable, ok)."""
    rules = pp.STATUS_RULES
    if not d.has_data or d.divergent_infinite_sum:
        return "undefined", tuple(r for r, hit in (
            ("no data", not d.has_data),
            ("divergent infinite-horizon sum", d.divergent_infinite_sum)) if hit)
    failed = tuple(r for r, hit in (
        ("not converged", not d.converged),
        ("non-finite likelihood", not d.finite_likelihood),
        ("separation without a penalty", d.separated_without_penalty)) if hit)
    if failed:
        return "failed", failed
    unstable = tuple(r for r, hit in (
        ("Hessian not positive definite", not d.hessian_positive_definite),
        ("parameter at the log-scale bound",
         d.max_abs_log_parameter > rules.log_parameter_bound),
        ("starting points disagree",
         d.start_log_likelihood_gap > rules.start_agreement_tolerance),
        ("too few events", d.n_events < rules.min_events),
        ("too few events per parameter",
         d.n_events < rules.min_events_per_parameter * d.n_parameters)) if hit)
    if unstable:
        return "unstable", unstable
    return "ok", ()
