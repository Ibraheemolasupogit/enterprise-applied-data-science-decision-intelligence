"""SIMULATED user-randomised A/B evaluation of recommendation policies (Milestone 9).

Every result here is **SIMULATED**: outcomes come from the M8 generator's click and purchase
model (:mod:`decision_intelligence.data.recommendation_experiments`), not from users.

The design is pre-specified in :data:`~decision_intelligence.recommendations.protocol.AB_DESIGN`
and analysed only with existing M6/M7 components:

* assignment — M6 :func:`~decision_intelligence.experimentation.assignment.hash_assignment`
  (equal shares over the arms, by user id);
* sample-ratio check — M6 :func:`~decision_intelligence.experimentation.srm.srm_test`;
* effects — M6 :func:`~decision_intelligence.experimentation.analysis.analyse_continuous`
  (Welch difference in means per user, intention to treat) for the primary metric (clicks per
  user), secondary metrics (purchases, margin) and guardrails;
* precision — M7 :func:`~decision_intelligence.experimentation.variance_reduction.cuped` with
  the pre-period clicks (secondary analysis; the decision uses the unadjusted primary);
* multiplicity — two primary comparisons, Bonferroni: each at ``alpha / 2`` with
  ``1 - alpha / 2`` intervals (tests and intervals dual); Holm-adjusted p-values also reported
  (M6 :func:`~decision_intelligence.statistics.multiple_testing.adjust_p_values`);
* guardrails and decision — M6 ``assess_guardrail`` (non-inferiority) and ``recommend``
  (LAUNCH / ITERATE / HOLD / STOP).

Policies are fixed before the simulated outcomes are drawn; the A/B outcome is never used to
choose a policy.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from decision_intelligence.data.recommendation_experiments import (
    OnlinePeriod,
    OnlineSessions,
    draw_sessions,
    simulate_online_period,
)
from decision_intelligence.data.recommendation_scenarios import RecommendationScenario
from decision_intelligence.experimentation.analysis import MetricResult, analyse_continuous
from decision_intelligence.experimentation.assignment import hash_assignment
from decision_intelligence.experimentation.decision import assess_guardrail, recommend
from decision_intelligence.experimentation.design import MetricSpec
from decision_intelligence.experimentation.srm import srm_test
from decision_intelligence.experimentation.variance_reduction import cuped
from decision_intelligence.recommendations.baselines import Recommender
from decision_intelligence.recommendations.candidates import allowed_mask
from decision_intelligence.recommendations.evaluation import top_k
from decision_intelligence.recommendations.interactions import (
    IntArray,
    InteractionData,
    deduplicate_impressions,
    index_of,
)
from decision_intelligence.recommendations.protocol import ABDesign
from decision_intelligence.statistics.multiple_testing import adjust_p_values

METRICS = {"clicks per user": "clicks", "purchases per user": "purchases",
           "margin per user": "margin"}


def experiment_users(scenario: RecommendationScenario, start: pd.Timestamp) -> IntArray:
    """Users signed up by the start (all of them can be randomised)."""
    users = scenario.logs.users.sort_values("user_id")
    return np.flatnonzero((users["signup_time"] <= start).to_numpy()).astype(np.int64)


def policy_lists(model: Recommender, history: InteractionData, users: IntArray, k: int
                 ) -> IntArray:
    """Each user's fixed top-``k`` list at the history cutoff (M8 eligibility and exclusion)."""
    return top_k(model.score(users), allowed_mask(history, users), k)


def pre_period_clicks(scenario: RecommendationScenario, history: InteractionData,
                      users: IntArray, days: float) -> np.ndarray:
    """Logged clicks per user in ``(start - days, start]`` (pre-treatment covariate)."""
    imp = deduplicate_impressions(scenario.logs.impressions)
    start = history.cutoff
    win = imp.loc[imp["clicked"] & (imp["timestamp"] > start - pd.to_timedelta(days, unit="D"))
                  & (imp["timestamp"] <= start)]
    idx = index_of(history.user_ids, win["user_id"].to_numpy())
    counts = np.bincount(idx, minlength=history.n_users).astype(np.float64)
    return np.asarray(counts[users])


@dataclass(frozen=True)
class Comparison:
    treatment: str
    results: dict[str, MetricResult]
    cuped_effect: tuple[float, float, float]
    """(estimate, lower, upper) of the CUPED-adjusted primary effect, 1 - alpha/2 interval."""
    cuped_variance_reduction: float
    p_value: float
    bonferroni_p: float
    holm_p: float
    guardrails: tuple[str, ...]
    decision: str
    reasons: tuple[str, ...]
    sate: dict[str, float]
    """Exact simulated sample average treatment effect per metric (SIMULATED truth)."""
    covered: dict[str, bool]


@dataclass(frozen=True)
class ABResult:
    period: OnlinePeriod
    arm_counts: dict[str, int]
    srm_p_value: float
    srm_passed: bool
    comparisons: list[Comparison]


def run_ab(
    scenario: RecommendationScenario, history: InteractionData,
    lists: Mapping[str, IntArray], users: IntArray, design: ABDesign, *,
    salt: str = "", click_seed: int | None = None, sessions: OnlineSessions | None = None,
) -> ABResult:
    """Assign, simulate and analyse one SIMULATED A/B test (control = ``design.arms[0]``)."""
    control, treatments = design.arms[0], design.arms[1:]
    if set(lists) != set(design.arms):
        raise ValueError("lists are needed for every arm")
    ids = history.user_ids[users]
    arm = hash_assignment(ids, experiment=design.experiment, arms=design.arms, salt=salt)
    if sessions is None:
        sessions = draw_sessions(scenario, users, design.start_day, design.duration_days,
                                 design.simulation_seed)
    period = simulate_online_period(scenario, users, arm, lists, sessions,
                                    design.simulation_seed + 1 if click_seed is None
                                    else click_seed)
    counts = {a: int(np.sum(arm == a)) for a in design.arms}
    srm = srm_test(counts, {a: 1.0 / len(design.arms) for a in design.arms},
                   alpha=design.srm_alpha)
    level = 1 - design.alpha / len(treatments)
    pre = pre_period_clicks(scenario, history, users, design.pre_period_days)
    comps: list[Comparison] = []
    raw_p = []
    for t in treatments:
        sel = (arm == control) | (arm == t)
        res = {}
        for metric, col in METRICS.items():
            y = getattr(period, col)[sel]
            res[metric] = analyse_continuous(y, arm[sel], metric=metric, control=control,
                                             treatment=t, confidence_level=level)
        adj = cuped(period.clicks[sel], arm[sel], {"pre_clicks": pre[sel]}, control=control,
                    treatment=t, confidence_level=level, metric=design.primary_metric)
        guards = [assess_guardrail(res[name], MetricSpec(name, "continuous", name,
                                                         "increase", margin))
                  for name, margin in design.guardrails]
        rec = recommend(res[design.primary_metric],
                        minimum_worthwhile_effect=design.minimum_worthwhile_effect,
                        guardrails=guards,
                        validity_failures=() if srm.p_value > design.srm_alpha
                        else ("sample ratio mismatch",))
        sate = {m: period.sate(t, control, col) for m, col in METRICS.items()}
        covered = {m: bool(res[m].effect.ci_lower <= sate[m] <= res[m].effect.ci_upper)
                   for m in METRICS}
        raw_p.append(res[design.primary_metric].p_value)
        comps.append(Comparison(
            t, res, (adj.effect.estimate, adj.effect.ci_lower, adj.effect.ci_upper),
            adj.realised_variance_reduction, res[design.primary_metric].p_value,
            min(1.0, res[design.primary_metric].p_value * len(treatments)), float("nan"),
            tuple(f"{g.metric}: {g.status}" for g in guards), rec.decision, rec.reasons,
            sate, covered))
    holm = adjust_p_values(np.array(raw_p), method="holm", alpha=design.alpha).adjusted_p_values
    comps = [Comparison(c.treatment, c.results, c.cuped_effect, c.cuped_variance_reduction,
                        c.p_value, c.bonferroni_p, float(h), c.guardrails, c.decision,
                        c.reasons, c.sate, c.covered) for c, h in zip(comps, holm, strict=True)]
    return ABResult(period, counts, srm.p_value, bool(srm.p_value > design.srm_alpha), comps)


@dataclass(frozen=True)
class RepeatedAB:
    rows: pd.DataFrame
    n_replications: int


def aa_study(scenario: RecommendationScenario, history: InteractionData, control_list: IntArray,
             users: IntArray, design: ABDesign, n_replications: int) -> RepeatedAB:
    """Both arms show the control lists; new assignment salt and clicks per replication.
    Records the two-arm primary p-value at the unadjusted level ``alpha``."""
    two = ABDesign(experiment=design.experiment, arms=("control", "copy"),
                   start_day=design.start_day, duration_days=design.duration_days,
                   simulation_seed=design.simulation_seed)
    sessions = draw_sessions(scenario, users, two.start_day, two.duration_days,
                             two.simulation_seed)
    rows = []
    for r in range(n_replications):
        ids = history.user_ids[users]
        arm = hash_assignment(ids, experiment=two.experiment, arms=two.arms, salt=f"aa{r}")
        period = simulate_online_period(scenario, users, arm,
                                        {"control": control_list, "copy": control_list},
                                        sessions, 70_000 + r)
        res = analyse_continuous(period.clicks, arm, metric="clicks per user",
                                 control="control", treatment="copy")
        rows.append({"replication": r, "p_value": res.p_value,
                     "estimate": res.effect.estimate})
    return RepeatedAB(pd.DataFrame(rows), n_replications)


def coverage_study(scenario: RecommendationScenario, history: InteractionData,
                   lists: Mapping[str, IntArray], users: IntArray, design: ABDesign,
                   treatment: str, n_replications: int) -> RepeatedAB:
    """Re-randomise and re-simulate clicks with sessions fixed (so the SATE is fixed); record
    whether the 95% and the Bonferroni (1 - alpha/2) intervals cover the exact SATE."""
    sessions = draw_sessions(scenario, users, design.start_day, design.duration_days,
                             design.simulation_seed)
    control = design.arms[0]
    rows = []
    for r in range(n_replications):
        ids = history.user_ids[users]
        arm = hash_assignment(ids, experiment=design.experiment, arms=design.arms,
                              salt=f"cov{r}")
        period = simulate_online_period(scenario, users, arm, lists, sessions, 80_000 + r)
        sel = (arm == control) | (arm == treatment)
        sate = period.sate(treatment, control, "clicks")
        row: dict[str, float] = {"replication": float(r), "sate": sate}
        for label, level in (("95", 0.95), ("bonferroni", 1 - design.alpha / 2)):
            res = analyse_continuous(period.clicks[sel], arm[sel], control=control,
                                     treatment=treatment, confidence_level=level)
            row[f"covered_{label}"] = float(res.effect.ci_lower <= sate <= res.effect.ci_upper)
            row["estimate"] = res.effect.estimate
        rows.append(row)
    return RepeatedAB(pd.DataFrame(rows), n_replications)
