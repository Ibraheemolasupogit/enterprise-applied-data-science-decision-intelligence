"""Experiment validity checks: is the logged experiment the one that was designed?

Each check returns a :class:`CheckResult` with a status:

* ``FAIL`` — the comparison cannot be interpreted until the problem is
  understood (SRM, conflicting assignments, post-treatment exclusions or
  selection, invalid metric values);
* ``WARN`` — interpretable, but the issue must be reported and its likely
  impact assessed (cross-over exposure, missing outcomes, chance imbalance);
* ``PASS`` — no problem detected. Passing a check is not proof of validity:
  every check has limited power, and some violations (interference,
  undetectable selection) are invisible in the data.

Pre-treatment balance checks under randomisation test the *assignment
mechanism*, not whether "the groups are comparable": chance imbalance is
expected (1 in 20 covariates at p < 0.05), and the remedy is never to
re-randomise, drop units or choose covariates after seeing outcomes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from decision_intelligence.causal_inference.diagnostics import balance_table
from decision_intelligence.causal_inference.estimators import CausalData
from decision_intelligence.experimentation.srm import srm_from_labels
from decision_intelligence.statistics.hypothesis_tests import (
    two_proportion_z_test,
    two_sample_t_test,
)
from decision_intelligence.statistics.linear_regression import fit_ols
from decision_intelligence.statistics.multiple_testing import rejects

Status = Literal["PASS", "WARN", "FAIL"]
_ORDER = {"PASS": 0, "WARN": 1, "FAIL": 2}


@dataclass(frozen=True)
class CheckResult:
    check: str
    status: Status
    detail: str
    value: float | None = None


def overall_status(checks: Sequence[CheckResult]) -> Status:
    """The worst status among the checks."""
    worst: Status = "PASS"
    for check in checks:
        if _ORDER[check.status] > _ORDER[worst]:
            worst = check.status
    return worst


def check_duplicates(log: pd.DataFrame, *, unit: str = "unit_id", arm: str = "arm") -> CheckResult:
    """Units assigned more than once; conflicting arms mean the unit saw both experiences."""
    counts = log.groupby(unit)[arm].agg(["size", "nunique"])
    duplicated = int((counts["size"] > 1).sum())
    conflicting = int((counts["nunique"] > 1).sum())
    if conflicting:
        return CheckResult(
            "duplicate units", "FAIL",
            f"{duplicated} units logged more than once, {conflicting} with conflicting arms "
            f"({conflicting / len(counts):.2%} of units): assignment is not unique — keep each "
            "unit's first assignment and report the contamination",
            conflicting / len(counts),
        )
    if duplicated:
        return CheckResult("duplicate units", "WARN",
                           f"{duplicated} units logged more than once with the same arm: "
                           "deduplicate before analysis", duplicated / len(counts))
    return CheckResult("duplicate units", "PASS", "every unit assigned exactly once", 0.0)


def first_assignment(
    log: pd.DataFrame, *, unit: str = "unit_id", time: str = "assigned_at"
) -> pd.DataFrame:
    """One row per unit: the earliest assignment event (the arm the unit was randomised to)."""
    ordered = log.sort_values([unit, time], kind="stable")
    return ordered.drop_duplicates(subset=[unit], keep="first").reset_index(drop=True)


def check_srm(
    labels: ArrayLike, expected_shares: Mapping[str, float], *, alpha: float = 0.001
) -> CheckResult:
    result = srm_from_labels(np.asarray(labels), expected_shares, alpha=alpha)
    shares = ", ".join(
        f"{a} {s:.4f}" for a, s in zip(result.arms, result.observed_shares, strict=True)
    )
    return CheckResult(
        "sample ratio mismatch", "FAIL" if result.detected else "PASS",
        f"observed shares {shares}; chi-square p = {result.p_value:.3g}. "
        + result.interpretation(),
        result.p_value,
    )


def check_assignment_leakage(
    assigned: ArrayLike, exposed: ArrayLike, *, tolerance: float = 0.01
) -> CheckResult:
    """Exposure inconsistent with assignment: control units exposed, treatment units unexposed."""
    z = np.asarray(assigned, dtype=float)
    d = np.asarray(exposed, dtype=float)
    leak = float(d[z == 0].mean())
    missed = float(d[z == 1].mean())
    detail = (
        f"control units exposed to treatment: {leak:.2%}; treatment units not exposed: "
        f"{1 - missed:.2%}. The ITT (by assignment) stays unbiased for the effect of "
        "assignment but is diluted; report the complier effect alongside it"
    )
    status: Status = "WARN" if leak > tolerance or 1 - missed > tolerance else "PASS"
    return CheckResult("assignment leakage / noncompliance", status, detail, leak)


def _numeric_covariates(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(frame, drop_first=True, dtype=float)


def check_pre_treatment_balance(
    covariates: pd.DataFrame, assigned: ArrayLike, *, smd_threshold: float = 0.1,
    joint_alpha: float = 0.001,
) -> tuple[CheckResult, pd.DataFrame]:
    """Standardised mean differences (M4 ``balance_table``) and a joint F test.

    The joint test regresses the assignment indicator on all covariates
    (M3 OLS, classical F — valid under the randomisation null because the
    indicator is then independent of the covariates and homoskedastic).
    """
    z = np.asarray(assigned, dtype=float)
    numeric = _numeric_covariates(covariates.reset_index(drop=True))
    data = CausalData.from_arrays(z, np.zeros(z.size), numeric)
    table = balance_table(data)
    fit = fit_ols(numeric.to_numpy(float), z, feature_names=list(numeric.columns))
    p = float(fit.f_p_value) if fit.f_p_value is not None else 1.0
    worst = float(table["smd_raw"].abs().max())
    if rejects(p, joint_alpha):
        status: Status = "FAIL"
        note = "joint test rejects: check the assignment mechanism"
    elif worst > smd_threshold:
        status = "WARN"
        note = "chance imbalance above the SMD threshold: report it; do not re-randomise"
    else:
        status = "PASS"
        note = "no imbalance beyond chance"
    detail = f"max |SMD| = {worst:.3f}; joint F-test p = {p:.3g}. {note}"
    return CheckResult("pre-treatment balance", status, detail, p), table


def check_exclusions(
    excluded: ArrayLike,
    assigned: ArrayLike,
    *,
    rule_timing: Mapping[str, Literal["pre", "post"]],
    alpha: float = 0.01,
) -> CheckResult:
    """Exclusion rules must use pre-treatment information only, and exclude equally by arm."""
    post = [name for name, timing in rule_timing.items() if timing == "post"]
    x = np.asarray(excluded, dtype=bool)
    z = np.asarray(assigned, dtype=float)
    k_t, n_t = int(x[z == 1].sum()), int((z == 1).sum())
    k_c, n_c = int(x[z == 0].sum()), int((z == 0).sum())
    test = two_proportion_z_test(k_t, n_t, k_c, n_c)
    rates = f"excluded: treatment {k_t / n_t:.2%}, control {k_c / n_c:.2%} (p = {test.p_value:.3g})"
    if post:
        return CheckResult(
            "sample exclusions", "FAIL",
            f"rules {post} use post-randomisation information: excluding on them conditions on "
            f"an outcome of treatment. {rates}", test.p_value,
        )
    if rejects(test.p_value, alpha):
        return CheckResult("sample exclusions", "FAIL",
                           f"pre-treatment rules exclude differently by arm — check how they "
                           f"were applied. {rates}", test.p_value)
    return CheckResult("sample exclusions", "PASS", f"pre-treatment rules only. {rates}",
                       test.p_value)


def check_metric_integrity(
    values: ArrayLike,
    *,
    metric: str,
    kind: Literal["binary", "continuous"],
    lower: float | None = None,
    upper: float | None = None,
    max_missing: float = 0.0,
) -> CheckResult:
    """Values in their domain, finite, and missing no more than allowed."""
    v = np.asarray(values, dtype=np.float64)
    missing = float(np.isnan(v).mean())
    present = v[~np.isnan(v)]
    problems: list[str] = []
    if np.isinf(present).any():
        problems.append("infinite values")
    if kind == "binary" and not np.isin(present, (0.0, 1.0)).all():
        problems.append("binary metric has values other than 0/1")
    if lower is not None and (present < lower).any():
        problems.append(f"{int((present < lower).sum())} values below {lower:g}")
    if upper is not None and (present > upper).any():
        problems.append(f"{int((present > upper).sum())} values above {upper:g}")
    if problems:
        return CheckResult(f"metric integrity: {metric}", "FAIL", "; ".join(problems), missing)
    if missing > max_missing:
        return CheckResult(f"metric integrity: {metric}", "WARN",
                           f"{missing:.2%} missing: analyse with attrition bounds, not by dropping",
                           missing)
    return CheckResult(f"metric integrity: {metric}", "PASS",
                       f"{present.size} valid values; {missing:.2%} missing", missing)


def check_post_randomisation_selection(
    variable: str,
    values: ArrayLike,
    assigned: ArrayLike,
    *,
    measured_after_assignment: bool,
) -> CheckResult:
    """Using a post-randomisation variable to filter or define the analysis population.

    Fails whenever the variable was measured after assignment — a test cannot
    prove the variable is unaffected by treatment (absence of evidence) — and
    reports the arm difference as evidence of how strongly selection differs.
    """
    v = np.asarray(values, dtype=np.float64)
    z = np.asarray(assigned, dtype=float)
    if np.isin(v, (0.0, 1.0)).all():
        test = two_proportion_z_test(int(v[z == 1].sum()), int((z == 1).sum()),
                                     int(v[z == 0].sum()), int((z == 0).sum()))
    else:
        test = two_sample_t_test(v[z == 1], v[z == 0])
    evidence = (f"treatment - control difference in {variable!r}: {test.estimate:+.4f} "
                f"(p = {test.p_value:.3g})")
    if measured_after_assignment:
        return CheckResult(
            f"post-randomisation selection: {variable}", "FAIL",
            f"{variable!r} is measured after assignment; filtering on it compares different "
            f"subpopulations in each arm. {evidence}", test.p_value,
        )
    return CheckResult(f"post-randomisation selection: {variable}", "PASS",
                       f"{variable!r} is pre-treatment. {evidence}", test.p_value)


__all__ = [
    "CheckResult",
    "Status",
    "check_assignment_leakage",
    "check_duplicates",
    "check_exclusions",
    "check_metric_integrity",
    "check_post_randomisation_selection",
    "check_pre_treatment_balance",
    "check_srm",
    "first_assignment",
    "overall_status",
]
