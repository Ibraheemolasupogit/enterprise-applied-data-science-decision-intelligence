"""M11 P0: frozen-protocol consistency, integrity and reserved-seed safeguards."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from decision_intelligence.customer_value import backtest
from decision_intelligence.customer_value import predictive_protocol as pp
from decision_intelligence.customer_value import protocol as m10
from decision_intelligence.customer_value.panel import Observable

REPO = Path(__file__).resolve().parents[3]
PROTOCOL_FILE = "src/decision_intelligence/customer_value/predictive_protocol.py"
SEED_NAMES = ("MAIN_SEED", "PILOT_SEED", "FRESH_SEED")

FROZEN_DIGEST = "49f3cd6f9e4895daecb9e80727696661a3a0af843feddf1d380041af2f547ce1"
"""SHA-256 of the protocol manifest (P0 as amended through A19). Any protocol change must update
this value in the same reviewed diff and be disclosed in the M11 evidence."""
CANDIDATE_CODE_DIGEST = "6120b81ebb2d9f0d19967266bb4b60905cc172b78f9d49b971e77a87a1f72e2d"
"""SHA-256 of the exact bytes of ``PROTOCOL_CODE_FILES`` (decision logic, not only values)."""


def test_protocol_is_internally_consistent() -> None:
    assert pp.validate_protocol() == []


def test_protocol_digest_is_pinned() -> None:
    assert pp.protocol_digest() == FROZEN_DIGEST


def test_protocol_code_digest_is_pinned() -> None:
    assert pp.protocol_code_digest() == CANDIDATE_CODE_DIGEST


def test_manifest_is_canonical_and_change_sensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = pp.protocol_manifest()
    json.dumps(manifest, sort_keys=True)  # serialisable
    assert pp.protocol_digest() == pp.protocol_digest()
    assert {"PRIMARY_HORIZON", "SELECTION", "ORACLE_GATE", "CLUSTERED_INFERENCE",
            "PENDING_BEFORE_MAIN", "FEATURES", "MODELS"} <= set(manifest)
    before = pp.protocol_digest()
    monkeypatch.setattr(pp, "TARGET_RELATIVE_MDE", 0.03)
    assert pp.protocol_digest() != before


def test_economics_and_stability_rules_are_m10s() -> None:
    assert pp.ANNUAL_DISCOUNT_RATE == m10.ANNUAL_DISCOUNT_RATE
    assert pp.MONTHLY_DISCOUNT_RATE == m10.MONTHLY_DISCOUNT_RATE
    assert pp.RECORDING_LAG_DAYS == m10.MAX_RECORDING_LAG_DAYS
    assert pp.COHORT_HORIZONS == m10.HORIZONS
    assert pp.STATUS_RULES.min_events == m10.MIN_CHURN_EVENTS
    assert pp.M10_CUSTOMER_METHODS == backtest.CUSTOMER_METHODS
    assert pp.scenario("m10_contractual").variants == tuple(
        f"frailty_variance={v}" for v in m10.HETEROGENEITY_GRID)


def test_complete_month_offset_matches_m10_on_the_real_calendar() -> None:
    empty = pd.DataFrame()
    for sc in pp.SCENARIOS:
        start = pd.Timestamp(sc.start)
        for c in range(2, sc.n_months):
            cutoff = start + pd.DateOffset(months=c + 1) - pd.to_timedelta(1, unit="s")
            obs = Observable(c, cutoff, start, empty, empty, empty, empty, empty)
            assert obs.complete_through() == c - pp.COMPLETE_MONTH_OFFSET


def test_landmarks_respect_label_completeness_and_tuning_separation() -> None:
    for sc in pp.SCENARIOS:
        pairs = [(c, pp.PRIMARY_HORIZON) for c in sc.evaluation_cutoffs]
        pairs += [(c, h) for h, c in sc.secondary_horizon]
        for c, h in pairs:
            assert c + h <= sc.n_months - 1
            train = pp.training_landmarks(c, h)
            lv = pp.validation_landmark(c, h)
            assert lv == max(train)
            assert all(L + h <= c - pp.COMPLETE_MONTH_OFFSET for L in train)
            assert lv + 1 + h > c - pp.COMPLETE_MONTH_OFFSET  # the latest admissible landmark
            for L in pp.tuning_landmarks(c, h):
                assert L + h <= lv - pp.COMPLETE_MONTH_OFFSET


def test_temporal_tuning_feasibility_is_recorded() -> None:
    """P0 finding (owner decision pending): with 48-month scenarios and the amendment-2
    separation, temporal penalty tuning is feasible only for the signals scenario at W=12."""
    feasible = {(sc.name, c, h): pp.temporal_tuning_feasible(c, h)
                for sc in pp.SCENARIOS
                for c, h in [(c, pp.PRIMARY_HORIZON) for c in sc.evaluation_cutoffs]
                + [(c, h) for h, c in sc.secondary_horizon]}
    assert feasible == {
        ("m10_contractual", 24, 12): False, ("m10_contractual", 30, 12): False,
        ("signals_contractual", 36, 12): True, ("signals_contractual", 42, 12): True,
        ("signals_contractual", 33, 24): False,
        ("purchases_noncontractual", 24, 12): False,
        ("purchases_noncontractual", 30, 12): False}
    assert pp.tuning_landmarks(36, 12) == (3, 4, 5, 6, 7, 8)


def test_information_tiers_and_feature_rules() -> None:
    a = {f.name for f in pp.features("contractual", "A")}
    b = {f.name for f in pp.features("contractual", "B")}
    assert a < b and all(f.tier == "A" for f in pp.features("contractual", "A"))
    assert {f.source for f in pp.FEATURES}.isdisjoint(pp.FORBIDDEN_SOURCES)
    for f in pp.FEATURES:
        if f.source == "signals":
            assert f.usable_through_offset >= pp.SIGNAL_COMPLETE_OFFSET >= 1
    assert pp.HAZARD_COVARIATE_LAG_MONTHS >= 1
    assert pp.scenario("purchases_noncontractual").tiers == ("A",)


def test_model_roles_and_selection_labels() -> None:
    candidates = {m.id for m in pp.MODELS if m.complexity_rank is not None}
    assert candidates == {"B1", "B2", "E1", "E2", "E3", "E4", "E5"}
    assert pp.model("S1").complexity_rank is None and pp.model("ORACLE").role == "reference"
    assert pp.NO_QUALIFIED_MODEL == "NO QUALIFIED MODEL"
    assert pp.REFERENCE_FALLBACK_MODEL == "E2"
    assert "not a selection" in pp.REFERENCE_FALLBACK_LABEL
    assert len(pp.PRIMARY_COMPARISONS) == 5
    assert pp.PLANNING_ALPHA == pytest.approx(0.01)


def _assert_seed_gate_holds_while_pending() -> None:
    """The gate's invariant at every stage: while any item is pending, the main and fresh seeds
    are refused (checked with a local pending item, independent of the live protocol)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pp, "PENDING_BEFORE_MAIN", ("pending item",))
        with pytest.raises(pp.ProtocolNotReady):
            pp.authorise_seed("main")
        with pytest.raises(pp.ProtocolNotReady):
            pp.authorise_seed("fresh", confirm_single_final_run=True)


def test_seed_gate_refuses_main_and_fresh_while_items_are_pending() -> None:
    _assert_seed_gate_holds_while_pending()
    assert pp.authorise_seed("pilot") == pp.PILOT_SEED
    with pytest.raises(ValueError):
        pp.authorise_seed("development")


def test_seed_gate_logic_once_nothing_is_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pp, "PENDING_BEFORE_MAIN", ())
    checks: list[bool] = []
    monkeypatch.setattr(pp, "require_launch_approval",
                        lambda *, require_clean=True: checks.append(require_clean))
    monkeypatch.setattr(pp, "MAIN_SEED", 6000)
    monkeypatch.setattr(pp, "FRESH_SEED", 6001)
    assert pp.authorise_seed("main") == 6000
    with pytest.raises(pp.ProtocolNotReady):
        pp.authorise_seed("fresh")
    assert pp.authorise_seed("fresh", confirm_single_final_run=True) == 6001
    assert checks == [True, True]


def test_earlier_protocol_seeds_are_listed() -> None:
    from decision_intelligence.recommendations import cases as rc
    from decision_intelligence.recommendations import protocol as rp

    for seed in (rp.MAIN_SEED, rp.FRESH_SEED, rc.FRESH_SEED,
                 m10.MAIN_SEED, m10.PILOT_SEED,
                 m10.FRESH_SEED):
        assert seed in pp.OTHER_PROTOCOL_SEEDS


def _python_files() -> list[Path]:
    return [p for root in ("src", "scripts", "tests") for p in (REPO / root).rglob("*.py")]


def _imports_predictive_protocol(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and (
                node.module.endswith("predictive_protocol")
                or any(a.name == "predictive_protocol" for a in node.names)):
            return True
        if isinstance(node, ast.Import) and any(
                a.name.endswith("predictive_protocol") for a in node.names):
            return True
    return False


def _seed_offenders(files: list[Path]) -> list[str]:
    values = {getattr(pp, n) for n in SEED_NAMES}
    blocks = (range(pp.COVERAGE_SEED_START, pp.COVERAGE_SEED_START + pp.COVERAGE_REPLICATIONS),
              range(pp.REFIT_STUDY_SEED_START,
                    pp.REFIT_STUDY_SEED_START + pp.REFIT_STUDY_REPLICATIONS))
    offenders: list[str] = []
    for path in files:
        rel = path.relative_to(REPO).as_posix() if path.is_relative_to(REPO) else path.name
        if rel == PROTOCOL_FILE:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and type(node.value) is int and (
                    node.value in values or any(node.value in b for b in blocks)):
                offenders.append(f"{rel}: literal {node.value}")
            if isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(
                    node.func, "id", "")) == "authorise_seed":
                arg = node.args[0] if node.args else None
                if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
                    offenders.append(f"{rel}: authorise_seed without a literal purpose")
                elif rel not in pp.AUTHORISE_ALLOWED_FILES.get(
                        arg.value, ("tests/unit/customer_value/test_predictive_protocol.py",)):
                    # an unknown purpose may appear only in this file's rejection test
                    offenders.append(f"{rel}: authorise_seed({arg.value!r})")
        if not _imports_predictive_protocol(tree):
            continue
        for node in ast.walk(tree):
            name = node.attr if isinstance(node, ast.Attribute) else (
                node.id if isinstance(node, ast.Name) else (
                    node.value if isinstance(node, ast.Constant)
                    and isinstance(node.value, str) else None))
            if name in SEED_NAMES and rel not in pp.SEED_ALLOWED_FILES[name]:
                offenders.append(f"{rel}: {name}")
    return offenders


def test_reserved_seeds_are_referenced_only_where_allowed() -> None:
    assert _seed_offenders(_python_files()) == []


def test_seed_guard_detects_violations(tmp_path: Path) -> None:
    bad = tmp_path / "bad_usage.py"
    bad.write_text(
        "from decision_intelligence.customer_value import predictive_protocol as pp\n"
        "a = pp.MAIN_SEED\n"
        "b = getattr(pp, 'FRESH_SEED')\n"
        f"c = {pp.COVERAGE_SEED_START + 3}\n"
        f"d = {pp.REFIT_STUDY_SEED_START}\n"
        "e = pp.authorise_seed('pilot')\n"
        "purpose = 'main'\n"
        "f = pp.authorise_seed(purpose)\n")
    found = _seed_offenders([bad])
    assert len(found) == 6, found


def test_coverage_levels_are_reported_not_asserted() -> None:
    """Coverage is a reporting criterion: the labelling helpers may be exercised only by the
    test of their own correctness, never used to assert a study's coverage."""
    allowed = "tests/unit/customer_value/test_precision_and_selection.py"
    users = []
    for path in (REPO / "tests").rglob("test_*.py"):
        rel = path.relative_to(REPO).as_posix()
        names = {n.attr if isinstance(n, ast.Attribute) else n.id
                 for n in ast.walk(ast.parse(path.read_text()))
                 if isinstance(n, ast.Attribute | ast.Name)}
        if names & {"classify_coverage", "coverage_acceptance_region"} and rel != allowed:
            users.append(rel)
    assert users == []
    assert "not acceptance tests" in pp.COVERAGE_REPORTING_RULE


def test_penalty_and_replication_conventions() -> None:
    c = pp.PENALTY_CONVENTION
    assert "TOTAL (not mean)" in c.objective
    assert "rows of the fit being made" in c.standardisation
    assert c.unpenalised == "intercept and the tenure-band basis"
    assert list(pp.PENALTY_GRID) == sorted(pp.PENALTY_GRID) and pp.DEFAULT_PENALTY == 1.0
    assert "fixed, predeclared DEFAULT_PENALTY" in pp.TUNING_RULE
    assert pp.COVERAGE_REPLICATIONS == 1000 >= pp.MIN_COVERAGE_REPLICATIONS
    assert pp.REFIT_STUDY_REPLICATIONS < pp.COVERAGE_REPLICATIONS
    assert "PROVISIONAL" in pp.PILOT_FINDINGS_SCOPE


@pytest.mark.parametrize("rel", [
    PROTOCOL_FILE,
    "src/decision_intelligence/customer_value/precision.py",
    "src/decision_intelligence/customer_value/model_selection.py",
    "scripts/plan_m11_precision.py",
])
def test_p0_code_never_reads_ground_truth(rel: str) -> None:
    tree = ast.parse((REPO / rel).read_text())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert not names & {"truth", "EconomicsTruth"}


def test_p1_amendments_are_recorded_and_consistent() -> None:
    assert [a.id for a in pp.PROTOCOL_AMENDMENTS[:4]] == ["A1", "A2", "A3", "A4"]
    assert all(a.before_main_seed and a.stage == "P1" for a in pp.PROTOCOL_AMENDMENTS[:4])
    assert "segment x channel" in pp.model("E4").name and "12 cells" in pp.E4_FIT_UNIT
    assert "MISSPECIFIED REFERENCE" in pp.E4_POOLED_REFERENCE
    assert pp.model("E4").complexity_rank is not None  # the per-cell fit is the candidate
    assert "kept as generated at P1" in pp.SIGNAL_PARAMETERS_DECISION
    assert "kept the repeated-sampling preflight" in pp.PROTOCOL_AMENDMENTS[3].change
    for name in ("signals_contractual", "purchases_noncontractual"):
        assert pp.scenario(name).parameters_status.startswith("fixed at P1")


def test_p2_amendment_pins_the_oracle_numerics_and_closes_its_item() -> None:
    assert [a.id for a in pp.PROTOCOL_AMENDMENTS[:5]] == ["A1", "A2", "A3", "A4", "A5"]
    a5 = pp.PROTOCOL_AMENDMENTS[4]
    assert a5.stage == "P2" and a5.before_main_seed
    on = pp.ORACLE_NUMERICS
    assert (on.bgnbd_jacobi_nodes, on.pareto_jacobi_nodes) == (64, 64)
    assert on.copula_nodes == (64, 48, 32)
    assert "doubled" in on.convergence_check
    assert on.monte_carlo_draws == 1_000_000 and "independent" in on.monte_carlo_check
    assert "closed pending item 'P2: oracle Monte Carlo" in a5.change
    assert not any("oracle Monte Carlo" in item for item in pp.PENDING_BEFORE_MAIN)
    _assert_seed_gate_holds_while_pending()


def test_p3_amendment_adds_only_the_narrow_e2_boundary_exception() -> None:
    assert [a.id for a in pp.PROTOCOL_AMENDMENTS[:6]] == ["A1", "A2", "A3", "A4", "A5", "A6"]
    a6 = pp.PROTOCOL_AMENDMENTS[5]
    assert a6.stage == "P3" and a6.before_main_seed
    rule = pp.E2_GEOMETRIC_BOUNDARY
    assert (rule.model, rule.parameter) == ("E2", "v")
    assert rule.max_score_per_customer == pp.STATUS_RULES.gradient_tolerance
    assert rule.log_likelihood_tolerance == pp.STATUS_RULES.relative_change_tolerance
    assert "Every other status and fallback rule is unchanged" in rule.statement
    # the frozen status rules and statuses themselves are untouched
    assert pp.STATUS_RULES.log_parameter_bound == 10.0 and pp.STATUS_RULES.n_starts == 5
    assert pp.FIT_STATUSES == ("ok", "fallback", "unstable", "failed", "undefined")
    assert "pooled fit of the same model" in pp.STATUS_RULES.fallback
    _assert_seed_gate_holds_while_pending()


def test_p3_amendment_a7_adds_only_the_narrow_e1_boundary_exception() -> None:
    assert [a.id for a in pp.PROTOCOL_AMENDMENTS[:7]] == ["A1", "A2", "A3", "A4", "A5", "A6",
                                                          "A7"]
    a7 = pp.PROTOCOL_AMENDMENTS[6]
    assert a7.stage == "P3" and a7.before_main_seed
    rule = pp.E1_GEOMETRIC_BOUNDARY
    assert rule.model == "E1" and "alpha + beta" in rule.parameter
    assert rule.label != pp.E2_GEOMETRIC_BOUNDARY.label
    assert (rule.max_score_per_customer, rule.log_likelihood_tolerance,
            rule.log_rate_tolerance) == (pp.E2_GEOMETRIC_BOUNDARY.max_score_per_customer,
                                         pp.E2_GEOMETRIC_BOUNDARY.log_likelihood_tolerance,
                                         pp.E2_GEOMETRIC_BOUNDARY.log_rate_tolerance)
    assert "A6 (E2) and every other status and fallback rule are unchanged" in rule.statement
    assert pp.E2_GEOMETRIC_BOUNDARY.model == "E2"  # A6 itself unchanged
    assert pp.FIT_STATUSES == ("ok", "fallback", "unstable", "failed", "undefined")
    _assert_seed_gate_holds_while_pending()


def test_p4_amendment_a8_excludes_three_e3_covariates_and_records_the_clarifications() -> None:
    ids = [a.id for a in pp.PROTOCOL_AMENDMENTS[:8]]
    assert ids == ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"]
    a8 = pp.PROTOCOL_AMENDMENTS[7]
    assert a8.stage == "P4" and a8.before_main_seed
    assert pp.E3_EXCLUDED_COVARIATES == ("acquisition_cohort", "tenure_months", "paid_months")
    # the registry itself is unchanged: every excluded name is still a Tier A feature
    names_a = [f.name for f in pp.features("contractual", "A")]
    assert set(pp.E3_EXCLUDED_COVARIATES) <= set(names_a)
    assert names_a == ["segment", "channel", "tenure_months", "acquisition_cohort",
                       "promo_months_remaining", "paid_months", "mean_contribution",
                       "last_contribution"]
    assert "ties to the larger penalty" in pp.E3_TUNING_LOSS
    assert "no censored validation" in pp.E3_SELECTION_ELIGIBILITY
    assert "registry is unchanged" in pp.DESIGN_DUPLICATE_RULE
    assert pp.TUNING_RULE.startswith("temporal: choose the penalty on the validation landmark")
    assert pp.PENALTY_GRID == (0.01, 0.1, 1.0, 10.0, 100.0) and pp.DEFAULT_PENALTY == 1.0
    assert pp.TENURE_BAND_EDGES == (0, 1, 2, 3, 6, 12, 24)
    assert pp.TENURE_BAND_MIN_EVENTS == 10
    _assert_seed_gate_holds_while_pending()


def test_p6_amendment_a9_authorises_e5_tuning_and_eligibility_only() -> None:
    ids = [a.id for a in pp.PROTOCOL_AMENDMENTS[:9]]
    assert ids == ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"]
    a9 = pp.PROTOCOL_AMENDMENTS[8]
    assert a9.stage == "P6" and a9.before_main_seed
    assert "'P4-amended' -> 'P6-amended'" in a9.change
    assert pp.E5_TUNING_LOSS == (
        "E5: one penalty shared by both parts, chosen under TUNING_RULE by squared error of the "
        "window (12-month at the primary horizon) value P(R > 0) x E[R | R > 0] at the "
        "validation landmark; ties to the larger penalty; if no tuning fit has both parts 'ok', "
        "DEFAULT_PENALTY, labelled 'fixed penalty: no tuning fit was ok'")
    assert pp.E5_SELECTION_ELIGIBILITY == (
        "E5 is a selection candidate at an evaluation cutoff only if a validation fit exists: "
        "at least one tuning landmark whose labels are completely recorded before the "
        "validation landmark. Otherwise E5 is 'not selection-eligible' there; no censored "
        "validation windows")
    # A9 changes nothing else: E3's A8 rules, the gates, tolerances and the penalty grid stand
    assert pp.E3_TUNING_LOSS.startswith("squared error of the window")
    assert pp.E3_SELECTION_ELIGIBILITY.startswith("E3 is a selection candidate")
    assert pp.SELECTION.slope_bounds == (0.85, 1.15)
    assert pp.SELECTION.calibration_in_the_large_tolerance == 0.05
    assert pp.PENALTY_GRID == (0.01, 0.1, 1.0, 10.0, 100.0) and pp.DEFAULT_PENALTY == 1.0
    assert pp.STATUS_RULES.gradient_tolerance == 1e-6
    _assert_seed_gate_holds_while_pending()


def test_p7_amendment_a10_fixes_b1_b2_reference_scope_tier_rule_and_e3_fallback() -> None:
    ids = [a.id for a in pp.PROTOCOL_AMENDMENTS[:10]]
    assert ids == ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"]
    a10 = pp.PROTOCOL_AMENDMENTS[9]
    assert a10.stage == "P7" and a10.before_main_seed
    assert "'P6-amended' -> 'P7-amended'" in a10.change
    assert pp.B1_DEFINITION == (
        "B1 at the customer level (existing_undiscounted_12; the selection candidate and PC1's "
        "comparator) is M10's 'Kaplan–Meier by cell' (backtest.customer_projections, "
        "unchanged) with W = PRIMARY_HORIZON: run rate x sum_{t=1..W} S(a0 + t) / S(a0), "
        "where the run rate is the customer's mean contribution over its last up to 3 "
        "completely recorded months (months <= complete_through(C)), a0 = C - acquisition "
        "month, and S is the paid-month Kaplan–Meier curve of the customer's segment x "
        "channel cell at C, truncated where fewer than MIN_AT_RISK customers remain at risk "
        "and extended by M10's geometric tail over the last KM_TAIL_MONTHS. Population: the "
        "contractual training_stack population (active at C, acquired by "
        "complete_through(C)). Cell status (M10's thresholds): 'undefined' if the cell has no "
        "usable curve (fewer than 2 ages with MIN_AT_RISK at risk); 'unstable' if fewer than "
        "MIN_CHURN_EVENTS churn events; otherwise 'ok'. No fallback (M10 unchanged): a cell "
        "that is not 'ok' counts against min_ok_cell_fraction, and a customer without a "
        "finite prediction is 'undefined' and prevents B1 from qualifying at that cutoff. "
        "Nothing is tuned. Selection inputs as E1/E2: computed from the tables as recorded by "
        "C, observed at the validation landmark L_v, scored on L_v's completely recorded "
        "labels; selection-eligible wherever a validation landmark exists. In PC2 B1 is H4 "
        "(cohort level), never a selection candidate. M10's other customer methods (run rate, "
        "historical average, constant churn pooled and by cell) and H1-H3 are reported "
        "descriptively only")
    assert pp.B2_DEFINITION == (
        "B2: for every customer acquired by T_C = complete_through(C) (E4's population), "
        "prediction of existing_undiscounted_12 = f x s x k, where f = the number of the "
        "customer's purchases (the acquisition purchase included) in the complete months "
        "max(acquisition month, T_C - PRIMARY_HORIZON + 1) .. T_C, carried forward unchanged "
        "to months C+1 .. C+12 (no churn, no annualisation); s = the customer's mean revenue "
        "per purchase over every completely recorded purchase (months <= T_C, the acquisition "
        "purchase included); k = the business's recorded contribution-to-revenue ratio over "
        "completely recorded purchases at C (E4's conversion). Nothing is fitted or tuned; "
        "status 'ok' where the prediction is finite, otherwise 'undefined'. Selection inputs: "
        "the same rule with the tables as recorded by the validation landmark L_v, scored on "
        "L_v's completely recorded labels (as E4); B2 is selection-eligible wherever a "
        "validation landmark exists. No other target")
    assert pp.REFERENCE_FALLBACK_SCOPE == (
        "the reference fallback applies only to scenarios whose kind is among "
        "model(REFERENCE_FALLBACK_MODEL).kinds (contractual); for a non-contractual scenario a "
        "NO QUALIFIED MODEL result is reported with no reference model")
    assert pp.TIER_A_IN_TIER_B == (
        "a Tier A model is a selection candidate in a Tier B selection of a scenario whose "
        "kind it supports: it uses only Tier A information, which Tier B includes")
    assert pp.E3_NO_OK_FIT_RULE == (
        "E3: if temporal tuning is feasible but no tuning fit is 'ok', DEFAULT_PENALTY, "
        "labelled 'fixed penalty: no tuning fit was ok'; the resulting fit is still subject "
        "to the SELECTION status gate")
    # A10 changes no other value: models, gates, tolerances, targets, comparisons, A8/A9 rules
    assert pp.model("B1").complexity_rank == 0 and pp.model("B2").complexity_rank == 0
    assert pp.model("B1").kinds == ("contractual",) and pp.model("B2").kinds == (
        "non_contractual",)
    assert pp.REFERENCE_FALLBACK_MODEL == "E2" and pp.model("E2").kinds == ("contractual",)
    assert pp.REFERENCE_FALLBACK_LABEL == "REFERENCE FALLBACK — not a selection"
    assert pp.SELECTION.slope_bounds == (0.85, 1.15)
    assert pp.SELECTION.calibration_in_the_large_tolerance == 0.05
    assert pp.SELECTION.min_ok_cell_fraction == 0.90
    assert pp.SELECTION.paired_interval_level == 0.95
    assert [pc.id for pc in pp.PRIMARY_COMPARISONS] == ["PC1", "PC2", "PC3", "PC4", "PC5"]
    assert sum(t.primary for t in pp.TARGETS) == 1 and pp.TARGETS[0].primary
    assert pp.E3_TUNING_LOSS == ("squared error of the window (12-month at the primary "
                                 "horizon) value at the validation landmark; ties to the "
                                 "larger penalty")
    assert pp.E5_TUNING_LOSS.startswith("E5: one penalty shared by both parts")
    assert pp.PENALTY_GRID == (0.01, 0.1, 1.0, 10.0, 100.0) and pp.DEFAULT_PENALTY == 1.0
    assert pp.validate_protocol() == []
    _assert_seed_gate_holds_while_pending()


def test_p7_amendment_a11_adds_contractual_outcome_noise_only() -> None:
    ids = [a.id for a in pp.PROTOCOL_AMENDMENTS[:11]]
    assert ids == ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11"]
    a11 = pp.PROTOCOL_AMENDMENTS[10]
    assert a11.stage == "P7" and a11.before_main_seed
    assert "PROTOCOL_STAGE stays 'P7-amended'" in a11.change
    assert pp.CONTRACTUAL_OUTCOME_NOISE == (
        "Contractual predictive intervals (E1, E2, E3): each simulated paid month's contribution "
        "is the ContributionModel expected contribution plus a residual drawn with replacement, "
        "independently per month, from the residuals (recorded contribution minus that mean) of "
        "completely recorded months (billing_month <= complete_through(C)) in the source group "
        "ContributionModel used for that mean (cell x promotion state, else segment x promotion "
        "state, else promotion state). Nothing is estimated from validation or holdout outcomes "
        "and no interval is rescaled (CALIBRATION.recalibration 'none in M11' stands). Intervals "
        "without the residual are kept and reported as the pre-A11 construction")
    # A11 changes nothing else
    assert pp.CALIBRATION.recalibration == "none in M11"
    assert pp.CALIBRATION.interval_levels == (0.80, 0.95)
    assert pp.SELECTION.slope_bounds == (0.85, 1.15)
    assert pp.SELECTION.calibration_in_the_large_tolerance == 0.05
    assert pp.SELECTION.min_ok_cell_fraction == 0.90
    assert [pc.id for pc in pp.PRIMARY_COMPARISONS] == ["PC1", "PC2", "PC3", "PC4", "PC5"]
    assert pp.B1_DEFINITION.startswith("B1 at the customer level")
    assert pp.B2_DEFINITION.startswith("B2: for every customer")
    assert pp.validate_protocol() == []
    _assert_seed_gate_holds_while_pending()


FROZEN_COMPARISONS = (
    ("PC1", "m10_contractual", "A", "existing_undiscounted_12", "E2", "B1"),
    ("PC2", "m10_contractual", "A", "cohort_expected_lifetime", "E2", "B1"),
    ("PC3", "signals_contractual", "B", "existing_undiscounted_12", "E3", "E3"),
    ("PC4", "purchases_noncontractual", "A", "existing_undiscounted_12", "E4", "B2"),
    ("PC5", "signals_contractual", "B", "existing_undiscounted_12", "E5", "E3"),
)


def test_p8_amendment_a12_fixes_the_p8_design_and_keeps_the_seeds_locked() -> None:
    ids = [a.id for a in pp.PROTOCOL_AMENDMENTS[:12]]
    assert ids == ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12"]
    a12 = pp.PROTOCOL_AMENDMENTS[11]
    assert a12.stage == "P8" and a12.before_main_seed
    assert pp.PROTOCOL_STAGE == "P8-amended"
    # safeguard 6: A12 did not unlock reserved seeds (it replaced the preflight item by the
    # dry-run item, which only A13 closed)
    assert "replaced by the P8 dry-run item, so the main and fresh seeds stay refused" in \
        " ".join(a12.change.split())
    _assert_seed_gate_holds_while_pending()
    # safeguard 2: frozen hypotheses, family and gates unchanged
    assert tuple((c.id, c.scenario, c.tier, c.target, c.model_a, c.model_b)
                 for c in pp.PRIMARY_COMPARISONS) == FROZEN_COMPARISONS
    assert pp.FAMILY_ALPHA == 0.05 and pp.PLANNING_ALPHA == pytest.approx(0.01)
    assert pp.SELECTION.slope_bounds == (0.85, 1.15)
    assert pp.SELECTION.calibration_in_the_large_tolerance == 0.05
    assert pp.SELECTION.min_ok_cell_fraction == 0.90
    assert pp.SELECTION.paired_interval_level == 0.95
    assert pp.ORACLE_GATE.consistency_z == 3.0 and pp.ORACLE_GATE.positive_interval_level == 0.95
    assert pp.STATUS_RULES.gradient_tolerance == 1e-6 and pp.STATUS_RULES.min_events == 30
    assert pp.CALIBRATION.recalibration == "none in M11"
    assert (pp.COVERAGE_REPLICATIONS, pp.MIN_COVERAGE_REPLICATIONS,
            pp.REFIT_STUDY_REPLICATIONS) == (1000, 500, 200)
    assert pp.CLUSTERED_INFERENCE.metric_bootstrap_resamples == 2000
    assert pp.CLUSTERED_INFERENCE.refit_bootstrap_resamples == 200
    b = pp.BUDGETS
    assert (b.evidence_target_min, b.evidence_ceiling_min, b.evidence_memory_gb) == (
        20.0, 180.0, 4.0)
    assert (b.new_tests_target_s, b.new_tests_ceiling_s, b.single_test_ceiling_s,
            b.test_memory_gb) == (90.0, 150.0, 10.0, 1.5)
    # PC definitions: exact wording of the test and of PC3's non-evaluable rule
    assert set(pp.PC_DEFINITIONS) == {"PC1", "PC2", "PC3", "PC4", "PC5"}
    assert pp.PC_TEST == (
        "PC1, PC3, PC4 and PC5: one two-sided test each of H0: E[D] = 0 on the main seed, D = "
        "(R - prediction of model_a)^2 - (R - prediction of model_b)^2, pooled over the "
        "comparison's evaluation cutoffs (customer x cutoff rows); estimate mean(D); p-value "
        "from the CR1 customer-clustered z (normal reference); a 95% customer-cluster "
        "percentile bootstrap interval (CLUSTERED_INFERENCE.metric_bootstrap_resamples) is "
        "reported; a customer without a finite prediction from either model is excluded and "
        "counted; a negative estimate favours model_a. PC2 follows PC_DEFINITIONS. Holm at "
        "FAMILY_ALPHA across the five p-values PC1-PC5")
    assert pp.PC_DEFINITIONS["PC3"] == (
        "E3 Tier B vs, at each of cutoffs 36 and 42 of signals_contractual, the Tier A model "
        "selected by the frozen SELECTION rule (Tier A candidates only); a cutoff whose Tier A "
        "selection is NO QUALIFIED MODEL is non-evaluable and its rows are excluded, never "
        "replaced by an unqualified model; PC_TEST pooled over the evaluable cutoffs; if neither "
        "cutoff is evaluable, PC3 enters Holm with p = 1, reported 'non-evaluable'")
    assert "inference across R1's independent generator draws" in pp.PC_DEFINITIONS["PC2"]
    assert "two-sided one-sample t test across the R1 draws" in pp.PC_DEFINITIONS["PC2"]
    assert "M10's common-support rule" in pp.PC_DEFINITIONS["PC1"]
    assert "E5 as specified (A9)" in pp.PC_DEFINITIONS["PC5"]
    # repeated sampling: approved counts, seeds inside their reserved blocks, shared seeds
    st = {r.id: r for r in pp.REPEATED_SAMPLING}
    assert {k: (v.replications, v.scales, v.role) for k, v in st.items()} == {
        "R1": (1000, (1.0,), "confirmatory"), "R2": (500, (1.0,), "reported"),
        "R3": (200, (0.5, 1.0, 2.0), "confirmatory"), "REFIT": (200, (1.0,), "reported")}
    assert st["R1"].seed_start == st["R2"].seed_start == st["R3"].seed_start == \
        pp.COVERAGE_SEED_START
    assert st["REFIT"].seed_start == pp.REFIT_STUDY_SEED_START
    assert st["REFIT"].content.startswith(
        "E2 on m10_contractual, Tier A, cutoff 30 only (cutoff 24 excluded for runtime; E3 not "
        "included, E3_REFIT_DISCLOSURE)")
    assert ("excluded for runtime: m10_contractual C = 24, signals_contractual C = 36, "
            "purchases_noncontractual C = 24") in st["R2"].content
    pc2 = pp.PC_DEFINITIONS["PC2"]
    for convention in ("truth = the generator's population_value for the channel",
                       "the P3 cohort_value with the cell's fit as used by E2's predictions",
                       "weighted by the channel's acquisitions in each segment over the "
                       "CHURN_WINDOW_MONTHS before C (M10's H3 cell weighting)",
                       "H4 = M10's H4 for the channel at C", "summed to INFINITE_SUM_MONTHS"):
        assert convention in pc2
    assert pp.BUDGET_SCOPE.startswith(
        "BUDGETS.evidence_target_min applies to one main-seed evidence run")
    assert "bounds the complete P8 execution" in pp.BUDGET_SCOPE
    assert pp.P8_DRY_RUN.startswith(
        "development seeds only (never a reserved seed or block): 10 draws for each of R1, R2 "
        "and REFIT, 10 per scale for R3, 20 refits per REFIT draw")
    assert "'dry run - not evidence'" in pp.P8_DRY_RUN
    for model in ("E1 and E2", "E3 on", "E4 on", "S2 with E5"):
        assert model in st["R2"].content
    assert "never pooled or combined" in pp.SHARED_SEEDS_DISCLOSURE
    # safeguard 1: E3 refit-bootstrap validation explicitly not performed
    assert pp.E3_REFIT_DISCLOSURE.startswith(
        "No refit-bootstrap validation of E3 is performed")
    # safeguard 3: confirmatory tests separated from exploratory diagnostics
    assert pp.CONFIRMATORY_TESTS == ("PC1-PC5 (PC_TEST, PC_DEFINITIONS), Holm at FAMILY_ALPHA",
                                     "ORACLE_GATE on R1", "ACCEPTANCE_GATE_4 on R3")
    assert "E3 Tier B vs E3 Tier A" in pp.EXPLORATORY_DIAGNOSTICS
    assert not set(pp.CONFIRMATORY_TESTS) & set(pp.EXPLORATORY_DIAGNOSTICS)
    assert "|z| < ORACLE_GATE.consistency_z" in pp.ACCEPTANCE_GATE_4
    assert "95% t interval entirely below 0" in pp.ACCEPTANCE_GATE_4
    # safeguards 4 and 5: original E4 quadrature outside the validated range; P7 unchanged
    assert pp.E4_ACCELERATED_MAX_X == 44
    assert "every other customer uses the original per-count Gauss-Jacobi rule" in \
        pp.E4_ACCELERATED_SIMULATION
    assert "P7's e4_simulate (and so P7's record) are unchanged" in pp.E4_ACCELERATED_SIMULATION
    assert (pp.P8_EXECUTION.workers, pp.P8_EXECUTION.threads_per_worker) == (3, 1)
    assert pp.validate_protocol() == []


@pytest.mark.parametrize("breakage", ["pc", "r2_low", "r1_count", "seed_block", "refit",
                                      "scales", "budget"])
def test_validate_protocol_rejects_a12_inconsistencies(monkeypatch: pytest.MonkeyPatch,
                                                       breakage: str) -> None:
    from dataclasses import replace
    studies = list(pp.REPEATED_SAMPLING)
    if breakage == "pc":
        monkeypatch.setattr(pp, "PC_DEFINITIONS", {k: v for k, v in pp.PC_DEFINITIONS.items()
                                                   if k != "PC3"})
    elif breakage == "r2_low":
        studies[1] = replace(studies[1], replications=pp.MIN_COVERAGE_REPLICATIONS - 1)
    elif breakage == "r1_count":
        studies[0] = replace(studies[0], replications=pp.COVERAGE_REPLICATIONS - 1)
    elif breakage == "seed_block":
        studies[2] = replace(studies[2], seed_start=pp.REFIT_STUDY_SEED_START)
    elif breakage == "refit":
        studies[3] = replace(studies[3], replications=pp.REFIT_STUDY_REPLICATIONS + 1)
    elif breakage == "scales":
        studies[2] = replace(studies[2], scales=(0.5, 2.0))
    else:
        monkeypatch.setattr(pp, "BUDGETS", replace(pp.BUDGETS, evidence_ceiling_min=10.0))
    monkeypatch.setattr(pp, "REPEATED_SAMPLING", tuple(studies))
    assert pp.validate_protocol() != []


def test_p8_amendment_a13_closes_the_dry_run_item_and_pins_the_execution_rules() -> None:
    ids = [a.id for a in pp.PROTOCOL_AMENDMENTS[:13]]
    assert ids == ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12",
                   "A13"]
    a13 = pp.PROTOCOL_AMENDMENTS[12]
    assert (a13.stage, a13.date, a13.before_main_seed) == ("P8", "2026-11-20", True)
    assert "closes the P8 dry-run item" in a13.change
    assert "PROTOCOL_STAGE stays 'P8-amended'" in a13.change
    assert pp.PROTOCOL_STAGE == "P8-amended"
    assert pp.COVERAGE_DRAW_LEVEL_RULE == (
        "For REPEATED_SAMPLING customer-level intervals (R2 and REFIT) the unit of replication "
        "is the generator draw: for each setting, model, interval, level and group, the mean of "
        "per-draw coverage, its two-sided 95% t interval across draws, and the label 'below "
        "nominal' (interval entirely below the nominal level), 'above nominal' (entirely "
        "above) or 'consistent with nominal'; the pooled customer proportion is descriptive "
        "only. precision.classify_coverage remains the rule for one interval per replication. "
        "Coverage is reported, never asserted, and never triggers recalibration or a change of "
        "method, seed, scale or replication count (COVERAGE_REPORTING_RULE)")
    assert pp.P8_SINGLE_ATTEMPT == (
        "The final P8 run (scripts/generate_m11_evidence.py --mode final with --expect-manifest "
        "and --expect-code equal to the digests pinned after A13) is attempted once in the "
        "repository: after protocol verification, and only if no final ledger exists anywhere "
        "under docs/evidence, it takes the fixed reservation 'final' in the attempt registry "
        "docs/evidence/m11-p8-attempts (an atomically created, immutable entry recorded in a "
        "hash-chained attempts.log written under an exclusive lock; the registry is validated "
        "before every reservation and any alteration, deletion or unlogged entry refuses every "
        "further attempt); only then is the run directory's ledger.json created and any "
        "reserved seed used. Phases run in the order main seed, REPEATED_SAMPLING, M10 "
        "comparability, fresh seed; the fresh seed is authorised and recorded only at that "
        "last phase. Raw results are kept outside Git with their SHA-256 in the ledger; the "
        "evidence is record.md, summary.json and CSV tables, written only when every phase is "
        "complete. A failed or interrupted run is recorded and is never resumed, repeated or "
        "moved to replacement seeds, in any directory, without a new amendment")
    assert pp.P8_APPROVED_DRY_RUN == (
        "docs/evidence/m11-p8-dry-run (reservation dry-run-a3cef63df948e378), generated under "
        "manifest e40970d2d725966e57f8c648b897528a932bc09a8a8f98c60c91392c792cd6eb and code "
        "b4a1554bd3a4fff2e418e8f359f85c64a66ed080764db7869b92841ee1587b4b: record.md sha256 "
        "57a645b1f159e9c64e6ceee7fc5f8b641f2692c2d6722abc9ad75a9dcdf406a4; ledger.json "
        "f1977321a727cc2e87b9594debccb537a526154d64e1e6925292d03c2269c5ae; summary.json "
        "472ded23d1eb6583c7f45c2c296143745691a6c1210cb2f19f0f6efdaec84816; 10 draws per study, "
        "wall 3.0 min, peak 2.31 GB, extrapolated full size 141 min (ceiling 180)")
    # The A13 historical record and the live seed gate are separate.
    _assert_seed_gate_holds_while_pending()
    # A13 changes nothing else
    assert [pc.id for pc in pp.PRIMARY_COMPARISONS] == ["PC1", "PC2", "PC3", "PC4", "PC5"]
    assert pp.SELECTION.slope_bounds == (0.85, 1.15) and pp.FAMILY_ALPHA == 0.05
    assert {s.id: s.replications for s in pp.REPEATED_SAMPLING} == {
        "R1": 1000, "R2": 500, "R3": 200, "REFIT": 200}
    assert pp.BUDGETS.evidence_ceiling_min == 180.0 and pp.validate_protocol() == []


def test_the_approved_dry_run_record_matches_its_pinned_hashes() -> None:
    import hashlib
    root = REPO / "docs" / "evidence" / "m11-p8-dry-run"
    pinned = dict(re.findall(r"(record\.md|ledger\.json|summary\.json)(?: sha256)? "
                             r"([0-9a-f]{64})", pp.P8_APPROVED_DRY_RUN))
    assert set(pinned) == {"record.md", "ledger.json", "summary.json"}
    for name, sha in pinned.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == sha, name


def test_p8_amendment_a14_records_the_failure_fixes_e4_and_authorises_one_retry() -> None:
    import hashlib

    from decision_intelligence.customer_value import predictive_cases as pc
    ids = [a.id for a in pp.PROTOCOL_AMENDMENTS]
    assert ids[:18] == ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11",
                        "A12", "A13", "A14", "A15", "A16", "A17", "A18"]
    assert ids[18:] == (["A19", "A20"] if pp.P8_FINAL_ATTEMPT_KEY == "final-a20" else ["A19"])
    a14 = next(a for a in pp.PROTOCOL_AMENDMENTS if a.id == "A14")
    # A14 followed the failed attempt's main-seed computation but preceded any inspection
    assert (a14.stage, a14.date) == ("P8", "2026-11-20")
    assert (a14.before_main_seed, a14.before_outcome_inspection) == (False, True)
    assert "PROTOCOL_STAGE stays 'P8-amended'" in a14.change
    assert pp.PROTOCOL_STAGE == "P8-amended"
    assert pp.P8_FAILED_ATTEMPTS[0] == (
        "m11-p8-final/ledger.json",
        "cd7c8391347df2f9a2cf802161f35a0cb6a9a889b29c8fce3825a22ebe5902a9")
    assert pp.E4_ACCELERATED_MIN_A == 0.4 == pc.E4_ACCELERATED_MIN_A
    assert pp.E4_ZERO_DROPOUT_RULE == (
        "In E4 outcome simulation (P7 e4_simulate and P8 e4_simulate_fast) a dropout probability "
        "that is exactly 0 in floating point is its limit, no dropout within the window "
        "(purchases until dropout = the largest int64, the value NumPy returns for every "
        "positive p below about 1e-300); the geometric draw is made with p replaced by 1, so "
        "every other draw and the random stream are unchanged. A non-finite or out-of-range p, "
        "or a non-finite or non-positive BG/NBD parameter draw, raises an error naming the "
        "task's study and seed")
    assert pp.E4_ACCELERATED_SMALL_A.startswith(
        "Parameter draws with a < E4_ACCELERATED_MIN_A use the original per-count Gauss-Jacobi "
        "nodes with the telescoped integrand (A(e_0) - A(e_H)) / p, finite at p = 0; for a >= "
        "0.4 the accelerated rule is within 1.3e-11 of the per-count rule (the A12 bound 2e-11 "
        "holds).")
    assert pp.E4_ACCELERATED_SMALL_A.endswith(
        "only R2's E4 interval coverage (reported, never asserted) uses it, never a point "
        "prediction, primary comparison, selection or gate")
    assert pp.P8_RETRY_RULE == (
        "Exactly one new final attempt: reservation final-a14 in docs/evidence/"
        "m11-p8-attempts and run directory docs/evidence/m11-p8-final-a14, on the same seed plan "
        "as the failed attempt (main, M10 comparability, coverage and refit blocks; the fresh "
        "seed still unused and last), allowed only if the only final ledgers under "
        "docs/evidence are those in P8_FAILED_ATTEMPTS with unchanged SHA-256. The failed "
        "attempt's ledger, raw results and reservation are kept unchanged as evidence and its "
        "main-seed results are never inspected. P8_SINGLE_ATTEMPT otherwise stands: no further "
        "attempt without a new amendment")
    assert pp.P8_FAILED_ATTEMPT_RECORD.startswith("First final attempt (reservation 'final'")
    for fragment in ("d514d59e4b8e813f23855c482f2f952818ebf202446bfef7"
                     "1a341b072cd23b11, never opened",
                     "the fresh seed was not consumed",
                     "docs/evidence/m11-p8-final-attempt-1.md"):
        assert fragment in pp.P8_FAILED_ATTEMPT_RECORD
    # the preserved evidence matches the amendment (hashes only; raw results are never read)
    evidence = REPO / "docs" / "evidence"
    for rel, sha in pp.P8_FAILED_ATTEMPTS:
        assert hashlib.sha256((evidence / rel).read_bytes()).hexdigest() == sha
    entry = evidence / "m11-p8-attempts" / "final.json"
    assert hashlib.sha256(entry.read_bytes()).hexdigest() in pp.P8_FAILED_ATTEMPT_RECORD.replace(
        "e5dc98f0402381f5c7308327" "8ca2cf140e6cf76fb77a3b6e5c5641f724c6ad04",
        "e5dc98f0402381f5c73083278ca2cf140e6cf76fb77a3b6e5c5641f724c6ad04")
    # A14 changes nothing else
    assert [c.id for c in pp.PRIMARY_COMPARISONS] == ["PC1", "PC2", "PC3", "PC4", "PC5"]
    assert pp.SELECTION.slope_bounds == (0.85, 1.15) and pp.FAMILY_ALPHA == 0.05
    assert pp.SELECTION.calibration_in_the_large_tolerance == 0.05
    assert {st.id: st.replications for st in pp.REPEATED_SAMPLING} == {
        "R1": 1000, "R2": 500, "R3": 200, "REFIT": 200}
    assert pp.E4_ACCELERATED_MAX_X == 44 and pp.BUDGETS.evidence_ceiling_min == 180.0
    assert pp.validate_protocol() == []


def test_amendment_timing_is_recorded_truthfully() -> None:
    """A14 followed the first main computation. A15 through A18 followed both."""
    timing = {a.id: (a.before_main_seed, a.before_outcome_inspection)
              for a in pp.PROTOCOL_AMENDMENTS}
    end = 21 if pp.P8_FINAL_ATTEMPT_KEY == "final-a20" else 20
    assert timing == {**{f"A{i}": (True, True) for i in range(1, 14)},
                      **{f"A{i}": (False, True) for i in range(14, end)}}
    assert pp.AMENDMENTS_AFTER_MAIN_SEED_COMPUTATION == tuple(f"A{i}" for i in range(14, end))
    assert pp.validate_protocol() == []


def test_a15_records_outcome_blindness() -> None:
    a15 = next(a for a in pp.PROTOCOL_AMENDMENTS if a.id == "A15")
    assert (a15.id, a15.stage) == ("A15", "P8")
    approval_date = date.fromisoformat(a15.date)
    assert approval_date.isoformat() == a15.date and approval_date >= date(2026, 11, 20)
    assert "Both failed attempts completed main before A15." in a15.reason
    assert "crash cause remains unconfirmed" in a15.reason
    assert "read as bytes for SHA-256 verification" in a15.reason
    assert "without deserialisation or outcome inspection" in a15.reason


def test_a18_requires_separate_launch_approval(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from decision_intelligence.customer_value import launch_approval

    monkeypatch.setattr(launch_approval, "_APPROVAL_DIR", tmp_path / "missing-approval")
    a16, a17, a18 = pp.PROTOCOL_AMENDMENTS[15:18]
    assert (a16.id, a16.stage, a16.date) == ("A16", "P8", "2026-11-21")
    assert "Retain the execution hold." in a16.change
    assert "was pending at A16 and recorded under A17" in a16.reason
    assert (a17.id, a17.stage, a17.date) == ("A17", "P8", "2026-11-22")
    assert "attempts remain uninspected" in a17.reason
    assert "Approval covers hold-release configuration only." in a17.reason
    assert "green CI on the exact release commit" in a17.reason
    assert "a passing final host preflight" in a17.reason
    assert "separate explicit owner authorisation for a single launch" in a17.reason
    assert "docs/evidence/m11-p8-final-a15/" in a17.change
    assert (a18.id, a18.stage, a18.date) == ("A18", "P8", "2026-11-22")
    assert "release preparation only" in a18.reason
    assert pp.PENDING_BEFORE_MAIN == ()
    assert pp.validate_protocol() == []
    monkeypatch.setattr(pp, "P8_FINAL_ATTEMPT_KEY", "final-a15")
    with pytest.raises(pp.ProtocolNotReady, match="permanently consumed"):
        pp.authorise_seed("main")
    with pytest.raises(pp.ProtocolNotReady, match="permanently consumed"):
        pp.authorise_seed("fresh", confirm_single_final_run=True)


def test_unconfirmed_a16_still_blocks_seeds_if_hold_is_cleared(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pp, "PENDING_BEFORE_MAIN", ())
    historical = pp.PROTOCOL_AMENDMENTS[:-2]
    monkeypatch.setattr(pp, "PROTOCOL_AMENDMENTS", (
        *historical[:-1], replace(historical[-1], before_outcome_inspection=None),
    ))
    monkeypatch.setattr(pp, "AMENDMENTS_AFTER_MAIN_SEED_COMPUTATION", ("A14", "A15", "A16"))
    problems = pp.validate_protocol()
    assert "amendments require confirmed outcome blindness before execution" in problems
    with pytest.raises(pp.ProtocolNotReady, match="outcome-blindness confirmation"):
        pp.authorise_seed("main")
    with pytest.raises(pp.ProtocolNotReady, match="outcome-blindness confirmation"):
        pp.authorise_seed("fresh", confirm_single_final_run=True)


@pytest.mark.parametrize("breakage", ["earlier_marked_after", "a14_marked_before",
                                      "inspected", "unconfirmed_a15",
                                      "unlisted_exception", "order"])
def test_validate_protocol_guards_amendment_timing(monkeypatch: pytest.MonkeyPatch,
                                                   breakage: str) -> None:
    amends = list(pp.PROTOCOL_AMENDMENTS)
    idx = {a.id: i for i, a in enumerate(amends)}
    if breakage == "earlier_marked_after":  # an earlier amendment may not claim the exception
        amends[idx["A13"]] = replace(amends[idx["A13"]], before_main_seed=False)
    elif breakage == "a14_marked_before":  # nor may A14 claim to precede the computation
        amends[idx["A14"]] = replace(amends[idx["A14"]], before_main_seed=True)
    elif breakage == "inspected":
        amends[idx["A14"]] = replace(amends[idx["A14"]], before_outcome_inspection=False)
    elif breakage == "unconfirmed_a15":
        amends[idx["A15"]] = replace(amends[idx["A15"]], before_outcome_inspection=None)
    elif breakage == "unlisted_exception":
        monkeypatch.setattr(pp, "AMENDMENTS_AFTER_MAIN_SEED_COMPUTATION", ())
    else:  # a new amendment incorrectly claims to precede main after A14–A18
        amends.append(replace(amends[idx["A12"]], id="A19"))
    monkeypatch.setattr(pp, "PROTOCOL_AMENDMENTS", tuple(amends))
    assert pp.validate_protocol() != []
