"""M11 P0: frozen-protocol consistency, integrity and reserved-seed safeguards."""

from __future__ import annotations

import ast
import json
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

FROZEN_DIGEST = "0a1ea3bdb7c60e0a82fd4b2f08ff7cde067da852ef69e5b0a3327921f4f608a8"
"""SHA-256 of the protocol manifest (P0 as amended through A9). Any protocol change must update
this value in the same reviewed diff and be disclosed in the M11 evidence."""
FROZEN_CODE_DIGEST = "244acc014ecd3acd5ca995e47bd3bc7866eb38946d1d220d7a8c68363d49ada0"
"""SHA-256 of the exact bytes of ``PROTOCOL_CODE_FILES`` (decision logic, not only values)."""


def test_protocol_is_internally_consistent() -> None:
    assert pp.validate_protocol() == []


def test_protocol_digest_is_pinned() -> None:
    assert pp.protocol_digest() == FROZEN_DIGEST


def test_protocol_code_digest_is_pinned() -> None:
    assert pp.protocol_code_digest() == FROZEN_CODE_DIGEST


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


def test_seed_gate_refuses_main_and_fresh_while_items_are_pending() -> None:
    assert pp.PENDING_BEFORE_MAIN
    assert pp.authorise_seed("pilot") == pp.PILOT_SEED
    with pytest.raises(pp.ProtocolNotReady):
        pp.authorise_seed("main")
    with pytest.raises(pp.ProtocolNotReady):
        pp.authorise_seed("fresh", confirm_single_final_run=True)
    with pytest.raises(ValueError):
        pp.authorise_seed("development")


def test_seed_gate_logic_once_nothing_is_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pp, "PENDING_BEFORE_MAIN", ())
    assert pp.authorise_seed("main") == pp.MAIN_SEED
    with pytest.raises(pp.ProtocolNotReady):
        pp.authorise_seed("fresh")
    assert pp.authorise_seed("fresh", confirm_single_final_run=True) == pp.FRESH_SEED


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
    assert any("COVERAGE_REPLICATIONS" in item for item in pp.PENDING_BEFORE_MAIN)
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
    assert len(pp.PENDING_BEFORE_MAIN) == 1
    assert "COVERAGE_REPLICATIONS" in pp.PENDING_BEFORE_MAIN[0]
    assert not any("oracle Monte Carlo" in item for item in pp.PENDING_BEFORE_MAIN)
    with pytest.raises(pp.ProtocolNotReady):  # one item still pending: seeds stay refused
        pp.authorise_seed("main")


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
    assert len(pp.PENDING_BEFORE_MAIN) == 1
    with pytest.raises(pp.ProtocolNotReady):
        pp.authorise_seed("main")


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
    with pytest.raises(pp.ProtocolNotReady):
        pp.authorise_seed("main")


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
    with pytest.raises(pp.ProtocolNotReady):
        pp.authorise_seed("main")


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
    assert len(pp.PENDING_BEFORE_MAIN) == 1
    with pytest.raises(pp.ProtocolNotReady):
        pp.authorise_seed("main")
    with pytest.raises(pp.ProtocolNotReady):
        pp.authorise_seed("fresh", confirm_single_final_run=True)


def test_p7_amendment_a10_fixes_b1_b2_reference_scope_tier_rule_and_e3_fallback() -> None:
    ids = [a.id for a in pp.PROTOCOL_AMENDMENTS[:10]]
    assert ids == ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"]
    a10 = pp.PROTOCOL_AMENDMENTS[9]
    assert a10.stage == "P7" and a10.before_main_seed
    assert pp.PROTOCOL_STAGE == "P7-amended"
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
    assert len(pp.PENDING_BEFORE_MAIN) == 1
    assert pp.validate_protocol() == []
    with pytest.raises(pp.ProtocolNotReady):
        pp.authorise_seed("main")
    with pytest.raises(pp.ProtocolNotReady):
        pp.authorise_seed("fresh", confirm_single_final_run=True)


def test_p7_amendment_a11_adds_contractual_outcome_noise_only() -> None:
    ids = [a.id for a in pp.PROTOCOL_AMENDMENTS]
    assert ids == ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11"]
    a11 = pp.PROTOCOL_AMENDMENTS[-1]
    assert a11.stage == "P7" and a11.before_main_seed
    assert "PROTOCOL_STAGE stays 'P7-amended'" in a11.change
    assert pp.PROTOCOL_STAGE == "P7-amended"
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
    assert len(pp.PENDING_BEFORE_MAIN) == 1
    assert pp.validate_protocol() == []
    with pytest.raises(pp.ProtocolNotReady):
        pp.authorise_seed("main")
    with pytest.raises(pp.ProtocolNotReady):
        pp.authorise_seed("fresh", confirm_single_final_run=True)
