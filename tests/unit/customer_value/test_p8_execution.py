"""M11 P8 final-mode orchestration (A12; proposed A13): phase order, the fresh seed once and last,
the repository-level attempt registry (duplicate, other-directory, interrupted and concurrent
attempts; tampering), the per-run ledger, evidence verification, digest checks, seed isolation
and the draw-level coverage rule. Fake draws and temporary registries only; development seeds
only; no reserved seed is generated or used."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from scipy import stats

from decision_intelligence.customer_value import predictive_cases as pc
from decision_intelligence.customer_value import predictive_protocol as pp

REPO = Path(__file__).resolve().parents[3]


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "generate_m11_evidence_exec", REPO / "scripts" / "generate_m11_evidence.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["generate_m11_evidence_exec"] = mod
    spec.loader.exec_module(mod)
    return mod


GEN = _script()


class Recorder:
    """A fake draw: records the order of tasks; optionally fails on one study."""

    def __init__(self, fail_on: str | None = None,
                 exc: type[BaseException] = RuntimeError) -> None:
        self.calls: list[pc.Task] = []
        self.fail_on, self.exc = fail_on, exc

    def __call__(self, task: pc.Task) -> dict[str, Any]:
        self.calls.append(task)
        if task.study == self.fail_on:
            raise self.exc(f"{task.study} failed")
        return {"task": task, "seed": task.seed, "seconds": 0.0}


def _stub_report(plan: Any, results: dict[str, Any], perf: dict[str, Any], run_dir: Path,
                 output: Path | None) -> list[Path]:
    out = run_dir / "record.md"
    out.write_text(f"stub {sorted(results)}")
    return [out]


def _plan() -> Any:
    return GEN.rehearsal_plan(draws=2, refits=2)


def _run(tmp_path: Path, run: str = "run", fn: Any = None, **kw: Any) -> Path:
    out: Path = GEN.execute(kw.pop("plan", _plan()), tmp_path / run, 1, fn=fn or Recorder(),
                            report_fn=_stub_report, registry=tmp_path / "registry",
                            evidence_root=tmp_path, **kw)
    return out


def _ledger(run_dir: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((run_dir / "ledger.json").read_text())
    return data


# --- order and the fresh seed ---------------------------------------------------------------
def test_phases_run_in_order_and_the_fresh_seed_is_used_once_last(tmp_path: Path) -> None:
    plan, rec = _plan(), Recorder()
    out = _run(tmp_path, fn=rec, plan=plan)
    studies = [t.study for t in rec.calls]
    seeds = [t.seed for t in rec.calls]
    assert studies[0] == "MAIN" and seeds[0] == plan.main
    assert studies[-2:] == ["M10", "MAIN"] and seeds[-2] == plan.m10_comparability
    fresh = seeds[-1]
    assert fresh == GEN.REHEARSAL_SEED_START + 2 and seeds.count(fresh) == 1
    assert studies[1:-2] == ["R1"] * 2 + ["R2"] * 2 + ["R3"] * 6 + ["REFIT"] * 2
    led = _ledger(tmp_path / "run")
    order = [(e["phase"], e["state"]) for e in led["events"]]
    assert [p for p, s in order if s == "done"] == ["verify", "main", "repeated_sampling",
                                                     "m10_comparability", "fresh", "record"]
    assert order.index(("fresh", "seed consumed")) < order.index(("fresh", "done"))
    assert led["status"] == "complete" and out.read_text().startswith("stub")
    assert led["reservation"] == GEN.reservation_key(plan)
    assert all(e["file"].startswith("raw/") for e in led["events"] if "sha256" in e)
    assert GEN.verify_run(tmp_path / "run") == []
    assert GEN.main(["--verify", str(tmp_path / "run")]) == 0


# --- single attempt across all run directories ---------------------------------------------------
def test_the_same_attempt_is_refused_in_any_run_directory(tmp_path: Path) -> None:
    _run(tmp_path)
    before = (tmp_path / "run" / "ledger.json").read_bytes()
    for other in ("run", "elsewhere"):
        rec = Recorder()
        with pytest.raises(GEN.RunRefused, match="already exists"):
            _run(tmp_path, run=other, fn=rec)
        assert rec.calls == []
    assert (tmp_path / "run" / "ledger.json").read_bytes() == before
    assert not (tmp_path / "elsewhere").exists()
    assert GEN.AttemptRegistry(tmp_path / "registry").validate() == []


@pytest.mark.parametrize("fail_on, exc, status, phase", [
    ("REFIT", RuntimeError, "failed", "repeated_sampling"),
    ("M10", RuntimeError, "failed", "m10_comparability"),
    ("R2", KeyboardInterrupt, "interrupted", "repeated_sampling")])
def test_failed_or_interrupted_attempts_are_recorded_and_never_rerun(
        tmp_path: Path, fail_on: str, exc: type[BaseException], status: str,
        phase: str) -> None:
    with pytest.raises(exc):
        _run(tmp_path, fn=Recorder(fail_on, exc))
    led = _ledger(tmp_path / "run")
    assert (led["status"], led["phase"]) == (status, phase)
    assert not any(e["phase"] == "fresh" for e in led["events"])  # fresh seed not consumed
    assert not (tmp_path / "run" / "record.md").exists()
    assert f"status {status}" in GEN.verify_run(tmp_path / "run")
    for other in ("run", "retry"):  # no silent restart, no replacement seeds, no bypass
        again = Recorder()
        with pytest.raises(GEN.RunRefused):
            _run(tmp_path, run=other, fn=again)
        assert again.calls == []


def test_concurrent_attempts_reserve_exactly_once(tmp_path: Path) -> None:
    reg = GEN.AttemptRegistry(tmp_path / "registry")
    barrier = threading.Barrier(8)
    outcomes: list[str] = []
    lock = threading.Lock()

    def attempt(i: int) -> None:
        barrier.wait()
        try:
            reg.reserve("final", {"run_dir": f"dir{i}"})
            result = "reserved"
        except GEN.RunRefused:
            result = "refused"
        with lock:
            outcomes.append(result)

    threads = [threading.Thread(target=attempt, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(outcomes) == ["refused"] * 7 + ["reserved"]
    assert reg.validate() == [] and not reg.lock.exists()
    assert len(reg.log.read_text().splitlines()) == 1


@pytest.mark.parametrize("tamper", ["alter_entry", "delete_entry", "drop_log_line",
                                    "edit_log", "unlogged_entry"])
def test_registry_tampering_refuses_every_further_attempt(tmp_path: Path, tamper: str) -> None:
    reg = GEN.AttemptRegistry(tmp_path / "registry")
    reg.reserve("first", {"run_dir": "a"})
    reg.reserve("second", {"run_dir": "b"})
    first = reg.root / "first.json"
    lines = reg.log.read_text().splitlines()
    if tamper == "alter_entry":
        first.write_text(first.read_text().replace('"a"', '"z"'))
    elif tamper == "delete_entry":
        first.unlink()
    elif tamper == "drop_log_line":
        reg.log.write_text(lines[1] + "\n")
    elif tamper == "edit_log":
        reg.log.write_text(lines[0] + "\n" + lines[1].replace('"second"', '"other"') + "\n")
    else:
        (reg.root / "sneaky.json").write_text("{}")
    assert reg.validate() != []
    with pytest.raises(GEN.RunRefused, match="tampered"):
        reg.reserve("third", {"run_dir": "c"})
    assert not (reg.root / "third.json").exists()


def test_a_held_lock_refuses_rather_than_waits_forever(tmp_path: Path) -> None:
    reg = GEN.AttemptRegistry(tmp_path / "registry")
    reg.root.mkdir(parents=True)
    reg.lock.write_text("")
    with pytest.raises(GEN.RunRefused, match="held"):
        reg._acquire(timeout_s=0.05)


def test_final_mode_refuses_if_any_final_ledger_exists(tmp_path: Path) -> None:
    old = tmp_path / "some-other-dir"
    old.mkdir()
    (old / "ledger.json").write_text(json.dumps({"mode": "final"}))
    (tmp_path / "rehearsal").mkdir()
    (tmp_path / "rehearsal" / "ledger.json").write_text(json.dumps({"mode": "rehearsal"}))
    assert GEN.prior_final_attempts(tmp_path) == [old / "ledger.json"]


# --- evidence -------------------------------------------------------------------------------
def test_incomplete_or_altered_evidence_is_detected(tmp_path: Path) -> None:
    _run(tmp_path)
    run_dir = tmp_path / "run"
    assert GEN.verify_run(run_dir) == []
    pkl = run_dir / "raw" / "repeated_sampling.pkl"
    pkl.write_bytes(pkl.read_bytes() + b"x")
    assert any("altered" in p for p in GEN.verify_run(run_dir))
    pkl.unlink()
    assert any("missing" in p for p in GEN.verify_run(run_dir))
    assert not any("repeated_sampling" in p
                   for p in GEN.verify_run(run_dir, require_raw=False))  # a clone without raw
    (run_dir / "record.md").unlink()
    assert any("record" in p for p in GEN.verify_run(run_dir, require_raw=False))
    entry = tmp_path / "registry" / f"{_ledger(run_dir)['reservation']}.json"
    entry.write_text(entry.read_text() + " ")
    assert "reservation missing or altered" in GEN.verify_run(run_dir, require_raw=False)
    assert GEN.verify_run(tmp_path / "missing") == ["no ledger"]
    assert GEN.main(["--verify", str(run_dir)]) == 1


def test_raw_results_are_ignored_by_git() -> None:
    assert "docs/evidence/m11-p8-*/raw/" in (REPO / ".gitignore").read_text().splitlines()


# --- refusal before any reservation, and seeds ------------------------------------------------
def test_digest_mismatch_and_final_requirements_refuse_before_any_reservation(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # never depend on the live protocol state: a test must not be able to start a final run
    monkeypatch.setattr(pp, "PENDING_BEFORE_MAIN", ("test: final runs are never executed",))
    with pytest.raises(GEN.RunRefused, match="manifest digest"):
        _run(tmp_path, expect_manifest="0" * 64)
    assert not (tmp_path / "run").exists() and not (tmp_path / "registry").exists()
    fake_final = GEN.SeedPlan("final", 6000, 6001, True, {"R1": [6002], "R2": [6002],
                                                           "R3": [6002], "REFIT": [6003]},
                              (0.5, 1.0, 2.0), 2)
    with pytest.raises(GEN.RunRefused, match="--expect-manifest"):
        GEN.verify_protocol(fake_final, None, None)
    with pytest.raises(GEN.RunRefused, match="pending"):
        GEN.verify_protocol(fake_final, pp.protocol_digest(), pp.protocol_code_digest())
    with pytest.raises(pp.ProtocolNotReady):
        GEN.final_plan()
    assert GEN.main(["--mode", "final", "--run-dir", str(tmp_path / "run"),
                     "--registry", str(tmp_path / "registry"),
                     "--expect-manifest", pp.protocol_digest(),
                     "--expect-code", pp.protocol_code_digest()]) == 2
    assert not (tmp_path / "run").exists() and not (tmp_path / "registry").exists()
    assert GEN.reservation_key(fake_final) == pp.P8_FINAL_ATTEMPT_KEY == "final-a14"


def test_development_plans_never_touch_reserved_seeds() -> None:
    reserved = GEN.reserved_seeds()
    for plan in (GEN.dry_run_plan(), GEN.rehearsal_plan(), GEN.rehearsal_plan(10, 20)):
        seeds = {plan.main, *[s for v in plan.draws.values() for s in v]}
        if plan.m10_comparability is not None:
            seeds.add(plan.m10_comparability)
        if plan.fresh_step:
            seeds.add(GEN.resolve_fresh_seed(plan))
        assert not seeds & reserved and max(seeds) <= pp.DEVELOPMENT_SEED_MAX
    assert {pp.COVERAGE_SEED_START, pp.REFIT_STUDY_SEED_START} <= reserved
    assert GEN.reservation_key(GEN.rehearsal_plan(2, 2)) != GEN.reservation_key(
        GEN.rehearsal_plan(3, 2))


def test_final_plan_maps_studies_to_their_reserved_blocks(monkeypatch: pytest.MonkeyPatch
                                                          ) -> None:
    """Seed mapping only (nothing is generated): the gate is opened in this process alone."""
    monkeypatch.setattr(pp, "PENDING_BEFORE_MAIN", ())
    plan = GEN.final_plan()
    st = {s.id: s for s in pp.REPEATED_SAMPLING}
    for k in ("R1", "R2", "R3", "REFIT"):
        assert plan.draws[k] == list(range(st[k].seed_start, st[k].seed_start
                                           + st[k].replications))
    assert plan.refits == pp.CLUSTERED_INFERENCE.refit_bootstrap_resamples and plan.fresh_step
    assert len(GEN.study_tasks(plan)) == 1000 + 500 + 3 * 200 + 200


def test_interrupted_cli_exit_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupted(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(GEN, "execute", interrupted)
    assert GEN.main(["--mode", "rehearsal", "--run-dir", str(tmp_path / "cli"),
                     "--registry", str(tmp_path / "registry")]) == 130


# --- draw-level coverage rule -------------------------------------------------------------
def test_draw_level_coverage_rule() -> None:
    assert pc.classify_draw_coverage(0.70, 0.78, 0.8) == "below nominal"
    assert pc.classify_draw_coverage(0.82, 0.90, 0.8) == "above nominal"
    assert pc.classify_draw_coverage(0.75, 0.85, 0.8) == "consistent with nominal"
    assert pc.classify_draw_coverage(float("nan"), float("nan"), 0.8) == "insufficient draws"
    rng = np.random.default_rng(6170)
    draws: list[dict[str, Any]] = []
    for i in range(20):
        n = int(rng.integers(200, 400))
        k = int(rng.binomial(n, 0.75))
        draws.append({"seed": i, "coverage": [{"setting": "s", "model": "m", "interval": "i",
                                               "level": 0.8, "group": "all", "covered": k,
                                               "n": n}]})
    row = pc.aggregate_coverage(draws).iloc[0]
    per = np.array([d["coverage"][0]["covered"] / d["coverage"][0]["n"] for d in draws])
    se = per.std(ddof=1) / np.sqrt(per.size)
    q = stats.t.ppf(0.975, per.size - 1)
    assert row["mean coverage"] == pytest.approx(per.mean())
    assert (row["95% lower"], row["95% upper"]) == pytest.approx((per.mean() - q * se,
                                                                  per.mean() + q * se))
    assert row["classification"] == pc.classify_draw_coverage(row["95% lower"],
                                                              row["95% upper"], 0.8)
    pooled = sum(d["coverage"][0]["covered"] for d in draws) / sum(d["coverage"][0]["n"]
                                                                  for d in draws)
    assert row["pooled coverage (descriptive)"] == pytest.approx(pooled)


# --- final mode: clean tree, test-process guard, one attempt (fake draws only) -------------------
def _fake_final() -> Any:
    return GEN.SeedPlan("final", 6000, 6001, True, {"R1": [6002], "R2": [6002], "R3": [6002],
                                                    "REFIT": [6003]}, (0.5, 1.0, 2.0), 2)


def _final(tmp_path: Path, run: str = "final", fn: Any = None) -> Path:
    out: Path = GEN.execute(_fake_final(), tmp_path / run, 1, fn=fn or Recorder(),
                            report_fn=_stub_report, registry=tmp_path / "registry",
                            evidence_root=tmp_path, expect_manifest=pp.protocol_digest(),
                            expect_code=pp.protocol_code_digest())
    return out


def test_final_mode_requires_a_clean_working_tree(tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pp, "PENDING_BEFORE_MAIN", ())  # exercise the clean-tree guard
    monkeypatch.setattr(GEN, "git_dirty_paths", lambda repo=None: ["?? untracked.txt"])
    rec = Recorder()
    with pytest.raises(GEN.RunRefused, match="not clean"):
        _final(tmp_path, fn=rec)
    assert rec.calls == [] and not (tmp_path / "registry").exists()
    assert not (tmp_path / "final").exists()
    assert GEN.main(["--mode", "final", "--registry", str(tmp_path / "registry"),
                     "--expect-manifest", pp.protocol_digest(),
                     "--expect-code", pp.protocol_code_digest()]) == 2


def test_a_real_final_run_can_never_start_inside_the_tests(tmp_path: Path,
                                                           monkeypatch: pytest.MonkeyPatch
                                                           ) -> None:
    monkeypatch.setattr(GEN, "git_dirty_paths", lambda repo=None: [])
    with pytest.raises(GEN.RunRefused, match="under pytest"):
        GEN.execute(_fake_final(), tmp_path / "final", 1, fn=pc.run_task,
                    registry=tmp_path / "registry", evidence_root=tmp_path,
                    expect_manifest=pp.protocol_digest(), expect_code=pp.protocol_code_digest())
    assert not (tmp_path / "registry").exists() and not (tmp_path / "final").exists()
    assert GEN.main(["--mode", "final", "--registry", str(tmp_path / "registry"),
                     "--run-dir", str(tmp_path / "final"),
                     "--expect-manifest", pp.protocol_digest(),
                     "--expect-code", pp.protocol_code_digest()]) == 2
    assert not (tmp_path / "registry").exists() and not (tmp_path / "final").exists()


def test_the_final_sequence_runs_once_with_fake_draws(tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pp, "PENDING_BEFORE_MAIN", ())  # synthetic authorised state only
    monkeypatch.setattr(GEN, "git_dirty_paths", lambda repo=None: [])
    monkeypatch.setattr(GEN, "resolve_fresh_seed", lambda plan, **kw: 6004)  # development only
    monkeypatch.setattr(GEN, "a15_gate_applies", lambda: False)  # historical replay only
    rec = Recorder()
    _final(tmp_path, fn=rec)
    assert [t.study for t in rec.calls][0] == "MAIN" and rec.calls[-1].seed == 6004
    led = _ledger(tmp_path / "final")
    assert led["reservation"] == pp.P8_FINAL_ATTEMPT_KEY and led["status"] == "complete"
    assert GEN.verify_run(tmp_path / "final") == []
    for other in ("final", "another"):  # any directory: the prior final ledger refuses
        again = Recorder()
        with pytest.raises(GEN.RunRefused, match="already attempted"):
            _final(tmp_path, run=other, fn=again)
        assert again.calls == []
    (tmp_path / "final" / "ledger.json").unlink()  # even without the ledger: the registry
    with pytest.raises(GEN.RunRefused, match="already exists"):
        _final(tmp_path, run="third")


def test_a15_execution_hold_refuses_even_with_matching_digests(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(GEN, "git_dirty_paths", lambda repo=None: [])
    rec = Recorder()
    with pytest.raises(GEN.RunRefused, match="protocol items are pending"):
        _final(tmp_path, fn=rec)
    with pytest.raises(pp.ProtocolNotReady, match="pending"):
        GEN.final_plan()
    assert rec.calls == []
    assert not (tmp_path / "registry").exists() and not (tmp_path / "final").exists()


@pytest.fixture
def candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    """A simulated future approval in a temporary tree. All task seeds are development seeds."""
    root = tmp_path / "evidence"
    registry = root / GEN.REGISTRY_DIR.name
    reg = GEN.AttemptRegistry(registry)
    pins = []
    for key in ("final", "final-a14"):
        directory = root / f"m11-p8-{key}"
        directory.mkdir(parents=True)
        meta = {"mode": "final", "manifest_digest": "fixture", "code_digest": "fixture"}
        sha = reg.reserve(key, {**meta, "run_dir": str(directory)})
        ledger = directory / "ledger.json"
        ledger.write_text(json.dumps({**meta, "status": "failed", "phase": "repeated_sampling",
                                      "reservation": key, "reservation_sha256": sha}))
        pins.append((ledger.relative_to(root).as_posix(),
                     hashlib.sha256(ledger.read_bytes()).hexdigest()))
    monkeypatch.setattr(pp, "P8_FAILED_ATTEMPTS", tuple(pins))
    monkeypatch.setattr(pp, "P8_FINAL_ATTEMPT_KEY", "final-a15")
    monkeypatch.setattr(pp, "PENDING_BEFORE_MAIN", ())
    monkeypatch.setattr(GEN, "git_dirty_paths", lambda repo=None: [])
    monkeypatch.setattr(pp, "authorise_seed",
                        lambda *a, **kw: pytest.fail("unexpected seed access"))
    return root, registry, root / "m11-p8-final-a15"


def _candidate(paths: tuple[Path, Path, Path], rec: Recorder) -> Path:
    root, registry, run = paths
    out: Path = GEN.execute(_fake_final(), run, 1, fn=rec, report_fn=_stub_report,
                            registry=registry, evidence_root=root,
                            expect_manifest=pp.protocol_digest(),
                            expect_code=pp.protocol_code_digest())
    return out


@pytest.mark.parametrize("damage, message", [
    ("missing", "missing or redirected"),
    ("altered", "ledger altered"),
    ("wrong_state", "failure or consumed reservation does not match"),
    ("wrong_reservation", "failure or consumed reservation does not match"),
    ("duplicate", "unexpected or duplicate"),
    ("redirected", "redirected directory"),
    ("unregistered", "unexpected or duplicate"),
    ("missing_reservation", "entry deleted"),
    ("unexpected_reservation", "unexpected final reservation"),
    ("duplicate_chain", "duplicate chain entry"),
])
def test_a15_refuses_incomplete_or_unexpected_history(
        candidate: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch,
        damage: str, message: str) -> None:
    root, registry, run = candidate
    ledger = root / "m11-p8-final-a14" / "ledger.json"
    if damage == "missing":
        ledger.unlink()
    elif damage == "altered":
        ledger.write_bytes(ledger.read_bytes() + b" ")
    elif damage in {"wrong_state", "wrong_reservation"}:
        record = json.loads(ledger.read_bytes())
        field, value = ("status", "complete") if damage == "wrong_state" else (
            "reservation_sha256", "0" * 64)
        record[field] = value
        ledger.write_text(json.dumps(record))
        pins = dict(pp.P8_FAILED_ATTEMPTS)
        pins[ledger.relative_to(root).as_posix()] = hashlib.sha256(ledger.read_bytes()).hexdigest()
        monkeypatch.setattr(pp, "P8_FAILED_ATTEMPTS", tuple(pins.items()))
    elif damage in {"duplicate", "unregistered"}:
        extra = root / "other" / "ledger.json"
        extra.parent.mkdir()
        extra.write_bytes(ledger.read_bytes() if damage == "duplicate" else b'{"mode":"final"}')
    elif damage == "redirected":
        (root / "copy").symlink_to(ledger.parent, target_is_directory=True)
    elif damage == "missing_reservation":
        (registry / "final-a14.json").unlink()
    elif damage == "unexpected_reservation":
        GEN.AttemptRegistry(registry).reserve("other", {"mode": "final"})
    else:
        path = registry / "attempts.log"
        last = json.loads(path.read_text().splitlines()[-1])
        last["prev"] = last["hash"]
        last["hash"] = GEN.AttemptRegistry._chain(
            last["prev"], last["key"], last["entry_sha256"], last["time"])
        with path.open("a") as out:
            out.write(json.dumps(last) + "\n")
    rec = Recorder()
    with pytest.raises(GEN.RunRefused, match=message):
        _candidate(candidate, rec)
    assert rec.calls == [] and not run.exists() and not (registry / "final-a15.json").exists()


@pytest.mark.parametrize("key", ("final", "final-a14"))
def test_a15_refuses_historical_keys_and_output_directories(
        candidate: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch, key: str) -> None:
    root, registry, run = candidate
    rec = Recorder()
    with pytest.raises(GEN.RunRefused, match="canonical registry and final-a15 output"):
        _candidate((root, registry, root / f"m11-p8-{key}"), rec)
    with pytest.raises(GEN.RunRefused, match="canonical registry and final-a15 output"):
        _candidate((root, root / "alternate-registry", run), rec)
    # "final" is already invalid in validate_protocol. Both historical keys must be refused.
    monkeypatch.setattr(pp, "P8_FINAL_ATTEMPT_KEY", key)
    with pytest.raises(GEN.RunRefused, match="new final-a15 reservation key|protocol invalid"):
        _candidate(candidate, rec)
    assert rec.calls == [] and not run.exists() and not (registry / "final-a15.json").exists()
    assert not (root / "alternate-registry").exists()


def test_a15_refuses_existing_output_without_a_ledger(candidate: tuple[Path, Path, Path]) -> None:
    _, registry, run = candidate
    run.mkdir()
    rec = Recorder()
    with pytest.raises(GEN.RunRefused, match="output directory already exists"):
        _candidate(candidate, rec)
    assert rec.calls == [] and not (registry / "final-a15.json").exists()


def test_a15_concurrent_candidates_dispatch_once_and_authorise_fresh_last(
        candidate: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    _, registry, run = candidate
    recs = [Recorder(), Recorder()]
    barrier = threading.Barrier(2)
    history = []

    def fresh(purpose: str, *, confirm_single_final_run: bool = False) -> int:
        assert purpose == "fresh" and confirm_single_final_run
        assert (registry / "final-a15.json").is_file()
        history.extend((e["phase"], e["state"]) for e in _ledger(run)["events"])
        return 6004

    def contender(rec: Recorder) -> str:
        barrier.wait(timeout=10)
        try:
            _candidate(candidate, rec)
        except GEN.RunRefused:
            return "refused"
        return "completed"

    monkeypatch.setattr(pp, "authorise_seed", fresh)
    with pytest.raises(GEN.RunRefused, match="after M10 comparability"):
        GEN.resolve_fresh_seed(_fake_final())
    early = GEN.Ledger(run / "ledger.json",
                       {"status": "started", "reservation": "final-a15",
                        "events": [{"phase": "main", "state": "done"}]})
    with pytest.raises(GEN.RunRefused, match="after M10 comparability"):
        GEN.resolve_fresh_seed(_fake_final(), ledger=early)
    with ThreadPoolExecutor(2) as pool:
        assert sorted(pool.map(contender, recs)) == ["completed", "refused"]
    assert GEN.AttemptRegistry(registry).validate() == []
    winner, loser = sorted(recs, key=lambda rec: len(rec.calls), reverse=True)
    assert loser.calls == [] and winner.calls[-1].seed == 6004
    assert [call.study for call in winner.calls[-2:]] == ["M10", "MAIN"]
    assert history[-1] == ("m10_comparability", "done") and len(history) == 7
    led = _ledger(run)
    assert sum(e["state"] == "seed consumed" for e in led["events"]) == 1
    with pytest.raises(GEN.RunRefused, match="after M10 comparability"):
        GEN.resolve_fresh_seed(_fake_final(), ledger=GEN.Ledger(run / "ledger.json", led))


@pytest.mark.parametrize("exc,status", [
    (RuntimeError, "failed"), (KeyboardInterrupt, "interrupted")])
def test_a15_failure_consumes_key_even_if_output_is_moved(
        candidate: tuple[Path, Path, Path], tmp_path: Path,
        exc: type[BaseException], status: str) -> None:
    _, registry, run = candidate
    with pytest.raises(exc, match="M10 failed"):
        _candidate(candidate, Recorder("M10", exc))
    led = _ledger(run)
    assert led["status"] == status and led["phase"] == "m10_comparability"
    assert all(e["phase"] != "fresh" for e in led["events"])
    reservation = (registry / "final-a15.json").read_bytes()
    run.rename(tmp_path / "moved")  # synthetic fixture only
    rec = Recorder()
    with pytest.raises(GEN.RunRefused, match="unexpected final reservation: final-a15"):
        _candidate(candidate, rec)
    assert rec.calls == [] and not run.exists()
    assert (registry / "final-a15.json").read_bytes() == reservation
