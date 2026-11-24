"""Development-only regression checks for hybrid scheduling and successor bindings."""

import hashlib
import json
import os
import pickle
import subprocess
from concurrent.futures import BrokenExecutor
from contextlib import closing
from types import SimpleNamespace

import pytest
from scripts import generate_m11_evidence as runner

from decision_intelligence.customer_value import (
    launch_approval,
    p8_candidate,
    p8_immutable,
    p8_private,
    p8_scheduler,
)
from decision_intelligence.customer_value.predictive_cases import Task


def _draw(task):
    return {"task": task, "pid": os.getpid()}


def _lose_worker(task):
    if task.seed == 6202:
        os._exit(41)
    return _draw(task)


def test_hybrid_order_isolation_and_legacy_raw_round_trip(tmp_path):
    tasks = [Task("R1", 6200), *(Task("R2", n) for n in range(6201, 6205)), Task("R3", 6200)]
    main = p8_scheduler.isolated_task(Task("MAIN", 6200), _draw)
    assert len(main) == 1 and main[0]["pid"] != os.getpid()
    repeated = runner._hybrid_results(tmp_path, tasks, _draw)
    assert [result["task"] for result in repeated] == tasks
    assert all(row["pid"] != main[0]["pid"] for row in repeated)
    assert {row["pid"] for row in repeated[1:3]}.isdisjoint(
        row["pid"] for row in repeated[3:5])
    ledger = runner.Ledger.create(tmp_path, {})
    runner._store(tmp_path, ledger, "repeated_sampling", repeated)
    path = tmp_path / "raw" / "repeated_sampling.pkl"
    assert pickle.loads(path.read_bytes()) == repeated
    assert path.read_bytes() == pickle.dumps(repeated, protocol=pickle.HIGHEST_PROTOCOL)
    event = ledger.data["events"][-1]
    assert event["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_bytes(path.read_bytes() + b"altered")
    assert runner._check_file(tmp_path, event["file"], event["sha256"], "draws", True)


def test_worker_loss_aborts_ordered_stream(tmp_path):
    tasks = [Task("R2", seed) for seed in range(6201, 6205)]
    with pytest.raises(BrokenExecutor):
        runner._hybrid_results(tmp_path, tasks, _lose_worker)
    assert list(tmp_path.iterdir()) == []


def test_pool_bounds_submission_and_cancels_on_early_close(monkeypatch):
    tasks = (Task("R1", seed) for seed in range(6200, 6220))
    submitted = []
    retired = []
    outstanding = 0

    def submit(fn, task):
        nonlocal outstanding
        outstanding += 1
        assert outstanding <= 6
        submitted.append(task)

        def result():
            nonlocal outstanding
            outstanding -= 1
            return fn(task)

        return SimpleNamespace(result=result)

    def executor(*, max_workers, mp_context):
        assert max_workers == 3 and mp_context.get_start_method() == "spawn"
        return SimpleNamespace(submit=submit, shutdown=lambda **kw: retired.append(kw))

    monkeypatch.setattr(p8_scheduler, "ProcessPoolExecutor", executor)
    with closing(p8_scheduler.ordered_pool(tasks, 3, _draw)) as rows:
        assert next(rows)["task"].seed == 6200 and len(submitted) == 7
    assert retired == [{"wait": True, "cancel_futures": True}]


def test_interrupted_consumer_closes_active_pool_and_does_not_open_next(monkeypatch):
    events = []

    def pool(tasks, workers, fn):
        batch = list(tasks)
        events.append(("open", workers, len(batch)))
        try:
            yield from map(fn, batch)
        finally:
            events.append(("close",))

    monkeypatch.setattr(p8_scheduler, "ordered_pool", pool)
    tasks = [Task("R2", seed) for seed in range(6201, 6205)]
    with pytest.raises(KeyboardInterrupt), closing(p8_scheduler.hybrid_tasks(tasks, _draw)) as rows:
        assert next(rows)["task"] == tasks[0]
        raise KeyboardInterrupt
    assert events == [("open", 2, 2), ("close",)]


def test_partial_pickle_writes_never_publish_a_result(tmp_path, monkeypatch):
    ledger = runner.Ledger.create(tmp_path, {})

    def failed_dump(obj, stream, **kwargs):
        stream.write(b"partial")
        raise OSError("fixture write failure")

    monkeypatch.setattr(runner.pickle, "dump", failed_dump)
    with pytest.raises(OSError, match="fixture write failure"):
        runner._store(tmp_path, ledger, "repeated_sampling", [])
    assert ledger.data["events"] == []
    assert list((tmp_path / "raw").iterdir()) == []


@pytest.mark.parametrize("attempt", ["final", "final-a14", "final-a15"])
def test_all_consumed_identities_refuse(attempt):
    with pytest.raises(runner.RunRefused, match="permanently consumed"):
        runner.require_successor_identity(attempt)


def test_successor_history_requires_the_incomplete_a15_metadata(p8_layout, monkeypatch):
    reg = runner.AttemptRegistry(p8_layout.history_registry)
    pins = {}
    for key in p8_candidate.HISTORICAL_PINS:
        directory = p8_layout.history / f"m11-p8-{key}"
        directory.mkdir(mode=0o700)
        header = {"mode": "final", "manifest_digest": "fixture", "code_digest": "fixture"}
        entry_sha = reg.reserve(key, {**header, "run_dir": str(directory)})
        state = ({"status": "started", "events": [
            {"phase": "repeated_sampling", "state": "started"}]}
            if key == "final-a15" else {"status": "failed", "phase": "repeated_sampling"})
        path = directory / "ledger.json"
        path.write_text(json.dumps({**header, **state, "reservation": key,
                                    "reservation_sha256": entry_sha}))
        pins[key] = (runner._sha256(path), entry_sha)
    monkeypatch.setattr(p8_candidate, "HISTORICAL_PINS", pins)
    monkeypatch.setattr(p8_candidate, "HISTORY_CHAIN_SHA256", runner._sha256(reg.log))
    chain = reg.log.read_bytes().splitlines(keepends=True)
    prefix = hashlib.sha256(b"".join(chain[:-1])).hexdigest()
    monkeypatch.setattr(p8_candidate, "HISTORY_PREFIX_SHA256", prefix)
    manifest = p8_layout.history / "manifest.json"
    record = {
        "schema": 1, "chain_sha256": runner._sha256(reg.log), "chain_prefix_sha256": prefix,
        "chain_tip": json.loads(chain[-1])["hash"],
        "attempts": {key: {"ledger_sha256": ledger, "reservation_sha256": entry}
                     for key, (ledger, entry) in pins.items()},
    }
    manifest.write_text(json.dumps(record))
    pin = runner._sha256(manifest)
    monkeypatch.setattr(p8_private, "_owner_pin", lambda name: pin)
    with monkeypatch.context() as probe:
        probe.setattr(runner.subprocess, "run",
                      lambda *a, **kw: pytest.fail("private history consulted Git"))
        assert runner.verify_successor_history(p8_layout) == pin
    manifest.write_text(json.dumps({**record, "chain_tip": "0" * 64}))
    with pytest.raises(p8_private.PrivateEvidenceError, match="owner pin"):
        runner.verify_successor_history(p8_layout)
    with monkeypatch.context() as probe:
        wrong_pin = runner._sha256(manifest)
        probe.setattr(p8_private, "_owner_pin", lambda name: wrong_pin)
        with pytest.raises(runner.RunRefused, match="owner manifest does not match"):
            runner.verify_successor_history(p8_layout)
    manifest.write_text(json.dumps(record))
    a15 = p8_layout.history / "m11-p8-final-a15" / "ledger.json"
    a15.write_bytes(a15.read_bytes() + b" ")
    with pytest.raises(runner.RunRefused, match="historical ledger altered"):
        runner.verify_successor_history(p8_layout)


def test_replayed_signature_and_approval_payload_refuse(p8_layout, tmp_path, monkeypatch):
    monkeypatch.setattr(p8_candidate.sys, "platform", "darwin")
    monkeypatch.setattr(launch_approval, "_APPROVAL_DIR", tmp_path)
    monkeypatch.setattr(launch_approval, "_trusted_path", lambda *a, **kw: None)
    attempted = "final-fixture-new"
    repo = tmp_path / "code"
    repo.mkdir()
    private = p8_private.open_private_evidence(repo)
    monkeypatch.setattr(p8_private.PrivateEvidence, "history_manifest",
                        lambda self: ({}, "fixture-history"))
    approval = tmp_path / f"{attempted}.json"
    approval.write_text(json.dumps(
        {"attempt": "final-a15", "ci_run_id": "123", "host_preflight_passed": True}))

    def wrong_namespace(argv, **kwargs):
        assert argv[argv.index("-n") + 1] == f"m11-p8-{attempted}"
        assert argv[argv.index("-s") + 1].endswith(f"{attempted}.json.sig")
        raise subprocess.CalledProcessError(1, argv)

    def verify():
        p8_candidate.require_owner_approval(
            repo, attempt=attempted, manifest="fixture", code="fixture", require_clean=True)

    with monkeypatch.context() as probe:
        probe.setattr(p8_candidate.subprocess, "run", wrong_namespace)
        with pytest.raises(launch_approval.LaunchApprovalError, match="invalid"):
            verify()
    monkeypatch.setattr(p8_candidate, "_verify_signature", lambda *a: None)
    monkeypatch.setattr(launch_approval, "_live_binding", lambda *a, **kw: {"attempt": attempted})
    monkeypatch.setattr(p8_immutable, "binding", lambda *a: {})
    source = repo / "src" / "decision_intelligence" / "customer_value"
    source.mkdir(parents=True)
    modules = ("p8_scheduler", "p8_candidate", "p8_private", "p8_journal", "p8_immutable")
    for name in modules:
        (source / f"{name}.py").write_text("fixture")
    with pytest.raises(launch_approval.LaunchApprovalError, match="release binding"):
        verify()

    record = {"attempt": attempted, "ci_run_id": "123", "host_preflight_passed": True,
              "registry": str(private.registry), "output": str(private.run_dir(attempted)),
              "history_sha256": "fixture-history", "worker_uid": str(os.geteuid()),
              "worker_gid": str(private.root.stat().st_gid)}
    record.update({f"{name}_sha256": runner._sha256(source / f"{name}.py")
                   for name in modules})
    approval.write_text(json.dumps(record))
    verify()
    (source / "p8_scheduler.py").write_text("altered implementation")
    with pytest.raises(launch_approval.LaunchApprovalError, match="release binding"):
        verify()
