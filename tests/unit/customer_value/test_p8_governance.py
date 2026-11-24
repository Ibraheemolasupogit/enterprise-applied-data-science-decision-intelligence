"""Native supervision and approval tests on development data and ephemeral fixture keys."""

import json
import os
import re
import runpy
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import generate_m11_evidence as runner
from scripts import prepare_m11_p8_a19 as renderer
from scripts import prepare_m11_p8_a20 as adoption

from decision_intelligence.customer_value import (
    launch_approval,
    p8_candidate,
    p8_journal,
)
from decision_intelligence.customer_value import (
    predictive_protocol as pp,
)

_WORKER = """
import errno
import json
import os
import sys
from pathlib import Path

from scripts import generate_m11_evidence as runner


def draw(task):
    return {"study": task.study, "seed": task.seed, "scale": task.scale}


def report(plan, results, perf, root, output):
    path = root / "fixture.md"
    path.write_text("development only")
    return [path]


if __name__ == "__main__":
    resources = runner.await_supervisor()
    try:
        os.pwrite(resources.fd, b"x", 0)
    except OSError as exc:
        assert exc.errno == errno.EBADF
    else:
        raise AssertionError("worker can change the supervisor's RSS sample")
    root, journal = map(Path, sys.argv[1:3])
    claim = json.loads(journal.read_text().splitlines()[0])
    registry = root.parent / "registry"
    plan = runner.rehearsal_plan(3, 2)
    ledger = runner._reserve(plan, root, 3, registry, claim["attempt"], claim=claim)
    runner._execute_phases(plan, root, 3, ledger, draw, report, None, resources)
    if sys.argv[3] == "duplicate-fresh":
        event = next(e for e in ledger.data["events"] if e["state"] == "seed consumed")
        ledger.data["events"].insert(-1, event)
        ledger.save()
"""


@pytest.mark.skipif(sys.platform != "darwin", reason="requires Darwin native resource accounting")
@pytest.mark.parametrize("scenario", [
    "complete", "duplicate-fresh", "memory_limit", "time_limit", "late-terminal", "rss-terminal",
])
def test_native_successor_receipt(tmp_path, monkeypatch, scenario):
    script = tmp_path / "fixture.py"
    script.write_text(_WORKER)
    journal_path = tmp_path / "journal"
    ledger_root = tmp_path / "run"
    # The development subprocess can read its claim but this test never uses the installed journal.
    fd = os.open(journal_path, os.O_CREAT | os.O_EXCL | os.O_APPEND | os.O_RDWR, 0o600)
    try:
        journal = p8_journal.Journal(fd, full_sync=True)
        if scenario in ("late-terminal", "rss-terminal"):
            append = journal.append

            def after_terminal(attempt, state, payload):
                digest = append(attempt, state, payload)
                if state == "terminal" and scenario == "late-terminal":
                    monkeypatch.setattr(
                        runner, "time", SimpleNamespace(monotonic=lambda: time.monotonic() + 100))
                if state == "terminal" and scenario == "rss-terminal":
                    monkeypatch.setattr(runner.resource_supervision, "DarwinRSS",
                                        lambda: SimpleNamespace(_rss=lambda _: 4_000_000_001))
                return digest

            monkeypatch.setattr(journal, "append", after_terminal)

        @contextmanager
        def read_fixture(_path):
            yield journal

        monkeypatch.setattr(runner.p8_journal, "open_journal", read_fixture)
        repo = Path(runner.__file__).resolve().parents[1]
        env = os.pathsep.join((str(repo), str(repo / "src"),
                               os.environ.get("PYTHONPATH", "")))
        monkeypatch.setenv("PYTHONPATH", env)
        command = [sys.executable, "-B", str(script), str(ledger_root), str(journal_path), scenario]
        timeout = 0.1 if scenario == "time_limit" else 90
        memory = 1 if scenario == "memory_limit" else 4_000_000_000
        limits = {"timeout_s": timeout, "memory_bytes": memory, "approval_sha256": "dev-approval"}
        if scenario == "duplicate-fresh":
            with pytest.raises(runner.RunRefused, match="completion"):
                runner._consume_and_supervise(journal, "fixture", command, ledger_root,
                                              "dev-pin", **limits)
        else:
            rc = runner._consume_and_supervise(journal, "fixture", command, ledger_root,
                                               "dev-pin", **limits)
            assert rc == (0 if scenario == "complete" else 1)
        consumed, terminal, *decision = journal.records()
        assert terminal["payload"]["claim_sha256"] == consumed["sha256"]
        expected = {"duplicate-fresh": "certification_failure", "complete": "supervision_complete",
                    "late-terminal": "supervision_complete",
                    "rss-terminal": "supervision_complete"}.get(scenario, scenario)
        assert terminal["payload"]["status"] == expected
        if scenario == "complete":
            assert len(decision) == 1 and decision[0]["state"] == "certified"
            assert decision[0]["payload"]["terminal_sha256"] == terminal["sha256"]
            assert runner.verify_run(ledger_root) == []
            path = ledger_root / "ledger.json"
            data = json.loads(path.read_text())
            data["events"][-1]["files"][0]["sha256"] = "0" * 64
            path.write_text(json.dumps(data))
            assert "protected supervisor receipt missing or mismatched" in runner.verify_run(
                ledger_root)
        else:
            assert not decision
        if scenario in ("late-terminal", "rss-terminal"):
            assert "protected supervisor receipt missing or mismatched" in runner.verify_run(
                ledger_root)
        with pytest.raises(p8_journal.JournalError, match="permanently consumed"):
            runner._consume_and_supervise(journal, "fixture", ["must-not-dispatch"],
                                          tmp_path / "different", "dev-pin", **limits)
    finally:
        os.close(fd)


def test_truncated_journal_fails_closed(tmp_path):
    path = tmp_path / "journal"
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_APPEND | os.O_RDWR, 0o600)
    try:
        journal = p8_journal.Journal(fd)
        claim = journal.append("fixture", "consumed", {})
        with pytest.raises(p8_journal.JournalError, match="does not match"):
            journal.append("fixture", "terminal", {"claim_sha256": "incorrect"})
        with pytest.raises(p8_journal.JournalError, match="does not match"):
            journal.append("fixture", "certified", {"terminal_sha256": claim})
        assert journal.records()[0]["sha256"] == claim
        os.write(fd, b"{")
        with pytest.raises(p8_journal.JournalError, match="partial"):
            journal.append("fixture-other", "consumed", {})
    finally:
        os.close(fd)


def test_a20_identity_alone_does_not_activate_successor(monkeypatch):
    assert p8_candidate.A20_ATTEMPT == "final-a20"
    monkeypatch.setattr(pp, "PROTOCOL_AMENDMENTS", pp.PROTOCOL_AMENDMENTS[:19])
    monkeypatch.setattr(pp, "AMENDMENTS_AFTER_MAIN_SEED_COMPUTATION",
                        pp.AMENDMENTS_AFTER_MAIN_SEED_COMPUTATION[:6])
    monkeypatch.setattr(pp, "P8_FINAL_ATTEMPT_KEY", p8_candidate.A20_ATTEMPT)
    with pytest.raises(pp.ProtocolNotReady, match="has not been adopted"):
        pp.require_successor_amendment()
    draft = replace(pp.PROTOCOL_AMENDMENTS[-1], id="A20", before_outcome_inspection=None)
    monkeypatch.setattr(pp, "PROTOCOL_AMENDMENTS", (*pp.PROTOCOL_AMENDMENTS, draft))
    monkeypatch.setattr(pp, "AMENDMENTS_AFTER_MAIN_SEED_COMPUTATION",
                        (*pp.AMENDMENTS_AFTER_MAIN_SEED_COMPUTATION, "A20"))
    with pytest.raises(pp.ProtocolNotReady, match="has not been adopted"):
        pp.require_successor_amendment()
    adopted = replace(draft, before_outcome_inspection=True)
    monkeypatch.setattr(pp, "PROTOCOL_AMENDMENTS", (*pp.PROTOCOL_AMENDMENTS[:-1], adopted))
    pp.require_successor_amendment()


def test_a20_adoption_patch_preserves_frozen_science_and_projects_both_pins(tmp_path):
    runbook = renderer.REPO / "docs/milestones/m11-p8-a20-owner-runbook.md"
    text = runbook.read_text()
    assert f"**Change.** {adoption.CHANGE}" in text
    assert f"**Reason.** {adoption.REASON}" in text
    if pp.P8_FINAL_ATTEMPT_KEY == "final-a20":
        assert pp.PROTOCOL_AMENDMENTS[-1].change == adoption.CHANGE
        assert pp.PROTOCOL_AMENDMENTS[-1].reason == adoption.REASON
        with pytest.raises(ValueError, match="held A19"):
            adoption.prepare_patch("2099-12-31")
        return

    with pytest.raises(ValueError, match="decision date"):
        adoption.prepare_patch("2020-01-01")
    with pytest.raises(pp.ProtocolNotReady, match="permanently consumed"):
        pp.require_successor_amendment()
    paths = [renderer.PROTOCOL_PATH.with_name(name) for name in pp.PROTOCOL_CODE_FILES]
    for path in (*paths, renderer.TEST_PATH):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(renderer.REPO / path, tmp_path / path)
    patch = adoption.prepare_patch("2099-12-31")
    subprocess.run(["git", "apply"], cwd=tmp_path, input=patch, text=True, check=True)
    fixture = runpy.run_path(str(tmp_path / renderer.PROTOCOL_PATH))
    assert fixture["validate_protocol"]() == []
    fixture["require_successor_amendment"]()
    before, after = pp.protocol_manifest(), fixture["protocol_manifest"]()
    assert {key for key in before.keys() | after.keys() if before[key] != after[key]} == {
        "P8_FINAL_ATTEMPT_KEY", "AMENDMENTS_AFTER_MAIN_SEED_COMPUTATION", "PROTOCOL_AMENDMENTS",
    }
    assert after["PROTOCOL_AMENDMENTS"][:-1] == before["PROTOCOL_AMENDMENTS"]
    assert after["P8_FINAL_ATTEMPT_KEY"] == "final-a20"
    pins = (tmp_path / renderer.TEST_PATH).read_text()
    for constant, digest in (("FROZEN_DIGEST", "protocol_digest"),
                             ("CANDIDATE_CODE_DIGEST", "protocol_code_digest")):
        assert re.search(rf'^{constant} = "{fixture[digest]()}"$', pins, re.MULTILINE)


def test_adopted_a20_still_requires_separate_signed_approval(tmp_path, monkeypatch):
    if pp.P8_FINAL_ATTEMPT_KEY != "final-a20":
        record = replace(pp.PROTOCOL_AMENDMENTS[-1], id="A20")
        monkeypatch.setattr(pp, "P8_FINAL_ATTEMPT_KEY", "final-a20")
        monkeypatch.setattr(pp, "PROTOCOL_AMENDMENTS", (*pp.PROTOCOL_AMENDMENTS, record))
        monkeypatch.setattr(pp, "AMENDMENTS_AFTER_MAIN_SEED_COMPUTATION",
                            (*pp.AMENDMENTS_AFTER_MAIN_SEED_COMPUTATION, "A20"))
    pp.require_successor_amendment()
    monkeypatch.setattr(p8_candidate.sys, "platform", "darwin")
    monkeypatch.setattr(launch_approval, "_APPROVAL_DIR", tmp_path)
    monkeypatch.setattr(launch_approval, "_trusted_path", lambda *args, **kwargs: None)
    monkeypatch.setattr(p8_candidate, "require_active_claim",
                        lambda *args, **kwargs: pytest.fail("claim reached without approval"))
    with pytest.raises(pp.ProtocolNotReady, match="owner approval is missing"):
        pp.require_launch_approval()
    assert list(tmp_path.iterdir()) == []


def test_a20_pins_a19_startup_failure_and_refuses_a_used_successor(tmp_path, monkeypatch):
    attempt = p8_candidate.A19_ATTEMPT
    private = SimpleNamespace(history_manifest=lambda: ({}, "history-pin"),
                              run_dir=lambda key: tmp_path / key)
    rows = [
        {"attempt": attempt, "state": "consumed", "payload": {
            "manifest_digest": p8_candidate._A19_MANIFEST,
            "code_digest": p8_candidate._A19_CODE,
            "approval_sha256": "a" * 64, "history_manifest_sha256": "history-pin",
            "run_dir": str(private.run_dir(attempt)),
        }},
        {"attempt": attempt, "state": "terminal", "sha256": "b" * 64,
         "payload": {"status": "child_failure", "returncode": 2}},
    ]

    @contextmanager
    def read_only(_path):
        yield SimpleNamespace(records=lambda: rows)

    monkeypatch.setattr(p8_journal, "open_journal", read_only)
    assert p8_candidate.verify_a19_consumption(private, require_unused=True) == "b" * 64
    rows[1]["payload"]["returncode"] = 1
    with pytest.raises(launch_approval.LaunchApprovalError, match="startup-failure"):
        p8_candidate.verify_a19_consumption(private)
    rows[1]["payload"]["returncode"] = 2
    rows.append({"attempt": p8_candidate.A20_ATTEMPT, "state": "consumed"})
    with pytest.raises(launch_approval.LaunchApprovalError, match="already consumed"):
        p8_candidate.verify_a19_consumption(private, require_unused=True)
    assert p8_candidate.verify_a19_consumption(private) == "b" * 64


_SIGNED_PROBE = """
import os, sys
from pathlib import Path
from types import SimpleNamespace
from scripts import generate_m11_evidence as runner

root, scenario = Path(sys.argv[1]), sys.argv[2]
attempt = "final-fixture-preflight"
runner.REPO = root
runner.pp.require_successor_amendment = lambda: None
runner.final_attempt_key = lambda: attempt
runner.p8_private.open_private_evidence = lambda repo: SimpleNamespace(
    root=SimpleNamespace(lstat=lambda: SimpleNamespace(
        st_uid=os.geteuid(), st_gid=os.getegid() + (scenario == "identity"))),
    run_dir=lambda key: root / key)
runner._require_private_available = lambda layout, path: None
runner.verify_successor_history = lambda private: "fixture-history"
runner.p8_candidate.verify_a19_consumption = lambda *a, **kw: "fixture-tip"
runner.launch_approval._APPROVAL_DIR = root
runner.launch_approval._trusted_path = lambda *a, **kw: None
runner.p8_candidate.release_binding = lambda *a, **kw: {"attempt": attempt}

def verify_git_index(repo):
    if scenario == "index":
        raise PermissionError("private fixture error")

def forbidden(*args, **kwargs):
    os._exit(97)

runner.p8_immutable.verify_git_index = verify_git_index
runner.p8_journal.open_journal = forbidden
runner.p8_candidate.require_active_claim = forbidden
runner.final_plan = forbidden
runner._reserve = forbidden
raise SystemExit(runner._preflight_child())
"""


@pytest.mark.skipif(sys.platform != "darwin" or not shutil.which("ssh-keygen"),
                    reason="requires native Darwin supervision and OpenSSH")
def test_native_signed_pre_dispatch_is_read_only(tmp_path, monkeypatch):
    root = Path(runner.__file__).resolve().parents[1]
    extra = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(root), str(root / "src"), extra)))
    key = tmp_path / "probe-fixture.key"
    subprocess.run(["/usr/bin/ssh-keygen", "-t", "ed25519", "-N", "", "-q", "-f", str(key)],
                   check=True)
    principal = launch_approval._PRINCIPAL
    (tmp_path / "allowed_signers").write_text(
        f"{principal} {Path(str(key) + '.pub').read_text().strip()}\n")
    record = tmp_path / "final-fixture-preflight.json"
    record.write_text(json.dumps({"attempt": "final-fixture-preflight", "ci_run_id": "123",
                                  "host_preflight_passed": True}))
    subprocess.run(["/usr/bin/ssh-keygen", "-Y", "sign", "-f", str(key),
                    "-n", "m11-p8-final-fixture-preflight", str(record)],
                   check=True, capture_output=True)
    script = tmp_path / "worker.py"
    script.write_text(_SIGNED_PROBE)
    before = set(tmp_path.iterdir())

    def run(scenario):
        return runner.supervise([sys.executable, "-B", str(script), str(tmp_path), scenario],
                                timeout_s=25, memory_bytes=4_000_000_000)

    expected = {"complete": ("complete", 0), "index": ("child_failure", 23),
                "identity": ("child_failure", 22)}
    for scenario, pair in expected.items():
        result = run(scenario)
        assert (result.status, result.returncode) == pair
    record.write_bytes(record.read_bytes() + b" ")
    result = run("signature")
    assert (result.status, result.returncode) == ("child_failure", 25)
    assert set(tmp_path.iterdir()) == before


def test_successor_cannot_hide_its_missing_receipt_by_changing_staging_mode(tmp_path):
    run = tmp_path / "m11-p8-final-fixture"
    run.mkdir()
    plan = runner.dry_run_plan(1, 1)
    events = [{"phase": phase, "state": state} for phase, state in (
        ("verify", "done"), ("main", "started"), ("main", "done"),
        ("repeated_sampling", "started"), ("repeated_sampling", "done"), ("record", "done"))]
    (run / "ledger.json").write_text(json.dumps({
        "mode": "dry-run", "plan": runner.protocol_header(plan, 1)["plan"],
        "status": "complete", "events": events,
    }))
    assert runner.verify_run(run) == ["protected supervisor receipt missing or mismatched"]


@pytest.mark.skipif(not shutil.which("ssh-keygen"), reason="requires OpenSSH")
def test_real_fixture_signature_rejects_payload_and_namespace_replay(tmp_path):
    key = tmp_path / "fixture"
    command = ["/usr/bin/ssh-keygen"]
    subprocess.run([*command, "-t", "ed25519", "-N", "", "-q", "-f", str(key)], check=True)
    public_key = key.with_suffix(".pub").read_text().strip()
    (tmp_path / "allowed_signers").write_text(f"{launch_approval._PRINCIPAL} {public_key}\n")
    original = "final-fixture-one"
    payload = tmp_path / f"{original}.json"
    payload.write_bytes(b'{"purpose":"development fixture"}\n')
    subprocess.run([*command, "-Y", "sign", "-f", str(key), "-n", f"m11-p8-{original}",
                    str(payload)], check=True, capture_output=True)
    p8_candidate._verify_signature(tmp_path, original, payload.read_bytes())
    with pytest.raises(subprocess.CalledProcessError):
        p8_candidate._verify_signature(tmp_path, original, payload.read_bytes() + b" ")
    replay = p8_candidate.A20_ATTEMPT
    shutil.copyfile(tmp_path / f"{original}.json.sig", tmp_path / f"{replay}.json.sig")
    with pytest.raises(subprocess.CalledProcessError):
        p8_candidate._verify_signature(tmp_path, replay, payload.read_bytes())
