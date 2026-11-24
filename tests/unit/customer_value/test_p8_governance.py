"""Native supervision and approval tests on development data and ephemeral fixture keys."""

import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
from scripts import generate_m11_evidence as runner

from decision_intelligence.customer_value import launch_approval, p8_candidate, p8_journal

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
@pytest.mark.parametrize("scenario", ["complete", "duplicate-fresh", "memory_limit", "time_limit"])
def test_native_successor_receipt(tmp_path, monkeypatch, scenario):
    script = tmp_path / "fixture.py"
    script.write_text(_WORKER)
    journal_path = tmp_path / "journal"
    ledger_root = tmp_path / "run"
    # The development subprocess can read its claim but this test never uses the installed journal.
    fd = os.open(journal_path, os.O_CREAT | os.O_EXCL | os.O_APPEND | os.O_RDWR, 0o600)
    try:
        journal = p8_journal.Journal(fd, full_sync=True)

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
        consumed, terminal = journal.records()
        assert terminal["payload"]["claim_sha256"] == consumed["sha256"]
        expected = "certification_failure" if scenario == "duplicate-fresh" else scenario
        assert terminal["payload"]["status"] == expected
        if scenario == "complete":
            assert runner.verify_run(ledger_root) == []
            path = ledger_root / "ledger.json"
            data = json.loads(path.read_text())
            data["events"][-1]["files"][0]["sha256"] = "0" * 64
            path.write_text(json.dumps(data))
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
        assert journal.records()[0]["sha256"] == claim
        os.write(fd, b"{")
        with pytest.raises(p8_journal.JournalError, match="partial"):
            journal.append("fixture-other", "consumed", {})
    finally:
        os.close(fd)


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
    replay = "final-fixture-two"
    shutil.copyfile(tmp_path / f"{original}.json.sig", tmp_path / f"{replay}.json.sig")
    with pytest.raises(subprocess.CalledProcessError):
        p8_candidate._verify_signature(tmp_path, replay, payload.read_bytes())
