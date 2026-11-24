"""Private-path and reviewed-export checks using synthetic data only."""

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
from scripts import generate_m11_evidence as runner

from decision_intelligence.customer_value import p8_private
from decision_intelligence.customer_value.resource_supervision import SupervisionResult


def test_checkout_and_paths_redirected_to_it_are_refused(p8_layout, tmp_path):
    repo = tmp_path / "checkout"
    repo.mkdir()
    (repo / ".git").mkdir()
    checkout_dir = repo / "private"
    checkout_dir.mkdir(mode=0o700)
    with pytest.raises(p8_private.PrivateEvidenceError, match="outside Git"):
        p8_private.PrivateEvidence(checkout_dir, p8_layout.repo)
    alias = tmp_path / "alias"
    alias.symlink_to(checkout_dir, target_is_directory=True)
    with pytest.raises(p8_private.PrivateEvidenceError, match="redirects"):
        p8_private.PrivateEvidence(alias, p8_layout.repo)
    staging = p8_layout.staging
    staging.chmod(0o750)
    with pytest.raises(p8_private.PrivateEvidenceError, match="owner-only"):
        p8_layout.run_dir("dry-run-fixture")
    staging.chmod(0o700)
    (staging / ".git").mkdir()
    with pytest.raises(p8_private.PrivateEvidenceError, match="outside Git"):
        p8_layout.run_dir("dry-run-fixture")


@pytest.mark.parametrize("option", ["--run-dir", "--registry", "--output"])
def test_cli_rejects_public_execution_destination(p8_layout, tmp_path, monkeypatch, option):
    repo = tmp_path / "checkout"
    repo.mkdir()
    monkeypatch.setattr(runner, "REPO", repo)
    target = repo / "unapproved"
    assert runner.main(["--mode", "dry-run", option, str(target)]) == 2
    assert not target.exists()
    assert list(p8_layout.registry.iterdir()) == []


def test_fixture_results_and_supervisor_receipt_stay_private(p8_layout, monkeypatch):
    plan = runner.dry_run_plan(draws=1, refits=1)
    key = runner.reservation_key(plan)
    run = p8_layout.run_dir(key)

    def draw(task):
        return {"task": task, "data": "synthetic output"}

    def report(plan, results, perf, directory, output):
        record = directory / "record.md"
        record.write_text("synthetic report")
        return [record]

    runner.execute(plan, run, 1, fn=draw, registry=p8_layout.registry,
                   report_fn=report, private=p8_layout)
    assert (run / "raw" / "repeated_sampling.pkl").is_file()
    assert (run / "record.md").is_file()
    assert (run / "ledger.json").is_file()
    assert (p8_layout.registry / f"{key}.json").is_file()
    assert run.stat().st_mode & 0o077 == 0
    monkeypatch.setattr(runner, "supervise",
                        lambda *a, **kw: SupervisionResult("memory_limit", -9, 2.0, 4_000_000_001))
    assert runner._run_under_supervision([], run, private=p8_layout) == 1
    receipts = list(p8_layout.supervisor_dir().glob("*.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text())["status"] == "memory_limit"
    assert not (run / "supervisor.json").exists()


@pytest.fixture
def export_fixture(p8_layout, tmp_path, monkeypatch):
    repo = tmp_path / "checkout"
    public = repo / "docs" / "evidence"
    public.mkdir(parents=True)
    layout = replace(p8_layout, repo=repo)
    reviewed = layout.root / "reviewed" / "qualification.json"
    pins = {}
    monkeypatch.setattr(p8_private, "_owner_pin", pins.__getitem__)
    return layout, public, reviewed, pins


_SUMMARY = {"schema": 1, "scope": "development-resource-summary",
            "elapsed_s": 36.25, "peak_bytes": 123_000_000, "draws": 16}


def test_reviewed_export_only_reads_the_prepared_summary(export_fixture, monkeypatch):
    layout, public, reviewed, pins = export_fixture
    raw = layout.staging / "raw-fixture.pkl"
    raw.write_bytes(b"PROTECTED FIXTURE")
    reviewed.write_text(json.dumps(_SUMMARY))
    pins["export-qualification.sha256"] = hashlib.sha256(reviewed.read_bytes()).hexdigest()
    read_bytes = Path.read_bytes

    def guarded_read(path):
        assert path == reviewed, "export attempted to read a different file"
        return read_bytes(path)

    with monkeypatch.context() as probe:
        probe.setattr(Path, "read_bytes", guarded_read)
        assert list(public.iterdir()) == []
        path = p8_private.export_reviewed(layout, "qualification")
        with pytest.raises(p8_private.PrivateEvidenceError, match="could not be published"):
            p8_private.export_reviewed(layout, "qualification")
    assert list(public.iterdir()) == [path]
    assert json.loads(path.read_text()) == _SUMMARY
    assert b"PROTECTED FIXTURE" not in path.read_bytes()


@pytest.mark.parametrize("damage", ["pin", "raw-field", "hard-link", "write-failure"])
def test_failed_export_leaves_no_public_payload(export_fixture, monkeypatch, damage):
    layout, public, reviewed, pins = export_fixture
    if damage == "hard-link":
        raw = layout.staging / "raw-fixture.pkl"
        raw.write_bytes(b"PROTECTED FIXTURE")
        os.link(raw, reviewed)
    else:
        reviewed.write_text(json.dumps({**_SUMMARY, "raw": "payload"})
                            if damage == "raw-field" else json.dumps(_SUMMARY))
    pins["export-qualification.sha256"] = (
        "0" * 64 if damage == "pin" else hashlib.sha256(reviewed.read_bytes()).hexdigest())

    def fail_link(*args):
        raise PermissionError("fixture publication failure")

    if damage == "write-failure":
        monkeypatch.setattr(p8_private.os, "link", fail_link)
    with pytest.raises(p8_private.PrivateEvidenceError):
        p8_private.export_reviewed(layout, "qualification")
    assert list(public.iterdir()) == []
