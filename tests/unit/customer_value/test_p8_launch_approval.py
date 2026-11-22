"""Exercise the launch verifier with isolated, deliberately inapplicable records."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from decision_intelligence.customer_value import launch_approval as approval

_BINDING = {
    "commit": "1" * 40,
    "manifest_sha256": "2" * 64,
    "code_sha256": "3" * 64,
    "runner_sha256": "4" * 64,
    "supervisor_sha256": "5" * 64,
    "host_uuid": "00000000-0000-0000-0000-000000000001",
    "attempt": "final-a15",
    "registry": "/fixture/registry",
    "output": "/fixture/output",
}


@pytest.mark.parametrize("field", [*_BINDING, "ci_run_id", "host_preflight_passed"])
@pytest.mark.parametrize("remove", [False, True])
def test_mismatched_or_incomplete_owner_approval_refuses(
        field: str, remove: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record: dict[str, str | bool] = {
        **_BINDING, "ci_run_id": "123", "host_preflight_passed": True}
    if remove:
        record.pop(field)
    else:
        record[field] = "incorrect"
    path = tmp_path / "final-a15.json"
    path.write_text(json.dumps(record))
    monkeypatch.setattr(approval.sys, "platform", "darwin")
    monkeypatch.setattr(approval, "_APPROVAL_DIR", tmp_path)
    monkeypatch.setattr(approval, "_trusted_path", lambda *args, **kwargs: None)
    monkeypatch.setattr(approval, "_verify_signature", lambda *args: None)
    monkeypatch.setattr(approval, "_live_binding", lambda *args, **kwargs: _BINDING)
    with pytest.raises(approval.LaunchApprovalError, match="release binding|CI run|preflight"):
        approval.require_owner_approval(tmp_path, attempt="final-a15", manifest="fixture",
                                        code="fixture", require_clean=True)


def test_approval_path_rejects_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text("fixture")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(approval.LaunchApprovalError, match="root-controlled"):
        approval._trusted_path(link)


@pytest.mark.skipif(not Path("/usr/bin/ssh-keygen").is_file(), reason="OpenSSH required")
def test_detached_signature_rejects_changed_payload_and_untrusted_key(tmp_path: Path) -> None:
    def key(name: str) -> Path:
        path = tmp_path / name
        subprocess.run(["/usr/bin/ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(path)],
                       check=True, capture_output=True, timeout=10)
        return path

    owner, other = key("owner"), key("other")
    payload = tmp_path / "final-a15.json"
    payload.write_bytes(b"deliberately invalid fixture payload")
    subprocess.run(["/usr/bin/ssh-keygen", "-Y", "sign", "-f", str(owner),
                    "-n", approval._NAMESPACE, str(payload)],
                   check=True, capture_output=True, timeout=10)
    signers = tmp_path / "allowed_signers"
    signers.write_text(f"{approval._PRINCIPAL} {owner.with_suffix('.pub').read_text()}")
    approval._verify_signature(tmp_path, payload.read_bytes())
    with pytest.raises(approval.LaunchApprovalError, match="signature"):
        approval._verify_signature(tmp_path, b"changed payload")
    signers.write_text(f"{approval._PRINCIPAL} {other.with_suffix('.pub').read_text()}")
    with pytest.raises(approval.LaunchApprovalError, match="signature"):
        approval._verify_signature(tmp_path, payload.read_bytes())
