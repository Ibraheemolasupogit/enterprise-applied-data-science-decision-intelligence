"""Verify the owner's detached approval for the one M11 P8 candidate launch."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

_APPROVAL_DIR = Path("/private/etc/m11-p8")
_PRINCIPAL = "m11-p8-owner"
_NAMESPACE = "m11-p8-final-a15"
_MAX_FILE_BYTES = 16_384
_COMMAND_ENV = {"PATH": "/usr/bin:/bin", "LC_ALL": "C", "GIT_CONFIG_GLOBAL": "/dev/null",
                "GIT_CONFIG_NOSYSTEM": "1", "GIT_OPTIONAL_LOCKS": "0"}


class LaunchApprovalError(RuntimeError):
    """The launch lacks a verifiable owner approval."""


def _trusted_path(path: Path, *, directory: bool = False) -> None:
    metadata = path.lstat()
    correct_type = stat.S_ISDIR if directory else stat.S_ISREG
    if (metadata.st_uid != 0 or metadata.st_mode & 0o022
            or (os.geteuid() != 0 and os.access(path, os.W_OK))
            or not correct_type(metadata.st_mode)
            or (not directory and not 0 < metadata.st_size <= _MAX_FILE_BYTES)):
        raise LaunchApprovalError("owner approval files must be root-controlled regular files")


def _verify_signature(directory: Path, payload: bytes) -> None:
    try:
        subprocess.run(
            ["/usr/bin/ssh-keygen", "-Y", "verify", "-f", str(directory / "allowed_signers"),
             "-I", _PRINCIPAL, "-n", _NAMESPACE, "-s", str(directory / "final-a15.json.sig")],
            input=payload, capture_output=True, check=True, timeout=5, env=_COMMAND_ENV,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LaunchApprovalError("owner launch approval signature could not be verified") from exc


def _unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    record = dict(pairs)
    if len(record) != len(pairs):
        raise LaunchApprovalError("owner approval contains duplicate fields")
    return record


def _live_binding(repo: Path, *, attempt: str, manifest: str, code: str,
                  require_clean: bool, successor: bool = False) -> dict[str, str]:
    git = ["/usr/bin/git", "-c", "core.fsmonitor=false", "-c", "core.untrackedCache=false"]
    if successor:
        git.extend(["-c", f"safe.directory={repo}", "-c", "core.hooksPath=/dev/null"])
    host = subprocess.run(["/usr/sbin/ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                          capture_output=True, check=True, text=True, timeout=5,
                          env=_COMMAND_ENV).stdout
    identities = re.findall(r'^\s*"IOPlatformUUID" = "([0-9a-fA-F-]{36})"\s*$',
                            host, flags=re.MULTILINE)
    if len(identities) != 1:
        raise LaunchApprovalError("cannot establish the designated Mac hardware identity")
    head = subprocess.run([*git, "rev-parse", "--verify", "HEAD"], cwd=repo,
                          capture_output=True, check=True, text=True, timeout=5,
                          env=_COMMAND_ENV).stdout.strip()
    dirty = require_clean and subprocess.run(
        [*git, "status", "--porcelain=v1", "--untracked-files=all"], cwd=repo,
        capture_output=True, check=True, timeout=5, env=_COMMAND_ENV,
    ).stdout
    if not re.fullmatch(r"[0-9a-f]{40}", head) or dirty:
        raise LaunchApprovalError("owner launch approval requires a clean exact Git commit")

    runner = repo / "scripts" / "generate_m11_evidence.py"
    supervisor = repo / "src" / "decision_intelligence" / "customer_value" / (
        "resource_supervision.py")
    registry = repo / "docs" / "evidence" / "m11-p8-attempts"
    output = registry.parent / f"m11-p8-{attempt}"
    if any(path.resolve() != path for path in (runner, supervisor, registry, output)):
        raise LaunchApprovalError("launch paths must be canonical and free of symlink redirects")
    return {
        "commit": head,
        "manifest_sha256": manifest,
        "code_sha256": code,
        "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
        "supervisor_sha256": hashlib.sha256(supervisor.read_bytes()).hexdigest(),
        "host_uuid": identities[0].lower(),
        "attempt": attempt,
        "registry": str(registry),
        "output": str(output),
    }


def require_owner_approval(repo: Path, *, attempt: str, manifest: str, code: str,
                           require_clean: bool) -> None:
    """Fail closed unless the designated owner's signed record matches this launch."""
    if sys.platform != "darwin":
        raise LaunchApprovalError("owner-approved final launch requires the designated Mac")
    payload_path = _APPROVAL_DIR / "final-a15.json"
    try:
        for path in (*reversed(_APPROVAL_DIR.parents), _APPROVAL_DIR):
            _trusted_path(path, directory=True)
        for path in (payload_path, _APPROVAL_DIR / "final-a15.json.sig",
                     _APPROVAL_DIR / "allowed_signers"):
            _trusted_path(path)
        payload = payload_path.read_bytes()
        _verify_signature(_APPROVAL_DIR, payload)
        record = json.loads(payload, object_pairs_hook=_unique_fields)
        if not isinstance(record, dict):
            raise LaunchApprovalError("invalid owner launch approval record")
        ci_run = record.pop("ci_run_id", None)
        if not isinstance(ci_run, str) or not re.fullmatch(r"[1-9][0-9]*", ci_run):
            raise LaunchApprovalError("owner approval must identify the green CI run")
        if record.pop("host_preflight_passed", None) is not True:
            raise LaunchApprovalError("owner approval must confirm the immediate host preflight")
        expected = _live_binding(repo, attempt=attempt, manifest=manifest, code=code,
                                 require_clean=require_clean)
        if record != expected:
            raise LaunchApprovalError("owner launch approval does not match the release binding")
    except (OSError, ValueError, RecursionError, subprocess.SubprocessError) as exc:
        raise LaunchApprovalError("owner launch approval is missing or unreadable") from exc
