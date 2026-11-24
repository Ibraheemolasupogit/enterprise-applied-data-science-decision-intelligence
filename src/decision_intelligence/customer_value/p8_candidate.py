"""Identity and detached approval binding for a future M11 P8 attempt."""

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import launch_approval as legacy
from . import p8_immutable, p8_journal, p8_private

A19_ATTEMPT = "final-a19"
HISTORICAL_PINS = {
    "final": (
        "cd7c8391347df2f9a2cf802161f35a0cb6a9a889b29c8fce3825a22ebe5902a9",
        "e5dc98f0402381f5c73083278ca2cf140e6cf76fb77a3b6e5c5641f724c6ad04",
    ),
    "final-a14": (
        "74a0cdd66175ab19f5ecf0dddef08114230713897d5f968d6295df390b678008",
        "806a2b126e94a903a170e7ea50e7b2ae0a8dd7039852772df950dc7dc9cc7ae3",
    ),
    "final-a15": (
        "1adb6b9ec8d7bbfd3d71b4b387b546612c66472b369cace55d732e97ee107649",
        "5dba9248b975e60773bf264db28fdd52e0dc52f1b772056589d57a3646a7531d",
    ),
}
HISTORY_CHAIN_SHA256 = "5dbcc3a8f3ee527cd75d50fcfb099f967968119b57d19985df7d197611259c2a"
HISTORY_PREFIX_SHA256 = "52c6031120ff02a58b6b773134f69ad9d48d322a20fb10ce9418c6af58eec2fb"


def require_new_identity(attempt: str) -> None:
    if attempt in HISTORICAL_PINS:
        raise legacy.LaunchApprovalError(f"attempt {attempt} is permanently consumed")
    if len(attempt) > 64 or re.fullmatch(r"final-[a-z0-9]+(?:-[a-z0-9]+)*", attempt) is None:
        raise legacy.LaunchApprovalError("a distinct canonical attempt identity is required")


def _verify_signature(directory: Path, attempt: str, payload: bytes) -> None:
    subprocess.run(
        ["/usr/bin/ssh-keygen", "-Y", "verify", "-f", str(directory / "allowed_signers"),
         "-I", legacy._PRINCIPAL, "-n", f"m11-p8-{attempt}",
         "-s", str(directory / f"{attempt}.json.sig")],
        input=payload, capture_output=True, check=True, timeout=5, env=legacy._COMMAND_ENV,
    )


def require_active_claim(repo: Path, *, attempt: str, manifest: str, code: str,
                         approval_sha256: str) -> dict[str, Any]:
    private = p8_private.open_private_evidence(repo)
    _, pin = private.history_manifest()
    expected = {"supervisor_pid": os.getppid(), "run_dir": str(private.run_dir(attempt)),
                "history_manifest_sha256": pin, "manifest_digest": manifest, "code_digest": code,
                "approval_sha256": approval_sha256}
    with p8_journal.open_journal(legacy._APPROVAL_DIR / "consumption.jsonl") as journal:
        claims = [row for row in journal.records() if row["attempt"] == attempt]
    if (os.geteuid() == 0 or os.getsid(0) != os.getpid()
            or len(claims) != 1 or claims[0]["state"] != "consumed"
            or any(claims[0]["payload"].get(key) != value for key, value in expected.items())):
        raise legacy.LaunchApprovalError("worker has no matching active supervisor claim")
    return claims[0]


def release_binding(repo: Path, *, attempt: str, manifest: str, code: str,
                    require_clean: bool = True, worker_uid: int | None = None) -> dict[str, str]:
    """Measure exact approval fields without signing or reserving anything."""
    require_new_identity(attempt)
    immutable = p8_immutable.binding(repo)
    binding = legacy._live_binding(repo, attempt=attempt, manifest=manifest, code=code,
                                   require_clean=require_clean, successor=True)
    private = p8_private.open_private_evidence(repo, owner=worker_uid)
    _, history_sha256 = private.history_manifest()
    worker = private.root.lstat()
    binding.update(registry=str(private.registry), output=str(private.run_dir(attempt)),
                   history_sha256=history_sha256, worker_uid=str(worker.st_uid),
                   worker_gid=str(worker.st_gid))
    binding.update(immutable)
    source = repo / "src" / "decision_intelligence" / "customer_value"
    for name in ("p8_scheduler", "p8_candidate", "p8_private", "p8_journal", "p8_immutable"):
        module = source / f"{name}.py"
        if module.resolve() != module:
            raise legacy.LaunchApprovalError("release bindings require canonical source paths")
        binding[f"{name}_sha256"] = hashlib.sha256(module.read_bytes()).hexdigest()
    return binding


def require_owner_approval(repo: Path, *, attempt: str, manifest: str, code: str,
                           require_clean: bool, supervisor_uid: int | None = None) -> str:
    """Require an attempt-specific signature and an exact release binding."""
    require_new_identity(attempt)
    if sys.platform != "darwin":
        raise legacy.LaunchApprovalError("owner-approved launch requires the designated Mac")
    directory = legacy._APPROVAL_DIR
    path = directory / f"{attempt}.json"
    try:
        for parent in (*reversed(directory.parents), directory):
            legacy._trusted_path(parent, directory=True)
        for file in (path, directory / f"{attempt}.json.sig", directory / "allowed_signers"):
            legacy._trusted_path(file)
        payload = path.read_bytes()
        _verify_signature(directory, attempt, payload)
        record = json.loads(payload, object_pairs_hook=legacy._unique_fields)
        if not isinstance(record, dict):
            raise legacy.LaunchApprovalError("invalid owner launch approval record")
        ci = record.pop("ci_run_id", None)
        if not isinstance(ci, str) or not re.fullmatch(r"[1-9][0-9]*", ci):
            raise legacy.LaunchApprovalError("owner approval must identify the green CI run")
        if record.pop("host_preflight_passed", None) is not True:
            raise legacy.LaunchApprovalError("owner approval must confirm immediate host preflight")
        binding = release_binding(repo, attempt=attempt, manifest=manifest, code=code,
                                  require_clean=require_clean, worker_uid=supervisor_uid)
        if record != binding:
            raise legacy.LaunchApprovalError("owner approval does not match the release binding")
        return hashlib.sha256(payload).hexdigest()
    except (OSError, ValueError, RecursionError, subprocess.SubprocessError) as exc:
        raise legacy.LaunchApprovalError(
            "owner approval is missing, invalid or unreadable") from exc
