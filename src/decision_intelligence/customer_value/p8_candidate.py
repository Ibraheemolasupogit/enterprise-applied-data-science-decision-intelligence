"""Identity and detached approval binding for a future M11 P8 attempt."""

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import launch_approval as legacy
from . import p8_immutable, p8_journal, p8_private

A19_ATTEMPT = "final-a19"
A20_ATTEMPT = "final-a20"
A21_ATTEMPT = "final-a21"
_A19_MANIFEST = "c930e20b123a52d6b14ad58309f6bc5ac690ba10ec18a7f6aa3b2c83a09b0999"
_A19_CODE = "66b1c18c6cd2346f9232d236b5aac93aa0796f857f20b4ee7f57406e74dd9abb"
_A20_MANIFEST = "49f3cd6f9e4895daecb9e80727696661a3a0af843feddf1d380041af2f547ce1"
_A20_CODE = "6120b81ebb2d9f0d19967266bb4b60905cc172b78f9d49b971e77a87a1f72e2d"
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
CONSUMED_IDENTITIES = frozenset((*HISTORICAL_PINS, A19_ATTEMPT, A20_ATTEMPT))


def require_new_identity(attempt: str) -> None:
    if attempt in CONSUMED_IDENTITIES:
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


def _verify_a19_prefix(private: p8_private.PrivateEvidence, rows: list[dict[str, Any]]) -> str:
    _, history_pin = private.history_manifest()
    if ([(r["attempt"], r["state"]) for r in rows[:2]]
            != [(A19_ATTEMPT, "consumed"), (A19_ATTEMPT, "terminal")]):
        raise legacy.LaunchApprovalError("protected A19 journal history is missing or unexpected")
    claim, terminal = rows[:2]
    a19_dir = private.run_dir(A19_ATTEMPT)
    expected = {"manifest_digest": _A19_MANIFEST, "code_digest": _A19_CODE,
                "history_manifest_sha256": history_pin, "run_dir": str(a19_dir)}
    approval = claim["payload"].get("approval_sha256")
    if (any(claim["payload"].get(k) != v for k, v in expected.items())
            or not isinstance(approval, str) or re.fullmatch(r"[0-9a-f]{64}", approval) is None
            or terminal["payload"].get("status") != "child_failure"
            or terminal["payload"].get("returncode") != 2
            or a19_dir.exists()):
        raise legacy.LaunchApprovalError("protected A19 startup-failure evidence does not match")
    return str(terminal["sha256"])


def verify_a19_consumption(private: p8_private.PrivateEvidence, *,
                           require_unused: bool = False) -> str:
    """Verify the protected A19 prefix and return its terminal chain digest."""
    with p8_journal.open_journal(legacy._APPROVAL_DIR / "consumption.jsonl") as journal:
        rows = journal.records()
    tip = _verify_a19_prefix(private, rows)
    if any(r["attempt"] != A20_ATTEMPT for r in rows[2:]):
        raise legacy.LaunchApprovalError("protected A19 journal history is unexpected")
    if require_unused and len(rows) != 2:
        raise legacy.LaunchApprovalError("final-a20 is already consumed")
    return tip


def _history_metadata(path: Path, owner: int) -> tuple[dict[str, Any], str]:
    """Read bounded ledger or reservation metadata without following links."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != owner or info.st_nlink != 1
                or info.st_size > 1_048_576):
            raise legacy.LaunchApprovalError("invalid A20 history metadata file")
        body = stream.read(1_048_577)
    if len(body) > 1_048_576:
        raise legacy.LaunchApprovalError("A20 history metadata exceeds size limit")
    data = json.loads(body, object_pairs_hook=legacy._unique_fields)
    if not isinstance(data, dict):
        raise legacy.LaunchApprovalError("invalid A20 history metadata object")
    return data, hashlib.sha256(body).hexdigest()


def verify_a20_consumption(private: p8_private.PrivateEvidence, *,
                           require_unused: bool = False) -> dict[str, str]:
    """Check the protected A19/A20 prefix and linked A20 failure metadata only."""
    with p8_journal.open_journal(legacy._APPROVAL_DIR / "consumption.jsonl") as journal:
        rows = journal.records()
    a19_tip = _verify_a19_prefix(private, rows)
    if ([(r["attempt"], r["state"]) for r in rows[2:4]]
            != [(A20_ATTEMPT, "consumed"), (A20_ATTEMPT, "terminal")]
            or any(r["attempt"] != A21_ATTEMPT for r in rows[4:])):
        raise legacy.LaunchApprovalError("protected A20 journal history is missing or unexpected")
    if require_unused and len(rows) != 4:
        raise legacy.LaunchApprovalError("final-a21 is already consumed")
    claim, terminal = rows[2:4]
    _, history_pin = private.history_manifest()
    run_dir = private.run_dir(A20_ATTEMPT)
    expected = {"manifest_digest": _A20_MANIFEST, "code_digest": _A20_CODE,
                "history_manifest_sha256": history_pin, "run_dir": str(run_dir)}
    approval = claim["payload"].get("approval_sha256")
    if (any(claim["payload"].get(k) != v for k, v in expected.items())
            or not isinstance(approval, str) or re.fullmatch(r"[0-9a-f]{64}", approval) is None
            or terminal["payload"].get("status") != "child_failure"
            or terminal["payload"].get("returncode") != 1):
        raise legacy.LaunchApprovalError("protected A20 failure receipt does not match")
    owner = private.root.lstat().st_uid
    ledger, ledger_sha = _history_metadata(run_dir / "ledger.json", owner)
    entry, entry_sha = _history_metadata(private.registry / f"{A20_ATTEMPT}.json", owner)
    error = ledger.get("error")
    if (any(entry.get(k) != v for k, v in expected.items())
            or entry.get("mode") != "final" or entry.get("key") != A20_ATTEMPT
            or entry.get("claim_sha256") != claim["sha256"]
            or ledger.get("mode") != "final" or ledger.get("reservation") != A20_ATTEMPT
            or ledger.get("claim_sha256") != claim["sha256"]
            or ledger.get("reservation_sha256") != entry_sha
            or ledger.get("manifest_digest") != _A20_MANIFEST
            or ledger.get("code_digest") != _A20_CODE
            or ledger.get("status") != "failed" or ledger.get("phase") != "repeated_sampling"
            or not isinstance(error, str) or not error.startswith("task R2 ")
            or not error.endswith(": OverflowError")):
        raise legacy.LaunchApprovalError(
            "A20 failed R2 metadata does not match the protected claim")
    return {"a19_journal_tip_sha256": a19_tip, "a20_journal_tip_sha256": str(terminal["sha256"]),
            "a20_ledger_sha256": ledger_sha, "a20_reservation_sha256": entry_sha}


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
    if attempt == A20_ATTEMPT:
        binding["a19_journal_tip_sha256"] = verify_a19_consumption(private)
    if attempt == A21_ATTEMPT:
        binding.update(verify_a20_consumption(private))
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
