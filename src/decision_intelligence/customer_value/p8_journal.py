"""Owner-protected, append-only consumption journal for the governed supervisor."""

import fcntl
import hashlib
import json
import os
import stat
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import launch_approval as trust


class JournalError(RuntimeError):
    """An attempt cannot be safely started or certified."""


class Journal:
    def __init__(self, fd: int, *, full_sync: bool = False) -> None:
        self.fd = fd
        self.full_sync = full_sync

    @contextmanager
    def _locked(self) -> Iterator[None]:
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(self.fd, fcntl.LOCK_UN)

    def _read(self) -> list[dict[str, Any]]:
        os.lseek(self.fd, 0, os.SEEK_SET)
        with os.fdopen(os.dup(self.fd), "rb") as stream:
            data = stream.read(1_048_577)
        if len(data) > 1_048_576 or (data and not data.endswith(b"\n")):
            raise JournalError("journal is oversized or contains a partial record")
        rows: list[dict[str, Any]] = []
        prior = "0" * 64
        states: dict[str, dict[str, Any]] = {}
        try:
            for line in data.splitlines():
                row = json.loads(line, object_pairs_hook=trust._unique_fields)
                digest = row.pop("sha256")
                key, state = row["attempt"], row["state"]
                if (type(key) is not str or row["previous"] != prior
                        or digest != self._digest(row)
                        or not self._extends(states.get(key), state, row["payload"])):
                    raise JournalError("journal chain or attempt sequence is invalid")
                row["sha256"] = digest
                states[key] = row
                prior = digest
                rows.append(row)
        except (TypeError, ValueError, KeyError, AttributeError, trust.LaunchApprovalError) as exc:
            raise JournalError("journal contains invalid records") from exc
        return rows

    @staticmethod
    def _digest(row: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(row, sort_keys=True, allow_nan=False).encode()).hexdigest()

    @staticmethod
    def _extends(prior: dict[str, Any] | None, state: str, payload: dict[str, Any]) -> bool:
        if prior is None:
            return state == "consumed"
        if state == "terminal":
            return bool(prior["state"] == "consumed"
                        and payload.get("claim_sha256") == prior["sha256"])
        return bool(state == "certified" and prior["state"] == "terminal"
                    and prior["payload"].get("status") == "supervision_complete"
                    and payload.get("terminal_sha256") == prior["sha256"])

    def records(self) -> list[dict[str, Any]]:
        with self._locked():
            return self._read()

    def append(self, attempt: str, state: str, payload: dict[str, Any]) -> str:
        if state not in ("consumed", "terminal", "certified"):
            raise JournalError("invalid attempt state")
        with self._locked():
            rows = self._read()
            matches = [row for row in rows if row["attempt"] == attempt]
            if state == "consumed" and matches:
                raise JournalError("attempt is permanently consumed")
            if not self._extends(matches[-1] if matches else None, state, payload):
                raise JournalError("receipt does not match the preceding attempt record")
            row = {"attempt": attempt, "state": state, "payload": payload,
                   "previous": rows[-1]["sha256"] if rows else "0" * 64}
            digest = self._digest(row)
            data = (json.dumps({**row, "sha256": digest}, sort_keys=True) + "\n").encode()
            if os.write(self.fd, data) != len(data):
                raise JournalError("short journal write; owner reconciliation required")
            if sys.platform == "darwin":
                if self.full_sync:
                    fcntl.fcntl(self.fd, fcntl.F_FULLFSYNC)
                    return digest
            os.fsync(self.fd)
            return digest


@contextmanager
def open_journal(path: Path, *, writable: bool = False) -> Iterator[Journal]:
    """Open the preprovisioned macOS system-append-only journal, without creating it."""
    for directory in reversed(path.parents):
        trust._trusted_path(directory, directory=True)
    if writable and os.geteuid() != 0:
        raise JournalError("only the privileged supervisor can commit journal records")
    if os.geteuid() != 0 and os.access(path, os.W_OK):
        raise JournalError("worker must have read-only access to the journal")
    access = os.O_RDWR | os.O_APPEND if writable else os.O_RDONLY
    fd = os.open(path, access | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_nlink != 1
                or stat.S_IMODE(info.st_mode) != 0o640
                or not getattr(info, "st_flags", 0) & getattr(stat, "SF_APPEND", 0)):
            raise JournalError("journal must be root-owned and system-append-only")
        yield Journal(fd, full_sync=writable)
    finally:
        os.close(fd)
