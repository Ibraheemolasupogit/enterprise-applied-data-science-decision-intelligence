"""Private P8 execution paths and explicit, reviewed public summaries.

The operator owns the 0700 evidence tree. Only the owner-controlled trust directory
may pin a history manifest or reviewed export. Historical replicas contain metadata
only and preserve the original bytes. Provisioning and copying are owner operations.
"""

import hashlib
import json
import math
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import launch_approval as trust

REGISTRY_NAME = "m11-p8-attempts"
_ROOT = Path.home() / "Library" / "Application Support" / "M11-P8-Private"


class PrivateEvidenceError(RuntimeError):
    """An evidence path or owner pin failed validation."""


def _canonical(path: Path) -> None:
    if not path.is_absolute() or path.resolve() != path:
        raise PrivateEvidenceError("evidence paths must be absolute and free of redirects")


def require_outside_checkout(path: Path, repo: Path) -> None:
    _canonical(path)
    if path.is_relative_to(repo.resolve()) or any(
        (p / ".git").exists() or (p / ".git").is_symlink() for p in (path, *path.parents)
    ):
        raise PrivateEvidenceError("private evidence must be outside Git checkouts")


def _private_dir(path: Path, owner: int | None = None) -> None:
    _canonical(path)
    try:
        info = path.lstat()
    except OSError as exc:
        raise PrivateEvidenceError("owner must provision the private evidence directories") from exc
    uid = os.geteuid() if owner is None else owner
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != uid or info.st_mode & 0o077:
        raise PrivateEvidenceError("private evidence directories must be owner-only")


def _owner_pin(filename: str) -> str:
    directory = trust._APPROVAL_DIR
    pin = directory / filename
    try:
        for path in (*reversed(directory.parents), directory):
            trust._trusted_path(path, directory=True)
        trust._trusted_path(pin)
        value = pin.read_text(encoding="ascii").strip()
    except (OSError, ValueError, trust.LaunchApprovalError) as exc:
        raise PrivateEvidenceError("owner evidence pin is unavailable or untrusted") from exc
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise PrivateEvidenceError("invalid owner evidence pin")
    return value


def _pinned_json(path: Path, pin: str) -> tuple[dict[str, Any], str]:
    _canonical(path)
    try:
        expected = _owner_pin(pin)
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 16_384:
            raise PrivateEvidenceError("expected a small, regular evidence manifest")
        body = path.read_bytes()
        digest = hashlib.sha256(body).hexdigest()
        if digest != expected:
            raise PrivateEvidenceError("private evidence does not match its owner pin")
        doc = json.loads(body, object_pairs_hook=trust._unique_fields)
    except (OSError, ValueError, RecursionError, trust.LaunchApprovalError) as exc:
        raise PrivateEvidenceError("cannot read the owner-pinned evidence manifest") from exc
    if not isinstance(doc, dict):
        raise PrivateEvidenceError("expected an evidence manifest object")
    return doc, digest


@dataclass(frozen=True)
class PrivateEvidence:
    root: Path
    repo: Path
    owner: int | None = None

    def __post_init__(self) -> None:
        _private_dir(self.root, self.owner)
        require_outside_checkout(self.root, self.repo)

    @property
    def history(self) -> Path:
        path = self.root / "history"
        _private_dir(path, self.owner)
        return path

    @property
    def staging(self) -> Path:
        path = self.root / "staging"
        _private_dir(path, self.owner)
        require_outside_checkout(path, self.repo)
        return path

    @property
    def registry(self) -> Path:
        path = self.staging / REGISTRY_NAME
        _private_dir(path, self.owner)
        require_outside_checkout(path, self.repo)
        return path

    @property
    def history_registry(self) -> Path:
        path = self.history / REGISTRY_NAME
        _private_dir(path, self.owner)
        return path

    def run_dir(self, key: str) -> Path:
        if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", key) is None:
            raise PrivateEvidenceError("invalid private run identity")
        path = self.staging / f"m11-p8-{key}"
        require_outside_checkout(path, self.repo)
        return path

    def check_target(self, key: str, run_dir: Path, registry: Path, output: Path | None) -> None:
        if run_dir != self.run_dir(key) or registry != self.registry or output is not None:
            raise PrivateEvidenceError("execution requires the canonical private run and registry")

    def history_manifest(self) -> tuple[dict[str, Any], str]:
        return _pinned_json(self.history / "manifest.json", "history.sha256")

    def supervisor_dir(self) -> Path:
        path = self.staging / "supervisor"
        _private_dir(path, self.owner)
        require_outside_checkout(path, self.repo)
        return path


def open_private_evidence(repo: Path, *, owner: int | None = None) -> PrivateEvidence:
    return PrivateEvidence(_ROOT, repo, owner)


def export_reviewed(layout: PrivateEvidence, slug: str) -> Path:
    """Publish only an owner-pinned, separately prepared development resource summary."""
    if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", slug) is None or len(slug) > 64:
        raise PrivateEvidenceError("invalid export identity")
    reviewed = layout.root / "reviewed"
    _private_dir(reviewed)
    require_outside_checkout(reviewed, layout.repo)
    doc, _ = _pinned_json(reviewed / f"{slug}.json", f"export-{slug}.sha256")
    if (doc.keys() != {"schema", "scope", "elapsed_s", "peak_bytes", "draws"}
            or type(doc["schema"]) is not int or doc["schema"] != 1
            or doc["scope"] != "development-resource-summary"
            or type(doc["elapsed_s"]) not in (float, int)
            or not 0 < doc["elapsed_s"] <= 86_400 or not math.isfinite(doc["elapsed_s"])
            or type(doc["peak_bytes"]) is not int or not 0 < doc["peak_bytes"] <= 2**50
            or type(doc["draws"]) is not int or not 0 < doc["draws"] <= 100_000):
        raise PrivateEvidenceError("reviewed export must contain only development resource totals")
    directory = layout.repo.resolve() / "docs" / "evidence"
    path = directory / f"m11-p8-public-{slug}.json"
    _canonical(path)
    # Publish a complete file exclusively. A failed export leaves no public temporary file.
    with tempfile.NamedTemporaryFile(dir=reviewed, prefix=".export-", delete=False) as f:
        staged = Path(f.name)
        try:
            f.write((json.dumps(doc, sort_keys=True, indent=2) + "\n").encode())
            f.flush()
            os.link(staged, path)
        except OSError as exc:
            raise PrivateEvidenceError("reviewed export could not be published") from exc
        finally:
            staged.unlink(missing_ok=True)
    return path
