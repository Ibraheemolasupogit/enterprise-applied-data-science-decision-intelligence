"""Measure and enforce the read-only source and runtime used by governed P8."""

import hashlib
import os
import stat
import subprocess
import sys
from pathlib import Path

from .launch_approval import LaunchApprovalError, _trusted_path

_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def tree_digest(root: Path) -> str:
    """Hash a protected directory tree, including internal link targets by name."""
    if root.resolve() != root:
        raise LaunchApprovalError("release tree requires a canonical path")
    for ancestor in (*reversed(root.parents), root):
        _trusted_path(ancestor, directory=True)
    digest = hashlib.sha256()

    def unreadable(error: OSError) -> None:
        raise error

    for directory, dirs, files in os.walk(root, onerror=unreadable):
        for name in sorted([*dirs, *files]):
            path = Path(directory) / name
            info = path.lstat()
            if info.st_uid != 0:
                raise LaunchApprovalError("release tree must be root-owned")
            digest.update(os.fsencode(path.relative_to(root)) + b"\0")
            if stat.S_ISLNK(info.st_mode):
                target = path.resolve(strict=True)
                if not target.is_relative_to(root):
                    raise LaunchApprovalError("release tree contains an external redirect")
                digest.update(b"link\0" + os.fsencode(os.readlink(path)) + b"\0")
            elif info.st_mode & 0o022 or (os.geteuid() != 0 and os.access(path, os.W_OK)):
                raise LaunchApprovalError("release inputs must be immutable to the operator")
            elif stat.S_ISDIR(info.st_mode):
                digest.update(b"dir\0")
            elif stat.S_ISREG(info.st_mode):
                with path.open("rb") as stream:
                    digest.update(b"file\0" + hashlib.file_digest(stream, "sha256").digest())
            else:
                raise LaunchApprovalError("release tree contains an unsupported input")
        dirs.sort()
    return digest.hexdigest()


def binding(repo: Path) -> dict[str, str]:
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        raise LaunchApprovalError("governed execution requires isolated Python")
    _trusted_path(repo, directory=True)
    runtime = Path(sys.prefix).resolve()
    executable = Path(sys.executable).resolve()
    environment = {"HOME": str(Path.home()), "PATH": "/usr/bin:/bin", "LC_ALL": "C",
                   "TMPDIR": "/private/tmp", **dict.fromkeys(_THREAD_ENV, "1")}
    live_env = dict(os.environ)
    live_env.pop("_M11_P8_SUPERVISOR_FD", None)
    # Darwin supplies this per-UID CoreFoundation setting at exec.
    live_env.pop("__CF_USER_TEXT_ENCODING", None)
    if (Path(sys.base_prefix).resolve() != runtime or not executable.is_relative_to(runtime)
            or Path(sys.prefix) != runtime or Path(sys.executable) != executable
            or any(not Path(p).resolve().is_relative_to(runtime)
                   and Path(p).resolve() != repo / "src" for p in sys.path)
            or live_env != environment):
        raise LaunchApprovalError("governed Python must use its approved self-contained runtime")
    _trusted_path(repo / ".git", directory=True)
    _trusted_path(repo / ".git" / "config")
    build = subprocess.run(["/usr/bin/sw_vers", "-buildVersion"], check=True,
                           capture_output=True, text=True, timeout=5,
                           env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"}).stdout.strip()
    if not build:
        raise LaunchApprovalError("cannot identify the macOS build")
    return {"runtime": str(runtime), "interpreter": str(executable),
            "runtime_sha256": tree_digest(runtime), "macos_build": build,
            "source_sha256": tree_digest(repo / "src"),
            "scripts_sha256": tree_digest(repo / "scripts")}
