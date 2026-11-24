"""Milestone 11 P8 evidence: the A12 design (repeated sampling, primary comparisons).

One orchestrator, three modes (``P8_DRY_RUN``):

* ``--mode dry-run`` - development seeds, reduced size, no M10 comparability or fresh step.
* ``--mode rehearsal`` - the complete final sequence (verification, main, repeated sampling,
  M10 comparability, fresh seed last) on development seeds only, at reduced size.
* ``--mode final`` - the reserved seeds through :func:`authorise_seed` (refused while
  ``PENDING_BEFORE_MAIN`` is non-empty), every ``REPEATED_SAMPLING`` study at its declared size;
  requires ``--expect-manifest`` and ``--expect-code`` equal to the live protocol digests.

**Private execution.** A privileged supervisor validates the owner-pinned private history,
commits the consumption claim, and starts an unprivileged worker. The worker keeps the
execution ledger, raw pickles and reports in private staging. The fresh seed is authorised
and recorded only in the last analysis phase. The protected journal holds terminal receipts.
``--verify RUN_DIR`` re-checks an existing run. ``--export-reviewed ID`` separately stages
an explicitly pinned development resource summary for public review.

    python scripts/generate_m11_evidence.py --mode dry-run
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pickle  # noqa: E402
import platform  # noqa: E402
import stat  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable, Sequence  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from contextlib import closing, nullcontext  # noqa: E402
from dataclasses import asdict, dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

if sys.flags.isolated:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from decision_intelligence.common.failure_trace import (  # noqa: E402
    TracedTaskError,
    failure_trace,
)
from decision_intelligence.customer_value import (  # noqa: E402
    launch_approval,
    p8_candidate,
    p8_immutable,
    p8_journal,
    p8_private,
    p8_scheduler,
    resource_supervision,
)
from decision_intelligence.customer_value import predictive_cases as pc  # noqa: E402
from decision_intelligence.customer_value import predictive_protocol as pp  # noqa: E402
from decision_intelligence.customer_value.resource_supervision import (  # noqa: E402
    MonitorUnavailable,
    SupervisedChild,
    SupervisionResult,
    await_supervisor,
    supervise,
)

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / "docs" / "evidence"
DRY_RUN_DRAWS = 10
DRY_RUN_REFITS = 20
DRY_RUN_SEED_START = 6200
"""Dry run: main analysis 6200; R1/R2/R3 draw r uses 6201 + r; REFIT draw r uses 6301 + r."""
REGISTRY_DIR = EVIDENCE / "m11-p8-attempts"
"""Repository-level attempt registry (tracked in Git; one reservation per attempt)."""
A15_ATTEMPT_KEY = "final-a15"
"""Candidate key only. The protocol must separately authorise it before it can be used."""
HISTORICAL_FINAL_KEYS = ("final", "final-a14")
RAW_DIR = "raw"
"""Per-run subdirectory for raw pickled results (Git-ignored; SHA-256 kept in the ledger)."""
REHEARSAL_SEED_START = 6400
"""Rehearsal: main 6400, M10 comparability 6401, fresh 6402; draws 6411 + r; REFIT 6511 + r."""
PHASES = ("verify", "main", "repeated_sampling", "m10_comparability", "fresh", "record")
TaskFn = Callable[[pc.Task], dict[str, Any]]


class RunRefused(RuntimeError):
    """The run must not start (duplicate attempt, digest mismatch, unauthorised seed)."""


# --- seeds -------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SeedPlan:
    mode: str
    main: int
    m10_comparability: int | None
    fresh_step: bool
    draws: dict[str, list[int]]
    r3_scales: tuple[float, ...]
    refits: int


def reserved_seeds() -> set[int]:
    out = {pp.MAIN_SEED, pp.PILOT_SEED, pp.FRESH_SEED, pp.M10_COMPARABILITY_SEED,
           *pp.OTHER_PROTOCOL_SEEDS}
    out |= set(range(pp.COVERAGE_SEED_START, pp.COVERAGE_SEED_START + pp.COVERAGE_REPLICATIONS))
    out |= set(range(pp.REFIT_STUDY_SEED_START,
                     pp.REFIT_STUDY_SEED_START + pp.REFIT_STUDY_REPLICATIONS))
    for start, stop in pp.OTHER_COVERAGE_BLOCKS:
        out |= set(range(start, stop))
    return out


def check_development_seeds(seeds: Sequence[int]) -> None:
    reserved = reserved_seeds()
    bad = [s for s in seeds if not 0 <= s <= pp.DEVELOPMENT_SEED_MAX or s in reserved]
    if bad:
        raise ValueError(f"not development seeds: {bad}")


def _study(study_id: str) -> pp.RepeatedStudy:
    return next(s for s in pp.REPEATED_SAMPLING if s.id == study_id)


def dry_run_plan(draws: int = DRY_RUN_DRAWS, refits: int = DRY_RUN_REFITS) -> SeedPlan:
    base = [DRY_RUN_SEED_START + 1 + r for r in range(draws)]
    refit = [DRY_RUN_SEED_START + 101 + r for r in range(draws)]
    plan = SeedPlan("dry-run", DRY_RUN_SEED_START, None, False,
                    {"R1": base, "R2": base, "R3": base, "REFIT": refit},
                    _study("R3").scales, refits)
    check_development_seeds([*base, *refit, plan.main])
    return plan


def rehearsal_plan(draws: int = DRY_RUN_DRAWS, refits: int = DRY_RUN_REFITS) -> SeedPlan:
    s0 = REHEARSAL_SEED_START
    base = [s0 + 11 + r for r in range(draws)]
    refit = [s0 + 111 + r for r in range(draws)]
    plan = SeedPlan("rehearsal", s0, s0 + 1, True,
                    {"R1": base, "R2": base, "R3": base, "REFIT": refit},
                    _study("R3").scales, refits)
    check_development_seeds([*base, *refit, s0, s0 + 1, s0 + 2])
    return plan


def final_plan() -> SeedPlan:
    """The reserved seeds; ``authorise_seed`` refuses while any protocol item is pending. The
    fresh seed is not authorised here (only at the last phase)."""
    main = pp.authorise_seed("main")
    draws = {s.id: [s.seed_start + r for r in range(s.replications)]
             for s in pp.REPEATED_SAMPLING}
    return SeedPlan("final", main, pp.M10_COMPARABILITY_SEED, True, draws,
                    _study("R3").scales, pp.CLUSTERED_INFERENCE.refit_bootstrap_resamples)


def resolve_fresh_seed(plan: SeedPlan, *, ledger: Ledger | None = None) -> int:
    if plan.mode == "final":
        required = [("verify", "done"), ("main", "started"), ("main", "done"),
                    ("repeated_sampling", "started"), ("repeated_sampling", "done"),
                    ("m10_comparability", "started"), ("m10_comparability", "done")]
        if (ledger is None or ledger.data.get("status") != "started" or not plan.fresh_step
                or ledger.data.get("reservation") != final_attempt_key()
                or [(e["phase"], e["state"]) for e in ledger.data.get("events", [])] != required):
            raise RunRefused("fresh seed requires the active final ledger after M10 comparability")
        return pp.authorise_seed("fresh", confirm_single_final_run=True)
    seed = REHEARSAL_SEED_START + 2
    check_development_seeds([seed])
    return seed


def study_tasks(plan: SeedPlan) -> list[pc.Task]:
    tasks = [pc.Task("R1", s) for s in plan.draws["R1"]]
    tasks += [pc.Task("R2", s) for s in plan.draws["R2"]]
    tasks += [pc.Task("R3", s, scale) for scale in plan.r3_scales for s in plan.draws["R3"]]
    tasks += [pc.Task("REFIT", s, refits=plan.refits) for s in plan.draws["REFIT"]]
    return tasks


# --- execution ---------------------------------------------------------------------------------
class MemoryMonitor:
    """Peak aggregate RSS (GB) of this process and its children, sampled every 0.5 s."""

    def __init__(self) -> None:
        self.peak_gb = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        me = os.getpid()
        while not self._stop.is_set():
            try:
                out = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,rss="], capture_output=True,
                                     text=True, check=True).stdout
            except (OSError, subprocess.CalledProcessError):
                return
            total = sum(int(p[2]) for p in (line.split() for line in out.splitlines())
                        if len(p) == 3 and me in (int(p[0]), int(p[1])))
            self.peak_gb = max(self.peak_gb, total * 1024 / 1e9)
            time.sleep(0.5)

    def __enter__(self) -> MemoryMonitor:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()


def run_tasks(tasks: Sequence[pc.Task], workers: int, fn: TaskFn = pc.run_task
              ) -> list[dict[str, Any]]:
    """Results in task order; any failure propagates."""
    if workers <= 1:
        return [fn(t) for t in tasks]
    with mp.get_context("spawn").Pool(workers) as pool:
        return pool.map(fn, list(tasks), chunksize=1)


def run_supervised_tasks(tasks: Sequence[pc.Task], workers: int, fn: TaskFn
                         ) -> list[dict[str, Any]]:
    """Propagate abrupt worker death instead of replacing the worker and waiting on its task."""
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        return list(pool.map(fn, tasks, chunksize=1))


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


@dataclass
class Ledger:
    """The run's audit trail; one per run directory, never overwritten."""

    path: Path
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, run_dir: Path, header: dict[str, Any]) -> Ledger:
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "ledger.json"
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)  # atomic: one attempt
        except FileExistsError as exc:
            raise RunRefused(f"{path} exists: a run was already attempted here; it is never "
                             "restarted (archive it for the owner's review)") from exc
        os.close(fd)
        led = cls(path, {**header, "status": "started", "started": _now(), "events": []})
        led.save()
        return led

    def save(self, *, deadline: float | None = None) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, default=str))
        if deadline is not None and time.monotonic() >= deadline:
            tmp.unlink()
            raise RunRefused("time limit expired before completion could be recorded")
        tmp.replace(self.path)

    def event(self, phase: str, state: str, **info: Any) -> None:
        self.data["events"].append({"phase": phase, "state": state, "time": _now(), **info})
        self.save()

    def finish(self, status: str, **info: Any) -> None:
        self.data.update(status=status, finished=_now(), **info)
        self.save()


def protocol_header(plan: SeedPlan, workers: int, *, signed_checkout: bool = False
                    ) -> dict[str, Any]:
    return {"mode": plan.mode, "protocol_stage": pp.PROTOCOL_STAGE,
            "manifest_digest": pp.protocol_digest(), "code_digest": pp.protocol_code_digest(),
            "pending_before_main": list(pp.PENDING_BEFORE_MAIN),
            "git": git_state(signed_checkout=signed_checkout),
            "plan": {**asdict(plan), "draws": {k: [v[0], v[-1], len(v)]
                                               for k, v in plan.draws.items()}},
            "workers": workers, "python": sys.version.split()[0], "numpy": np.__version__}


def git_dirty_paths(repo: Path = REPO) -> list[str]:
    """Uncommitted or untracked paths (Git-ignored files excluded); raises if Git is unusable."""
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"],
                             cwd=repo, check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RunRefused(f"cannot read the Git working tree: {exc}") from exc
    return [line for line in out.splitlines() if line.strip()]


def refuse_real_final_under_test(plan_mode: str, fn: TaskFn) -> None:
    """A final run with real draws never starts inside a test process (pytest sets
    ``PYTEST_CURRENT_TEST``); tests may exercise final mode with fake draws only."""
    if plan_mode == "final" and fn is pc.run_task and os.environ.get("PYTEST_CURRENT_TEST"):
        raise RunRefused("a final run with real draws cannot start under pytest")


def require_clean_tree() -> None:
    """Final mode only: the working tree must be clean (everything committed) before any
    attempt is reserved or any reserved seed is consumed."""
    dirty = git_dirty_paths()
    if dirty:
        raise RunRefused(f"the Git working tree is not clean ({len(dirty)} paths, e.g. "
                         f"{dirty[:3]}); commit first: the final run starts from a commit")


def verify_protocol(mode: str, expect_manifest: str | None, expect_code: str | None,
                    *, signed_checkout: bool = False
                    ) -> None:
    problems = pp.validate_protocol()
    if problems:
        raise RunRefused(f"protocol invalid: {problems}")
    if mode == "final":
        if expect_manifest is None or expect_code is None:
            raise RunRefused("final mode needs --expect-manifest and --expect-code")
        if pp.PENDING_BEFORE_MAIN:
            raise RunRefused("protocol items are pending")
        if not signed_checkout:
            require_clean_tree()
        source = REPO / "src" / "decision_intelligence" / "customer_value"
        for module, filename in ((pp, "predictive_protocol.py"),
                                 (launch_approval, "launch_approval.py"),
                                 (p8_candidate, "p8_candidate.py"),
                                 (p8_immutable, "p8_immutable.py"),
                                 (p8_journal, "p8_journal.py"),
                                 (p8_private, "p8_private.py"),
                                 (p8_scheduler, "p8_scheduler.py"),
                                 (resource_supervision, "resource_supervision.py")):
            if module.__file__ is None or Path(module.__file__).resolve() != source / filename:
                raise RunRefused("final launch must import its controls from this checkout")
    for name, want, got in (("manifest", expect_manifest, pp.protocol_digest()),
                            ("code", expect_code, pp.protocol_code_digest())):
        if want is not None and want != got:
            raise RunRefused(f"{name} digest {got} differs from the approved {want}")


# --- repository-level attempt registry ----------------------------------------------------------
class AttemptRegistry:
    """Repository-wide single-attempt reservations, shared by every run directory.

    One immutable entry file per reservation key, created with ``O_CREAT | O_EXCL`` (atomic: of
    concurrent attempts exactly one succeeds), and a hash-chained ``attempts.log`` written under
    an exclusive lock. Before any reservation the whole registry is validated: an altered,
    deleted or unlogged entry, or a broken chain, refuses every further attempt. The final run's
    key is fixed by the protocol, regardless of output directory."""

    GENESIS = "0" * 64

    def __init__(self, root: Path) -> None:
        self.root = root
        self.log = root / "attempts.log"
        self.lock = root / ".lock"

    @staticmethod
    def _chain(prev: str, key: str, entry_sha256: str, when: str) -> str:
        return hashlib.sha256(f"{prev}|{key}|{entry_sha256}|{when}".encode()).hexdigest()

    def _acquire(self, timeout_s: float = 30.0) -> None:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                os.close(os.open(self.lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644))
                return
            except FileExistsError:
                if time.monotonic() > deadline:
                    raise RunRefused(f"{self.lock} is held: another attempt is running or "
                                     "crashed while reserving; the owner must review") from None
                time.sleep(0.01)

    def validate(self) -> list[str]:
        if not self.root.exists():
            return []
        problems: list[str] = []
        prev, logged = self.GENESIS, {}
        if self.log.is_symlink():
            return ["attempts.log is a symlink"]
        lines = self.log.read_text().splitlines() if self.log.exists() else []
        for i, line in enumerate(lines):
            try:
                rec = json.loads(line)
                key = rec["key"]
                ok = (isinstance(key, str) and bool(key) and Path(key).name == key
                      and rec["prev"] == prev and rec["hash"] == self._chain(
                          prev, key, rec["entry_sha256"], rec["time"]))
            except (ValueError, KeyError, TypeError):
                ok = False
            if not ok:
                problems.append(f"attempts.log line {i + 1}: broken hash chain")
                break
            prev = rec["hash"]
            if rec["key"] in logged:
                problems.append(f"reservation {rec['key']}: duplicate chain entry")
            logged[rec["key"]] = rec["entry_sha256"]
        for key, sha in logged.items():
            f = self.root / f"{key}.json"
            if f.is_symlink():
                problems.append(f"reservation {key}: entry is a symlink")
            elif not f.is_file():
                problems.append(f"reservation {key}: entry deleted")
            elif _sha256(f) != sha:
                problems.append(f"reservation {key}: entry altered")
        for f in sorted(self.root.glob("*.json")):
            if f.stem not in logged:
                problems.append(f"reservation {f.stem}: entry not in attempts.log")
        return problems

    def reserve(self, key: str, info: dict[str, Any], *,
                preflight: Callable[[], None] | None = None) -> str:
        """Reserve ``key`` once; returns the entry's SHA-256. Refuses a tampered registry or a
        key already reserved (whatever the output directory)."""
        self.root.mkdir(parents=True, exist_ok=True)
        self._acquire()
        try:
            problems = self.validate()
            if problems:
                raise RunRefused(f"attempt registry tampered: {problems}")
            if preflight is not None:
                preflight()
            path = self.root / f"{key}.json"
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            except FileExistsError as exc:
                raise RunRefused(f"reservation {key!r} already exists: this attempt was "
                                 "already made; it is never rerun") from exc
            with os.fdopen(fd, "w") as fh:
                fh.write(json.dumps({"key": key, "reserved": _now(), **info}, indent=1,
                                    default=str))
            sha = _sha256(path)
            lines = self.log.read_text().splitlines() if self.log.exists() else []
            prev = json.loads(lines[-1])["hash"] if lines else self.GENESIS
            when = _now()
            rec = {"key": key, "entry_sha256": sha, "time": when, "prev": prev,
                   "hash": self._chain(prev, key, sha, when)}
            with self.log.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            return sha
        finally:
            self.lock.unlink(missing_ok=True)


def final_attempt_key() -> str:
    """The reservation key of the authorised final attempt (A14: ``P8_FINAL_ATTEMPT_KEY``; the
    first attempt used ``final``)."""
    return str(pp.P8_FINAL_ATTEMPT_KEY)


def authorised_failed_attempts() -> dict[str, str]:
    """Failed final ledgers an amendment has recorded (path relative to ``docs/evidence`` ->
    SHA-256): only these may exist when the authorised new attempt starts."""
    return {str(path): str(sha) for path, sha in pp.P8_FAILED_ATTEMPTS}


def unauthorised_final_attempts(evidence_root: Path) -> list[Path]:
    allowed = authorised_failed_attempts()
    out = []
    for f in prior_final_attempts(evidence_root):
        rel = f.relative_to(evidence_root).as_posix()
        if allowed.get(rel) != _sha256(f):
            out.append(f)
    return out


def reservation_key(plan: SeedPlan) -> str:
    if plan.mode == "final":
        return final_attempt_key()
    digest = hashlib.sha256(json.dumps(asdict(plan), sort_keys=True).encode()).hexdigest()
    return f"{plan.mode}-{digest[:16]}"


def _evidence_walk_error(exc: OSError) -> None:
    raise RunRefused("cannot scan the evidence directory") from exc


def prior_final_attempts(evidence_root: Path) -> list[Path]:
    """Any ledger of a final run under ``evidence_root`` (whatever its directory)."""
    found: list[Path] = []
    if not evidence_root.exists():
        return found  # legacy temporary run directories can start empty
    for parent, dirs, files in os.walk(evidence_root, onerror=_evidence_walk_error):
        if RAW_DIR in dirs:
            dirs.remove(RAW_DIR)
        if "ledger.json" not in files:
            continue
        f = Path(parent) / "ledger.json"
        try:
            if f.is_symlink() or json.loads(f.read_text()).get("mode") == "final":
                found.append(f)
        except (OSError, ValueError, AttributeError):
            found.append(f)  # unreadable: treated as an attempt
    return found


def a15_gate_applies() -> bool:
    return any(amendment.id == "A15" for amendment in pp.PROTOCOL_AMENDMENTS)


def verify_a15_history(evidence_root: Path, reg: AttemptRegistry) -> None:
    """Check pinned historical metadata under the reservation lock. Raw results stay closed."""
    expected = {f"m11-p8-{key}/ledger.json": key for key in HISTORICAL_FINAL_KEYS}
    allowed = authorised_failed_attempts()
    if allowed.keys() != expected.keys():
        raise RunRefused("A15 requires both historical ledger pins")
    _verify_history(evidence_root, reg, expected, allowed)


def verify_successor_history(private: p8_private.PrivateEvidence) -> str:
    """Require the intact three-attempt chain for a successor, including the incomplete A15."""
    manifest, owner_pin = private.history_manifest()
    reg = AttemptRegistry(private.history_registry)
    problems = reg.validate()
    if problems:
        raise RunRefused(f"attempt registry tampered: {problems}")
    if not reg.log.is_file() or _sha256(reg.log) != p8_candidate.HISTORY_CHAIN_SHA256:
        raise RunRefused("consumed attempt chain differs from the successor history pin")
    lines = reg.log.read_bytes().splitlines(keepends=True)
    prefix = hashlib.sha256(b"".join(lines[:-1])).hexdigest()
    if prefix != p8_candidate.HISTORY_PREFIX_SHA256:
        raise RunRefused("consumed attempt chain prefix differs from the history pin")
    if manifest != {
        "schema": 1,
        "chain_sha256": p8_candidate.HISTORY_CHAIN_SHA256,
        "chain_prefix_sha256": prefix,
        "chain_tip": json.loads(lines[-1])["hash"],
        "attempts": {key: {"ledger_sha256": ledger, "reservation_sha256": entry}
                     for key, (ledger, entry) in p8_candidate.HISTORICAL_PINS.items()},
    }:
        raise RunRefused("owner manifest does not match the consumed attempt history")
    expected = {f"m11-p8-{key}/ledger.json": key for key in p8_candidate.HISTORICAL_PINS}
    _verify_history(private.history, reg, expected,
                    {rel: p8_candidate.HISTORICAL_PINS[key][0] for rel, key in expected.items()},
                    successor=True)
    return owner_pin


def _verify_history(evidence_root: Path, reg: AttemptRegistry, expected: dict[str, str],
                    allowed: dict[str, str], *, successor: bool = False) -> None:
    for rel, key in expected.items():
        ledger_path = evidence_root / rel
        entry_path = reg.root / f"{key}.json"
        for path in (ledger_path, entry_path):
            if path.resolve() != path.absolute() or not path.is_file():
                raise RunRefused(f"historical evidence missing or redirected: {path}")
        ledger_bytes, entry_bytes = ledger_path.read_bytes(), entry_path.read_bytes()
        if hashlib.sha256(ledger_bytes).hexdigest() != allowed[rel]:
            raise RunRefused(f"historical ledger altered: {rel}")
        try:
            led, entry = json.loads(ledger_bytes), json.loads(entry_bytes)
            entry_sha = hashlib.sha256(entry_bytes).hexdigest()
            if key == A15_ATTEMPT_KEY:
                last = led["events"][-1]
                stopped = (led["status"], last["phase"], last["state"]) == (
                    "started", "repeated_sampling", "started")
            else:
                stopped = (led["status"], led["phase"]) == ("failed", "repeated_sampling")
            matches = (
                stopped and (led["mode"], led["reservation"]) == ("final", key)
                and led["reservation_sha256"] == entry_sha
                and (not successor or entry_sha == p8_candidate.HISTORICAL_PINS[key][1])
                and (entry["key"], entry["mode"]) == (key, "final")
                # The historical absolute path belongs to the checkout that ran the attempt.
                and Path(entry["run_dir"]).name == ledger_path.parent.name
                and all(entry[d] == led[d] for d in ("manifest_digest", "code_digest"))
            )
        except (ValueError, TypeError, KeyError, IndexError):
            matches = False
        if not matches:
            raise RunRefused(f"historical failure or consumed reservation does not match: {key}")
        if not successor and evidence_root == EVIDENCE.resolve():
            for path, content in ((ledger_path, ledger_bytes), (entry_path, entry_bytes)):
                relpath = path.relative_to(REPO.resolve())
                committed = subprocess.run(["git", "show", f"HEAD:{relpath.as_posix()}"],
                                           cwd=REPO, capture_output=True, check=False)
                if committed.returncode != 0 or committed.stdout != content:
                    raise RunRefused(f"historical evidence differs from HEAD: {relpath}")
    historical_dirs = {(evidence_root / rel).parent for rel in expected}
    for parent, dirs, _ in os.walk(evidence_root, onerror=_evidence_walk_error):
        if RAW_DIR in dirs:
            if successor:
                raise RunRefused("private history replica must contain metadata only")
            dirs.remove(RAW_DIR)  # Never scan protected raw directories.
        for directory in dirs:
            if (Path(parent) / directory).is_symlink():
                raise RunRefused("redirected directory under canonical evidence root")
            if (directory.startswith("m11-p8-final")
                    and Path(parent) / directory not in historical_dirs):
                raise RunRefused("unexpected final-attempt directory")
    actual = {p.relative_to(evidence_root).as_posix() for p in prior_final_attempts(evidence_root)}
    if actual != expected.keys():
        raise RunRefused("unexpected or duplicate final-attempt ledger evidence")
    for path in reg.root.glob("*.json"):
        try:
            entry = json.loads(path.read_bytes())
            is_final = entry["mode"] == "final"
        except (ValueError, TypeError, KeyError) as exc:
            raise RunRefused(f"invalid registry entry: {path.name}") from exc
        if is_final and path.stem not in expected.values():
            raise RunRefused(f"unexpected final reservation: {path.stem}")


def _store(run_dir: Path, ledger: Ledger, phase: str, results: Any) -> None:
    """Raw results go to ``run_dir/raw`` (kept outside Git); the ledger keeps their SHA-256."""
    raw = run_dir / RAW_DIR
    raw.mkdir(exist_ok=True)
    path = raw / f"{phase}.pkl"
    staged = None
    try:
        with tempfile.NamedTemporaryFile(dir=raw, prefix=f".{phase}-", delete=False) as stream:
            staged = Path(stream.name)
            pickle.dump(results, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(staged, path)
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)
    ledger.event(phase, "done", file=f"{RAW_DIR}/{path.name}", sha256=_sha256(path))


def _hybrid_results(run_dir: Path, tasks: Sequence[pc.Task], fn: TaskFn) -> list[dict[str, Any]]:
    """Spool while workers are live. Rebuild the legacy list after all pools exit."""
    with tempfile.TemporaryFile(dir=run_dir) as spool:
        with closing(p8_scheduler.hybrid_tasks(tasks, fn)) as results:
            for result, task in zip(results, tasks, strict=True):
                if "task" in result and result["task"] != task:
                    raise RunRefused("repeated-sampling result differs from the ordered task plan")
                pickle.dump(result, spool, protocol=pickle.HIGHEST_PROTOCOL)
        spool.seek(0)
        return [pickle.load(spool) for _ in tasks]


def _inline_task(task: pc.Task, fn: TaskFn) -> list[dict[str, Any]]:
    return [fn(task)]


ReportFn = Callable[[SeedPlan, dict[str, Any], dict[str, Any], Path, "Path | None"], list[Path]]


def require_final_target(root: Path, registry: Path, run_dir: Path, output: Path | None,
                         *, attempt: str = A15_ATTEMPT_KEY) -> None:
    canonical_registry = root / REGISTRY_DIR.name
    canonical_run = root / f"m11-p8-{attempt}"
    if (registry.absolute() != canonical_registry or registry.resolve() != canonical_registry
            or run_dir.absolute() != canonical_run or run_dir.resolve() != canonical_run
            or output is not None):
        raise RunRefused(f"requires the canonical registry and {attempt} output directory")


def require_successor_identity(attempt: str) -> None:
    try:
        p8_candidate.require_new_identity(attempt)
    except launch_approval.LaunchApprovalError as exc:
        raise RunRefused(str(exc)) from exc


def execute(plan: SeedPlan, run_dir: Path, workers: int, *, fn: TaskFn = pc.run_task,
            registry: Path = REGISTRY_DIR, evidence_root: Path = EVIDENCE,
            expect_manifest: str | None = None, expect_code: str | None = None,
            output: Path | None = None, report_fn: ReportFn | None = None,
            resources: SupervisedChild | None = None,
            private: p8_private.PrivateEvidence | None = None) -> Path:
    """Execute only a validated development plan. Governed execution has its own entry point."""
    refuse_real_final_under_test(plan.mode, fn)
    if plan.mode not in ("dry-run", "rehearsal"):
        raise RunRefused("governed execution cannot accept a supplied plan or callbacks")
    seeds = [plan.main, *(seed for block in plan.draws.values() for seed in block)]
    if plan.m10_comparability is not None:
        seeds.append(plan.m10_comparability)
    check_development_seeds(seeds)
    verify_protocol(plan.mode, expect_manifest, expect_code)
    key = reservation_key(plan)
    fixture = fn is not pc.run_task and bool(os.environ.get("PYTEST_CURRENT_TEST"))
    if private is not None or not fixture:
        private = private or p8_private.open_private_evidence(REPO)
        private.check_target(key, run_dir, registry, output)
        if run_dir.exists():
            raise RunRefused("private run directory already exists and cannot be reused")
    else:
        for path in (run_dir, registry, output):
            if path is not None:
                p8_private.require_outside_checkout(path, REPO)
    entry = _reserve(plan, run_dir, workers, registry, key, private=private)
    return _execute_phases(plan, run_dir, workers, entry, fn, report_fn or write_report,
                           output, resources)


def _reserve(plan: SeedPlan, run_dir: Path, workers: int, registry: Path, key: str, *,
             private: p8_private.PrivateEvidence | None = None,
             claim: dict[str, Any] | None = None) -> Ledger:
    entry = {"run_dir": str(run_dir), "mode": plan.mode, "manifest_digest": pp.protocol_digest(),
             "code_digest": pp.protocol_code_digest()}
    if claim is not None:
        entry.update(claim_sha256=claim["sha256"],
                     history_manifest_sha256=claim["payload"]["history_manifest_sha256"])
    entry_sha = AttemptRegistry(registry).reserve(key, entry)
    if private is not None:
        run_dir.mkdir(mode=0o700)
    header = {**protocol_header(plan, workers, signed_checkout=claim is not None),
              "registry": str(registry),
              "reservation": key, "reservation_sha256": entry_sha}
    if claim is not None:
        header["claim_sha256"] = claim["sha256"]
    return Ledger.create(run_dir, header)


def _execute_phases(plan: SeedPlan, run_dir: Path, workers: int, ledger: Ledger,
                    fn: TaskFn, report_fn: ReportFn, output: Path | None,
                    resources: SupervisedChild | None) -> Path:
    """Compute phases after the caller has established a durable reservation."""
    ledger.event("verify", "done")
    phase = "main"
    t0 = time.perf_counter()
    results: dict[str, Any] = {}
    hybrid = resources is not None and workers == 3
    isolated = p8_scheduler.isolated_task if hybrid else _inline_task
    try:
        with (nullcontext(resources) if resources is not None else MemoryMonitor()) as mon:
            ledger.event(phase, "started", seed=plan.main)
            results["main"] = isolated(pc.Task("MAIN", plan.main), fn)
            _store(run_dir, ledger, phase, results["main"])
            phase = "repeated_sampling"
            tasks = study_tasks(plan)
            ledger.event(phase, "started", tasks=len(tasks))
            if hybrid:
                results["repeated_sampling"] = _hybrid_results(run_dir, tasks, fn)
            else:
                dispatch = (run_supervised_tasks if resources is not None and workers > 1
                            else run_tasks)
                results["repeated_sampling"] = dispatch(tasks, workers, fn)
            _store(run_dir, ledger, phase, results["repeated_sampling"])
            if plan.m10_comparability is not None:
                phase = "m10_comparability"
                ledger.event(phase, "started", seed=plan.m10_comparability)
                results["m10_comparability"] = isolated(pc.Task("M10", plan.m10_comparability), fn)
                _store(run_dir, ledger, phase, results["m10_comparability"])
            if plan.fresh_step:
                phase = "fresh"
                fresh = resolve_fresh_seed(plan, ledger=ledger)
                ledger.event(phase, "seed consumed", seed=fresh)  # before any use: once only
                results["fresh"] = isolated(pc.Task("MAIN", fresh), fn)
                _store(run_dir, ledger, phase, results["fresh"])
            phase = "record"
            wall = time.perf_counter() - t0
            perf: dict[str, Any] = {"wall": wall, "peak_gb": mon.peak_gb, "workers": workers}
            if resources is not None:
                perf["resource_snapshot"] = "before report generation"
            if hybrid:
                perf["r2_workers"] = 2
                perf["r2_pool_tasks"] = 2
            files = report_fn(plan, results, perf, run_dir, output)
            ledger.event(phase, "done", files=[
                {"file": str(f.relative_to(run_dir)) if f.is_relative_to(run_dir) else str(f),
                 "sha256": _sha256(f)} for f in files])
    except KeyboardInterrupt:
        ledger.finish("interrupted", phase=phase)
        raise
    except BaseException as exc:
        # Exception messages can contain outcomes. Only our task wrapper has a safe message.
        error = str(exc) if isinstance(exc, TracedTaskError) else type(exc).__name__
        ledger.finish("failed", phase=phase, error=error, failure_trace=failure_trace(exc))
        raise
    if resources is None:
        ledger.finish("complete", wall_s=wall, peak_gb=mon.peak_gb)
    else:
        # Only the parent can certify completion after monitoring the runner's entire lifetime.
        ledger.finish("supervision_pending", supervisor_pid=os.getppid())
    return files[0]


def _check_file(run_dir: Path, name: str, sha: str, label: str, require: bool) -> list[str]:
    f = Path(name) if Path(name).is_absolute() else run_dir / name
    if not f.exists():
        return [f"{label}: {f.name} missing"] if require else []
    return [] if _sha256(f) == sha else [f"{label}: {f.name} altered"]


def phase_sequence_ok(data: dict[str, Any]) -> bool:
    try:
        plan, events = data["plan"], data["events"]
        expected = [("verify", "done"), ("main", "started"), ("main", "done"),
                    ("repeated_sampling", "started"), ("repeated_sampling", "done")]
        if plan["m10_comparability"] is not None:
            expected.extend([("m10_comparability", "started"), ("m10_comparability", "done")])
        if data.get("mode") == "final" and (not plan["fresh_step"]
                                           or plan["m10_comparability"] is None):
            return False
        if plan["fresh_step"]:
            expected.extend([("fresh", "seed consumed"), ("fresh", "done")])
        expected.append(("record", "done"))
        return [(event["phase"], event["state"]) for event in events] == expected
    except (TypeError, KeyError):
        return False


def _phase_digest(data: dict[str, Any]) -> str:
    fixed = {k: data[k] for k in (
        "mode", "plan", "events", "registry", "reservation", "reservation_sha256", "claim_sha256")}
    return hashlib.sha256(json.dumps(fixed, sort_keys=True).encode()).hexdigest()


def _journal_path() -> Path:
    return launch_approval._APPROVAL_DIR / "consumption.jsonl"


def _certified(rows: list[dict[str, Any]], attempt: str) -> bool:
    if len(rows) != 3 or rows[-1]["attempt"] != attempt or rows[-1]["state"] != "certified":
        return False
    elapsed = rows[-1]["payload"].get("elapsed_s")
    peak = rows[-1]["payload"].get("peak_bytes")
    return (type(elapsed) in (float, int) and 0 <= elapsed < pp.BUDGETS.evidence_ceiling_min * 60
            and type(peak) is int and 0 < peak <= int(pp.BUDGETS.evidence_memory_gb * 1e9))


def verify_run(run_dir: Path, *, require_raw: bool = True) -> list[str]:
    """Problems with a run: incomplete status or phases, altered or missing files, and a
    reservation that is missing or differs from the ledger. ``require_raw=False`` accepts a
    clone without the raw (Git-ignored) results; their hashes are still compared if present."""
    path = run_dir / "ledger.json"
    if not path.exists():
        return ["no ledger"]
    led = json.loads(path.read_text())
    problems = [] if led.get("status") == "complete" else [f"status {led.get('status')}"]
    if not phase_sequence_ok(led):
        problems.append("invalid phase sequence or fresh consumption")
        return problems
    identity = run_dir.name.removeprefix("m11-p8-")
    governed = ((identity.startswith("final-") and identity not in p8_candidate.HISTORICAL_PINS)
                or (led.get("mode") == "final"
                    and led.get("reservation") not in p8_candidate.HISTORICAL_PINS))
    if governed and not led.get("claim_sha256"):
        return [*problems, "protected supervisor receipt missing or mismatched"]
    if governed or led.get("claim_sha256"):
        try:
            with p8_journal.open_journal(_journal_path()) as journal:
                rows = [row for row in journal.records() if row["attempt"] == led["reservation"]]
            confirmed = (_certified(rows, led["reservation"])
                         and rows[0]["payload"]["run_dir"] == str(run_dir.resolve())
                         and rows[0]["sha256"] == led["claim_sha256"]
                         and rows[1]["payload"]["phase_sha256"] == _phase_digest(led))
        except (OSError, p8_journal.JournalError, launch_approval.LaunchApprovalError,
                TypeError, KeyError):
            confirmed = False
        if not confirmed:
            return [*problems, "protected supervisor receipt missing or mismatched"]
        if led["status"] == "supervision_pending":
            problems = []
    done = {e["phase"]: e for e in led["events"] if e["state"] == "done"}
    needed = ["verify", "main", "repeated_sampling", "record"]
    if led["plan"]["m10_comparability"] is not None:
        needed.append("m10_comparability")
    if led["plan"]["fresh_step"]:
        needed.append("fresh")
    problems += [f"phase {p} incomplete" for p in needed if p not in done]
    for p, e in done.items():
        if "sha256" in e:
            problems += _check_file(run_dir, e["file"], e["sha256"], f"phase {p}", require_raw)
        for f in e.get("files", []):
            problems += _check_file(run_dir, f["file"], f["sha256"], f"phase {p}", True)
    reg = AttemptRegistry(Path(led.get("registry", REGISTRY_DIR)))
    entry = reg.root / f"{led.get('reservation')}.json"
    if not entry.exists() or _sha256(entry) != led.get("reservation_sha256"):
        problems.append("reservation missing or altered")
    problems += [f"registry: {q}" for q in reg.validate()]
    return problems


# --- analysis and record ------------------------------------------------------------------------
def analyse(results: dict[str, Any]) -> dict[str, Any]:
    rs = results["repeated_sampling"]
    by = {k: [r for r in rs if r["task"].study == k] for k in ("R1", "R2", "R3", "REFIT")}
    main = results["main"][0]
    pc2 = pc.aggregate_pc2(by["R1"])
    p = pc.primary_p_values(main["tests"], pc2["p_value"])
    bias, slope = pc.aggregate_gate4(by["R3"])
    out = {"main": main, "pc2": pc2, "p": p, "holm": pc.holm(p),
           "oracle_gate": pc.aggregate_oracle_gate(by["R1"]), "gate4_bias": bias,
           "gate4_slope": slope, "r2": pc.aggregate_coverage(by["R2"]),
           "refit": pc.aggregate_coverage(by["REFIT"]), "tasks": rs}
    if "m10_comparability" in results:
        out["m10"] = results["m10_comparability"][0]
    if "fresh" in results:
        out["fresh"] = results["fresh"][0]
    return out


def extrapolate(tasks: Sequence[dict[str, Any]], main_s: float, wall: float, workers: int,
                plan: SeedPlan) -> pd.DataFrame:
    busy = sum(r["seconds"] for r in tasks)
    eff = min(1.0, busy / (max(wall - main_s, 1e-9) * max(workers, 1)))
    full = {s.id: s.replications * (len(s.scales) if s.id == "R3" else 1)
            for s in pp.REPEATED_SAMPLING}
    refit_scale = pp.CLUSTERED_INFERENCE.refit_bootstrap_resamples / max(plan.refits, 1)
    rows, total = [], 0.0
    for study in ("R1", "R2", "R3", "REFIT"):
        secs = [r["seconds"] for r in tasks if r["task"].study == study]
        per = float(np.mean(secs)) * (refit_scale if study == "REFIT" else 1.0)
        wall_full = per * full[study] / (workers * eff)
        total += wall_full
        rows.append([study, len(secs), float(np.mean(secs)), full[study], wall_full / 60.0])
    rows.append(["main, M10, fresh", 1, main_s, 3, 3 * main_s / 60.0])
    total += 3 * main_s
    rows.append(["total", len(tasks) + 1, math.nan, math.nan, total / 60.0])
    out = pd.DataFrame(rows, columns=["study", "tasks run", "mean task s", "full-size tasks",
                                      "extrapolated wall min"])
    out.attrs["efficiency"] = eff
    return out


def table(frame: pd.DataFrame, digits: int = 4) -> str:
    def cell(v: object) -> str:
        if isinstance(v, float):
            return "nan" if not math.isfinite(v) else f"{v:.{digits}g}"
        return str(v)
    head = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    sep = "|" + "---|" * len(frame.columns)
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def git_state(*, signed_checkout: bool = False) -> str:
    try:
        git = (["/usr/bin/git", "-c", f"safe.directory={REPO}"] if signed_checkout else ["git"])
        head = subprocess.run([*git, "rev-parse", "--short", "HEAD"], cwd=REPO, check=True,
                              capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run([*git, "status", "--porcelain"], cwd=REPO, check=True,
                               capture_output=True, text=True).stdout.strip().splitlines()
        return f"{head} + {len(dirty)} uncommitted or untracked paths" if dirty else head
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _tests_table(analysis: dict[str, Any], tests: dict[str, Any], with_pc2: bool
                 ) -> pd.DataFrame:
    rows = []
    for k in ("PC1", "PC2", "PC3", "PC4", "PC5"):
        if k == "PC2":
            if not with_pc2:
                continue
            q = analysis["pc2"]
            rows.append([k, q["mean S"], q["ci_lower"], q["ci_upper"], analysis["p"][k],
                         analysis["holm"][k], f"t test, {int(q['draws'])} draws, "
                                              f"{int(q['excluded'])} excluded"])
        elif tests[k] is None:
            rows.append([k, math.nan, math.nan, math.nan, 1.0,
                         analysis["holm"][k] if with_pc2 else math.nan,
                         "non-evaluable (no Tier A model selected at either cutoff)"])
        else:
            t = tests[k]
            rows.append([k, t.estimate, t.ci_lower, t.ci_upper, t.p_value,
                         analysis["holm"][k] if with_pc2 else math.nan,
                         f"z {t.z:.2f}; {t.rows} rows, {t.customers} customers, "
                         f"{t.excluded} excluded"])
    return pd.DataFrame(rows, columns=["comparison", "estimate", "95% lower", "95% upper", "p",
                                       "Holm p", "note"])


def _selection_table(selection: Sequence[Any]) -> pd.DataFrame:
    return pd.DataFrame([[s, c, t, r.status, r.selected or "-", r.reference_fallback or "-",
                          "; ".join(f"{m} ({w})" for m, w in r.excluded) or "-"]
                         for s, c, t, r in selection],
                        columns=["scenario", "cutoff", "tier", "status", "selected",
                                 "reference", "excluded"])


def render(plan: SeedPlan, results: dict[str, Any], perf: dict[str, Any]) -> str:
    a = analyse(results)
    label = {"dry-run": "DRY RUN - NOT EVIDENCE", "rehearsal": "REHEARSAL - NOT EVIDENCE",
             "final": "FINAL"}[plan.mode]
    lines = [f"# Milestone 11 P8 record ({label})", ""]
    if plan.mode != "final":
        lines += [f"Generated by `python scripts/generate_m11_evidence.py --mode {plan.mode}`. "
                  "**Development seeds only**; every statistic exercises the A12 code path at "
                  "reduced size and is **not evidence**.", ""]
    lines += [table(pd.DataFrame([
        ["mode", plan.mode], ["protocol stage", pp.PROTOCOL_STAGE],
        ["protocol manifest digest", f"`{pp.protocol_digest()}`"],
        ["protocol code digest", f"`{pp.protocol_code_digest()}`"],
        ["pending before main", str(len(pp.PENDING_BEFORE_MAIN))],
        ["main-analysis seed", str(plan.main)],
        ["M10 comparability seed", str(plan.m10_comparability)],
        ["fresh step", str(plan.fresh_step)],
        ["draw seeds", f"R1/R2/R3 {plan.draws['R1'][0]}-{plan.draws['R1'][-1]}; REFIT "
                       f"{plan.draws['REFIT'][0]}-{plan.draws['REFIT'][-1]}"],
        ["draws per study", ", ".join(f"{k} {len(v)}" for k, v in plan.draws.items())],
        ["R3 scales", ", ".join(f"{s:g}" for s in plan.r3_scales)],
        ["refits per REFIT draw", str(plan.refits)],
        ["workers / threads", f"{perf['workers']} / 1"], ["git", git_state()],
        ["python / numpy / pandas", f"{sys.version.split()[0]} / {np.__version__} / "
                                    f"{pd.__version__}"],
        ["platform", platform.platform()]], columns=["item", "value"])), ""]
    ex = extrapolate(a["tasks"], results["main"][0]["seconds"], perf["wall"], perf["workers"],
                     plan)
    lines += ["## 1. Performance", "",
              f"Wall time {perf['wall'] / 60:.2f} min; peak aggregate memory "
              f"{perf['peak_gb']:.2f} GB (budget {pp.BUDGETS.evidence_memory_gb:g} GB); parallel "
              f"efficiency {ex.attrs['efficiency']:.2f}. Full-size extrapolation (`BUDGET_SCOPE`): "
              f"ceiling {pp.BUDGETS.evidence_ceiling_min:g} min.", "", table(ex), ""]
    if "resource_snapshot" in perf:
        lines += ["The report's resource figures are a pre-report snapshot. The external "
                  "supervisor measures through report and ledger completion.", ""]
    if "r2_pool_tasks" in perf:
        lines += ["Hybrid execution uses isolated main, M10 and fresh analyses, with two workers "
                  "per fresh R2 pair and three workers in other studies. The extrapolation "
                  "assumes a uniform pool size across studies.", ""]
    lines += ["## 2. Primary comparisons (PC_TEST, PC_DEFINITIONS; Holm across PC1-PC5)", "",
              f"PC3 comparator by cutoff: {a['main']['pc3_comparator']}.", "",
              table(_tests_table(a, a["main"]["tests"], True)), "",
              "## 3. Selection (main analysis)", "",
              table(_selection_table(a["main"]["selection"])), "",
              "## 4. ORACLE_GATE (R1)", "", table(a["oracle_gate"]), "",
              "## 5. ACCEPTANCE_GATE_4 (R3)", "", table(a["gate4_bias"]), "",
              table(a["gate4_slope"]), ""]
    keep = ["setting", "model", "interval", "level", "draws", "mean coverage", "95% lower",
            "95% upper", "classification", "pooled coverage (descriptive)"]
    lines += ["## 6. R2 coverage (draw-level rule; reported, never used for recalibration)", "",
              table(a["r2"].loc[a["r2"]["group"] == "all", keep]), "",
              "## 7. REFIT (E2 only; E3 not performed: `E3_REFIT_DISCLOSURE`)", "",
              table(a["refit"].loc[a["refit"]["group"] == "all", keep]), ""]
    if "m10" in a:
        lines += ["## 8. M10 comparability replication (descriptive; not a Holm member)", "",
                  table(_tests_table(a, a["m10"]["tests"] | {"PC3": None, "PC4": None,
                                                             "PC5": None}, False).iloc[:1]),
                  "", table(_selection_table(a["m10"]["selection"])), ""]
    if "fresh" in a:
        lines += ["## 9. Fresh seed (evaluated once, last)", "",
                  table(_tests_table(a, a["fresh"]["tests"], False)), "",
                  table(_selection_table(a["fresh"]["selection"])), ""]
    lines += ["## Disclosures", "", f"- {pp.SHARED_SEEDS_DISCLOSURE}.",
              f"- {pp.E3_REFIT_DISCLOSURE}.", f"- {pp.E4_ACCELERATED_SIMULATION}.",
              "- Coverage: mean per-draw coverage with its 95% t interval across independent "
              "draws, classified below / consistent with / above nominal; pooled customer "
              "coverage is descriptive.", ""]
    return "\n".join(lines)


def _jsonable(v: Any) -> Any:
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, (np.floating, np.integer)):
        return _jsonable(v.item())
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def _tests_json(tests: dict[str, Any]) -> dict[str, Any]:
    return {k: (None if t is None else asdict(t)) for k, t in tests.items()}


def write_report(plan: SeedPlan, results: dict[str, Any], perf: dict[str, Any], run_dir: Path,
                 output: Path | None) -> list[Path]:
    """The record (Markdown) plus portable summaries: ``summary.json`` and CSV tables. Raw
    pickles stay in ``raw/`` (Git-ignored); their hashes are in the ledger."""
    a = analyse(results)
    out = output or (run_dir / "record.md")
    files = [out]
    tables = {"oracle_gate": a["oracle_gate"], "gate4_bias": a["gate4_bias"],
              "gate4_slope": a["gate4_slope"], "coverage_r2": a["r2"],
              "coverage_refit": a["refit"],
              "primary_comparisons": _tests_table(a, a["main"]["tests"], True),
              "selection": _selection_table(a["main"]["selection"])}
    if "fresh" in a:
        tables["fresh_comparisons"] = _tests_table(a, a["fresh"]["tests"], False)
        tables["fresh_selection"] = _selection_table(a["fresh"]["selection"])
    if "m10" in a:
        tables["m10_selection"] = _selection_table(a["m10"]["selection"])
    for name, frame in tables.items():
        path = run_dir / f"{name}.csv"
        frame.to_csv(path, index=False, float_format="%.10g")
        files.append(path)
    summary = {
        "mode": plan.mode, "label": "final" if plan.mode == "final" else "not evidence",
        "protocol": {"stage": pp.PROTOCOL_STAGE, "manifest_digest": pp.protocol_digest(),
                     "code_digest": pp.protocol_code_digest()},
        "plan": {**asdict(plan), "draws": {k: [v[0], v[-1], len(v)]
                                           for k, v in plan.draws.items()}},
        "performance": perf, "pc3_comparator": a["main"]["pc3_comparator"],
        "primary_tests": _tests_json(a["main"]["tests"]), "pc2": a["pc2"], "p": a["p"],
        "holm": a["holm"],
        "fresh_tests": _tests_json(a["fresh"]["tests"]) if "fresh" in a else None,
        "m10_comparability_tests": _tests_json(a["m10"]["tests"]) if "m10" in a else None,
        "tables": {name: tables[name].to_dict(orient="records") for name in tables}}
    path = run_dir / "summary.json"
    path.write_text(json.dumps(_jsonable(summary), indent=1, sort_keys=True))
    files.append(path)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(render(plan, results, perf))
    tmp.replace(out)
    return files


def _run_under_supervision(argv: Sequence[str], run_dir: Path, *,
                           private: p8_private.PrivateEvidence | None = None) -> int:
    receipt_dir = None
    if private is not None:
        private.check_target(run_dir.name.removeprefix("m11-p8-"), run_dir, private.registry, None)
        receipt_dir = private.supervisor_dir()
    timeout_s = pp.BUDGETS.evidence_ceiling_min * 60
    deadline = time.monotonic() + timeout_s
    result = supervise([sys.executable, str(Path(__file__).resolve()), *argv,
                        "--_supervised-child"], timeout_s=timeout_s,
                       memory_bytes=int(pp.BUDGETS.evidence_memory_gb * 1e9))
    if receipt_dir is not None:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=receipt_dir,
                                         prefix=f"{run_dir.name}-", suffix=".pending") as receipt:
            json.dump({**asdict(result), "run": run_dir.name, "reported_at": _now()}, receipt)
            receipt.write("\n")
            receipt.flush()
            os.link(receipt.name, Path(receipt.name).with_suffix(".json"))
    if result.status == "complete":
        path = run_dir / "ledger.json"
        data = json.loads(path.read_text())
        if data.get("status") != "supervision_pending" or data.get("supervisor_pid") != os.getpid():
            raise RunRefused("no matching supervised completion ledger")
        data.update(status="complete", finished=_now(), wall_s=result.elapsed_s,
                    peak_gb=result.peak_bytes / 1e9)
        Ledger(path, data).save(deadline=deadline)
    print(f"supervision {result.status}, elapsed {result.elapsed_s:.3f} s, "
          f"sampled peak RSS {result.peak_bytes} bytes",
          file=sys.stdout if result.status == "complete" else sys.stderr)
    if result.status == "interrupted":
        return 130
    return 0 if result.status == "complete" else 1


def _pending_ledger(run_dir: Path, worker_uid: int) -> dict[str, Any]:
    """Accept bounded metadata only and never follow a worker-controlled symlink."""
    directory = os.open(run_dir, os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fd = os.open("ledger.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != worker_uid or info.st_nlink != 1
                    or info.st_size > 1_048_576):
                raise RunRefused("invalid pending ledger file")
            body = stream.read(1_048_577)
        if len(body) > 1_048_576:
            raise RunRefused("pending ledger exceeds size ceiling")
        data = json.loads(body, object_pairs_hook=launch_approval._unique_fields)
        if not isinstance(data, dict):
            raise RunRefused("invalid pending ledger object")
        return data
    finally:
        os.close(directory)


def _consume_and_supervise(journal: p8_journal.Journal, attempt: str, command: Sequence[str],
                           run_dir: Path, history_pin: str, *, timeout_s: float,
                           memory_bytes: int, approval_sha256: str,
                           credentials: tuple[int, int] | None = None) -> int:
    """Commit a claim before dispatch and a terminal receipt before certification."""
    start = time.monotonic()
    deadline = start + timeout_s
    claim = journal.append(attempt, "consumed", {
        "supervisor_pid": os.getpid(), "run_dir": str(run_dir),
        "manifest_digest": pp.protocol_digest(), "code_digest": pp.protocol_code_digest(),
        "history_manifest_sha256": history_pin, "approval_sha256": approval_sha256,
        "started_at": _now(),
    })
    try:
        if time.monotonic() >= deadline:
            result = SupervisionResult("time_limit", None, time.monotonic() - start, 0)
        else:
            result = supervise(command, timeout_s=timeout_s, memory_bytes=memory_bytes,
                               credentials=credentials, deadline=deadline)
        receipt = asdict(result)
        if result.status == "complete":
            candidate = _pending_ledger(run_dir, credentials[0] if credentials else os.geteuid())
            if (candidate.get("status") != "supervision_pending"
                    or candidate.get("supervisor_pid") != os.getpid()
                    or candidate.get("reservation") != attempt
                    or candidate.get("claim_sha256") != claim
                    or not phase_sequence_ok(candidate)
                    or result.returncode != 0):
                raise RunRefused("governed completion could not be verified")
            receipt["phase_sha256"] = _phase_digest(candidate)
            receipt["status"] = "supervision_complete"
    except KeyboardInterrupt:
        journal.append(attempt, "terminal", {"claim_sha256": claim,
                       "status": "interrupted", "reported_at": _now()})
        return 130
    except (OSError, TypeError, ValueError, KeyError, RecursionError, RunRefused,
            launch_approval.LaunchApprovalError) as exc:
        journal.append(attempt, "terminal", {"claim_sha256": claim,
                       "status": "certification_failure", "reported_at": _now(),
                       "error": type(exc).__name__})
        raise
    terminal = journal.append(attempt, "terminal", {
        **receipt, "claim_sha256": claim, "reported_at": _now(),
    })
    status, peak = result.status, result.peak_bytes
    if status == "complete":
        try:
            peak = max(peak, resource_supervision.DarwinRSS()._rss(os.getpid()))
        except (OSError, MonitorUnavailable):
            status = "monitor_failure"
        if status == "complete" and peak > memory_bytes:
            status = "memory_limit"
    elapsed = time.monotonic() - start
    if status == "complete":
        if elapsed >= timeout_s:
            status = "time_limit"
        else:
            # The elapsed value includes durable receipt I/O. This row publishes the decision.
            journal.append(attempt, "certified", {"terminal_sha256": terminal,
                           "elapsed_s": elapsed, "peak_bytes": peak, "reported_at": _now()})
    print(f"supervision {status}, elapsed {elapsed:.3f} s, sampled peak RSS {peak} bytes")
    return 130 if status == "interrupted" else (0 if status == "complete" else 1)



def _bounded_staging_mode(path: Path) -> str:
    """Read staging metadata without following symlinks or accepting oversized files."""
    import stat

    limit = 1_048_576
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)

    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise RunRefused("invalid or oversized staging metadata")

        body = stream.read(limit + 1)
        if len(body) > limit:
            raise RunRefused("staging metadata exceeds size ceiling")

    data = json.loads(body)
    if not isinstance(data, dict):
        raise RunRefused("invalid staging metadata object")
    mode = data.get("mode")
    if not isinstance(mode, str):
        raise RunRefused("invalid staging metadata mode")
    return mode


def _require_private_available(layout: p8_private.PrivateEvidence, run_dir: Path) -> None:
    registry = AttemptRegistry(layout.registry)
    prior_keys = {p8_candidate.A20_ATTEMPT} if run_dir == layout.run_dir(
        p8_candidate.A21_ATTEMPT) else set()
    prior_ledgers = {layout.run_dir(key) / "ledger.json" for key in prior_keys}
    if (registry.validate() or run_dir.exists()
            or set(prior_final_attempts(layout.staging)) != prior_ledgers
            or any(path.stem not in prior_keys and _bounded_staging_mode(path) == "final"
                   for path in registry.root.glob("*.json"))):
        raise RunRefused("private history or registry is invalid, or a successor already exists")


_PREFLIGHT_ERRORS = {
    21: "native supervisor handshake failed",
    22: "dedicated worker identity or private staging check failed",
    23: "worker cannot verify .git/index; restore root-owned worker-readable permissions",
    24: "worker could not verify the pinned historical records or unused A21 journal state",
    25: "worker rejected the immutable release, Git status or signed launch binding",
}


def _preflight_child() -> int:
    """Read-only check on the real supervisor boundary, without a live claim."""
    stage = 21
    try:
        await_supervisor()
        stage = 22
        pp.require_successor_amendment()
        private = p8_private.open_private_evidence(REPO)
        worker = private.root.lstat()
        if os.geteuid() == 0 or (os.geteuid(), os.getegid()) != (worker.st_uid, worker.st_gid):
            return stage
        attempt = final_attempt_key()
        _require_private_available(private, private.run_dir(attempt))
        stage = 23
        p8_immutable.verify_git_index(REPO)
        stage = 24
        verify_successor_history(private)
        p8_candidate.verify_a20_consumption(private, require_unused=True)
        stage = 25
        p8_candidate.require_owner_approval(
            REPO, attempt=attempt, manifest=pp.protocol_digest(), code=pp.protocol_code_digest(),
            require_clean=True)
    except (OSError, ValueError, RunRefused, MonitorUnavailable, pp.ProtocolNotReady,
            p8_journal.JournalError, p8_private.PrivateEvidenceError,
            launch_approval.LaunchApprovalError):
        return stage
    return 0


def _require_pre_dispatch(private: p8_private.PrivateEvidence, uid: int, gid: int) -> None:
    if os.geteuid() != 0 or uid == 0:
        raise RunRefused("signed preflight requires root and a dedicated unprivileged worker")
    _require_private_available(private, private.run_dir(final_attempt_key()))
    p8_candidate.verify_a20_consumption(private, require_unused=True)
    result = supervise([sys.executable, "-I", "-B", str(Path(__file__).resolve()),
                        "--_preflight-child"], timeout_s=120,
                       memory_bytes=int(pp.BUDGETS.evidence_memory_gb * 1e9),
                       credentials=(uid, gid))
    if result.status != "complete" or result.returncode != 0:
        detail = _PREFLIGHT_ERRORS.get(result.returncode or 0, result.status)
        if result.status == "child_failure" and result.returncode == 1:
            detail = "worker could not start the probe; check checkout traversal and imports"
        raise RunRefused(f"pre-dispatch check failed before consumption: {detail}")


def _signed_preflight() -> None:
    pp.require_successor_amendment()
    if os.geteuid() != 0:
        raise RunRefused("signed preflight requires the privileged supervisor")
    worker = p8_private._ROOT.lstat()
    private = p8_private.open_private_evidence(REPO, owner=worker.st_uid)
    verify_successor_history(private)
    p8_candidate.require_owner_approval(
        REPO, attempt=final_attempt_key(), manifest=pp.protocol_digest(),
        code=pp.protocol_code_digest(), require_clean=True, supervisor_uid=worker.st_uid)
    _require_pre_dispatch(private, worker.st_uid, worker.st_gid)


def _run_successor(args: argparse.Namespace, argv: Sequence[str]) -> int:
    """The only governed entry. No supplied plans, draws or reporting callbacks."""
    refuse_real_final_under_test("final", pc.run_task)
    attempt = final_attempt_key()
    require_successor_identity(attempt)
    pp.require_successor_amendment()
    if (os.geteuid() == 0) == args._supervised_child:
        raise RunRefused("governed execution requires a privileged supervisor and isolated worker")
    metadata = p8_private._ROOT.lstat()
    uid, gid = metadata.st_uid, metadata.st_gid
    if uid == 0 or (args._supervised_child and uid != os.geteuid()):
        raise RunRefused("private evidence requires a dedicated unprivileged worker")
    verify_protocol("final", args.expect_manifest, args.expect_code, signed_checkout=True)
    if args.workers != pp.P8_EXECUTION.workers:
        raise RunRefused("final execution requires the frozen worker count")
    private = p8_private.open_private_evidence(REPO, owner=uid)
    run_dir = args.run_dir or private.run_dir(attempt)
    registry = args.registry or private.registry
    private.check_target(attempt, run_dir, registry, args.output)
    approval_sha256 = p8_candidate.require_owner_approval(
        REPO, attempt=attempt, manifest=pp.protocol_digest(), code=pp.protocol_code_digest(),
        require_clean=True, supervisor_uid=uid)
    history_pin = verify_successor_history(private)
    if run_dir.exists() or run_dir.is_symlink():
        raise RunRefused("private run directory already exists and cannot be reused")
    if not args._supervised_child:
        _require_pre_dispatch(private, uid, gid)
        with p8_journal.open_journal(_journal_path(), writable=True) as journal:
            return _consume_and_supervise(
                journal, attempt, [sys.executable, "-I", "-B", str(Path(__file__).resolve()),
                                   *argv, "--_supervised-child"], run_dir, history_pin,
                timeout_s=pp.BUDGETS.evidence_ceiling_min * 60,
                memory_bytes=int(pp.BUDGETS.evidence_memory_gb * 1e9),
                approval_sha256=approval_sha256, credentials=(uid, gid))
    resources = await_supervisor()
    claim = p8_candidate.require_active_claim(REPO, attempt=attempt, manifest=pp.protocol_digest(),
                                               code=pp.protocol_code_digest(),
                                               approval_sha256=approval_sha256)
    _require_private_available(private, run_dir)
    plan = final_plan()
    ledger = _reserve(plan, run_dir, args.workers, registry, attempt, private=private,
                      claim=claim)
    _execute_phases(plan, run_dir, args.workers, ledger, pc.run_task, write_report, None, resources)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--_supervised-child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--_preflight-child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--preflight", action="store_true",
                    help="verify the signed launch read-only before consumption")
    ap.add_argument("--mode", choices=("dry-run", "rehearsal", "final"))
    ap.add_argument("--workers", type=int, default=pp.P8_EXECUTION.workers)
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--draws", type=int, default=DRY_RUN_DRAWS,
                    help="draws per study (dry-run and rehearsal only)")
    ap.add_argument("--refits", type=int, default=DRY_RUN_REFITS,
                    help="refits per REFIT draw (dry-run and rehearsal only)")
    ap.add_argument("--expect-manifest", default=None)
    ap.add_argument("--expect-code", default=None)
    ap.add_argument("--registry", type=Path)
    ap.add_argument("--export-reviewed", metavar="ID",
                    help="stage an explicitly owner-reviewed development resource summary")
    ap.add_argument("--verify", type=Path, default=None, help="check a completed run directory")
    ap.add_argument("--no-raw", action="store_true",
                    help="with --verify: accept a clone without the raw (Git-ignored) results")
    args = ap.parse_args(argv)
    if args.preflight or args._preflight_child:
        if (args.preflight and args._preflight_child
                or any((args.mode, args.verify, args.export_reviewed, args._supervised_child,
                        args.run_dir, args.registry, args.output))):
            ap.error("preflight is separate from execution, verification and export")
        if args._preflight_child:
            return _preflight_child()
        try:
            _signed_preflight()
        except (OSError, ValueError, RunRefused, pp.ProtocolNotReady,
                p8_journal.JournalError, p8_private.PrivateEvidenceError,
                launch_approval.LaunchApprovalError) as exc:
            print(f"preflight refused: {type(exc).__name__}" if isinstance(exc, OSError)
                  else f"preflight refused: {exc}", file=sys.stderr)
            return 2
        print("Signed pre-dispatch check passed. No claim was created.")
        return 0
    if args.export_reviewed is not None:
        if any((args.mode, args.verify, args.run_dir, args.registry, args.output,
                args._supervised_child)):
            ap.error("--export-reviewed is separate from execution and verification")
        try:
            staged = p8_private.export_reviewed(p8_private.open_private_evidence(REPO),
                                                args.export_reviewed)
        except (p8_private.PrivateEvidenceError, OSError) as exc:
            print(f"export refused: {exc}", file=sys.stderr)
            return 2
        print(f"staged reviewed export {staged}")
        return 0
    if args.verify is not None:
        problems = verify_run(args.verify, require_raw=not args.no_raw)
        print("\n".join(problems) if problems else "run verified: complete and unaltered")
        return 1 if problems else 0
    if args.mode is None:
        ap.error("--mode is required unless --verify is given")
    try:
        if args.mode == "final":
            return _run_successor(args, argv)
        plan = (rehearsal_plan(args.draws, args.refits) if args.mode == "rehearsal"
                else dry_run_plan(args.draws, args.refits))
        private = p8_private.open_private_evidence(REPO)
        run_dir = args.run_dir or private.run_dir(reservation_key(plan))
        registry = args.registry or private.registry
        out = execute(plan, run_dir, args.workers, registry=registry,
                      expect_manifest=args.expect_manifest, expect_code=args.expect_code,
                      output=args.output, private=private)
    except (pp.ProtocolNotReady, RunRefused, p8_private.PrivateEvidenceError,
            p8_journal.JournalError, launch_approval.LaunchApprovalError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted: inspect the private journal and staging before any further action"
              if args.mode == "final"
              else "interrupted: recorded in the ledger; the run will not be resumed",
              file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - any failed phase aborts the run
        error = str(exc) if isinstance(exc, TracedTaskError) else type(exc).__name__
        print(f"failed: {error}; inspect the private journal and staging"
              if args.mode == "final"
              else f"failed: {error}; recorded in the ledger; no record written", file=sys.stderr)
        return 1
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
