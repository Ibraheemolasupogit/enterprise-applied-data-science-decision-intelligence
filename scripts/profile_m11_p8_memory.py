#!/usr/bin/env python3
"""Profile or qualify P8 scheduling on development seeds in a new macOS scratch directory.

Run from the repository root with PYTHONPATH=src and the pinned Python. This harness
never uses the evidence CLI, attempt registry, final plan, or protected artifacts.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pickle  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import deque  # noqa: E402
from collections.abc import Iterable, Iterator, Mapping  # noqa: E402
from concurrent.futures import Future, ProcessPoolExecutor  # noqa: E402
from contextlib import ExitStack  # noqa: E402
from functools import partial, wraps  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, BinaryIO  # noqa: E402
from unittest.mock import patch  # noqa: E402

import numpy as np  # noqa: E402
from generate_m11_evidence import (  # noqa: E402
    SeedPlan,
    check_development_seeds,
    run_supervised_tasks,
    study_tasks,
    write_report,
)

from decision_intelligence.customer_value import (  # noqa: E402
    btyd,
    oracle,
    predictive,
    predictive_evaluation,
    purchase_oracle,
    survival,
)
from decision_intelligence.customer_value import predictive_cases as pc  # noqa: E402
from decision_intelligence.customer_value import predictive_protocol as pp  # noqa: E402
from decision_intelligence.customer_value import resource_supervision as rs  # noqa: E402

Result = dict[str, Any]
PHASE_TASKS = {"main": pc.Task("MAIN", 6200), "m10_comparability": pc.Task("M10", 7501),
               "fresh": pc.Task("MAIN", 7502)}


def event(path: Path, name: str, **fields: Any) -> None:
    row = {"ns": time.monotonic_ns(), "event": name, "pid": os.getpid(), **fields}
    line = (json.dumps(row) + "\n").encode()
    # One append syscall per event, including concurrent writes from spawned workers.
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        if os.write(fd, line) != len(line):
            raise OSError("incomplete profiling event")
    finally:
        os.close(fd)


class TraceRSS(rs.DarwinRSS):
    """Record the native sampler's exact per-PID contributions, including the watchdog."""

    def __init__(self, output: Any) -> None:
        super().__init__()
        self.output = output
        self.started_ns = time.monotonic_ns()
        self.members: dict[int, int] = {}
        self.runner = 0
        self.awaiting_watchdog = False

    def __call__(self, group: int, root_alive: bool) -> tuple[int, bool]:
        self.members = {}
        self.runner = group
        self.awaiting_watchdog = False
        rss, live = super().__call__(group, root_alive)
        self.awaiting_watchdog = True
        return rss, live

    def _rss(self, pid: int) -> int:
        rss = super()._rss(pid)
        self.members[pid] = rss
        # The A18 controller probes the watchdog immediately after the group sample.
        if self.awaiting_watchdog:
            self.awaiting_watchdog = False
            now = time.monotonic_ns()
            row = {"ns": now, "elapsed_s": (now - self.started_ns) / 1e9,
                   "parent": self.runner, "supervisor": os.getpid(), "watchdog": pid,
                   "aggregate_bytes": sum(self.members.values()), "rss_bytes": self.members}
            self.output.write(json.dumps(row) + "\n")
        return rss


def trace_r2_call(owner: Any, name: str, events: Path) -> Any:
    original = getattr(owner, name)

    @wraps(original)
    def traced(*args: Any, **kwargs: Any) -> Any:
        event(events, "r2_enter", stage=name)
        try:
            return original(*args, **kwargs)
        finally:
            event(events, "r2_exit", stage=name)

    return patch.object(owner, name, traced)


def timed_task(task: pc.Task, *, events: Path) -> Result:
    identity = {"study": task.study, "seed": task.seed, "scale": task.scale, "refits": task.refits}
    event(events, "task_start", **identity)
    create_rng = pc._rng
    streams: list[tuple[int, int, Mapping[str, Any], np.random.Generator]] = []

    def record_rng(seed: int, stream: int) -> np.random.Generator:
        generator = create_rng(seed, stream)
        streams.append((seed, stream, generator.bit_generator.state, generator))
        return generator

    with ExitStack() as stack:
        stack.enter_context(patch.object(pc, "_rng", record_rng))
        if task.study == "R2" and os.environ.get("M11_P8_PROFILE_R2_STAGES") == "1":
            for owner, name in (
                (predictive_evaluation, "simulate_contractual"),
                (predictive_evaluation, "simulate_e3"),
                (predictive_evaluation.SimulatedValue, "interval"),
                (pc, "e4_window_purchases"), (pc, "e4_simulate_fast"),
                (btyd, "predict"), (predictive, "fit_e5"), (predictive, "fit_s2"),
                (survival, "fit_e3"), (oracle, "signals_oracle"), (oracle, "m10_oracle"),
                (survival, "_landmark_data"), (survival, "_fit_at_validation"),
                (survival, "fit_hazard"), (survival, "person_months"),
                (survival, "build_design"), (survival, "fit_penalised_logistic"),
                (survival, "_paid"), (predictive_evaluation, "e3_cr1"),
                (purchase_oracle, "purchases_oracle"), (predictive, "training_stack"),
                (predictive, "fit_parts"), (predictive, "build_design"),
                (predictive, "fit_penalised_logistic"), (predictive, "fit_penalised_gamma"),
                (btyd, "purchase_data"), (btyd, "fit_cells"), (btyd, "bgnbd_conditional"),
            ):
                stack.enter_context(trace_r2_call(owner, name, events))
        result = pc.run_task(task)
    states = [(seed, stream, initial, generator.bit_generator.state)
              for seed, stream, initial, generator in streams]
    rng_sha256 = hashlib.sha256(json.dumps(states, sort_keys=True).encode()).hexdigest()
    event(events, "task_done", **identity, seconds=result["seconds"],
          rng_streams=len(states), rng_sha256=rng_sha256)
    return result


def worker_ready(events: Path, phase: str, wave: int) -> None:
    event(events, "worker_ready", phase=phase, wave=wave)


def bounded_tasks(tasks: Iterable[pc.Task], workers: int, events: Path,
                  phase: str = "all", wave: int = 0,
                  third_delay: float = 0.0) -> Iterator[Result]:
    """Submit at most two tasks per worker, yield in order and join every worker."""
    source = iter(tasks)
    window = 2 * workers
    event(events, "pool_open", phase=phase, wave=wave)
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                             initializer=worker_ready, initargs=(events, phase, wave)) as pool:
        futures: deque[Future[Result]] = deque()
        for position, task in enumerate(itertools.islice(source, window)):
            if position == 2 and third_delay:
                event(events, "r2_delay", wave=wave, seconds=third_delay)
                time.sleep(third_delay)
            futures.append(pool.submit(timed_task, task, events=events))
        while futures:
            result = futures.popleft().result()
            next_task = next(source, None)
            if next_task is not None:
                futures.append(pool.submit(timed_task, next_task, events=events))
            yield result
        event(events, "pool_drained", phase=phase, wave=wave)
    event(events, "pool_closed", phase=phase, wave=wave)


def replacement_tasks(tasks: Iterable[pc.Task], workers: int, events: Path,
                      *, r2_workers: int, r2_batch_size: int,
                      r2_third_delay: float) -> Iterator[Result]:
    """Drain and join each pool before starting the next study or R2 batch."""
    for study, group in itertools.groupby(tasks, key=lambda task: task.study):
        if study == "R2":
            wave = 1
            while batch := tuple(itertools.islice(group, r2_batch_size)):
                yield from bounded_tasks(batch, r2_workers, events, study, wave, r2_third_delay)
                wave += 1
        else:
            yield from bounded_tasks(group, workers, events, study)


def result_payload(result: Result) -> bytes:
    return pickle.dumps({k: v for k, v in result.items() if k not in ("pid", "seconds")},
                        protocol=pickle.HIGHEST_PROTOCOL)


def digest_rows(results: Iterable[Result], spool: BinaryIO | None = None) -> dict[str, Any]:
    digest = hashlib.sha256()
    studies = {study: hashlib.sha256() for study in ("R1", "R2", "R3", "REFIT")}
    sizes: dict[str, list[int]] = {}
    for result in results:
        payload = result_payload(result)
        digest.update(payload)
        study = result["task"].study
        studies[study].update(payload)
        sizes.setdefault(study, []).append(len(payload))
        if spool is not None:
            pickle.dump(result, spool, protocol=pickle.HIGHEST_PROTOCOL)
    return {"ordered_result_sha256": digest.hexdigest(),
            "ordered_study_sha256": {study: studies[study].hexdigest() for study in sizes},
            "serialized_result_bytes": {
                study: {"count": len(values), "total": sum(values), "max": max(values)}
                for study, values in sizes.items()}}


def from_spool(spool: BinaryIO, count: int) -> Iterator[Result]:
    spool.seek(0)
    for _ in range(count):
        yield pickle.load(spool)  # Generated in this process from development inputs only.


def aggregate(results: Iterable[Result]) -> None:
    """Run the same study aggregators, materialising one study at a time."""
    for study, group in itertools.groupby(results, key=lambda row: row["task"].study):
        draws = list(group)
        if study == "R1":
            pc.aggregate_pc2(draws)
            pc.aggregate_oracle_gate(draws)
        elif study == "R3":
            pc.aggregate_gate4(draws)
        else:
            pc.aggregate_coverage(draws)


def qualification_plan(full_count: bool) -> SeedPlan:
    counts = ({study.id: study.replications for study in pp.REPEATED_SAMPLING}
              if full_count else {"R1": 6, "R2": 2, "R3": 2, "REFIT": 2})
    draws = {study: list(range(start, start + counts[study])) for study, start in
             (("R1", 6201), ("R2", 6201), ("R3", 6201), ("REFIT", 7301))}
    scales = next(study.scales for study in pp.REPEATED_SAMPLING if study.id == "R3")
    check_development_seeds([*(task.seed for task in PHASE_TASKS.values()),
                             *(seed for study in draws.values() for seed in study)])
    return SeedPlan("rehearsal", PHASE_TASKS["main"].seed,
                    PHASE_TASKS["m10_comparability"].seed, True, draws, scales,
                    pp.CLUSTERED_INFERENCE.refit_bootstrap_resamples)


def task_key(task: pc.Task) -> str:
    return f"{task.study}/{task.seed}/{task.scale:g}/{task.refits}"


def qualified_results(tasks: list[pc.Task], results: Iterable[Result], events: Path,
                      hashes: dict[str, str], reference: dict[str, str]) -> Iterator[Result]:
    for task, result in zip(tasks, results, strict=True):
        if result["task"] != task:
            raise RuntimeError("development result arrived out of plan order")
        key = task_key(task)
        digest = hashlib.sha256(result_payload(result)).hexdigest()
        if key in reference and digest != reference[key]:
            raise RuntimeError(f"development reference mismatch for {key}")
        hashes[key] = digest
        event(events, "result_hash", task=key, sha256=digest)
        yield result


def qualification(args: argparse.Namespace, directory: Path) -> None:
    resources = rs.await_supervisor()
    full_count = args.qualification == "hybrid"
    events = directory / "events.jsonl"
    started = time.monotonic()
    plan = qualification_plan(full_count)
    tasks = study_tasks(plan)
    reference: dict[str, str] = {}
    reference_rng: dict[str, tuple[int, str]] = {}
    if full_count:
        saved = json.loads(args.reference.read_text())
        reference = saved["task_hashes"]
        reference_rng = {k: tuple(v) for k, v in saved["rng"].items()}
        expected_reference = {task_key(t) for t in
                              [*PHASE_TASKS.values(), *study_tasks(qualification_plan(False))]}
        if reference.keys() != expected_reference or reference_rng.keys() != expected_reference:
            raise RuntimeError("incomplete development reference")
    hashes: dict[str, str] = {}
    raw = directory / "raw"
    raw.mkdir()

    def isolated(phase: str) -> None:
        event(events, "phase", phase=phase)
        task = PHASE_TASKS[phase]
        path = raw / f"{phase}.pkl"
        with path.open("xb") as output:
            subprocess.run(
                [sys.executable, "-B", str(Path(__file__).resolve()), "--output", str(directory),
                 "--strategy", "replace", "--workers", "3", "--isolated-phase", phase],
                stdout=output, check=True,
            )
        with path.open("rb") as output:
            list(qualified_results([task], [pickle.load(output)], events, hashes, reference))

    isolated("main")
    event(events, "phase", phase="repeated_sampling")
    stream = (replacement_tasks(tasks, 3, events, r2_workers=2, r2_batch_size=2,
                                r2_third_delay=0) if full_count
              else bounded_tasks(tasks, 1, events, "serial_reference"))
    spool = raw / "repeated_sampling.pkl"
    with spool.open("xb") as output:
        report = digest_rows(qualified_results(tasks, stream, events, hashes, reference), output)
    isolated("m10_comparability")
    isolated("fresh")

    event(events, "phase", phase="prepare_report")
    results: dict[str, Any] = {}
    for phase in PHASE_TASKS:
        with (raw / f"{phase}.pkl").open("rb") as stored:
            results[phase] = [pickle.load(stored)]
    with spool.open("rb") as stored:
        results["repeated_sampling"] = list(from_spool(stored, len(tasks)))
    rng = {}
    for line in events.read_text().splitlines():
        row = json.loads(line)
        if row["event"] == "task_done":
            key = task_key(pc.Task(row["study"], row["seed"], row["scale"], row["refits"]))
            rng[key] = (row["rng_streams"], row["rng_sha256"])
    if rng.keys() != hashes.keys() or any(rng[k] != v for k, v in reference_rng.items()):
        raise RuntimeError("development RNG audit failed")
    event(events, "phase", phase="reporting")
    files = write_report(plan, results, {
        "wall": time.monotonic() - started, "peak_gb": resources.peak_gb,
        "workers": 3 if full_count else 1, "r2_workers": 2 if full_count else 1,
        "resource_snapshot": "before report generation", "full_count_development": full_count,
    }, directory, None)
    record = files[0]
    lines = record.read_text().splitlines(keepends=True)
    label = "FULL-COUNT DEVELOPMENT QUALIFICATION" if full_count else "REDUCED SERIAL REFERENCE"
    lines[0] = f"# Milestone 11 P8 {label}\n"
    lines[2] = ("Development inputs only. Use the external supervisor and phase events for "
                "measured resource use. The generic extrapolation below assumes a single "
                "worker count across studies.\n")
    record.write_text("".join(lines))
    event(events, "phase", phase="checksums_and_result")
    report.update(task_hashes=hashes, rng=rng, reference_rows_verified=len(reference),
                  full_count_development=full_count, repeated_tasks=len(tasks))
    report["files"] = {}
    for file in [*files, *raw.iterdir()]:
        with file.open("rb") as content:
            report["files"][file.relative_to(directory).as_posix()] = (
                hashlib.file_digest(content, "sha256").hexdigest())
    (directory / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    event(events, "phase", phase="teardown")


def benchmark(args: argparse.Namespace, directory: Path) -> None:
    await_resources = rs.await_supervisor()
    events = directory / "events.jsonl"
    os.environ["M11_P8_PROFILE_R2_STAGES"] = "1" if args.trace_r2_stages else "0"
    tasks = [pc.Task(study, 6201 + index, scale, refits)
             for study, count, scales, refits in (
                 ("R1", args.r1, (1.0,), 0), ("R2", args.r2, (1.0,), 0),
                 ("R3", args.r3, (0.5, 1.0, 2.0), 0),
                 ("REFIT", args.refit, (1.0,), pp.CLUSTERED_INFERENCE.refit_bootstrap_resamples))
             for scale in scales for index in range(count)]
    check_development_seeds([6200, *(task.seed for task in tasks)])
    event(events, "phase", phase="development_main")
    with tempfile.TemporaryFile(dir=directory) as main_file:
        if args.isolate_main:
            subprocess.run(
                [sys.executable, "-B", str(Path(__file__).resolve()), "--output", str(directory),
                 "--strategy", args.strategy, "--workers", str(args.workers), "--main-worker"],
                stdout=main_file, check=True,
            )
            main_file.seek(0)
            main = pickle.load(main_file)
        else:
            main = timed_task(pc.Task("MAIN", 6200), events=events)
            # Match the release's main serialization and retention during repeated sampling.
            main_file.write(pickle.dumps([main], protocol=pickle.HIGHEST_PROTOCOL))
    event(events, "phase", phase="repeated_sampling")
    if args.strategy in ("stream", "replace"):
        results_iterator = (
            replacement_tasks(tasks, args.workers, events, r2_workers=args.r2_workers,
                              r2_batch_size=args.r2_batch_size, r2_third_delay=args.r2_third_delay)
            if args.strategy == "replace" else bounded_tasks(tasks, args.workers, events)
        )
        with tempfile.TemporaryFile(dir=directory) as spool:
            report = digest_rows(results_iterator, spool)
            event(events, "phase", phase="aggregation")
            aggregate(from_spool(spool, len(tasks)))
    else:
        if args.strategy == "eager":
            results = run_supervised_tasks(tasks, args.workers, partial(timed_task, events=events))
        else:
            results = list(bounded_tasks(tasks, args.workers, events))
        event(events, "phase", phase="serialization")
        with tempfile.TemporaryFile(dir=directory) as batch:
            batch.write(pickle.dumps(results, protocol=pickle.HIGHEST_PROTOCOL))
        report = digest_rows(results)
        event(events, "phase", phase="aggregation")
        aggregate(results)
    event(events, "phase", phase="done")
    main_payload = pickle.dumps({k: v for k, v in main.items() if k not in ("pid", "seconds")},
                                protocol=pickle.HIGHEST_PROTOCOL)
    report.update(main_seconds=main["seconds"],
                  main_sha256=hashlib.sha256(main_payload).hexdigest(),
                  workers=args.workers, strategy=args.strategy,
                  r2_workers=args.r2_workers, r2_batch_size=args.r2_batch_size,
                  r2_third_delay=args.r2_third_delay,
                  isolated_main=args.isolate_main,
                  sampled_peak_gb_before_exit=await_resources.peak_gb)
    (directory / "result.json").write_text(json.dumps(report, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strategy", choices=("eager", "bounded", "stream", "replace"),
                        required=True)
    parser.add_argument("--workers", type=int, choices=(2, 3), required=True)
    parser.add_argument("--r2-workers", type=int, choices=(2, 3),
                        help="R2-only worker limit for replacement profiling")
    parser.add_argument("--r2-batch-size", type=int)
    parser.add_argument("--r2-third-delay", type=float, default=0,
                        help="Seconds before submitting the third draw in a three-draw R2 wave")
    parser.add_argument("--r1", type=int, default=120)
    parser.add_argument("--r2", type=int, default=3)
    parser.add_argument("--r3", type=int, default=2)
    parser.add_argument("--refit", type=int, default=3)
    parser.add_argument("--isolate-main", action="store_true")
    parser.add_argument("--trace-r2-stages", action="store_true")
    parser.add_argument("--qualification", choices=("serial-reference", "hybrid"))
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--main-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--isolated-phase", choices=tuple(PHASE_TASKS), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.strategy != "replace" and (args.r2_workers is not None or args.r2_third_delay):
        parser.error("R2 scheduling options require --strategy replace")
    if args.r2_workers is None:
        args.r2_workers = args.workers
    if args.r2_workers > args.workers:
        parser.error("--r2-workers must be at most the default worker count")
    if args.r2_batch_size is None:
        args.r2_batch_size = args.r2_workers
    if args.r2_batch_size < args.r2_workers:
        parser.error("--r2-batch-size must be at least the R2 worker count")
    if not 0 <= args.r2_third_delay < 60:
        parser.error("--r2-third-delay must be nonnegative and below 60 seconds")
    if args.r2_third_delay and (args.r2_workers != 3 or args.r2_batch_size != 3):
        parser.error("--r2-third-delay requires three workers and three draws per R2 wave")
    if (args.strategy == "replace" and not args.isolate_main and not args.main_worker
            and not args.isolated_phase and not args.qualification):
        parser.error("replacement profiling requires --isolate-main")
    scratch_root = Path("/tmp").resolve()
    directory = args.output.resolve()
    if directory == scratch_root or not directory.is_relative_to(scratch_root):
        parser.error("output must be a new directory under /tmp")
    if args.isolated_phase:
        task = PHASE_TASKS[args.isolated_phase]
        check_development_seeds([task.seed])
        isolated_result = timed_task(task, events=directory / "events.jsonl")
        pickle.dump(isolated_result, sys.stdout.buffer, protocol=pickle.HIGHEST_PROTOCOL)
        return 0
    if args.qualification:
        qualification_plan(args.qualification == "hybrid")
        if (args.strategy, args.workers, args.r2_workers, args.r2_batch_size,
                args.r2_third_delay) != ("replace", 3, 2, 2, 0):
            parser.error("qualification requires replacement, 3 workers and fresh R2 pairs")
        if args.qualification == "hybrid":
            if args.reference is None:
                parser.error("hybrid qualification requires a serial development reference")
            args.reference = args.reference.resolve()
            if (not args.reference.is_relative_to(scratch_root)
                    or args.reference.name != "result.json"):
                parser.error("reference must be a scratch result.json")
            certificate = args.reference.with_name("supervisor.json")
            if (not certificate.is_file()
                    or json.loads(certificate.read_text())["status"] != "complete"):
                parser.error("reference requires a complete supervision certificate")
            if json.loads(args.reference.read_text())["full_count_development"]:
                parser.error("reference must be the reduced serial development run")
    counts = (args.r1, args.r2, args.r3, args.refit)
    study_limits = {study.id: study.replications for study in pp.REPEATED_SAMPLING}
    if any(not 2 <= count < study_limits[study] for study, count in
           zip(("R1", "R2", "R3", "REFIT"), counts, strict=True)):
        parser.error("each development study count must be at least 2 and below its protocol count")
    if args.main_worker:
        check_development_seeds([6200])
        main_result = timed_task(pc.Task("MAIN", 6200), events=directory / "events.jsonl")
        pickle.dump(main_result, sys.stdout.buffer, protocol=pickle.HIGHEST_PROTOCOL)
        return 0
    if args.child:
        if args.qualification:
            try:
                qualification(args, directory)
            except Exception:
                (directory / "development-error.txt").write_text(traceback.format_exc())
                raise
        else:
            benchmark(args, directory)
        return 0
    directory.mkdir(mode=0o700)
    timeout_s = pp.BUDGETS.evidence_ceiling_min * 60 if args.qualification == "hybrid" else 600
    memory_bytes = int(pp.BUDGETS.evidence_memory_gb * 1e9)
    with (directory / "rss.jsonl").open("x", buffering=1) as output:
        trace = TraceRSS(output)
        with patch.object(rs, "DarwinRSS", return_value=trace):
            result = rs.supervise(
                [sys.executable, "-B", str(Path(__file__).resolve()), *sys.argv[1:], "--child"],
                timeout_s=timeout_s, memory_bytes=memory_bytes,
            )
    status = ("time_limit" if result.status == "complete" and result.elapsed_s > timeout_s
              else result.status)
    summary = {"status": status, "returncode": result.returncode,
               "elapsed_s": result.elapsed_s, "peak_bytes": result.peak_bytes}
    (directory / "supervisor.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"directory": str(directory), **summary}), flush=True)
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
