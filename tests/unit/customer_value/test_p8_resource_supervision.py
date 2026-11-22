"""Resource failures use isolated subprocesses, fake draws and temporary reservations."""

from __future__ import annotations

import ctypes
import errno
import importlib.util
import json
import os
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from decision_intelligence.customer_value import resource_supervision as resources
from decision_intelligence.customer_value.resource_supervision import (
    DarwinRSS,
    MonitorUnavailable,
    SupervisionResult,
    supervise,
)

REPO = Path(__file__).resolve().parents[3]
HOST = pytest.mark.skipif(sys.platform != "darwin", reason="macOS libproc is required")
FIXTURE = """
import os
import subprocess
import sys
import time
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

from decision_intelligence.customer_value.resource_supervision import await_supervisor
from scripts import generate_m11_evidence as gen


def draw(task):
    if mode in {"worker", "worker_without_registry"} and task.study == "R2":
        os._exit(41)
    return {"study": task.study}


class SlowPickle:
    def __reduce__(self):
        (root / "serializing").touch()
        time.sleep(30)
        return str, ("done",)


def serializing_draw(task):
    return {"result": SlowPickle()}


def report(plan, results, perf, run_dir, output):
    (root / "reporting").touch()
    if mode == "report":
        time.sleep(30)
    record = run_dir / "record.md"
    record.write_text("development fixture")
    return [record]


if __name__ == "__main__":
    mode, root = sys.argv[1], Path(sys.argv[2])
    resources = await_supervisor()
    if mode in {"memory", "monitor", "interrupt"}:
        child = subprocess.Popen([sys.executable, "-c",
            "import pathlib,sys,time; b=bytearray(80_000_000); "
            "pathlib.Path(sys.argv[1]).touch(); time.sleep(30); "
            "pathlib.Path(sys.argv[2]).touch()",
            str(root / "grandchild_ready"), str(root / "grandchild_survived")])
        (root / "child_pid").write_text(str(child.pid))
        child.wait()
    elif mode == "worker_without_registry":
        os.environ["TEST_P8_MODE"] = mode
        tasks = [gen.pc.Task(study, seed)
                 for study, seed in (("R2", 6750), ("R3", 6751), ("R3", 6752))]
        try:
            gen.run_supervised_tasks(tasks, 3, draw)
        except BrokenProcessPool:
            (root / "worker_failure").touch()
            raise
    else:
        # Make arguments available in spawn workers without running the main block there.
        os.environ["TEST_P8_MODE"] = mode
        gen.execute(gen.dry_run_plan(1, 1), root / "run", 3 if mode == "worker" else 1,
                    registry=root / "registry", evidence_root=root, resources=resources,
                    fn=serializing_draw if mode == "serialize" else draw, report_fn=report)
else:
    mode = os.environ.get("TEST_P8_MODE")
"""


@pytest.fixture
def command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    script = tmp_path / "fixture.py"
    script.write_text(FIXTURE)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(REPO), str(REPO / "src"))))
    return lambda mode: [sys.executable, str(script), mode, str(tmp_path)]


def _ledger(tmp_path: Path):
    return json.loads((tmp_path / "run" / "ledger.json").read_text())


@pytest.mark.parametrize(
    "task_errno, last_state, expected",
    [(errno.ESRCH, "zombie", True), (errno.ESRCH, "reaped", True),
     (errno.ESRCH, "still_exiting", False), (errno.ESRCH, "live", False),
     (errno.ESRCH, "wrong_pid", False), (errno.ESRCH, "short_reply", False),
     (errno.EPERM, "zombie", False)],
)
def test_task_info_exit_race_requires_verified_exit(
        monkeypatch: pytest.MonkeyPatch, task_errno: int, last_state: str,
        expected: bool) -> None:
    sampler = DarwinRSS.__new__(DarwinRSS)
    clock = [0.0]
    bsd_calls = 0
    exiting_pid, live_pid, group = 200, 201, 202

    def sleep(seconds: float) -> None:
        clock[0] += seconds

    def members(_group: int) -> set[int]:
        return {live_pid} if last_state == "reaped" and bsd_calls >= 2 else {
            live_pid, exiting_pid}

    def pidinfo(pid, flavor, arg, buf, size):
        nonlocal bsd_calls
        if flavor == 4:
            if pid == exiting_pid:
                ctypes.set_errno(task_errno)
                return 0
            struct.pack_into("=Q", buf, 8, 200 if pid == live_pid else 100)
            return size
        assert (pid, flavor, arg, size) == (exiting_pid, 3, 1, 136)
        bsd_calls += 1
        if last_state == "reaped" and bsd_calls >= 2:
            ctypes.set_errno(errno.ESRCH)
            return 0
        if last_state == "short_reply":
            return 12
        flags = 0 if last_state == "live" else 4
        status = 5 if last_state == "wrong_pid" or (
            last_state == "zombie" and bsd_calls >= 2) else 2
        pid = exiting_pid + 10 if last_state == "wrong_pid" else exiting_pid
        struct.pack_into("=IIII", buf, 0, flags, status, 0, pid)
        return size

    monkeypatch.setattr(sampler, "lib", SimpleNamespace(proc_pidinfo=pidinfo), raising=False)
    monkeypatch.setattr(sampler, "_members", members)
    monkeypatch.setattr(resources, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep))
    if expected:
        assert sampler(group, False) == (300, True)
        assert bsd_calls == 2 and 0 < clock[0] < resources._POLL_S
    else:
        with pytest.raises(MonitorUnavailable, match="cannot inspect a listed execution process"):
            sampler(group, False)
        if last_state == "still_exiting":
            assert bsd_calls > 1 and clock[0] == pytest.approx(resources._POLL_S)
        else:
            assert bsd_calls == 1 and clock[0] == 0


@HOST
def test_native_memory_limit_counts_grandchild_and_stops_group(tmp_path, command):
    native = DarwinRSS()
    baseline = 0

    def sample(pid, alive):
        nonlocal baseline
        rss, live = native(pid, alive)
        if not (tmp_path / "child_pid").exists():
            baseline = max(baseline, rss)
        # Include the actual imported runner. Calibrate against the same sample without its
        # grandchild so this stays meaningful across different Darwin Python builds.
        if (tmp_path / "grandchild_ready").exists():
            child = int((tmp_path / "child_pid").read_text())
            assert native._rss(child) > 75_000_000
            assert rss > baseline + 70_000_000
        return rss, live

    # Use a native baseline from the same runner after imports, with no allocation.
    measured = supervise(command("success"), timeout_s=15, memory_bytes=4_000_000_000)
    assert measured.status == "complete"
    result = supervise(command("memory"), timeout_s=15,
                       memory_bytes=measured.peak_bytes + 45_000_000, sample=sample)
    assert result.status == "memory_limit"
    assert result.peak_bytes > measured.peak_bytes + 45_000_000
    assert result.returncode != 0
    _assert_grandchild_stopped(tmp_path)


def _assert_grandchild_stopped(tmp_path: Path) -> None:
    child = int((tmp_path / "child_pid").read_text())
    for _ in range(50):
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.02)
    else:
        pytest.fail("grandchild survived group termination")
    assert not (tmp_path / "grandchild_survived").exists()


@HOST
@pytest.mark.parametrize("stage", ["serialize", "report"])
def test_timeout_in_output_phase_consumes_reservation_once(tmp_path, command, stage):
    result = supervise(command(stage), timeout_s=4, memory_bytes=4_000_000_000)
    assert result.status == "time_limit"
    assert 4 <= result.elapsed_s < 6
    assert (tmp_path / ("serializing" if stage == "serialize" else "reporting")).exists()
    assert _ledger(tmp_path)["status"] != "complete"
    registry = {p.name: p.read_bytes() for p in (tmp_path / "registry").glob("*.json*")}
    assert registry
    retry = supervise(command("success"), timeout_s=10, memory_bytes=4_000_000_000)
    assert retry.status == "child_failure"
    assert registry == {p.name: p.read_bytes() for p in (tmp_path / "registry").glob("*.json*")}


@HOST
@pytest.mark.parametrize("mode,expected", [("monitor", "monitor_failure"),
                                           ("interrupt", "interrupted")])
def test_lost_monitor_or_interrupt_terminates_live_descendants(tmp_path, command, mode, expected):
    native = DarwinRSS()

    def sample(pid, alive):
        if (tmp_path / "grandchild_ready").exists():
            if mode == "monitor":
                raise OSError("sensitive diagnostic")
            os.kill(os.getpid(), signal.SIGTERM)
        return native(pid, alive)

    result = supervise(command(mode), timeout_s=10, memory_bytes=4_000_000_000, sample=sample)
    assert result.status == expected
    assert result.returncode != 0
    _assert_grandchild_stopped(tmp_path)


@HOST
def test_abrupt_worker_death_aborts_without_completing_or_retrying(tmp_path, command):
    result = supervise(command("worker"), timeout_s=15, memory_bytes=4_000_000_000)
    assert result.status == "child_failure"
    ledger = _ledger(tmp_path)
    assert (ledger["status"], ledger["phase"], ledger["error"]) == (
        "failed", "repeated_sampling", "BrokenProcessPool")
    assert [e["phase"] for e in ledger["events"]].count("repeated_sampling") == 1
    assert not (tmp_path / "reporting").exists()


@HOST
def test_native_supervisor_reports_worker_crash_without_a_reservation(tmp_path, command):
    result = supervise(command("worker_without_registry"), timeout_s=15,
                       memory_bytes=4_000_000_000)
    assert result.status == "child_failure" and result.returncode == 1
    assert (tmp_path / "worker_failure").is_file()
    assert not (tmp_path / "registry").exists()


def test_initial_monitor_failure_does_not_open_gate(tmp_path, command, capsys):
    def sample(_pid, _alive):
        raise OSError("sensitive diagnostic")

    result = supervise(command("success"), timeout_s=10, memory_bytes=4_000_000_000, sample=sample)
    assert result.status == "monitor_failure"
    assert not (tmp_path / "registry").exists()
    assert capsys.readouterr() == ("", "")


@HOST
def test_runner_leaves_completion_pending_for_controller(tmp_path, command):
    result = supervise(command("success"), timeout_s=15, memory_bytes=4_000_000_000)
    assert result.status == "complete" and result.peak_bytes > 0
    assert _ledger(tmp_path)["status"] == "supervision_pending"
    assert (tmp_path / "run" / "record.md").is_file()


@HOST
def test_termination_during_report_keeps_reservation_and_records_interrupt(tmp_path, command):
    native = DarwinRSS()

    def sample(pid, alive):
        if (tmp_path / "reporting").exists():
            os.kill(os.getpid(), signal.SIGTERM)
        return native(pid, alive)

    result = supervise(command("report"), timeout_s=15, memory_bytes=4_000_000_000, sample=sample)
    assert result.status == "interrupted"
    assert _ledger(tmp_path)["status"] == "interrupted"
    reservation = tmp_path / "registry" / f"{_ledger(tmp_path)['reservation']}.json"
    assert reservation.is_file()


@HOST
def test_watchdog_stops_tree_if_controller_is_killed(tmp_path, command):
    with subprocess.Popen([
        sys.executable, "-c",
        "import sys; "
        "from decision_intelligence.customer_value.resource_supervision import supervise; "
        "supervise(sys.argv[1:], timeout_s=15, memory_bytes=4_000_000_000)",
        *command("interrupt"),
    ]) as controller:
        try:
            deadline = time.monotonic() + 10
            while not (tmp_path / "grandchild_ready").exists() and time.monotonic() < deadline:
                assert controller.poll() is None
                time.sleep(0.02)
            assert (tmp_path / "grandchild_ready").exists()
        finally:
            controller.kill()
        assert controller.wait(timeout=2) == -signal.SIGKILL
    _assert_grandchild_stopped(tmp_path)


@HOST
def test_watchdog_stops_tree_if_rss_probe_stalls(tmp_path, command):
    native = DarwinRSS()

    def sample(pid, alive):
        if (tmp_path / "grandchild_ready").exists():
            time.sleep(1.5)
        return native(pid, alive)

    result = supervise(command("monitor"), timeout_s=15, memory_bytes=4_000_000_000, sample=sample)
    assert result.status == "monitor_failure"
    _assert_grandchild_stopped(tmp_path)


@pytest.mark.parametrize("status", ["complete", "memory_limit"])
def test_only_successful_supervision_can_complete_the_ledger(tmp_path, monkeypatch, status):
    spec = importlib.util.spec_from_file_location(
        "generate_m11_evidence_supervision", REPO / "scripts" / "generate_m11_evidence.py")
    assert spec is not None and spec.loader is not None
    gen = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, gen)
    spec.loader.exec_module(gen)

    ledger = gen.Ledger.create(tmp_path, {})
    ledger.finish("supervision_pending", supervisor_pid=os.getpid())
    monkeypatch.setattr(gen, "supervise", lambda *args, **kwargs:
                        SupervisionResult(status, 0, 2.0, 123_000_000))
    success = status == "complete"
    assert gen._run_under_supervision([], tmp_path) == (0 if success else 1)
    data = json.loads(ledger.path.read_text())
    assert data["status"] == ("complete" if success else "supervision_pending")
    if success:
        assert data["wall_s"] == 2.0 and data["peak_gb"] == 0.123
