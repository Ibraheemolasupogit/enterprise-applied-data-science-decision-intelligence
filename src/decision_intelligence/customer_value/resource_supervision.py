"""Out-of-process resource enforcement for the macOS P8 execution host."""

from __future__ import annotations

import ctypes
import errno
import os
import select
import signal
import struct
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass

_PEAK_FD = "_M11_P8_SUPERVISOR_FD"
_RSS = struct.Struct("=Q")
_GO = b"\x01"
_POLL_S = 0.05
_STOP_GRACE_S = 1.0
_HEARTBEAT_S = 1.0

# In a separate session so loss of the controller still stops the execution group.
# A pipe EOF, missing heartbeat or expired deadline all mean the group must stop.
_WATCHDOG = """
import os, select, signal, sys, time
group, deadline, heartbeat = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
os.write(1, b"1")
while True:
    wait = max(0, min(heartbeat, deadline - time.monotonic()))
    ready, _, _ = select.select([0], [], [], wait)
    command = os.read(0, 1) if ready else b""
    if command == b"0":
        break
    if command != b"1" or time.monotonic() >= deadline:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        sys.exit(1)
"""


class MonitorUnavailable(RuntimeError):
    """The complete process group's RSS could not be established."""


class DarwinRSS:
    """Sample the dedicated execution group and the supervising process via libproc."""

    lib: ctypes.CDLL

    def __init__(self) -> None:
        if sys.platform != "darwin":
            raise MonitorUnavailable("resource supervision requires macOS libproc")
        self.lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        self.lib.proc_listpids.argtypes = (
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int)
        self.lib.proc_listpids.restype = ctypes.c_int
        self.lib.proc_pidinfo.argtypes = (
            ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int)
        self.lib.proc_pidinfo.restype = ctypes.c_int

    def _members(self, group: int) -> set[int]:
        capacity = 64
        while capacity <= 65536:
            buf = (ctypes.c_int * capacity)()
            ctypes.set_errno(0)
            # PROC_PGRP_ONLY; zero with no errno is a valid empty group.
            size = self.lib.proc_listpids(2, group, buf, ctypes.sizeof(buf))
            if size < 0 or ctypes.get_errno() or size % ctypes.sizeof(ctypes.c_int):
                break
            if size < ctypes.sizeof(buf):
                members = buf[:size // ctypes.sizeof(ctypes.c_int)]
                if any(pid <= 0 for pid in members):
                    break
                return set(members)
            capacity *= 2
        raise MonitorUnavailable("cannot enumerate the execution process group")

    def _rss(self, pid: int) -> int:
        # PROC_PIDTASKINFO / struct proc_taskinfo: virtual size followed by resident size.
        buf = ctypes.create_string_buffer(96)
        ctypes.set_errno(0)
        size = self.lib.proc_pidinfo(pid, 4, 0, buf, len(buf))
        if size == len(buf):
            return int(_RSS.unpack_from(buf, _RSS.size)[0])
        if size == 0 and ctypes.get_errno() == errno.ESRCH:
            raise ProcessLookupError
        raise MonitorUnavailable("cannot inspect execution RSS")

    def _exited(self, pid: int, group: int) -> bool:
        # PROC_PIDTBSDINFO with a nonzero arg includes zombies. These have no task RSS
        # but can remain in PROC_PGRP_ONLY until their parent reaps them.
        buf = ctypes.create_string_buffer(136)
        size = self.lib.proc_pidinfo(pid, 3, 1, buf, len(buf))
        if size == len(buf):
            _, status, _, found_pid = struct.unpack_from("=IIII", buf)
            if found_pid == pid and status == 5:  # SZOMB
                return True
        return pid not in self._members(group)

    def __call__(self, group: int, root_alive: bool) -> tuple[int, bool]:
        members = self._members(group)
        if root_alive and group not in members:
            # The root may have exited since poll; prove it is gone before accepting the sample.
            try:
                os.kill(group, 0)
            except ProcessLookupError:
                pass
            else:
                raise MonitorUnavailable("execution root is missing from its group")
        total = self._rss(os.getpid())
        live = False
        for pid in members:
            try:
                rss = self._rss(pid)
            except (ProcessLookupError, MonitorUnavailable):
                if not self._exited(pid, group):
                    raise MonitorUnavailable("cannot inspect a listed execution process") from None
            else:
                total += rss
                live = True
        return total, live


@dataclass(frozen=True)
class SupervisionResult:
    status: str
    returncode: int | None
    elapsed_s: float
    peak_bytes: int


@dataclass(frozen=True)
class SupervisedChild:
    """Read the controller's peak at report time, while it continues to sample."""

    fd: int

    @property
    def peak_gb(self) -> float:
        return float(_RSS.unpack(os.pread(self.fd, _RSS.size, 0))[0]) / 1e9


def await_supervisor() -> SupervisedChild:
    """Block the isolated runner until its parent has made a valid initial sample."""
    if os.getsid(0) != os.getpid() or sys.stdin.buffer.read(1) != _GO:
        raise MonitorUnavailable("runner has no resource supervisor")
    child = SupervisedChild(int(os.environ[_PEAK_FD]))
    if child.peak_gb <= 0:
        raise MonitorUnavailable("runner has no resource sample")
    return child


def _signal_group(group: int, signum: int) -> None:
    try:
        os.killpg(group, signum)
    except ProcessLookupError:
        pass


def _stop_tree(proc: subprocess.Popen[bytes], grace_s: float, native: DarwinRSS | None) -> None:
    if grace_s > 0 and proc.poll() is None:
        try:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=grace_s)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass
    # Even a runner that handled SIGINT can leave a worker or tracker behind.
    try:
        _signal_group(proc.pid, signal.SIGKILL)
    except PermissionError:
        # Darwin can report EPERM when the watchdog has already killed the group and
        # only unreaped processes remain. Suppress it only after verifying that state.
        if native is None or any(not native._exited(pid, proc.pid)
                                 for pid in native._members(proc.pid)):
            raise
    proc.wait(timeout=_STOP_GRACE_S)


def supervise(command: Sequence[str], *, timeout_s: float, memory_bytes: int,
              sample: Callable[[int, bool], tuple[int, bool]] | None = None
              ) -> SupervisionResult:
    """Start one isolated runner, with no retries and no child output in diagnostics."""
    start = time.monotonic()
    peak = 0
    proc = None
    watchdog = None
    native = None
    status = "monitor_failure"
    interrupted = False
    execution_exited = False

    def interrupt(_signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True

    handlers = {sig: signal.signal(sig, interrupt)
                for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        native = DarwinRSS() if sys.platform == "darwin" or sample is None else None
        monitor = sample if sample is not None else native
        assert monitor is not None
        with tempfile.TemporaryFile() as stats:
            fd = stats.fileno()
            proc = subprocess.Popen(
                command, start_new_session=True, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                env={**os.environ, _PEAK_FD: str(fd)}, pass_fds=(fd,))
            assert proc.stdin is not None
            watchdog = subprocess.Popen(
                [sys.executable, "-S", "-c", _WATCHDOG, str(proc.pid),
                 str(start + timeout_s), str(_HEARTBEAT_S)],
                start_new_session=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL)
            assert watchdog.stdin is not None and watchdog.stdout is not None
            with proc.stdin as gate, watchdog.stdout as ready:
                started = False
                watchdog_ready = False
                while True:
                    if interrupted:
                        status = "interrupted"
                        break
                    if time.monotonic() - start >= timeout_s:
                        status = "time_limit"
                        break
                    returncode = proc.poll()
                    try:
                        rss, live = monitor(proc.pid, returncode is None)
                        if rss <= 0:
                            raise MonitorUnavailable("empty execution RSS sample")
                        if native is not None:
                            rss += native._rss(watchdog.pid)
                    except Exception:  # noqa: BLE001 - any probe failure invalidates the run
                        raise MonitorUnavailable("RSS probe failed") from None
                    if returncode is None and not live and proc.poll() is None:
                        raise MonitorUnavailable("execution group has no inspectable live tasks")
                    execution_exited = returncode is not None and not live
                    if rss > peak:
                        peak = rss
                        if os.pwrite(fd, _RSS.pack(peak), 0) != _RSS.size:
                            raise MonitorUnavailable("cannot publish the RSS sample")
                    if peak > memory_bytes:
                        status = "memory_limit"
                        break
                    if time.monotonic() - start >= timeout_s:
                        status = "time_limit"
                        break
                    if interrupted:
                        status = "interrupted"
                        break
                    if watchdog.poll() is not None:
                        raise MonitorUnavailable("execution watchdog exited")
                    if not watchdog_ready and select.select([ready], [], [], 0)[0]:
                        watchdog_ready = ready.read(1) == b"1"
                        if not watchdog_ready:
                            raise MonitorUnavailable("execution watchdog failed to start")
                    if watchdog_ready:
                        watchdog.stdin.write(b"1")
                        watchdog.stdin.flush()
                    if returncode is not None and (returncode != 0 or not live):
                        if returncode == 0 and started:
                            watchdog.stdin.write(b"0")
                            watchdog.stdin.flush()
                            watchdog.wait(timeout=_HEARTBEAT_S)
                            if watchdog.returncode != 0:
                                raise MonitorUnavailable("execution watchdog did not certify exit")
                            if interrupted:
                                status = "interrupted"
                            else:
                                status = ("time_limit" if time.monotonic() - start >= timeout_s
                                          else "complete")
                        else:
                            status = "child_failure"
                        break
                    if not started and returncode is None and watchdog_ready:
                        gate.write(_GO)
                        gate.flush()
                        started = True
                    time.sleep(_POLL_S)
    except (OSError, MonitorUnavailable, ValueError, subprocess.TimeoutExpired):
        status = "time_limit" if time.monotonic() - start >= timeout_s else "monitor_failure"
    finally:
        try:
            if proc is not None and not execution_exited:
                grace_s = min(_STOP_GRACE_S, timeout_s - (time.monotonic() - start))
                try:
                    _stop_tree(proc, grace_s if status == "interrupted" else 0, native)
                except (OSError, MonitorUnavailable, subprocess.TimeoutExpired):
                    status = "termination_failure"
        finally:
            try:
                if watchdog is not None:
                    assert watchdog.stdin is not None
                    with suppress(BrokenPipeError):
                        watchdog.stdin.close()
                    try:
                        watchdog.wait(timeout=_HEARTBEAT_S)
                    except subprocess.TimeoutExpired:
                        watchdog.kill()
                        watchdog.wait()
            finally:
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)
    return SupervisionResult(status, None if proc is None else proc.returncode,
                             time.monotonic() - start, peak)
