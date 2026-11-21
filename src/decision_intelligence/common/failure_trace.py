"""Outcome-free exception locations, captured before a worker traceback is lost in transport."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class TracedTaskError(RuntimeError):
    """RuntimeError with a pickle-safe, message-free originating exception chain."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.failure_trace: list[dict[str, Any]] = []


def failure_trace(exc: BaseException) -> list[dict[str, Any]]:
    """Types and frame locations only: never messages, source lines, locals or remote text.

    A multiprocessing RemoteTraceback embeds arbitrary exception messages/source text; do not
    parse it. TracedTaskError carries the original structured chain across the process boundary.
    Explicit causes take precedence over unsuppressed implicit contexts, as in Python tracebacks.
    """
    chain: list[dict[str, Any]] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    relation = "raised"
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        frames: list[dict[str, Any]] = []
        tb = current.__traceback__
        while tb is not None:
            code = tb.tb_frame.f_code
            frames.append({"file": Path(code.co_filename).name, "line": tb.tb_lineno,
                           "function": code.co_name})
            tb = tb.tb_next
        chain.append({"type": type(current).__name__, "relation": relation, "frames": frames})
        if isinstance(current, TracedTaskError):
            chain.extend(current.failure_trace)
            break
        if current.__cause__ is not None:
            current, relation = current.__cause__, "cause"
        elif not current.__suppress_context__:
            current, relation = current.__context__, "context"
        else:
            break
    return chain
