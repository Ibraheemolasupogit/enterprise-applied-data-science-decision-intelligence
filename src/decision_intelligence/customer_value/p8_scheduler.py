"""Ordered, bounded process scheduling for M11 P8."""

from collections import deque
from collections.abc import Callable, Generator, Iterable
from concurrent.futures import ProcessPoolExecutor
from itertools import groupby, islice
from multiprocessing import get_context
from typing import Any

from .predictive_cases import Task

Result = dict[str, Any]
TaskFn = Callable[[Task], Result]


def ordered_pool(tasks: Iterable[Task], workers: int, fn: TaskFn
                 ) -> Generator[Result, None, None]:
    """Yield in submission order and join the pool even on a consumer failure."""
    source = iter(tasks)
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"))
    try:
        pending = deque(pool.submit(fn, task) for task in islice(source, 2 * workers))
        while pending:
            result = pending.popleft().result()
            next_task = next(source, None)
            if next_task is not None:
                pending.append(pool.submit(fn, next_task))
            yield result
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def hybrid_tasks(tasks: Iterable[Task], fn: TaskFn) -> Generator[Result, None, None]:
    """Join each R2 pair before launching its replacement. Other studies use three workers."""
    for study, group in groupby(tasks, key=lambda task: task.study):
        if study != "R2":
            yield from ordered_pool(group, 3, fn)
            continue
        while pair := tuple(islice(group, 2)):
            yield from ordered_pool(pair, 2, fn)


def isolated_task(task: Task, fn: TaskFn) -> list[Result]:
    """Keep the fit outside the runner and preserve the phase's one-element output list."""
    return list(ordered_pool((task,), 1, fn))
