"""Report trace-loading progress to whichever run owns the current task."""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

# ponytail: a context variable reaches the sources without threading a callback through every
# loader signature; asyncio copies it into the tasks a load spawns.
_REPORTER: ContextVar[Callable[[int, int], None] | None] = ContextVar('trace_load_progress', default=None)


def report_load_progress(done: int, total: int) -> None:
    """Tell the owning run how many traces are loaded so far; a no-op outside a run."""
    reporter = _REPORTER.get()
    if reporter is not None:
        reporter(done, total)


def set_load_reporter(reporter: Callable[[int, int], None]) -> None:
    """Bind a reporter for the current context, before the loading task is created."""
    _REPORTER.set(reporter)
