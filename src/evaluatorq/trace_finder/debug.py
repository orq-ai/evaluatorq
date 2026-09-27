"""Opt-in trace-finder request and response diagnostics."""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

_cli_debug: ContextVar[bool] = ContextVar('trace_finder_cli_debug', default=False)


def enabled() -> bool:
    """Show trace content only for an explicit CLI flag or dashboard DEBUG level."""
    return _cli_debug.get() or os.environ.get('EVALUATORQ_LOG_LEVEL', '').upper() == 'DEBUG'


@contextmanager
def cli_debug(*, active: bool) -> Iterator[None]:
    """Scope CLI diagnostics to one find invocation, including its async tasks."""
    token = _cli_debug.set(active)
    try:
        yield
    finally:
        _cli_debug.reset(token)
