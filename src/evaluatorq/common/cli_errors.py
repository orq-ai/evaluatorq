"""Shared CLI error helpers for evaluatorq."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from collections.abc import Callable

_ISSUES_URL = 'https://github.com/orq-ai/evaluatorq/issues'


def emit_error(exc: object) -> None:
    """Print a one-line CLI error to stderr."""
    typer.echo(f'Error: {exc}', err=True)


def run_guarded(app_callable: Callable[[], object]) -> None:
    """Run a CLI app and render unexpected exceptions as a concise error."""
    try:
        app_callable()
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception as exc:
        if os.environ.get('EQ_DEBUG'):
            raise
        emit_error(exc)
        typer.echo(f'Set EQ_DEBUG=1 to see the full traceback; report at {_ISSUES_URL}', err=True)
        raise SystemExit(1) from exc
