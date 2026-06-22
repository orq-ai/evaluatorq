"""Launcher for the FastHTML dashboard.

Exposes ``serve(roots, *, host, port)`` which wires the loguru bridge,
then hands off to uvicorn.  Import-time side-effects are kept to a minimum
so this module can be imported safely even when fasthtml / uvicorn are absent
(``ensure_fasthtml`` performs the runtime guard).
"""

from __future__ import annotations

import logging
from pathlib import Path  # noqa: TC003

from loguru import logger


def ensure_fasthtml() -> None:
    """Exit with code 1 + install hint when fasthtml or uvicorn are missing."""
    try:
        import fasthtml  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError:
        import typer

        typer.echo(
            'The dashboard requires "dashboard" extra:\n'
            '  pip install "evaluatorq[dashboard]"',
            err=True,
        )
        raise typer.Exit(code=1)


class _InterceptHandler(logging.Handler):
    """Canonical loguru bridge: map levelno to a loguru level with numeric fallback.

    Walks the call-stack to find the real caller depth so loguru reports the
    originating file/line rather than the logging-module internals.
    """

    def emit(self, record: logging.LogRecord) -> None:
        # Resolve the loguru level name; fall back to the numeric level when
        # the name is not registered (e.g. for custom numeric levels).
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Walk up the stack to find the frame that actually issued the log
        # call, skipping frames that belong to the stdlib logging machinery.
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def serve(
    roots: list[Path] | None,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> None:
    """Start the FastHTML dashboard under uvicorn.

    Args:
        roots: Directories to scan for run reports.  ``None`` uses the
            production defaults defined in ``evaluatorq.dashboard.library``.
        host:  Bind address (default ``127.0.0.1``).
        port:  TCP port (default ``8080``).
    """
    ensure_fasthtml()

    # Route all stdlib logging (uvicorn access-log, starlette, etc.) through
    # loguru.  ``force=True`` replaces any pre-existing root-logger handlers.
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)

    import uvicorn

    from evaluatorq.dashboard.app import build_app

    uvicorn.run(build_app(roots), host=host, port=port)
