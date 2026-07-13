"""Launcher for the FastHTML dashboard.

Exposes ``serve(roots, *, host, port)`` which wires the loguru bridge,
then hands off to uvicorn.  Import-time side-effects are kept to a minimum
so this module can be imported safely even when fasthtml / uvicorn are absent
(``ensure_fasthtml`` performs the runtime guard).
"""

from __future__ import annotations

import logging
from pathlib import Path

from loguru import logger


def ensure_fasthtml() -> None:
    """Exit with code 1 + install hint when fasthtml or uvicorn are missing."""
    try:
        import fasthtml  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError:
        import typer

        typer.echo(
            'The dashboard requires "dashboard" extra:\n  pip install "evaluatorq[dashboard]"',
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
        while frame is not None and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def _install_log_bridge() -> None:
    """Route all stdlib logging (uvicorn access-log, starlette, etc.) through
    loguru.  ``force=True`` replaces any pre-existing root-logger handlers."""
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)


# Env var carrying the JSON-encoded roots to the reload subprocess. uvicorn's
# --reload re-imports the app in a child process, so runtime args can't be
# passed directly — the factory below reads them back from here.
_ROOTS_ENV = 'EVALUATORQ_DASHBOARD_ROOTS'


def build_app_from_env():
    """uvicorn factory used under ``--reload``.  Runs in the reloader's worker
    subprocess, so it (re)installs the loguru bridge and rebuilds the app from
    the roots stashed in ``_ROOTS_ENV`` by :func:`serve`."""
    import json
    import os

    from evaluatorq.dashboard.app import build_app

    _install_log_bridge()
    raw = os.environ.get(_ROOTS_ENV)
    roots = [Path(p) for p in json.loads(raw)] if raw else None
    return build_app(roots)


def serve(
    roots: list[Path] | None,
    *,
    host: str = '127.0.0.1',
    port: int = 8080,
) -> None:
    """Start the FastHTML dashboard under uvicorn with hot-reload always on.

    Reload watches the ``evaluatorq`` package source and restarts the worker on
    any change, so an edit is picked up without a manual kill/restart (the
    dashboard is a dev preview; a stale in-memory build was a recurring
    footgun). ``roots`` are handed to the reload worker via ``_ROOTS_ENV`` since
    uvicorn re-imports the app in a subprocess.

    Args:
        roots: Directories to scan for run reports.  ``None`` uses the
            production defaults defined in ``evaluatorq.dashboard.library``.
        host:  Bind address (default ``127.0.0.1``).
        port:  TCP port (default ``8080``).
    """
    import os

    ensure_fasthtml()
    _install_log_bridge()

    import json

    import uvicorn

    import evaluatorq

    if roots is None:
        os.environ.pop(_ROOTS_ENV, None)
    else:
        os.environ[_ROOTS_ENV] = json.dumps([str(p) for p in roots])

    pkg_dir = str(Path(evaluatorq.__file__).parent)
    uvicorn.run(
        'evaluatorq.dashboard.launch:build_app_from_env',
        factory=True,
        host=host,
        port=port,
        reload=True,
        reload_dirs=[pkg_dir],
    )
