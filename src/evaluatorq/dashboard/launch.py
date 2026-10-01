"""Launcher for the FastHTML dashboard.

Exposes ``serve(roots, *, host, port, open_browser)`` which wires the loguru bridge,
then hands off to uvicorn.  Import-time side-effects are kept to a minimum
so this module can be imported safely even when fasthtml / uvicorn are absent
(``ensure_fasthtml`` performs the runtime guard).
"""

from __future__ import annotations

import logging
import os
import queue
import secrets
import socket
import sys
import threading
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO
from urllib.request import urlopen

from loguru import logger


def ensure_fasthtml() -> None:
    """Exit with code 1 + install hint when fasthtml or uvicorn are missing."""
    try:
        import fasthtml  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError:
        import typer

        typer.echo(
            'The dashboard requires "dashboard" extra. Install with: uv add "evaluatorq[dashboard]" '
            '(or: python -m pip install "evaluatorq[dashboard]")',
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


class _DroppingConsoleSink:
    """Write log lines from a bounded queue on a thread; drop lines while the console stops draining.

    A terminal that stops reading stdout (a hidden pane, a paused pipe) makes the
    next write block, and a blocking write on the event loop hangs every route and
    ignores SIGTERM until the reader catches up. loguru's ``enqueue=True`` does not
    help: its queue is a pipe that fills the same way. A bounded in-process queue
    never blocks the caller; a note of what was dropped follows once writes resume.
    """

    def __init__(self, stream: TextIO, maxsize: int = 10_000) -> None:
        self._queue: queue.Queue[str] = queue.Queue(maxsize)
        self._dropped = 0
        self._dropped_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._stop_requested = threading.Event()
        self._thread = threading.Thread(target=self._drain, args=(stream,), name='console-log', daemon=True)
        self._thread.start()

    def __call__(self, message: str) -> None:
        with self._state_lock:
            if self._stop_requested.is_set():
                return
            try:
                self._queue.put_nowait(str(message))
            except queue.Full:
                with self._dropped_lock:
                    self._dropped += 1

    def write(self, message: str) -> None:
        """Use Loguru's stream sink so ``logger.remove()`` calls ``stop()``."""
        self(message)

    @property
    def active(self) -> bool:
        """Whether the writer can still receive lines from a reused bridge."""
        return not self._stop_requested.is_set() and self._thread.is_alive()

    def stop(self) -> None:
        """Drain queued lines on removal without hanging forever on a stalled terminal."""
        with self._state_lock:
            self._stop_requested.set()
        self._thread.join(timeout=2)
        if self._thread.is_alive():
            self._disable()

    def _disable(self) -> None:
        """Stop accepting logs after the output stream breaks and release queued text."""
        with self._state_lock:
            self._stop_requested.set()
            self._queue = queue.Queue(self._queue.maxsize)

    def _drain(self, stream: TextIO) -> None:
        while not self._stop_requested.is_set() or not self._queue.empty():
            try:
                line = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            with self._dropped_lock:
                dropped, self._dropped = self._dropped, 0
            try:
                if dropped:
                    stream.write(f'... dropped {dropped} log lines while the console was not draining\n')
                stream.write(line)
                stream.flush()
            except (BrokenPipeError, OSError, ValueError):
                self._disable()
                return
        with self._dropped_lock:
            dropped, self._dropped = self._dropped, 0
        if dropped:
            try:
                stream.write(f'... dropped {dropped} log lines while the console was not draining\n')
                stream.flush()
            except (BrokenPipeError, OSError, ValueError):
                self._disable()


@dataclass(frozen=True)
class _ConsoleBridge:
    logger: object
    stream: TextIO
    level: int | str
    sink: _DroppingConsoleSink


_console_bridge: _ConsoleBridge | None = None


def _install_log_bridge() -> None:
    """Route all stdlib logging (uvicorn access-log, starlette, etc.) through
    loguru.  ``force=True`` replaces any pre-existing root-logger handlers.

    Defaults to INFO so third-party DEBUG chatter (watchfiles reload scans,
    uvicorn internals) stays out of the console, and quiets httpx's per-request
    INFO line. Override with ``EVALUATORQ_LOG_LEVEL=DEBUG`` (or any level name)
    when diagnosing; the override shows httpx lines again.
    """
    global _console_bridge
    override = os.environ.get('EVALUATORQ_LOG_LEVEL', '').upper()
    level: int | str = override or logging.INFO
    bridge = _console_bridge
    if not (
        bridge is not None
        and bridge.logger is logger
        and bridge.stream is sys.stderr
        and bridge.level == level
        and bridge.sink.active
    ):
        logger.remove()
        sink = _DroppingConsoleSink(sys.stderr)
        logger.add(sink, level=level, colorize=True)
        _console_bridge = _ConsoleBridge(logger, sys.stderr, level, sink)
    logging.basicConfig(handlers=[_InterceptHandler()], level=level, force=True)
    # One INFO line per Orq call; a run classifies hundreds of traces.
    # Undo that suppression when a later install requests diagnostic output.
    logging.getLogger('httpx').setLevel(logging.NOTSET if override else logging.WARNING)


# Env var carrying the JSON-encoded roots to the reload subprocess. uvicorn's
# --reload re-imports the app in a child process, so runtime args can't be
# passed directly — the factory below reads them back from here.
_ROOTS_ENV = 'EVALUATORQ_DASHBOARD_ROOTS'


def _load_local_env() -> None:
    """Read launch-directory settings in both the reloader and its worker."""
    from dotenv import load_dotenv

    load_dotenv(Path.cwd() / '.env', override=False)


def _browser_host(host: str) -> str:
    """Use a local address when uvicorn binds to every interface."""
    return {'0.0.0.0': '127.0.0.1', '::': '::1'}.get(host, host)  # noqa: S104


_BROWSER_NONCE_ENV = 'EVALUATORQ_DASHBOARD_LAUNCH_NONCE'


def _open_browser_when_ready(host: str, port: int, stop: threading.Event, nonce: str | None = None) -> None:
    """Open the dashboard once this server answers a bounded readiness probe."""
    browser_host = _browser_host(host)
    url_host = f'[{browser_host}]' if ':' in browser_host else browser_host
    url = f'http://{url_host}:{port}/'
    for _ in range(100):
        if stop.wait(0.1):
            return
        try:
            if nonce is None:
                with socket.create_connection((browser_host, port), timeout=0.2):
                    pass
            else:
                with urlopen(f'{url}_dashboard-ready', timeout=0.2) as response:  # noqa: S310
                    if response.read().decode() != nonce:
                        continue
        except (OSError, ValueError):
            continue
        try:
            if not webbrowser.open(url):
                logger.warning('Could not open a browser automatically; open {}', url)
        except (OSError, webbrowser.Error) as exc:
            logger.warning('Could not open a browser automatically: {}; open {}', exc, url)
        return
    logger.warning('Dashboard did not become ready for browser launch; open {} when it starts', url)


def build_app_from_env():
    """uvicorn factory used under ``--reload``.  Runs in the reloader's worker
    subprocess, so it reads the local environment, (re)installs the loguru
    bridge, and rebuilds the app from the roots stashed by ``serve``."""
    import json
    import os

    from evaluatorq.dashboard.app import build_app

    _load_local_env()
    _install_log_bridge()
    raw = os.environ.get(_ROOTS_ENV)
    roots = [Path(p) for p in json.loads(raw)] if raw else None
    app = build_app(roots)
    nonce = os.environ.get(_BROWSER_NONCE_ENV)
    if nonce:
        from starlette.responses import PlainTextResponse

        @app.get('/_dashboard-ready')
        def dashboard_ready() -> PlainTextResponse:
            return PlainTextResponse(nonce)

    return app


def serve(
    roots: list[Path] | None,
    *,
    host: str = '127.0.0.1',
    port: int = 8080,
    open_browser: bool = True,
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
        open_browser: Open the dashboard once it starts (default ``True``).
    """
    import os

    ensure_fasthtml()
    _load_local_env()
    _install_log_bridge()

    import json

    import uvicorn

    import evaluatorq

    if roots is None:
        os.environ.pop(_ROOTS_ENV, None)
    else:
        os.environ[_ROOTS_ENV] = json.dumps([str(p) for p in roots])

    pkg_dir = str(Path(evaluatorq.__file__).parent)
    nonce = secrets.token_urlsafe(24) if open_browser else None
    if nonce is not None:
        os.environ[_BROWSER_NONCE_ENV] = nonce
    else:
        os.environ.pop(_BROWSER_NONCE_ENV, None)
    stop = threading.Event()
    browser_thread: threading.Thread | None = None
    if open_browser:
        try:
            with socket.create_connection((_browser_host(host), port), timeout=0.2):
                logger.warning('Port {} is already in use; skipping automatic browser launch', port)
        except OSError:
            browser_thread = threading.Thread(
                target=_open_browser_when_ready, args=(host, port, stop, nonce), name='dashboard-browser', daemon=True
            )
            browser_thread.start()
    try:
        uvicorn.run(
            'evaluatorq.dashboard.launch:build_app_from_env',
            factory=True,
            host=host,
            port=port,
            reload=True,
            reload_dirs=[pkg_dir],
            # uvicorn's own dictConfig writes the access log straight to stdout, past the
            # loguru bridge and its queue thread; None leaves its loggers propagating to root.
            log_config=None,
        )
    finally:
        stop.set()
        if browser_thread is not None:
            browser_thread.join(timeout=0.3)
        os.environ.pop(_BROWSER_NONCE_ENV, None)
