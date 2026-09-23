"""Tests for evaluatorq.dashboard.launch (FastHTML launcher + loguru bridge)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner


# ---------------------------------------------------------------------------
# ensure_fasthtml
# ---------------------------------------------------------------------------


def test_ensure_fasthtml_exits_with_hint_when_fasthtml_missing() -> None:
    """ensure_fasthtml() raises typer.Exit(1) + prints install hint when fasthtml import fails."""
    import typer

    # Mask fasthtml so the import inside ensure_fasthtml raises ImportError.
    with patch.dict(sys.modules, {'fasthtml': None}):
        from evaluatorq.dashboard.launch import ensure_fasthtml

        with pytest.raises(typer.Exit) as exc:
            ensure_fasthtml()

    assert exc.value.exit_code == 1


def test_ensure_fasthtml_exits_with_hint_when_uvicorn_missing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ensure_fasthtml() raises typer.Exit(1) when uvicorn is absent."""
    import typer

    with patch.dict(sys.modules, {'uvicorn': None}):
        from evaluatorq.dashboard.launch import ensure_fasthtml

        with pytest.raises(typer.Exit) as exc:
            ensure_fasthtml()

    assert exc.value.exit_code == 1
    # The hint should mention the dashboard extra.
    err = capsys.readouterr().err
    assert 'evaluatorq[dashboard]' in err


# ---------------------------------------------------------------------------
# _InterceptHandler
# ---------------------------------------------------------------------------


def test_intercept_handler_forwards_record_without_raising() -> None:
    """_InterceptHandler.emit() should not raise for a standard log record."""
    from evaluatorq.dashboard.launch import _InterceptHandler

    handler = _InterceptHandler()
    record = logging.LogRecord(
        name='uvicorn',
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='test message',
        args=(),
        exc_info=None,
    )

    # ``logger`` is bound at module level via ``from loguru import logger``;
    # patch the binding in evaluatorq.dashboard.launch to avoid real I/O.
    with patch('evaluatorq.dashboard.launch.logger') as mock_logger:
        mock_logger.level.return_value.name = 'INFO'
        mock_logger.opt.return_value.log = MagicMock()
        handler.emit(record)
        # Verify opt() was called (with depth + exception keyword args).
        assert mock_logger.opt.called


def test_intercept_handler_uses_numeric_fallback_for_unknown_level() -> None:
    """_InterceptHandler.emit() uses the numeric level when the name is unregistered."""
    from evaluatorq.dashboard.launch import _InterceptHandler

    handler = _InterceptHandler()
    record = logging.LogRecord(
        name='test',
        level=42,
        pathname=__file__,
        lineno=1,
        msg='custom numeric level',
        args=(),
        exc_info=None,
    )
    record.levelname = 'CUSTOM_UNKNOWN'

    with patch('evaluatorq.dashboard.launch.logger') as mock_logger:
        mock_logger.level.side_effect = ValueError('unknown level')
        mock_logger.opt.return_value.log = MagicMock()
        handler.emit(record)
        # opt().log() should have been called with the numeric level 42.
        call_args = mock_logger.opt.return_value.log.call_args
        assert call_args.args[0] == 42


# ---------------------------------------------------------------------------
# _install_log_bridge — default level
# ---------------------------------------------------------------------------


def test_log_bridge_defaults_to_info(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bridge installs at INFO by default so third-party DEBUG stays silent."""
    from evaluatorq.dashboard import launch

    monkeypatch.delenv('EVALUATORQ_LOG_LEVEL', raising=False)
    with patch('evaluatorq.dashboard.launch.logging.basicConfig') as mock_cfg, patch('evaluatorq.dashboard.launch.logger'):
        launch._install_log_bridge()

    assert mock_cfg.call_args.kwargs['level'] == logging.INFO


def test_log_bridge_respects_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """EVALUATORQ_LOG_LEVEL=DEBUG re-enables debug capture."""
    from evaluatorq.dashboard import launch

    monkeypatch.setenv('EVALUATORQ_LOG_LEVEL', 'debug')
    with patch('evaluatorq.dashboard.launch.logging.basicConfig') as mock_cfg, patch('evaluatorq.dashboard.launch.logger'):
        launch._install_log_bridge()

    assert mock_cfg.call_args.kwargs['level'] == 'DEBUG'


def test_log_bridge_writes_the_console_off_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stalled terminal must not block log writes, and with them every request."""
    from evaluatorq.dashboard import launch

    monkeypatch.delenv('EVALUATORQ_LOG_LEVEL', raising=False)
    with patch('evaluatorq.dashboard.launch.logging.basicConfig'), patch('evaluatorq.dashboard.launch.logger') as mock_logger:
        launch._install_log_bridge()

    mock_logger.remove.assert_called_once_with()
    assert isinstance(mock_logger.add.call_args.args[0], launch._DroppingConsoleSink)


def test_console_sink_drops_lines_instead_of_blocking_when_the_console_stalls() -> None:
    """Log calls return at once while the console stalls; the drop count is reported once it resumes."""
    import io
    import threading

    from evaluatorq.dashboard import launch

    stalled = threading.Event()
    resume = threading.Event()
    written = io.StringIO()

    class StalledStream(io.TextIOBase):
        def write(self, text: str) -> int:
            stalled.set()
            resume.wait(timeout=5)
            return written.write(text)

        def flush(self) -> None:
            return None

    sink = launch._DroppingConsoleSink(StalledStream(), maxsize=2)
    sink('first\n')
    assert stalled.wait(timeout=2), 'the drain thread never reached the console'
    finished = threading.Event()

    def log_more() -> None:
        for n in range(5):
            sink(f'line {n}\n')
        finished.set()

    threading.Thread(target=log_more, daemon=True).start()
    assert finished.wait(timeout=2), 'logging blocked on a stalled console'
    resume.set()
    deadline = threading.Event()
    for _ in range(50):
        if 'dropped' in written.getvalue() and written.getvalue().count('line ') == 2:
            break
        deadline.wait(0.05)
    output = written.getvalue()
    assert output.startswith('first\n')
    assert '... dropped 3 log lines while the console was not draining\n' in output
    assert output.count('line ') == 2


def test_console_sink_flushes_queued_lines_when_removed() -> None:
    import io

    from loguru import logger

    from evaluatorq.dashboard import launch

    stream = io.StringIO()
    sink = launch._DroppingConsoleSink(stream)
    handler_id = logger.add(sink, colorize=False)
    try:
        for index in range(100):
            logger.info('queued line {}', index)
    finally:
        logger.remove(handler_id)

    assert not sink._thread.is_alive()
    assert stream.getvalue().count('queued line ') == 100


def test_console_sink_disables_itself_when_output_pipe_breaks() -> None:
    import io
    import threading

    from evaluatorq.dashboard import launch

    failed = threading.Event()

    class BrokenStream(io.TextIOBase):
        def write(self, text: str) -> int:
            failed.set()
            raise BrokenPipeError('reader exited')

        def flush(self) -> None:
            return None

    sink = launch._DroppingConsoleSink(BrokenStream())
    sink.write('first\n')
    assert failed.wait(timeout=2)
    sink._thread.join(timeout=2)
    assert not sink._thread.is_alive()
    assert sink._stop_requested.is_set()
    sink.write('ignored after stream failure\n')
    assert sink._queue.empty()


@pytest.mark.parametrize(
    ('override', 'expected'),
    [(None, logging.WARNING), ('DEBUG', logging.NOTSET)],
)
def test_log_bridge_quiets_httpx_only_at_the_default_level(
    monkeypatch: pytest.MonkeyPatch, override: str | None, expected: int
) -> None:
    """httpx logs one INFO line per Orq call; a run makes hundreds, so the default level drops them."""
    from evaluatorq.dashboard import launch

    if override is None:
        monkeypatch.delenv('EVALUATORQ_LOG_LEVEL', raising=False)
    else:
        monkeypatch.setenv('EVALUATORQ_LOG_LEVEL', override)
    httpx_logger = logging.getLogger('httpx')
    monkeypatch.setattr(httpx_logger, 'level', logging.NOTSET)
    with patch('evaluatorq.dashboard.launch.logging.basicConfig'), patch('evaluatorq.dashboard.launch.logger'):
        launch._install_log_bridge()

    assert httpx_logger.level == expected


# ---------------------------------------------------------------------------
# CLI help smoke-test
# ---------------------------------------------------------------------------


def test_eq_dashboard_help() -> None:
    """eq dashboard --help names all three stores scanned with no path."""
    from evaluatorq.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ['dashboard', '--help'])
    assert result.exit_code == 0
    for store in ('.evaluatorq/runs/', '.evaluatorq/sim-runs/', '.evaluatorq/pairwise-runs/'):
        assert store in result.output


# ---------------------------------------------------------------------------
# serve() wiring (uvicorn.run patched — do NOT actually start a server)
# ---------------------------------------------------------------------------


def test_serve_calls_uvicorn_run(tmp_path: Path) -> None:
    """serve() calls uvicorn.run with the FastHTML app and correct host/port.

    ``build_app`` is a lazy import inside ``serve()`` so we patch it at its
    source (``evaluatorq.dashboard.app.build_app``) rather than on the
    ``launch`` module, which avoids importing fasthtml at test-collection time.
    """
    import uvicorn

    from evaluatorq.dashboard.launch import serve

    fake_asgi_app = MagicMock()

    with (
        patch('evaluatorq.dashboard.launch.ensure_fasthtml'),
        patch('evaluatorq.dashboard.launch.logging.basicConfig'),
        patch.object(uvicorn, 'run') as mock_run,
        patch('evaluatorq.dashboard.app.build_app', return_value=fake_asgi_app),
    ):
        serve([tmp_path], host='0.0.0.0', port=9999)

    assert mock_run.called
    call_kwargs = mock_run.call_args
    assert call_kwargs.kwargs.get('host') == '0.0.0.0'
    assert call_kwargs.kwargs.get('port') == 9999
    assert call_kwargs.kwargs.get('log_config') is None, 'uvicorn must log through the loguru bridge'


def test_eq_dashboard_accepts_multiple_paths(tmp_path: Path) -> None:
    """Multiple directory arguments are merged into one combined roots list."""
    from evaluatorq.cli import app

    a = tmp_path / 'repo-a'
    b = tmp_path / 'repo-b'
    a.mkdir()
    b.mkdir()

    runner = CliRunner()
    with patch('evaluatorq.dashboard.launch.serve') as mock_serve:
        result = runner.invoke(app, ['dashboard', str(a), str(b)])
    assert result.exit_code == 0
    roots = mock_serve.call_args.args[0]
    assert a in roots
    assert b in roots
    assert a / '.evaluatorq' / 'runs' in roots
    assert b / '.evaluatorq' / 'sim-runs' in roots
    assert a / 'pairwise-runs' in roots
    assert b / '.evaluatorq' / 'pairwise-runs' in roots


def test_eq_dashboard_rejects_nonexistent_path(tmp_path: Path) -> None:
    """A typo'd path must error, not silently become a parent root with a
    confident Direct-report URL for a file that does not exist."""
    from evaluatorq.cli import app

    runner = CliRunner()
    with patch('evaluatorq.dashboard.launch.serve') as mock_serve:
        result = runner.invoke(app, ['dashboard', str(tmp_path / 'no-such-dir')])
    assert result.exit_code != 0
    assert 'does not exist' in result.output
    mock_serve.assert_not_called()
