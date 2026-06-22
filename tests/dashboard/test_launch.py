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
    with patch.dict(sys.modules, {"fasthtml": None}):
        from evaluatorq.dashboard.launch import ensure_fasthtml

        with pytest.raises(typer.Exit) as exc:
            ensure_fasthtml()

    assert exc.value.exit_code == 1


def test_ensure_fasthtml_exits_with_hint_when_uvicorn_missing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ensure_fasthtml() raises typer.Exit(1) when uvicorn is absent."""
    import typer

    with patch.dict(sys.modules, {"uvicorn": None}):
        from evaluatorq.dashboard.launch import ensure_fasthtml

        with pytest.raises(typer.Exit) as exc:
            ensure_fasthtml()

    assert exc.value.exit_code == 1
    # The hint should mention the dashboard extra.
    err = capsys.readouterr().err
    assert "evaluatorq[dashboard]" in err


# ---------------------------------------------------------------------------
# _InterceptHandler
# ---------------------------------------------------------------------------


def test_intercept_handler_forwards_record_without_raising() -> None:
    """_InterceptHandler.emit() should not raise for a standard log record."""
    from evaluatorq.dashboard.launch import _InterceptHandler

    handler = _InterceptHandler()
    record = logging.LogRecord(
        name="uvicorn",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="test message",
        args=(),
        exc_info=None,
    )

    # ``logger`` is bound at module level via ``from loguru import logger``;
    # patch the binding in evaluatorq.dashboard.launch to avoid real I/O.
    with patch("evaluatorq.dashboard.launch.logger") as mock_logger:
        mock_logger.level.return_value.name = "INFO"
        mock_logger.opt.return_value.log = MagicMock()
        handler.emit(record)
        # Verify opt() was called (with depth + exception keyword args).
        assert mock_logger.opt.called


def test_intercept_handler_uses_numeric_fallback_for_unknown_level() -> None:
    """_InterceptHandler.emit() uses the numeric level when the name is unregistered."""
    from evaluatorq.dashboard.launch import _InterceptHandler

    handler = _InterceptHandler()
    record = logging.LogRecord(
        name="test",
        level=42,
        pathname=__file__,
        lineno=1,
        msg="custom numeric level",
        args=(),
        exc_info=None,
    )
    record.levelname = "CUSTOM_UNKNOWN"

    with patch("evaluatorq.dashboard.launch.logger") as mock_logger:
        mock_logger.level.side_effect = ValueError("unknown level")
        mock_logger.opt.return_value.log = MagicMock()
        handler.emit(record)
        # opt().log() should have been called with the numeric level 42.
        call_args = mock_logger.opt.return_value.log.call_args
        assert call_args.args[0] == 42


# ---------------------------------------------------------------------------
# CLI help smoke-tests (eq ui, redteam ui, sim ui)
# ---------------------------------------------------------------------------


def test_eq_ui_help() -> None:
    """eq ui --help exits 0 and lists the ui command."""
    # The module-level ``app`` already has the ``ui`` command registered.
    from evaluatorq.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["ui", "--help"])
    assert result.exit_code == 0
    # The help text should mention the fasthtml dashboard.
    assert "dashboard" in result.output.lower() or "path" in result.output.lower()


def test_redteam_ui_help() -> None:
    """redteam ui --help exits 0."""
    from evaluatorq.redteam.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["ui", "--help"])
    assert result.exit_code == 0


def test_sim_ui_help() -> None:
    """sim ui --help exits 0."""
    from evaluatorq.simulation.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["ui", "--help"])
    assert result.exit_code == 0


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
        patch("evaluatorq.dashboard.launch.ensure_fasthtml"),
        patch("evaluatorq.dashboard.launch.logging.basicConfig"),
        patch.object(uvicorn, "run") as mock_run,
        patch("evaluatorq.dashboard.app.build_app", return_value=fake_asgi_app),
    ):
        serve([tmp_path], host="0.0.0.0", port=9999)

    assert mock_run.called
    call_kwargs = mock_run.call_args
    assert call_kwargs.kwargs.get("host") == "0.0.0.0"
    assert call_kwargs.kwargs.get("port") == 9999
