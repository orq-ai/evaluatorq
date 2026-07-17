# ruff: noqa: S101

"""Tests for top-level CLI error handling."""

from __future__ import annotations

import pytest

from evaluatorq.common.cli_errors import run_guarded


def test_unexpected_exception_becomes_exit_1(capsys: pytest.CaptureFixture[str]) -> None:
    def boom() -> None:
        raise RuntimeError('kaboom')

    with pytest.raises(SystemExit) as exc_info:
        run_guarded(boom)

    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert 'kaboom' in err
    assert 'EQ_DEBUG=1' in err


def test_systemexit_code_preserved() -> None:
    def usage_error() -> None:
        raise SystemExit(2)

    with pytest.raises(SystemExit) as exc_info:
        run_guarded(usage_error)

    assert exc_info.value.code == 2


def test_keyboardinterrupt_propagates() -> None:
    def interrupted() -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_guarded(interrupted)


def test_debug_env_reraises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('EQ_DEBUG', '1')

    def boom() -> None:
        raise RuntimeError('kaboom')

    with pytest.raises(RuntimeError, match='kaboom'):
        run_guarded(boom)


def test_main_converts_unexpected_error(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from evaluatorq import cli as cli_module

    @cli_module.app.command('boom-test', hidden=True)
    def _boom() -> None:  # pragma: no cover - registered for this test only
        raise RuntimeError('e2e kaboom')

    monkeypatch.setattr('sys.argv', ['evaluatorq', 'boom-test'])

    with pytest.raises(SystemExit) as exc_info:
        cli_module.main()

    assert exc_info.value.code == 1
    assert 'e2e kaboom' in capsys.readouterr().err
