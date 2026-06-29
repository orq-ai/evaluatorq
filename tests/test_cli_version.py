"""Tests for top-level CLI behaviour (RES-1005): `eq --version` and bare `eq`."""

from __future__ import annotations

from importlib.metadata import version

from typer.testing import CliRunner

from evaluatorq.cli import app

runner = CliRunner()


def test_version_flag_prints_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert version("evaluatorq") in result.stdout


def test_bare_invocation_shows_help() -> None:
    # no_args_is_help=True → bare `eq` prints usage and exits non-zero.
    result = runner.invoke(app, [])
    assert "Usage:" in result.stdout
