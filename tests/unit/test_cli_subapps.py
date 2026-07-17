# ruff: noqa: S101, SLF001

import typer
from typer.testing import CliRunner

from evaluatorq import cli as cli_module


def test_subapps_are_registered():
    app = typer.Typer()
    cli_module._register_subapps(app)
    result = CliRunner().invoke(app, ['--help'])
    assert result.exit_code == 0
    assert 'redteam' in result.output
    assert 'sim' in result.output
