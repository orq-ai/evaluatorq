# ruff: noqa: S101

from typer.testing import CliRunner

from evaluatorq.redteam.cli import app as redteam_app


def test_dash_h_shows_help_on_subcommand():
    result = CliRunner().invoke(redteam_app, ['run', '-h'])

    assert result.exit_code == 0
    assert 'Usage' in result.output
