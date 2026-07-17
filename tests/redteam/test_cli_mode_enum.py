from typer.testing import CliRunner

from evaluatorq.redteam.cli import app as redteam_app


def test_bogus_mode_rejected_early():
    result = CliRunner().invoke(redteam_app, ['run', '-t', 'agent:x', '--mode', 'bogus'])

    assert result.exit_code == 2  # usage error, not a late runtime failure
    assert 'bogus' in result.output or 'dynamic' in result.output
