# ruff: noqa: S101

from __future__ import annotations

import json

from typer.testing import CliRunner

from evaluatorq.simulation.cli import app as simulation_app


def test_runs_json_skips_valid_json_with_an_invalid_report_shape(tmp_path):
    (tmp_path / 'invalid-report.json').write_text('[]')

    result = CliRunner().invoke(simulation_app, ['runs', str(tmp_path), '--json'])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == []
    assert 'Warning: 1 malformed file(s) skipped.' in result.stderr
