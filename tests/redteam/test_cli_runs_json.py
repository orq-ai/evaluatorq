# ruff: noqa: S101

from __future__ import annotations

import json

from typer.testing import CliRunner

from evaluatorq.redteam.cli import app as redteam_app


def test_runs_json_empty_dir_emits_empty_array(tmp_path):
    result = CliRunner().invoke(redteam_app, ['runs', str(tmp_path), '--json'])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == []


def test_runs_json_serialises_raw_fields(tmp_path):
    (tmp_path / 'r.json').write_text(
        json.dumps({
            'run_name': 'demo',
            'created_at': '2026-07-17T12:00:00Z',
            'pipeline': 'dynamic',
            'tested_agents': ['agent:x'],
            'summary': {'total_attacks': 3, 'vulnerability_rate': 0.42},
        })
    )

    result = CliRunner().invoke(redteam_app, ['runs', str(tmp_path), '--json'])

    assert result.exit_code == 0
    records = json.loads(result.stdout)
    assert records[0]['run_name'] == 'demo'
    assert records[0]['created_at'] == '2026-07-17T12:00:00Z'
    assert records[0]['vulnerability_rate'] == 0.42
    assert records[0]['total_attacks'] == 3
    assert records[0]['report_id']


def test_runs_json_skips_valid_json_with_an_invalid_report_shape(tmp_path):
    (tmp_path / 'invalid-report.json').write_text('[]')

    result = CliRunner().invoke(redteam_app, ['runs', str(tmp_path), '--json'])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == []
    assert 'Warning: 1 file(s) could not be parsed and were skipped.' in result.stderr


def test_runs_json_skips_valid_json_with_an_invalid_summary_shape(tmp_path):
    (tmp_path / 'invalid-summary.json').write_text(json.dumps({'summary': []}))

    result = CliRunner().invoke(redteam_app, ['runs', str(tmp_path), '--json'])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == []
    assert 'Warning: 1 file(s) could not be parsed and were skipped.' in result.stderr
