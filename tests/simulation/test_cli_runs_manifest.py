# ruff: noqa: S101
"""Manifest-first behavior of the simulation `runs` CLI listing."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from evaluatorq.common.run_manifest import start_manifest
from evaluatorq.simulation.cli import app as sim_app

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


def test_runs_lists_manifest_backed_status_and_stats(tmp_path: Path) -> None:
    runs = tmp_path / 'sim-runs'
    runs.mkdir()
    report = runs / 'sim_20260101.json'
    report.write_text('{"mode": "run"}', encoding='utf-8')
    start_manifest(run_id='s1', surface='sim', run_name='manifest-sim', runs_dir=runs).complete(
        report_path=report,
        summary={
            'mode': 'run',
            'target_kind': 'orq_agent',
            'total_results': 4,
            'scorer_averages': {'goal_achieved': 0.75},
        },
    )

    result = runner.invoke(sim_app, ['runs', str(runs)])
    assert result.exit_code == 0, result.output
    assert 'manifest-sim' in result.stdout
    assert 'completed' in result.stdout
    assert 'goal_achieved' in result.stdout


def test_runs_shows_in_flight_run(tmp_path: Path) -> None:
    runs = tmp_path / 'sim-runs'
    runs.mkdir()
    w = start_manifest(run_id='live', surface='sim', run_name='live-sim', runs_dir=runs)
    w.start_stage('Simulating')

    result = runner.invoke(sim_app, ['runs', str(runs)])
    assert result.exit_code == 0, result.output
    assert 'live-sim' in result.stdout
    assert 'running' in result.stdout


def test_runs_json_does_not_crash_on_in_flight(tmp_path: Path) -> None:
    runs = tmp_path / 'sim-runs'
    runs.mkdir()
    start_manifest(run_id='live', surface='sim', run_name='live-sim', runs_dir=runs)

    result = runner.invoke(sim_app, ['runs', str(runs), '--json'])
    assert result.exit_code == 0, result.output
    records = json.loads(result.stdout)
    assert len(records) == 1
    assert records[0]['run_name'] == 'live-sim'
    assert records[0]['status'] == 'running'
    assert records[0]['total_results'] is None


def test_runs_legacy_fallback_still_works(tmp_path: Path) -> None:
    runs = tmp_path / 'sim-runs'
    runs.mkdir()
    (runs / 'legacy_20250101.json').write_text(
        json.dumps({
            'run_name': 'legacy-sim',
            'created_at': '2025-01-01T00:00:00+00:00',
            'mode': 'run',
            'target_kind': 'openai_model',
            'total_results': 1,
            'scorer_averages': {'goal_achieved': 0.5},
            'results': [],
        }),
        encoding='utf-8',
    )

    result = runner.invoke(sim_app, ['runs', str(runs)])
    assert result.exit_code == 0, result.output
    assert 'legacy-sim' in result.stdout
    assert 'completed' in result.stdout
