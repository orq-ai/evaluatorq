# ruff: noqa: S101
"""Manifest-first behavior of the red team `runs` CLI listing."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from evaluatorq.common.run_manifest import start_manifest
from evaluatorq.redteam.cli import app as redteam_app

if TYPE_CHECKING:
    from pathlib import Path


def test_runs_lists_manifest_backed_status_and_stats(tmp_path: Path) -> None:
    runs = tmp_path / 'runs'
    runs.mkdir()
    report = runs / 'rt_20260101.json'
    report.write_text('{"pipeline": "dynamic", "summary": {}}', encoding='utf-8')
    start_manifest(run_id='r1', surface='redteam', run_name='manifest-run', runs_dir=runs).complete(
        report_path=report,
        summary={
            'pipeline': 'dynamic',
            'total_results': 5,
            'total_attacks': 5,
            'vulnerability_rate': 0.4,
            'tested_agents': ['agent:x'],
        },
    )

    result = CliRunner().invoke(redteam_app, ['runs', str(runs)])
    assert result.exit_code == 0, result.output
    assert 'manifest-run' in result.stdout
    assert 'completed' in result.stdout
    assert '40%' in result.stdout


def test_runs_shows_in_flight_run_status(tmp_path: Path) -> None:
    runs = tmp_path / 'runs'
    runs.mkdir()
    w = start_manifest(run_id='live', surface='redteam', run_name='live-run', runs_dir=runs)
    w.start_stage('Executing Attacks')

    result = CliRunner().invoke(redteam_app, ['runs', str(runs)])
    assert result.exit_code == 0, result.output
    assert 'live-run' in result.stdout
    assert 'running' in result.stdout


def test_runs_json_does_not_crash_on_in_flight(tmp_path: Path) -> None:
    runs = tmp_path / 'runs'
    runs.mkdir()
    start_manifest(run_id='live', surface='redteam', run_name='live-run', runs_dir=runs)

    result = CliRunner().invoke(redteam_app, ['runs', str(runs), '--json'])
    assert result.exit_code == 0, result.output
    records = json.loads(result.stdout)
    assert len(records) == 1
    assert records[0]['run_name'] == 'live-run'
    assert records[0]['status'] == 'running'
    assert records[0]['total_attacks'] is None


def test_runs_legacy_fallback_still_works(tmp_path: Path) -> None:
    runs = tmp_path / 'runs'
    runs.mkdir()
    (runs / 'legacy_20250101.json').write_text(
        json.dumps({
            'run_name': 'legacy',
            'created_at': '2025-01-01T00:00:00Z',
            'pipeline': 'static',
            'tested_agents': [],
            'summary': {'total_attacks': 2, 'vulnerability_rate': 0.0},
        }),
        encoding='utf-8',
    )

    result = CliRunner().invoke(redteam_app, ['runs', str(runs)])
    assert result.exit_code == 0, result.output
    assert 'legacy' in result.stdout
    assert 'completed' in result.stdout


def test_backfilled_sidecar_keeps_the_full_read_stats(tmp_path: Path) -> None:
    """Opening the dashboard writes a manifest sidecar for a legacy report; the
    `runs` row must then carry the same stats it carried from the full read.

    Regression: the first backfill wrote only ``total_results``, so browsing the
    dashboard silently blanked this table's pipeline / agents / rate columns.
    """
    from evaluatorq.dashboard.library import scan

    runs = tmp_path / 'runs'
    runs.mkdir()
    (runs / 'rt_20260101_000000.json').write_text(
        json.dumps({
            'version': '2.0.0',
            'created_at': '2026-01-01T00:00:00+00:00',
            'pipeline': 'dynamic',
            'categories_tested': ['ASI01'],
            'tested_agents': ['agent:x'],
            'total_results': 4,
            'results': [],
            'summary': {'total_attacks': 4, 'vulnerability_rate': 0.25, 'resistance_rate': 0.75},
        }),
        encoding='utf-8',
    )
    stats = ('pipeline', 'tested_agents', 'total_attacks', 'vulnerability_rate')

    def _row() -> dict[str, object]:
        result = CliRunner().invoke(redteam_app, ['runs', str(runs), '--json'])
        assert result.exit_code == 0, result.output
        rows = json.loads(result.stdout)
        assert len(rows) == 1
        return {k: rows[0][k] for k in stats}

    before = _row()
    scan([runs])  # the dashboard visit that backfills the sidecar
    assert (runs / '.manifests' / 'rt_20260101_000000.json').exists()
    assert _row() == before


def test_runs_ignores_stage_artifacts(tmp_path: Path) -> None:
    """`save='detail'` writes 01_/02_/03_ stage artifacts beside reports. They are
    not runs: listing them would spend --limit slots and emit skip warnings for
    files nobody asked about."""
    runs = tmp_path / 'runs'
    runs.mkdir()
    (runs / 'rt_20260101_000000.json').write_text(
        json.dumps({'pipeline': 'static', 'summary': {}, 'total_results': 1}), encoding='utf-8'
    )
    (runs / '01_all_datapoints.json').write_text('[]', encoding='utf-8')
    (runs / '02_attack_results.json').write_text('[]', encoding='utf-8')

    result = CliRunner().invoke(redteam_app, ['runs', str(runs), '--json'])
    assert result.exit_code == 0, result.output
    rows = json.loads(result.stdout)
    assert [r['file'] for r in rows] == ['rt_20260101_000000.json']
