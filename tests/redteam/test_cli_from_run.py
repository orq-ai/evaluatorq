"""CLI wiring for `eq redteam run --from-run` (RES-1126)."""
# ruff: noqa: S101

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from evaluatorq.redteam.cli import app

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


def test_from_run_is_forwarded_as_previous_run() -> None:
    report = MagicMock()
    report.model_dump.return_value = {}

    with patch('evaluatorq.redteam.red_team', new=AsyncMock(return_value=report)) as mock_rt:
        result = runner.invoke(
            app,
            ['run', '--target', 'agent:test-agent', '--from-run', 'latest', '--yes'],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    assert mock_rt.call_args.kwargs['previous_run'] == 'latest'


def test_from_run_with_a_data_selection_flag_exits_cleanly(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ['run', '--target', 'agent:test-agent', '--from-run', 'latest', '--category', 'ASI01', '--yes'],
    )

    assert result.exit_code == 1
    assert 'cannot be combined with data-selection arguments' in result.output + (result.stderr or '')


def test_unresolvable_from_run_exits_cleanly(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path / '.evaluatorq'))
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')

    result = runner.invoke(app, ['run', '--target', 'agent:test-agent', '--from-run', 'nope', '--yes'])

    assert result.exit_code == 1
    assert 'Could not resolve previous red team run' in result.output + (result.stderr or '')


def test_from_run_resolves_a_saved_run_by_name(tmp_path: Path, monkeypatch) -> None:
    """A stored run is found by its run name, not just by file name."""
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path / '.evaluatorq'))

    from evaluatorq.redteam.replay import load_redteam_replay
    from evaluatorq.redteam.runner import get_runs_dir

    runs_dir = get_runs_dir()
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / 'nightly_20260101_000000.json').write_text(
        json.dumps({'run_name': 'nightly', 'pipeline': 'dynamic', 'datapoints': [{'id': 'a', 'category': 'ASI01', 'strategy': {'name': 's'}}]}),
        encoding='utf-8',
    )

    replay = load_redteam_replay('nightly', runs_dir)
    assert replay.run_name == 'nightly'
    assert len(replay.datapoints) == 1
