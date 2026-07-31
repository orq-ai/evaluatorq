"""Replaying a prior red-team run (`previous_run=` / `--from-run`)."""
# ruff: noqa: S101

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from evaluatorq.common.replay import ReplayError
from evaluatorq.redteam.contracts import Pipeline
from evaluatorq.redteam.replay import load_redteam_replay

if TYPE_CHECKING:
    from pathlib import Path

DYNAMIC_INPUTS = {
    'id': 'dynamic_goal_hijacking_role_confusion',
    'vulnerability': 'goal_hijacking',
    'category': 'ASI01',
    'strategy': {'name': 'role_confusion', 'is_generated': False},
    'objective': 'make the agent adopt a new goal',
}
STATIC_INPUTS = {
    'id': 'OWASP-ASI03-0001',
    'vulnerability': 'privilege_compromise',
    'category': 'ASI03',
    'messages': [{'role': 'user', 'content': 'escalate me'}],
}


def _save_run(runs_dir: Path, name: str, payload: dict[str, Any]) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f'{name}.json'
    path.write_text(json.dumps(payload), encoding='utf-8')
    return path


def test_replay_rebuilds_datapoints_and_pipeline(tmp_path: Path) -> None:
    runs = tmp_path / 'runs'
    _save_run(
        runs,
        'rt_20260101_000000',
        {'run_name': 'rt', 'pipeline': 'dynamic', 'datapoints': [DYNAMIC_INPUTS]},
    )

    replay = load_redteam_replay('latest', runs)

    assert replay.pipeline == Pipeline.DYNAMIC
    assert replay.run_name == 'rt'
    assert [dp.inputs for dp in replay.datapoints] == [DYNAMIC_INPUTS]
    assert replay.categories == ['ASI01']


def test_replay_preserves_hybrid_split(tmp_path: Path) -> None:
    runs = tmp_path / 'runs'
    _save_run(
        runs,
        'rt_20260101_000000',
        {
            'pipeline': 'hybrid',
            'datapoints': [
                {**DYNAMIC_INPUTS, 'hybrid_source': 'dynamic'},
                {**STATIC_INPUTS, 'hybrid_source': 'static'},
            ],
        },
    )

    replay = load_redteam_replay('latest', runs)

    assert replay.pipeline == Pipeline.HYBRID
    assert [dp.inputs['hybrid_source'] for dp in replay.datapoints] == ['dynamic', 'static']
    assert replay.categories == ['ASI01', 'ASI03']


def test_pipeline_is_inferred_when_the_stored_value_is_unusable(tmp_path: Path) -> None:
    runs = tmp_path / 'runs'
    _save_run(runs, 'rt_20260101_000000', {'datapoints': [{**STATIC_INPUTS, 'hybrid_source': 'static'}]})

    assert load_redteam_replay('latest', runs).pipeline == Pipeline.STATIC


def test_run_without_stored_datapoints_cannot_be_replayed(tmp_path: Path) -> None:
    runs = tmp_path / 'runs'
    _save_run(runs, 'legacy_20260101_000000', {'pipeline': 'dynamic', 'results': []})

    with pytest.raises(ReplayError, match='stores no datapoints'):
        load_redteam_replay('latest', runs)


def test_malformed_datapoint_entry_is_rejected(tmp_path: Path) -> None:
    runs = tmp_path / 'runs'
    _save_run(runs, 'rt_20260101_000000', {'pipeline': 'dynamic', 'datapoints': ['not-an-object']})

    with pytest.raises(ReplayError, match='datapoint 0 is not an object'):
        load_redteam_replay('latest', runs)


def test_missing_run_is_reported_clearly(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match='Could not resolve previous red team run'):
        load_redteam_replay('does-not-exist', tmp_path / 'runs')


def test_auto_save_persists_the_executed_datapoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.redteam import runner as runner_mod

    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path / '.evaluatorq'))

    report = _minimal_report()
    path = runner_mod._auto_save_run(report, name='rt', datapoints=[DYNAMIC_INPUTS])

    assert path is not None
    stored = json.loads(path.read_text(encoding='utf-8'))
    assert stored['datapoints'] == [DYNAMIC_INPUTS]

    replay = load_redteam_replay('latest', runner_mod.get_runs_dir())
    assert [dp.inputs for dp in replay.datapoints] == [DYNAMIC_INPUTS]


def _minimal_report() -> Any:
    from datetime import datetime, timezone

    from evaluatorq.redteam.contracts import RedTeamReport, ReportSummary

    return RedTeamReport(
        created_at=datetime.now(tz=timezone.utc),
        pipeline=Pipeline.DYNAMIC,
        categories_tested=['ASI01'],
        total_results=0,
        results=[],
        summary=ReportSummary(),
    )


@pytest.mark.asyncio
async def test_previous_run_rejects_data_selection_arguments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.redteam import red_team

    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path / '.evaluatorq'))
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')

    with pytest.raises(ValueError, match='cannot be combined with data-selection arguments: dataset, categories'):
        await red_team(target='agent:demo', previous_run='latest', dataset='local.json', categories=['ASI01'])


@pytest.mark.asyncio
async def test_previous_run_reports_an_unresolvable_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.redteam import red_team

    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path / '.evaluatorq'))
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')

    with pytest.raises(ReplayError, match='Could not resolve previous red team run'):
        await red_team(target='agent:demo', previous_run='nope')
