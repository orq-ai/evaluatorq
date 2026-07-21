"""Lifecycle-manifest read/write behaviour."""
# ruff: noqa: S101

from __future__ import annotations

from typing import TYPE_CHECKING

from evaluatorq.common.run_manifest import (
    list_manifests,
    start_manifest,
)
from evaluatorq.contracts import ManifestStatus, ManifestSurface

if TYPE_CHECKING:
    from pathlib import Path


def test_running_manifest_written_to_sidecar(tmp_path: Path) -> None:
    runs = tmp_path / 'runs'
    w = start_manifest(run_id='abc123', surface='sim', run_name='demo', runs_dir=runs)

    assert w.path == runs / '.manifests' / 'abc123.json'
    assert w.path.exists()
    # Sidecar dir keeps manifests out of the non-recursive report glob.
    assert list(runs.glob('*.json')) == []

    [m] = list_manifests(runs)
    assert m.status == 'running'
    assert m.stage is None


def test_per_stage_status_and_timing(tmp_path: Path) -> None:
    runs = tmp_path / 'runs'
    w = start_manifest(run_id='r1', surface='redteam', run_name='rt', runs_dir=runs)

    w.start_stage('Generating Datapoints')
    w.end_stage('Generating Datapoints')
    w.start_stage('Executing Attacks')
    m = list_manifests(runs)[0]
    assert m.stage == 'Executing Attacks'
    assert [s.name for s in m.stages] == ['Generating Datapoints', 'Executing Attacks']
    # First stage closed, second still running.
    assert m.stages[0].status == 'completed'
    assert m.stages[0].ended_at is not None
    first_stage_duration = m.stages[0].duration_seconds
    assert first_stage_duration is not None
    assert first_stage_duration >= 0
    assert m.stages[1].status == 'running'
    assert m.stages[1].ended_at is None

    w.complete(report_path=runs / 'rt_20250101.json')
    m = list_manifests(runs)[0]
    assert m.status == 'completed'
    assert m.report_path is not None
    assert m.report_path.endswith('rt_20250101.json')
    # complete() closes any dangling stage and stamps the run end time.
    assert all(s.status == 'completed' and s.ended_at is not None for s in m.stages)
    assert m.ended_at is not None
    run_duration = m.duration_seconds
    assert run_duration is not None
    assert run_duration >= 0


def test_stage_is_noop_after_terminal(tmp_path: Path) -> None:
    runs = tmp_path / 'runs'
    w = start_manifest(run_id='r1', surface='sim', run_name='x', runs_dir=runs)
    w.complete()
    w.start_stage('late')  # must not revert a completed run to running
    m = list_manifests(runs)[0]
    assert m.status == 'completed'
    assert m.stages == []


def test_fail_marks_open_stage_errored(tmp_path: Path) -> None:
    runs = tmp_path / 'runs'
    w = start_manifest(run_id='r2', surface='redteam', run_name='rt', runs_dir=runs)
    w.start_stage('attack')
    w.fail('boom')
    m = list_manifests(runs)[0]
    assert m.status == 'error'
    assert m.error == 'boom'
    assert m.ended_at is not None
    assert m.stages[-1].name == 'attack'
    assert m.stages[-1].status == 'error'
    assert m.stages[-1].ended_at is not None


def test_list_empty_when_no_dir(tmp_path: Path) -> None:
    assert list_manifests(tmp_path / 'nope') == []


def test_end_stage_with_error_marks_stage_errored(tmp_path: Path) -> None:
    runs = tmp_path / 'runs'
    w = start_manifest(run_id='r1', surface='sim', run_name='x', runs_dir=runs)
    w.start_stage('simulate')
    w.end_stage('simulate', error=RuntimeError('kaboom'))
    m = list_manifests(runs)[0]
    # Stage recorded error, but the run itself stays running (end_stage never
    # flips overall status — the runner owns terminal transitions).
    assert m.stages[-1].status == 'error'
    assert m.stages[-1].ended_at is not None
    assert m.status == 'running'


def test_cancel_leaves_completed_stage_completed(tmp_path: Path) -> None:
    runs = tmp_path / 'runs'
    w = start_manifest(run_id='r1', surface='sim', run_name='x', runs_dir=runs)
    w.start_stage('generate')
    w.end_stage('generate')
    w.cancel()
    m = list_manifests(runs)[0]
    assert m.status == 'cancelled'
    assert m.ended_at is not None
    # Cancellation happens outside a stage — a finished stage stays truthful.
    assert m.stages[-1].name == 'generate'
    assert m.stages[-1].status == 'completed'


def test_cancel_is_idempotent_and_terminal(tmp_path: Path) -> None:
    runs = tmp_path / 'runs'
    w = start_manifest(run_id='r1', surface='sim', run_name='x', runs_dir=runs)
    w.cancel()
    w.complete()  # must not override a terminal cancelled run
    w.fail('nope')
    m = list_manifests(runs)[0]
    assert m.status == 'cancelled'


def test_fail_with_closed_stage_and_no_open_stage(tmp_path: Path) -> None:
    """R1 revised: a post-stage failure never relabels a succeeded stage."""
    runs = tmp_path / 'runs'
    w = start_manifest(run_id='r1', surface='redteam', run_name='rt', runs_dir=runs)
    w.start_stage('attack')
    w.end_stage('attack')  # stage succeeded and closed
    w.fail('glue exploded')  # failure outside any open stage
    m = list_manifests(runs)[0]
    assert m.status == 'error'
    assert m.error == 'glue exploded'
    # The closed stage keeps its truthful 'completed' status.
    assert m.stages[-1].name == 'attack'
    assert m.stages[-1].status == 'completed'


def test_fail_closes_all_open_stages(tmp_path: Path) -> None:
    """R2: multiple open stages at failure are all closed as error."""
    runs = tmp_path / 'runs'
    w = start_manifest(run_id='r1', surface='redteam', run_name='rt', runs_dir=runs)
    # Two concurrent open stages via distinct targets (start_stage only closes a
    # dangling stage sharing the same target).
    w.start_stage('prepare', target='agent-a')
    w.start_stage('prepare', target='agent-b')
    w.fail('boom')
    m = list_manifests(runs)[0]
    assert all(s.status == 'error' and s.ended_at is not None for s in m.stages)


def test_per_target_stages_do_not_close_each_other(tmp_path: Path) -> None:
    """Dec2: concurrent stages keyed by target stay independent."""
    runs = tmp_path / 'runs'
    w = start_manifest(run_id='r1', surface='redteam', run_name='rt', runs_dir=runs)
    w.start_stage('prepare', target='agent-a')
    w.start_stage('prepare', target='agent-b')
    # Ending agent-a must not touch agent-b's open stage.
    w.end_stage('prepare', target='agent-a')
    m = list_manifests(runs)[0]
    by_target = {s.target: s for s in m.stages}
    assert by_target['agent-a'].status == 'completed'
    assert by_target['agent-a'].ended_at is not None
    assert by_target['agent-b'].status == 'running'
    assert by_target['agent-b'].ended_at is None


def test_targetless_end_does_not_close_targeted_stage(tmp_path: Path) -> None:
    """A missing target is its own key, not a wildcard for concurrent stages."""
    runs = tmp_path / 'runs'
    w = start_manifest(run_id='r1', surface='redteam', run_name='rt', runs_dir=runs)
    w.start_stage('cleanup', target='agent-a')
    w.start_stage('cleanup', target='agent-b')

    w.end_stage('cleanup')

    assert all(stage.status == 'running' for stage in w.manifest.stages)


def test_status_strenum_round_trips(tmp_path: Path) -> None:
    runs = tmp_path / 'runs'
    w = start_manifest(run_id='r1', surface=ManifestSurface.REDTEAM, run_name='rt', runs_dir=runs)
    w.start_stage('attack')
    # StrEnum members survive the model_dump_json / model_validate_json round-trip
    # performed by list_manifests, and compare equal to both enum and str.
    dumped = w.manifest.model_dump_json()
    assert '"running"' in dumped
    m = list_manifests(runs)[0]
    assert m.surface == ManifestSurface.REDTEAM
    assert m.surface == 'redteam'
    assert m.status == ManifestStatus.RUNNING
    assert m.status == 'running'
    assert isinstance(m.status, ManifestStatus)


def test_summary_round_trips_through_manifest(tmp_path: Path) -> None:
    runs = tmp_path / 'runs'
    w = start_manifest(run_id='s1', surface='redteam', run_name='rt', runs_dir=runs)
    summary = {
        'pipeline': 'dynamic',
        'total_results': 12,
        'total_attacks': 12,
        'vulnerability_rate': 0.25,
        'resistance_rate': 0.75,
        'tested_agents': ['agent:x'],
    }
    w.complete(report_path=runs / 'rt_20250101.json', summary=summary)

    m = list_manifests(runs)[0]
    assert m.status == 'completed'
    assert m.summary == summary


def test_list_run_records_manifest_first_dedups_and_falls_back(tmp_path: Path) -> None:
    from evaluatorq.common.run_manifest import list_run_records

    runs = tmp_path / 'runs'
    runs.mkdir()
    # Completed run with a manifest pointing at a report on disk.
    report = runs / 'done_20250101.json'
    report.write_text('{"pipeline": "dynamic", "summary": {}}', encoding='utf-8')
    start_manifest(run_id='done', surface='redteam', run_name='done', runs_dir=runs).complete(
        report_path=report, summary={'total_results': 1}
    )
    # In-flight run — manifest only, no report.
    start_manifest(run_id='live', surface='redteam', run_name='live', runs_dir=runs)
    # Legacy report with no manifest at all.
    legacy = runs / 'legacy_20240101.json'
    legacy.write_text('{"pipeline": "static", "summary": {}}', encoding='utf-8')

    records = list_run_records(runs)
    # Three distinct runs, none listed twice.
    assert len(records) == 3
    # The completed run's report is covered by its manifest → not re-listed as legacy.
    manifests = [m for m, _ in records if m is not None]
    legacy_paths = [p for m, p in records if m is None]
    assert {m.run_name for m in manifests} == {'done', 'live'}
    assert legacy_paths == [legacy]
    # The in-flight manifest carries no report path.
    live = next(m for m, _ in records if m is not None and m.run_name == 'live')
    assert live.report_path is None
    assert live.status == 'running'
