"""Lifecycle-manifest read/write behaviour."""
# ruff: noqa: S101

from __future__ import annotations

from typing import TYPE_CHECKING

from evaluatorq.common.run_manifest import (
    active_manifests,
    format_active_lines,
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


def test_active_excludes_completed(tmp_path: Path) -> None:
    runs = tmp_path / 'runs'
    start_manifest(run_id='run', surface='sim', run_name='running-one', runs_dir=runs)
    start_manifest(run_id='err', surface='sim', run_name='errored-one', runs_dir=runs).fail('x')
    start_manifest(run_id='done', surface='sim', run_name='done-one', runs_dir=runs).complete()

    names = {m.run_name for m in active_manifests(runs)}
    assert names == {'running-one', 'errored-one'}

    lines = format_active_lines(runs)
    assert len(lines) == 2
    assert any('running-one' in ln and 'running' in ln for ln in lines)
    assert any('errored-one' in ln and 'error' in ln for ln in lines)


def test_list_empty_when_no_dir(tmp_path: Path) -> None:
    assert list_manifests(tmp_path / 'nope') == []
    assert format_active_lines(tmp_path / 'nope') == []


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


def test_cancelled_run_shown_in_active_lines(tmp_path: Path) -> None:
    runs = tmp_path / 'runs'
    start_manifest(run_id='c', surface='sim', run_name='cancelled-one', runs_dir=runs).cancel()
    names = {m.run_name for m in active_manifests(runs)}
    assert 'cancelled-one' in names
    lines = format_active_lines(runs)
    assert any('cancelled-one' in ln and 'cancelled' in ln for ln in lines)
