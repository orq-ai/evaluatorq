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
