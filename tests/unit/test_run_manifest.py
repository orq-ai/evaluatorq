"""Lifecycle-manifest read/write behaviour."""
# ruff: noqa: S101

from __future__ import annotations

import json
import threading
import time
from typing import TYPE_CHECKING, cast

import pytest
from pydantic import ValidationError

from evaluatorq.common import run_manifest
from evaluatorq.common.run_manifest import (
    ManifestWriter,
    fail_if_running,
    list_manifests,
    read_manifest,
    start_manifest,
    update_manifest,
)
from evaluatorq.contracts import ManifestStatus, ManifestSurface, RunManifest, RunSummary

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


@pytest.mark.parametrize(('completed', 'total'), [(-1, 4), (1, -1), (5, 4)])
def test_persisted_manifest_rejects_invalid_stage_progress(completed: int, total: int) -> None:
    persisted = {
        'run_id': 'r1',
        'surface': 'sim',
        'run_name': 'demo',
        'started_at': '2025-01-01T00:00:00Z',
        'updated_at': '2025-01-01T00:00:00Z',
        'stages': [
            {
                'name': 'population',
                'started_at': '2025-01-01T00:00:00Z',
                'completed': completed,
                'total': total,
            }
        ],
    }

    with pytest.raises(ValidationError):
        RunManifest.model_validate_json(json.dumps(persisted))


def test_persisted_manifest_allows_missing_stage_progress() -> None:
    persisted = {
        'run_id': 'r1',
        'surface': 'sim',
        'run_name': 'demo',
        'started_at': '2025-01-01T00:00:00Z',
        'updated_at': '2025-01-01T00:00:00Z',
        'stages': [{'name': 'population', 'started_at': '2025-01-01T00:00:00Z'}],
    }

    [stage] = RunManifest.model_validate_json(json.dumps(persisted)).stages
    assert stage.completed is None
    assert stage.total is None


def test_stage_progress_throttles_and_always_writes_the_last_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = [0.0]
    started = start_manifest(run_id='r', surface='insights', run_name='demo', runs_dir=tmp_path)
    writer = ManifestWriter(started.manifest, started.path, clock=lambda: now[0])
    writer.start_stage('label')
    flushes: list[int] = []
    original = writer.flush
    monkeypatch.setattr(writer, 'flush', lambda: (flushes.append(1), original())[1])

    for done in range(1, 101):
        writer.stage_progress('label', done, 100)

    assert len(flushes) == 2
    now[0] = 2.0
    record = writer.manifest.stages[-1]
    assert (record.completed, record.total) == (100, 100)


def test_first_progress_for_each_stage_is_flushed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    now = [0.0]
    started = start_manifest(run_id='r', surface='insights', run_name='demo', runs_dir=tmp_path)
    writer = ManifestWriter(started.manifest, started.path, clock=lambda: now[0])
    writer.start_stage('label')
    writer.stage_progress('label', 1, 3)
    writer.end_stage('label')

    flushes: list[int] = []
    original = writer.flush
    monkeypatch.setattr(writer, 'flush', lambda: (flushes.append(1), original())[1])
    writer.start_stage('summary')
    flushes.clear()  # Ignore the stage transition flush; check its first progress update.

    writer.stage_progress('summary', 1, 3)

    assert len(flushes) == 1
    [manifest] = list_manifests(tmp_path)
    record = manifest.stages[-1]
    assert (record.completed, record.total) == (1, 3)


def test_stage_progress_updates_only_matching_target(tmp_path: Path) -> None:
    writer = start_manifest(run_id='target-progress', surface='sim', run_name='demo', runs_dir=tmp_path)
    writer.start_stage('prepare', target='agent-a')
    writer.start_stage('prepare', target='agent-b')

    writer.stage_progress('prepare', 2, 5, target='agent-a')

    by_target = {stage.target: stage for stage in writer.manifest.stages}
    assert (by_target['agent-a'].completed, by_target['agent-a'].total) == (2, 5)
    assert (by_target['agent-b'].completed, by_target['agent-b'].total) == (None, None)


def test_targetless_stage_progress_does_not_update_targeted_stages(tmp_path: Path) -> None:
    writer = start_manifest(run_id='targetless-progress', surface='sim', run_name='demo', runs_dir=tmp_path)
    writer.start_stage('prepare', target='agent-a')
    writer.start_stage('prepare', target='agent-b')

    writer.stage_progress('prepare', 2, 5)

    assert all((stage.completed, stage.total) == (None, None) for stage in writer.manifest.stages)


def test_failed_progress_flush_retries_latest_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    now = [0.0]
    started = start_manifest(run_id='failed-progress', surface='insights', run_name='demo', runs_dir=tmp_path)
    writer = ManifestWriter(started.manifest, started.path, clock=lambda: now[0])
    writer.start_stage('label')
    attempts = [0]

    replace = run_manifest.os.replace

    def fail_once(*args, **kwargs) -> None:
        attempts[0] += 1
        if attempts[0] == 1:
            raise OSError('temporary disk error')
        replace(*args, **kwargs)

    monkeypatch.setattr(run_manifest.os, 'replace', fail_once)
    writer.stage_progress('label', 1, 3)
    assert list_manifests(tmp_path)[0].stages[-1].completed is None
    now[0] = 0.1
    writer.stage_progress('label', 2, 3)

    assert attempts[0] == 2
    record = list_manifests(tmp_path)[0].stages[-1]
    assert (record.completed, record.total) == (2, 3)


def test_failed_progress_flush_after_a_success_retries_inside_the_throttle_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = [0.0]
    started = start_manifest(run_id='failed-after-ok', surface='insights', run_name='demo', runs_dir=tmp_path)
    writer = ManifestWriter(started.manifest, started.path, clock=lambda: now[0])
    writer.start_stage('label')
    writer.stage_progress('label', 1, 5)  # succeeds, arms the throttle

    replace = run_manifest.os.replace
    fail = [True]

    def flaky(*args, **kwargs) -> None:
        if fail[0]:
            raise OSError('temporary disk error')
        replace(*args, **kwargs)

    monkeypatch.setattr(run_manifest.os, 'replace', flaky)
    now[0] = 2.0
    writer.stage_progress('label', 2, 5)  # past the window, flush fails
    fail[0] = False
    now[0] = 2.1
    writer.stage_progress('label', 3, 5)  # inside the old window, must still retry

    record = list_manifests(tmp_path)[0].stages[-1]
    assert (record.completed, record.total) == (3, 5)


def test_stage_progress_ignores_invalid_counts_without_persisting_them(tmp_path: Path) -> None:
    started = start_manifest(run_id='invalid-progress', surface='insights', run_name='demo', runs_dir=tmp_path)
    writer = ManifestWriter(started.manifest, started.path)
    writer.start_stage('label')
    writer.stage_progress('label', 1, 3)

    for completed, total in ((-1, 3), (4, 3), (1, -1), (1.5, 3), (True, 3)):
        writer.stage_progress('label', cast('int', completed), total)

    [manifest] = list_manifests(tmp_path)
    record = manifest.stages[-1]
    assert (record.completed, record.total) == (1, 3)


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


def test_read_manifest_finds_a_run_by_id(tmp_path: Path) -> None:
    start_manifest(run_id='abc123', surface='sim', run_name='demo', runs_dir=tmp_path)
    start_manifest(run_id='other', surface='sim', run_name='other', runs_dir=tmp_path)

    found = read_manifest(tmp_path, 'abc123')

    assert found is not None
    assert found.run_id == 'abc123'
    assert read_manifest(tmp_path, 'missing') is None


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
    summary: RunSummary = {
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
        report_path=report, summary={'pipeline': 'dynamic', 'total_results': 1, 'total_attacks': 1}
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


def test_list_run_records_demotes_an_unstamped_summary_to_a_full_read(tmp_path: Path) -> None:
    """A summary written before versioning (or by an older shape) can't be trusted
    to fill a runs row, so the record falls back to the report instead of listing
    with blank columns."""
    import json

    from evaluatorq.common.run_manifest import list_run_records

    runs = tmp_path / 'runs'
    runs.mkdir()
    report = runs / 'thin_20250101.json'
    report.write_text('{"pipeline": "dynamic", "summary": {}}', encoding='utf-8')
    w = start_manifest(run_id='thin', surface='redteam', run_name='thin', runs_dir=runs)
    w.complete(report_path=report, summary={'total_results': 1})
    aged = json.loads(w.path.read_text())
    del aged['summary_version']  # as every sidecar written before the stamp is
    w.path.write_text(json.dumps(aged))

    records = list_run_records(runs)
    assert records == [(None, report)]  # listed once, as a legacy full-read row


def test_complete_stamps_the_summary_version(tmp_path: Path) -> None:
    """The stamp is written wherever a summary is, so no writer can persist an
    unversioned one — that is what makes the reader's check a comparison and not
    a shape guess."""
    from evaluatorq.contracts import RUN_SUMMARY_VERSION

    runs = tmp_path / 'runs'
    w = start_manifest(run_id='r1', surface='sim', run_name='x', runs_dir=runs)
    w.complete(summary={'total_results': 2, 'mode': 'run'})

    m = list_manifests(runs)[0]
    assert m.summary_version == RUN_SUMMARY_VERSION


def test_concurrent_updates_to_different_fields_both_persist(tmp_path: Path) -> None:
    w = start_manifest(run_id='race', surface='sim', run_name='before', runs_dir=tmp_path / 'runs')
    first_inside = threading.Event()

    def slow_report_path(m: RunManifest) -> None:
        first_inside.set()
        time.sleep(0.3)  # hold the read-modify-write open so the other writer overlaps it
        m.report_path = 'report.json'

    def rename(m: RunManifest) -> None:
        m.run_name = 'after'

    def second_writer() -> None:
        first_inside.wait(timeout=5)
        update_manifest(w.path, rename)

    other = threading.Thread(target=second_writer)
    other.start()
    update_manifest(w.path, slow_report_path)
    other.join(timeout=10)

    [m] = list_manifests(tmp_path / 'runs')
    assert (m.report_path, m.run_name) == ('report.json', 'after')


def test_fail_if_running_does_not_clobber_a_finished_run(tmp_path: Path) -> None:
    w = start_manifest(run_id='done', surface='sim', run_name='r', runs_dir=tmp_path / 'runs')
    w.complete(report_path=None, summary=None)

    assert fail_if_running(w.path, 'late failure') is False
    assert list_manifests(tmp_path / 'runs')[0].status == ManifestStatus.COMPLETED

    running = start_manifest(run_id='live', surface='sim', run_name='r', runs_dir=tmp_path / 'runs')
    assert fail_if_running(running.path, 'boom', stage='setup') is True
    stored = next(m for m in list_manifests(tmp_path / 'runs') if m.run_id == 'live')
    assert (stored.status, stored.error) == (ManifestStatus.ERROR, 'boom')


def test_running_writer_flush_does_not_overwrite_a_failure_written_by_another_process(tmp_path: Path) -> None:
    w = start_manifest(run_id='stale', surface='sim', run_name='r', runs_dir=tmp_path / 'runs')
    fail_if_running(w.path, 'worker died')

    w.stage_progress('x', 1, 2)  # no open stage: harmless
    w.flush()

    assert w.manifest.status == ManifestStatus.ERROR
    assert list_manifests(tmp_path / 'runs')[0].error == 'worker died'


def test_terminal_transition_does_not_overwrite_another_writers_terminal_state(tmp_path: Path) -> None:
    runs = tmp_path / 'runs'
    first = start_manifest(run_id='terminal-race', surface='sim', run_name='r', runs_dir=runs)
    late = ManifestWriter(first.manifest.model_copy(deep=True), first.path)

    first.complete(report_path='report.json')
    late.fail('late failure')

    stored = list_manifests(runs)[0]
    assert stored.status == ManifestStatus.COMPLETED
    assert stored.report_path == 'report.json'
    assert late.manifest.status == ManifestStatus.COMPLETED


def test_stale_writer_flush_preserves_newer_independent_fields(tmp_path: Path) -> None:
    runs = tmp_path / 'runs'
    writer = start_manifest(run_id='stale-fields', surface='sim', run_name='before', runs_dir=runs)
    writer.start_stage('population')
    stale = ManifestWriter(writer.manifest.model_copy(deep=True), writer.path)

    def add_progress(manifest: RunManifest) -> None:
        manifest.summary = {'total_results': 3}
        manifest.stage_labels['population'] = 'Population'
        manifest.stages[0].completed = 2
        manifest.stages[0].total = 3

    update_manifest(writer.path, add_progress)
    stale.manifest.run_name = 'after'
    assert stale.flush()

    stored = list_manifests(runs)[0]
    assert stored.run_name == 'after'
    assert stored.summary == {'total_results': 3}
    assert stored.stage_labels == {'population': 'Population'}
    assert (stored.stages[0].completed, stored.stages[0].total) == (2, 3)
