"""Dashboard run creation and manifest-backed Insights progress."""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient

from evaluatorq.common.orq_client import OrqProfile
from evaluatorq.common.run_manifest import list_manifests, start_manifest
from evaluatorq.dashboard import insights_routes
from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.insights_launch import (
    MAX_FINDER_EXPORT_BYTES,
    InsightsLaunchPayload,
    InsightsLaunchSpec,
    ensure_private_finder_reference_dir,
    finder_export_reference_path,
    launch_insights,
    read_private_finder_reference,
)
from evaluatorq.insights.models import InsightsPopulation
from evaluatorq.insights.presets import CUSTOMER_SATISFACTION, SENTIMENT
from evaluatorq.insights.progress import stage_plan
from evaluatorq.insights.store import save_run
from evaluatorq.trace_finder.models import FacetCatalogue, FacetSelection
from evaluatorq.trace_finder.settings import DashboardSettings
from tests.dashboard.test_insights_page import minimal_run
from tests.insights.test_population import _run_export


def test_stage_plan_follows_source_labels_and_dimensions() -> None:
    query_plan = dict(stage_plan(InsightsPopulation(query='refunds'), [SENTIMENT], ['intent', 'failure']))
    finder_plan = dict(stage_plan(InsightsPopulation.from_finder_export(Path('finder.json')), [], ['sentiment']))
    priority_plan = dict(stage_plan(InsightsPopulation(), [CUSTOMER_SATISFACTION], ['intent']))

    assert query_plan['population'] == 'Find matching traces'
    assert query_plan['label'] == 'Match and classify traces'
    assert list(query_plan).count('dimension:intent') == 1
    assert 'dimension:failure' in query_plan
    assert 'dimension:sentiment' not in query_plan
    assert finder_plan['population'] == 'Load Finder matches'
    assert finder_plan['label'] == 'Classify traces'
    assert priority_plan['priority'] == 'Build priority matrix'


def test_launch_persists_plan_before_spawning_worker(tmp_path: Path) -> None:
    spec = InsightsLaunchSpec(source='query', query='refunds', labels=['sentiment'], dimensions=['intent', 'failure'])
    with patch('evaluatorq.dashboard.insights_launch.subprocess.Popen') as spawn:
        run_id = launch_insights(spec, tmp_path)

    manifest = list_manifests(tmp_path)[0]
    assert manifest.run_id == run_id
    assert manifest.status == 'running'
    assert manifest.planned_stages == ['population', 'label', 'summary', 'dimension:intent', 'dimension:failure', 'priority', 'write']
    assert manifest.stage_labels['label'] == 'Match and classify traces'
    command = spawn.call_args.args[0]
    assert command[1] == '-c'
    assert 'runpy.run_module' in command[2]
    assert spawn.call_args.kwargs['env']['EVALUATORQ_INSIGHTS_MANIFEST'] == str(
        tmp_path / '.manifests' / f'{run_id}.json'
    )
    payload = InsightsLaunchPayload.model_validate_json(spawn.call_args.kwargs['env']['EVALUATORQ_INSIGHTS_LAUNCH_REQUEST'])
    assert payload.run_id == run_id
    assert payload.spec.query == 'refunds'
    assert spawn.call_args.kwargs['start_new_session'] is True


def test_spawn_failure_is_visible_in_manifest(tmp_path: Path) -> None:
    with patch('evaluatorq.dashboard.insights_launch.subprocess.Popen', side_effect=OSError('worker unavailable')):
        run_id = launch_insights(InsightsLaunchSpec(), tmp_path)

    manifest = list_manifests(tmp_path)[0]
    assert manifest.run_id == run_id
    assert manifest.status == 'error'
    assert manifest.error is not None and 'worker unavailable' in manifest.error


def test_dashboard_reconciles_worker_killed_before_start_and_releases_finder_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import json
    import tempfile
    import time

    from evaluatorq.dashboard.insights_launch import _write_worker_state, worker_state_path

    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
    runs_dir = tmp_path / 'insights-runs'
    writer = start_manifest(run_id='worker-killed', surface='insights', run_name='demo', runs_dir=runs_dir)
    reference = finder_export_reference_path(runs_dir, 'worker-killed')
    ensure_private_finder_reference_dir(reference.parent)
    reference.write_text(json.dumps({'finder_export': '/private/trace-finder-export.json'}), encoding='utf-8')
    reference.chmod(0o600)
    snapshot_dir = Path(tempfile.mkdtemp(prefix='evaluatorq-finder-snapshot-'))
    snapshot = snapshot_dir / 'finder-export.json'
    snapshot.write_text('{}', encoding='utf-8')
    snapshot.chmod(0o600)
    state_path = worker_state_path(runs_dir, 'worker-killed')
    _write_worker_state(state_path, {'pid': 12345, 'heartbeat_at': time.time() - 120, 'snapshot_path': str(snapshot)})

    entries, _, manifests = insights_routes._entries(runs_dir)

    assert ('worker-killed', 'demo', 'error') in entries
    assert manifests['worker-killed'].status.value == 'error'
    assert manifests['worker-killed'].error is not None
    assert 'stopped before completing' in manifests['worker-killed'].error
    assert not reference.exists()
    assert not state_path.exists()
    assert not snapshot_dir.exists()
    assert writer.manifest.status.value == 'running'


def test_non_oserror_setup_failure_marks_manifest_and_cleans_finder_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import json
    import tempfile

    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
    exports = tmp_path / 'finder-exports'
    exports.mkdir()
    export_path = exports / 'trace-finder-source.json'
    export_path.write_text(_run_export(['trace-1']).model_dump_json(), encoding='utf-8')
    spec = InsightsLaunchSpec(source='finder', finder_export=export_path.name)

    real_dumps = json.dumps

    def fail_worker_request(value: object, *args: object, **kwargs: object) -> str:
        if isinstance(value, dict) and 'run_id' in value:
            raise TypeError('worker request is not serializable')
        return real_dumps(value, *args, **kwargs)

    with (
        patch('evaluatorq.dashboard.insights_launch.json.dumps', side_effect=fail_worker_request),
        patch('evaluatorq.dashboard.insights_launch.subprocess.Popen') as spawn,
    ):
        run_id = launch_insights(spec, tmp_path / 'insights-runs')

    manifest = list_manifests(tmp_path / 'insights-runs')[0]
    assert manifest.run_id == run_id
    assert manifest.status == 'error'
    assert manifest.error is not None and 'worker request is not serializable' in manifest.error
    assert not finder_export_reference_path(tmp_path / 'insights-runs', run_id).exists()
    assert list(tmp_path.glob('evaluatorq-finder-snapshot-*')) == []
    assert list(tmp_path.glob(f'.{run_id}.*.tmp')) == []
    spawn.assert_not_called()


def test_worker_import_failure_marks_manifest_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import runpy
    import tempfile

    from evaluatorq.dashboard.insights_launch import _WORKER_BOOTSTRAP, _MANIFEST_ENV, _REQUEST_ENV, _SNAPSHOT_ENV

    writer = start_manifest(run_id='import-failure', surface='insights', run_name='demo', runs_dir=tmp_path)
    reference = finder_export_reference_path(tmp_path, 'import-failure')
    ensure_private_finder_reference_dir(reference.parent)
    reference.write_text('{}', encoding='utf-8')
    reference.chmod(0o600)
    monkeypatch.setenv(_MANIFEST_ENV, str(writer.path))
    snapshot_directory = Path(tempfile.mkdtemp(prefix='evaluatorq-finder-snapshot-'))
    snapshot_path = snapshot_directory / 'finder-export.json'
    snapshot_path.write_text('{}', encoding='utf-8')
    monkeypatch.setenv(_REQUEST_ENV, '{truncated request')
    monkeypatch.setenv(_SNAPSHOT_ENV, str(snapshot_path))

    def fail_import(*args, **kwargs):
        raise ImportError('missing optional dependency')

    with patch.object(runpy, 'run_module', side_effect=fail_import), pytest.raises(ImportError):
        exec(_WORKER_BOOTSTRAP, {})

    manifest = list_manifests(tmp_path)[0]
    assert manifest.status == 'error'
    assert manifest.stage == 'setup'
    assert manifest.error is not None and 'missing optional dependency' in manifest.error
    assert not snapshot_directory.exists()
    assert not reference.exists()


def test_successful_worker_exit_does_not_recover_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import runpy

    from evaluatorq.dashboard.insights_launch import _WORKER_BOOTSTRAP, _MANIFEST_ENV, _REQUEST_ENV

    writer = start_manifest(run_id='worker-success', surface='insights', run_name='demo', runs_dir=tmp_path)
    monkeypatch.setenv(_MANIFEST_ENV, str(writer.path))
    monkeypatch.setenv(_REQUEST_ENV, '{invalid request')

    def exit_cleanly(*args, **kwargs):
        raise SystemExit(0)

    monkeypatch.setattr(runpy, 'run_module', exit_cleanly)

    with pytest.raises(SystemExit) as exit_info:
        exec(_WORKER_BOOTSTRAP, {})

    assert exit_info.value.code == 0
    assert list_manifests(tmp_path)[0].status == 'running'


def test_worker_payload_decode_failure_marks_manifest_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.dashboard import insights_worker
    from evaluatorq.dashboard.insights_launch import _MANIFEST_ENV, _REQUEST_ENV

    writer = start_manifest(run_id='decode-failure', surface='insights', run_name='demo', runs_dir=tmp_path)
    monkeypatch.delenv(_REQUEST_ENV, raising=False)
    monkeypatch.setenv(_MANIFEST_ENV, str(writer.path))

    assert insights_worker.main() == 1
    manifest = list_manifests(tmp_path)[0]
    assert manifest.status == 'error'
    assert manifest.stage == 'setup'
    assert manifest.error == 'Missing Insights launch request'


def test_worker_cleans_private_snapshot_when_request_json_is_truncated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tempfile

    from evaluatorq.dashboard import insights_worker
    from evaluatorq.dashboard.insights_launch import _MANIFEST_ENV, _REQUEST_ENV, _SNAPSHOT_ENV

    writer = start_manifest(run_id='truncated-request', surface='insights', run_name='demo', runs_dir=tmp_path)
    snapshot_directory = Path(tempfile.mkdtemp(prefix='evaluatorq-finder-snapshot-'))
    snapshot = snapshot_directory / 'finder-export.json'
    snapshot.write_text('{}', encoding='utf-8')
    snapshot.chmod(0o600)
    monkeypatch.setenv(_REQUEST_ENV, '{truncated request')
    monkeypatch.setenv(_SNAPSHOT_ENV, str(snapshot))
    monkeypatch.setenv(_MANIFEST_ENV, str(writer.path))

    assert insights_worker.main() == 1
    assert not snapshot_directory.exists()
    assert list_manifests(tmp_path)[0].status == 'error'


def test_worker_cleans_private_snapshot_when_payload_validation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    import tempfile

    from evaluatorq.dashboard import insights_worker
    from evaluatorq.dashboard.insights_launch import _MANIFEST_ENV, _REQUEST_ENV

    writer = start_manifest(run_id='invalid-payload', surface='insights', run_name='demo', runs_dir=tmp_path)
    snapshot_directory = Path(tempfile.mkdtemp(prefix='evaluatorq-finder-snapshot-'))
    snapshot = snapshot_directory / 'finder-export.json'
    snapshot.write_text('{}', encoding='utf-8')
    snapshot.chmod(0o600)
    monkeypatch.setenv(
        _REQUEST_ENV,
        json.dumps({'spec': {'source': 'finder'}, 'finder_export_snapshot': str(snapshot)}),
    )
    monkeypatch.setenv(_MANIFEST_ENV, str(writer.path))

    assert insights_worker.main() == 1

    assert not snapshot_directory.exists()
    manifest = list_manifests(tmp_path)[0]
    assert manifest.status == 'error'
    assert manifest.stage == 'setup'


def test_worker_does_not_clean_arbitrary_snapshot_from_invalid_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from evaluatorq.dashboard import insights_worker
    from evaluatorq.dashboard.insights_launch import _MANIFEST_ENV, _REQUEST_ENV

    writer = start_manifest(run_id='unsafe-payload', surface='insights', run_name='demo', runs_dir=tmp_path)
    arbitrary_directory = tmp_path / 'evaluatorq-finder-snapshot-user-data'
    arbitrary_directory.mkdir(mode=0o700)
    arbitrary_file = arbitrary_directory / 'finder-export.json'
    arbitrary_file.write_text('user data', encoding='utf-8')
    monkeypatch.setenv(
        _REQUEST_ENV,
        json.dumps({'spec': {'source': 'finder'}, 'finder_export_snapshot': str(arbitrary_file)}),
    )
    monkeypatch.setenv(_MANIFEST_ENV, str(writer.path))

    assert insights_worker.main() == 1

    assert arbitrary_file.read_text(encoding='utf-8') == 'user data'


def test_startup_failure_shows_in_run_page(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    directory = tmp_path / 'insights-runs'
    with patch('evaluatorq.dashboard.insights_launch.subprocess.Popen', side_effect=OSError('worker unavailable')):
        run_id = launch_insights(InsightsLaunchSpec(), directory)

    response = TestClient(build_app()).get(f'/insights/{run_id}')
    assert response.status_code == 200
    assert 'worker unavailable' in response.text
    assert '<span class="insights-stage-label">Start</span><span class="sr-only">error</span>' in response.text
    assert 'Filter recent traces' in response.text


def test_wizard_validates_source_before_launch() -> None:
    with pytest.raises(ValidationError, match='Enter a question'):
        InsightsLaunchSpec(source='query', query='  ')
    with pytest.raises(ValidationError, match='Select at least one'):
        InsightsLaunchSpec(labels=[], dimensions=[])
    with pytest.raises(ValidationError, match='Finder export already fixes'):
        InsightsLaunchSpec(source='finder', facets=FacetSelection(status=frozenset({'error'})))


def test_dashboard_finder_export_must_be_in_approved_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    allowed_dir = tmp_path / 'finder-exports'
    allowed_dir.mkdir()
    allowed = allowed_dir / 'finder.json'
    outside = tmp_path / 'outside.json'
    payload = _run_export(['trace-1']).model_dump_json()
    allowed.write_text(payload)
    outside.write_text(payload)
    (allowed_dir / 'link.json').symlink_to(outside)

    spec = InsightsLaunchSpec(source='finder', finder_export='finder.json')
    assert spec.population().finder_export == allowed.resolve()
    assert InsightsLaunchSpec(source='finder', finder_export=str(allowed)).population().finder_export == allowed.resolve()
    for path in (str(outside), '../outside.json', 'link.json'):
        with pytest.raises(ValidationError, match='Finder exports must be in'):
            InsightsLaunchSpec(source='finder', finder_export=path)


def test_dashboard_finder_export_is_size_limited(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    exports = tmp_path / 'finder-exports'
    exports.mkdir()
    oversized = exports / 'large.json'
    oversized.write_bytes(b' ' * (MAX_FINDER_EXPORT_BYTES + 1))

    with pytest.raises(ValidationError, match='exceeds the 10 MiB size limit'):
        InsightsLaunchSpec(source='finder', finder_export='large.json')


def test_finder_insights_launch_records_source_until_run_finishes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import json

    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    export_dir = tmp_path / 'finder-exports'
    export_dir.mkdir()
    export_path = export_dir / 'trace-finder-source.json'
    export_path.write_text(_run_export(['trace-1']).model_dump_json(), encoding='utf-8')
    spec = InsightsLaunchSpec(source='finder', finder_export=export_path.name)
    with patch('evaluatorq.dashboard.insights_launch.subprocess.Popen'):
        run_id = launch_insights(spec, tmp_path / 'insights-runs')

    reference = finder_export_reference_path(tmp_path / 'insights-runs', run_id)
    assert json.loads(reference.read_text(encoding='utf-8')) == {'finder_export': str(export_path)}
    assert reference.stat().st_mode & 0o777 == 0o600
    assert reference.parent.stat().st_mode & 0o777 == 0o700
    assert read_private_finder_reference(reference) == {'finder_export': str(export_path)}
    assert len(list_manifests(tmp_path / 'insights-runs')) == 1


def test_finder_reference_rejects_shared_directory_and_file_modes(tmp_path: Path) -> None:
    reference = finder_export_reference_path(tmp_path / 'runs', 'private-run')
    ensure_private_finder_reference_dir(reference.parent)
    reference.write_text('{"finder_export":"/tmp/export.json"}', encoding='utf-8')
    reference.chmod(0o644)

    with pytest.raises(OSError, match='not private'):
        read_private_finder_reference(reference)

    reference.chmod(0o600)
    reference.parent.chmod(0o755)
    with pytest.raises(OSError, match='not private'):
        read_private_finder_reference(reference)


def test_stale_reconciliation_leaves_a_live_worker_and_its_finder_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import json
    import tempfile
    import time

    from evaluatorq.dashboard import insights_launch
    from evaluatorq.dashboard.insights_launch import _write_worker_state, worker_state_path

    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
    runs_dir = tmp_path / 'runs'
    writer = start_manifest(run_id='paused-worker', surface='insights', run_name='demo', runs_dir=runs_dir)
    reference = finder_export_reference_path(runs_dir, 'paused-worker')
    ensure_private_finder_reference_dir(reference.parent)
    reference.write_text(json.dumps({'finder_export': '/private/export.json'}), encoding='utf-8')
    reference.chmod(0o600)
    snapshot_dir = Path(tempfile.mkdtemp(prefix='evaluatorq-finder-snapshot-'))
    snapshot = snapshot_dir / 'finder-export.json'
    snapshot.write_text('{}', encoding='utf-8')
    snapshot.chmod(0o600)
    state_path = worker_state_path(runs_dir, 'paused-worker')
    _write_worker_state(
        state_path,
        {
            'pid': 12345,
            'heartbeat_at': time.time() - 120,
            'snapshot_path': str(snapshot),
        },
    )
    monkeypatch.setattr(insights_launch, '_worker_process_is_alive', lambda _pid, _identity=None: True)

    assert not insights_launch.reconcile_stale_worker(runs_dir, 'paused-worker')
    assert list_manifests(runs_dir)[0].status.value == 'running'
    assert reference.exists()
    assert state_path.exists()
    assert snapshot_dir.exists()
    assert writer.manifest.status.value == 'running'


def test_stale_reconciliation_rechecks_worker_liveness_before_failing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import time

    from evaluatorq.dashboard import insights_launch
    from evaluatorq.dashboard.insights_launch import _write_worker_state, worker_state_path

    runs_dir = tmp_path / 'runs'
    writer = start_manifest(run_id='resumed-worker', surface='insights', run_name='demo', runs_dir=runs_dir)
    _write_worker_state(
        worker_state_path(runs_dir, 'resumed-worker'),
        {
            'pid': 12345,
            'heartbeat_at': time.time() - 120,
            'snapshot_path': None,
        },
    )
    checks = iter((False, True))
    monkeypatch.setattr(insights_launch, '_worker_process_is_alive', lambda _pid, _identity=None: next(checks))

    assert not insights_launch.reconcile_stale_worker(runs_dir, 'resumed-worker')
    assert list_manifests(runs_dir)[0].status.value == 'running'
    assert writer.manifest.status.value == 'running'


def test_stale_reconciliation_detects_reused_pid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    from evaluatorq.dashboard import insights_launch
    from evaluatorq.dashboard.insights_launch import _write_worker_state, worker_state_path

    runs_dir = tmp_path / 'runs'
    start_manifest(run_id='reused-worker-pid', surface='insights', run_name='demo', runs_dir=runs_dir)
    _write_worker_state(
        worker_state_path(runs_dir, 'reused-worker-pid'),
        {
            'pid': 12345,
            'process_identity': 'linux:100',
            'heartbeat_at': time.time() - 120,
            'snapshot_path': None,
        },
    )
    monkeypatch.setattr(insights_launch.os, 'kill', lambda _pid, _signal: None)
    monkeypatch.setattr(insights_launch, '_read_worker_process_identity', lambda _pid: ('linux:200', False))

    assert insights_launch.reconcile_stale_worker(runs_dir, 'reused-worker-pid')
    assert list_manifests(runs_dir)[0].status.value == 'error'


def test_stale_reconciliation_recovers_zombie_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    from evaluatorq.dashboard import insights_launch
    from evaluatorq.dashboard.insights_launch import _write_worker_state, worker_state_path

    runs_dir = tmp_path / 'runs'
    start_manifest(run_id='zombie-worker', surface='insights', run_name='demo', runs_dir=runs_dir)
    _write_worker_state(
        worker_state_path(runs_dir, 'zombie-worker'),
        {
            'pid': 12345,
            'process_identity': 'linux:100',
            'heartbeat_at': time.time() - 120,
            'snapshot_path': None,
        },
    )
    monkeypatch.setattr(insights_launch.os, 'kill', lambda _pid, _signal: None)
    monkeypatch.setattr(insights_launch, '_read_worker_process_identity', lambda _pid: ('linux:100', True))

    assert insights_launch.reconcile_stale_worker(runs_dir, 'zombie-worker')
    assert list_manifests(runs_dir)[0].status.value == 'error'


def test_macos_process_identity_parses_start_time_and_state(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from evaluatorq.dashboard import insights_launch

    monkeypatch.setattr(insights_launch.sys, 'platform', 'darwin')
    monkeypatch.setattr(
        insights_launch.subprocess,
        'run',
        lambda *_args, **_kwargs: SimpleNamespace(stdout='Sun Sep 27 23:04:52 2026 Z+'),
    )

    assert insights_launch._read_worker_process_identity(12345) == ('darwin:Sun Sep 27 23:04:52 2026', True)


def test_macos_process_identity_failure_is_conservative(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.dashboard import insights_launch

    monkeypatch.setattr(insights_launch.sys, 'platform', 'darwin')

    def unavailable(*_args: object, **_kwargs: object) -> None:
        raise OSError('ps unavailable')

    monkeypatch.setattr(insights_launch.subprocess, 'run', unavailable)
    monkeypatch.setattr(insights_launch.os, 'kill', lambda _pid, _signal: None)

    assert insights_launch._read_worker_process_identity(12345) is None
    assert insights_launch._worker_process_is_alive(12345, 'darwin:original-start')


def test_linux_process_identity_handles_parenthesis_in_command_name(monkeypatch: pytest.MonkeyPatch) -> None:
    from pathlib import Path

    from evaluatorq.dashboard import insights_launch

    monkeypatch.setattr(insights_launch.sys, 'platform', 'linux')
    tail = ['S', *(['0'] * 18), '987654']
    monkeypatch.setattr(Path, 'read_text', lambda _path, **_kwargs: f'123 (worker ) name) {" ".join(tail)}')

    assert insights_launch._read_worker_process_identity(123) == ('linux:987654', False)


def test_windows_worker_liveness_check_does_not_call_os_kill(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.dashboard import insights_launch

    monkeypatch.setattr(insights_launch.sys, 'platform', 'win32')

    def unexpected_kill(_pid: int, _signal: int) -> None:
        raise AssertionError('Windows os.kill(pid, 0) terminates the worker')

    monkeypatch.setattr(insights_launch.os, 'kill', unexpected_kill)

    assert insights_launch._worker_process_is_alive(12345, 'old-start-marker')


def test_stale_reconciliation_treats_overflowing_worker_pid_as_uncertain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import time

    from evaluatorq.dashboard import insights_launch
    from evaluatorq.dashboard.insights_launch import _write_worker_state, worker_state_path

    runs_dir = tmp_path / 'runs'
    writer = start_manifest(run_id='corrupt-pid', surface='insights', run_name='demo', runs_dir=runs_dir)
    _write_worker_state(
        worker_state_path(runs_dir, 'corrupt-pid'),
        {
            'pid': 10**100,
            'heartbeat_at': time.time() - 120,
            'snapshot_path': None,
        },
    )

    def overflow(_pid: int, _signal: int) -> None:
        raise OverflowError('pid does not fit in a platform pid_t')

    monkeypatch.setattr(insights_launch.os, 'kill', overflow)

    assert not insights_launch.reconcile_stale_worker(runs_dir, 'corrupt-pid')
    assert list_manifests(runs_dir)[0].status.value == 'running'
    assert writer.manifest.status.value == 'running'


def test_finder_worker_uses_validated_snapshot_after_export_is_replaced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from evaluatorq.dashboard import insights_worker
    from evaluatorq.dashboard.insights_launch import _REQUEST_ENV, read_launch_payload, worker_state_path

    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    exports = tmp_path / 'finder-exports'
    exports.mkdir()
    export_path = exports / 'finder.json'
    approved_export = _run_export(['approved-trace']).model_dump_json()
    export_path.write_text(approved_export, encoding='utf-8')
    spec = InsightsLaunchSpec(source='finder', finder_export='finder.json', labels=[], dimensions=['intent'])

    export_path.unlink()
    export_path.symlink_to(tmp_path / 'outside.json')
    (tmp_path / 'outside.json').write_text(_run_export(['replacement-trace']).model_dump_json(), encoding='utf-8')

    with patch('evaluatorq.dashboard.insights_launch.subprocess.Popen') as spawn:
        run_id = launch_insights(spec, tmp_path / 'runs')
    reference = finder_export_reference_path(tmp_path / 'runs', run_id)
    assert reference.exists()
    payload_json = spawn.call_args.kwargs['env'][_REQUEST_ENV]
    monkeypatch.setenv(_REQUEST_ENV, payload_json)
    payload = read_launch_payload()
    assert payload.finder_export_snapshot is not None
    assert payload.finder_export_snapshot.read_text(encoding='utf-8') == approved_export

    consumed: list[str] = []
    consumed_source: list[Path | None] = []

    async def capture_population(population, **kwargs):
        from evaluatorq.common.run_manifest import ManifestWriter
        from evaluatorq.contracts import RunManifest

        assert population.finder_export is not None
        consumed_source.append(kwargs.get('_finder_export_source'))
        consumed.append(population.finder_export.read_text(encoding='utf-8'))
        manifest_path = tmp_path / 'runs' / '.manifests' / f'{payload.run_id}.json'
        manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding='utf-8'))
        ManifestWriter(manifest, manifest_path).complete()
        return type('CompletedRun', (), {'status': 'completed'})()

    monkeypatch.setattr(insights_worker, 'insights', capture_population)
    monkeypatch.setenv(_REQUEST_ENV, payload_json)
    assert insights_worker.main() == 0
    assert consumed == [approved_export]
    assert consumed_source == [export_path]
    assert not payload.finder_export_snapshot.exists()
    assert not reference.exists()
    assert not worker_state_path(tmp_path / 'runs', payload.run_id).exists()
    assert list_manifests(tmp_path / 'runs')[0].status == 'completed'


def test_finder_snapshot_cleanup_without_getuid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from evaluatorq.dashboard import insights_worker

    directory = tmp_path / 'evaluatorq-finder-snapshot-test'
    directory.mkdir(mode=0o700)
    snapshot = directory / 'finder-export.json'
    snapshot.write_text('{}', encoding='utf-8')
    snapshot.chmod(0o600)
    monkeypatch.setattr(insights_worker.tempfile, 'gettempdir', lambda: str(tmp_path))
    monkeypatch.setattr(insights_worker, 'os', SimpleNamespace(name='nt'))

    insights_worker._cleanup_snapshot(snapshot)

    assert not directory.exists()


def test_live_facet_selection_reaches_population_and_worker(tmp_path: Path) -> None:
    facets = FacetSelection(status=frozenset({'error'}), provider=frozenset({'openai', 'anthropic'}))
    spec = InsightsLaunchSpec(source='query', query='refunds', window_days=3, facets=facets)
    assert spec.population().facets == facets
    assert spec.population().window_days == 3
    with patch('evaluatorq.dashboard.insights_launch.subprocess.Popen') as spawn:
        launch_insights(spec, tmp_path)
    payload = InsightsLaunchPayload.model_validate_json(spawn.call_args.kwargs['env']['EVALUATORQ_INSIGHTS_LAUNCH_REQUEST'])
    assert payload.spec.population().facets == facets


@pytest.mark.parametrize(
    ('host', 'expected_host'),
    [('https://profile.example', 'https://profile.example'), (None, 'https://my.orq.ai')],
)
def test_selected_profile_controls_facets_and_worker_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, host: str | None, expected_host: str
) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    monkeypatch.setenv('ORQ_API_KEY', 'environment-key')
    monkeypatch.setenv('ORQ_BASE_URL', 'https://environment.example')
    client = TestClient(build_app())
    profile = OrqProfile('staging', 'profile-key', host, False)
    getattr(client.app, 'state').finder_settings = DashboardSettings.model_validate({'orq_profile': 'staging'})
    getattr(client.app, 'state').finder_profile = profile
    page = client.get('/insights/new')
    token = re.search(r'name="csrf" value="([^"]+)"', page.text)
    assert token is not None

    with (
        patch('evaluatorq.dashboard.insights_routes.resolve_orq_client', return_value=object()) as resolve,
        patch('evaluatorq.dashboard.insights_routes.load_facet_catalogue', new_callable=AsyncMock, return_value=FacetCatalogue()),
        patch('evaluatorq.dashboard.insights_routes.close_orq_client', new_callable=AsyncMock),
        patch('evaluatorq.dashboard.insights_launch.subprocess.Popen') as spawn,
    ):
        assert client.get('/insights/facets?window_days=7').status_code == 200
        response = client.post(
            '/insights/runs',
            data={'csrf': token.group(1), 'source': 'recent', 'dimensions': 'intent'},
            follow_redirects=False,
        )

    assert response.status_code == 303
    resolve.assert_called_once_with('profile-key', base_url=expected_host)
    worker_env = spawn.call_args.kwargs['env']
    assert worker_env['ORQ_API_KEY'] == 'profile-key'
    assert worker_env['ORQ_BASE_URL'] == expected_host
    assert 'profile-key' not in list_manifests(tmp_path / 'insights-runs')[0].model_dump_json()


def test_missing_selected_profile_never_uses_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    monkeypatch.setenv('ORQ_API_KEY', 'environment-key')
    client = TestClient(build_app())
    getattr(client.app, 'state').finder_settings = DashboardSettings.model_validate({'orq_profile': 'deleted'})
    getattr(client.app, 'state').finder_profile = None
    page = client.get('/insights/new')
    token = re.search(r'name="csrf" value="([^"]+)"', page.text)
    assert token is not None

    with (
        patch('evaluatorq.dashboard.insights_routes.resolve_orq_client') as resolve,
        patch('evaluatorq.dashboard.insights_routes.launch_insights') as launch,
    ):
        facets = client.get('/insights/facets?window_days=7')
        start = client.post(
            '/insights/runs',
            data={'csrf': token.group(1), 'source': 'recent', 'dimensions': 'intent'},
            follow_redirects=False,
        )

    assert 'Facet values are unavailable' in facets.text
    assert start.status_code == 422
    assert 'Orq profile deleted is unavailable' in start.text
    resolve.assert_not_called()
    launch.assert_not_called()


def test_facet_cache_separates_profiles(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    client = TestClient(build_app())
    getattr(client.app, 'state').finder_settings = DashboardSettings.model_validate({'orq_profile': 'first'})
    getattr(client.app, 'state').finder_profile = OrqProfile('first', 'first-key', None, False)
    with (
        patch('evaluatorq.dashboard.insights_routes.resolve_orq_client', return_value=object()) as resolve,
        patch('evaluatorq.dashboard.insights_routes.load_facet_catalogue', new_callable=AsyncMock, return_value=FacetCatalogue()),
        patch('evaluatorq.dashboard.insights_routes.close_orq_client', new_callable=AsyncMock),
    ):
        client.get('/insights/facets?window_days=7')
        getattr(client.app, 'state').finder_settings = DashboardSettings.model_validate({'orq_profile': 'second'})
        getattr(client.app, 'state').finder_profile = OrqProfile('second', 'second-key', None, False)
        client.get('/insights/facets?window_days=7')

    assert resolve.call_count == 2
    assert resolve.call_args_list[0].args[0] == 'first-key'
    assert resolve.call_args_list[1].args[0] == 'second-key'


@pytest.mark.asyncio
async def test_concurrent_insights_facet_misses_share_one_provider_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    app = build_app()
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0
    catalogue = FacetCatalogue(project=('shared',))

    async def load(*args: object, **kwargs: object) -> FacetCatalogue:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return catalogue

    with (
        patch('evaluatorq.dashboard.insights_routes.resolve_orq_client', return_value=object()),
        patch('evaluatorq.dashboard.insights_routes.load_facet_catalogue', load),
        patch('evaluatorq.dashboard.insights_routes.close_orq_client', new_callable=AsyncMock),
    ):
        first = asyncio.create_task(insights_routes._catalogue(app, 7))
        await started.wait()
        second = asyncio.create_task(insights_routes._catalogue(app, 7))
        await asyncio.sleep(0)
        release.set()
        assert await asyncio.gather(first, second) == [catalogue, catalogue]

    assert calls == 1
    assert len(app.state.insights_facet_catalogues) == 1


@pytest.mark.asyncio
async def test_settings_generation_change_discards_stale_insights_facet_fetch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    app = build_app()
    started = asyncio.Event()
    release = asyncio.Event()

    async def load(*args: object, **kwargs: object) -> FacetCatalogue:
        started.set()
        await release.wait()
        return FacetCatalogue(project=('old-profile',))

    with (
        patch('evaluatorq.dashboard.insights_routes.resolve_orq_client', return_value=object()),
        patch('evaluatorq.dashboard.insights_routes.load_facet_catalogue', load),
        patch('evaluatorq.dashboard.insights_routes.close_orq_client', new_callable=AsyncMock),
    ):
        stale = asyncio.create_task(insights_routes._catalogue(app, 7))
        await started.wait()
        # This is the cache invalidation performed by the successful settings save route.
        app.state.finder_generation += 1
        del app.state.insights_facet_catalogues
        release.set()
        assert await stale is None

    assert not hasattr(app.state, 'insights_facet_catalogues')


def test_facet_cache_key_tracks_same_profile_credentials_and_endpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    client = TestClient(build_app())
    getattr(client.app, 'state').finder_settings = DashboardSettings.model_validate({'orq_profile': 'same'})
    getattr(client.app, 'state').finder_profile = OrqProfile('same', 'first-secret', 'https://first.example', False)
    with (
        patch('evaluatorq.dashboard.insights_routes.resolve_orq_client', return_value=object()) as resolve,
        patch('evaluatorq.dashboard.insights_routes.load_facet_catalogue', new_callable=AsyncMock, return_value=FacetCatalogue()),
        patch('evaluatorq.dashboard.insights_routes.close_orq_client', new_callable=AsyncMock),
    ):
        client.get('/insights/facets?window_days=7')
        getattr(client.app, 'state').finder_profile = OrqProfile('same', 'second-secret', 'https://second.example', False)
        client.get('/insights/facets?window_days=7')

    assert resolve.call_count == 2
    cache = getattr(client.app, 'state').insights_facet_catalogues
    assert 'first-secret' not in repr(cache)
    assert 'second-secret' not in repr(cache)


def test_facet_options_use_selected_window_and_keep_values(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    client = TestClient(build_app())
    catalogue = FacetCatalogue(status=('ok',), provider=('openai',))
    load = AsyncMock(return_value=catalogue)
    with (
        patch('evaluatorq.dashboard.insights_routes.resolve_orq_client', return_value=object()),
        patch('evaluatorq.dashboard.insights_routes.load_facet_catalogue', load),
        patch('evaluatorq.dashboard.insights_routes.close_orq_client', new_callable=AsyncMock),
    ):
        response = client.get('/insights/facets?window_days=3&facet_status=error')
        cached = client.get('/insights/facets?window_days=3')
        changed = client.get('/insights/facets?window_days=4')

    assert response.status_code == cached.status_code == changed.status_code == 200
    assert 'name="facet_status" value="error" checked' in response.text
    assert 'name="facet_status" value="ok"' in response.text
    assert load.await_count == 2
    first = load.await_args_list[0].kwargs
    assert first['limit'] == 50
    assert isinstance(first['start'], datetime)
    assert isinstance(first['end'], datetime)
    assert abs(first['end'] - first['start'] - timedelta(days=3)) < timedelta(seconds=1)
    assert client.get('/insights/facets?window_days=91').status_code == 422


def test_facet_catalogue_failure_is_visible(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    client = TestClient(build_app())
    with (
        patch('evaluatorq.dashboard.insights_routes.resolve_orq_client', return_value=object()),
        patch('evaluatorq.dashboard.insights_routes.load_facet_catalogue', new_callable=AsyncMock, side_effect=RuntimeError('Orq unavailable')),
        patch('evaluatorq.dashboard.insights_routes.close_orq_client', new_callable=AsyncMock),
    ):
        response = client.get('/insights/facets?window_days=7&facet_status=error')
    assert response.status_code == 200
    assert 'Facet values are unavailable' in response.text
    assert 'name="facet_status" value="error" checked' in response.text


def test_new_run_form_rejects_bad_input_and_launches_valid_request(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    client = TestClient(build_app())
    page = client.get('/insights/new')
    assert page.status_code == 200
    assert 'New Insights run' in page.text
    token = re.search(r'name="csrf" value="([^"]+)"', page.text)
    assert token is not None

    with patch('evaluatorq.dashboard.insights_routes.launch_insights', return_value='launched-id') as launch:
        invalid = client.post('/insights/runs', data={'csrf': token.group(1), 'source': 'query', 'dimensions': 'intent'})
        assert invalid.status_code == 422
        assert 'Enter a question' in invalid.text
        forbidden = client.post('/insights/runs', data={'source': 'recent', 'dimensions': 'intent'})
        assert forbidden.status_code == 403
        valid = client.post(
            '/insights/runs',
            data={'csrf': token.group(1), 'source': 'recent', 'dimensions': 'intent', 'limit': '5', 'facet_status': 'error'},
            follow_redirects=False,
        )

    assert valid.status_code == 303
    assert valid.headers['location'] == '/insights/launched-id'
    assert launch.call_args.args[0].limit == 5
    assert launch.call_args.args[0].population().facets.status == frozenset({'error'})


def test_running_and_completed_pages_show_manifest_stages(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    directory = tmp_path / 'insights-runs'
    writer = start_manifest(
        run_id='run-1',
        surface='insights',
        run_name='Sample run',
        runs_dir=directory,
        planned_stages=['population', 'summary', 'dimension:intent', 'write'],
        stage_labels={'population': 'Load recent traces', 'summary': 'Summarize traces', 'dimension:intent': 'Cluster and map intent', 'write': 'Save run'},
    )
    writer.start_stage('population')
    writer.end_stage('population')
    writer.start_stage('summary')
    client = TestClient(build_app())

    running = client.get('/insights/run-1')
    overview = client.get('/insights')
    assert running.status_code == overview.status_code == 200
    assert 'Summarize traces' in running.text
    assert 'pending' in running.text
    assert 'Summarize traces' in overview.text
    assert 'location.reload()' in running.text

    writer.end_stage('summary')
    report = save_run(minimal_run.__wrapped__(), directory)
    writer.complete(report)
    completed = client.get('/insights/run-1')
    assert completed.status_code == 200
    assert 'Load recent traces' in completed.text
    assert 'Cluster and map intent' in completed.text
    assert 'skipped' in completed.text
