"""Dashboard run creation and manifest-backed Insights progress."""

from __future__ import annotations

import asyncio
import os
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
from evaluatorq.dashboard.auth import DashboardAuth, auth_identity
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
from tests.insights.test_population import _run_export, make_trace


def test_stage_plan_follows_source_labels_and_dimensions() -> None:
    query_plan = dict(stage_plan(InsightsPopulation(query='refunds'), [SENTIMENT], ['intent', 'failure']))
    finder_plan = dict(stage_plan(InsightsPopulation.from_finder_export(Path('finder.json')), [], ['sentiment']))
    snapshot_plan = dict(stage_plan(InsightsPopulation.from_snapshot(Path('traces.json')), [], ['intent']))
    priority_plan = dict(stage_plan(InsightsPopulation(), [CUSTOMER_SATISFACTION], ['intent']))

    assert query_plan['population'] == 'Find matching traces'
    assert query_plan['label'] == 'Match and classify traces'
    assert list(query_plan).count('dimension:intent') == 1
    assert 'dimension:failure' in query_plan
    assert 'dimension:sentiment' not in query_plan
    assert finder_plan['population'] == 'Load Finder matches'
    assert finder_plan['label'] == 'Classify traces'
    assert snapshot_plan['population'] == 'Load local traces'
    assert priority_plan['priority'] == 'Build priority matrix'


def test_worker_rejects_changed_credential_before_building_clients(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from evaluatorq.dashboard import insights_worker

    settings = DashboardSettings.model_validate({'orq_auth_method': 'environment'})
    expected = auth_identity(DashboardAuth('environment', 'key-at-launch', 'https://my.orq.ai'), settings)
    payload = InsightsLaunchPayload(
        run_id='run-1',
        run_name='Auth change test',
        runs_dir=tmp_path,
        spec=InsightsLaunchSpec(),
        auth_method='environment',
        auth_identity=expected,
    )
    monkeypatch.setenv('ORQ_API_KEY', 'different-key')
    monkeypatch.setattr(insights_worker, 'read_launch_payload', lambda: payload)
    monkeypatch.setattr(insights_worker, 'effective_settings', lambda: settings)
    monkeypatch.setattr(insights_worker, '_fail_running', lambda *_args: None)

    def must_not_build(*_args: object, **_kwargs: object) -> object:
        raise AssertionError('worker must not build clients after a credential change')

    monkeypatch.setattr(insights_worker, 'build_auth_clients', must_not_build)
    assert insights_worker.main() == 1


def test_oauth_identity_changes_with_signed_in_account(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.common import cli_oauth

    settings = DashboardSettings.model_validate({'orq_auth_method': 'cli_oauth'})
    auth = DashboardAuth('cli_oauth', None, 'https://my.orq.ai')
    monkeypatch.setattr(cli_oauth, 'oauth_subject', lambda _server: {'user_id': 'user-one'})
    first = auth_identity(auth, settings)
    monkeypatch.setattr(cli_oauth, 'oauth_subject', lambda _server: {'user_id': 'user-two'})
    second = auth_identity(auth, settings)

    assert first != second
    assert 'user-one' not in first
    assert 'user-two' not in second


def test_launch_persists_plan_before_spawning_worker(tmp_path: Path) -> None:
    spec = InsightsLaunchSpec(source='query', query='refunds', labels=['sentiment'], dimensions=['intent', 'failure'])
    with (
        patch('evaluatorq.dashboard.insights_launch.subprocess.Popen') as spawn,
        patch('evaluatorq.dashboard.insights_launch._worker_process_identity', return_value='test:1'),
    ):
        spawn.return_value.pid = 43210
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


def test_worker_starts_only_after_parent_records_its_pid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    import os
    import subprocess
    from types import SimpleNamespace

    from evaluatorq.dashboard.insights_launch import worker_state_path

    observed: list[object] = []
    state_file: Path | None = None

    class WorkerInput:
        def write(self, data: bytes) -> int:
            assert state_file is not None
            state = json.loads(state_file.read_text(encoding='utf-8'))
            assert state['pid'] == 43210
            observed.append(data)
            return len(data)

        def close(self) -> None:
            observed.append('closed')

    def spawn(*_args: object, **kwargs: object) -> SimpleNamespace:
        nonlocal state_file
        observed.append('spawned')
        assert kwargs['stdin'] == subprocess.PIPE
        environment = kwargs['env']
        assert isinstance(environment, dict)
        manifest_path = Path(str(environment['EVALUATORQ_INSIGHTS_MANIFEST']))
        state_file = worker_state_path(tmp_path, manifest_path.stem)
        state = json.loads(state_file.read_text(encoding='utf-8'))
        assert state['pid'] is None
        assert state['launcher_pid'] == os.getpid()
        assert state['launcher_process_identity'] == 'test:1'
        return SimpleNamespace(pid=43210, stdin=WorkerInput())

    monkeypatch.setattr('evaluatorq.dashboard.insights_launch.subprocess.Popen', spawn)
    monkeypatch.setattr('evaluatorq.dashboard.insights_launch._worker_process_identity', lambda _pid: 'test:1')

    run_id = launch_insights(InsightsLaunchSpec(), tmp_path)
    state_file = worker_state_path(tmp_path, run_id)

    assert observed == ['spawned', b'1', 'closed']
    assert json.loads(state_file.read_text(encoding='utf-8'))['pid'] == 43210


def test_worker_is_not_released_when_pid_state_cannot_be_saved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from evaluatorq.dashboard import insights_launch

    observed: list[object] = []

    class WorkerInput:
        def write(self, data: bytes) -> int:
            observed.append(data)
            return len(data)

        def close(self) -> None:
            observed.append('closed')

    original_write = insights_launch._write_worker_state

    def fail_pid_write(path: Path, state: dict[str, object]) -> None:
        if state.get('pid') is not None:
            raise OSError('state unavailable')
        original_write(path, state)

    monkeypatch.setattr(insights_launch, '_write_worker_state', fail_pid_write)
    monkeypatch.setattr(insights_launch, '_worker_process_identity', lambda _pid: 'test:1')
    monkeypatch.setattr(
        insights_launch.subprocess, 'Popen', lambda *_args, **_kwargs: SimpleNamespace(pid=43210, stdin=WorkerInput())
    )

    run_id = launch_insights(InsightsLaunchSpec(), tmp_path)

    assert observed == ['closed']
    manifest = list_manifests(tmp_path)[0]
    assert manifest.run_id == run_id
    assert manifest.status.value == 'error'
    assert manifest.error is not None and 'state unavailable' in manifest.error


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


def test_insights_listing_does_not_reconcile_another_surface_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import time

    from evaluatorq.dashboard.insights_launch import _write_worker_state, worker_state_path

    runs_dir = tmp_path / 'runs'
    writer = start_manifest(run_id='redteam-worker', surface='redteam', run_name='other surface', runs_dir=runs_dir)
    state_path = worker_state_path(runs_dir, 'redteam-worker')
    _write_worker_state(state_path, {'pid': 12345, 'heartbeat_at': time.time() - 120})
    monkeypatch.setattr('evaluatorq.dashboard.insights_launch._worker_process_is_alive', lambda *_: False)

    entries, _, manifests = insights_routes._entries(runs_dir)

    assert ('redteam-worker', 'other surface', 'running') not in entries
    assert manifests == {}
    assert writer.manifest.status.value == 'running'
    assert list_manifests(runs_dir)[0].status.value == 'running'
    assert state_path.exists()


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


def test_snapshot_descriptor_closes_when_fdopen_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import os
    import tempfile

    from evaluatorq.dashboard import insights_launch

    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
    exports = tmp_path / 'finder-exports'
    exports.mkdir()
    (exports / 'approved.json').write_text(_run_export(['trace-1']).model_dump_json(), encoding='utf-8')
    spec = InsightsLaunchSpec(source='finder', finder_export='approved.json')
    opened_descriptors: list[int] = []

    def fail_fdopen(descriptor: int, _mode: str, **_kwargs: object):
        opened_descriptors.append(descriptor)
        raise OSError('snapshot fdopen failed')

    monkeypatch.setattr(insights_launch.os, 'fdopen', fail_fdopen)

    run_id = launch_insights(spec, tmp_path / 'runs')

    manifest = list_manifests(tmp_path / 'runs')[0]
    assert manifest.run_id == run_id
    assert manifest.status.value == 'error'
    assert manifest.error is not None and 'snapshot fdopen failed' in manifest.error
    assert len(opened_descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(opened_descriptors[0])
    assert not finder_export_reference_path(tmp_path / 'runs', run_id).exists()
    assert list(tmp_path.glob('evaluatorq-finder-snapshot-*')) == []


def test_worker_import_failure_marks_manifest_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import io
    import runpy
    import sys
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
    snapshot_path.chmod(0o600)
    monkeypatch.setenv(_REQUEST_ENV, '{truncated request')
    monkeypatch.setenv(_SNAPSHOT_ENV, str(snapshot_path))
    monkeypatch.setattr(sys, 'stdin', io.TextIOWrapper(io.BytesIO(b'1')))

    def fail_import(*args, **kwargs):
        raise ImportError('missing optional dependency')

    with patch.object(runpy, 'run_module', side_effect=fail_import), pytest.raises(ImportError):
        exec(_WORKER_BOOTSTRAP, {})

    manifest = list_manifests(tmp_path)[0]
    assert manifest.status == 'error'
    assert manifest.stage == 'setup'
    assert manifest.error is not None and 'missing optional dependency' in manifest.error
    if os.name == 'nt':
        # The stdlib-only bootstrap cannot safely delete snapshots when worker
        # import itself fails, so it leaves the private temporary directory.
        assert snapshot_directory.exists()
        snapshot_path.unlink()
        snapshot_directory.rmdir()
    else:
        assert not snapshot_directory.exists()
    assert not reference.exists()


def test_worker_bootstrap_snapshot_cleanup_does_not_follow_directory_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import io
    import runpy
    import sys
    import tempfile

    from evaluatorq.dashboard.insights_launch import _WORKER_BOOTSTRAP, _MANIFEST_ENV, _REQUEST_ENV, _SNAPSHOT_ENV

    writer = start_manifest(run_id='bootstrap-snapshot-swap', surface='insights', run_name='demo', runs_dir=tmp_path)
    monkeypatch.setenv(_MANIFEST_ENV, str(writer.path))
    monkeypatch.setenv(_REQUEST_ENV, '{truncated request')
    monkeypatch.setattr(sys, 'stdin', io.TextIOWrapper(io.BytesIO(b'1')))

    snapshot_directory = Path(tempfile.mkdtemp(prefix='evaluatorq-finder-snapshot-'))
    snapshot_path = snapshot_directory / 'finder-export.json'
    snapshot_path.write_text('{}', encoding='utf-8')
    snapshot_path.chmod(0o600)
    replacement = Path(tempfile.mkdtemp(prefix='snapshot-replacement-'))
    protected = replacement / 'keep.txt'
    protected.write_text('keep', encoding='utf-8')
    original_directory = snapshot_directory.with_name('original-bootstrap-snapshot')
    monkeypatch.setenv(_SNAPSHOT_ENV, str(snapshot_path))

    original_open = os.open
    swapped = False

    def swap_after_directory_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if path == snapshot_directory.name and dir_fd is not None and not swapped:
            swapped = True
            snapshot_directory.rename(original_directory)
            replacement.rename(snapshot_directory)
        return descriptor

    monkeypatch.setattr(os, 'open', swap_after_directory_open)

    def fail_import(*args, **kwargs):
        raise ImportError('missing optional dependency')

    with patch.object(runpy, 'run_module', side_effect=fail_import), pytest.raises(ImportError):
        exec(_WORKER_BOOTSTRAP, {})

    if os.name == 'nt':
        # Bootstrap cleanup intentionally avoids path-based deletion on Windows.
        assert not swapped
        assert snapshot_path.exists()
        assert protected.read_text(encoding='utf-8') == 'keep'
        snapshot_path.unlink()
        snapshot_directory.rmdir()
        protected.unlink()
        replacement.rmdir()
    else:
        assert swapped
        assert (snapshot_directory / 'keep.txt').read_text(encoding='utf-8') == 'keep'
        assert not (original_directory / 'finder-export.json').exists()
        snapshot_directory.joinpath('keep.txt').unlink()
        snapshot_directory.rmdir()
        original_directory.rmdir()
    assert list_manifests(tmp_path)[0].status == 'error'


def test_successful_worker_exit_does_not_recover_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import io
    import runpy
    import sys

    from evaluatorq.dashboard.insights_launch import _WORKER_BOOTSTRAP, _MANIFEST_ENV, _REQUEST_ENV

    writer = start_manifest(run_id='worker-success', surface='insights', run_name='demo', runs_dir=tmp_path)
    monkeypatch.setenv(_MANIFEST_ENV, str(writer.path))
    monkeypatch.setenv(_REQUEST_ENV, '{invalid request')
    monkeypatch.setattr(sys, 'stdin', io.TextIOWrapper(io.BytesIO(b'1')))

    def exit_cleanly(*args, **kwargs):
        raise SystemExit(0)

    monkeypatch.setattr(runpy, 'run_module', exit_cleanly)

    with pytest.raises(SystemExit) as exit_info:
        exec(_WORKER_BOOTSTRAP, {})

    assert exit_info.value.code == 0
    assert list_manifests(tmp_path)[0].status == 'running'


def test_worker_does_not_start_when_launcher_closes_handshake_without_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import io
    import runpy
    import sys

    from evaluatorq.dashboard.insights_launch import _WORKER_BOOTSTRAP, _MANIFEST_ENV, _REQUEST_ENV

    start_manifest(run_id='launcher-died-before-release', surface='insights', run_name='demo', runs_dir=tmp_path)
    monkeypatch.setenv(_MANIFEST_ENV, str(tmp_path / '.manifests' / 'launcher-died-before-release.json'))
    monkeypatch.setenv(_REQUEST_ENV, '{}')
    monkeypatch.setattr(sys, 'stdin', io.TextIOWrapper(io.BytesIO(b'')))
    monkeypatch.setattr(runpy, 'run_module', lambda *_args, **_kwargs: pytest.fail('worker must not start'))

    with pytest.raises(SystemExit) as exit_info:
        exec(_WORKER_BOOTSTRAP, {})

    assert exit_info.value.code == 0
    assert list_manifests(tmp_path)[0].status.value == 'running'


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


def test_worker_payload_decode_failure_releases_finder_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evaluatorq.dashboard import insights_worker
    from evaluatorq.dashboard.insights_launch import _MANIFEST_ENV, _REQUEST_ENV

    runs_dir = tmp_path / 'insights-runs'
    writer = start_manifest(run_id='decode-finder-failure', surface='insights', run_name='demo', runs_dir=runs_dir)
    reference = finder_export_reference_path(runs_dir, writer.manifest.run_id)
    ensure_private_finder_reference_dir(reference.parent)
    reference.write_text('{"finder_export":"/private/export.json"}', encoding='utf-8')
    reference.chmod(0o600)
    monkeypatch.setenv(_REQUEST_ENV, '{truncated request')
    monkeypatch.setenv(_MANIFEST_ENV, str(writer.path))

    assert insights_worker.main() == 1

    assert not reference.exists()
    manifest = list_manifests(runs_dir)[0]
    assert manifest.status == 'error'


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
    with pytest.raises(ValidationError, match='Enter the path to a local trace snapshot'):
        InsightsLaunchSpec(source='snapshot')


def test_wizard_accepts_local_snapshot_and_rejects_raw_sessions(tmp_path: Path) -> None:
    from evaluatorq.trace_finder.models import Snapshot

    path = tmp_path / 'traces.json'
    path.write_text(Snapshot(traces=(make_trace('local'),)).model_dump_json(), encoding='utf-8')
    spec = InsightsLaunchSpec(source='snapshot', snapshot_path=str(path))
    assert spec.population().snapshot_path == path

    path.write_text(Snapshot(traces=()).model_dump_json(), encoding='utf-8')
    with pytest.raises(ValidationError, match='contains no traces'):
        InsightsLaunchSpec(source='snapshot', snapshot_path=str(path))

    path.write_text('[{"session_id": "one", "thread": []}]', encoding='utf-8')
    with pytest.raises(ValidationError, match='valid local trace snapshot'):
        InsightsLaunchSpec(source='snapshot', snapshot_path=str(path))


def test_dashboard_starts_local_snapshot_run_without_trace_lookup(tmp_path: Path) -> None:
    from evaluatorq.trace_finder.models import Snapshot

    path = tmp_path / 'traces.json'
    path.write_text(Snapshot(traces=(make_trace('local'),)).model_dump_json(), encoding='utf-8')
    client = TestClient(build_app())
    page = client.get('/insights/new')
    assert 'Local trace file' in page.text
    token = re.search(r'name="csrf" value="([^"]+)"', page.text)
    assert token is not None

    with (
        patch('evaluatorq.dashboard.insights_routes.selected_dashboard_auth', return_value=DashboardAuth('environment', 'key', 'https://my.orq.ai')),
        patch('evaluatorq.dashboard.insights_routes.launch_insights', return_value='run-1') as launch,
    ):
        response = client.post(
            '/insights/runs',
            data={'csrf': token.group(1), 'source': 'snapshot', 'snapshot_path': str(path), 'dimensions': 'intent'},
            follow_redirects=False,
        )

    assert response.status_code == 303
    assert launch.call_args.args[0].population().snapshot_path == path


def test_local_snapshot_preview_reports_omissions_before_start(tmp_path: Path) -> None:
    from evaluatorq.trace_finder.models import Snapshot

    trace = make_trace('long').model_copy(
        update={'messages': ({'role': 'user', 'content': 'a' * 60_000}, {'role': 'assistant', 'content': 'done'})}
    )
    path = tmp_path / 'traces.json'
    path.write_text(Snapshot(traces=(trace,)).model_dump_json(), encoding='utf-8')
    client = TestClient(build_app())
    page = client.get('/insights/new')
    token = re.search(r'name="csrf" value="([^"]+)"', page.text)
    assert token is not None
    assert 'insights-snapshot-preview' in page.text

    preview = client.post('/insights/snapshot-preview', data={'csrf': token.group(1), 'snapshot_path': str(path)})

    assert preview.status_code == 200
    assert '1 of 1 trace exceeds' in preview.text
    assert '1 of 2 whole messages omitted (50.0%)' in preview.text
    assert 'Serialized source:' in preview.text
    assert client.post('/insights/snapshot-preview', data={'snapshot_path': str(path)}).status_code == 403
    assert client.post(
        '/insights/snapshot-preview', data={'csrf': token.group(1), 'snapshot_path': str(tmp_path / 'missing.json')}
    ).status_code == 422
    path.write_text(Snapshot(traces=()).model_dump_json(), encoding='utf-8')
    empty = client.post('/insights/snapshot-preview', data={'csrf': token.group(1), 'snapshot_path': str(path)})
    assert empty.status_code == 422
    assert 'contains no traces' in empty.text


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


def test_windows_path_normalization_resolves_short_path_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from evaluatorq.dashboard import insights_launch

    long_path = r'C:\Users\runneradmin\AppData\Local\Temp'
    short_path = r'C:\Users\RUNNER~1\AppData\Local\Temp'

    def realpath(value: str) -> str:
        return long_path if value == short_path else value

    monkeypatch.setattr(
        insights_launch,
        'os',
        SimpleNamespace(name='nt', path=SimpleNamespace(realpath=realpath)),
    )

    assert insights_launch._normalized_windows_path(short_path) == insights_launch._normalized_windows_path(
        long_path
    )
    extended_path = '\\\\?\\' + long_path
    assert insights_launch._normalized_windows_path(extended_path) == insights_launch._normalized_windows_path(long_path)


@pytest.mark.skipif(os.name == 'nt', reason='POSIX dir_fd swap; Windows reparse swap has dedicated coverage')
def test_finder_export_replacement_with_symlink_during_open_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from evaluatorq.dashboard import insights_launch

    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    exports = tmp_path / 'finder-exports'
    exports.mkdir()
    approved = exports / 'approved.json'
    approved.write_text(_run_export(['approved-trace']).model_dump_json(), encoding='utf-8')
    outside = tmp_path / 'outside.json'
    outside.write_text(_run_export(['outside-trace']).model_dump_json(), encoding='utf-8')
    original_open = insights_launch.os.open

    def replace_before_open(path: Path, flags: int, *args: object, **kwargs: object) -> int:
        if Path(path) == approved or (Path(path) == Path('approved.json') and 'dir_fd' in kwargs):
            approved.unlink()
            approved.symlink_to(outside)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(insights_launch.os, 'open', replace_before_open)

    with pytest.raises(ValidationError, match='Could not read a valid Finder export'):
        InsightsLaunchSpec(source='finder', finder_export='approved.json')


@pytest.mark.skipif(os.name == 'nt', reason='POSIX dir_fd swap; Windows reparse swap has dedicated coverage')
def test_finder_export_directory_replacement_during_open_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from evaluatorq.dashboard import insights_launch

    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    exports = tmp_path / 'finder-exports'
    exports.mkdir()
    (exports / 'approved.json').write_text(_run_export(['approved-trace']).model_dump_json(), encoding='utf-8')
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'approved.json').write_text(_run_export(['outside-trace']).model_dump_json(), encoding='utf-8')
    original_open = insights_launch.os.open

    def replace_directory_before_open(path: Path, flags: int, *args: object, **kwargs: object) -> int:
        if Path(path) == exports:
            exports.rename(tmp_path / 'old-exports')
            exports.symlink_to(outside, target_is_directory=True)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(insights_launch.os, 'open', replace_directory_before_open)

    with pytest.raises(ValidationError, match='Could not read a valid Finder export'):
        InsightsLaunchSpec(source='finder', finder_export='approved.json')


def test_windows_finder_export_reparse_swap_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import ctypes
    import types

    from evaluatorq.dashboard import insights_launch

    exports = tmp_path / 'finder-exports'
    exports.mkdir()
    approved = exports / 'approved.json'
    approved.write_text('{}', encoding='utf-8')
    calls: list[str] = []

    def create_file(*_args: object) -> int:
        calls.append('open-without-following-reparse')
        return 123

    def get_info(_handle: object, _kind: int, info_pointer: object, _size: int) -> int:
        info = ctypes.cast(info_pointer, ctypes.POINTER(type_info)).contents
        info.attributes = 0x400  # FILE_ATTRIBUTE_REPARSE_POINT on the opened handle.
        return 1

    class type_info(ctypes.Structure):
        _fields_ = [('attributes', ctypes.c_uint32), ('reparse_tag', ctypes.c_uint32)]

    def unused(*_args: object) -> int:
        raise AssertionError('A reparse point must be rejected before resolving or reading its target.')

    api = types.SimpleNamespace(
        CreateFileW=create_file,
        GetFileInformationByHandleEx=get_info,
        GetFinalPathNameByHandleW=unused,
        CloseHandle=lambda _handle: 1,
    )
    monkeypatch.setattr(ctypes, 'WinDLL', lambda *_args, **_kwargs: api, raising=False)
    monkeypatch.setattr(insights_launch, '_windows_directory_identity', lambda _root: (1, 2, 3))
    monkeypatch.setitem(__import__('sys').modules, 'msvcrt', types.SimpleNamespace())

    with pytest.raises(OSError, match='reparse point'):
        insights_launch._open_windows_approved_regular_file(exports, approved)
    assert calls == ['open-without-following-reparse', 'open-without-following-reparse']


def test_windows_finder_export_pins_parent_without_delete_sharing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import ctypes
    import types

    from evaluatorq.dashboard import insights_launch

    exports = tmp_path / 'finder-exports'
    exports.mkdir()
    approved = exports / 'approved.json'
    approved.write_text('{}', encoding='utf-8')
    calls: list[tuple[object, ...]] = []
    closed: list[int] = []

    def create_file(*args: object) -> int:
        calls.append(args)
        return 123 if args[0] == str(exports) else 456

    class type_info(ctypes.Structure):
        _fields_ = [('attributes', ctypes.c_uint32), ('reparse_tag', ctypes.c_uint32)]

    def get_info(_handle: object, _kind: int, info_pointer: object, _size: int) -> int:
        info = ctypes.cast(info_pointer, ctypes.POINTER(type_info)).contents
        info.attributes = 0
        return 1

    def get_final_path(_handle: object, buffer: object, _size: int, _flags: int) -> int:
        value = ctypes.create_unicode_buffer(str(approved))
        ctypes.memmove(buffer, value, ctypes.sizeof(value))
        return len(value.value)

    api = types.SimpleNamespace(
        CreateFileW=create_file,
        GetFileInformationByHandleEx=get_info,
        GetFinalPathNameByHandleW=get_final_path,
        CloseHandle=lambda handle: closed.append(handle) or 1,
    )
    monkeypatch.setattr(ctypes, 'WinDLL', lambda *_args, **_kwargs: api, raising=False)
    monkeypatch.setattr(insights_launch, '_windows_directory_identity', lambda _root: (1, 2, 3))
    monkeypatch.setitem(
        __import__('sys').modules,
        'msvcrt',
        types.SimpleNamespace(open_osfhandle=lambda *_args: 789),
    )

    descriptor, identity = insights_launch._open_windows_approved_regular_file(exports, approved)

    assert (descriptor, identity) == (789, (1, 2, 3))
    assert len(calls) == 2
    assert calls[0][0] == str(exports)
    assert calls[0][1] == 0x80  # FILE_READ_ATTRIBUTES
    assert calls[0][2] == 0x1 | 0x2  # Share read/write, deny delete sharing.
    assert calls[0][4] == 3  # OPEN_EXISTING
    assert calls[0][5] == 0x00200000 | 0x02000000  # OPEN_REPARSE_POINT | BACKUP_SEMANTICS
    assert calls[1][0] == str(approved)
    assert closed == [123]


def test_windows_finder_export_directory_swap_during_create_file_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import ctypes
    import types

    from evaluatorq.dashboard import insights_launch

    exports = tmp_path / 'finder-exports'
    exports.mkdir()
    approved = exports / 'approved.json'
    approved.write_text('{}', encoding='utf-8')
    replacement = tmp_path / 'replacement'
    replacement.mkdir()
    (replacement / approved.name).write_text('outside', encoding='utf-8')
    root_identities: list[tuple[int, int, int]] = []

    def directory_identity(root: Path) -> tuple[int, int, int]:
        info = root.stat()
        identity = (info.st_dev, info.st_ino, 0)
        root_identities.append(identity)
        return identity

    monkeypatch.setattr(insights_launch, '_windows_directory_identity', directory_identity)

    def create_file(path: str, *_args: object) -> int:
        if path == str(approved):
            exports.rename(tmp_path / 'original-exports')
            replacement.rename(exports)
        return 123

    def get_info(_handle: object, _kind: int, info_pointer: object, _size: int) -> int:
        info = ctypes.cast(info_pointer, ctypes.POINTER(type_info)).contents
        info.attributes = 0
        return 1

    class type_info(ctypes.Structure):
        _fields_ = [('attributes', ctypes.c_uint32), ('reparse_tag', ctypes.c_uint32)]

    def get_final_path(_handle: object, buffer: object, _size: int, _flags: int) -> int:
        value = ctypes.create_unicode_buffer(str(approved))
        ctypes.memmove(buffer, value, ctypes.sizeof(value))
        return len(value.value)

    api = types.SimpleNamespace(
        CreateFileW=create_file,
        GetFileInformationByHandleEx=get_info,
        GetFinalPathNameByHandleW=get_final_path,
        CloseHandle=lambda _handle: 1,
    )
    monkeypatch.setattr(ctypes, 'WinDLL', lambda *_args, **_kwargs: api, raising=False)
    monkeypatch.setitem(
        __import__('sys').modules,
        'msvcrt',
        types.SimpleNamespace(open_osfhandle=lambda *_args: 456),
    )

    with pytest.raises(OSError, match='directory changed while opening'):
        insights_launch._open_windows_approved_regular_file(exports, approved)
    assert len(root_identities) == 3
    assert root_identities[0] == root_identities[1]
    assert root_identities[0] != root_identities[2]


def test_windows_finder_lease_read_rejects_reparse_point_before_path_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import ctypes
    import types

    from evaluatorq.dashboard import insights_launch

    lease_dir = tmp_path / 'finder-references'
    lease_dir.mkdir(mode=0o700)
    lease = lease_dir / 'run.json'
    lease.write_text('{"finder_export":"approved.json"}', encoding='utf-8')
    opened: list[str] = []

    def create_file(path: str, *_args: object) -> int:
        opened.append(path)
        return 123

    class type_info(ctypes.Structure):
        _fields_ = [('attributes', ctypes.c_uint32), ('reparse_tag', ctypes.c_uint32)]

    def get_info(_handle: object, _kind: int, info_pointer: object, _size: int) -> int:
        info = ctypes.cast(info_pointer, ctypes.POINTER(type_info)).contents
        info.attributes = 0x400
        return 1

    def unused(*_args: object) -> int:
        raise AssertionError('A reparse point must be rejected before resolving or reading its target.')

    api = types.SimpleNamespace(
        CreateFileW=create_file,
        GetFileInformationByHandleEx=get_info,
        GetFinalPathNameByHandleW=unused,
        CloseHandle=lambda _handle: 1,
    )
    monkeypatch.setattr(ctypes, 'WinDLL', lambda *_args, **_kwargs: api, raising=False)
    monkeypatch.setattr(insights_launch.os, 'name', 'nt')
    monkeypatch.setattr(insights_launch, '_windows_directory_identity', lambda _root: (1, 2, 3))
    monkeypatch.setitem(__import__('sys').modules, 'msvcrt', types.SimpleNamespace())
    monkeypatch.setattr(
        insights_launch.os,
        'open',
        lambda *_args, **_kwargs: pytest.fail('Windows lease reads must not use pathname os.open.'),
    )

    with pytest.raises(OSError, match='reparse point'):
        insights_launch.read_private_finder_reference(lease)
    assert opened == [str(lease.parent), str(lease)]


def test_finder_export_descriptor_closes_when_fdopen_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import os

    from evaluatorq.dashboard import insights_launch

    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    exports = tmp_path / 'finder-exports'
    exports.mkdir()
    (exports / 'approved.json').write_text(_run_export(['trace-1']).model_dump_json(), encoding='utf-8')
    opened_descriptors: list[int] = []

    def fail_fdopen(descriptor: int, _mode: str):
        opened_descriptors.append(descriptor)
        raise OSError('fdopen failed')

    monkeypatch.setattr(insights_launch.os, 'fdopen', fail_fdopen)

    with pytest.raises(ValidationError, match='Could not read a valid Finder export'):
        InsightsLaunchSpec(source='finder', finder_export='approved.json')

    assert len(opened_descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(opened_descriptors[0])


def test_dashboard_finder_export_is_size_limited(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    exports = tmp_path / 'finder-exports'
    exports.mkdir()
    oversized = exports / 'large.json'
    oversized.write_bytes(b' ' * (MAX_FINDER_EXPORT_BYTES + 1))

    with pytest.raises(ValidationError, match='exceeds the 10 MiB size limit'):
        InsightsLaunchSpec(source='finder', finder_export='large.json')


@pytest.mark.timeout(10)
def test_finder_export_fifo_is_rejected_without_blocking(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import os

    if not hasattr(os, 'mkfifo'):
        pytest.skip('FIFOs are unavailable on this platform')
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    exports = tmp_path / 'finder-exports'
    exports.mkdir()
    os.mkfifo(exports / 'pipe.json')

    with pytest.raises(ValidationError, match='regular file'):
        InsightsLaunchSpec(source='finder', finder_export='pipe.json')


def test_finder_insights_launch_records_source_until_run_finishes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import json

    import tempfile

    # Keep the private Finder snapshot directory inside tmp_path so it is removed with the test.
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
    monkeypatch.setattr('evaluatorq.dashboard.insights_launch._worker_process_identity', lambda _pid: 'test:1')
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    export_dir = tmp_path / 'finder-exports'
    export_dir.mkdir()
    export_path = export_dir / 'trace-finder-source.json'
    export_path.write_text(_run_export(['trace-1']).model_dump_json(), encoding='utf-8')
    spec = InsightsLaunchSpec(source='finder', finder_export=export_path.name)
    with patch('evaluatorq.dashboard.insights_launch.subprocess.Popen') as spawn:
        spawn.return_value.pid = 43210
        run_id = launch_insights(spec, tmp_path / 'insights-runs')

    reference = finder_export_reference_path(tmp_path / 'insights-runs', run_id)
    assert json.loads(reference.read_text(encoding='utf-8')) == {'finder_export': str(export_path)}
    if os.name != 'nt':
        assert reference.stat().st_mode & 0o777 == 0o600
        assert reference.parent.stat().st_mode & 0o777 == 0o700
    assert read_private_finder_reference(reference) == {'finder_export': str(export_path)}
    assert len(list_manifests(tmp_path / 'insights-runs')) == 1


@pytest.mark.skipif(os.name == 'nt', reason='chmod does not configure Windows file or directory ACLs')
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


def test_finder_reference_open_stays_in_validated_directory_if_path_is_swapped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import os
    import json

    from evaluatorq.dashboard import insights_launch

    if os.name == 'nt' or not hasattr(os, 'O_DIRECTORY') or not hasattr(os, 'O_NOFOLLOW'):
        pytest.skip('POSIX directory-relative open is unavailable')

    reference = finder_export_reference_path(tmp_path / 'runs', 'private-run')
    ensure_private_finder_reference_dir(reference.parent)
    reference.write_text(json.dumps({'source': 'approved'}), encoding='utf-8')
    reference.chmod(0o600)

    attacker = tmp_path / 'attacker'
    attacker.mkdir(mode=0o700)
    (attacker / reference.name).write_text(json.dumps({'source': 'attacker'}), encoding='utf-8')
    (attacker / reference.name).chmod(0o600)

    original_directory = reference.parent.with_name('leases-original')
    real_open = os.open
    swapped = False

    def swap_then_open(file: str | bytes | os.PathLike[str], flags: int, *args: object, **kwargs: object) -> int:
        nonlocal swapped
        if kwargs.get('dir_fd') is not None and file == reference.name and not swapped:
            reference.parent.rename(original_directory)
            reference.parent.symlink_to(attacker, target_is_directory=True)
            swapped = True
        return real_open(file, flags, *args, **kwargs)

    monkeypatch.setattr(insights_launch.os, 'open', swap_then_open)
    try:
        assert read_private_finder_reference(reference) == {'source': 'approved'}
        assert swapped
    finally:
        reference.parent.unlink()
        original_directory.rename(reference.parent)


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


def test_stale_reconciliation_recovers_parent_dying_before_worker_pid_is_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    from evaluatorq.dashboard import insights_launch
    from evaluatorq.dashboard.insights_launch import _write_worker_state, worker_state_path

    runs_dir = tmp_path / 'runs'
    writer = start_manifest(run_id='launcher-died', surface='insights', run_name='demo', runs_dir=runs_dir)
    _write_worker_state(
        worker_state_path(runs_dir, 'launcher-died'),
        {
            'pid': None,
            'launcher_pid': 12345,
            'launcher_process_identity': 'linux:100',
            'heartbeat_at': time.time() - 120,
            'snapshot_path': None,
        },
    )
    monkeypatch.setattr(insights_launch, '_worker_process_is_alive', lambda _pid, _identity=None: False)

    assert insights_launch.reconcile_stale_worker(runs_dir, 'launcher-died')
    assert list_manifests(runs_dir)[0].status.value == 'error'
    assert writer.manifest.status.value == 'running'


def test_stale_reconciliation_keeps_live_launcher_with_unpublished_worker_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    from evaluatorq.dashboard import insights_launch
    from evaluatorq.dashboard.insights_launch import _write_worker_state, worker_state_path

    runs_dir = tmp_path / 'runs'
    writer = start_manifest(run_id='launcher-still-starting', surface='insights', run_name='demo', runs_dir=runs_dir)
    _write_worker_state(
        worker_state_path(runs_dir, 'launcher-still-starting'),
        {
            'pid': None,
            'launcher_pid': 12345,
            'launcher_process_identity': 'linux:100',
            'heartbeat_at': time.time() - 120,
            'snapshot_path': None,
        },
    )
    monkeypatch.setattr(insights_launch, '_worker_process_is_alive', lambda _pid, _identity=None: True)

    assert not insights_launch.reconcile_stale_worker(runs_dir, 'launcher-still-starting')
    assert list_manifests(runs_dir)[0].status.value == 'running'
    assert writer.manifest.status.value == 'running'


def test_stale_reconciliation_recovers_reused_launcher_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    from evaluatorq.dashboard import insights_launch
    from evaluatorq.dashboard.insights_launch import _write_worker_state, worker_state_path

    runs_dir = tmp_path / 'runs'
    writer = start_manifest(run_id='reused-launcher-pid', surface='insights', run_name='demo', runs_dir=runs_dir)
    _write_worker_state(
        worker_state_path(runs_dir, 'reused-launcher-pid'),
        {
            'pid': None,
            'launcher_pid': 12345,
            'launcher_process_identity': 'darwin:original-start',
            'heartbeat_at': time.time() - 120,
            'snapshot_path': None,
        },
    )
    monkeypatch.setattr(insights_launch.os, 'kill', lambda _pid, _signal: None)
    monkeypatch.setattr(insights_launch, '_read_worker_process_identity', lambda _pid: ('darwin:reused-pid', False))

    assert insights_launch.reconcile_stale_worker(runs_dir, 'reused-launcher-pid')
    assert list_manifests(runs_dir)[0].status.value == 'error'
    assert writer.manifest.status.value == 'running'


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


def test_windows_worker_liveness_uses_read_only_process_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    import ctypes

    from evaluatorq.dashboard import insights_launch

    monkeypatch.setattr(insights_launch.sys, 'platform', 'win32')

    def unexpected_kill(_pid: int, _signal: int) -> None:
        raise AssertionError('Windows os.kill(pid, 0) terminates the worker')

    monkeypatch.setattr(insights_launch.os, 'kill', unexpected_kill)

    class ApiFunction:
        def __init__(self, call):
            self.call = call

        def __call__(self, *args):
            return self.call(*args)

    def process_times(_handle, created, *_rest):
        words = ctypes.cast(created, ctypes.POINTER(ctypes.c_uint32))
        words[0] = 1234
        words[1] = 5
        return 1

    kernel32 = type('Kernel32', (), {
        'OpenProcess': ApiFunction(lambda *_args: 1),
        'WaitForSingleObject': ApiFunction(lambda *_args: 258),
        'GetProcessTimes': ApiFunction(process_times),
        'CloseHandle': ApiFunction(lambda _handle: 1),
    })()
    monkeypatch.setattr(ctypes, 'WinDLL', lambda *_args, **_kwargs: kernel32, raising=False)

    identity = insights_launch._worker_process_identity(12345)
    assert identity == 'windows:21474837714'
    assert insights_launch._worker_process_is_alive(12345, identity)
    assert not insights_launch._worker_process_is_alive(12345, 'windows:9')


def test_windows_worker_liveness_recovers_exited_and_missing_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    import ctypes

    from evaluatorq.dashboard import insights_launch

    monkeypatch.setattr(insights_launch.sys, 'platform', 'win32')
    monkeypatch.setattr(insights_launch.os, 'kill', lambda *_args: pytest.fail('must not signal a Windows process'))

    class ApiFunction:
        def __init__(self, call):
            self.call = call

        def __call__(self, *args):
            return self.call(*args)

    kernel32 = type('Kernel32', (), {
        'OpenProcess': ApiFunction(lambda *_args: 1),
        'WaitForSingleObject': ApiFunction(lambda *_args: 0),
        'GetProcessTimes': ApiFunction(lambda *_args: pytest.fail('exited process has no start identity')),
        'CloseHandle': ApiFunction(lambda _handle: 1),
    })()
    monkeypatch.setattr(ctypes, 'WinDLL', lambda *_args, **_kwargs: kernel32, raising=False)
    assert not insights_launch._worker_process_is_alive(12345, 'windows:old')

    monkeypatch.setattr(insights_launch.ctypes, 'get_last_error', lambda: 87, raising=False)
    kernel32.OpenProcess = ApiFunction(lambda *_args: None)
    assert not insights_launch._worker_process_is_alive(12345, 'windows:old')


def test_windows_worker_liveness_keeps_run_when_process_api_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    import ctypes

    from evaluatorq.dashboard import insights_launch

    monkeypatch.setattr(insights_launch.sys, 'platform', 'win32')

    def unavailable(*_args, **_kwargs):
        raise OSError('kernel32 unavailable')

    monkeypatch.setattr(ctypes, 'WinDLL', unavailable, raising=False)
    assert insights_launch._worker_process_is_alive(12345, 'windows:old')


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
    from evaluatorq.trace_finder.settings import DashboardSettings

    monkeypatch.setattr('evaluatorq.dashboard.insights_launch._worker_process_identity', lambda _pid: 'test:1')
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    monkeypatch.setenv('ORQ_API_KEY', 'test-key')
    exports = tmp_path / 'finder-exports'
    exports.mkdir()
    export_path = exports / 'finder.json'
    approved_export = _run_export(['approved-trace']).model_dump_json()
    export_path.write_text(approved_export, encoding='utf-8')
    spec = InsightsLaunchSpec(
        source='finder',
        finder_export='finder.json',
        source_name='my-export.json',
        labels=[],
        coding_labels=['task_type', 'verified'],
        dimensions=['intent'],
    )

    export_path.unlink()
    export_path.symlink_to(tmp_path / 'outside.json')
    (tmp_path / 'outside.json').write_text(_run_export(['replacement-trace']).model_dump_json(), encoding='utf-8')

    with patch('evaluatorq.dashboard.insights_launch.subprocess.Popen') as spawn:
        spawn.return_value.pid = 43210
        run_id = launch_insights(spec, tmp_path / 'runs')
    reference = finder_export_reference_path(tmp_path / 'runs', run_id)
    assert reference.exists()
    payload_json = spawn.call_args.kwargs['env'][_REQUEST_ENV]
    monkeypatch.setenv(_REQUEST_ENV, payload_json)
    payload = read_launch_payload()
    assert payload.finder_export_snapshot is not None
    assert payload.finder_export_snapshot.read_text(encoding='utf-8') == approved_export

    replacement_export = _run_export(['replacement-after-validation']).model_dump_json()
    original_from_finder_export = InsightsPopulation.from_finder_export
    snapshot_replacement: list[str] = []

    def replace_snapshot_after_validation(cls, path: Path, *, export=None):
        path.write_text(replacement_export, encoding='utf-8')
        snapshot_replacement.append(path.read_text(encoding='utf-8'))
        return original_from_finder_export(path, export=export)

    monkeypatch.setattr(InsightsPopulation, 'from_finder_export', classmethod(replace_snapshot_after_validation))

    consumed: list[str] = []
    consumed_source: list[Path | None] = []
    forwarded: list[dict[str, object]] = []
    forwarded_specs: list[InsightsLaunchSpec] = []
    selected_auth_runner = insights_worker._run_with_selected_auth

    async def fake_insights(*_args, **kwargs):
        from types import SimpleNamespace

        assert kwargs['coding_labels'] == ['task_type', 'verified']
        return SimpleNamespace(status='completed')

    monkeypatch.setattr(insights_worker, 'effective_settings', lambda: DashboardSettings.model_validate({}))
    monkeypatch.setattr(
        insights_worker,
        'resolve_dashboard_auth',
        lambda _settings: DashboardAuth('environment', 'test-key', 'https://my.orq.ai'),
    )
    monkeypatch.setattr(insights_worker, 'build_auth_clients', lambda *_args, **_kwargs: (object(), object()))
    monkeypatch.setattr(insights_worker, 'close_orq_client', AsyncMock())
    monkeypatch.setattr(insights_worker, 'insights', fake_insights)

    async def capture_population(_payload, population, **kwargs):
        from evaluatorq.common.run_manifest import ManifestWriter
        from evaluatorq.contracts import RunManifest

        assert population.finder_export is not None
        snapshot = population.finder_export_snapshot()
        assert snapshot is not None
        consumed_source.append(kwargs.get('_finder_export_source'))
        forwarded.append(kwargs)
        forwarded_specs.append(_payload.spec)
        consumed.extend(snapshot.matched_trace_ids)
        assert await selected_auth_runner(_payload, population, **kwargs)
        manifest_path = tmp_path / 'runs' / '.manifests' / f'{payload.run_id}.json'
        manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding='utf-8'))
        ManifestWriter(manifest, manifest_path).complete()
        return True

    monkeypatch.setattr(insights_worker, '_run_with_selected_auth', capture_population)
    monkeypatch.setenv(_REQUEST_ENV, payload_json)
    assert insights_worker.main() == 0
    assert consumed == ['approved-trace']
    assert snapshot_replacement == [replacement_export]
    assert consumed_source == [export_path]
    assert forwarded[0]['_source_name'] == 'my-export.json'
    assert forwarded_specs[0].coding_labels == ['task_type', 'verified']
    assert not payload.finder_export_snapshot.exists()
    assert not reference.exists()
    assert not worker_state_path(tmp_path / 'runs', payload.run_id).exists()
    assert list_manifests(tmp_path / 'runs')[0].status == 'completed'


def test_finder_worker_rejects_snapshot_replaced_after_payload_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import hashlib

    from evaluatorq.dashboard import insights_worker
    from evaluatorq.dashboard.insights_launch import _REQUEST_ENV, read_launch_payload

    monkeypatch.setattr('evaluatorq.dashboard.insights_launch._worker_process_identity', lambda _pid: 'test:1')
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    exports = tmp_path / 'finder-exports'
    exports.mkdir()
    (exports / 'finder.json').write_text(_run_export(['approved-trace']).model_dump_json(), encoding='utf-8')
    spec = InsightsLaunchSpec(source='finder', finder_export='finder.json', labels=[], dimensions=['intent'])

    with patch('evaluatorq.dashboard.insights_launch.subprocess.Popen') as spawn:
        spawn.return_value.pid = 43210
        run_id = launch_insights(spec, tmp_path / 'runs')
    payload_json = spawn.call_args.kwargs['env'][_REQUEST_ENV]
    monkeypatch.setenv(_REQUEST_ENV, payload_json)
    payload = read_launch_payload()
    snapshot = payload.finder_export_snapshot
    assert snapshot is not None
    assert payload.finder_export_snapshot_sha256 == hashlib.sha256(snapshot.read_bytes()).hexdigest()

    outside = tmp_path / 'outside.json'
    outside.write_text(_run_export(['replacement-trace']).model_dump_json(), encoding='utf-8')
    outside.chmod(0o600)

    def replace_snapshot_after_validation():
        outside.replace(snapshot)
        return payload

    monkeypatch.setattr(insights_worker, 'read_launch_payload', replace_snapshot_after_validation)
    monkeypatch.setattr(insights_worker, 'insights', lambda *_args, **_kwargs: pytest.fail('unsafe snapshot was used'))

    assert insights_worker.main() == 1
    manifest = list_manifests(tmp_path / 'runs')[0]
    assert manifest.status == 'error'
    assert manifest.error is not None and 'snapshot changed after launch' in manifest.error
    assert not outside.exists()
    assert not snapshot.exists()


def test_windows_snapshot_read_checks_directory_identity_after_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os
    import tempfile
    from types import SimpleNamespace

    from evaluatorq.dashboard import insights_worker

    snapshot_directory = Path(tempfile.mkdtemp(prefix='evaluatorq-finder-snapshot-'))
    snapshot = snapshot_directory / 'finder-export.json'
    snapshot.write_text('{}', encoding='utf-8')
    snapshot.chmod(0o600)
    file_descriptor = os.open(snapshot, os.O_RDONLY)
    identities = iter(((1, 2, 3), (4, 5, 6)))

    def open_approved_snapshot(*_args, **_kwargs):
        return file_descriptor, next(identities)

    monkeypatch.setattr(
        insights_worker,
        'os',
        SimpleNamespace(name='nt', O_RDONLY=os.O_RDONLY, fstat=os.fstat, fdopen=os.fdopen, close=os.close),
    )
    monkeypatch.setattr(
        insights_worker,
        '_open_windows_approved_regular_file',
        open_approved_snapshot,
    )
    monkeypatch.setattr(insights_worker, '_windows_directory_identity', lambda _path: next(identities))

    with pytest.raises(ValueError, match='directory changed while reading'):
        insights_worker._read_private_snapshot(snapshot)
    snapshot.unlink()
    snapshot_directory.rmdir()


def test_finder_launch_plan_uses_validated_export_snapshot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from evaluatorq.dashboard import insights_launch

    import tempfile

    # Keep the private Finder snapshot directory inside tmp_path so it is removed with the test.
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
    monkeypatch.setattr(insights_launch, '_worker_process_identity', lambda _pid: 'test:1')
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    exports = tmp_path / 'finder-exports'
    exports.mkdir()
    approved_path = exports / 'approved.json'
    approved_path.write_text(_run_export(['approved-trace']).model_dump_json(), encoding='utf-8')
    replacement_path = exports / 'replacement.json'
    replacement_path.write_text(_run_export(['replacement-trace']).model_dump_json(), encoding='utf-8')
    spec = InsightsLaunchSpec(source='finder', finder_export='approved.json', labels=[], dimensions=['intent'])
    spec.finder_export = str(replacement_path)

    populations: list[InsightsPopulation] = []
    original_stage_plan = insights_launch.stage_plan

    def capture_plan(population, labels, dimensions):
        populations.append(population)
        return original_stage_plan(population, labels, dimensions)

    monkeypatch.setattr(insights_launch, 'stage_plan', capture_plan)
    with patch('evaluatorq.dashboard.insights_launch.subprocess.Popen') as spawn:
        spawn.return_value.pid = 43210
        launch_insights(spec, tmp_path / 'runs')

    assert len(populations) == 1
    snapshot = populations[0].finder_export_snapshot()
    assert snapshot is not None
    assert snapshot.matched_trace_ids == ['approved-trace']


def test_finder_snapshot_cleanup_without_getuid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import ctypes
    from types import SimpleNamespace

    from evaluatorq.dashboard import insights_worker

    directory = tmp_path / 'evaluatorq-finder-snapshot-test'
    directory.mkdir(mode=0o700)
    snapshot = directory / 'finder-export.json'
    snapshot.write_text('{}', encoding='utf-8')
    snapshot.chmod(0o600)
    monkeypatch.setattr(insights_worker.tempfile, 'gettempdir', lambda: str(tmp_path))
    deleted = False
    monkeypatch.setattr(
        insights_worker,
        'os',
        SimpleNamespace(name='nt', listdir=lambda _path: [] if deleted else [snapshot.name]),
    )
    monkeypatch.setattr(
        insights_worker,
        '_windows_directory_identity',
        lambda _path: (1, 0, 3),
    )

    handles = {str(directory): 101, str(snapshot): 102}
    handle_paths = {handle: path for path, handle in handles.items()}
    create_calls: list[tuple[str, int, int, int]] = []
    deleted_handles: list[int] = []
    closed_handles: list[int] = []

    def create_file(path, access, share, _security, _creation, flags, _template):
        create_calls.append((path, access, share, flags))
        return handles[path]

    def get_file_info(handle, info_pointer):
        info = info_pointer._obj
        info.attributes = 0x10 if handle == handles[str(directory)] else 0
        info.volume_serial = 1 if handle == handles[str(directory)] else 9
        info.index_high = 0
        info.index_low = 3 if handle == handles[str(directory)] else 4
        return 1

    def get_final_path(handle, buffer, _length, _flags):
        value = handle_paths[handle]
        buffer.value = value
        return len(value)

    def set_file_info(handle, _kind, info_pointer, _size):
        nonlocal deleted
        assert info_pointer._obj.delete_file == 1
        deleted_handles.append(handle)
        if handle == handles[str(snapshot)]:
            deleted = True
            snapshot.unlink()
        return 1

    def close_handle(handle):
        closed_handles.append(handle)
        return 1

    kernel32 = SimpleNamespace(
        CreateFileW=create_file,
        GetFileInformationByHandle=get_file_info,
        GetFinalPathNameByHandleW=get_final_path,
        SetFileInformationByHandle=set_file_info,
        CloseHandle=close_handle,
    )
    monkeypatch.setattr(ctypes, 'WinDLL', lambda *_args, **_kwargs: kernel32, raising=False)

    insights_worker._cleanup_snapshot(snapshot)

    assert [call[0] for call in create_calls] == [str(directory), str(snapshot)]
    assert all(call[1] == 0x10080 and call[2] == 0x3 for call in create_calls)
    assert create_calls[0][3] == 0x00200000 | 0x02000000
    assert create_calls[1][3] == 0x00200000
    assert deleted_handles == [handles[str(snapshot)], handles[str(directory)]]
    assert closed_handles == [handles[str(snapshot)], handles[str(directory)]]
    if snapshot.exists():
        snapshot.unlink()
    directory.rmdir()


@pytest.mark.skipif(os.name == 'nt', reason='POSIX descriptor-relative cleanup')
def test_snapshot_cleanup_does_not_follow_directory_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evaluatorq.dashboard import insights_worker

    temp_root = tmp_path / 'tmp'
    temp_root.mkdir()
    directory = temp_root / 'evaluatorq-finder-snapshot-test'
    directory.mkdir(mode=0o700)
    snapshot = directory / 'finder-export.json'
    snapshot.write_text('{}', encoding='utf-8')
    snapshot.chmod(0o600)
    replacement = temp_root / 'replacement'
    replacement.mkdir()
    protected = replacement / 'keep.txt'
    protected.write_text('keep', encoding='utf-8')
    monkeypatch.setattr(insights_worker.tempfile, 'gettempdir', lambda: str(temp_root))

    original_open = insights_worker.os.open
    swapped = False

    def swap_after_directory_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if path == directory.name and dir_fd is not None and not swapped:
            swapped = True
            directory.rename(temp_root / 'original-snapshot')
            replacement.rename(directory)
        return descriptor

    monkeypatch.setattr(insights_worker.os, 'open', swap_after_directory_open)
    insights_worker._cleanup_snapshot(snapshot)

    assert swapped
    assert (directory / 'keep.txt').read_text(encoding='utf-8') == 'keep'
    assert directory.exists()
    assert (temp_root / 'original-snapshot').exists()


def test_live_facet_selection_reaches_population_and_worker(tmp_path: Path) -> None:
    facets = FacetSelection(status=frozenset({'error'}), provider=frozenset({'openai', 'anthropic'}))
    spec = InsightsLaunchSpec(source='query', query='refunds', window_days=3, facets=facets)
    assert spec.population().facets == facets
    assert spec.population().window_days == 3
    with (
        patch('evaluatorq.dashboard.insights_launch.subprocess.Popen') as spawn,
        patch('evaluatorq.dashboard.insights_launch._worker_process_identity', return_value='test:1'),
    ):
        spawn.return_value.pid = 43210
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
    monkeypatch.setattr('evaluatorq.dashboard.insights_launch._worker_process_identity', lambda _pid: 'test:1')
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
        spawn.return_value.pid = 43210
        assert client.get('/insights/facets?window_days=7').status_code == 200
        response = client.post(
            '/insights/runs',
            data={'csrf': token.group(1), 'source': 'recent', 'dimensions': 'intent'},
            follow_redirects=False,
        )

    assert response.status_code == 303
    resolve.assert_called_once_with('profile-key', base_url=expected_host)
    worker_env = spawn.call_args.kwargs['env']
    assert worker_env['ORQ_API_KEY'] == 'environment-key'
    assert worker_env['ORQ_BASE_URL'] == 'https://environment.example'
    assert 'profile-key' not in worker_env['EVALUATORQ_INSIGHTS_LAUNCH_REQUEST']
    assert InsightsLaunchPayload.model_validate_json(worker_env['EVALUATORQ_INSIGHTS_LAUNCH_REQUEST']).auth_method == 'cli_profile'
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

    assert 'Could not load filter choices from Orq' in facets.text
    assert '<a href="/settings" target="_blank" rel="noopener">Settings → Authentication</a>' in facets.text
    assert 'data-retry-facets' in facets.text
    assert start.status_code == 422
    assert 'Orq profile deleted is unavailable' in start.text
    resolve.assert_not_called()
    launch.assert_not_called()


def test_reloaded_selected_profile_is_used_and_named_when_orq_rejects_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    settings_path = tmp_path / 'dashboard-settings.json'
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(settings_path))
    from evaluatorq.dashboard.trace_finder import routes as finder_routes
    from evaluatorq.trace_finder.settings import save_settings

    profile = OrqProfile('research', 'profile-secret-for-test', 'https://profile.example', False)
    save_settings(DashboardSettings.model_validate({'orq_profile': profile.name}), settings_path)
    monkeypatch.setattr(finder_routes, 'list_orq_profiles', lambda: (profile,))

    class Unauthorized(Exception):
        status_code = 401

    with (
        patch('evaluatorq.dashboard.insights_routes.resolve_orq_client', return_value=object()) as resolve,
        patch(
            'evaluatorq.dashboard.insights_routes.load_facet_catalogue',
            new_callable=AsyncMock,
            side_effect=Unauthorized('private provider response'),
        ),
        patch('evaluatorq.dashboard.insights_routes.close_orq_client', new_callable=AsyncMock),
    ):
        response = TestClient(build_app()).get('/insights/facets?window_days=7')

    assert resolve.call_args.args == ('profile-secret-for-test',)
    assert resolve.call_args.kwargs == {'base_url': 'https://profile.example'}
    assert 'profile <code>research</code>' in response.text
    assert 'Orq rejected the key' in response.text
    assert 'workspace selected in Settings' in response.text
    assert 'profile-secret-for-test' not in response.text
    assert 'private provider response' not in response.text


def test_facet_auth_error_names_oauth_credential() -> None:
    from evaluatorq.dashboard.insights_run_form import facet_options

    html = facet_options(None, FacetSelection(), profile_name='CLI OAuth', credential_rejected=True)

    assert 'CLI OAuth sign-in' in html
    assert 'orq auth login' in html
    assert 'key for profile' not in html


def test_facet_options_renders_removable_chips_for_selected_values() -> None:
    from evaluatorq.dashboard.insights_run_form import facet_options

    html = facet_options(
        FacetCatalogue(agent_name=('a', 'b'), value_counts={'agent_name': {'a': 7, 'b': 2}}),
        FacetSelection(agent_name=frozenset({'a'})),
    )

    selected = html[html.index('class="insights-selected-facets"') :]
    assert 'data-chip-name="facet_agent_name" data-finder-value="a"' in selected
    assert 'value="a" checked><span>a</span><span class="facet-n"' in html
    assert '>7</span>' in html
    assert 'hx-get="/find/facets' not in html


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
        results = await asyncio.gather(first, second)
        assert [result[0] for result in results] == [catalogue, catalogue]

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
        assert (await stale)[0] is None

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
    assert 'class="finder-facets"' in response.text
    assert 'class="facet-item" data-facet="status"' in response.text
    assert 'class="facet-search"' in response.text
    assert 'form="insights-new-form" type="checkbox" name="facet_status"' in response.text
    assert 'name="tokens_min"' not in response.text
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
    assert 'Could not load filter choices from Orq' in response.text
    assert 'data-retry-facets' in response.text
    assert 'name="facet_status" value="error" checked' in response.text


def test_facet_retry_bypasses_failed_cache_and_preserves_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    client = TestClient(build_app())
    load = AsyncMock(side_effect=[RuntimeError('Orq unavailable'), FacetCatalogue(status=('ok',))])
    with (
        patch('evaluatorq.dashboard.insights_routes.resolve_orq_client', return_value=object()),
        patch('evaluatorq.dashboard.insights_routes.load_facet_catalogue', load),
        patch('evaluatorq.dashboard.insights_routes.close_orq_client', new_callable=AsyncMock),
    ):
        failed = client.get('/insights/facets?window_days=7&facet_status=error')
        cached = client.get('/insights/facets?window_days=7&facet_status=error')
        retried = client.get('/insights/facets?window_days=7&facet_status=error&retry=1')

    assert 'data-retry-facets' in failed.text
    assert 'data-retry-facets' in cached.text
    assert load.await_count == 2
    assert 'data-retry-facets' not in retried.text
    assert 'name="facet_status" value="error" checked' in retried.text
    assert 'name="facet_status" value="ok"' in retried.text


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


def test_chosen_models_reach_insights_and_blank_ones_keep_the_pipeline_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from evaluatorq.dashboard import insights_worker

    seen: dict[str, object] = {}

    async def fake_insights(population: object, **kwargs: object) -> object:
        seen.update(kwargs)
        return type('Run', (), {'status': 'completed'})()

    settings = DashboardSettings.model_validate({'orq_auth_method': 'environment'})
    monkeypatch.setattr(insights_worker, 'effective_settings', lambda: settings)
    monkeypatch.setattr(insights_worker, 'resolve_dashboard_auth', lambda _settings: DashboardAuth('environment', 'k', 'https://my.orq.ai'))
    monkeypatch.setattr(insights_worker, 'build_auth_clients', lambda *_a, **_k: (object(), type('C', (), {'close': AsyncMock()})()))
    monkeypatch.setattr(insights_worker, 'close_orq_client', AsyncMock())
    monkeypatch.setattr(insights_worker, 'insights', fake_insights)
    spec = InsightsLaunchSpec(summary_model=' acme/summary ', embedding_model='acme/embed', classifier_model='  ')
    payload = InsightsLaunchPayload(run_id='r', run_name='n', runs_dir=tmp_path, spec=spec)

    assert asyncio.run(insights_worker._run_with_selected_auth(payload, InsightsPopulation())) is True

    assert seen['summary_model'] == 'acme/summary'
    assert seen['embedding_model'] == 'acme/embed'
    assert 'classifier_model' not in seen
    assert 'compiler_model' not in seen
