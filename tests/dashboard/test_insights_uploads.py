"""Validated upload handling for Insights source files."""

from __future__ import annotations

from datetime import datetime
import json
import os
import re
from pathlib import Path
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from evaluatorq.dashboard.app import build_app
from evaluatorq.trace_finder.models import Snapshot
from tests.insights.test_population import _run_export, make_trace


def _token(client: TestClient) -> str:
    page = client.get('/insights/new')
    match = re.search(r'name="csrf" value="([^"]+)"', page.text)
    assert match is not None
    return match.group(1)


def test_upload_validates_and_stores_finder_export_under_runs_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    client = TestClient(build_app())

    response = client.post(
        '/insights/uploads',
        data={'csrf': _token(client)},
        files={'file': ('export.json', _run_export(['trace-1']).model_dump_json(), 'application/json')},
    )

    assert response.status_code == 201
    stored = Path(response.json()['path'])
    assert stored.parent == tmp_path / 'insights-runs' / '.uploads'
    assert stored.exists()
    if os.name == 'posix':
        assert stored.stat().st_mode & 0o777 == 0o600
    assert json.loads(stored.read_text())['schema_version'] == 2


def test_upload_rejects_bad_csrf_type_and_invalid_content(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    client = TestClient(build_app())
    token = _token(client)

    assert (
        client.post('/insights/uploads', data={'kind': 'snapshot'}, files={'file': ('x.json', '{}')}).status_code == 403
    )
    for body in ('{}', '[1]', 'not json', '{"traces": []}'):
        rejected = client.post('/insights/uploads', data={'csrf': token}, files={'file': ('x.json', body)})
        assert rejected.status_code == 422, body
    assert 'neither a Finder export nor a trace snapshot' in client.post(
        '/insights/uploads', data={'csrf': token}, files={'file': ('x.json', '{}')}
    ).json()['error']
    assert not list((tmp_path / 'insights-runs' / '.uploads').glob('*'))


def test_upload_rejects_oversized_files_before_storing(tmp_path: Path, monkeypatch) -> None:
    from evaluatorq.dashboard.insights_uploads import MAX_INSIGHTS_UPLOAD_BYTES

    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    client = TestClient(build_app())
    token = _token(client)
    body = b' ' * (MAX_INSIGHTS_UPLOAD_BYTES + 1)

    response = client.post(
        '/insights/uploads', data={'csrf': token}, files={'file': ('large.json', body)}
    )

    assert response.status_code == 413
    assert not list((tmp_path / 'insights-runs' / '.uploads').glob('*'))


def test_finder_upload_rejects_10_mib_plus_one_while_snapshot_keeps_100_mib_limit(
    tmp_path: Path, monkeypatch
) -> None:
    from evaluatorq.dashboard.insights_uploads import MAX_FINDER_EXPORT_BYTES, store_upload

    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    client = TestClient(build_app())
    token = _token(client)
    export = _run_export(['trace-1']).model_dump_json().encode()
    export += b' ' * (MAX_FINDER_EXPORT_BYTES + 1 - len(export))

    response = client.post(
        '/insights/uploads', data={'csrf': token}, files={'file': ('large.json', export)}
    )

    assert response.status_code == 413
    assert '10 MiB size limit' in response.json()['error']
    assert not list((tmp_path / 'insights-runs' / '.uploads').glob('*'))

    snapshot = Snapshot(traces=(make_trace('trace-1'),)).model_dump_json().encode()
    snapshot += b' ' * (MAX_FINDER_EXPORT_BYTES + 1 - len(snapshot))
    stored = store_upload(tmp_path, snapshot, 'snapshot')
    assert stored.read_bytes() == snapshot


def test_upload_detects_kind_from_content_and_ignores_client_kind(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    client = TestClient(build_app())
    token = _token(client)
    export = _run_export(['trace-1']).model_dump_json()
    snapshot = Snapshot(traces=(make_trace('trace-1'),)).model_dump_json()

    for label, body, expected in (
        (None, export, 'finder'),
        (None, snapshot, 'snapshot'),
        ('snapshot', export, 'finder'),
        ('finder', snapshot, 'snapshot'),
        ('bogus', export, 'finder'),
    ):
        data = {'csrf': token} if label is None else {'csrf': token, 'kind': label}
        response = client.post('/insights/uploads', data=data, files={'file': ('f.json', body)})
        assert response.status_code == 201, (label, response.text)
        assert response.json()['kind'] == expected
        assert Path(response.json()['path']).name.startswith(f'{expected}-')


def test_upload_rejects_empty_snapshot(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    client = TestClient(build_app())
    empty = Snapshot(traces=()).model_dump_json()
    response = client.post('/insights/uploads', data={'csrf': _token(client)}, files={'file': ('f.json', empty)})
    assert response.status_code == 422
    assert not list((tmp_path / 'insights-runs' / '.uploads').glob('*'))


@pytest.mark.anyio
async def test_request_body_limit_stops_consuming_oversized_stream() -> None:
    import asyncio

    from evaluatorq.dashboard.insights_uploads import (
        MAX_INSIGHTS_UPLOAD_BYTES,
        UploadRequestTooLargeError,
        limit_request_body,
    )

    chunks = [b'x' * (MAX_INSIGHTS_UPLOAD_BYTES // 2)] * 3
    calls = 0

    async def receive():
        nonlocal calls
        await asyncio.sleep(0)
        calls += 1
        if calls <= len(chunks):
            return {'type': 'http.request', 'body': chunks[calls - 1], 'more_body': True}
        return {'type': 'http.request', 'body': b'', 'more_body': False}

    class RequestStub:
        _receive = staticmethod(receive)

    request = RequestStub()
    limit_request_body(request)

    async def consume_until_rejected() -> None:
        wrapped_receive = getattr(request, '_receive')
        await wrapped_receive()
        await wrapped_receive()
        await wrapped_receive()

    with pytest.raises(UploadRequestTooLargeError):
        await consume_until_rejected()
    assert calls == 3


def test_expired_retained_uploads_and_metadata_are_removed_safely(tmp_path: Path) -> None:
    import os

    from evaluatorq.dashboard.insights_uploads import (
        UPLOAD_RETENTION_SECONDS,
        cleanup_expired_uploads,
        store_upload,
        uploads_dir,
    )

    runs_dir = tmp_path / 'runs'
    live = store_upload(runs_dir, Snapshot(traces=(make_trace('live'),)).model_dump_json().encode(), 'snapshot')
    stale = store_upload(runs_dir, Snapshot(traces=(make_trace('stale'),)).model_dump_json().encode(), 'snapshot')
    os.utime(stale, (100, 100))
    unrelated = uploads_dir(runs_dir) / 'keep-me.txt'
    unrelated.write_text('keep')

    assert cleanup_expired_uploads(runs_dir, now=100 + UPLOAD_RETENTION_SECONDS + 1) == 1
    assert live.exists()
    assert not stale.exists()
    assert not stale.with_name(stale.name + '.meta.json').exists()
    assert live.with_name(live.name + '.meta.json').exists()
    assert unrelated.exists()


def test_uploaded_snapshot_path_is_revalidated_if_removed_before_worker_opens_it(tmp_path: Path) -> None:
    from evaluatorq.dashboard.insights_launch import InsightsLaunchSpec
    from evaluatorq.dashboard.insights_uploads import store_upload

    path = store_upload(tmp_path, Snapshot(traces=(make_trace('one'),)).model_dump_json().encode(), 'snapshot')
    InsightsLaunchSpec(source='snapshot', snapshot_path=str(path))
    path.unlink()

    with pytest.raises(Exception, match='Could not read a valid local trace snapshot'):
        InsightsLaunchSpec(source='snapshot', snapshot_path=str(path))


def test_worker_retains_route_upload_after_success_and_failure(tmp_path: Path, monkeypatch) -> None:
    from evaluatorq.dashboard.insights_launch import InsightsLaunchPayload, InsightsLaunchSpec
    from evaluatorq.dashboard.insights_uploads import store_upload
    from evaluatorq.dashboard.insights_worker import main

    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    monkeypatch.setenv('ORQ_API_KEY', 'test-key')
    runs_dir = tmp_path / 'runs'
    source = store_upload(runs_dir, Snapshot(traces=(make_trace('one'),)).model_dump_json().encode(), 'snapshot')
    spec = InsightsLaunchSpec(source='snapshot', snapshot_path=str(source))
    payload = InsightsLaunchPayload(run_id='run-1', run_name='test', runs_dir=runs_dir, spec=spec)
    monkeypatch.delenv('EVALUATORQ_INSIGHTS_MANIFEST', raising=False)
    with (
        patch('evaluatorq.dashboard.insights_worker.read_launch_payload', return_value=payload),
        patch('evaluatorq.dashboard.insights_worker.insights', side_effect=_completed_run),
    ):
        assert main() == 0
    assert source.exists()
    assert source.with_name(source.name + '.meta.json').exists()

    source = store_upload(runs_dir, Snapshot(traces=(make_trace('two'),)).model_dump_json().encode(), 'snapshot')
    spec = InsightsLaunchSpec(source='snapshot', snapshot_path=str(source))
    payload = InsightsLaunchPayload(run_id='run-2', run_name='test', runs_dir=runs_dir, spec=spec)
    with (
        patch('evaluatorq.dashboard.insights_worker.read_launch_payload', return_value=payload),
        patch('evaluatorq.dashboard.insights_worker.insights', side_effect=_failed_run),
    ):
        assert main() == 1
    assert source.exists()
    assert source.with_name(source.name + '.meta.json').exists()


async def _completed_run(*args, **kwargs):
    return type('Run', (), {'status': 'completed'})()


async def _failed_run(*args, **kwargs):
    raise RuntimeError('pipeline failed')


def test_recent_files_lists_original_names_counts_kinds_and_newest_first(tmp_path: Path, monkeypatch) -> None:
    import time

    from evaluatorq.dashboard.insights_launch import get_finder_exports_dir
    from evaluatorq.dashboard.insights_uploads import store_upload
    from evaluatorq.insights.store import get_insights_runs_dir

    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    runs_dir = get_insights_runs_dir()
    snapshot = store_upload(
        runs_dir=runs_dir, contents=Snapshot(traces=(make_trace('one'), make_trace('two'))).model_dump_json().encode(),
        kind='snapshot', display_name='sessions.json',
    )
    uploaded_finder = store_upload(
        runs_dir=runs_dir, contents=_run_export(['one']).model_dump_json().encode(),
        kind='finder', display_name='upload-export.json',
    )
    finder_dir = get_finder_exports_dir()
    finder_dir.mkdir(parents=True, exist_ok=True)
    finder = finder_dir / 'finder-export.json'
    finder.write_text(_run_export(['a', 'b', 'c']).model_dump_json())
    now = time.time()
    for path, age in ((snapshot, 300), (uploaded_finder, 200), (finder, 100)):
        modified = now - age
        os.utime(path, (modified, modified))
        if path in (snapshot, uploaded_finder):
            metadata_path = path.with_name(path.name + '.meta.json')
            metadata = json.loads(metadata_path.read_text())
            metadata['modified_at'] = datetime.fromtimestamp(modified).astimezone().isoformat()
            metadata_path.write_text(json.dumps(metadata))

    client = TestClient(build_app())
    page = client.get('/insights/files')
    assert page.status_code == 200
    assert page.text.index('finder-export.json') < page.text.index('upload-export.json') < page.text.index('sessions.json')
    assert f'data-path="{snapshot}"' in page.text
    assert 'Trace snapshot' in page.text and 'Finder export' in page.text
    assert 'data-label="Traces">2</td>' in page.text and 'data-label="Traces">3</td>' in page.text
    assert 'UTC' in page.text
    assert not re.search(r'data-display-name="snapshot-[a-f0-9]{32}', page.text)
    assert client.get('/insights/new').text.count('data-recent-files') == 1


def test_upload_persists_private_display_metadata(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    client = TestClient(build_app())
    response = client.post(
        '/insights/uploads', data={'csrf': _token(client)},
        files={'file': ('C:\\fakepath\\<session>.json', Snapshot(traces=(make_trace('one'),)).model_dump_json())},
    )
    assert response.status_code == 201
    stored = Path(response.json()['path'])
    metadata = stored.with_name(stored.name + '.meta.json')
    metadata_data = json.loads(metadata.read_text())
    assert metadata_data['display_name'] == '<session>.json'
    assert metadata_data['trace_count'] == 1
    assert metadata_data['kind'] == 'snapshot'
    assert metadata_data['modified_at']
    if os.name == 'posix':
        assert metadata.stat().st_mode & 0o777 == 0o600
        assert stored.parent.stat().st_mode & 0o777 == 0o700
    files = client.get('/insights/files').text
    assert '&lt;session&gt;.json' in files
    assert 'fakepath' not in files


def test_recent_files_prunes_expired_uploads_but_not_finder_exports(tmp_path: Path, monkeypatch) -> None:
    import time

    from evaluatorq.dashboard.insights_launch import get_finder_exports_dir
    from evaluatorq.dashboard.insights_uploads import UPLOAD_RETENTION_SECONDS, store_upload
    from evaluatorq.insights.store import get_insights_runs_dir

    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    upload = store_upload(
        runs_dir=get_insights_runs_dir(), contents=Snapshot(traces=(make_trace('one'),)).model_dump_json().encode(),
        kind='snapshot', display_name='expired.json',
    )
    finder_dir = get_finder_exports_dir()
    finder_dir.mkdir(parents=True, exist_ok=True)
    finder = finder_dir / 'older-finder.json'
    finder.write_text(_run_export(['one']).model_dump_json())
    old = time.time() - UPLOAD_RETENTION_SECONDS - 1
    for path in (upload, finder):
        os.utime(path, (old, old))
    page = TestClient(build_app()).get('/insights/files')
    assert 'expired.json' not in page.text
    assert 'older-finder.json' in page.text
    assert not upload.exists()
    assert not upload.with_name(upload.name + '.meta.json').exists()
    assert finder.exists()


def test_worker_retention_extends_seven_days_from_run_finish(tmp_path: Path) -> None:
    import time

    from evaluatorq.dashboard.insights_uploads import UPLOAD_RETENTION_SECONDS, cleanup_expired_uploads, store_upload
    from evaluatorq.dashboard.insights_worker import _retain_consumed_upload

    source = store_upload(
        runs_dir=tmp_path, contents=Snapshot(traces=(make_trace('one'),)).model_dump_json().encode(), kind='snapshot',
    )
    old = time.time() - UPLOAD_RETENTION_SECONDS + 10
    os.utime(source, (old, old))
    _retain_consumed_upload(tmp_path, 'snapshot', '', str(source))
    finished = source.stat().st_mtime
    assert finished > old
    metadata = json.loads(source.with_name(source.name + '.meta.json').read_text())
    assert datetime.fromisoformat(metadata['modified_at'].replace('Z', '+00:00')).timestamp() == pytest.approx(finished, abs=1)
    assert cleanup_expired_uploads(tmp_path, now=finished + UPLOAD_RETENTION_SECONDS + 1) == 1


def test_recent_files_skip_symlinks_unsafe_metadata_and_foreign_paths(tmp_path: Path, monkeypatch) -> None:
    from evaluatorq.dashboard.insights_launch import get_finder_exports_dir
    from evaluatorq.dashboard.insights_uploads import read_uploaded_source, store_upload, uploads_dir
    from evaluatorq.insights.store import get_insights_runs_dir

    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    runs_dir = get_insights_runs_dir()
    contents = Snapshot(traces=(make_trace('one'),)).model_dump_json().encode()
    source = store_upload(runs_dir=runs_dir, contents=contents, kind='snapshot', display_name='hidden.json')
    outside = tmp_path / 'outside.json'
    outside.write_bytes(contents)
    symlink = uploads_dir(runs_dir) / ('snapshot-' + 'a' * 32 + '.json')
    try:
        symlink.symlink_to(outside)
    except OSError:
        pytest.skip('symlink creation unavailable')
    metadata = source.with_name(source.name + '.meta.json')
    metadata.unlink()
    metadata.symlink_to(outside)
    finder_dir = get_finder_exports_dir()
    finder_dir.mkdir(parents=True, exist_ok=True)
    (finder_dir / 'linked.json').symlink_to(outside)
    page = TestClient(build_app()).get('/insights/files')
    assert 'data-recent-file ' not in page.text
    assert 'No recent trace files' in page.text
    with pytest.raises(ValueError):
        read_uploaded_source(runs_dir=runs_dir, path=symlink, kind='snapshot')
    with pytest.raises(ValueError):
        read_uploaded_source(runs_dir=runs_dir, path=outside, kind='snapshot')


@pytest.mark.skipif(os.name != 'posix', reason='POSIX permission checks')
def test_recent_files_skip_nonprivate_uploads(tmp_path: Path, monkeypatch) -> None:
    from evaluatorq.dashboard.insights_uploads import store_upload
    from evaluatorq.insights.store import get_insights_runs_dir

    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    source = store_upload(
        runs_dir=get_insights_runs_dir(), contents=Snapshot(traces=(make_trace('one'),)).model_dump_json().encode(),
        kind='snapshot', display_name='unsafe.json',
    )
    source.chmod(0o644)
    assert 'unsafe.json' not in TestClient(build_app()).get('/insights/files').text


def test_uploaded_snapshot_preview_rejects_replaced_symlink(tmp_path: Path, monkeypatch) -> None:
    from evaluatorq.dashboard.insights_uploads import store_upload
    from evaluatorq.insights.store import get_insights_runs_dir

    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    contents = Snapshot(traces=(make_trace('one'),)).model_dump_json().encode()
    source = store_upload(runs_dir=get_insights_runs_dir(), contents=contents, kind='snapshot')
    outside = tmp_path / 'outside.json'
    outside.write_bytes(contents)
    source.unlink()
    try:
        source.symlink_to(outside)
    except OSError:
        pytest.skip('symlink creation unavailable')
    client = TestClient(build_app())
    preview = client.post('/insights/snapshot-preview', data={'csrf': _token(client), 'snapshot_path': str(source)})
    assert preview.status_code == 422

def test_finder_preview_accepts_retained_upload_and_approved_export_only(tmp_path: Path, monkeypatch) -> None:
    from evaluatorq.dashboard.insights_launch import get_finder_exports_dir
    from evaluatorq.dashboard.insights_uploads import store_upload
    from evaluatorq.insights.store import get_insights_runs_dir

    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    client = TestClient(build_app())
    token = _token(client)
    raw = _run_export(['one', 'two']).model_dump_json().encode()
    uploaded = store_upload(
        runs_dir=get_insights_runs_dir(), contents=raw, kind='finder', display_name='retained.json'
    )
    export_dir = get_finder_exports_dir()
    export_dir.mkdir(parents=True, exist_ok=True)
    approved = export_dir / 'saved.json'
    approved.write_bytes(raw)
    outside = tmp_path / 'outside.json'
    outside.write_bytes(raw)

    for path in (uploaded, approved):
        preview = client.post('/insights/finder-preview', data={'csrf': token, 'finder_path': str(path)})
        assert preview.status_code == 200
        assert '2 matched traces' in preview.text
    relative = client.post('/insights/finder-preview', data={'csrf': token, 'finder_path': 'saved.json'})
    assert relative.status_code == 200
    assert '2 matched traces' in relative.text
    rejected = client.post('/insights/finder-preview', data={'csrf': token, 'finder_path': str(outside)})
    assert rejected.status_code == 422
    linked = export_dir / 'linked.json'
    try:
        linked.symlink_to(outside)
    except OSError:
        pytest.skip('symlink creation unavailable')
    rejected_link = client.post('/insights/finder-preview', data={'csrf': token, 'finder_path': str(linked)})
    assert rejected_link.status_code == 422


