"""The Insights run form's Local sessions picker: server-side search and freezing a selection into a snapshot."""

# ruff: noqa: S101

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from evaluatorq.dashboard import insights_routes
from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.insights_uploads import is_uploaded_source
from evaluatorq.insights.store import get_insights_runs_dir
from evaluatorq.local_sessions import SessionSearchResult, clear_summary_cache
from evaluatorq.trace_finder.models import Snapshot
from tests.local_sessions.conftest import write_jsonl


@pytest.fixture(autouse=True)
def _fresh_summary_cache() -> None:
    clear_summary_cache()


def _utc(day: int, hour: int = 10) -> datetime:
    return datetime(2026, 10, day, hour, tzinfo=timezone.utc)


def _claude(sid: str, *, text: str = 'hello there', day: int = 1) -> Path:
    root = Path(os.environ['CLAUDE_CONFIG_DIR']) / 'projects'
    start = _utc(day)
    base = {'sessionId': sid, 'cwd': '/work/proj', 'version': '2.0', 'entrypoint': 'cli'}
    path = write_jsonl(
        root / '-work-proj' / f'{sid}.jsonl',
        [
            {
                **base,
                'type': 'user',
                'uuid': 'u1',
                'parentUuid': None,
                'timestamp': start.isoformat(),
                'message': {'role': 'user', 'content': text},
            },
            {
                **base,
                'type': 'assistant',
                'uuid': 'a1',
                'parentUuid': 'u1',
                'timestamp': (start + timedelta(minutes=5)).isoformat(),
                'message': {'id': 'm1', 'model': 'claude-x', 'content': [{'type': 'text', 'text': 'ok'}]},
            },
        ],
    )
    os.utime(path, (start.timestamp(), start.timestamp()))
    return path


def _codex(sid: str, *, text: str = 'codex prompt', day: int = 2, hour: int = 10) -> Path:
    home = Path(os.environ['CODEX_HOME'])
    ts = _utc(day, hour).isoformat()
    path = write_jsonl(
        home / 'sessions' / '2026' / '10' / f'{day:02d}' / f'rollout-2026-10-{day:02d}T10-00-00-{sid}.jsonl',
        [
            {'timestamp': ts, 'type': 'session_meta', 'payload': {'id': sid, 'timestamp': ts, 'cwd': '/work/codex'}},
            {
                'timestamp': ts,
                'type': 'response_item',
                'payload': {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': text}]},
            },
        ],
    )
    os.utime(path, (_utc(day, hour).timestamp(), _utc(day, hour).timestamp()))
    return path


def _ref(source: str, path: Path) -> str:
    return f'{source}:{path}'


@pytest.fixture
def client() -> TestClient:
    return TestClient(build_app())


@pytest.fixture
def token(client: TestClient) -> str:
    match = re.search(r'name="csrf" value="([^"]+)"', client.get('/insights/new').text)
    assert match is not None
    return match.group(1)


def _search(client: TestClient, token: str, **fields: Any) -> Any:
    return client.post('/insights/sessions/search', data={'csrf': token, 'session_source': ['claude-code', 'codex'], **fields})


def test_search_lists_claude_and_codex_sessions_with_checkboxes(client: TestClient, token: str) -> None:
    claude = _claude('c1')
    codex = _codex('x1')

    response = _search(client, token)

    assert response.status_code == 200
    assert response.text.count('name="session"') == 2
    assert _ref('claude-code', claude) in response.text
    assert _ref('codex', codex) in response.text
    assert 'hello there' in response.text and 'codex prompt' in response.text
    assert 'data-sessions-all' in response.text
    assert '2 sessions' in response.text and '0 selected' in response.text
    assert 'data-utc="2026-10-02T10:00:00"' in response.text


def test_search_is_handled_by_the_sessions_route_not_the_run_page(client: TestClient, token: str) -> None:
    response = _search(client, token)

    assert response.status_code == 200
    assert 'Insights run not found' not in response.text


def test_source_filter_limits_the_results(client: TestClient, token: str) -> None:
    _claude('c1')
    _codex('x1')

    response = _search(client, token, session_source=['codex'])

    assert response.text.count('name="session"') == 1
    assert 'codex prompt' in response.text and 'hello there' not in response.text


def test_date_bounds_use_the_browser_offset(client: TestClient, token: str) -> None:
    _claude('c1')
    _codex('x1')

    only_second = _search(client, token, session_from='2026-10-02', session_to='2026-10-02', tz_offset='0')
    nothing_late = _search(client, token, session_from='2026-10-03', session_to='2026-10-04', tz_offset='0')

    assert only_second.text.count('name="session"') == 1 and 'codex prompt' in only_second.text
    assert 'No local sessions match these filters.' in nothing_late.text


def test_empty_states_tell_missing_folders_from_no_matches(client: TestClient, token: str) -> None:
    missing = _search(client, token)
    assert 'No Claude Code, Claude desktop, Codex or omp session folders were found on this computer.' in missing.text

    (Path(os.environ['CLAUDE_CONFIG_DIR']) / 'projects').mkdir(parents=True)
    empty = _search(client, token)
    assert 'No local sessions match these filters.' in empty.text


def test_a_cut_short_search_says_how_much_it_scanned(
    client: TestClient, token: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        insights_routes,
        'search_sessions',
        lambda query, *, cancelled=None: SessionSearchResult(sessions=(), scanned_files=3, candidate_files=10, complete=False),
    )

    response = _search(client, token)

    assert (
        'Searched 3 of 10 session files before the 20-second limit. Narrow the dates or text to see the rest.'
        in response.text
    )


def test_selected_sessions_stay_checked(client: TestClient, token: str) -> None:
    claude = _claude('c1')
    _codex('x1')

    response = _search(client, token, selected=[_ref('claude-code', claude)])

    assert response.text.count(' checked') == 1
    assert re.search(rf'value="{re.escape(_ref("claude-code", claude))}" checked', response.text)
    assert '1 selected' in response.text


def test_search_and_snapshot_need_the_csrf_token(client: TestClient, token: str) -> None:
    assert client.post('/insights/sessions/search', data={'session_source': ['codex']}).status_code == 403
    assert client.post('/insights/sessions/snapshot', data={'session': ['codex:/x']}).status_code == 403


def test_search_rejects_a_bad_source_date_or_offset(client: TestClient, token: str) -> None:
    assert _search(client, token, session_source=['bogus']).status_code == 422
    assert client.post('/insights/sessions/search', data={'csrf': token}).status_code == 422
    assert _search(client, token, session_from='yesterday', tz_offset='0').status_code == 422
    assert _search(client, token, session_from='2026-10-02', tz_offset='abc').status_code == 422
    assert _search(client, token, session_from='2026-10-05', session_to='2026-10-02', tz_offset='0').status_code == 422


def test_snapshot_freezes_two_sessions_into_an_upload(client: TestClient, token: str) -> None:
    claude = _claude('c1')
    codex = _codex('x1')

    response = client.post(
        '/insights/sessions/snapshot',
        data={'csrf': token, 'session': [_ref('claude-code', claude), _ref('codex', codex)]},
    )

    assert response.status_code == 201
    body = response.json()
    assert body['n_sessions'] == 2 and body['failed'] == []
    path = Path(body['path'])
    assert body['bytes'] == path.stat().st_size
    snapshot = Snapshot.model_validate_json(path.read_bytes())
    assert len(snapshot.documents) == 2
    assert is_uploaded_source(get_insights_runs_dir(), path)


def test_snapshot_rejects_a_path_outside_the_session_folders(client: TestClient, token: str) -> None:
    response = client.post('/insights/sessions/snapshot', data={'csrf': token, 'session': ['codex:/etc/hosts']})

    assert response.status_code == 422
    assert 'error' in response.json()


def test_snapshot_rejects_no_selection_and_more_than_the_limit(client: TestClient, token: str) -> None:
    assert client.post('/insights/sessions/snapshot', data={'csrf': token}).status_code == 422
    many = client.post('/insights/sessions/snapshot', data={'csrf': token, 'session': ['codex:/x'] * 1001})
    assert many.status_code == 422
    assert 'at most 1000' in many.json()['error']
    assert client.post('/insights/sessions/snapshot', data={'csrf': token, 'session': ['nonsense']}).status_code == 422
    full = client.post('/insights/sessions/snapshot', data={'csrf': token, 'session': ['codex:/x'] * 1000})
    assert full.status_code == 422, 'a full selection is not cut off by the form field cap'
    assert 'could be loaded' in full.json()['error']


def test_an_oversized_snapshot_is_refused_with_its_size(
    client: TestClient, token: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    claude = _claude('c1')
    monkeypatch.setattr(insights_routes, 'MAX_INSIGHTS_UPLOAD_BYTES', 10)

    response = client.post('/insights/sessions/snapshot', data={'csrf': token, 'session': [_ref('claude-code', claude)]})

    assert response.status_code == 413
    assert response.json()['error'].startswith('The selected sessions make a ')
    assert 'Select fewer sessions.' in response.json()['error']
    assert not (get_insights_runs_dir() / '.uploads').exists() or not list((get_insights_runs_dir() / '.uploads').iterdir())


def test_a_broken_session_is_reported_with_its_reason_but_no_content(client: TestClient, token: str) -> None:
    good = _codex('x1')
    broken = write_jsonl(
        Path(os.environ['CODEX_HOME']) / 'sessions' / '2026' / '10' / '02' / 'rollout-2026-10-02T10-00-00-broken.jsonl',
        ['secret user words that must not be echoed'],
    )

    response = client.post(
        '/insights/sessions/snapshot', data={'csrf': token, 'session': [_ref('codex', good), _ref('codex', broken)]}
    )

    assert response.status_code == 201
    body = response.json()
    assert body['n_sessions'] == 1
    assert [item['path'] for item in body['failed']] == [str(broken)]
    reason = body['failed'][0]['error']
    assert str(broken) in reason
    assert 'not a main session' in reason or 'unreadable session' in reason
    assert 'secret user words' not in json.dumps(body)


def test_a_ref_outside_the_session_folders_is_reported_with_its_reason(client: TestClient, token: str) -> None:
    good = _codex('x1')

    response = client.post(
        '/insights/sessions/snapshot', data={'csrf': token, 'session': [_ref('codex', good), 'codex:/etc/hosts']}
    )

    assert response.status_code == 201
    assert 'not a main session file inside the codex session folders' in response.json()['failed'][0]['error']


def test_date_bounds_are_local_midnights_in_the_browser_offset(client: TestClient, token: str) -> None:
    _codex('late', day=3, hour=2)

    utc_day = _search(client, token, session_from='2026-10-02', session_to='2026-10-02', tz_offset='0')
    west_day = _search(client, token, session_from='2026-10-02', session_to='2026-10-02', tz_offset='300')

    assert 'No local sessions match these filters.' in utc_day.text
    assert west_day.text.count('name="session"') == 1, '02:00Z on the 3rd is still the 2nd at UTC-5'


def test_a_session_title_is_escaped(client: TestClient, token: str) -> None:
    _claude('c1', text='<script>alert(1)</script>')

    response = _search(client, token)

    assert '<script>alert(1)' not in response.text
    assert '&lt;script&gt;alert(1)' in response.text


def test_plan_before_sessions_are_frozen_asks_for_a_selection(client: TestClient) -> None:
    compact = client.get('/insights/new/plan', params={'source': 'sessions', 'dimensions': 'intent', 'compact': '1'})
    full = client.get('/insights/new/plan', params={'source': 'sessions', 'dimensions': 'intent'})

    assert 'Estimate unavailable: Select local sessions and press Next.' in compact.text
    assert full.status_code == 422
    assert 'Select local sessions and press Next.' in full.text
