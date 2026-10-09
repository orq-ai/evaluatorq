"""The Settings-gated compression inspector's source, privacy, and rendering routes."""

from __future__ import annotations

import html
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from evaluatorq.common.trace_document import TraceDocument
from evaluatorq.dashboard import compression_inspector
from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.compression_inspector_data import CompressionPreview
from evaluatorq.local_sessions import SessionRef, SessionSearchResult, load_session_document
from evaluatorq.trace_finder.settings import DashboardSettings, effective_settings, save_settings
from tests.local_sessions.conftest import write_jsonl


def _claude_session(session_id: str, *, prompt: str, answer: str) -> Path:
    started = datetime(2026, 10, 1, 10, tzinfo=timezone.utc)
    base = {'sessionId': session_id, 'cwd': '/work/private-project', 'version': '2.0', 'entrypoint': 'cli'}
    return write_jsonl(
        Path(os.environ['CLAUDE_CONFIG_DIR']) / 'projects' / '-work-private-project' / f'{session_id}.jsonl',
        [
            {
                **base,
                'type': 'user',
                'uuid': 'user-1',
                'parentUuid': None,
                'timestamp': started.isoformat(),
                'message': {'role': 'user', 'content': prompt},
            },
            {
                **base,
                'type': 'assistant',
                'uuid': 'assistant-1',
                'parentUuid': 'user-1',
                'timestamp': (started + timedelta(minutes=1)).isoformat(),
                'message': {'id': 'message-1', 'model': 'claude-test', 'content': answer},
            },
        ],
    )

def _codex_session(session_id: str, *, prompt: str, answer: str) -> Path:
    timestamp = '2026-10-01T10:00:00.000Z'
    root = Path(os.environ['CODEX_HOME']) / 'sessions' / '2026' / '10' / '01'
    path = root / f'rollout-2026-10-01T10-00-00-{session_id}.jsonl'
    records = [
        {'timestamp': timestamp, 'type': 'session_meta', 'payload': {'id': session_id, 'timestamp': timestamp, 'cwd': '/work/codex'}},
        {'timestamp': timestamp, 'type': 'response_item', 'payload': {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': prompt}]}},
        {'timestamp': timestamp, 'type': 'response_item', 'payload': {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': answer}]}},
    ]
    return write_jsonl(path, records)



@pytest.fixture
def client() -> TestClient:
    save_settings(DashboardSettings.model_validate({'compression_inspector_enabled': True}))
    return TestClient(build_app())


def _csrf(client: TestClient) -> str:
    response = client.get('/compression-inspector')
    assert response.status_code == 200
    match = re.search(r'name="csrf" value="([^"]+)"', response.text)
    assert match is not None
    return match.group(1)


def test_every_route_returns_not_found_when_persisted_switch_is_disabled(client: TestClient) -> None:
    save_settings(DashboardSettings.model_validate({'compression_inspector_enabled': False}))

    responses = (
        client.get('/compression-inspector'),
        client.post('/compression-inspector/source/orq'),
        client.post('/compression-inspector/sessions/search'),
        client.post('/compression-inspector/sessions/load'),
    )

    assert all(response.status_code == 404 for response in responses)


def test_get_offers_orq_and_local_sources_with_limits_and_no_question_notice(client: TestClient) -> None:
    response = client.get('/compression-inspector')

    assert response.status_code == 200
    assert 'Orq trace ID' in response.text
    assert 'Local sessions' in response.text
    assert 'most recent 7 days' in response.text
    assert '500,000' in response.text
    assert '112,000' in response.text
    assert '128,000' in response.text and '256,000' in response.text
    assert 'includes no question' in response.text.lower()


def test_all_post_paths_reject_missing_csrf_before_source_work(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        compression_inspector,
        'fetch_orq_document',
        lambda *_args, **_kwargs: pytest.fail('CSRF must be checked before fetching a trace'),
    )

    responses = (
        client.post('/compression-inspector/source/orq', data={'trace_id': 'secret-id'}),
        client.post('/compression-inspector/sessions/search', data={}),
        client.post('/compression-inspector/sessions/load', data={'session': 'claude-code:/private/session.jsonl'}),
    )

    assert all(response.status_code == 403 for response in responses)
    assert all('secret-id' not in response.text for response in responses)
    assert all('/private/session.jsonl' not in response.text for response in responses)


def test_orq_load_uses_recent_seven_day_window_and_renders_actual_preview(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _claude_session('orq-fixture', prompt='<script>alert(1)</script>', answer='safe answer')
    document = load_session_document(SessionRef(source='claude-code', path=path))
    calls: list[tuple[datetime | None, datetime | None]] = []

    async def fetch(
        _app: Any, trace_id: str, *, start: datetime | None = None, end: datetime | None = None
    ) -> TraceDocument:
        assert trace_id == 'trace-42'
        calls.append((start, end))
        return document

    monkeypatch.setattr(compression_inspector, 'fetch_orq_document', fetch)
    response = client.post(
        '/compression-inspector/source/orq',
        data={'csrf': _csrf(client), 'trace_id': 'trace-42'},
    )

    assert response.status_code == 200
    assert calls == [(None, None)]
    assert '&lt;script&gt;alert(1)&lt;/script&gt;' in response.text
    assert '<script>alert(1)</script>' not in response.text
    assert 'Original readable trace' in response.text
    assert 'Jev state' in response.text
    assert 'trace-42' not in response.text



def test_orq_explicit_dates_allow_older_trace_and_safe_source_errors_do_not_echo_identifier(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evaluatorq.dashboard.compression_inspector_data import InspectorSourceError

    observed: list[tuple[datetime | None, datetime | None]] = []

    async def missing(
        _app: Any, _trace_id: str, *, start: datetime | None = None, end: datetime | None = None
    ) -> None:
        observed.append((start, end))
        raise InspectorSourceError('fixed safe failure text')

    monkeypatch.setattr(compression_inspector, 'fetch_orq_document', missing)
    response = client.post(
        '/compression-inspector/source/orq',
        data={
            'csrf': _csrf(client),
            'trace_id': 'sensitive-trace-id',
            'start_date': '2020-01-02',
            'end_date': '2020-01-04',
        },
    )

    assert response.status_code == 200
    assert observed == [(datetime(2020, 1, 2, tzinfo=timezone.utc), datetime(2020, 1, 5, tzinfo=timezone.utc))]
    assert 'sensitive-trace-id' not in response.text
    assert 'fixed safe failure text' not in response.text
    assert 'could not be loaded' in response.text.lower()



def test_missing_orq_trace_shows_safe_message_without_echoing_id(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def missing(
        _app: Any, _trace_id: str, *, start: datetime | None = None, end: datetime | None = None
    ) -> None:
        return None

    monkeypatch.setattr(compression_inspector, 'fetch_orq_document', missing)
    response = client.post(
        '/compression-inspector/source/orq',
        data={'csrf': _csrf(client), 'trace_id': 'private-missing-id'},
    )

    assert response.status_code == 200
    assert 'No Orq trace matched' in response.text
    assert 'private-missing-id' not in response.text

def test_orq_date_overflow_returns_safe_validation_error(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def unexpected_fetch(
        _app: Any, _trace_id: str, *, start: datetime | None = None, end: datetime | None = None
    ) -> TraceDocument | None:
        pytest.fail('an overflowing date range must be rejected before fetching')

    monkeypatch.setattr(compression_inspector, 'fetch_orq_document', unexpected_fetch)
    response = client.post(
        '/compression-inspector/source/orq',
        data={
            'csrf': _csrf(client),
            'trace_id': 'private-overflow-id',
            'start_date': '9999-12-30',
            'end_date': '9999-12-31',
        },
    )

    assert response.status_code == 422
    assert 'Enter both valid dates' in response.text
    assert 'private-overflow-id' not in response.text
    assert '9999-12-31' not in response.text
def test_local_search_and_load_use_real_claude_and_codex_readers_and_escape_content(client: TestClient) -> None:
    prompt = "</pre><script>alert('prompt')</script>"
    answer = '<img src=x onerror=alert(2)>'
    claude_path = _claude_session('private-session-id', prompt=prompt, answer=answer)
    codex_path = _codex_session(
        '019a0000-0000-7000-8000-000000000001', prompt='Codex prompt', answer='Codex answer'
    )
    search = client.post(
        '/compression-inspector/sessions/search',
        data={'csrf': _csrf(client)},
    )

    assert search.status_code == 200
    visible_text = html.unescape(re.sub(r'<[^>]*>', '', search.text))
    assert 'private-session-id' not in visible_text
    assert f'>{claude_path}<' not in search.text
    assert html.escape(f'claude-code:{claude_path}', quote=True) in search.text
    assert html.escape(f'codex:{codex_path}', quote=True) in search.text
    loaded = client.post(
        '/compression-inspector/sessions/load',
        data={'csrf': _csrf(client), 'session': f'claude-code:{claude_path}'},
    )

    assert loaded.status_code == 200
    assert html.escape(prompt) in loaded.text
    assert html.escape(answer) in loaded.text
    assert '<script>alert(' not in loaded.text
    assert '<img src=x onerror=' not in loaded.text
    assert 'Jev state' in loaded.text
    assert 'serialized jev state characters' in loaded.text.lower()
    assert 'global character cap' in loaded.text.lower()
    assert 'question-aware' in loaded.text.lower()

    codex_loaded = client.post(
        '/compression-inspector/sessions/load',
        data={'csrf': _csrf(client), 'session': f'codex:{codex_path}'},
    )
    assert codex_loaded.status_code == 200
    assert 'Codex prompt' in codex_loaded.text
    assert 'Codex answer' in codex_loaded.text



def test_local_search_empty_and_invalid_selection_are_safe(client: TestClient) -> None:
    empty = client.post('/compression-inspector/sessions/search', data={'csrf': _csrf(client)})
    private_path = '/private/home/secret.jsonl'
    invalid = client.post(
        '/compression-inspector/sessions/load',
        data={'csrf': _csrf(client), 'session': f'claude-code:{private_path}'},
    )

    assert empty.status_code == 200
    assert 'No Claude Code' in empty.text
    assert invalid.status_code == 200
    assert private_path not in invalid.text
    assert 'local session could not be loaded' in invalid.text.lower()



def test_incomplete_empty_session_search_does_not_claim_no_matches_or_folders(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        compression_inspector,
        'search_sessions',
        lambda **_kwargs: SessionSearchResult(sessions=(), scanned_files=2, candidate_files=5, complete=False),
    )
    response = client.post('/compression-inspector/sessions/search', data={'csrf': _csrf(client)})

    assert response.status_code == 200
    assert 'stopped before finding a local session' in response.text.lower()
    assert 'try again' in response.text.lower()
    assert 'narrow' in response.text.lower()
    assert 'session folders were found' not in response.text.lower()
    assert 'no local sessions matched' not in response.text.lower()


def test_preview_explains_when_original_has_no_readable_messages() -> None:
    preview = CompressionPreview(
        original='',
        compressed={'messages': []},
        original_chars=0,
        compressed_chars=15,
        message_count=0,
        omitted_messages=0,
        global_cap=500_000,
    )

    rendered = compression_inspector._preview_html(preview)

    assert 'No readable input or output is available for this source.' in rendered
    assert '<h3 id="compression-original-title">Original readable trace</h3><p' in rendered
    assert '<h3 id="compression-original-title">Original readable trace</h3><pre></pre>' not in rendered
def test_rendered_preview_uses_exact_backend_counts_and_structured_state(client: TestClient) -> None:
    path = _claude_session('count-check', prompt='a prompt', answer='an answer')
    trace_document = load_session_document(SessionRef(source='claude-code', path=path))
    preview = compression_inspector.preview_document(
        trace_document,
        global_char_cap=effective_settings().trace_input_chars,
    )
    ref = f'claude-code:{path}'
    response = client.post(
        '/compression-inspector/sessions/load',
        data={'csrf': _csrf(client), 'session': ref},
    )
    state = html.escape(json.dumps(preview.compressed, ensure_ascii=False, indent=2), quote=True)

    assert response.headers['cache-control'] == 'no-store'
    assert f'{preview.original_chars:,}' in response.text
    assert f'{preview.compressed_chars:,}' in response.text
    assert str(preview.omitted_messages) in response.text
    assert state in response.text
