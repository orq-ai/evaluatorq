"""Canonical compression previews and safe dashboard trace loading."""

from __future__ import annotations

import asyncio
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evaluatorq.common.model_input import jev_state, serialized_chars
from evaluatorq.common.trace_document import TraceDocument, ensure_trace_document, prompt_messages
from evaluatorq.dashboard import compression_inspector_data as inspector_data
from evaluatorq.dashboard.auth import DashboardAuth
from evaluatorq.insights.transcript import full_conversation_view_with_message_spans
from evaluatorq.local_sessions import SessionRef, load_session_document
from evaluatorq.trace_finder.orq_source import PAGE_SIZE
from tests.insights.test_population import make_trace
from tests.local_sessions.conftest import write_jsonl
from tests.trace_finder.test_orq_source import FakeOrq, FakeTraces, summary

UTC = timezone.utc
START = datetime(2026, 9, 20, tzinfo=UTC)
END = datetime(2026, 9, 21, tzinfo=UTC)


@pytest.fixture
def claude_projects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    config = tmp_path / 'claude'
    monkeypatch.setenv('CLAUDE_CONFIG_DIR', str(config))
    projects = config / 'projects'
    projects.mkdir(parents=True)
    return projects


@pytest.fixture
def codex_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / 'codex'
    monkeypatch.setenv('CODEX_HOME', str(home))
    return home


def _claude_document(root: Path, *, prompt: str) -> TraceDocument:
    timestamp = '2026-10-01T10:00:00Z'
    base = {'sessionId': 'compression-claude', 'cwd': '/work/claude', 'version': '2.0', 'entrypoint': 'cli'}
    path = write_jsonl(
        root / '-work-claude' / 'compression-claude.jsonl',
        [
            {
                **base,
                'type': 'user',
                'uuid': 'user-1',
                'parentUuid': None,
                'timestamp': timestamp,
                'message': {'role': 'user', 'content': prompt},
            },
            {
                **base,
                'type': 'assistant',
                'uuid': 'assistant-1',
                'parentUuid': 'user-1',
                'timestamp': '2026-10-01T10:00:05Z',
                'message': {
                    'id': 'message-1',
                    'model': 'claude-x',
                    'content': [{'type': 'text', 'text': 'Finished.'}],
                },
            },
        ],
    )
    return load_session_document(SessionRef(source='claude-code', path=path))


def _codex_document(root: Path, *, prompt: str) -> TraceDocument:
    timestamp = '2026-10-02T10:00:00Z'
    path = write_jsonl(
        root / 'sessions' / '2026' / '10' / '02' / 'rollout-2026-10-02T10-00-00-codex-session.jsonl',
        [
            {
                'timestamp': timestamp,
                'type': 'session_meta',
                'payload': {'id': 'codex-session', 'timestamp': timestamp, 'cwd': '/work/codex'},
            },
            {
                'timestamp': timestamp,
                'type': 'response_item',
                'payload': {
                    'type': 'message',
                    'role': 'user',
                    'content': [{'type': 'input_text', 'text': prompt}],
                },
            },
            {
                'timestamp': timestamp,
                'type': 'response_item',
                'payload': {
                    'type': 'message',
                    'role': 'assistant',
                    'content': [{'type': 'output_text', 'text': 'Codex finished.'}],
                },
            },
        ],
    )
    return load_session_document(SessionRef(source='codex', path=path))



def _document_with_messages(contents: list[str]) -> TraceDocument:
    record = make_trace('compression-preview').model_copy(update={
        'messages': tuple(
            {'role': 'user' if index % 2 == 0 else 'assistant', 'content': content}
            for index, content in enumerate(contents)
        ),
    })
    return ensure_trace_document(record)


def test_preview_renders_media_markers_without_fetching_urls() -> None:
    image_url = 'https://example.test/private-image.png'
    record = make_trace('media').model_copy(
        update={
            'messages': ({
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'Describe this'},
                    {'type': 'image_url', 'image_url': {'url': image_url}},
                ],
            },),
        }
    )
    document = ensure_trace_document(record)

    preview = inspector_data.preview_document(document, global_char_cap=10_000)

    assert '[image]' in preview.original
    assert '[image, size unknown]' in str(preview.compressed)
    assert image_url not in preview.original
    assert image_url not in str(preview.compressed)


def test_preview_document_uses_uncapped_transcript_and_one_jev_state(monkeypatch: pytest.MonkeyPatch) -> None:
    state_calls: list[tuple[list[dict[str, Any]], int]] = []
    build_state = inspector_data.jev_state

    def counted_state(messages: list[dict[str, Any]], *, global_char_cap: int) -> dict[str, Any]:
        state_calls.append((messages, global_char_cap))
        return build_state(messages, global_char_cap=global_char_cap)

    monkeypatch.setattr(inspector_data, 'jev_state', counted_state)
    document = _document_with_messages(['detail ' * 100 for _ in range(20)])
    original_document = document.model_dump(mode='python')
    messages = prompt_messages(document)
    original, _, source_message_count = full_conversation_view_with_message_spans(document)
    expected_state = jev_state(messages, global_char_cap=4096)

    preview = inspector_data.preview_document(document, global_char_cap=4096)

    assert len(state_calls) == 1
    assert preview.original == original
    assert preview.original_chars == len(original)
    assert preview.compressed == expected_state
    assert preview.compressed_chars == serialized_chars(expected_state)
    assert preview.message_count == source_message_count == len(messages)
    marker = expected_state.get('omission')
    marker_match = re.fullmatch(r'\[\.\.\. (\d+) messages left out \.\.\.\]', marker) if marker else None
    marker_count = int(marker_match.group(1)) if marker_match else 0
    assert preview.omitted_messages == marker_count
    assert preview.omitted_messages > 0
    assert preview.original_chars > preview.global_cap
    assert preview.compressed_chars <= preview.global_cap
    assert preview.global_cap == 4096
    assert document.model_dump(mode='python') == original_document


def test_preview_document_keeps_original_text_uncapped_and_redacted() -> None:
    secret = 'API_TOKEN=gXeRk29vLq0Pz8Wd'
    document = _document_with_messages([
        secret + ' ' + ('context ' * 120),
        *('context ' * 120 for _ in range(19)),
    ])

    preview = inspector_data.preview_document(document, global_char_cap=4096)

    assert len(preview.original) == preview.original_chars
    assert preview.original_chars > preview.global_cap
    assert 'gXeRk29vLq0Pz8Wd' not in preview.original
    assert 'gXeRk29vLq0Pz8Wd' not in str(preview.compressed)
    assert 'context' in preview.original
    assert preview.compressed_chars <= preview.global_cap


def test_compression_preview_uses_real_codex_reader(codex_home: Path) -> None:
    document = _codex_document(codex_home, prompt='Codex source prompt')

    preview = inspector_data.preview_document(document, global_char_cap=10_000)

    assert document.metadata.capture_metadata['source'] == 'local:codex'
    assert 'Codex source prompt' in preview.original
    assert preview.message_count == 2
    assert preview.omitted_messages == 0


def test_compression_preview_uses_real_claude_reader(claude_projects: Path) -> None:
    document = _claude_document(claude_projects, prompt='Claude source prompt')

    preview = inspector_data.preview_document(document, global_char_cap=10_000)

    assert document.metadata.capture_metadata['source'] == 'local:claude-code'
    assert 'Claude source prompt' in preview.original
    assert preview.message_count == 2
    assert preview.omitted_messages == 0


def _app() -> SimpleNamespace:
    return SimpleNamespace(
        state=SimpleNamespace(finder_settings=SimpleNamespace(orq_workspace='workspace-a', orq_project_id='project-a'))
    )


@pytest.mark.asyncio
async def test_fetch_orq_document_uses_targeted_source_window_and_selected_oauth_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    traces = FakeTraces({
        None: ([summary('older-target', messages=[{'role': 'user', 'content': 'hello'}])], False, None),
    })
    client = FakeOrq(traces)
    auth = DashboardAuth(method='cli_oauth', api_key=None, base_url='https://my.orq.ai')
    seen: list[tuple[Any, dict[str, Any]]] = []
    source_closes: list[None] = []
    original_close = inspector_data.OrqTraceSource.close

    def close_source(source: inspector_data.OrqTraceSource) -> None:
        original_close(source)
        source_closes.append(None)

    monkeypatch.setattr(inspector_data.OrqTraceSource, 'close', close_source)
    monkeypatch.setattr(inspector_data, 'selected_dashboard_auth', lambda _app: auth)
    monkeypatch.setattr(
        inspector_data,
        'build_orq_client',
        lambda selected, **kwargs: seen.append((selected, kwargs)) or client,
    )
    document = await inspector_data.fetch_orq_document(_app(), 'older-target', start=START, end=END)

    assert document is not None
    assert document.metadata.trace_id == 'older-target'
    assert seen == [(auth, {'workspace': 'workspace-a', 'project': 'project-a'})]
    [query] = traces.query_calls
    assert query['from_'] == START
    assert query['to'] == END
    assert query['limit'] == PAGE_SIZE
    assert 'filter trace_id in ("older-target")' in query['oql']
    assert len(client.exit_calls) == 1
    assert len(source_closes) == 1



@pytest.mark.asyncio
async def test_fetch_orq_document_uses_sdk_http_transport_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx
    from orq_ai_sdk import Orq

    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith('/traces/query'):
            return httpx.Response(
                200,
                json={
                    'search': {
                        'data': [{
                            'trace_id': 'wire-target',
                            'span_id': 'span-wire-target',
                            'root_span_id': 'span-wire-target',
                            'started_at': START.isoformat(),
                            'ended_at': END.isoformat(),
                            'project_id': 'project-a',
                            'status': 'completed',
                            'product': 'chat',
                            'type': 'trace',
                            'providers': ['openai'],
                            'models': ['openai/gpt-4o'],
                            'attributes': {
                                'gen_ai': {
                                    'input': {'messages': [{'role': 'user', 'content': 'from SDK transport'}]},
                                },
                            },
                        }],
                        'has_more': False,
                        'next_page_token': None,
                    },
                },
            )
        return httpx.Response(200, json={'object': 'list', 'data': [], 'has_more': False})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    client = Orq(api_key='fixture-key', server_url='https://orq.test', async_client=http_client)
    auth = DashboardAuth(method='environment', api_key='fixture-key', base_url='https://orq.test')
    monkeypatch.setattr(inspector_data, 'selected_dashboard_auth', lambda _app: auth)
    monkeypatch.setattr(inspector_data, 'build_orq_client', lambda *_args, **_kwargs: client)

    document = await inspector_data.fetch_orq_document(_app(), 'wire-target', start=START, end=END)

    assert document is not None
    assert document.metadata.trace_id == 'wire-target'
    assert 'from SDK transport' in str(document.messages)
    [query] = [request for request in requests if request.url.path.endswith('/traces/query')]
    assert b'wire-target' in query.content


@pytest.mark.asyncio
async def test_fetch_orq_document_defaults_to_a_timezone_aware_seven_day_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    traces = FakeTraces({None: ([], False, None)})
    client = FakeOrq(traces)
    monkeypatch.setattr(
        inspector_data,
        'selected_dashboard_auth',
        lambda _app: DashboardAuth('environment', 'key', 'https://my.orq.ai'),
    )
    monkeypatch.setattr(inspector_data, 'build_orq_client', lambda *_args, **_kwargs: client)

    assert await inspector_data.fetch_orq_document(_app(), 'not-present') is None

    [query] = traces.query_calls
    assert query['to'].tzinfo is not None and query['to'].utcoffset() == timedelta(0)
    assert query['to'] - query['from_'] == timedelta(days=7)
    assert 0 < query['timeout_ms'] <= 30_000
    assert len(client.exit_calls) == 1


@pytest.mark.asyncio
async def test_fetch_orq_document_sanitizes_source_failures_and_closes_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingTraces(FakeTraces):
        async def query_async(self, **_kwargs: Any) -> Any:
            raise RuntimeError('private payload /home/private/path with token=secret and trace=target-id')

    client = FakeOrq(FailingTraces({}))
    monkeypatch.setattr(
        inspector_data,
        'selected_dashboard_auth',
        lambda _app: DashboardAuth('environment', 'key', 'https://my.orq.ai'),
    )
    monkeypatch.setattr(inspector_data, 'build_orq_client', lambda *_args, **_kwargs: client)

    with pytest.raises(inspector_data.InspectorSourceError) as error:
        await inspector_data.fetch_orq_document(_app(), 'target-id', start=START, end=END)

    assert str(error.value) == 'The selected Orq trace could not be loaded.'
    assert all(
        secret not in str(error.value)
        for secret in ('private payload', '/home/private/path', 'secret', 'target-id')
    )
    assert len(client.exit_calls) == 1


@pytest.mark.asyncio
async def test_fetch_orq_document_sanitizes_auth_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_auth(_app: Any) -> DashboardAuth:
        raise RuntimeError('selected auth at /private/credentials.json contains secret auth material')

    monkeypatch.setattr(inspector_data, 'selected_dashboard_auth', fail_auth)
    with pytest.raises(inspector_data.InspectorSourceError) as error:
        await inspector_data.fetch_orq_document(_app(), 'private-trace-id')

    assert str(error.value) == 'The selected Orq trace could not be loaded.'
    assert all(
        value not in str(error.value) for value in ('/private/credentials.json', 'auth material', 'private-trace-id')
    )



@pytest.mark.asyncio
async def test_fetch_orq_document_closes_client_if_source_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeOrq(FakeTraces({}))

    def fail_source(_client: Any) -> Any:
        raise RuntimeError('source initialization failed at /private/session with secret content')

    monkeypatch.setattr(
        inspector_data,
        'selected_dashboard_auth',
        lambda _app: DashboardAuth('environment', 'key', 'https://my.orq.ai'),
    )
    monkeypatch.setattr(inspector_data, 'build_orq_client', lambda *_args, **_kwargs: client)
    monkeypatch.setattr(inspector_data, 'OrqTraceSource', fail_source)

    with pytest.raises(inspector_data.InspectorSourceError) as error:
        await inspector_data.fetch_orq_document(_app(), 'private-trace-id')

    assert str(error.value) == 'The selected Orq trace could not be loaded.'
    assert len(client.exit_calls) == 1


@pytest.mark.asyncio
async def test_cleanup_logs_only_exception_types(monkeypatch: pytest.MonkeyPatch) -> None:
    from loguru import logger

    client = FakeOrq(FakeTraces({None: ([summary('cleanup-target')], False, None)}))
    messages: list[str] = []

    def fail_source_close(_source: Any) -> None:
        raise RuntimeError('source output from /private/session is secret')

    async def fail_client_close(_client: Any) -> None:
        raise RuntimeError('credential key from /private/settings is secret')

    monkeypatch.setattr(
        inspector_data,
        'selected_dashboard_auth',
        lambda _app: DashboardAuth('environment', 'key', 'https://my.orq.ai'),
    )
    monkeypatch.setattr(inspector_data, 'build_orq_client', lambda *_args, **_kwargs: client)
    monkeypatch.setattr(inspector_data.OrqTraceSource, 'close', fail_source_close)
    monkeypatch.setattr(inspector_data, 'close_orq_client', fail_client_close)
    sink_id = logger.add(messages.append)
    try:
        document = await inspector_data.fetch_orq_document(_app(), 'cleanup-target', start=START, end=END)
    finally:
        logger.remove(sink_id)

    assert document is not None
    output = ''.join(messages)
    assert 'source cleanup failed (RuntimeError)' in output
    assert 'client cleanup failed (RuntimeError)' in output
    assert all(secret not in output for secret in ('/private/session', 'source output', 'credential key', '/private/settings'))


@pytest.mark.asyncio
async def test_fetch_orq_document_does_not_swallow_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeOrq(FakeTraces({}))
    source_closes: list[None] = []
    original_close = inspector_data.OrqTraceSource.close

    def close_source(source: inspector_data.OrqTraceSource) -> None:
        original_close(source)
        source_closes.append(None)

    async def cancel_load(_source: inspector_data.OrqTraceSource, *_args: Any, **_kwargs: Any) -> Any:
        raise asyncio.CancelledError

    monkeypatch.setattr(
        inspector_data,
        'selected_dashboard_auth',
        lambda _app: DashboardAuth('environment', 'key', 'https://my.orq.ai'),
    )
    monkeypatch.setattr(inspector_data, 'build_orq_client', lambda *_args, **_kwargs: client)
    monkeypatch.setattr(inspector_data.OrqTraceSource, 'close', close_source)
    monkeypatch.setattr(inspector_data.OrqTraceSource, 'load_async', cancel_load)

    with pytest.raises(asyncio.CancelledError):
        await inspector_data.fetch_orq_document(_app(), 'target-id', start=START, end=END)

    assert len(source_closes) == 1
    assert len(client.exit_calls) == 1


@pytest.mark.asyncio
async def test_fetch_orq_document_rejects_naive_dates_without_leaking_trace_id() -> None:
    with pytest.raises(inspector_data.InspectorSourceError) as error:
        await inspector_data.fetch_orq_document(
            _app(), 'sensitive-trace-id', start=datetime(2026, 9, 20), end=END
        )

    assert 'sensitive-trace-id' not in str(error.value)
