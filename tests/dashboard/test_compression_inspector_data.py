"""Canonical compression previews and safe dashboard trace loading."""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evaluatorq.common.model_input import cap_text, jev_state, serialized_chars
from evaluatorq.common.trace_document import TraceDocument, ensure_trace_document, prompt_messages
from evaluatorq.dashboard import compression_inspector_data as inspector_data
from evaluatorq.dashboard.auth import DashboardAuth
from evaluatorq.dashboard.compression_inspector_data import CompressionKnobs
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

    preview = inspector_data.preview_document(document, knobs=CompressionKnobs.defaults(10_000))

    assert '[image]' in preview.readable
    assert '[image, size unknown]' in str(preview.compressed)
    assert image_url not in preview.readable
    assert image_url not in str(preview.compressed)


def test_preview_document_defaults_show_the_whole_transcript_beside_one_jev_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_calls: list[tuple[list[dict[str, Any]], int]] = []
    build_state = inspector_data.jev_state

    def counted_state(messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        state_calls.append((messages, kwargs['global_char_cap']))
        return build_state(messages, **kwargs)

    monkeypatch.setattr(inspector_data, 'jev_state', counted_state)
    document = _document_with_messages(['detail ' * 100 for _ in range(20)])
    original_document = document.model_dump(mode='python')
    messages = prompt_messages(document)
    original, _, source_message_count = full_conversation_view_with_message_spans(document)
    expected_state = jev_state(messages, global_char_cap=4096)

    preview = inspector_data.preview_document(document, knobs=CompressionKnobs(global_char_cap=4096, budget=4096))

    assert len(state_calls) == 1
    assert preview.transcript_chars == len(original)
    assert preview.readable == cap_text(original, 4096)[0]
    assert preview.readable_chars == 4096
    assert preview.compressed == expected_state
    assert preview.compressed_chars == serialized_chars(expected_state)
    assert preview.message_count == source_message_count == len(messages)
    marker = expected_state.get('omission')
    marker_match = re.fullmatch(r'\[\.\.\. (\d+) messages left out \.\.\.\]', marker) if marker else None
    marker_count = int(marker_match.group(1)) if marker_match else 0
    assert preview.omitted_messages == marker_count
    assert preview.omitted_messages > 0
    assert preview.transcript_chars > preview.global_cap
    assert preview.compressed_chars <= preview.global_cap
    assert preview.global_cap == 4096
    assert document.model_dump(mode='python') == original_document


def test_preview_document_scrubs_secrets_from_both_views() -> None:
    secret = 'API_TOKEN=gXeRk29vLq0Pz8Wd'
    document = _document_with_messages([
        secret + ' ' + ('context ' * 120),
        *('context ' * 120 for _ in range(19)),
    ])

    knobs = CompressionKnobs(global_char_cap=4096, budget=100_000)
    preview = inspector_data.preview_document(document, knobs=knobs)

    assert preview.readable_chars == preview.transcript_chars > preview.global_cap
    assert 'gXeRk29vLq0Pz8Wd' not in preview.readable
    assert 'gXeRk29vLq0Pz8Wd' not in str(preview.compressed)
    assert 'context' in preview.readable
    assert preview.compressed_chars <= preview.global_cap


def test_compression_preview_uses_real_codex_reader(codex_home: Path) -> None:
    document = _codex_document(codex_home, prompt='Codex source prompt')

    preview = inspector_data.preview_document(document, knobs=CompressionKnobs.defaults(10_000))

    assert document.metadata.capture_metadata['source'] == 'local:codex'
    assert 'Codex source prompt' in preview.readable
    assert preview.message_count == 2
    assert preview.omitted_messages == 0


def test_compression_preview_uses_real_claude_reader(claude_projects: Path) -> None:
    document = _claude_document(claude_projects, prompt='Claude source prompt')

    preview = inspector_data.preview_document(document, knobs=CompressionKnobs.defaults(10_000))

    assert document.metadata.capture_metadata['source'] == 'local:claude-code'
    assert 'Claude source prompt' in preview.readable
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


@pytest.mark.parametrize(
    ('start', 'end', 'test_id'),
    [
        pytest.param(datetime(2026, 9, 20), END, 'naive-start', id='naive-start'),
        pytest.param(START, datetime(2026, 9, 21), 'naive-end', id='naive-end'),
        pytest.param(datetime(2026, 9, 20), datetime(2026, 9, 21), 'both-naive', id='both-naive'),
    ],
)
@pytest.mark.asyncio
async def test_fetch_orq_document_rejects_naive_dates_without_leaking_trace_id(
    start: datetime, end: datetime, test_id: str
) -> None:
    with pytest.raises(inspector_data.InspectorSourceError) as error:
        await inspector_data.fetch_orq_document(_app(), 'sensitive-trace-id', start=start, end=end)
    assert 'sensitive-trace-id' not in str(error.value)


def test_reduction_uses_full_json_independent_of_display_and_compression_caps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _document_with_messages(['A readable message. ' * 200, 'A long response. ' * 200])
    baseline = len(document.model_dump_json())
    monkeypatch.setattr(inspector_data, 'ORIGINAL_CHARS', 100)
    knobs = CompressionKnobs.defaults(10_000)
    initial = inspector_data.preview_document(document, knobs=knobs)
    reduced = inspector_data.preview_document(
        document, knobs=replace(knobs, original_view='readable', budget=1_000,
                                jev=replace(knobs.jev, state_chars=1_500)),
    )
    assert initial.original_chars <= 100
    assert initial.full_json_chars == reduced.full_json_chars == baseline
    assert reduced.readable_reduction_percent > initial.readable_reduction_percent
    assert reduced.jev_reduction_percent > initial.jev_reduction_percent
    assert reduced.readable_reduction_percent == pytest.approx(100 * (baseline - reduced.readable_chars) / baseline)
    assert reduced.jev_reduction_percent == pytest.approx(100 * (baseline - reduced.compressed_chars) / baseline)


def test_reduction_can_be_negative_when_formatting_expands_the_trace() -> None:
    document = _document_with_messages(['short'])
    preview = inspector_data.preview_document(document, knobs=CompressionKnobs.defaults(10_000))
    expanded = replace(preview, readable_chars=preview.full_json_chars * 2,
                       compressed_chars=preview.full_json_chars * 3)
    assert expanded.readable_reduction_percent == -100
    assert expanded.jev_reduction_percent == -200
