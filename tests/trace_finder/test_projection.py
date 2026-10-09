"""Classifier input tests at the consumer-visible trace boundary."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluatorq.trace_finder.models import TraceRecord
from evaluatorq.trace_finder.projection import estimate_tokens, project_trace
from evaluatorq.trace_finder.settings import DashboardSettings, save_settings


def _trace(messages: tuple[dict[str, Any], ...]) -> TraceRecord:
    return TraceRecord(
        schema_version=1,
        trace_id='input-contract',
        span_id='span-1',
        timestamp=datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc),
        messages=messages,
        project='alpha',
        model='gpt-5',
        provider='openai',
        status='completed',
        product='chat',
        trace_type='conversation',
    )


def _persist_cap(monkeypatch: Any, tmp_path: Path, cap: int) -> None:
    settings_path = tmp_path / 'dashboard-settings.json'
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(settings_path))
    save_settings(DashboardSettings.model_validate({'trace_input_chars': cap}), settings_path)


def test_jev_classifier_receives_role_excerpt_after_persisted_global_cap(
    monkeypatch: Any, tmp_path: Path
) -> None:
    _persist_cap(monkeypatch, tmp_path, 500_000)
    content = 'source-head:' + 'x' * 600_000 + ':source-tail'
    trace = _trace(({'role': 'user', 'content': content},))

    projection = project_trace(trace, model='typesafe/jev-latest')
    state = projection.payload

    assert state['messages'][0]['text'].startswith('source-head:')
    assert state['messages'][0]['text'].endswith(':source-tail')
    assert state['messages'][0]['text'] == (
        f'{content[:1000]}[... {len(content) - 2_000} chars left out ...]{content[-1000:]}'
    )
    assert projection.omitted_messages == 0
    assert len(projection.serialized) <= 500_000
    assert trace.messages[0]['content'] == content


def test_non_jev_classifier_receives_full_readable_conversation_with_persisted_cap(
    monkeypatch: Any, tmp_path: Path
) -> None:
    _persist_cap(monkeypatch, tmp_path, 500_000)
    trace = _trace((
        {'role': 'user', 'content': 'REQUEST-BEGIN first question.'},
        {
            'role': 'assistant',
            'tool_calls': [{'id': 'lookup-1', 'function': {'name': 'search', 'arguments': '{"query":"middle-input"}'}}],
        },
        {'role': 'tool', 'tool_call_id': 'lookup-1', 'content': 'MIDDLE-RESULT found one record.'},
        {'role': 'assistant', 'content': 'FINAL-ANSWER resolved.'},
    ))

    projection = project_trace(trace, model='openai/gpt-6-luna')
    conversation = projection.payload['conversation']

    assert 'REQUEST-BEGIN first question.' in conversation
    assert 'middle-input' in conversation
    assert 'MIDDLE-RESULT found one record.' in conversation
    assert 'FINAL-ANSWER resolved.' in conversation
    assert len(projection.serialized) <= 500_000


def test_classifier_input_character_limit_is_persisted_and_keeps_both_edges(
    monkeypatch: Any, tmp_path: Path
) -> None:
    _persist_cap(monkeypatch, tmp_path, 20_000)
    content = 'beginning-' + 'z' * 40_000 + '-ending'
    trace = _trace(({'role': 'user', 'content': content},))

    projection = project_trace(trace, model='openai/gpt-6-luna')

    assert len(projection.serialized) <= 20_000
    assert 'beginning-' in projection.payload['conversation']
    assert '-ending' in projection.payload['conversation']
    assert trace.messages[0]['content'] == content



def test_non_jev_projection_fits_escaped_content_by_serialized_size() -> None:
    content = 'ESCAPED-HEAD' + chr(92) * 10_000 + 'ESCAPED-TAIL'
    trace = _trace(({'role': 'user', 'content': content},))

    projection = project_trace(trace, model='openai/gpt-6-luna', trace_input_chars=5_000)

    assert len(projection.serialized) <= 5_000
    assert 'ESCAPED-HEAD' in projection.payload['conversation']
    assert 'ESCAPED-TAIL' in projection.payload['conversation']
    assert '[... ' in projection.payload['conversation']

def test_token_estimate_uses_four_characters_per_estimated_token() -> None:
    assert estimate_tokens('a' * 4_001) == 1_001
