"""Characterization snapshots for the Insights classifier views and summary prompt."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from evaluatorq.common.trace_document import ensure_trace_document
from evaluatorq.insights.summarize import _build_prompt
from evaluatorq.insights.transcript import tool_activity_chunks, tool_inventory
from evaluatorq.trace_finder.models import TraceRecord


FIXTURE = Path(__file__).parent / 'fixtures' / 'atif_characterization_expected.json'


def _call(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    return {'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': arguments}}


def _trace(messages: list[dict[str, Any]], trace_id: str = 'atif-characterization') -> TraceRecord:
    return TraceRecord(
        schema_version=1,
        trace_id=trace_id,
        span_id='span-1',
        timestamp=datetime(2026, 9, 1, tzinfo=timezone.utc),
        messages=tuple(messages),
        project='default',
        model='claude',
        provider='anthropic',
        status='completed',
        product='agent',
        trace_type='conversation',
    )


def _fixtures() -> tuple[TraceRecord, TraceRecord]:
    trace = _trace([
        {'role': 'user', 'content': 'Inspect the deployment. <system-reminder>secret reminder</system-reminder>'},
        {
            'role': 'assistant',
            'content': 'I will inspect both services.',
            'tool_calls': [
                _call('a', 'Bash', '{"command":"uv run pytest tests/test_api.py -q"}'),
                _call('b', 'Lookup', '{"query":"tenant 7"}'),
                _call('c', 'Edit', '{"file_path":"src/api.py","bad":'),
                _call('d', 'Read', '{"file_path":"orphan.py"}'),
            ],
        },
        {
            'role': 'tool',
            'tool_call_id': 'a',
            'content': 'Error: timed out',
            'trace_finder_metadata': {'tool_result': {'status': 'failed'}},
        },
        {
            'role': 'tool',
            'tool_call_id': 'b',
            'content': 'credential=sk-live-secret-value; customer private payload',
            'status': 'completed',
        },
        {'role': 'tool', 'tool_call_id': 'orphan', 'content': 'Error: permission denied'},
        {'role': 'assistant', 'content': 'x' * 350 + ' middle assistant details ' + 'y' * 350},
        {'role': 'user', 'content': 'Second request.'},
        {'role': 'assistant', 'tool_calls': [_call('e', 'Bash', '{"command":"git push --force origin main"}')]},
        {'role': 'tool', 'tool_call_id': 'e', 'content': 'Error: permission denied; token=sk-live-secret-value'},
    ])

    long_messages: list[dict[str, Any]] = [{'role': 'user', 'content': 'THE ORIGINAL TASK'}]
    for index in range(12):
        call_id = f'l{index}'
        long_messages.extend([
            {
                'role': 'assistant',
                'tool_calls': [_call(call_id, 'Read', json.dumps({'file_path': f'path/{index}.py', 'note': 'n' * 35}))],
            },
            {'role': 'tool', 'tool_call_id': call_id, 'content': 'x' * 120},
        ])
    return trace, _trace(long_messages, trace_id='long')


@pytest.mark.parametrize('use_atif', [False, True], ids=['legacy-source', 'canonical-document'])
def test_current_classifier_views_match_characterization_snapshots(use_atif: bool) -> None:
    expected = json.loads(FIXTURE.read_text(encoding='utf-8'))
    expected.pop('finder_classifier_state', None)
    expected.pop('summary_prompt', None)
    trace, long_trace = (ensure_trace_document(trace) if use_atif else trace for trace in _fixtures())

    actual = {
        'tool_inventory': tool_inventory(trace),
        'tool_activity_chunks': tool_activity_chunks(trace),
        'tool_activity_chunks_small_budget': tool_activity_chunks(long_trace, budget=450),
    }
    summary_prompt = _build_prompt(trace)

    assert actual == expected
    serialized = json.dumps(actual, ensure_ascii=False)
    assert 'secret reminder' not in serialized
    assert '<conversation>0 user: Inspect the deployment. [harness reminder omitted]' in summary_prompt
    assert 'middle assistant details' in summary_prompt
    assert 'credential=<API_KEY>' in summary_prompt
    assert 'sk-live-secret-value' not in summary_prompt



@pytest.mark.parametrize('use_atif', [False, True], ids=['legacy-source', 'canonical-document'])
def test_classifier_state_uses_current_input_builder(use_atif: bool) -> None:
    from evaluatorq.common.model_input import JEV_STATE_CHARS, serialized_chars
    from evaluatorq.insights.labeling import prepare_classifier_state

    trace = _fixtures()[0]
    if use_atif:
        trace = ensure_trace_document(trace)
    state = prepare_classifier_state(trace, model='typesafe/jev-latest', char_cap=500_000)
    serialized = json.dumps(state, ensure_ascii=False)

    assert isinstance(state, dict)
    assert 'Inspect the deployment.' in serialized
    assert serialized_chars(state) <= JEV_STATE_CHARS
    assert 'sk-live-secret-value' not in serialized
    assert 'secret reminder' not in serialized
