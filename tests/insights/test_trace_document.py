from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from evaluatorq.common.trace_document import ensure_trace_document, prompt_messages
from evaluatorq.common.trace_document import TraceDocument, TraceMetadata
from evaluatorq.formats.atif import AtifAgent, AtifStep, AtifTrajectory
from evaluatorq.signals import compute_signals
from evaluatorq.trace_finder.models import TraceRecord


def test_trace_metadata_mirrors_trace_record_fields() -> None:
    assert set(TraceMetadata.model_fields) - {'signals'} == set(TraceRecord.model_fields) - {'messages'}


def _trace(*messages: dict[str, Any]) -> TraceRecord:
    return TraceRecord(
        schema_version=1,
        trace_id='trace-1',
        span_id='span-1',
        timestamp=datetime(2026, 10, 5, tzinfo=timezone.utc),
        messages=messages,
        project='project',
        model='model',
        provider='provider',
        status='ok',
        product='product',
        trace_type='span.responses',
    )


def test_trace_document_reconstructs_exact_existing_projection() -> None:
    trace = _trace(
        {'role': 'system', 'content': 'rules'},
        {'role': 'developer', 'content': [{'type': 'text', 'text': 'context'}]},
        {
            'role': 'assistant',
            'content': None,
            'tool_calls': [
                {
                    'id': 'call-1',
                    'type': 'function',
                    'function': {'name': 'run', 'arguments': '{ "cmd" : "pwd" }'},
                }
            ],
        },
        {'role': 'tool', 'tool_call_id': 'call-1', 'name': 'run', 'content': 'done',
         'trace_finder_metadata': {'tool_result': {'status': 'ok'}}},
        {'role': 'assistant', 'content': 'finished'},
    )

    document = ensure_trace_document(trace)

    assert prompt_messages(document) == list(trace.messages)
    assert document.trajectory.steps[2].observation is not None
    assert len(document.trajectory.steps[2].observation.results) == 1


def test_prompt_renderer_reads_native_atif_message_content() -> None:
    document = ensure_trace_document(_trace({'role': 'user', 'content': 'before'}))
    changed = document.model_copy(
        update={
            'trajectory': document.trajectory.model_copy(
                update={'steps': [document.trajectory.steps[0].model_copy(update={'message': 'after'})]}
            )
        }
    )

    assert prompt_messages(document) == [{'role': 'user', 'content': 'before'}]
    assert prompt_messages(changed) == [{'role': 'user', 'content': 'after'}]


def test_orphan_tool_result_is_stored_once_as_observation() -> None:
    document = ensure_trace_document(_trace({'role': 'tool', 'tool_call_id': 'missing', 'content': 'orphan'}))

    assert prompt_messages(document) == [{'role': 'tool', 'content': 'orphan', 'tool_call_id': 'missing'}]
    assert len(document.trajectory.steps) == 1
    assert document.trajectory.steps[0].observation is not None
    report = compute_signals(document.trajectory, only=['user_message_count', 'assistant_message_count', 'llm_call_count'])
    assert {name: result.value for name, result in report.results.items()} == {
        'user_message_count': 0,
        'assistant_message_count': 0,
        'llm_call_count': 0,
    }


def test_reused_call_id_result_pairs_with_nearest_preceding_call() -> None:
    trace = _trace(
        {'role': 'assistant', 'tool_calls': [{'id': 'same', 'function': {'name': 'first', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'same', 'content': 'first result'},
        {'role': 'assistant', 'tool_calls': [{'id': 'same', 'function': {'name': 'second', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'same', 'content': 'second result'},
    )

    document = ensure_trace_document(trace)
    call_steps = [step for step in document.trajectory.steps if step.tool_calls]

    assert [step.observation.results[0].content for step in call_steps if step.observation] == [
        'first result',
        'second result',
    ]


def test_empty_projection_does_not_create_activity() -> None:
    document = TraceDocument(
        metadata=TraceMetadata(
            trace_id='empty',
            span_id='empty-span',
            timestamp=datetime(2026, 10, 5, tzinfo=timezone.utc),
        ),
        trajectory=AtifTrajectory(
            trajectory_id='empty',
            agent=AtifAgent(name='unknown', version='unknown'),
            steps=[AtifStep(step_id=1, source='system', message='', extra={'evaluatorq.insights.hidden': True})],
        ),
    )

    assert prompt_messages(document) == []
    report = compute_signals(document.trajectory, only=['user_message_count', 'assistant_message_count', 'llm_call_count'])
    assert {name: result.value for name, result in report.results.items()} == {
        'user_message_count': 0,
        'assistant_message_count': 0,
        'llm_call_count': 0,
    }


def test_native_observation_retains_legacy_error_status_for_signals() -> None:
    document = ensure_trace_document(
        _trace(
            {'role': 'assistant', 'tool_calls': [{'id': 'c1', 'function': {'name': 'run', 'arguments': '{}'}}]},
            {
                'role': 'tool',
                'tool_call_id': 'c1',
                'content': 'failed',
                'trace_finder_metadata': {'tool_result': {'status': 'failed'}},
            },
        )
    )

    report = compute_signals(document.trajectory, only=['tool_error_count'])

    assert report.results['tool_error_count'].value == 1
    assert prompt_messages(document)[1]['trace_finder_metadata'] == {'tool_result': {'status': 'failed'}}


def test_text_from_mixed_multimodal_content_uses_native_atif_part() -> None:
    document = ensure_trace_document(
        _trace({'role': 'user', 'content': [
            {'type': 'text', 'text': 'before'},
            {'type': 'image_url', 'image_url': {'url': 'https://example.test/a.png', 'media_type': 'image/png'}},
            {'type': 'text', 'text': 'after'},
        ]})
    )
    step = document.trajectory.steps[0]

    assert isinstance(step.message, list)
    assert [part.text for part in step.message if part.type == 'text'] == ['before', 'after']
    assert step.message[1].type == 'image'
    assert prompt_messages(document)[0]['content'][0]['text'] == 'before'


@pytest.mark.parametrize('arguments', ['[]', '"scalar"', 'not json'])
def test_non_object_argument_text_roundtrips_and_keeps_signal_provenance(arguments: str) -> None:
    document = ensure_trace_document(
        _trace({'role': 'assistant', 'tool_calls': [
            {'id': 'c1', 'type': 'function', 'function': {'name': 'run', 'arguments': arguments}}
        ]})
    )

    calls = document.trajectory.steps[0].tool_calls
    assert calls is not None
    call = calls[0]
    assert call.arguments == {}
    assert call.extra is not None
    assert call.extra['evaluatorq.raw_arguments'] == arguments
    assert prompt_messages(document)[0]['tool_calls'][0]['function']['arguments'] == arguments


def test_idless_tool_call_keeps_no_fabricated_prompt_id() -> None:
    document = ensure_trace_document(
        _trace({'role': 'assistant', 'tool_calls': [
            {'type': 'function', 'function': {'name': 'run', 'arguments': '{}'}}
        ]})
    )

    calls = document.trajectory.steps[0].tool_calls
    assert calls is not None
    call = calls[0]
    assert call.tool_call_id.startswith('unknown-')
    assert prompt_messages(document)[0]['tool_calls'] == [
        {'type': 'function', 'function': {'name': 'run', 'arguments': '{}'}}
    ]


def test_prompt_renderer_uses_native_call_and_observation_ids() -> None:
    document = ensure_trace_document(_trace(
        {'role': 'assistant', 'tool_calls': [
            {'id': 'before', 'type': 'function', 'function': {'name': 'run', 'arguments': '{}'}}
        ]},
        {'role': 'tool', 'tool_call_id': 'before', 'content': 'done'},
    ))
    step = document.trajectory.steps[0]
    assert step.tool_calls is not None and step.observation is not None
    call = step.tool_calls[0].model_copy(update={'tool_call_id': 'after'})
    observation = step.observation.model_copy(update={
        'results': [step.observation.results[0].model_copy(update={'source_call_id': 'after'})],
    })
    changed = document.model_copy(update={'trajectory': document.trajectory.model_copy(update={
        'steps': [step.model_copy(update={'tool_calls': [call], 'observation': observation})],
    })})
    rendered = prompt_messages(changed)
    assert rendered[0]['tool_calls'][0]['id'] == 'after'
    assert rendered[1]['tool_call_id'] == 'after'
