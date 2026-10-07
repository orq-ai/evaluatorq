from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from pydantic import ValidationError

from evaluatorq.common.trace_document import (
    DatasetRef,
    LabelSource,
    Outcome,
    TraceDocument,
    TraceMetadata,
    TrajectoryCounts,
    check_trace_document,
    ensure_trace_document,
    prompt_messages,
    trace_document_with_signals,
    trajectory_counts,
)
from evaluatorq.formats.atif import (
    AtifAgent,
    AtifAudioSource,
    AtifContentPart,
    AtifObservation,
    AtifObservationResult,
    AtifStep,
    AtifToolCall,
    AtifTrajectory,
)
from evaluatorq.signals import SignalReport, compute_signals
from evaluatorq.trace_finder.models import TraceRecord


def test_trace_metadata_mirrors_trace_record_fields() -> None:
    assert set(TraceMetadata.model_fields) - {'signals', 'dataset', 'outcome'} == set(TraceRecord.model_fields) - {'messages'}


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


def test_legacy_trace_trajectory_id_is_stable_and_separate_from_source_identity() -> None:
    trace = _trace({'role': 'user', 'content': 'hello'})

    first = ensure_trace_document(trace)
    repeated = ensure_trace_document(trace)

    assert first.trajectory.trajectory_id == repeated.trajectory.trajectory_id
    assert first.trajectory.trajectory_id != trace.trace_id


def test_legacy_trace_trajectory_id_includes_selected_span_identity() -> None:
    trace = _trace({'role': 'user', 'content': 'hello'})
    other_span = trace.model_copy(update={'span_id': 'span-2'})

    first_id = ensure_trace_document(trace).trajectory.trajectory_id
    second_id = ensure_trace_document(other_span).trajectory.trajectory_id
    assert first_id != second_id


def test_native_trace_document_trajectory_id_is_preserved() -> None:
    document = TraceDocument(
        metadata=TraceMetadata(
            trace_id='source-trace',
            span_id='source-span',
            timestamp=datetime(2026, 10, 5, tzinfo=timezone.utc),
        ),
        trajectory=AtifTrajectory(
            trajectory_id='native-trajectory',
            agent=AtifAgent(name='agent', version='1'),
            steps=[AtifStep(step_id=1, source='user', message='hello')],
        ),
    )

    assert ensure_trace_document(document).trajectory.trajectory_id == 'native-trajectory'


def test_signal_attachment_checks_atif_trajectory_id() -> None:
    document = ensure_trace_document(_trace({'role': 'user', 'content': 'hello'}))
    report = compute_signals(document.trajectory, only=[])

    assert trace_document_with_signals(document, report).metadata.signals is report
    with pytest.raises(ValueError, match='trajectory_id does not match'):
        trace_document_with_signals(
            document,
            SignalReport(trajectory_id=document.metadata.trace_id, results={}, config_version='1'),
        )


def test_multimessage_response_metadata_is_distributed_without_double_counting() -> None:
    source_messages = [
        {'role': 'assistant', 'content': 'alpha', 'tool_calls': [
            {'id': 'call-a', 'type': 'function', 'function': {'name': 'first', 'arguments': '{}'}},
        ]},
        {'role': 'assistant', 'content': 'beta', 'tool_calls': [
            {'id': 'call-b', 'type': 'function', 'function': {'name': 'second', 'arguments': '{}'}},
        ]},
        {'role': 'tool', 'tool_call_id': 'call-a', 'content': 'first result'},
        {'role': 'tool', 'tool_call_id': 'call-b', 'content': 'second result'},
    ]
    started_at = '2026-10-05T00:00:00+00:00'
    ended_at = '2026-10-05T00:00:02+00:00'
    raw_output = {
        '0': [{'type': 'message', 'id': 'message-a'}],
        '1': [{'type': 'message', 'id': 'message-b'}],
    }
    document = ensure_trace_document(
        _trace(*source_messages).model_copy(update={'capture_metadata': {
            'signal_selected_span': {
                'step_orders': [0, 1],
                'step_order': None,
                'output_items_by_order': raw_output,
                'metrics': {'prompt_tokens': 40, 'completion_tokens': 12},
                'started_at': started_at,
                'start_timestamp': datetime.fromisoformat(started_at).timestamp(),
                'end_timestamp': datetime.fromisoformat(ended_at).timestamp(),
                'tool_definitions': [{'type': 'function', 'name': 'lookup'}],
                'tool_definitions_present': True,
            },
        }}),
    )

    first, second = document.trajectory.steps
    assert prompt_messages(document) == source_messages
    assert first.metrics is not None
    assert (first.metrics.prompt_tokens, first.metrics.completion_tokens) == (40, 12)
    assert second.metrics is None
    assert (first.llm_call_count, second.llm_call_count) == (1, 0)
    assert first.timestamp == started_at
    assert second.timestamp is None
    assert first.extra is not None and first.extra['invocation']['start_timestamp']
    assert second.extra is not None and 'invocation' not in second.extra
    assert first.extra['evaluatorq.responses_tools'] == second.extra['evaluatorq.responses_tools']
    assert first.extra['evaluatorq.responses_output_items'] == raw_output['0']
    assert second.extra['evaluatorq.responses_output_items'] == raw_output['1']
    call_counts = compute_signals(document.trajectory, only=['llm_call_count'])
    assert call_counts.results['llm_call_count'].value == 1


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


def test_orphan_result_owner_uses_source_order() -> None:
    document = ensure_trace_document(_trace(
        {'role': 'user', 'content': 'before'},
        {'role': 'tool', 'tool_call_id': 'missing', 'content': 'orphan'},
        {'role': 'assistant', 'content': 'after', 'tool_calls': [
            {'id': 'after-call', 'type': 'function', 'function': {'name': 'run', 'arguments': '{}'}},
        ]},
        {'role': 'tool', 'tool_call_id': 'after-call', 'content': 'result'},
    ))

    assert prompt_messages(document) == [
        {'role': 'user', 'content': 'before'},
        {'role': 'tool', 'content': 'orphan', 'tool_call_id': 'missing'},
        {'role': 'assistant', 'content': 'after', 'tool_calls': [
            {'id': 'after-call', 'type': 'function', 'function': {'name': 'run', 'arguments': '{}'}},
        ]},
        {'role': 'tool', 'tool_call_id': 'after-call', 'content': 'result'},
    ]
    assert [step.step_id for step in document.trajectory.steps] == [1, 2, 3]
    assert [step.source for step in document.trajectory.steps] == ['user', 'system', 'agent']
    assert document.trajectory.steps[1].observation is not None
    assert document.trajectory.steps[2].observation is not None
    assert document.trajectory.steps[2].observation.results[0].source_call_id == 'after-call'


@pytest.mark.parametrize('bad_order', [True, -1, 'invalid'])
def test_prompt_renderer_falls_back_for_invalid_source_order(bad_order: Any) -> None:
    document = TraceDocument(
        metadata=TraceMetadata(
            trace_id='external',
            span_id='external-span',
            timestamp=datetime(2026, 10, 5, tzinfo=timezone.utc),
        ),
        trajectory=AtifTrajectory(
            trajectory_id='external',
            agent=AtifAgent(name='agent', version='1'),
            steps=[
                AtifStep(
                    step_id=1,
                    source='user',
                    message='first',
                    extra={'evaluatorq.insights.source_order': bad_order},
                    observation=AtifObservation(results=[AtifObservationResult(
                        content='orphan',
                        extra={'evaluatorq.insights.source_order': bad_order},
                    )]),
                ),
                AtifStep(step_id=2, source='agent', message='second'),
            ],
        ),
    )

    assert prompt_messages(document) == [
        {'role': 'user', 'content': 'first'},
        {'role': 'tool', 'content': 'orphan'},
        {'role': 'assistant', 'content': 'second'},
    ]


def test_signal_span_enrichment_accumulates_for_calls_on_same_step() -> None:
    trace = _trace(
        {'role': 'assistant', 'tool_calls': [
            {'id': 'call-a', 'type': 'function', 'function': {'name': 'first', 'arguments': '{}'}},
            {'id': 'call-b', 'type': 'function', 'function': {'name': 'second', 'arguments': '{}'}},
        ]},
        {'role': 'tool', 'tool_call_id': 'call-a', 'content': 'first result'},
        {'role': 'tool', 'tool_call_id': 'call-b', 'content': 'second result'},
    ).model_copy(update={'capture_metadata': {'signal_tool_spans': {
        'call-a': {'start_timestamp': '2026-10-05T00:00:01Z'},
        'call-b': {'start_timestamp': '2026-10-05T00:00:02Z'},
    }}})

    document = ensure_trace_document(trace)
    observation = document.trajectory.steps[0].observation
    assert observation is not None
    enriched = {}
    for result in observation.results:
        assert result.source_call_id is not None
        assert result.extra is not None
        enriched[result.source_call_id] = result.extra['start_timestamp']
    assert enriched == {
        'call-a': '2026-10-05T00:00:01Z',
        'call-b': '2026-10-05T00:00:02Z',
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


@pytest.mark.parametrize(
    'image_part',
    [
        {'type': 'input_image', 'image_url': 'https://example.test/a.png'},
        {'type': 'image', 'url': 'https://example.test/a.png', 'media_type': 'image/png'},
        {'type': 'image', 'url': 'https://example.test/a.png', 'path': 'extra-path', 'media_type': 'image/png'},
        {'type': 'input_image', 'source': {'path': 'asset.png', 'media_type': 'image/png'}},
    ],
)
def test_image_parts_keep_native_content_and_exact_source_shape(image_part: dict[str, Any]) -> None:
    document = ensure_trace_document(_trace({'role': 'user', 'content': [image_part]}))

    message = document.trajectory.steps[0].message
    assert isinstance(message, list)
    part = message[0]
    assert part.type == 'image'
    assert prompt_messages(document)[0]['content'] == [image_part]


def test_image_data_uri_infers_jpeg_without_changing_prompt_shape() -> None:
    image_part = {'type': 'input_image', 'image_url': 'data:image/jpeg;base64,/9j/2Q=='}
    document = ensure_trace_document(_trace({'role': 'user', 'content': [image_part]}))

    message = document.trajectory.steps[0].message
    assert isinstance(message, list)
    part = message[0]
    assert part.type == 'image'
    assert part.source is not None
    assert part.source.media_type == 'image/jpeg'
    assert prompt_messages(document)[0]['content'] == [image_part]


@pytest.mark.parametrize('url', ['https://example.test/image?token=secret', 'https://[invalid'])
def test_unknown_or_malformed_image_url_stays_raw(url: str) -> None:
    image_part = {'type': 'input_image', 'image_url': url}
    document = ensure_trace_document(_trace({'role': 'user', 'content': [image_part]}))

    message = document.trajectory.steps[0].message
    assert isinstance(message, list)
    assert message[0].type == 'text'
    assert prompt_messages(document)[0]['content'] == [image_part]


def test_input_audio_is_stored_as_atif_audio_and_roundtrips_exactly() -> None:
    audio_part = {'type': 'input_audio', 'input_audio': {'data': 'c291bmQ=', 'format': 'wav'}}
    document = ensure_trace_document(_trace({'role': 'user', 'content': [audio_part]}))

    message = document.trajectory.steps[0].message
    assert isinstance(message, list)
    part = message[0]
    assert part.type == 'audio'
    assert part.source is not None
    assert part.source.media_type == 'audio/wav'
    assert part.source.path == 'data:audio/wav;base64,c291bmQ='
    assert prompt_messages(document)[0]['content'] == [audio_part]
    serialized_trajectory = document.model_dump(mode='json')['trajectory']
    restored = AtifTrajectory.from_json(serialized_trajectory)
    assert restored.schema_version == 'ATIF-v1.8'
    assert restored == document.trajectory


def test_native_audio_datapoint_uses_atif_audio_version() -> None:
    document = TraceDocument(
        metadata=TraceMetadata(
            trace_id='native-audio',
            span_id='native-audio-span',
            timestamp=datetime(2026, 10, 5, tzinfo=timezone.utc),
        ),
        trajectory=AtifTrajectory(
            trajectory_id='native-audio',
            agent=AtifAgent(name='agent', version='1'),
            steps=[AtifStep(
                step_id=1,
                source='user',
                message=[AtifContentPart(
                    type='audio',
                    source=AtifAudioSource(media_type='audio/wav', path='clip.wav'),
                )],
            )],
        ),
    )

    datapoint = document.to_datapoint()
    restored = AtifTrajectory.from_json(datapoint.inputs['trajectory'])
    assert restored.schema_version == 'ATIF-v1.8'
    assert restored.has_audio()


@pytest.mark.parametrize(
    'audio_part',
    [
        {'type': 'audio', 'source': {'path': 'clip.wav', 'media_type': 'audio/wav'}},
        {'type': 'audio', 'source': 'https://example.test/clip.wav', 'media_type': 'audio/wav'},
    ],
)
def test_audio_source_layouts_are_native_and_losslessly_rendered(audio_part: dict[str, Any]) -> None:
    document = ensure_trace_document(_trace({'role': 'user', 'content': [audio_part]}))

    message = document.trajectory.steps[0].message
    assert isinstance(message, list)
    part = message[0]
    assert part.type == 'audio'
    assert prompt_messages(document)[0]['content'] == [audio_part]


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


@pytest.mark.parametrize(
    ('raw_call', 'mutated_call'),
    [
        (
            {'call_id': 'c', 'tool_name': 'run', 'input': {'cmd': 'pwd'}},
            {'call_id': 'c', 'tool_name': 'renamed', 'input': {'cmd': 'ls'}},
        ),
        (
            {'id': 'c', 'function': {'tool_name': 'run', 'input': {'cmd': 'pwd'}}},
            {'id': 'c', 'function': {'tool_name': 'renamed', 'input': {'cmd': 'ls'}}},
        ),
        (
            {'id': 'c', 'function': {'name': 'run', 'arguments': {'cmd': 'pwd'}}},
            {'id': 'c', 'function': {'name': 'renamed', 'arguments': {'cmd': 'ls'}}},
        ),
    ],
)
def test_tool_call_shape_roundtrips_and_mutations_use_native_values(
    raw_call: dict[str, Any], mutated_call: dict[str, Any]
) -> None:
    document = ensure_trace_document(_trace({'role': 'assistant', 'tool_calls': [raw_call]}))
    step = document.trajectory.steps[0]
    assert step.tool_calls is not None
    call = step.tool_calls[0]

    assert prompt_messages(document)[0]['tool_calls'] == [raw_call]
    changed_call = call.model_copy(update={'function_name': 'renamed', 'arguments': {'cmd': 'ls'}})
    changed = document.model_copy(update={
        'trajectory': document.trajectory.model_copy(update={
            'steps': [step.model_copy(update={'tool_calls': [changed_call]})],
        }),
    })
    assert prompt_messages(changed)[0]['tool_calls'] == [mutated_call]


@pytest.mark.parametrize(
    'raw_call',
    [
        {'id': 'c'},
        {'id': 'c', 'function': {}},
    ],
)
def test_tool_call_missing_name_and_arguments_stay_missing(raw_call: dict[str, Any]) -> None:
    document = ensure_trace_document(_trace({'role': 'assistant', 'tool_calls': [raw_call]}))

    assert prompt_messages(document)[0]['tool_calls'] == [raw_call]


def test_prompt_renderer_preserves_absent_content_and_explicit_null_content() -> None:
    document = ensure_trace_document(_trace(
        {'role': 'assistant'},
        {'role': 'assistant', 'content': None},
    ))

    assert prompt_messages(document) == [
        {'role': 'assistant'},
        {'role': 'assistant', 'content': None},
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


_SWE_ROW = DatasetRef(name='nebius/SWE-agent-trajectories', revision='68195a1450865274106246d0d0296a1d6807b88e', split='train', row_id='7')
_RESOLVED = Outcome(passed=True, score=1.0, source='programmatic', definition='SWE-bench issue resolved by the agent patch')


def _agent_trajectory(*, subagent: AtifTrajectory | None = None, trajectory_id: str | None = None) -> AtifTrajectory:
    return AtifTrajectory(
        trajectory_id=trajectory_id,
        agent=AtifAgent(name='swe-agent', version='unknown'),
        steps=[
            AtifStep(step_id=1, source='user', message='fix the bug'),
            AtifStep(
                step_id=2,
                source='agent',
                message='',
                reasoning_content='look at the file first',
                tool_calls=[AtifToolCall(tool_call_id='c1', function_name='bash', arguments={'cmd': 'cat a.py'})],
                observation=AtifObservation(results=[AtifObservationResult(source_call_id='c1', content='print(1)')]),
            ),
        ],
        subagent_trajectories=[subagent] if subagent else None,
    )


def test_dataset_row_needs_no_span_or_timestamp() -> None:
    metadata = TraceMetadata(trace_id='swe-agent:django__django-1:7', dataset=_SWE_ROW, outcome=_RESOLVED)

    assert metadata.span_id is None
    assert metadata.timestamp is None


def test_trace_without_dataset_still_needs_span_and_timestamp() -> None:
    with pytest.raises(ValidationError, match='needs span_id and timestamp'):
        TraceMetadata(trace_id='trace-1')


@pytest.mark.parametrize(
    ('span_id', 'timestamp'),
    [
        (None, datetime(2026, 10, 5, tzinfo=timezone.utc)),
        ('span-1', None),
    ],
)
def test_orq_trace_requires_span_id_and_timestamp_independently(
    span_id: str | None,
    timestamp: datetime | None,
) -> None:
    with pytest.raises(ValidationError, match='needs span_id and timestamp'):
        TraceMetadata(trace_id='trace-1', span_id=span_id, timestamp=timestamp)


def test_dataset_timestamp_must_be_timezone_aware() -> None:
    with pytest.raises(ValidationError, match='timestamp must include a timezone offset'):
        TraceMetadata(
            trace_id='row-7',
            timestamp=datetime(2026, 10, 5),
            dataset=_SWE_ROW,
            outcome=_RESOLVED,
        )


def test_dataset_row_without_outcome_is_rejected() -> None:
    with pytest.raises(ValidationError, match='needs an outcome'):
        TraceMetadata(trace_id='row-7', dataset=_SWE_ROW)


@pytest.mark.parametrize(('passed', 'source'), [(None, 'programmatic'), (True, 'none')])
def test_outcome_unlabelled_must_be_explicit(passed: bool | None, source: LabelSource) -> None:
    with pytest.raises(ValidationError, match="exactly when source is 'none'"):
        Outcome(passed=passed, source=source, definition='x')


def test_unlabelled_dataset_row_is_valid() -> None:
    outcome = Outcome(passed=None, source='none', definition='Orq traces carry no ground truth')

    assert TraceMetadata(trace_id='row-1', dataset=_SWE_ROW, outcome=outcome).outcome == outcome


def test_to_datapoint_carries_trajectory_and_outcome() -> None:
    document = TraceDocument(
        metadata=TraceMetadata(trace_id='row-7', dataset=_SWE_ROW, outcome=_RESOLVED),
        trajectory=_agent_trajectory(),
    )

    datapoint = document.to_datapoint()

    assert AtifTrajectory.model_validate(datapoint.inputs['trajectory']) == document.trajectory
    assert datapoint.inputs['dataset'] == {
        'name': 'nebius/SWE-agent-trajectories',
        'revision': '68195a1450865274106246d0d0296a1d6807b88e',
        'split': 'train',
        'row_id': '7',
    }
    assert datapoint.expected_output == {
        'passed': True,
        'score': 1.0,
        'source': 'programmatic',
        'definition': 'SWE-bench issue resolved by the agent patch',
    }


def test_to_datapoint_without_outcome_has_no_expected_output() -> None:
    document = ensure_trace_document(_trace({'role': 'user', 'content': 'hi'}))

    assert document.to_datapoint().expected_output is None


def test_trajectory_counts_include_subagents() -> None:
    trajectory = _agent_trajectory(subagent=_agent_trajectory(trajectory_id='sub-1'))
    counts = trajectory_counts(trajectory)

    assert counts == TrajectoryCounts(tool_calls=2, tool_results=2, reasoning_steps=2)


def test_check_trace_document_preserves_json_and_source_measured_content() -> None:
    document = TraceDocument(
        metadata=TraceMetadata(trace_id='row-7', dataset=_SWE_ROW, outcome=_RESOLVED),
        trajectory=_agent_trajectory(subagent=_agent_trajectory(trajectory_id='sub-1')),
    )
    expected_counts = TrajectoryCounts(tool_calls=2, tool_results=2, reasoning_steps=2)

    check_trace_document(document, expected_counts=expected_counts, expected_outcome=_RESOLVED)
    restored = TraceDocument.model_validate_json(document.model_dump_json())

    assert restored == document
    assert restored.metadata.dataset == _SWE_ROW
    assert restored.metadata.outcome == _RESOLVED
    assert restored.trajectory.steps[1].tool_calls == document.trajectory.steps[1].tool_calls
    assert restored.trajectory.steps[1].observation == document.trajectory.steps[1].observation
    assert restored.trajectory.steps[1].reasoning_content == document.trajectory.steps[1].reasoning_content
    assert restored.trajectory.subagent_trajectories == document.trajectory.subagent_trajectories


@pytest.mark.parametrize('lost_field', ['tool_call', 'tool_result', 'reasoning'])
def test_check_trace_document_rejects_lost_trajectory_content(lost_field: str) -> None:
    document = TraceDocument(
        metadata=TraceMetadata(trace_id='row-7', dataset=_SWE_ROW, outcome=_RESOLVED),
        trajectory=_agent_trajectory(),
    )
    agent_step = document.trajectory.steps[1]
    updates: dict[str, Any]
    if lost_field == 'tool_call':
        updates = {'tool_calls': None}
    elif lost_field == 'tool_result':
        assert agent_step.observation is not None
        updates = {'observation': agent_step.observation.model_copy(update={'results': []})}
    else:
        updates = {'reasoning_content': None}
    changed = document.model_copy(
        update={
            'trajectory': document.trajectory.model_copy(
                update={'steps': [document.trajectory.steps[0], agent_step.model_copy(update=updates)]}
            )
        }
    )

    with pytest.raises(ValueError, match='trajectory counts differ'):
        check_trace_document(
            changed,
            expected_counts=TrajectoryCounts(tool_calls=1, tool_results=1, reasoning_steps=1),
            expected_outcome=_RESOLVED,
        )


def test_check_trace_document_rejects_lost_outcome() -> None:
    document = TraceDocument(
        metadata=TraceMetadata(trace_id='row-7', dataset=_SWE_ROW, outcome=_RESOLVED),
        trajectory=_agent_trajectory(),
    )
    changed_outcome = Outcome(
        passed=False,
        score=0.0,
        source='programmatic',
        definition='SWE-bench issue resolved by the agent patch',
    )
    changed = document.model_copy(
        update={'metadata': document.metadata.model_copy(update={'outcome': changed_outcome})}
    )

    with pytest.raises(ValueError, match='outcome differs'):
        check_trace_document(
            changed,
            expected_counts=TrajectoryCounts(tool_calls=1, tool_results=1, reasoning_steps=1),
            expected_outcome=_RESOLVED,
        )
