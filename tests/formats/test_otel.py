"""OTel GenAI parts/messages/spans and OtelTrace.from_orq."""

# ruff: noqa: S101

from __future__ import annotations

import json
from datetime import timezone
from pathlib import Path
from typing import Any

import pytest

from evaluatorq.formats.otel import (
    OtelGenericPart,
    OtelTextPart,
    OtelToolCallPart,
    OtelToolCallResponsePart,
    OtelTrace,
    flatten_attributes,
    parse_part,
    parse_time,
    span_usage,
)

FIXTURE = Path(__file__).parent / 'fixtures' / 'otel' / 'orq_agent_subagent.json'
REAL_FIXTURE = Path(__file__).parent / 'fixtures' / 'otel' / 'orq_agent_real.json'


def _raw() -> list[dict[str, Any]]:
    return json.loads(FIXTURE.read_text())


@pytest.mark.parametrize(
    ('raw', 'cls'),
    [
        ({'type': 'text', 'content': 'a'}, OtelTextPart),
        ({'type': 'tool_call', 'id': 'c', 'name': 'f', 'arguments': {'x': 1}}, OtelToolCallPart),
    ],
)
def test_known_parts(raw: dict[str, Any], cls: type) -> None:
    assert isinstance(parse_part(raw), cls)


def test_unknown_part_type_kept_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    part = parse_part({'type': 'hologram', 'x': 1})
    assert isinstance(part, OtelGenericPart)
    assert part.model_dump()['x'] == 1
    assert 'hologram' in caplog.text


@pytest.mark.parametrize('kind', ['refusal', 'data'])
def test_orq_refusal_and_data_parts_are_generic(kind: str, caplog: pytest.LogCaptureFixture) -> None:
    assert isinstance(parse_part({'type': kind, 'content': 'no'}), OtelGenericPart)
    assert 'unknown' not in caplog.text.lower()


def test_blob_requires_modality_and_content() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        parse_part({'type': 'blob', 'content': 'AAAA'})


def test_from_orq_builds_tree_and_typed_messages() -> None:
    trace = OtelTrace.from_orq(_raw())
    by_id = {s.span_id: s for s in trace.spans}
    assert [s.span_id for s in trace.roots()] == ['a1']
    assert by_id['c1'].operation == 'chat'
    assert by_id['t1'].operation == 'execute_tool'
    assert [c.span_id for c in trace.children('a1')] == ['c1', 't1', 'c3']
    out = by_id['c1'].output_messages
    assert out is not None
    assert out[0].parts[0].type == 'reasoning'
    assert by_id['c1'].system_instructions is not None
    assert by_id['c1'].start_time is not None
    assert by_id['c1'].start_time.tzinfo is not None


def test_nested_and_flat_attributes_both_accepted() -> None:
    trace = OtelTrace.from_orq(_raw())
    tool = next(s for s in trace.spans if s.span_id == 't1')
    assert tool.attributes['gen_ai.tool.call.id'] == 'call_1'


def test_flatten_attributes() -> None:
    assert flatten_attributes({'gen_ai': {'a': {'b': 1}}, 'x.y': 2}) == {'gen_ai.a.b': 1, 'x.y': 2}


def test_bad_message_attribute_keeps_raw_attribute(caplog: pytest.LogCaptureFixture) -> None:
    span = {'span_id': 's', 'attributes': {'gen_ai.operation.name': 'chat', 'gen_ai.input.messages': '{not json'}}
    trace = OtelTrace.from_orq([span])
    assert trace.spans[0].input_messages is None
    assert trace.spans[0].attributes['gen_ai.input.messages'] == '{not json'
    assert 'gen_ai.input.messages' in caplog.text


def test_json_string_messages_are_parsed() -> None:
    raw = json.dumps([{'role': 'user', 'parts': [{'type': 'text', 'content': 'hi'}]}])
    trace = OtelTrace.from_orq([{'span_id': 's', 'attributes': {'gen_ai.input.messages': raw}}])
    assert trace.spans[0].input_messages is not None


def test_invalid_instruction_parts_are_skipped_and_valid_instruction_reaches_atif(
    caplog: pytest.LogCaptureFixture,
) -> None:
    trace = OtelTrace.from_orq([
        {'span_id': 'root', 'attributes': {'gen_ai.operation.name': 'invoke_agent'}},
        {
            'span_id': 'chat',
            'parent_span_id': 'root',
            'attributes': {
                'gen_ai.operation.name': 'chat',
                'gen_ai.system_instructions': [
                    {'type': 'text', 'content': 'keep this instruction'},
                    {'type': 'text', 'content': 7},
                ],
            },
        },
    ])

    instructions = trace.spans[1].system_instructions
    assert instructions is not None
    assert len(instructions) == 1
    assert trace.to_atif().steps[0].message == 'keep this instruction'
    assert 'Could not parse OTel part 1' in caplog.text


def test_invalid_messages_are_skipped_without_losing_valid_messages(caplog: pytest.LogCaptureFixture) -> None:
    valid = {'role': 'user', 'parts': [{'type': 'text', 'content': 'keep me'}]}
    invalid = {'role': 3, 'parts': [{'type': 'text', 'content': 'bad'}]}
    trace = OtelTrace.from_orq([{'span_id': 's', 'attributes': {
        'gen_ai.input.messages': [valid, invalid, valid]}}])
    parsed = trace.spans[0].input_messages
    assert parsed is not None and [getattr(m.parts[0], 'content', None) for m in parsed] == ['keep me', 'keep me']
    assert 'skipping it' in caplog.text


def test_all_invalid_messages_keep_raw_attribute(caplog: pytest.LogCaptureFixture) -> None:
    trace = OtelTrace.from_orq([{'span_id': 's', 'attributes': {
        'gen_ai.input.messages': [{'role': 3, 'parts': []}]}}])
    assert trace.spans[0].input_messages is None
    assert 'No valid OTel messages' in caplog.text


def test_usage_reads_dotted_keys_and_legacy_names() -> None:
    trace = OtelTrace.from_orq(_raw())
    usage = span_usage(next(s for s in trace.spans if s.span_id == 'c1'))
    assert (usage.input_tokens, usage.output_tokens, usage.cached_tokens) == (100, 20, 40)
    legacy = OtelTrace.from_orq(
        [{'span_id': 'x', 'attributes': {'gen_ai.usage.prompt_tokens': 3, 'gen_ai.usage.completion_tokens': 4}}]
    )
    assert span_usage(legacy.spans[0]).input_tokens == 3


@pytest.mark.parametrize(
    ('value', 'iso'),
    [('2026-04-20T10:00:00Z', '2026-04-20T10:00:00+00:00'), (1_776_679_200, '2026-04-20T10:00:00+00:00'),
     (1_776_679_200_000, '2026-04-20T10:00:00+00:00')],
)
def test_epoch_millis_and_seconds_parse(value: object, iso: str) -> None:
    parsed = parse_time(value)
    assert parsed is not None
    assert parsed.astimezone(timezone.utc).isoformat() == iso


def test_unparseable_time_is_none_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    assert parse_time('soon') is None
    assert 'soon' in caplog.text


def test_status_and_error_type() -> None:
    span = {'span_id': 's', 'status': 'ERROR', 'attributes': {'error.type': 'Timeout'}}
    parsed = OtelTrace.from_orq([span]).spans[0]
    assert (parsed.status, parsed.error_type) == ('error', 'Timeout')


def test_span_without_id_gets_synthetic_id() -> None:
    trace = OtelTrace.from_orq([{'name': 'x'}, {'name': 'y'}])
    assert len({s.span_id for s in trace.spans}) == 2


def test_orphan_parent_is_treated_as_root() -> None:
    trace = OtelTrace.from_orq([{'span_id': 's', 'parent_span_id': 'gone'}])
    assert [s.span_id for s in trace.roots()] == ['s']


def _real() -> list[dict[str, Any]]:
    return json.loads(REAL_FIXTURE.read_text())['spans']


def test_real_orq_trace_tree_and_conversation() -> None:
    trace = OtelTrace.from_orq(_real())
    [root] = trace.roots()
    assert root.operation == 'AgentInvoke'
    assert root.trace_id == '01M3RHSVX9X29KGKFVQREXFF74'
    [agent] = trace.children(root.span_id)
    assert agent.operation == 'invoke_agent'
    chats = trace.children(agent.span_id)
    assert [c.operation for c in chats] == ['chat', 'chat', 'chat']
    last = chats[-1]
    assert last.input_messages is not None
    assert last.output_messages is not None
    conversation = last.input_messages + last.output_messages
    assert [m.role for m in conversation] == ['system', 'user', 'assistant', 'tool', 'assistant', 'tool', 'assistant']
    first_call, sub_call = conversation[2].parts[0], conversation[4].parts[0]
    assert isinstance(first_call, OtelToolCallPart)
    assert (first_call.name, first_call.arguments) == ('retrieve_agents', {'include_descriptions': True})
    assert isinstance(sub_call, OtelToolCallPart)
    assert sub_call.name == 'call_sub_agent'
    for call, result in ((first_call, conversation[3].parts[0]), (sub_call, conversation[5].parts[0])):
        assert isinstance(result, OtelToolCallResponsePart)
        assert result.id == call.id
    answer = conversation[6].parts[0]
    assert isinstance(answer, OtelTextPart)
    assert answer.content.startswith('Child says: ')
    assert 'gen_ai.input.messages' not in last.attributes
    assert 'gen_ai.input.messages.0.role' not in last.attributes


def test_real_orq_usage_matches_span_summaries() -> None:
    raw = _real()
    trace = OtelTrace.from_orq(raw)
    chat_ids = {s['span_id'] for s in raw if s['type'] == 'span.chat_completion'}
    chats = [s for s in trace.spans if s.span_id in chat_ids]
    usages = [span_usage(s) for s in chats]
    assert sum(u.input_tokens or 0 for u in usages) == sum(s['usage']['prompt_tokens'] for s in raw if s['span_id'] in chat_ids)
    assert sum(u.output_tokens or 0 for u in usages) == sum(
        s['usage']['completion_tokens'] for s in raw if s['span_id'] in chat_ids
    )
    by_id = {s['span_id']: s for s in raw}
    assert [u.cost_usd for u in usages] == [by_id[s.span_id]['cost']['total'] for s in chats]


def test_real_orq_root_singular_input_message() -> None:
    [root] = OtelTrace.from_orq(_real()).roots()
    assert root.input_messages is not None
    [message] = root.input_messages
    assert message.role == 'user'
    part = message.parts[0]
    assert isinstance(part, OtelTextPart)
    assert part.content.startswith('What is the date today')


@pytest.mark.parametrize('container', ['list', 'single', 'indexed'])
def test_router_root_direct_messages_parse_a2a_parts(container: str, caplog: pytest.LogCaptureFixture) -> None:
    message = {'role': 'agent', 'parts': [{'kind': 'text', 'text': 'Hello.'}]}
    value: Any = [message] if container == 'list' else message if container == 'single' else {'0': message}
    [span] = OtelTrace.from_orq([{'span_id': 'root', 'type': 'trace', 'attributes': {
        'gen_ai.operation.name': 'chat', 'gen_ai.output': value}}]).spans
    assert span.output_messages is not None
    assert [parsed.role for parsed in span.output_messages] == ['agent']
    assert span.output_messages[0].parts == [OtelTextPart(type='text', content='Hello.')]
    assert 'Unknown OTel message part type' not in caplog.text


def test_usage_falls_back_to_span_summary() -> None:
    span = {
        'span_id': 's',
        'usage': {'prompt_tokens': 7, 'completion_tokens': 2, 'prompt_cached_tokens': 1, 'completion_reasoning_tokens': 3},
        'cost': {'total': 0.5},
    }
    usage = span_usage(OtelTrace.from_orq([span]).spans[0])
    assert (usage.input_tokens, usage.output_tokens, usage.cached_tokens, usage.reasoning_tokens, usage.cost_usd) == (
        7,
        2,
        1,
        3,
        0.5,
    )


def test_unreported_usage_stays_none() -> None:
    usage = span_usage(OtelTrace.from_orq([{'span_id': 's'}]).spans[0])
    assert usage.model_dump() == dict.fromkeys(usage.model_dump())


@pytest.mark.parametrize(
    ('attributes', 'kind', 'operation'),
    [
        ({'gen_ai.operation.name': 'chat-completion'}, None, 'chat'),
        ({}, 'span.chat_completion', 'chat'),
        ({}, 'span.agent_execution', 'invoke_agent'),
        ({'gen_ai.operation.name': 'AgentInvoke'}, 'trace', 'AgentInvoke'),
    ],
)
def test_orq_operation_names_are_normalised(attributes: dict[str, Any], kind: str | None, operation: str) -> None:
    span = OtelTrace.from_orq([{'span_id': 's', 'type': kind, 'attributes': attributes}]).spans[0]
    assert span.operation == operation


def test_chat_completions_message_with_json_string_arguments() -> None:
    messages = {
        '1': {'role': 'tool', 'tool_call_id': 'c1', 'content': 'sunny'},
        '0': {
            'role': 'assistant',
            'content': 'Checking.',
            'tool_calls': {'0': {'id': 'c1', 'type': 'function', 'function': {'name': 'weather', 'arguments': '{"city": "Oslo"}'}}},
        },
    }
    span = OtelTrace.from_orq([{'span_id': 's', 'attributes': {'gen_ai.input.messages': messages}}]).spans[0]
    assert span.input_messages is not None
    assistant, tool = span.input_messages
    assert [p.type for p in assistant.parts] == ['text', 'tool_call']
    call = assistant.parts[1]
    assert isinstance(call, OtelToolCallPart)
    assert call.arguments == {'city': 'Oslo'}
    result = tool.parts[0]
    assert isinstance(result, OtelToolCallResponsePart)
    assert (result.id, result.response) == ('c1', 'sunny')


@pytest.mark.parametrize('kind', [{'a': 1}, ['x'], 3])
def test_unhashable_or_non_string_part_type_is_kept_generic(kind: Any, caplog: pytest.LogCaptureFixture) -> None:
    span = {'span_id': 's', 'attributes': {'gen_ai.operation.name': 'chat', 'gen_ai.input.messages': [
        {'role': 'user', 'parts': [{'type': kind}, {'type': 'text', 'content': 'hi'}]}]}}
    [parsed] = OtelTrace.from_orq([span]).spans
    assert parsed.input_messages is not None
    parts = parsed.input_messages[0].parts
    assert isinstance(parts[0], OtelGenericPart) and parts[0].type == str(kind)
    assert isinstance(parts[1], OtelTextPart)
    assert 'Unknown OTel message part type' in caplog.text


def test_non_dict_part_is_kept_generic() -> None:
    part = parse_part('loose text')
    assert isinstance(part, OtelGenericPart)
    assert part.model_dump() == {'type': 'unknown', 'value': 'loose text'}


def test_nanosecond_iso_time_parses() -> None:
    parsed = parse_time('2026-04-20T10:00:00.123456789Z')
    assert parsed is not None and parsed.microsecond == 123456 and parsed.tzinfo is not None


@pytest.mark.parametrize('value', [1e20, float('nan')])
def test_out_of_range_epoch_is_unset_with_warning(value: float, caplog: pytest.LogCaptureFixture) -> None:
    assert parse_time(value) is None
    assert 'out of range' in caplog.text


def test_spans_from_two_trace_ids_are_rejected() -> None:
    raw = _raw()
    raw[0]['trace_id'] = 'trace-a'
    raw[1]['trace_id'] = 'trace-b'
    with pytest.raises(ValueError, match='2 trace ids'):
        OtelTrace.from_orq(raw)
    spans = OtelTrace.from_orq(_raw()).spans
    with pytest.raises(ValueError, match='2 trace ids'):
        OtelTrace(spans=[spans[0].model_copy(update={'trace_id': 'x'}), spans[1].model_copy(update={'trace_id': 'y'})])


def test_several_roots_in_one_trace_id_are_allowed() -> None:
    raw = [dict(span, trace_id='t') for span in _raw()]
    raw.append(dict(raw[4], span_id='c9', parent_span_id=None))
    raw.append({'span_id': 'no-trace', 'attributes': {}})
    assert len(OtelTrace.from_orq(raw).roots()) == 3
