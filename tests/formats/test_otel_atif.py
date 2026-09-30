"""OTel <-> ATIF conversion."""

# ruff: noqa: S101

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import pytest

from evaluatorq.contracts import Message
from evaluatorq.formats.atif import (
    AtifAgent,
    AtifObservation,
    AtifObservationResult,
    AtifStep,
    AtifSubagentRef,
    AtifToolCall,
    AtifTrajectory,
)
from evaluatorq.formats._shared import RAW_ARGUMENTS_EXTRA_KEY
from evaluatorq.formats.chat import ChatConversation
from evaluatorq.formats.otel import OtelTrace

FIX = Path(__file__).parent / 'fixtures'


def _trace() -> OtelTrace:
    return OtelTrace.from_orq(json.loads((FIX / 'otel' / 'orq_agent_subagent.json').read_text()))


def _real() -> OtelTrace:
    return OtelTrace.from_orq(json.loads((FIX / 'otel' / 'orq_agent_real.json').read_text())['spans'])


def test_otel_to_atif_step_layout() -> None:
    traj = _trace().to_atif(agent_name='planner')
    assert traj.session_id == 'sess-1'
    assert [s.source for s in traj.steps] == ['system', 'user', 'agent', 'agent']
    step = traj.steps[2]
    assert step.reasoning_content == 'I should delegate.'
    assert step.model_name == 'gpt-x' and step.llm_call_count == 1
    assert step.metrics is not None and (step.metrics.prompt_tokens, step.metrics.cached_tokens) == (100, 40)
    assert step.tool_calls is not None and step.tool_calls[0].arguments == {'topic': 'ATIF'}


def test_tool_result_attaches_to_issuing_step_with_subagent_ref() -> None:
    traj = _trace().to_atif()
    obs = traj.steps[2].observation
    assert obs is not None
    result = obs.results[0]
    assert result.source_call_id == 'call_1' and result.content == 'done'
    assert result.subagent_trajectory_ref is not None
    ref = result.subagent_trajectory_ref[0]
    assert traj.subagent_trajectories is not None
    assert traj.subagent_trajectories[0].trajectory_id == ref.trajectory_id
    assert traj.subagent_trajectories[0].agent.name == 'researcher'


def test_step_extra_follows_nemo_layout() -> None:
    extra = _trace().to_atif().steps[2].extra
    assert extra is not None
    assert extra['ancestry']['function_id'] == 'c1' and extra['ancestry']['parent_id'] == 'a1'
    assert extra['invocation']['status'] == 'ok' or extra['invocation']['status'] == 'unset'


def test_parent_usage_is_not_summed_into_final_metrics() -> None:
    raw = json.loads((FIX / 'otel' / 'orq_agent_subagent.json').read_text())
    raw[0]['attributes']['gen_ai.usage.input_tokens'] = 9999
    fm = OtelTrace.from_orq(raw).to_atif().final_metrics
    assert fm is not None and fm.total_prompt_tokens == 100 + 150


def test_atif_to_otel_tree_shape() -> None:
    traj = AtifTrajectory.from_json((FIX / 'atif' / 'phoenix_v17_embedded_subagents.json').read_text())
    trace = traj.to_otel()
    ops = [s.operation for s in trace.spans]
    assert ops.count('invoke_agent') == 2
    root = trace.roots()[0]
    assert root.operation == 'invoke_agent'
    kinds = {s.operation for s in trace.children(root.span_id)}
    assert kinds <= {'chat', 'execute_tool', 'invoke_agent'}
    assert len({s.trace_id for s in trace.spans}) == 1


def test_conversion_is_deterministic() -> None:
    traj = AtifTrajectory.from_json((FIX / 'atif' / 'phoenix_v17_embedded_subagents.json').read_text())
    assert traj.to_otel().model_dump_json() == traj.to_otel().model_dump_json()


def test_subagent_ids_differ_from_parent() -> None:
    traj = AtifTrajectory.from_json((FIX / 'atif' / 'phoenix_v17_embedded_subagents.json').read_text())
    ids = [s.span_id for s in traj.to_otel().spans]
    assert len(ids) == len(set(ids))


def test_otel_atif_otel_preserves_tree_and_tool_linkage() -> None:
    once = _trace().to_atif().to_otel()
    twice = once.to_atif().to_otel()
    shape = lambda t: sorted((s.operation, s.name) for s in t.spans)  # noqa: E731
    assert shape(once) == shape(twice)
    tools = [s for s in once.spans if s.operation == 'execute_tool']
    assert tools[0].attributes['gen_ai.tool.call.id'] == 'call_1'


def test_two_roots_warns(caplog: pytest.LogCaptureFixture) -> None:
    raw: list[dict[str, Any]] = json.loads((FIX / 'otel' / 'orq_agent_subagent.json').read_text())
    extra_root = dict(raw[4], span_id='c9', parent_span_id=None)
    OtelTrace.from_orq([*raw, extra_root]).to_atif()
    assert 'roots' in caplog.text


def test_no_chat_span_raises() -> None:
    with pytest.raises(ValueError, match='no chat spans'):
        OtelTrace.from_orq([{'span_id': 'a', 'attributes': {'gen_ai.operation.name': 'invoke_agent'}}]).to_atif()


def test_chat_span_without_output_gives_no_agent_step() -> None:
    span = {'span_id': 'c', 'attributes': {'gen_ai.operation.name': 'chat', 'gen_ai.input.messages': [
        {'role': 'user', 'parts': [{'type': 'text', 'content': 'hi'}]}]}}
    traj = OtelTrace.from_orq([span]).to_atif()
    assert [s.source for s in traj.steps] == ['user']


def test_per_turn_chat_inputs_are_not_dropped() -> None:
    raw = [
        _chat('c1', [_text('user', 'first')], [_text('assistant', 'answer one')]),
        _chat('c2', [_text('user', 'second')], [_text('assistant', 'answer two')]),
    ]
    traj = OtelTrace.from_orq(raw).to_atif()
    assert [(step.source, step.message) for step in traj.steps] == [
        ('user', 'first'), ('agent', 'answer one'), ('user', 'second'), ('agent', 'answer two')
    ]


def test_cumulative_chat_input_without_prior_output_keeps_new_turn() -> None:
    raw = [
        _chat('c1', [_text('user', 'first')], [_text('assistant', 'answer one')]),
        _chat('c2', [_text('user', 'first'), _text('user', 'second')], [_text('assistant', 'answer two')]),
    ]
    traj = OtelTrace.from_orq(raw).to_atif()
    assert [(step.source, step.message) for step in traj.steps] == [
        ('user', 'first'), ('agent', 'answer one'), ('user', 'second'), ('agent', 'answer two')
    ]


def test_changed_system_instructions_become_a_later_system_step() -> None:
    first = _chat('c1', [_text('user', 'first')], [_text('assistant', 'answer one')])
    second = _chat('c2', [_text('user', 'second')], [_text('assistant', 'answer two')])
    first['attributes']['gen_ai.system_instructions'] = [{'type': 'text', 'content': 'rule one'}]
    second['attributes']['gen_ai.system_instructions'] = [{'type': 'text', 'content': 'rule two'}]
    raw = [first, second]
    traj = OtelTrace.from_orq(raw).to_atif()
    assert [(step.source, step.message) for step in traj.steps] == [
        ('system', 'rule one'), ('user', 'first'), ('agent', 'answer one'),
        ('system', 'rule two'), ('user', 'second'), ('agent', 'answer two'),
    ]


def test_error_tool_span_records_error_type() -> None:
    raw = json.loads((FIX / 'otel' / 'orq_agent_subagent.json').read_text())
    raw[2]['status'] = 'error'
    raw[2]['attributes'] = {'gen_ai.operation.name': 'execute_tool', 'gen_ai.tool.call.id': 'call_1',
                            'gen_ai.tool.call.result': 'boom', 'error.type': 'ToolError'}
    obs = OtelTrace.from_orq(raw).to_atif().steps[2].observation
    assert obs is not None
    assert obs.results[0].extra == {'error_type': 'ToolError'}


# --- the real Orq export: AgentInvoke wrapper, system prompt as a message, results only in later inputs ---


def test_real_orq_trace_scope_and_steps(caplog: pytest.LogCaptureFixture) -> None:
    traj = _real().to_atif()
    assert 'roots' not in caplog.text and 'scope' not in caplog.text
    assert traj.session_id == '01M3RHSVX9X29KGKFVQREXFF74'
    assert traj.agent.name == 'trace-probe-parent'
    assert [s.source for s in traj.steps] == ['system', 'user', 'agent', 'agent', 'agent']
    assert isinstance(traj.steps[0].message, str) and 'You never know the date yourself' in traj.steps[0].message
    assert traj.steps[1].message == 'What is the date today, and what day of the week is it?'
    assert traj.subagent_trajectories is None


def test_real_orq_trace_results_come_from_later_chat_inputs() -> None:
    steps = _real().to_atif().steps
    for step, name in ((steps[2], 'retrieve_agents'), (steps[3], 'call_sub_agent')):
        assert step.tool_calls is not None and [c.function_name for c in step.tool_calls] == [name]
        assert step.observation is not None and len(step.observation.results) == 1
        result = step.observation.results[0]
        assert result.source_call_id == step.tool_calls[0].tool_call_id
        assert isinstance(result.content, str) and result.content
    last = steps[4]
    assert isinstance(last.message, str) and last.message.startswith('Child says: ')
    assert last.tool_calls is None and last.observation is None
    assert steps[3].observation is not None and '"result"' in str(steps[3].observation.results[0].content)


def test_real_orq_trace_metrics_come_from_chat_spans_only() -> None:
    traj = _real().to_atif()
    assert [s.metrics.prompt_tokens for s in traj.steps if s.metrics] == [305, 363, 497]
    fm = traj.final_metrics
    assert fm is not None
    assert (fm.total_prompt_tokens, fm.total_completion_tokens, fm.total_steps) == (1165, 78, 5)
    assert fm.total_cost_usd == pytest.approx(4e-05 + 5.53e-05 + 6.02e-05)
    assert traj.steps[2].model_name == 'gpt-6-luna'
    assert traj.steps[2].extra is not None and traj.steps[2].extra['finish_reasons'] == ['tool_calls']


def test_real_orq_trace_round_trips_through_otel() -> None:
    first = _real().to_atif()
    second = first.to_otel().to_atif()
    assert [s.source for s in second.steps] == [s.source for s in first.steps]
    assert [s.message for s in second.steps] == [s.message for s in first.steps]
    assert [s.observation for s in second.steps] == [s.observation for s in first.steps]
    assert second.final_metrics == first.final_metrics


# --- edge cases ---


def _chat(span_id: str, inputs: list[dict[str, Any]], outputs: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {'span_id': span_id, **extra, 'attributes': {
        'gen_ai.operation.name': 'chat', 'gen_ai.input.messages': inputs, 'gen_ai.output.messages': outputs}}


def _text(role: str, text: str) -> dict[str, Any]:
    return {'role': role, 'parts': [{'type': 'text', 'content': text}]}


def test_history_before_first_chat_span_is_warned_and_dropped(caplog: pytest.LogCaptureFixture) -> None:
    span = _chat('c', [_text('user', 'a'), _text('assistant', 'b'), _text('user', 'c')], [_text('assistant', 'd')])
    traj = OtelTrace.from_orq([span]).to_atif()
    assert [(s.source, s.message) for s in traj.steps] == [('user', 'a'), ('user', 'c'), ('agent', 'd')]
    assert 'history of 1 messages before the first chat span is not converted' in caplog.text


def test_call_without_id_gets_stable_id_and_no_result_entry() -> None:
    call = {'role': 'assistant', 'parts': [{'type': 'tool_call', 'name': 'f', 'arguments': 'not json'}]}
    raw = [_chat('c', [_text('user', 'a')], [call])]
    first = OtelTrace.from_orq(raw).to_atif()
    step = first.steps[1]
    assert step.tool_calls is not None
    assert step.tool_calls[0].tool_call_id.startswith('call_') and len(step.tool_calls[0].tool_call_id) == 21
    assert step.tool_calls[0].arguments == {}
    assert step.tool_calls[0].extra == {RAW_ARGUMENTS_EXTRA_KEY: 'not json'}
    assert step.observation is None
    assert OtelTrace.from_orq(raw).to_atif() == first


def test_raw_arguments_are_written_back_as_the_string() -> None:
    traj = AtifTrajectory(
        schema_version='ATIF-v1.7',
        agent=AtifAgent(name='a', version='1'),
        steps=[
            AtifStep(step_id=1, source='user', message='hi'),
                AtifStep(step_id=2, source='agent', message='', tool_calls=[
                    AtifToolCall(tool_call_id='c1', function_name='f', arguments={},
                                 extra={RAW_ARGUMENTS_EXTRA_KEY: 'not json'})]),
        ],
    )
    tool = next(s for s in traj.to_otel().spans if s.operation == 'execute_tool')
    assert tool.attributes['gen_ai.tool.call.arguments'] == 'not json'
    assert 'gen_ai.tool.call.result' not in tool.attributes


def test_valid_arguments_matching_legacy_raw_sentinel_round_trip() -> None:
    arguments = {'_raw': {'evaluatorq_raw_value': 'literal'}}
    raw = [_chat('c', [_text('user', 'a')], [{
        'role': 'assistant',
        'parts': [{'type': 'tool_call', 'id': 'c1', 'name': 'f', 'arguments': arguments}],
    }])]
    round_trip = OtelTrace.from_orq(raw).to_atif().to_otel().to_atif()
    assert round_trip.steps[1].tool_calls is not None
    call = round_trip.steps[1].tool_calls[0]
    assert call.arguments == arguments
    assert call.extra is None


def test_multiple_results_for_one_call_round_trip() -> None:
    traj = AtifTrajectory(
        agent=AtifAgent(name='a', version='1'),
        steps=[
            AtifStep(step_id=1, source='user', message='go'),
            AtifStep(
                step_id=2,
                source='agent',
                message='',
                tool_calls=[AtifToolCall(tool_call_id='c1', function_name='f', arguments={})],
                observation=AtifObservation(results=[
                    AtifObservationResult(source_call_id='c1', content='first'),
                    AtifObservationResult(source_call_id='c1', content='second', extra={'part': 2}),
                ]),
            ),
        ],
    )
    round_trip = traj.to_otel().to_atif()
    assert round_trip.steps[1].observation is not None
    assert traj.steps[1].observation is not None
    assert round_trip.steps[1].observation.results == traj.steps[1].observation.results


def test_reused_call_id_keeps_subagent_under_referencing_step() -> None:
    subagents = [
        AtifTrajectory(trajectory_id='sub1', agent=AtifAgent(name='sub1', version='1'),
                       steps=[AtifStep(step_id=1, source='agent', message='one')]),
        AtifTrajectory(trajectory_id='sub2', agent=AtifAgent(name='sub2', version='1'),
                       steps=[AtifStep(step_id=1, source='agent', message='two')]),
    ]
    traj = AtifTrajectory(
        agent=AtifAgent(name='a', version='1'),
        steps=[
            AtifStep(step_id=1, source='user', message='go'),
            AtifStep(step_id=2, source='agent', message='',
                     tool_calls=[AtifToolCall(tool_call_id='same', function_name='f', arguments={})],
                     observation=AtifObservation(results=[AtifObservationResult(
                         source_call_id='same', subagent_trajectory_ref=[AtifSubagentRef(trajectory_id='sub1')]
                     )])),
            AtifStep(step_id=3, source='agent', message='',
                     tool_calls=[AtifToolCall(tool_call_id='same', function_name='f', arguments={})],
                     observation=AtifObservation(results=[AtifObservationResult(
                         source_call_id='same', subagent_trajectory_ref=[AtifSubagentRef(trajectory_id='sub2')]
                     )])),
        ],
        subagent_trajectories=subagents,
    )
    trace = traj.to_otel()
    tools = [span for span in trace.spans if span.operation == 'execute_tool']
    subs = [span for span in trace.spans if span.operation == 'invoke_agent' and span.attributes.get('gen_ai.agent.name') != 'a']
    assert len(subs) == 2
    assert [span.parent_span_id for span in subs] == [span.span_id for span in tools]

    round_trip = trace.to_atif()
    assert round_trip.subagent_trajectories is not None
    names_by_id = {sub.trajectory_id: sub.agent.name for sub in round_trip.subagent_trajectories}
    step_refs = []
    for step in round_trip.steps:
        refs = step.observation.results[0].subagent_trajectory_ref if step.observation else None
        step_refs.append(names_by_id[refs[0].trajectory_id] if refs else None)
    assert step_refs == [None, 'sub1', 'sub2']


def test_atif_to_otel_keeps_history_usage_and_times() -> None:
    traj = _trace().to_atif()
    trace = traj.to_otel()
    chats = [s for s in trace.spans if s.operation == 'chat' and s.parent_span_id == trace.roots()[0].span_id]
    assert [len(c.input_messages or []) for c in chats] == [1, 3]
    assert chats[1].input_messages is not None and chats[1].input_messages[2].role == 'tool'
    assert chats[0].system_instructions is not None
    assert chats[0].attributes['gen_ai.usage.input_tokens'] == 100
    assert chats[0].attributes['gen_ai.usage.cache_read.input_tokens'] == 40
    assert chats[0].attributes['gen_ai.response.finish_reasons'] == ['tool_calls']
    assert chats[0].start_time is not None and chats[0].end_time is not None
    assert chats[0].output_messages is not None
    assert [p.type for p in chats[0].output_messages[0].parts] == ['reasoning', 'tool_call']


def test_unmatched_tool_results_are_warned_and_kept(caplog: pytest.LogCaptureFixture) -> None:
    call = {'role': 'assistant', 'parts': [{'type': 'tool_call', 'id': 'c1', 'name': 'f', 'arguments': {}}]}
    first_in = [_text('user', 'a')]
    orphan = {'role': 'tool', 'parts': [{'type': 'tool_call_response', 'id': 'nope', 'response': 'R'}]}
    matched = {'role': 'tool', 'parts': [{'type': 'tool_call_response', 'id': 'c1', 'response': 'ok'}]}
    raw = [
        {'span_id': 'a', 'attributes': {'gen_ai.operation.name': 'invoke_agent'}},
        _chat('c', first_in, [call], parent_span_id='a', started_at=1),
        {'span_id': 't', 'parent_span_id': 'a', 'started_at': 2, 'attributes': {
            'gen_ai.operation.name': 'execute_tool', 'gen_ai.tool.call.result': 'lost'}},
        _chat('d', [*first_in, call, matched, orphan], [_text('assistant', 'done')], parent_span_id='a', started_at=3),
    ]
    step = OtelTrace.from_orq(raw).to_atif().steps[1]
    assert step.observation is not None
    results = [(r.source_call_id, r.content, r.extra) for r in step.observation.results]
    assert results == [('c1', 'ok', None), (None, 'lost', None), (None, 'R', {'orphan_call_id': 'nope'})]
    assert '2 tool results match no tool call' in caplog.text


def test_reused_tool_call_id_consumes_execute_spans_in_order() -> None:
    call = {'role': 'assistant', 'parts': [{'type': 'tool_call', 'id': 'same', 'name': 'f', 'arguments': {}}]}
    raw = [
        {'span_id': 'a', 'attributes': {'gen_ai.operation.name': 'invoke_agent'}},
        _chat('c1', [_text('user', 'first')], [call], parent_span_id='a', started_at=1),
        {'span_id': 't1', 'parent_span_id': 'a', 'started_at': 2, 'attributes': {
            'gen_ai.operation.name': 'execute_tool', 'gen_ai.tool.call.id': 'same', 'gen_ai.tool.call.result': 'first result'}},
        _chat('c2', [_text('user', 'second')], [call], parent_span_id='a', started_at=3),
        {'span_id': 't2', 'parent_span_id': 'a', 'started_at': 4, 'attributes': {
            'gen_ai.operation.name': 'execute_tool', 'gen_ai.tool.call.id': 'same', 'gen_ai.tool.call.result': 'second result'}},
    ]
    agents = [step for step in OtelTrace.from_orq(raw).to_atif().steps if step.source == 'agent']
    assert [step.observation.results[0].content for step in agents if step.observation] == [
        'first result', 'second result']


def test_atif_to_otel_warns_about_dropped_results(caplog: pytest.LogCaptureFixture) -> None:
    traj = AtifTrajectory.model_validate({
        'schema_version': 'ATIF-v1.7', 'trajectory_id': 't', 'agent': {'name': 'a', 'version': '1'},
        'steps': [
            {'step_id': 1, 'source': 'system', 'message': 's', 'observation': {'results': [{'content': 'x'}]}},
            {'step_id': 2, 'source': 'agent', 'message': 'm', 'observation': {'results': [
                {'content': 'y'}, {'content': 'z'},
                {'subagent_trajectory_ref': [{'trajectory_path': 'elsewhere.json'}]}]}},
        ]})
    traj.to_otel()
    assert 'dropping 2 observation results with no source_call_id' in caplog.text
    assert 'dropping 1 observation results on user or system steps' in caplog.text
    assert 'dropping 1 references to subagents that are not embedded' in caplog.text


def test_image_result_content_becomes_a_warned_marker(caplog: pytest.LogCaptureFixture) -> None:
    traj = AtifTrajectory.model_validate({
        'schema_version': 'ATIF-v1.7', 'agent': {'name': 'a', 'version': '1'},
        'steps': [{'step_id': 1, 'source': 'agent', 'message': '', 'tool_calls': [
            {'tool_call_id': 'c', 'function_name': 'f', 'arguments': {}}],
            'observation': {'results': [{'source_call_id': 'c', 'content': [
                {'type': 'text', 'text': 'see'},
                {'type': 'image', 'source': {'media_type': 'image/png', 'path': 'p.png'}}]}]}}]})
    tool = next(s for s in traj.to_otel().spans if s.operation == 'execute_tool')
    assert tool.attributes['gen_ai.tool.call.result'] == 'see\n[image: p.png]'
    assert caplog.text.count('p.png') == 1


# --- user and system steps the OTel trace has to carry ---


def _steps(*specs: tuple[Literal['system', 'user', 'agent'], str]) -> AtifTrajectory:
    return AtifTrajectory(
        session_id='s',
        agent=AtifAgent(name='a', version='1'),
        steps=[AtifStep(step_id=n + 1, source=source, message=text) for n, (source, text) in enumerate(specs)],
    )


def _sig(traj: AtifTrajectory) -> list[tuple[str, Any]]:
    return [(s.source, s.message) for s in traj.steps]


def test_steps_after_the_last_agent_step_round_trip() -> None:
    traj = _steps(('user', 'q'), ('agent', 'a'), ('user', 'thanks'), ('system', 'wrap up'))
    trace = traj.to_otel()
    tail = [s for s in trace.spans if s.operation == 'chat'][-1]
    assert tail.output_messages is None
    assert _sig(trace.to_atif()) == _sig(traj)


def test_system_step_after_the_first_agent_step_keeps_its_place() -> None:
    traj = _steps(('system', 'be brief'), ('user', 'q'), ('agent', 'a'), ('system', 'now be formal'),
                  ('user', 'q2'), ('agent', 'a2'))
    assert _sig(traj.to_otel().to_atif()) == _sig(traj)


def test_conversation_with_no_agent_turn_round_trips() -> None:
    chat = ChatConversation(messages=[Message(role='system', content='be kind'), Message(role='user', content='hi')])
    trace = chat.to_otel()
    assert [s.operation for s in trace.spans] == ['invoke_agent', 'chat']
    assert _sig(trace.to_atif()) == [('system', 'be kind'), ('user', 'hi')]


def test_developer_step_and_empty_message_round_trip() -> None:
    traj = AtifTrajectory(
        agent=AtifAgent(name='a', version='1'),
        steps=[
            AtifStep(step_id=1, source='system', message='dev', extra={'original_role': 'developer'}),
            AtifStep(step_id=2, source='user', message=''),
            AtifStep(step_id=3, source='agent', message='ok'),
        ],
    )
    back = traj.to_otel().to_atif()
    assert _sig(back) == _sig(traj)
    assert back.steps[0].extra == {'original_role': 'developer'}


def test_multi_part_messages_join_with_newlines_in_every_role() -> None:
    parts = [{'type': 'text', 'content': 'a'}, {'type': 'text', 'content': 'b'}]
    span = _chat('c', [{'role': 'user', 'parts': parts}], [{'role': 'assistant', 'parts': parts}])
    span['attributes']['gen_ai.system_instructions'] = parts
    assert _sig(OtelTrace.from_orq([span]).to_atif()) == [('system', 'a\nb'), ('user', 'a\nb'), ('agent', 'a\nb')]


def _extras_traj() -> AtifTrajectory:
    return AtifTrajectory.model_validate({
        'schema_version': 'ATIF-v1.7',
        'session_id': 's-extra',
        'agent': {'name': 'a', 'version': '1'},
        'extra': {'run': {'seed': 7}},
        'final_metrics': {'total_steps': 2, 'total_cost_usd': 0.5, 'extra': {'judge': 'j1'}},
        'steps': [
            {'step_id': 1, 'source': 'user', 'message': 'go'},
            {
                'step_id': 2,
                'source': 'agent',
                'message': '',
                'extra': {'custom': [1, 2], 'finish_reasons': ['tool_calls']},
                'tool_calls': [
                    {'tool_call_id': 'c1', 'function_name': 'f', 'arguments': {}, 'extra': {'origin': 'mcp'}}
                ],
                'observation': {'results': [{'source_call_id': 'c1', 'content': 'ok', 'extra': {'latency': 3}}]},
            },
        ],
    })


def test_atif_extras_and_final_metrics_round_trip_through_otel() -> None:
    src = _extras_traj()
    trace = src.to_otel()
    root = trace.roots()[0]
    assert json.loads(root.attributes['evaluatorq.atif.trajectory.extra']) == {'run': {'seed': 7}}
    chat = next(s for s in trace.spans if s.operation == 'chat')
    assert json.loads(chat.attributes['evaluatorq.atif.step.extra']) == {'custom': [1, 2]}
    back = trace.to_atif()
    assert back.extra == src.extra
    assert back.final_metrics == src.final_metrics
    step = back.steps[1]
    assert step.extra is not None and step.extra['custom'] == [1, 2] and 'ancestry' in step.extra
    assert step.tool_calls is not None and step.tool_calls[0].extra == {'origin': 'mcp'}
    assert step.observation is not None and step.observation.results[0].extra == {'latency': 3}


def test_empty_extras_write_no_attribute() -> None:
    trace = _steps(('user', 'hi'), ('agent', 'yo')).to_otel()
    assert not any(key.startswith('evaluatorq.atif.') for span in trace.spans for key in span.attributes)


def test_values_the_trace_yields_win_over_carried_ones(caplog: pytest.LogCaptureFixture) -> None:
    src = _extras_traj()
    trace = src.to_otel()
    chat = next(s for s in trace.spans if s.operation == 'chat')
    chat.attributes['evaluatorq.atif.step.extra'] = json.dumps({'custom': 1, 'finish_reasons': ['stop']})
    step = trace.to_atif().steps[1]
    assert step.extra is not None and step.extra['finish_reasons'] == ['tool_calls'] and step.extra['custom'] == 1
    assert 'finish_reasons' in caplog.text


def test_object_valued_invocation_keeps_times_and_status() -> None:
    from types import SimpleNamespace

    invocation = SimpleNamespace(start_timestamp=1_700_000_000.0, end_timestamp=1_700_000_002.0, status='error')
    traj = AtifTrajectory(
        session_id='s-obj',  # the id seed would otherwise JSON-dump the object
        agent=AtifAgent(name='a', version='1'),
        steps=[AtifStep(step_id=1, source='agent', message='x', extra={'invocation': invocation})],
    )
    chat = next(s for s in traj.to_otel().spans if s.operation == 'chat')
    assert chat.start_time is not None and chat.start_time.timestamp() == 1_700_000_000.0
    assert chat.end_time is not None and chat.end_time.timestamp() == 1_700_000_002.0
    assert chat.status == 'error'


def test_multi_part_text_result_joins_with_newlines() -> None:
    traj = AtifTrajectory.model_validate({
        'schema_version': 'ATIF-v1.7', 'agent': {'name': 'a', 'version': '1'},
        'steps': [{'step_id': 1, 'source': 'agent', 'message': '', 'tool_calls': [
            {'tool_call_id': 'c', 'function_name': 'f', 'arguments': {}}],
            'observation': {'results': [{'source_call_id': 'c', 'content': [
                {'type': 'text', 'text': 'a'}, {'type': 'text', 'text': 'b'}]}]}}]})
    tool = next(s for s in traj.to_otel().spans if s.operation == 'execute_tool')
    assert tool.attributes['gen_ai.tool.call.result'] == 'a\nb'
