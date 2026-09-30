"""Responses <-> ATIF conversion."""

# ruff: noqa: S101

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from openai.types.responses import Response

from evaluatorq.formats.atif import AtifTrajectory
from evaluatorq.formats.responses import ResponsesConversation

FIXTURES = Path(__file__).parent / 'fixtures' / 'atif'

_ITEMS: list[dict[str, Any]] = [
    {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'weather in Berlin?'}]},
    {'type': 'reasoning', 'id': 'rs_1', 'summary': [{'type': 'summary_text', 'text': 'need the tool'}]},
    {'type': 'function_call', 'id': 'fc_1', 'call_id': 'c1', 'name': 'get_weather', 'arguments': '{"city":"Berlin"}'},
    {'type': 'function_call_output', 'call_id': 'c1', 'output': '12C'},
    {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'It is 12C.'}]},
]


def _response(model: str, usage: dict[str, Any] | None) -> Response:
    return Response.model_validate({
        'id': 'resp_1', 'created_at': 1_776_679_200.0, 'model': model, 'object': 'response', 'output': [],
        'parallel_tool_calls': False, 'tool_choice': 'auto', 'tools': [], 'status': 'completed', 'usage': usage,
    })


def test_items_become_user_and_two_agent_steps() -> None:
    traj = ResponsesConversation(items=_ITEMS).to_atif()
    assert [s.source for s in traj.steps] == ['user', 'agent', 'agent']
    first = traj.steps[1]
    assert first.reasoning_content == 'need the tool'
    assert first.tool_calls is not None and first.tool_calls[0].arguments == {'city': 'Berlin'}
    assert first.observation is not None and first.observation.results[0].source_call_id == 'c1'
    assert traj.steps[2].message == 'It is 12C.'


def test_usage_maps_to_metrics_and_extra() -> None:
    usage = {'input_tokens': 100, 'output_tokens': 20, 'total_tokens': 120,
             'input_tokens_details': {'cached_tokens': 40}, 'output_tokens_details': {'reasoning_tokens': 7}}
    conv = ResponsesConversation(items=_ITEMS, responses=[_response('gpt-x', usage), _response('gpt-x', None)])
    steps = conv.to_atif().steps
    metrics = steps[1].metrics
    assert metrics is not None
    assert (metrics.prompt_tokens, metrics.completion_tokens, metrics.cached_tokens) == (100, 20, 40)
    assert metrics.extra == {'reasoning_tokens': 7, 'total_tokens': 120}
    assert steps[1].model_name == 'gpt-x' and steps[1].llm_call_count == 1
    assert steps[2].metrics is None


def test_response_count_mismatch_ignores_responses(caplog: pytest.LogCaptureFixture) -> None:
    conv = ResponsesConversation(items=_ITEMS, responses=[_response('gpt-x', None)])
    assert conv.to_atif().steps[1].model_name is None
    assert 'responses' in caplog.text


def test_orphan_function_call_output_warns_and_attaches(caplog: pytest.LogCaptureFixture) -> None:
    items = [*_ITEMS[:1], {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'a'}]},
             {'type': 'function_call_output', 'call_id': 'ghost', 'output': 'r'}]
    traj = ResponsesConversation(items=items).to_atif()
    result = traj.steps[1].observation.results[0]  # pyright: ignore[reportOptionalMemberAccess]
    assert result.source_call_id is None and result.extra == {'orphan_call_id': 'ghost'}
    assert 'ghost' in caplog.text


def test_empty_items_raise_value_error() -> None:
    with pytest.raises(ValueError, match='no items'):
        ResponsesConversation(items=[]).to_atif()


def test_developer_message_becomes_system_step_with_original_role() -> None:
    items: list[dict[str, Any]] = [{'type': 'message', 'role': 'developer', 'content': 'rules'}, *_ITEMS]
    step = ResponsesConversation(items=items).to_atif().steps[0]
    assert step.source == 'system' and step.extra == {'original_role': 'developer'}


def test_non_json_arguments_survive_via_raw_wrapper(caplog: pytest.LogCaptureFixture) -> None:
    items: list[dict[str, Any]] = [
        {'type': 'function_call', 'call_id': 'c', 'name': 'f', 'arguments': 'not json'},
        {'type': 'function_call_output', 'call_id': 'c', 'output': 'r'},
    ]
    traj = ResponsesConversation(items=items).to_atif()
    assert traj.steps[0].tool_calls is not None and traj.steps[0].tool_calls[0].arguments == {'_raw': 'not json'}
    back = traj.to_responses()
    assert next(i for i in back.items if i['type'] == 'function_call')['arguments'] == 'not json'


def test_data_url_image_survives_as_path() -> None:
    url = 'data:image/png;base64,AAAA'
    items: list[dict[str, Any]] = [{'type': 'message', 'role': 'user', 'content': [{'type': 'input_image', 'image_url': url}]}]
    step = ResponsesConversation(items=items).to_atif().steps[0]
    assert isinstance(step.message, list) and step.message[0].source is not None and step.message[0].source.path == url


def test_audio_part_becomes_marker_in_chat(caplog: pytest.LogCaptureFixture) -> None:
    part = {'type': 'audio', 'source': {'media_type': 'audio/wav', 'path': 'a.wav'}}
    traj = AtifTrajectory.model_validate({
        'schema_version': 'ATIF-v1.8', 'agent': {'name': 'a', 'version': '1'},
        'steps': [{'step_id': 1, 'source': 'user', 'message': [part]}]})
    content = traj.to_responses().to_chat().messages[0].content
    assert 'audio' in str(content) and 'a.wav' in str(content)  # marker text
    assert 'audio' in caplog.text


def test_null_source_call_id_dropped_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    traj = AtifTrajectory.model_validate({
        'schema_version': 'ATIF-v1.7', 'agent': {'name': 'a', 'version': '1'},
        'steps': [{'step_id': 1, 'source': 'agent', 'message': 'x', 'observation': {'results': [{'content': 'r'}]}}]})
    items = traj.to_responses().items
    assert all(i.get('type') != 'function_call_output' for i in items)
    assert 'source_call_id' in caplog.text


def test_atif_to_responses_carries_metrics_into_response() -> None:
    traj = AtifTrajectory.model_validate({
        'schema_version': 'ATIF-v1.7', 'agent': {'name': 'a', 'version': '1'}, 'session_id': 's',
        'steps': [{'step_id': 1, 'source': 'agent', 'message': 'x', 'model_name': 'gpt-x', 'llm_call_count': 1,
                   'metrics': {'prompt_tokens': 5, 'completion_tokens': 2, 'cached_tokens': 1}}]})
    responses = traj.to_responses().responses
    assert responses is not None and responses[0].usage is not None
    assert (responses[0].usage.input_tokens, responses[0].usage.input_tokens_details.cached_tokens) == (5, 1)


def test_responses_atif_responses_roundtrip_preserves_transcript() -> None:
    conv = ResponsesConversation(items=_ITEMS)
    back = conv.to_atif().to_responses()
    assert [i.get('type', 'message') for i in back.items] == [i.get('type', 'message') for i in _ITEMS]
    call = next(i for i in back.items if i['type'] == 'function_call')
    assert call['call_id'] == 'c1' and call['id'] == 'fc_1' and call['arguments'] == '{"city":"Berlin"}'


def test_fixture_trajectory_converts_to_responses_and_back() -> None:
    traj = AtifTrajectory.from_json((FIXTURES / 'harbor_invalid_json_v18.json').read_text())
    again = traj.to_responses().to_atif()
    assert len(again.steps) >= 1


def test_response_timing_status_and_final_metrics() -> None:
    usage = {'input_tokens': 10, 'output_tokens': 2, 'total_tokens': 12,
             'input_tokens_details': {'cached_tokens': 4}, 'output_tokens_details': {'reasoning_tokens': 0}}
    failed = _response('gpt-x', usage).model_copy(update={'status': 'incomplete', 'id': 'resp_2'})
    conv = ResponsesConversation(items=_ITEMS, responses=[_response('gpt-x', usage), failed])
    traj = conv.to_atif()
    assert traj.steps[1].timestamp == '2026-04-20T10:00:00+00:00'
    assert traj.steps[1].extra == {'fc_item_ids': {'c1': 'fc_1'}, 'response_id': 'resp_1'}
    assert traj.steps[2].extra == {'response_id': 'resp_2', 'status': 'incomplete'}
    assert traj.final_metrics is not None
    assert (traj.final_metrics.total_prompt_tokens, traj.final_metrics.total_cached_tokens) == (20, 8)
    back = traj.to_responses().responses
    assert back is not None and [r.id for r in back] == ['resp_1', 'resp_2']
    assert back[1].status == 'incomplete' and back[0].created_at == 1_776_679_200.0


def test_encrypted_only_reasoning_is_flagged_not_invented() -> None:
    items: list[dict[str, Any]] = [
        {'type': 'reasoning', 'id': 'rs_x', 'summary': [], 'encrypted_content': 'opaque'},
        {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'hi'}]},
    ]
    step = ResponsesConversation(items=items).to_atif().steps[0]
    assert step.reasoning_content is None and step.extra == {'encrypted_reasoning': True}


def test_input_file_becomes_marker_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    items: list[dict[str, Any]] = [{'type': 'message', 'role': 'user', 'content': [
        {'type': 'input_text', 'text': 'see'}, {'type': 'input_file', 'filename': 'a.pdf', 'file_data': 'x'}]}]
    assert ResponsesConversation(items=items).to_atif().steps[0].message == 'see\n[file: a.pdf]'
    assert 'input_file' in caplog.text


def test_developer_role_survives_the_round_trip() -> None:
    items: list[dict[str, Any]] = [{'type': 'message', 'role': 'developer', 'content': 'rules'}, *_ITEMS]
    back = ResponsesConversation(items=items).to_atif().to_responses()
    assert back.items[0] == {'type': 'message', 'role': 'developer', 'content': 'rules'}


def test_embedded_subagent_is_warned_as_lost(caplog: pytest.LogCaptureFixture) -> None:
    traj = AtifTrajectory.from_json((FIXTURES / 'phoenix_v17_embedded_subagents.json').read_text())
    traj.to_responses()
    assert 'not representable in Responses' in caplog.text


def test_one_response_per_agent_step_keeps_metadata_aligned(caplog: pytest.LogCaptureFixture) -> None:
    traj = AtifTrajectory.model_validate({
        'schema_version': 'ATIF-v1.7', 'agent': {'name': 'a', 'version': '1'}, 'session_id': 's',
        'steps': [
            {'step_id': 1, 'source': 'user', 'message': 'weather?'},
            {'step_id': 2, 'source': 'agent', 'message': '',
             'tool_calls': [{'tool_call_id': 'c1', 'function_name': 'get_weather', 'arguments': {'city': 'Berlin'}}],
             'observation': {'results': [{'source_call_id': 'c1', 'content': '12C'}]}},
            {'step_id': 3, 'source': 'agent', 'message': 'It is 12C.', 'model_name': 'gpt-x', 'llm_call_count': 1,
             'metrics': {'prompt_tokens': 5, 'completion_tokens': 2, 'cached_tokens': 1}},
        ]})
    conv = traj.to_responses()
    assert conv.responses is not None and len(conv.responses) == 2
    again = conv.to_atif()
    assert [s.model_name for s in again.steps] == [None, None, 'gpt-x']
    assert again.steps[1].metrics is None
    metrics = again.steps[2].metrics
    assert metrics is not None and (metrics.prompt_tokens, metrics.completion_tokens, metrics.cached_tokens) == (5, 2, 1)
    assert again.steps[1].timestamp is None
    assert not [r for r in caplog.records if r.levelname == 'WARNING']


def test_foreign_step_extra_values_are_dropped_not_crashing(caplog: pytest.LogCaptureFixture) -> None:
    traj = AtifTrajectory.model_validate({
        'schema_version': 'ATIF-v1.7', 'agent': {'name': 'a', 'version': '1'}, 'session_id': 's',
        'steps': [{'step_id': 1, 'source': 'agent', 'message': 'x', 'model_name': 'gpt-x',
                   'tool_calls': [{'tool_call_id': 'c1', 'function_name': 'f', 'arguments': {}}],
                   'metrics': {'prompt_tokens': 5, 'extra': {'total_tokens': 'many', 'reasoning_tokens': 1.5}},
                   'extra': {'error': 'boom', 'status': 'error', 'incomplete_details': {'reason': 7},
                             'response_id': 3, 'fc_item_ids': ['fc_1']}}]})
    conv = traj.to_responses()
    assert conv.responses is not None
    response = conv.responses[0]
    assert (response.status, response.error, response.incomplete_details) == ('completed', None, None)
    assert response.id.startswith('resp_') and response.usage is not None
    assert (response.usage.total_tokens, response.usage.output_tokens_details.reasoning_tokens) == (5, 0)
    assert 'id' not in next(i for i in conv.items if i['type'] == 'function_call')
    for key in ('error', 'status', 'incomplete_details', 'response_id', 'fc_item_ids', 'total_tokens', 'reasoning_tokens'):
        assert f'extra.{key}' in caplog.text


def _agent_step_with_extra(extra: dict[str, Any]) -> AtifTrajectory:
    return AtifTrajectory.model_validate({
        'schema_version': 'ATIF-v1.7', 'agent': {'name': 'a', 'version': '1'}, 'session_id': 's',
        'steps': [{'step_id': 1, 'source': 'agent', 'message': 'x', 'model_name': 'gpt-x', 'extra': extra}]})


def test_unhashable_status_is_dropped_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    responses = _agent_step_with_extra({'status': ['x']}).to_responses().responses
    assert responses is not None and responses[0].status == 'completed'
    assert 'extra.status' in caplog.text


def test_valid_status_error_and_incomplete_details_pass_through() -> None:
    error = {'code': 'server_error', 'message': 'boom'}
    incomplete = {'reason': 'max_output_tokens'}
    traj = _agent_step_with_extra({'status': 'failed', 'error': error, 'incomplete_details': incomplete})
    responses = traj.to_responses().responses
    assert responses is not None
    response = responses[0]
    assert response.status == 'failed'
    assert response.error is not None and response.error.model_dump() == error
    assert response.incomplete_details is not None and response.incomplete_details.model_dump() == incomplete


# --- ATIF -> Responses -> ATIF does not invent data ---


def _agent_traj(**step: Any) -> AtifTrajectory:
    return AtifTrajectory.model_validate({
        'agent': {'name': 'a', 'version': '1'}, 'session_id': 's',
        'steps': [{'step_id': 1, 'source': 'user', 'message': 'q'},
                  {'step_id': 2, 'source': 'agent', 'message': 'x', **step}]})


def test_placeholder_response_reads_back_as_absent() -> None:
    traj = _agent_traj(llm_call_count=0)
    back = traj.to_responses().to_atif().steps[1]
    assert (back.llm_call_count, back.model_name, back.timestamp, back.metrics, back.extra) == (None,) * 5


def test_unset_token_counts_stay_unset() -> None:
    traj = _agent_traj(model_name='gpt-x', metrics={'prompt_tokens': 10})
    responses = traj.to_responses().responses
    assert responses is not None and responses[0].metadata is not None
    back = traj.to_responses().to_atif().steps[1].metrics
    assert back is not None
    assert (back.prompt_tokens, back.completion_tokens, back.cached_tokens, back.extra) == (10, None, None, None)


def test_known_token_counts_including_zero_survive() -> None:
    metrics = {'prompt_tokens': 0, 'completion_tokens': 3, 'cached_tokens': 0,
               'extra': {'reasoning_tokens': 0, 'total_tokens': 3}}
    traj = _agent_traj(model_name='gpt-x', metrics=metrics)
    responses = traj.to_responses().responses
    assert responses is not None and responses[0].metadata is None
    back = traj.to_responses().to_atif().steps[1].metrics
    assert back is not None and back.model_dump(exclude_none=True) == metrics


def test_cost_and_token_ids_are_dropped_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    traj = _agent_traj(model_name='gpt-x', metrics={'prompt_tokens': 1, 'completion_tokens': 1, 'cost_usd': 0.5,
                                                    'logprobs': [-0.1]})
    back = traj.to_responses().to_atif().steps[1].metrics
    assert back is not None and back.cost_usd is None and back.logprobs is None
    assert "'cost_usd': 1" in caplog.text and "'logprobs': 1" in caplog.text


def test_nanosecond_timestamp_converts_to_created_at() -> None:
    traj = _agent_traj(model_name='gpt-x', timestamp='2026-04-20T10:00:00.123456789Z')
    responses = traj.to_responses().responses
    assert responses is not None and responses[0].created_at == pytest.approx(1_776_679_200.123456)


def test_session_id_is_passed_through() -> None:
    assert ResponsesConversation(items=_ITEMS).to_atif(session_id='mine').session_id == 'mine'


# --- one text-joining rule on every route ---


def test_multi_part_messages_agree_across_routes() -> None:
    parts = [{'type': 'output_text', 'text': 'a'}, {'type': 'output_text', 'text': ''},
             {'type': 'output_text', 'text': 'b'}]
    conv = ResponsesConversation(items=[
        {'type': 'message', 'role': 'user', 'content': [{**p, 'type': 'input_text'} for p in parts]},
        {'type': 'message', 'role': 'assistant', 'content': parts},
    ])
    direct = [(m.role, m.content) for m in conv.to_chat().messages]
    assert direct == [('user', 'a\nb'), ('assistant', 'a\nb')]
    assert [(m.role, m.content) for m in conv.to_atif().to_chat().messages] == direct


def test_multi_part_text_tool_result_joins_with_newlines() -> None:
    traj = AtifTrajectory.model_validate({
        'schema_version': 'ATIF-v1.7', 'agent': {'name': 'a', 'version': '1'},
        'steps': [{'step_id': 1, 'source': 'agent', 'message': '', 'tool_calls': [
            {'tool_call_id': 'c', 'function_name': 'f', 'arguments': {}}],
            'observation': {'results': [{'source_call_id': 'c', 'content': [
                {'type': 'text', 'text': 'a'}, {'type': 'text', 'text': 'b'}]}]}}]})
    output = next(i for i in traj.to_responses().items if i.get('type') == 'function_call_output')
    assert output['output'] == 'a\nb'


def test_malformed_assistant_text_part_is_warned_not_repr_ed(caplog: pytest.LogCaptureFixture) -> None:
    conv = ResponsesConversation(items=[
        {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': {'bad': 1}}]},
        {'type': 'message', 'role': 'assistant', 'content': [
            {'type': 'output_text', 'text': 'ok'}, {'type': 'output_text', 'text': ['bad']}]},
    ])
    traj = conv.to_atif()
    assert traj.steps[0].message == ''
    assert traj.steps[1].message == 'ok'
    assert caplog.text.count('carries no text') == 2
