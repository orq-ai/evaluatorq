"""The Responses face of the Anthropic adapter: request conversion, response shape, parse, and the simulation agents."""

from __future__ import annotations

import asyncio
import copy
import json
from typing import Any

import pytest
from anthropic.types import Message, TextBlock, ThinkingBlock, ToolUseBlock, Usage
from loguru import logger
from openai.types.responses import ParsedResponse, Response
from pydantic import BaseModel, ValidationError

from evaluatorq.anthropic import AnthropicLLMClient
from evaluatorq.common.llm_call import execute_response
from evaluatorq.common.responses import first_responses_refusal, parse_responses_response, responses_stop_reason
from evaluatorq.common.retry import without_client_retries
from evaluatorq.contracts import AgentResponse, FunctionCall, LLMCallConfig, Message as SimMessage, StrategyToolCall
from evaluatorq.openresponses.input_items import messages_to_responses_input
from evaluatorq.simulation.agents.judge import JudgeAgent
from evaluatorq.simulation.agents.user_simulator import UserSimulatorAgent

from .test_chat_adapter import FakeAnthropic, call as chat_call, make_message
from .test_structured_output import FakeClient


def adapter(fake: Any) -> Any:
    """The adapter typed as ``Any``: callers take it duck-typed where they declare an ``AsyncOpenAI``."""
    return AnthropicLLMClient(fake)


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def respond(reply: Message | None = None, **params: Any) -> tuple[Any, dict[str, Any]]:
    fake = FakeAnthropic(reply)
    params.setdefault('model', 'claude-sonnet-5-5')
    params.setdefault('input', 'hi')
    response = run(AnthropicLLMClient(fake).responses.create(**params))
    return response, fake.messages.calls[0]


def transcript() -> list[SimMessage]:
    return [
        SimMessage(role='user', content='Where is my order?'),
        SimMessage(
            role='assistant',
            content='Let me look.',
            tool_calls=[StrategyToolCall(id='toolu_1', function=FunctionCall(name='lookup', arguments='{"id": 7}'))],
        ),
        SimMessage(role='tool', content='shipped', tool_call_id='toolu_1'),
        SimMessage(role='assistant', content='It shipped.'),
        SimMessage(role='user', content='Thanks.'),
    ]


# --- request ---


def test_string_input_and_instructions():
    _, sent = respond(input='hello', instructions='Be brief.')
    assert sent['system'] == 'Be brief.'
    assert sent['messages'] == [{'role': 'user', 'content': [{'type': 'text', 'text': 'hello'}]}]
    assert sent['max_tokens'] == 16000
    assert sent['model'] == 'claude-sonnet-5-5'
    for key in ('temperature', 'top_p', 'output_config', 'tools', 'metadata'):
        assert key not in sent


def test_transcript_from_messages_to_responses_input():
    _, sent = respond(input=messages_to_responses_input(transcript()), instructions='sys')
    assert sent['messages'] == [
        {'role': 'user', 'content': [{'type': 'text', 'text': 'Where is my order?'}]},
        {
            'role': 'assistant',
            'content': [
                {'type': 'text', 'text': 'Let me look.'},
                {'type': 'tool_use', 'id': 'toolu_1', 'name': 'lookup', 'input': {'id': 7}},
            ],
        },
        {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'toolu_1', 'content': 'shipped'}]},
        {
            'role': 'assistant',
            'content': [{'type': 'text', 'text': 'It shipped.'}],
        },
        {'role': 'user', 'content': [{'type': 'text', 'text': 'Thanks.'}]},
    ]


def test_tool_result_and_next_user_turn_merge_and_adjacent_calls_merge():
    items = [
        {'role': 'user', 'content': 'go'},
        {'type': 'function_call', 'call_id': 'a', 'name': 'f', 'arguments': '{}'},
        {'type': 'function_call', 'call_id': 'b', 'name': 'g', 'arguments': '{"x": 1}'},
        {'type': 'function_call_output', 'call_id': 'a', 'output': 'ra'},
        {'type': 'function_call_output', 'call_id': 'b', 'output': [{'type': 'input_text', 'text': 'rb'}]},
        {'role': 'user', 'content': [{'type': 'input_text', 'text': 'and?'}]},
    ]
    _, sent = respond(input=items)
    assert [m['role'] for m in sent['messages']] == ['user', 'assistant', 'user']
    assert [b['type'] for b in sent['messages'][1]['content']] == ['tool_use', 'tool_use']
    assert sent['messages'][2]['content'] == [
        {'type': 'tool_result', 'tool_use_id': 'a', 'content': 'ra'},
        {'type': 'tool_result', 'tool_use_id': 'b', 'content': 'rb'},
        {'type': 'text', 'text': 'and?'},
    ]


def test_reasoning_items_are_dropped_and_cache_control_is_kept():
    items = [
        {'role': 'user', 'content': [{'type': 'input_text', 'text': 'a', 'cache_control': {'type': 'ephemeral'}}]},
        {'type': 'reasoning', 'id': 'rs_1', 'summary': []},
        {'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'b'}]},
    ]
    _, sent = respond(input=items)
    assert sent['messages'][0]['content'][0]['cache_control'] == {'type': 'ephemeral'}
    assert len(sent['messages']) == 2


def test_cache_control_on_instructions_part_survives_in_system_blocks():
    items = [
        {'role': 'developer', 'content': [{'type': 'input_text', 'text': 's', 'cache_control': {'type': 'ephemeral'}}]},
        {'role': 'user', 'content': 'u'},
    ]
    _, sent = respond(input=items, instructions='first')
    assert sent['system'] == [
        {'type': 'text', 'text': 'first'},
        {'type': 'text', 'text': 's', 'cache_control': {'type': 'ephemeral'}},
    ]


def test_flat_tools_become_anthropic_tools():
    tool = {'type': 'function', 'name': 'lookup', 'description': 'Find it', 'parameters': {'type': 'object', 'properties': {}}}
    _, sent = respond(tools=[tool], tool_choice='auto')
    assert sent['tools'] == [{'name': 'lookup', 'description': 'Find it', 'input_schema': {'type': 'object', 'properties': {}}}]
    assert sent['tool_choice'] == {'type': 'auto'}


def test_forced_tool_choice_is_translated_and_non_function_tools_raise():
    tool = {'type': 'function', 'name': 'f', 'parameters': {'type': 'object', 'properties': {}}}
    _, sent = respond(tools=[tool], tool_choice='required')
    assert sent['tool_choice'] == {'type': 'any'}
    _, sent = respond(tools=[tool], tool_choice={'type': 'function', 'name': 'f'})
    assert sent['tool_choice'] == {'type': 'tool', 'name': 'f'}
    with pytest.raises(ValueError, match='function tools'):
        respond(tools=[{'type': 'web_search_preview'}])


def test_input_that_starts_with_an_assistant_turn_gets_a_user_turn_first():
    # From turn 2 on, the user simulator's request is the transcript with roles swapped, so it opens with its own line.
    from evaluatorq.anthropic.convert import LEADING_USER_TEXT

    items = [
        {'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'Hi, I need help.'}]},
        {'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'With my order.'}]},
        {'role': 'user', 'content': [{'type': 'input_text', 'text': 'Sure, what is the number?'}]},
    ]
    _, sent = respond(input=items)
    assert [turn['role'] for turn in sent['messages']] == ['user', 'assistant', 'user']
    assert sent['messages'][0]['content'] == [{'type': 'text', 'text': LEADING_USER_TEXT}]
    assert [block['text'] for block in sent['messages'][1]['content']] == ['Hi, I need help.', 'With my order.']


def test_params_map():
    _, sent = respond(
        max_output_tokens=500,
        temperature=0.3,
        top_p=0.9,
        reasoning={'effort': 'xhigh'},
        metadata={'run': 'x'},
        extra_headers={'h': '1'},
        timeout=12.0,
        extra_body={'thread': {'id': 't'}, 'cache_control': {'type': 'ephemeral'}},
    )
    assert sent['max_tokens'] == 500
    assert sent['temperature'] == 0.3
    assert sent['top_p'] == 0.9
    assert sent['output_config'] == {'effort': 'max'}
    assert 'metadata' not in sent
    assert sent['extra_headers'] == {'h': '1'}
    assert sent['timeout'] == 12.0
    assert sent['extra_body'] == {'cache_control': {'type': 'ephemeral'}}


def test_text_format_json_schema_becomes_output_config_format():
    schema = {'type': 'object', 'properties': {'n': {'type': 'integer', 'minimum': 0}}, 'required': ['n']}
    _, sent = respond(text={'format': {'type': 'json_schema', 'name': 'N', 'strict': True, 'schema': schema}})
    assert sent['output_config'] == {
        'format': {
            'type': 'json_schema',
            'schema': {'type': 'object', 'properties': {'n': {'type': 'integer'}}, 'required': ['n'], 'additionalProperties': False},
        }
    }


def test_text_format_text_sends_nothing_and_unknown_keys_raise():
    _, sent = respond(text={'format': {'type': 'text'}})
    assert 'output_config' not in sent
    with pytest.raises(TypeError, match='previous_response_id'):
        respond(previous_response_id='r')
    with pytest.raises(TypeError, match='verbosity'):
        respond(text={'verbosity': 'low'})
    with pytest.raises(ValueError, match='no user or assistant content'):
        respond(input=[{'type': 'reasoning', 'id': 'r', 'summary': []}])
    with pytest.raises(ValueError, match='other/model'):
        respond(model='other/model')
    with pytest.raises(ValueError, match='input_image'):
        respond(input=[{'role': 'user', 'content': [{'type': 'input_image', 'image_url': 'x'}]}])


def test_model_prefix_is_stripped():
    _, sent = respond(model='anthropic/claude-opus-5-5')
    assert sent['model'] == 'claude-opus-5-5'


# --- response ---


def test_text_response_is_a_genuine_response():
    usage = Usage(input_tokens=10, output_tokens=4, cache_read_input_tokens=5, cache_creation_input_tokens=2)
    response, _ = respond(make_message(usage=usage))
    assert isinstance(response, Response)
    assert response.status == 'completed'
    assert response.incomplete_details is None
    assert response.model == 'anthropic/claude-sonnet-5-5'
    assert response.output_text == 'hello'
    assert response.output[0].type == 'message'
    assert response.usage is not None
    assert response.usage.input_tokens == 17
    assert response.usage.output_tokens == 4
    assert response.usage.total_tokens == 21
    assert response.usage.input_tokens_details.cached_tokens == 5
    assert response.usage.output_tokens_details.reasoning_tokens == 0
    assert response.usage.model_dump()['cache_creation_input_tokens'] == 2
    assert responses_stop_reason(response) is None
    assert first_responses_refusal(response) is None


def test_tool_use_response_and_thinking_dropped():
    content = [
        ThinkingBlock(type='thinking', thinking='hm', signature='s'),
        TextBlock(type='text', text='Looking.'),
        ToolUseBlock(type='tool_use', id='toolu_9', name='lookup', input={'id': 3}),
    ]
    response, _ = respond(make_message(content=content, stop_reason='tool_use'))
    assert [item.type for item in response.output] == ['message', 'function_call']
    call = response.output[1]
    assert (call.call_id, call.name, json.loads(call.arguments)) == ('toolu_9', 'lookup', {'id': 3})
    parsed = AgentResponse.from_openresponses(response)
    assert [getattr(item, 'text', None) for item in parsed.output if hasattr(item, 'text')] == ['Looking.']
    assert [(getattr(i, 'call_id', None), getattr(i, 'name', None)) for i in parsed.output if hasattr(i, 'call_id')] == [
        ('toolu_9', 'lookup')
    ]


def test_tool_use_only_has_no_message_item():
    content = [ToolUseBlock(type='tool_use', id='t', name='f', input={})]
    response, _ = respond(make_message(content=content, stop_reason='tool_use'))
    assert [item.type for item in response.output] == ['function_call']


def test_max_tokens_is_incomplete():
    response, _ = respond(make_message(stop_reason='max_tokens'))
    assert response.status == 'incomplete'
    assert response.incomplete_details is not None
    assert response.incomplete_details.reason == 'max_output_tokens'
    assert responses_stop_reason(response) == 'length'


def test_pause_turn_is_completed_like_the_chat_face():
    response, _ = respond(make_message(stop_reason='pause_turn'))
    assert response.status == 'completed'
    assert response.incomplete_details is None
    assert response.output_text == 'hello'
    chat, _ = chat_call(make_message(stop_reason='pause_turn'))
    assert chat.choices[0].finish_reason == 'stop'


def test_refusal_is_a_refusal_part():
    response, _ = respond(make_message(content=[TextBlock(type='text', text='No.')], stop_reason='refusal'))
    assert first_responses_refusal(response) == 'No.'
    assert response.output_text == ''
    empty, _ = respond(make_message(content=[], stop_reason='refusal'))
    assert first_responses_refusal(empty) == 'The model declined to answer this request.'


def test_refusal_warning_shared_with_chat_face_and_clones():
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m)), level='WARNING')
    try:
        fake = FakeAnthropic(make_message(content=[], stop_reason='refusal'))
        client = adapter(fake)
        clone = without_client_retries(client)
        run(clone.responses.create(model='claude-sonnet-5-5', input='x'))
        run(client.responses.create(model='claude-sonnet-5-5', input='x'))
        run(client.chat.completions.create(model='claude-sonnet-5-5', messages=[{'role': 'user', 'content': 'x'}]))
    finally:
        logger.remove(sink)
    assert len([m for m in messages if 'refused' in m]) == 1


# --- parse ---


class Verdict(BaseModel):
    ok: bool
    note: str


def parse(body: str, stop_reason: str = 'end_turn', **params: Any) -> tuple[Any, dict[str, Any]]:
    fake = FakeAnthropic(make_message(content=[TextBlock(type='text', text=body)], stop_reason=stop_reason))
    result = run(
        AnthropicLLMClient(fake).responses.parse(
            model='claude-sonnet-5-5', input='x', text_format=Verdict, reasoning={'effort': 'high'}, **params
        )
    )
    return result, fake.messages.calls[0]


def test_parse_fills_output_parsed_and_sends_strict_schema():
    result, sent = parse('{"ok": true, "note": "fine"}')
    assert isinstance(result, ParsedResponse)
    assert result.output_parsed == Verdict(ok=True, note='fine')
    assert sent['output_config']['effort'] == 'high'
    schema = sent['output_config']['format']['schema']
    assert schema['additionalProperties'] is False
    assert schema['required'] == ['ok', 'note']


def test_parse_invalid_json_raises_validation_error():
    with pytest.raises(ValidationError):
        parse('{"ok": "maybe"}')


def test_parse_refusal_leaves_output_parsed_none():
    fake = FakeAnthropic(make_message(content=[TextBlock(type='text', text='No.')], stop_reason='refusal'))
    result = run(AnthropicLLMClient(fake).responses.parse(model='claude-sonnet-5-5', input='x', text_format=Verdict))
    assert result.output_parsed is None
    assert first_responses_refusal(result) == 'No.'


def test_parse_responses_response_reads_a_created_response():
    schema = Verdict.model_json_schema()
    response, _ = respond(
        make_message(content=[TextBlock(type='text', text='{"ok": false, "note": "n"}')]),
        text={'format': {'type': 'json_schema', 'name': 'Verdict', 'schema': schema}},
    )
    assert parse_responses_response(response, Verdict).output_parsed == Verdict(ok=False, note='n')


def test_execute_response_runs_on_the_adapter_and_leaves_inputs_untouched():
    items = messages_to_responses_input(transcript())
    snapshot = copy.deepcopy(items)
    fake = FakeAnthropic(make_message(usage=Usage(input_tokens=10, output_tokens=4)))
    response, usage = run(
        execute_response(
            client=adapter(fake),
            model='claude-sonnet-5-5',
            messages=items,
            span=None,
            timeout_s=30,
            instructions='sys',
            max_output_tokens=300,
        )
    )
    assert response.output_text == 'hello'
    assert (usage.input_tokens, usage.output_tokens) == (10, 4)
    assert items == snapshot
    assert fake.messages.calls[0]['max_tokens'] == 300


# --- fallbacks ---


def test_fallbacks_route_responses_to_the_beta_call():
    fake = FakeClient()
    run(AnthropicLLMClient(fake).responses.create(model='claude-sonnet-5-5', input='x', fallbacks='default'))
    assert fake.messages.calls == []
    call = fake.beta.messages.calls[0]
    assert call['betas'] == ['server-side-fallback-2026-07-01']
    assert call['extra_body'] == {'fallbacks': 'default'}


def test_fallbacks_rejected_on_bedrock_for_responses():
    from .test_structured_output import AsyncAnthropicBedrock

    with pytest.raises(ValueError, match='Claude API only'):
        run(AnthropicLLMClient(AsyncAnthropicBedrock()).responses.create(model='m', input='x', fallbacks='default'))


# --- the simulation agents, end to end ---

class SharedFake(FakeAnthropic):
    """A fake whose ``with_options`` clones (made by ``without_client_retries``) record into the same call list."""

    def with_options(self, **kwargs: Any) -> SharedFake:
        clone = SharedFake()
        clone.messages = self.messages
        return clone


JUDGE_REPLY = json.dumps({
    'reason': 'goal met',
    'should_terminate': True,
    'goal_achieved': True,
    'goal_completion_score': 1.0,
    'criteria_verdicts': [],
    'response_quality': 0.9,
    'hallucination_risk': 0.1,
    'tone_appropriateness': 0.9,
    'factual_accuracy': 0.9,
})


def test_judge_agent_completes_a_turn_on_the_adapter():
    fake = SharedFake(make_message(content=[TextBlock(type='text', text=JUDGE_REPLY)]))
    judge = JudgeAgent(
        LLMCallConfig(model='anthropic/claude-sonnet-5-5', client=adapter(fake), reasoning_effort='high'),
        goal='Find the order',
    )
    judgment = run(judge.evaluate(transcript()))
    assert judgment.should_terminate is True
    assert judgment.goal_achieved is True
    sent = fake.messages.calls[0]
    assert sent['model'] == 'claude-sonnet-5-5'
    assert 'Find the order' in sent['system']
    assert sent['output_config']['effort'] == 'high'
    assert sent['output_config']['format']['type'] == 'json_schema'
    assert sent['output_config']['format']['schema']['additionalProperties'] is False
    roles = [m['role'] for m in sent['messages']]
    assert roles == ['user', 'assistant', 'user', 'assistant', 'user']
    blocks = [b['type'] for m in sent['messages'] for b in m['content']]
    assert blocks == ['text', 'text', 'tool_use', 'tool_result', 'text', 'text', 'text']
    assert sent['messages'][-1]['content'][-1]['text'].startswith('Evaluate the conversation above')
    assert judge.get_usage().input_tokens == 10


def test_user_simulator_turn_on_the_adapter():
    fake = SharedFake(make_message(content=[TextBlock(type='text', text='Hi, where is my order?')]))
    agent = UserSimulatorAgent(
        LLMCallConfig(model='claude-sonnet-5-5', client=adapter(fake), reasoning_effort='low'),
        system_prompt='You are Sam.',
    )
    text = run(agent.generate_first_message())
    assert text == 'Hi, where is my order?'
    sent = fake.messages.calls[0]
    assert 'You are Sam.' in sent['system']
    assert sent['output_config'] == {'effort': 'low'}
    assert sent['messages'][-1]['content'][0]['text'].startswith('Generate your first message')
    assert 'tools' not in sent

