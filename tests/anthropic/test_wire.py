"""What the real Anthropic SDK puts on the wire when evaluatorq calls it through the adapter (RES-1874).

Every test builds a genuine SDK client on an ``httpx.MockTransport`` and reads the HTTP request the SDK produced, so a rename or move inside the SDK fails here instead of passing against a hand-written fake.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import anthropic
import httpx
import pytest
from pydantic import BaseModel

from evaluatorq.anthropic import AnthropicLLMClient
from evaluatorq.anthropic.client import FALLBACK_BETA
from evaluatorq.common.llm_client import LLMClient, as_llm_client
from evaluatorq.common.prompt_cache import apply_cache_breakpoints, mark_responses_input

_REPLY = {
    'id': 'msg_1',
    'type': 'message',
    'role': 'assistant',
    'model': 'claude-sonnet-5-5',
    'content': [{'type': 'text', 'text': '{"ok": true}'}],
    'stop_reason': 'end_turn',
    'stop_sequence': None,
    'usage': {'input_tokens': 3, 'output_tokens': 2},
}


class Wire:
    """Records every HTTP request the SDK sends and answers each with a minimal valid Messages reply."""

    def __init__(self, reply: dict[str, Any] | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self.reply = reply or _REPLY

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json=self.reply)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    @property
    def sent(self) -> httpx.Request:
        assert len(self.requests) == 1, f'expected one request, saw {len(self.requests)}'
        return self.requests[0]

    @property
    def body(self) -> dict[str, Any]:
        return json.loads(self.sent.content)


def _client(wire: Wire) -> tuple[anthropic.AsyncAnthropic, Any]:
    raw = anthropic.AsyncAnthropic(api_key='test', max_retries=0, http_client=httpx.AsyncClient(transport=wire.transport))
    return raw, as_llm_client(raw)


_LONG = 'The quick brown fox jumps over the lazy dog. ' * 400
"""Long enough to clear the 1024-token minimum below which no cache breakpoint is placed."""


class Verdict(BaseModel):
    ok: bool


def test_chat_create_sends_a_bare_model_id_the_default_max_tokens_and_no_temperature() -> None:
    wire = Wire()
    _, client = _client(wire)
    asyncio.run(
        client.chat.completions.create(
            model='anthropic/claude-sonnet-5-5', messages=[{'role': 'user', 'content': 'hi'}]
        )
    )
    assert wire.sent.url.path == '/v1/messages'
    assert wire.sent.headers['x-api-key'] == 'test'
    body = wire.body
    assert body['model'] == 'claude-sonnet-5-5'
    assert body['max_tokens'] == 16000
    assert 'temperature' not in body
    assert body['messages'] == [{'role': 'user', 'content': [{'type': 'text', 'text': 'hi'}]}]


def test_chat_create_sends_temperature_and_max_tokens_when_set() -> None:
    wire = Wire()
    _, client = _client(wire)
    asyncio.run(
        client.chat.completions.create(
            model='claude-sonnet-5-5',
            messages=[{'role': 'user', 'content': 'hi'}],
            temperature=0.2,
            max_completion_tokens=123,
        )
    )
    assert wire.body['temperature'] == 0.2
    assert wire.body['max_tokens'] == 123


def test_chat_parse_sends_the_schema_as_output_config_format() -> None:
    wire = Wire()
    _, client = _client(wire)
    parsed = asyncio.run(
        client.chat.completions.parse(
            model='claude-sonnet-5-5', messages=[{'role': 'user', 'content': 'hi'}], response_format=Verdict
        )
    )
    output_format = wire.body['output_config']['format']
    assert output_format['type'] == 'json_schema'
    assert output_format['schema']['properties']['ok'] == {'title': 'Ok', 'type': 'boolean'}
    assert output_format['schema']['additionalProperties'] is False
    assert parsed.choices[0].message.parsed == Verdict(ok=True)


def test_responses_parse_sends_the_schema_as_output_config_format() -> None:
    wire = Wire()
    _, client = _client(wire)
    parsed = asyncio.run(
        client.responses.parse(model='claude-sonnet-5-5', input='hi', text_format=Verdict, instructions='Be terse.')
    )
    body = wire.body
    assert body['output_config']['format']['type'] == 'json_schema'
    assert body['system'] == 'Be terse.'
    assert parsed.output_parsed == Verdict(ok=True)


@pytest.mark.parametrize(('effort', 'sent'), [('minimal', 'low'), ('high', 'high'), ('xhigh', 'max')])
def test_reasoning_effort_becomes_output_config_effort(effort: str, sent: str) -> None:
    wire = Wire()
    _, client = _client(wire)
    asyncio.run(
        client.chat.completions.create(
            model='claude-sonnet-5-5', messages=[{'role': 'user', 'content': 'hi'}], reasoning_effort=effort
        )
    )
    assert wire.body['output_config'] == {'effort': sent}


def test_reasoning_effort_none_sends_no_output_config() -> None:
    wire = Wire()
    _, client = _client(wire)
    asyncio.run(
        client.chat.completions.create(
            model='claude-sonnet-5-5', messages=[{'role': 'user', 'content': 'hi'}], reasoning_effort='none'
        )
    )
    assert 'output_config' not in wire.body


def test_fallbacks_go_to_the_beta_endpoint_with_the_beta_header_and_the_body_field() -> None:
    wire = Wire()
    _, client = _client(wire)
    asyncio.run(
        client.chat.completions.create(
            model='claude-sonnet-5-5', messages=[{'role': 'user', 'content': 'hi'}], fallbacks='default'
        )
    )
    request = wire.sent
    assert request.url.path == '/v1/messages'
    assert request.url.params.get('beta') == 'true'
    assert FALLBACK_BETA in request.headers['anthropic-beta']
    assert wire.body['fallbacks'] == 'default'


def test_a_plain_call_does_not_use_the_beta_endpoint() -> None:
    wire = Wire()
    _, client = _client(wire)
    asyncio.run(client.chat.completions.create(model='claude-sonnet-5-5', messages=[{'role': 'user', 'content': 'hi'}]))
    assert 'beta' not in wire.sent.url.params
    assert 'anthropic-beta' not in wire.sent.headers


def test_orq_only_request_fields_never_reach_the_wire() -> None:
    wire = Wire()
    _, client = _client(wire)
    asyncio.run(
        client.chat.completions.create(
            model='claude-sonnet-5-5',
            messages=[{'role': 'user', 'content': 'hi'}],
            metadata={'run': '1'},
            extra_body={'retry': {'count': 2}, 'thread': {'id': 't'}, 'memory': {'entity_id': 'e'}},
        )
    )
    body = wire.body
    assert not {'metadata', 'retry', 'thread', 'memory'} & set(body)


def test_every_orq_only_field_the_adapter_strips_is_logged_at_debug() -> None:
    from loguru import logger

    wire = Wire()
    _, client = _client(wire)
    seen: list[str] = []
    sink = logger.add(lambda message: seen.append(str(message)), level='DEBUG')
    try:
        asyncio.run(
            client.chat.completions.create(
                model='claude-sonnet-5-5',
                messages=[{'role': 'user', 'content': 'hi'}],
                metadata={'run': '1'},
                reasoning_effort='none',
                extra_body={'retry': {'count': 2}, 'thread': {'id': 't'}, 'memory': {'entity_id': 'e'}},
            )
        )
    finally:
        logger.remove(sink)
    text = '\n'.join(seen)
    for name in ("'metadata'", "extra_body['retry']", "extra_body['thread']", "extra_body['memory']", "reasoning_effort='none'"):
        assert name in text, name


def test_marked_responses_input_reaches_the_wire_as_cache_control_blocks() -> None:
    wire = Wire()
    _, client = _client(wire)
    items = [
        {'role': 'user', 'content': [{'type': 'input_text', 'text': _LONG}]},
        {'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'second'}]},
        {'role': 'user', 'content': [{'type': 'input_text', 'text': 'third'}]},
    ]
    marked = mark_responses_input(items, volatile_items=1)
    asyncio.run(client.responses.create(model='claude-sonnet-5-5', input=marked, instructions='sys'))
    messages = wire.body['messages']
    assert messages[1]['content'] == [{'type': 'text', 'text': 'second', 'cache_control': {'type': 'ephemeral'}}]
    assert 'cache_control' not in json.dumps(messages[0])
    assert 'cache_control' not in json.dumps(messages[2])


def test_marked_chat_messages_reach_the_wire_as_cache_control_blocks() -> None:
    wire = Wire()
    _, client = _client(wire)
    messages = [
        {'role': 'system', 'content': _LONG},
        {'role': 'user', 'content': 'first'},
        {'role': 'assistant', 'content': 'second'},
        {'role': 'user', 'content': 'third'},
    ]
    asyncio.run(
        client.chat.completions.create(
            model='claude-sonnet-5-5', messages=apply_cache_breakpoints(messages, volatile_tail=0)
        )
    )
    body = wire.body
    assert body['system'] == [{'type': 'text', 'text': _LONG, 'cache_control': {'type': 'ephemeral'}}]
    assert body['messages'][-1]['content'] == [
        {'type': 'text', 'text': 'third', 'cache_control': {'type': 'ephemeral'}}
    ]
    assert 'cache_control' not in json.dumps(body['messages'][:-1])


def test_bedrock_and_vertex_clients_reject_fallbacks_before_any_request() -> None:
    wire = Wire()
    http = httpx.AsyncClient(transport=wire.transport)
    clients = (
        anthropic.AsyncAnthropicBedrock(
            aws_region='eu-west-1', aws_access_key='a', aws_secret_key='b', http_client=http
        ),
        anthropic.AsyncAnthropicVertex(region='europe-west1', project_id='p', access_token='t', http_client=http),
    )
    for raw in clients:
        client = as_llm_client(raw)
        assert isinstance(client, AnthropicLLMClient)
        with pytest.raises(ValueError, match='Claude API only'):
            asyncio.run(
                client.chat.completions.create(
                    model='claude-sonnet-5-5', messages=[{'role': 'user', 'content': 'hi'}], fallbacks='default'
                )
            )
    assert wire.requests == []


def test_a_subclass_of_a_sdk_client_is_wrapped() -> None:
    class MyAnthropic(anthropic.AsyncAnthropic):
        pass

    assert isinstance(as_llm_client(MyAnthropic(api_key='test')), AnthropicLLMClient)


def test_a_sdk_client_and_its_wrapper_are_both_valid_llm_clients_for_the_type_checker() -> None:
    from evaluatorq.anthropic import wrap_anthropic_client

    raw: LLMClient = anthropic.AsyncAnthropic(api_key='test')
    wrapped: LLMClient = wrap_anthropic_client(anthropic.AsyncAnthropic(api_key='test'))
    assert raw is not wrapped


# ---------------------------------------------------------------------------
# Conversion error paths: each raises at the call, before a request is sent
# ---------------------------------------------------------------------------


def _chat(wire: Wire, **params: Any) -> None:
    _, client = _client(wire)
    params.setdefault('model', 'claude-sonnet-5-5')
    asyncio.run(client.chat.completions.create(**params))


def _responses(wire: Wire, **params: Any) -> None:
    _, client = _client(wire)
    params.setdefault('model', 'claude-sonnet-5-5')
    asyncio.run(client.responses.create(**params))


def _assistant_call(arguments: str) -> dict[str, Any]:
    return {
        'role': 'assistant',
        'content': None,
        'tool_calls': [{'id': 'call_1', 'type': 'function', 'function': {'name': 'lookup', 'arguments': arguments}}],
    }


def test_chat_tool_call_arguments_that_are_not_json_raise_a_value_error() -> None:
    wire = Wire()
    with pytest.raises(ValueError, match='not valid JSON'):
        _chat(wire, messages=[{'role': 'user', 'content': 'x'}, _assistant_call('{not json')])
    assert wire.requests == []


def test_chat_tool_call_arguments_that_are_not_an_object_raise_a_type_error() -> None:
    wire = Wire()
    with pytest.raises(TypeError, match='not a JSON object'):
        _chat(wire, messages=[{'role': 'user', 'content': 'x'}, _assistant_call('[1, 2]')])
    assert wire.requests == []


def test_a_tool_result_part_that_is_not_text_raises() -> None:
    wire = Wire()
    messages = [
        {'role': 'user', 'content': 'x'},
        _assistant_call('{}'),
        {
            'role': 'tool',
            'tool_call_id': 'call_1',
            'content': [{'type': 'image_url', 'image_url': {'url': 'https://example.com/a.png'}}],
        },
    ]
    with pytest.raises(ValueError, match='only text parts are supported'):
        _chat(wire, messages=messages)
    assert wire.requests == []


@pytest.mark.parametrize('tool_choice', ['required', {'type': 'function', 'function': {'name': 'lookup'}}])
def test_a_forced_tool_choice_without_tools_raises(tool_choice: Any) -> None:
    wire = Wire()
    with pytest.raises(ValueError, match='cannot send tool_choice'):
        _chat(wire, messages=[{'role': 'user', 'content': 'x'}], tool_choice=tool_choice)
    assert wire.requests == []


def test_a_responses_function_call_with_bad_arguments_raises() -> None:
    wire = Wire()
    items = [
        {'role': 'user', 'content': 'x'},
        {'type': 'function_call', 'call_id': 'call_1', 'name': 'lookup', 'arguments': '{oops'},
    ]
    with pytest.raises(ValueError, match='not valid JSON'):
        _responses(wire, input=items)
    assert wire.requests == []


def test_flat_responses_tools_are_sent_as_anthropic_tools() -> None:
    wire = Wire()
    tool = {
        'type': 'function',
        'name': 'lookup',
        'description': 'Find an order.',
        'parameters': {'type': 'object', 'properties': {'id': {'type': 'integer'}}},
    }
    _responses(wire, input='hi', tools=[tool])
    assert wire.body['tools'] == [
        {
            'name': 'lookup',
            'description': 'Find an order.',
            'input_schema': {'type': 'object', 'properties': {'id': {'type': 'integer'}}},
        }
    ]


def test_a_non_function_responses_tool_raises() -> None:
    wire = Wire()
    with pytest.raises(ValueError, match='only supports function tools'):
        _responses(wire, input='hi', tools=[{'type': 'web_search'}])
    assert wire.requests == []



# ---------------------------------------------------------------------------
# Stop reasons the installed SDK does not know yet still arrive through its real parser
# ---------------------------------------------------------------------------


def test_a_context_window_stop_reads_as_a_truncation_on_both_faces() -> None:
    wire = Wire({**_REPLY, 'stop_reason': 'model_context_window_exceeded'})
    _, client = _client(wire)
    completion = asyncio.run(
        client.chat.completions.create(model='claude-sonnet-5-5', messages=[{'role': 'user', 'content': 'hi'}])
    )
    assert completion.choices[0].finish_reason == 'length'
    response = asyncio.run(client.responses.create(model='claude-sonnet-5-5', input='hi'))
    assert response.status == 'incomplete'
    assert response.incomplete_details is not None
    assert response.incomplete_details.reason == 'max_output_tokens'


def test_an_unknown_stop_reason_is_logged_once_and_read_as_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    from loguru import logger

    from evaluatorq.anthropic import convert

    monkeypatch.setattr(convert, '_unknown_stop_reasons_logged', set())
    seen: list[str] = []
    sink = logger.add(lambda message: seen.append(str(message)), level='WARNING')
    try:
        wire = Wire({**_REPLY, 'stop_reason': 'brand_new_reason'})
        _, client = _client(wire)
        for _ in range(2):
            completion = asyncio.run(
                client.chat.completions.create(model='claude-sonnet-5-5', messages=[{'role': 'user', 'content': 'hi'}])
            )
            assert completion.choices[0].finish_reason == 'stop'
        response = asyncio.run(client.responses.create(model='claude-sonnet-5-5', input='hi'))
        assert response.status == 'completed'
    finally:
        logger.remove(sink)
    assert len([line for line in seen if 'brand_new_reason' in line]) == 1
