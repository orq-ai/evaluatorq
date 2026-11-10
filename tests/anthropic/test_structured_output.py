"""Structured output, refusal logging and server-side fallbacks of the Anthropic adapter."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from loguru import logger
from openai import LengthFinishReasonError
from openai.types.chat import ParsedChatCompletion
from pydantic import BaseModel, Field, ValidationError

from evaluatorq.anthropic import AnthropicLLMClient
from evaluatorq.anthropic.convert import adapt_schema
from evaluatorq.common.structured_output import StructuredGenerationError, generate_structured

from .test_chat_adapter import make_message


class Answer(BaseModel):
    value: str
    score: int = Field(ge=0, le=10)


class QueueMessages:
    """``messages.create`` that replies from a queue and records every request."""

    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


class FakeClient:
    base_url = 'https://api.anthropic.com'
    max_retries = 2

    def __init__(self, replies: list[Any] | None = None, beta_replies: list[Any] | None = None) -> None:
        self.messages = QueueMessages(replies or [make_message()])
        self.beta = MagicMock()
        self.beta.messages = QueueMessages(beta_replies or [make_message()])

    def with_options(self, **kwargs: Any) -> FakeClient:
        return self


class AsyncAnthropicBedrock(FakeClient):
    pass


class AsyncAnthropicVertex(FakeClient):
    pass


def text(body: str, **kwargs: Any) -> Any:
    from anthropic.types import TextBlock

    return make_message(content=[TextBlock(type='text', text=body)], **kwargs)


def parse(fake: Any, **params: Any) -> Any:
    client = AnthropicLLMClient(fake)
    params.setdefault('model', 'claude-sonnet-5-5')
    params.setdefault('messages', [{'role': 'user', 'content': 'hi'}])
    return asyncio.run(client.chat.completions.parse(**params))


def test_parse_with_a_pydantic_class_sends_the_strict_schema_and_validates() -> None:
    fake = FakeClient([text('{"value": "ok", "score": 3}')])
    result = parse(fake, response_format=Answer, reasoning_effort='high')
    assert isinstance(result, ParsedChatCompletion)
    assert result.choices[0].message.parsed == Answer(value='ok', score=3)
    sent = fake.messages.calls[0]
    assert sent['output_config']['effort'] == 'high'
    fmt = sent['output_config']['format']
    assert fmt['type'] == 'json_schema'
    assert fmt['schema']['additionalProperties'] is False
    assert 'minimum' not in json.dumps(fmt['schema'])
    assert 'maximum' not in json.dumps(fmt['schema'])
    assert 'response_format' not in sent


def test_parse_with_a_json_schema_dict_leaves_parsed_empty() -> None:
    schema = {'type': 'object', 'properties': {'a': {'type': 'string', 'maxLength': 3}}, 'required': ['a']}
    fake = FakeClient([text('{"a": "x"}')])
    result = parse(fake, response_format={'type': 'json_schema', 'json_schema': {'name': 'n', 'schema': schema}})
    assert result.choices[0].message.parsed is None
    assert result.choices[0].message.content == '{"a": "x"}'
    sent_schema = fake.messages.calls[0]['output_config']['format']['schema']
    assert sent_schema == {
        'type': 'object',
        'properties': {'a': {'type': 'string'}},
        'required': ['a'],
        'additionalProperties': False,
    }


def test_invalid_json_raises_the_pydantic_validation_error_openai_raises() -> None:
    with pytest.raises(ValidationError):
        parse(FakeClient([text('not json')]), response_format=Answer)


def test_truncation_raises_length_finish_reason_error_with_the_completion() -> None:
    with pytest.raises(LengthFinishReasonError) as error:
        parse(FakeClient([text('{"value"', stop_reason='max_tokens')]), response_format=Answer)
    assert error.value.completion.usage is not None


def test_a_refusal_comes_back_as_message_refusal_not_an_exception() -> None:
    result = parse(FakeClient([text('I cannot help.', stop_reason='refusal')]), response_format=Answer)
    message = result.choices[0].message
    assert message.parsed is None
    assert message.refusal == 'I cannot help.'


def test_json_object_on_create_sends_a_plain_call_and_returns_text_unchanged() -> None:
    fake = FakeClient([text('```json\n{"a": 1}\n```')])
    client = AnthropicLLMClient(fake)
    completion = asyncio.run(
        client.chat.completions.create(
            model='claude-sonnet-5-5',
            messages=[{'role': 'user', 'content': 'hi'}],
            response_format={'type': 'json_object'},
        )
    )
    assert 'output_config' not in fake.messages.calls[0]
    assert completion.choices[0].message.content == '```json\n{"a": 1}\n```'


def test_json_schema_on_create_maps_like_parse() -> None:
    fake = FakeClient([text('{"value": "v", "score": 1}')])
    client = AnthropicLLMClient(fake)
    asyncio.run(
        client.chat.completions.create(
            model='claude-sonnet-5-5',
            messages=[{'role': 'user', 'content': 'hi'}],
            response_format={
                'type': 'json_schema',
                'json_schema': {'name': 'Answer', 'strict': False, 'schema': Answer.model_json_schema()},
            },
        )
    )
    schema = fake.messages.calls[0]['output_config']['format']['schema']
    assert schema['additionalProperties'] is False
    assert 'maximum' not in json.dumps(schema)
    assert '$defs' not in schema or isinstance(schema['$defs'], dict)


def test_adapt_schema_keeps_properties_named_like_keywords_and_walks_defs() -> None:
    schema = {
        'type': 'object',
        'properties': {
            'minimum': {'type': 'integer', 'minimum': 1},
            'xs': {'type': 'array', 'items': {'$ref': '#/$defs/A'}, 'minItems': 2},
        },
        '$defs': {'A': {'type': 'object', 'properties': {}, 'maxItems': 4}},
    }
    out = adapt_schema(schema)
    assert 'minimum' in out['properties']
    assert out['properties']['minimum'] == {'type': 'integer'}
    assert 'minItems' not in out['properties']['xs']
    assert out['$defs']['A'] == {'type': 'object', 'properties': {}, 'additionalProperties': False}


async def _ladder(client: Any) -> Any:
    return await generate_structured(
        client,
        model='claude-sonnet-5-5',
        messages=[{'role': 'user', 'content': 'go'}],
        response_format=Answer,
        max_tokens=64,
        label='ladder',
    )


def test_ladder_skips_the_forced_tool_rung_for_the_adapter() -> None:
    empty = text('')
    fake = FakeClient([empty, empty, text('{"value": "ok", "score": 2}')])
    result = asyncio.run(_ladder(AnthropicLLMClient(fake)))
    assert result.parsed == Answer(value='ok', score=2)
    assert len(fake.messages.calls) == 3
    assert all('tools' not in call and 'tool_choice' not in call for call in fake.messages.calls)


def test_ladder_still_forces_a_tool_call_for_an_openai_style_client() -> None:
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.parsed_chat_completion import ParsedChatCompletionMessage, ParsedChoice

    def parsed_none() -> ParsedChatCompletion[Answer]:
        return ParsedChatCompletion[Answer](
            id='c',
            created=0,
            model='m',
            object='chat.completion',
            choices=[
                ParsedChoice[Answer](
                    index=0,
                    finish_reason='stop',
                    message=ParsedChatCompletionMessage[Answer](role='assistant', content=None),
                )
            ],
        )

    def plain(body: str) -> ChatCompletion:
        return ChatCompletion(
            id='c',
            created=0,
            model='m',
            object='chat.completion',
            choices=[
                Choice(index=0, finish_reason='stop', message=ChatCompletionMessage(role='assistant', content=body))
            ],
        )

    client = MagicMock()
    client.chat.completions.parse = AsyncMock(side_effect=[parsed_none(), parsed_none()])
    client.chat.completions.create = AsyncMock(side_effect=[plain('nope'), plain('{"value": "ok", "score": 2}')])
    result = asyncio.run(_ladder(client))
    assert result.parsed == Answer(value='ok', score=2)
    assert client.chat.completions.parse.await_count == 2
    assert client.chat.completions.parse.await_args_list[1].kwargs['tool_choice']['type'] == 'function'


def test_ladder_reads_an_adapter_refusal_as_a_model_refusal() -> None:
    fake = FakeClient([text('No.', stop_reason='refusal')])
    with pytest.raises(StructuredGenerationError, match='refused'):
        asyncio.run(_ladder(AnthropicLLMClient(fake)))


def _capture() -> tuple[list[str], int]:
    seen: list[str] = []
    return seen, logger.add(lambda m: seen.append(str(m)), level='WARNING')


def test_refusal_warning_is_logged_once_per_adapter_instance() -> None:
    seen, handler = _capture()
    try:
        client = AnthropicLLMClient(FakeClient([text('No.', stop_reason='refusal')]))
        for _ in range(3):
            asyncio.run(
                client.chat.completions.create(model='claude-sonnet-5-5', messages=[{'role': 'user', 'content': 'x'}])
            )
        other = AnthropicLLMClient(FakeClient([text('No.', stop_reason='refusal')]))
        asyncio.run(
            other.chat.completions.create(model='claude-sonnet-5-5', messages=[{'role': 'user', 'content': 'x'}])
        )
    finally:
        logger.remove(handler)
    refusals = [line for line in seen if 'refused' in line]
    assert len(refusals) == 2
    assert 'fallbacks' in refusals[0]
    assert "'default'" in refusals[0] or '"default"' in refusals[0]


def test_no_refusal_warning_for_a_normal_reply() -> None:
    seen, handler = _capture()
    try:
        asyncio.run(
            AnthropicLLMClient(FakeClient()).chat.completions.create(
                model='claude-sonnet-5-5', messages=[{'role': 'user', 'content': 'x'}]
            )
        )
    finally:
        logger.remove(handler)
    assert not any('refused' in line for line in seen)


def test_fallbacks_route_to_the_beta_call_through_extra_body() -> None:
    fake = FakeClient()
    client = AnthropicLLMClient(fake)
    completion = asyncio.run(
        client.chat.completions.create(
            model='claude-opus-5-5',
            messages=[{'role': 'user', 'content': 'x'}],
            fallbacks='default',
            extra_body={'keep': 1, 'retry': {'count': 1}},
        )
    )
    assert fake.messages.calls == []
    sent = fake.beta.messages.calls[0]
    assert sent['betas'] == ['server-side-fallback-2026-07-01']
    assert sent['extra_body'] == {'keep': 1, 'fallbacks': 'default'}
    assert 'fallbacks' not in sent
    assert completion.choices[0].message.content == 'hello'


def test_fallbacks_work_for_parse_too() -> None:
    fake = FakeClient(beta_replies=[text('{"value": "ok", "score": 1}')])
    result = parse(fake, response_format=Answer, fallbacks='default')
    assert result.choices[0].message.parsed == Answer(value='ok', score=1)
    assert fake.beta.messages.calls[0]['output_config']['format']['type'] == 'json_schema'


def test_without_fallbacks_the_plain_call_is_used() -> None:
    fake = FakeClient()
    asyncio.run(
        AnthropicLLMClient(fake).chat.completions.create(
            model='claude-sonnet-5-5', messages=[{'role': 'user', 'content': 'x'}]
        )
    )
    assert len(fake.messages.calls) == 1
    assert fake.beta.messages.calls == []


@pytest.mark.parametrize('cls', [AsyncAnthropicBedrock, AsyncAnthropicVertex])
def test_fallbacks_on_bedrock_or_vertex_raise_value_error(cls: type[FakeClient]) -> None:
    fake = cls()
    client = AnthropicLLMClient(fake)
    with pytest.raises(ValueError, match='Claude API only'):
        asyncio.run(
            client.chat.completions.create(
                model='claude-opus-5-5', messages=[{'role': 'user', 'content': 'x'}], fallbacks='default'
            )
        )
    assert fake.messages.calls == [] and fake.beta.messages.calls == []
