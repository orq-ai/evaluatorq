from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import textwrap
from typing import Any, cast

import anthropic
import pytest
from anthropic.types import Message, TextBlock, ThinkingBlock, ToolUseBlock, Usage
from openai.types.chat import ChatCompletion

from evaluatorq.anthropic import AnthropicLLMClient, wrap_anthropic_client
from evaluatorq.common.llm_client import client_routes_through_orq, is_anthropic_client
from evaluatorq.common.retry import without_client_retries


def make_message(content=None, stop_reason='end_turn', usage=None, model='claude-sonnet-5-5') -> Message:
    return Message(
        id='msg_1',
        type='message',
        role='assistant',
        model=model,
        content=content if content is not None else [TextBlock(type='text', text='hello')],
        stop_reason=stop_reason,
        stop_sequence=None,
        usage=usage or Usage(input_tokens=10, output_tokens=4),
    )


class FakeMessages:
    def __init__(self, reply: Message) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Message:
        self.calls.append(kwargs)
        return self.reply


class FakeAnthropic:
    base_url = 'https://api.anthropic.com'
    max_retries = 2

    def __init__(self, reply: Message | None = None) -> None:
        self.messages = FakeMessages(reply or make_message())
        self.closed = False
        self.options: list[dict[str, Any]] = []

    def with_options(self, **kwargs: Any) -> FakeAnthropic:
        self.options.append(kwargs)
        clone = FakeAnthropic(self.messages.reply)
        clone.max_retries = kwargs.get('max_retries', self.max_retries)
        return clone

    async def close(self) -> None:
        self.closed = True


def run(coro):
    return asyncio.run(coro)


def call(reply: Message | None = None, **params: Any):
    fake = FakeAnthropic(reply)
    client = AnthropicLLMClient(fake)
    params.setdefault('model', 'claude-sonnet-5-5')
    params.setdefault('messages', [{'role': 'user', 'content': 'hi'}])
    completion = run(client.chat.completions.create(**params))
    return completion, fake.messages.calls[0]


def test_returns_genuine_chat_completion():
    completion, sent = call()
    assert isinstance(completion, ChatCompletion)
    assert completion.choices[0].message.content == 'hello'
    assert completion.model == 'anthropic/claude-sonnet-5-5'
    assert sent['messages'] == [{'role': 'user', 'content': [{'type': 'text', 'text': 'hi'}]}]


def test_system_and_developer_messages_go_to_system():
    _, sent = call(
        messages=[
            {'role': 'system', 'content': 'be brief'},
            {'role': 'developer', 'content': 'be kind'},
            {'role': 'user', 'content': 'hi'},
        ]
    )
    assert sent['system'] == 'be brief\n\nbe kind'
    assert [m['role'] for m in sent['messages']] == ['user']


def test_system_keeps_cache_control_parts():
    parts = [{'type': 'text', 'text': 'rules', 'cache_control': {'type': 'ephemeral'}}]
    _, sent = call(messages=[{'role': 'system', 'content': parts}, {'role': 'user', 'content': 'hi'}])
    assert sent['system'] == [{'type': 'text', 'text': 'rules', 'cache_control': {'type': 'ephemeral'}}]


def test_assistant_tool_calls_and_merged_tool_results():
    messages = [
        {'role': 'user', 'content': 'go'},
        {
            'role': 'assistant',
            'content': 'calling',
            'tool_calls': [
                {'id': 't1', 'type': 'function', 'function': {'name': 'a', 'arguments': '{"x": 1}'}},
                {'id': 't2', 'type': 'function', 'function': {'name': 'b', 'arguments': ''}},
            ],
        },
        {'role': 'tool', 'tool_call_id': 't1', 'content': 'one'},
        {'role': 'tool', 'tool_call_id': 't2', 'content': [{'type': 'text', 'text': 'two'}]},
        {'role': 'user', 'content': 'thanks'},
    ]
    _, sent = call(messages=messages)
    assistant = sent['messages'][1]
    assert assistant['content'][1:] == [
        {'type': 'tool_use', 'id': 't1', 'name': 'a', 'input': {'x': 1}},
        {'type': 'tool_use', 'id': 't2', 'name': 'b', 'input': {}},
    ]
    assert sent['messages'][2] == {
        'role': 'user',
        'content': [
            {'type': 'tool_result', 'tool_use_id': 't1', 'content': 'one'},
            {'type': 'tool_result', 'tool_use_id': 't2', 'content': 'two'},
            {'type': 'text', 'text': 'thanks'},
        ],
    }
    assert len(sent['messages']) == 3


def test_consecutive_user_messages_merge_into_one_turn():
    _, sent = call(messages=[{'role': 'user', 'content': 'a'}, {'role': 'user', 'content': 'b'}])
    assert sent['messages'] == [
        {'role': 'user', 'content': [{'type': 'text', 'text': 'a'}, {'type': 'text', 'text': 'b'}]}
    ]


def test_consecutive_assistant_messages_merge_into_one_turn():
    _, sent = call(
        messages=[
            {'role': 'user', 'content': 'go'},
            {'role': 'assistant', 'content': 'one'},
            {'role': 'assistant', 'content': 'two'},
        ]
    )
    assert [m['role'] for m in sent['messages']] == ['user', 'assistant']
    assert sent['messages'][1]['content'] == [{'type': 'text', 'text': 'one'}, {'type': 'text', 'text': 'two'}]


def test_empty_message_list_raises_before_the_api_call():
    with pytest.raises(ValueError, match='no user or assistant content'):
        call(messages=[{'role': 'system', 'content': 'rules'}])


def test_empty_content_message_is_dropped_with_a_debug_log():
    from loguru import logger

    seen: list[str] = []
    handler = logger.add(lambda m: seen.append(str(m)), level='DEBUG')
    try:
        _, sent = call(messages=[{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': ''}])
    finally:
        logger.remove(handler)
    assert [m['role'] for m in sent['messages']] == ['user']
    assert any('empty content' in line for line in seen)


def test_image_part_raises_naming_the_type():
    with pytest.raises(ValueError, match='image_url'):
        call(messages=[{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'x'}}]}])


def test_tools_convert_and_auto_choice_passes():
    tools = [{'type': 'function', 'function': {'name': 'f', 'description': 'd', 'parameters': {'type': 'object', 'properties': {'a': {}}}}}]
    _, sent = call(tools=tools, tool_choice='auto')
    assert sent['tools'] == [{'name': 'f', 'description': 'd', 'input_schema': {'type': 'object', 'properties': {'a': {}}}}]
    assert sent['tool_choice'] == {'type': 'auto'}
    _, sent = call(tools=tools, tool_choice='none')
    assert sent['tool_choice'] == {'type': 'none'}
    _, sent = call(tools=tools)
    assert 'tool_choice' not in sent


@pytest.mark.parametrize('choice', ['required', {'type': 'function', 'function': {'name': 'f'}}])
def test_forced_tool_choice_raises(choice):
    tools = [{'type': 'function', 'function': {'name': 'f', 'parameters': {}}}]
    with pytest.raises(ValueError, match='tool_choice'):
        call(tools=tools, tool_choice=choice)


def test_tool_use_response_becomes_tool_calls():
    reply = make_message(
        [
            ThinkingBlock(type='thinking', thinking='hmm', signature='s'),
            TextBlock(type='text', text='a'),
            TextBlock(type='text', text='b'),
            ToolUseBlock(type='tool_use', id='tu1', name='f', input={'k': 'v'}),
        ],
        stop_reason='tool_use',
    )
    completion, _ = call(reply)
    choice = completion.choices[0]
    assert choice.finish_reason == 'tool_calls'
    assert choice.message.content == 'ab'
    call_ = choice.message.tool_calls[0]
    assert (call_.id, call_.function.name, json.loads(call_.function.arguments)) == ('tu1', 'f', {'k': 'v'})


@pytest.mark.parametrize(
    ('stop_reason', 'finish'),
    [
        ('end_turn', 'stop'),
        ('stop_sequence', 'stop'),
        ('pause_turn', 'stop'),
        ('max_tokens', 'length'),
        ('tool_use', 'tool_calls'),
        ('refusal', 'content_filter'),
    ],
)
def test_finish_reason_mapping(stop_reason, finish):
    completion, _ = call(make_message(stop_reason=stop_reason))
    assert completion.choices[0].finish_reason == finish


def test_refusal_uses_text_or_fixed_sentence():
    completion, _ = call(make_message([TextBlock(type='text', text='no can do')], stop_reason='refusal'))
    assert completion.choices[0].message.refusal == 'no can do'
    completion, _ = call(make_message([], stop_reason='refusal'))
    assert completion.choices[0].message.refusal


def test_usage_mapping_and_extract():
    from evaluatorq.contracts import Usage as EqUsage

    usage = Usage(input_tokens=10, output_tokens=4, cache_read_input_tokens=100, cache_creation_input_tokens=20)
    completion, _ = call(make_message(usage=usage))
    assert completion.usage.prompt_tokens == 130
    assert completion.usage.completion_tokens == 4
    assert completion.usage.total_tokens == 134
    assert completion.usage.prompt_tokens_details.cached_tokens == 100
    extracted = EqUsage.extract(completion.usage)
    assert extracted is not None
    assert extracted.input_tokens == 130
    assert extracted.cache_creation_tokens == 20


@pytest.mark.parametrize(
    ('given', 'sent'),
    [
        ('claude-sonnet-5-5', 'claude-sonnet-5-5'),
        ('anthropic/claude-sonnet-5-5', 'claude-sonnet-5-5'),
        ('anthropic/claude-opus-5-5', 'claude-opus-5-5'),
        (
            'arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-opus-5-5-v1:0',
            'arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-opus-5-5-v1:0',
        ),
        ('us.anthropic.claude-opus-5-5-v1:0', 'us.anthropic.claude-opus-5-5-v1:0'),
        ('claude-opus-5-5@20260101', 'claude-opus-5-5@20260101'),
    ],
)
def test_model_prefix_accepted(given, sent):
    _, request = call(model=given)
    assert request['model'] == sent


@pytest.mark.parametrize(
    ('model', 'named', 'not_named'),
    [
        ('openai/gpt-6-luna', 'EVALUATORQ_FAST_MODEL', 'EVALUATORQ_SMART_MODEL'),
        ('openai/gpt-6-sol', 'EVALUATORQ_SMART_MODEL', 'EVALUATORQ_FAST_MODEL'),
        ('orq/foo', 'EVALUATORQ_FAST_MODEL', None),
    ],
)
def test_other_provider_prefix_raises_naming_the_role_variable_that_resolves_to_it(model, named, not_named):
    with pytest.raises(ValueError, match=named) as caught:
        call(model=model)
    if not_named is not None:
        assert not_named not in str(caught.value)


def test_sampling_and_max_tokens_defaults():
    _, sent = call()
    assert 'temperature' not in sent
    assert 'top_p' not in sent
    assert sent['max_tokens'] == 16000
    _, sent = call(temperature=0.2, top_p=0.9, max_completion_tokens=50)
    assert (sent['temperature'], sent['top_p'], sent['max_tokens']) == (0.2, 0.9, 50)
    _, sent = call(max_tokens=70, temperature=None)
    assert sent['max_tokens'] == 70
    assert 'temperature' not in sent


def test_reasoning_effort_becomes_output_config():
    _, sent = call(reasoning_effort='high')
    assert sent['output_config'] == {'effort': 'high'}


@pytest.mark.parametrize(
    ('given', 'sent'),
    [('minimal', 'low'), ('low', 'low'), ('medium', 'medium'), ('high', 'high'), ('xhigh', 'max'), ('max', 'max')],
)
def test_reasoning_effort_mapping(given, sent):
    _, request = call(reasoning_effort=given)
    assert request['output_config'] == {'effort': sent}


def test_reasoning_effort_none_is_omitted():
    _, request = call(reasoning_effort='none')
    assert 'output_config' not in request


def test_unknown_reasoning_effort_raises_naming_allowed_values():
    with pytest.raises(ValueError, match='xhigh'):
        call(reasoning_effort='extreme')


def test_metadata_dropped_headers_and_timeout_pass_through():
    _, sent = call(metadata={'a': 'b'}, extra_headers={'x': '1'}, timeout=5)
    assert 'metadata' not in sent
    assert sent['extra_headers'] == {'x': '1'}
    assert sent['timeout'] == 5


def test_orq_only_extra_body_keys_dropped():
    _, sent = call(extra_body={'retry': {'count': 1}, 'thread': {'id': 't'}, 'memory': {'e': 1}, 'keep': 1})
    assert sent['extra_body'] == {'keep': 1}
    _, sent = call(extra_body={'retry': {'count': 1}})
    assert 'extra_body' not in sent


def test_unknown_key_raises_type_error():
    with pytest.raises(TypeError, match='seed'):
        call(seed=1)


def test_client_attributes_retry_and_close():
    fake = FakeAnthropic()
    client = wrap_anthropic_client(fake)
    assert wrap_anthropic_client(client) is client
    assert client.evaluatorq_provider == 'anthropic'
    assert is_anthropic_client(client)
    assert client.base_url == 'https://api.anthropic.com'
    assert not client_routes_through_orq(cast(Any, client))
    assert client.api_key is None
    assert client.max_retries == 2
    quiet = without_client_retries(cast(Any, client))
    assert quiet is not client
    assert isinstance(quiet, AnthropicLLMClient)
    assert quiet.max_retries == 0
    assert fake.options == [{'max_retries': 0}]
    assert without_client_retries(cast(Any, quiet)) is quiet
    run(client.close())
    assert fake.closed


def test_embeddings_not_supported():
    client = AnthropicLLMClient(FakeAnthropic())
    with pytest.raises(NotImplementedError, match='embedding_client'):
        client.embeddings.create


def test_rejects_object_without_messages_create():
    with pytest.raises(TypeError):
        AnthropicLLMClient(object())


def test_real_anthropic_client_wraps():
    client = wrap_anthropic_client(anthropic.AsyncAnthropic(api_key='k'))
    assert str(client.base_url).startswith('https://api.anthropic.com')
    assert client.max_retries == 2
    assert without_client_retries(cast(Any, client)).max_retries == 0


def test_works_with_anthropic_uninstalled():
    code = textwrap.dedent(
        """
        import sys
        sys.modules['anthropic'] = None
        import evaluatorq
        from evaluatorq.common.llm_client import is_anthropic_client
        class C:
            evaluatorq_provider = 'anthropic'
        assert is_anthropic_client(C()) and not is_anthropic_client(object())
        from evaluatorq.anthropic import wrap_anthropic_client
        async def make_create(**kwargs): ...
        wrap_anthropic_client(type('Fake', (), {'messages': type('M', (), {'create': staticmethod(make_create)})()})())
        """
    )
    result = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
