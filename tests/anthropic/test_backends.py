"""Red-team target backends on an Anthropic client."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, cast
from unittest.mock import patch

import anthropic
import httpx
import pytest

from evaluatorq.anthropic import AnthropicLLMClient
from evaluatorq.common.llm_client import as_llm_client
from evaluatorq.common.retry import without_client_retries
from evaluatorq.common.thread_context import evaluatorq_pipeline
from evaluatorq.contracts import Message
from evaluatorq.redteam.backends.openai import OpenAIBackend, OpenAIModelTarget
from evaluatorq.redteam.backends.openresponses import OpenResponsesBackend
from evaluatorq.redteam.backends.orq import ORQBackend
from evaluatorq.redteam.backends.registry import resolve_backend

from .test_wiring import RawAnthropic, _orchestrator, _run_attack, _SPANS, _target, scripted, text


def test_without_client_retries_turns_the_sdk_default_into_zero_on_the_adapter() -> None:
    raw = anthropic.AsyncAnthropic(api_key='sk-ant-not-used')
    assert raw.max_retries == 2
    adapter = as_llm_client(raw)
    assert isinstance(adapter, AnthropicLLMClient)
    stripped = without_client_retries(adapter)
    assert isinstance(stripped, AnthropicLLMClient)
    assert stripped is not adapter
    assert stripped.max_retries == 0
    assert stripped._client.max_retries == 0
    assert raw.max_retries == 2, 'the injected client is never mutated'


def test_the_openai_backend_target_runs_on_the_anthropic_client_with_the_agent_key_as_model() -> None:
    fake = RawAnthropic([text('Hello from Claude.')])
    # A raw client reaches the factory unwrapped when it is called directly.
    backend = resolve_backend(backend='openai', llm_client=cast('Any', fake))
    assert isinstance(backend, OpenAIBackend)
    target = backend.create_target('claude-sonnet-5-5')
    assert isinstance(target, OpenAIModelTarget)
    assert isinstance(target.client, AnthropicLLMClient)

    async def go() -> Any:
        with patch('evaluatorq.redteam.backends.openai.with_llm_span', side_effect=_noop_span_cm):
            return await target.respond([Message(role='user', content='hi')])

    response = asyncio.run(go())
    assert response.text == 'Hello from Claude.'
    (request,) = fake.messages.calls
    assert request['model'] == 'claude-sonnet-5-5'


def test_the_openai_backend_does_not_stack_sdk_retries_under_the_target_retry() -> None:
    raw = anthropic.AsyncAnthropic(api_key='sk-ant-not-used')
    backend = OpenAIBackend(client=cast('Any', raw))
    assert backend._client.max_retries == 0
    assert backend.create_target('claude-sonnet-5-5').client.max_retries == 0
    assert raw.max_retries == 2


@pytest.mark.parametrize('reasoning_effort', [None, 'low', 'high'])
def test_the_target_request_fields_are_accepted_or_dropped_by_the_chat_face(reasoning_effort: str | None) -> None:
    fake = RawAnthropic([text('ok')])
    target = OpenAIModelTarget(
        model='claude-sonnet-5-5',
        client=cast('Any', as_llm_client(fake)),
        max_tokens=1234,
        reasoning_effort=reasoning_effort,
    )
    async def go() -> None:
        with evaluatorq_pipeline('red_teaming'), patch(
            'evaluatorq.redteam.backends.openai.with_llm_span', side_effect=_noop_span_cm
        ):
            await target.respond([Message(role='user', content='hi')])

    asyncio.run(go())
    (request,) = fake.messages.calls
    assert request['max_tokens'] == 1234
    assert 'metadata' not in request, 'Anthropic metadata takes only user_id, so the run tags are dropped'
    assert 'reasoning_effort' not in request
    if reasoning_effort:
        assert request['output_config']['effort'] == reasoning_effort
    else:
        assert 'effort' not in request.get('output_config', {})


def test_router_gated_extra_body_never_reaches_anthropic() -> None:
    fake = RawAnthropic([text('ok')])
    adapter = as_llm_client(fake)
    with patch('evaluatorq.redteam.backends.openai.client_routes_through_orq', return_value=True), patch(
        'evaluatorq.redteam.backends.openai.thread_body_param', return_value={'thread': {'id': 't1'}}
    ), patch('evaluatorq.redteam.backends.openai.with_llm_span', side_effect=_noop_span_cm):
        target = OpenAIModelTarget(model='claude-sonnet-5-5', client=cast('Any', adapter))
        asyncio.run(target.respond([Message(role='user', content='hi')]))
    (request,) = fake.messages.calls
    assert 'thread' not in request.get('extra_body', {})


def test_the_openresponses_backend_rejects_an_anthropic_client_at_construction() -> None:
    for client in (RawAnthropic(), as_llm_client(RawAnthropic())):
        with pytest.raises(ValueError, match="backend='openai'") as info:
            OpenResponsesBackend(client=cast('Any', client))
        assert 'Orq router' in str(info.value)
    with pytest.raises(ValueError, match='openresponses'):
        resolve_backend(backend='openresponses', llm_client=cast('Any', RawAnthropic()))


def test_an_orq_backend_built_beside_an_anthropic_client_leaves_the_attacker_on_the_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('ORQ_API_KEY', 'orq-not-used')
    fake = scripted([text('Here is the attack prompt.')])
    backend = resolve_backend(backend='orq', llm_client=cast('Any', as_llm_client(fake)))
    assert isinstance(backend, ORQBackend)
    target = _target()
    with _SPANS[0], _SPANS[1]:
        result = _run_attack(_orchestrator(fake), target)
    assert result.error is None
    assert len(fake.messages.calls) >= 1
    assert target.respond.call_count == 1


def test_a_claude_target_that_refuses_replies_with_the_refusal_text() -> None:
    fake = RawAnthropic([text('I cannot help with that.', stop_reason='refusal')])
    target = OpenAIModelTarget(model='claude-sonnet-5-5', client=cast('Any', as_llm_client(fake)))

    async def go() -> Any:
        with patch('evaluatorq.redteam.backends.openai.with_llm_span', side_effect=_noop_span_cm):
            return await target.respond([Message(role='user', content='do the bad thing')])

    response = asyncio.run(go())
    assert response.text == 'I cannot help with that.'
    assert response.finish_reason == 'content_filter'


@asynccontextmanager
async def _noop_span_cm(*args: object, **kwargs: object):  # noqa: RUF029
    yield None


def test_an_anthropic_target_labels_its_errors_and_context_as_anthropic() -> None:
    target = OpenAIModelTarget(model='claude-sonnet-5-5', client=cast('Any', as_llm_client(RawAnthropic())))
    error = anthropic.InternalServerError(
        'boom', response=httpx.Response(500, request=httpx.Request('POST', 'https://x')), body=None
    )
    code, message = target.map_error(error)
    assert code == 'anthropic.http.500'
    assert 'InternalServerError' in message
    assert asyncio.run(target.get_agent_context()).description == 'Anthropic model target'


def test_an_openai_target_keeps_its_openai_labels() -> None:
    target = OpenAIModelTarget(model='gpt-5-mini', client=cast('Any', _FakeOpenAI()))
    assert target.map_error(RuntimeError('x')) == ('openai.unknown', 'RuntimeError: x')
    assert asyncio.run(target.get_agent_context()).description == 'OpenAI model target'


class _FakeOpenAI:
    base_url = 'https://api.openai.com/v1'
    max_retries = 0

    def with_options(self, **kwargs: Any) -> _FakeOpenAI:
        return self
