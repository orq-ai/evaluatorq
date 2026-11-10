"""Red-team target backends on an Anthropic client."""

from __future__ import annotations

import asyncio
from typing import Any, cast
from unittest.mock import patch

import anthropic
import pytest

from evaluatorq.anthropic import AnthropicLLMClient
from evaluatorq.common.llm_client import as_llm_client
from evaluatorq.contracts import Message
from evaluatorq.common.thread_context import evaluatorq_pipeline
from evaluatorq.common.retry import without_client_retries
from evaluatorq.redteam.backends.openai import OpenAIBackend, OpenAIModelTarget
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
    from evaluatorq.redteam.backends.openresponses import OpenResponsesBackend

    for client in (RawAnthropic(), as_llm_client(RawAnthropic())):
        with pytest.raises(ValueError, match="backend='openai'") as info:
            OpenResponsesBackend(client=cast('Any', client))
        assert 'Orq router' in str(info.value)
    with pytest.raises(ValueError, match='openresponses'):
        resolve_backend(backend='openresponses', llm_client=cast('Any', RawAnthropic()))


def test_the_orq_backend_ignores_an_anthropic_client_and_the_attacker_uses_the_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('ORQ_API_KEY', 'orq-not-used')
    from evaluatorq.redteam.backends.orq import ORQBackend

    fake = scripted([text('Here is the attack prompt.')])
    backend = resolve_backend(backend='orq', llm_client=cast('Any', as_llm_client(fake)))
    assert isinstance(backend, ORQBackend)
    target = _target()
    with _SPANS[0], _SPANS[1]:
        result = _run_attack(_orchestrator(fake), target)
    assert result.error is None
    assert len(fake.messages.calls) >= 1
    assert target.respond.call_count == 1


from contextlib import asynccontextmanager  # noqa: E402


@asynccontextmanager
async def _noop_span_cm(*args: object, **kwargs: object):  # noqa: RUF029
    yield None
