"""Tests for the shared chat-completion mechanic in ``common.llm_call``."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai import BadRequestError

from evaluatorq.common.llm_call import execute_chat_completion


def _bad_request(message: str) -> BadRequestError:
    request = httpx.Request('POST', 'https://example/v1/chat/completions')
    response = httpx.Response(400, request=request)
    return BadRequestError(message, response=response, body={'error': {'message': message}})


def _fake_response() -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = 'ok'
    resp.usage = None
    return resp


@pytest.mark.asyncio
async def test_builds_params_and_returns_response_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('evaluatorq.common.llm_call.get_trace_context_headers', AsyncMock(return_value={}))
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_fake_response())

    response, usage = await execute_chat_completion(
        client=client,
        model='gpt-x',
        messages=[{'role': 'user', 'content': 'hi'}],
        span=None,
        timeout_s=5.0,
        temperature=0.0,
        max_tokens=128,
        response_format={'type': 'json_object'},
    )

    assert response.choices[0].message.content == 'ok'
    assert usage is None
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs['model'] == 'gpt-x'
    assert kwargs['temperature'] == 0.0
    assert kwargs['max_tokens'] == 128
    assert kwargs['response_format'] == {'type': 'json_object'}


@pytest.mark.asyncio
async def test_injects_trace_headers_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        'evaluatorq.common.llm_call.get_trace_context_headers',
        AsyncMock(return_value={'traceparent': 'abc'}),
    )
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_fake_response())

    await execute_chat_completion(
        client=client,
        model='m',
        messages=[{'role': 'user', 'content': 'x'}],
        span=None,
        timeout_s=5.0,
        inject_trace_headers=True,
    )
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs['extra_headers'] == {'traceparent': 'abc'}


@pytest.mark.asyncio
async def test_trace_headers_merged_with_existing_extra_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        'evaluatorq.common.llm_call.get_trace_context_headers',
        AsyncMock(return_value={'traceparent': 'trace-val'}),
    )
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_fake_response())

    await execute_chat_completion(
        client=client,
        model='m',
        messages=[{'role': 'user', 'content': 'x'}],
        span=None,
        timeout_s=5.0,
        inject_trace_headers=True,
        extra_kwargs={'extra_headers': {'x-custom': 'existing'}},
    )
    kwargs = client.chat.completions.create.call_args.kwargs
    # Caller header survives; trace header is added (and wins on key conflict).
    assert kwargs['extra_headers'] == {'x-custom': 'existing', 'traceparent': 'trace-val'}


@pytest.mark.asyncio
async def test_no_trace_headers_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        'evaluatorq.common.llm_call.get_trace_context_headers', AsyncMock(return_value={'traceparent': 'abc'})
    )
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_fake_response())
    await execute_chat_completion(
        client=client,
        model='m',
        messages=[{'role': 'user', 'content': 'x'}],
        span=None,
        timeout_s=5.0,
        inject_trace_headers=False,
    )
    assert 'extra_headers' not in client.chat.completions.create.call_args.kwargs


@pytest.mark.asyncio
async def test_does_not_swallow_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('evaluatorq.common.llm_call.get_trace_context_headers', AsyncMock(return_value={}))
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=RuntimeError('boom'))
    with pytest.raises(RuntimeError, match='boom'):
        await execute_chat_completion(
            client=client,
            model='m',
            messages=[{'role': 'user', 'content': 'x'}],
            span=None,
            timeout_s=5.0,
        )


@pytest.mark.asyncio
async def test_extra_kwargs_and_tools_merged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('evaluatorq.common.llm_call.get_trace_context_headers', AsyncMock(return_value={}))
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_fake_response())
    await execute_chat_completion(
        client=client,
        model='m',
        messages=[{'role': 'user', 'content': 'x'}],
        span=None,
        timeout_s=5.0,
        tools=[{'type': 'function'}],
        extra_kwargs={'seed': 7},
    )
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs['tools'] == [{'type': 'function'}]
    assert kwargs['tool_choice'] == 'auto'
    assert kwargs['seed'] == 7


@pytest.mark.asyncio
async def test_records_input_and_response_on_span(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('evaluatorq.common.llm_call.get_trace_context_headers', AsyncMock(return_value={}))
    record_input = MagicMock()
    record_response = MagicMock()
    monkeypatch.setattr('evaluatorq.common.llm_call.record_llm_input', record_input)
    monkeypatch.setattr('evaluatorq.common.llm_call.record_llm_response', record_response)
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_fake_response())
    span = MagicMock()
    await execute_chat_completion(
        client=client,
        model='m',
        messages=[{'role': 'user', 'content': 'x'}],
        span=span,
        timeout_s=5.0,
    )
    record_input.assert_called_once()
    record_response.assert_called_once()


@pytest.mark.asyncio
async def test_drops_reasoning_effort_and_retries_when_model_rejects_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('evaluatorq.common.llm_call.get_trace_context_headers', AsyncMock(return_value={}))
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[_bad_request('Unknown parameter: reasoning_effort'), _fake_response()]
    )

    response, _ = await execute_chat_completion(
        client=client,
        model='m',
        messages=[{'role': 'user', 'content': 'x'}],
        span=None,
        timeout_s=5.0,
        extra_kwargs={'reasoning_effort': 'low'},
    )

    assert response.choices[0].message.content == 'ok'
    assert client.chat.completions.create.await_count == 2
    # The retry must not carry reasoning_effort.
    assert client.chat.completions.create.await_args is not None
    assert 'reasoning_effort' not in client.chat.completions.create.await_args.kwargs


@pytest.mark.asyncio
async def test_reraises_unrelated_400_without_stripping_reasoning_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('evaluatorq.common.llm_call.get_trace_context_headers', AsyncMock(return_value={}))
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=_bad_request('Invalid tools schema'))

    with pytest.raises(BadRequestError, match='Invalid tools schema'):
        await execute_chat_completion(
            client=client,
            model='m',
            messages=[{'role': 'user', 'content': 'x'}],
            span=None,
            timeout_s=5.0,
            extra_kwargs={'reasoning_effort': 'low'},
        )
    # No reasoning-stripped retry — an unrelated 400 is not masked.
    assert client.chat.completions.create.await_count == 1


@pytest.mark.asyncio
async def test_reraises_400_when_reasoning_effort_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('evaluatorq.common.llm_call.get_trace_context_headers', AsyncMock(return_value={}))
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=_bad_request('reasoning not supported'))

    with pytest.raises(BadRequestError):
        await execute_chat_completion(
            client=client,
            model='m',
            messages=[{'role': 'user', 'content': 'x'}],
            span=None,
            timeout_s=5.0,
        )
    assert client.chat.completions.create.await_count == 1


@pytest.mark.asyncio
async def test_propagates_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('evaluatorq.common.llm_call.get_trace_context_headers', AsyncMock(return_value={}))
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=asyncio.TimeoutError)
    with pytest.raises(asyncio.TimeoutError):
        await execute_chat_completion(
            client=client,
            model='m',
            messages=[{'role': 'user', 'content': 'x'}],
            span=None,
            timeout_s=5.0,
        )
