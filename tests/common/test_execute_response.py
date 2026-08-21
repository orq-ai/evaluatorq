"""Tests for the Responses-API mechanic in ``common.llm_call.execute_response``.

Mirrors ``tests/common/test_llm_call.py`` (the Chat Completions sibling), plus the
Responses-only bits that aren't exercised elsewhere: the ``.parse``/``.create``
branch, the nested ``reasoning`` block drop-retry-once + memoization, and that an
unrelated 400 propagates untouched (the judge's chat fallback depends on that
propagation to know the Responses endpoint itself is fine).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai import BadRequestError
from pydantic import BaseModel

from evaluatorq.common import llm_call
from evaluatorq.common.llm_call import execute_response
from evaluatorq.common.responses import responses_text_config


class _Verdict(BaseModel):
    value: bool
    explanation: str


def _bad_request(message: str) -> BadRequestError:
    request = httpx.Request('POST', 'https://example/v1/responses')
    response = httpx.Response(400, request=request)
    return BadRequestError(message, response=response, body={'error': {'message': message}})


def _fake_response() -> MagicMock:
    resp = MagicMock()
    resp.output_text = 'ok'
    resp.usage = None
    return resp


def _fake_parsed_response(parsed: _Verdict | None = None) -> MagicMock:
    resp = MagicMock()
    resp.output_parsed = parsed if parsed is not None else _Verdict(value=True, explanation='ok')
    resp.usage = None
    return resp


def _client() -> MagicMock:
    client = MagicMock()
    client.responses.create = AsyncMock(return_value=_fake_response())
    client.responses.parse = AsyncMock(return_value=_fake_parsed_response())
    return client


@pytest.mark.asyncio
async def test_uses_create_when_no_response_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('evaluatorq.common.llm_call.get_trace_context_headers', AsyncMock(return_value={}))
    client = _client()

    response, usage = await execute_response(
        client=client,
        model='gpt-x',
        messages=[{'role': 'user', 'content': 'hi'}],
        span=None,
        timeout_s=5.0,
    )

    assert response.output_text == 'ok'
    assert usage is None
    client.responses.create.assert_awaited_once()
    client.responses.parse.assert_not_awaited()
    kwargs = client.responses.create.call_args.kwargs
    assert kwargs['model'] == 'gpt-x'
    assert kwargs['input'] == [{'role': 'user', 'content': 'hi'}]


@pytest.mark.asyncio
async def test_uses_parse_when_response_model_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('evaluatorq.common.llm_call.get_trace_context_headers', AsyncMock(return_value={}))
    client = _client()

    response, _usage = await execute_response(
        client=client,
        model='gpt-x',
        messages=[{'role': 'user', 'content': 'hi'}],
        span=None,
        timeout_s=5.0,
        response_model=_Verdict,
    )

    assert response.output_parsed == _Verdict(value=True, explanation='ok')
    client.responses.parse.assert_awaited_once()
    client.responses.create.assert_not_awaited()
    kwargs = client.responses.parse.call_args.kwargs
    assert kwargs['text_format'] is _Verdict
    assert kwargs['model'] == 'gpt-x'


@pytest.mark.asyncio
async def test_can_return_raw_response_while_sending_the_model_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('evaluatorq.common.llm_call.get_trace_context_headers', AsyncMock(return_value={}))
    client = _client()

    await execute_response(
        client=client,
        model='gpt-x',
        messages=[{'role': 'user', 'content': 'hi'}],
        span=None,
        timeout_s=5.0,
        response_text_format=_Verdict,
    )

    client.responses.create.assert_awaited_once()
    client.responses.parse.assert_not_awaited()
    text_config = client.responses.create.call_args.kwargs['text']
    assert text_config == responses_text_config(_Verdict)
    assert text_config['format']['schema']['additionalProperties'] is False


@pytest.mark.asyncio
async def test_drops_reasoning_and_retries_once_on_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('evaluatorq.common.llm_call.get_trace_context_headers', AsyncMock(return_value={}))
    # `logger.warning` on the rejection path uses loguru-style `{}` placeholders,
    # which stdlib `logging` (used in tests) chokes on when a handler formats the
    # record eagerly (e.g. pytest's log capture) — pre-existing elsewhere (see
    # tests/common/test_llm_call.py's analogous chat-path test). Not this test's
    # concern, so the logger is stubbed rather than exercising that bug here.
    monkeypatch.setattr('evaluatorq.common.llm_call.logger', MagicMock())
    client = _client()
    client.responses.create = AsyncMock(
        side_effect=[_bad_request("Unknown parameter: 'reasoning'"), _fake_response()]
    )

    response, _usage = await execute_response(
        client=client,
        model='m',
        messages=[{'role': 'user', 'content': 'x'}],
        span=None,
        timeout_s=5.0,
        extra_kwargs={'reasoning': {'effort': 'low'}},
    )

    assert response.output_text == 'ok'
    assert client.responses.create.await_count == 2
    retry_call = client.responses.create.await_args
    assert retry_call is not None
    assert 'reasoning' not in retry_call.kwargs
    # Memoized so a later call to this (model, has_tools) shape strips up front.
    assert ('m', False) in llm_call._RESPONSES_REASONING_REJECTORS


@pytest.mark.asyncio
async def test_memoized_rejection_strips_reasoning_up_front_no_second_400(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr('evaluatorq.common.llm_call.get_trace_context_headers', AsyncMock(return_value={}))
    llm_call._RESPONSES_REASONING_REJECTORS.add(('m', False))
    client = _client()

    await execute_response(
        client=client,
        model='m',
        messages=[{'role': 'user', 'content': 'x'}],
        span=None,
        timeout_s=5.0,
        extra_kwargs={'reasoning': {'effort': 'low'}},
    )

    assert client.responses.create.await_count == 1
    args = client.responses.create.await_args
    assert args is not None
    assert 'reasoning' not in args.kwargs


@pytest.mark.asyncio
async def test_unrelated_400_propagates_without_stripping_reasoning(monkeypatch: pytest.MonkeyPatch) -> None:
    """A content-policy or schema 400 must not be mistaken for a reasoning rejection.

    This matters beyond this function: the judge's Responses->chat fallback memo
    is fed by exactly this propagation. Getting it backwards either poisons the
    memo (treating an unrelated failure as "this model rejects reasoning") or,
    inverted, could loop retrying forever.
    """
    monkeypatch.setattr('evaluatorq.common.llm_call.get_trace_context_headers', AsyncMock(return_value={}))
    client = _client()
    client.responses.create = AsyncMock(side_effect=_bad_request('content management policy violation'))

    with pytest.raises(BadRequestError, match='content management policy violation'):
        await execute_response(
            client=client,
            model='m',
            messages=[{'role': 'user', 'content': 'x'}],
            span=None,
            timeout_s=5.0,
            extra_kwargs={'reasoning': {'effort': 'low'}},
        )

    assert client.responses.create.await_count == 1
    assert ('m', False) not in llm_call._RESPONSES_REASONING_REJECTORS


@pytest.mark.asyncio
async def test_reraises_400_when_reasoning_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('evaluatorq.common.llm_call.get_trace_context_headers', AsyncMock(return_value={}))
    client = _client()
    client.responses.create = AsyncMock(side_effect=_bad_request('reasoning not supported'))

    with pytest.raises(BadRequestError):
        await execute_response(
            client=client,
            model='m',
            messages=[{'role': 'user', 'content': 'x'}],
            span=None,
            timeout_s=5.0,
        )
    assert client.responses.create.await_count == 1


@pytest.mark.asyncio
async def test_propagates_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('evaluatorq.common.llm_call.get_trace_context_headers', AsyncMock(return_value={}))
    client = _client()
    client.responses.create = AsyncMock(side_effect=asyncio.TimeoutError)
    with pytest.raises(asyncio.TimeoutError):
        await execute_response(
            client=client,
            model='m',
            messages=[{'role': 'user', 'content': 'x'}],
            span=None,
            timeout_s=5.0,
        )


@pytest.mark.asyncio
async def test_timeout_s_is_applied_to_a_slow_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """A call that outlasts ``timeout_s`` is cut off by ``asyncio.wait_for``."""
    monkeypatch.setattr('evaluatorq.common.llm_call.get_trace_context_headers', AsyncMock(return_value={}))
    client = _client()

    async def _slow(**_kwargs: object) -> MagicMock:
        await asyncio.sleep(10)
        return _fake_response()

    client.responses.create = AsyncMock(side_effect=_slow)

    with pytest.raises(asyncio.TimeoutError):
        await execute_response(
            client=client,
            model='m',
            messages=[{'role': 'user', 'content': 'x'}],
            span=None,
            timeout_s=0.01,
        )


@pytest.mark.asyncio
async def test_records_input_and_response_on_span(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('evaluatorq.common.llm_call.get_trace_context_headers', AsyncMock(return_value={}))
    record_input = MagicMock()
    record_response = MagicMock()
    monkeypatch.setattr('evaluatorq.common.llm_call.record_llm_input', record_input)
    monkeypatch.setattr('evaluatorq.common.llm_call.record_llm_response', record_response)
    client = _client()
    span = MagicMock()
    await execute_response(
        client=client,
        model='m',
        messages=[{'role': 'user', 'content': 'x'}],
        span=span,
        timeout_s=5.0,
    )
    record_input.assert_called_once()
    record_response.assert_called_once()


@pytest.mark.asyncio
async def test_injects_trace_headers_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        'evaluatorq.common.llm_call.get_trace_context_headers',
        AsyncMock(return_value={'traceparent': 'abc'}),
    )
    client = _client()

    await execute_response(
        client=client,
        model='m',
        messages=[{'role': 'user', 'content': 'x'}],
        span=None,
        timeout_s=5.0,
        inject_trace_headers=True,
    )
    kwargs = client.responses.create.call_args.kwargs
    assert kwargs['extra_headers'] == {'traceparent': 'abc'}


@pytest.mark.asyncio
async def test_trace_headers_merged_with_existing_extra_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        'evaluatorq.common.llm_call.get_trace_context_headers',
        AsyncMock(return_value={'traceparent': 'trace-val'}),
    )
    client = _client()

    await execute_response(
        client=client,
        model='m',
        messages=[{'role': 'user', 'content': 'x'}],
        span=None,
        timeout_s=5.0,
        inject_trace_headers=True,
        extra_kwargs={'extra_headers': {'x-custom': 'existing'}},
    )
    kwargs = client.responses.create.call_args.kwargs
    assert kwargs['extra_headers'] == {'x-custom': 'existing', 'traceparent': 'trace-val'}


@pytest.mark.asyncio
async def test_no_trace_headers_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        'evaluatorq.common.llm_call.get_trace_context_headers', AsyncMock(return_value={'traceparent': 'abc'})
    )
    client = _client()
    await execute_response(
        client=client,
        model='m',
        messages=[{'role': 'user', 'content': 'x'}],
        span=None,
        timeout_s=5.0,
        inject_trace_headers=False,
    )
    assert 'extra_headers' not in client.responses.create.call_args.kwargs


@pytest.mark.asyncio
async def test_trace_headers_reach_parse_path_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ``.parse`` branch must get the same trace-header treatment as ``.create``."""
    monkeypatch.setattr(
        'evaluatorq.common.llm_call.get_trace_context_headers',
        AsyncMock(return_value={'traceparent': 'abc'}),
    )
    client = _client()

    await execute_response(
        client=client,
        model='m',
        messages=[{'role': 'user', 'content': 'x'}],
        span=None,
        timeout_s=5.0,
        response_model=_Verdict,
    )
    kwargs = client.responses.parse.call_args.kwargs
    assert kwargs['extra_headers'] == {'traceparent': 'abc'}


def _orq_client() -> MagicMock:
    client = MagicMock()
    client.base_url = 'https://my.orq.ai/v3/router'
    client.responses.create = AsyncMock(return_value=_fake_response())
    client.responses.parse = AsyncMock(return_value=_fake_parsed_response())
    return client


@pytest.mark.asyncio
async def test_pipeline_metadata_reaches_responses_params(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('evaluatorq.common.llm_call.get_trace_context_headers', AsyncMock(return_value={}))
    from evaluatorq.common.thread_context import evaluatorq_pipeline

    client = _orq_client()
    with evaluatorq_pipeline('red_teaming'):
        await execute_response(
            client=client,
            model='m',
            messages=[{'role': 'user', 'content': 'x'}],
            span=None,
            timeout_s=5.0,
        )
    kwargs = client.responses.create.call_args.kwargs
    assert kwargs['metadata'] == {'evaluatorq_pipeline': 'red_teaming'}


@pytest.mark.asyncio
async def test_pipeline_metadata_reaches_parse_path_too(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('evaluatorq.common.llm_call.get_trace_context_headers', AsyncMock(return_value={}))
    from evaluatorq.common.thread_context import evaluatorq_pipeline

    client = _orq_client()
    with evaluatorq_pipeline('red_teaming'):
        await execute_response(
            client=client,
            model='m',
            messages=[{'role': 'user', 'content': 'x'}],
            span=None,
            timeout_s=5.0,
            response_model=_Verdict,
        )
    kwargs = client.responses.parse.call_args.kwargs
    assert kwargs['metadata'] == {'evaluatorq_pipeline': 'red_teaming'}


@pytest.mark.asyncio
async def test_caller_metadata_wins_over_pipeline_label(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('evaluatorq.common.llm_call.get_trace_context_headers', AsyncMock(return_value={}))
    from evaluatorq.common.thread_context import evaluatorq_pipeline

    client = _orq_client()
    with evaluatorq_pipeline('red_teaming'):
        await execute_response(
            client=client,
            model='m',
            messages=[{'role': 'user', 'content': 'x'}],
            span=None,
            timeout_s=5.0,
            extra_kwargs={'metadata': {'evaluatorq_pipeline': 'caller', 'k': 'v'}},
        )
    kwargs = client.responses.create.call_args.kwargs
    assert kwargs['metadata'] == {'evaluatorq_pipeline': 'caller', 'k': 'v'}
