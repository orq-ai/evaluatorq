from __future__ import annotations

import asyncio
from typing import Any

import pytest

from evaluatorq.common.target_call import (
    call_target_with_retry,
    classify_error_type,
    default_map_error,
)
from evaluatorq.contracts import AgentResponse, AgentResponseError, Message

class _Target:
    """Minimal AgentTarget double: respond() returns queued items or raises them."""

    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)
        self.calls = 0

    async def respond(self, messages: list[Message]) -> AgentResponse:
        self.calls += 1
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _ok(text: str) -> AgentResponse:
    return AgentResponse(text=text)


def _err(msg: str) -> AgentResponse:
    return AgentResponse(
        text=msg, error=AgentResponseError(message=msg, error_type='target_error', code='x')
    )


@pytest.mark.asyncio
async def test_success_first_try():
    t = _Target([_ok('hi')])
    r = await call_target_with_retry(t, [Message(role='user', content='q')], target_agent_timeout_ms=1000, max_target_retries=2)
    assert r.succeeded is True
    assert r.error is None
    assert r.attempts == 1
    assert r.response.text == 'hi'
    assert t.calls == 1


@pytest.mark.asyncio
async def test_retry_then_succeed():
    t = _Target([_err('boom'), _ok('hi')])
    r = await call_target_with_retry(t, [Message(role='user', content='q')], target_agent_timeout_ms=1000, max_target_retries=2)
    assert r.succeeded is True
    assert r.attempts == 2
    assert t.calls == 2


@pytest.mark.asyncio
async def test_error_marker_exhausted():
    t = _Target([_err('boom'), _err('boom'), _err('boom')])
    r = await call_target_with_retry(t, [Message(role='user', content='q')], target_agent_timeout_ms=1000, max_target_retries=2)
    assert r.succeeded is False
    assert r.error is not None
    assert r.attempts == 3
    assert t.calls == 3


@pytest.mark.asyncio
async def test_timeout_becomes_synthetic_error():
    async def _hang(_messages):
        await asyncio.sleep(10)

    class _Hang:
        async def respond(self, messages):
            return await _hang(messages)

    r = await call_target_with_retry(_Hang(), [Message(role='user', content='q')], target_agent_timeout_ms=10, max_target_retries=0)
    assert r.succeeded is False
    assert r.error is not None
    assert r.error.error_type == 'timeout'
    assert r.error.code == 'target.timeout'


@pytest.mark.asyncio
async def test_generic_exception_mapped_and_classified():
    t = _Target([ConnectionError('connection reset')])
    r = await call_target_with_retry(t, [Message(role='user', content='q')], target_agent_timeout_ms=1000, max_target_retries=0)
    assert r.succeeded is False
    assert r.error is not None
    # classify_error_type maps 'connection' -> 'network_error'
    assert r.error.error_type == 'network_error'
    assert r.error_details is not None
    assert r.error_details['exception_type'] == 'ConnectionError'
    assert 'connection reset' in str(r.error_details['raw_message'])


@pytest.mark.asyncio
async def test_map_error_none_falls_back_to_default():
    t = _Target([RuntimeError('weird')])
    r = await call_target_with_retry(
        t, [Message(role='user', content='q')], target_agent_timeout_ms=1000, max_target_retries=0,
        map_error=lambda exc: None,  # pyright: ignore[reportArgumentType]
    )
    assert r.succeeded is False
    assert r.error is not None
    assert r.error.code == 'target_error'  # default_map_error code


@pytest.mark.asyncio
async def test_on_attempt_called_per_attempt():
    from contextlib import asynccontextmanager

    seen: list[int] = []

    @asynccontextmanager
    async def _span(i: int):
        seen.append(i)
        yield object()

    t = _Target([_err('boom'), _ok('hi')])
    await call_target_with_retry(
        t, [Message(role='user', content='q')], target_agent_timeout_ms=1000, max_target_retries=2, on_attempt=_span
    )
    assert seen == [0, 1]


@pytest.mark.asyncio
async def test_on_attempt_response_receives_each_target_response():
    from contextlib import asynccontextmanager

    seen: list[tuple[int, str]] = []

    @asynccontextmanager
    async def _span(i: int):
        yield i

    def _record_response(span: int, response: AgentResponse) -> None:
        seen.append((span, response.text))

    t = _Target([_err('retry me'), _ok('hi')])
    await call_target_with_retry(
        t,
        [Message(role='user', content='q')],
        target_agent_timeout_ms=1000,
        max_target_retries=2,
        on_attempt=_span,
        on_attempt_response=_record_response,
    )

    assert seen == [(0, 'retry me'), (1, 'hi')]


@pytest.mark.asyncio
async def test_cancelled_error_propagates():
    class _Cancel:
        async def respond(self, messages):
            raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await call_target_with_retry(_Cancel(), [Message(role='user', content='q')], target_agent_timeout_ms=1000, max_target_retries=2)


def test_default_map_error_shape():
    code, msg = default_map_error(ValueError('x'))
    assert code == 'target_error'
    assert 'ValueError' in msg


def test_classify_error_type_moved():
    assert classify_error_type('rate limit exceeded') == 'rate_limit'
    assert classify_error_type(None) is None
    assert classify_error_type('nonsense') == 'unknown'
