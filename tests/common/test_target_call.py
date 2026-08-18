from __future__ import annotations

import asyncio
from typing import Any

import pytest

from evaluatorq.common.target_call import (
    call_target_with_retry,
    classify_error_type,
    close_target,
    default_map_error,
    extract_status_code,
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


class _ClosingTarget:
    async def close(self) -> None:
        raise RuntimeError('close boom')


def _ok(text: str) -> AgentResponse:
    return AgentResponse(text=text)


def _err(msg: str) -> AgentResponse:
    return AgentResponse(
        text=msg, error=AgentResponseError(message=msg, error_type='target_error', code='x')
    )


@pytest.mark.asyncio
async def test_close_target_logs_and_swallows_cleanup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    warnings: list[tuple[object, ...]] = []
    monkeypatch.setattr('evaluatorq.common.target_call.logger.warning', lambda *args: warnings.append(args))

    await close_target(_ClosingTarget())

    assert len(warnings) == 1
    assert 'ClosingTarget' in str(warnings[0])
    assert 'close boom' in str(warnings[0])


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


class _ClientError(Exception):
    """Stub of an SDK client error carrying an HTTP status_code (e.g. openai.APIStatusError)."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f'client error {status_code}')
        self.status_code = status_code


@pytest.mark.asyncio
async def test_client_error_is_not_retried():
    t = _Target([_ClientError(400), _ok('unreachable')])
    r = await call_target_with_retry(t, [Message(role='user', content='q')], target_agent_timeout_ms=1000, max_target_retries=2)
    assert r.succeeded is False
    assert r.attempts == 1
    assert t.calls == 1


@pytest.mark.asyncio
async def test_permission_error_is_not_retried():
    t = _Target([_ClientError(403), _ok('unreachable')])
    r = await call_target_with_retry(t, [Message(role='user', content='q')], target_agent_timeout_ms=1000, max_target_retries=2)
    assert r.succeeded is False
    assert r.attempts == 1
    assert t.calls == 1


@pytest.mark.asyncio
async def test_rate_limit_429_still_retried():
    t = _Target([_ClientError(429), _ok('hi')])
    r = await call_target_with_retry(t, [Message(role='user', content='q')], target_agent_timeout_ms=1000, max_target_retries=2)
    assert r.succeeded is True
    assert r.attempts == 2
    assert t.calls == 2


@pytest.mark.asyncio
async def test_server_error_500_still_retried():
    t = _Target([_ClientError(500), _ClientError(500), _ClientError(500)])
    r = await call_target_with_retry(t, [Message(role='user', content='q')], target_agent_timeout_ms=1000, max_target_retries=2)
    assert r.succeeded is False
    assert r.attempts == 3
    assert t.calls == 3


class _HttpxStyleError(Exception):
    """Stub of httpx.HTTPStatusError: status only under .response.status_code."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f'http status error {status_code}')
        self.response = type('Resp', (), {'status_code': status_code})()


class _AiohttpStyleError(Exception):
    """Stub of an aiohttp-style error carrying .status."""

    def __init__(self, status: int) -> None:
        super().__init__(f'aiohttp error {status}')
        self.status = status


def _wrapped(status_code: int) -> RuntimeError:
    """A wrapper exception hiding the client error under __cause__."""
    wrapper = RuntimeError('target adapter failed')
    wrapper.__cause__ = _ClientError(status_code)
    return wrapper


def _implicitly_wrapped(status_code: int) -> RuntimeError:
    """A wrapper raised inside an except block WITHOUT ``from``: the original
    lands under __context__, not __cause__ (implicit exception chaining)."""
    try:
        raise _ClientError(status_code)
    except _ClientError:
        try:
            raise RuntimeError('target adapter failed')  # noqa: TRY301
        except RuntimeError as wrapper:
            return wrapper


class _BoolStatusError(Exception):
    """Pathological shape: a truthy non-status value on the status attribute."""

    status_code = True


def test_extract_status_code_shapes():
    assert extract_status_code(_ClientError(400)) == 400
    assert extract_status_code(_HttpxStyleError(403)) == 403
    assert extract_status_code(_AiohttpStyleError(429)) == 429
    assert extract_status_code(_wrapped(400)) == 400
    assert extract_status_code(_implicitly_wrapped(400)) == 400
    assert extract_status_code(RuntimeError('no status anywhere')) is None
    # bool is an int subclass but never a status
    assert extract_status_code(_BoolStatusError()) is None


@pytest.mark.asyncio
async def test_httpx_style_client_error_is_not_retried():
    # Vercel targets raise httpx.HTTPStatusError: status lives on .response.
    t = _Target([_HttpxStyleError(400), _ok('unreachable')])
    r = await call_target_with_retry(t, [Message(role='user', content='q')], target_agent_timeout_ms=1000, max_target_retries=2)
    assert r.succeeded is False
    assert r.attempts == 1
    assert t.calls == 1


@pytest.mark.asyncio
async def test_wrapped_client_error_is_not_retried():
    # Adapters may wrap the SDK error; the status hides under __cause__.
    t = _Target([_wrapped(403), _ok('unreachable')])
    r = await call_target_with_retry(t, [Message(role='user', content='q')], target_agent_timeout_ms=1000, max_target_retries=2)
    assert r.succeeded is False
    assert r.attempts == 1
    assert t.calls == 1


@pytest.mark.asyncio
async def test_aiohttp_style_429_still_retried():
    t = _Target([_AiohttpStyleError(429), _ok('hi')])
    r = await call_target_with_retry(t, [Message(role='user', content='q')], target_agent_timeout_ms=1000, max_target_retries=2)
    assert r.succeeded is True
    assert r.attempts == 2
    assert t.calls == 2

@pytest.mark.asyncio
async def test_implicitly_chained_client_error_is_not_retried():
    # A wrapper raised without `from` hides the 4xx under __context__; the
    # extractor must still find it so the call fails fast instead of burning
    # all retries.
    t = _Target([_implicitly_wrapped(400), _ok('unreachable')])
    r = await call_target_with_retry(t, [Message(role='user', content='q')], target_agent_timeout_ms=1000, max_target_retries=2)
    assert r.succeeded is False
    assert r.attempts == 1
    assert t.calls == 1
