from __future__ import annotations

import asyncio
from typing import Any

import pytest

from evaluatorq.common.target_call import (
    call_target_with_retry,
    classify_error_type,
    default_map_error,
    extract_status_code,
)
from evaluatorq.contracts import AgentResponse, AgentResponseError, Message, Usage

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


# ---------------------------------------------------------------------------
# billed_usage accumulator (RES-1307 audit task 2)
# ---------------------------------------------------------------------------


def _usage(total: int, *, calls: int = 0) -> Usage:
    return Usage(input_tokens=total, output_tokens=0, total_tokens=total, calls=calls)


def _err_with_usage(msg: str, usage: Usage) -> AgentResponse:
    """An error marker that still carries a billed usage block — the case the
    old `usage if succeeded else None` read silently dropped."""
    return AgentResponse(
        text=msg, usage=usage, error=AgentResponseError(message=msg, error_type='target_error', code='x')
    )


@pytest.mark.asyncio
async def test_billed_usage_none_when_no_attempt_reported_usage():
    t = _Target([_ok('hi')])
    r = await call_target_with_retry(t, [Message(role='user', content='q')], target_agent_timeout_ms=1000, max_target_retries=2)
    assert r.billed_usage is None
    assert r.usage_attempts == 0


@pytest.mark.asyncio
async def test_billed_usage_matches_response_usage_on_single_success():
    """The single-attempt success path: accumulator == response.usage exactly.

    This is what makes it safe for callers to REPLACE the `response.usage` read
    rather than add to it — adding both here would double every run total.
    """
    t = _Target([AgentResponse(text='hi', usage=_usage(30, calls=1))])
    r = await call_target_with_retry(t, [Message(role='user', content='q')], target_agent_timeout_ms=1000, max_target_retries=2)
    assert r.billed_usage is not None
    assert r.billed_usage.total_tokens == 30
    assert r.billed_usage.total_tokens == r.response.usage.total_tokens  # pyright: ignore[reportOptionalMemberAccess]
    assert r.usage_attempts == 1


@pytest.mark.asyncio
async def test_billed_usage_sums_failed_attempt_then_success():
    """Tokens burned by a usage-bearing error attempt survive a later success."""
    t = _Target([_err_with_usage('boom', _usage(7, calls=1)), AgentResponse(text='hi', usage=_usage(11, calls=1))])
    r = await call_target_with_retry(t, [Message(role='user', content='q')], target_agent_timeout_ms=1000, max_target_retries=2)
    assert r.succeeded is True
    assert r.attempts == 2
    assert r.billed_usage is not None
    assert r.billed_usage.total_tokens == 18
    assert r.billed_usage.calls == 2
    assert r.usage_attempts == 2
    # `response` still means the surviving response, not the aggregate.
    assert r.response.usage is not None
    assert r.response.usage.total_tokens == 11


@pytest.mark.asyncio
async def test_billed_usage_recorded_when_every_attempt_fails_with_usage():
    t = _Target([_err_with_usage('boom', _usage(5)) for _ in range(3)])
    r = await call_target_with_retry(t, [Message(role='user', content='q')], target_agent_timeout_ms=1000, max_target_retries=2)
    assert r.succeeded is False
    assert r.billed_usage is not None
    assert r.billed_usage.total_tokens == 15
    assert r.usage_attempts == 3


@pytest.mark.asyncio
async def test_synthetic_timeout_and_exception_contribute_no_usage():
    """`_synthetic()` carries `usage=None`, so those branches stay unknown-not-zero:
    a usage-bearing attempt before them is still counted, the synthetic ones add
    nothing rather than a fabricated 0-cost call."""

    class _OnceThenRaise:
        def __init__(self) -> None:
            self.calls = 0

        async def respond(self, messages: list[Message]) -> AgentResponse:
            self.calls += 1
            if self.calls == 1:
                return _err_with_usage('boom', _usage(9, calls=1))
            raise ConnectionError('connection reset')

    t = _OnceThenRaise()
    r = await call_target_with_retry(t, [Message(role='user', content='q')], target_agent_timeout_ms=1000, max_target_retries=2)
    assert r.succeeded is False
    assert t.calls == 3
    assert r.billed_usage is not None
    assert r.billed_usage.total_tokens == 9
    assert r.usage_attempts == 1


# ---------------------------------------------------------------------------
# per-attempt call-counter normalisation (RES-1307 audit fix wave)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_untracked_call_count_normalised_to_one_per_attempt():
    """A target reporting `calls=0` billed one exchange per attempt, not zero.

    Consumers read `billed_usage` verbatim; leaving `calls=0` there gave a run
    total with a cost and no call count, which `cost_is_partial` cannot qualify
    and `cost_source` reads as unknown.
    """
    t = _Target([_err_with_usage('boom', _usage(4)), AgentResponse(text='hi', usage=_usage(7))])
    r = await call_target_with_retry(t, [Message(role='user', content='q')], target_agent_timeout_ms=1000, max_target_retries=2)
    assert r.billed_usage is not None
    assert r.billed_usage.total_tokens == 11
    assert r.billed_usage.calls == 2
    assert r.billed_usage.priced_calls == 0
    assert r.usage_attempts == 2


@pytest.mark.asyncio
async def test_mixed_cost_reporting_across_retry_stays_partial():
    """The reviewer's reproduction: two billed attempts, one reported a cost.

    `with_calls(2)` would have widened `priced_calls` to 2 alongside `calls`,
    rendering a half-known cost as fully provider-billed. Per-attempt
    normalisation keeps `priced_calls` at 1, so the figure stays qualified.
    """
    priced = Usage(input_tokens=7, output_tokens=0, total_tokens=7, total_cost=0.01, calls=0)
    t = _Target([_err_with_usage('boom', _usage(4)), AgentResponse(text='hi', usage=priced)])
    r = await call_target_with_retry(t, [Message(role='user', content='q')], target_agent_timeout_ms=1000, max_target_retries=2)
    assert r.billed_usage is not None
    assert r.billed_usage.calls == 2
    assert r.billed_usage.priced_calls == 1
    assert r.billed_usage.total_cost == 0.01
    assert r.billed_usage.cost_is_partial is True
    assert r.billed_usage.cost_source == 'provider'


@pytest.mark.asyncio
async def test_target_reported_counters_are_left_alone():
    """A target that tracks its own counters must come through untouched.

    orq sums tool-continuation rounds into one usage block, so `calls=3` for a
    single attempt is correct and must not be flattened to 1. A catalogue-priced
    block keeps its `estimated_calls` too.
    """
    honest = Usage(
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        total_cost=0.02,
        calls=3,
        priced_calls=2,
        estimated_calls=1,
    )
    t = _Target([AgentResponse(text='hi', usage=honest)])
    r = await call_target_with_retry(t, [Message(role='user', content='q')], target_agent_timeout_ms=1000, max_target_retries=2)
    assert r.billed_usage is not None
    assert r.billed_usage.calls == 3
    assert r.billed_usage.priced_calls == 2
    assert r.billed_usage.estimated_calls == 1
    assert r.billed_usage.cost_source == 'mixed'
    assert r.usage_attempts == 1


@pytest.mark.asyncio
async def test_untracked_estimated_calls_clamped_to_priced_calls():
    """`estimated_calls <= priced_calls <= calls` survives normalisation.

    A `calls=0` block that nonetheless claims an estimated call but carries no
    cost has nothing to have estimated — the counter is dropped, not preserved
    into an aggregate where it would out-number `priced_calls`.
    """
    bogus = Usage(input_tokens=2, output_tokens=0, total_tokens=2, calls=0, priced_calls=0, estimated_calls=1)
    t = _Target([AgentResponse(text='hi', usage=bogus)])
    r = await call_target_with_retry(t, [Message(role='user', content='q')], target_agent_timeout_ms=1000, max_target_retries=2)
    assert r.billed_usage is not None
    assert r.billed_usage.calls == 1
    assert r.billed_usage.priced_calls == 0
    assert r.billed_usage.estimated_calls == 0
    assert r.billed_usage.cost_source is None


@pytest.mark.asyncio
async def test_hand_built_usage_with_cost_but_no_priced_calls_warns(caplog: pytest.LogCaptureFixture):
    """A custom AgentTarget can build a Usage whose cost has no priced call.

    `_attempt_usage` leaves a self-tracking target's counters alone, so the cost
    reaches the run total with `priced_calls=0` and renders unqualified — i.e. as
    fully provider-billed. Widening `priced_calls` here would over-claim for a
    genuinely partial aggregate, so the degraded path announces itself instead.
    """
    hand_built = Usage(input_tokens=10, output_tokens=0, total_tokens=10, calls=2, total_cost=0.02)
    t = _Target([AgentResponse(text='hi', usage=hand_built)])
    with caplog.at_level('WARNING'):
        r = await call_target_with_retry(
            t, [Message(role='user', content='q')], target_agent_timeout_ms=1000, max_target_retries=2
        )
    assert r.billed_usage is not None
    assert (r.billed_usage.calls, r.billed_usage.priced_calls) == (2, 0)
    assert 'priced_calls=0' in caplog.text


@pytest.mark.asyncio
async def test_self_tracking_usage_with_priced_calls_does_not_warn(caplog: pytest.LogCaptureFixture):
    """The honest shape must stay silent — two branches must not differ only in logging."""
    honest = Usage(input_tokens=10, output_tokens=0, total_tokens=10, calls=2, priced_calls=2, total_cost=0.02)
    t = _Target([AgentResponse(text='hi', usage=honest)])
    with caplog.at_level('WARNING'):
        r = await call_target_with_retry(
            t, [Message(role='user', content='q')], target_agent_timeout_ms=1000, max_target_retries=2
        )
    assert r.billed_usage is not None
    assert r.billed_usage.priced_calls == 2
    assert 'priced_calls=0' not in caplog.text
