"""A target exception marked NonRetryableTargetError ends the retry loop on the first attempt."""

from __future__ import annotations

import pytest

from evaluatorq.common.target_call import NonRetryableTargetError, call_target_with_retry
from evaluatorq.contracts import AgentResponse, Message


class _Fatal(NonRetryableTargetError):
    pass


class _CountingTarget:
    def __init__(self, exc: Exception) -> None:
        self.calls = 0
        self._exc = exc

    async def respond(self, messages: list[Message]) -> AgentResponse:
        self.calls += 1
        raise self._exc


@pytest.mark.asyncio
async def test_non_retryable_marker_stops_after_one_attempt() -> None:
    target = _CountingTarget(_Fatal('binary missing'))
    result = await call_target_with_retry(
        target,
        [Message(role='user', content='hi')],
        target_agent_timeout_ms=1000,
        max_target_retries=3,
        map_error=lambda exc: ('cli.not_found', str(exc)),
    )
    assert target.calls == 1
    assert result.attempts == 1
    assert result.error is not None
    assert result.error.code == 'cli.not_found'


@pytest.mark.asyncio
async def test_plain_exception_still_retries() -> None:
    target = _CountingTarget(RuntimeError('flaky'))
    result = await call_target_with_retry(
        target,
        [Message(role='user', content='hi')],
        target_agent_timeout_ms=1000,
        max_target_retries=2,
    )
    assert target.calls == 3
    assert result.attempts == 3
