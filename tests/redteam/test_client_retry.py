"""LLMConfig.client_retry: client-side backoff for pipeline-internal LLM calls.

Router-side retry (``retry_extra_body``) is a no-op on a plain OpenAI client
and never covers network errors; this helper closes that gap. The tests pin
that it follows the config's retry budget and re-raises non-retryable errors
immediately.
"""

from __future__ import annotations

import httpx
import pytest
from openai import APIStatusError

from evaluatorq.common import retry as retry_module
from evaluatorq.redteam.contracts import LLMConfig

pytestmark = pytest.mark.asyncio


def _status_error(status: int) -> APIStatusError:
    request = httpx.Request('POST', 'https://router.example/v3/router')
    response = httpx.Response(status, request=request)
    return APIStatusError(f'status {status}', response=response, body=None)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch):
    """Skip real backoff sleeps so retry tests run instantly."""

    async def _instant(_seconds: float) -> None:
        return None

    monkeypatch.setattr(retry_module.asyncio, 'sleep', _instant)


async def test_retries_retryable_status_then_succeeds() -> None:
    cfg = LLMConfig(retry_count=2)
    calls = 0

    async def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise _status_error(429)
        return 'ok'

    assert await cfg.client_retry(flaky, label='test') == 'ok'
    assert calls == 3


async def test_gives_up_after_retry_count_attempts() -> None:
    cfg = LLMConfig(retry_count=1)
    calls = 0

    async def always_503() -> str:
        nonlocal calls
        calls += 1
        raise _status_error(503)

    with pytest.raises(APIStatusError):
        await cfg.client_retry(always_503, label='test')
    # retry_count=1 -> one initial attempt plus one retry.
    assert calls == 2


async def test_non_retryable_status_raises_immediately() -> None:
    cfg = LLMConfig(retry_count=3)
    calls = 0

    async def bad_request() -> str:
        nonlocal calls
        calls += 1
        raise _status_error(400)

    with pytest.raises(APIStatusError):
        await cfg.client_retry(bad_request, label='test')
    assert calls == 1


async def test_custom_retry_on_codes_are_honored() -> None:
    cfg = LLMConfig(retry_count=1, retry_on_codes=[418])
    calls = 0

    async def teapot_once() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _status_error(418)
        return 'ok'

    assert await cfg.client_retry(teapot_once, label='test') == 'ok'
    assert calls == 2
