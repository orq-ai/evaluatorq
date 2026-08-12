"""Tests for evaluatorq.common.retry._is_retryable_status.

Verifies that retry_statuses AUGMENTS the default set (429 + 5xx) rather
than replacing it — passing retry_statuses={429} must not disable 503 retries.
"""

from __future__ import annotations

from evaluatorq.common.retry import _is_retryable_status


def test_default_retries_429():
    assert _is_retryable_status(429) is True


def test_default_retries_500():
    assert _is_retryable_status(500) is True


def test_default_retries_503():
    assert _is_retryable_status(503) is True


def test_default_does_not_retry_404():
    assert _is_retryable_status(404) is False


def test_custom_status_added():
    """A caller-supplied status code is retried in addition to defaults."""
    assert _is_retryable_status(418, retry_statuses={418}) is True


def test_default_5xx_still_retried_when_custom_set_given():
    """retry_statuses augments the default; passing {429} must not drop 503."""
    assert _is_retryable_status(503, retry_statuses={429}) is True


def test_default_429_still_retried_when_custom_set_given():
    assert _is_retryable_status(429, retry_statuses={418}) is True


def test_non_retryable_not_added_by_custom_set():
    """404 stays non-retryable even when a custom set is supplied."""
    assert _is_retryable_status(404, retry_statuses={429}) is False


def test_none_status_never_retried():
    assert _is_retryable_status(None) is False
    assert _is_retryable_status(None, retry_statuses={429}) is False


# ---------------------------------------------------------------------------
# with_retry — network errors, timeout passthrough, backoff timing
# ---------------------------------------------------------------------------

import asyncio

import httpx
import pytest
from openai import APIConnectionError

from evaluatorq.common import retry as retry_module
from evaluatorq.common.retry import RETRY_MAX_WAIT_S, RETRY_MIN_WAIT_S, with_retry


def _connection_error_with_cause() -> APIConnectionError:
    """An APIConnectionError wrapping an httpx.ConnectError, as the SDK raises it."""
    request = httpx.Request("POST", "https://router.example/v3/router")
    err = APIConnectionError(request=request)
    err.__cause__ = httpx.ConnectError("connection reset", request=request)
    return err


@pytest.mark.asyncio
async def test_network_error_with_cause_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """The stated motivation for client-side retry: an httpx connection error
    wrapped by the SDK (matched via __cause__ class name) must be retried."""

    async def _instant(_seconds: float) -> None:
        return None

    monkeypatch.setattr(retry_module.asyncio, "sleep", _instant)
    calls = 0

    async def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _connection_error_with_cause()
        return "ok"

    assert await with_retry(flaky, max_attempts=3, label="test") == "ok"
    assert calls == 2


def _connection_error_wrapping(cause: Exception) -> APIConnectionError:
    """An APIConnectionError wrapping an arbitrary httpx transport error, as the
    SDK raises it. The __cause__ class name is deliberately NOT in the httpx
    allowlist, so retry must come from the SDK class, not the name match."""
    request = httpx.Request("POST", "https://router.example/v3/router")
    err = APIConnectionError(request=request)
    err.__cause__ = cause
    return err


@pytest.mark.asyncio
async def test_remote_protocol_error_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """A server that disconnects mid-response arrives as APIConnectionError with
    an httpx.RemoteProtocolError cause. That class is NOT in the name allowlist,
    so this only retries because APIConnectionError itself is treated as a
    transport failure — the regression currentlycodinng flagged on RES-832."""

    async def _instant(_seconds: float) -> None:
        return None

    monkeypatch.setattr(retry_module.asyncio, "sleep", _instant)
    request = httpx.Request("POST", "https://router.example/v3/router")
    calls = 0

    async def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _connection_error_wrapping(httpx.RemoteProtocolError("server disconnected", request=request))
        return "ok"

    assert await with_retry(flaky, max_attempts=3, label="test") == "ok"
    assert calls == 2


@pytest.mark.asyncio
async def test_api_timeout_error_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """APITimeoutError (a subclass of APIConnectionError) is a transport failure
    and must retry."""
    from openai import APITimeoutError

    async def _instant(_seconds: float) -> None:
        return None

    monkeypatch.setattr(retry_module.asyncio, "sleep", _instant)
    request = httpx.Request("POST", "https://router.example/v3/router")
    calls = 0

    async def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise APITimeoutError(request=request)
        return "ok"

    assert await with_retry(flaky, max_attempts=3, label="test") == "ok"
    assert calls == 2


@pytest.mark.asyncio
async def test_asyncio_timeout_is_never_retried() -> None:
    """Per-attempt timeouts stay per-attempt: a hung call fails immediately."""
    calls = 0

    async def hung() -> str:
        nonlocal calls
        calls += 1
        raise asyncio.TimeoutError

    with pytest.raises(asyncio.TimeoutError):
        await with_retry(hung, max_attempts=5, label="test")
    assert calls == 1


def _status_error(status: int):
    from openai import APIStatusError

    request = httpx.Request("POST", "https://router.example/v3/router")
    response = httpx.Response(status, request=request)
    return APIStatusError(f"status {status}", response=response, body=None)


@pytest.mark.asyncio
async def test_backoff_curve_doubles_and_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sleeps follow 2/4/8/16/32/60 with the 60s cap applied; jitter disabled."""
    sleeps: list[float] = []

    async def _record(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(retry_module.asyncio, "sleep", _record)
    monkeypatch.setattr(retry_module.random, "uniform", lambda _a, _b: 0.0)

    async def always_503() -> str:
        raise _status_error(503)

    with pytest.raises(Exception):
        await with_retry(always_503, max_attempts=7, label="test")

    assert sleeps == [2.0, 4.0, 8.0, 16.0, 32.0, RETRY_MAX_WAIT_S]
    assert sleeps[0] == RETRY_MIN_WAIT_S


@pytest.mark.asyncio
async def test_jitter_bounded_at_quarter_of_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """Jitter is drawn from [0, wait*0.25] — pin the bound passed to random.uniform."""
    uniform_calls: list[tuple[float, float]] = []

    async def _instant(_seconds: float) -> None:
        return None

    def _uniform(a: float, b: float) -> float:
        uniform_calls.append((a, b))
        return 0.0

    monkeypatch.setattr(retry_module.asyncio, "sleep", _instant)
    monkeypatch.setattr(retry_module.random, "uniform", _uniform)

    async def always_503() -> str:
        raise _status_error(503)

    with pytest.raises(Exception):
        await with_retry(always_503, max_attempts=3, label="test")

    assert uniform_calls == [(0, 2.0 * 0.25), (0, 4.0 * 0.25)]
