"""Shared retry helper for LLM API calls.

Canonical retry for every surface — do not add a second layer. An OpenAI/Orq
client already carries its own ``max_retries``; wrapping such a call in
``with_retry`` multiplies the two budgets, so the default ``MAX_RETRY_ATTEMPTS``
of 5 over a client with the SDK default of 2 retries is 15 requests, not 5. The
SDK's wall-clock guard is measured from before the first attempt, so the outer
layer can also exhaust it. Per call path, either configure the client's
``max_retries`` or use ``with_retry`` — not both — and name the choice in the
caller's docstring.
"""

from __future__ import annotations

import asyncio
import random
from typing import TYPE_CHECKING, TypeVar

from loguru import logger
from openai import APIConnectionError, APIStatusError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable

    from openai import AsyncOpenAI

MAX_RETRY_ATTEMPTS = 5
RETRY_MIN_WAIT_S = 2.0
RETRY_MAX_WAIT_S = 60.0
# Fraction of the computed wait added as random jitter, to stop a fleet of
# concurrent datapoints retrying in lockstep against the same rate limit.
RETRY_JITTER_FRACTION = 0.25

T = TypeVar('T')

# httpx connection-error class names, kept only as a defensive fallback for a
# raw httpx error raised outside the OpenAI SDK. SDK calls surface every
# transport failure as APIConnectionError (see _is_retryable_error), so this
# list is not the primary matcher and must not be relied on to enumerate httpx.
_RETRYABLE_NETWORK_ERRORS = (
    'ConnectError',
    'ConnectTimeout',
    'ReadTimeout',
    'WriteTimeout',
    'PoolTimeout',
)


def without_client_retries(client: AsyncOpenAI) -> AsyncOpenAI:
    """Return an OpenAI client clone with SDK retries disabled.

    This belongs beside ``with_retry`` because it enforces the boundary between
    the two retry owners: a caller that wraps an SDK operation in ``with_retry``
    must disarm the SDK budget first. ``with_options`` creates a new client and
    reuses the caller's transport, authentication, base URL, headers, and
    timeout, so an injected client is never mutated and its lifecycle remains
    caller-owned. Clients that already have no integer retry budget are returned
    unchanged, which keeps lightweight test doubles usable. Disarming ignores the
    caller's own attempt count: ``retry_count=0`` means one attempt, not "let the
    SDK retry instead".
    """
    max_retries = getattr(client, 'max_retries', 0)
    if not isinstance(max_retries, int) or max_retries <= 0:
        return client
    return client.with_options(max_retries=0)


def _is_retryable_status(
    status: int | None,
    retry_statuses: set[int] | None = None,
) -> bool:
    if status is None:
        return False
    if status in (retry_statuses or set()):
        return True
    return status == 429 or status >= 500


def _is_retryable_error(
    err: Exception,
    retry_statuses: set[int] | None = None,
) -> bool:
    """Check if an error is retryable (API status or transport failure)."""
    # API errors with retryable status codes.
    if isinstance(err, APIStatusError):
        return _is_retryable_status(err.status_code, retry_statuses)

    # Transport failures. The OpenAI SDK raises APIConnectionError (and its
    # APITimeoutError subclass) ONLY for transport-level failures — connection
    # reset, read/write error, proxy error, DNS, timeout, and a server that
    # disconnects mid-response (httpx.RemoteProtocolError) — wrapping the
    # underlying httpx error as __cause__. Retrying on the SDK class covers
    # every one of them at once, where an httpx class-name allowlist silently
    # drops the next error type the SDK wraps (e.g. RemoteProtocolError, the
    # ordinary way a long router call dies mid-flight).
    if isinstance(err, APIConnectionError):
        return True

    # Defensive fallback for a raw httpx error raised outside the SDK: match the
    # connection-error class names directly, or through a single __cause__ hop.
    if type(err).__name__ in _RETRYABLE_NETWORK_ERRORS:
        return True
    cause = err.__cause__
    return cause is not None and type(cause).__name__ in _RETRYABLE_NETWORK_ERRORS


async def with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = MAX_RETRY_ATTEMPTS,
    min_wait_s: float = RETRY_MIN_WAIT_S,
    max_wait_s: float = RETRY_MAX_WAIT_S,
    jitter_fraction: float = RETRY_JITTER_FRACTION,
    retry_statuses: Iterable[int] | None = None,
    label: str = 'API call',
) -> T:
    """Execute an async callable with exponential backoff on retryable errors.

    Retries on rate-limit (429), server errors (500+), and network errors
    (connection reset, timeout, DNS). All other errors are raised immediately.
    ``asyncio.TimeoutError`` and ``asyncio.CancelledError`` are never retried.

    The backoff curve is ``min_wait_s * 2 ** (attempt - 1)``, capped at
    ``max_wait_s``, plus up to ``jitter_fraction`` of that wait chosen at random.
    All three are parameters because a provider with a long rate-limit window
    needs a different curve than a flaky local endpoint, and the module defaults
    used to be the only option.

    This is one of the package's two retry layers and they compose
    multiplicatively — a call path uses this **or** the SDK's own ``max_retries``,
    never both.
    """
    if max_attempts < 1:
        raise ValueError(f'with_retry: max_attempts must be >= 1, got {max_attempts}')
    if min_wait_s < 0 or max_wait_s < 0:
        raise ValueError(f'with_retry: waits must be >= 0, got min={min_wait_s}, max={max_wait_s}')
    if jitter_fraction < 0:
        raise ValueError(f'with_retry: jitter_fraction must be >= 0, got {jitter_fraction}')
    last_error: Exception = RuntimeError('with_retry: no attempts made')
    retry_status_set = set(retry_statuses) if retry_statuses is not None else None

    for attempt in range(1, max_attempts + 1):
        try:
            return await fn()
        except (asyncio.TimeoutError, asyncio.CancelledError):  # noqa: PERF203
            raise
        except Exception as err:
            last_error = err

            if not _is_retryable_error(err, retry_status_set):
                raise

            if attempt < max_attempts:
                base_wait = min_wait_s * (2 ** (attempt - 1))
                wait_s = min(base_wait, max_wait_s)
                jitter = random.uniform(0, wait_s * jitter_fraction)
                logger.warning(
                    '{}: attempt {}/{} failed ({}), retrying in {:.1f}s',
                    label,
                    attempt,
                    max_attempts,
                    type(err).__name__,
                    wait_s + jitter,
                )
                await asyncio.sleep(wait_s + jitter)

    raise last_error
