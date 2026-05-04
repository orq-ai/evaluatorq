"""Shared retry helper for LLM API calls."""

from __future__ import annotations

import asyncio
import logging
import random
from typing import TYPE_CHECKING, TypeVar

from openai import APIStatusError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

MAX_RETRY_ATTEMPTS = 5
RETRY_MIN_WAIT_S = 2.0
RETRY_MAX_WAIT_S = 60.0

T = TypeVar("T")

# httpx connection errors that mirror TS network error retry
_RETRYABLE_NETWORK_ERRORS = (
    "ConnectError",
    "ConnectTimeout",
    "ReadTimeout",
    "WriteTimeout",
    "PoolTimeout",
)


def _is_retryable_status(status: int | None) -> bool:
    if status is None:
        return False
    return status == 429 or status >= 500


def _is_retryable_error(err: Exception) -> bool:
    """Check if an error is retryable (API status or network error)."""
    # API errors with retryable status codes
    if isinstance(err, APIStatusError):
        return _is_retryable_status(err.status_code)

    # Network/connection errors from httpx (used by openai SDK)
    err_type = type(err).__name__
    if err_type in _RETRYABLE_NETWORK_ERRORS:
        return True

    # Check wrapped cause for connection errors
    if err.__cause__ is not None:
        cause_type = type(err.__cause__).__name__
        if cause_type in _RETRYABLE_NETWORK_ERRORS:
            return True

    return False


async def with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = MAX_RETRY_ATTEMPTS,
    label: str = "API call",
) -> T:
    """Execute an async callable with exponential backoff on retryable errors.

    Retries on rate-limit (429), server errors (500+), and network errors
    (connection reset, timeout, DNS). All other errors are raised immediately.
    ``asyncio.TimeoutError`` and ``asyncio.CancelledError`` are never retried.
    """
    last_error: Exception = RuntimeError("with_retry: no attempts made")

    for attempt in range(1, max_attempts + 1):
        try:
            return await fn()
        except (asyncio.TimeoutError, asyncio.CancelledError):  # noqa: PERF203
            raise
        except Exception as err:
            last_error = err

            if not _is_retryable_error(err):
                raise

            if attempt < max_attempts:
                base_wait = RETRY_MIN_WAIT_S * (2 ** (attempt - 1))
                wait_s = min(base_wait, RETRY_MAX_WAIT_S)
                jitter = random.uniform(0, wait_s * 0.25)
                logger.warning(
                    "%s: attempt %d/%d failed (%s), retrying in %.1fs",
                    label,
                    attempt,
                    max_attempts,
                    type(err).__name__,
                    wait_s + jitter,
                )
                await asyncio.sleep(wait_s + jitter)

    raise last_error
