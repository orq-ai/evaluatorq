"""Neutral target-call helper shared by red-team and simulation.

Owns bounded retry, per-call timeout, error-marker detection, exception
mapping, and per-attempt tracing for any ``AgentTarget.respond()`` call.
MUST NOT import from ``evaluatorq.redteam`` or ``evaluatorq.simulation`` —
those depend on this module, not the reverse.
"""

from __future__ import annotations

import asyncio
import re
from contextlib import AbstractAsyncContextManager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq.contracts import AgentResponse, AgentResponseError, Message, TextOutputItem

if TYPE_CHECKING:
    from collections.abc import Callable

_ERROR_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r'content[_ ]filter|content management policy'), 'content_filter'),
    (re.compile(r'rate limit|(?<!\d)429(?!\d)'), 'rate_limit'),
    (re.compile(r'timed out|timeout'), 'timeout'),
    (re.compile(r'status[ _]?5\d\d'), 'server_error'),
    (re.compile(r'status[ _]?4\d\d'), 'client_error'),
    (re.compile(r'connection'), 'network_error'),
]


def classify_error_type(error: str | None, *, existing_type: str | None = None) -> str | None:
    """Infer a coarse ``error_type`` from a raw error string when not already set.

    Returns ``existing_type`` unchanged when provided, ``None`` for an empty
    error, the first matching pattern's type, or ``'unknown'`` when nothing
    matches. Shared by the orchestrator (per-response ``AgentResponseError``)
    and report converters (run-level rollup) so both classify identically.
    """
    if existing_type:
        return existing_type
    if not error:
        return None
    lower = error.lower()
    for pattern, etype in _ERROR_PATTERNS:
        if pattern.search(lower):
            return etype
    return 'unknown'


def default_map_error(exc: Exception) -> tuple[str, str]:
    """Fallback (code, message) mapping; identical to ``Backend.map_error`` base."""
    return 'target_error', f'{type(exc).__name__}: {exc}'


_STATUS_CHAIN_DEPTH = 5  # how far down __cause__ to look for a wrapped HTTP error


def extract_status_code(exc: BaseException) -> int | None:
    """Best-effort HTTP status from the heterogeneous exception shapes targets raise.

    Checks, on the exception and then down its chained originals — explicit
    ``__cause__`` (``raise ... from``) first, falling back to the implicit
    ``__context__`` (a bare ``raise`` inside an ``except`` block):

    * ``exc.status_code`` — openai ``APIStatusError`` and friends;
    * ``exc.status`` — aiohttp-style errors;
    * ``exc.response.status_code`` — ``httpx.HTTPStatusError`` (Vercel targets).

    Returns ``None`` when no integer status is found. ``bool`` is excluded
    (it is an ``int`` subclass but never a status).
    """
    current: BaseException | None = exc
    for _ in range(_STATUS_CHAIN_DEPTH):
        if current is None:
            return None
        for attr in ('status_code', 'status'):
            value = getattr(current, attr, None)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        response = getattr(current, 'response', None)
        value = getattr(response, 'status_code', None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        current = current.__cause__ or current.__context__
    return None


def _coerce_to_agent_response(raw: Any) -> AgentResponse:
    """Wrap a plain ``str`` return into ``AgentResponse`` for legacy targets."""
    if isinstance(raw, AgentResponse):
        return raw
    text_item = TextOutputItem(text=str(raw) if raw is not None else '', annotations=[])
    return AgentResponse(output=[text_item])


@dataclass(slots=True)
class TargetCallResult:
    """In-process control-flow value returned by :func:`call_target_with_retry`.

    ``response`` is always populated: the target's real ``AgentResponse`` on
    success or a returned-error attempt, or a synthetic one on timeout/exception.
    ``error`` is ``None`` iff ``succeeded``. ``error_details`` carries the raw
    exception info callers persist into ``AttackOutput.error_details``.
    """

    response: AgentResponse
    succeeded: bool
    attempts: int
    error: AgentResponseError | None
    error_details: dict[str, object] | None


def _synthetic(text: str, *, error_type: str, code: str) -> AgentResponse:
    return AgentResponse(
        output=[TextOutputItem(text=text, annotations=[])],
        error=AgentResponseError(message=text, error_type=error_type, code=code),
    )


async def call_target_with_retry(
    target: Any,
    messages: list[Message],
    *,
    target_agent_timeout_ms: float,
    max_target_retries: int,
    map_error: Callable[[Exception], tuple[str, str] | None] = default_map_error,
    on_attempt: Callable[[int], AbstractAsyncContextManager[Any]] | None = None,
    on_attempt_response: Callable[[Any, AgentResponse], None] | None = None,
) -> TargetCallResult:
    """Call ``target.respond(messages)`` with bounded retry + per-call timeout.

    Retries the SAME exchange on a returned ``.error`` marker, a per-call
    timeout, or a generic ``Exception``. ``asyncio.CancelledError`` is NOT
    caught (it is a ``BaseException``) so an outer-ceiling cancellation
    propagates cleanly. ``on_attempt(i)`` wraps each attempt in a
    caller-supplied span (0-based index). ``on_attempt_response`` receives the
    caller-supplied context value and each returned response while that context
    is still open, so callers can annotate per-attempt spans. Returns a uniform
    :class:`TargetCallResult`.
    """
    timeout_s = target_agent_timeout_ms / 1000.0
    max_attempts = max(1, max_target_retries + 1)
    last_response: AgentResponse = AgentResponse()
    last_error: AgentResponseError | None = None
    last_details: dict[str, object] | None = None

    attempt = 0  # max_attempts >= 1, but the type checker can't prove the loop runs
    for attempt in range(max_attempts):
        ctx = on_attempt(attempt) if on_attempt is not None else nullcontext()
        try:
            async with ctx as attempt_context:
                raw = await asyncio.wait_for(target.respond(messages), timeout=timeout_s)
                resp = _coerce_to_agent_response(raw)
                if on_attempt_response is not None:
                    on_attempt_response(attempt_context, resp)
            if resp.error is None:
                return TargetCallResult(
                    response=resp, succeeded=True, attempts=attempt + 1, error=None, error_details=None
                )
            last_response = resp
            last_error = resp.error
            last_details = {
                'response_error_type': resp.error.error_type,
                'raw_message': resp.error.message,
                'attempts': attempt + 1,
            }
        except asyncio.TimeoutError:
            text = f'[ERROR: Target agent timed out after {timeout_s:.0f}s]'
            last_response = _synthetic(text, error_type='timeout', code='target.timeout')
            last_error = last_response.error
            last_details = {'timeout_ms': target_agent_timeout_ms, 'attempts': attempt + 1}
        except Exception as exc:
            mapped = map_error(exc)
            code, msg = mapped if mapped is not None else default_map_error(exc)
            classified = classify_error_type(msg)
            text = f'[ERROR: {msg}]'
            last_response = _synthetic(
                text,
                error_type=classified if classified and classified != 'unknown' else 'target_error',
                code=code,
            )
            last_error = last_response.error
            last_details = {'exception_type': type(exc).__name__, 'raw_message': str(exc), 'attempts': attempt + 1}
            # 4xx client errors are non-retryable by default — bad request,
            # auth, permission scope, conflict etc. are deterministic, so a
            # retry replays the same rejection. The only exceptions are 408
            # (timeout) and 429 (rate limit), which stay retryable.
            status = extract_status_code(exc)
            if status is not None and 400 <= status < 500 and status not in (408, 429):
                logger.warning(f'Target call failed with non-retryable client error ({status}); not retrying')
                break

        if attempt + 1 < max_attempts:
            logger.warning(f'Target call failed (attempt {attempt + 1}/{max_attempts}); retrying same exchange')

    return TargetCallResult(
        response=last_response, succeeded=False, attempts=attempt + 1, error=last_error, error_details=last_details
    )
