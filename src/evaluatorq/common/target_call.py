"""Neutral target-call helper shared by red-team and simulation.

Owns bounded retry, per-call timeout, error-marker detection, exception
mapping, and per-attempt tracing for any ``AgentTarget.respond()`` call.
MUST NOT import from ``evaluatorq.redteam`` or ``evaluatorq.simulation`` —
those depend on this module, not the reverse.
"""

from __future__ import annotations

import asyncio
import inspect
import re
from contextlib import AbstractAsyncContextManager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

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


async def close_target(target: object) -> None:
    """Best-effort close a target without letting cleanup replace its result."""
    try:
        target_close = getattr(target, 'close', None)
        if not callable(target_close):
            return
        maybe = target_close()
        if inspect.isawaitable(maybe):
            await maybe
    except Exception as exc:
        logger.warning('Failed to close target {}: {}', type(target).__name__, exc)


_STATUS_CHAIN_DEPTH = 5  # how far down __cause__ to look for a wrapped HTTP error


_STATUS_TEXT_PATTERNS = (
    r'\bstatus(?:_code)?\s*[=:]\s*(\d{3})\b',
    r'\bHTTP\s*(\d{3})\b',
    r'\bcode\s*[=:]\s*(\d{3})\b',
)


def _is_status(value: object) -> bool:
    """A plausible HTTP status: an int in range, and not a ``bool``."""
    return isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599


def extract_status_code(exc: BaseException) -> int | None:
    """Best-effort HTTP status, checked down the exception chain then in ``str(exc)``.

    The single status extractor: the retry boundary and the backends' ``map_error``
    must classify one exception the same way, or a reported 4xx gets retried anyway.
    """
    current: BaseException | None = exc
    for _ in range(_STATUS_CHAIN_DEPTH):
        if current is None:
            break
        # Response first: if the two disagree, the actual HTTP response wins.
        response = getattr(current, 'response', None)
        value = getattr(response, 'status_code', None)
        if _is_status(value):
            return cast('int', value)
        for attr in ('status_code', 'status'):
            value = getattr(current, attr, None)
            if _is_status(value):
                return cast('int', value)
        current = current.__cause__ or current.__context__

    text = str(exc)
    for pattern in _STATUS_TEXT_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match and _is_status(int(match.group(1))):
            return int(match.group(1))
    return None


def _coerce_to_agent_response(raw: Any) -> AgentResponse:
    """Wrap a plain ``str`` return into ``AgentResponse`` for legacy targets."""
    if isinstance(raw, AgentResponse):
        return raw
    text_item = TextOutputItem(text=str(raw) if raw is not None else '', annotations=[])
    return AgentResponse(output=[text_item])


@dataclass(slots=True)
class TargetCallResult:
    """In-process control-flow value returned by `call_target_with_retry`.

    ``response`` is always populated: the target's real ``AgentResponse`` on
    success or a returned-error attempt, or a synthetic one on timeout/exception.
    ``error_details`` carries the raw exception info callers persist into
    ``AttackOutput.error_details``.
    """

    response: AgentResponse
    attempts: int
    error: AgentResponseError | None
    error_details: dict[str, object] | None

    @property
    def succeeded(self) -> bool:
        """Derived, not stored, so it cannot disagree with the ``error`` payload keys off."""
        return self.error is None

    def error_payload(self, *, context: str = '', turn: int = 1) -> dict[str, Any]:
        """Return this result's error as the flat fields the report layer stores.

        ``JobOutputPayload.error`` and ``AttackOutput.error`` are ``str``, so the
        ``AgentResponseError`` object must be flattened before it leaves the call
        site — handing back the object fails validation and takes down report
        generation for the whole run, after every attack has already been billed.
        Every consumer of a target call needs the same six fields, so they are
        derived here rather than re-spelled per call site.

        All keys are always present (``None`` on success) so dict consumers can
        index without guarding. ``context`` is appended to the message before the
        target's own text, e.g. ``' on turn 2/5'``; ``turn`` sets ``error_turn``.
        """
        err = self.error
        if err is None:
            return dict.fromkeys(('error', 'error_type', 'error_stage', 'error_code', 'error_turn', 'error_details'))
        return {
            'error': f'Target agent failed after {self.attempts} attempt(s){context}: {err.message}',
            # error_stage = where (target_call); error_type = what
            # (timeout/network_error/rate_limit/... classified above).
            'error_type': err.error_type,
            'error_stage': 'target_call',
            'error_code': err.code or 'target_error',
            'error_turn': turn,
            'error_details': self.error_details,
        }


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
    `TargetCallResult`.
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
                return TargetCallResult(response=resp, attempts=attempt + 1, error=None, error_details=None)
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

    return TargetCallResult(response=last_response, attempts=attempt + 1, error=last_error, error_details=last_details)
