"""Provider content-filter detection on the OpenAI-compatible wire protocol.

Domain-neutral: any caller making chat-completion calls through the Orq router (or
OpenAI directly) can use these to tell a *structural* content-filter block apart from
a model's natural-language self-refusal. Two shapes exist, both normalized to the
``content_filter`` code by the router:

1. 200 response with ``finish_reason='content_filter'`` (canonical OpenAI signal).
2. HTTP error with structured ``code='content_filter'`` — verified empirically: Azure
   via the Orq router raises ``openai.BadRequestError`` (HTTP 400) this way.

A model that self-censors in prose is deliberately NOT detected: at the protocol level
it is indistinguishable from a real response (``finish_reason='stop'``, no ``refusal``
field — verified for Anthropic and Gemini via the Orq router), so it is left for the
caller to forward downstream rather than treated as a block.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

from loguru import logger
from openai import APIConnectionError, APIStatusError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

_ResponseT = TypeVar('_ResponseT')


def classify_finish_reason(finish_reason: str | None) -> Literal['content_filter'] | None:
    """Return ``'content_filter'`` when the provider blocked generation, else ``None``."""
    return 'content_filter' if finish_reason == 'content_filter' else None


def is_content_filter_error(exc: BaseException) -> bool:
    """Return True if ``exc`` is a provider content-filter block surfaced as an error.

    Keys on the structured error ``code`` (exposed by the OpenAI SDK as ``exc.code`` and
    mirrored in the JSON body), NOT on message prose, so it stays provider-agnostic for
    any block the router normalizes to that code.
    """
    if getattr(exc, 'code', None) == 'content_filter':
        return True
    body = getattr(exc, 'body', None)
    return isinstance(body, dict) and body.get('code') == 'content_filter'


async def regenerate_on_content_filter(
    call: Callable[[int], Awaitable[_ResponseT]],
    *,
    max_attempts: int,
    label: str = 'attacker generation',
) -> _ResponseT:
    """Invoke ``call`` repeatedly, regenerating while the provider content-filters it.

    ``call(attempt)`` makes one chat-completion request (and owns its own span / token
    accounting — the attempt index is passed so it can tag retries). A block is detected
    in either shape: a 200 response whose ``finish_reason='content_filter'``, or a raised
    error carrying ``code='content_filter'``. Each is retried up to ``max_attempts`` total.

    Returns the first usable response. When every attempt is content-filtered, returns the
    final filtered *response* (200-form) so the caller can inspect it and stop cleanly — but
    a content-filter *error* on the final attempt (and any non-filter error) propagates, so
    the caller's own error handling classifies it. ``max_attempts`` is clamped to >= 1.
    """
    attempts = max(1, max_attempts)
    for attempt in range(attempts):
        try:
            response = await call(attempt)
        except (APIConnectionError, APIStatusError) as e:
            if is_content_filter_error(e) and attempt < attempts - 1:
                logger.warning(f'{label} content-filtered (error form); regenerating ({attempt + 1}/{attempts - 1})')
                continue
            raise
        choices = getattr(response, 'choices', None) or []
        finish_reason = getattr(choices[0], 'finish_reason', None) if choices else None
        if classify_finish_reason(finish_reason) is None or attempt >= attempts - 1:
            return response
        logger.warning(f'{label} content-filtered (finish_reason); regenerating ({attempt + 1}/{attempts - 1})')
    raise RuntimeError('regenerate_on_content_filter: loop exhausted without returning')  # unreachable


__all__ = ['classify_finish_reason', 'is_content_filter_error', 'regenerate_on_content_filter']
