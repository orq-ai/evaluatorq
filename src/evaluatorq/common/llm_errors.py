"""SDK-neutral LLM exception classes.

Each tuple names the matching exception class from the ``openai`` SDK and, when
it is installed, from the ``anthropic`` SDK, so an ``except`` clause or an
``isinstance`` check recognises an error from either one. When ``anthropic`` is
not installed the tuples hold only the OpenAI classes. An installed
``anthropic`` that cannot be imported raises at import time instead of being
skipped.

``openai.LengthFinishReasonError`` is deliberately not here. It is a
client-side parse error, not an API error, and stays imported from ``openai``.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import openai

API_ERRORS: tuple[type[BaseException], ...] = (openai.APIError,)
STATUS_ERRORS: tuple[type[BaseException], ...] = (openai.APIStatusError,)
CONNECTION_ERRORS: tuple[type[BaseException], ...] = (openai.APIConnectionError,)
TIMEOUT_ERRORS: tuple[type[BaseException], ...] = (openai.APITimeoutError,)
BAD_REQUEST_ERRORS: tuple[type[BaseException], ...] = (openai.BadRequestError,)
SDK_ERRORS: tuple[type[BaseException], ...] = (openai.OpenAIError,)

# Gate on installation, not importability: an installed but broken anthropic fails loudly here.
if importlib.util.find_spec('anthropic') is not None:
    import anthropic

    API_ERRORS = (openai.APIError, anthropic.APIError)
    STATUS_ERRORS = (openai.APIStatusError, anthropic.APIStatusError)
    CONNECTION_ERRORS = (openai.APIConnectionError, anthropic.APIConnectionError)
    TIMEOUT_ERRORS = (openai.APITimeoutError, anthropic.APITimeoutError)
    BAD_REQUEST_ERRORS = (openai.BadRequestError, anthropic.BadRequestError)
    SDK_ERRORS = (openai.OpenAIError, anthropic.AnthropicError)


def status_code(exc: BaseException) -> int | None:
    """Return the HTTP status of a status error from either SDK, else ``None``."""
    if isinstance(exc, STATUS_ERRORS):
        code = getattr(exc, 'status_code', None)
        return code if isinstance(code, int) else None
    return None


def error_body(exc: BaseException) -> dict[str, Any] | None:
    """Return the inner error object of an SDK error's JSON body, else ``None``.

    The two SDKs store the body at different depths: OpenAI unwraps the
    ``{"error": {...}}`` envelope and stores the inner object, while Anthropic
    stores the whole envelope (``{"type": "error", "error": {...}}``). This
    returns the inner object for either, so ``code`` and ``type`` are read from
    the same place.
    """
    body = getattr(exc, 'body', None)
    if not isinstance(body, dict):
        return None
    inner = body.get('error')
    return inner if isinstance(inner, dict) else body


def error_text(exc: BaseException) -> str:
    """Lowercased error text for keyword matching: the inner error body, else the raw body or message."""
    source = error_body(exc) or getattr(exc, 'body', None) or getattr(exc, 'message', '') or ''
    return str(source).lower()


__all__ = [
    'API_ERRORS',
    'BAD_REQUEST_ERRORS',
    'CONNECTION_ERRORS',
    'SDK_ERRORS',
    'STATUS_ERRORS',
    'TIMEOUT_ERRORS',
    'error_body',
    'error_text',
    'status_code',
]
