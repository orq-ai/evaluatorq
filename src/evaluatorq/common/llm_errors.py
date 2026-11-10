"""SDK-neutral LLM exception classes.

Each tuple names the matching exception class from the ``openai`` SDK and, when
it is installed, from the ``anthropic`` SDK, so an ``except`` clause or an
``isinstance`` check recognises an error from either one. ``anthropic`` is an
optional extra: when it is absent the tuples hold only the OpenAI classes.

``openai.LengthFinishReasonError`` is deliberately not here. It is a
client-side parse error, not an API error, and stays imported from ``openai``.
"""

from __future__ import annotations

import openai

API_ERRORS: tuple[type[BaseException], ...] = (openai.APIError,)
STATUS_ERRORS: tuple[type[BaseException], ...] = (openai.APIStatusError,)
CONNECTION_ERRORS: tuple[type[BaseException], ...] = (openai.APIConnectionError,)
TIMEOUT_ERRORS: tuple[type[BaseException], ...] = (openai.APITimeoutError,)
BAD_REQUEST_ERRORS: tuple[type[BaseException], ...] = (openai.BadRequestError,)
SDK_ERRORS: tuple[type[BaseException], ...] = (openai.OpenAIError,)

try:
    import anthropic
except ImportError:  # anthropic is an optional extra
    pass
else:
    API_ERRORS += (anthropic.APIError,)
    STATUS_ERRORS += (anthropic.APIStatusError,)
    CONNECTION_ERRORS += (anthropic.APIConnectionError,)
    TIMEOUT_ERRORS += (anthropic.APITimeoutError,)
    BAD_REQUEST_ERRORS += (anthropic.BadRequestError,)
    SDK_ERRORS += (anthropic.AnthropicError,)


def status_code(exc: BaseException) -> int | None:
    """Return the HTTP status of a status error from either SDK, else ``None``."""
    if isinstance(exc, STATUS_ERRORS):
        code = getattr(exc, 'status_code', None)
        return code if isinstance(code, int) else None
    return None


__all__ = [
    'API_ERRORS',
    'BAD_REQUEST_ERRORS',
    'CONNECTION_ERRORS',
    'SDK_ERRORS',
    'STATUS_ERRORS',
    'TIMEOUT_ERRORS',
    'status_code',
]
