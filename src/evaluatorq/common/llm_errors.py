"""SDK-neutral LLM exception classes.

Each tuple names the matching exception class from the ``openai`` SDK and, when
it is installed, from the ``anthropic`` SDK, so an ``except`` clause or an
``isinstance`` check recognises an error from either one. ``anthropic`` is an
optional extra: when it is not installed the tuples hold only the OpenAI classes.
An installed ``anthropic`` that cannot be imported raises at import time instead
of being skipped.

``openai.LengthFinishReasonError`` is deliberately not here. It is a
client-side parse error, not an API error, and stays imported from ``openai``.
"""

from __future__ import annotations

import importlib.util

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


__all__ = [
    'API_ERRORS',
    'BAD_REQUEST_ERRORS',
    'CONNECTION_ERRORS',
    'SDK_ERRORS',
    'STATUS_ERRORS',
    'TIMEOUT_ERRORS',
    'status_code',
]
