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
from loguru import logger

API_ERRORS: tuple[type[BaseException], ...] = (openai.APIError,)
STATUS_ERRORS: tuple[type[BaseException], ...] = (openai.APIStatusError,)
CONNECTION_ERRORS: tuple[type[BaseException], ...] = (openai.APIConnectionError,)
TIMEOUT_ERRORS: tuple[type[BaseException], ...] = (openai.APITimeoutError,)
BAD_REQUEST_ERRORS: tuple[type[BaseException], ...] = (openai.BadRequestError,)
SDK_ERRORS: tuple[type[BaseException], ...] = (openai.OpenAIError,)

try:
    import anthropic

    _ANTHROPIC_ERRORS = (
        anthropic.APIError,
        anthropic.APIStatusError,
        anthropic.APIConnectionError,
        anthropic.APITimeoutError,
        anthropic.BadRequestError,
        anthropic.AnthropicError,
    )
except ImportError:  # anthropic is an optional extra
    pass
except AttributeError as exc:  # installed, but not the API this module expects
    logger.warning(
        'The installed anthropic package lacks an expected error class ({}); '
        'only OpenAI error classes are recognised, so Anthropic errors will not be caught as LLM errors',
        exc,
    )
else:
    API_ERRORS += (_ANTHROPIC_ERRORS[0],)
    STATUS_ERRORS += (_ANTHROPIC_ERRORS[1],)
    CONNECTION_ERRORS += (_ANTHROPIC_ERRORS[2],)
    TIMEOUT_ERRORS += (_ANTHROPIC_ERRORS[3],)
    BAD_REQUEST_ERRORS += (_ANTHROPIC_ERRORS[4],)
    SDK_ERRORS += (_ANTHROPIC_ERRORS[5],)


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
