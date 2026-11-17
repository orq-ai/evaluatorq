"""Builders for LLM SDK errors from both the OpenAI and the Anthropic SDK, so a test can run against each."""

from __future__ import annotations

import anthropic
import httpx
import openai

_REQUEST = httpx.Request('POST', 'https://example.test/v1')


def _response(code: int) -> httpx.Response:
    return httpx.Response(code, request=_REQUEST)


def openai_status(code: int, message: str = 'boom') -> openai.APIStatusError:
    cls = openai.BadRequestError if code == 400 else openai.APIStatusError
    return cls(message, response=_response(code), body=None)


def anthropic_status(code: int, message: str = 'boom') -> anthropic.APIStatusError:
    cls = anthropic.BadRequestError if code == 400 else anthropic.APIStatusError
    return cls(message, response=_response(code), body=None)


def openai_connection() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=_REQUEST)


def anthropic_connection() -> anthropic.APIConnectionError:
    return anthropic.APIConnectionError(request=_REQUEST)


def openai_timeout() -> openai.APITimeoutError:
    return openai.APITimeoutError(request=_REQUEST)


def anthropic_timeout() -> anthropic.APITimeoutError:
    return anthropic.APITimeoutError(request=_REQUEST)


STATUS_FACTORIES = [openai_status, anthropic_status]
CONNECTION_FACTORIES = [openai_connection, anthropic_connection]
# APITimeoutError subclasses APIConnectionError in both SDKs, so only clause order tells a timeout from a connection error.
TIMEOUT_FACTORIES = [openai_timeout, anthropic_timeout]
