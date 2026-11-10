from __future__ import annotations

import importlib
import sys

import anthropic
import httpx
import openai
import pytest

from evaluatorq.common import llm_errors
from evaluatorq.common.llm_errors import (
    API_ERRORS,
    BAD_REQUEST_ERRORS,
    CONNECTION_ERRORS,
    SDK_ERRORS,
    STATUS_ERRORS,
    TIMEOUT_ERRORS,
    status_code,
)

_REQUEST = httpx.Request('POST', 'https://example.test/v1')


def _response(code: int) -> httpx.Response:
    return httpx.Response(code, request=_REQUEST)


def _openai_status(code: int) -> openai.APIStatusError:
    cls = openai.BadRequestError if code == 400 else openai.APIStatusError
    return cls('boom', response=_response(code), body=None)


def _anthropic_status(code: int) -> anthropic.APIStatusError:
    cls = anthropic.BadRequestError if code == 400 else anthropic.APIStatusError
    return cls('boom', response=_response(code), body=None)


STATUS_FACTORIES = [_openai_status, _anthropic_status]


@pytest.mark.parametrize('make', STATUS_FACTORIES)
def test_bad_request_is_status_and_api_error_but_not_connection(make):
    exc = make(400)
    assert isinstance(exc, BAD_REQUEST_ERRORS)
    assert isinstance(exc, STATUS_ERRORS)
    assert isinstance(exc, API_ERRORS)
    assert isinstance(exc, SDK_ERRORS)
    assert not isinstance(exc, CONNECTION_ERRORS)
    assert not isinstance(exc, TIMEOUT_ERRORS)


@pytest.mark.parametrize('make', STATUS_FACTORIES)
def test_non_400_status_is_not_bad_request(make):
    exc = make(500)
    assert isinstance(exc, STATUS_ERRORS)
    assert not isinstance(exc, BAD_REQUEST_ERRORS)


@pytest.mark.parametrize(
    'exc',
    [
        openai.APIConnectionError(request=_REQUEST),
        anthropic.APIConnectionError(request=_REQUEST),
    ],
)
def test_connection_error_is_api_error_but_not_status(exc):
    assert isinstance(exc, CONNECTION_ERRORS)
    assert isinstance(exc, API_ERRORS)
    assert not isinstance(exc, STATUS_ERRORS)
    assert not isinstance(exc, TIMEOUT_ERRORS)


@pytest.mark.parametrize(
    'exc',
    [openai.APITimeoutError(_REQUEST), anthropic.APITimeoutError(_REQUEST)],
)
def test_timeout_is_connection_error(exc):
    assert isinstance(exc, TIMEOUT_ERRORS)
    assert isinstance(exc, CONNECTION_ERRORS)


def test_sdk_errors_cover_client_side_errors():
    assert isinstance(openai.OpenAIError('x'), SDK_ERRORS)
    assert isinstance(anthropic.AnthropicError('x'), SDK_ERRORS)
    assert not isinstance(openai.OpenAIError('x'), API_ERRORS)
    assert not isinstance(ValueError('x'), SDK_ERRORS)


@pytest.mark.parametrize('make', STATUS_FACTORIES)
def test_status_code_returns_http_status(make):
    assert status_code(make(429)) == 429


def test_status_code_none_for_non_status_errors():
    assert status_code(ValueError('x')) is None
    assert status_code(openai.APIConnectionError(request=_REQUEST)) is None
    assert status_code(anthropic.APIConnectionError(request=_REQUEST)) is None


def test_tuples_contain_only_openai_when_anthropic_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, 'anthropic', None)
    try:
        reloaded = importlib.reload(llm_errors)
        for name in (
            'API_ERRORS',
            'STATUS_ERRORS',
            'CONNECTION_ERRORS',
            'TIMEOUT_ERRORS',
            'BAD_REQUEST_ERRORS',
            'SDK_ERRORS',
        ):
            members = getattr(reloaded, name)
            assert len(members) == 1, name
            assert members[0].__module__.split('.')[0] == 'openai', name
        assert reloaded.status_code(_openai_status(404)) == 404
    finally:
        monkeypatch.undo()
        importlib.reload(llm_errors)
    assert len(llm_errors.API_ERRORS) == 2
