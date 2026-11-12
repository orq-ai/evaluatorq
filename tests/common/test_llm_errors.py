from __future__ import annotations

import importlib
import importlib.util

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
    error_body,
    error_text,
    status_code,
)
from tests.common.llm_error_factories import STATUS_FACTORIES, openai_status

anthropic = pytest.importorskip('anthropic')

_REQUEST = httpx.Request('POST', 'https://example.test/v1')


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


def test_tuples_contain_only_openai_when_anthropic_not_installed(monkeypatch):
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        'find_spec',
        lambda name, *args: None if name == 'anthropic' else real_find_spec(name, *args),
    )
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
            assert all(m.__module__.split('.')[0] == 'openai' for m in members), name
        assert reloaded.status_code(openai_status(404)) == 404
    finally:
        monkeypatch.undo()
        importlib.reload(llm_errors)
    assert anthropic.APIError in llm_errors.API_ERRORS


_INNER = {'type': 'invalid_request_error', 'code': 'content_filter', 'message': 'Response_Format Rejected'}


def test_error_body_returns_the_inner_object_for_both_sdk_body_shapes():
    """OpenAI stores the inner error object; Anthropic stores the whole envelope around it."""
    response = httpx.Response(400, request=_REQUEST)
    openai_exc = openai.BadRequestError('boom', response=response, body=dict(_INNER))
    anthropic_exc = anthropic.BadRequestError('boom', response=response, body={'type': 'error', 'error': dict(_INNER)})

    assert error_body(openai_exc) == _INNER
    assert error_body(anthropic_exc) == _INNER
    assert error_text(openai_exc) == error_text(anthropic_exc)
    assert 'response_format rejected' in error_text(anthropic_exc)


def test_error_body_and_text_without_a_dict_body():
    response = httpx.Response(400, request=_REQUEST)
    assert error_body(ValueError('x')) is None
    assert error_body(openai.BadRequestError('Raw Message', response=response, body=None)) is None
    assert error_text(openai.BadRequestError('Raw Message', response=response, body=None)) == 'raw message'
    assert error_text(openai.BadRequestError('m', response=response, body='Plain Body')) == 'plain body'
