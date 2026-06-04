"""Unit tests for redteam backends: base.py, openai.py, orq.py."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from evaluatorq.contracts import Message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_exc(
    *,
    message: str = "error",
    status_code: int | None = None,
    status: int | None = None,
    response_status_code: int | None = None,
    code: str | None = None,
    error_code: str | None = None,
    type_attr: str | None = None,
    body: dict[str, Any] | None = None,
) -> Exception:
    """Create a mock exception with arbitrary structured attributes."""
    exc = Exception(message)
    if response_status_code is not None:
        mock_response = MagicMock()
        mock_response.status_code = response_status_code
        exc.response = mock_response  # pyright: ignore[reportAttributeAccessIssue]
    if status_code is not None:
        exc.status_code = status_code  # pyright: ignore[reportAttributeAccessIssue]
    if status is not None:
        exc.status = status  # pyright: ignore[reportAttributeAccessIssue]
    if code is not None:
        exc.code = code  # pyright: ignore[reportAttributeAccessIssue]
    if error_code is not None:
        exc.error_code = error_code  # pyright: ignore[reportAttributeAccessIssue]
    if type_attr is not None:
        exc.type = type_attr  # pyright: ignore[reportAttributeAccessIssue]
    if body is not None:
        exc.body = body  # pyright: ignore[reportAttributeAccessIssue]
    return exc


# ===========================================================================
# backends/base.py — extract_status_code
# ===========================================================================


class TestExtractStatusCode:
    """Tests for extract_status_code()."""

    def test_from_response_status_code_attribute(self):
        from evaluatorq.redteam.backends._errors import extract_status_code

        exc = _make_exc(response_status_code=429)
        assert extract_status_code(exc) == 429

    def test_from_exc_status_code_direct(self):
        from evaluatorq.redteam.backends._errors import extract_status_code

        exc = _make_exc(status_code=500)
        assert extract_status_code(exc) == 500

    def test_from_exc_status_direct(self):
        from evaluatorq.redteam.backends._errors import extract_status_code

        exc = _make_exc(status=403)
        assert extract_status_code(exc) == 403

    def test_from_regex_http_prefix(self):
        from evaluatorq.redteam.backends._errors import extract_status_code

        exc = Exception("Request failed: HTTP 429 Too Many Requests")
        assert extract_status_code(exc) == 429

    def test_from_regex_status_code_equals(self):
        from evaluatorq.redteam.backends._errors import extract_status_code

        exc = Exception("status_code=500 Internal Server Error")
        assert extract_status_code(exc) == 500

    def test_from_regex_code_equals(self):
        from evaluatorq.redteam.backends._errors import extract_status_code

        exc = Exception("code=403 Forbidden")
        assert extract_status_code(exc) == 403

    def test_returns_none_when_no_code(self):
        from evaluatorq.redteam.backends._errors import extract_status_code

        exc = Exception("some generic error with no status")
        assert extract_status_code(exc) is None

    def test_returns_none_for_out_of_range_code(self):
        from evaluatorq.redteam.backends._errors import extract_status_code

        exc = Exception("code=999 not a valid HTTP status")
        assert extract_status_code(exc) is None

    def test_response_status_code_out_of_range_falls_through(self):
        from evaluatorq.redteam.backends._errors import extract_status_code

        # response.status_code is out of 100-599 range; should fall through
        exc = _make_exc(message="error")
        mock_response = MagicMock()
        mock_response.status_code = 99
        exc.response = mock_response  # pyright: ignore[reportAttributeAccessIssue]
        # No other code present → should return None
        assert extract_status_code(exc) is None

    def test_response_status_code_lower_boundary(self):
        from evaluatorq.redteam.backends._errors import extract_status_code

        exc = _make_exc(response_status_code=100)
        assert extract_status_code(exc) == 100

    def test_response_status_code_upper_boundary(self):
        from evaluatorq.redteam.backends._errors import extract_status_code

        exc = _make_exc(response_status_code=599)
        assert extract_status_code(exc) == 599

    def test_prefers_response_status_code_over_direct(self):
        """response.status_code is checked first and wins over exc.status_code."""
        from evaluatorq.redteam.backends._errors import extract_status_code

        exc = _make_exc(response_status_code=503, status_code=429)
        assert extract_status_code(exc) == 503


# ===========================================================================
# backends/base.py — extract_provider_error_code
# ===========================================================================


class TestExtractProviderErrorCode:
    """Tests for extract_provider_error_code()."""

    def test_from_exc_code_attribute(self):
        from evaluatorq.redteam.backends._errors import extract_provider_error_code

        exc = _make_exc(code="rate_limit_exceeded")
        assert extract_provider_error_code(exc) == "rate_limit_exceeded"

    def test_from_exc_error_code_attribute(self):
        from evaluatorq.redteam.backends._errors import extract_provider_error_code

        exc = _make_exc(error_code="invalid_api_key")
        assert extract_provider_error_code(exc) == "invalid_api_key"

    def test_from_exc_type_attribute(self):
        from evaluatorq.redteam.backends._errors import extract_provider_error_code

        exc = _make_exc(type_attr="authentication_error")
        assert extract_provider_error_code(exc) == "authentication_error"

    def test_from_body_error_code(self):
        from evaluatorq.redteam.backends._errors import extract_provider_error_code

        exc = _make_exc(body={"error": {"code": "content_filter", "message": "blocked"}})
        assert extract_provider_error_code(exc) == "content_filter"

    def test_from_body_direct_code(self):
        from evaluatorq.redteam.backends._errors import extract_provider_error_code

        exc = _make_exc(body={"code": "model_not_found"})
        assert extract_provider_error_code(exc) == "model_not_found"

    def test_from_body_error_type(self):
        from evaluatorq.redteam.backends._errors import extract_provider_error_code

        exc = _make_exc(body={"error": {"type": "invalid_request_error"}})
        assert extract_provider_error_code(exc) == "invalid_request_error"

    def test_from_regex_in_error_text(self):
        from evaluatorq.redteam.backends._errors import extract_provider_error_code

        exc = Exception("Request failed: error_code=rate_limit_exceeded")
        result = extract_provider_error_code(exc)
        assert result == "rate_limit_exceeded"

    def test_returns_none_when_no_code(self):
        from evaluatorq.redteam.backends._errors import extract_provider_error_code

        exc = Exception("Something went wrong")
        assert extract_provider_error_code(exc) is None

    def test_strips_whitespace_from_code(self):
        from evaluatorq.redteam.backends._errors import extract_provider_error_code

        exc = _make_exc(code="  rate_limit  ")
        assert extract_provider_error_code(exc) == "rate_limit"

    def test_lowercases_code(self):
        from evaluatorq.redteam.backends._errors import extract_provider_error_code

        exc = _make_exc(code="RateLimitExceeded")
        assert extract_provider_error_code(exc) == "ratelimitexceeded"

    def test_empty_string_code_ignored(self):
        from evaluatorq.redteam.backends._errors import extract_provider_error_code

        exc = _make_exc(code="   ")
        # Whitespace-only code should be treated as missing
        assert extract_provider_error_code(exc) is None


# ===========================================================================
# backends/openai.py — OpenAIModelTarget
# ===========================================================================


class TestOpenAIModelTarget:
    """Tests for OpenAIModelTarget."""

    def test_new_returns_new_instance_with_same_config(self):
        """new() produces a new OpenAIModelTarget with identical config."""
        from evaluatorq.redteam.backends.openai import OpenAIModelTarget

        mock_client = MagicMock()
        target = OpenAIModelTarget(
            model="gpt-4o-mini",
            client=mock_client,
            system_prompt="Custom prompt",
        )
        fresh = target.new()

        assert isinstance(fresh, OpenAIModelTarget)
        assert fresh is not target
        assert fresh.model == target.model
        assert fresh.system_prompt == target.system_prompt
        assert fresh.client is target.client  # Same client instance is shared


# ===========================================================================
# backends/openai.py — _openai_map_error
# ===========================================================================


class TestOpenAIErrorMapper:
    """Tests for _openai_map_error()."""

    def test_maps_http_status_code(self):
        from evaluatorq.redteam.backends.openai import _openai_map_error

        exc = _make_exc(status_code=429)
        code, msg = _openai_map_error(exc)
        assert code == "openai.http.429"
        assert "Exception" in msg

    def test_maps_provider_error_code(self):
        from evaluatorq.redteam.backends.openai import _openai_map_error

        exc = _make_exc(code="content_filter")
        code, msg = _openai_map_error(exc)
        assert code == "openai.code.content_filter"

    def test_maps_rate_limit_by_exception_name(self):
        from evaluatorq.redteam.backends.openai import _openai_map_error

        class RateLimitError(Exception):
            pass

        exc = RateLimitError("rate limit hit")
        code, msg = _openai_map_error(exc)
        assert code == "openai.rate_limit"

    def test_maps_authentication_error_by_exception_name(self):
        from evaluatorq.redteam.backends.openai import _openai_map_error

        class AuthenticationError(Exception):
            pass

        exc = AuthenticationError("bad key")
        code, msg = _openai_map_error(exc)
        assert code == "openai.auth"

    def test_maps_timeout_by_exception_name(self):
        from evaluatorq.redteam.backends.openai import _openai_map_error

        class TimeoutError(Exception):
            pass

        exc = TimeoutError("timed out")
        code, msg = _openai_map_error(exc)
        assert code == "openai.timeout"

    def test_maps_unknown_fallback(self):
        from evaluatorq.redteam.backends.openai import _openai_map_error

        exc = Exception("something unexpected")
        code, msg = _openai_map_error(exc)
        assert code == "openai.unknown"
        assert "Exception" in msg

    def test_message_includes_exception_type_and_text(self):
        from evaluatorq.redteam.backends.openai import _openai_map_error

        exc = ValueError("bad value encountered")
        code, msg = _openai_map_error(exc)
        assert "ValueError" in msg
        assert "bad value encountered" in msg


# ===========================================================================
# backends/orq.py — _orq_map_error
# ===========================================================================


class TestORQErrorMapper:
    """Tests for _orq_map_error()."""

    def test_maps_http_status_code(self):
        from evaluatorq.redteam.backends.orq import _orq_map_error

        exc = _make_exc(status_code=503)
        code, msg = _orq_map_error(exc)
        assert code == "orq.http.503"

    def test_maps_provider_error_code(self):
        from evaluatorq.redteam.backends.orq import _orq_map_error

        exc = _make_exc(code="model_unavailable")
        code, msg = _orq_map_error(exc)
        assert code == "orq.code.model_unavailable"

    def test_maps_timeout_by_exception_name(self):
        from evaluatorq.redteam.backends.orq import _orq_map_error

        class TimeoutError(Exception):
            pass

        exc = TimeoutError("request timed out")
        code, msg = _orq_map_error(exc)
        assert code == "orq.timeout"

    def test_maps_timeout_by_text(self):
        from evaluatorq.redteam.backends.orq import _orq_map_error

        exc = Exception("connection timed out after 30s")
        code, msg = _orq_map_error(exc)
        assert code == "orq.timeout"

    def test_maps_auth_by_exception_name(self):
        from evaluatorq.redteam.backends.orq import _orq_map_error

        class AuthError(Exception):
            pass

        exc = AuthError("invalid credentials")
        code, msg = _orq_map_error(exc)
        assert code == "orq.auth"

    def test_maps_auth_by_unauthorized_text(self):
        from evaluatorq.redteam.backends.orq import _orq_map_error

        exc = Exception("unauthorized access to resource")
        code, msg = _orq_map_error(exc)
        assert code == "orq.auth"

    def test_maps_auth_by_forbidden_text(self):
        from evaluatorq.redteam.backends.orq import _orq_map_error

        exc = Exception("forbidden - insufficient permissions")
        code, msg = _orq_map_error(exc)
        assert code == "orq.auth"

    def test_maps_rate_limit_by_exception_name(self):
        from evaluatorq.redteam.backends.orq import _orq_map_error

        class RateLimitError(Exception):
            pass

        exc = RateLimitError("too many requests")
        code, msg = _orq_map_error(exc)
        assert code == "orq.rate_limit"

    def test_maps_rate_limit_by_429_in_text(self):
        """When '429' appears in error text without a structured status pattern,
        the ratelimit keyword check catches it as orq.rate_limit."""
        from evaluatorq.redteam.backends.orq import _orq_map_error

        exc = Exception("received 429 from server")
        code, msg = _orq_map_error(exc)
        assert code == "orq.rate_limit"

    def test_maps_unknown_fallback(self):
        from evaluatorq.redteam.backends.orq import _orq_map_error

        exc = Exception("something else entirely")
        code, msg = _orq_map_error(exc)
        assert code == "orq.unknown"

    def test_message_contains_exception_type_and_text(self):
        from evaluatorq.redteam.backends.orq import _orq_map_error

        exc = RuntimeError("runtime failure")
        code, msg = _orq_map_error(exc)
        assert "RuntimeError" in msg
        assert "runtime failure" in msg


# ===========================================================================
# backends/orq.py — _get_orq_server_url
# ===========================================================================


class TestGetOrqServerUrl:
    """Tests for _get_orq_server_url()."""

    def test_strips_router_suffix(self, monkeypatch):
        from evaluatorq.redteam.backends.orq import _get_orq_server_url

        monkeypatch.setenv("ORQ_BASE_URL", "https://my.orq.ai/v3/router")
        assert _get_orq_server_url() == "https://my.orq.ai"

    def test_strips_trailing_slash_and_suffix(self, monkeypatch):
        from evaluatorq.redteam.backends.orq import _get_orq_server_url

        monkeypatch.setenv("ORQ_BASE_URL", "https://my.orq.ai/v3/router/")
        assert _get_orq_server_url() == "https://my.orq.ai"

    def test_returns_url_unchanged_when_no_suffix(self, monkeypatch):
        from evaluatorq.redteam.backends.orq import _get_orq_server_url

        monkeypatch.setenv("ORQ_BASE_URL", "https://my.orq.ai")
        assert _get_orq_server_url() == "https://my.orq.ai"

    def test_uses_default_when_env_not_set(self, monkeypatch):
        from evaluatorq.redteam.backends.orq import _get_orq_server_url

        monkeypatch.delenv("ORQ_BASE_URL", raising=False)
        result = _get_orq_server_url()
        assert result == "https://my.orq.ai"


# ===========================================================================
# backends/orq.py — _get_orq_api_key
# ===========================================================================


class TestGetOrqApiKey:
    """Tests for _get_orq_api_key()."""

    def test_returns_key_when_set(self, monkeypatch):
        from evaluatorq.redteam.backends.orq import _get_orq_api_key

        monkeypatch.setenv("ORQ_API_KEY", "sk-test-key")
        assert _get_orq_api_key() == "sk-test-key"

    def test_raises_runtime_error_when_not_set(self, monkeypatch):
        from evaluatorq.redteam.backends.orq import _get_orq_api_key
        from evaluatorq.redteam.exceptions import CredentialError

        monkeypatch.delenv("ORQ_API_KEY", raising=False)
        with pytest.raises(CredentialError, match="ORQ_API_KEY"):
            _get_orq_api_key()

    def test_raises_runtime_error_when_empty_string(self, monkeypatch):
        from evaluatorq.redteam.backends.orq import _get_orq_api_key
        from evaluatorq.redteam.exceptions import CredentialError

        monkeypatch.setenv("ORQ_API_KEY", "")
        with pytest.raises(CredentialError, match="ORQ_API_KEY"):
            _get_orq_api_key()


# ===========================================================================
# backends/orq.py — ORQAgentTarget._extract_tool_call_items
# ===========================================================================


class TestORQExtractToolCallItems:
    """Tests for the _extract_tool_call_items inner function via ORQAgentTarget.respond."""

    def _make_target(self) -> Any:
        from evaluatorq.redteam.backends.orq import ORQAgentTarget

        return ORQAgentTarget(agent_key="test-agent", orq_client=MagicMock())

    def _make_response(self, pending_tool_calls: list[Any]) -> MagicMock:
        resp = MagicMock()
        resp.task_id = None
        resp.pending_tool_calls = pending_tool_calls
        resp.output = []
        resp.usage = None
        resp.model = None
        return resp

    def _make_call(self, name: str, arguments: Any) -> MagicMock:
        call = MagicMock()
        call.name = name
        call.arguments = arguments
        call.id = "call_1"
        return call

    @pytest.mark.asyncio
    async def test_extracts_tool_call_with_dict_arguments(self) -> None:
        from unittest.mock import patch

        target = self._make_target()
        call = self._make_call("search", {"query": "test"})
        first_resp = self._make_response([call])
        # Second response (after synthetic tool_result) has no pending calls
        second_resp = self._make_response([])
        _part = MagicMock()
        _part.kind = "text"
        _part.text = "result"
        _item = MagicMock()
        _item.parts = [_part]
        second_resp.output = [_item]

        async def fake_to_thread(fn, **kwargs):  # type: ignore[return]
            return fn(**kwargs)

        with patch("evaluatorq.redteam.backends.orq.asyncio.to_thread", side_effect=fake_to_thread):
            target.orq_client.agents.responses.create = MagicMock(side_effect=[first_resp, second_resp])
            response = await target.respond([Message(role="user", content="find something")])

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "search"
        assert response.tool_calls[0].arguments_dict == {"query": "test"}
        assert response.text == "result"

    @pytest.mark.asyncio
    async def test_extracts_tool_call_with_json_string_arguments(self) -> None:
        import asyncio
        from unittest.mock import patch

        target = self._make_target()
        call = self._make_call("send_email", '{"to": "user@example.com", "subject": "hi"}')
        first_resp = self._make_response([call])
        second_resp = self._make_response([])
        _part = MagicMock()
        _part.kind = "text"
        _part.text = "sent"
        _item = MagicMock()
        _item.parts = [_part]
        second_resp.output = [_item]

        async def fake_to_thread(fn, **kwargs):  # type: ignore[return]
            return fn(**kwargs)

        with patch("evaluatorq.redteam.backends.orq.asyncio.to_thread", side_effect=fake_to_thread):
            target.orq_client.agents.responses.create = MagicMock(side_effect=[first_resp, second_resp])
            response = await target.respond([Message(role="user", content="send a message")])

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "send_email"
        assert response.tool_calls[0].arguments_dict == {"to": "user@example.com", "subject": "hi"}
        assert response.text == "sent"

    @pytest.mark.asyncio
    async def test_json_parse_fallback_for_invalid_string_arguments(self) -> None:
        import asyncio
        from unittest.mock import patch

        target = self._make_target()
        call = self._make_call("bad_tool", "not-valid-json")
        first_resp = self._make_response([call])
        second_resp = self._make_response([])
        _part = MagicMock()
        _part.kind = "text"
        _part.text = "done"
        _item = MagicMock()
        _item.parts = [_part]
        second_resp.output = [_item]

        async def fake_to_thread(fn, **kwargs):  # type: ignore[return]
            return fn(**kwargs)

        with patch("evaluatorq.redteam.backends.orq.asyncio.to_thread", side_effect=fake_to_thread):
            target.orq_client.agents.responses.create = MagicMock(side_effect=[first_resp, second_resp])
            response = await target.respond([Message(role="user", content="trigger bad tool")])

        assert len(response.tool_calls) == 1
        # Invalid JSON args wrapped under 'raw' key so arguments_dict still parses
        assert response.tool_calls[0].arguments_dict == {"raw": "not-valid-json"}
        assert response.text == "done"
