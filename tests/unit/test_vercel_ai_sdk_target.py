"""Unit tests for Vercel AI SDK red teaming target."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from evaluatorq.integrations.vercel_ai_sdk_integration import VercelAISdkTarget
from evaluatorq.integrations.vercel_ai_sdk_integration.target import (
    _parse_data_stream,
    _parse_json_response,
)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class TestParseDataStream:
    def test_single_chunk(self) -> None:
        raw = '0:"Hello world"\n'
        assert _parse_data_stream(raw) == "Hello world"

    def test_multiple_chunks(self) -> None:
        raw = '0:"Hello"\n0:" world"\n'
        assert _parse_data_stream(raw) == "Hello world"

    def test_ignores_non_text_lines(self) -> None:
        raw = (
            '0:"Hello"\n'
            'e:{"finishReason":"stop","usage":{"promptTokens":10}}\n'
            'd:{"finishReason":"stop"}\n'
            '0:" world"\n'
        )
        assert _parse_data_stream(raw) == "Hello world"

    def test_empty_stream(self) -> None:
        assert _parse_data_stream("") == ""

    def test_only_metadata_lines(self) -> None:
        raw = 'e:{"finishReason":"stop"}\nd:{"finishReason":"stop"}\n'
        assert _parse_data_stream(raw) == ""


class TestParseJsonResponse:
    def test_message_dict(self) -> None:
        raw = '{"message": {"content": "Hello"}}'
        assert _parse_json_response(raw) == "Hello"

    def test_message_string(self) -> None:
        raw = '{"message": "Hello"}'
        assert _parse_json_response(raw) == "Hello"

    def test_text_key(self) -> None:
        raw = '{"text": "Hello"}'
        assert _parse_json_response(raw) == "Hello"

    def test_content_key(self) -> None:
        raw = '{"content": "Hello"}'
        assert _parse_json_response(raw) == "Hello"

    def test_openai_compat_choices(self) -> None:
        raw = '{"choices": [{"message": {"content": "Hello"}}]}'
        assert _parse_json_response(raw) == "Hello"

    def test_plain_string(self) -> None:
        raw = '"Hello"'
        assert _parse_json_response(raw) == "Hello"

    def test_invalid_json(self) -> None:
        raw = "not json at all"
        assert _parse_json_response(raw) == "not json at all"


# ---------------------------------------------------------------------------
# Target protocol
# ---------------------------------------------------------------------------


def _mock_response(text: str, content_type: str = "text/plain") -> httpx.Response:
    return httpx.Response(
        status_code=200,
        text=text,
        headers={"content-type": content_type},
        request=httpx.Request("POST", "http://test"),
    )


class TestVercelAISdkTarget:
    @pytest.mark.asyncio
    async def test_send_prompt_returns_response(self) -> None:
        target = VercelAISdkTarget("http://test/api/chat")
        mock_response = _mock_response('0:"Hello"\n')

        with patch("evaluatorq.integrations.vercel_ai_sdk_integration.target.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            response = await target.send_prompt("hello")

        assert response.text == "Hello"
        assert response.tool_calls == []

    @pytest.mark.asyncio
    async def test_sends_messages_format(self) -> None:
        target = VercelAISdkTarget("http://test/api/chat")
        mock_response = _mock_response('0:"ok"\n')

        with patch("evaluatorq.integrations.vercel_ai_sdk_integration.target.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            await target.send_prompt("hello")

        call_kwargs = mock_client.post.call_args
        body = call_kwargs.kwargs["json"]
        assert body["messages"] == [{"role": "user", "content": "hello"}]

    @pytest.mark.asyncio
    async def test_multi_turn_sends_history(self) -> None:
        target = VercelAISdkTarget("http://test/api/chat")

        responses = [
            _mock_response('0:"first response"\n'),
            _mock_response('0:"second response"\n'),
        ]

        with patch("evaluatorq.integrations.vercel_ai_sdk_integration.target.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=responses)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            await target.send_prompt("first")
            await target.send_prompt("second")

        second_call = mock_client.post.call_args_list[1]
        messages = second_call.kwargs["json"]["messages"]
        assert messages == [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "first response"},
            {"role": "user", "content": "second"},
        ]

    @pytest.mark.asyncio
    async def test_extra_body_merged(self) -> None:
        target = VercelAISdkTarget("http://test/api/chat", extra_body={"model": "gpt-4o"})
        mock_response = _mock_response('0:"ok"\n')

        with patch("evaluatorq.integrations.vercel_ai_sdk_integration.target.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            await target.send_prompt("hello")

        body = mock_client.post.call_args.kwargs["json"]
        assert body["model"] == "gpt-4o"

    def test_reset_clears_history(self) -> None:
        target = VercelAISdkTarget("http://test/api/chat")
        target._history = [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old reply"},
        ]
        target.reset_conversation()
        assert target._history == []

    def test_clone_returns_independent_instance(self) -> None:
        target = VercelAISdkTarget(
            "http://test/api/chat",
            headers={"Authorization": "Bearer sk-test"},
            extra_body={"model": "gpt-4o"},
            timeout=60.0,
        )
        target._history = [{"role": "user", "content": "old"}]

        cloned = target.clone()

        assert cloned is not target
        assert cloned._history == []
        assert cloned._url == "http://test/api/chat"
        assert cloned._headers == {"Authorization": "Bearer sk-test"}
        assert cloned._headers is not target._headers
        assert cloned._extra_body == {"model": "gpt-4o"}
        assert cloned._extra_body is not target._extra_body
        assert cloned._timeout == 60.0

    def test_parse_json_response(self) -> None:
        response = _mock_response(
            '{"message": {"content": "Hello"}}',
            content_type="application/json",
        )
        text = VercelAISdkTarget._parse_response(response)
        assert text == "Hello"

    def test_parse_data_stream_response(self) -> None:
        response = _mock_response('0:"Hello"\n0:" world"\n')
        text = VercelAISdkTarget._parse_response(response)
        assert text == "Hello world"


class TestVercelAISdkTargetAgentContext:
    @pytest.mark.asyncio
    async def test_get_agent_context_default_is_minimal(self) -> None:
        target = VercelAISdkTarget("http://test/api/chat")
        ctx = await target.get_agent_context()
        assert ctx.key == "http://test/api/chat"
        assert ctx.tools == []
        assert ctx.memory_stores == []
        assert ctx.description == "opaque Vercel AI SDK HTTP target"

    @pytest.mark.asyncio
    async def test_get_agent_context_strips_credentials_from_url_key(self) -> None:
        target = VercelAISdkTarget("https://user:secret@api.example.com/chat?token=abc")
        ctx = await target.get_agent_context()
        assert "secret" not in ctx.key
        assert "token" not in ctx.key
        assert ctx.key == "https://api.example.com/chat"

    @pytest.mark.asyncio
    async def test_get_agent_context_returns_override(self) -> None:
        from evaluatorq.redteam.contracts import AgentContext, ToolInfo

        override = AgentContext(
            key="vercel-bot",
            tools=[ToolInfo(name="http_tool")],
        )
        target = VercelAISdkTarget("http://test/api/chat", agent_context=override)
        ctx = await target.get_agent_context()
        assert ctx is override

    def test_clone_preserves_agent_context(self) -> None:
        from evaluatorq.redteam.contracts import AgentContext

        override = AgentContext(key="k")
        target = VercelAISdkTarget("http://test/api/chat", agent_context=override)
        cloned = target.clone()
        assert cloned._agent_context is override
