"""Unit tests for Vercel AI SDK red teaming target."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from evaluatorq.contracts import FunctionCall, Message, StrategyToolCall
from evaluatorq.integrations.vercel_ai_sdk_integration import VercelAISdkTarget
from evaluatorq.integrations.vercel_ai_sdk_integration.target import (
    _parse_data_stream,
    _parse_json_response,
)
from evaluatorq.redteam.contracts import TokenUsage


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class TestParseDataStream:
    def test_single_chunk(self) -> None:
        raw = '0:"Hello world"\n'
        text, usage = _parse_data_stream(raw)
        assert text == "Hello world"
        assert usage is None

    def test_multiple_chunks(self) -> None:
        raw = '0:"Hello"\n0:" world"\n'
        text, usage = _parse_data_stream(raw)
        assert text == "Hello world"
        assert usage is None

    def test_ignores_non_text_lines(self) -> None:
        raw = (
            '0:"Hello"\n'
            'e:{"finishReason":"stop","usage":{"promptTokens":10}}\n'
            'd:{"finishReason":"stop"}\n'
            '0:" world"\n'
        )
        text, usage = _parse_data_stream(raw)
        assert text == "Hello world"

    def test_empty_stream(self) -> None:
        text, usage = _parse_data_stream("")
        assert text == ""
        assert usage is None

    def test_only_metadata_lines(self) -> None:
        raw = 'e:{"finishReason":"stop"}\nd:{"finishReason":"stop"}\n'
        text, usage = _parse_data_stream(raw)
        assert text == ""

    def test_usage_extracted_from_e_frame(self) -> None:
        """Usage from an 'e:' finish frame is captured in the returned TokenUsage."""
        raw = (
            '0:"Hello"\n'
            '0:" world"\n'
            'e:{"finishReason":"stop","usage":{"promptTokens":10,"completionTokens":5,"totalTokens":15}}\n'
        )
        text, usage = _parse_data_stream(raw)
        assert text == "Hello world"
        assert isinstance(usage, TokenUsage)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 5
        assert usage.total_tokens == 15
        assert usage.calls == 1

    def test_usage_extracted_from_d_frame(self) -> None:
        """Usage from a 'd:' done frame is captured in the returned TokenUsage."""
        raw = (
            '0:"Reply"\n'
            'd:{"finishReason":"stop","usage":{"promptTokens":20,"completionTokens":8}}\n'
        )
        text, usage = _parse_data_stream(raw)
        assert text == "Reply"
        assert isinstance(usage, TokenUsage)
        assert usage.prompt_tokens == 20
        assert usage.completion_tokens == 8
        # totalTokens absent → falls back to sum
        assert usage.total_tokens == 28

    def test_no_usage_field_in_finish_frame_yields_none(self) -> None:
        """Finish frame without 'usage' key does not populate TokenUsage."""
        raw = '0:"Hi"\ne:{"finishReason":"stop"}\n'
        _text, usage = _parse_data_stream(raw)
        assert usage is None


class TestParseJsonResponse:
    def test_message_dict(self) -> None:
        raw = '{"message": {"content": "Hello"}}'
        text, usage = _parse_json_response(raw)
        assert text == "Hello"
        assert usage is None

    def test_message_string(self) -> None:
        raw = '{"message": "Hello"}'
        text, usage = _parse_json_response(raw)
        assert text == "Hello"
        assert usage is None

    def test_text_key(self) -> None:
        raw = '{"text": "Hello"}'
        text, usage = _parse_json_response(raw)
        assert text == "Hello"
        assert usage is None

    def test_content_key(self) -> None:
        raw = '{"content": "Hello"}'
        text, usage = _parse_json_response(raw)
        assert text == "Hello"
        assert usage is None

    def test_openai_compat_choices(self) -> None:
        raw = '{"choices": [{"message": {"content": "Hello"}}]}'
        text, usage = _parse_json_response(raw)
        assert text == "Hello"
        assert usage is None

    def test_plain_string(self) -> None:
        raw = '"Hello"'
        text, usage = _parse_json_response(raw)
        assert text == "Hello"
        assert usage is None

    def test_invalid_json(self) -> None:
        raw = "not json at all"
        text, usage = _parse_json_response(raw)
        assert text == "not json at all"
        assert usage is None

    def test_usage_extracted_openai_snake_case(self) -> None:
        """OpenAI-style usage with snake_case keys is captured."""
        raw = '{"text": "hi", "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}'
        text, usage = _parse_json_response(raw)
        assert text == "hi"
        assert isinstance(usage, TokenUsage)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 5
        assert usage.total_tokens == 15
        assert usage.calls == 1

    def test_usage_extracted_vercel_camel_case(self) -> None:
        """Vercel-style usage with camelCase keys is captured as fallback."""
        raw = '{"text": "bye", "usage": {"promptTokens": 7, "completionTokens": 3, "totalTokens": 10}}'
        text, usage = _parse_json_response(raw)
        assert text == "bye"
        assert isinstance(usage, TokenUsage)
        assert usage.prompt_tokens == 7
        assert usage.completion_tokens == 3
        assert usage.total_tokens == 10

    def test_usage_absent_yields_none(self) -> None:
        """JSON without a 'usage' key returns usage=None."""
        raw = '{"text": "no usage here"}'
        _text, usage = _parse_json_response(raw)
        assert usage is None


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
    async def test_respond_returns_response(self) -> None:
        target = VercelAISdkTarget("http://test/api/chat")
        mock_response = _mock_response('0:"Hello"\n')

        with patch("evaluatorq.integrations.vercel_ai_sdk_integration.target.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await target.respond([Message(role="user", content="hello")])

        assert result.text == "Hello"
        assert result.tool_calls == []

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

            await target.respond([Message(role="user", content="hello")])

        call_kwargs = mock_client.post.call_args
        body = call_kwargs.kwargs["json"]
        assert body["messages"] == [{"role": "user", "content": "hello"}]

    @pytest.mark.asyncio
    async def test_sends_tool_calls_as_ai_sdk_parts(self) -> None:
        """A replayed transcript with tool calls becomes AI SDK v5 CoreMessage parts."""
        target = VercelAISdkTarget("http://test/api/chat")
        mock_response = _mock_response('0:"ok"\n')

        with patch("evaluatorq.integrations.vercel_ai_sdk_integration.target.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            await target.respond([
                Message(role="user", content="q1"),
                Message(
                    role="assistant",
                    content=None,
                    tool_calls=[StrategyToolCall(id="c1", function=FunctionCall(name="lookup", arguments='{"q": "x"}'))],
                ),
                Message(role="tool", tool_call_id="c1", name="lookup", content="r"),
            ])

        body = mock_client.post.call_args.kwargs["json"]
        # Plain user turn unchanged
        assert {"role": "user", "content": "q1"} in body["messages"]
        # Assistant tool call rendered as a v5 tool-call part with parsed `input`
        assistant = next(m for m in body["messages"] if m["role"] == "assistant")
        assert assistant["content"] == [
            {"type": "tool-call", "toolCallId": "c1", "toolName": "lookup", "input": {"q": "x"}}
        ]
        # Tool result rendered as a v5 tool-result part with typed `output`
        tool_row = next(m for m in body["messages"] if m["role"] == "tool")
        assert tool_row["content"] == [
            {
                "type": "tool-result",
                "toolCallId": "c1",
                "toolName": "lookup",
                "output": {"type": "text", "value": "r"},
            }
        ]

    @pytest.mark.asyncio
    async def test_sends_tool_calls_as_v4_parts(self) -> None:
        """message_format='v4' renders tool turns with legacy `args` / bare `result`."""
        target = VercelAISdkTarget("http://test/api/chat", message_format="v4")
        mock_response = _mock_response('0:"ok"\n')

        with patch("evaluatorq.integrations.vercel_ai_sdk_integration.target.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            await target.respond([
                Message(role="user", content="q1"),
                Message(
                    role="assistant",
                    content=None,
                    tool_calls=[StrategyToolCall(id="c1", function=FunctionCall(name="lookup", arguments='{"q": "x"}'))],
                ),
                Message(role="tool", tool_call_id="c1", name="lookup", content="r"),
            ])

        body = mock_client.post.call_args.kwargs["json"]
        assert {"role": "user", "content": "q1"} in body["messages"]
        # v4 tool-call uses `args`, not `input`
        assistant = next(m for m in body["messages"] if m["role"] == "assistant")
        assert assistant["content"] == [
            {"type": "tool-call", "toolCallId": "c1", "toolName": "lookup", "args": {"q": "x"}}
        ]
        # v4 tool-result uses a bare `result`, not the typed `output` wrapper
        tool_row = next(m for m in body["messages"] if m["role"] == "tool")
        assert tool_row["content"] == [
            {"type": "tool-result", "toolCallId": "c1", "toolName": "lookup", "result": "r"}
        ]

    def test_new_preserves_message_format(self) -> None:
        """new() forwards the configured tool wire format to the fresh instance."""
        target = VercelAISdkTarget("http://test/api/chat", message_format="v4")
        assert target.new()._message_format == "v4"

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

            await target.respond([Message(role="user", content="hello")])

        body = mock_client.post.call_args.kwargs["json"]
        assert body["model"] == "gpt-4o"

    def test_clone_returns_independent_instance(self) -> None:
        target = VercelAISdkTarget(
            "http://test/api/chat",
            headers={"Authorization": "Bearer sk-test"},
            extra_body={"model": "gpt-4o"},
            timeout=60.0,
        )

        cloned = target.new()

        assert cloned is not target
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
        text, usage = VercelAISdkTarget._parse_response(response)
        assert text == "Hello"
        assert usage is None

    def test_parse_data_stream_response(self) -> None:
        response = _mock_response('0:"Hello"\n0:" world"\n')
        text, usage = VercelAISdkTarget._parse_response(response)
        assert text == "Hello world"
        assert usage is None

    @pytest.mark.asyncio
    async def test_respond_plumbs_stream_usage(self) -> None:
        """Usage from the data stream 'e:' frame is returned in AgentResponse.usage."""
        target = VercelAISdkTarget("http://test/api/chat")
        stream = (
            '0:"Hello"\n'
            'e:{"finishReason":"stop","usage":{"promptTokens":12,"completionTokens":4,"totalTokens":16}}\n'
        )
        mock_response = _mock_response(stream)

        with patch("evaluatorq.integrations.vercel_ai_sdk_integration.target.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await target.respond([Message(role="user", content="hi")])

        assert result.text == "Hello"
        assert isinstance(result.usage, TokenUsage)
        assert result.usage.prompt_tokens == 12
        assert result.usage.completion_tokens == 4
        assert result.usage.total_tokens == 16
        assert result.usage.calls == 1

    @pytest.mark.asyncio
    async def test_respond_plumbs_json_usage(self) -> None:
        """Usage from an OpenAI-compat JSON response is returned in AgentResponse.usage."""
        target = VercelAISdkTarget("http://test/api/chat")
        body = '{"text": "ok", "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10}}'
        mock_response = _mock_response(body, content_type="application/json")

        with patch("evaluatorq.integrations.vercel_ai_sdk_integration.target.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await target.respond([Message(role="user", content="hello")])

        assert result.text == "ok"
        assert isinstance(result.usage, TokenUsage)
        assert result.usage.prompt_tokens == 8
        assert result.usage.completion_tokens == 2

    @pytest.mark.asyncio
    async def test_respond_usage_none_without_usage_frame(self) -> None:
        """When stream has no finish frame with usage, AgentResponse.usage is None."""
        target = VercelAISdkTarget("http://test/api/chat")
        mock_response = _mock_response('0:"Plain response"\n')

        with patch("evaluatorq.integrations.vercel_ai_sdk_integration.target.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await target.respond([Message(role="user", content="hi")])

        assert result.text == "Plain response"
        assert result.usage is None

    @pytest.mark.asyncio
    async def test_respond_propagates_http_error(self) -> None:
        """An HTTP error from the endpoint propagates out of respond()."""
        target = VercelAISdkTarget("http://x/agent")

        error_response = httpx.Response(
            status_code=500,
            text="Internal Server Error",
            request=httpx.Request("POST", "http://x/agent"),
        )

        with patch("evaluatorq.integrations.vercel_ai_sdk_integration.target.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=error_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            with pytest.raises(httpx.HTTPStatusError):
                await target.respond([Message(role="user", content="hi")])


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
        cloned = target.new()
        assert cloned._agent_context is override
