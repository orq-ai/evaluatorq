"""Tests for BaseAgent._call_llm dispatch based on config.api.

Verifies that:
- config.api == "chat_completions" routes to _call_chat_completions
- config.api == "responses" routes to _call_responses
- Default config (no api=) uses chat_completions path
"""

from __future__ import annotations

# ruff: noqa: S101, SLF001
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from openai import BadRequestError

import pytest

from evaluatorq.contracts import LLMCallConfig
from evaluatorq.simulation.agents.base import BaseAgent, LLMResult
from evaluatorq.simulation.types import Message

# ---------------------------------------------------------------------------
# Concrete subclass for testing (BaseAgent is abstract)
# ---------------------------------------------------------------------------


class _ConcreteAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "TestAgent"

    @property
    def system_prompt(self) -> str:
        return "You are a test agent."


def _make_client() -> MagicMock:
    """Build a minimal mock AsyncOpenAI client."""
    client = MagicMock()
    # chat.completions.create is a coroutine
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock()
    # responses.create is a coroutine
    client.responses = MagicMock()
    client.responses.create = AsyncMock()
    return client


def _make_messages() -> list[Message]:
    return [Message(role="user", content="hello")]


def _chat_response(content: str | None) -> MagicMock:
    mock_message = MagicMock()
    mock_message.content = content
    mock_message.tool_calls = None
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = None
    return mock_response


def _bad_request(message: str) -> BadRequestError:
    request = httpx.Request("POST", "https://example/v1/responses")
    response = httpx.Response(400, request=request)
    return BadRequestError(message, response=response, body={"error": {"message": message}})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCallLlmDispatch:
    @pytest.mark.asyncio
    async def test_chat_completions_api_calls_chat_completions(self):
        """config.api == 'chat_completions' must call _call_chat_completions."""
        client = _make_client()
        config = LLMCallConfig(model="gpt-4o", api="chat_completions", client=client)
        agent = _ConcreteAgent(config)

        expected_result = LLMResult(content="chat response")
        with (
            patch.object(agent, "_call_chat_completions", new=AsyncMock(return_value=expected_result)) as mock_cc,
            patch.object(agent, "_call_responses", new=AsyncMock()) as mock_resp,
        ):
            result = await agent._call_llm(_make_messages())

        mock_cc.assert_awaited_once()
        mock_resp.assert_not_awaited()
        assert result.content == "chat response"

    @pytest.mark.asyncio
    async def test_responses_api_calls_responses(self):
        """config.api == 'responses' must call _call_responses."""
        client = _make_client()
        config = LLMCallConfig(model="gpt-4o", api="responses", client=client)
        agent = _ConcreteAgent(config)

        expected_result = LLMResult(content="responses response")
        with (
            patch.object(agent, "_call_responses", new=AsyncMock(return_value=expected_result)) as mock_resp,
            patch.object(agent, "_call_chat_completions", new=AsyncMock()) as mock_cc,
        ):
            result = await agent._call_llm(_make_messages())

        mock_resp.assert_awaited_once()
        mock_cc.assert_not_awaited()
        assert result.content == "responses response"

    @pytest.mark.asyncio
    async def test_default_config_uses_chat_completions(self):
        """LLMCallConfig with no api= specified defaults to chat_completions."""
        client = _make_client()
        # Explicitly confirm the default is "chat_completions"
        config = LLMCallConfig(model="gpt-4o", client=client)
        assert config.api == "chat_completions"

        agent = _ConcreteAgent(config)

        expected_result = LLMResult(content="default response")
        with (
            patch.object(agent, "_call_chat_completions", new=AsyncMock(return_value=expected_result)) as mock_cc,
            patch.object(agent, "_call_responses", new=AsyncMock()) as mock_resp,
        ):
            result = await agent._call_llm(_make_messages())

        mock_cc.assert_awaited_once()
        mock_resp.assert_not_awaited()
        assert result.content == "default response"

    @pytest.mark.asyncio
    async def test_responses_path_via_real_sdk_mock(self):
        """End-to-end: _call_responses uses client.responses.create, not chat.completions."""
        client = _make_client()

        # Build a minimal response object that _call_responses can parse
        mock_response = MagicMock()
        mock_response.output = []
        mock_response.usage = None
        client.responses.create = AsyncMock(return_value=mock_response)

        config = LLMCallConfig(model="gpt-4o", api="responses", client=client)
        agent = _ConcreteAgent(config)

        await agent._call_llm(_make_messages())

        client.responses.create.assert_awaited_once()
        client.chat.completions.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_responses_path_converts_function_tools_to_chat_style_result(self):
        client = _make_client()

        mock_response = MagicMock()
        mock_response.output = [
            {
                "type": "function_call",
                "call_id": "call_123",
                "name": "finish_conversation",
                "arguments": '{"done": true}',
            }
        ]
        mock_response.usage = None
        client.responses.create = AsyncMock(return_value=mock_response)

        config = LLMCallConfig(model="gpt-4o", api="responses", client=client)
        agent = _ConcreteAgent(config)

        result = await agent._call_llm(
            _make_messages(),
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "finish_conversation",
                        "description": "finish",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )

        assert client.responses.create.await_args is not None
        sent = client.responses.create.await_args.kwargs
        assert sent["tools"] == [
            {
                "type": "function",
                "name": "finish_conversation",
                "description": "finish",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        assert result.tool_calls is not None
        assert result.tool_calls[0].id == "call_123"
        assert result.tool_calls[0].function.name == "finish_conversation"
        assert result.tool_calls[0].function.arguments == '{"done": true}'

    @pytest.mark.asyncio
    async def test_chat_completions_path_via_real_sdk_mock(self):
        """End-to-end: _call_chat_completions uses client.chat.completions.create, not responses."""
        client = _make_client()

        # Build a minimal chat completion response
        client.chat.completions.create = AsyncMock(return_value=_chat_response("hello"))

        config = LLMCallConfig(model="gpt-4o", api="chat_completions", client=client)
        agent = _ConcreteAgent(config)

        await agent._call_llm(_make_messages())

        client.chat.completions.create.assert_awaited_once()
        client.responses.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_chat_completions_retries_once_on_empty_response(self):
        client = _make_client()
        client.chat.completions.create = AsyncMock(
            side_effect=[_chat_response(""), _chat_response("second response")]
        )

        config = LLMCallConfig(model="gpt-4o", api="chat_completions", client=client)
        agent = _ConcreteAgent(config)

        result = await agent._call_llm(_make_messages())

        assert result.content == "second response"
        assert client.chat.completions.create.await_count == 2

    @pytest.mark.asyncio
    async def test_responses_drops_reasoning_and_retries_when_rejected(self, monkeypatch):
        monkeypatch.setattr("evaluatorq.simulation.agents.base.DEFAULT_REASONING_EFFORT", "low")
        client = _make_client()
        ok = MagicMock()
        ok.output = []
        ok.usage = None
        client.responses.create = AsyncMock(
            side_effect=[_bad_request("Unsupported parameter: reasoning"), ok]
        )

        config = LLMCallConfig(model="gpt-4o", api="responses", client=client)
        agent = _ConcreteAgent(config)

        await agent._call_llm(_make_messages())

        assert client.responses.create.await_count == 2
        # The retry must not carry reasoning.
        assert client.responses.create.await_args is not None
        assert "reasoning" not in client.responses.create.await_args.kwargs

    @pytest.mark.asyncio
    async def test_responses_reraises_unrelated_400(self, monkeypatch):
        monkeypatch.setattr("evaluatorq.simulation.agents.base.DEFAULT_REASONING_EFFORT", "low")
        client = _make_client()
        client.responses.create = AsyncMock(side_effect=_bad_request("Invalid tool schema"))

        config = LLMCallConfig(model="gpt-4o", api="responses", client=client)
        agent = _ConcreteAgent(config)

        with pytest.raises(BadRequestError, match="Invalid tool schema"):
            await agent._call_llm(_make_messages())
        # No reasoning-stripped retry — an unrelated 400 is not masked.
        assert client.responses.create.await_count == 1
