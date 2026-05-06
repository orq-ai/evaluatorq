"""Tests for OrqResponsesTarget (simulation target backed by Responses API).

Verifies:
- __call__(messages) returns str text
- send_prompt returns AgentResponse
- previous_response_id threading across calls
- new() returns a fresh instance
- get_usage() accumulates token counts
- Timeout is applied via asyncio.wait_for
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evaluatorq.contracts import AgentResponse, LLMCallConfig
from evaluatorq.simulation.target import OrqResponsesTarget
from evaluatorq.simulation.types import ChatMessage, TokenUsage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client() -> MagicMock:
    """Return a mock AsyncOpenAI client with a stub responses.create."""
    client = MagicMock()
    client.responses = MagicMock()
    client.responses.create = AsyncMock()
    return client


def _make_response(
    text: str = "hello",
    response_id: str = "resp-123",
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> MagicMock:
    """Build a mock Responses API response object."""
    # Usage object
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens

    # Content part with output_text type
    part = MagicMock()
    part.type = "output_text"
    part.text = text

    # Message output item
    msg_item = MagicMock()
    msg_item.type = "message"
    msg_item.content = [part]

    response = MagicMock()
    response.id = response_id
    response.usage = usage
    response.output = [msg_item]
    return response


def _make_target(
    client: MagicMock | None = None,
    instructions: str | None = None,
    timeout_ms: int = 30_000,
) -> OrqResponsesTarget:
    """Create an OrqResponsesTarget with an injected mock client."""
    if client is None:
        client = _make_client()
    config = LLMCallConfig(model="gpt-4o", timeout_ms=timeout_ms)
    return OrqResponsesTarget(config, instructions=instructions, client=client)


def _make_messages(content: str = "hi") -> list[ChatMessage]:
    return [ChatMessage(role="user", content=content)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOrqResponsesTargetCall:
    @pytest.mark.asyncio
    async def test_call_returns_str(self):
        """__call__(messages) must return a plain str, not AgentResponse."""
        client = _make_client()
        client.responses.create = AsyncMock(return_value=_make_response(text="world"))
        target = _make_target(client=client)

        result = await target(_make_messages())

        assert isinstance(result, str)
        assert result == "world"

    @pytest.mark.asyncio
    async def test_call_passes_full_message_list_as_input(self):
        """__call__ converts the message list and passes it as input= to the SDK."""
        client = _make_client()
        client.responses.create = AsyncMock(return_value=_make_response())
        target = _make_target(client=client)

        messages = [
            ChatMessage(role="user", content="turn 1"),
            ChatMessage(role="assistant", content="reply"),
            ChatMessage(role="user", content="turn 2"),
        ]
        await target(messages)

        call_kwargs = client.responses.create.call_args.kwargs
        assert call_kwargs["input"] == [
            {"role": "user", "content": "turn 1"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "turn 2"},
        ]

    @pytest.mark.asyncio
    async def test_call_raises_error_on_no_output_items(self):
        """__call__ raises RuntimeError when response has no output items."""
        client = _make_client()
        empty_response = MagicMock()
        empty_response.id = "resp-empty"
        empty_response.usage = None
        empty_response.output = []
        client.responses.create = AsyncMock(return_value=empty_response)
        target = _make_target(client=client)

        with pytest.raises(RuntimeError, match="response contained no extractable output items"):
            await target(_make_messages())


class TestOrqResponsesTargetSendPrompt:
    @pytest.mark.asyncio
    async def test_send_prompt_returns_agent_response(self):
        """send_prompt must return an AgentResponse instance."""
        client = _make_client()
        client.responses.create = AsyncMock(return_value=_make_response(text="I'm fine"))
        target = _make_target(client=client)

        result = await target.send_prompt("hello")

        assert isinstance(result, AgentResponse)
        assert result.text == "I'm fine"

    @pytest.mark.asyncio
    async def test_send_prompt_passes_string_as_input(self):
        """send_prompt passes the prompt string directly as input=."""
        client = _make_client()
        client.responses.create = AsyncMock(return_value=_make_response())
        target = _make_target(client=client)

        await target.send_prompt("attack prompt")

        call_kwargs = client.responses.create.call_args.kwargs
        assert call_kwargs["input"] == "attack prompt"


class TestOrqResponsesTargetPreviousResponseId:
    @pytest.mark.asyncio
    async def test_previous_response_id_is_none_on_fresh_instance(self):
        """A fresh OrqResponsesTarget has _previous_response_id == None."""
        target = _make_target()
        assert target._previous_response_id is None

    @pytest.mark.asyncio
    async def test_previous_response_id_set_after_first_call(self):
        """After a call, _previous_response_id is set from response.id."""
        client = _make_client()
        client.responses.create = AsyncMock(return_value=_make_response(response_id="resp-abc"))
        target = _make_target(client=client)

        assert target._previous_response_id is None
        await target.send_prompt("first turn")
        assert target._previous_response_id == "resp-abc"

    @pytest.mark.asyncio
    async def test_second_call_sends_previous_response_id(self):
        """Second call must include previous_response_id= in SDK kwargs."""
        client = _make_client()
        client.responses.create = AsyncMock(
            side_effect=[
                _make_response(response_id="resp-first"),
                _make_response(response_id="resp-second"),
            ]
        )
        target = _make_target(client=client)

        await target.send_prompt("turn 1")
        await target.send_prompt("turn 2")

        second_call_kwargs = client.responses.create.call_args_list[1].kwargs
        assert second_call_kwargs.get("previous_response_id") == "resp-first"

    @pytest.mark.asyncio
    async def test_first_call_does_not_send_previous_response_id(self):
        """First call must NOT include previous_response_id= in SDK kwargs."""
        client = _make_client()
        client.responses.create = AsyncMock(return_value=_make_response())
        target = _make_target(client=client)

        await target.send_prompt("turn 1")

        first_call_kwargs = client.responses.create.call_args.kwargs
        assert "previous_response_id" not in first_call_kwargs


class TestOrqResponsesTargetNew:
    def test_new_returns_fresh_instance(self):
        """new() returns a new OrqResponsesTarget, not the same object."""
        target = _make_target()
        fresh = target.new()
        assert fresh is not target
        assert isinstance(fresh, OrqResponsesTarget)

    @pytest.mark.asyncio
    async def test_new_fresh_instance_has_previous_response_id_none(self):
        """new() returns an instance with _previous_response_id == None."""
        client = _make_client()
        client.responses.create = AsyncMock(return_value=_make_response(response_id="resp-xyz"))
        target = _make_target(client=client)

        # Simulate one call to set state on original
        await target.send_prompt("setup")

        fresh = target.new()
        assert fresh._previous_response_id is None

    def test_new_propagates_injected_client(self):
        """new() propagates injected (non-owned) client to the fresh instance."""
        client = _make_client()
        target = _make_target(client=client)

        # Confirm the client is not owned (it was injected)
        assert not target._client_owned

        fresh = target.new()
        assert fresh._client is client

    def test_new_preserves_config(self):
        """new() preserves the config from the original instance."""
        client = _make_client()
        config = LLMCallConfig(model="gpt-4o-special", timeout_ms=60_000)
        target = OrqResponsesTarget(config, client=client)

        fresh = target.new()
        assert fresh.config.model == "gpt-4o-special"
        assert fresh.config.timeout_ms == 60_000

    def test_new_preserves_instructions(self):
        """new() preserves instructions from the original instance."""
        client = _make_client()
        target = _make_target(client=client, instructions="Be concise.")
        fresh = target.new()
        assert fresh.instructions == "Be concise."


class TestOrqResponsesTargetGetUsage:
    @pytest.mark.asyncio
    async def test_get_usage_returns_zeros_on_fresh_instance(self):
        """A fresh target returns all-zero TokenUsage."""
        target = _make_target()
        usage = target.get_usage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    @pytest.mark.asyncio
    async def test_get_usage_accumulates_token_counts(self):
        """get_usage() returns non-zero counts after one or more calls."""
        client = _make_client()
        client.responses.create = AsyncMock(
            return_value=_make_response(input_tokens=10, output_tokens=5)
        )
        target = _make_target(client=client)

        await target.send_prompt("hello")

        usage = target.get_usage()
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 5
        assert usage.total_tokens == 15

    @pytest.mark.asyncio
    async def test_get_usage_accumulates_across_multiple_calls(self):
        """Token counts accumulate across multiple invocations."""
        client = _make_client()
        client.responses.create = AsyncMock(
            side_effect=[
                _make_response(input_tokens=10, output_tokens=5),
                _make_response(input_tokens=20, output_tokens=8),
            ]
        )
        target = _make_target(client=client)

        await target.send_prompt("turn 1")
        await target.send_prompt("turn 2")

        usage = target.get_usage()
        assert usage.prompt_tokens == 30
        assert usage.completion_tokens == 13
        assert usage.total_tokens == 43


class TestOrqResponsesTargetTimeout:
    @pytest.mark.asyncio
    async def test_timeout_is_applied_via_wait_for(self):
        """config.timeout_ms is converted to seconds and passed to asyncio.wait_for."""
        import asyncio

        client = _make_client()
        client.responses.create = AsyncMock(return_value=_make_response())
        config = LLMCallConfig(model="gpt-4o", timeout_ms=5_000)
        target = OrqResponsesTarget(config, client=client)

        with patch("evaluatorq.simulation.target.asyncio.wait_for", wraps=asyncio.wait_for) as mock_wait:
            await target.send_prompt("hi")

        mock_wait.assert_awaited_once()
        _, kwargs = mock_wait.call_args
        assert kwargs.get("timeout") == pytest.approx(5.0)

    @pytest.mark.asyncio
    async def test_timeout_exceeded_raises(self):
        """RuntimeError (wrapping asyncio.TimeoutError) is raised when SDK call exceeds timeout_ms."""
        import asyncio

        async def _slow(*args: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(10)

        client = _make_client()
        client.responses.create = _slow

        config = LLMCallConfig(model="gpt-4o", timeout_ms=10)  # 10ms — will expire
        target = OrqResponsesTarget(config, client=client)

        with pytest.raises(RuntimeError, match="timed out"):
            await target.send_prompt("hi")


class TestOrqResponsesTargetInstructions:
    @pytest.mark.asyncio
    async def test_instructions_passed_to_sdk_when_set(self):
        """When instructions are set, they're included in the SDK call."""
        client = _make_client()
        client.responses.create = AsyncMock(return_value=_make_response())
        target = _make_target(client=client, instructions="Be helpful.")

        await target.send_prompt("prompt")

        call_kwargs = client.responses.create.call_args.kwargs
        assert call_kwargs.get("instructions") == "Be helpful."

    @pytest.mark.asyncio
    async def test_instructions_omitted_when_none(self):
        """When instructions is None, it's not included in the SDK call kwargs."""
        client = _make_client()
        client.responses.create = AsyncMock(return_value=_make_response())
        target = _make_target(client=client, instructions=None)

        await target.send_prompt("prompt")

        call_kwargs = client.responses.create.call_args.kwargs
        assert "instructions" not in call_kwargs
