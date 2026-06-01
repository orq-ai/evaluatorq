"""Tests for OrqResponsesTarget (stateless target backed by the Responses API).

After RES-877 Task 9 the target is fully stateless:
- respond(messages) returns AgentResponse; the message list is sent verbatim
- send_prompt shim removed; respond is the sole response method
- no previous_response_id threading, no get_usage accumulation
- new() returns a fresh instance; client lifecycle preserved
- timeout applied via asyncio.wait_for; retry wraps the SDK call
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evaluatorq.contracts import AgentResponse, LLMCallConfig, Message
from evaluatorq.simulation.target import OrqResponsesTarget



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
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens

    part = MagicMock()
    part.type = "output_text"
    part.text = text

    msg_item = MagicMock()
    msg_item.type = "message"
    msg_item.content = [part]

    response = MagicMock()
    response.id = response_id
    response.usage = usage
    response.output = [msg_item]
    return response


def _make_dict_response(
    text: str = "hello",
    response_id: str = "resp-dict",
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> dict[str, Any]:
    return {
        "id": response_id,
        "model": "gpt-4o",
        "status": "completed",
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
    }


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


def _make_messages(content: str = "hi") -> list[Message]:
    return [Message(role="user", content=content)]


# ---------------------------------------------------------------------------
# respond (sole response method; send_prompt shim removed in RES-877)
# ---------------------------------------------------------------------------


class TestOrqResponsesTargetRespond:
    @pytest.mark.asyncio
    async def test_respond_returns_agent_response(self):
        client = _make_client()
        client.responses.create = AsyncMock(return_value=_make_response(text="world"))
        target = _make_target(client=client)

        result = await target.respond(_make_messages())

        assert isinstance(result, AgentResponse)
        assert result.text == "world"

    @pytest.mark.asyncio
    async def test_respond_passes_full_message_list_as_input(self):
        client = _make_client()
        client.responses.create = AsyncMock(return_value=_make_response())
        target = _make_target(client=client)

        messages = [
            Message(role="user", content="turn 1"),
            Message(role="assistant", content="reply"),
            Message(role="user", content="turn 2"),
        ]
        await target.respond(messages)

        call_kwargs = client.responses.create.call_args.kwargs
        assert call_kwargs["input"] == [
            {"role": "user", "content": "turn 1"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "turn 2"},
        ]

    @pytest.mark.asyncio
    async def test_respond_serializes_tool_calls_and_assistant_null_content(self):
        """_messages_to_input must preserve tool-call structure and assistant content=None.

        Assistant messages with tool_calls require content: null per OpenAI's
        spec; tool messages carry tool_call_id + name. respond passes the whole
        transcript, so these must survive into the SDK `input` payload.
        """
        from evaluatorq.contracts import FunctionCall, StrategyToolCall

        client = _make_client()
        client.responses.create = AsyncMock(return_value=_make_response())
        target = _make_target(client=client)

        tool_call = StrategyToolCall(id="c1", function=FunctionCall(name="f", arguments="{}"))
        await target.respond(
            [
                Message(role="user", content="hi"),
                Message(role="assistant", content=None, tool_calls=[tool_call]),
                Message(role="tool", content="result", tool_call_id="c1", name="f"),
            ]
        )

        sent = client.responses.create.call_args.kwargs["input"]
        assert sent[0] == {"role": "user", "content": "hi"}
        # Assistant with tool_calls keeps content=None (not coerced to "").
        assert sent[1]["role"] == "assistant"
        assert sent[1]["content"] is None
        assert sent[1]["tool_calls"][0]["function"] == {"name": "f", "arguments": "{}"}
        # Tool message carries tool_call_id + name.
        assert sent[2] == {
            "role": "tool",
            "content": "result",
            "tool_call_id": "c1",
            "name": "f",
        }

    @pytest.mark.asyncio
    async def test_respond_is_stateless_no_previous_response_id(self):
        """Consecutive respond calls never thread previous_response_id."""
        client = _make_client()
        client.responses.create = AsyncMock(
            side_effect=[_make_response(response_id="r1"), _make_response(response_id="r2")]
        )
        target = _make_target(client=client)

        await target.respond(_make_messages("turn 1"))
        await target.respond(_make_messages("turn 2"))

        for call in client.responses.create.call_args_list:
            assert "previous_response_id" not in call.kwargs

    @pytest.mark.asyncio
    async def test_respond_raises_error_on_no_output_items(self):
        client = _make_client()
        empty_response = MagicMock()
        empty_response.id = "resp-empty"
        empty_response.usage = None
        empty_response.output = []
        client.responses.create = AsyncMock(return_value=empty_response)
        target = _make_target(client=client)

        with pytest.raises(RuntimeError, match="response contained no extractable output items"):
            await target.respond(_make_messages())

    @pytest.mark.asyncio
    async def test_respond_with_single_user_message(self):
        client = _make_client()
        response = _make_response(text="I'm fine")
        response.model = "gpt-4o"
        client.responses.create = AsyncMock(return_value=response)
        target = _make_target(client=client)

        result = await target.respond([Message(role="user", content="hello")])

        assert isinstance(result, AgentResponse)
        assert result.text == "I'm fine"
        assert result.usage is not None
        assert result.model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_respond_wraps_single_user_message(self):
        client = _make_client()
        client.responses.create = AsyncMock(return_value=_make_response())
        target = _make_target(client=client)

        await target.respond([Message(role="user", content="attack prompt")])

        call_kwargs = client.responses.create.call_args.kwargs
        assert call_kwargs["input"] == [{"role": "user", "content": "attack prompt"}]
        assert "previous_response_id" not in call_kwargs


# ---------------------------------------------------------------------------
# new()
# ---------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_dict_response_usage_is_returned(self):
        """respond() with a dict-shaped response returns correct usage."""
        client = _make_client()
        client.responses.create = AsyncMock(
            return_value=_make_dict_response(text="one", response_id="resp-dict-1")
        )
        target = _make_target(client=client)

        result = await target.respond(_make_messages("turn 1"))

        assert result.text == "one"
        assert result.usage is not None
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5


class TestOrqResponsesTargetNew:
    def test_new_returns_fresh_instance(self):
        target = _make_target()
        fresh = target.new()
        assert fresh is not target
        assert isinstance(fresh, OrqResponsesTarget)

    def test_new_memory_entity_id_none_when_unset(self):
        target = _make_target()
        assert target.new().memory_entity_id is None

    def test_new_propagates_injected_client(self):
        client = _make_client()
        target = _make_target(client=client)
        assert not target._client_owned
        assert target.new()._client is client

    def test_new_preserves_config(self):
        client = _make_client()
        config = LLMCallConfig(model="gpt-4o-special", timeout_ms=60_000)
        target = OrqResponsesTarget(config, client=client)

        fresh = target.new()
        assert fresh.config.model == "gpt-4o-special"
        assert fresh.config.timeout_ms == 60_000

    def test_new_preserves_instructions(self):
        client = _make_client()
        target = _make_target(client=client, instructions="Be concise.")
        assert target.new().instructions == "Be concise."

    def test_new_mints_fresh_memory_entity_id_when_set(self):
        import uuid

        client = _make_client()
        target = OrqResponsesTarget(
            LLMCallConfig(model="gpt-4o"),
            memory_entity_id="original-uuid-abc",
            client=client,
        )

        fresh = target.new()

        assert fresh.memory_entity_id is not None
        assert fresh.memory_entity_id != target.memory_entity_id
        parsed = uuid.UUID(fresh.memory_entity_id, version=4)
        assert parsed.version == 4

    def test_new_preserves_tools_parameter(self):
        tools = [{"type": "function", "function": {"name": "foo"}}]
        client = _make_client()
        target = OrqResponsesTarget(LLMCallConfig(model="gpt-4o"), tools=tools, client=client)
        assert target.new().tools == target.tools

    def test_new_does_not_share_self_owned_client(self, monkeypatch):
        monkeypatch.setenv("ORQ_API_KEY", "orq-test-key")

        captured_clients: list[Any] = []

        def fake_async_openai(**kwargs):
            mock = MagicMock()
            captured_clients.append(mock)
            return mock

        with patch("openai.AsyncOpenAI", side_effect=fake_async_openai):
            target = OrqResponsesTarget(LLMCallConfig(model="gpt-4o"))
            assert target._client_owned
            fresh = target.new()

        assert fresh._client is not target._client


# ---------------------------------------------------------------------------
# instructions / tools forwarding
# ---------------------------------------------------------------------------


class TestOrqResponsesTargetInstructions:
    @pytest.mark.asyncio
    async def test_instructions_passed_to_sdk_when_set(self):
        client = _make_client()
        client.responses.create = AsyncMock(return_value=_make_response())
        target = _make_target(client=client, instructions="Be helpful.")

        await target.respond(_make_messages())

        assert client.responses.create.call_args.kwargs.get("instructions") == "Be helpful."

    @pytest.mark.asyncio
    async def test_instructions_omitted_when_none(self):
        client = _make_client()
        client.responses.create = AsyncMock(return_value=_make_response())
        target = _make_target(client=client, instructions=None)

        await target.respond(_make_messages())

        assert "instructions" not in client.responses.create.call_args.kwargs


class TestOrqResponsesTargetTools:
    @pytest.mark.asyncio
    async def test_tools_forwarded_to_sdk_when_set(self):
        client = _make_client()
        client.responses.create = AsyncMock(return_value=_make_response())
        tool_spec = [{"type": "function", "name": "lookup", "parameters": {}}]
        target = OrqResponsesTarget(LLMCallConfig(model="gpt-4o"), tools=tool_spec, client=client)

        await target.respond(_make_messages())

        assert client.responses.create.call_args.kwargs.get("tools") == tool_spec

    @pytest.mark.asyncio
    async def test_tools_omitted_when_none_or_empty(self):
        client = _make_client()
        client.responses.create = AsyncMock(return_value=_make_response())
        config = LLMCallConfig(model="gpt-4o")

        target_none = OrqResponsesTarget(config, tools=None, client=client)
        await target_none.respond(_make_messages())
        assert "tools" not in client.responses.create.call_args.kwargs

        client.responses.create.reset_mock()
        target_empty = OrqResponsesTarget(config, tools=[], client=client)
        await target_empty.respond(_make_messages())
        assert "tools" not in client.responses.create.call_args.kwargs


# ---------------------------------------------------------------------------
# timeout
# ---------------------------------------------------------------------------


class TestOrqResponsesTargetTimeout:
    @pytest.mark.asyncio
    async def test_timeout_is_applied_via_wait_for(self):
        import asyncio

        client = _make_client()
        client.responses.create = AsyncMock(return_value=_make_response())
        config = LLMCallConfig(model="gpt-4o", timeout_ms=5_000)
        target = OrqResponsesTarget(config, client=client)

        with patch(
            "evaluatorq.simulation.target.asyncio.wait_for", wraps=asyncio.wait_for
        ) as mock_wait:
            await target.respond(_make_messages())

        mock_wait.assert_awaited_once()
        _, kwargs = mock_wait.call_args
        assert kwargs.get("timeout") == pytest.approx(5.0)

    @pytest.mark.asyncio
    async def test_timeout_exceeded_raises(self):
        import asyncio

        async def _slow(*args: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(10)

        client = _make_client()
        client.responses.create = _slow

        config = LLMCallConfig(model="gpt-4o", timeout_ms=10)  # 10ms — will expire
        target = OrqResponsesTarget(config, client=client)

        with pytest.raises(RuntimeError, match="timed out"):
            await target.respond(_make_messages())


# ---------------------------------------------------------------------------
# retry
# ---------------------------------------------------------------------------


class TestOrqResponsesTargetRetry:
    @pytest.mark.asyncio
    async def test_retries_on_rate_limit_then_succeeds(self, monkeypatch):
        from openai import APIStatusError

        monkeypatch.setattr(
            "evaluatorq.common.retry.asyncio.sleep",
            AsyncMock(return_value=None),
        )

        client = _make_client()
        rate_limit = APIStatusError(
            "rate limited",
            response=MagicMock(status_code=429, headers={}),
            body=None,
        )
        good = _make_response(response_id="resp-after-retry")
        client.responses.create = AsyncMock(side_effect=[rate_limit, good])
        target = _make_target(client=client)

        result = await target.respond(_make_messages())

        assert client.responses.create.await_count == 2
        assert isinstance(result, AgentResponse)

    @pytest.mark.asyncio
    async def test_does_not_retry_on_non_retryable_error(self, monkeypatch):
        from openai import APIStatusError

        sleep_mock = AsyncMock(return_value=None)
        monkeypatch.setattr("evaluatorq.common.retry.asyncio.sleep", sleep_mock)

        client = _make_client()
        bad_request = APIStatusError(
            "bad request",
            response=MagicMock(status_code=400, headers={}),
            body=None,
        )
        client.responses.create = AsyncMock(side_effect=bad_request)
        target = _make_target(client=client)

        with pytest.raises(APIStatusError):
            await target.respond(_make_messages())

        assert client.responses.create.await_count == 1
        assert sleep_mock.await_count == 0


# ---------------------------------------------------------------------------
# client lifecycle
# ---------------------------------------------------------------------------


class TestOrqResponsesTargetClose:
    @pytest.mark.asyncio
    async def test_close_closes_owned_client(self, monkeypatch):
        from evaluatorq.simulation import target as target_mod

        owned_client = MagicMock()
        owned_client.close = AsyncMock()
        monkeypatch.setattr(
            target_mod, "build_simulation_client",
            lambda _client: (owned_client, True),  # pyright: ignore[reportUnknownLambdaType]
        )
        t = OrqResponsesTarget(LLMCallConfig(model="m", api="responses"))

        await t.close()

        owned_client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_does_not_close_injected_client(self):
        injected = _make_client()
        injected.close = AsyncMock()
        target = _make_target(client=injected)

        await target.close()

        injected.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self, monkeypatch):
        from evaluatorq.simulation import target as target_mod

        owned_client = MagicMock()
        owned_client.close = AsyncMock()
        monkeypatch.setattr(
            target_mod, "build_simulation_client",
            lambda _client: (owned_client, True),  # pyright: ignore[reportUnknownLambdaType]
        )
        t = OrqResponsesTarget(LLMCallConfig(model="m", api="responses"))

        await t.close()
        await t.close()

        owned_client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_async_context_manager_closes_owned_client(self, monkeypatch):
        from evaluatorq.simulation import target as target_mod

        owned_client = MagicMock()
        owned_client.close = AsyncMock()
        monkeypatch.setattr(
            target_mod, "build_simulation_client",
            lambda _client: (owned_client, True),  # pyright: ignore[reportUnknownLambdaType]
        )

        async with OrqResponsesTarget(LLMCallConfig(model="m", api="responses")) as t:
            assert t._client_owned is True

        owned_client.close.assert_awaited_once()
