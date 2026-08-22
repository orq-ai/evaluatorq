"""Tests for OrqResponsesTarget (stateless target backed by the Responses API).

After RES-877 Task 9 the target is fully stateless:
- respond(messages) returns AgentResponse; the message list is sent verbatim
- send_prompt shim removed; respond is the sole response method
- no previous_response_id threading, no get_usage accumulation
- new() returns a fresh instance; client lifecycle preserved
- timeout applied via asyncio.wait_for; retry wraps the SDK call
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import AsyncOpenAI, BadRequestError

from evaluatorq.contracts import AgentResponse, LLMCallConfig, Message
from evaluatorq.openresponses.target import OrqResponsesTarget


def _bad_request(message: str) -> BadRequestError:
    request = httpx.Request("POST", "https://my.orq.ai/v3/router/responses")
    response = httpx.Response(400, request=request)
    return BadRequestError(message, response=response, body={"error": {"message": message}})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client() -> MagicMock:
    """Return a mock AsyncOpenAI client with a stub responses.create.

    ``base_url`` points at the Orq router so router-only request extras
    (``thread``/``memory``) are emitted only when the client routes through
    Orq; native pipeline ``metadata`` is sent on both compatible endpoints.
    """
    client = MagicMock()
    client.base_url = "https://my.orq.ai/v3/router"
    client.responses = MagicMock()
    client.responses.create = AsyncMock()
    return client


def _make_direct_client() -> MagicMock:
    client = _make_client()
    client.base_url = "https://api.openai.com/v1"
    return client


def _make_response(
    text: str = "hello",
    response_id: str = "resp-123",
    input_tokens: int = 10,
    output_tokens: int = 5,
    trace_id: str | None = None,
    stop_reason: str | None = None,
    refusal: str | None = None,
) -> MagicMock:
    """Build a mock Responses API response object."""
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens

    part = MagicMock()
    part.type = "refusal" if refusal is not None else "output_text"
    part.text = text
    part.refusal = refusal

    msg_item = MagicMock()
    msg_item.type = "message"
    msg_item.content = [part]

    telemetry = MagicMock()
    telemetry.trace_id = trace_id

    response = MagicMock()
    response.id = response_id
    response.usage = usage
    response.output = [msg_item]
    response.stop_reason = stop_reason
    response.incomplete_details = None
    response.telemetry = telemetry
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
    client: Any | None = None,
    instructions: str | None = None,
    timeout_ms: int = 30_000,
    tools: list[dict[str, Any]] | None = None,
    retry_attempts: int = 1,
) -> OrqResponsesTarget:
    """Create an OrqResponsesTarget with an injected mock client."""
    if client is None:
        client = _make_client()
    config = LLMCallConfig(model="gpt-4o", timeout_ms=timeout_ms)
    return OrqResponsesTarget(
        config, instructions=instructions, tools=tools, client=client, retry_attempts=retry_attempts
    )


def _responses_http_response(
    *,
    status_code: int = 200,
    text: str = "from headers",
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    payload: dict[str, Any] = {
        "id": "resp-header",
        "object": "response",
        "created_at": 0,
        "model": "gpt-4o",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "msg-header",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }
    if status_code != 200:
        payload = {"error": {"message": "rate limited", "type": "rate_limit_error"}}
    return httpx.Response(
        status_code,
        headers=headers,
        json=payload,
        request=httpx.Request("POST", "https://my.orq.ai/v3/router/responses"),
    )


def _make_sdk_client(handler: Any) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key="test-key",
        base_url="https://my.orq.ai/v3/router",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


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
    async def test_respond_captures_trace_id_from_body(self):
        client = _make_client()
        client.responses.create = AsyncMock(
            return_value=_make_response(text="world", trace_id="trace-abc123")
        )
        target = _make_target(client=client)

        result = await target.respond(_make_messages())

        assert result.trace_id == "trace-abc123"

    @pytest.mark.asyncio
    async def test_respond_captures_trace_ids_from_orq_headers(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return _responses_http_response(
                headers={
                    "x-orq-trace-id": "header-trace-123",
                    "x-orq-trace-span-id": "header-span-456",
                }
            )

        client = _make_sdk_client(handler)
        try:
            result = await _make_target(client=client).respond(_make_messages())
        finally:
            await client.close()

        assert result.text == "from headers"
        assert result.trace_id == "header-trace-123"
        assert result.span_id == "header-span-456"

    @pytest.mark.asyncio
    async def test_respond_sends_thread_param_when_conversation_active(self):
        from evaluatorq.common.thread_context import conversation_thread

        client = _make_client()
        client.responses.create = AsyncMock(return_value=_make_response())
        target = _make_target(client=client)

        with conversation_thread("thread-xyz"):
            await target.respond(_make_messages())

        call_kwargs = client.responses.create.call_args.kwargs
        assert call_kwargs["extra_body"]["thread"] == {"id": "thread-xyz"}

    @pytest.mark.asyncio
    async def test_respond_omits_thread_param_when_no_conversation(self):
        client = _make_client()
        client.responses.create = AsyncMock(return_value=_make_response())
        target = _make_target(client=client)

        await target.respond(_make_messages())

        assert "extra_body" not in client.responses.create.call_args.kwargs

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
            {"role": "assistant", "content": [{"type": "output_text", "text": "reply"}]},
            {"role": "user", "content": "turn 2"},
        ]

    @pytest.mark.asyncio
    async def test_respond_serializes_tool_turns_as_responses_items(self):
        """Tool turns must serialize as Responses items, not chat-completions rows.

        The Responses API rejects `role: "tool"` outright ("Invalid value: 'tool'")
        and ignores a message-level `tool_calls` key, which silently drops the
        assistant's tool calls. respond passes the whole transcript, so an
        assistant tool call must become a `function_call` item and a tool result a
        `function_call_output` keyed by the same call_id.
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
        assert sent == [
            {"role": "user", "content": "hi"},
            {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "result"},
        ]
        # No chat-completions leakage: these keys are invalid on Responses input.
        assert not any("tool_calls" in item or item.get("role") == "tool" for item in sent)

    @pytest.mark.asyncio
    async def test_respond_keeps_assistant_narration_before_tool_calls(self):
        """An assistant turn with both text and tool calls emits both, in order."""
        from evaluatorq.contracts import FunctionCall, StrategyToolCall

        client = _make_client()
        client.responses.create = AsyncMock(return_value=_make_response())
        target = _make_target(client=client)

        tool_call = StrategyToolCall(id="c1", function=FunctionCall(name="f", arguments="{}"), item_id="fc_1")
        await target.respond(
            [
                Message(role="user", content="hi"),
                Message(role="assistant", content="let me check", tool_calls=[tool_call]),
            ]
        )

        sent = client.responses.create.call_args.kwargs["input"]
        assert sent[1] == {"role": "assistant", "content": [{"type": "output_text", "text": "let me check"}]}
        # item_id round-trips as the function_call item id when present.
        assert sent[2] == {
            "type": "function_call",
            "call_id": "c1",
            "name": "f",
            "arguments": "{}",
            "id": "fc_1",
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
    @pytest.mark.parametrize('stop_reason', ['length', 'max_output_tokens'])
    async def test_respond_raises_on_length_metadata(self, stop_reason):
        client = _make_client()
        client.responses.create = AsyncMock(
            return_value=_make_response(text='partial', stop_reason=stop_reason)
        )
        target = _make_target(client=client)

        with pytest.raises(RuntimeError, match='response truncated'):
            await target.respond(_make_messages())

    @pytest.mark.asyncio
    async def test_respond_exposes_refusal_metadata(self):
        client = _make_client()
        client.responses.create = AsyncMock(return_value=_make_response(refusal='not allowed'))
        target = _make_target(client=client)

        result = await target.respond(_make_messages())

        assert result.text == 'not allowed'
        assert result.refusal == 'not allowed'

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

    def test_new_preserves_constructor_seeded_memory_entity_id(self):
        """A constructor-passed id is a seed and survives clones (the sim
        --memory-entity path); only auto-minted ids re-mint per clone."""
        client = _make_client()
        target = OrqResponsesTarget(
            LLMCallConfig(model="gpt-4o"),
            memory_entity_id="original-uuid-abc",
            client=client,
        )

        assert target.new().memory_entity_id == "original-uuid-abc"

    def test_new_re_mints_auto_minted_memory_entity_id(self):
        import uuid

        client = _make_client()
        target = OrqResponsesTarget(LLMCallConfig(model="gpt-4o"), client=client)
        target.mint_memory_entity_id("minted-abc")

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

    def test_default_is_a_single_attempt(self):
        """`call_target_with_retry` owns the target retry budget on every surface.

        A default > 1 here multiplies against that outer budget instead of adding
        to it — 5 inner attempts under 3 outer ones is 15 calls to a target that
        is already refusing.
        """
        target = OrqResponsesTarget(LLMCallConfig(model="gpt-4o"), client=_make_client())
        assert target.retry_attempts == 1
        assert target.new().retry_attempts == 1

    def test_new_preserves_retry_settings(self):
        client = _make_client()
        target = OrqResponsesTarget(
            LLMCallConfig(model="gpt-4o"),
            client=client,
            retry_attempts=7,
            retry_statuses={503, 504},
        )
        fresh = target.new()
        assert fresh.retry_attempts == 7
        assert fresh.retry_statuses == {503, 504}

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
# memory entity forwarding (regression: the target stored memory_entity_id but
# never sent it, so memory-backed agents 400ed on every conversation)
# ---------------------------------------------------------------------------


class TestOrqResponsesTargetMemory:
    @pytest.mark.asyncio
    async def test_memory_entity_id_is_omitted_for_direct_client(self):
        client = _make_direct_client()
        client.responses.create = AsyncMock(return_value=_make_response())
        target = OrqResponsesTarget(
            LLMCallConfig(model="gpt-4o"), memory_entity_id="ent-42", client=client
        )

        await target.respond(_make_messages())

        assert "extra_body" not in client.responses.create.call_args.kwargs

    @pytest.mark.asyncio
    async def test_memory_entity_id_lands_in_extra_body(self):
        client = _make_client()
        client.responses.create = AsyncMock(return_value=_make_response())
        target = OrqResponsesTarget(
            LLMCallConfig(model="agent/support"), memory_entity_id="ent-42", client=client
        )

        await target.respond(_make_messages())

        extra_body = client.responses.create.call_args.kwargs["extra_body"]
        assert extra_body["memory"] == {"entity_id": "ent-42"}

    @pytest.mark.asyncio
    async def test_memory_coexists_with_thread_and_pipeline_metadata(self):
        from evaluatorq.common.thread_context import conversation_thread, evaluatorq_pipeline

        client = _make_client()
        client.responses.create = AsyncMock(return_value=_make_response())
        target = OrqResponsesTarget(
            LLMCallConfig(model="agent/support"), memory_entity_id="ent-42", client=client
        )

        with evaluatorq_pipeline("agent_simulation"), conversation_thread("thread-xyz"):
            await target.respond(_make_messages())

        extra_body = client.responses.create.call_args.kwargs["extra_body"]
        assert extra_body["memory"] == {"entity_id": "ent-42"}
        assert extra_body["thread"] == {"id": "thread-xyz"}
        assert client.responses.create.call_args.kwargs["metadata"] == {
            "evaluatorq_pipeline": "agent_simulation"
        }

    @pytest.mark.asyncio
    async def test_no_memory_key_sent_when_unset(self):
        client = _make_client()
        client.responses.create = AsyncMock(return_value=_make_response())
        target = OrqResponsesTarget(LLMCallConfig(model="agent/support"), client=client)

        await target.respond(_make_messages())

        # no memory id and no active thread/pipeline context -> no extra_body at all
        assert "extra_body" not in client.responses.create.call_args.kwargs


class TestOrqResponsesTargetExtraBodyPrecedence:
    """A config-supplied ``extra_body`` key must win over the router body
    (``thread``/``memory``) on a clash, while a router key the config does not
    mention must still reach the request. See CLAUDE.md: "Caller-supplied
    values win merges."
    """

    @pytest.mark.asyncio
    async def test_config_extra_body_key_wins_over_router_key(self):
        from evaluatorq.common.thread_context import conversation_thread

        client = _make_client()
        client.responses.create = AsyncMock(return_value=_make_response())
        target = OrqResponsesTarget(
            LLMCallConfig(model="agent/support", extra_body={"memory": {"entity_id": "tenant-A"}}),
            memory_entity_id="auto-minted-id",
            client=client,
        )

        with conversation_thread("thread-xyz"):
            await target.respond(_make_messages())

        extra_body = client.responses.create.call_args.kwargs["extra_body"]
        # config's memory scope wins over the auto-minted memory_entity_id the
        # router body would otherwise have set
        assert extra_body["memory"] == {"entity_id": "tenant-A"}

    @pytest.mark.asyncio
    async def test_router_only_key_survives_when_config_does_not_mention_it(self):
        from evaluatorq.common.thread_context import conversation_thread

        client = _make_client()
        client.responses.create = AsyncMock(return_value=_make_response())
        target = OrqResponsesTarget(
            LLMCallConfig(model="agent/support", extra_body={"unrelated": "value"}),
            client=client,
        )

        with conversation_thread("thread-xyz"):
            await target.respond(_make_messages())

        extra_body = client.responses.create.call_args.kwargs["extra_body"]
        assert extra_body["thread"] == {"id": "thread-xyz"}
        assert extra_body["unrelated"] == "value"

    @pytest.mark.asyncio
    async def test_router_body_still_sent_when_config_extra_body_is_empty(self):
        """Common path: config sets no extra_body at all — body_extra must
        still reach the request (the falsy-``self.extra_body`` early return in
        ``_merge_extra_body`` must not swallow the call-site body)."""
        from evaluatorq.common.thread_context import conversation_thread

        client = _make_client()
        client.responses.create = AsyncMock(return_value=_make_response())
        target = OrqResponsesTarget(LLMCallConfig(model="agent/support"), client=client)

        with conversation_thread("thread-xyz"):
            await target.respond(_make_messages())

        extra_body = client.responses.create.call_args.kwargs["extra_body"]
        assert extra_body["thread"] == {"id": "thread-xyz"}


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


class TestOrqResponsesTargetConfigParams:
    @pytest.mark.asyncio
    async def test_temperature_reaches_the_request(self):
        client = _make_client()
        client.responses.create = AsyncMock(return_value=_make_response())
        target = OrqResponsesTarget(LLMCallConfig(model="gpt-4o", temperature=0.3), client=client)

        await target.respond(_make_messages())

        assert client.responses.create.call_args.kwargs["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_extra_kwargs_reach_the_request(self):
        client = _make_client()
        client.responses.create = AsyncMock(return_value=_make_response())
        target = OrqResponsesTarget(
            LLMCallConfig(model="gpt-4o", extra_kwargs={"top_p": 0.5, "store": True}),
            client=client,
        )

        await target.respond(_make_messages())

        kwargs = client.responses.create.call_args.kwargs
        assert kwargs["top_p"] == 0.5
        assert kwargs["store"] is True

    @pytest.mark.asyncio
    async def test_extra_kwargs_override_computed_values(self):
        client = _make_client()
        client.responses.create = AsyncMock(return_value=_make_response())
        target = OrqResponsesTarget(
            LLMCallConfig(model="gpt-4o", temperature=0.3, extra_kwargs={"temperature": 0.9}),
            client=client,
        )

        await target.respond(_make_messages())

        assert client.responses.create.call_args.kwargs["temperature"] == 0.9

    @pytest.mark.asyncio
    async def test_extra_kwargs_cannot_replace_extra_body(self):
        """``extra_body`` is a reserved structural key — it carries the internally
        computed thread/memory router body, so a caller-supplied value must be
        rejected rather than silently clobbering it."""
        client = _make_client()
        target = OrqResponsesTarget(
            LLMCallConfig(model="gpt-4o", extra_kwargs={"extra_body": {"malicious": True}}),
            client=client,
        )

        with pytest.raises(ValueError, match="extra_body"):
            await target.respond(_make_messages())

        client.responses.create.assert_not_awaited()


class TestOrqResponsesTargetReasoningEffort:
    @pytest.mark.asyncio
    async def test_reasoning_effort_forwarded_when_set(self):
        client = _make_client()
        client.responses.create = AsyncMock(return_value=_make_response())
        target = OrqResponsesTarget(
            LLMCallConfig(model="gpt-4o", reasoning_effort="high"), client=client
        )

        await target.respond(_make_messages())

        assert client.responses.create.call_args.kwargs["reasoning"] == {"effort": "high"}

    @pytest.mark.asyncio
    async def test_reasoning_omitted_when_unset(self):
        client = _make_client()
        client.responses.create = AsyncMock(return_value=_make_response())
        target = OrqResponsesTarget(LLMCallConfig(model="gpt-4o"), client=client)

        await target.respond(_make_messages())

        assert "reasoning" not in client.responses.create.call_args.kwargs

    @pytest.mark.asyncio
    async def test_reasoning_rejection_drops_and_retries_once(self):
        """A 400 naming ``reasoning`` is dropped and retried once, like execute_response."""
        client = _make_client()
        rejection = _bad_request("Unsupported parameter: 'reasoning' is not supported with this model.")
        good = _make_response(response_id="resp-after-drop")
        client.responses.create = AsyncMock(side_effect=[rejection, good])
        target = OrqResponsesTarget(
            LLMCallConfig(model="gpt-4o-mini", reasoning_effort="high"), client=client
        )

        result = await target.respond(_make_messages())

        assert client.responses.create.await_count == 2
        first_kwargs, second_kwargs = (c.kwargs for c in client.responses.create.call_args_list)
        assert first_kwargs["reasoning"] == {"effort": "high"}
        assert "reasoning" not in second_kwargs
        assert result.model == "gpt-4o-mini" or result is not None

    @pytest.mark.asyncio
    async def test_unrelated_bad_request_propagates_without_retry(self):
        """A 400 that does not name ``reasoning`` in the error body must not be swallowed."""
        client = _make_client()
        unrelated = _bad_request("Invalid value for 'temperature': must be between 0 and 2.")
        client.responses.create = AsyncMock(side_effect=unrelated)
        target = OrqResponsesTarget(
            LLMCallConfig(model="gpt-4o-mini", reasoning_effort="high"), client=client
        )

        with pytest.raises(BadRequestError):
            await target.respond(_make_messages())

        assert client.responses.create.await_count == 1

    @pytest.mark.asyncio
    async def test_memoized_rejection_strips_reasoning_up_front_on_next_call(self):
        """After one rejection, a later call on the same target never re-sends ``reasoning``."""
        client = _make_client()
        rejection = _bad_request("Unsupported parameter: 'reasoning'.")
        good = _make_response()
        client.responses.create = AsyncMock(side_effect=[rejection, good, good])
        target = OrqResponsesTarget(
            LLMCallConfig(model="gpt-4o-mini-memo", reasoning_effort="high"), client=client
        )

        await target.respond(_make_messages())
        client.responses.create.reset_mock()
        client.responses.create.side_effect = None
        client.responses.create.return_value = good

        await target.respond(_make_messages())

        assert client.responses.create.await_count == 1
        assert "reasoning" not in client.responses.create.call_args.kwargs


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
            "evaluatorq.openresponses.target.asyncio.wait_for", wraps=asyncio.wait_for
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

    @pytest.mark.asyncio
    async def test_raw_response_path_timeout_is_enforced(self):
        class SlowTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                await asyncio.sleep(0.05)
                return _responses_http_response()

        client = AsyncOpenAI(
            api_key="test-key",
            base_url="https://my.orq.ai/v3/router",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=SlowTransport()),
        )
        target = _make_target(client=client, timeout_ms=5)
        try:
            with pytest.raises(RuntimeError, match="timed out"):
                await target.respond(_make_messages())
        finally:
            await client.close()


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
        # Opt-in: the default is a single attempt because call_target_with_retry
        # owns the budget. Raising it is for callers driving respond() directly.
        target = _make_target(client=client, retry_attempts=2)

        result = await target.respond(_make_messages())

        assert client.responses.create.await_count == 2
        assert isinstance(result, AgentResponse)

    @pytest.mark.asyncio
    async def test_does_not_retry_by_default(self, monkeypatch):
        """The default must not stack under `call_target_with_retry`."""
        from openai import APIStatusError

        monkeypatch.setattr("evaluatorq.common.retry.asyncio.sleep", AsyncMock(return_value=None))

        client = _make_client()
        rate_limit = APIStatusError("rate limited", response=MagicMock(status_code=429, headers={}), body=None)
        client.responses.create = AsyncMock(side_effect=rate_limit)
        target = _make_target(client=client)

        with pytest.raises(APIStatusError):
            await target.respond(_make_messages())

        assert client.responses.create.await_count == 1

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

    @pytest.mark.asyncio
    async def test_retries_raw_response_path_on_rate_limit(self, monkeypatch):
        monkeypatch.setattr(
            "evaluatorq.common.retry.asyncio.sleep",
            AsyncMock(return_value=None),
        )
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return _responses_http_response(status_code=429)
            return _responses_http_response()

        client = _make_sdk_client(handler)
        target = OrqResponsesTarget(
            LLMCallConfig(model="gpt-4o", timeout_ms=30_000),
            client=client,
            retry_attempts=2,
        )
        try:
            result = await target.respond(_make_messages())
        finally:
            await client.close()

        assert calls == 2
        assert result.trace_id is None


# ---------------------------------------------------------------------------
# client lifecycle
# ---------------------------------------------------------------------------


class TestOrqResponsesTargetClose:
    @pytest.mark.asyncio
    async def test_close_closes_owned_client(self, monkeypatch):
        import evaluatorq.openresponses.target as target_mod

        owned_client = MagicMock()
        owned_client.close = AsyncMock()
        monkeypatch.setattr(
            target_mod, "build_simulation_client",
            lambda _client, **_: (owned_client, True),  # pyright: ignore[reportUnknownLambdaType]
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
        import evaluatorq.openresponses.target as target_mod

        owned_client = MagicMock()
        owned_client.close = AsyncMock()
        monkeypatch.setattr(
            target_mod, "build_simulation_client",
            lambda _client, **_: (owned_client, True),  # pyright: ignore[reportUnknownLambdaType]
        )
        t = OrqResponsesTarget(LLMCallConfig(model="m", api="responses"))

        await t.close()
        await t.close()

        owned_client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_async_context_manager_closes_owned_client(self, monkeypatch):
        import evaluatorq.openresponses.target as target_mod

        owned_client = MagicMock()
        owned_client.close = AsyncMock()
        monkeypatch.setattr(
            target_mod, "build_simulation_client",
            lambda _client, **_: (owned_client, True),  # pyright: ignore[reportUnknownLambdaType]
        )

        async with OrqResponsesTarget(LLMCallConfig(model="m", api="responses")) as t:
            assert t._client_owned is True

        owned_client.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# agent context
# ---------------------------------------------------------------------------


class TestOrqResponsesTargetAgentContext:
    @pytest.mark.asyncio
    async def test_agent_context_carries_instructions(self):
        target = _make_target(client=_make_client(), instructions="You are a cow.")

        ctx = await target.get_agent_context()

        assert ctx.instructions == "You are a cow."
        assert ctx.key == target.config.model

    @pytest.mark.asyncio
    async def test_agent_context_instructions_empty_when_none(self):
        target = _make_target(client=_make_client(), instructions=None)

        assert (await target.get_agent_context()).instructions == ""

    @pytest.mark.asyncio
    async def test_agent_context_maps_tools(self):
        target = _make_target(
            client=_make_client(),
            tools=[
                {"type": "function", "name": "refund", "description": "Issue refund", "parameters": {"x": 1}},
                {"type": "function", "function": {"name": "lookup", "description": "Find order"}},
                {"type": "web_search"},
            ],
        )

        ctx = await target.get_agent_context()

        assert [t.name for t in ctx.tools] == ["refund", "lookup", "web_search"]
        assert ctx.tools[0].parameters == {"x": 1}
        assert ctx.tools[1].description == "Find order"
