"""Tests for OrqResponsesTarget (simulation target backed by Responses API).

Verifies:
- __call__(messages) returns str text and is stateless w.r.t. instance state
- send_prompt returns AgentResponse and threads via previous_response_id
- previous_response_id threading across send_prompt calls
- new() returns a fresh instance
- get_usage() accumulates token counts from send_prompt only
- Timeout is applied via asyncio.wait_for
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evaluatorq.contracts import AgentResponse, LLMCallConfig
from evaluatorq.simulation.target import OrqResponsesTarget
from evaluatorq.simulation.types import ChatMessage


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

    @pytest.mark.asyncio
    async def test_call_does_not_mutate_previous_response_id(self):
        """__call__ is stateless w.r.t. self._previous_response_id.

        Pins the partial-stateless contract: mixed send_prompt / __call__ usage
        on the same instance must not corrupt redteam threading state.
        """
        client = _make_client()
        client.responses.create = AsyncMock(
            side_effect=[
                _make_response(response_id="resp-from-send"),
                _make_response(response_id="resp-from-call"),
            ]
        )
        target = _make_target(client=client)

        # Establish redteam thread state via send_prompt
        await target.send_prompt("turn 1")
        assert target._previous_response_id == "resp-from-send"

        # Sim call must not overwrite or clear that state
        await target(_make_messages("sim turn"))
        assert target._previous_response_id == "resp-from-send"

    @pytest.mark.asyncio
    async def test_call_does_not_accumulate_usage_on_instance(self):
        """__call__ is stateless w.r.t. self._accumulated_usage.

        Sim mode does not contribute to instance-level usage; the sim runner
        owns its own accounting. get_usage() reflects send_prompt calls only.
        """
        client = _make_client()
        client.responses.create = AsyncMock(
            side_effect=[
                _make_response(input_tokens=10, output_tokens=5),   # send_prompt
                _make_response(input_tokens=99, output_tokens=99),  # __call__
            ]
        )
        target = _make_target(client=client)

        await target.send_prompt("redteam turn")
        await target(_make_messages("sim turn"))

        usage = target.get_usage()
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 5
        assert usage.total_tokens == 15


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
        assert result.usage is not None
        assert result.response_id == "resp-123"
        assert result.model == "gpt-4o"

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

    @pytest.mark.asyncio
    async def test_dict_response_id_threads_and_usage_is_returned(self):
        client = _make_client()
        client.responses.create = AsyncMock(
            side_effect=[
                _make_dict_response(text="one", response_id="resp-dict-1"),
                _make_dict_response(text="two", response_id="resp-dict-2"),
            ]
        )
        target = _make_target(client=client)

        result = await target.send_prompt("turn 1")
        await target.send_prompt("turn 2")

        second_call_kwargs = client.responses.create.call_args_list[1].kwargs
        assert second_call_kwargs.get("previous_response_id") == "resp-dict-1"
        assert target._previous_response_id == "resp-dict-2"
        assert result.text == "one"
        assert result.response_id == "resp-dict-1"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5


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


class TestOrqResponsesTargetNewExtended:
    """Additional new() branch tests from PR review."""

    def test_new_mints_fresh_memory_entity_id_when_set(self):
        """new() produces a different non-None memory_entity_id when one was set."""
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
        # Must be a valid UUID4
        parsed = uuid.UUID(fresh.memory_entity_id, version=4)
        assert parsed.version == 4

    def test_new_preserves_tools_parameter(self):
        """new() propagates tools= to the returned instance unchanged."""
        tools = [{"type": "function", "function": {"name": "foo"}}]
        client = _make_client()
        target = OrqResponsesTarget(
            LLMCallConfig(model="gpt-4o"),
            tools=tools,
            client=client,
        )

        fresh = target.new()

        assert fresh.tools == target.tools

    def test_new_propagates_externally_owned_client(self):
        """new() shares the same client object when it was externally injected."""
        client = _make_client()
        target = OrqResponsesTarget(LLMCallConfig(model="gpt-4o"), client=client)

        assert not target._client_owned  # externally injected → not owned

        fresh = target.new()

        assert fresh._client is target._client

    def test_new_does_not_share_self_owned_client(self, monkeypatch):
        """new() does NOT propagate a self-owned client; each instance owns its own."""
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

        # Two distinct client objects must have been created — one per instance
        assert fresh._client is not target._client


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


class TestOrqResponsesTargetTools:
    @pytest.mark.asyncio
    async def test_tools_forwarded_to_sdk_when_set(self):
        client = _make_client()
        client.responses.create = AsyncMock(return_value=_make_response())
        config = LLMCallConfig(model="gpt-4o")
        tool_spec = [{"type": "function", "name": "lookup", "parameters": {}}]
        target = OrqResponsesTarget(config, tools=tool_spec, client=client)

        await target.send_prompt("prompt")

        call_kwargs = client.responses.create.call_args.kwargs
        assert call_kwargs.get("tools") == tool_spec

    @pytest.mark.asyncio
    async def test_tools_omitted_when_none_or_empty(self):
        client = _make_client()
        client.responses.create = AsyncMock(return_value=_make_response())
        config = LLMCallConfig(model="gpt-4o")

        target_none = OrqResponsesTarget(config, tools=None, client=client)
        await target_none.send_prompt("p")
        assert "tools" not in client.responses.create.call_args.kwargs

        client.responses.create.reset_mock()
        target_empty = OrqResponsesTarget(config, tools=[], client=client)
        await target_empty.send_prompt("p")
        assert "tools" not in client.responses.create.call_args.kwargs


class TestOrqResponsesTargetMissingResponseId:
    @pytest.mark.asyncio
    async def test_missing_id_disables_threading_and_warns(self, caplog):
        client = _make_client()
        resp = _make_response()
        resp.id = None
        client.responses.create = AsyncMock(return_value=resp)
        target = _make_target(client=client)

        await target.send_prompt("first")

        # _previous_response_id stays None, so the next call must NOT include
        # previous_response_id in kwargs.
        assert target._previous_response_id is None

        client.responses.create.reset_mock()
        await target.send_prompt("second")
        call_kwargs = client.responses.create.call_args.kwargs
        assert "previous_response_id" not in call_kwargs

    @pytest.mark.asyncio
    async def test_empty_string_id_disables_threading(self):
        client = _make_client()
        resp = _make_response()
        resp.id = ""
        client.responses.create = AsyncMock(return_value=resp)
        target = _make_target(client=client)

        await target.send_prompt("first")

        assert target._previous_response_id is None


class TestOrqResponsesTargetSimPathKwargs:
    @pytest.mark.asyncio
    async def test_sim_call_does_not_send_previous_response_id_kwarg(self):
        """Sim path must NEVER include previous_response_id in SDK kwargs."""
        client = _make_client()
        client.responses.create = AsyncMock(return_value=_make_response())
        target = _make_target(client=client)

        # Prime instance state via send_prompt to ensure sim path doesn't leak it
        await target.send_prompt("first")
        client.responses.create.reset_mock()
        await target(_make_messages("sim turn"))

        call_kwargs = client.responses.create.call_args.kwargs
        assert "previous_response_id" not in call_kwargs

    @pytest.mark.asyncio
    async def test_interleaved_send_call_send_preserves_redteam_thread(self):
        """send_prompt -> __call__ -> send_prompt: redteam thread unaffected."""
        client = _make_client()
        responses = [
            _make_response(response_id="resp-1"),  # send #1
            _make_response(response_id="resp-2"),  # __call__
            _make_response(response_id="resp-3"),  # send #2 — must use resp-1
        ]
        client.responses.create = AsyncMock(side_effect=responses)
        target = _make_target(client=client)

        await target.send_prompt("a")
        await target(_make_messages("sim"))
        await target.send_prompt("b")

        # Third call kwargs must thread from resp-1 (post-send #1), not resp-2
        third_kwargs = client.responses.create.call_args_list[2].kwargs
        assert third_kwargs.get("previous_response_id") == "resp-1"
        # Final state reflects only send_prompt calls
        assert target._previous_response_id == "resp-3"


class TestOrqResponsesTargetStatefulMissingId:
    @pytest.mark.asyncio
    async def test_stateful_with_missing_response_id_keeps_prior_thread(self):
        """send_prompt with malformed response (no id): thread unchanged, usage updated."""
        client = _make_client()
        good = _make_response(response_id="resp-good", input_tokens=10, output_tokens=5)
        bad = _make_response(input_tokens=2, output_tokens=3)
        bad.id = None
        client.responses.create = AsyncMock(side_effect=[good, bad])
        target = _make_target(client=client)

        await target.send_prompt("first")
        assert target._previous_response_id == "resp-good"
        assert target.threading_disabled is False
        await target.send_prompt("second")
        # Thread id must NOT be overwritten by the bad response
        assert target._previous_response_id == "resp-good"
        # Threading-loss flag must be set so callers can detect degradation
        assert target.threading_disabled is True
        # Usage still accumulates (asymmetric but documented)
        assert target.get_usage().prompt_tokens == 12
        assert target.get_usage().completion_tokens == 8

    @pytest.mark.asyncio
    async def test_threading_disabled_logged_at_error_level(self, caplog):
        """Missing response.id must escalate to logger.error so silent degradation is visible."""
        import logging

        client = _make_client()
        bad = _make_response()
        bad.id = None
        client.responses.create = AsyncMock(return_value=bad)
        target = _make_target(client=client)

        with caplog.at_level(logging.ERROR, logger="evaluatorq.simulation.target"):
            await target.send_prompt("hi")

        assert target.threading_disabled is True
        assert any(
            r.levelno >= logging.ERROR and "threading disabled" in r.getMessage()
            for r in caplog.records
        ), f"expected ERROR log mentioning 'threading disabled'; got {[(r.levelname, r.getMessage()) for r in caplog.records]}"

    @pytest.mark.asyncio
    async def test_threading_disabled_error_logged_once(self, caplog):
        """Repeated missing-id calls must not spam the error log."""
        import logging

        client = _make_client()
        bad1, bad2 = _make_response(), _make_response()
        bad1.id = None
        bad2.id = None
        client.responses.create = AsyncMock(side_effect=[bad1, bad2])
        target = _make_target(client=client)

        with caplog.at_level(logging.ERROR, logger="evaluatorq.simulation.target"):
            await target.send_prompt("a")
            await target.send_prompt("b")

        error_records = [
            r for r in caplog.records
            if r.levelno >= logging.ERROR and "threading disabled" in r.getMessage()
        ]
        assert len(error_records) == 1, f"expected exactly one ERROR log; got {len(error_records)}"


class TestOrqResponsesTargetRetry:
    """Verify retry wrapping of the Responses API call (rate-limit / 5xx / network)."""

    @pytest.mark.asyncio
    async def test_retries_on_rate_limit_then_succeeds(self, monkeypatch):
        """A 429 followed by a success must be retried transparently."""
        from openai import APIStatusError

        monkeypatch.setattr(
            "evaluatorq.simulation.utils.retry.asyncio.sleep",
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

        result = await target.send_prompt("hi")

        assert client.responses.create.await_count == 2
        assert target._previous_response_id == "resp-after-retry"
        assert isinstance(result, AgentResponse)

    @pytest.mark.asyncio
    async def test_does_not_retry_on_non_retryable_error(self, monkeypatch):
        """A 400 (client error) must surface immediately without retry."""
        from openai import APIStatusError

        sleep_mock = AsyncMock(return_value=None)
        monkeypatch.setattr(
            "evaluatorq.simulation.utils.retry.asyncio.sleep", sleep_mock
        )

        client = _make_client()
        bad_request = APIStatusError(
            "bad request",
            response=MagicMock(status_code=400, headers={}),
            body=None,
        )
        client.responses.create = AsyncMock(side_effect=bad_request)
        target = _make_target(client=client)

        with pytest.raises(APIStatusError):
            await target.send_prompt("hi")

        assert client.responses.create.await_count == 1
        assert sleep_mock.await_count == 0


class TestOrqResponsesTargetClose:
    """Verify HTTP client lifecycle — owned clients must close, injected must not."""

    @pytest.mark.asyncio
    async def test_close_closes_owned_client(self, monkeypatch):
        """Self-built client must be closed on close()."""
        from evaluatorq.simulation import target as target_mod

        owned_client = MagicMock()
        owned_client.close = AsyncMock()
        monkeypatch.setattr(
            target_mod, "build_simulation_client",
            lambda _client: (owned_client, True),  # pyright: ignore[reportUnknownLambdaType]
        )
        config = LLMCallConfig(model="m", api="responses")
        t = OrqResponsesTarget(config)

        await t.close()

        owned_client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_does_not_close_injected_client(self):
        """Externally-injected client lifecycle is the caller's concern."""
        injected = _make_client()
        injected.close = AsyncMock()
        target = _make_target(client=injected)

        await target.close()

        injected.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self, monkeypatch):
        """Calling close twice must not double-close the underlying client."""
        from evaluatorq.simulation import target as target_mod

        owned_client = MagicMock()
        owned_client.close = AsyncMock()
        monkeypatch.setattr(
            target_mod, "build_simulation_client",
            lambda _client: (owned_client, True),  # pyright: ignore[reportUnknownLambdaType]
        )
        config = LLMCallConfig(model="m", api="responses")
        t = OrqResponsesTarget(config)

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
        config = LLMCallConfig(model="m", api="responses")

        async with OrqResponsesTarget(config) as t:
            assert t._client_owned is True

        owned_client.close.assert_awaited_once()
