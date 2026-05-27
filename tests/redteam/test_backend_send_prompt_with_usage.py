"""Unit tests for backend respond happy-path and error paths.

Covers:
- OpenAIModelTarget (backends/openai.py) — uses respond()
- ORQAgentTarget (backends/orq.py)
- LangGraphTarget (integrations/langgraph_integration/target.py)
- CallableTarget (integrations/callable_integration/target.py)
- OpenAIAgentTarget (integrations/openai_agents_integration/target.py)
- VercelAISdkTarget (integrations/vercel_ai_sdk_integration/target.py)

This module is the cross-target conformance harness for RES-715: every target
must return a ``SendResult`` whose ``usage`` is either ``None`` or a populated
``TokenUsage``. Adding a new target without an entry here is a process bug,
not a code bug — the integration should not ship until the conformance check
is extended.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("openai")

from evaluatorq.contracts import Message
from evaluatorq.redteam.contracts import SendResult, TokenUsage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_openai_response(
    content: str = "test response",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    total_tokens: int = 15,
    model: str = "gpt-4o-mini",
) -> MagicMock:
    """Build a minimal fake OpenAI chat completion response."""
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = total_tokens

    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    response.model = model
    response.id = "chatcmpl-test"
    return response


def _make_orq_response(
    text: str = "orq response",
    task_id: str = "task-001",
    prompt_tokens: int = 20,
    completion_tokens: int = 10,
    total_tokens: int = 30,
    pending_tool_calls: list[Any] | None = None,
    model: str | None = None,
) -> MagicMock:
    """Build a minimal fake ORQ agent response."""
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = total_tokens

    part = MagicMock()
    part.kind = "text"
    part.text = text

    output_item = MagicMock()
    output_item.parts = [part]

    response = MagicMock()
    response.task_id = task_id
    response.output = [output_item]
    response.usage = usage
    response.pending_tool_calls = pending_tool_calls or []
    response.model = model
    return response


# ===========================================================================
# OpenAIModelTarget
# ===========================================================================


class TestOpenAIModelTargetRespondWithUsage:
    @pytest.mark.asyncio
    async def test_happy_path_returns_send_result(self) -> None:
        """Happy path: returns SendResult with text, usage, model populated."""
        from evaluatorq.redteam.backends.openai import OpenAIModelTarget

        mock_client = MagicMock()
        response = _make_openai_response(
            content="Hello!", prompt_tokens=100, completion_tokens=50, total_tokens=150
        )
        mock_client.chat.completions.create = AsyncMock(return_value=response)

        target = OpenAIModelTarget(model="gpt-4o-mini", client=mock_client)

        with patch("evaluatorq.redteam.tracing.get_tracer", return_value=None):
            result = await target.respond([Message(role="user", content="test prompt")])

        assert isinstance(result, SendResult)
        assert result.text == "Hello!"
        assert result.usage is not None
        assert result.usage.prompt_tokens == 100
        assert result.usage.completion_tokens == 50
        assert result.usage.total_tokens == 150
        assert result.usage.calls == 1
        assert result.model == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_usage_none_when_response_has_no_usage(self) -> None:
        """SendResult.usage is None when the response has no usage attribute."""
        from evaluatorq.redteam.backends.openai import OpenAIModelTarget

        mock_client = MagicMock()
        response = _make_openai_response()
        response.usage = None  # Simulate no usage info
        mock_client.chat.completions.create = AsyncMock(return_value=response)

        target = OpenAIModelTarget(model="gpt-4o-mini", client=mock_client)
        with patch("evaluatorq.redteam.tracing.get_tracer", return_value=None):
            result = await target.respond([Message(role="user", content="prompt")])

        assert result.usage is None


# ===========================================================================
# ORQAgentTarget
# ===========================================================================


class TestORQAgentTargetSendPromptWithUsage:
    def _make_target(self) -> tuple[Any, Any]:
        """Return (target, orq_client_stub)."""
        from evaluatorq.redteam.backends.orq import ORQAgentTarget

        mock_orq_client = MagicMock()
        target = ORQAgentTarget(
            agent_key="test-agent",
            orq_client=mock_orq_client,
            memory_entity_id="mem-001",
        )
        return target, mock_orq_client

    @pytest.mark.asyncio
    async def test_happy_path_single_response(self) -> None:
        """Single response (no pending tool calls) populates SendResult correctly."""
        target, orq_client = self._make_target()
        response = _make_orq_response(
            text="agent says hi",
            prompt_tokens=20,
            completion_tokens=10,
            total_tokens=30,
        )

        with (
            patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread,
            patch("evaluatorq.redteam.tracing.get_tracer", return_value=None),
        ):
            mock_to_thread.return_value = response
            result = await target.respond([Message(role="user", content="hello")])

        assert isinstance(result, SendResult)
        assert result.text == "agent says hi"
        assert result.usage is not None
        assert result.usage.prompt_tokens == 20
        assert result.usage.completion_tokens == 10
        assert result.usage.total_tokens == 30
        assert result.usage.calls == 1

    @pytest.mark.asyncio
    async def test_tool_continuation_accumulates_usage(self) -> None:
        """Usage accumulates across initial call + tool continuation."""
        target, orq_client = self._make_target()

        pending_call = MagicMock()
        pending_call.id = "tool-call-001"

        first_response = _make_orq_response(
            text="",
            task_id="task-001",
            prompt_tokens=15,
            completion_tokens=5,
            total_tokens=20,
            pending_tool_calls=[pending_call],
        )
        second_response = _make_orq_response(
            text="final answer",
            task_id="task-001",
            prompt_tokens=10,
            completion_tokens=8,
            total_tokens=18,
            pending_tool_calls=[],
        )

        call_count = 0

        async def fake_to_thread(fn: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return first_response
            return second_response

        with (
            patch("asyncio.to_thread", side_effect=fake_to_thread),
            patch("evaluatorq.redteam.tracing.get_tracer", return_value=None),
        ):
            result = await target.respond([Message(role="user", content="prompt")])

        assert isinstance(result, SendResult)
        assert result.text == "final answer"
        assert result.usage is not None
        assert result.usage.calls == 2
        assert result.usage.prompt_tokens == 25  # 15 + 10
        assert result.usage.completion_tokens == 13  # 5 + 8

    @pytest.mark.asyncio
    async def test_continuation_exhaustion_raises_runtime_error(self) -> None:
        """5 continuations all returning pending_tool_calls raises RuntimeError."""
        target, orq_client = self._make_target()

        pending_call = MagicMock()
        pending_call.id = "tool-call-persist"

        # All responses keep returning pending tool calls
        continuing_response = _make_orq_response(
            text="",
            task_id="task-001",
            prompt_tokens=5,
            completion_tokens=2,
            total_tokens=7,
            pending_tool_calls=[pending_call],
        )

        async def always_pending(fn: Any, **kwargs: Any) -> Any:
            return continuing_response

        with (
            patch("asyncio.to_thread", side_effect=always_pending),
            patch("evaluatorq.redteam.tracing.get_tracer", return_value=None),
            pytest.raises(RuntimeError, match="Unresolved pending tool calls"),
        ):
            await target.respond([Message(role="user", content="prompt")])

    @pytest.mark.asyncio
    async def test_no_usage_in_response_yields_none(self) -> None:
        """SendResult.usage is None when response carries no usage."""
        target, _ = self._make_target()
        response = _make_orq_response()
        response.usage = None  # Strip usage

        async def fake_to_thread(fn: Any, **kwargs: Any) -> Any:
            return response

        with (
            patch("asyncio.to_thread", side_effect=fake_to_thread),
            patch("evaluatorq.redteam.tracing.get_tracer", return_value=None),
        ):
            result = await target.respond([Message(role="user", content="prompt")])

        assert result.usage is None

    @pytest.mark.asyncio
    async def test_orq_partial_usage_on_error(self) -> None:
        """record_token_usage is called on span with partial totals when mid-loop exception fires."""
        from evaluatorq.redteam.backends.orq import ORQAgentTarget

        mock_orq_client = MagicMock()
        target = ORQAgentTarget(
            agent_key="test-agent",
            orq_client=mock_orq_client,
            memory_entity_id="mem-001",
        )

        pending_call = MagicMock()
        pending_call.id = "tool-call-001"

        # First call succeeds and accumulates usage, returns a pending tool call
        first_response = _make_orq_response(
            text="",
            task_id="task-001",
            prompt_tokens=15,
            completion_tokens=5,
            total_tokens=20,
            pending_tool_calls=[pending_call],
        )

        call_count = 0

        async def fake_to_thread(fn: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return first_response
            raise RuntimeError("network failure mid-loop")

        with (
            patch("asyncio.to_thread", side_effect=fake_to_thread),
            patch("evaluatorq.redteam.tracing.get_tracer", return_value=None),
            patch("evaluatorq.redteam.backends.orq.record_token_usage") as mock_record,
        ):
            with pytest.raises(RuntimeError, match="network failure mid-loop"):
                await target.respond([Message(role="user", content="prompt")])

        # Partial usage from the first call must have been recorded before the exception propagated
        mock_record.assert_called_once()
        _, kwargs_called = mock_record.call_args
        assert kwargs_called["prompt_tokens"] == 15
        assert kwargs_called["completion_tokens"] == 5
        assert kwargs_called["total_tokens"] == 20
        assert kwargs_called["calls"] == 1

    @pytest.mark.asyncio
    async def test_orq_total_zero_falls_back_to_sum(self) -> None:
        """When provider returns total_tokens=0, total is computed as prompt + completion."""
        target, _ = self._make_target()
        response = _make_orq_response(
            text="some text",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=0,  # Provider bug: zero total despite non-zero component tokens
        )

        async def fake_to_thread(fn: Any, **kwargs: Any) -> Any:
            return response

        with (
            patch("asyncio.to_thread", side_effect=fake_to_thread),
            patch("evaluatorq.redteam.tracing.get_tracer", return_value=None),
        ):
            result = await target.respond([Message(role="user", content="prompt")])

        assert result.usage is not None
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.usage.total_tokens == 15  # falls back to prompt + completion

    @pytest.mark.asyncio
    async def test_orq_uses_pipeline_config_max_continuations(self) -> None:
        """Loop bails after PIPELINE_CONFIG.max_tool_continuations continuations."""
        from evaluatorq.redteam import contracts as contracts_module

        target, _ = self._make_target()

        pending_call = MagicMock()
        pending_call.id = "tool-call-persist"

        continuing_response = _make_orq_response(
            text="",
            task_id="task-001",
            prompt_tokens=5,
            completion_tokens=2,
            total_tokens=7,
            pending_tool_calls=[pending_call],
        )

        call_count = 0

        async def count_calls(fn: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            return continuing_response

        original_max = contracts_module.PIPELINE_CONFIG.max_tool_continuations
        try:
            # Monkeypatch to 2 so the loop stops after 2 continuations instead of 5
            contracts_module.PIPELINE_CONFIG.max_tool_continuations = 2

            with (
                patch("asyncio.to_thread", side_effect=count_calls),
                patch("evaluatorq.redteam.tracing.get_tracer", return_value=None),
                pytest.raises(RuntimeError, match="Unresolved pending tool calls"),
            ):
                await target.respond([Message(role="user", content="prompt")])
        finally:
            contracts_module.PIPELINE_CONFIG.max_tool_continuations = original_max

        # 1 initial call + 2 continuation calls = 3 total
        assert call_count == 3


# ===========================================================================
# LangGraphTarget — respond
# ===========================================================================


pytest.importorskip("langgraph")


class TestLangGraphTargetSendPromptWithUsage:
    def _make_graph(self, response_content: str = "graph response") -> MagicMock:
        graph = MagicMock()
        graph.name = "test_graph"
        msg = MagicMock()
        msg.content = response_content
        graph.ainvoke = AsyncMock(return_value={"messages": [msg]})
        return graph

    @pytest.mark.asyncio
    async def test_happy_path_returns_send_result(self) -> None:
        """Happy path: respond returns SendResult with text."""
        from evaluatorq.integrations.langgraph_integration import LangGraphTarget

        graph = self._make_graph("hello from graph")
        target = LangGraphTarget(graph)

        result = await target.respond([Message(role="user", content="test prompt")])

        assert isinstance(result, SendResult)
        assert result.text == "hello from graph"

    @pytest.mark.asyncio
    async def test_happy_path_usage_none_when_no_callbacks_fire(self) -> None:
        """Without LLM callbacks firing, usage is None."""
        from evaluatorq.integrations.langgraph_integration import LangGraphTarget

        graph = self._make_graph()
        target = LangGraphTarget(graph)

        result = await target.respond([Message(role="user", content="prompt")])

        # MagicMock graph does not fire callbacks → no usage captured
        assert result.usage is None

    @pytest.mark.asyncio
    async def test_exception_usage_captured_in_finally(self) -> None:
        """When ainvoke raises after callbacks fire, partial usage is captured in the finally block."""
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, LLMResult

        from evaluatorq.integrations.langgraph_integration import LangGraphTarget
        from evaluatorq.integrations.langgraph_integration.target import _TokenUsageCollector

        graph = MagicMock()
        graph.name = "failing_graph"

        async def failing_ainvoke(input_dict: Any, config: Any) -> Any:
            # Fire the collector before raising so partial usage is captured
            callbacks = config.get("callbacks") or []
            for cb in callbacks:
                if isinstance(cb, _TokenUsageCollector):
                    msg = AIMessage(
                        content="partial",
                        usage_metadata={"input_tokens": 40, "output_tokens": 5, "total_tokens": 45},
                    )
                    gen = ChatGeneration(message=msg)
                    cb.on_llm_end(LLMResult(generations=[[gen]]))
            raise RuntimeError("graph failed")

        graph.ainvoke = failing_ainvoke

        target = LangGraphTarget(graph)

        # The collector drains in the inner finally; the exception propagates.
        # We can't read usage from the exception path, but this verifies
        # the finally block runs without error.
        with pytest.raises(RuntimeError, match="graph failed"):
            await target.respond([Message(role="user", content="hi")])

    @pytest.mark.asyncio
    async def test_no_messages_key_raises_value_error(self) -> None:
        """Graph returning dict without 'messages' key raises ValueError."""
        from evaluatorq.integrations.langgraph_integration import LangGraphTarget

        graph = MagicMock()
        graph.name = "bad_graph"
        graph.ainvoke = AsyncMock(return_value={"output": "no messages"})

        target = LangGraphTarget(graph)

        with pytest.raises(ValueError, match="'messages' key"):
            await target.respond([Message(role="user", content="hi")])

    @pytest.mark.asyncio
    async def test_empty_messages_raises_value_error(self) -> None:
        """Graph returning empty messages list raises ValueError."""
        from evaluatorq.integrations.langgraph_integration import LangGraphTarget

        graph = MagicMock()
        graph.name = "empty_graph"
        graph.ainvoke = AsyncMock(return_value={"messages": []})

        target = LangGraphTarget(graph)

        with pytest.raises(ValueError, match="empty 'messages' list"):
            await target.respond([Message(role="user", content="hi")])

    @pytest.mark.asyncio
    async def test_usage_persisted_in_send_result_on_success(self) -> None:
        """SendResult.usage is populated after a successful call with usage."""
        from langchain_core.messages import AIMessage
        from langchain_core.messages.ai import UsageMetadata
        from langchain_core.outputs import ChatGeneration, LLMResult

        from evaluatorq.integrations.langgraph_integration import LangGraphTarget
        from evaluatorq.integrations.langgraph_integration.target import _TokenUsageCollector

        graph = MagicMock()
        graph.name = "test_graph"

        def _fake_ainvoke(input_dict: Any, *, config: Any) -> dict[str, Any]:
            callbacks = config.get("callbacks", [])
            for cb in (callbacks if isinstance(callbacks, list) else []):
                if isinstance(cb, _TokenUsageCollector):
                    meta = UsageMetadata(input_tokens=12, output_tokens=3, total_tokens=15)
                    msg = AIMessage(content="ok", usage_metadata=meta)
                    gen = ChatGeneration(message=msg)
                    cb.on_llm_end(LLMResult(generations=[[gen]]))
            mock_msg = MagicMock()
            mock_msg.content = "ok"
            return {"messages": [mock_msg]}

        graph.ainvoke = AsyncMock(side_effect=_fake_ainvoke)

        target = LangGraphTarget(graph)
        result = await target.respond([Message(role="user", content="hi")])

        assert result.usage is not None
        assert result.usage.prompt_tokens == 12


# ===========================================================================
# CallableTarget — respond
# ===========================================================================


class TestCallableTargetSendPromptWithUsage:
    @pytest.mark.asyncio
    async def test_async_callable_returns_send_result(self) -> None:
        from evaluatorq.integrations.callable_integration import CallableTarget

        async def agent(prompt: str) -> str:
            return f"reply:{prompt}"

        target = CallableTarget(agent)
        result = await target.respond([Message(role="user", content="hi")])
        assert isinstance(result, SendResult)
        assert result.text == "reply:hi"
        assert result.usage is None

    @pytest.mark.asyncio
    async def test_sync_callable_returns_send_result(self) -> None:
        from evaluatorq.integrations.callable_integration import CallableTarget

        target = CallableTarget(lambda prompt: f"sync:{prompt}")
        result = await target.respond([Message(role="user", content="hi")])
        assert isinstance(result, SendResult)
        assert result.text == "sync:hi"
        assert result.usage is None

    @pytest.mark.asyncio
    async def test_usage_fn_populates_token_usage(self) -> None:
        from evaluatorq.integrations.callable_integration import CallableTarget

        async def agent(prompt: str) -> str:
            return "ok"

        def usage_fn(prompt: str, response: str) -> TokenUsage:
            return TokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10, calls=1)

        target = CallableTarget(agent, usage_fn=usage_fn)
        result = await target.respond([Message(role="user", content="hi")])
        assert result.usage is not None
        assert result.usage.prompt_tokens == 7
        assert result.usage.completion_tokens == 3
        assert result.usage.total_tokens == 10
        assert result.usage.calls == 1

    @pytest.mark.asyncio
    async def test_usage_fn_exception_yields_none_usage(self) -> None:
        from evaluatorq.integrations.callable_integration import CallableTarget

        async def agent(prompt: str) -> str:
            return "ok"

        def bad_usage_fn(prompt: str, response: str) -> TokenUsage:
            raise RuntimeError("boom")

        target = CallableTarget(agent, usage_fn=bad_usage_fn)
        result = await target.respond([Message(role="user", content="hi")])
        assert result.usage is None
        assert result.text == "ok"


# ===========================================================================
# OpenAIAgentTarget — respond
# ===========================================================================


# Skip if the optional openai-agents SDK isn't available
pytest.importorskip("agents")


class TestOpenAIAgentTargetSendPromptWithUsage:
    def _make_run_result(
        self,
        *,
        final_output: str = "agent reply",
        input_tokens: int | None = 30,
        output_tokens: int | None = 12,
        total_tokens: int | None = 42,
    ) -> MagicMock:
        result = MagicMock()
        result.final_output = final_output
        result.to_input_list.return_value = []
        if input_tokens is None and output_tokens is None and total_tokens is None:
            result.context_wrapper = None
        else:
            usage = MagicMock()
            usage.input_tokens = input_tokens
            usage.output_tokens = output_tokens
            usage.total_tokens = total_tokens
            ctx = MagicMock()
            ctx.usage = usage
            result.context_wrapper = ctx
        return result

    @pytest.mark.asyncio
    async def test_happy_path_returns_send_result_with_usage(self) -> None:
        from evaluatorq.integrations.openai_agents_integration import OpenAIAgentTarget

        agent = MagicMock()
        target = OpenAIAgentTarget(agent)

        run_result = self._make_run_result(
            final_output="hello",
            input_tokens=30,
            output_tokens=12,
            total_tokens=42,
        )

        with patch(
            "evaluatorq.integrations.openai_agents_integration.target.Runner.run",
            new=AsyncMock(return_value=run_result),
        ):
            result = await target.respond([Message(role="user", content="hi")])

        assert isinstance(result, SendResult)
        assert result.text == "hello"
        assert result.usage is not None
        assert result.usage.prompt_tokens == 30
        assert result.usage.completion_tokens == 12
        assert result.usage.total_tokens == 42
        assert result.usage.calls == 1

    @pytest.mark.asyncio
    async def test_total_tokens_zero_falls_back_to_sum(self) -> None:
        from evaluatorq.integrations.openai_agents_integration import OpenAIAgentTarget

        agent = MagicMock()
        target = OpenAIAgentTarget(agent)

        run_result = self._make_run_result(
            final_output="hello",
            input_tokens=20,
            output_tokens=10,
            total_tokens=0,
        )

        with patch(
            "evaluatorq.integrations.openai_agents_integration.target.Runner.run",
            new=AsyncMock(return_value=run_result),
        ):
            result = await target.respond([Message(role="user", content="hi")])

        assert result.usage is not None
        assert result.usage.total_tokens == 30  # falls back to prompt + completion

    @pytest.mark.asyncio
    async def test_no_usage_in_context_yields_none(self) -> None:
        from evaluatorq.integrations.openai_agents_integration import OpenAIAgentTarget

        agent = MagicMock()
        target = OpenAIAgentTarget(agent)

        run_result = self._make_run_result(
            final_output="hello",
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
        )

        with patch(
            "evaluatorq.integrations.openai_agents_integration.target.Runner.run",
            new=AsyncMock(return_value=run_result),
        ):
            result = await target.respond([Message(role="user", content="hi")])

        assert result.text == "hello"
        assert result.usage is None


# ===========================================================================
# VercelAISdkTarget — respond
# ===========================================================================


class TestVercelAISdkTargetSendPromptWithUsage:
    def _make_response(self, *, content_type: str, text: str) -> MagicMock:
        response = MagicMock()
        response.headers = {"content-type": content_type}
        response.text = text
        response.raise_for_status = MagicMock()
        return response

    @pytest.mark.asyncio
    async def test_data_stream_protocol_returns_text_and_usage(self) -> None:
        from evaluatorq.integrations.vercel_ai_sdk_integration import VercelAISdkTarget

        target = VercelAISdkTarget("http://example.local/api/chat")

        stream = (
            '0:"Hello"\n'
            '0:" world"\n'
            'e:{"finishReason":"stop","usage":{"promptTokens":10,"completionTokens":5,"totalTokens":15}}\n'
        )
        response = self._make_response(content_type="text/plain", text=stream)

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=response)

        with patch(
            "evaluatorq.integrations.vercel_ai_sdk_integration.target.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await target.respond([Message(role="user", content="hi")])

        assert isinstance(result, SendResult)
        assert result.text == "Hello world"
        assert result.usage is not None
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.usage.total_tokens == 15
        assert result.usage.calls == 1

    @pytest.mark.asyncio
    async def test_json_response_returns_text_and_usage(self) -> None:
        from evaluatorq.integrations.vercel_ai_sdk_integration import VercelAISdkTarget

        target = VercelAISdkTarget("http://example.local/api/chat")

        body = (
            '{"text":"hello there",'
            '"usage":{"prompt_tokens":20,"completion_tokens":4,"total_tokens":24}}'
        )
        response = self._make_response(content_type="application/json", text=body)

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=response)

        with patch(
            "evaluatorq.integrations.vercel_ai_sdk_integration.target.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await target.respond([Message(role="user", content="hi")])

        assert isinstance(result, SendResult)
        assert result.usage is not None
        assert result.usage.prompt_tokens == 20
        assert result.usage.completion_tokens == 4
        assert result.usage.total_tokens == 24

    @pytest.mark.asyncio
    async def test_plain_text_response_yields_none_usage(self) -> None:
        from evaluatorq.integrations.vercel_ai_sdk_integration import VercelAISdkTarget

        target = VercelAISdkTarget("http://example.local/api/chat")
        response = self._make_response(content_type="text/html", text="plain reply")

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=response)

        with patch(
            "evaluatorq.integrations.vercel_ai_sdk_integration.target.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await target.respond([Message(role="user", content="hi")])

        assert isinstance(result, SendResult)
        assert result.text == "plain reply"
        assert result.usage is None
