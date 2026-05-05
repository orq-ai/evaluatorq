"""Unit tests for backend send_prompt_with_usage happy-path and error paths.

Covers:
- OpenAIModelTarget (backends/openai.py)
- ORQAgentTarget (backends/orq.py)
- LangGraphTarget (integrations/langgraph_integration/target.py)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("openai")

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


class TestOpenAIModelTargetSendPromptWithUsage:
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
            result = await target.send_prompt_with_usage("test prompt")

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
            result = await target.send_prompt_with_usage("prompt")

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
            result = await target.send_prompt_with_usage("hello")

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
            result = await target.send_prompt_with_usage("prompt")

        assert isinstance(result, SendResult)
        assert result.text == "final answer"
        assert result.usage is not None
        assert result.usage.calls == 2
        assert result.usage.prompt_tokens == 25  # 15 + 10
        assert result.usage.completion_tokens == 13  # 5 + 8

    @pytest.mark.asyncio
    async def test_continuation_exhaustion_raises_runtime_error(self) -> None:
        """5 continuations all returning pending_tool_calls raises RuntimeError
        so the orchestrator's existing circuit-breaker counts it as an agent
        error and continues with other runs (rather than treating an empty/
        partial response as a successful answer)."""
        target, orq_client = self._make_target()

        pending_call = MagicMock()
        pending_call.id = "tool-call-persist"

        continuing_response = _make_orq_response(
            text="partial",
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
            await target.send_prompt_with_usage("prompt")

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
            result = await target.send_prompt_with_usage("prompt")

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
                await target.send_prompt_with_usage("prompt")

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
            result = await target.send_prompt_with_usage("prompt")

        assert result.usage is not None
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.usage.total_tokens == 15  # falls back to prompt + completion

    @pytest.mark.asyncio
    async def test_orq_continuation_loop_call_count(self) -> None:
        """Hardcoded cap is 5 — 1 initial call + 5 continuation calls = 6 total
        before exhaustion raises."""
        target, _ = self._make_target()

        pending_call = MagicMock()
        pending_call.id = "tool-call-persist"

        continuing_response = _make_orq_response(
            text="partial",
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

        with (
            patch("asyncio.to_thread", side_effect=count_calls),
            patch("evaluatorq.redteam.tracing.get_tracer", return_value=None),
            pytest.raises(RuntimeError, match="Unresolved pending tool calls"),
        ):
            await target.send_prompt_with_usage("prompt")

        assert call_count == 6


# ===========================================================================
# LangGraphTarget — send_prompt_with_usage
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
        """Happy path: send_prompt_with_usage returns SendResult with text."""
        from evaluatorq.integrations.langgraph_integration import LangGraphTarget

        graph = self._make_graph("hello from graph")
        target = LangGraphTarget(graph)

        result = await target.send_prompt_with_usage("test prompt")

        assert isinstance(result, SendResult)
        assert result.text == "hello from graph"

    @pytest.mark.asyncio
    async def test_happy_path_usage_none_when_no_callbacks_fire(self) -> None:
        """Without LLM callbacks firing, usage is None."""
        from evaluatorq.integrations.langgraph_integration import LangGraphTarget

        graph = self._make_graph()
        target = LangGraphTarget(graph)

        result = await target.send_prompt_with_usage("prompt")

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
            await target.send_prompt_with_usage("hi")

    @pytest.mark.asyncio
    async def test_no_messages_key_raises_value_error(self) -> None:
        """Graph returning dict without 'messages' key raises ValueError."""
        from evaluatorq.integrations.langgraph_integration import LangGraphTarget

        graph = MagicMock()
        graph.name = "bad_graph"
        graph.ainvoke = AsyncMock(return_value={"output": "no messages"})

        target = LangGraphTarget(graph)

        with pytest.raises(ValueError, match="'messages' key"):
            await target.send_prompt_with_usage("hi")

    @pytest.mark.asyncio
    async def test_empty_messages_raises_value_error(self) -> None:
        """Graph returning empty messages list raises ValueError."""
        from evaluatorq.integrations.langgraph_integration import LangGraphTarget

        graph = MagicMock()
        graph.name = "empty_graph"
        graph.ainvoke = AsyncMock(return_value={"messages": []})

        target = LangGraphTarget(graph)

        with pytest.raises(ValueError, match="empty 'messages' list"):
            await target.send_prompt_with_usage("hi")

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
        result = await target.send_prompt_with_usage("hi")

        assert result.usage is not None
        assert result.usage.prompt_tokens == 12
