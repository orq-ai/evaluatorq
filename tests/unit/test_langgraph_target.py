"""Unit tests for LangGraph red teaming target."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("langgraph")

from evaluatorq.integrations.langgraph_integration import LangGraphTarget  # noqa: E402


def _make_graph(response_content: str = "I'm fine") -> MagicMock:
    graph = MagicMock()
    graph.name = "test_graph"
    msg = MagicMock()
    msg.content = response_content
    graph.ainvoke = AsyncMock(return_value={"messages": [msg]})
    return graph


class TestLangGraphTarget:
    @pytest.mark.asyncio
    async def test_send_prompt_returns_response(self) -> None:
        graph = _make_graph("hello back")
        target = LangGraphTarget(graph)
        result = await target.send_prompt("hello")
        assert result == "hello back"

    @pytest.mark.asyncio
    async def test_send_prompt_passes_user_message(self) -> None:
        graph = _make_graph()
        target = LangGraphTarget(graph)
        await target.send_prompt("test prompt")

        call_args = graph.ainvoke.call_args
        messages = call_args[0][0]["messages"]
        assert messages == [{"role": "user", "content": "test prompt"}]

    @pytest.mark.asyncio
    async def test_send_prompt_passes_memory_entity_id(self) -> None:
        graph = _make_graph()
        target = LangGraphTarget(graph)
        await target.send_prompt("hi")

        config = graph.ainvoke.call_args[1]["config"]
        assert "thread_id" in config["configurable"]

    @pytest.mark.asyncio
    async def test_reset_preserves_memory_entity_id(self) -> None:
        graph = _make_graph()
        target = LangGraphTarget(graph)
        old_thread = target.memory_entity_id
        target.new()
        assert target.memory_entity_id == old_thread

    @pytest.mark.asyncio
    async def test_extra_config_is_preserved(self) -> None:
        graph = _make_graph()
        target = LangGraphTarget(graph, config={"recursion_limit": 50})
        await target.send_prompt("hi")

        config = graph.ainvoke.call_args[1]["config"]
        assert config["recursion_limit"] == 50

    @pytest.mark.asyncio
    async def test_raises_on_missing_messages_key(self) -> None:
        graph = MagicMock()
        graph.ainvoke = AsyncMock(return_value={"output": "no messages here"})
        target = LangGraphTarget(graph)

        with pytest.raises(ValueError, match="'messages' key"):
            await target.send_prompt("hi")

    @pytest.mark.asyncio
    async def test_raises_on_empty_messages_list(self) -> None:
        graph = MagicMock()
        graph.ainvoke = AsyncMock(return_value={"messages": []})
        target = LangGraphTarget(graph)

        with pytest.raises(ValueError, match="empty 'messages' list"):
            await target.send_prompt("hi")

    @pytest.mark.asyncio
    async def test_handles_dict_messages(self) -> None:
        graph = MagicMock()
        graph.ainvoke = AsyncMock(
            return_value={"messages": [{"role": "assistant", "content": "dict msg"}]}
        )
        target = LangGraphTarget(graph)
        result = await target.send_prompt("hi")
        assert result == "dict msg"

    def test_clone_returns_independent_instance(self) -> None:
        graph = _make_graph()
        target = LangGraphTarget(graph, config={"recursion_limit": 50})
        cloned = target.new()
        assert cloned is not target
        assert cloned.memory_entity_id != target.memory_entity_id
        assert cloned._graph is graph
        assert cloned._extra_config is not target._extra_config
        assert cloned._extra_config == {"recursion_limit": 50}

    def test_clone_gets_fresh_memory_entity_id(self) -> None:
        graph = _make_graph()
        target = LangGraphTarget(graph)
        cloned = target.new()
        assert cloned.memory_entity_id != target.memory_entity_id
        assert cloned.memory_entity_id  # non-empty

    @pytest.mark.asyncio
    async def test_configurable_key_collision_preserves_user_keys(self) -> None:
        """config={"configurable": {"custom_key": "val"}} must not be overwritten by memory_entity_id injection."""
        graph = _make_graph()
        target = LangGraphTarget(graph, config={"configurable": {"custom_key": "val"}})
        await target.send_prompt("hi")

        config = graph.ainvoke.call_args[1]["config"]
        assert config["configurable"]["custom_key"] == "val"
        assert "thread_id" in config["configurable"]

    @pytest.mark.asyncio
    async def test_non_string_content_is_coerced_to_str(self) -> None:
        """List-type content (e.g. multimodal messages) must be coerced to str."""
        graph = MagicMock()
        msg = MagicMock()
        msg.content = [{"type": "text", "text": "multimodal content"}]
        graph.ainvoke = AsyncMock(return_value={"messages": [msg]})
        target = LangGraphTarget(graph)
        result = await target.send_prompt("hi")
        assert isinstance(result, str)
        assert "multimodal content" in result


class TestLangGraphTargetAgentContext:
    @pytest.mark.asyncio
    async def test_get_agent_context_from_react_agent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Tools bound via create_react_agent show up in the agent context."""
        from langchain_core.tools import tool
        from langchain_openai import ChatOpenAI
        from langgraph.prebuilt import create_react_agent

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-stub")

        @tool
        def add(a: int, b: int) -> int:
            """Add two integers."""
            return a + b

        graph = create_react_agent(ChatOpenAI(model="gpt-4o-mini"), tools=[add])
        target = LangGraphTarget(graph)

        ctx = await target.get_agent_context()
        assert ctx.key.startswith("LangGraph_")
        tool_names = ctx.get_tool_names()
        assert "add" in tool_names
        # create_react_agent does not set a checkpointer by default → no memory entries
        assert ctx.memory_stores == []

    @pytest.mark.asyncio
    async def test_get_agent_context_handles_graph_without_tools_node(self) -> None:
        graph = _make_graph()
        graph.nodes = {}
        graph.checkpointer = None
        target = LangGraphTarget(graph)
        ctx = await target.get_agent_context()
        assert ctx.key.startswith("test_graph_")
        assert ctx.tools == []
        assert ctx.memory_stores == []

    @pytest.mark.asyncio
    async def test_get_agent_context_emits_memory_store_when_checkpointer_present(self) -> None:
        from evaluatorq.redteam.contracts import MemoryStoreInfo

        graph = _make_graph()
        graph.nodes = {}
        checkpointer = MagicMock()
        checkpointer.__class__.__name__ = "InMemorySaver"
        graph.checkpointer = checkpointer
        target = LangGraphTarget(graph)

        ctx = await target.get_agent_context()
        assert len(ctx.memory_stores) == 1
        assert isinstance(ctx.memory_stores[0], MemoryStoreInfo)
        assert ctx.memory_stores[0].id == target.memory_entity_id

    @pytest.mark.asyncio
    async def test_get_agent_context_dedupes_tools_across_nodes(self) -> None:
        """Same tool registered in multiple ToolNodes must yield a single entry."""
        shared_tool = MagicMock()
        shared_tool.description = "shared"
        bound_a = MagicMock()
        bound_a.tools_by_name = {"shared": shared_tool}
        bound_b = MagicMock()
        bound_b.tools_by_name = {"shared": shared_tool, "extra": MagicMock(description=None)}
        node_a = MagicMock(bound=bound_a)
        node_b = MagicMock(bound=bound_b)

        graph = _make_graph()
        graph.nodes = {"tools_a": node_a, "tools_b": node_b}
        graph.checkpointer = None
        target = LangGraphTarget(graph)

        ctx = await target.get_agent_context()
        names = [t.name for t in ctx.tools]
        assert names.count("shared") == 1
        assert sorted(names) == ["extra", "shared"]

    @pytest.mark.asyncio
    async def test_get_agent_context_override_returns_verbatim(self) -> None:
        from evaluatorq.redteam.contracts import AgentContext, ToolInfo

        override = AgentContext(
            key="my-custom-agent",
            tools=[ToolInfo(name="custom_tool")],
            description="explicitly-provided context",
        )
        graph = _make_graph()
        target = LangGraphTarget(graph, agent_context=override)

        ctx = await target.get_agent_context()
        assert ctx is override


class TestLangGraphTargetTokenUsage:
    """Tests for callback-handler-based token usage capture."""

    def test_direct_collector_accumulates_usage(self) -> None:
        """Collector accumulates tokens from a real LLMResult with ChatGeneration."""
        from langchain_core.messages import AIMessage
        from langchain_core.messages.ai import UsageMetadata
        from langchain_core.outputs import ChatGeneration, LLMResult

        from evaluatorq.integrations.langgraph_integration.target import _TokenUsageCollector

        collector = _TokenUsageCollector()
        msg = AIMessage(
            content="hi",
            usage_metadata=UsageMetadata(input_tokens=100, output_tokens=10, total_tokens=110),
        )
        gen = ChatGeneration(message=msg)
        result = LLMResult(generations=[[gen]])
        collector.on_llm_end(result)

        assert collector.prompt_tokens == 100
        assert collector.completion_tokens == 10
        assert collector.total_tokens == 110
        assert collector.calls == 1

        usage = collector.to_token_usage()
        assert usage is not None
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 10
        assert usage.total_tokens == 110
        assert usage.calls == 1

    def test_total_tokens_zero_not_re_synthesized(self) -> None:
        """total_tokens=0 must be kept as 0, not replaced by prompt+completion sum."""
        from langchain_core.messages import AIMessage
        from langchain_core.messages.ai import UsageMetadata
        from langchain_core.outputs import ChatGeneration, LLMResult

        from evaluatorq.integrations.langgraph_integration.target import _TokenUsageCollector

        collector = _TokenUsageCollector()
        msg = AIMessage(
            content="hi",
            usage_metadata=UsageMetadata(input_tokens=50, output_tokens=50, total_tokens=0),
        )
        gen = ChatGeneration(message=msg)
        result = LLMResult(generations=[[gen]])
        collector.on_llm_end(result)

        assert collector.total_tokens == 0
        usage = collector.to_token_usage()
        assert usage is not None
        assert usage.total_tokens == 0

    def test_n_greater_than_1_no_double_count(self) -> None:
        """When n>1, only the first candidate in each inner list is counted."""
        from langchain_core.messages import AIMessage
        from langchain_core.messages.ai import UsageMetadata
        from langchain_core.outputs import ChatGeneration, LLMResult

        from evaluatorq.integrations.langgraph_integration.target import _TokenUsageCollector

        collector = _TokenUsageCollector()
        meta = UsageMetadata(input_tokens=20, output_tokens=5, total_tokens=25)
        msg1 = AIMessage(content="candidate 1", usage_metadata=meta)
        msg2 = AIMessage(content="candidate 2", usage_metadata=meta)
        gen1 = ChatGeneration(message=msg1)
        gen2 = ChatGeneration(message=msg2)
        # Both candidates are in the same inner list (same API call, n=2)
        result = LLMResult(generations=[[gen1, gen2]])
        collector.on_llm_end(result)

        # Only gen1 should be counted
        assert collector.calls == 1
        assert collector.prompt_tokens == 20
        assert collector.completion_tokens == 5
        assert collector.total_tokens == 25

    def test_on_llm_error_does_not_crash(self) -> None:
        """on_llm_error must not raise; to_token_usage returns None when nothing captured."""
        from evaluatorq.integrations.langgraph_integration.target import _TokenUsageCollector

        collector = _TokenUsageCollector()
        collector.on_llm_error(Exception("boom"))
        assert collector.to_token_usage() is None

    def test_on_llm_error_after_success_preserves_prior_usage(self) -> None:
        """on_llm_error must not wipe usage captured by a prior on_llm_end call."""
        from langchain_core.messages import AIMessage
        from langchain_core.messages.ai import UsageMetadata
        from langchain_core.outputs import ChatGeneration, LLMResult

        from evaluatorq.integrations.langgraph_integration.target import _TokenUsageCollector

        collector = _TokenUsageCollector()
        msg = AIMessage(
            content="ok",
            usage_metadata=UsageMetadata(input_tokens=10, output_tokens=2, total_tokens=12),
        )
        result = LLMResult(generations=[[ChatGeneration(message=msg)]])
        collector.on_llm_end(result)

        collector.on_llm_error(Exception("later error"))

        usage = collector.to_token_usage()
        assert usage is not None
        assert usage.calls == 1
        assert usage.prompt_tokens == 10

    @pytest.mark.asyncio
    async def test_callbacks_as_list_integration(self) -> None:
        """Collector is appended after existing list callbacks; both reach ainvoke."""
        from langchain_core.callbacks import BaseCallbackHandler

        class SentinelHandler(BaseCallbackHandler):
            pass

        sentinel = SentinelHandler()
        graph = _make_graph("response")
        target = LangGraphTarget(graph, config={"callbacks": [sentinel]})
        await target.send_prompt("hi")

        config_passed = graph.ainvoke.call_args[1]["config"]
        callbacks = config_passed["callbacks"]
        assert isinstance(callbacks, list)
        assert len(callbacks) == 2
        assert callbacks[0] is sentinel

        from evaluatorq.integrations.langgraph_integration.target import _TokenUsageCollector
        assert isinstance(callbacks[1], _TokenUsageCollector)

    @pytest.mark.asyncio
    async def test_callbacks_as_manager_not_mutated(self) -> None:
        """Original BaseCallbackManager must not be mutated across send_prompt calls.

        The implementation must copy the manager before adding the per-call collector
        so that stale collectors do not accumulate on the original instance or on
        .new() clones that share the same _extra_config reference.
        """
        from langchain_core.callbacks.manager import AsyncCallbackManager

        from evaluatorq.integrations.langgraph_integration.target import _TokenUsageCollector

        original_manager = MagicMock(spec=AsyncCallbackManager)
        # copy() must return a fresh mock so we can assert the original was not touched.
        manager_copy = MagicMock(spec=AsyncCallbackManager)
        original_manager.copy.return_value = manager_copy

        graph = _make_graph("response")
        target = LangGraphTarget(graph, config={"callbacks": original_manager})

        await target.send_prompt("first")
        await target.send_prompt("second")

        # copy() called once per send_prompt — never mutate the original.
        assert original_manager.copy.call_count == 2
        original_manager.add_handler.assert_not_called()

        # add_handler was called on the copy, not the original.
        assert manager_copy.add_handler.call_count == 2
        first_arg = manager_copy.add_handler.call_args_list[0][0][0]
        assert isinstance(first_arg, _TokenUsageCollector)

    @pytest.mark.asyncio
    async def test_new_yields_independent_token_state(self) -> None:
        """Parent and clone have independent _last_token_usage after respective calls."""
        from langchain_core.messages import AIMessage
        from langchain_core.messages.ai import UsageMetadata
        from langchain_core.outputs import ChatGeneration, LLMResult

        from evaluatorq.integrations.langgraph_integration.target import _TokenUsageCollector

        # Set up a graph that fires on_llm_end via a real collector invocation.
        # We use a custom ainvoke side-effect to simulate the callback being fired.
        captured_collectors: list[_TokenUsageCollector] = []

        def _fake_ainvoke(input_dict: dict[str, Any], *, config: dict[str, Any]) -> dict[str, Any]:
            callbacks = config.get("callbacks", [])
            for cb in (callbacks if isinstance(callbacks, list) else []):
                if isinstance(cb, _TokenUsageCollector):
                    captured_collectors.append(cb)
                    meta = UsageMetadata(input_tokens=7, output_tokens=3, total_tokens=10)
                    msg = AIMessage(content="ok", usage_metadata=meta)
                    gen = ChatGeneration(message=msg)
                    cb.on_llm_end(LLMResult(generations=[[gen]]))
            return {"messages": [MagicMock(content="ok")]}

        graph = MagicMock()
        graph.name = "test_graph"
        graph.ainvoke = AsyncMock(side_effect=_fake_ainvoke)

        parent = LangGraphTarget(graph)
        await parent.send_prompt("p1")
        parent_usage = parent._last_token_usage

        clone = parent.new()
        await clone.send_prompt("p2")
        clone_usage = clone._last_token_usage

        assert parent_usage is not None
        assert clone_usage is not None
        assert parent_usage is not clone_usage
        # Parent's state is unaffected by clone's call
        assert parent._last_token_usage is parent_usage

    @pytest.mark.asyncio
    async def test_usage_persisted_when_ainvoke_raises(self) -> None:
        """Token usage captured before ainvoke fails must still be recorded.

        on_llm_error can fire before ainvoke propagates the exception.
        The try/finally in send_prompt must persist the collector's state even
        when ainvoke raises, so error-path token spend is not silently dropped.
        """
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, LLMResult

        from evaluatorq.integrations.langgraph_integration.target import _TokenUsageCollector

        ai_msg = AIMessage(
            content="partial",
            usage_metadata={"input_tokens": 40, "output_tokens": 5, "total_tokens": 45},
        )
        gen = ChatGeneration(message=ai_msg)

        graph = MagicMock()
        graph.name = "test_graph"

        # Collector fires on_llm_end before ainvoke raises — simulate by calling
        # on_llm_end ourselves then having ainvoke raise.
        collector_ref: list[_TokenUsageCollector] = []

        async def failing_ainvoke(input: Any, config: Any) -> Any:  # noqa: A002
            # Grab the collector injected into config and fire on_llm_end on it.
            handlers = config.get("callbacks") or []
            for h in handlers:
                if isinstance(h, _TokenUsageCollector):
                    collector_ref.append(h)
                    h.on_llm_end(LLMResult(generations=[[gen]]))
            raise RuntimeError("provider error")

        graph.ainvoke = failing_ainvoke

        target = LangGraphTarget(graph)
        with pytest.raises(RuntimeError, match="provider error"):
            await target.send_prompt("hi")

        # Despite the exception, token usage must be stored.
        usage = target.consume_last_token_usage()
        assert usage is not None
        assert usage.prompt_tokens == 40
        assert usage.completion_tokens == 5

    @pytest.mark.asyncio
    async def test_consume_clears_state(self) -> None:
        """consume_last_token_usage() returns usage first call, None on second."""
        from langchain_core.messages import AIMessage
        from langchain_core.messages.ai import UsageMetadata
        from langchain_core.outputs import ChatGeneration, LLMResult

        from evaluatorq.integrations.langgraph_integration.target import _TokenUsageCollector

        def _fake_ainvoke(input_dict: dict[str, Any], *, config: dict[str, Any]) -> dict[str, Any]:
            callbacks = config.get("callbacks", [])
            for cb in (callbacks if isinstance(callbacks, list) else []):
                if isinstance(cb, _TokenUsageCollector):
                    meta = UsageMetadata(input_tokens=5, output_tokens=2, total_tokens=7)
                    msg = AIMessage(content="ok", usage_metadata=meta)
                    gen = ChatGeneration(message=msg)
                    cb.on_llm_end(LLMResult(generations=[[gen]]))
            return {"messages": [MagicMock(content="ok")]}

        graph = MagicMock()
        graph.name = "test_graph"
        graph.ainvoke = AsyncMock(side_effect=_fake_ainvoke)

        target = LangGraphTarget(graph)
        await target.send_prompt("hi")

        first = target.consume_last_token_usage()
        assert first is not None
        assert first.total_tokens == 7

        second = target.consume_last_token_usage()
        assert second is None

    @pytest.mark.asyncio
    async def test_no_usage_metadata_returns_none(self) -> None:
        """When graph fires no LLM callbacks, to_token_usage returns None."""
        graph = _make_graph("no usage")
        target = LangGraphTarget(graph)
        await target.send_prompt("hi")
        # ainvoke mock doesn't fire callbacks, so collector gets no calls
        assert target.consume_last_token_usage() is None
