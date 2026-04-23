"""Unit tests for LangGraph red teaming target."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("langgraph")

from evaluatorq.integrations.langgraph_integration import LangGraphTarget  # noqa: E402
from evaluatorq.redteam.contracts import AgentResponse  # noqa: E402


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
        assert isinstance(result, AgentResponse)
        assert result.text == "hello back"

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
        target.reset_conversation()
        assert target.memory_entity_id == old_thread

    @pytest.mark.asyncio
    async def test_reset_conversation_resets_prev_msg_count(self) -> None:
        graph = _make_graph()
        target = LangGraphTarget(graph)
        await target.send_prompt("hi")
        assert target._prev_msg_count > 0
        target.reset_conversation()
        assert target._prev_msg_count == 0

    @pytest.mark.asyncio
    async def test_multi_turn_tool_calls_excludes_previous_turns(self) -> None:
        """Turn N must not include tool calls from turns 1..N-1 (checkpointer accumulates state)."""
        tool_a = MagicMock()
        tool_a.content = "turn 1 result"
        tool_a.tool_calls = [{"name": "tool_A", "args": {"x": 1}}]

        final_1 = MagicMock()
        final_1.content = "done turn 1"
        final_1.tool_calls = []

        tool_b = MagicMock()
        tool_b.content = "turn 2 result"
        tool_b.tool_calls = [{"name": "tool_B", "args": {"y": 2}}]

        final_2 = MagicMock()
        final_2.content = "done turn 2"
        final_2.tool_calls = []

        call_count = 0

        async def fake_ainvoke(state, config):  # noqa: ANN001
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"messages": [tool_a, final_1]}
            # Checkpointer returns full accumulated state
            return {"messages": [tool_a, final_1, tool_b, final_2]}

        graph = MagicMock()
        graph.name = "test"
        graph.ainvoke = fake_ainvoke

        target = LangGraphTarget(graph)
        r1 = await target.send_prompt("first")
        assert len(r1.tool_calls) == 1
        assert r1.tool_calls[0].name == "tool_A"

        r2 = await target.send_prompt("second")
        assert len(r2.tool_calls) == 1
        assert r2.tool_calls[0].name == "tool_B"

    @pytest.mark.asyncio
    async def test_reset_then_send_extracts_all_tool_calls(self) -> None:
        """After reset, _prev_msg_count=0 so all messages are treated as new."""
        msg_with_tool = MagicMock()
        msg_with_tool.content = "result"
        msg_with_tool.tool_calls = [{"name": "tool_X", "args": {}}]

        graph = MagicMock()
        graph.name = "test"
        graph.ainvoke = AsyncMock(return_value={"messages": [msg_with_tool]})

        target = LangGraphTarget(graph)
        await target.send_prompt("first")  # _prev_msg_count becomes 1
        target.reset_conversation()        # _prev_msg_count resets to 0

        # Next send scans from index 0 — all returned messages are "new"
        result = await target.send_prompt("after reset")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "tool_X"

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
        assert result.text == "dict msg"

    def test_clone_returns_independent_instance(self) -> None:
        graph = _make_graph()
        target = LangGraphTarget(graph, config={"recursion_limit": 50})
        cloned = target.clone()
        assert cloned is not target
        assert cloned.memory_entity_id != target.memory_entity_id
        assert cloned._graph is graph
        assert cloned._extra_config is not target._extra_config
        assert cloned._extra_config == {"recursion_limit": 50}

    def test_clone_gets_fresh_memory_entity_id(self) -> None:
        graph = _make_graph()
        target = LangGraphTarget(graph)
        cloned = target.clone()
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
        assert isinstance(result.text, str)
        assert "multimodal content" in result.text


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
