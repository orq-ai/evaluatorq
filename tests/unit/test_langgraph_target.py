"""Unit tests for LangGraph red teaming target."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("langgraph")

from evaluatorq.integrations.langgraph_integration import LangGraphTarget  # noqa: E402


def _make_graph(response_content: str = "I'm fine") -> MagicMock:
    graph = MagicMock()
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
    async def test_send_prompt_passes_thread_id(self) -> None:
        graph = _make_graph()
        target = LangGraphTarget(graph)
        await target.send_prompt("hi")

        config = graph.ainvoke.call_args[1]["config"]
        assert "thread_id" in config["configurable"]

    @pytest.mark.asyncio
    async def test_reset_changes_thread_id(self) -> None:
        graph = _make_graph()
        target = LangGraphTarget(graph)
        old_thread = target._thread_id
        target.reset_conversation()
        assert target._thread_id != old_thread

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
        cloned = target.clone()
        assert cloned is not target
        assert cloned._thread_id != target._thread_id
        assert cloned._graph is graph
        assert cloned._extra_config is not target._extra_config
        assert cloned._extra_config == {"recursion_limit": 50}

    def test_clone_with_memory_entity_id_uses_it_as_thread_id(self) -> None:
        graph = _make_graph()
        target = LangGraphTarget(graph)
        cloned = target.clone(memory_entity_id="entity-abc")
        assert cloned._thread_id == "entity-abc"

    @pytest.mark.asyncio
    async def test_configurable_key_collision_preserves_user_keys(self) -> None:
        """config={"configurable": {"custom_key": "val"}} must not be overwritten by thread_id injection."""
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
