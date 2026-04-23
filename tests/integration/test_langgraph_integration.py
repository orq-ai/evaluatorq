"""Integration test: LangGraphTarget with a real LangGraph agent using a fake LLM.

This creates a real LangGraph ReAct-style graph but uses FakeListChatModel
so no API keys or network calls are needed. It validates that the full
wiring works: graph creation -> invoke -> response extraction -> reset -> clone.
"""

from __future__ import annotations

import pytest

pytest.importorskip("langgraph")

from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402
from langgraph.graph import END, MessagesState, StateGraph  # noqa: E402
from langgraph.graph.state import CompiledStateGraph  # noqa: E402

from evaluatorq.integrations.langgraph_integration import LangGraphTarget  # noqa: E402


def _build_echo_graph() -> CompiledStateGraph:  # pyright: ignore[reportMissingTypeArgument]
    """Build a minimal LangGraph that uses a fake LLM to respond."""
    model = FakeListChatModel(responses=["I am a helpful assistant.", "Sure, I can help with that."])

    def agent_node(state: MessagesState) -> dict[str, object]:
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)
    return graph.compile()


class TestLangGraphIntegration:
    @pytest.mark.asyncio
    async def test_send_prompt_gets_response(self) -> None:
        """Full round-trip: send a prompt through a real graph, get a response."""
        graph = _build_echo_graph()
        target = LangGraphTarget(graph)

        response = await target.send_prompt("Hello")
        assert isinstance(response.text, str)
        assert len(response.text) > 0

    @pytest.mark.asyncio
    async def test_reset_starts_fresh_conversation(self) -> None:
        """After reset, the graph should not see previous messages."""
        graph = _build_echo_graph()
        target = LangGraphTarget(graph)

        await target.send_prompt("First message")
        target.reset_conversation()
        response = await target.send_prompt("Second message")

        assert isinstance(response.text, str)
        assert len(response.text) > 0

    @pytest.mark.asyncio
    async def test_clone_works_independently(self) -> None:
        """Cloned targets should work independently."""
        graph = _build_echo_graph()
        target = LangGraphTarget(graph)
        cloned = target.clone()

        response_original = await target.send_prompt("Hello from original")
        response_cloned = await cloned.send_prompt("Hello from clone")

        assert isinstance(response_original.text, str)
        assert isinstance(response_cloned.text, str)
