"""Red teaming target wrapper for LangGraph agents."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from evaluatorq.redteam.backends.base import AgentTarget


class LangGraphTarget(AgentTarget):
    """Wraps a LangGraph CompiledStateGraph as a red teaming target.

    Each instance generates its own ``memory_entity_id`` used as the LangGraph
    ``thread_id`` — this is the checkpointer's isolation key, so parallel
    attacks never share thread state. The pipeline reads ``memory_entity_id``
    off the target rather than injecting it.

    Usage::

        from langgraph.prebuilt import create_react_agent
        from evaluatorq.integrations.langgraph_integration import LangGraphTarget

        graph = create_react_agent(model, tools=[...])
        target = LangGraphTarget(graph)

        # Pass to red teaming
        config = DynamicRunConfig(targets=[target])
    """

    def __init__(
        self,
        graph: CompiledStateGraph[Any, Any, Any, Any],
        *,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Create a LangGraph red teaming target.

        Args:
            graph: A compiled LangGraph state graph.
            config: Optional extra LangGraph RunnableConfig keys
                (e.g. ``{"recursion_limit": 50}``). The ``thread_id``
                is managed automatically — do not pass it here.
        """
        self._graph = graph
        self._extra_config = config or {}
        self.memory_entity_id: str = uuid4().hex

    def _build_config(self) -> RunnableConfig:
        """Build the RunnableConfig with the current thread_id."""
        extra = {k: v for k, v in self._extra_config.items() if k != "configurable"}
        return RunnableConfig(
            **extra,
            configurable={
                **self._extra_config.get("configurable", {}),
                "thread_id": self.memory_entity_id,
            },
        )

    async def send_prompt(self, prompt: str) -> str:
        """Send a prompt to the LangGraph agent and return its text response."""
        result = await self._graph.ainvoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config=self._build_config(),
        )
        messages = result.get("messages")
        if messages is None:
            raise ValueError(
                "LangGraphTarget requires a graph whose state has a 'messages' key "
                "(e.g. built with MessagesState). Got state keys: "
                + str(list(result.keys()))
            )
        if not messages:
            raise ValueError(
                "LangGraphTarget: graph returned an empty 'messages' list. "
                "Ensure every execution path appends at least one AI message."
            )
        last = messages[-1]
        # Support both dict and LangChain BaseMessage
        if isinstance(last, dict):
            content = last.get("content", "")
        else:
            content = getattr(last, "content", "")
        if not isinstance(content, str):
            content = str(content)
        return content

    def reset_conversation(self) -> None:
        """Reset conversation state. Thread ID is fixed at construction; clone() provides isolation."""
        pass

    def clone(self) -> LangGraphTarget:
        """Create an independent copy for parallel red teaming jobs.

        The clone gets a fresh ``memory_entity_id`` (and thus a fresh LangGraph
        thread), so parallel workers never share checkpointer state.
        """
        return LangGraphTarget(self._graph, config=dict(self._extra_config))
