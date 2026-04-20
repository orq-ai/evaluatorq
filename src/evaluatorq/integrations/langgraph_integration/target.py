"""Red teaming target wrapper for LangGraph agents."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from evaluatorq.redteam.backends.base import AgentTarget


class LangGraphTarget(AgentTarget):
    """Wraps a LangGraph CompiledStateGraph as a red teaming target.

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
        self._thread_id = uuid4().hex

    def _build_config(self) -> RunnableConfig:
        """Build the RunnableConfig with the current thread_id."""
        extra = {k: v for k, v in self._extra_config.items() if k != "configurable"}
        return RunnableConfig(
            **extra,
            configurable={
                **self._extra_config.get("configurable", {}),
                "thread_id": self._thread_id,
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
        """Reset conversation state by starting a new thread."""
        self._thread_id = uuid4().hex

    def clone(self, memory_entity_id: str | None = None) -> LangGraphTarget:
        """Create an independent copy for parallel red teaming jobs.

        If ``memory_entity_id`` is provided it is used as the thread_id,
        ensuring each memory entity gets its own isolated conversation thread.
        """
        cloned = LangGraphTarget(self._graph, config=dict(self._extra_config))
        if memory_entity_id is not None:
            cloned._thread_id = memory_entity_id
        return cloned
