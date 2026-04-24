"""Red teaming target wrapper for LangGraph agents."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.callbacks.base import BaseCallbackManager
from langchain_core.outputs import LLMResult
from langchain_core.runnables import RunnableConfig

from evaluatorq.redteam.backends.base import AgentTarget
from evaluatorq.redteam.contracts import AgentContext, MemoryStoreInfo, TokenUsage, ToolInfo

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


class _TokenUsageCollector(BaseCallbackHandler):
    """Sync LangChain callback handler that accumulates token usage across LLM calls.

    Subclasses sync ``BaseCallbackHandler`` (not ``AsyncCallbackHandler``) so it
    is invoked correctly from both ``invoke`` and ``ainvoke`` call paths.
    """

    def __init__(self) -> None:
        super().__init__()
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self.calls: int = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Accumulate token usage from a completed LLM call."""
        try:
            for inner in response.generations:
                if not inner:
                    continue
                # Only read first candidate — higher-n sampling reuses the same
                # underlying API call; iterating all candidates would multiply-count.
                gen = inner[0]
                meta = getattr(getattr(gen, "message", None), "usage_metadata", None)
                if meta is None:
                    continue
                prompt = int(meta.get("input_tokens")) if meta.get("input_tokens") is not None else 0
                completion = int(meta.get("output_tokens")) if meta.get("output_tokens") is not None else 0
                raw_total = meta.get("total_tokens")
                # CRITICAL: use `is not None`, not truthiness — total_tokens=0 is valid.
                total = int(raw_total) if raw_total is not None else prompt + completion
                self.prompt_tokens += prompt
                self.completion_tokens += completion
                self.total_tokens += total
                self.calls += 1
        except Exception as exc:
            logger.warning("_TokenUsageCollector.on_llm_end: failed to extract usage: %s", exc)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Handle LLM errors — try to extract partial usage, otherwise no-op."""
        try:
            # Some providers attach a partial LLMResult / usage on the exception.
            response = getattr(error, "response", None)
            if response is not None:
                partial_result = getattr(response, "llm_result", None) or getattr(response, "llm_output", None)
                if isinstance(partial_result, LLMResult):
                    self.on_llm_end(partial_result)
        except Exception as exc:
            logger.warning("_TokenUsageCollector.on_llm_error: failed to extract partial usage: %s", exc)

    def to_token_usage(self) -> TokenUsage | None:
        """Return aggregated usage, or None if no calls were recorded."""
        if self.calls == 0:
            return None
        return TokenUsage(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
            calls=self.calls,
        )


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
        agent_context: AgentContext | None = None,
    ) -> None:
        """Create a LangGraph red teaming target.

        Args:
            graph: A compiled LangGraph state graph.
            config: Optional extra LangGraph RunnableConfig keys
                (e.g. ``{"recursion_limit": 50}``). The ``thread_id``
                is managed automatically — do not pass it here.
            agent_context: Optional :class:`AgentContext` override. When
                provided, this context is returned from
                :meth:`get_agent_context` verbatim. When omitted, the target
                introspects the compiled graph (tools from a ``ToolNode``,
                checkpointer presence) on a best-effort basis.
        """
        self._graph = graph
        self._extra_config = config or {}
        self.memory_entity_id: str = uuid4().hex
        self._agent_context = agent_context
        # NOTE: _last_token_usage is not safe for concurrent send_prompt calls on
        # the same instance — use .new() to get independent instances for parallel use.
        self._last_token_usage: TokenUsage | None = None
        graph_name: str = getattr(graph, "name", None) or "langgraph_target"
        self._key = f"{graph_name}_{uuid4().hex[:8]}"

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
        collector = _TokenUsageCollector()

        base_config = self._build_config()
        existing = base_config.get("callbacks")

        if existing is None:
            new_callbacks: Any = [collector]
        elif isinstance(existing, list):
            new_callbacks = [*existing, collector]
        elif isinstance(existing, BaseCallbackManager) and hasattr(existing, "copy"):
            # Copy before mutating — avoid accumulating stale collectors on the
            # original manager across repeated send_prompt calls and .new() clones.
            manager_copy = existing.copy()
            manager_copy.add_handler(collector, inherit=True)
            new_callbacks = manager_copy
        else:
            # Unknown type — wrap alongside existing without mutating.
            logger.warning(
                "LangGraphTarget: unrecognised callbacks type %s; wrapping in list. "
                "Pass a list or BaseCallbackManager instead.",
                type(existing).__name__,
            )
            new_callbacks = [existing, collector]

        new_config: RunnableConfig = {**base_config, "callbacks": new_callbacks}

        try:
            result = await self._graph.ainvoke(
                {"messages": [{"role": "user", "content": prompt}]},
                config=new_config,
            )
        finally:
            # Always persist usage — including when ainvoke raises — so that
            # error-path token spend is not silently dropped.
            self._last_token_usage = collector.to_token_usage()

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
        content = last.get("content", "") if isinstance(last, dict) else getattr(last, "content", "")
        if not isinstance(content, str):
            content = str(content)
        return content

    def consume_last_token_usage(self) -> TokenUsage | None:
        """Return and clear token usage from the last send_prompt() call.

        Must be called between send_prompt() calls to avoid silently dropping
        usage from earlier calls — each new send_prompt() overwrites the stored
        value regardless of whether it has been consumed.
        """
        usage = self._last_token_usage
        self._last_token_usage = None
        return usage

    async def get_agent_context(self) -> AgentContext:
        """Return agent context introspected from the compiled graph.

        If an ``agent_context`` was passed at construction time, returns it
        unchanged. Otherwise performs best-effort introspection:

        * tools: extracted from any node whose ``bound`` is a ``ToolNode`` via
          ``tools_by_name``
        * memory_stores: single synthetic entry pointing at the checkpointer
          thread when a checkpointer is attached

        Callers needing stronger guarantees (custom graph shapes, non-standard
        tool nodes) should pass ``agent_context`` explicitly.
        """
        if self._agent_context is not None:
            return self._agent_context

        tools = _introspect_tools(self._graph)
        memory_stores = _introspect_memory_stores(self._graph, self.memory_entity_id)

        return AgentContext(
            key=self._key,
            description="LangGraph compiled state graph target",
            tools=tools,
            memory_stores=memory_stores,
        )

    def new(self) -> LangGraphTarget:
        """Return an independent instance for parallel red teaming jobs.

        Each call gets a fresh ``memory_entity_id`` (and thus a fresh LangGraph
        thread), so parallel workers never share checkpointer state.
        """
        cloned = LangGraphTarget(
            self._graph,
            config=dict(self._extra_config),
            agent_context=self._agent_context,
        )
        cloned._key = self._key
        return cloned


def _introspect_tools(graph: CompiledStateGraph[Any, Any, Any, Any]) -> list[ToolInfo]:
    """Best-effort extraction of tool metadata from a compiled graph.

    Looks for nodes whose ``bound`` attribute exposes ``tools_by_name``
    (LangGraph's ``ToolNode``). Falls back to an empty list on anything else.
    """
    tools: list[ToolInfo] = []
    seen: set[str] = set()
    nodes = getattr(graph, "nodes", None)
    if not nodes:
        return tools
    for node in nodes.values():
        bound = getattr(node, "bound", None)
        tools_by_name = getattr(bound, "tools_by_name", None)
        if not isinstance(tools_by_name, dict):
            continue
        for name, tool in tools_by_name.items():
            name_str = str(name)
            if name_str in seen:
                continue
            seen.add(name_str)
            tools.append(
                ToolInfo(
                    name=name_str,
                    description=getattr(tool, "description", None),
                    parameters=None,
                )
            )
    return tools


def _introspect_memory_stores(
    graph: CompiledStateGraph[Any, Any, Any, Any], thread_id: str
) -> list[MemoryStoreInfo]:
    """Return a single synthetic memory store entry when a checkpointer is attached."""
    checkpointer = getattr(graph, "checkpointer", None)
    if checkpointer is None:
        return []
    return [
        MemoryStoreInfo(
            id=thread_id,
            key="langgraph_checkpointer",
            description=f"LangGraph checkpointer ({type(checkpointer).__name__})",
        )
    ]
