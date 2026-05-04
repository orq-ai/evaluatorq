"""Red teaming target wrapper for OpenAI Agents SDK."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from agents import Agent, Runner

from evaluatorq.redteam.backends.base import AgentTarget
from evaluatorq.redteam.contracts import (
    AgentContext,
    AgentResponse,
    OutputMessage,
    TextOutputItem,
    ToolCallOutputItem,
    ToolInfo,
)


class OpenAIAgentTarget(AgentTarget):
    """Wraps an OpenAI Agents SDK Agent as a red teaming target.

    Usage::

        from agents import Agent
        from evaluatorq.integrations.openai_agents_integration import OpenAIAgentTarget

        agent = Agent(name="my-agent", instructions="You are a helpful assistant.")
        target = OpenAIAgentTarget(agent)

        # Pass to red teaming
        config = DynamicRunConfig(targets=[target])
    """

    memory_entity_id: str | None = None
    """OpenAI Agents SDK keeps conversation state client-side in ``_history``;
    there is no server-side memory entity to isolate across parallel jobs."""

    def __init__(self, agent: Agent, *, run_kwargs: dict[str, Any] | None = None) -> None:
        """Create an OpenAI Agents SDK red teaming target.

        Args:
            agent: An OpenAI Agents SDK Agent instance.
            run_kwargs: Optional extra keyword arguments passed to ``Runner.run()``
                (e.g. ``{"max_turns": 10}``).
        """
        self._agent = agent
        self._run_kwargs = run_kwargs or {}
        self._history: list[Any] = []

    async def send_prompt(self, prompt: str) -> AgentResponse:
        """Send a prompt to the agent and return its response with tool calls."""
        input_data: str | list[Any] = prompt
        prev_len = len(self._history)
        if self._history:
            input_data = [*self._history, {"role": "user", "content": prompt}]

        try:
            result = await Runner.run(self._agent, input_data, **self._run_kwargs)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            raise
        except Exception as exc:
            raise RuntimeError(
                f"OpenAIAgentTarget: Runner.run() raised an error: {exc}"
            ) from exc

        if result.final_output is None:
            raise ValueError(
                "OpenAIAgentTarget: Runner.run() returned final_output=None. "
                "Ensure the agent produces a text response."
            )

        self._history = result.to_input_list()

        # Build output items directly from new history items, preserving original
        # Responses API 'id'/'call_id' fields for trace correlation.
        output_items: list[OutputMessage] = []
        new_items_start = min(prev_len, len(self._history))
        for item in self._history[new_items_start:]:
            # Agents SDK to_input_list() returns Responses API items for tool calls,
            # e.g. {"type": "function_call", "id": "fc_...", "call_id": "call_...", "name": "...", "arguments": "..."}.
            if isinstance(item, dict) and item.get("type") == "function_call":
                item_id = item.get("id", "")
                call_id = item.get("call_id", "")
                name = str(item.get("name", ""))
                arguments = _normalize_args_str(item.get("arguments", "{}"))
                output_items.append(
                    ToolCallOutputItem(name=name, arguments=arguments, id=item_id, call_id=call_id)
                    if item_id and call_id
                    else ToolCallOutputItem(name=name, arguments=arguments)
                )
            # Handle dict format (standard OpenAI message format)
            elif isinstance(item, dict) and item.get("role") == "assistant":
                for tc in (item.get("tool_calls") or []):
                    if isinstance(tc, dict):
                        func = tc.get("function", {})
                        name = func.get("name", "") if isinstance(func, dict) else ""
                        args_raw = _normalize_args_str(func.get("arguments", "{}") if isinstance(func, dict) else "{}")
                        tc_id = tc.get("id", "")
                        output_items.append(
                            ToolCallOutputItem(name=name, arguments=args_raw, id=tc_id, call_id=tc_id)
                            if tc_id
                            else ToolCallOutputItem(name=name, arguments=args_raw)
                        )
            # Handle typed SDK objects (future-proofing if to_input_list() returns non-dict items)
            elif not isinstance(item, dict):
                if getattr(item, "type", None) == "function_call":
                    item_id = getattr(item, "id", "")
                    call_id = getattr(item, "call_id", "")
                    name = str(getattr(item, "name", ""))
                    arguments = _normalize_args_str(getattr(item, "arguments", "{}"))
                    output_items.append(
                        ToolCallOutputItem(name=name, arguments=arguments, id=item_id, call_id=call_id)
                        if item_id and call_id
                        else ToolCallOutputItem(name=name, arguments=arguments)
                    )
                elif getattr(item, "role", None) == "assistant":
                    for tc in (getattr(item, "tool_calls", None) or []):
                        tc_name = getattr(tc, "name", None) or getattr(getattr(tc, "function", None), "name", "") or ""
                        tc_args_raw = _normalize_args_str(getattr(tc, "arguments", None) or getattr(getattr(tc, "function", None), "arguments", "{}"))
                        tc_id = getattr(tc, "id", "")
                        output_items.append(
                            ToolCallOutputItem(name=str(tc_name), arguments=tc_args_raw, id=tc_id, call_id=tc_id)
                            if tc_id
                            else ToolCallOutputItem(name=str(tc_name), arguments=tc_args_raw)
                        )

        output_items.append(TextOutputItem(text=str(result.final_output), annotations=[]))
        return AgentResponse(output=output_items)

    async def get_agent_context(self) -> AgentContext:
        """Return agent context derived from the wrapped Agent instance.

        Maps the SDK ``Agent`` fields onto :class:`AgentContext`:
        ``name`` → ``key``/``display_name``, ``instructions`` → ``system_prompt``,
        ``model`` → ``model``, ``tools`` → ``tools`` (via duck-typed introspection).
        There is no server-side memory, so ``memory_stores`` stays empty.
        """
        agent = self._agent
        key = str(getattr(agent, "name", None) or "openai_agent")
        instructions = getattr(agent, "instructions", None)
        system_prompt = instructions if isinstance(instructions, str) else None

        model_attr = getattr(agent, "model", None)
        if isinstance(model_attr, str):
            model = model_attr
        elif model_attr is None:
            model = None
        else:
            model = getattr(model_attr, "model", None) or str(model_attr)

        tools: list[ToolInfo] = []
        for tool in getattr(agent, "tools", []) or []:
            name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
            if not name:
                continue
            params = getattr(tool, "params_json_schema", None)
            tools.append(
                ToolInfo(
                    name=str(name),
                    description=getattr(tool, "description", None),
                    parameters=params if isinstance(params, dict) else None,
                )
            )

        return AgentContext(
            key=key,
            display_name=key,
            description="OpenAI Agents SDK target",
            system_prompt=system_prompt,
            tools=tools,
            model=model,
        )

    def reset_conversation(self) -> None:
        """Clear accumulated conversation history for a fresh attack turn."""
        self._history = []

    def clone(self) -> OpenAIAgentTarget:
        """Return a fresh instance with empty history for parallel job safety."""
        return OpenAIAgentTarget(self._agent, run_kwargs=dict(self._run_kwargs))

    def new(self) -> OpenAIAgentTarget:
        """Return an independent instance for parallel red teaming jobs."""
        return self.clone()


def _normalize_args_str(raw: Any) -> str:
    """Convert SDK tool-call arguments to a valid JSON string.

    Handles str (valid or invalid JSON), dict, or other types.
    Invalid JSON strings are wrapped as ``{"raw": "<original>"}`` so that the
    ``AgentResponse.tool_calls`` property can always parse ``arguments`` back
    to a dict without silently discarding the original value.
    """
    if isinstance(raw, dict):
        return json.dumps(raw)
    if not isinstance(raw, str):
        return "{}"
    try:
        json.loads(raw)
        return raw
    except (json.JSONDecodeError, ValueError):
        return json.dumps({"raw": raw})
