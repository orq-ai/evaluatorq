"""Red teaming target wrapper for OpenAI Agents SDK."""

from __future__ import annotations

from typing import Any

from agents import Agent, Runner

from evaluatorq.redteam.backends.base import AgentTarget
from evaluatorq.redteam.contracts import AgentContext, ToolInfo


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

    async def send_prompt(self, prompt: str) -> str:
        """Send a prompt to the agent and return its text response."""
        input_data: str | list[Any] = prompt
        if self._history:
            input_data = [*self._history, {"role": "user", "content": prompt}]

        try:
            result = await Runner.run(self._agent, input_data, **self._run_kwargs)
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
        return str(result.final_output)

    def reset_conversation(self) -> None:
        """Reset conversation state by clearing the message history."""
        self._history = []

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

    def clone(self) -> OpenAIAgentTarget:
        """Create an independent copy for parallel red teaming jobs."""
        return OpenAIAgentTarget(self._agent, run_kwargs=dict(self._run_kwargs))
