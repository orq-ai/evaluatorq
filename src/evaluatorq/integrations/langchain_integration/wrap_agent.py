"""Wrapper function for LangChain agents to integrate with evaluatorq."""

from __future__ import annotations

from typing import Any

from evaluatorq.types import DataPoint
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from .convert import convert_to_open_responses


def _extract_schema_parameters(
    schema_obj: type[BaseModel] | dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Extract JSON schema from Pydantic model or dict schema."""
    if schema_obj is None:
        return None
    # Already a dict schema
    if isinstance(schema_obj, dict):
        return schema_obj
    # Pydantic model class - use model_json_schema (Pydantic v2)
    return schema_obj.model_json_schema()


def wrap_langchain_agent(
    agent: CompiledStateGraph[Any, Any, Any, Any],
    *,
    name: str = "agent",
    prompt_key: str = "prompt",
    tools: list[dict[str, Any]] | None = None,
) -> Any:
    """
    Creates an evaluatorq Job from a LangChain agent.

    The job will:
    - Execute the agent with the prompt from data.inputs[prompt_key]
    - Convert the result to OpenResponses format (industry standard)
    - Return the OpenResponses resource for backend integration

    Args:
        agent: A LangChain agent or runnable with an invoke() method.
        name: The name of the job (defaults to "agent").
        prompt_key: The key in data.inputs to use as the prompt (defaults to "prompt").
        tools: Optional list of tool definitions for the OpenResponses output.

    Returns:
        An async function compatible with evaluatorq's Job type.

    Example:
        ```python
        from langchain_openai import ChatOpenAI
        from langchain.agents import create_agent
        from langchain_core.tools import tool
        from evaluatorq import evaluatorq
        from evaluatorq.integrations.langchain import wrap_langchain_agent

        @tool
        def weather(location: str) -> dict:
            \"\"\"Get the weather in a location.\"\"\"
            return {"location": location, "temperature": 72}

        model = ChatOpenAI(model="gpt-4o")
        agent = create_agent(model, tools=[weather])

        result = await evaluatorq("weather-agent-eval", {
            "data": [
                {"inputs": {"prompt": "What is the weather in SF?"}},
            ],
            "jobs": [wrap_langchain_agent(agent)],
            "evaluators": [
                {
                    "name": "response-quality",
                    "scorer": async def score(params):
                        result = params["output"]
                        # Access OpenResponses format
                        output_items = result.get("output", [])
                        has_message = any(
                            item.get("type") == "message"
                            for item in output_items
                        )
                        return {
                            "value": 1 if has_message else 0,
                            "explanation": "Agent produced a response",
                        }
                },
            ],
        })
        ```
    """

    async def job(data: DataPoint, _row: int) -> dict[str, Any]:
        prompt = data.inputs.get(prompt_key)

        if not isinstance(prompt, str):
            raise ValueError(
                f"Expected data.inputs.{prompt_key} to be a string, got {type(prompt).__name__}"
            )

        # Invoke the LangChain agent
        result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})

        # Extract messages from result
        messages: list[BaseMessage] = result.get("messages", [])

        # Get tools from agent if not provided explicitly
        resolved_tools = tools if tools is not None else extract_tools_from_agent(agent)

        # Convert to OpenResponses format
        open_responses_output = convert_to_open_responses(messages, resolved_tools)

        return {
            "name": name,
            "output": open_responses_output,
        }

    return job


def extract_tools_from_agent(agent: CompiledStateGraph[Any, Any, Any, Any]) -> list[dict[str, Any]]:
    """
    Extract tool definitions from a LangChain/LangGraph agent.

    For LangGraph CompiledStateGraph, iterates over nodes to find ToolNodes
    and extracts tool definitions from them.

    Args:
        agent: A LangChain agent or LangGraph compiled graph.

    Returns:
        A list of tool definitions in OpenResponses FunctionTool format.
    """
    tools: list[dict[str, Any]] = []

    # CompiledStateGraph has nodes attribute
    if not hasattr(agent, "nodes"):
        return tools

    for node in agent.nodes.values():
        bound = getattr(node, "bound", None)
        if bound is None:
            continue

        # Check for tools_by_name attribute (works for both ToolNode and _ToolNode)
        tools_by_name = getattr(bound, "tools_by_name", None)
        if not tools_by_name:
            continue

        for tool_obj in tools_by_name.values():
            if isinstance(tool_obj, BaseTool):
                tool_schema: dict[str, Any] = {
                    "type": "function",
                    "name": tool_obj.name,
                    "description": tool_obj.description,
                    "parameters": _extract_schema_parameters(tool_obj.args_schema),
                    "strict": True,
                }
                tools.append(tool_schema)

    return tools


# Alias for LangGraph users
wrap_langgraph_agent = wrap_langchain_agent
