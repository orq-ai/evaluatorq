"""Wrapper function for LangChain agents to integrate with evaluatorq."""

from __future__ import annotations

from collections.abc import Callable
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


def _normalize_message(msg: Any) -> dict[str, Any]:
    """Convert an Orq SDK message object or dict to a plain dict."""
    if isinstance(msg, dict):
        return msg
    # Pydantic BaseModel (Orq SDK message types)
    if hasattr(msg, "model_dump"):
        return msg.model_dump(exclude_none=True)
    # Fallback: duck-type role + content
    return {"role": getattr(msg, "role", "user"), "content": getattr(msg, "content", "")}


def _extract_messages_from_data(
    data: DataPoint,
) -> list[dict[str, Any]] | None:
    """Safely extract messages from a DataPoint.

    Returns None when data.inputs["messages"] is missing, not a list, or empty.
    Each message is normalized to a plain dict to handle Orq SDK Pydantic models.
    """
    messages = data.inputs.get("messages")
    if not isinstance(messages, list) or len(messages) == 0:
        return None
    return [_normalize_message(m) for m in messages]


def wrap_langchain_agent(
    agent: CompiledStateGraph[Any, Any, Any, Any],
    *,
    name: str = "agent",
    prompt_key: str = "prompt",
    instructions: str | Callable[[DataPoint], str] | None = None,
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
        instructions: System instructions to prepend to the messages sent to the agent.
            Can be a static string or a callable that receives the data point and returns
            the instructions.
        tools: Optional list of tool definitions for the OpenResponses output.

    Returns:
        An async function compatible with evaluatorq's Job type.
    """

    async def job(data: DataPoint, _row: int) -> dict[str, Any]:
        input_messages = _extract_messages_from_data(data)
        has_messages = input_messages is not None
        prompt = data.inputs.get(prompt_key)
        has_prompt = isinstance(prompt, str) and bool(prompt)

        if instructions is not None:
            # Resolve instructions (static string or dynamic callable)
            resolved = instructions(data) if callable(instructions) else instructions
            system_message: dict[str, str] = {"role": "system", "content": resolved}

            if has_messages and has_prompt:
                messages = [system_message, *input_messages, {"role": "user", "content": prompt}]
            elif has_prompt:
                messages = [system_message, {"role": "user", "content": prompt}]
            elif has_messages:
                messages = [system_message, *input_messages]
            else:
                raise ValueError(
                    f"Expected data.inputs.messages (list) or data.inputs.{prompt_key} (str), but neither was provided"
                )
        elif has_messages and has_prompt:
            messages = [*input_messages, {"role": "user", "content": prompt}]
        elif has_prompt:
            messages = [{"role": "user", "content": prompt}]
        elif has_messages:
            messages = list(input_messages)
        else:
            raise ValueError(
                f"Expected data.inputs.messages (list) or data.inputs.{prompt_key} (str), but neither was provided"
            )

        # Invoke the LangChain agent
        result = agent.invoke({"messages": messages})

        # Extract messages from result
        result_messages: list[BaseMessage] = result.get("messages", [])

        # Get tools from agent if not provided explicitly
        resolved_tools = tools if tools is not None else extract_tools_from_agent(agent)

        # Convert to OpenResponses format
        open_responses_output = convert_to_open_responses(result_messages, resolved_tools)

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
