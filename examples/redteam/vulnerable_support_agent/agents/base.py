"""Tool-capable support agent implementing evaluatorq's AgentTarget protocol.

AgentTarget (evaluatorq.contracts) is message-based: the caller owns the
transcript and calls

    async def respond(self, messages: list[Message]) -> AgentResponse

Each subclass overrides ``system_prompt``. The target is stateless — it replays
the provided messages, prepends its own system prompt, and runs a local
tool-call loop against the orq router via AsyncOpenAI. ``new()`` returns a fresh
instance for parallel attack jobs.
"""

from __future__ import annotations

import json
from typing import Any

from evaluatorq.common.retry import without_client_retries
from evaluatorq.contracts import (
    AgentContext,
    AgentResponse,
    AgentResponseError,
    AgentTarget,
    Message,
    TextOutputItem,
    ToolInfo,
)
from evaluatorq.redteam.backends.registry import create_async_llm_client
from loguru import logger
from openai import AsyncOpenAI

from config import TARGET_MODEL
from tools import ToolCall, issue_refund, lookup_order

MAX_TOOL_ITERATIONS = 10

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up an order and its customer details by order id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Issue a refund against an order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "amount": {"type": "string"},
                },
                "required": ["order_id", "amount"],
            },
        },
    },
]


def _dispatch(name: str, arguments: Any) -> ToolCall:
    # Explicit args only: the model must not be able to inject extra kwargs.
    if not isinstance(arguments, dict):
        raise TypeError(f"arguments must be a JSON object, got {type(arguments).__name__}")
    if name == "lookup_order":
        return lookup_order(order_id=arguments["order_id"])
    if name == "issue_refund":
        return issue_refund(order_id=arguments["order_id"], amount=arguments["amount"])
    raise ValueError(f"unknown tool: {name}")


class SupportAgent(AgentTarget):
    """Tool-capable agent implementing evaluatorq's ``AgentTarget`` protocol."""

    system_prompt: str = ""
    model: str = TARGET_MODEL

    def __init__(self, *, client: AsyncOpenAI | None = None) -> None:
        # call_target_with_retry is the single retry owner for target calls, so
        # the client used here disables its own SDK retries.
        super().__init__(memory_entity_id=None)
        self.client = without_client_retries(client) if client is not None else create_async_llm_client(max_retries=0)

    def new(self) -> SupportAgent:
        return type(self)(client=self.client)

    async def get_agent_context(self) -> AgentContext:
        return AgentContext(
            key=type(self).__name__,
            display_name=type(self).__name__,
            system_prompt=self.system_prompt,
            tools=[
                ToolInfo(name=t["function"]["name"], description=t["function"]["description"]) for t in TOOL_SCHEMAS
            ],
        )

    async def respond(self, messages: list[Message]) -> AgentResponse:
        # Stateless: prepend our own system prompt, then replay the caller's
        # transcript (dropping any leading system message to avoid duplication).
        user_visible = [m for m in messages if m.role != "system"]
        conversation: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            *[m.to_chat_completion() for m in user_visible],
        ]

        text = ""
        for _ in range(MAX_TOOL_ITERATIONS):
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=conversation,
                tools=TOOL_SCHEMAS,
            )
            msg = completion.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None)

            if not tool_calls:
                text = msg.content or ""
                break

            conversation.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in tool_calls
                    ],
                }
            )

            for tc in tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments or "{}")
                    content = _dispatch(tc.function.name, arguments).result
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    # Feed the failure back so the model can recover instead of aborting the turn.
                    content = f"error: {type(exc).__name__}: {exc}"
                conversation.append({"role": "tool", "tool_call_id": tc.id, "content": content})
        else:
            message = f"max tool iterations ({MAX_TOOL_ITERATIONS}) reached without a final answer"
            logger.warning("{}: {}", type(self).__name__, message)
            return AgentResponse(
                output=[TextOutputItem(text=f"[{message}]", annotations=[])],
                error=AgentResponseError(message=message, error_type="target_error"),
            )

        return AgentResponse(text=text)
