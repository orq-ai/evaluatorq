"""Tool-capable DemoAgent implementing evaluatorq.redteam.backends.base.AgentTarget.

AgentTarget protocol (from spike findings):
    async def send_prompt(self, prompt: str) -> str
    def reset_conversation(self) -> None

Agents hold their own conversation state. reset_conversation clears it.
"""

from __future__ import annotations

import json
import os
from typing import Any

from evaluatorq.redteam import AgentTarget
from evaluatorq.redteam.contracts import AgentContext, ToolInfo
from openai import AsyncOpenAI

from tools import ToolCall, run_shell, send_crypto, send_email

MAX_TOOL_ITERATIONS = 10

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email on Alice's behalf.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_crypto",
            "description": "Transfer cryptocurrency between known wallets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_id": {"type": "string"},
                    "to_id": {"type": "string"},
                    "amount": {"type": "string"},
                    "currency": {"type": "string", "default": "BTC"},
                },
                "required": ["from_id", "to_id", "amount"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Execute a shell command on the host system.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                },
                "required": ["command"],
            },
        },
    },
]


def _dispatch(name: str, arguments: dict[str, Any]) -> ToolCall:
    if name == "send_email":
        return send_email(**arguments)
    if name == "send_crypto":
        return send_crypto(**arguments)
    if name == "run_shell":
        return run_shell(**arguments)
    raise ValueError(f"unknown tool: {name}")


class DemoAgent(AgentTarget):
    """Tool-capable agent implementing evaluatorq's AgentTarget protocol.

    Subclasses override `system_prompt`.
    """

    system_prompt: str = ""
    model: str = "gpt-5-mini"

    def __init__(self) -> None:
        base_url = os.environ.get("ORQ_BASE_URL", "https://my.orq.ai").rstrip("/") + "/v2/proxy"
        self.client = AsyncOpenAI(base_url=base_url, api_key=os.environ["ORQ_API_KEY"])
        self._conversation: list[dict[str, Any]] = []
        self._last_tool_calls: list[ToolCall] = []

    def reset_conversation(self) -> None:
        self._conversation = []
        self._last_tool_calls = []

    def clone(self, memory_entity_id: str | None = None) -> "DemoAgent":
        return type(self)()

    async def get_agent_context(self) -> AgentContext:
        return AgentContext(
            key=type(self).__name__,
            system_prompt=self.system_prompt,
            tools=[
                ToolInfo(name=t["function"]["name"], description=t["function"]["description"]) for t in TOOL_SCHEMAS
            ],
        )

    async def send_prompt(self, prompt: str) -> str:
        if not self._conversation:
            self._conversation.append({"role": "system", "content": self.system_prompt})
        self._conversation.append({"role": "user", "content": prompt})
        self._last_tool_calls = []

        for _ in range(MAX_TOOL_ITERATIONS):
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=self._conversation,
                tools=TOOL_SCHEMAS,
            )
            msg = completion.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None)

            if not tool_calls:
                text = msg.content or ""
                self._conversation.append({"role": "assistant", "content": text})
                return text

            self._conversation.append(
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
                arguments = json.loads(tc.function.arguments or "{}")
                result = _dispatch(tc.function.name, arguments)
                self._last_tool_calls.append(result)
                self._conversation.append({"role": "tool", "tool_call_id": tc.id, "content": result.result})

        return "[max tool iterations reached]"
