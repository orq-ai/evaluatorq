"""Simulation target implementations backed by the Orq Responses API."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from openai import AsyncOpenAI

from evaluatorq.contracts import AgentResponse, LLMCallConfig, OutputMessage, TextOutputItem, ToolCallOutputItem
from evaluatorq.simulation.types import ChatMessage, TokenUsage


class OrqResponsesTarget:
    """Wraps the Orq Responses v3 API as a simulation target.

    Implements both the sim callable shape (__call__) and the redteam
    AgentTarget protocol (send_prompt + new).

    Multi-turn state is threaded via previous_response_id — each instance
    tracks its own conversation thread. Call new() to start fresh.
    """

    memory_entity_id: str | None  # AgentTarget protocol field

    def __init__(
        self,
        config: LLMCallConfig,
        *,
        instructions: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        memory_entity_id: str | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.config = config
        self.instructions = instructions
        self.tools = tools
        self.memory_entity_id = memory_entity_id
        self._previous_response_id: str | None = None
        self._accumulated_usage = TokenUsage()
        if client is not None:
            self._client = client
            self._client_owned = False
        else:
            self._client = self._build_client()

    # ---------------------------------------------------------------------------
    # Public API — sim callable shape
    # ---------------------------------------------------------------------------

    async def __call__(self, messages: list[ChatMessage]) -> str:
        """Sim target_callback shape. Sends full message list each turn."""
        result = await self._invoke(input_=self._messages_to_input(messages))
        return result.text

    # ---------------------------------------------------------------------------
    # Public API — redteam AgentTarget protocol
    # ---------------------------------------------------------------------------

    async def send_prompt(self, prompt: str) -> AgentResponse:
        """Redteam AgentTarget protocol shape."""
        return await self._invoke(input_=prompt)

    def new(self) -> OrqResponsesTarget:
        """Return a fresh instance with identical config but cleared state.

        Externally-injected clients (client_owned=False) are propagated to the
        new instance so callers sharing a single HTTP connection continue to do
        so. Self-owned clients are not propagated — the new instance builds its
        own from env vars, keeping connection lifetimes independent.
        """
        return OrqResponsesTarget(
            self.config,
            instructions=self.instructions,
            tools=self.tools,
            memory_entity_id=self.memory_entity_id,
            client=self._client if not self._client_owned else None,
        )

    # ---------------------------------------------------------------------------
    # Core invocation
    # ---------------------------------------------------------------------------

    async def _invoke(self, *, input_: str | list[dict[str, Any]]) -> AgentResponse:
        """Core invocation: calls client.responses.create, threads previous_response_id."""
        timeout_s = self.config.timeout_ms / 1000.0

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "input": input_,
            "tools": self.tools or [],
        }
        if self.instructions is not None:
            kwargs["instructions"] = self.instructions
        if self._previous_response_id is not None:
            kwargs["previous_response_id"] = self._previous_response_id

        response = await asyncio.wait_for(
            self._client.responses.create(**kwargs),
            timeout=timeout_s,
        )

        # Thread the conversation via previous_response_id
        response_id = getattr(response, "id", None)
        if isinstance(response_id, str) and response_id:
            self._previous_response_id = response_id

        # Accumulate token usage
        usage_obj = getattr(response, "usage", None)
        input_toks = int(getattr(usage_obj, "input_tokens", 0) or 0)
        output_toks = int(getattr(usage_obj, "output_tokens", 0) or 0)
        self._accumulated_usage = TokenUsage(
            prompt_tokens=self._accumulated_usage.prompt_tokens + input_toks,
            completion_tokens=self._accumulated_usage.completion_tokens + output_toks,
            total_tokens=self._accumulated_usage.total_tokens + input_toks + output_toks,
        )

        # Extract output items
        output_items = self._extract_output(response)

        return AgentResponse(output=output_items)

    # ---------------------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------------------

    def _extract_output(self, response: Any) -> list[OutputMessage]:
        """Extract ordered output items from a Responses API response object."""
        items: list[OutputMessage] = []
        raw_output = getattr(response, "output", None) or []

        for item in raw_output:
            item_type = getattr(item, "type", None)

            # Text message output items: type == "message", carry a content list
            if item_type == "message":
                content = getattr(item, "content", None) or []
                for part in content:
                    part_type = getattr(part, "type", None)
                    if part_type == "output_text":
                        text = getattr(part, "text", None)
                        if isinstance(text, str):
                            items.append(TextOutputItem(text=text, annotations=[]))
                    elif hasattr(part, "text"):
                        # Fallback: any part with a text attribute
                        text = getattr(part, "text", None)
                        if isinstance(text, str):
                            items.append(TextOutputItem(text=text, annotations=[]))

            # Function call output items: type == "function_call"
            elif item_type == "function_call":
                name = getattr(item, "name", None) or ""
                arguments = getattr(item, "arguments", None) or "{}"
                call_id = getattr(item, "call_id", None) or getattr(item, "id", None) or ""
                result = getattr(item, "result", None)
                items.append(
                    ToolCallOutputItem(
                        name=str(name),
                        arguments=arguments if isinstance(arguments, str) else "{}",
                        call_id=str(call_id),
                        result=str(result) if result is not None else None,
                    )
                )

        return items

    def _build_client(self) -> AsyncOpenAI:
        """Construct an AsyncOpenAI client from config or environment variables.

        Resolution order:
        1. ``self.config.client`` — injected client, used as-is (not owned).
        2. ``ORQ_API_KEY`` env var — routes through the Orq router.
        3. ``OPENAI_API_KEY`` env var — uses the OpenAI SDK default base URL.
        """
        if self.config.client is not None:
            self._client_owned = False
            return self.config.client  # type: ignore[return-value]

        orq_api_key = os.environ.get("ORQ_API_KEY")
        resolved_api_key = orq_api_key or os.environ.get("OPENAI_API_KEY")

        if not resolved_api_key:
            raise ValueError(
                "No API key found. Set ORQ_API_KEY or OPENAI_API_KEY, "
                "or pass a pre-built client in LLMCallConfig."
            )

        base_url: str | None = (
            f"{os.environ.get('ORQ_BASE_URL', 'https://api.orq.ai')}/v2/router"
            if orq_api_key
            else None
        )

        self._client_owned = True
        return AsyncOpenAI(base_url=base_url, api_key=resolved_api_key)

    @staticmethod
    def _messages_to_input(messages: list[ChatMessage]) -> list[dict[str, Any]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def get_usage(self) -> TokenUsage:
        """Return cumulative token usage across all _invoke calls on this instance."""
        return self._accumulated_usage.model_copy()


__all__ = ["OrqResponsesTarget"]
