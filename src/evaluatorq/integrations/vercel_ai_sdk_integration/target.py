"""Red teaming target wrapper for Vercel AI SDK agents served over HTTP.

The Vercel AI SDK is a TypeScript library, so Python integration works by
calling an HTTP endpoint that serves the agent (e.g. a Next.js route handler).

The standard protocol:
- POST ``{"messages": [{"role": "user", "content": "..."}]}``
- Response uses the AI SDK Data Stream Protocol (text chunks prefixed ``0:``)
  or plain text/JSON.

See: https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from evaluatorq.redteam.backends.base import AgentTarget


class VercelAISdkTarget(AgentTarget):
    """Wraps a Vercel AI SDK HTTP endpoint as a red teaming target.

    The endpoint must accept POST requests with a JSON body containing
    ``messages`` in the standard chat format and return a response using
    the AI SDK Data Stream Protocol or plain text.

    Usage::

        from evaluatorq.integrations.vercel_ai_sdk_integration import VercelAISdkTarget

        # Point to your AI SDK agent endpoint
        target = VercelAISdkTarget("http://localhost:3000/api/chat")

        # With custom headers (e.g. authentication)
        target = VercelAISdkTarget(
            "https://my-app.vercel.app/api/chat",
            headers={"Authorization": "Bearer sk-..."},
        )

        # Pass to red teaming
        config = DynamicRunConfig(targets=[target])
    """

    memory_entity_id: str | None = None
    """Vercel AI SDK state lives inside the HTTP handler (stateless to us);
    conversation history is tracked client-side in ``_history``."""

    def __init__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float = 120.0,
    ) -> None:
        """Create a Vercel AI SDK red teaming target.

        Args:
            url: The HTTP endpoint URL serving the AI SDK agent.
            headers: Optional HTTP headers (e.g. for authentication).
            extra_body: Optional extra fields merged into the request body
                alongside ``messages`` (e.g. ``{"model": "gpt-4o"}``).
            timeout: HTTP request timeout in seconds.
        """
        self._url = url
        self._headers = headers or {}
        self._extra_body = extra_body or {}
        self._timeout = timeout
        self._history: list[dict[str, str]] = []

    async def send_prompt(self, prompt: str) -> str:
        """Send a prompt to the AI SDK endpoint and return its text response."""
        self._history.append({"role": "user", "content": prompt})

        body: dict[str, Any] = {
            "messages": list(self._history),
            **self._extra_body,
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                self._url,
                json=body,
                headers={"Content-Type": "application/json", **self._headers},
            )
            response.raise_for_status()

        text = self._parse_response(response)
        self._history.append({"role": "assistant", "content": text})
        return text

    def reset_conversation(self) -> None:
        """Reset conversation state by clearing the message history."""
        self._history = []

    def clone(self) -> VercelAISdkTarget:
        """Create an independent copy for parallel red teaming jobs."""
        return VercelAISdkTarget(
            self._url,
            headers=dict(self._headers),
            extra_body=dict(self._extra_body),
            timeout=self._timeout,
        )

    @staticmethod
    def _parse_response(response: httpx.Response) -> str:
        """Parse an AI SDK response, handling both Data Stream Protocol and plain formats.

        The Data Stream Protocol prefixes text chunks with ``0:`` and JSON-encodes
        them. Example stream::

            0:"Hello"
            0:" world"
            e:{"finishReason":"stop","usage":{"promptTokens":10,"completionTokens":5}}

        Plain text or JSON responses are returned as-is.
        """
        content_type = response.headers.get("content-type", "")
        raw = response.text

        # AI SDK Data Stream Protocol: lines prefixed with type codes
        if "text/plain" in content_type or raw.startswith("0:"):
            return _parse_data_stream(raw)

        # JSON response (e.g. {"message": {"content": "..."}} or {"text": "..."})
        if "application/json" in content_type:
            return _parse_json_response(raw)

        # Fallback: treat as plain text
        return raw.strip()


def _parse_data_stream(raw: str) -> str:
    """Extract text content from AI SDK Data Stream Protocol."""
    chunks: list[str] = []
    for line in raw.splitlines():
        if not line.startswith("0:"):
            continue
        # Text chunks are JSON-encoded strings after the prefix
        payload = line[2:]
        try:
            chunks.append(json.loads(payload))
        except (json.JSONDecodeError, TypeError):
            chunks.append(payload)
    return "".join(chunks)


def _parse_json_response(raw: str) -> str:
    """Extract text from common JSON response formats."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw.strip()

    # Common patterns from AI SDK endpoints
    if isinstance(data, str):
        return data

    if isinstance(data, dict):
        # {"message": {"content": "..."}} or {"message": "..."}
        msg = data.get("message")
        if isinstance(msg, dict):
            return str(msg.get("content", ""))
        if isinstance(msg, str):
            return msg

        # {"text": "..."} or {"content": "..."}
        for key in ("text", "content", "response", "output"):
            if key in data and isinstance(data[key], str):
                return data[key]

        # {"choices": [{"message": {"content": "..."}}]} (OpenAI-compat)
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            choice_msg = choices[0].get("message", {})
            if isinstance(choice_msg, dict):
                return str(choice_msg.get("content", ""))

    return raw.strip()
