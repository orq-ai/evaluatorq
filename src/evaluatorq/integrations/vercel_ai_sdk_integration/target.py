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
from urllib.parse import urlparse

import httpx

from evaluatorq.redteam.backends.base import AgentTarget
from evaluatorq.redteam.contracts import AgentContext, AgentResponse, OutputMessage, TextOutputItem, TokenUsage


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
        agent_context: AgentContext | None = None,
    ) -> None:
        """Create a Vercel AI SDK red teaming target.

        Args:
            url: The HTTP endpoint URL serving the AI SDK agent.
            headers: Optional HTTP headers (e.g. for authentication).
            extra_body: Optional extra fields merged into the request body
                alongside ``messages`` (e.g. ``{"model": "gpt-4o"}``).
            timeout: HTTP request timeout in seconds.
            agent_context: Optional :class:`AgentContext` describing the
                remote agent's tools, memory, system prompt, etc. The red
                teaming pipeline uses this for capability-aware strategy
                filtering — without it, all strategies (including
                nonsensical ones) will be applied. The HTTP handler cannot
                be introspected from Python, so this must be supplied by
                the caller when capability-aware filtering matters.
        """
        self._url = url
        self._headers = headers or {}
        self._extra_body = extra_body or {}
        self._timeout = timeout
        self._history: list[dict[str, str]] = []
        self._agent_context = agent_context

    async def send_prompt(self, prompt: str) -> AgentResponse:
        """Send a prompt to the AI SDK endpoint and return a structured response with usage.

        The Vercel AI SDK Data Stream Protocol does not expose tool call details
        in a structured form, so ``tool_calls`` is always empty. The full text
        response is available via ``.text``.
        """
        self._history.append({"role": "user", "content": prompt})

        body: dict[str, Any] = {
            "messages": list(self._history),
            **self._extra_body,
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    self._url,
                    json=body,
                    headers={"Content-Type": "application/json", **self._headers},
                )
                response.raise_for_status()
        except (httpx.HTTPError, ValueError, KeyError):
            # Roll back the user turn so the next call doesn't forward a
            # malformed conversation with a hanging user message. Catches
            # HTTP errors (network/status) and parser errors (ValueError
            # from json.loads, KeyError from missing fields). Programming
            # bugs (TypeError, AttributeError) propagate without rollback
            # so they aren't masked as transient failures.
            self._history.pop()
            raise

        text, usage = self._parse_response(response)
        self._history.append({"role": "assistant", "content": text})
        text_item: OutputMessage = TextOutputItem(text=text, annotations=[])
        return AgentResponse(output=[text_item], usage=usage)

    async def get_agent_context(self) -> AgentContext:
        """Return the user-provided agent context, or a minimal placeholder."""
        if self._agent_context is not None:
            return self._agent_context
        # Strip userinfo and query/fragment so embedded credentials never leak
        # into logs/reports, and to keep the key free of URL-reserved chars
        # that may break downstream consumers (file naming, dict keying).
        parsed = urlparse(self._url)
        host = parsed.hostname or parsed.netloc.split("@")[-1]
        port = f":{parsed.port}" if parsed.port else ""
        safe_key = f"{parsed.scheme}://{host}{port}{parsed.path}".rstrip("/") or self._url
        return AgentContext(key=safe_key, description="opaque Vercel AI SDK HTTP target")

    def new(self) -> VercelAISdkTarget:
        """Return an independent instance for parallel red teaming jobs."""
        return VercelAISdkTarget(
            self._url,
            headers=dict(self._headers),
            extra_body=dict(self._extra_body),
            timeout=self._timeout,
            agent_context=self._agent_context,
        )

    @staticmethod
    def _parse_response(response: httpx.Response) -> tuple[str, TokenUsage | None]:
        """Parse an AI SDK response, handling both Data Stream Protocol and plain formats.

        The Data Stream Protocol prefixes text chunks with ``0:`` and JSON-encodes
        them. Example stream::

            0:"Hello"
            0:" world"
            e:{"finishReason":"stop","usage":{"promptTokens":10,"completionTokens":5}}

        Plain text or JSON responses are returned as-is.

        Returns:
            A ``(text, usage)`` tuple. ``usage`` is ``None`` when the response
            does not contain token counts.
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
        return raw.strip(), None


def _parse_data_stream(raw: str) -> tuple[str, TokenUsage | None]:
    """Extract text content and token usage from AI SDK Data Stream Protocol.

    Returns:
        A ``(text, usage)`` tuple. ``usage`` is ``None`` when no finish/done
        frame carrying usage data is found in the stream.
    """
    parts: list[str] = []
    usage: TokenUsage | None = None
    for line in raw.splitlines():
        if not line:
            continue
        prefix, _, payload = line.partition(':')
        if not _:
            continue
        if prefix == '0':
            # Text chunks are JSON-encoded strings after the prefix
            try:
                parts.append(json.loads(payload))
            except (json.JSONDecodeError, TypeError):
                parts.append(payload)
        elif prefix in ('e', 'd'):  # finish/done frames
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            u = obj.get('usage') if isinstance(obj, dict) else None
            if isinstance(u, dict):
                p = int(u.get('promptTokens', 0) or 0)
                c = int(u.get('completionTokens', 0) or 0)
                if p > 0 or c > 0:
                    t = u.get('totalTokens')
                    total = int(t) if isinstance(t, (int, float)) and t > 0 else p + c
                    usage = TokenUsage(
                        prompt_tokens=p,
                        completion_tokens=c,
                        total_tokens=total,
                        calls=1,
                    )
    return ''.join(parts), usage


def _parse_json_response(raw: str) -> tuple[str, TokenUsage | None]:
    """Extract text and token usage from common JSON response formats.

    Returns:
        A ``(text, usage)`` tuple. ``usage`` is ``None`` when no recognised
        usage block is present.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw.strip(), None

    usage: TokenUsage | None = None

    # Common patterns from AI SDK endpoints
    if isinstance(data, str):
        return data, None

    if isinstance(data, dict):
        # Extract usage if present — support both OpenAI shape
        # (prompt_tokens/completion_tokens) and Vercel shape (promptTokens/completionTokens)
        u = data.get("usage")
        if isinstance(u, dict):
            # OpenAI-compat shape
            p = int(u.get("prompt_tokens", 0) or 0)
            c = int(u.get("completion_tokens", 0) or 0)
            # Fall back to Vercel camelCase shape when snake_case is absent
            if p == 0 and c == 0:
                p = int(u.get("promptTokens", 0) or 0)
                c = int(u.get("completionTokens", 0) or 0)
            t = u.get("total_tokens") or u.get("totalTokens")
            total = int(t) if isinstance(t, (int, float)) and t > 0 else p + c
            if p > 0 or c > 0:
                usage = TokenUsage(
                    prompt_tokens=p,
                    completion_tokens=c,
                    total_tokens=total,
                    calls=1,
                )

        # {"message": {"content": "..."}} or {"message": "..."}
        msg = data.get("message")
        if isinstance(msg, dict):
            return str(msg.get("content", "")), usage
        if isinstance(msg, str):
            return msg, usage

        # {"text": "..."} or {"content": "..."}
        for key in ("text", "content", "response", "output"):
            if key in data and isinstance(data[key], str):
                return data[key], usage

        # {"choices": [{"message": {"content": "..."}}]} (OpenAI-compat)
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            choice_msg = choices[0].get("message", {})
            if isinstance(choice_msg, dict):
                return str(choice_msg.get("content", "")), usage

    return raw.strip(), usage
