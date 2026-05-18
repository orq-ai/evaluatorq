"""OpenResponses output parsing for red teaming.

Extracts the agent's textual response and tool calls from an OpenResponses
``ResponseResource`` (or compatible shape) so safety classifiers and judges
can score it. Accepts:

- ``ResponseResource`` dicts (raw OpenResponses wire format)
- ``AgentResponse`` instances (the internal canonical shape; its
  ``output`` items already match OpenResponses item discriminators:
  ``output_text``, ``function_call``, ``reasoning``)
- Pydantic ``Message`` / ``FunctionCall`` instances from
  ``evaluatorq.openresponses.convert_models``

Used by:
- Judges and safety classifiers that consume a single ``response: str``.
- Multi-turn orchestrators that need to append the agent's reply back into
  an OpenResponses ``input`` array.
"""

from __future__ import annotations

from typing import Any

from evaluatorq.contracts import (
    AgentResponse,
    ReasoningOutputItem,
    TextOutputItem,
    ToolCallOutputItem,
)


def _iter_output_items(response: Any) -> list[Any]:
    """Return the ordered list of output items from any OpenResponses-shaped value.

    Handles:
    - ``AgentResponse`` → ``.output``
    - ``ResponseResource`` dict → ``response["output"]``
    - bare ``list`` → returned as-is
    """
    if isinstance(response, AgentResponse):
        return list(response.output)
    if isinstance(response, dict):
        items = response.get("output")
        if isinstance(items, list):
            return items
        return []
    if isinstance(response, list):
        return list(response)
    return []


def _item_type(item: Any) -> str | None:
    """Read the OpenResponses ``type`` discriminator from any item shape."""
    if isinstance(item, dict):
        t = item.get("type")
        return str(t) if t is not None else None
    return getattr(item, "type", None)


def _text_from_message_dict(item: dict[str, Any]) -> str:
    """Extract concatenated ``output_text`` content from a ``message`` item dict.

    OpenResponses ``message`` items carry a ``content`` list where each block
    may be ``{"type": "output_text", "text": "..."}``. Other block types
    (annotations, etc.) are skipped.
    """
    parts: list[str] = []
    for block in item.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "output_text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
        elif getattr(block, "type", None) == "output_text":
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def extract_assistant_text(response: Any) -> str:
    """Extract the concatenated assistant-visible text from an OpenResponses response.

    This is the canonical input to safety classifiers / red team judges that
    expect a single string. Reasoning and function-call items are ignored —
    only ``output_text`` content (from ``message`` items) and direct
    ``TextOutputItem`` content (from ``AgentResponse``) contribute.

    Returns the empty string when the response carries no textual output.
    """
    parts: list[str] = []
    for item in _iter_output_items(response):
        # Internal canonical shape: TextOutputItem (OutputTextContent)
        if isinstance(item, TextOutputItem):
            parts.append(item.text)
            continue
        # OpenResponses wire shape: message item with output_text content blocks
        item_type = _item_type(item)
        if item_type == "message" and isinstance(item, dict):
            parts.append(_text_from_message_dict(item))
            continue
        # OpenResponses wire shape: bare output_text content block
        if item_type == "output_text" and isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
            continue
    return "".join(parts)


def extract_tool_calls(response: Any) -> list[dict[str, Any]]:
    """Extract function/tool calls from an OpenResponses response.

    Returns a list of ``{"name", "arguments", "call_id"}`` dicts. ``arguments``
    is the raw JSON string as produced by the model (callers that need a
    parsed dict should use ``json.loads`` and handle malformed JSON
    explicitly).
    """
    calls: list[dict[str, Any]] = []
    for item in _iter_output_items(response):
        if isinstance(item, ToolCallOutputItem):
            calls.append({
                "name": item.name,
                "arguments": item.arguments,
                "call_id": item.call_id,
            })
            continue
        if _item_type(item) == "function_call" and isinstance(item, dict):
            calls.append({
                "name": str(item.get("name", "")),
                "arguments": str(item.get("arguments", "")),
                "call_id": str(item.get("call_id") or item.get("id") or ""),
            })
    return calls


def extract_reasoning(response: Any) -> list[str]:
    """Extract reasoning trace text from an OpenResponses response, if present."""
    out: list[str] = []
    for item in _iter_output_items(response):
        if isinstance(item, ReasoningOutputItem):
            out.append(item.text)
            continue
        if _item_type(item) == "reasoning" and isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str):
                out.append(text)
    return out


__all__ = [
    "extract_assistant_text",
    "extract_reasoning",
    "extract_tool_calls",
]
