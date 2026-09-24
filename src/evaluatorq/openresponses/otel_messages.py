"""Responses items -> OTel GenAI ``parts`` messages.

``gen_ai.input.messages`` / ``gen_ai.output.messages`` take a list of
``{role, parts:[{type, ...}]}`` objects, where a tool call is a *part* of an
assistant message. Responses items are a flat, differently-shaped list in which
``function_call``, ``function_call_output`` and ``reasoning`` entries carry a
``type`` but no ``role`` — every consumer that keys on ``role`` drops them, so
writing the raw items into those attributes silently loses the tool calls.

Derived from ``openResponsesItemToInputMessages`` /
``openResponsesItemToOutputMessages`` in orquesta-web
``libs/go/gateway/tracing.go``, which Orq's gateway runs before emitting a
Responses span. Which item type is a tool call, a tool result or reasoning is
decided by `evaluatorq.openresponses.items.parse_item`; this module only renders
that decision as OTel parts. Deliberate differences from the Go source: SDK model
items are accepted, an MCP tool name is read from ``name`` before ``tool_name``,
an ``orq:*`` item with no name is named from its type, and a result recorded on a
call item is emitted once even when a separate result item repeats it.
"""

from __future__ import annotations

import json
from typing import Any

from evaluatorq.common.fields import get_field
from evaluatorq.openresponses.items import ReasoningItem, ToolCallItem, classify_part, parse_item


def _parse_json_or_value(raw: Any) -> Any:
    """Return ``raw`` parsed as JSON when it is a JSON string, else unchanged."""
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return raw


def _generic_part(value: Any) -> dict[str, Any]:
    return {'type': 'data', 'content': value}


def _content_to_parts(content: Any, *, as_reasoning: bool = False) -> list[dict[str, Any]]:
    """Convert one item's ``content`` to OTel parts.

    ``as_reasoning`` retags text parts as ``reasoning``, matching the Go
    ``reasoningTaggedParts`` behaviour for reasoning items.
    """
    text_type = 'reasoning' if as_reasoning else 'text'
    if isinstance(content, str):
        return [{'type': text_type, 'content': content}] if content else []
    if not isinstance(content, list):
        return [] if content is None else [_generic_part(content)]

    parts: list[dict[str, Any]] = []
    for raw in content:
        kind, text = classify_part(raw)
        if kind == 'text':
            parts.append({'type': text_type, 'content': text})
        elif kind in ('reasoning', 'refusal'):
            parts.append({'type': kind, 'content': text})
        else:
            parts.append(_generic_part(raw))
    return parts


def _tool_call_part(item: ToolCallItem) -> dict[str, Any]:
    return {
        'type': 'tool_call',
        'id': item.ref,
        'name': item.name,
        'arguments': _parse_json_or_value(item.arguments),
    }


def _reasoning_parts(item: ReasoningItem) -> list[dict[str, Any]]:
    parts = _content_to_parts(get_field(item.raw, 'content'), as_reasoning=True)
    for summary_part in _content_to_parts(get_field(item.raw, 'summary'), as_reasoning=True):
        if summary_part not in parts:
            parts.append(summary_part)
    if not parts and get_field(item.raw, 'encrypted_content'):
        parts.append({'type': 'reasoning', 'content': '[encrypted]'})
    return parts


def _tool_response_message(call_id: Any, response: Any) -> dict[str, Any]:
    return {
        'role': 'tool',
        'parts': [{'type': 'tool_call_response', 'id': call_id or '', 'response': _parse_json_or_value(response)}],
    }


def items_to_input_messages(items: Any, *, default_role: str = 'user') -> list[dict[str, Any]]:
    """Convert Responses items to ``gen_ai.input.messages`` objects, tool results included.

    ``default_role`` is the role of a bare string and of a message with no role.
    A result item that repeats the result already recorded on its call item is
    skipped, so the call is answered once.
    """
    if isinstance(items, str):
        return [{'role': default_role, 'parts': [{'type': 'text', 'content': items}]}]
    if not isinstance(items, list):
        return []

    messages: list[dict[str, Any]] = []
    answered: dict[str, Any] = {}
    for raw in items:
        item = parse_item(raw)
        if item.kind == 'message':
            messages.append({'role': item.role or default_role, 'parts': _content_to_parts(item.content)})
        elif item.kind == 'tool_call':
            messages.append({'role': 'assistant', 'parts': [_tool_call_part(item)]})
            if item.result is not None:
                messages.append(_tool_response_message(item.ref, item.result))
                if item.call_id:
                    answered[item.call_id] = item.result
        elif item.kind == 'tool_result':
            if item.output is None or (item.call_id in answered and answered[item.call_id] == item.output):
                continue  # parse_item logged a missing output; a repeat adds nothing
            messages.append(_tool_response_message(item.call_id, item.output))
        elif item.kind == 'reasoning':
            messages.append({'role': 'assistant', 'parts': _reasoning_parts(item)})
        else:
            messages.append({'role': 'assistant', 'parts': [_generic_part(item.raw)]})
    return messages


def items_to_output_messages(items: Any, finish_reason: str = '') -> list[dict[str, Any]]:
    """Convert Responses output items to ``gen_ai.output.messages`` objects, for span emission.

    Tool *results* are dropped: an output list carries the model's calls, and the
    Go side omits them here too so a result is never attributed to the model.
    Anything that reads an output list back as a transcript (trace import) wants
    `items_to_input_messages`, which keeps them.
    """
    if not isinstance(items, list):
        return []

    messages: list[dict[str, Any]] = []
    for raw in items:
        item = parse_item(raw)
        if item.kind == 'tool_result':
            continue
        role = 'assistant'
        if item.kind == 'message':
            parts = _content_to_parts(item.content)
            role = item.role or 'assistant'
        elif item.kind == 'tool_call':
            parts = [_tool_call_part(item)]
        elif item.kind == 'reasoning':
            parts = _reasoning_parts(item)
        else:
            parts = [_generic_part(item.raw)]
        messages.append({'role': role, 'parts': parts, 'finish_reason': finish_reason})
    return messages
