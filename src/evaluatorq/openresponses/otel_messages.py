"""Responses items -> OTel GenAI ``parts`` messages.

``gen_ai.input.messages`` / ``gen_ai.output.messages`` take a list of
``{role, parts:[{type, ...}]}`` objects, where a tool call is a *part* of an
assistant message. Responses items are a flat, differently-shaped list in which
``function_call``, ``function_call_output`` and ``reasoning`` entries carry a
``type`` but no ``role`` — every consumer that keys on ``role`` drops them, so
writing the raw items into those attributes silently loses the tool calls.

This is a port of ``openResponsesItemToInputMessages`` /
``openResponsesItemToOutputMessages`` in orquesta-web
``libs/go/gateway/tracing.go``, which is what Orq's own gateway runs before
emitting a Responses span. Which item type is a tool call, a tool result or
reasoning is decided by `evaluatorq.openresponses.items.parse_item`; this module
only renders that decision as OTel parts.
"""

from __future__ import annotations

import json
from typing import Any

from evaluatorq.common.fields import get_field
from evaluatorq.openresponses.items import ParsedItem, parse_item

_TEXT_PART_TYPES = frozenset({'input_text', 'output_text', 'text', 'summary_text'})
_REASONING_PART_TYPES = frozenset({'reasoning_text', 'reasoning'})


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
        if not isinstance(raw, dict):
            parts.append(_generic_part(raw))
            continue
        part_type = raw.get('type')
        if part_type in _TEXT_PART_TYPES:
            parts.append({'type': text_type, 'content': raw.get('text') or ''})
        elif part_type in _REASONING_PART_TYPES:
            parts.append({'type': 'reasoning', 'content': raw.get('text') or raw.get('reasoning') or ''})
        elif part_type == 'refusal':
            parts.append({'type': 'refusal', 'content': raw.get('refusal') or ''})
        else:
            parts.append(_generic_part(raw))
    return parts


def _tool_call_part(item: ParsedItem) -> dict[str, Any]:
    return {
        'type': 'tool_call',
        'id': item.call_id,
        'name': item.name,
        'arguments': _parse_json_or_value(item.arguments),
    }


def _reasoning_parts(item: Any) -> list[dict[str, Any]]:
    parts = _content_to_parts(get_field(item, 'content'), as_reasoning=True)
    for summary_part in _content_to_parts(get_field(item, 'summary'), as_reasoning=True):
        if summary_part not in parts:
            parts.append(summary_part)
    if not parts and get_field(item, 'encrypted_content'):
        parts.append({'type': 'reasoning', 'content': '[encrypted]'})
    return parts


def _tool_response_message(call_id: Any, response: Any) -> dict[str, Any]:
    return {
        'role': 'tool',
        'parts': [{'type': 'tool_call_response', 'id': call_id or '', 'response': _parse_json_or_value(response)}],
    }


def items_to_input_messages(items: Any) -> list[dict[str, Any]]:
    """Convert Responses input items to ``gen_ai.input.messages`` objects."""
    if isinstance(items, str):
        return [{'role': 'user', 'parts': [{'type': 'text', 'content': items}]}]
    if not isinstance(items, list):
        return []

    messages: list[dict[str, Any]] = []
    for raw in items:
        item = parse_item(raw)
        if item.kind == 'message':
            messages.append({'role': item.role or 'user', 'parts': _content_to_parts(item.content)})
        elif item.kind == 'tool_call':
            messages.append({'role': 'assistant', 'parts': [_tool_call_part(item)]})
            if item.result is not None:
                messages.append(_tool_response_message(item.call_id, item.result))
        elif item.kind == 'tool_result':
            messages.append(_tool_response_message(item.call_id, item.result))
        elif item.kind == 'reasoning':
            messages.append({'role': 'assistant', 'parts': _reasoning_parts(raw)})
        else:
            messages.append({'role': 'assistant', 'parts': [_generic_part(raw)]})
    return messages


def items_to_output_messages(items: Any, finish_reason: str = '') -> list[dict[str, Any]]:
    """Convert Responses output items to ``gen_ai.output.messages`` objects.

    Tool *results* are dropped: an output list carries the model's calls, and the
    Go side omits them here too so a result is never attributed to the model.
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
            parts = _reasoning_parts(raw)
        else:
            parts = [_generic_part(raw)]
        messages.append({'role': role, 'parts': parts, 'finish_reason': finish_reason})
    return messages
