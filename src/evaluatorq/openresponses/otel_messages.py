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
emitting a Responses span. Kept deliberately close to that source so the two can
be diffed; item types the gateway knows and we never emit (MCP, custom and Orq
tool calls) fall into the generic ``data`` branch rather than being guessed at.
"""

from __future__ import annotations

import json
from typing import Any

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


def _tool_call_part(item: dict[str, Any], *, name_key: str = 'name', args_key: str = 'arguments') -> dict[str, Any]:
    return {
        'type': 'tool_call',
        'id': item.get('call_id') or item.get('id') or '',
        'name': item.get(name_key) or '',
        'arguments': _parse_json_or_value(item.get(args_key)),
    }


def _reasoning_parts(item: dict[str, Any]) -> list[dict[str, Any]]:
    parts = _content_to_parts(item.get('content'), as_reasoning=True)
    for summary_part in _content_to_parts(item.get('summary'), as_reasoning=True):
        if summary_part not in parts:
            parts.append(summary_part)
    if not parts and item.get('encrypted_content'):
        parts.append({'type': 'reasoning', 'content': '[encrypted]'})
    return parts


def _is_orq_tool_call(item_type: str) -> bool:
    """Orq's built-in tools are items typed ``orq:<tool name>`` — there is no single literal."""
    return item_type.startswith('orq:')


def _tool_response_message(call_id: Any, response: Any) -> dict[str, Any]:
    return {
        'role': 'tool',
        'parts': [{'type': 'tool_call_response', 'id': call_id or '', 'response': _parse_json_or_value(response)}],
    }


def _item_type(item: dict[str, Any]) -> str:
    declared = item.get('type')
    if isinstance(declared, str) and declared:
        return declared
    # A bare {'role': ..., 'content': ...} dict is a message; the Responses API
    # accepts it as input shorthand and callers hand us that shape too.
    return 'message' if 'role' in item else 'unknown'


def items_to_input_messages(items: Any) -> list[dict[str, Any]]:
    """Convert Responses input items to ``gen_ai.input.messages`` objects."""
    if isinstance(items, str):
        return [{'role': 'user', 'parts': [{'type': 'text', 'content': items}]}]
    if not isinstance(items, list):
        return []

    messages: list[dict[str, Any]] = []
    for raw in items:
        if not isinstance(raw, dict):
            messages.append({'role': 'assistant', 'parts': [_generic_part(raw)]})
            continue
        item_type = _item_type(raw)
        if item_type == 'message':
            messages.append({'role': raw.get('role') or 'user', 'parts': _content_to_parts(raw.get('content'))})
        elif item_type == 'function_call':
            messages.append({'role': 'assistant', 'parts': [_tool_call_part(raw)]})
        elif item_type in {'function_call_output', 'custom_tool_call_output'}:
            messages.append(_tool_response_message(raw.get('call_id'), raw.get('output')))
        elif item_type == 'custom_tool_call':
            messages.append({'role': 'assistant', 'parts': [_tool_call_part(raw, args_key='input')]})
        elif item_type == 'mcp_call':
            messages.append({'role': 'assistant', 'parts': [_tool_call_part(raw, name_key='tool_name')]})
            error = raw.get('error')
            if error:
                messages.append(_tool_response_message(raw.get('call_id'), {'error': error}))
            elif raw.get('output'):
                messages.append(_tool_response_message(raw.get('call_id'), raw.get('output')))
        elif _is_orq_tool_call(item_type):
            messages.append({'role': 'assistant', 'parts': [_tool_call_part(raw)]})
            if raw.get('result') is not None:
                messages.append(_tool_response_message(raw.get('call_id'), raw.get('result')))
        elif item_type == 'reasoning':
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
        if not isinstance(raw, dict):
            messages.append({'role': 'assistant', 'parts': [_generic_part(raw)], 'finish_reason': finish_reason})
            continue
        item_type = _item_type(raw)
        if item_type in {'function_call_output', 'custom_tool_call_output'}:
            continue
        if item_type == 'message':
            parts = _content_to_parts(raw.get('content'))
            role = raw.get('role') or 'assistant'
        elif item_type == 'function_call' or _is_orq_tool_call(item_type):
            parts = [_tool_call_part(raw)]
            role = 'assistant'
        elif item_type == 'custom_tool_call':
            parts = [_tool_call_part(raw, args_key='input')]
            role = 'assistant'
        elif item_type == 'mcp_call':
            parts = [_tool_call_part(raw, name_key='tool_name')]
            role = 'assistant'
        elif item_type == 'reasoning':
            parts = _reasoning_parts(raw)
            role = 'assistant'
        else:
            parts = [_generic_part(raw)]
            role = 'assistant'
        messages.append({'role': role, 'parts': parts, 'finish_reason': finish_reason})
    return messages
