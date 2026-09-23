"""The one place that decides what each Responses item type means.

A Responses ``input`` / ``output`` list is a flat run of typed items. Every
consumer needs the same answer to "is this text, a tool call, a tool result or
reasoning?", but renders it into a different shape: `AgentResponse.from_output_items`
builds canonical `OutputMessage` items, `otel_messages` builds OTel ``gen_ai``
message dicts. Both call `parse_item` and branch on its ``kind``, so a new item
type is taught here once and every consumer picks it up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from evaluatorq.common.fields import get_field

ItemKind = Literal['message', 'tool_call', 'tool_result', 'reasoning', 'unknown']

# Item types whose payload is a tool call. ``orq:<tool name>`` is matched by prefix.
_TOOL_CALL_TYPES = frozenset({'function_call', 'custom_tool_call', 'mcp_call'})
_TOOL_RESULT_TYPES = frozenset({'function_call_output', 'custom_tool_call_output'})

RESPONSES_ITEM_TYPES = frozenset({'message', 'reasoning', *_TOOL_CALL_TYPES, *_TOOL_RESULT_TYPES})


@dataclass(frozen=True)
class ParsedItem:
    """One Responses item, reduced to what every consumer needs.

    ``raw`` keeps the original item for consumers that read more than this
    (reasoning summaries, an unknown item's payload). ``result`` is ``None`` when
    the item carries no tool result.
    """

    kind: ItemKind
    raw: Any
    item_type: str = ''
    role: str | None = None
    content: Any = None
    call_id: str = ''
    name: str = ''
    arguments: Any = None
    result: Any = None


def is_orq_tool_call(item_type: Any) -> bool:
    """Orq's built-in tools are items typed ``orq:<tool name>`` — there is no single literal."""
    return isinstance(item_type, str) and item_type.startswith('orq:')


def is_responses_item(value: Any) -> bool:
    """Whether *value* is a typed Responses API item rather than a chat message or content part."""
    item_type = value.get('type') if isinstance(value, dict) else None
    return isinstance(item_type, str) and (item_type in RESPONSES_ITEM_TYPES or is_orq_tool_call(item_type))


def _item_type(item: Any) -> str:
    declared = get_field(item, 'type')
    if isinstance(declared, str) and declared:
        return declared
    # A bare {'role': ..., 'content': ...} dict is a message; the Responses API
    # accepts it as input shorthand and callers hand us that shape too.
    return 'message' if get_field(item, 'role') is not None else 'unknown'


def _tool_result(item: Any, item_type: str) -> Any:
    if item_type == 'mcp_call':
        # An MCP call carries its own outcome: an error wins, an empty output is no result.
        error = get_field(item, 'error')
        if error:
            return {'error': error}
        return get_field(item, 'output') or None
    result = get_field(item, 'result')
    return get_field(item, 'output') if result is None else result


def parse_item(item: Any) -> ParsedItem:
    """Classify one Responses item (a dict or an SDK model) by its ``type``."""
    item_type = _item_type(item)
    if item_type == 'message':
        return ParsedItem(
            kind='message',
            raw=item,
            item_type=item_type,
            role=get_field(item, 'role'),
            content=get_field(item, 'content'),
        )
    if item_type in _TOOL_CALL_TYPES or is_orq_tool_call(item_type):
        # The OpenAI SDK names an MCP tool in ``name``; Orq's gateway uses ``tool_name``.
        name = get_field(item, 'name') or get_field(item, 'tool_name')
        if not name and is_orq_tool_call(item_type):
            name = item_type.split(':', 1)[1]
        return ParsedItem(
            kind='tool_call',
            raw=item,
            item_type=item_type,
            call_id=str(get_field(item, 'call_id') or get_field(item, 'id') or ''),
            name=str(name or ''),
            arguments=get_field(item, 'input' if item_type == 'custom_tool_call' else 'arguments'),
            result=_tool_result(item, item_type),
        )
    if item_type in _TOOL_RESULT_TYPES:
        return ParsedItem(
            kind='tool_result',
            raw=item,
            item_type=item_type,
            call_id=str(get_field(item, 'call_id') or ''),
            result=get_field(item, 'output'),
        )
    if item_type == 'reasoning':
        return ParsedItem(kind='reasoning', raw=item, item_type=item_type)
    return ParsedItem(kind='unknown', raw=item, item_type=item_type)
