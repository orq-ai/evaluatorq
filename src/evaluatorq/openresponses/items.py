"""The one place that decides what each Responses item type means.

A Responses ``input`` / ``output`` list is a flat run of typed items. Every
consumer needs the same answer to "is this text, a tool call, a tool result or
reasoning?", but renders it into a different shape: `AgentResponse.from_output_items`
builds canonical `OutputMessage` items, `otel_messages` builds OTel ``gen_ai``
message dicts, `common.trace_input` recovers item ids, `openresponses.dataset`
replays seed transcripts and the Agents SDK adapter builds replayable calls.
All of them call `parse_item` and branch on its ``kind``; message content parts
go through `classify_part` the same way.

`parse_item` also normalises the shapes callers hand it, so no consumer has to:
an SDK model becomes its dict, a message's string ``content`` becomes a one-part
list, and a tool result given as content parts becomes text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel

from evaluatorq.common.fields import get_field
from evaluatorq.common.messages import coerce_content_text, content_part_text

# Item types whose payload is a tool call. ``orq:<tool name>`` is matched by prefix.
_TOOL_CALL_TYPES = frozenset({'function_call', 'custom_tool_call', 'mcp_call'})
_TOOL_RESULT_TYPES = frozenset({'function_call_output', 'custom_tool_call_output'})
# Types whose outcome travels on the call item itself rather than in a separate result item.
_INLINE_RESULT_TYPES = frozenset({'mcp_call'})

RESPONSES_ITEM_TYPES = frozenset({'message', 'reasoning', *_TOOL_CALL_TYPES, *_TOOL_RESULT_TYPES})

_REASONING_PART_TYPES = frozenset({'reasoning_text', 'reasoning'})

PartKind = Literal['text', 'refusal', 'reasoning', 'other']


@dataclass(frozen=True)
class MessageItem:
    """A ``message`` item. ``content`` is always a list of part dicts."""

    role: str | None
    content: list[Any]
    kind: Literal['message'] = 'message'


@dataclass(frozen=True)
class ToolCallItem:
    """A tool call: ``function_call``, ``custom_tool_call``, ``mcp_call`` or ``orq:<tool>``.

    ``call_id`` pairs the call with its result and is ``None`` when the item has
    none. ``item_id`` is the item's own ``id``. ``result`` is set only for the
    types that carry their outcome inline (``mcp_call``, ``orq:*``, and a
    ``function_call`` recorded with a ``result``); ``None`` means no result.
    """

    item_type: str
    call_id: str | None
    item_id: str | None
    name: str
    arguments: Any
    result: Any
    kind: Literal['tool_call'] = 'tool_call'

    @property
    def ref(self) -> str:
        """The best identifier to show for the call: its ``call_id``, else its item ``id``."""
        return self.call_id or self.item_id or ''


@dataclass(frozen=True)
class ToolResultItem:
    """A separate tool result item. ``output`` is ``None`` when the item carries none."""

    item_type: str
    call_id: str | None
    output: Any
    kind: Literal['tool_result'] = 'tool_result'


@dataclass(frozen=True)
class ReasoningItem:
    """A ``reasoning`` item; ``raw`` holds its content, summary and encrypted payload."""

    raw: Any
    kind: Literal['reasoning'] = 'reasoning'


@dataclass(frozen=True)
class UnknownItem:
    """An item whose type no consumer understands. `parse_item` has already logged it."""

    item_type: str
    raw: Any
    kind: Literal['unknown'] = 'unknown'


ParsedItem = MessageItem | ToolCallItem | ToolResultItem | ReasoningItem | UnknownItem


def _is_orq_tool_call(item_type: str) -> bool:
    """Orq's built-in tools are items typed ``orq:<tool name>`` — there is no single literal."""
    return item_type.startswith('orq:')


def is_responses_item(value: Any) -> bool:
    """Whether *value* is a typed Responses API item rather than a chat message or content part."""
    item_type = get_field(value, 'type')
    return isinstance(item_type, str) and (item_type in RESPONSES_ITEM_TYPES or _is_orq_tool_call(item_type))


def _as_dict(item: Any) -> Any:
    """Return an SDK (pydantic) model as its dict so consumers read one shape; anything else unchanged."""
    return item.model_dump() if isinstance(item, BaseModel) else item


def _item_type(item: Any) -> str:
    declared = get_field(item, 'type')
    if isinstance(declared, str) and declared:
        return declared
    # A bare {'role': ..., 'content': ...} dict is a message; the Responses API
    # accepts it as input shorthand and callers hand us that shape too.
    has_role = 'role' in item if isinstance(item, dict) else isinstance(get_field(item, 'role'), str)
    return 'message' if has_role else 'unknown'


def _message_content(content: Any, role: str | None) -> list[Any]:
    if content is None or content == '':
        return []
    if isinstance(content, str):
        return [{'type': 'output_text' if role == 'assistant' else 'input_text', 'text': content}]
    return [_as_dict(part) for part in content] if isinstance(content, list) else [content]


def _result_value(value: Any) -> Any:
    """A tool result given as content parts is text to every reader, never a JSON dump of the parts."""
    if isinstance(value, list):
        return coerce_content_text([_as_dict(part) for part in value])
    return value


def _inline_result(item: Any, item_type: str) -> Any:
    if item_type in _INLINE_RESULT_TYPES:
        # An MCP call carries its own outcome: an error wins; an empty string is still a result.
        error = get_field(item, 'error')
        return {'error': error} if error else _result_value(get_field(item, 'output'))
    # ``orq:*`` items carry ``result``; evaluatorq's own recorded calls may carry ``result`` or ``output``.
    result = get_field(item, 'result')
    return _result_value(get_field(item, 'output') if result is None else result)


def _optional_str(value: Any) -> str | None:
    return str(value) if value else None


def parse_item(raw: Any) -> ParsedItem:
    """Classify one Responses item (a dict or an SDK model) by its ``type``.

    An unknown type, a tool call with no name and a result with no output are
    logged here, once, so consumers do not each decide whether to announce them.
    """
    item = _as_dict(raw)
    item_type = _item_type(item)
    if item_type == 'message':
        role = get_field(item, 'role')
        return MessageItem(role=role, content=_message_content(get_field(item, 'content'), role))
    if item_type in _TOOL_CALL_TYPES or _is_orq_tool_call(item_type):
        # The OpenAI SDK names an MCP tool in ``name``; Orq's gateway uses ``tool_name``.
        name = get_field(item, 'name') or get_field(item, 'tool_name')
        if not name and _is_orq_tool_call(item_type):
            name = item_type.split(':', 1)[1]
        item_id = _optional_str(get_field(item, 'id'))
        # An MCP call has no ``call_id``: its own ``id`` is the only handle on it.
        call_id = _optional_str(get_field(item, 'call_id')) or (item_id if item_type == 'mcp_call' else None)
        if not name:
            logger.warning('parse_item: {} item {!r} has no tool name', item_type, call_id or item_id)
        return ToolCallItem(
            item_type=item_type,
            call_id=call_id,
            item_id=item_id,
            name=str(name or ''),
            arguments=get_field(item, 'input' if item_type == 'custom_tool_call' else 'arguments'),
            result=_inline_result(item, item_type),
        )
    if item_type in _TOOL_RESULT_TYPES:
        output = get_field(item, 'output')
        if output is None:
            logger.warning('parse_item: {} for call_id={!r} has no output', item_type, get_field(item, 'call_id'))
        return ToolResultItem(
            item_type=item_type, call_id=_optional_str(get_field(item, 'call_id')), output=_result_value(output)
        )
    if item_type == 'reasoning':
        return ReasoningItem(raw=item)
    logger.warning('parse_item: unknown Responses item type={!r}', item_type)
    return UnknownItem(item_type=item_type, raw=item)


def classify_part(part: Any) -> tuple[PartKind, str]:
    """Classify one message content part and return its text (``''`` for ``other``)."""
    part_type = get_field(part, 'type')
    text = content_part_text(part)
    if part_type == 'refusal':
        return 'refusal', text or ''
    if text is not None:
        return 'text', text
    if part_type in _REASONING_PART_TYPES:
        value = get_field(part, 'text') or get_field(part, 'reasoning')
        return 'reasoning', value if isinstance(value, str) else ''
    return 'other', ''


def normalize_tool_arguments(raw: Any) -> str:
    """Return a JSON string for a Responses function call, preserving freeform custom input."""
    if isinstance(raw, dict):
        return json.dumps(raw)
    if not isinstance(raw, str):
        return '{}'
    try:
        json.loads(raw)
        return raw
    except (json.JSONDecodeError, ValueError):
        return json.dumps({'raw': raw})
