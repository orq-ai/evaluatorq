"""Shared helpers for normalizing chat-message content.

Canonical for every surface. Content is typed ``str | list[ContentPart]``:
never call ``str()`` on it — that renders a Python repr into a transcript a
judge then scores. Use these helpers or ``contracts.content_to_text``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Iterable


_TEXT_PART_TYPES = frozenset({'text', 'input_text', 'output_text', 'summary_text', 'refusal'})

_REASONING_PART_TYPES = frozenset({'reasoning', 'reasoning_content', 'thinking'})
_MEDIA_PART_TYPES = {
    'image': 'image',
    'image_url': 'image',
    'input_image': 'image',
    'file': 'file',
    'input_file': 'file',
    'audio': 'audio',
    'input_audio': 'audio',
    'output_audio': 'audio',
    'video': 'video',
    'input_video': 'video',
}


@dataclass(frozen=True)
class MediaContent:
    """A recognized non-text content part that renderers can safely summarize."""

    kind: str
    value: Any


def _field(part: Any, key: str) -> Any:
    return part.get(key) if isinstance(part, Mapping) else getattr(part, key, None)


def content_part_text(part: Any) -> str | None:
    """Read text only from a recognized provider text-part type."""
    part_type = _field(part, 'type')
    if not isinstance(part_type, str) or part_type not in _TEXT_PART_TYPES:
        return None
    keys = ('refusal', 'text', 'content') if part_type == 'refusal' else ('text', 'content')
    for key in keys:
        if isinstance(value := _field(part, key), str):
            return value
    return ''


def _json_or_text(value: Any) -> str:
    """Render mappings and typed models as JSON instead of a Python repr."""
    if isinstance(value, Mapping):
        return json.dumps(dict(value), default=str)
    if isinstance(value, (list, tuple)):
        return json.dumps(value, default=str)
    model_dump = getattr(value, 'model_dump', None)
    if callable(model_dump):
        return json.dumps(model_dump(), default=str)
    return '' if value is None else str(value)


def _media_kind(part: Any) -> str | None:
    part_type = _field(part, 'type')
    kind = _MEDIA_PART_TYPES.get(part_type) if isinstance(part_type, str) else None
    if kind:
        return kind
    image_url = _field(part, 'image_url')
    return 'image' if part_type is None and _is_image_url_shape(image_url) else None


def _render_part(part: Any) -> Iterator[str | MediaContent]:
    part_type = _field(part, 'type')
    if isinstance(part_type, str) and part_type in _REASONING_PART_TYPES:
        return
    if (text := content_part_text(part)) is not None:
        yield text
        return
    if kind := _media_kind(part):
        yield MediaContent(kind, part)
        return
    yield _json_or_text(part)


def _is_image_url_shape(image_url: Any) -> bool:
    if isinstance(image_url, str):
        return bool(image_url)
    if isinstance(image_url, Mapping):
        return any(isinstance(image_url.get(key), str) and image_url[key] for key in ('url', 'data'))
    return any(isinstance(value := getattr(image_url, key, None), str) and value for key in ('url', 'data'))


def _is_content_part(part: Any) -> bool:
    if isinstance(part, str):
        return True
    part_type = _field(part, 'type')
    return (
        isinstance(part_type, str)
        and (part_type in _TEXT_PART_TYPES or part_type in _REASONING_PART_TYPES or part_type in _MEDIA_PART_TYPES)
    ) or _media_kind(part) is not None


def iter_content_parts(content: Any) -> Iterator[str | MediaContent]:
    """Yield text and recognized media parts in source order for every renderer.

    Yields:
        Text strings and typed media parts in source order.
    """
    if content is None:
        return
    if isinstance(content, (list, tuple)):
        if not all(_is_content_part(part) for part in content if part is not None):
            yield _json_or_text(content)
            return
        for part in content:
            if part is not None:
                yield from _render_part(part)
        return
    yield from _render_part(content)


def contains_media_content(content: Any) -> bool:
    """Whether content contains a recognized media part rather than structured business data."""
    parts = content if isinstance(content, (list, tuple)) else (content,)
    return any(_media_kind(part) is not None for part in parts if part is not None)


def reasoning_text(message: Mapping[str, Any]) -> str:
    """Join a message's reasoning: its `reasoning_content`, `reasoning` or `thinking` field, then reasoning parts.

    Content renderers skip reasoning parts; this is the one place that reads them.
    """
    # Providers mirror one reasoning text under several names, so each source contributes its first string field.
    keys = ('reasoning_content', 'reasoning', 'thinking')
    texts = [next((value for key in keys if isinstance(value := message.get(key), str)), '')]
    content = message.get('content')
    for part in content if isinstance(content, (list, tuple)) else ():
        if _field(part, 'type') not in _REASONING_PART_TYPES:
            continue
        part_keys = ('thinking', 'text', 'reasoning', 'content')
        texts.append(next((value for key in part_keys if isinstance(value := _field(part, key), str)), ''))
        summary = _field(part, 'summary')
        if isinstance(summary, list):
            texts.extend(text for item in summary if isinstance(text := _field(item, 'text'), str))
    return '\n\n'.join(text for text in texts if text.strip())


def coerce_content_text(content: Any) -> str:
    """Flatten content, preserving structured values as JSON and marking known media."""
    rendered = (f'[{part.kind}]' if isinstance(part, MediaContent) else part for part in iter_content_parts(content))
    return '\n'.join(rendered)


UNKNOWN_ARGUMENTS = ''
UNKNOWN_ARGUMENTS_TEXT = 'arguments unavailable'


def _tool_call_text(call: Any) -> str:
    """Render one tool call as a marker a judge can read."""
    function = call.get('function') if isinstance(call, dict) else getattr(call, 'function', None)
    source = function if function is not None else call
    if isinstance(source, dict):
        name, arguments = source.get('name'), source.get('arguments')
    else:
        name, arguments = getattr(source, 'name', None), getattr(source, 'arguments', None)
    rendered = arguments if isinstance(arguments, str) else _json_or_text(arguments)
    if arguments == UNKNOWN_ARGUMENTS:
        rendered = UNKNOWN_ARGUMENTS_TEXT
    return f'[tool_call: {name or "unknown"}({rendered or ""})]'


def _legacy_result_text(result: Any) -> str:
    """Render a legacy ``function_call_output`` value as text.

    A mapping that is not itself the result carries it under ``output`` or
    ``result``; anything still structured is rendered as JSON rather than a
    Python repr, which a judge would otherwise score verbatim.
    """
    value = result.get('output', result.get('result', result)) if isinstance(result, dict) else result
    return _json_or_text(value)


def messages_to_text(messages: Iterable[Any]) -> str:
    """Render a message list as one plain-text transcript.

    Canonical for every surface that hands a recorded conversation to a judge or
    a scorer. Three properties the naive ``''.join(content)`` it replaced lacked:
    messages are newline-separated (they used to be glued into one run-on word),
    tool calls are rendered rather than dropped (a turn that only called a tool
    used to render as the empty string, so a working agent scored as silent), and
    non-assistant turns are labelled (tool JSON used to read as the agent's own
    answer). A single assistant text message still renders as its bare text.

    Content is rendered exactly as recorded — ``strip()`` decides only whether a
    turn is blank, never what it contains, because an exact-match scorer reading
    a single-message output would otherwise score trimmed text the trace never
    had. The legacy ``function_call`` / ``function_call_output`` fields render
    alongside ``tool_calls``: `has_meaningful_output` counts them, so skipping
    them here would hand a judge an empty string for an agent that acted.
    """
    lines: list[str] = []
    for message in messages:
        if isinstance(message, dict):
            role, content, tool_calls = message.get('role'), message.get('content'), message.get('tool_calls')
            legacy_call, legacy_result = message.get('function_call'), message.get('function_call_output')
        else:
            role = getattr(message, 'role', None)
            content = getattr(message, 'content', None)
            tool_calls = getattr(message, 'tool_calls', None)
            legacy_call = getattr(message, 'function_call', None)
            legacy_result = getattr(message, 'function_call_output', None)
        text = coerce_content_text(content)
        parts = [text] if text.strip() else []
        calls = [*(tool_calls or []), *([legacy_call] if legacy_call is not None else [])]
        parts.extend(_tool_call_text(call) for call in calls)
        if legacy_result is not None:
            parts.append(f'[tool_result: {_legacy_result_text(legacy_result)}]')
        if not parts:
            continue
        body = '\n'.join(parts)
        lines.append(body if role in (None, 'assistant') else f'[{role}] {body}')
    return '\n'.join(lines)


def parse_json_or_value(raw: Any) -> Any:
    """Return ``raw`` parsed as JSON when it is a JSON string, else unchanged."""
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return raw


def tool_result_to_text(result: object) -> str:
    """Render a tool's return value as text.

    Tools routinely return structured data (``{"temp": 12}``), and a result has to
    be text because it ends up as a tool ``Message.content``. ``str()`` would give
    the Python repr — single-quoted, ``None``/``True`` instead of ``null``/``true``
    — which no downstream reader can parse back. JSON keeps it readable by both a
    model and a parser.
    """
    if isinstance(result, str):
        return result
    model_dump_json = getattr(result, 'model_dump_json', None)
    if callable(model_dump_json):
        try:
            rendered = model_dump_json()
        except Exception as exc:  # noqa: BLE001 - any renderer failure must reach the text fallback
            logger.warning(
                'tool_result_to_text: {}.model_dump_json() failed with {}: {}; falling back to JSON encoding',
                type(result).__name__,
                type(exc).__name__,
                exc,
            )
        else:
            if isinstance(rendered, str):
                return rendered
            logger.warning(
                'tool_result_to_text: {}.model_dump_json() returned {}; falling back to JSON encoding',
                type(result).__name__,
                type(rendered).__name__,
            )
    # default=str: a value JSON cannot encode degrades to its repr rather than
    # failing the whole response over one odd tool return.
    return json.dumps(result, default=str)
