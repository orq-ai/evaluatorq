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

from evaluatorq.common.thread_context import conversation_warned_keys

if TYPE_CHECKING:
    from collections.abc import Iterable

    from evaluatorq.contracts import Message


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
NO_RESULT_TEXT = 'no result recorded'


def tool_call_text(call: Any, *, incomplete: bool = False, result: str | None = None) -> str:
    """Render one tool call as a marker a judge can read.

    The ``-> no result recorded`` suffix comes from the recorded fact, an ``incomplete`` flag on
    the call or the argument, never from which neighbours happen to be in the same slice.
    """
    function = call.get('function') if isinstance(call, dict) else getattr(call, 'function', None)
    source = function if function is not None else call
    if isinstance(source, dict):
        name, arguments = source.get('name'), source.get('arguments')
    else:
        name, arguments = getattr(source, 'name', None), getattr(source, 'arguments', None)
    rendered = arguments if isinstance(arguments, str) else _json_or_text(arguments)
    if arguments == UNKNOWN_ARGUMENTS:
        rendered = UNKNOWN_ARGUMENTS_TEXT
    if result is not None:
        suffix = f' -> {result}'
    else:
        suffix = f' -> {NO_RESULT_TEXT}' if incomplete or _field(call, 'incomplete') is True else ''
    return f'[tool_call: {name or "unknown"}({rendered or ""}){suffix}]'


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
    them here would hand a judge an empty string for an agent that acted. A tool call
    flagged ``incomplete`` (its result was never recorded) renders with ``-> no result recorded``,
    so the judge reads an attempt that never returned instead of a call whose result was lost.
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
        parts.extend(tool_call_text(call) for call in calls)
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


@dataclass(frozen=True)
class ToolPairing:
    """Which tool calls and tool rows of a transcript answer each other, by position."""

    paired: dict[int, set[str]]
    rows: set[int]
    row_of: dict[tuple[int, str], int]


def pair_tool_rows(messages: list[Message]) -> ToolPairing:
    """Pair tool rows to calls by position, for provider replay.

    A block is the consecutive assistant turns that carry tool calls (parallel calls may be split
    across rows) plus the tool rows that directly follow them; system rows between the pieces are
    skipped. A tool row answers the first call in its block with that id, and the first row per id
    wins. A reused id, a result after a later user turn, a result before its call and a duplicate
    result do not pair.

    Position is authoritative for replay only. What the judge reads as "no result recorded" is the
    recorded fact, `StrategyToolCall.incomplete`; the display pairing in
    `common.model_input.pair_tool_call_results` is a different job (FIFO within a call group for
    rendering) and stays separate.
    """
    paired: dict[int, set[str]] = {}
    rows: set[int] = set()
    row_of: dict[tuple[int, str], int] = {}
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.role != 'assistant' or not message.tool_calls:
            index += 1
            continue
        owner: dict[str, int] = {}
        cursor = index
        while cursor < len(messages) and messages[cursor].role in ('assistant', 'system'):
            if messages[cursor].role == 'assistant':
                if not messages[cursor].tool_calls:
                    break
                paired.setdefault(cursor, set())
                for tc in messages[cursor].tool_calls or []:
                    owner.setdefault(tc.id, cursor)
            cursor += 1
        while cursor < len(messages) and messages[cursor].role in ('tool', 'system'):
            call_id = messages[cursor].tool_call_id
            if messages[cursor].role == 'tool' and call_id in owner and call_id not in paired[owner[call_id]]:
                paired[owner[call_id]].add(call_id)
                rows.add(cursor)
                row_of[owner[call_id], call_id] = cursor
            cursor += 1
        index = cursor
    return ToolPairing(paired=paired, rows=rows, row_of=row_of)


WIRE_EMPTY_ARGUMENTS = '{}'


def _wire_call(call: Any) -> Any:
    """Return ``call`` as it goes on the wire: unknown arguments become ``'{}'`` on a copy.

    Anthropic behind the Orq router rejects ``''`` (``tool_use.input: Input should be an object``);
    the recorded call keeps ``''`` so evidence still reads "arguments unavailable".
    """
    if call.function.arguments != UNKNOWN_ARGUMENTS:
        return call
    return call.model_copy(update={'function': call.function.model_copy(update={'arguments': WIRE_EMPTY_ARGUMENTS})})


def drop_unreplayable_tool_records(messages: list[Message], *, incomplete_as_text: bool = False) -> list[Message]:
    """Return a copy of ``messages`` without the tool records a provider would reject.

    The recorded transcript is evidence and keeps an incomplete tool call; this builds the copy
    that is sent. An assistant tool call needs a tool row after it with the same ``tool_call_id``
    (see `pair_tool_rows`), and a tool row needs the call it answers. Every other call and row is
    removed. A turn left with no text and no calls stays as an empty assistant row, so the
    user/assistant alternation survives. A paired call with unknown arguments is kept, and the copy
    sends ``'{}'`` for it (a provider-required object); the recorded call keeps ``''``.
    Run this on the whole transcript, never on a slice.

    With ``incomplete_as_text`` a call without a result is not lost: it is appended to its turn's
    text as ``[tool_call: name(args) -> no result recorded]``, for a reader (the judge) that must
    still see the attempt, and only an orphan result warns.

    One warning per record per conversation: keys already in `conversation_warned_keys` log at
    debug. A call is keyed by id, name and arguments, a result by id.
    """
    pairing = pair_tool_rows(messages)
    dropped_calls: list[tuple[str, str]] = []
    dropped_results: list[tuple[str, str]] = []
    out: list[Message] = []
    for index, m in enumerate(messages):
        if m.role == 'tool':
            if index in pairing.rows:
                out.append(m)
            else:
                call_id = m.tool_call_id or '<no id>'
                dropped_results.append((call_id, f'result|{call_id}'))
            continue
        if m.role == 'assistant' and m.tool_calls:
            kept = [tc for tc in m.tool_calls if tc.id in pairing.paired[index]]
            lost = [tc for tc in m.tool_calls if tc.id not in pairing.paired[index]]
            wire_kept = [_wire_call(tc) for tc in kept]
            if lost or any(a is not b for a, b in zip(wire_kept, kept, strict=True)):
                content = m.content
                if incomplete_as_text and lost:
                    from evaluatorq.contracts import InputTextContent

                    marker = '\n'.join(tool_call_text(tc, incomplete=True) for tc in lost)
                    if isinstance(content, list):
                        content = [*content, InputTextContent(type='input_text', text=marker)]
                    else:
                        content = f'{content}\n{marker}' if content and content.strip() else marker
                else:
                    dropped_calls.extend(
                        (tc.id, f'call|{tc.id}|{tc.function.name}|{tc.function.arguments}') for tc in lost
                    )
                    if not kept and not coerce_content_text(content).strip():
                        content = ''
                m = m.model_copy(update={'content': content, 'tool_calls': wire_kept or None})
        out.append(m)
    _warn_dropped(dropped_calls, dropped_results)
    return out


def _warn_dropped(calls: list[tuple[str, str]], results: list[tuple[str, str]]) -> None:
    if not calls and not results:
        return
    seen = conversation_warned_keys()
    fresh = [(i, k) for i, k in [*calls, *results] if k not in seen]
    seen.update(k for _, k in [*calls, *results])
    shown = fresh or [*calls, *results]
    fresh_calls = sorted({i for i, k in shown if k.startswith('call|')})
    fresh_results = sorted({i for i, k in shown if k.startswith('result|')})
    parts = []
    if fresh_calls:
        parts.append(f'tool call(s) {fresh_calls} (no result follows them)')
    if fresh_results:
        parts.append(f'tool result(s) {fresh_results} (no call precedes them)')
    report = logger.warning if fresh else logger.debug
    report(
        'Not replaying {} to the target. They stay in the recorded transcript.',
        ' and '.join(parts),
    )
