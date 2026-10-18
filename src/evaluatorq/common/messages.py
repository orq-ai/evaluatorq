"""Shared helpers for normalizing chat-message content.

Canonical for every surface. Content is typed ``str | list[ContentPart]``:
never call ``str()`` on it — that renders a Python repr into a transcript a
judge then scores. Use these helpers or ``contracts.content_to_text``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq.common.thread_context import warn_once

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
EMPTY_ASSISTANT_TEXT = '[The agent sent no reply]'


def tool_call_text(call: Any, *, incomplete: bool | None = None, result: str | None = None) -> str:
    """Render one tool call as a marker a judge can read.

    ``incomplete`` decides the ``-> no result recorded`` suffix; ``None`` reads the recorded
    ``incomplete`` flag on the call. A ``result`` renders inside the brackets instead.
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
        flagged = _field(call, 'incomplete') is True if incomplete is None else incomplete
        suffix = f' -> {NO_RESULT_TEXT}' if flagged else ''
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
    without a result renders with ``-> no result recorded`` (`call_is_incomplete`), so the judge
    reads an attempt that never returned instead of a call whose result was lost. Over a list of
    `Message` the result is looked up by position; pass the whole transcript, because a slice that
    cuts a call from its result row marks the call. Anything else falls back to the recorded
    ``incomplete`` flag.
    """
    from evaluatorq.contracts import Message

    messages = list(messages)
    pairing = pair_tool_rows(messages) if messages and all(isinstance(m, Message) for m in messages) else None
    lines: list[str] = []
    for index, message in enumerate(messages):
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
        answered = pairing.results_for(index) if pairing is not None else {}
        parts.extend(
            tool_call_text(
                call,
                incomplete=call_is_incomplete(call, paired=position in answered if pairing is not None else None),
            )
            for position, call in enumerate(calls)
        )
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


class ToolPairing:
    """Which tool row answers which call of a transcript, by position."""

    def __init__(self, row_of: dict[int, dict[int, int]]) -> None:
        self.row_of = row_of
        self.rows = {row for answers in row_of.values() for row in answers.values()}

    def results_for(self, index: int) -> dict[int, int]:
        """Map each answered call position of the assistant row ``index`` to its tool row index."""
        return self.row_of.get(index, {})


def match_tool_results(
    rows: Sequence[tuple[str | None, Sequence[str | None], str | None]],
    *,
    tolerant: bool = False,
) -> tuple[dict[tuple[int, int], list[int]], list[int]]:
    """Match tool rows to calls by id inside a turn window; the one rule every pairing reader shares.

    Each row is ``(role, call_ids, tool_call_id)``: the ids of an assistant row's calls (``None`` for
    one without a string id) and the id a tool row answers. A tool row answers a call when the call's
    assistant row comes before it and no user row sits between them; any rows may sit in between. A
    result before its call, after a later user row, or with no call in its window is an orphan.
    Returns ``({(row, position): [result_row, ...]}, [orphan_row, ...])``.

    Strict (replay): the first call per id owns it while unanswered, the first result per id wins and
    a duplicate result is an orphan, so every call has at most one result. ``tolerant`` (display of
    logs): calls repeating an id in one row queue up and take results in order, a later row's call replaces an unanswered earlier one, and a surplus result attaches to
    the last call it answered instead of being an orphan.

    The window follows what providers accept, probed live through the Orq router: Anthropic wants the
    result after its call and before the next user turn but not adjacent, OpenAI chat completions wants
    it directly after the call (`drop_unreplayable_tool_records` reorders the replay copy for that),
    and OpenAI Responses accepts any order.
    """
    open_calls: dict[str, list[tuple[int, int]]] = {}
    answered: dict[str, tuple[int, int]] = {}
    pairs: dict[tuple[int, int], list[int]] = {}
    orphans: list[int] = []
    for index, (role, call_ids, result_id) in enumerate(rows):
        if role == 'user':
            open_calls.clear()
            answered.clear()
        elif role == 'assistant':
            for position, call_id in enumerate(call_ids):
                if call_id is None:
                    continue
                queue = open_calls.setdefault(call_id, [])
                if tolerant and queue and queue[0][0] != index:
                    queue.clear()
                if tolerant or not queue:
                    queue.append((index, position))
        elif role == 'tool':
            queue = open_calls.get(result_id or '')
            if queue:
                owner = queue.pop(0)
                answered[result_id or ''] = owner
            else:
                owner = answered.get(result_id or '') if tolerant else None
            if owner is None:
                orphans.append(index)
            else:
                pairs.setdefault(owner, []).append(index)
    return pairs, orphans


def pair_tool_rows(messages: list[Message]) -> ToolPairing:
    """Pair tool rows to calls by id inside a turn window, for provider replay.

    See `match_tool_results` for the rule and the provider findings behind it. This is the single
    implementation: the display pairing in `common.model_input.pair_tool_call_results` calls it too.

    Position is authoritative for replay and for `call_is_incomplete` over a full transcript. The
    recorded `StrategyToolCall.incomplete` flag only stands in where no position is available (a
    loose dict, an export).
    """
    pairs, _ = match_tool_results([(m.role, [tc.id for tc in m.tool_calls or []], m.tool_call_id) for m in messages])
    row_of: dict[int, dict[int, int]] = {}
    for (row, position), results in pairs.items():
        row_of.setdefault(row, {})[position] = results[0]
    return ToolPairing(row_of)


def call_is_incomplete(call: Any, *, paired: bool | None) -> bool:
    """Whether ``call`` has no result: the one answer every reader of a transcript uses.

    ``paired`` is whether a tool row answers it by position, or ``None`` when no transcript
    position is available, in which case the recorded ``incomplete`` flag decides. A real tool
    row always wins over the flag.
    """
    if paired is not None:
        return not paired
    return _field(call, 'incomplete') is True


def wire_arguments(arguments: str) -> str:
    """Return ``arguments`` as a provider accepts it: a JSON object, else ``'{}'``.

    Unknown (``''``), truncated or non-object arguments become ``'{}'``. Anthropic behind the Orq
    router rejects anything else (``tool_use.input: Input should be an object``). Only the
    serializers call this; the recorded call keeps what it had.
    """
    try:
        parsed = json.loads(arguments)
    except (TypeError, ValueError):
        return '{}'
    return arguments if isinstance(parsed, dict) else '{}'


def drop_unreplayable_tool_records(messages: list[Message], *, incomplete_as_text: bool = False) -> list[Message]:
    """Return a shallow copy of ``messages`` without the tool records a provider would reject.

    Rows that need no change are the same objects; a changed row is a `model_copy`, so the
    recorded transcript is never mutated and keeps its incomplete calls. An assistant tool call
    needs a tool row after it in its turn (see `pair_tool_rows`) and a tool row needs the call it
    answers; every other call and row is removed, each kept result is moved to sit directly after
    its call's row (OpenAI chat completions requires that; parallel results keep call order), and a repeated call id in one row keeps only its first
    call. A turn left with no text and no calls (and any other empty assistant turn) gets
    `EMPTY_ASSISTANT_TEXT`, which keeps the user/assistant alternation and satisfies providers
    that reject an empty assistant message. Arguments are not touched here; the serializers send
    `wire_arguments`, so a target still sees an unknown call as unknown.

    With ``incomplete_as_text`` (the judge's copy) nothing is lost silently: a removed call is
    appended to its turn's text as ``[tool_call: name(args) -> no result recorded]`` and a kept
    call with unknown arguments as ``[tool_call: name(arguments unavailable)]``. Only an orphan
    result warns then.

    One warning per record per conversation (`warn_once`); repeats log at debug.
    """
    from evaluatorq.contracts import InputTextContent

    pairing = pair_tool_rows(messages)
    dropped_calls: list[tuple[str, str, str, str]] = []
    dropped_results: list[tuple[str, str]] = []
    out: list[Message] = []
    for index, m in enumerate(messages):
        if m.role == 'tool':
            if index not in pairing.rows:
                dropped_results.append(('result', m.tool_call_id or '<no id>'))
            continue
        if m.role != 'assistant':
            out.append(m)
            continue
        answered = pairing.results_for(index)
        calls = m.tool_calls or []
        kept = [tc for position, tc in enumerate(calls) if position in answered]
        lost = [tc for position, tc in enumerate(calls) if position not in answered]
        unknown = [tc for tc in kept if tc.function.arguments == UNKNOWN_ARGUMENTS]
        silent = not coerce_content_text(m.content).strip()
        results = [messages[row] for _, row in sorted(answered.items())]
        if not lost and not (incomplete_as_text and unknown) and (calls or not silent):
            out.extend([m, *results])
            continue
        content = m.content
        if incomplete_as_text:
            notes = [
                *(tool_call_text(tc, incomplete=True) for tc in lost),
                *(tool_call_text(tc, incomplete=False) for tc in unknown),
            ]
            if notes:
                marker = '\n'.join(notes)
                if isinstance(content, list):
                    content = [*content, InputTextContent(type='input_text', text=marker)]
                else:
                    content = f'{content}\n{marker}' if content and content.strip() else marker
        else:
            dropped_calls.extend(('call', tc.id, tc.function.name, tc.function.arguments) for tc in lost)
        if not kept and not coerce_content_text(content).strip():
            content = EMPTY_ASSISTANT_TEXT
        out.extend([m.model_copy(update={'content': content, 'tool_calls': kept or None}), *results])
    _warn_dropped(dropped_calls, dropped_results)
    return out


def _warn_dropped(calls: list[tuple[str, str, str, str]], results: list[tuple[str, str]]) -> None:
    if not calls and not results:
        return
    parts = []
    if calls:
        parts.append(f'tool call(s) {sorted({c[1] for c in calls})} (no result follows them)')
    if results:
        parts.append(f'tool result(s) {sorted({r[1] for r in results})} (no call precedes them)')
    warn_once(
        [*calls, *results],
        'Not replaying {} to the target. They stay in the recorded transcript.',
        ' and '.join(parts),
    )
