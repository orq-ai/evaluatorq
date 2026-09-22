"""Shared helpers for normalizing chat-message content.

Canonical for every surface. Content is typed ``str | list[ContentPart]``:
never call ``str()`` on it — that renders a Python repr into a transcript a
judge then scores. Use these helpers or ``contracts.content_to_text``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable


def coerce_content_text(content: Any) -> str:
    """Flatten message content to a plain text string.

    Multi-part content (e.g. tool/result messages shaped like
    ``[{"type": "text", "text": "..."}]``) surfaces the joined text rather than a
    Python ``repr`` of the list. ``None`` becomes ``""``; plain strings (and anything
    else) pass through ``str``.

    Unlike `evaluatorq.contracts.content_to_text`, this best-effort helper
    does not raise on non-text parts (it is used in report/transcript rendering).
    Image and file parts are surfaced as a ``[image]`` / ``[file]`` placeholder,
    and any other (unknown/future) part type as ``[<type>]``, so every part is
    visibly accounted for rather than silently dropped.
    """
    if isinstance(content, list):
        texts: list[str] = []
        for part in content:
            part_type = part.get('type') if isinstance(part, dict) else getattr(part, 'type', None)
            # Both the chat-completions ("text") and Responses ("input_text")
            # shapes carry their text under a "text" key.
            if part_type in ('text', 'input_text'):
                texts.append(part.get('text', '') if isinstance(part, dict) else getattr(part, 'text', ''))
            elif part_type in ('image_url', 'input_image'):
                texts.append('[image]')
            elif part_type in ('file', 'input_file'):
                texts.append('[file]')
            else:
                # Unknown/future part shapes (e.g. audio, output_text) are still
                # surfaced as a placeholder rather than vanishing silently.
                texts.append(f'[{part_type or "unknown"}]')
        return '\n'.join(texts)
    return str(content or '')


def _json_or_text(value: Any) -> str:
    """Render a value as text, using JSON for mappings.

    ``str()`` on a dict renders a Python repr that a judge then scores as the
    agent's words, so structured values go through ``json.dumps``.
    """
    if isinstance(value, dict):
        return json.dumps(value, default=str)
    return coerce_content_text(value)


def _tool_call_text(call: Any) -> str:
    """Render one tool call as a marker a judge can read."""
    function = call.get('function') if isinstance(call, dict) else getattr(call, 'function', None)
    source = function if function is not None else call
    if isinstance(source, dict):
        name, arguments = source.get('name'), source.get('arguments')
    else:
        name, arguments = getattr(source, 'name', None), getattr(source, 'arguments', None)
    rendered = arguments if isinstance(arguments, str) else _json_or_text(arguments)
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
            legacy_call, legacy_result = None, None
        text = coerce_content_text(content)
        parts = [text] if text.strip() else []
        parts.extend(_tool_call_text(call) for call in [*(tool_calls or []), *([legacy_call] if legacy_call else [])])
        if legacy_result:
            parts.append(f'[tool_result: {_legacy_result_text(legacy_result)}]')
        if not parts:
            continue
        body = '\n'.join(parts)
        lines.append(body if role in (None, 'assistant') else f'[{role}] {body}')
    return '\n'.join(lines)
