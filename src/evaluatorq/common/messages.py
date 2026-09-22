"""Shared helpers for normalizing chat-message content.

Canonical for every surface. Content is typed ``str | list[ContentPart]``:
never call ``str()`` on it — that renders a Python repr into a transcript a
judge then scores. Use these helpers or ``contracts.content_to_text``.
"""

from __future__ import annotations

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


def _tool_call_text(call: Any) -> str:
    """Render one tool call as a marker a judge can read."""
    function = call.get('function') if isinstance(call, dict) else getattr(call, 'function', None)
    source = function if function is not None else call
    if isinstance(source, dict):
        name, arguments = source.get('name'), source.get('arguments')
    else:
        name, arguments = getattr(source, 'name', None), getattr(source, 'arguments', None)
    rendered = arguments if isinstance(arguments, str) else coerce_content_text(arguments)
    return f'[tool_call: {name or "unknown"}({rendered or ""})]'


def messages_to_text(messages: Iterable[Any]) -> str:
    """Render a message list as one plain-text transcript.

    Canonical for every surface that hands a recorded conversation to a judge or
    a scorer. Three properties the naive ``''.join(content)`` it replaced lacked:
    messages are newline-separated (they used to be glued into one run-on word),
    tool calls are rendered rather than dropped (a turn that only called a tool
    used to render as the empty string, so a working agent scored as silent), and
    non-assistant turns are labelled (tool JSON used to read as the agent's own
    answer). A single assistant text message still renders as its bare text.
    """
    lines: list[str] = []
    for message in messages:
        if isinstance(message, dict):
            role, content, tool_calls = message.get('role'), message.get('content'), message.get('tool_calls')
        else:
            role = getattr(message, 'role', None)
            content = getattr(message, 'content', None)
            tool_calls = getattr(message, 'tool_calls', None)
        parts = [text] if (text := coerce_content_text(content).strip()) else []
        parts.extend(_tool_call_text(call) for call in tool_calls or [])
        if not parts:
            continue
        body = '\n'.join(parts)
        lines.append(body if role in (None, 'assistant') else f'[{role}] {body}')
    return '\n'.join(lines)
