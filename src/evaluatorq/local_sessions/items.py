"""Responses item constructors: the only place readers build item dicts."""

from __future__ import annotations

import json
from typing import Any

from evaluatorq.local_sessions.models import MAX_TOOL_TEXT_CHARS

_MARKER_ALLOWANCE = 100  # room for one `[truncated N chars]` marker and the JSON around a single trimmed value


def trim(text: str) -> str:
    if len(text) <= MAX_TOOL_TEXT_CHARS:
        return text
    half = MAX_TOOL_TEXT_CHARS // 2
    return f'{text[:half]}\n[truncated {len(text) - 2 * half} chars]\n{text[-half:]}'


def blocks_text(blocks: object) -> str:
    """Join the text of raw content blocks; a string passes through, non-text blocks render as `[image]` or `[<type>]`."""
    if isinstance(blocks, str):
        return blocks
    if not isinstance(blocks, list):
        return ''
    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        kind = block.get('type')
        if kind in ('text', 'input_text', 'output_text') and isinstance(block.get('text'), str):
            parts.append(block['text'])
        elif kind in ('image', 'input_image'):
            parts.append('[image]')
        elif isinstance(kind, str):
            parts.append(f'[{kind}]')
    return '\n'.join(part for part in parts if part)


def user_text(text: str) -> dict[str, Any]:
    return {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': text}]}


def system_text(text: str) -> dict[str, Any]:
    return {'type': 'message', 'role': 'system', 'content': [{'type': 'input_text', 'text': text}]}


def assistant_text(text: str) -> dict[str, Any]:
    return {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': text}]}


def reasoning(text: str) -> dict[str, Any]:
    return {'type': 'reasoning', 'summary': [{'type': 'summary_text', 'text': text}]}


def _trim_leaves(value: object) -> object:
    if isinstance(value, str):
        return trim(value)
    if isinstance(value, dict):
        return {key: _trim_leaves(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_trim_leaves(inner) for inner in value]
    return value


def function_call(*, call_id: str, name: str, arguments: object) -> dict[str, Any]:
    """Arguments stay valid JSON after trimming: only string leaves are cut, never the serialised text."""
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except ValueError:
            parsed = None
        if not isinstance(parsed, dict):
            return {'type': 'function_call', 'call_id': call_id, 'name': name, 'arguments': trim(arguments)}
        arguments = parsed
    raw = json.dumps(_trim_leaves(arguments if arguments is not None else {}), ensure_ascii=False)
    if len(raw) > MAX_TOOL_TEXT_CHARS + _MARKER_ALLOWANCE:
        raw = json.dumps({'_truncated': trim(raw)}, ensure_ascii=False)
    return {'type': 'function_call', 'call_id': call_id, 'name': name, 'arguments': raw}


def function_call_output(*, call_id: str, output: str, is_error: bool = False) -> dict[str, Any]:
    # `Error: ` is what the transcript's tool-status check recognizes; it goes on before trimming so it survives.
    return {
        'type': 'function_call_output',
        'call_id': call_id,
        'output': trim(f'Error: {output}' if is_error else output),
    }


def compaction_summary(summary: str) -> dict[str, Any]:
    return system_text(f'Conversation compacted. Summary:\n{summary}' if summary else 'Conversation compacted.')
