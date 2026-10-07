"""Responses item constructors: the only place readers build item dicts."""

from __future__ import annotations

import json
from typing import Any

from evaluatorq.local_sessions.models import MAX_TOOL_TEXT_CHARS


def trim(text: str) -> str:
    if len(text) <= MAX_TOOL_TEXT_CHARS:
        return text
    return f'{text[:MAX_TOOL_TEXT_CHARS]}\n[truncated {len(text) - MAX_TOOL_TEXT_CHARS} chars]'


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


def function_call(*, call_id: str, name: str, arguments: object) -> dict[str, Any]:
    raw = (
        arguments
        if isinstance(arguments, str)
        else json.dumps(arguments if arguments is not None else {}, ensure_ascii=False)
    )
    return {'type': 'function_call', 'call_id': call_id, 'name': name, 'arguments': trim(raw)}


def function_call_output(*, call_id: str, output: str) -> dict[str, Any]:
    return {'type': 'function_call_output', 'call_id': call_id, 'output': trim(output)}


def compaction_summary(summary: str) -> dict[str, Any]:
    return system_text(f'Conversation compacted. Summary:\n{summary}' if summary else 'Conversation compacted.')
