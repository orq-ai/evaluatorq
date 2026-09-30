"""Split a conversation into token-sized, kind-coloured segments for the trajectory view."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, get_args

from evaluatorq.common.messages import content_part_text

Kind = Literal['system', 'user', 'assistant', 'reasoning', 'call', 'result', 'other']
KIND_LABELS: Mapping[Kind, str] = MappingProxyType({
    'system': 'System',
    'user': 'User',
    'assistant': 'Assistant',
    'reasoning': 'Reasoning',
    'call': 'Tool call',
    'result': 'Tool result',
    'other': 'Other',
})
if set(KIND_LABELS) != set(get_args(Kind)):
    raise RuntimeError('KIND_LABELS must cover every trajectory kind')

PREVIEW_CHARS = 240
_ROLE_KINDS: Mapping[str, Kind] = MappingProxyType({
    'system': 'system',
    'developer': 'system',
    'user': 'user',
    'assistant': 'assistant',
    'agent': 'assistant',
    'model': 'assistant',
    'tool': 'result',
})
_TEXT_TYPES = frozenset({'text', 'input_text', 'output_text', 'summary_text', 'refusal'})
_CALL_TYPES = frozenset({'tool_call', 'function_call', 'function', 'tool_use'})
_RESULT_TYPES = frozenset({'tool_call_response', 'function_call_output', 'tool_result'})


@dataclass(frozen=True)
class Segment:
    kind: Kind
    tokens: int
    index: int
    label: str | None
    preview: str


def preview(text: str) -> str:
    """Pretty-print a JSON body and cut the result at 240 characters."""
    with suppress(ValueError, TypeError):
        text = json.dumps(json.loads(text), indent=1, ensure_ascii=False)
    return text if len(text) <= PREVIEW_CHARS else text[:PREVIEW_CHARS] + '…'


def _text(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _segment(kind: Kind, text: str, index: int, label: str | None = None) -> Segment | None:
    if not text:
        return None
    return Segment(kind, max(1, math.ceil(len(text) / 4)), index, label, preview(text))


def _part(part: Any, role_kind: Kind, index: int, tool_name: str | None) -> Segment | None:
    if isinstance(part, str):
        return _segment(role_kind, part, index, tool_name if role_kind == 'result' else None)
    if not isinstance(part, Mapping):
        return _segment('other', _text(part), index)
    kind = part.get('type') or part.get('kind')  # Orq-native parts carry `kind`, OpenAI-style ones `type`
    call: Mapping[str, Any] = part
    function = part.get('function')
    if isinstance(function, Mapping):
        call = function
    shared_text = content_part_text(part)
    if shared_text is not None:
        return _segment(
            role_kind,
            shared_text,
            index,
            tool_name if role_kind == 'result' else None,
        )
    if kind in _TEXT_TYPES:  # Orq-native parts use `kind` instead of `type`.
        return _segment(role_kind, _text(part.get('content') or part.get('text')), index)
    if kind == 'reasoning':
        return _segment(
            'reasoning',
            _text(part.get('content') or part.get('text') or part.get('reasoning') or part.get('summary')),
            index,
        )
    if kind in _CALL_TYPES:
        return _segment(
            'call',
            _text(call.get('arguments') or call.get('input')),
            index,
            _text(call.get('name') or call.get('tool_name')) or None,
        )
    if kind in _RESULT_TYPES:
        body = part.get('response', part.get('output', part.get('result', part.get('content'))))
        return _segment('result', _text(body), index, _text(part.get('name')) or tool_name)
    return _segment('other', _text(part), index)


def segments(messages: Sequence[Mapping[str, Any]]) -> list[Segment]:
    """One segment per text/reasoning/tool part; size is characters / 4.

    Unknown part types become kind ``other``; the caller counts and logs them once per render.
    """
    out: list[Segment] = []
    for index, message in enumerate(messages, start=1):
        role_kind = _ROLE_KINDS.get(str(message.get('role', '')), 'other')
        tool_name = _text(message.get('name')) or None
        content = message.get('content')
        body = content if content not in (None, '', []) else message.get('parts')
        parts = body if isinstance(body, list) else [body] if body not in (None, '') else []
        for part in parts:
            segment = _part(part, role_kind, index, tool_name)
            if segment is not None:
                out.append(segment)
        for call in message.get('tool_calls') or []:
            segment = _part({'type': 'function', **call} if isinstance(call, Mapping) else call, 'call', index, None)
            if segment is not None:
                out.append(segment)
    return out
