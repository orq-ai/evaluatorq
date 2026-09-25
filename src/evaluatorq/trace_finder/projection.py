"""Bounded, auditable projections of traces for classification."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .models import TraceProjection, TraceRecord

MAX_TOKEN_BUDGET = 25_000
MAX_PROJECTED_TOOL_CALLS = 32
MAX_TOOL_FIELD_BYTES = 128
OMISSION_MARKER = '[... earlier bytes omitted ...]'
REASONING_KEYS = frozenset({'reasoning', 'reasoning_content', 'thinking'})
ERROR_STATUSES = frozenset({'error', 'failed', 'failure', 'cancelled', 'canceled'})


@dataclass(frozen=True)
class ProjectionUnit:
    """One indivisible source-conversation unit for suffix selection."""

    messages: tuple[dict[str, Any], ...]
    source_messages: tuple[dict[str, Any], ...]


def serialize_projection(payload: dict[str, Any]) -> str:
    """Return the stable representation that classifier receives and the drawer displays."""

    return _canonical_json(payload)


def estimate_tokens(serialized: str) -> int:
    """Bound byte-level tokenizer tokens by the serialized UTF-8 byte count."""

    return len(serialized.encode('utf-8'))


def project_trace(trace: TraceRecord, token_budget: int = MAX_TOKEN_BUDGET) -> TraceProjection:
    """Project one complete trace into a newest-first-fitting classifier conversation state."""

    if not 1 <= token_budget <= MAX_TOKEN_BUDGET:
        raise ValueError(f'token_budget must be between 1 and {MAX_TOKEN_BUDGET}')

    payload: dict[str, Any] = {'trace_status': trace.status, 'messages': []}
    if estimate_tokens(serialize_projection(payload)) > token_budget:
        raise ValueError('token_budget is too small for required trace status metadata')

    units = _projection_units(trace.messages)
    selected_messages: tuple[dict[str, Any], ...] = ()
    omitted_units: tuple[ProjectionUnit, ...] = ()
    truncated_bytes = 0

    for index in range(len(units) - 1, -1, -1):
        unit = units[index]
        candidate_messages = unit.messages + selected_messages
        candidate_payload = {'trace_status': trace.status, 'messages': list(candidate_messages)}
        if estimate_tokens(serialize_projection(candidate_payload)) <= token_budget:
            selected_messages = candidate_messages
            continue

        if not selected_messages:
            selected_messages, truncated_bytes = _tail_truncate_unit(unit, trace.status, token_budget)
            omitted_units = tuple(units[:index])
        else:
            omitted_units = tuple(units[: index + 1])
        break

    payload['messages'] = list(selected_messages)
    serialized = serialize_projection(payload)
    estimated_tokens = estimate_tokens(serialized)
    omitted_messages = sum(len(unit.source_messages) for unit in omitted_units)
    omitted_bytes = sum(_source_message_bytes(message) for unit in omitted_units for message in unit.source_messages)

    return TraceProjection(
        payload=payload,
        serialized=serialized,
        estimated_tokens=estimated_tokens,
        omitted_messages=omitted_messages,
        omitted_bytes=omitted_bytes + truncated_bytes,
    )


def _projection_units(messages: tuple[dict[str, Any], ...]) -> tuple[ProjectionUnit, ...]:
    units: list[ProjectionUnit] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        tool_calls = message.get('tool_calls')
        if message.get('role') == 'assistant' and isinstance(tool_calls, list) and tool_calls:
            result_end = index + 1
            while result_end < len(messages) and messages[result_end].get('role') == 'tool':
                result_end += 1
            results = messages[index + 1 : result_end]
            units.append(
                ProjectionUnit(
                    messages=(_project_assistant(message, results),),
                    source_messages=messages[index:result_end],
                )
            )
            index = result_end
            continue

        if message.get('role') == 'tool':
            units.append(ProjectionUnit(messages=(), source_messages=(message,)))
        else:
            units.append(ProjectionUnit(messages=(_project_message(message),), source_messages=(message,)))
        index += 1
    return tuple(units)


def _project_assistant(message: dict[str, Any], results: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    source_calls = message['tool_calls']
    projected_calls = [_project_tool_call(call, results) for call in source_calls[:MAX_PROJECTED_TOOL_CALLS]]
    if len(source_calls) > MAX_PROJECTED_TOOL_CALLS:
        projected_calls.append({
            'id': None,
            'name': None,
            'arguments': f'{len(source_calls) - MAX_PROJECTED_TOOL_CALLS} tool calls omitted',
            'status': 'omitted',
        })
    projected = {
        'role': message.get('role'),
        'content': _project_content(message.get('content')),
        'tool_calls': projected_calls,
    }
    if 'parts' in message:
        projected['parts'] = _project_content(message['parts'])
    return projected


def _project_message(message: dict[str, Any]) -> dict[str, Any]:
    """Keep conversation fields while dropping unbounded unrelated source metadata."""

    role = message.get('role')
    projected: dict[str, Any] = {
        'role': role if role in {'system', 'developer', 'user', 'assistant', 'tool'} else 'unknown'
    }
    for name in ('content', 'parts'):
        if name in message:
            projected[name] = _project_content(message[name])
    return projected


def _project_content(content: Any) -> Any:
    """Render text blocks and mark non-text media without copying large media payloads."""

    if isinstance(content, list):
        blocks = [_project_content_block(block) for block in content[:16]]
        if len(content) > 16:
            blocks.append({'type': 'omission', 'omitted': f'{len(content) - 16} content blocks'})
        return blocks
    if isinstance(content, dict):
        return [_project_content_block(content)]
    return content if isinstance(content, str) or content is None else None


def _project_content_block(block: Any) -> dict[str, Any]:
    if isinstance(block, str):
        return {'type': 'text', 'text': block}
    if not isinstance(block, dict):
        return {'type': 'unknown', 'omitted': 'non-text content'}
    block_type = block.get('type')
    kind = block_type[:80] if isinstance(block_type, str) else 'unknown'
    text = block.get('text')
    if isinstance(text, str):
        return {'type': kind, 'text': text}
    if isinstance(text, dict) and isinstance(text.get('value'), str):
        return {'type': kind, 'text': {'value': text['value']}}
    try:
        if len(_canonical_json(block).encode('utf-8')) <= 512:
            return _strip_reasoning(block)
    except (TypeError, ValueError):
        pass
    return {'type': kind, 'omitted': 'non-text content'}


def _project_tool_call(call: Any, results: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    if not isinstance(call, dict):
        return {'id': None, 'name': None, 'arguments': _strip_reasoning(call), 'status': 'pending'}

    function = call.get('function')
    function_data = function if isinstance(function, dict) else {}
    call_id = _strip_reasoning(call.get('id'))
    arguments = _parse_arguments(function_data.get('arguments', call.get('arguments')))
    return {
        'id': _bounded_tool_field(call_id),
        'name': _bounded_tool_field(function_data.get('name', call.get('name'))),
        'arguments': arguments,
        'status': _tool_call_status(call_id, results),
    }


def _bounded_tool_field(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    if len(value.encode('utf-8')) <= MAX_TOOL_FIELD_BYTES:
        return value
    return _tail_truncate_text(value, MAX_TOOL_FIELD_BYTES - len(OMISSION_MARKER.encode('utf-8')))[0]


def _parse_arguments(arguments: Any) -> Any:
    if not isinstance(arguments, str):
        return _strip_reasoning(arguments)
    try:
        return _strip_reasoning(json.loads(arguments))
    except json.JSONDecodeError:
        return arguments


def _tool_call_status(call_id: Any, results: tuple[dict[str, Any], ...]) -> str:
    result = next((item for item in results if item.get('tool_call_id') == call_id), None)
    if result is None:
        return 'pending'
    status = result.get('status')
    if isinstance(status, str) and status.casefold() in ERROR_STATUSES:
        return 'error'
    if result.get('is_error') is True or result.get('error') not in (None, False, ''):
        return 'error'
    return 'completed'


def _tail_truncate_unit(
    unit: ProjectionUnit,
    trace_status: str,
    token_budget: int,
) -> tuple[tuple[dict[str, Any], ...], int]:
    text_lengths = _truncatable_text_lengths(unit.messages)
    if not text_lengths:
        return _omit_structural_unit(unit, trace_status, token_budget)

    def fits(retained_bytes: int) -> bool:
        truncated, _ = _truncate_messages(unit.messages, retained_bytes)
        payload = {'trace_status': trace_status, 'messages': list(truncated)}
        return estimate_tokens(serialize_projection(payload)) <= token_budget

    if not fits(0):
        return _omit_structural_unit(unit, trace_status, token_budget)

    lower = 0
    upper = max(text_lengths)
    while lower < upper:
        midpoint = (lower + upper + 1) // 2
        if fits(midpoint):
            lower = midpoint
        else:
            upper = midpoint - 1
    return _truncate_messages(unit.messages, lower)


def _omit_structural_unit(
    unit: ProjectionUnit, trace_status: str, token_budget: int
) -> tuple[tuple[dict[str, Any], ...], int]:
    role = unit.messages[0].get('role', 'unknown') if unit.messages else 'unknown'
    marker = ({'role': role, 'content': OMISSION_MARKER},)
    payload = {'trace_status': trace_status, 'messages': list(marker)}
    if estimate_tokens(serialize_projection(payload)) > token_budget:
        raise ValueError('token_budget is too small for the newest structural projection unit')
    return marker, sum(_source_message_bytes(message) for message in unit.source_messages)


def _truncatable_text_lengths(messages: tuple[dict[str, Any], ...]) -> tuple[int, ...]:
    lengths: list[int] = []
    for message in messages:
        for name in ('content', 'parts'):
            content = message.get(name)
            if isinstance(content, str) and content:
                lengths.append(len(content.encode('utf-8')))
            lengths.extend(len(container[key].encode('utf-8')) for container, key in _content_text_slots(content))
        tool_calls = message.get('tool_calls')
        if isinstance(tool_calls, list):
            lengths.extend(
                len(_argument_text_for_truncation(call['arguments']).encode('utf-8'))
                for call in tool_calls
                if isinstance(call, dict) and 'arguments' in call and call['arguments'] != ''
            )
    return tuple(lengths)


def _content_text_slots(content: Any) -> list[tuple[dict[str, Any], str]]:
    """Locate text in supported content blocks without treating metadata as prose."""
    slots = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            text = block.get('text')
            if isinstance(text, str):
                slots.append((block, 'text'))
            elif isinstance(text, dict) and isinstance(text.get('value'), str):
                slots.append((text, 'value'))
    return slots


def _truncate_messages(
    messages: tuple[dict[str, Any], ...], retained_bytes: int
) -> tuple[tuple[dict[str, Any], ...], int]:
    truncated_messages: list[dict[str, Any]] = []
    omitted_bytes = 0
    for message in messages:
        truncated = dict(message)
        for name in ('content', 'parts'):
            content = truncated.get(name)
            if isinstance(content, str):
                truncated[name], omitted = _tail_truncate_text(content, retained_bytes)
                omitted_bytes += omitted
            elif isinstance(content, list):
                truncated[name] = deepcopy(content)
                for container, key in _content_text_slots(truncated[name]):
                    container[key], omitted = _tail_truncate_text(container[key], retained_bytes)
                    omitted_bytes += omitted
        tool_calls = truncated.get('tool_calls')
        if isinstance(tool_calls, list):
            truncated_calls: list[Any] = []
            for call in tool_calls:
                if not isinstance(call, dict):
                    truncated_calls.append(call)
                    continue
                truncated_call = dict(call)
                if 'arguments' in truncated_call:
                    arguments = truncated_call['arguments']
                    arguments_text = _argument_text_for_truncation(arguments)
                    if len(arguments_text.encode('utf-8')) > retained_bytes:
                        truncated_call['arguments'], omitted = _tail_truncate_text(arguments_text, retained_bytes)
                        omitted_bytes += omitted
                truncated_calls.append(truncated_call)
            truncated['tool_calls'] = truncated_calls
        truncated_messages.append(truncated)
    return tuple(truncated_messages), omitted_bytes


def _tail_truncate_text(value: str, retained_bytes: int) -> tuple[str, int]:
    encoded = value.encode('utf-8')
    if len(encoded) <= retained_bytes:
        return value, 0
    tail = encoded[-retained_bytes:] if retained_bytes else b''
    retained_tail = tail.decode('utf-8', errors='ignore')
    return OMISSION_MARKER + retained_tail, len(encoded) - len(retained_tail.encode('utf-8'))


def _argument_text_for_truncation(arguments: Any) -> str:
    return arguments if isinstance(arguments, str) else _canonical_json(arguments)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _strip_reasoning(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_reasoning(item) for key, item in value.items() if key not in REASONING_KEYS}
    if isinstance(value, list):
        return [_strip_reasoning(item) for item in value]
    if isinstance(value, tuple):
        return [_strip_reasoning(item) for item in value]
    return value


def _source_message_bytes(message: dict[str, Any]) -> int:
    return len(serialize_projection(message).encode('utf-8'))
