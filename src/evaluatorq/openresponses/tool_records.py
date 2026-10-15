"""Pair the split tool records of an Orq router response.

The Orq router can report one tool call as two wire records: a ``function_call``
(the arguments) and an ``orq:<tool>`` record (the result), tied together by
``call_id``; a native ``function_call_output`` item is one more result record for its ``call_id``. Every reader of those records needs the same pairing rules, so they
live here, on plain wire dicts or objects, with no dependency on ``contracts``.
``normalize_tool_records`` is the entry point for readers: it rewrites the
records into standard ``function_call`` / ``function_call_output`` items.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from loguru import logger

from evaluatorq.common.fields import get_field as _gf
from evaluatorq.common.messages import tool_result_to_text


@dataclass(frozen=True)
class PairedToolCall:
    """One tool call with the records that described it merged.

    ``source_index`` is the input position of the ``function_call`` record (of the
    first record when there is none), so a reader can place the call among the
    non-tool items it also handles. ``has_arguments`` is False for a result whose
    call record never arrived; ``arguments`` is then ``'{}'``. ``item_id`` is the
    wire ``id`` of the ``function_call`` record, or of the lone record when no call
    record exists. ``indices`` are the input positions of every record merged into the call.
    """

    call_id: str
    item_id: str | None
    name: str
    arguments: str
    has_arguments: bool
    result: str | None
    source_index: int
    indices: tuple[int, ...]


@dataclass(frozen=True)
class _ToolRecord:
    """One wire tool record, normalised for pairing."""

    key: str | None
    wire_id: str | None
    is_function_call: bool
    is_native_output: bool
    explicit_name: str | None
    derived_name: str
    arguments: str | None
    result: str | None
    error: str | None
    failed: bool


def is_tool_record(item_type: Any) -> bool:
    """Whether a wire item type is a tool record: a ``function_call`` or an ``orq:<tool>``."""
    return isinstance(item_type, str) and (item_type == 'function_call' or item_type.startswith('orq:'))


def parse_json_or_value(raw: Any) -> Any:
    """Return ``raw`` parsed as JSON when it is a JSON string, else unchanged."""
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return raw


def _has_error(raw: Any) -> bool:
    """Whether an ``error`` field reports a failure: a non-empty string, dict or list."""
    return bool(raw) and isinstance(raw, (str, dict, list))


def _is_placeholder_arguments(text: str) -> bool:
    try:
        parsed = json.loads(text)
    except ValueError:
        return False
    return parsed is None or parsed in ({}, [], '')


def _read_tool_record(item: Any, item_type: str) -> _ToolRecord:
    explicit_call_id = _gf(item, 'call_id')
    wire_id = _gf(item, 'id')
    is_function_call = item_type == 'function_call'
    is_native_output = item_type == 'function_call_output'
    key = explicit_call_id or (wire_id if is_function_call else None)
    raw_args = _gf(item, 'arguments')
    arguments = (
        None
        if raw_args is None or raw_args == ''
        else raw_args
        if isinstance(raw_args, str)
        else json.dumps(raw_args, default=str)
    )
    result = _gf(item, 'result')
    if result is None:
        result = _gf(item, 'output')
    raw_error = _gf(item, 'error')
    error = tool_result_to_text(raw_error) if _has_error(raw_error) else None
    return _ToolRecord(
        key=str(key) if key else None,
        wire_id=wire_id if isinstance(wire_id, str) and wire_id else None,
        is_function_call=is_function_call,
        is_native_output=is_native_output,
        explicit_name=str(_gf(item, 'name')) if _gf(item, 'name') else None,
        derived_name=item_type.split(':', 1)[1] if item_type.startswith('orq:') else '',
        arguments=arguments,
        result=tool_result_to_text(result) if result is not None else None,
        error=error,
        failed=str(_gf(item, 'status')).lower() in ('failed', 'error'),
    )


def _json_key(text: str) -> tuple[str, str]:
    try:
        return ('json', json.dumps(json.loads(text), sort_keys=True))
    except ValueError:
        return ('raw', text)


def _pick(field: str, values: list[str], call_id: str) -> str | None:
    """The first of ``values``; one warning naming the discarded values when they disagree as JSON."""
    if not values:
        return None
    kept = values[0]
    discarded = [v for v in values[1:] if _json_key(v) != _json_key(kept)]
    if discarded:
        logger.warning(
            'Conflicting {} for call_id {!r}: kept {!r}, discarded {}',
            field,
            call_id,
            kept,
            ', '.join(repr(v) for v in dict.fromkeys(discarded)),
        )
    return kept


def _merge(members: list[tuple[int, _ToolRecord]], *, call_id: str) -> PairedToolCall:
    """Merge the records of one call; a function_call record's fields win."""
    ordered = sorted(members, key=lambda m: not m[1].is_function_call)
    primary_index, primary = ordered[0]
    records = [r for _, r in ordered]
    call_arguments = [r.arguments for r in records if r.is_function_call and r.arguments is not None]
    # An orq: record's empty arguments are a placeholder only when a function_call carries real ones.
    argument_values = (
        call_arguments
        + [
            r.arguments
            for r in records
            if not r.is_function_call and r.arguments and not _is_placeholder_arguments(r.arguments)
        ]
        if call_arguments
        else [r.arguments for r in records if r.arguments is not None]
    )
    arguments = _pick('arguments', argument_values, call_id)
    name = _pick('name', [r.explicit_name for r in records if r.explicit_name], call_id) or next(
        (r.derived_name for r in records if r.derived_name), ''
    )
    result = _pick('result', [r.result for r in records if r.result is not None], call_id)
    error = _pick('error', [r.error for r in records if r.error is not None], call_id)
    if error is not None or any(r.failed for r in records):
        payload: dict[str, Any] = {'status': 'failed', 'error': error}
        if result is not None:
            payload['result'] = parse_json_or_value(result)
        result = json.dumps(payload, default=str)
    return PairedToolCall(
        call_id=call_id,
        item_id=primary.wire_id,
        name=name,
        arguments='{}' if arguments is None else arguments,
        has_arguments=arguments is not None,
        result=result,
        source_index=primary_index,
        indices=tuple(sorted(index for index, _ in members)),
    )


def _warn_unmatched(records: list[_ToolRecord], *, label: str, response_has_call: bool, response_has_orq: bool) -> None:
    has_args = any(r.arguments is not None for r in records)
    has_result = any(r.result is not None or r.error is not None or r.failed for r in records)
    # A lone function_call is the normal Responses shape when the caller runs the tool
    # itself; only the router's split shape (an orq: record in the response) makes a
    # missing result worth a warning.
    if has_args and not has_result:
        if response_has_orq:
            logger.warning('Tool call {!r} has no result record', label)
    elif has_result and not has_args and response_has_call:
        logger.warning('Tool result {!r} has no matching call record; its arguments are unknown', label)


def _starts_turn(item: Any) -> bool:
    item_type = _gf(item, 'type')
    is_message = item_type == 'message' or (item_type is None and _gf(item, 'role') is not None)
    return is_message and _gf(item, 'role') == 'user'


def pair_tool_records(items: Any) -> list[PairedToolCall]:
    """Pair the tool records in ``items`` into calls, in input order; other items are ignored.

    Pairing is scoped to one turn: a ``user`` message starts a new scope, so the
    same ``call_id`` in another turn (or another list) never pairs. The wire ``id``
    is a correlation key only for ``function_call`` records, and a record without a
    usable ``call_id`` gets the id ``call_unpaired_<index>`` and never pairs.
    Records that share a ``call_id`` become one call: a ``function_call`` record's
    fields win, otherwise the first value seen, and a field on which they disagree
    logs one warning naming the discarded values. A native ``function_call_output``
    item is a result record like an ``orq:<tool>`` one, so it merges into the call
    with its ``call_id`` through the same rules; one with no call or ``orq:`` record
    to join is left out of the result.
    """
    items = list(items)
    records: list[tuple[int, int, _ToolRecord]] = []
    scope = 0
    for index, item in enumerate(items):
        if _starts_turn(item):
            scope += 1
        item_type = _gf(item, 'type')
        if is_tool_record(item_type) or item_type == 'function_call_output':
            records.append((scope, index, _read_tool_record(item, str(item_type))))
    response_has_call = any(r.is_function_call for _, _, r in records)
    response_has_orq = any(not r.is_function_call and not r.is_native_output for _, _, r in records)
    groups: dict[tuple[int, str], list[tuple[int, _ToolRecord]]] = {}
    for scope, index, record in records:
        key = record.key or f'call_unpaired_{index}'
        groups.setdefault((scope, key), []).append((index, record))
    paired: list[PairedToolCall] = []
    for members in groups.values():
        if all(r.is_native_output for _, r in members):
            continue
        key = members[0][1].key or f'call_unpaired_{members[0][0]}'
        paired.append(_merge(members, call_id=key))
        _warn_unmatched(
            [r for _, r in members],
            label=key,
            response_has_call=response_has_call,
            response_has_orq=response_has_orq,
        )
    return sorted(paired, key=lambda call: call.source_index)


def normalize_tool_records(items: Any) -> list[Any]:
    """Return ``items`` with every tool record rewritten as standard Responses items.

    Each call becomes one ``function_call`` item (``type``, ``id`` when the wire id
    is known, ``call_id``, ``name``, ``arguments``) followed by a
    ``function_call_output`` item when a result exists, at the position of the
    call's ``function_call`` record. Other items pass through untouched, in order.
    """
    items = list(items)
    paired = pair_tool_records(items)
    calls = {call.source_index: call for call in paired}
    consumed = {index for call in paired for index in call.indices}
    normalized: list[Any] = []
    for index, item in enumerate(items):
        call = calls.get(index)
        if call is not None:
            function_call: dict[str, Any] = {'type': 'function_call'}
            if call.item_id:
                function_call['id'] = call.item_id
            function_call.update(call_id=call.call_id, name=call.name, arguments=call.arguments)
            normalized.append(function_call)
            if call.result is not None:
                normalized.append({'type': 'function_call_output', 'call_id': call.call_id, 'output': call.result})
        elif index not in consumed:
            normalized.append(item)
    return normalized
