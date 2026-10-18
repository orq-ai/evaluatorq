"""Pair the split tool records of an Orq router response.

The Orq router can report one tool call as two wire records: a ``function_call``
(the arguments) and an ``orq:<tool>`` record (the result), tied together by
``call_id``; a native ``function_call_output`` item is one more result record for
its ``call_id``. Every reader of those records needs the same pairing rules, so
they live here, on plain wire dicts or objects, with no dependency on
``contracts``. ``iter_paired`` is the one walk readers build on, and
``normalize_tool_records`` rewrites the records into standard ``function_call`` /
``function_call_output`` items.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq.common.fields import get_field as _gf
from evaluatorq.common.messages import UNKNOWN_ARGUMENTS, parse_json_or_value, tool_result_to_text

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True)
class PairedToolCall:
    """One tool call with the records that described it merged.

    ``source_index`` is the input position of the ``function_call`` record (of the
    first record when there is none), so a reader can place the call among the
    non-tool items it also handles. ``arguments`` is ``None`` when no record carried
    real arguments (a result whose call record never arrived, or an ``orq:`` placeholder
    such as ``{}``); ``'{}'`` is kept only for a call that was made with no arguments.
    ``item_id`` is the wire ``id`` of the ``function_call``
    record, or of the lone record when no call record exists. ``indices`` are the
    input positions of every record merged into the call.
    """

    call_id: str
    item_id: str | None
    name: str
    arguments: str | None
    result: str | None
    source_index: int
    indices: tuple[int, ...]


@dataclass(frozen=True)
class _ToolRecord:
    """One wire tool record, normalised for pairing."""

    key: str | None
    keyed_by_wire_id: bool
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


def _has_error(raw: Any) -> bool:
    """Whether an ``error`` field reports a failure: a non-empty string, dict or list."""
    return bool(raw) and isinstance(raw, (str, dict, list))


def _is_placeholder_arguments(text: str) -> bool:
    parsed = parse_json_or_value(text)
    return parsed is None or parsed in ({}, [], '')


def _read_tool_record(item: Any, item_type: str) -> _ToolRecord:
    explicit_call_id = _gf(item, 'call_id')
    wire_id = _gf(item, 'id')
    is_function_call = item_type == 'function_call'
    is_native_output = item_type == 'function_call_output'
    name = _gf(item, 'name')
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
        keyed_by_wire_id=not explicit_call_id and bool(key),
        wire_id=wire_id if isinstance(wire_id, str) and wire_id else None,
        is_function_call=is_function_call,
        is_native_output=is_native_output,
        explicit_name=str(name) if name else None,
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


def failure_payload(result: str | None, error: str | None) -> str:
    """The one JSON envelope for a failed tool call, shared by every reader of tool records."""
    payload: dict[str, Any] = {'status': 'failed', 'error': error}
    if result is not None:
        payload['result'] = parse_json_or_value(result)
    return json.dumps(payload, default=str)


def _merge(members: list[tuple[int, _ToolRecord]], *, call_id: str) -> PairedToolCall:
    """Merge the records of one call; a function_call record's fields win."""
    ordered = sorted(members, key=lambda m: not m[1].is_function_call)
    primary_index, primary = ordered[0]
    records = [r for _, r in ordered]
    argument_values = [r.arguments for r in records if r.is_function_call and r.arguments is not None] + [
        r.arguments
        for r in records
        if not r.is_function_call and r.arguments and not _is_placeholder_arguments(r.arguments)
    ]
    arguments = _pick('arguments', argument_values, call_id)
    name = _pick('name', [r.explicit_name for r in records if r.explicit_name], call_id) or next(
        (r.derived_name for r in records if r.derived_name), ''
    )
    result = _pick('result', [r.result for r in records if r.result is not None], call_id)
    error = _pick('error', [r.error for r in records if r.error is not None], call_id)
    if error is not None or any(r.failed for r in records if not r.is_function_call):
        result = failure_payload(result, error)
    return PairedToolCall(
        call_id=call_id,
        item_id=primary.wire_id,
        name=name,
        arguments=arguments,
        result=result,
        source_index=primary_index,
        indices=tuple(sorted(index for index, _ in members)),
    )


def _warn_unmatched(records: list[_ToolRecord], call: PairedToolCall, *, scope_orq_names: frozenset[str]) -> None:
    has_args = any(r.arguments is not None for r in records)
    has_result = any(r.result is not None or r.error is not None or r.failed for r in records)
    # A lone function_call is the normal Responses shape when the caller runs the tool
    # itself; only a call the router plausibly ran (an orq: record of the same tool in
    # the same turn) makes a missing result worth a warning.
    if has_args and not has_result:
        if any(r.explicit_name in scope_orq_names for r in records if r.is_function_call and r.explicit_name):
            logger.warning('Tool call {!r} has no result record', call.call_id)
    elif has_result and call.arguments is None:
        logger.warning(
            'Tool result for {!r} (call_id {!r}) has no matching call record; its arguments are unavailable',
            call.name,
            call.call_id,
        )


def _starts_turn(item: Any) -> bool:
    item_type = _gf(item, 'type')
    is_message = item_type == 'message' or (item_type is None and _gf(item, 'role') is not None)
    return is_message and _gf(item, 'role') == 'user'


def _mint_ids(records: list[tuple[int, int, _ToolRecord]]) -> dict[int, str]:
    """Content-derived ids for records without a usable id, unique within one list."""
    minted: dict[int, str] = {}
    used: set[str] = set()
    for _, index, record in records:
        if record.key:
            continue
        digest = hashlib.sha1(  # noqa: S324
            json.dumps(
                [
                    record.is_function_call,
                    record.is_native_output,
                    record.explicit_name or record.derived_name,
                    record.arguments,
                    record.result,
                    record.error,
                ],
                default=str,
            ).encode()
        ).hexdigest()[:8]
        call_id = f'call_unpaired_{digest}'
        if call_id in used:
            call_id = f'{call_id}_{index}'
        used.add(call_id)
        minted[index] = call_id
    return minted


def _pair(items: list[Any]) -> tuple[list[PairedToolCall], dict[int, str]]:
    records: list[tuple[int, int, _ToolRecord]] = []
    scope = 0
    for index, item in enumerate(items):
        if _starts_turn(item):
            scope += 1
        item_type = _gf(item, 'type')
        if is_tool_record(item_type) or item_type == 'function_call_output':
            records.append((scope, index, _read_tool_record(item, str(item_type))))
    minted = _mint_ids(records)
    scope_orq_names: dict[int, set[str]] = {}
    for scope, _, r in records:
        if not r.is_function_call and not r.is_native_output:
            scope_orq_names.setdefault(scope, set()).add(r.derived_name)
    groups: dict[tuple[Any, ...], list[tuple[int, _ToolRecord]]] = {}
    generation: dict[tuple[Any, ...], int] = {}
    has_call: set[tuple[Any, ...]] = set()
    for scope, index, record in records:
        if not record.key:
            groups[scope, False, None, index] = [(index, record)]
            continue
        base = (scope, record.keyed_by_wire_id, record.key)
        if record.is_function_call:
            if (base, generation.get(base, 0)) in has_call:
                generation[base] = generation.get(base, 0) + 1
            has_call.add((base, generation.get(base, 0)))
        groups.setdefault((*base, generation.get(base, 0)), []).append((index, record))
    paired: list[PairedToolCall] = []
    for group_key, members in groups.items():
        if all(r.is_native_output for _, r in members):
            continue
        key = members[0][1].key or minted[members[0][0]]
        scope = group_key[0]
        call = _merge(members, call_id=key)
        paired.append(call)
        _warn_unmatched([r for _, r in members], call, scope_orq_names=frozenset(scope_orq_names.get(scope, ())))
    return sorted(paired, key=lambda call: call.source_index), minted


def pair_tool_records(items: Any) -> list[PairedToolCall]:
    """Pair the tool records in ``items`` into calls, in input order; other items are ignored.

    Pairing is scoped to one turn: a ``user`` message starts a new scope, so the
    same ``call_id`` in another turn (or another list) never pairs. The wire ``id``
    names the call only for a ``function_call`` without a ``call_id``, and then it
    pairs only with records that name the call the same way, never with an
    ``orq:`` record's ``call_id``. A second ``function_call`` for a ``call_id`` that
    already has one starts a new call, and later results join that newest call. A
    record with no usable id never pairs and gets the id ``call_unpaired_<hash>``,
    where the hash derives from its content, so the same record always gets the same
    id (``_<index>`` is appended only when two records in one list would collide).
    Records that share a ``call_id`` become one call: a ``function_call`` record's
    fields win, otherwise the first value seen, and a field on which they disagree
    logs one warning naming the discarded values. A native ``function_call_output``
    item is a result record like an ``orq:<tool>`` one, so it merges into the call
    with its ``call_id`` through the same rules; one with no call or ``orq:`` record
    to join is left out of the result. Only a result-side record (an ``orq:`` record
    or a native output) can mark a call failed.
    """
    return _pair(list(items))[0]


def _failed_output(record: _ToolRecord, call_id: str) -> dict[str, Any]:
    return {'type': 'function_call_output', 'call_id': call_id, 'output': failure_payload(record.result, record.error)}


def iter_paired(items: Any) -> Iterator[Any]:
    """Walk ``items`` in order, yielding each merged call once and every other item untouched.

    A ``PairedToolCall`` is yielded at the position of its ``function_call`` record
    (its first record when there is none) and the rest of its records are skipped.
    A native ``function_call_output`` with no call to join passes through, except
    one that reports a failure: it is yielded as a ``function_call_output`` dict
    whose output carries the failure payload. That dict is consumed only by the
    OTel readers; ``AgentResponse.from_output_items`` still skips such an output
    with its existing warning.

    Yields:
        A ``PairedToolCall`` per merged call, otherwise the input item.
    """
    items = list(items)
    calls_list, minted = _pair(items)
    calls = {call.source_index: call for call in calls_list}
    consumed = {index for call in calls.values() for index in call.indices}
    for index, item in enumerate(items):
        if index in calls:
            yield calls[index]
        elif index in consumed:
            continue
        elif _gf(item, 'type') == 'function_call_output':
            record = _read_tool_record(item, 'function_call_output')
            if record.error is not None or record.failed:
                yield _failed_output(record, record.key or minted[index])
            else:
                yield item
        else:
            yield item


def normalize_tool_records(items: Any) -> list[Any]:
    """Return ``items`` with split Orq tool records rewritten as standard Responses items.

    A call with an ``orq:`` result becomes one ``function_call`` item (``type``,
    ``id`` when the wire id is known, ``call_id``, ``name``, ``arguments``) at the
    position of its ``function_call`` record, followed by one ``function_call_output``
    item when a result exists. Calls made only of native records keep every record
    at its original index, so parallel calls keep their order and wire fields; a
    native output that reports a failure is replaced in place by its failure
    payload. Other items pass through untouched, in order.
    """
    items = list(items)
    calls, minted = _pair(items)
    rewritten = {
        call.source_index: call
        for call in calls
        if any(str(_gf(items[i], 'type')).startswith('orq:') for i in call.indices)
    }
    consumed = {index for call in rewritten.values() for index in call.indices}
    normalized: list[Any] = []
    for index, item in enumerate(items):
        if index in rewritten:
            entry = rewritten[index]
            function_call: dict[str, Any] = {'type': 'function_call'}
            if entry.item_id:
                function_call['id'] = entry.item_id
            function_call.update(
                call_id=entry.call_id,
                name=entry.name,
                arguments=UNKNOWN_ARGUMENTS if entry.arguments is None else entry.arguments,
            )
            normalized.append(function_call)
            if entry.result is not None:
                normalized.append({'type': 'function_call_output', 'call_id': entry.call_id, 'output': entry.result})
        elif index in consumed:
            continue
        elif _gf(item, 'type') == 'function_call_output':
            record = _read_tool_record(item, 'function_call_output')
            if record.error is not None or record.failed:
                normalized.append(_failed_output(record, record.key or minted[index]))
            else:
                normalized.append(item)
        else:
            normalized.append(item)
    return normalized
