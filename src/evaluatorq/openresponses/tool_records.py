"""Pair the split tool records of an Orq router response.

The Orq router can report one tool call as two wire records: a ``function_call``
(the arguments) and an ``orq:<tool>`` record (the result), tied together by
``call_id``. Every reader of those records needs the same pairing rules, so they
live here, on plain wire dicts or objects, with no dependency on ``contracts``.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, replace
from typing import Any

from loguru import logger

from evaluatorq.common.fields import get_field as _gf


@dataclass(frozen=True)
class PairedToolCall:
    """One tool call with the records that described it merged.

    ``source_index`` is the input position of the ``function_call`` record (of the
    first record when there is none), so a reader can place the call among the non-tool items it also handles. ``has_arguments``
    is False for a result whose call record never arrived; ``arguments`` is then
    ``'{}'``. ``item_id`` is the wire ``id`` of the ``function_call`` record, or of
    the lone record when no call record exists.
    """

    call_id: str
    item_id: str | None
    name: str
    arguments: str
    has_arguments: bool
    result: str | None
    failed: bool
    source_index: int


@dataclass(frozen=True)
class _ToolRecord:
    """One wire tool record, normalised for pairing."""

    key: str | None
    wire_id: str | None
    is_function_call: bool
    explicit_name: str | None
    derived_name: str
    arguments: str | None
    result: str | None
    error: str | None
    failed: bool


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


def is_tool_record(item_type: Any) -> bool:
    """Whether a wire item type is a tool record: a ``function_call`` or an ``orq:<tool>``."""
    return isinstance(item_type, str) and (item_type == 'function_call' or item_type.startswith('orq:'))


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
    key = explicit_call_id or (wire_id if is_function_call else None)
    raw_args = _gf(item, 'arguments')
    arguments = (
        None if raw_args is None or raw_args == '' else raw_args if isinstance(raw_args, str) else json.dumps(raw_args)
    )
    # An orq: record's empty arguments are a placeholder, so the function_call's win.
    if arguments is not None and not is_function_call and _is_placeholder_arguments(arguments):
        arguments = None
    result = _gf(item, 'result')
    if result is None:
        result = _gf(item, 'output')
    raw_error = _gf(item, 'error')
    error = tool_result_to_text(raw_error) if raw_error not in (None, '') else None
    return _ToolRecord(
        key=str(key) if key else None,
        wire_id=wire_id if isinstance(wire_id, str) and wire_id else None,
        is_function_call=is_function_call,
        explicit_name=str(_gf(item, 'name')) if _gf(item, 'name') else None,
        derived_name='' if is_function_call else item_type.split(':', 1)[1],
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


def _conflicting_fields(records: list[_ToolRecord]) -> list[str]:
    """Fields on which the records of one call_id disagree, comparing JSON by value."""
    arguments = {_json_key(r.arguments) for r in records if r.arguments is not None}
    names = {r.explicit_name for r in records if r.explicit_name}
    results = {_json_key(r.result) for r in records if r.result is not None}
    errors = {_json_key(r.error) for r in records if r.error is not None}
    candidates = (('arguments', arguments), ('name', names), ('result', results), ('error', errors))
    return [name for name, values in candidates if len(values) > 1]


def _outcome_buckets(records: list[_ToolRecord], conflicts: list[str]) -> list[list[_ToolRecord]]:
    """Group the records into one bucket per distinct result or error; the rest join the first."""
    if 'result' not in conflicts and 'error' not in conflicts:
        return [records]

    def signature(record: _ToolRecord) -> Any:
        result = _json_key(record.result) if record.result is not None else None
        error = _json_key(record.error) if record.error is not None else None
        if 'result' in conflicts and 'error' in conflicts:
            return None if result is None and error is None else (result, error)
        return result if 'result' in conflicts else error

    buckets: dict[Any, list[_ToolRecord]] = {}
    rest: list[_ToolRecord] = []
    for record in records:
        sig = signature(record)
        if sig is None:
            rest.append(record)
        else:
            buckets.setdefault(sig, []).append(record)
    ordered = list(buckets.values())
    ordered[0].extend(rest)
    return ordered


def _merge(group: list[_ToolRecord], *, call_id: str, source_index: int) -> PairedToolCall:
    """Merge the records of ``group`` into one call; a function_call record's fields win."""
    ordered = sorted(group, key=lambda r: not r.is_function_call)
    result = next((r.result for r in ordered if r.result is not None), None)
    error = next((r.error for r in ordered if r.error is not None), None)
    failed = any(r.failed for r in ordered)
    if failed:
        payload: dict[str, Any] = {'status': 'failed', 'error': error}
        if result is not None:
            payload['result'] = result
        result = json.dumps(payload)
    elif result is None:
        result = error
    arguments = next((r.arguments for r in ordered if r.arguments is not None), None)
    return PairedToolCall(
        call_id=call_id,
        item_id=ordered[0].wire_id,
        name=next((r.explicit_name for r in ordered if r.explicit_name), None)
        or next((r.derived_name for r in ordered if r.derived_name), ''),
        arguments='{}' if arguments is None else arguments,
        has_arguments=arguments is not None,
        result=result,
        failed=failed,
        source_index=source_index,
    )


def _warn_unmatched(records: list[_ToolRecord], *, label: str, response_has_call: bool, response_has_orq: bool) -> None:
    has_args = any(r.arguments is not None for r in records)
    has_result = any(r.result is not None or r.error is not None or r.failed for r in records)
    # A lone function_call is the normal Responses shape when the caller runs the tool
    # itself; only the router's split shape (an orq: record in the response) makes a
    # missing result worth a warning.
    if has_args and not has_result:
        if not response_has_orq:
            return
        logger.warning('Tool call {!r} has no result record', label)
    elif has_result and not has_args and response_has_call:
        logger.warning('Tool result {!r} has no matching call record; its arguments are unknown', label)


def pair_tool_records(items: Any) -> list[PairedToolCall]:
    """Pair the tool records in ``items`` into calls, in input order; other items are ignored.

    Pairing is scoped to one list, so the same ``call_id`` in another turn never
    pairs. The wire ``id`` is a correlation key only for ``function_call`` records,
    and a record without a usable ``call_id`` gets a unique one and never pairs.
    Records that agree on a field are merged. Records whose results or errors
    conflict become separate calls with derived ids (``<call_id>#2``, ``#3``), each
    keeping the call's arguments and wire id. Conflicting arguments or names keep
    the call's value. Each conflict and each unmatched record logs one warning.
    """
    positions: list[tuple[int, _ToolRecord]] = []
    for index, item in enumerate(items):
        item_type = _gf(item, 'type')
        if is_tool_record(item_type):
            positions.append((index, _read_tool_record(item, str(item_type))))
    response_has_call = any(record.is_function_call for _, record in positions)
    response_has_orq = any(not record.is_function_call for _, record in positions)
    groups: dict[str, list[tuple[int, _ToolRecord]]] = {}
    paired: list[PairedToolCall] = []
    for index, record in positions:
        if record.key is None:
            call = _merge([record], call_id=f'call_{uuid.uuid4().hex}', source_index=index)
            paired.append(call)
            _warn_unmatched(
                [record], label=call.call_id, response_has_call=response_has_call, response_has_orq=response_has_orq
            )
        else:
            groups.setdefault(record.key, []).append((index, record))
    for key, members in groups.items():
        records = [record for _, record in members]
        call_record = next((r for r in records if r.is_function_call), records[0])
        index = next(i for i, r in members if r is call_record)
        conflicts = _conflicting_fields(records)
        for conflict in conflicts:
            logger.warning('Conflicting {} for call_id {!r}', conflict, key)
        call_only = replace(call_record, result=None, error=None, failed=False)
        for n, bucket in enumerate(_outcome_buckets(records, conflicts), start=1):
            group = bucket if any(r is call_record for r in bucket) else [call_only, *bucket]
            paired.append(_merge(group, call_id=key if n == 1 else f'{key}#{n}', source_index=index))
        _warn_unmatched(records, label=key, response_has_call=response_has_call, response_has_orq=response_has_orq)
    return sorted(paired, key=lambda call: call.source_index)
