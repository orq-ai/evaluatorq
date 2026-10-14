"""Pairing of split Orq tool records, tested on the pure helper (no AgentResponse)."""

# ruff: noqa: S101

from __future__ import annotations

import json
import logging

import pytest

from evaluatorq.openresponses.tool_records import is_tool_record, pair_tool_records

CALL = {'type': 'function_call', 'id': 'fc_1', 'call_id': 'c1', 'name': 'refund', 'arguments': '{"order_id":"a2"}'}
RESULT = {'type': 'orq:refund', 'id': 'orq_9', 'call_id': 'c1', 'name': 'refund', 'result': {'ok': False}}
TEXT = {'type': 'message', 'content': [{'type': 'output_text', 'text': 'hi'}]}


def _warnings(caplog) -> str:
    return '\n'.join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)


def test_is_tool_record() -> None:
    assert is_tool_record('function_call')
    assert is_tool_record('orq:refund')
    assert not is_tool_record('function_call_output')
    assert not is_tool_record('message')
    assert not is_tool_record(None)


def test_split_pair_becomes_one_call() -> None:
    (call,) = pair_tool_records([CALL, RESULT])
    assert (call.call_id, call.item_id, call.name) == ('c1', 'fc_1', 'refund')
    assert call.arguments == '{"order_id":"a2"}'
    assert call.has_arguments
    assert call.result == '{"ok": false}'
    assert not call.failed
    assert call.source_index == 0


def test_reordered_pair_pairs_without_warning_at_the_call_position(caplog) -> None:
    (call,) = pair_tool_records([RESULT, TEXT, CALL])
    assert call.arguments == '{"order_id":"a2"}'
    assert call.item_id == 'fc_1'
    assert call.source_index == 2
    assert not _warnings(caplog)


def test_non_tool_items_are_ignored_and_indices_refer_to_the_input() -> None:
    calls = pair_tool_records([TEXT, CALL, TEXT, RESULT])
    assert [c.source_index for c in calls] == [1]


def test_calls_come_back_in_input_order() -> None:
    items = [
        {**CALL, 'id': 'fc_a', 'call_id': 'a', 'arguments': '{"n":1}'},
        {**CALL, 'id': 'fc_b', 'call_id': 'b', 'arguments': '{"n":2}'},
        {**RESULT, 'call_id': 'b', 'result': 'rb'},
        {**RESULT, 'call_id': 'a', 'result': 'ra'},
    ]
    assert [(c.call_id, c.arguments, c.result, c.source_index) for c in pair_tool_records(items)] == [
        ('a', '{"n":1}', 'ra', 0),
        ('b', '{"n":2}', 'rb', 1),
    ]


def test_orphan_result_has_no_arguments_and_keeps_its_wire_id(caplog) -> None:
    (call,) = pair_tool_records([RESULT])
    assert not call.has_arguments
    assert call.arguments == '{}'
    assert call.item_id == 'orq_9'
    assert not _warnings(caplog)


def test_unmatched_result_warns_when_the_list_has_a_call(caplog) -> None:
    calls = pair_tool_records([CALL, {**RESULT, 'call_id': 'zz'}])
    assert len(calls) == 2
    assert 'has no matching call record' in _warnings(caplog)


def test_unmatched_call_warns_beside_an_orq_record_but_a_lone_call_does_not(caplog) -> None:
    other = {'type': 'orq:other', 'call_id': 'c2', 'arguments': '{}', 'result': 'ok'}
    pair_tool_records([CALL, other])
    assert 'has no result record' in _warnings(caplog)
    caplog.clear()
    (call,) = pair_tool_records([CALL])
    assert call.result is None
    assert not _warnings(caplog)


def test_record_without_call_id_gets_a_unique_id_and_never_pairs(caplog) -> None:
    a = {'type': 'orq:t', 'arguments': '{"a":1}', 'result': 'x'}
    b = {'type': 'orq:t', 'arguments': '{"b":2}', 'result': 'y'}
    first, second = pair_tool_records([a, b])
    assert first.call_id and second.call_id
    assert first.call_id != second.call_id


def test_unmatched_result_without_call_id_warns_with_the_minted_id(caplog) -> None:
    calls = pair_tool_records([CALL, {'type': 'orq:refund', 'result': 'x'}])
    minted = next(c.call_id for c in calls if c.call_id != 'c1')
    assert f'{minted!r} has no matching call record' in _warnings(caplog)


@pytest.mark.parametrize('placeholder', [{}, '{}', [], None, ''])
def test_empty_orq_arguments_are_a_placeholder(placeholder, caplog) -> None:
    (call,) = pair_tool_records([CALL, {**RESULT, 'arguments': placeholder}])
    assert call.arguments == '{"order_id":"a2"}'
    assert 'Conflicting' not in _warnings(caplog)


def test_conflicting_arguments_keep_the_calls_and_warn(caplog) -> None:
    (call,) = pair_tool_records([CALL, {**RESULT, 'arguments': '{"x":1}'}])
    assert call.arguments == '{"order_id":"a2"}'
    assert 'Conflicting arguments for call_id' in _warnings(caplog)


def test_conflicting_results_split_into_derived_ids(caplog) -> None:
    calls = pair_tool_records([CALL, RESULT, {**RESULT, 'result': 'other'}, {**RESULT, 'result': 'third'}])
    assert [c.call_id for c in calls] == ['c1', 'c1#2', 'c1#3']
    assert [c.result for c in calls] == ['{"ok": false}', 'other', 'third']
    assert {c.arguments for c in calls} == {'{"order_id":"a2"}'}
    assert {c.item_id for c in calls} == {'fc_1'}
    assert {c.source_index for c in calls} == {0}
    assert _warnings(caplog).count('Conflicting result for call_id') == 1


def test_results_are_compared_as_json() -> None:
    calls = pair_tool_records([CALL, {**RESULT, 'result': '{"a": 1, "b": 2}'}, {**RESULT, 'result': '{"b":2,"a":1}'}])
    assert len(calls) == 1


def test_explicit_name_conflict_keeps_the_calls_name(caplog) -> None:
    (call,) = pair_tool_records([CALL, {**RESULT, 'name': 'other'}])
    assert call.name == 'refund'
    assert 'Conflicting name for call_id' in _warnings(caplog)


def test_failed_record_carries_its_error_and_keeps_a_real_result_under_result() -> None:
    (plain,) = pair_tool_records([CALL, {'type': 'orq:refund', 'call_id': 'c1', 'status': 'failed', 'error': 'boom'}])
    assert plain.failed
    assert json.loads(plain.result or '') == {'status': 'failed', 'error': 'boom'}
    (with_result,) = pair_tool_records([CALL, {**RESULT, 'status': 'failed', 'error': 'boom'}])
    assert json.loads(with_result.result or '') == {'status': 'failed', 'error': 'boom', 'result': '{"ok": false}'}


def test_same_call_id_in_separate_lists_is_not_cross_paired() -> None:
    (only_call,) = pair_tool_records([CALL])
    (only_result,) = pair_tool_records([RESULT])
    assert only_call.result is None
    assert not only_result.has_arguments
