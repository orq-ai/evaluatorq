"""Pairing of split Orq tool records, tested on the pure helper (no AgentResponse)."""

# ruff: noqa: S101

from __future__ import annotations

import json
import logging

import pytest

from evaluatorq.openresponses.tool_records import (
    PairedToolCall,
    is_tool_record,
    iter_paired,
    normalize_tool_records,
    pair_tool_records,
)

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
    assert call.result == '{"ok": false}'
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
    assert call.arguments is None
    assert call.item_id == 'orq_9'
    warnings = _warnings(caplog)
    assert warnings.count('has no matching call record') == 1
    assert "'refund'" in warnings
    assert "'c1'" in warnings


def test_orphan_result_in_normalized_items_has_unknown_empty_arguments(caplog) -> None:
    function_call, _ = normalize_tool_records([RESULT])
    assert function_call['arguments'] == ''


def test_call_with_real_arguments_and_orq_result_does_not_warn(caplog) -> None:
    (call,) = pair_tool_records([CALL, RESULT])
    assert call.arguments == '{"order_id":"a2"}'
    assert not _warnings(caplog)


def test_explicit_empty_call_arguments_stay_empty_object() -> None:
    (call,) = pair_tool_records([{**CALL, 'arguments': '{}'}, RESULT])
    assert call.arguments == '{}'


def test_unmatched_result_warns_when_the_list_has_a_call(caplog) -> None:
    calls = pair_tool_records([CALL, {**RESULT, 'call_id': 'zz'}])
    assert len(calls) == 2
    assert 'has no matching call record' in _warnings(caplog)


def test_unmatched_call_warns_beside_an_orq_record_but_a_lone_call_does_not(caplog) -> None:
    other = {'type': 'orq:refund', 'call_id': 'c2', 'arguments': '{}', 'result': 'ok'}
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
    assert f'call_id {minted!r}) has no matching call record' in _warnings(caplog)


@pytest.mark.parametrize('placeholder', [{}, '{}', [], None, ''])
def test_empty_orq_arguments_are_a_placeholder(placeholder, caplog) -> None:
    (call,) = pair_tool_records([CALL, {**RESULT, 'arguments': placeholder}])
    assert call.arguments == '{"order_id":"a2"}'
    assert 'Conflicting' not in _warnings(caplog)


def test_conflicting_arguments_keep_the_calls_and_warn(caplog) -> None:
    (call,) = pair_tool_records([CALL, {**RESULT, 'arguments': '{"x":1}'}])
    assert call.arguments == '{"order_id":"a2"}'
    assert 'Conflicting arguments for call_id' in _warnings(caplog)


def test_conflicting_results_keep_the_call_records_result_and_warn_once(caplog) -> None:
    (call,) = pair_tool_records([CALL, RESULT, {**RESULT, 'result': 'other'}, {**RESULT, 'result': 'third'}])
    assert call.call_id == 'c1'
    assert call.result == '{"ok": false}'
    warnings = _warnings(caplog)
    assert warnings.count('Conflicting result for call_id') == 1
    assert "'other'" in warnings
    assert "'third'" in warnings
    assert '#' not in call.call_id


def test_result_on_the_function_call_record_beats_an_orq_result(caplog) -> None:
    (call,) = pair_tool_records([{**RESULT, 'result': 'orq'}, {**CALL, 'result': 'fc'}])
    assert call.result == 'fc'
    assert 'Conflicting result' in _warnings(caplog)


def test_second_function_call_for_a_call_id_is_its_own_call(caplog) -> None:
    first, second = pair_tool_records([CALL, {**CALL, 'id': 'fc_2', 'arguments': '{"order_id":"zz"}', 'name': 'other'}])
    assert (first.arguments, first.name, first.item_id) == ('{"order_id":"a2"}', 'refund', 'fc_1')
    assert (second.arguments, second.name, second.item_id) == ('{"order_id":"zz"}', 'other', 'fc_2')
    assert 'Conflicting' not in _warnings(caplog)


def test_conflicting_errors_keep_the_first_and_warn_once(caplog) -> None:
    one = {'type': 'orq:refund', 'call_id': 'c1', 'error': 'boom'}
    two = {'type': 'orq:refund', 'call_id': 'c1', 'error': 'bang'}
    (call,) = pair_tool_records([CALL, one, two])
    assert json.loads(call.result or '') == {'status': 'failed', 'error': 'boom'}
    assert _warnings(caplog).count('Conflicting error for call_id') == 1


def test_results_are_compared_as_json() -> None:
    calls = pair_tool_records([CALL, {**RESULT, 'result': '{"a": 1, "b": 2}'}, {**RESULT, 'result': '{"b":2,"a":1}'}])
    assert len(calls) == 1


def test_explicit_name_conflict_keeps_the_calls_name(caplog) -> None:
    (call,) = pair_tool_records([CALL, {**RESULT, 'name': 'other'}])
    assert call.name == 'refund'
    assert 'Conflicting name for call_id' in _warnings(caplog)


def test_failed_record_carries_its_error_and_the_parsed_result() -> None:
    (plain,) = pair_tool_records([CALL, {'type': 'orq:refund', 'call_id': 'c1', 'status': 'failed', 'error': 'boom'}])
    assert json.loads(plain.result or '') == {'status': 'failed', 'error': 'boom'}
    (with_result,) = pair_tool_records([CALL, {**RESULT, 'status': 'failed', 'error': 'boom'}])
    assert json.loads(with_result.result or '') == {'status': 'failed', 'error': 'boom', 'result': {'ok': False}}


def test_failed_record_with_a_non_json_result_keeps_the_string() -> None:
    (call,) = pair_tool_records([CALL, {**RESULT, 'result': 'plain text', 'status': 'failed'}])
    assert json.loads(call.result or '') == {'status': 'failed', 'error': None, 'result': 'plain text'}


def test_record_with_an_error_but_no_failed_status_is_wrapped_as_failed() -> None:
    (call,) = pair_tool_records([CALL, {'type': 'orq:refund', 'call_id': 'c1', 'error': 'boom'}])
    assert json.loads(call.result or '') == {'status': 'failed', 'error': 'boom'}


def test_lone_orq_record_with_placeholder_arguments_is_unknown() -> None:
    (call,) = pair_tool_records([{'type': 'orq:kb', 'call_id': 'k', 'arguments': {}, 'result': 'x'}])
    assert call.arguments is None


def test_non_json_arguments_are_encoded_with_a_str_fallback() -> None:
    (call,) = pair_tool_records([{**CALL, 'arguments': {'when': object}}])
    assert call.arguments is not None
    assert call.arguments.startswith('{"when": "<class')


def test_same_call_id_in_two_turns_is_two_calls_without_a_conflict(caplog) -> None:
    user = {'type': 'message', 'role': 'user', 'content': 'again'}
    items = [
        CALL,
        RESULT,
        user,
        {**CALL, 'arguments': '{"order_id":"b3"}'},
        {**RESULT, 'result': 'second'},
    ]
    first, second = pair_tool_records(items)
    assert (first.arguments, first.result) == ('{"order_id":"a2"}', '{"ok": false}')
    assert (second.arguments, second.result) == ('{"order_id":"b3"}', 'second')
    assert not _warnings(caplog)
    assert [i.get('call_id') for i in normalize_tool_records(items) if i['type'] == 'function_call'] == ['c1', 'c1']


def test_function_call_output_item_counts_as_the_result(caplog) -> None:
    items = [
        {'type': 'function_call', 'call_id': 'a', 'name': 'x', 'arguments': '{}'},
        {'type': 'function_call_output', 'call_id': 'a', 'output': 'done'},
        {'type': 'orq:kb', 'call_id': 'b', 'arguments': '{"q":1}', 'result': 'hit'},
    ]
    pair_tool_records(items)
    assert not _warnings(caplog)


def test_minted_ids_are_content_derived_and_unique_within_a_list() -> None:
    first = {'type': 'orq:t', 'arguments': '{"a":1}', 'result': 'x'}
    second = {'type': 'orq:t', 'result': 'y'}
    ids = [c.call_id for c in pair_tool_records([TEXT, first, second])]
    assert len(set(ids)) == 2
    assert all(i.startswith('call_unpaired_') for i in ids)
    assert ids == [c.call_id for c in pair_tool_records([TEXT, first, second])]
    assert normalize_tool_records([first, second]) == normalize_tool_records([first, second])
    twins = [c.call_id for c in pair_tool_records([second, second])]
    assert len(set(twins)) == 2


def test_minted_ids_do_not_repeat_across_responses() -> None:
    one = [{'type': 'orq:t', 'arguments': '{"a":1}', 'result': 'x'}]
    other = [{'type': 'orq:t', 'arguments': '{"a":2}', 'result': 'y'}]
    (first,) = pair_tool_records([TEXT, *one])
    (again,) = pair_tool_records(one)
    (different,) = pair_tool_records([TEXT, *other])
    assert first.call_id == again.call_id
    assert first.call_id != different.call_id


def test_reused_call_id_in_one_turn_starts_a_new_call() -> None:
    items = [CALL, RESULT, {**CALL, 'arguments': '{"order_id":"b3"}'}, {**RESULT, 'result': 'second'}]
    first, second = pair_tool_records(items)
    assert (first.arguments, first.result) == ('{"order_id":"a2"}', '{"ok": false}')
    assert (second.arguments, second.result) == ('{"order_id":"b3"}', 'second')
    identical = pair_tool_records([CALL, CALL, RESULT])
    assert [c.result for c in identical] == [None, '{"ok": false}']
    assert [c.source_index for c in identical] == [0, 1]


def test_only_a_result_side_record_marks_a_call_failed() -> None:
    (call,) = pair_tool_records([{**CALL, 'status': 'failed'}, RESULT])
    assert call.result == '{"ok": false}'
    (failed,) = pair_tool_records([CALL, {**RESULT, 'status': 'failed'}])
    assert json.loads(failed.result or '')['status'] == 'failed'


def test_incomplete_status_is_not_a_failure() -> None:
    (call,) = pair_tool_records([CALL, {**RESULT, 'status': 'incomplete'}])
    assert call.result == '{"ok": false}'


def test_failed_native_output_without_a_call_is_yielded_as_a_failure_payload() -> None:
    orphan = {'type': 'function_call_output', 'call_id': 'z', 'output': 'x', 'status': 'failed'}
    (entry,) = list(iter_paired([orphan]))
    assert entry['call_id'] == 'z'
    assert json.loads(entry['output'])['status'] == 'failed'


def test_normalize_rewrites_records_in_place_and_passes_other_items_through() -> None:
    items = [RESULT, TEXT, CALL, {'type': 'reasoning', 'id': 'r'}]
    assert normalize_tool_records(items) == [
        TEXT,
        {'type': 'function_call', 'id': 'fc_1', 'call_id': 'c1', 'name': 'refund', 'arguments': '{"order_id":"a2"}'},
        {'type': 'function_call_output', 'call_id': 'c1', 'output': '{"ok": false}'},
        {'type': 'reasoning', 'id': 'r'},
    ]


def test_normalize_leaves_a_plain_responses_transcript_unchanged() -> None:
    plain = [
        {'type': 'function_call', 'id': 'fc_1', 'call_id': 'c1', 'name': 't', 'arguments': '{}'},
        {'type': 'function_call_output', 'call_id': 'c1', 'output': 'ok'},
        TEXT,
    ]
    assert normalize_tool_records(plain) == plain


def test_same_call_id_in_separate_lists_is_not_cross_paired() -> None:
    (only_call,) = pair_tool_records([CALL])
    (only_result,) = pair_tool_records([RESULT])
    assert only_call.result is None
    assert only_result.arguments is None


def test_parallel_native_calls_pair_by_call_id_not_adjacency(caplog) -> None:
    items = [
        {'type': 'function_call', 'call_id': 'a', 'name': 'x', 'arguments': '{}'},
        {'type': 'function_call', 'call_id': 'b', 'name': 'x', 'arguments': '{}'},
        {'type': 'function_call_output', 'call_id': 'a', 'output': 'ra'},
        {'type': 'function_call_output', 'call_id': 'b', 'output': 'rb'},
    ]
    assert [(c.call_id, c.result) for c in pair_tool_records(items)] == [('a', 'ra'), ('b', 'rb')]
    assert normalize_tool_records(items) == items
    assert not _warnings(caplog)


def test_orq_result_and_native_output_with_one_call_id_collapse_to_one_output(caplog) -> None:
    same = {'type': 'function_call_output', 'call_id': 'c1', 'output': '{"ok": false}'}
    assert [i['type'] for i in normalize_tool_records([CALL, RESULT, same])] == ['function_call', 'function_call_output']
    assert not _warnings(caplog)
    differing = {**same, 'output': 'other'}
    normalized = normalize_tool_records([CALL, RESULT, differing])
    assert [i['output'] for i in normalized if i['type'] == 'function_call_output'] == ['{"ok": false}']
    assert _warnings(caplog).count('Conflicting result for call_id') == 1


def test_native_output_without_a_call_passes_through_unpaired() -> None:
    orphan = {'type': 'function_call_output', 'call_id': 'z', 'output': 'x'}
    assert pair_tool_records([orphan]) == []
    assert normalize_tool_records([orphan]) == [orphan]


@pytest.mark.parametrize('empty', [{}, [], '', None, False])
def test_empty_error_on_a_successful_record_stays_a_success(empty) -> None:
    (call,) = pair_tool_records([CALL, {**RESULT, 'error': empty}])
    assert call.result == '{"ok": false}'


@pytest.mark.parametrize('error', [{'message': 'boom'}, ['boom'], 'boom'])
def test_non_empty_error_marks_the_call_failed(error) -> None:
    (call,) = pair_tool_records([CALL, {**RESULT, 'error': error}])
    assert json.loads(call.result or '')['status'] == 'failed'


def test_lone_failed_function_call_gets_no_invented_output() -> None:
    (call,) = pair_tool_records([{**CALL, 'status': 'failed'}])
    assert call.result is None
    (with_error,) = pair_tool_records([{**CALL, 'error': 'boom'}])
    assert json.loads(with_error.result or '') == {'status': 'failed', 'error': 'boom'}


def test_unpaired_native_output_with_an_error_keeps_it() -> None:
    orphan = {'type': 'function_call_output', 'call_id': 'z', 'error': 'boom'}
    (out,) = normalize_tool_records([orphan])
    assert out['call_id'] == 'z'
    assert json.loads(out['output']) == {'status': 'failed', 'error': 'boom'}


def test_iter_paired_yields_a_call_at_its_first_record_and_passes_the_rest_through() -> None:
    entries = list(iter_paired([TEXT, RESULT, CALL]))
    assert entries[0] == TEXT
    assert isinstance(entries[1], PairedToolCall)
    assert len(entries) == 2


def test_warning_flags_are_scoped_to_the_turn(caplog) -> None:
    user = {'type': 'message', 'role': 'user', 'content': 'next'}
    other = {'type': 'orq:other', 'call_id': 'c9', 'arguments': '{}', 'result': 'ok'}
    pair_tool_records([other, user, CALL])
    assert 'has no result record' not in _warnings(caplog)


def test_pending_client_call_beside_an_orq_tool_stays_silent(caplog) -> None:
    other = {'type': 'orq:other', 'call_id': 'c9', 'arguments': '{}', 'result': 'ok'}
    pair_tool_records([CALL, other])
    assert 'has no result record' not in _warnings(caplog)


def test_call_without_call_id_does_not_pair_with_an_orq_call_id_equal_to_its_wire_id() -> None:
    bare = {'type': 'function_call', 'id': 'fc_1', 'name': 'refund', 'arguments': '{}'}
    first, second = pair_tool_records([bare, {'type': 'orq:refund', 'call_id': 'fc_1', 'result': 'x'}])
    assert (first.result, second.result) == (None, 'x')
