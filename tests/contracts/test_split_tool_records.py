"""Orq returns one invocation as a ``function_call`` plus an ``orq:<tool>`` result sharing a call_id."""

from __future__ import annotations

from loguru import logger

from evaluatorq.contracts import AgentResponse

CALL = {'type': 'function_call', 'id': 'fc_1', 'call_id': 'c1', 'name': 'refund', 'arguments': '{"order_id":"a2"}'}
RESULT = {'type': 'orq:refund', 'id': 'orq_9', 'call_id': 'c1', 'name': 'refund', 'result': {'ok': False}}


def _warnings(fn):
    seen: list[str] = []
    sink = logger.add(lambda m: seen.append(str(m)), level='WARNING')
    try:
        return fn(), seen
    finally:
        logger.remove(sink)


def test_split_records_become_one_call_with_args_and_result():
    r = AgentResponse.from_output_items([CALL, RESULT])
    assert len(r.tool_calls) == 1
    tc = r.tool_calls[0]
    assert (tc.call_id, tc.id, tc.name) == ('c1', 'fc_1', 'refund')
    assert tc.arguments == '{"order_id":"a2"}'
    assert tc.result == '{"ok": false}'


def test_result_before_call_pairs_too():
    r, warns = _warnings(lambda: AgentResponse.from_output_items([RESULT, CALL]))
    assert len(r.tool_calls) == 1
    assert r.tool_calls[0].arguments == '{"order_id":"a2"}'
    assert not warns


def test_several_calls_keep_their_own_arguments():
    items = [
        {**CALL, 'id': 'fc_a', 'call_id': 'a', 'arguments': '{"n":1}'},
        {**CALL, 'id': 'fc_b', 'call_id': 'b', 'arguments': '{"n":2}'},
        {**RESULT, 'call_id': 'b', 'result': 'rb'},
        {**RESULT, 'call_id': 'a', 'result': 'ra'},
    ]
    got = {t.call_id: (t.arguments, t.result) for t in AgentResponse.from_output_items(items).tool_calls}
    assert got == {'a': ('{"n":1}', 'ra'), 'b': ('{"n":2}', 'rb')}


def test_empty_string_result_is_kept():
    r = AgentResponse.from_output_items([CALL, {**RESULT, 'result': ''}])
    assert r.tool_calls[0].result == ''


def test_identical_duplicate_is_collapsed():
    assert len(AgentResponse.from_output_items([CALL, CALL, RESULT, RESULT]).tool_calls) == 1


def test_conflicting_arguments_are_not_paired():
    r, warns = _warnings(lambda: AgentResponse.from_output_items([CALL, {**CALL, 'id': 'fc_2', 'arguments': '{"x":1}'}]))
    assert len(r.tool_calls) == 2
    assert any('Conflicting' in w for w in warns)


def test_orphan_result_warns_that_arguments_are_unknown():
    r, warns = _warnings(lambda: AgentResponse.from_output_items([RESULT]))
    assert len(r.tool_calls) == 1
    assert any('no matching call' in w for w in warns)


def test_unmatched_call_is_kept_without_result():
    r = AgentResponse.from_output_items([CALL])
    assert r.tool_calls[0].result is None


def test_missing_call_ids_never_pair():
    a = {'type': 'orq:t', 'arguments': '{"a":1}', 'result': 'x'}
    b = {'type': 'orq:t', 'arguments': '{"b":2}', 'result': 'y'}
    r, warns = _warnings(lambda: AgentResponse.from_output_items([a, b]))
    assert len(r.tool_calls) == 2
    assert len(warns) == 2


def test_orq_record_id_is_not_a_correlation_key():
    r = AgentResponse.from_output_items([{**RESULT, 'call_id': None, 'id': 'fc_1'}, {**CALL, 'call_id': None}])
    assert len(r.tool_calls) == 2


def test_function_call_id_fallback_still_pairs():
    call = {'type': 'function_call', 'id': 'c7', 'name': 't', 'arguments': '{"k":1}'}
    res = {'type': 'orq:t', 'call_id': 'c7', 'result': 'ok'}
    r = AgentResponse.from_output_items([call, res])
    assert len(r.tool_calls) == 1 and r.tool_calls[0].arguments == '{"k":1}'
    assert r.tool_calls[0].call_id == 'c7'


def test_same_call_id_in_separate_responses_is_not_cross_paired():
    a = AgentResponse.from_output_items([CALL])
    b = AgentResponse.from_output_items([RESULT])
    assert a.tool_calls[0].result is None
    assert b.tool_calls[0].arguments == '{}'
