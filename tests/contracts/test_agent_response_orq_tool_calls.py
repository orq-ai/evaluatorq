"""Unit tests for AgentResponse.from_openresponses parsing of Orq-router
``orq:<tool_name>`` tool-call items.

Covers the ``orq:`` branch of ``AgentResponse.from_output_items`` (added
alongside the standard ``function_call`` shape), the ``result`` -> ``output``
fallback, the ``name`` derivation from the ``orq:`` prefix, and the pairing of
split ``function_call`` / ``orq:<tool>`` records by ``call_id``.
"""

from __future__ import annotations

import json
import logging

from evaluatorq.contracts import (
    AgentResponse,
    TextOutputItem,
    ToolCallOutputItem,
)


def _resp(**kw):
    base = {"output": [], "usage": None, "model": None, "status": None, "id": None}
    base.update(kw)
    return base


def test_orq_prefixed_item_becomes_tool_call_with_name_from_prefix():
    """'orq:query_knowledge_base' is parsed as a tool call named 'query_knowledge_base'."""
    r = AgentResponse.from_openresponses(_resp(
        output=[
            {
                "type": "orq:query_knowledge_base",
                "arguments": "{}",
                "call_id": "c1",
            }
        ],
    ))
    assert [type(i) for i in r.output] == [ToolCallOutputItem]
    tool_call = r.tool_calls[0]
    assert tool_call.name == "query_knowledge_base"
    assert tool_call.call_id == "c1"
    # The normalized item type is always 'function_call' regardless of the
    # Orq-router wire type, so downstream consumers see a uniform shape.
    assert tool_call.type == "function_call"


def test_orq_prefixed_item_explicit_name_wins_over_prefix():
    """When the item also carries an explicit 'name' field, that value wins."""
    r = AgentResponse.from_openresponses(_resp(
        output=[
            {
                "type": "orq:some_tool",
                "name": "explicit_name",
                "arguments": "{}",
                "call_id": "c1",
            }
        ],
    ))
    assert r.tool_calls[0].name == "explicit_name"


def test_orq_result_field_populates_output():
    """The 'result' field, when present, becomes the tool call's output."""
    r = AgentResponse.from_openresponses(_resp(
        output=[
            {
                "type": "orq:query_knowledge_base",
                "arguments": "{}",
                "call_id": "c1",
                "result": "42 documents found",
            }
        ],
    ))
    assert r.tool_calls[0].result == "42 documents found"


def test_orq_result_falls_back_to_output_field():
    """When 'result' is absent/None, 'output' is used instead."""
    r = AgentResponse.from_openresponses(_resp(
        output=[
            {
                "type": "orq:query_knowledge_base",
                "arguments": "{}",
                "call_id": "c1",
                "result": None,
                "output": "fallback output value",
            }
        ],
    ))
    assert r.tool_calls[0].result == "fallback output value"


def test_orq_result_and_output_both_absent_yields_none():
    r = AgentResponse.from_openresponses(_resp(
        output=[
            {
                "type": "orq:query_knowledge_base",
                "arguments": "{}",
                "call_id": "c1",
            }
        ],
    ))
    assert r.tool_calls[0].result is None


def test_orq_item_without_name_or_recognizable_prefix_suffix_uses_split():
    """Name derivation splits on the first colon only (tool names may contain none)."""
    r = AgentResponse.from_openresponses(_resp(
        output=[
            {
                "type": "orq:namespace:tool",
                "arguments": "{}",
                "call_id": "c1",
            }
        ],
    ))
    # split(':', 1) on 'orq:namespace:tool' -> ['orq', 'namespace:tool']
    assert r.tool_calls[0].name == "namespace:tool"


def test_normal_function_call_item_still_parses_regression():
    """Non-'orq:' standard 'function_call' items are unaffected by the orq branch."""
    r = AgentResponse.from_openresponses(_resp(
        output=[
            {"type": "function_call", "name": "lookup", "arguments": "{}", "call_id": "c1", "result": "ok"}
        ],
    ))
    assert [type(i) for i in r.output] == [ToolCallOutputItem]
    assert r.tool_calls[0].name == "lookup"
    assert r.tool_calls[0].result == "ok"


def test_orq_tool_call_interleaved_with_text_output_regression():
    """An orq: tool call alongside a normal text message; both parse and preserve order."""
    r = AgentResponse.from_openresponses(_resp(
        output=[
            {"type": "message", "content": [{"type": "output_text", "text": "before"}]},
            {"type": "orq:query_knowledge_base", "arguments": "{}", "call_id": "c1", "result": "hit"},
        ],
    ))
    assert len(r.output) == 2
    assert isinstance(r.output[0], TextOutputItem)
    assert r.output[0].text == "before"
    assert isinstance(r.output[1], ToolCallOutputItem)
    assert r.output[1].name == "query_knowledge_base"
    assert r.output[1].result == "hit"


# ---------------------------------------------------------------------------
# Split records: a function_call (arguments) plus an orq:<tool> record (result)
# ---------------------------------------------------------------------------

CALL = {'type': 'function_call', 'id': 'fc_1', 'call_id': 'c1', 'name': 'refund', 'arguments': '{"order_id":"a2"}'}
RESULT = {'type': 'orq:refund', 'id': 'orq_9', 'call_id': 'c1', 'name': 'refund', 'result': {'ok': False}}


def _messages(caplog) -> str:
    return '\n'.join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)


def test_split_records_become_one_call_with_args_and_result():
    r = AgentResponse.from_output_items([CALL, RESULT])
    assert len(r.tool_calls) == 1
    tc = r.tool_calls[0]
    assert (tc.call_id, tc.id, tc.name) == ('c1', 'fc_1', 'refund')
    assert tc.arguments == '{"order_id":"a2"}'
    assert tc.result == '{"ok": false}'


def test_result_before_call_pairs_too(caplog):
    r = AgentResponse.from_output_items([RESULT, CALL])
    assert len(r.tool_calls) == 1
    assert r.tool_calls[0].arguments == '{"order_id":"a2"}'
    assert not _messages(caplog)


def test_id_is_the_function_call_id_in_both_orders():
    for items in ([CALL, RESULT], [RESULT, CALL]):
        assert AgentResponse.from_output_items(items).tool_calls[0].id == 'fc_1'


def test_orq_only_record_keeps_its_wire_id():
    assert AgentResponse.from_output_items([RESULT]).tool_calls[0].id == 'orq_9'


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


def test_dict_and_string_arguments_pair():
    result = {**RESULT, 'arguments': {'order_id': 'a2'}}
    r = AgentResponse.from_output_items([{**CALL, 'arguments': '{ "order_id": "a2" }'}, result])
    assert len(r.tool_calls) == 1
    assert r.tool_calls[0].arguments == '{ "order_id": "a2" }'


def test_empty_string_arguments_are_filled_in_by_the_other_record():
    r = AgentResponse.from_output_items([{**RESULT, 'arguments': ''}, CALL])
    assert len(r.tool_calls) == 1
    assert r.tool_calls[0].arguments == '{"order_id":"a2"}'


def test_empty_function_call_arguments_dict_keeps_the_old_behaviour():
    r = AgentResponse.from_output_items([{**CALL, 'arguments': {}}])
    assert r.tool_calls[0].arguments == '{}'


def test_conflicting_results_keep_one_call_and_warn_once(caplog):
    r = AgentResponse.from_output_items([CALL, RESULT, {**RESULT, 'result': 'other'}, {**RESULT, 'result': 'third'}])
    assert [t.call_id for t in r.tool_calls] == ['c1']
    assert r.tool_calls[0].result == '{"ok": false}'
    assert r.tool_calls[0].arguments == '{"order_id":"a2"}'
    assert _messages(caplog).count('Conflicting result for call_id') == 1


def test_conflicts_land_at_the_function_call_position():
    r = AgentResponse.from_output_items(
        [RESULT, {'type': 'message', 'content': [{'type': 'output_text', 'text': 'hi'}]}, CALL]
    )
    assert isinstance(r.output[0], TextOutputItem)
    assert isinstance(r.output[1], ToolCallOutputItem)


def test_name_derived_from_orq_type_is_not_compared():
    result = {'type': 'orq:renamed', 'call_id': 'c1', 'result': 'ok'}
    r = AgentResponse.from_output_items([CALL, result])
    assert len(r.tool_calls) == 1
    assert r.tool_calls[0].name == 'refund'


def test_failed_orq_record_carries_its_error_as_the_result():
    failed = {'type': 'orq:refund', 'call_id': 'c1', 'status': 'failed', 'error': 'boom'}
    r = AgentResponse.from_output_items([CALL, failed])
    assert len(r.tool_calls) == 1
    assert r.tool_calls[0].arguments == '{"order_id":"a2"}'
    assert json.loads(r.tool_calls[0].result or '') == {'status': 'failed', 'error': 'boom'}


def test_orq_record_with_only_an_id_gets_a_unique_call_id_and_never_pairs():
    orq = {'type': 'orq:refund', 'id': 'fc_1', 'result': 'x'}
    r = AgentResponse.from_output_items([{**CALL, 'call_id': None}, orq])
    assert len(r.tool_calls) == 2
    ids = [t.call_id for t in r.tool_calls]
    assert all(ids)
    assert ids[0] == 'fc_1'
    assert ids[1] not in ('', 'fc_1')


def test_function_call_wire_id_does_not_pair_with_an_orq_call_id():
    call = {'type': 'function_call', 'id': 'c7', 'name': 't', 'arguments': '{"k":1}'}
    res = {'type': 'orq:t', 'call_id': 'c7', 'result': 'ok'}
    r = AgentResponse.from_output_items([call, res])
    assert [t.result for t in r.tool_calls] == [None, 'ok']


def test_lone_orq_record_with_real_empty_arguments_agrees_across_readers():
    from evaluatorq.openresponses.otel_messages import items_to_input_messages

    record = {'type': 'orq:kb', 'call_id': 'k', 'arguments': {}, 'result': 'x'}
    assert AgentResponse.from_output_items([record]).tool_calls[0].arguments == '{}'
    assert items_to_input_messages([record])[0]['parts'][0]['arguments'] == {}


def test_plain_function_call_with_its_function_call_output_item_gets_the_result():
    items = [
        {'type': 'function_call', 'call_id': 'a', 'name': 'x', 'arguments': '{}'},
        {'type': 'function_call_output', 'call_id': 'a', 'output': 'done'},
    ]
    assert AgentResponse.from_output_items(items).tool_calls[0].result == 'done'




def test_parallel_native_calls_keep_their_own_results(caplog):
    items = [
        {'type': 'function_call', 'call_id': 'a', 'name': 'x', 'arguments': '{"n":1}'},
        {'type': 'function_call', 'call_id': 'b', 'name': 'x', 'arguments': '{"n":2}'},
        {'type': 'function_call_output', 'call_id': 'a', 'output': 'ra'},
        {'type': 'function_call_output', 'call_id': 'b', 'output': 'rb'},
    ]
    r = AgentResponse.from_output_items(items)
    assert [(t.call_id, t.arguments, t.result) for t in r.tool_calls] == [('a', '{"n":1}', 'ra'), ('b', '{"n":2}', 'rb')]
    assert not _messages(caplog)


def test_orq_result_and_native_output_for_one_call_id_make_one_result(caplog):
    items = [CALL, RESULT, {'type': 'function_call_output', 'call_id': 'c1', 'output': '{"ok": false}'}]
    r = AgentResponse.from_output_items(items)
    assert len(r.output) == 1
    assert r.tool_calls[0].result == '{"ok": false}'
    assert not _messages(caplog)


def test_orphan_native_output_is_still_skipped_with_a_warning(caplog):
    r = AgentResponse.from_output_items([{'type': 'function_call_output', 'call_id': 'z', 'output': 'x'}])
    assert r.output == []
    assert 'unknown item type' in _messages(caplog)
