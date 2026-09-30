from __future__ import annotations

from evaluatorq.trace_finder.trajectory import preview, segments


def test_chat_shape_with_tool_call_and_result() -> None:
    messages = [
        {'role': 'system', 'content': 'You are helpful.'},
        {'role': 'user', 'content': 'x' * 40},
        {'role': 'assistant', 'content': None, 'tool_calls': [{'type': 'function', 'function': {'name': 'lookup', 'arguments': '{"id": 1}'}}]},
        {'role': 'tool', 'name': 'lookup', 'content': '{"ok": true}'},
        {'role': 'assistant', 'content': 'Done.'},
    ]
    result = segments(messages)
    assert [(s.kind, s.index) for s in result] == [('system', 1), ('user', 2), ('call', 3), ('result', 4), ('assistant', 5)]
    assert result[1].tokens == 10
    assert result[2].label == 'lookup'
    assert result[3].label == 'lookup'


def test_otel_parts_shape() -> None:
    messages = [
        {'role': 'user', 'parts': [{'type': 'text', 'content': 'hi'}]},
        {'role': 'assistant', 'parts': [
            {'type': 'reasoning', 'content': 'thinking'},
            {'type': 'tool_call', 'name': 'search', 'arguments': {'q': 'a'}},
        ]},
        {'role': 'tool', 'parts': [{'type': 'tool_call_response', 'name': 'search', 'response': 'found'}]},
    ]
    result = segments(messages)
    assert [s.kind for s in result] == ['user', 'reasoning', 'call', 'result']
    assert result[2].label == 'search'


def test_responses_summary_and_refusal_parts_render_as_text() -> None:
    result = segments([
        {'role': 'assistant', 'content': [
            {'type': 'summary_text', 'text': 'Reasoning summary'},
            {'type': 'refusal', 'refusal': 'I cannot help with that.'},
        ]},
    ])
    assert [segment.kind for segment in result] == ['assistant', 'assistant']
    assert 'Reasoning summary' in result[0].preview
    assert 'I cannot help with that.' in result[1].preview


def test_unknown_part_becomes_other() -> None:
    result = segments([{'role': 'assistant', 'parts': [{'type': 'hologram', 'content': 'zzz'}]}])
    assert [s.kind for s in result] == ['other']


def test_empty_content_yields_no_segment() -> None:
    assert segments([{'role': 'assistant', 'content': ''}]) == []


def test_preview_pretty_prints_json_and_truncates() -> None:
    assert preview('{"a":1}') == '{\n "a": 1\n}'
    long = preview('y' * 500)
    assert len(long) == 241
    assert long.endswith('…')


def test_orq_native_kind_parts_map_to_their_kinds() -> None:
    messages = [
        {'role': 'user', 'parts': [{'_id': 'text_1', 'kind': 'text', 'text': 'Find the refund'}]},
        {
            'role': 'agent',
            'parts': [
                {'kind': 'reasoning', 'reasoning': 'Need the order first'},
                {'kind': 'tool_call', 'tool_name': 'lookup', 'tool_call_id': 'c1', 'arguments': {'id': 7}},
            ],
        },
        {'role': 'tool', 'parts': [{'kind': 'tool_result', 'tool_call_id': 'c1', 'result': {'status': 'paid'}}]},
        {'role': 'agent', 'parts': [{'_id': 'text_2', 'kind': 'text', 'text': 'Refund issued'}]},
    ]

    out = segments(messages)

    assert [s.kind for s in out] == ['user', 'reasoning', 'call', 'result', 'assistant']
    assert out[2].label == 'lookup'
    assert 'paid' in out[3].preview
