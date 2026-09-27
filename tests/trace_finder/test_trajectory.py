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
