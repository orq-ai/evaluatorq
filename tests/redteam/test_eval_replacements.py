from __future__ import annotations

import json

from evaluatorq.contracts import TextOutputItem, ToolCallOutputItem
from evaluatorq.redteam.judge import build_eval_replacements


def _text(s: str) -> TextOutputItem:
    return TextOutputItem(text=s, annotations=[])


def _tool(name: str, args: str, result: str | None, id_: str) -> ToolCallOutputItem:
    return ToolCallOutputItem(id=id_, call_id=id_, name=name, arguments=args, result=result)


def test_input_messages_map_to_all_messages() -> None:
    rep = build_eval_replacements(
        input_messages=[{'role': 'user', 'content': 'hi'}],
        output_messages=[_text('hello')],
    )
    assert rep['input']['all_messages'] == [{'role': 'user', 'content': 'hi'}]


def test_output_response_joins_all_text() -> None:
    rep = build_eval_replacements(
        input_messages=[],
        output_messages=[_text('part one '), _text('part two')],
    )
    assert rep['output']['response'] == 'part one part two'


def test_tool_calls_arguments_stay_parsed_object() -> None:
    rep = build_eval_replacements(
        input_messages=[],
        output_messages=[_tool('read_file', '{"path": "/etc"}', 'ok', 'call_1')],
    )
    tc = rep['output']['tools_called'][0]
    assert tc['arguments'] == {'path': '/etc'}
    assert tc['name'] == 'read_file'
    assert tc['result'] == 'ok'
    assert tc['id'] == 'call_1'


def test_output_messages_drops_reasoning_and_excludes_input() -> None:
    rep = build_eval_replacements(
        input_messages=[{'role': 'user', 'content': 'hi'}],
        output_messages=[_text('answer'), _tool('t', '{}', None, 'c1')],
    )
    rendered = json.dumps(rep['output']['messages'])
    assert 'answer' in rendered
    assert 'hi' not in rendered


def test_tools_called_flat_override_is_json_string() -> None:
    rep = build_eval_replacements(
        input_messages=[],
        output_messages=[_tool('t', '{}', None, 'c1')],
    )
    assert isinstance(rep['output.tools_called'], str)
