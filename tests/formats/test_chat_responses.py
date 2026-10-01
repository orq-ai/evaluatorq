# pyright: reportOptionalSubscript=false
"""chat <-> Responses conversion."""

# ruff: noqa: S101

from __future__ import annotations

import json
from typing import Any

import pytest

from evaluatorq.contracts import FunctionCall, InputFileContent, InputImageContent, InputTextContent, Message, StrategyToolCall
from evaluatorq.formats.chat import ChatConversation
from evaluatorq.formats.responses import ResponsesConversation


def _tool_chat() -> ChatConversation:
    return ChatConversation(
        messages=[
            Message(role='system', content='be brief'),
            Message(role='user', content='weather?'),
            Message(
                role='assistant',
                content='checking',
                tool_calls=[StrategyToolCall(id='c1', function=FunctionCall(name='w', arguments='{"city":"Berlin"}'), item_id='fc_1')],
            ),
            Message(role='tool', tool_call_id='c1', name='w', content='12C'),
            Message(role='assistant', content='12C in Berlin'),
        ]
    )


def test_chat_to_responses_uses_output_text_for_assistant() -> None:
    items = _tool_chat().to_responses().items
    assistant = [i for i in items if i.get('role') == 'assistant']
    assert assistant[0]['content'][0]['type'] == 'output_text'
    assert [i['type'] for i in items if i.get('type') in ('function_call', 'function_call_output')] == [
        'function_call', 'function_call_output',
    ]


def test_chat_roundtrip_is_exact_for_tool_transcript() -> None:
    chat = _tool_chat()
    back = chat.to_responses().to_chat()
    assert [m.model_dump(exclude_none=True) for m in back.messages] == [m.model_dump(exclude_none=True) for m in chat.messages]


def test_empty_chat_to_responses_is_empty() -> None:
    assert ChatConversation(messages=[]).to_responses().items == []


def test_developer_role_maps_to_system_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    conv = ResponsesConversation(items=[{'type': 'message', 'role': 'developer', 'content': 'rules'}])
    assert conv.to_chat().messages[0].role == 'system'
    assert 'developer' in caplog.text


def test_reasoning_dropped_with_one_warning(caplog: pytest.LogCaptureFixture) -> None:
    items: list[dict[str, Any]] = [
        {'type': 'reasoning', 'id': 'rs_1', 'summary': []},
        {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'hi'}]},
    ]
    chat = ResponsesConversation(items=items).to_chat()
    assert [m.role for m in chat.messages] == ['assistant']
    assert caplog.text.count('dropped 1 reasoning items') == 1


def test_orphan_function_call_output_keeps_tool_message_without_name() -> None:
    items: list[dict[str, Any]] = [{'type': 'function_call_output', 'call_id': 'zz', 'output': 'r'}]
    msg = ResponsesConversation(items=items).to_chat().messages[0]
    assert (msg.role, msg.tool_call_id, msg.name) == ('tool', 'zz', None)


def test_function_call_output_keeps_typed_content_parts() -> None:
    items: list[dict[str, Any]] = [
        {'type': 'function_call_output', 'call_id': 'c', 'output': [
            {'type': 'input_text', 'text': 'found'},
            {'type': 'input_image', 'image_url': 'https://x/image.png'},
            {'type': 'input_file', 'file_id': 'file_1'},
        ]},
    ]
    content = ResponsesConversation(items=items).to_chat().messages[0].content
    assert isinstance(content, list)
    assert [part.type for part in content] == ['input_text', 'input_image', 'input_file']


@pytest.mark.parametrize('output', [[], [{'temperature': 12}, {'unit': 'C'}]])
def test_function_call_output_serializes_arbitrary_lists_as_json(output: list[dict[str, Any]]) -> None:
    items: list[dict[str, Any]] = [
        {'type': 'function_call_output', 'call_id': 'c', 'output': output},
    ]
    content = ResponsesConversation(items=items).to_chat().messages[0].content
    assert content == json.dumps(output)


def test_function_call_keeps_nested_raw_key_json_arguments_verbatim() -> None:
    raw_arguments = '{"_raw":{"evaluatorq_raw_value":"literal"}}'
    items: list[dict[str, Any]] = [
        {'type': 'function_call', 'call_id': 'c', 'name': 'f', 'arguments': raw_arguments},
    ]
    call = ResponsesConversation(items=items).to_chat().messages[0].tool_calls[0]  # ty: ignore[not-subscriptable]
    assert call.function.arguments == raw_arguments


def test_function_call_without_call_id_is_skipped(caplog: pytest.LogCaptureFixture) -> None:
    items: list[dict[str, Any]] = [{'type': 'function_call', 'name': 'f', 'arguments': '{}'}]
    assert ResponsesConversation(items=items).to_chat().messages == []
    assert 'call_id' in caplog.text


def test_consecutive_calls_share_one_assistant_message() -> None:
    items: list[dict[str, Any]] = [
        {'type': 'function_call', 'call_id': 'a', 'name': 'f', 'arguments': '{}'},
        {'type': 'function_call', 'call_id': 'b', 'name': 'g', 'arguments': '{}'},
    ]
    msgs = ResponsesConversation(items=items).to_chat().messages
    assert len(msgs) == 1 and msgs[0].tool_calls is not None and len(msgs[0].tool_calls) == 2


def test_fc_item_id_preserved_and_other_ids_dropped() -> None:
    items: list[dict[str, Any]] = [
        {'type': 'function_call', 'id': 'fc_9', 'call_id': 'a', 'name': 'f', 'arguments': '{}'},
        {'type': 'function_call', 'id': 'toolu_1', 'call_id': 'b', 'name': 'g', 'arguments': '{}'},
    ]
    calls = ResponsesConversation(items=items).to_chat().messages[0].tool_calls
    assert calls is not None and [c.item_id for c in calls] == ['fc_9', None]


def test_multimodal_user_content_keeps_image_part() -> None:
    items: list[dict[str, Any]] = [
        {'type': 'message', 'role': 'user', 'content': [
            {'type': 'input_text', 'text': 'look'}, {'type': 'input_image', 'image_url': 'https://x/y.png'}]},
    ]
    content = ResponsesConversation(items=items).to_chat().messages[0].content
    assert isinstance(content, list) and [p.type for p in content] == ['input_text', 'input_image']


def test_invalid_media_parts_degrade_to_markers(caplog: pytest.LogCaptureFixture) -> None:
    items: list[dict[str, Any]] = [{'type': 'message', 'role': 'user', 'content': [
        {'type': 'input_image', 'image_url': 'https://x/y.png', 'detail': {'bad': 1}},
        {'type': 'input_file', 'file_id': 42},
        {'type': 'input_text', 'text': 'still here'},
    ]}]
    content = ResponsesConversation(items=items).to_chat().messages[0].content
    assert isinstance(content, list)
    assert isinstance(content[0], InputImageContent)
    assert content[0].image_url == 'https://x/y.png'
    assert isinstance(content[1], InputTextContent)
    assert content[1].text == '[input_file]'
    assert isinstance(content[2], InputTextContent)
    assert content[2].text == 'still here'
    assert 'Invalid input_image detail' in caplog.text
    assert 'Invalid input_file' in caplog.text


def test_malformed_optional_media_fields_do_not_drop_valid_sources(caplog: pytest.LogCaptureFixture) -> None:
    conv = ResponsesConversation(items=[{'type': 'message', 'role': 'user', 'content': [
        {'type': 'input_image', 'image_url': 'https://x/y.png', 'detail': 'invalid'},
        {'type': 'input_file', 'file_data': 'encoded-file', 'file_id': {'bad': 1}, 'filename': 42},
    ]}])
    content = conv.to_chat().messages[0].content
    assert isinstance(content, list)
    assert isinstance(content[0], InputImageContent)
    assert isinstance(content[1], InputFileContent)
    assert content[0].image_url == 'https://x/y.png'
    assert content[0].detail == 'auto'
    assert content[1].file_data == 'encoded-file'
    assert content[1].file_id is None
    assert content[1].filename is None
    assert 'Invalid input_image detail' in caplog.text
    assert 'Invalid input_file file_id' in caplog.text and 'Invalid input_file filename' in caplog.text


def test_unknown_item_type_warns_and_is_skipped(caplog: pytest.LogCaptureFixture) -> None:
    conv = ResponsesConversation(items=[{'type': 'web_search_call', 'id': 'ws'}])
    assert conv.to_chat().messages == []
    assert 'web_search_call' in caplog.text


def test_dict_arguments_are_kept_as_json() -> None:
    items: list[dict[str, Any]] = [{'type': 'function_call', 'call_id': 'c', 'name': 'f', 'arguments': {'x': 1}}]
    [message] = ResponsesConversation(items=items).to_chat().messages
    assert message.tool_calls is not None and message.tool_calls[0].function.arguments == '{"x":1}'


def test_unencodable_arguments_warn_instead_of_failing(caplog: pytest.LogCaptureFixture) -> None:
    items: list[dict[str, Any]] = [{'type': 'function_call', 'call_id': 'c', 'name': 'f', 'arguments': {'x': {1, 2}}}]
    [message] = ResponsesConversation(items=items).to_chat().messages
    assert message.tool_calls is not None and message.tool_calls[0].function.arguments == '{"x":"{1, 2}"}'
    assert 'not JSON-encodable' in caplog.text


def test_chat_to_atif_takes_a_session_id() -> None:
    assert ChatConversation(messages=[Message(role='user', content='hi')]).to_atif(session_id='mine').session_id == 'mine'


def test_malformed_text_part_is_warned_not_repr_ed(caplog: pytest.LogCaptureFixture) -> None:
    conv = ResponsesConversation(items=[
        {'type': 'message', 'role': 'user', 'content': [
            {'type': 'input_text', 'text': 'hi'}, {'type': 'input_text', 'text': {'bad': 1}}]},
    ])
    assert conv.to_chat().messages[0].content == 'hi'
    assert 'carries no text' in caplog.text
