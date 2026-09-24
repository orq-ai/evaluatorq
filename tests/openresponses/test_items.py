"""`parse_item` is the contract every Responses consumer branches on; pin it directly."""

# ruff: noqa: S101

from __future__ import annotations

from typing import Any

import pytest
from openai.types.responses import ResponseOutputMessage, ResponseOutputText

from evaluatorq.openresponses.items import (
    RESPONSES_ITEM_TYPES,
    MessageItem,
    ToolCallItem,
    classify_part,
    is_responses_item,
    parse_item,
)


@pytest.mark.parametrize(
    ('item', 'kind'),
    [
        ({'type': 'message', 'role': 'user', 'content': 'hi'}, 'message'),
        ({'role': 'user', 'content': 'hi'}, 'message'),
        ({'role': None, 'content': 'hi'}, 'message'),
        ({'type': 'function_call', 'call_id': 'c', 'name': 'f', 'arguments': '{}'}, 'tool_call'),
        ({'type': 'custom_tool_call', 'call_id': 'c', 'name': 'f', 'input': 'x'}, 'tool_call'),
        ({'type': 'mcp_call', 'id': 'm', 'name': 'f', 'arguments': '{}'}, 'tool_call'),
        ({'type': 'orq:kb', 'call_id': 'c'}, 'tool_call'),
        ({'type': 'function_call_output', 'call_id': 'c', 'output': 'x'}, 'tool_result'),
        ({'type': 'custom_tool_call_output', 'call_id': 'c', 'output': 'x'}, 'tool_result'),
        ({'type': 'reasoning', 'summary': []}, 'reasoning'),
        ({'type': 'web_search_call', 'id': 'w'}, 'unknown'),
    ],
)
def test_each_type_maps_to_one_kind(item: dict[str, Any], kind: str) -> None:
    assert parse_item(item).kind == kind


def test_every_known_type_is_classified_and_recognised() -> None:
    for item_type in RESPONSES_ITEM_TYPES:
        item = {'type': item_type, 'call_id': 'c', 'name': 'f', 'output': 'x'}
        assert is_responses_item(item)
        assert parse_item(item).kind != 'unknown', item_type


def test_unknown_type_is_logged_once(caplog: pytest.LogCaptureFixture) -> None:
    parse_item({'type': 'web_search_call', 'id': 'w'})

    assert caplog.text.count("type='web_search_call'") == 1


def test_sdk_model_and_string_content_become_part_dicts() -> None:
    sdk = ResponseOutputMessage(
        id='m', type='message', role='assistant', status='completed',
        content=[ResponseOutputText(type='output_text', text='done', annotations=[])],
    )

    from_sdk = parse_item(sdk)
    from_str = parse_item({'type': 'message', 'role': 'assistant', 'content': 'done'})

    assert isinstance(from_sdk, MessageItem)
    assert isinstance(from_str, MessageItem)
    assert [classify_part(p) for p in from_sdk.content] == [('text', 'done')]
    assert [classify_part(p) for p in from_str.content] == [('text', 'done')]


def test_call_id_and_item_id_are_kept_apart() -> None:
    both = parse_item({'type': 'function_call', 'call_id': 'call_1', 'id': 'fc_1', 'name': 'f'})
    id_only = parse_item({'type': 'function_call', 'id': 'fc_1', 'name': 'f'})
    mcp = parse_item({'type': 'mcp_call', 'id': 'mcp_1', 'name': 'f'})

    assert isinstance(both, ToolCallItem)
    assert isinstance(id_only, ToolCallItem)
    assert isinstance(mcp, ToolCallItem)
    assert (both.call_id, both.item_id) == ('call_1', 'fc_1')
    assert (id_only.call_id, id_only.item_id, id_only.ref) == (None, 'fc_1', 'fc_1')
    # An MCP call has no call_id field; its id is how its (inline) result is referenced.
    assert mcp.call_id == 'mcp_1'


@pytest.mark.parametrize(
    ('fields', 'result'),
    [
        ({'output': 'x', 'error': None}, 'x'),
        ({'output': '', 'error': None}, ''),
        ({'output': 'x', 'error': 'down'}, {'error': 'down'}),
        ({'output': None, 'error': None}, None),
    ],
)
def test_mcp_result_precedence(fields: dict[str, Any], result: Any) -> None:
    item = parse_item({'type': 'mcp_call', 'id': 'm', 'name': 'f', **fields})

    assert item.kind == 'tool_call'
    assert item.result == result


def test_names_come_from_name_then_tool_name_then_orq_suffix() -> None:
    both = parse_item({'type': 'mcp_call', 'id': 'm', 'name': 'sdk', 'tool_name': 'gateway'})
    gateway = parse_item({'type': 'mcp_call', 'id': 'm', 'tool_name': 'gateway'})
    orq = parse_item({'type': 'orq:query_kb', 'call_id': 'c'})

    assert isinstance(both, ToolCallItem)
    assert isinstance(gateway, ToolCallItem)
    assert isinstance(orq, ToolCallItem)
    assert (both.name, gateway.name, orq.name) == ('sdk', 'gateway', 'query_kb')


def test_result_given_as_content_parts_is_text() -> None:
    item = parse_item({
        'type': 'function_call_output', 'call_id': 'c', 'output': [{'type': 'input_text', 'text': 'hi'}]
    })

    assert item.kind == 'tool_result'
    assert item.output == 'hi'


def test_result_without_output_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    item = parse_item({'type': 'function_call_output', 'call_id': 'c'})

    assert item.kind == 'tool_result'
    assert item.output is None
    assert 'has no output' in caplog.text
