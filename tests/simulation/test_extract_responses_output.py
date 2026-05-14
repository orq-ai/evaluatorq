"""Unit tests for `extract_responses_output` helper.

Covers branches that were previously only exercised indirectly:
- function_call output items
- reasoning item skip
- unknown item.type warning
- empty/None text parts
- missing usage object
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

from evaluatorq.contracts import TextOutputItem, ToolCallOutputItem
from evaluatorq.simulation._client import extract_responses_output


def _msg_item(text: str | None) -> MagicMock:
    part = MagicMock()
    part.type = "output_text"
    part.text = text
    item = MagicMock()
    item.type = "message"
    item.content = [part]
    return item


def _fc_item(name: str, args: "dict[str, Any] | str", call_id: str = "call_1", id_: str = "fc_1") -> MagicMock:
    item = MagicMock()
    item.type = "function_call"
    item.name = name
    item.arguments = args
    item.call_id = call_id
    item.id = id_
    return item


def _reasoning_item() -> MagicMock:
    item = MagicMock()
    item.type = "reasoning"
    return item


def _unknown_item(t: str = "mystery") -> MagicMock:
    item = MagicMock()
    item.type = t
    return item


_SENTINEL = object()


def _make_response(items, *, usage: MagicMock | None | object = _SENTINEL) -> MagicMock:
    if usage is _SENTINEL:
        usage = MagicMock(input_tokens=3, output_tokens=2)
    r = MagicMock()
    r.output = items
    r.usage = usage
    return r


class TestExtractResponsesOutput:
    def test_function_call_item_becomes_tool_call_output_item_with_dict_args(self):
        response = _make_response([_fc_item("lookup", {"q": "x"}, call_id="call_42", id_="fc_42")])
        items, usage = extract_responses_output(response)
        assert len(items) == 1
        assert isinstance(items[0], ToolCallOutputItem)
        assert items[0].name == "lookup"
        assert items[0].call_id == "call_42"
        assert json.loads(items[0].arguments) == {"q": "x"}
        assert usage is not None
        assert usage.prompt_tokens == 3
        assert usage.completion_tokens == 2

    def test_function_call_with_string_arguments_preserved(self):
        response = _make_response([_fc_item("lookup", '{"q": "raw"}')])
        items, _ = extract_responses_output(response)
        assert isinstance(items[0], ToolCallOutputItem)
        assert items[0].arguments == '{"q": "raw"}'

    def test_function_call_id_fallback_when_call_id_missing(self):
        item = _fc_item("lookup", {}, id_="fc_fallback_id")
        item.call_id = None
        response = _make_response([item])
        items, _ = extract_responses_output(response)
        assert isinstance(items[0], ToolCallOutputItem)
        assert items[0].call_id == "fc_fallback_id"

    def test_reasoning_item_is_skipped(self):
        response = _make_response([_reasoning_item(), _msg_item("hi")])
        items, _ = extract_responses_output(response)
        assert len(items) == 1
        assert isinstance(items[0], TextOutputItem)
        assert items[0].text == "hi"

    def test_unknown_item_type_logged_and_skipped(self):
        response = _make_response([_unknown_item("mystery"), _msg_item("ok")])
        items, _ = extract_responses_output(response)
        assert len(items) == 1
        assert isinstance(items[0], TextOutputItem)

    def test_empty_text_part_is_not_emitted_as_text_item(self):
        response = _make_response([_msg_item("")])
        items, _ = extract_responses_output(response)
        assert items == []

    def test_none_text_part_is_not_emitted(self):
        response = _make_response([_msg_item(None)])
        items, _ = extract_responses_output(response)
        assert items == []

    def test_missing_usage_returns_none(self):
        """Missing usage must propagate None so cost reports do not log fake-zero usage for billed calls."""
        response = _make_response([_msg_item("hi")], usage=None)
        _, usage = extract_responses_output(response)
        assert usage is None

    def test_interleaved_text_and_tool_call_order_preserved(self):
        items_in = [
            _msg_item("before"),
            _fc_item("t", {}),
            _msg_item("after"),
        ]
        items, _ = extract_responses_output(_make_response(items_in))
        assert len(items) == 3
        assert isinstance(items[0], TextOutputItem)
        assert isinstance(items[1], ToolCallOutputItem)
        assert isinstance(items[2], TextOutputItem)
        assert items[0].text == "before"
        assert items[2].text == "after"
