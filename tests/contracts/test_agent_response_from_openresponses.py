"""Unit tests for AgentResponse.from_openresponses (RES-897 parse consolidation)."""

from __future__ import annotations

from evaluatorq.contracts import (
    AgentResponse,
    TextOutputItem,
    ToolCallOutputItem,
)


def _resp(**kw):
    base = {"output": [], "usage": None, "model": None, "status": None, "id": None}
    base.update(kw)
    return base


def test_message_text_becomes_text_output_item():
    r = AgentResponse.from_openresponses(_resp(
        output=[{"type": "message", "content": [{"type": "output_text", "text": "hi"}]}],
    ))
    assert [type(i) for i in r.output] == [TextOutputItem]
    assert r.text == "hi"


def test_function_call_becomes_tool_call_output_item():
    r = AgentResponse.from_openresponses(_resp(
        output=[{"type": "function_call", "name": "f", "arguments": "{}", "call_id": "c1"}],
    ))
    assert [type(i) for i in r.output] == [ToolCallOutputItem]
    assert r.tool_calls[0].name == "f"
    assert r.tool_calls[0].call_id == "c1"


def test_reasoning_item_skipped():
    r = AgentResponse.from_openresponses(_resp(output=[{"type": "reasoning"}]))
    assert r.output == []


def test_none_usage_stays_none():
    r = AgentResponse.from_openresponses(_resp(
        output=[{"type": "message", "content": [{"type": "output_text", "text": "x"}]}],
        usage=None,
    ))
    assert r.usage is None


def test_usage_parsed_without_calls_accounting():
    r = AgentResponse.from_openresponses(_resp(
        output=[{"type": "message", "content": [{"type": "output_text", "text": "x"}]}],
        usage={"input_tokens": 3, "output_tokens": 5},
    ))
    assert r.usage is not None
    assert r.usage.prompt_tokens == 3
    assert r.usage.completion_tokens == 5
    assert r.usage.total_tokens == 8
    assert r.usage.calls == 0  # pure parse; call sites bump calls


def test_model_finish_reason_and_response_id_populated():
    # response_id is the RESTORED field — previously dropped on every parse path.
    r = AgentResponse.from_openresponses(_resp(
        output=[{"type": "message", "content": [{"type": "output_text", "text": "x"}]}],
        model="azure/gpt-4o",
        status="completed",
        id="resp_abc123",
    ))
    assert r.model == "azure/gpt-4o"
    assert r.finish_reason == "completed"
    assert r.response_id == "resp_abc123"


def test_function_call_with_string_arguments_preserved():
    r = AgentResponse.from_openresponses(_resp(
        output=[{"type": "function_call", "name": "lookup", "arguments": '{"q": "raw"}', "call_id": "c1"}],
    ))
    assert r.tool_calls[0].arguments == '{"q": "raw"}'


def test_function_call_id_fallback_when_call_id_missing():
    """call_id falls back to item id when call_id is None/absent."""
    from unittest.mock import MagicMock
    item = MagicMock()
    item.type = "function_call"
    item.name = "lookup"
    item.arguments = "{}"
    item.call_id = None
    item.id = "fc_fallback_id"
    item.result = None
    r = AgentResponse.from_openresponses({"output": [item], "usage": None})
    assert r.tool_calls[0].call_id == "fc_fallback_id"


def test_unknown_item_type_logged_and_skipped():
    """Unknown item types are skipped; known items after them still appear."""
    r = AgentResponse.from_openresponses(_resp(
        output=[
            {"type": "mystery"},
            {"type": "message", "content": [{"type": "output_text", "text": "ok"}]},
        ],
    ))
    assert len(r.output) == 1
    assert isinstance(r.output[0], TextOutputItem)


def test_empty_text_part_is_not_emitted():
    r = AgentResponse.from_openresponses(_resp(
        output=[{"type": "message", "content": [{"type": "output_text", "text": ""}]}],
    ))
    assert r.output == []


def test_none_text_part_is_not_emitted():
    r = AgentResponse.from_openresponses(_resp(
        output=[{"type": "message", "content": [{"type": "output_text", "text": None}]}],
    ))
    assert r.output == []


def test_interleaved_text_and_tool_call_order_preserved():
    r = AgentResponse.from_openresponses(_resp(
        output=[
            {"type": "message", "content": [{"type": "output_text", "text": "before"}]},
            {"type": "function_call", "name": "t", "arguments": "{}", "call_id": "c1"},
            {"type": "message", "content": [{"type": "output_text", "text": "after"}]},
        ],
    ))
    assert len(r.output) == 3
    assert isinstance(r.output[0], TextOutputItem)
    assert isinstance(r.output[1], ToolCallOutputItem)
    assert isinstance(r.output[2], TextOutputItem)
    assert r.output[0].text == "before"
    assert r.output[2].text == "after"


def test_accepts_dict_and_object_shaped_response_items():
    """Dict-shaped payloads (wire format) are handled correctly."""
    import json
    resp = {
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "hello"}]},
            {"type": "function_call", "name": "lookup", "arguments": {"q": "x"}, "call_id": "call_1"},
        ],
        "usage": {"input_tokens": 4, "output_tokens": 3},
        "model": None,
        "status": None,
        "id": None,
    }
    r = AgentResponse.from_openresponses(resp)
    assert len(r.output) == 2
    assert isinstance(r.output[0], TextOutputItem)
    assert r.output[0].text == "hello"
    assert isinstance(r.output[1], ToolCallOutputItem)
    assert json.loads(r.output[1].arguments) == {"q": "x"}
    assert r.usage is not None
    assert r.usage.total_tokens == 7
