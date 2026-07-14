"""Unit tests for AgentResponse.from_openresponses parsing of Orq-router
``orq:<tool_name>`` tool-call items.

Covers src/evaluatorq/contracts.py:765 (the ``item_type.startswith('orq:')``
branch added alongside the standard ``function_call`` shape) plus the
``result`` -> ``output`` fallback at src/evaluatorq/contracts.py:768-770 and
the ``name`` derivation from the ``orq:`` prefix at
src/evaluatorq/contracts.py:771.
"""

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
