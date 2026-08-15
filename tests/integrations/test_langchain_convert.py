"""Tests for LangChain -> OpenResponses conversion (convert_to_open_responses)."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from evaluatorq.integrations.langchain_integration.convert import convert_to_open_responses


def test_langchain_ai_message_with_text_and_tool_calls_keeps_text():
    """An AIMessage with both prose content and tool_calls must keep the prose.

    Previously the AI-message branch only emitted the output-text message in
    the `else` of `if tool_calls:`, so any assistant text alongside tool calls
    was silently dropped from the converted output.
    """
    msg = AIMessage(
        content='I will look that up.',
        tool_calls=[{'name': 'search', 'args': {}, 'id': 'c1'}],
    )
    result = convert_to_open_responses([msg])
    output = result.get('output', [])

    types = [item['type'] for item in output]
    assert 'function_call' in types

    texts = [item.get('content') for item in output]
    assert any('I will look that up.' in str(t) for t in texts)
