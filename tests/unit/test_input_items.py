"""Direct tests for the shared Message -> Responses-API input converter.

Two independent targets (``OrqResponsesTarget`` and ``OpenAIAgentTarget``) depend
on this module's exact output shape, and getting it wrong produced both a hard
400 and silent loss of tool calls from replayed history. Tested here directly
rather than only through a target, so a break points at the converter.
"""

from __future__ import annotations

import json

import pytest

from evaluatorq.contracts import (
    FunctionCall,
    InputImageContent,
    InputTextContent,
    Message,
    StrategyToolCall,
)
from evaluatorq.openresponses.input_items import (
    message_to_responses_input_items,
    messages_to_responses_input,
)


def _tool_call(call_id: str = "call_1", name: str = "search", item_id: str | None = None) -> StrategyToolCall:
    return StrategyToolCall(
        id=call_id,
        type="function",
        function=FunctionCall(name=name, arguments='{"q": "x"}'),
        item_id=item_id,
    )


class TestPlainMessages:
    @pytest.mark.parametrize("role", ["user", "assistant", "system", "developer"])
    def test_string_content_roundtrips_per_role(self, role: str) -> None:
        items = message_to_responses_input_items(Message(role=role, content="hello"))
        assert items == [{"role": role, "content": "hello"}]

    @pytest.mark.parametrize("role", ["user", "assistant", "system", "developer"])
    def test_multipart_content_roundtrips_per_role(self, role: str) -> None:
        m = Message(
            role=role,
            content=[
                InputTextContent(type="input_text", text="look"),
                InputImageContent(type="input_image", image_url="https://x/y.png"),
            ],
        )
        items = message_to_responses_input_items(m)
        assert items[0]["role"] == role
        assert [p["type"] for p in items[0]["content"]] == ["input_text", "input_image"]

    def test_none_content_becomes_empty_string(self) -> None:
        # An assistant turn with no text and no tool calls still has to be a
        # valid item; ``content: None`` is not accepted by the API.
        assert message_to_responses_input_items(Message(role="assistant", content=None)) == [
            {"role": "assistant", "content": ""}
        ]


class TestToolTurns:
    def test_assistant_tool_call_becomes_function_call(self) -> None:
        m = Message(role="assistant", content=None, tool_calls=[_tool_call()])
        assert message_to_responses_input_items(m) == [
            {"type": "function_call", "call_id": "call_1", "name": "search", "arguments": '{"q": "x"}'}
        ]

    def test_narration_precedes_the_function_call(self) -> None:
        m = Message(role="assistant", content="let me look", tool_calls=[_tool_call()])
        items = message_to_responses_input_items(m)
        assert items[0] == {"role": "assistant", "content": "let me look"}
        assert items[1]["type"] == "function_call"

    def test_item_id_is_echoed_so_the_call_roundtrips(self) -> None:
        m = Message(role="assistant", content=None, tool_calls=[_tool_call(item_id="fc_abc")])
        assert message_to_responses_input_items(m)[0]["id"] == "fc_abc"

    def test_multiple_tool_calls_each_become_an_item(self) -> None:
        m = Message(
            role="assistant",
            content=None,
            tool_calls=[_tool_call("c1", "a"), _tool_call("c2", "b")],
        )
        items = message_to_responses_input_items(m)
        assert [i["call_id"] for i in items] == ["c1", "c2"]

    def test_tool_result_becomes_function_call_output(self) -> None:
        m = Message(role="tool", tool_call_id="call_1", content="42")
        assert message_to_responses_input_items(m) == [
            {"type": "function_call_output", "call_id": "call_1", "output": "42"}
        ]

    def test_tool_result_keeps_image_parts_instead_of_flattening(self) -> None:
        # ``output`` accepts a parts list, so a tool that returns a screenshot
        # must not be reduced to its text (which would drop the image entirely).
        m = Message(
            role="tool",
            tool_call_id="call_1",
            content=[
                InputTextContent(type="input_text", text="see:"),
                InputImageContent(type="input_image", image_url="https://x/shot.png"),
            ],
        )
        output = message_to_responses_input_items(m)[0]["output"]
        assert [p["type"] for p in output] == ["input_text", "input_image"]

    def test_orphan_tool_result_is_dropped_with_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        # No call_id means the API rejects the item outright; dropping one bad
        # row beats failing the turn, but it must not happen silently.
        m = Message(role="tool", tool_call_id=None, content="stranded")
        assert message_to_responses_input_items(m) == []
        assert "tool_call_id" in caplog.text

    def test_tool_calls_on_a_non_assistant_role_are_not_emitted(self) -> None:
        # The model only accepts tool calls from the assistant; a user message
        # carrying them is malformed, and silently ignoring them is intended.
        m = Message(role="user", content="hi", tool_calls=[_tool_call()])
        assert message_to_responses_input_items(m) == [{"role": "user", "content": "hi"}]


class TestFullTranscript:
    def test_no_chat_completions_shape_leaks(self) -> None:
        # The original bug: role="tool" is rejected by the API, and a
        # message-level tool_calls key is silently ignored, losing the calls.
        messages = [
            Message(role="user", content="what is 2+2"),
            Message(role="assistant", content=None, tool_calls=[_tool_call()]),
            Message(role="tool", tool_call_id="call_1", content="4"),
        ]
        items = messages_to_responses_input(messages)
        assert [i.get("type") or i["role"] for i in items] == ["user", "function_call", "function_call_output"]
        assert not any("tool_calls" in i or i.get("role") == "tool" for i in items)

    def test_output_is_json_serializable(self) -> None:
        # Pydantic models left in the payload raise inside the SDK's encoder,
        # killing the request before it is ever sent.
        messages = [
            Message(
                role="assistant",
                content=[InputTextContent(type="input_text", text="looking")],
                tool_calls=[_tool_call()],
            ),
            Message(role="tool", tool_call_id="call_1", content=[InputTextContent(type="input_text", text="done")]),
        ]
        json.dumps(messages_to_responses_input(messages))
