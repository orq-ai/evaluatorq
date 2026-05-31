"""Tests for OpenResponses output parsing in the red teaming pipeline.

Covers RES-540 touchpoint 3: parse OpenResponses output format to extract
the agent's response for evaluation by safety classifiers and red teaming
judges.
"""

from __future__ import annotations

from evaluatorq.contracts import (
    AgentResponse,
    ReasoningOutputItem,
    TextOutputItem,
    ToolCallOutputItem,
)
from evaluatorq.redteam.openresponses_adapter import agent_response_from_openresponses
from evaluatorq.redteam.parsing import (
    extract_assistant_text,
    extract_reasoning,
    extract_tool_calls,
)


class TestExtractAssistantText:
    def test_agent_response_concatenates_text_items(self):
        resp = AgentResponse(output=[
            TextOutputItem(text="Hello, ", annotations=[]),
            TextOutputItem(text="world.", annotations=[]),
        ])
        assert extract_assistant_text(resp) == "Hello, world."

    def test_agent_response_skips_function_calls_and_reasoning(self):
        resp = AgentResponse(output=[
            ReasoningOutputItem(text="think think"),
            TextOutputItem(text="visible answer", annotations=[]),
            ToolCallOutputItem(name="x", call_id="c1", arguments="{}"),
        ])
        assert extract_assistant_text(resp) == "visible answer"

    def test_openresponses_resource_dict_extracts_message_text(self):
        response_resource = {
            "id": "resp_1",
            "object": "response",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "first "},
                    {"type": "output_text", "text": "second"},
                ],
            }],
        }
        assert extract_assistant_text(response_resource) == "first second"

    def test_returns_empty_string_for_no_text(self):
        assert extract_assistant_text(AgentResponse(output=[])) == ""
        assert extract_assistant_text({}) == ""
        assert extract_assistant_text({"output": []}) == ""

    def test_skips_non_output_text_content_blocks(self):
        response_resource = {
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "refusal", "refusal": "I can't help with that."},
                    {"type": "output_text", "text": "alternative"},
                ],
            }],
        }
        assert extract_assistant_text(response_resource) == "alternative"


class TestExtractToolCalls:
    def test_extracts_tool_calls_from_agent_response(self):
        resp = AgentResponse(output=[
            TextOutputItem(text="checking", annotations=[]),
            ToolCallOutputItem(name="search", call_id="call_1", arguments='{"q":"x"}'),
            ToolCallOutputItem(name="fetch", call_id="call_2", arguments='{}'),
        ])
        calls = extract_tool_calls(resp)
        assert [c["name"] for c in calls] == ["search", "fetch"]
        assert calls[0]["arguments"] == '{"q":"x"}'
        assert calls[0]["call_id"] == "call_1"

    def test_extracts_tool_calls_from_openresponses_dict(self):
        response_resource = {
            "output": [{
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_xyz",
                "name": "lookup",
                "arguments": '{"id": 1}',
            }],
        }
        calls = extract_tool_calls(response_resource)
        assert calls == [{"name": "lookup", "arguments": '{"id": 1}', "call_id": "call_xyz"}]

    def test_falls_back_to_id_when_call_id_missing(self):
        response_resource = {
            "output": [{
                "type": "function_call",
                "id": "fc_1",
                "name": "lookup",
                "arguments": "{}",
            }],
        }
        calls = extract_tool_calls(response_resource)
        assert calls[0]["call_id"] == "fc_1"


class TestExtractReasoning:
    def test_pulls_reasoning_items(self):
        resp = AgentResponse(output=[
            ReasoningOutputItem(text="step 1"),
            TextOutputItem(text="answer", annotations=[]),
            ReasoningOutputItem(text="step 2"),
        ])
        assert extract_reasoning(resp) == ["step 1", "step 2"]

    def test_handles_openresponses_dict_reasoning_items(self):
        response_resource = {
            "output": [
                {"type": "reasoning", "text": "thinking..."},
                {"type": "message", "role": "assistant", "content": [
                    {"type": "output_text", "text": "answer"},
                ]},
            ],
        }
        assert extract_reasoning(response_resource) == ["thinking..."]


class TestAgentResponseFromOpenResponses:
    def test_converts_message_and_function_call_items(self):
        resource = {
            "id": "resp_1",
            "model": "agent-id",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "answer"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "search",
                    "arguments": '{"q": "x"}',
                },
            ],
        }
        agent_response = agent_response_from_openresponses(resource)
        assert agent_response.text == "answer"
        assert len(agent_response.tool_calls) == 1
        assert agent_response.tool_calls[0].name == "search"
        assert agent_response.tool_calls[0].call_id == "call_1"
        assert agent_response.response_id == "resp_1"
        assert agent_response.model == "agent-id"
        assert agent_response.finish_reason == "completed"

    def test_drops_empty_message_items(self):
        resource = {
            "output": [
                {"type": "message", "role": "assistant", "content": []},
            ],
        }
        result = agent_response_from_openresponses(resource)
        assert result.output == []
