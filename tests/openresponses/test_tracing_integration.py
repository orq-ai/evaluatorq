"""Tests for OpenResponses trace integration.

Covers RES-540 touchpoint 4: red teaming runs produce proper traces in the
OpenResponses span format so results are visible in the observability layer.

These tests exercise the span-attribute recording helpers without booting
a real OTel SDK — a stub span captures attribute writes for assertions.
"""

from __future__ import annotations

import json
from typing import Any

from evaluatorq.contracts import AgentResponse, TextOutputItem, ToolCallOutputItem
from evaluatorq.redteam.openresponses_adapter import (
    build_openresponses_request,
    record_openresponses_request,
    record_openresponses_response,
)


class StubSpan:
    """Minimal span-like object that captures set_attribute calls."""

    def __init__(self) -> None:
        self.attrs: dict[str, Any] = {}

    def set_attribute(self, key: str, value: Any) -> None:
        self.attrs[key] = value


class TestRecordOpenResponsesRequest:
    def test_sets_gen_ai_input_messages_with_input_array(self):
        payload = build_openresponses_request(model="agent-id", prompt="attack")
        span = StubSpan()
        record_openresponses_request(span, payload)
        assert "gen_ai.input.messages" in span.attrs
        recovered = json.loads(span.attrs["gen_ai.input.messages"])
        assert recovered == [{"role": "user", "content": "attack"}]

    def test_stores_full_request_under_orq_namespace(self):
        payload = build_openresponses_request(model="agent-id", prompt="attack")
        span = StubSpan()
        record_openresponses_request(span, payload)
        assert "orq.openresponses.request" in span.attrs
        recovered = json.loads(span.attrs["orq.openresponses.request"])
        assert recovered["model"] == "agent-id"
        assert recovered["input"][0]["content"] == "attack"

    def test_sets_request_model_attribute(self):
        payload = build_openresponses_request(model="agent-id", prompt="attack")
        span = StubSpan()
        record_openresponses_request(span, payload)
        assert span.attrs["gen_ai.request.model"] == "agent-id"

    def test_none_span_is_noop(self):
        record_openresponses_request(None, {"model": "x", "input": []})


class TestRecordOpenResponsesResponse:
    def test_records_assistant_output_for_resource_dict(self):
        response = {
            "id": "resp_1",
            "model": "agent-id",
            "status": "completed",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "agent reply"}],
            }],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            },
        }
        span = StubSpan()
        record_openresponses_response(span, response)

        output_messages = json.loads(span.attrs["gen_ai.output.messages"])
        assert output_messages == [{"role": "assistant", "content": "agent reply"}]

        full = json.loads(span.attrs["orq.openresponses.response"])
        assert full["id"] == "resp_1"

        assert span.attrs["gen_ai.response.id"] == "resp_1"
        assert span.attrs["gen_ai.response.model"] == "agent-id"
        assert span.attrs["gen_ai.usage.input_tokens"] == 10
        assert span.attrs["gen_ai.usage.output_tokens"] == 5
        assert span.attrs["gen_ai.usage.total_tokens"] == 15

    def test_records_agent_response_instance(self):
        agent_response = AgentResponse(
            output=[
                TextOutputItem(text="partial ", annotations=[]),
                TextOutputItem(text="answer.", annotations=[]),
                ToolCallOutputItem(name="x", call_id="c1", arguments="{}"),
            ],
            model="agent-id",
            response_id="resp_2",
        )
        span = StubSpan()
        record_openresponses_response(span, agent_response)

        output_messages = json.loads(span.attrs["gen_ai.output.messages"])
        assert output_messages == [{"role": "assistant", "content": "partial answer."}]
        assert span.attrs["gen_ai.response.id"] == "resp_2"
        assert span.attrs["gen_ai.response.model"] == "agent-id"

    def test_handles_empty_response(self):
        span = StubSpan()
        record_openresponses_response(span, {"output": []})
        recovered = json.loads(span.attrs["gen_ai.output.messages"])
        assert recovered == [{"role": "assistant", "content": ""}]

    def test_none_span_is_noop(self):
        record_openresponses_response(None, {"output": []})

    def test_missing_usage_does_not_set_token_attrs(self):
        span = StubSpan()
        record_openresponses_response(span, {
            "id": "r",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ok"}],
            }],
        })
        assert "gen_ai.usage.input_tokens" not in span.attrs

    def test_dict_response_falls_back_to_sum_when_total_missing(self):
        span = StubSpan()
        record_openresponses_response(span, {
            "output": [],
            "usage": {"input_tokens": 7, "output_tokens": 3},
        })
        assert span.attrs["gen_ai.usage.total_tokens"] == 10
        assert span.attrs["total_tokens"] == 10

    def test_agent_response_records_token_attrs(self):
        from evaluatorq.redteam.contracts import TokenUsage
        agent_response = AgentResponse(
            output=[TextOutputItem(text="ok", annotations=[])],
            usage=TokenUsage(prompt_tokens=11, completion_tokens=4, total_tokens=15),
            model="agent-id",
            response_id="resp_x",
        )
        span = StubSpan()
        record_openresponses_response(span, agent_response)
        # Both OTel-canonical and bare keys are written, matching the
        # chat-completions span convention used elsewhere in redteam/tracing.
        assert span.attrs["gen_ai.usage.input_tokens"] == 11
        assert span.attrs["gen_ai.usage.output_tokens"] == 4
        assert span.attrs["gen_ai.usage.total_tokens"] == 15
        assert span.attrs["input_tokens"] == 11
        assert span.attrs["output_tokens"] == 4
        assert span.attrs["total_tokens"] == 15

    def test_agent_response_falls_back_to_sum_when_total_zero(self):
        from evaluatorq.redteam.contracts import TokenUsage
        agent_response = AgentResponse(
            output=[TextOutputItem(text="ok", annotations=[])],
            usage=TokenUsage(prompt_tokens=6, completion_tokens=2, total_tokens=0),
        )
        span = StubSpan()
        record_openresponses_response(span, agent_response)
        assert span.attrs["gen_ai.usage.total_tokens"] == 8
