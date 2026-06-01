"""Tests for OpenResponses input format adaptation in the red teaming pipeline.

Covers RES-540 touchpoint 1: attack payloads / datasets are structured in
OpenResponses format ``{"model": "agent-id", "input": [{"role": "user",
"content": "..."}]}``.
"""

from __future__ import annotations

import pytest

from evaluatorq.openresponses import (
    assistant_input_item,
    build_openresponses_request,
    system_input_item,
    user_input_item,
)


class TestBuildOpenResponsesRequest:
    def test_minimal_request_has_model_and_input(self):
        payload = build_openresponses_request(model="agent-id", prompt="adversarial prompt here")
        assert payload == {
            "model": "agent-id",
            "input": [{"role": "user", "content": "adversarial prompt here"}],
        }

    def test_input_array_is_a_list_of_role_content_dicts(self):
        payload = build_openresponses_request(model="agent-id", prompt="x")
        assert isinstance(payload["input"], list)
        assert payload["input"][0] == {"role": "user", "content": "x"}

    def test_instructions_field_is_passed_through(self):
        payload = build_openresponses_request(
            model="m",
            prompt="p",
            instructions="be helpful",
        )
        assert payload["instructions"] == "be helpful"

    def test_extra_fields_are_merged_at_top_level(self):
        payload = build_openresponses_request(
            model="m",
            prompt="p",
            extra={"temperature": 0.0, "store": False, "tools": []},
        )
        assert payload["temperature"] == 0.0
        assert payload["store"] is False
        assert payload["tools"] == []

    def test_conversation_is_prepended_to_input(self):
        conversation = [
            user_input_item("first attack"),
            assistant_input_item("agent response"),
        ]
        payload = build_openresponses_request(
            model="m", prompt="follow-up", conversation=conversation,
        )
        assert payload["input"] == [
            {"role": "user", "content": "first attack"},
            {"role": "assistant", "content": "agent response"},
            {"role": "user", "content": "follow-up"},
        ]

    def test_conversation_only_is_accepted_without_prompt(self):
        conversation = [user_input_item("only attack")]
        payload = build_openresponses_request(model="m", conversation=conversation)
        assert payload["input"] == [{"role": "user", "content": "only attack"}]

    def test_empty_inputs_raise(self):
        with pytest.raises(ValueError, match="at least one"):
            build_openresponses_request(model="m")

    def test_helpers_produce_simple_role_content_dicts(self):
        assert user_input_item("hi") == {"role": "user", "content": "hi"}
        assert assistant_input_item("hello") == {"role": "assistant", "content": "hello"}
        assert system_input_item("be safe") == {"role": "system", "content": "be safe"}
