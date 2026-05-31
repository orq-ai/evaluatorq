"""Tests for the OpenResponses → RedTeamSample dataset adapter.

Datasets stored in OpenResponses format can be loaded into the existing
RedTeamSample schema (which stores attack conversations as OpenAI chat
messages) without re-authoring.
"""

from __future__ import annotations

import pytest

from evaluatorq.redteam.contracts import (
    RedTeamInput,
    Severity,
    TurnType,
    VulnerabilityDomain,
)
from evaluatorq.redteam.openresponses_adapter import (
    messages_from_openresponses_input,
    redteam_sample_from_openresponses,
)


def _input() -> RedTeamInput:
    return RedTeamInput(
        id="s1",
        vulnerability="prompt_injection",
        severity=Severity.HIGH,
        vulnerability_domain=VulnerabilityDomain.MODEL,
        turn_type=TurnType.SINGLE,
        source="unit-test",
    )


class TestMessagesFromOpenResponsesInput:
    def test_simple_user_assistant_pair_round_trips(self):
        result = messages_from_openresponses_input([
            {"role": "user", "content": "initial attack"},
            {"role": "assistant", "content": "agent response"},
        ])
        assert result == [
            {"role": "user", "content": "initial attack"},
            {"role": "assistant", "content": "agent response"},
        ]

    def test_system_role_passes_through(self):
        result = messages_from_openresponses_input([
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "x"},
        ])
        assert result == [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "x"},
        ]

    def test_message_item_with_output_text_blocks(self):
        result = messages_from_openresponses_input([
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "first "},
                    {"type": "output_text", "text": "second"},
                ],
            },
        ])
        assert result == [{"role": "assistant", "content": "first second"}]

    def test_message_item_with_input_text_blocks(self):
        result = messages_from_openresponses_input([
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "attack"}],
            },
        ])
        assert result == [{"role": "user", "content": "attack"}]

    def test_function_call_items_are_dropped(self):
        result = messages_from_openresponses_input([
            {"role": "user", "content": "use the search tool"},
            {"type": "function_call", "name": "search", "arguments": "{}", "call_id": "c1"},
            {"role": "user", "content": "follow-up"},
        ])
        assert result == [
            {"role": "user", "content": "use the search tool"},
            {"role": "user", "content": "follow-up"},
        ]

    def test_list_content_is_flattened(self):
        result = messages_from_openresponses_input([
            {"role": "user", "content": [{"type": "input_text", "text": "hello"}]},
        ])
        assert result == [{"role": "user", "content": "hello"}]


class TestRedTeamSampleFromOpenResponses:
    def test_builds_valid_sample(self):
        sample = redteam_sample_from_openresponses(
            input=_input(),
            openresponses_input=[
                {"role": "user", "content": "attack prompt"},
            ],
        )
        assert sample.input.id == "s1"
        assert len(sample.messages) == 1
        assert sample.messages[0].role == "user"
        assert sample.messages[0].content == "attack prompt"

    def test_multi_turn_conversation(self):
        sample = redteam_sample_from_openresponses(
            input=_input(),
            openresponses_input=[
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": "second"},
            ],
        )
        assert [m.role for m in sample.messages] == ["user", "assistant", "user"]

    def test_raises_when_no_messages_extracted(self):
        with pytest.raises(ValueError, match="produced no messages"):
            redteam_sample_from_openresponses(
                input=_input(),
                openresponses_input=[
                    {"type": "function_call", "name": "x", "arguments": "{}"},
                ],
            )
