"""Tests for multi-turn OpenResponses conversation state in red teaming.

Covers RES-540 touchpoint 2: the adversarial agent maintains conversation
state in OpenResponses format, appending assistant responses and follow-up
attacks.
"""

from __future__ import annotations

from typing import Any

from evaluatorq.contracts import AgentResponse, TextOutputItem, ToolCallOutputItem
from evaluatorq.openresponses import (
    append_assistant_turn,
    append_user_followup,
    build_openresponses_request,
    turns_to_openresponses_input,
    user_input_item,
)
from evaluatorq.redteam.contracts import AttackerResponse, Turn


def _agent_text(text: str) -> AgentResponse:
    return AgentResponse(output=[TextOutputItem(text=text, annotations=[])])


def _turn(attacker_prompt: str, agent_text: str) -> Turn:
    return Turn(
        attacker=AttackerResponse(generated_prompt=attacker_prompt),
        target=_agent_text(agent_text),
    )


class TestAppendAssistantTurn:
    def test_appends_assistant_text_item(self):
        input_array: list[dict[str, Any]] = [user_input_item("initial attack")]
        append_assistant_turn(input_array, _agent_text("agent response"))
        assert input_array == [
            {"role": "user", "content": "initial attack"},
            {"role": "assistant", "content": "agent response"},
        ]

    def test_appends_nothing_when_response_has_no_text(self):
        input_array: list[dict[str, Any]] = [user_input_item("x")]
        append_assistant_turn(input_array, AgentResponse(output=[]))
        assert input_array == [{"role": "user", "content": "x"}]

    def test_accepts_openresponses_resource_dict(self):
        response_resource = {
            "id": "resp_1",
            "object": "response",
            "model": "agent",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ok"}],
            }],
        }
        input_array: list[dict[str, Any]] = []
        append_assistant_turn(input_array, response_resource)
        assert input_array == [{"role": "assistant", "content": "ok"}]

    def test_preserves_assistant_tool_calls(self):
        input_array: list[dict[str, Any]] = [user_input_item("use a tool")]
        response = AgentResponse(
            output=[
                ToolCallOutputItem(
                    name="lookup",
                    arguments='{"query":"x"}',
                    call_id="call-1",
                )
            ]
        )

        append_assistant_turn(input_array, response)

        assert input_array == [
            {"role": "user", "content": "use a tool"},
            {
                "type": "function_call",
                "name": "lookup",
                "arguments": '{"query":"x"}',
                "call_id": "call-1",
            },
        ]


class TestAppendUserFollowup:
    def test_appends_user_role_item(self):
        input_array: list[dict[str, Any]] = [
            user_input_item("initial attack"),
            {"role": "assistant", "content": "agent response"},
        ]
        append_user_followup(input_array, "follow-up attack")
        assert input_array[-1] == {"role": "user", "content": "follow-up attack"}


class TestTurnsToOpenResponsesInput:
    def test_two_turn_conversation_round_trips_to_input_array(self):
        turns = [
            _turn("initial attack", "agent response"),
            _turn("follow-up attack", "second agent response"),
        ]
        result = turns_to_openresponses_input(turns)
        assert result == [
            {"role": "user", "content": "initial attack"},
            {"role": "assistant", "content": "agent response"},
            {"role": "user", "content": "follow-up attack"},
            {"role": "assistant", "content": "second agent response"},
        ]

    def test_include_final_assistant_false_omits_last_response(self):
        turns = [_turn("u1", "a1"), _turn("u2", "a2")]
        result = turns_to_openresponses_input(turns, include_final_assistant=False)
        assert result == [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ]

    def test_matches_ticket_example_shape(self):
        # Mirrors the example from the RES-540 description:
        # { "model": "agent-id",
        #   "input": [{"role": "user", "content": "initial attack"},
        #             {"role": "assistant", "content": "agent response"},
        #             {"role": "user", "content": "follow-up attack"}] }
        conversation = turns_to_openresponses_input(
            [_turn("initial attack", "agent response")],
        )
        request = build_openresponses_request(
            model="agent-id",
            prompt="follow-up attack",
            conversation=conversation,
        )
        assert request == {
            "model": "agent-id",
            "input": [
                {"role": "user", "content": "initial attack"},
                {"role": "assistant", "content": "agent response"},
                {"role": "user", "content": "follow-up attack"},
            ],
        }
