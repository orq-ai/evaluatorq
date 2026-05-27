"""turns_to_messages: list[Turn] -> list[Message], with error filtering — RES-877."""

from __future__ import annotations

from evaluatorq.contracts import (
    AgentResponse,
    AgentResponseError,
    TextOutputItem,
    ToolCallOutputItem,
)
from evaluatorq.redteam.contracts import AttackerResponse, Turn, turns_to_messages


def _turn(prompt: str, reply: str, *, error: bool = False) -> Turn:
    err = AgentResponseError(message=reply, error_type="timeout") if error else None
    return Turn(
        attacker=AttackerResponse(generated_prompt=prompt),
        target=AgentResponse(output=[TextOutputItem(text=reply, annotations=[])], error=err),
    )


def test_empty_turns_gives_empty_list():
    assert turns_to_messages([]) == []


def test_single_turn_user_then_assistant():
    msgs = turns_to_messages([_turn("hi", "hello")])
    assert [(m.role, m.content) for m in msgs] == [("user", "hi"), ("assistant", "hello")]


def test_multi_turn_alternation():
    msgs = turns_to_messages([_turn("q1", "a1"), _turn("q2", "a2")])
    assert [(m.role, m.content) for m in msgs] == [
        ("user", "q1"), ("assistant", "a1"),
        ("user", "q2"), ("assistant", "a2"),
    ]


def test_tool_calls_preserved():
    turn = Turn(
        attacker=AttackerResponse(generated_prompt="go"),
        target=AgentResponse(output=[
            ToolCallOutputItem(name="lookup", arguments='{"q":"x"}', id="c1", call_id="c1"),
        ]),
    )
    msgs = turns_to_messages([turn])
    assert msgs[0].role == "user"
    assistant_with_tool = next(m for m in msgs if m.role == "assistant" and m.tool_calls)
    assert assistant_with_tool.tool_calls[0].function.name == "lookup"


def test_empty_output_emits_empty_assistant_row():
    """An empty target output still yields a user/assistant pair (alternation invariant)."""
    turn = Turn(
        attacker=AttackerResponse(generated_prompt="q"),
        target=AgentResponse(output=[]),
    )
    msgs = turns_to_messages([turn])
    assert [(m.role, m.content) for m in msgs] == [("user", "q"), ("assistant", "")]


def test_tool_result_emits_following_tool_row():
    turn = Turn(
        attacker=AttackerResponse(generated_prompt="go"),
        target=AgentResponse(output=[
            ToolCallOutputItem(name="lookup", arguments='{"q":"x"}', id="c1", call_id="c1", result="found"),
        ]),
    )
    msgs = turns_to_messages([turn])
    tool_row = next(m for m in msgs if m.role == "tool")
    assert tool_row.tool_call_id == "c1"
    assert tool_row.content == "found"
    # the tool row immediately follows its assistant tool_calls row
    assistant_idx = next(i for i, m in enumerate(msgs) if m.role == "assistant" and m.tool_calls)
    assert msgs[assistant_idx + 1] is tool_row


def test_skip_errors_drops_errored_turn():
    turns = [_turn("q1", "a1"), _turn("q2", "[ERROR: boom]", error=True), _turn("q3", "a3")]
    kept = turns_to_messages(turns, skip_errors=True)
    assert [(m.role, m.content) for m in kept] == [
        ("user", "q1"), ("assistant", "a1"),
        ("user", "q3"), ("assistant", "a3"),
    ]


def test_default_includes_errored_turn():
    turns = [_turn("q1", "a1"), _turn("q2", "[ERROR: boom]", error=True)]
    everything = turns_to_messages(turns)  # skip_errors defaults False
    assert any(m.content == "[ERROR: boom]" for m in everything)
