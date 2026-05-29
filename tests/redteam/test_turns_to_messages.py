"""turns_to_messages: list[Turn] -> list[Message], with error filtering — RES-877."""

from __future__ import annotations

from evaluatorq.contracts import (
    AgentResponse,
    AgentResponseError,
    ReasoningOutputItem,
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
    assert assistant_with_tool.tool_calls is not None
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


def test_reasoning_only_output_emits_empty_assistant_row():
    """All-reasoning output still yields a user/assistant pair (alternation invariant).

    Reasoning items are dropped by turns_to_messages; without an assistant row the
    transcript would have a bare user turn and downstream chat-completions APIs
    would 400 on the broken alternation.
    """
    turn = Turn(
        attacker=AttackerResponse(generated_prompt="q"),
        target=AgentResponse(output=[ReasoningOutputItem(text="thinking...")]),
    )
    msgs = turns_to_messages([turn])
    assert [(m.role, m.content) for m in msgs] == [("user", "q"), ("assistant", "")]


def test_tool_call_with_result_does_not_emit_extra_assistant_row():
    """A turn ending in a tool result must not get a trailing empty assistant row."""
    turn = Turn(
        attacker=AttackerResponse(generated_prompt="go"),
        target=AgentResponse(output=[
            ToolCallOutputItem(name="lookup", arguments='{"q":"x"}', id="c1", call_id="c1", result="found"),
        ]),
    )
    msgs = turns_to_messages([turn])
    assert [m.role for m in msgs] == ["user", "assistant", "tool"]


def test_tool_call_preserves_responses_item_id():
    """Responses-API fc_* item id round-trips through StrategyToolCall.item_id."""
    turn = Turn(
        attacker=AttackerResponse(generated_prompt="go"),
        target=AgentResponse(output=[
            ToolCallOutputItem(name="lookup", arguments='{"q":"x"}', id="fc_abc", call_id="call_xyz"),
        ]),
    )
    msgs = turns_to_messages([turn])
    assistant = next(m for m in msgs if m.role == "assistant" and m.tool_calls)
    assert assistant.tool_calls is not None
    tc = assistant.tool_calls[0]
    assert tc.id == "call_xyz"
    assert tc.item_id == "fc_abc"


def test_interleaved_text_tool_text_emits_three_assistant_rows():
    """The docstring claims ``Text -> ToolCall -> Text`` yields three assistant
    rows in order. A refactor that merged the bracketing text rows would silently
    break multi-turn replay for agents that narrate around tool use.
    """
    turn = Turn(
        attacker=AttackerResponse(generated_prompt="search"),
        target=AgentResponse(output=[
            TextOutputItem(text="thinking...", annotations=[]),
            ToolCallOutputItem(name="lookup", arguments="{}", id="c1", call_id="c1"),
            TextOutputItem(text="found it", annotations=[]),
        ]),
    )
    msgs = turns_to_messages([turn])
    # user, assistant("thinking..."), assistant(tool_calls), assistant("found it")
    assert [m.role for m in msgs] == ["user", "assistant", "assistant", "assistant"]
    assert msgs[1].content == "thinking..."
    assert msgs[2].tool_calls is not None and msgs[2].tool_calls[0].function.name == "lookup"
    assert msgs[3].content == "found it"


def test_consecutive_text_items_collapse_into_one_assistant_row():
    """Two adjacent ``TextOutputItem``s join into a single assistant message.
    Documented in ``turns_to_messages``; pinning the behavior here.
    """
    turn = Turn(
        attacker=AttackerResponse(generated_prompt="hi"),
        target=AgentResponse(output=[
            TextOutputItem(text="part one ", annotations=[]),
            TextOutputItem(text="part two", annotations=[]),
        ]),
    )
    msgs = turns_to_messages([turn])
    assert [m.role for m in msgs] == ["user", "assistant"]
    assert msgs[1].content == "part one part two"


def test_tool_result_empty_string_still_emits_tool_row():
    """``result=""`` is a legitimate tool return (e.g. an action with no
    payload). The current check is ``if item.result is not None``; a refactor
    to ``if item.result:`` would silently drop the row and break alternation.
    """
    turn = Turn(
        attacker=AttackerResponse(generated_prompt="go"),
        target=AgentResponse(output=[
            ToolCallOutputItem(name="noop", arguments="{}", id="c1", call_id="c1", result=""),
        ]),
    )
    msgs = turns_to_messages([turn])
    assert [m.role for m in msgs] == ["user", "assistant", "tool"]
    assert msgs[2].content == ""


def test_reasoning_interleaved_with_text_and_tool_dropped_silently():
    """Reasoning items are dropped, but the surrounding text/tool ordering must
    not collapse — flushing the text buffer on a reasoning item would split
    what should be one assistant text row.
    """
    from evaluatorq.contracts import ReasoningOutputItem

    turn = Turn(
        attacker=AttackerResponse(generated_prompt="q"),
        target=AgentResponse(output=[
            TextOutputItem(text="hello", annotations=[]),
            ReasoningOutputItem(text="(internal)"),
            ToolCallOutputItem(name="lookup", arguments="{}", id="c1", call_id="c1", result="ok"),
        ]),
    )
    msgs = turns_to_messages([turn])
    # user, assistant("hello"), assistant(tool_calls), tool
    assert [m.role for m in msgs] == ["user", "assistant", "assistant", "tool"]
    assert msgs[1].content == "hello"
