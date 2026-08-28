"""Cache breakpoints on the adversarial loop — RES-1360.

This is the primary guard for the placement, not a nice-to-have. The failure it
exists to catch is hoisting `apply_cache_breakpoints` above the turn loop: the
helper returns a *copy*, so a hoisted call freezes the turn-1 two-message
snapshot and sends it for the rest of the attack, turning a multi-turn attack
into a sequence of disconnected single-turn ones. That is a correctness bug, it
is silent, and no other test in the suite would notice — the marker count stays
a valid 2 the whole time.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from evaluatorq.common.prompt_cache import CACHE_MIN_PROMPT_TOKENS

from tests.redteam.test_orchestrator_transcript import (
    _PATCH_LLM_SPAN,
    _PATCH_RECORD_LLM,
    _PATCH_REDTEAM_SPAN,
    _RecordingTarget,
    _make_completion,
    _make_context,
    _make_orchestrator,
    _make_strategy,
    _noop_span_ctx,
)

ROUTER_URL = "https://my.orq.ai/v3/router"

# The helper places nothing below CACHE_MIN_PROMPT_TOKENS, so the system prompt
# has to clear the floor for any of this to be observable. Sized off the constant
# rather than a magic number so a change to the floor fails loudly here.
_LONG_DESCRIPTION = "capabilities and constraints. " * (CACHE_MIN_PROMPT_TOKENS // 2)


def _cached_orchestrator(model: str = "anthropic/claude-sonnet-4-6") -> Any:
    orchestrator, mock_llm = _make_orchestrator()
    mock_llm.base_url = ROUTER_URL
    orchestrator.model = model
    return orchestrator, mock_llm


def _sent_messages(mock_llm: Any) -> list[list[dict[str, Any]]]:
    """The `messages` kwarg of each request, in order.

    These are the recorded *references*. Under the correct implementation each
    turn is handed a distinct copy, so per-turn assertions are meaningful; under
    an implementation that mutates the shared transcript every entry reports the
    final state instead. That still fails these tests loudly, just not with the
    per-turn numbers the message implies — read a failure as "it mutated", not as
    "turn 1 already had six markers".
    """
    return [call.kwargs["messages"] for call in mock_llm.chat.completions.create.call_args_list]


def _marked_indices(messages: list[dict[str, Any]]) -> list[int]:
    """Indices carrying a breakpoint, in list order."""
    marked = []
    for i, message in enumerate(messages):
        content = message.get("content")
        if isinstance(content, list) and any("cache_control" in part for part in content):
            marked.append(i)
    return marked


async def _run(orchestrator: Any, mock_llm: Any, turns: int) -> None:
    mock_llm.chat.completions.create.side_effect = [_make_completion(f"q{n + 1}") for n in range(turns)]
    await orchestrator.run_attack(
        target=_RecordingTarget([f"a{n + 1}" for n in range(turns)]),
        strategy=_make_strategy(),
        objective="Test cache breakpoint placement",
        agent_context=_make_context(description=_LONG_DESCRIPTION),
        max_turns=turns,
    )


@pytest.mark.asyncio
@patch(_PATCH_RECORD_LLM)
@patch(_PATCH_LLM_SPAN, side_effect=_noop_span_ctx)
@patch(_PATCH_REDTEAM_SPAN, side_effect=_noop_span_ctx)
async def test_marker_follows_the_growing_transcript(_rs: Any, _ls: Any, _rl: Any) -> None:
    """The prefix marker sits on the last message of *this* turn's list, every turn.

    The hoisting regression: every request would carry the same two messages and
    the same marked indices `[0, 1]` forever.
    """
    orchestrator, mock_llm = _cached_orchestrator()
    await _run(orchestrator, mock_llm, turns=3)

    requests = _sent_messages(mock_llm)
    assert len(requests) == 3

    lengths = [len(messages) for messages in requests]
    assert lengths == sorted(lengths) and len(set(lengths)) == 3, (
        f"transcript did not grow across turns: {lengths} — the marked copy was hoisted"
    )

    for turn, messages in enumerate(requests, start=1):
        assert _marked_indices(messages) == [0, len(messages) - 1], (
            f"turn {turn}: expected the system message and the prefix end, "
            f"got {_marked_indices(messages)} of {len(messages)} messages"
        )


@pytest.mark.asyncio
@patch(_PATCH_RECORD_LLM)
@patch(_PATCH_LLM_SPAN, side_effect=_noop_span_ctx)
@patch(_PATCH_REDTEAM_SPAN, side_effect=_noop_span_ctx)
async def test_marker_count_never_accumulates(_rs: Any, _ls: Any, _rl: Any) -> None:
    """Exactly two breakpoints on every request, well inside Anthropic's limit of four.

    A count that climbs means the marked copy reached the transcript the loop
    appends to, which the API rejects outright a few turns in.
    """
    orchestrator, mock_llm = _cached_orchestrator()
    await _run(orchestrator, mock_llm, turns=5)

    counts = [len(_marked_indices(messages)) for messages in _sent_messages(mock_llm)]
    assert counts == [2] * 5, f"breakpoint count is not constant across turns: {counts}"


@pytest.mark.asyncio
@patch(_PATCH_RECORD_LLM)
@patch(_PATCH_LLM_SPAN, side_effect=_noop_span_ctx)
@patch(_PATCH_REDTEAM_SPAN, side_effect=_noop_span_ctx)
async def test_transcript_is_not_mutated_by_marking(_rs: Any, _ls: Any, _rl: Any) -> None:
    """The appended transcript keeps plain-string content; only the sent copy is marked.

    Marked messages reaching the orchestrator's own list is how the count starts
    climbing, and it would also re-shape content the report layer reads as text.
    """
    orchestrator, mock_llm = _cached_orchestrator()
    await _run(orchestrator, mock_llm, turns=3)

    # Every request is built from the transcript, so a mutation would show up as a
    # marked message at an index that is no longer the prefix end.
    for messages in _sent_messages(mock_llm):
        for message in messages[1:-1]:
            assert isinstance(message["content"], str), (
                f"a middle message arrived as content parts: {message['content']!r} — "
                "the marked copy was written back to the transcript"
            )


@pytest.mark.asyncio
@patch(_PATCH_RECORD_LLM)
@patch(_PATCH_LLM_SPAN, side_effect=_noop_span_ctx)
@patch(_PATCH_REDTEAM_SPAN, side_effect=_noop_span_ctx)
async def test_no_markers_for_a_non_anthropic_model(_rs: Any, _ls: Any, _rl: Any) -> None:
    """`caching_applies` gates the whole thing: OpenAI caches on prefix, no marker needed."""
    orchestrator, mock_llm = _cached_orchestrator(model="openai/gpt-5.6-luna")
    await _run(orchestrator, mock_llm, turns=2)

    for messages in _sent_messages(mock_llm):
        assert _marked_indices(messages) == []


@pytest.mark.asyncio
@patch(_PATCH_RECORD_LLM)
@patch(_PATCH_LLM_SPAN, side_effect=_noop_span_ctx)
@patch(_PATCH_REDTEAM_SPAN, side_effect=_noop_span_ctx)
async def test_no_markers_when_the_client_bypasses_the_router(_rs: Any, _ls: Any, _rl: Any) -> None:
    """A direct-OpenAI client never receives the marker, whatever the model id says.

    `cache_control` inside a content part is outside the direct OpenAI schema.
    """
    orchestrator, mock_llm = _cached_orchestrator()
    mock_llm.base_url = "https://api.openai.com/v1"
    await _run(orchestrator, mock_llm, turns=2)

    for messages in _sent_messages(mock_llm):
        assert _marked_indices(messages) == []
