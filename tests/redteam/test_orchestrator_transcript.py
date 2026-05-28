"""Orchestrator builds the target transcript across turns — RES-877.

Replaces the per-target _history multi-turn tests: conversation memory now
lives in the orchestrator, which threads the growing transcript into
target.respond(messages) and skips errored turns.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evaluatorq.contracts import AgentResponse, AgentTarget, Message, TextOutputItem, ToolCallOutputItem
from evaluatorq.redteam.contracts import (
    AgentContext,
    AttackStrategy,
    AttackTechnique,
    DeliveryMethod,
    TurnType,
)


# ---------------------------------------------------------------------------
# _RecordingTarget
# ---------------------------------------------------------------------------


class _RecordingTarget(AgentTarget):
    """Records the messages list handed to each respond() call."""

    def __init__(self, replies: list[str]) -> None:
        super().__init__(memory_entity_id=None)
        self._original_replies = list(replies)
        self._replies = list(replies)
        self.calls: list[list[Message]] = []

    async def respond(self, messages: list[Message]) -> AgentResponse:
        self.calls.append(list(messages))
        text = self._replies.pop(0)
        return AgentResponse(output=[TextOutputItem(text=text, annotations=[])])

    def new(self) -> "_RecordingTarget":
        return _RecordingTarget(self._original_replies)


# ---------------------------------------------------------------------------
# Helpers (mirrors test_orchestrator_coverage.py style)
# ---------------------------------------------------------------------------


def _make_strategy(**overrides: Any) -> AttackStrategy:
    defaults: dict[str, Any] = {
        "category": "ASI01",
        "name": "test-strategy",
        "description": "Test strategy description",
        "attack_technique": AttackTechnique.INDIRECT_INJECTION,
        "delivery_methods": [DeliveryMethod.CRESCENDO],
        "turn_type": TurnType.MULTI,
        "objective_template": "Convince {agent_name} to follow instructions",
    }
    defaults.update(overrides)
    return AttackStrategy(**defaults)  # pyright: ignore[reportArgumentType]


def _make_context(**overrides: Any) -> AgentContext:
    defaults: dict[str, Any] = {
        "key": "test-agent",
        "display_name": "Test Agent",
        "description": "A helpful test assistant",
    }
    defaults.update(overrides)
    return AgentContext(**defaults)


def _make_completion(content: str, finish_reason: str = "stop", usage: Any = None) -> Any:
    """Create a mock OpenAI chat completion response."""
    choice = MagicMock()
    choice.message.content = content
    choice.finish_reason = finish_reason
    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    response.model = "test-model"
    response.id = "resp-123"
    return response


@asynccontextmanager
async def _noop_span_ctx(*args: Any, **kwargs: Any):  # type: ignore[misc]
    """Async context manager that yields None — replaces tracing spans."""
    yield None


def _make_orchestrator() -> Any:
    from evaluatorq.redteam.adaptive.orchestrator import MultiTurnOrchestrator

    mock_llm = MagicMock()
    mock_llm.chat = MagicMock()
    mock_llm.chat.completions = MagicMock()
    mock_llm.chat.completions.create = AsyncMock()

    orchestrator = MultiTurnOrchestrator(
        llm_client=mock_llm,
        model="test-model",
    )
    return orchestrator, mock_llm


_PATCH_REDTEAM_SPAN = "evaluatorq.redteam.adaptive.orchestrator.with_redteam_span"
_PATCH_LLM_SPAN = "evaluatorq.redteam.adaptive.orchestrator.with_llm_span"
_PATCH_RECORD_LLM = "evaluatorq.redteam.adaptive.orchestrator.record_llm_response"


# ---------------------------------------------------------------------------
# Behavioral tests
# ---------------------------------------------------------------------------


class TestOrchestratorTranscript:
    """The orchestrator builds and grows the transcript across turns (RES-877)."""

    @pytest.mark.asyncio
    @patch(_PATCH_RECORD_LLM)
    @patch(_PATCH_LLM_SPAN, side_effect=_noop_span_ctx)
    @patch(_PATCH_REDTEAM_SPAN, side_effect=_noop_span_ctx)
    async def test_transcript_grows_across_turns(self, _rs: Any, _ls: Any, _rl: Any) -> None:
        """Adversarial LLM emits q1, q2, q3 over 3 turns; transcript is threaded correctly.

        Turn 2's call to target.respond should include turn 1 as prior context.
        Turn 3's call should include turns 1 and 2 as prior context.
        """
        orchestrator, mock_llm = _make_orchestrator()
        target = _RecordingTarget(["a1", "a2", "a3"])

        mock_llm.chat.completions.create.side_effect = [
            _make_completion("q1"),
            _make_completion("q2"),
            _make_completion("q3"),
        ]

        await orchestrator.run_attack(
            target=target,
            strategy=_make_strategy(),
            objective="Test transcript growth",
            agent_context=_make_context(),
            max_turns=3,
        )

        assert len(target.calls) == 3

        # Turn 1: only the current user message ("q1")
        assert [(m.role, m.content) for m in target.calls[0]] == [
            ("user", "q1"),
        ]

        # Turn 2: prior context (q1/a1) + current user message
        assert [(m.role, m.content) for m in target.calls[1]] == [
            ("user", "q1"), ("assistant", "a1"), ("user", "q2"),
        ]

        # Turn 3: prior context (q1/a1/q2/a2) + current user message
        assert [(m.role, m.content) for m in target.calls[2]][:4] == [
            ("user", "q1"), ("assistant", "a1"), ("user", "q2"), ("assistant", "a2"),
        ]
        # Last message is the new attack prompt
        assert target.calls[2][-1] == Message(role="user", content="q3")

    @pytest.mark.asyncio
    @patch(_PATCH_RECORD_LLM)
    @patch(_PATCH_LLM_SPAN, side_effect=_noop_span_ctx)
    @patch(_PATCH_REDTEAM_SPAN, side_effect=_noop_span_ctx)
    async def test_errored_turn_excluded_from_transcript(self, _rs: Any, _ls: Any, _rl: Any) -> None:
        """A recovered timeout error (single, not consecutive) is excluded from the transcript.

        Turn 1: success (q1 / a1).
        Turn 2: target raises asyncio.TimeoutError — the orchestrator records an error
                AgentResponse with AgentResponseError, recovers, and continues.
        Turn 3: success (q3 / a3). The transcript passed to turn 3's target.respond
                must NOT include q2 or the timed-out reply, because skip_errors=True
                filters it out.
        """
        orchestrator, mock_llm = _make_orchestrator()

        # Target: turn 1 succeeds, turn 2 times out, turn 3 succeeds.
        # We use a custom target that raises TimeoutError on the 2nd call.
        class _TimingOutTarget(AgentTarget):
            def __init__(self) -> None:
                super().__init__(memory_entity_id=None)
                self.calls: list[list[Message]] = []
                self._call_count = 0

            async def respond(self, messages: list[Message]) -> AgentResponse:
                self.calls.append(list(messages))
                self._call_count += 1
                if self._call_count == 2:
                    raise asyncio.TimeoutError
                text = f"a{self._call_count}"
                return AgentResponse(output=[TextOutputItem(text=text, annotations=[])])

            def new(self) -> "_TimingOutTarget":
                return _TimingOutTarget()

        target = _TimingOutTarget()

        mock_llm.chat.completions.create.side_effect = [
            _make_completion("q1"),
            _make_completion("q2"),
            _make_completion("q3"),
        ]

        result = await orchestrator.run_attack(
            target=target,
            strategy=_make_strategy(),
            objective="Test error exclusion",
            agent_context=_make_context(),
            max_turns=3,
        )

        # Attack should have completed (single timeout, not consecutive)
        assert result.error_type is None
        assert result.n_turns == 3

        # The recorded errored turn carries a typed timeout marker (this is what
        # drives skip_errors — a mislabeled marker would silently change replay).
        errored = [t for t in result.turns if t.target.error is not None]
        assert len(errored) == 1
        assert errored[0].target.error is not None
        assert errored[0].target.error.error_type == "timeout"
        assert errored[0].target.error.code == "target.timeout"

        # Turn 3's transcript should NOT contain q2 or its failed reply
        turn3_messages = target.calls[2]
        turn3_pairs = [(m.role, m.content) for m in turn3_messages]

        # q2 should be absent (errored turn was skipped)
        assert ("user", "q2") not in turn3_pairs

        # But q1/a1 context is still there
        assert ("user", "q1") in turn3_pairs
        assert ("assistant", "a1") in turn3_pairs

        # And the current prompt q3 is the last user message
        assert turn3_messages[-1] == Message(role="user", content="q3")

    @pytest.mark.asyncio
    @patch(_PATCH_RECORD_LLM)
    @patch(_PATCH_LLM_SPAN, side_effect=_noop_span_ctx)
    @patch(_PATCH_REDTEAM_SPAN, side_effect=_noop_span_ctx)
    async def test_exception_turn_carries_classified_error_type(self, _rs: Any, _ls: Any, _rl: Any) -> None:
        """A target exception (not a timeout) is recorded with a classified error_type
        derived from the message, while code carries the mapped backend code."""
        orchestrator, mock_llm = _make_orchestrator()

        class _RaisingTarget(AgentTarget):
            def __init__(self) -> None:
                super().__init__(memory_entity_id=None)
                self.calls: list[list[Message]] = []
                self._call_count = 0

            async def respond(self, messages: list[Message]) -> AgentResponse:
                self.calls.append(list(messages))
                self._call_count += 1
                if self._call_count == 2:
                    raise RuntimeError("rate limit exceeded")
                text = f"a{self._call_count}"
                return AgentResponse(output=[TextOutputItem(text=text, annotations=[])])

            def new(self) -> "_RaisingTarget":
                return _RaisingTarget()

        target = _RaisingTarget()
        mock_llm.chat.completions.create.side_effect = [
            _make_completion("q1"),
            _make_completion("q2"),
            _make_completion("q3"),
        ]

        result = await orchestrator.run_attack(
            target=target,
            strategy=_make_strategy(),
            objective="Test exception classification",
            agent_context=_make_context(),
            max_turns=3,
        )

        assert result.n_turns == 3
        errored = [t for t in result.turns if t.target.error is not None]
        assert len(errored) == 1
        assert errored[0].target.error is not None
        # "rate limit exceeded" classifies to rate_limit (not the generic target_error)
        assert errored[0].target.error.error_type == "rate_limit"
        # _default_map_error (no backend) returns code "target_error"
        assert errored[0].target.error.code == "target_error"

    @pytest.mark.asyncio
    @patch(_PATCH_RECORD_LLM)
    @patch(_PATCH_LLM_SPAN, side_effect=_noop_span_ctx)
    @patch(_PATCH_REDTEAM_SPAN, side_effect=_noop_span_ctx)
    async def test_tool_calls_round_trip_through_transcript(self, _rs: Any, _ls: Any, _rl: Any) -> None:
        """RES-877: a ToolCallOutputItem emitted by the target on turn 1 must appear
        as an assistant message with tool_calls in the transcript delivered to turn 2.

        This tests the end-to-end seam: real target emits ToolCallOutputItem →
        orchestrator records the turn → turns_to_messages builds the transcript →
        next turn's respond() receives a message with tool_calls intact.
        """
        orchestrator, mock_llm = _make_orchestrator()

        class _ToolCallTarget(AgentTarget):
            """On first respond() returns a ToolCallOutputItem + text; thereafter plain text."""

            def __init__(self) -> None:
                super().__init__(memory_entity_id=None)
                self.calls: list[list[Message]] = []
                self._call_count = 0

            async def respond(self, messages: list[Message]) -> AgentResponse:
                self.calls.append(list(messages))
                self._call_count += 1
                if self._call_count == 1:
                    return AgentResponse(output=[
                        ToolCallOutputItem(
                            name="lookup",
                            arguments='{"q":"x"}',
                            id="call_1",
                            call_id="call_1",
                        ),
                        TextOutputItem(text="a1", annotations=[]),
                    ])
                return AgentResponse(output=[TextOutputItem(text=f"a{self._call_count}", annotations=[])])

            def new(self) -> _ToolCallTarget:
                return _ToolCallTarget()

        target = _ToolCallTarget()

        mock_llm.chat.completions.create.side_effect = [
            _make_completion("q1"),
            _make_completion("q2"),
        ]

        await orchestrator.run_attack(
            target=target,
            strategy=_make_strategy(),
            objective="Test tool_calls round-trip",
            agent_context=_make_context(),
            max_turns=2,
        )

        assert len(target.calls) == 2

        # Turn 2's messages must contain an assistant message carrying the tool call
        # from turn 1 — proving tool_calls survive the orchestrator transcript round-trip.
        turn2_messages = target.calls[1]
        assistant_tool_msg = next(
            (m for m in turn2_messages if m.role == "assistant" and m.tool_calls),
            None,
        )
        assert assistant_tool_msg is not None, (
            "Turn 2 transcript is missing an assistant message with tool_calls. "
            "The ToolCallOutputItem from turn 1 was not preserved through the orchestrator transcript."
        )
        assert assistant_tool_msg.tool_calls is not None
        assert assistant_tool_msg.tool_calls[0].function.name == "lookup"
