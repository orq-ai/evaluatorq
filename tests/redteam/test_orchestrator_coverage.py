"""Additional unit tests for the multi-turn orchestrator (coverage supplement)."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evaluatorq.redteam.contracts import (
    AgentContext,
    AttackStrategy,
    AttackTechnique,
    DeliveryMethod,
    MemoryStoreInfo,
    ToolInfo,
    TurnType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_strategy(**overrides: Any) -> AttackStrategy:
    """Create an AttackStrategy with sensible defaults."""
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
    return AttackStrategy(**defaults)


def _make_context(**overrides: Any) -> AgentContext:
    """Create an AgentContext with sensible defaults."""
    defaults: dict[str, Any] = {
        "key": "test-agent",
        "display_name": "Test Agent",
        "description": "A helpful test assistant",
    }
    defaults.update(overrides)
    return AgentContext(**defaults)


# ===========================================================================
# orchestrator._build_adversarial_system_prompt
# ===========================================================================


class TestBuildAdversarialSystemPrompt:
    """Tests for _build_adversarial_system_prompt()."""

    def test_with_tools_listed_in_prompt(self):
        from evaluatorq.redteam.adaptive.orchestrator import _build_adversarial_system_prompt

        context = _make_context(
            tools=[
                ToolInfo(name="search", description="Search the web"),
                ToolInfo(name="calculator", description="Perform calculations"),
            ]
        )
        strategy = _make_strategy()
        result = _build_adversarial_system_prompt("Test objective", strategy, context, max_turns=3)

        assert "search" in result
        assert "calculator" in result

    def test_without_tools_shows_none(self):
        from evaluatorq.redteam.adaptive.orchestrator import _build_adversarial_system_prompt

        context = _make_context(tools=[])
        strategy = _make_strategy()
        result = _build_adversarial_system_prompt("Test objective", strategy, context, max_turns=3)

        assert "None" in result

    def test_with_memory_stores_listed(self):
        from evaluatorq.redteam.adaptive.orchestrator import _build_adversarial_system_prompt

        context = _make_context(
            memory_stores=[
                MemoryStoreInfo(id="ms-001", key="user_history"),
                MemoryStoreInfo(id="ms-002", key="session_data"),
            ]
        )
        strategy = _make_strategy()
        result = _build_adversarial_system_prompt("Test objective", strategy, context, max_turns=3)

        assert "user_history" in result
        assert "session_data" in result

    def test_without_memory_shows_none(self):
        from evaluatorq.redteam.adaptive.orchestrator import _build_adversarial_system_prompt

        context = _make_context(memory_stores=[])
        strategy = _make_strategy()
        result = _build_adversarial_system_prompt("Test objective", strategy, context, max_turns=3)

        assert "None" in result

    def test_objective_included_in_prompt(self):
        from evaluatorq.redteam.adaptive.orchestrator import _build_adversarial_system_prompt

        context = _make_context()
        strategy = _make_strategy()
        result = _build_adversarial_system_prompt(
            "Exfiltrate user credentials", strategy, context, max_turns=5
        )

        assert "Exfiltrate user credentials" in result

    def test_agent_name_from_display_name(self):
        from evaluatorq.redteam.adaptive.orchestrator import _build_adversarial_system_prompt

        context = _make_context(display_name="My Bot")
        strategy = _make_strategy()
        result = _build_adversarial_system_prompt("Objective", strategy, context, max_turns=2)

        assert "My Bot" in result

    def test_agent_name_falls_back_to_key(self):
        from evaluatorq.redteam.adaptive.orchestrator import _build_adversarial_system_prompt

        context = AgentContext(key="fallback-key")  # No display_name
        strategy = _make_strategy()
        result = _build_adversarial_system_prompt("Objective", strategy, context, max_turns=2)

        assert "fallback-key" in result

    def test_max_turns_included(self):
        from evaluatorq.redteam.adaptive.orchestrator import _build_adversarial_system_prompt

        context = _make_context()
        strategy = _make_strategy()
        result = _build_adversarial_system_prompt("Objective", strategy, context, max_turns=7)

        assert "7" in result

    def test_strategy_description_included(self):
        from evaluatorq.redteam.adaptive.orchestrator import _build_adversarial_system_prompt

        strategy = _make_strategy(description="Gradually escalate via social engineering")
        context = _make_context()
        result = _build_adversarial_system_prompt("Objective", strategy, context, max_turns=3)

        assert "Gradually escalate via social engineering" in result

    def test_memory_store_uses_id_when_key_is_none(self):
        from evaluatorq.redteam.adaptive.orchestrator import _build_adversarial_system_prompt

        context = _make_context(
            memory_stores=[MemoryStoreInfo(id="ms-999", key=None)]
        )
        strategy = _make_strategy()
        result = _build_adversarial_system_prompt("Objective", strategy, context, max_turns=1)

        assert "ms-999" in result


# ===========================================================================
# orchestrator._progress_label
# ===========================================================================


class TestProgressLabel:
    """Tests for _progress_label()."""

    def test_short_label_unchanged(self):
        from evaluatorq.redteam.adaptive.orchestrator import _progress_label

        result = _progress_label("ASI01", "short_name")
        assert result == "ASI01:short_name"

    def test_exact_max_length_unchanged(self):
        from evaluatorq.redteam.adaptive.orchestrator import (
            _PROGRESS_LABEL_MAX_LEN,
            _progress_label,
        )

        # Build a label that is exactly _PROGRESS_LABEL_MAX_LEN characters
        name_len = _PROGRESS_LABEL_MAX_LEN - len("ASI01:")
        name = "x" * name_len
        result = _progress_label("ASI01", name)
        assert len(result) == _PROGRESS_LABEL_MAX_LEN
        assert not result.endswith("…")

    def test_long_label_truncated_with_ellipsis(self):
        from evaluatorq.redteam.adaptive.orchestrator import (
            _PROGRESS_LABEL_MAX_LEN,
            _progress_label,
        )

        long_name = "a" * 100
        result = _progress_label("ASI01", long_name)
        assert len(result) == _PROGRESS_LABEL_MAX_LEN
        assert result.endswith("…")

    def test_truncated_label_starts_with_category(self):
        from evaluatorq.redteam.adaptive.orchestrator import _progress_label

        result = _progress_label("LLM07", "a" * 100)
        assert result.startswith("LLM07:")


# ===========================================================================
# TokenUsage.from_completion
# ===========================================================================


class TestExtractUsageFromCompletion:
    """Tests for TokenUsage.from_completion()."""

    def test_handles_partial_usage_fields(self):
        """When some usage fields are missing, defaults to 0."""
        from evaluatorq.redteam.contracts import TokenUsage

        usage = MagicMock(spec=["prompt_tokens"])
        usage.prompt_tokens = 42

        response = MagicMock()
        response.usage = usage

        result = TokenUsage.from_completion(response)
        assert result is not None
        assert result.prompt_tokens == 42
        assert result.completion_tokens == 0
        assert result.total_tokens == 42


# ===========================================================================
# orchestrator._merge_usage
# ===========================================================================


class TestMergeUsage:
    """Tests for _merge_usage()."""

    def test_merges_two_usages(self):
        from evaluatorq.redteam.adaptive.orchestrator import _merge_usage
        from evaluatorq.redteam.contracts import TokenUsage

        u1 = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30, calls=1)
        u2 = TokenUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15, calls=2)

        result = _merge_usage(u1, u2)
        assert result is not None
        assert result.prompt_tokens == 15
        assert result.completion_tokens == 30
        assert result.total_tokens == 45
        assert result.calls == 3

    def test_returns_none_when_all_none(self):
        from evaluatorq.redteam.adaptive.orchestrator import _merge_usage

        assert _merge_usage(None, None) is None

    def test_skips_none_entries(self):
        from evaluatorq.redteam.adaptive.orchestrator import _merge_usage
        from evaluatorq.redteam.contracts import TokenUsage

        u1 = TokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10, calls=1)
        result = _merge_usage(u1, None)

        assert result is not None
        assert result.prompt_tokens == 7
        assert result.calls == 1

    def test_single_usage_returned_correctly(self):
        from evaluatorq.redteam.adaptive.orchestrator import _merge_usage
        from evaluatorq.redteam.contracts import TokenUsage

        u = TokenUsage(prompt_tokens=50, completion_tokens=100, total_tokens=150, calls=5)
        result = _merge_usage(u)

        assert result is not None
        assert result.total_tokens == 150
        assert result.calls == 5

    def test_no_args_returns_none(self):
        from evaluatorq.redteam.adaptive.orchestrator import _merge_usage

        assert _merge_usage() is None


# ===========================================================================
# MultiTurnOrchestrator.run_attack — integration-style unit tests
# ===========================================================================


def _make_completion(content: str, finish_reason: str = "stop", usage=None):
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
async def _noop_span_ctx(*args, **kwargs):
    """Async context manager that yields None — replaces tracing spans."""
    yield None


def _make_orchestrator():
    """Create a MultiTurnOrchestrator with a mocked AsyncOpenAI client."""
    from evaluatorq.redteam.adaptive.orchestrator import MultiTurnOrchestrator
    from evaluatorq.redteam.backends.base import DefaultErrorMapper

    mock_llm = MagicMock()
    mock_llm.chat = MagicMock()
    mock_llm.chat.completions = MagicMock()
    mock_llm.chat.completions.create = AsyncMock()

    orchestrator = MultiTurnOrchestrator(
        llm_client=mock_llm,
        model="test-model",
        error_mapper=DefaultErrorMapper(),
    )
    return orchestrator, mock_llm


def _make_target():
    """Create a mock AgentTarget with the required interface."""
    target = MagicMock()
    target.send_prompt = AsyncMock()
    target.new = MagicMock()
    target.consume_last_token_usage = MagicMock(return_value=None)
    return target


_PATCH_REDTEAM_SPAN = "evaluatorq.redteam.adaptive.orchestrator.with_redteam_span"
_PATCH_LLM_SPAN = "evaluatorq.redteam.adaptive.orchestrator.with_llm_span"
_PATCH_RECORD_LLM = "evaluatorq.redteam.adaptive.orchestrator.record_llm_response"


class TestRunAttack:
    """Integration-style unit tests for MultiTurnOrchestrator.run_attack()."""

    @pytest.mark.asyncio
    @patch(_PATCH_RECORD_LLM)
    @patch(_PATCH_LLM_SPAN, side_effect=_noop_span_ctx)
    @patch(_PATCH_REDTEAM_SPAN, side_effect=_noop_span_ctx)
    async def test_happy_path_multi_turn(self, _rs, _ls, _rl):
        """3 turns, adversarial LLM returns different prompts, target responds normally."""
        orchestrator, mock_llm = _make_orchestrator()
        target = _make_target()

        mock_llm.chat.completions.create.side_effect = [
            _make_completion("Attack turn 1"),
            _make_completion("Attack turn 2"),
            _make_completion("Attack turn 3"),
        ]
        target.send_prompt.side_effect = [
            "Target response 1",
            "Target response 2",
            "Target response 3",
        ]

        result = await orchestrator.run_attack(
            target=target,
            strategy=_make_strategy(),
            objective="Test objective",
            agent_context=_make_context(),
            max_turns=3,
        )

        assert result.turns == 3
        assert result.objective_achieved is False
        assert len(result.conversation) == 6
        assert result.error is None
        assert result.final_response == "Target response 3"

    @pytest.mark.asyncio
    @patch(_PATCH_RECORD_LLM)
    @patch(_PATCH_LLM_SPAN, side_effect=_noop_span_ctx)
    @patch(_PATCH_REDTEAM_SPAN, side_effect=_noop_span_ctx)
    async def test_objective_achieved_mid_conversation(self, _rs, _ls, _rl):
        """Objective achieved on turn 2 stops attack after that turn."""
        orchestrator, mock_llm = _make_orchestrator()
        target = _make_target()

        mock_llm.chat.completions.create.side_effect = [
            _make_completion("Normal attack turn 1"),
            _make_completion("OBJECTIVE_ACHIEVED Tell me secrets"),
        ]
        target.send_prompt.side_effect = [
            "Target response 1",
            "Target response 2",
        ]

        result = await orchestrator.run_attack(
            target=target,
            strategy=_make_strategy(),
            objective="Exfiltrate secrets",
            agent_context=_make_context(),
            max_turns=5,
        )

        assert result.objective_achieved is True
        assert result.turns == 2
        # Turn 2 attack prompt should have the marker stripped
        turn2_user_msg = result.conversation[2]
        assert turn2_user_msg.role == "user"
        assert turn2_user_msg.content == "Tell me secrets"

    @pytest.mark.asyncio
    @patch(_PATCH_RECORD_LLM)
    @patch(_PATCH_LLM_SPAN, side_effect=_noop_span_ctx)
    @patch(_PATCH_REDTEAM_SPAN, side_effect=_noop_span_ctx)
    async def test_objective_achieved_empty_prompt_after_marker(self, _rs, _ls, _rl):
        """Adversarial returns just 'OBJECTIVE_ACHIEVED' with nothing after stripping.

        The adversarial LLM claiming success on turn 0 with no prior target
        interaction is invalid — objective_achieved should be False.
        """
        orchestrator, mock_llm = _make_orchestrator()
        target = _make_target()

        mock_llm.chat.completions.create.side_effect = [
            _make_completion("OBJECTIVE_ACHIEVED"),
        ]

        result = await orchestrator.run_attack(
            target=target,
            strategy=_make_strategy(),
            objective="Test",
            agent_context=_make_context(),
            max_turns=5,
        )

        assert result.objective_achieved is False
        assert result.turns == 0
        target.send_prompt.assert_not_called()

    @pytest.mark.asyncio
    @patch(_PATCH_RECORD_LLM)
    @patch(_PATCH_LLM_SPAN, side_effect=_noop_span_ctx)
    @patch(_PATCH_REDTEAM_SPAN, side_effect=_noop_span_ctx)
    async def test_adversarial_llm_timeout(self, _rs, _ls, _rl):
        """Timeout from adversarial LLM sets error fields correctly."""
        orchestrator, mock_llm = _make_orchestrator()
        target = _make_target()

        mock_llm.chat.completions.create.side_effect = asyncio.TimeoutError

        result = await orchestrator.run_attack(
            target=target,
            strategy=_make_strategy(),
            objective="Test",
            agent_context=_make_context(),
            max_turns=3,
        )

        assert result.error_code == "adversarial.timeout"
        assert result.error_stage == "adversarial_generation"
        assert result.error_type == "llm_error"

    @pytest.mark.asyncio
    @patch(_PATCH_RECORD_LLM)
    @patch(_PATCH_LLM_SPAN, side_effect=_noop_span_ctx)
    @patch(_PATCH_REDTEAM_SPAN, side_effect=_noop_span_ctx)
    async def test_adversarial_llm_exception(self, _rs, _ls, _rl):
        """Generic exception from adversarial LLM is caught and mapped."""
        orchestrator, mock_llm = _make_orchestrator()
        target = _make_target()

        mock_llm.chat.completions.create.side_effect = RuntimeError("API broke")

        result = await orchestrator.run_attack(
            target=target,
            strategy=_make_strategy(),
            objective="Test",
            agent_context=_make_context(),
            max_turns=3,
        )

        assert result.error_code == "adversarial.llm_exception"
        assert result.error_type == "llm_error"
        assert result.error is not None
        assert "API broke" in result.error

    @pytest.mark.asyncio
    @patch(_PATCH_RECORD_LLM)
    @patch(_PATCH_LLM_SPAN, side_effect=_noop_span_ctx)
    @patch(_PATCH_REDTEAM_SPAN, side_effect=_noop_span_ctx)
    async def test_adversarial_empty_content_filter(self, _rs, _ls, _rl):
        """Empty content due to content_filter finish_reason maps correctly."""
        orchestrator, mock_llm = _make_orchestrator()
        target = _make_target()

        mock_llm.chat.completions.create.return_value = _make_completion(
            "", finish_reason="content_filter"
        )

        result = await orchestrator.run_attack(
            target=target,
            strategy=_make_strategy(),
            objective="Test",
            agent_context=_make_context(),
            max_turns=3,
        )

        assert result.error_type == "content_filter"
        assert result.error_code == "adversarial.content_filter"

    @pytest.mark.asyncio
    @patch(_PATCH_RECORD_LLM)
    @patch(_PATCH_LLM_SPAN, side_effect=_noop_span_ctx)
    @patch(_PATCH_REDTEAM_SPAN, side_effect=_noop_span_ctx)
    async def test_adversarial_empty_max_tokens(self, _rs, _ls, _rl):
        """Empty content due to max-tokens truncation maps to 'max_tokens' error."""
        orchestrator, mock_llm = _make_orchestrator()
        target = _make_target()

        mock_llm.chat.completions.create.return_value = _make_completion(
            "", finish_reason="length"
        )

        result = await orchestrator.run_attack(
            target=target,
            strategy=_make_strategy(),
            objective="Test",
            agent_context=_make_context(),
            max_turns=3,
        )

        assert result.error_type == "max_tokens"
        assert result.error_code == "adversarial.max_tokens"

    @pytest.mark.asyncio
    @patch(_PATCH_RECORD_LLM)
    @patch(_PATCH_LLM_SPAN, side_effect=_noop_span_ctx)
    @patch(_PATCH_REDTEAM_SPAN, side_effect=_noop_span_ctx)
    async def test_target_timeout_single_recovers(self, _rs, _ls, _rl):
        """A single target timeout does not abort — attack continues to completion."""
        orchestrator, mock_llm = _make_orchestrator()
        target = _make_target()

        mock_llm.chat.completions.create.side_effect = [
            _make_completion("Attack turn 1"),
            _make_completion("Attack turn 2"),
            _make_completion("Attack turn 3"),
        ]
        target.send_prompt.side_effect = [
            asyncio.TimeoutError,
            "OK response",
            "Another response",
        ]

        result = await orchestrator.run_attack(
            target=target,
            strategy=_make_strategy(),
            objective="Test",
            agent_context=_make_context(),
            max_turns=3,
        )

        assert result.error is None
        assert result.turns == 3
        # Turn 1 recorded an error message placeholder in the conversation
        turn1_assistant = result.conversation[1]
        assert turn1_assistant.content is not None
        assert "ERROR" in turn1_assistant.content or "timed out" in turn1_assistant.content.lower()

    @pytest.mark.asyncio
    @patch(_PATCH_RECORD_LLM)
    @patch(_PATCH_LLM_SPAN, side_effect=_noop_span_ctx)
    @patch(_PATCH_REDTEAM_SPAN, side_effect=_noop_span_ctx)
    async def test_target_timeout_consecutive_aborts(self, _rs, _ls, _rl):
        """Two consecutive target timeouts abort the attack."""
        orchestrator, mock_llm = _make_orchestrator()
        target = _make_target()

        mock_llm.chat.completions.create.side_effect = [
            _make_completion("Attack turn 1"),
            _make_completion("Attack turn 2"),
        ]
        target.send_prompt.side_effect = [
            asyncio.TimeoutError,
            asyncio.TimeoutError,
        ]

        result = await orchestrator.run_attack(
            target=target,
            strategy=_make_strategy(),
            objective="Test",
            agent_context=_make_context(),
            max_turns=5,
        )

        assert result.error_code == "target.timeout"
        assert result.error_type == "target_error"
        assert result.turns == 2

    @pytest.mark.asyncio
    @patch(_PATCH_RECORD_LLM)
    @patch(_PATCH_LLM_SPAN, side_effect=_noop_span_ctx)
    @patch(_PATCH_REDTEAM_SPAN, side_effect=_noop_span_ctx)
    async def test_target_exception_consecutive_aborts(self, _rs, _ls, _rl):
        """Two consecutive target exceptions abort the attack with target_error."""
        orchestrator, mock_llm = _make_orchestrator()
        target = _make_target()

        mock_llm.chat.completions.create.side_effect = [
            _make_completion("Attack turn 1"),
            _make_completion("Attack turn 2"),
        ]
        target.send_prompt.side_effect = [
            RuntimeError("connection refused"),
            RuntimeError("connection refused"),
        ]

        result = await orchestrator.run_attack(
            target=target,
            strategy=_make_strategy(),
            objective="Test",
            agent_context=_make_context(),
            max_turns=5,
        )

        assert result.error_type == "target_error"
        assert result.error_stage == "target_call"
        assert result.turns == 2

    @pytest.mark.asyncio
    @patch(_PATCH_RECORD_LLM)
    @patch(_PATCH_LLM_SPAN, side_effect=_noop_span_ctx)
    @patch(_PATCH_REDTEAM_SPAN, side_effect=_noop_span_ctx)
    async def test_truncation_warning_tracked(self, _rs, _ls, _rl):
        """Turns where finish_reason='length' are tracked in truncated_turns."""
        orchestrator, mock_llm = _make_orchestrator()
        target = _make_target()

        mock_llm.chat.completions.create.side_effect = [
            _make_completion("Attack turn 1", finish_reason="length"),
            _make_completion("Attack turn 2", finish_reason="stop"),
            _make_completion("Attack turn 3", finish_reason="length"),
        ]
        target.send_prompt.side_effect = [
            "Target response 1",
            "Target response 2",
            "Target response 3",
        ]

        result = await orchestrator.run_attack(
            target=target,
            strategy=_make_strategy(),
            objective="Test",
            agent_context=_make_context(),
            max_turns=3,
        )

        assert result.truncated_turns == [1, 3]

    @pytest.mark.asyncio
    @patch(_PATCH_RECORD_LLM)
    @patch(_PATCH_LLM_SPAN, side_effect=_noop_span_ctx)
    @patch(_PATCH_REDTEAM_SPAN, side_effect=_noop_span_ctx)
    async def test_single_turn_attack(self, _rs, _ls, _rl):
        """max_turns=1 produces exactly 1 exchange in the conversation."""
        orchestrator, mock_llm = _make_orchestrator()
        target = _make_target()

        mock_llm.chat.completions.create.return_value = _make_completion("Single attack prompt")
        target.send_prompt.return_value = "Single response"

        result = await orchestrator.run_attack(
            target=target,
            strategy=_make_strategy(),
            objective="Test",
            agent_context=_make_context(),
            max_turns=1,
        )

        assert result.turns == 1
        assert len(result.conversation) == 2
