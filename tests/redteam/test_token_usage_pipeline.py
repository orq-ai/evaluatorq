"""End-to-end token usage validation for the red teaming pipeline.

Mocks both the adversarial LLM endpoint and the target backend, runs a full
``MultiTurnOrchestrator.run_attack`` pipeline, and asserts that the final
aggregated counts on :class:`OrchestratorResult` match the per-call values
that were emitted upstream.

The motivation is RES-596 / PR #127: after consolidating ``TokenUsage`` and
``Message`` across simulation and redteam, multiple aggregation paths exist
(adversarial accumulator, target accumulator, ``_merge_usage``,
``TokenUsage.from_completion``, ``TokenUsage.__add__``). These tests pin the
contract so that future refactors cannot silently drift call/token totals.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from evaluatorq.redteam.adaptive.orchestrator import MultiTurnOrchestrator
from evaluatorq.contracts import AgentTarget
from evaluatorq.redteam.contracts import (
    AgentContext,
    AgentInfo,
    AgentResponse,
    AttackInfo,
    AttackStrategy,
    AttackSource,
    AttackTechnique,
    DeliveryMethod,
    ExecutionDetails,
    Framework,
    RedTeamResult,
    Severity,
    TokenUsage,
    TurnType,
    UnifiedEvaluationResult,
)
from evaluatorq.redteam.reports.converters import compute_report_summary


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


# Per-call usage budgets used across tests. Distinct prime-ish numbers so any
# accidental cross-attribution between adversarial and target accumulators
# shows up as an arithmetic mismatch.
ADVERSARIAL_PROMPT = 11
ADVERSARIAL_COMPLETION = 7
ADVERSARIAL_TOTAL = 18  # = 11 + 7
TARGET_PROMPT = 23
TARGET_COMPLETION = 13
TARGET_TOTAL = 36  # = 23 + 13


def _make_chat_completion(content: str = "next attack prompt") -> MagicMock:
    """Build a fake ``ChatCompletion`` with deterministic usage."""
    choice = MagicMock()
    choice.message.content = content
    choice.finish_reason = "stop"
    response = MagicMock()
    response.choices = [choice]
    response.model = "fake/model"
    usage = MagicMock()
    usage.prompt_tokens = ADVERSARIAL_PROMPT
    usage.completion_tokens = ADVERSARIAL_COMPLETION
    usage.total_tokens = ADVERSARIAL_TOTAL
    response.usage = usage
    return response


class _FakeTarget(AgentTarget):
    """Minimal :class:`AgentTarget` returning fixed token usage each call."""

    def __init__(self, usage: TokenUsage | None) -> None:
        super().__init__()
        self._usage = usage
        self.call_count = 0

    async def send_prompt(self, prompt: str) -> AgentResponse:
        self.call_count += 1
        return AgentResponse(text=f"target reply {self.call_count}", usage=self._usage)

    def new(self) -> "_FakeTarget":
        return _FakeTarget(self._usage)


def _make_strategy() -> AttackStrategy:
    return AttackStrategy(
        category="ASI01",
        name="token-usage-pipeline-test",
        description="Token-usage pipeline test strategy",
        attack_technique=AttackTechnique.INDIRECT_INJECTION,
        delivery_methods=[DeliveryMethod.CRESCENDO],
        turn_type=TurnType.MULTI,
        objective_template="Validate token accounting",
    )


def _make_orchestrator(create_mock: Any) -> MultiTurnOrchestrator:
    llm = AsyncMock()
    llm.chat.completions.create = create_mock
    return MultiTurnOrchestrator(llm_client=llm, model="fake/model")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPipelineTokenUsageAggregation:
    """Validate that ``OrchestratorResult`` aggregates exactly match upstream emissions."""

    @pytest.mark.asyncio
    async def test_full_run_attacker_and_target_counts_balance(self) -> None:
        """N turns × (adversarial + target) usage must equal aggregated totals.

        Catches: any drift between per-call accumulation in ``run_attack`` and
        the final ``TokenUsage(...)`` construction at the bottom of the loop.
        """
        max_turns = 3
        create_mock = AsyncMock(side_effect=[_make_chat_completion() for _ in range(max_turns)])
        target_usage = TokenUsage(
            prompt_tokens=TARGET_PROMPT,
            completion_tokens=TARGET_COMPLETION,
            total_tokens=TARGET_TOTAL,
            calls=1,
        )
        target = _FakeTarget(target_usage)
        orchestrator = _make_orchestrator(create_mock)

        result = await orchestrator.run_attack(
            target=target,
            strategy=_make_strategy(),
            objective="Test",
            agent_context=AgentContext(key="fake_agent"),
            max_turns=max_turns,
        )

        assert result.error is None
        assert result.n_turns == max_turns
        assert target.call_count == max_turns
        assert create_mock.await_count == max_turns

        adv = result.token_usage_adversarial
        assert adv is not None
        assert adv.calls == max_turns
        assert adv.prompt_tokens == ADVERSARIAL_PROMPT * max_turns
        assert adv.completion_tokens == ADVERSARIAL_COMPLETION * max_turns
        assert adv.total_tokens == ADVERSARIAL_TOTAL * max_turns

        tgt = result.token_usage_target
        assert tgt is not None
        assert tgt.calls == max_turns
        assert tgt.prompt_tokens == TARGET_PROMPT * max_turns
        assert tgt.completion_tokens == TARGET_COMPLETION * max_turns
        assert tgt.total_tokens == TARGET_TOTAL * max_turns

        # The merged total must equal the arithmetic sum of the two components.
        # Use TokenUsage.__add__ so any future change to merge semantics is
        # caught here too.
        expected_total = adv + tgt
        assert result.token_usage is not None
        assert result.token_usage.prompt_tokens == expected_total.prompt_tokens
        assert result.token_usage.completion_tokens == expected_total.completion_tokens
        assert result.token_usage.total_tokens == expected_total.total_tokens
        assert result.token_usage.calls == expected_total.calls
        assert result.token_usage.calls == 2 * max_turns

    @pytest.mark.asyncio
    async def test_target_without_usage_yields_none_target_totals(self) -> None:
        """Target returning ``usage=None`` must leave ``token_usage_target=None``.

        Catches: a regression where ``target_calls`` accidentally increments
        from ``calls or 0 or 1`` when ``target_usage`` is ``None`` (it must
        short-circuit before the ``or 1`` fallback).
        """
        max_turns = 2
        create_mock = AsyncMock(side_effect=[_make_chat_completion() for _ in range(max_turns)])
        target = _FakeTarget(usage=None)
        orchestrator = _make_orchestrator(create_mock)

        result = await orchestrator.run_attack(
            target=target,
            strategy=_make_strategy(),
            objective="Test",
            agent_context=AgentContext(key="fake_agent"),
            max_turns=max_turns,
        )

        assert result.error is None
        assert target.call_count == max_turns
        assert result.token_usage_target is None

        adv = result.token_usage_adversarial
        assert adv is not None
        assert adv.calls == max_turns

        # Merged total degenerates to the adversarial component alone.
        assert result.token_usage is not None
        assert result.token_usage.calls == adv.calls
        assert result.token_usage.total_tokens == adv.total_tokens

    @pytest.mark.asyncio
    async def test_provider_total_trusted_when_diverges_from_components(self) -> None:
        """Provider-reported ``total_tokens`` is propagated through aggregation.

        Catches: any silent normalization that would replace the reported
        total with ``prompt + completion`` (which would corrupt cached/
        reasoning/audio token accounting upstream).
        """
        max_turns = 2
        # Provider reports an "unusual" total (e.g. includes cached/reasoning
        # tokens) that is *not* prompt+completion. Aggregation must preserve it.
        divergent_total = ADVERSARIAL_PROMPT + ADVERSARIAL_COMPLETION + 100

        def _make_divergent() -> MagicMock:
            r = _make_chat_completion()
            r.usage.total_tokens = divergent_total
            return r

        create_mock = AsyncMock(side_effect=[_make_divergent() for _ in range(max_turns)])
        target = _FakeTarget(usage=None)
        orchestrator = _make_orchestrator(create_mock)

        result = await orchestrator.run_attack(
            target=target,
            strategy=_make_strategy(),
            objective="Test",
            agent_context=AgentContext(key="fake_agent"),
            max_turns=max_turns,
        )

        adv = result.token_usage_adversarial
        assert adv is not None
        assert adv.total_tokens == divergent_total * max_turns
        # prompt+completion is unchanged (proves the divergent total was not
        # silently rewritten to prompt+completion).
        assert adv.prompt_tokens == ADVERSARIAL_PROMPT * max_turns
        assert adv.completion_tokens == ADVERSARIAL_COMPLETION * max_turns
        assert adv.total_tokens != adv.prompt_tokens + adv.completion_tokens

    @pytest.mark.asyncio
    async def test_partial_run_after_target_failure_counts_consistent(self) -> None:
        """When the target fails consecutively, partial counts must still balance.

        The orchestrator aborts after two consecutive target errors. Adversarial
        calls = turns attempted; target calls = turns where the target actually
        returned usage (zero here, since every call raises).
        """
        max_turns = 4
        create_mock = AsyncMock(side_effect=[_make_chat_completion() for _ in range(max_turns)])

        class _FailingTarget(AgentTarget):
            def __init__(self) -> None:
                super().__init__()
                self.attempts = 0

            async def send_prompt(self, prompt: str) -> AgentResponse:
                self.attempts += 1
                raise RuntimeError("target down")

            def new(self) -> "_FailingTarget":
                return _FailingTarget()

        target = _FailingTarget()
        orchestrator = _make_orchestrator(create_mock)

        result = await orchestrator.run_attack(
            target=target,
            strategy=_make_strategy(),
            objective="Test",
            agent_context=AgentContext(key="fake_agent"),
            max_turns=max_turns,
        )

        # Aborted with a target error after 2 consecutive failures.
        assert result.error is not None
        assert result.error_type == "target_error"
        assert target.attempts == 2

        # Adversarial LLM was called once per attempted turn (2 turns before abort).
        adv = result.token_usage_adversarial
        assert adv is not None
        assert adv.calls == 2
        assert adv.total_tokens == ADVERSARIAL_TOTAL * 2

        # Target never produced usage, so the target aggregate is absent.
        assert result.token_usage_target is None

        # Merged total equals adversarial only.
        assert result.token_usage is not None
        assert result.token_usage.calls == adv.calls
        assert result.token_usage.total_tokens == adv.total_tokens

    @pytest.mark.asyncio
    async def test_aggregated_total_equals_sum_of_per_turn_usage(self) -> None:
        """Sum of per-turn ``AttackerResponse.usage`` must equal ``token_usage_adversarial``.

        Pins the invariant that the canonical per-turn record and the
        aggregated counter cannot drift.
        """
        max_turns = 3
        create_mock = AsyncMock(side_effect=[_make_chat_completion() for _ in range(max_turns)])
        target = _FakeTarget(
            TokenUsage(
                prompt_tokens=TARGET_PROMPT,
                completion_tokens=TARGET_COMPLETION,
                total_tokens=TARGET_TOTAL,
                calls=1,
            )
        )
        orchestrator = _make_orchestrator(create_mock)

        result = await orchestrator.run_attack(
            target=target,
            strategy=_make_strategy(),
            objective="Test",
            agent_context=AgentContext(key="fake_agent"),
            max_turns=max_turns,
        )

        per_turn_attacker_sum = sum(
            (t.attacker.usage for t in result.turns if t.attacker.usage is not None),
            start=TokenUsage(),
        )
        per_turn_target_sum = sum(
            (t.target.usage for t in result.turns if t.target.usage is not None),
            start=TokenUsage(),
        )

        adv = result.token_usage_adversarial
        tgt = result.token_usage_target
        assert adv is not None
        assert tgt is not None

        assert per_turn_attacker_sum.calls == adv.calls
        assert per_turn_attacker_sum.prompt_tokens == adv.prompt_tokens
        assert per_turn_attacker_sum.completion_tokens == adv.completion_tokens
        assert per_turn_attacker_sum.total_tokens == adv.total_tokens

        assert per_turn_target_sum.calls == tgt.calls
        assert per_turn_target_sum.prompt_tokens == tgt.prompt_tokens
        assert per_turn_target_sum.completion_tokens == tgt.completion_tokens
        assert per_turn_target_sum.total_tokens == tgt.total_tokens


# ---------------------------------------------------------------------------
# Report-level aggregation
# ---------------------------------------------------------------------------


def _make_result(
    *,
    category: str = "ASI01",
    execution_usage: TokenUsage | None,
    evaluation_usage: TokenUsage | None = None,
    has_execution: bool = True,
    vulnerable: bool = False,
) -> RedTeamResult:
    """Build a minimal :class:`RedTeamResult` with explicit usage on each layer."""
    attack = AttackInfo(
        id=f"{category}-test",
        vulnerability="goal_hijacking",
        category=category,
        framework=Framework.OWASP_ASI,
        attack_technique=AttackTechnique.INDIRECT_INJECTION,
        delivery_methods=[DeliveryMethod.CRESCENDO],
        turn_type=TurnType.MULTI,
        severity=Severity.MEDIUM,
        source=AttackSource.TEMPLATE_DYNAMIC,
        strategy_name="agg-test",
    )
    execution = (
        ExecutionDetails(turns=1, token_usage=execution_usage) if has_execution else None
    )
    evaluation = UnifiedEvaluationResult(
        passed=not vulnerable,
        evaluator_id="test",
        token_usage=evaluation_usage,
    )
    return RedTeamResult(
        attack=attack,
        agent=AgentInfo(key="fake"),
        messages=[],
        evaluation=evaluation,
        vulnerable=vulnerable,
        execution=execution,
    )


class TestReportLevelAggregation:
    """Validate ``ReportSummary.token_usage_total`` across many results."""

    def test_summary_token_usage_total_equals_sum_of_execution_usage(self) -> None:
        """N results with known execution usage → summary total = arithmetic sum.

        Catches: a regression in ``_aggregate_token_usage`` that miscounts
        across results (e.g. field-by-field int accumulator drifting from
        ``TokenUsage.__add__`` semantics).
        """
        usages = [
            TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15, calls=1),
            TokenUsage(prompt_tokens=20, completion_tokens=8, total_tokens=28, calls=2),
            TokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10, calls=1),
        ]
        results = [_make_result(execution_usage=u) for u in usages]
        expected = sum(usages, start=TokenUsage())

        summary = compute_report_summary(results)

        assert summary.token_usage_total is not None
        assert summary.token_usage_total.prompt_tokens == expected.prompt_tokens
        assert summary.token_usage_total.completion_tokens == expected.completion_tokens
        assert summary.token_usage_total.total_tokens == expected.total_tokens
        assert summary.token_usage_total.calls == expected.calls

    def test_summary_skips_results_without_execution(self) -> None:
        """Results lacking ``execution`` block must not contribute to the total."""
        with_exec = _make_result(
            execution_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15, calls=1),
        )
        without_exec = _make_result(execution_usage=None, has_execution=False)

        summary = compute_report_summary([with_exec, without_exec])

        assert summary.token_usage_total is not None
        assert summary.token_usage_total.calls == 1
        assert summary.token_usage_total.total_tokens == 15

    def test_summary_skips_execution_without_token_usage(self) -> None:
        """Execution present but ``token_usage=None`` must be a no-op."""
        with_usage = _make_result(
            execution_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15, calls=1),
        )
        without_usage = _make_result(execution_usage=None)  # execution exists, usage None

        summary = compute_report_summary([with_usage, without_usage])

        assert summary.token_usage_total is not None
        assert summary.token_usage_total.calls == 1
        assert summary.token_usage_total.total_tokens == 15

    def test_summary_total_none_when_no_usage_anywhere(self) -> None:
        """All results without usage → ``token_usage_total`` is ``None`` (not zero)."""
        results = [_make_result(execution_usage=None) for _ in range(3)]
        summary = compute_report_summary(results)
        assert summary.token_usage_total is None

    def test_summary_includes_evaluator_token_usage(self) -> None:
        """Evaluator tokens must be included in the grand total alongside execution."""
        execution_usage = TokenUsage(
            prompt_tokens=10, completion_tokens=5, total_tokens=15, calls=1
        )
        evaluator_usage = TokenUsage(
            prompt_tokens=4, completion_tokens=2, total_tokens=6, calls=1
        )
        results = [
            _make_result(execution_usage=execution_usage, evaluation_usage=evaluator_usage)
        ]

        summary = compute_report_summary(results)

        assert summary.token_usage_total is not None
        # Bug: only execution_usage flows in. Expected: execution + evaluator.
        assert summary.token_usage_total.calls == 2
        assert summary.token_usage_total.total_tokens == 21
