"""A failed strategy generation is recorded, and stops the run when it leaves a vulnerability untested.

The objective generator used to swallow its own failures and return no objectives, so the
planner never recorded a ``generation_error`` and a vulnerability could run with fewer
strategies, or none, with no warning that said why (RES-1963).
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

from evaluatorq.redteam.adaptive import strategy_planner
from evaluatorq.redteam.adaptive.objective_generator import (
    GeneratedObjective,
    GeneratedObjectives,
    generate_objectives_for_vulnerability,
)
from evaluatorq.redteam.contracts import (
    AgentContext,
    AttackStrategy,
    DeliveryMethod,
    LLMConfig,
    TurnType,
    Vulnerability,
)
from evaluatorq.redteam.exceptions import RedTeamError, StrategyGenerationError

_MODEL = 'azure/gpt-4.1'


async def _plan(*, monkeypatch: pytest.MonkeyPatch, hardcoded: list[AttackStrategy]) -> tuple[Any, Any]:
    async def _blocked(**kwargs: Any) -> list[AttackStrategy]:
        msg = f'the content filter of attacker model {kwargs["model"]!r} blocked objective generation on all 3 attempt(s)'
        raise StrategyGenerationError(msg)

    monkeypatch.setattr(strategy_planner, 'generate_strategies_for_vulnerability', _blocked)
    monkeypatch.setattr(strategy_planner, 'select_applicable_strategies_for_vulnerability', lambda **_: hardcoded)
    strategies, selection, _ = await strategy_planner.plan_strategies_for_vulnerabilities(
        agent_context=AgentContext(key='test-agent'),
        vulnerabilities=[Vulnerability.GOAL_HIJACKING],
        llm_client=cast(Any, object()),  # unused: generation is stubbed
        attack_model=_MODEL,
        max_turns=5,
        max_per_category=None,
        generate_additional_strategies=True,
        generated_strategy_count=2,
    )
    return strategies, selection


@pytest.mark.asyncio
async def test_generation_failure_with_hardcoded_strategies_is_recorded_and_the_run_goes_on(monkeypatch):
    hardcoded = strategy_planner.get_strategies_for_vulnerability(Vulnerability.GOAL_HIJACKING)[:1]
    assert hardcoded, 'goal_hijacking ships curated strategies'

    strategies, selection = await _plan(monkeypatch=monkeypatch, hardcoded=hardcoded)

    assert strategies[Vulnerability.GOAL_HIJACKING] == hardcoded
    meta = selection[Vulnerability.GOAL_HIJACKING]
    assert meta['generated_count'] == 0
    # The runner's end-of-run warning reads this field; it must name the blocked model.
    assert _MODEL in meta['generation_error']


@pytest.mark.asyncio
async def test_generation_failure_that_leaves_no_strategy_stops_the_run(monkeypatch):
    with pytest.raises(StrategyGenerationError) as info:
        await _plan(monkeypatch=monkeypatch, hardcoded=[])

    message = str(info.value)
    assert 'goal_hijacking' in message
    assert _MODEL in message
    assert 'before any attack was sent' in message
    # Callers catching the package's base error (the CLI does) see it too.
    assert isinstance(info.value, RedTeamError)


@pytest.mark.asyncio
async def test_no_applicable_strategies_without_a_generation_error_does_not_raise(monkeypatch):
    """Zero strategies because nothing applies to the agent is not a failure of generation."""
    monkeypatch.setattr(strategy_planner, 'generate_strategies_for_vulnerability', AsyncMock(return_value=[]))
    monkeypatch.setattr(strategy_planner, 'select_applicable_strategies_for_vulnerability', lambda **_: [])

    strategies, selection, _ = await strategy_planner.plan_strategies_for_vulnerabilities(
        agent_context=AgentContext(key='test-agent'),
        vulnerabilities=[Vulnerability.GOAL_HIJACKING],
        llm_client=cast(Any, object()),
        attack_model=_MODEL,
        max_turns=5,
        max_per_category=None,
        generate_additional_strategies=True,
        generated_strategy_count=2,
    )

    assert strategies[Vulnerability.GOAL_HIJACKING] == []
    assert selection[Vulnerability.GOAL_HIJACKING]['generation_error'] is None


def _response() -> MagicMock:
    choice = MagicMock()
    choice.message.parsed = GeneratedObjectives(
        objectives=[
            GeneratedObjective(
                objective='Redirect the agent goal via injected content',
                turn_type=TurnType.MULTI,
                delivery_method=DeliveryMethod.DIRECT_REQUEST,
                requires_tools=False,
                requires_memory=False,
            )
        ]
    )
    choice.finish_reason = 'stop'
    response = MagicMock()
    response.choices = [choice]
    response.usage = None
    return response


@pytest.mark.asyncio
async def test_a_later_batch_failure_keeps_the_objectives_of_earlier_batches():
    client = MagicMock()
    client.base_url = 'https://api.openai.com/v1'
    # Batch 1 succeeds; batch 2 is filtered on every attempt.
    client.chat.completions.parse = AsyncMock(side_effect=[_response(), *[openai.ContentFilterFinishReasonError()] * 3])

    with patch('evaluatorq.redteam.adaptive.objective_generator.with_llm_span') as span:
        span.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
        span.return_value.__aexit__ = AsyncMock(return_value=None)
        objectives = await generate_objectives_for_vulnerability(
            vuln=Vulnerability.GOAL_HIJACKING,
            agent_context=AgentContext(key='test-agent'),
            llm_client=client,
            model=_MODEL,
            count=2,
            pipeline_config=LLMConfig(max_objectives_per_llm_call=1, max_content_filter_retries=2),
        )

    assert len(objectives) == 1
    assert client.chat.completions.parse.call_count == 4
