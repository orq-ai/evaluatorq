"""Generated (dynamic) strategies must always be planned as multi-turn.

A single-turn generated attack gets one shot with no chance to adapt, so the
planner forces ``TurnType.MULTI``; the orchestrator still terminates early once
the objective is achieved.
"""

from typing import Any, cast

import pytest

from evaluatorq.redteam.contracts import AgentContext, MemoryStoreInfo, ToolInfo, TurnType, Vulnerability


@pytest.mark.asyncio
async def test_planner_forces_multi_turn_for_generated_strategies(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.redteam.adaptive import strategy_planner

    captured: dict[str, Any] = {}

    async def _fake_generate(**kwargs: Any) -> list[Any]:
        captured.update(kwargs)
        return []

    monkeypatch.setattr(strategy_planner, 'generate_strategies_for_vulnerability', _fake_generate)

    await strategy_planner.plan_strategies_for_vulnerabilities(
        agent_context=AgentContext(
            key='test-agent',
            tools=[ToolInfo(name='search')],
            memory_stores=[MemoryStoreInfo(id='ms_001', key='history')],
        ),
        vulnerabilities=[Vulnerability.GOAL_HIJACKING],
        llm_client=cast(Any, object()),  # unused: generation is stubbed
        attack_model='test-model',
        max_turns=5,
        max_per_category=None,
        generate_additional_strategies=True,
        generated_strategy_count=2,
    )

    assert captured['turn_type'] == TurnType.MULTI
