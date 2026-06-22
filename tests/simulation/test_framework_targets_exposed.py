"""RES-907: framework integration targets are decoupled from redteam and
exposed via evaluatorq.simulation."""

from __future__ import annotations

import subprocess
import sys

import pytest


def test_importing_callable_target_does_not_import_redteam() -> None:
    """An integration target must not drag in the whole redteam package."""
    code = (
        "import evaluatorq.integrations.callable_integration.target\n"
        "import sys\n"
        "leaked = sorted(m for m in sys.modules if m.startswith('evaluatorq.redteam'))\n"
        "assert not leaked, leaked\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize(
    "name",
    ["OpenAIAgentTarget", "LangGraphTarget", "VercelAISdkTarget", "CallableTarget"],
)
def test_framework_target_exposed_from_simulation(name: str) -> None:
    import evaluatorq.simulation as sim

    assert hasattr(sim, name), f"{name} not exposed from evaluatorq.simulation"
    assert name in sim.__all__


@pytest.mark.asyncio
async def test_integration_target_reports_token_usage() -> None:
    """A simulation driven by an integration target (the target_agent path, not
    the str-callback path) aggregates the target's token usage into the result."""
    from unittest.mock import AsyncMock, MagicMock

    from evaluatorq.contracts import Message as ContractMessage
    from evaluatorq.contracts import TokenUsage
    from evaluatorq.integrations.callable_integration import CallableTarget
    from evaluatorq.simulation.runner.simulation import SimulationRunner
    from evaluatorq.simulation.types import (
        CommunicationStyle,
        Datapoint,
        Persona,
        Scenario,
    )

    def agent(messages: list[ContractMessage]) -> str:
        return "agent reply"

    def usage_fn(messages: list[ContractMessage], response: str) -> TokenUsage:
        return TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15, calls=1)

    target = CallableTarget(agent, usage_fn=usage_fn)

    # User-simulator and judge contribute zero usage, so any non-zero total comes
    # from the integration target — proving the target_agent path is exercised.
    sim = MagicMock()
    sim.generate_first_message = AsyncMock(return_value="hello")
    sim.respond_async = AsyncMock(return_value="thanks")
    sim.get_usage = MagicMock(return_value=TokenUsage())

    judgment = MagicMock()
    judgment.should_terminate = True
    judgment.goal_achieved = True
    judgment.goal_completion_score = 1.0
    judgment.rules_broken = []
    judgment.reason = "done"
    judgment.response_quality = 0.9
    judgment.hallucination_risk = 0.1
    judgment.tone_appropriateness = 0.9
    judgment.factual_accuracy = 0.9
    judge = MagicMock()
    judge.evaluate = AsyncMock(return_value=judgment)
    judge.get_usage = MagicMock(return_value=TokenUsage())

    runner = SimulationRunner(
        target_agent=target,
        model="test",
        max_turns=1,
        user_simulator=sim,
        judge=judge,
    )
    dp = Datapoint(
        id="dp-1",
        persona=Persona(
            name="P",
            patience=0.5,
            assertiveness=0.5,
            politeness=0.5,
            technical_level=0.5,
            communication_style=CommunicationStyle.casual,
            background="b",
        ),
        scenario=Scenario(name="S", goal="g"),
        user_system_prompt="sys",
        first_message="hi",
    )
    result = await runner.run(datapoint=dp)

    assert result.token_usage.total_tokens >= 15
