"""RES-907: framework integration targets are decoupled from redteam and
exposed via evaluatorq.simulation."""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "evaluatorq.integrations.callable_integration.target",
        "evaluatorq.integrations.crewai_integration.target",
        "evaluatorq.integrations.langgraph_integration.target",
        "evaluatorq.integrations.openai_agents_integration.target",
        "evaluatorq.integrations.pydantic_ai_integration.target",
        "evaluatorq.integrations.vercel_ai_sdk_integration.target",
    ],
)
def test_importing_integration_target_does_not_import_redteam(module: str) -> None:
    """No exposed integration target may drag in the whole redteam package.

    Checked per-module because each target has its own independent import block —
    a redteam import re-added to any one of them must fail the suite, not just
    the one that happens to be exercised.
    """
    code = (
        f"import {module}\n"
        "import sys\n"
        "leaked = sorted(m for m in sys.modules if m.startswith('evaluatorq.redteam'))\n"
        "assert not leaked, leaked\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,  # a hung optional-SDK import must fail this test, not stall the suite
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize(
    "submodule",
    [
        "callable_integration",
        "crewai_integration",
        "langchain_integration",
        "langgraph_integration",
        "openai_agents_integration",
        "pydantic_ai_integration",
        "vercel_ai_sdk_integration",
    ],
)
def test_from_integrations_import_submodule(submodule: str) -> None:
    """`from evaluatorq.integrations import X` must not recurse.

    The lazy loader used to do `from . import X` inside `__getattr__`; the
    import machinery answers that by calling getattr on the package, which
    re-enters `__getattr__` — RecursionError for every sub-module on 3.13.
    Run in a subprocess so the sub-module is genuinely not yet imported;
    in-process the attribute may already be bound by an earlier test.
    """
    proc = subprocess.run(
        [sys.executable, "-c", f"from evaluatorq.integrations import {submodule}"],
        capture_output=True,
        text=True,
        timeout=120,  # RecursionError is fast, but a hung import must not stall the suite
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize(
    ("name", "canonical"),
    [
        ("OpenAIAgentTarget", "evaluatorq.integrations.openai_agents_integration"),
        ("LangGraphTarget", "evaluatorq.integrations.langgraph_integration"),
        ("VercelAISdkTarget", "evaluatorq.integrations.vercel_ai_sdk_integration"),
        ("CallableTarget", "evaluatorq.integrations.callable_integration"),
        ("CrewAITarget", "evaluatorq.integrations.crewai_integration"),
        ("PydanticAITarget", "evaluatorq.integrations.pydantic_ai_integration"),
    ],
)
def test_framework_target_exposed_from_simulation(name: str, canonical: str) -> None:
    """Identity, not just presence: a lazy mapping that resolved `CrewAITarget`
    to `CallableTarget` would satisfy `hasattr` while handing callers the wrong
    class."""
    import importlib

    import evaluatorq.simulation as sim

    assert hasattr(sim, name), f"{name} not exposed from evaluatorq.simulation"
    assert name in sim.__all__
    assert getattr(sim, name) is getattr(importlib.import_module(canonical), name)


@pytest.mark.parametrize(
    ("name", "extra"),
    [
        ("LangGraphTarget", "langgraph"),
        ("OpenAIAgentTarget", "openai-agents"),
    ],
)
def test_missing_optional_dep_gives_actionable_error(
    name: str, extra: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing integration extra surfaces as an install hint, not a raw
    ImportError about the third-party package."""
    import evaluatorq.simulation as sim

    def _boom(_module: str) -> object:
        raise ModuleNotFoundError(f"No module named '{extra}'")

    monkeypatch.setattr(sim.importlib, "import_module", _boom)
    with pytest.raises(ImportError, match=rf"evaluatorq\[{extra}\]"):
        sim.__getattr__(name)


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
        Persona,
        Scenario,
        SimulationDatapoint,
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
    dp = SimulationDatapoint(
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
