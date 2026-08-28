"""Seeded persona/scenario generation — the intermediate tier.

Verifies that a short archetype seed is threaded into the generator prompt and
that the public wrappers return one fully-built object per seed. The LLM call is
mocked at the ``generate_structured`` layer so no network/key is needed.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from evaluatorq.common.structured_output import StructuredResult
from evaluatorq.contracts import LLMCallConfig
from evaluatorq.simulation import (
    generate_persona,
    generate_personas,
    generate_scenario,
    generate_scenarios,
)
from evaluatorq.simulation.types import CommunicationStyle, Persona, Scenario


class _Parsed:
    def __init__(self, **kw: object) -> None:
        self.__dict__.update(kw)


def _persona(name: str = "Seeded") -> Persona:
    return Persona(
        name=name,
        patience=0.2,
        assertiveness=0.8,
        politeness=0.3,
        technical_level=0.4,
        background="bg",
        communication_style=CommunicationStyle.terse,
    )


def _scenario(name: str = "Seeded") -> Scenario:
    return Scenario(name=name, goal="goal")


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    """Patch the LLM layer; record each call's user prompt."""
    prompts: dict[str, list[str]] = {"prompts": []}

    async def fake_persona(client, *, messages, **_kw) -> StructuredResult[Any]:  # noqa: ANN001, ANN002
        prompts["prompts"].append(messages[-1]["content"])
        return StructuredResult(cast(Any, _Parsed(personas=[_persona()])), "")

    async def fake_scenario(client, *, messages, **_kw) -> StructuredResult[Any]:  # noqa: ANN001, ANN002
        prompts["prompts"].append(messages[-1]["content"])
        return StructuredResult(cast(Any, _Parsed(scenarios=[_scenario()])), "")

    monkeypatch.setattr(
        "evaluatorq.openresponses.client.build_simulation_client",
        lambda _c=None, **_kw: (object(), False),
    )
    monkeypatch.setattr(
        "evaluatorq.simulation.generators.persona_generator.generate_structured",
        fake_persona,
    )
    monkeypatch.setattr(
        "evaluatorq.simulation.generators.scenario_generator.generate_structured",
        fake_scenario,
    )
    return prompts


@pytest.mark.asyncio
async def test_generate_persona_threads_seed_into_prompt(captured):
    p = await generate_persona("angry customer", agent_description="support agent")
    assert isinstance(p, Persona)
    assert "angry customer" in captured["prompts"][0]


@pytest.mark.asyncio
async def test_generate_scenario_threads_seed_into_prompt(captured):
    s = await generate_scenario("disputes a refund denial")
    assert isinstance(s, Scenario)
    assert "disputes a refund denial" in captured["prompts"][0]


@pytest.mark.asyncio
async def test_batch_returns_one_per_seed(captured):
    ps = await generate_personas(["a", "b", "c"])
    assert len(ps) == 3
    assert len(captured["prompts"]) == 3


@pytest.mark.asyncio
async def test_empty_seeds_raise(captured):
    with pytest.raises(ValueError):
        await generate_personas([])
    with pytest.raises(ValueError):
        await generate_scenarios([])


@pytest.mark.asyncio
async def test_generate_personas_scenarios_seeds_override_num(captured):
    """_generate_personas_scenarios: seeded dimension = one per seed (ignores num_*),
    other dimension still auto-generates."""
    from evaluatorq.simulation.api import _generate_personas_scenarios

    personas, scenarios, _usage = await _generate_personas_scenarios(
        agent_description="support agent",
        num_personas=99,  # ignored — seeds win
        num_scenarios=3,
        llm_config=LLMCallConfig(model="m"),
        generation_client=object(),  # pyright: ignore[reportArgumentType]
        persona_seeds=["angry retiree", "fraud dispute"],
    )
    # One persona per seed, each seed threaded into its prompt.
    assert len(personas) == 2
    joined = " ".join(captured["prompts"])
    assert "angry retiree" in joined
    assert "fraud dispute" in joined
    # Scenarios auto-generated (no seeds) — still produced.
    assert len(scenarios) >= 1


@pytest.mark.asyncio
async def test_generate_personas_scenarios_threads_edge_case_percentage(captured):
    """F6: edge_case_percentage was hardcoded to each generator's own default
    (0.2 / 0.3) with no way for a caller of _generate_personas_scenarios to
    override it. An explicit value must reach the auto-generated (non-seeded)
    prompt for both dimensions."""
    from evaluatorq.simulation.api import _generate_personas_scenarios

    await _generate_personas_scenarios(
        agent_description="support agent",
        num_personas=5,
        num_scenarios=5,
        llm_config=LLMCallConfig(model="m"),
        generation_client=object(),  # pyright: ignore[reportArgumentType]
        edge_case_percentage=0.6,
    )
    joined = " ".join(captured["prompts"])
    # int(5 * 0.6) == 3 edge cases, vs int(5 * 0.2) == 1 / int(5 * 0.3) == 1 at defaults.
    assert "3 edge case" in joined


@pytest.mark.asyncio
async def test_generate_personas_scenarios_default_edge_case_percentage_unchanged(captured):
    """Omitting edge_case_percentage must still fall through to each generator's
    own default (0.2 for personas, 0.3 for scenarios), not some new literal."""
    from evaluatorq.simulation.api import _generate_personas_scenarios

    await _generate_personas_scenarios(
        agent_description="support agent",
        num_personas=5,
        num_scenarios=5,
        llm_config=LLMCallConfig(model="m"),
        generation_client=object(),  # pyright: ignore[reportArgumentType]
    )
    joined = " ".join(captured["prompts"])
    assert "1 edge case" in joined
    assert "3 edge case" not in joined
