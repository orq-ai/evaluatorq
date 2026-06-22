"""Seeded persona/scenario generation — the intermediate tier.

Verifies that a short archetype seed is threaded into the generator prompt and
that the public wrappers return one fully-built object per seed. The LLM call is
mocked at the ``generate_structured`` layer so no network/key is needed.
"""

from __future__ import annotations

import pytest

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

    async def fake_persona(client, *, messages, **_kw):  # noqa: ANN001, ANN002
        prompts["prompts"].append(messages[-1]["content"])
        return _Parsed(personas=[_persona()]), None

    async def fake_scenario(client, *, messages, **_kw):  # noqa: ANN001, ANN002
        prompts["prompts"].append(messages[-1]["content"])
        return _Parsed(scenarios=[_scenario()]), None

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
