from unittest.mock import AsyncMock, patch

import pytest

from evaluatorq.simulation.api import generate_and_simulate, simulate
from evaluatorq.simulation.types import CommunicationStyle, Persona, Scenario


def _persona() -> Persona:
    return Persona(
        name="p",
        patience=0.5,
        assertiveness=0.5,
        politeness=0.5,
        technical_level=0.5,
        communication_style=CommunicationStyle.casual,
        background="b",
    )


def _scenario() -> Scenario:
    return Scenario(name="s", goal="g")


@pytest.mark.asyncio
async def test_generate_and_simulate_accepts_generation_client_without_orq(monkeypatch):
    monkeypatch.delenv("ORQ_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from openai import AsyncOpenAI

    injected = AsyncOpenAI(api_key="sk-test", base_url="https://example.test/v1")

    with patch(
        "evaluatorq.simulation.generators.PersonaGenerator.generate",
        new=AsyncMock(side_effect=RuntimeError("reached-generation")),
    ):
        with pytest.raises(RuntimeError, match="reached-generation"):
            await generate_and_simulate(
                agent_description="a test agent",
                target=lambda messages: "ok",
                num_personas=1,
                num_scenarios=1,
                generation_client=injected,
            )


@pytest.mark.asyncio
async def test_simulate_first_message_uses_generation_client_without_orq(monkeypatch):
    monkeypatch.delenv("ORQ_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from openai import AsyncOpenAI

    injected = AsyncOpenAI(api_key="sk-test", base_url="https://example.test/v1")

    mock_gen = AsyncMock(side_effect=RuntimeError("reached-first-message"))
    with patch(
        "evaluatorq.simulation.generators.FirstMessageGenerator.generate",
        new=mock_gen,
    ):
        # The batch loop swallows per-pair generation failures and raises its
        # own "produced no datapoints" RuntimeError. Reaching that path (and
        # the mock being called) proves first-message generation ran via the
        # injected client without an ORQ key.
        with pytest.raises(RuntimeError, match="produced no datapoints"):
            await simulate(
                personas=[_persona()],
                scenarios=[_scenario()],
                target=lambda messages: "ok",
                generation_client=injected,
            )
    mock_gen.assert_awaited()


@pytest.mark.asyncio
async def test_sim_model_is_the_public_param(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with pytest.raises(TypeError):
        await simulate(
            datapoints=[],
            target=lambda messages: "ok",
            model="x",  # old name removed -> unexpected kwarg
        )
