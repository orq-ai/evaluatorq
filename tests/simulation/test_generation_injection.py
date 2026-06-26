from unittest.mock import AsyncMock, patch

import pytest

from evaluatorq.simulation.api import generate, generate_and_simulate, simulate
from evaluatorq.simulation.types import CommunicationStyle, Judgment, Persona, Scenario, SimulationDatapoint


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
async def test_generate_accepts_generation_client_without_orq(monkeypatch):
    # SDK generate() runs the same env-free path with an injected client and
    # reaches persona/scenario generation (symmetry with generate_and_simulate).
    monkeypatch.delenv("ORQ_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from openai import AsyncOpenAI

    injected = AsyncOpenAI(api_key="sk-test", base_url="https://example.test/v1")

    with patch(
        "evaluatorq.simulation.generators.PersonaGenerator.generate",
        new=AsyncMock(side_effect=RuntimeError("reached-generation")),
    ):
        with pytest.raises(RuntimeError, match="reached-generation"):
            await generate(
                agent_description="a test agent",
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
    # Pass the old name dynamically so the static type checker doesn't flag the
    # deliberately-invalid kwarg; we assert the *runtime* rejection of `model`.
    bad_kwargs: dict[str, object] = {
        "datapoints": [],
        "target": lambda messages: "ok",
        "model": "x",
    }
    with pytest.raises(TypeError):
        await simulate(**bad_kwargs)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]


def _make_datapoint(dp_id: str = "dp-0") -> SimulationDatapoint:
    return SimulationDatapoint(
        id=dp_id,
        persona=_persona(),
        scenario=_scenario(),
        user_system_prompt="You are a user.",
        first_message="Hello",
    )


@pytest.mark.asyncio
async def test_generate_and_simulate_emit_datapoints_called_once(monkeypatch):
    """emit_datapoints is invoked exactly once with the generated list[SimulationDatapoint]."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    fake_datapoints = [_make_datapoint("dp-0"), _make_datapoint("dp-1")]

    emitted: list[list[SimulationDatapoint]] = []

    def sink(dps: list[SimulationDatapoint]) -> None:
        emitted.append(dps)

    with (
        patch(
            "evaluatorq.simulation.api._generate_personas_scenarios",
            new=AsyncMock(return_value=([_persona()], [_scenario()])),
        ),
        patch(
            "evaluatorq.simulation.api._resolve_or_generate_datapoints",
            new=AsyncMock(return_value=fake_datapoints),
        ),
        patch(
            "evaluatorq.simulation.api._simulate_core",
            new=AsyncMock(return_value=[]),
        ),
    ):
        await generate_and_simulate(
            agent_description="test agent",
            target=lambda messages: "ok",
            emit_datapoints=sink,
        )

    assert len(emitted) == 1, f"sink called {len(emitted)} times (expected 1)"
    assert emitted[0] is fake_datapoints
    assert isinstance(emitted[0], list)
    assert all(isinstance(dp, SimulationDatapoint) for dp in emitted[0])


@pytest.mark.asyncio
async def test_simulate_uses_generation_client_for_default_user_and_judge(monkeypatch):
    monkeypatch.delenv("ORQ_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from openai import AsyncOpenAI

    injected = AsyncOpenAI(api_key="sk-test", base_url="https://example.test/v1")

    async def fake_user_response(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        assert self._client is injected
        return "next user message"

    async def fake_judge_evaluate(self, messages):  # noqa: ANN001
        assert self._client is injected
        return Judgment(
            should_terminate=True,
            reason="done",
            goal_achieved=True,
            rules_broken=[],
            goal_completion_score=1.0,
        )

    def fake_build_simulation_client(config_client=None, **kwargs):  # noqa: ANN001, ANN003
        if config_client is not injected:
            raise RuntimeError("runner built its own client")
        return injected, False

    with (
        patch(
            "evaluatorq.openresponses.client.build_simulation_client",
            side_effect=fake_build_simulation_client,
        ),
        patch(
            "evaluatorq.simulation.agents.user_simulator.UserSimulatorAgent.respond_async",
            new=fake_user_response,
        ),
        patch(
            "evaluatorq.simulation.agents.judge.JudgeAgent.evaluate",
            new=fake_judge_evaluate,
        ),
    ):
        results = await simulate(
            datapoints=[_make_datapoint()],
            target=lambda messages: "target reply",
            generation_client=injected,
            upload_results=False,
        )

    assert results[0].goal_achieved is True
