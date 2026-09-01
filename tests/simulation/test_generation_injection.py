from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evaluatorq.contracts import AgentContext
from evaluatorq.simulation.api import generate, generate_and_simulate, simulate
from evaluatorq.simulation.types import CommunicationStyle, Judgment, Persona, Scenario, SimulationDatapoint
from evaluatorq.contracts import LLMCallConfig


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
@pytest.mark.parametrize("target", ["agent:support-agent", "support-agent"])
async def test_generate_and_simulate_uses_orq_agent_description_when_omitted(monkeypatch, target):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    backend = MagicMock()
    backend.resolve_context = AsyncMock(
        return_value=AgentContext(key="support-agent", description="Handles customer refunds")
    )
    generated_descriptions: list[object] = []

    def capture_generation(**kwargs: object) -> tuple[list[Persona], list[Scenario], None]:
        generated_descriptions.append(kwargs["agent_description"])
        return [_persona()], [_scenario()], None

    with (
        patch(
            "evaluatorq.redteam.backends.registry.make_agent_backend",
            return_value=backend,
        ),
        patch(
            "evaluatorq.simulation.api._generate_personas_scenarios",
            new=AsyncMock(side_effect=capture_generation),
        ),
        patch(
            "evaluatorq.simulation.api._resolve_or_generate_datapoints",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "evaluatorq.simulation.api._simulate_core",
            new=AsyncMock(return_value=MagicMock(results=[])),
        ),
    ):
        await generate_and_simulate(target=target)

    assert generated_descriptions == ["Handles customer refunds"]
    backend.resolve_context.assert_awaited_once_with("support-agent")


@pytest.mark.asyncio
async def test_generate_and_simulate_prefers_an_explicit_description(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    generated_descriptions: list[object] = []

    def capture_generation(**kwargs: object) -> tuple[list[Persona], list[Scenario], None]:
        generated_descriptions.append(kwargs["agent_description"])
        return [_persona()], [_scenario()], None

    with (
        patch("evaluatorq.redteam.backends.registry.make_agent_backend") as make_backend,
        patch(
            "evaluatorq.simulation.api._generate_personas_scenarios",
            new=AsyncMock(side_effect=capture_generation),
        ),
        patch(
            "evaluatorq.simulation.api._resolve_or_generate_datapoints",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "evaluatorq.simulation.api._simulate_core",
            new=AsyncMock(return_value=MagicMock(results=[])),
        ),
    ):
        await generate_and_simulate(
            agent_description="Explicit description",
            target="agent:support-agent",
        )

    assert generated_descriptions == ["Explicit description"]
    make_backend.assert_not_called()


@pytest.mark.asyncio
async def test_generate_and_simulate_requires_an_explicit_or_target_description():
    with pytest.raises(ValueError, match=r"agent_description.*description"):
        await generate_and_simulate(target=lambda messages: "ok")


@pytest.mark.asyncio
async def test_generate_and_simulate_rejects_orq_agents_without_a_description():
    backend = MagicMock()
    backend.resolve_context = AsyncMock(return_value=AgentContext(key="support-agent"))

    with (
        patch("evaluatorq.redteam.backends.registry.make_agent_backend", return_value=backend),
        pytest.raises(ValueError, match=r"agent_description.*description"),
    ):
        await generate_and_simulate(target="agent:support-agent")


@pytest.mark.asyncio
async def test_generate_and_simulate_rejects_target_callback(monkeypatch):
    monkeypatch.delenv("ORQ_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from openai import AsyncOpenAI

    injected = AsyncOpenAI(api_key="sk-test", base_url="https://example.test/v1")
    with pytest.raises(TypeError, match="target_callback"):
        await generate_and_simulate(
            agent_description="a test agent",
            target_callback=lambda messages: "ok",  # pyright: ignore[reportCallIssue]
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
            new=AsyncMock(return_value=([_persona()], [_scenario()], None)),
        ),
        patch(
            "evaluatorq.simulation.api._resolve_or_generate_datapoints",
            new=AsyncMock(return_value=fake_datapoints),
        ),
        patch(
            "evaluatorq.simulation.api._simulate_core",
            new=AsyncMock(return_value=MagicMock(results=[])),
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
        assert self._client is not injected
        assert self._client.max_retries == 0
        return "next user message"

    async def fake_judge_evaluate(self, messages):  # noqa: ANN001
        assert self._client is not injected
        assert self._client.max_retries == 0
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
        return injected.with_options(max_retries=0), False

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
            # Off because it is incidental here: the summary step resolves its
            # own client rather than using the injected one, so leaving it on
            # makes this test reach the live router.
            executive_summary=False,
        )

    assert results[0].goal_achieved


# ---------------------------------------------------------------------------
# Behavioral regression tests: sim_model -> config.model, and owned
# generation-client close semantics through generate_and_simulate().
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sim_model_propagates_to_config_model(monkeypatch):
    """sim_model flows through to SimulationConfig.model at the innermost seam.

    A broken `model=sim_model` mapping in _generate_and_simulate_run (e.g.
    reverting to DEFAULT_MODEL) makes this assertion fail.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    fake_datapoints = [_make_datapoint("dp-0")]
    captured: dict[str, object] = {}

    def capture_simulate_via_evaluatorq(*, config, **kwargs):  # noqa: ANN003
        captured["model"] = config.model
        return []

    with (
        patch(
            "evaluatorq.simulation.api._generate_datapoints_inner",
            new=AsyncMock(return_value=(fake_datapoints, MagicMock(), False, None)),
        ),
        patch(
            "evaluatorq.simulation.api._simulate_via_evaluatorq",
            new=AsyncMock(side_effect=capture_simulate_via_evaluatorq),
        ),
    ):
        await generate_and_simulate(
            agent_description="test agent",
            target=lambda messages: "ok",
            llm_config=LLMCallConfig(model="distinct-model-xyz"),
        )

    assert captured["model"] == "distinct-model-xyz"


@pytest.mark.asyncio
async def test_generate_and_simulate_closes_owned_generation_client_once(monkeypatch):
    """An internally-built (owned) generation client is closed exactly once on success."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    mock_client = MagicMock()
    mock_client.close = AsyncMock()

    with (
        patch(
            "evaluatorq.openresponses.client.build_simulation_client",
            return_value=(mock_client, True),
        ),
        patch(
            "evaluatorq.simulation.api._generate_personas_scenarios",
            new=AsyncMock(return_value=([_persona()], [_scenario()], None)),
        ),
        patch(
            "evaluatorq.simulation.api._resolve_or_generate_datapoints",
            new=AsyncMock(return_value=[_make_datapoint()]),
        ),
        patch(
            "evaluatorq.simulation.api._simulate_via_evaluatorq",
            new=AsyncMock(return_value=[]),
        ),
    ):
        await generate_and_simulate(
            agent_description="test agent",
            target=lambda messages: "ok",
        )

    mock_client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_and_simulate_does_not_close_injected_generation_client(monkeypatch):
    """An injected (not owned) generation client is never closed by the pipeline."""
    monkeypatch.delenv("ORQ_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from openai import AsyncOpenAI

    injected = AsyncOpenAI(api_key="sk-test", base_url="https://example.test/v1")
    injected.close = AsyncMock()  # type: ignore[method-assign]

    with (
        patch(
            "evaluatorq.simulation.api._generate_personas_scenarios",
            new=AsyncMock(return_value=([_persona()], [_scenario()], None)),
        ),
        patch(
            "evaluatorq.simulation.api._resolve_or_generate_datapoints",
            new=AsyncMock(return_value=[_make_datapoint()]),
        ),
        patch(
            "evaluatorq.simulation.api._simulate_via_evaluatorq",
            new=AsyncMock(return_value=[]),
        ),
    ):
        await generate_and_simulate(
            agent_description="test agent",
            target=lambda messages: "ok",
            generation_client=injected,
        )

    injected.close.assert_not_called()


@pytest.mark.asyncio
async def test_generate_and_simulate_closes_owned_client_when_emit_datapoints_raises(monkeypatch):
    """Regression: emit_datapoints raising must still close the owned generation client.

    Before the fix, the close call lived in a try block the exception from
    emit_datapoints skipped over, leaking the client.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    mock_client = MagicMock()
    mock_client.close = AsyncMock()

    def failing_sink(dps: list[SimulationDatapoint]) -> None:
        raise RuntimeError("sink exploded")

    with (
        patch(
            "evaluatorq.openresponses.client.build_simulation_client",
            return_value=(mock_client, True),
        ),
        patch(
            "evaluatorq.simulation.api._generate_personas_scenarios",
            new=AsyncMock(return_value=([_persona()], [_scenario()], None)),
        ),
        patch(
            "evaluatorq.simulation.api._resolve_or_generate_datapoints",
            new=AsyncMock(return_value=[_make_datapoint()]),
        ),
        pytest.raises(RuntimeError, match="sink exploded"),
    ):
        await generate_and_simulate(
            agent_description="test agent",
            target=lambda messages: "ok",
            emit_datapoints=failing_sink,
        )

    mock_client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_and_simulate_threads_seeds_and_edge_case_percentage(monkeypatch):
    """F6: generate_and_simulate() previously had no persona_seeds/scenario_seeds/
    edge_case_percentage parameters at all -- verify they now reach
    _generate_datapoints_inner (the shared generation seam)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    fake_datapoints = [_make_datapoint("dp-0")]
    captured: dict[str, object] = {}

    async def capture_generate_datapoints_inner(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return fake_datapoints, MagicMock(), False, None

    with (
        patch(
            "evaluatorq.simulation.api._generate_datapoints_inner",
            new=capture_generate_datapoints_inner,
        ),
        patch(
            "evaluatorq.simulation.api._simulate_via_evaluatorq",
            new=AsyncMock(return_value=[]),
        ),
    ):
        await generate_and_simulate(
            agent_description="test agent",
            target=lambda messages: "ok",
            persona_seeds=["angry retiree"],
            scenario_seeds=["refund dispute"],
            edge_case_percentage=0.5,
        )

    assert captured["persona_seeds"] == ["angry retiree"]
    assert captured["scenario_seeds"] == ["refund dispute"]
    assert captured["edge_case_percentage"] == 0.5


@pytest.mark.asyncio
async def test_generate_and_simulate_threads_target_agent_knobs(monkeypatch):
    """F3/F5: target_agent_timeout_ms / max_target_retries / target_reasoning_effort
    must reach SimulationConfig, not stay pinned at SimulationRunner's literals."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    fake_datapoints = [_make_datapoint("dp-0")]
    captured: dict[str, object] = {}

    def capture_simulate_via_evaluatorq(*, config, **kwargs):  # noqa: ANN003
        captured["target_agent_timeout_ms"] = config.target_agent_timeout_ms
        captured["max_target_retries"] = config.max_target_retries
        captured["target_reasoning_effort"] = config.target_reasoning_effort
        return []

    with (
        patch(
            "evaluatorq.simulation.api._generate_datapoints_inner",
            new=AsyncMock(return_value=(fake_datapoints, MagicMock(), False, None)),
        ),
        patch(
            "evaluatorq.simulation.api._simulate_via_evaluatorq",
            new=AsyncMock(side_effect=capture_simulate_via_evaluatorq),
        ),
    ):
        await generate_and_simulate(
            agent_description="test agent",
            target=lambda messages: "ok",
            target_agent_timeout_ms=999_000,
            max_target_retries=9,
            target_reasoning_effort="low",
        )

    assert captured["target_agent_timeout_ms"] == 999_000
    assert captured["max_target_retries"] == 9
    assert captured["target_reasoning_effort"] == "low"


@pytest.mark.asyncio
async def test_simulate_threads_target_agent_knobs(monkeypatch):
    """F3/F5, simulate() path: same knobs must reach SimulationConfig here too."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    captured: dict[str, object] = {}

    def capture_simulate_via_evaluatorq(*, config, **kwargs):  # noqa: ANN003
        captured["target_agent_timeout_ms"] = config.target_agent_timeout_ms
        captured["max_target_retries"] = config.max_target_retries
        captured["target_reasoning_effort"] = config.target_reasoning_effort
        return []

    with patch(
        "evaluatorq.simulation.api._simulate_via_evaluatorq",
        new=AsyncMock(side_effect=capture_simulate_via_evaluatorq),
    ):
        await simulate(
            datapoints=[_make_datapoint("dp-0")],
            target=lambda messages: "ok",
            target_agent_timeout_ms=999_000,
            max_target_retries=9,
            target_reasoning_effort="low",
        )

    assert captured["target_agent_timeout_ms"] == 999_000
    assert captured["max_target_retries"] == 9
    assert captured["target_reasoning_effort"] == "low"
