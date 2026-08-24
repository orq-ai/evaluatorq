"""``llm_config`` reaches every simulation-side LLM call.

The temperature default was unset repo-wide (`LLMCallConfig.temperature` is
``None``), which removed the only way a caller could pin sampling on the
user simulator, the judge or the generators. These tests hold the replacement
surface: one config in at the entry point, honoured at each call site.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from evaluatorq.contracts import LLMCallConfig
from evaluatorq.simulation.agents.base import AgentConfig
from evaluatorq.simulation.agents.judge import JudgeAgentConfig
from evaluatorq.simulation.agents.user_simulator import UserSimulatorAgentConfig
from evaluatorq.simulation._config import resolve_sim_llm_config
from evaluatorq.simulation.generators import (
    FirstMessageGenerator,
    PersonaGenerator,
    ScenarioGenerator,
)
from evaluatorq.simulation.runner.simulation import SimulationRunner
from evaluatorq.simulation.types import DEFAULT_MODEL, CommunicationStyle, Persona, Scenario


def _persona() -> Persona:
    return Persona(
        name='Test User',
        patience=0.5,
        assertiveness=0.5,
        politeness=0.5,
        technical_level=0.5,
        communication_style=CommunicationStyle.casual,
        background='bg',
    )


def test_sim_model_alone_sets_only_the_model() -> None:
    resolved = resolve_sim_llm_config(sim_model='openai/gpt-4o-mini', llm_config=None)
    assert resolved.model == 'openai/gpt-4o-mini'
    assert resolved.model_fields_set == {'model'}


def test_llm_config_wins_over_sim_model(caplog: pytest.LogCaptureFixture) -> None:
    cfg = LLMCallConfig(model='openai/gpt-4o', temperature=0.3)
    with caplog.at_level('WARNING'):
        resolved = resolve_sim_llm_config(sim_model='openai/gpt-4o-mini', llm_config=cfg)
    assert resolved is cfg
    assert 'contradicts' in caplog.text


def test_no_warning_when_sim_model_is_untouched(caplog: pytest.LogCaptureFixture) -> None:
    cfg = LLMCallConfig(model='openai/gpt-4o')
    with caplog.at_level('WARNING'):
        assert resolve_sim_llm_config(sim_model=DEFAULT_MODEL, llm_config=cfg) is cfg
    assert caplog.text == ''


@pytest.mark.parametrize('cls', [AgentConfig, UserSimulatorAgentConfig, JudgeAgentConfig])
def test_from_call_config_carries_only_set_fields(cls: type[AgentConfig]) -> None:
    cfg = LLMCallConfig(model='openai/gpt-4o', temperature=0.25)
    agent_cfg = cls.from_call_config(cfg)
    assert agent_cfg.model == 'openai/gpt-4o'
    assert agent_cfg.temperature == 0.25
    # Untouched on the LLMCallConfig, so they stay None here and the per-call-site
    # literal keeps applying — copying the pydantic defaults would shadow it.
    assert agent_cfg.max_tokens is None
    assert agent_cfg.reasoning_effort is None


def test_from_call_config_round_trips_through_base_agent() -> None:
    from evaluatorq.simulation.agents.base import _config_from_agent_config

    cfg = LLMCallConfig(model='openai/gpt-4o', temperature=0.25, reasoning_effort='high')
    back, _api_key = _config_from_agent_config(AgentConfig.from_call_config(cfg))
    assert back.model_fields_set == {'model', 'client', 'api', 'temperature', 'reasoning_effort'}
    assert back.temperature == 0.25
    assert back.reasoning_effort == 'high'


def test_from_call_config_takes_subclass_overrides() -> None:
    judge_cfg = JudgeAgentConfig.from_call_config(LLMCallConfig(model='m'), goal='refund the order')
    assert judge_cfg.goal == 'refund the order'
    assert judge_cfg.api == 'responses'


def test_runner_builds_its_agents_from_the_config() -> None:
    cfg = LLMCallConfig(model='openai/gpt-4o', temperature=0.4)
    runner = SimulationRunner(target=lambda _messages: 'hi', llm_config=cfg)
    assert runner._model == 'openai/gpt-4o'
    assert runner._llm_config is cfg


def test_runner_model_shorthand_still_works() -> None:
    runner = SimulationRunner(target=lambda _messages: 'hi', model='openai/gpt-4o-mini')
    assert runner._llm_config.model == 'openai/gpt-4o-mini'
    assert runner._llm_config.temperature is None


@pytest.mark.parametrize('cls', [PersonaGenerator, ScenarioGenerator, FirstMessageGenerator])
def test_generators_take_a_config(cls: Any) -> None:
    cfg = LLMCallConfig(model='openai/gpt-4o', temperature=0.9)
    gen = cls(client=MagicMock(), config=cfg)
    assert gen._config is cfg
    assert gen._model == 'openai/gpt-4o'


@pytest.mark.asyncio
async def test_persona_generator_forwards_the_config(monkeypatch: pytest.MonkeyPatch) -> None:
    import evaluatorq.simulation.generators.persona_generator as mod

    seen: dict[str, Any] = {}

    async def fake(*_args: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return MagicMock(parsed=mod.PersonaListResponse(personas=[]), raw='[]', usage=None)

    monkeypatch.setattr(mod, 'generate_structured', fake)
    cfg = LLMCallConfig(model='openai/gpt-4o', temperature=0.9)
    await PersonaGenerator(client=MagicMock(), config=cfg).generate(agent_description='a bot', num_personas=1)
    assert seen['config'] is cfg


@pytest.mark.asyncio
async def test_first_message_generator_sends_the_configured_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    import evaluatorq.simulation.generators.first_message_generator as mod

    seen: dict[str, Any] = {}

    async def fake(**kwargs: Any) -> tuple[Any, None]:
        seen.update(kwargs)
        response = MagicMock()
        response.output_text = 'hello'
        return response, None

    monkeypatch.setattr(mod, 'execute_response', AsyncMock(side_effect=fake))
    gen = FirstMessageGenerator(client=MagicMock(), config=LLMCallConfig(model='m', temperature=0.55))
    await gen.generate(_persona(), Scenario(name='S', goal='g'))
    assert seen['temperature'] == 0.55


@pytest.mark.asyncio
async def test_first_message_generator_sends_no_temperature_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    import evaluatorq.simulation.generators.first_message_generator as mod

    seen: dict[str, Any] = {}

    async def fake(**kwargs: Any) -> tuple[Any, None]:
        seen.update(kwargs)
        response = MagicMock()
        response.output_text = 'hello'
        return response, None

    monkeypatch.setattr(mod, 'execute_response', AsyncMock(side_effect=fake))
    gen = FirstMessageGenerator(client=MagicMock(), config=LLMCallConfig(model='m'))
    await gen.generate(_persona(), Scenario(name='S', goal='g'))
    assert seen['temperature'] is None


def test_job_builder_hands_the_config_to_the_runner() -> None:
    from evaluatorq.simulation.api import _build_simulation_job_and_cache

    cfg = LLMCallConfig(model='openai/gpt-4o', temperature=0.4)
    _job, _cache, runner = _build_simulation_job_and_cache(
        job_name='simulation',
        sim_dp_by_id={},
        target=lambda _messages: 'hi',
        target_agent=None,
        model='ignored/when-config-given',
        llm_config=cfg,
        max_turns=1,
        user_simulator=None,
        judge=None,
        generation_client=None,
        hooks=None,
    )
    assert runner._llm_config is cfg


@pytest.mark.asyncio
async def test_executive_summary_forwards_the_configured_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    import evaluatorq.common.reports.executive_summary as summary_mod
    from evaluatorq.simulation.reports.executive_summary import populate_run_executive_summary

    seen: dict[str, Any] = {}

    async def fake(_facts: str, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return summary_mod.ExecutiveSummary(text='ok', usage=None)

    monkeypatch.setattr(summary_mod, 'generate_executive_summary', fake)
    run = MagicMock()
    run.results = [MagicMock()]
    await populate_run_executive_summary(
        run,
        enabled=True,
        model='m',
        llm_config=LLMCallConfig(model='m', temperature=0.15),
        resolve_client=lambda: MagicMock(),
    )
    assert seen['temperature'] == 0.15


@pytest.mark.asyncio
async def test_executive_summary_sends_no_temperature_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    import evaluatorq.common.reports.executive_summary as summary_mod
    from evaluatorq.simulation.reports.executive_summary import populate_run_executive_summary

    seen: dict[str, Any] = {}

    async def fake(_facts: str, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return summary_mod.ExecutiveSummary(text='ok', usage=None)

    monkeypatch.setattr(summary_mod, 'generate_executive_summary', fake)
    run = MagicMock()
    run.results = [MagicMock()]
    await populate_run_executive_summary(run, enabled=True, model='m', resolve_client=lambda: MagicMock())
    # Absent, not None: an unset field must leave generate_executive_summary's own default alone.
    assert 'temperature' not in seen


@pytest.mark.asyncio
@pytest.mark.parametrize('entry', ['summarize_conversations', 'datapoints_from_traces', 'extend_from_traces'])
async def test_trace_helpers_use_the_config_model(monkeypatch: pytest.MonkeyPatch, entry: str) -> None:
    """`generate_structured` ignores `config.model`, so the entry point must resolve it.

    Without this the temperature applied and the model silently did not — the run
    billed the default model while the caller read their own in the config.
    """
    import evaluatorq.simulation.traces as mod

    seen: list[str] = []

    async def fake(*_args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs['model'])
        raise RuntimeError('stop')

    monkeypatch.setattr(mod, 'generate_structured', fake)
    monkeypatch.setattr(mod, 'build_simulation_client', lambda *_a, **_k: (MagicMock(), False), raising=False)
    conversation = mod.TraceConversation(trace_id='t1', messages=[{'role': 'user', 'content': 'hello'}])
    fn = getattr(mod, entry)
    kwargs: dict[str, Any] = {'llm_config': LLMCallConfig(model='chosen/model'), 'client': MagicMock()}
    if entry == 'extend_from_traces':
        kwargs['num_datapoints'] = 1
    try:
        await fn([conversation], **kwargs)
    except Exception:  # noqa: BLE001 — the fake raises once the model is observed
        pass
    assert seen and all(m == 'chosen/model' for m in seen), seen


def test_extend_from_experiment_warns_on_a_contradicting_sim_model(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level('WARNING'):
        resolve_sim_llm_config(
            sim_model='openai/gpt-4o-mini',
            llm_config=LLMCallConfig(model='openai/gpt-4o'),
            caller='extend_from_experiment',
        )
    assert 'extend_from_experiment(): sim_model=' in caplog.text


def test_runner_uses_the_config_client_without_env_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller who brings their own client must not need credentials in the environment."""
    monkeypatch.delenv('ORQ_API_KEY', raising=False)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    mine = MagicMock()
    runner = SimulationRunner(target=lambda _messages: 'hi', llm_config=LLMCallConfig(model='m', client=mine))
    assert runner._get_shared_client() is mine
    # Not ours to close.
    assert runner._client_owned is False


@pytest.mark.parametrize('cls', [PersonaGenerator, ScenarioGenerator, FirstMessageGenerator])
def test_generators_use_the_config_client_without_env_credentials(
    monkeypatch: pytest.MonkeyPatch, cls: Any
) -> None:
    monkeypatch.delenv('ORQ_API_KEY', raising=False)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    mine = MagicMock()
    gen = cls(config=LLMCallConfig(model='m', client=mine))
    assert gen._client is mine
    assert gen._client_owned is False


@pytest.mark.asyncio
async def test_recommendations_use_the_config(monkeypatch: pytest.MonkeyPatch) -> None:
    import evaluatorq.simulation.reports.recommendations as mod

    seen: dict[str, Any] = {}

    async def fake(**kwargs: Any) -> Any:
        seen.update(kwargs)
        raise RuntimeError('stop')

    monkeypatch.setattr(mod, 'generate_structured', fake)
    result = MagicMock()
    result.rules_broken = ['be polite']
    result.goal_achieved = False
    monkeypatch.setattr(mod, 'find_triggers', lambda *_a, **_k: [('goal_not_achieved', 'evidence')])
    cfg = LLMCallConfig(model='config/model', temperature=0.2)
    assert await mod.generate_recommendations([result], MagicMock(), 'explicit/model', llm_config=cfg) == []
    assert seen['config'] is cfg
    # Not passed by this caller, so the config's value must survive to the merge.
    assert isinstance(seen['temperature'], mod.Unset)
    assert seen['model'] == 'explicit/model'
