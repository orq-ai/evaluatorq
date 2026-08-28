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
from evaluatorq.common.structured_output import UNSET
from evaluatorq.simulation.agents.base import AgentConfig, _config_from_agent_config
from evaluatorq.simulation.agents.judge import JudgeAgent
from evaluatorq.simulation.agents.user_simulator import UserSimulatorAgent
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


def test_legacy_agent_config_round_trips_into_a_call_config() -> None:
    """`AgentConfig` is deprecated but still accepted; only fields the caller set
    on it may reach the `LLMCallConfig`, or its `None` defaults would shadow the
    per-call-site literals."""
    from evaluatorq.simulation.agents.base import _config_from_agent_config

    back, _api_key = _config_from_agent_config(
        AgentConfig(model='openai/gpt-4o', temperature=0.25, reasoning_effort='high')
    )
    assert back.model_fields_set == {'model', 'api', 'temperature', 'reasoning_effort'}
    assert back.temperature == 0.25
    assert back.reasoning_effort == 'high'
    assert back.max_tokens not in back.model_fields_set
    assert 'client' not in back.model_fields_set


def test_legacy_agent_config_only_marks_client_set_when_caller_supplied_one() -> None:
    """A legacy `AgentConfig` left at its `client=None` default must not mark
    `client` as caller-set on the resulting `LLMCallConfig`, or it would shadow
    the per-call-site client resolution the same way a stray `None` does for
    every other mirrored field."""
    from evaluatorq.simulation.agents.base import _config_from_agent_config

    without_client, _ = _config_from_agent_config(AgentConfig(model='openai/gpt-4o'))
    assert 'client' not in without_client.model_fields_set

    injected_client = MagicMock()
    with_client, _ = _config_from_agent_config(AgentConfig(model='openai/gpt-4o', client=injected_client))
    assert 'client' in with_client.model_fields_set
    assert with_client.client is injected_client


def test_agents_default_to_the_responses_api_on_a_bare_call_config() -> None:
    """`LLMCallConfig` defaults to chat_completions, which rejects function tools
    plus reasoning_effort together with a 400 on the judge's models."""
    cfg = LLMCallConfig(model='m', client=MagicMock())
    assert JudgeAgent(cfg).config.api == 'responses'
    assert UserSimulatorAgent(cfg).config.api == 'responses'
    assert JudgeAgent(cfg.model_copy(update={'api': 'chat_completions'})).config.api == 'chat_completions'


def test_the_legacy_config_classes_default_to_the_same_endpoint() -> None:
    """They set `api` themselves, because `_config_from_agent_config` always writes
    it and so `BaseAgent.DEFAULT_API` never reaches them — a second literal there
    would drift the deprecated path off the endpoint the agents actually speak."""
    from evaluatorq.simulation.agents.base import BaseAgent
    from evaluatorq.simulation.agents.judge import JudgeAgentConfig
    from evaluatorq.simulation.agents.user_simulator import UserSimulatorAgentConfig

    assert JudgeAgentConfig().api == BaseAgent.DEFAULT_API
    assert UserSimulatorAgentConfig().api == BaseAgent.DEFAULT_API


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
    # Absent, not present-and-None: `set_values` keeps an unset temperature out of the request.
    assert 'temperature' not in seen


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
    # Handed over whole, not exploded into keywords the callee rebuilds into a config.
    assert 'temperature' not in seen
    assert seen['config'].temperature == 0.15


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
    # No config at all, so generate_executive_summary keeps every one of its own defaults.
    assert seen['config'] is None
    assert 'temperature' not in seen


@pytest.mark.asyncio
async def test_executive_summary_config_overrides_only_the_fields_it_sets(monkeypatch: pytest.MonkeyPatch) -> None:
    """The receiving end of the config hand-over: a set `temperature=None` must
    stay None (send no temperature), while `max_tokens` keeps the module default."""
    import evaluatorq.common.reports.executive_summary as summary_mod

    seen: dict[str, Any] = {}

    async def fake_call(**kwargs: Any) -> Any:
        seen.update(kwargs)
        response = MagicMock()
        response.choices[0].message.content = 'summary'
        return response, None

    monkeypatch.setattr(summary_mod, 'execute_chat_completion', fake_call)
    await summary_mod.generate_executive_summary(
        'facts',
        llm_client=MagicMock(),
        model='m',
        config=LLMCallConfig(model='ignored', temperature=None, timeout_ms=1500),
    )
    assert seen['temperature'] is None
    assert seen['timeout_s'] == 1.5
    assert seen['max_completion_tokens'] == summary_mod.EXECUTIVE_SUMMARY_MAX_TOKENS
    # `model` is the caller's, never the config's.
    assert seen['model'] == 'm'


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


def test_extend_from_experiment_hands_the_config_to_the_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    """The config must reach `DatapointGenerator`, not merely be resolved.

    The previous version of this test called `resolve_sim_llm_config` directly with
    a `caller` string it supplied itself, so `extend_from_experiment` could have
    been deleted and it would still have passed.
    """
    import asyncio

    from evaluatorq.simulation import experiments as mod
    from evaluatorq.simulation import generators as gen_mod

    seen: dict[str, Any] = {}

    class _Gen:
        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)

        async def generate_from_description(self, **_kwargs: Any) -> list[Any]:
            return []

        async def close(self) -> None:
            return None

    monkeypatch.setattr(gen_mod, 'DatapointGenerator', _Gen)
    monkeypatch.setattr(mod, 'datapoints_from_experiment', AsyncMock(return_value=[]))

    cfg = LLMCallConfig(model='chosen/model', temperature=0.25)
    asyncio.run(mod.extend_from_experiment('exp-1', llm_config=cfg))
    assert seen.get('config') is cfg


def test_sim_model_applies_when_the_config_leaves_the_model_unset(caplog: pytest.LogCaptureFixture) -> None:
    """`sim_model=` for the model plus `llm_config=` for the sampling is the
    composition the entry-point docstrings recommend, and it must not be read as
    a contradiction: `LLMCallConfig.model` has a non-`None` default, so a value
    check calls it one and silently runs the default model."""
    with caplog.at_level('WARNING'):
        resolved = resolve_sim_llm_config(
            sim_model='openai/gpt-4o-mini',
            llm_config=LLMCallConfig(temperature=0.2),
            caller='simulate',
        )
    assert resolved.model == 'openai/gpt-4o-mini'
    assert resolved.temperature == 0.2
    assert 'contradicts' not in caplog.text


def test_matching_sim_model_and_config_model_do_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level('WARNING'):
        resolve_sim_llm_config(
            sim_model='openai/gpt-4o',
            llm_config=LLMCallConfig(model='openai/gpt-4o'),
            caller='simulate',
        )
    assert 'contradicts' not in caplog.text


def test_the_runner_config_reaches_both_agents_intact() -> None:
    """The runner hands its `LLMCallConfig` to each agent whole. An explicitly set
    `temperature=None` — "send no temperature" — is the case a detour through the
    legacy `AgentConfig` loses, because that class spells unset as `None` too."""
    cfg = LLMCallConfig(model='chosen/model', temperature=None, reasoning_effort='high', client=MagicMock())
    sim = UserSimulatorAgent(cfg, system_prompt='be brief')
    judge = JudgeAgent(cfg, goal='g', criteria=[], ground_truth='')
    for agent in (sim, judge):
        assert agent.config.model == 'chosen/model'
        assert 'temperature' in agent.config.model_fields_set
        assert agent.config.temperature is None
        assert agent.config.reasoning_effort == 'high'
    assert sim._custom_system_prompt == 'be brief'
    assert judge._goal == 'g'


def test_injected_agents_warn_that_the_config_cannot_reach_them(caplog: pytest.LogCaptureFixture) -> None:
    """An injected agent arrives fully built, so `llm_config` cannot configure it.
    Silence there is indistinguishable from the config having been applied."""
    from evaluatorq.simulation.agents.judge import JudgeAgent

    judge = JudgeAgent.__new__(JudgeAgent)
    with caplog.at_level('WARNING'):
        SimulationRunner(
            target=lambda _m: 'hi',
            judge=judge,
            llm_config=LLMCallConfig(model='chosen/model', temperature=0.0),
        )
    assert 'do not reach it' in caplog.text


def test_the_first_message_generator_says_which_config_fields_it_ignores(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """It sizes its own budget and speaks one endpoint, so `max_tokens`, `api` and
    `retry_count` never reach the request. Dropping them silently makes a config
    that did nothing look like one that worked."""
    with caplog.at_level('WARNING'):
        FirstMessageGenerator(
            client=MagicMock(),
            config=LLMCallConfig(model='m', max_tokens=8000, retry_count=3, temperature=0.2),
        )
    assert 'FirstMessageGenerator ignores llm_config max_tokens, retry_count' in caplog.text
    assert 'temperature' not in caplog.text.split('—')[0]


def test_config_timeout_reaches_generate_structured_in_seconds() -> None:
    """A unit slip here is invisible: 30000 ms read as seconds is a 500-minute
    timeout that looks like a hang, and 30 ms read as seconds looks like it works."""
    from evaluatorq.common.structured_output import _fold_config

    settings = _fold_config(
        config=LLMCallConfig(model='m', timeout_ms=1500),
        temperature=UNSET,
        extra_kwargs=UNSET,
        extra_body=UNSET,
        reasoning_effort=UNSET,
        timeout_s=UNSET,
    )
    assert settings.timeout_s == 1.5


def test_a_field_set_to_a_falsy_value_survives_the_agent_round_trip() -> None:
    """`temperature=0.0` is the case that separates `model_fields_set` keying from
    a truthiness or `is not None` check — every other test here uses 0.25."""
    agent_cfg = AgentConfig(model='m', temperature=0.0)
    back, _api_key = _config_from_agent_config(agent_cfg)
    assert back.temperature == 0.0
    assert 'temperature' in back.model_fields_set


def test_runner_uses_the_config_client_without_env_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller who brings their own client must not need credentials in the environment."""
    mine = MagicMock()
    runner = SimulationRunner(target=lambda _messages: 'hi', llm_config=LLMCallConfig(model='m', client=mine))
    assert runner._get_shared_client() is mine
    # Not ours to close.
    assert runner._client_owned is False


@pytest.mark.parametrize('cls', [PersonaGenerator, ScenarioGenerator, FirstMessageGenerator])
def test_generators_use_the_config_client_without_env_credentials(
    monkeypatch: pytest.MonkeyPatch, cls: Any
) -> None:
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


@pytest.mark.asyncio
async def test_the_executive_summary_says_it_ignores_retry_count(caplog: pytest.LogCaptureFixture) -> None:
    """Retry on this call is the client's own budget, so `retry_count` never reaches the
    request. `simulate(llm_config=LLMCallConfig(retry_count=5))` hands the whole run config
    here, and a silent drop makes a config that did nothing look like one that worked."""
    from evaluatorq.common.reports.executive_summary import generate_executive_summary

    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=RuntimeError('no provider in a unit test'))
    with caplog.at_level('WARNING'):
        summary = await generate_executive_summary(
            'facts',
            llm_client=client,
            model='m',
            config=LLMCallConfig(model='m', retry_count=5, temperature=0.2),
        )
    assert summary.text is None
    assert 'generate_executive_summary ignores llm_config retry_count' in caplog.text
    # The fields it does read must not be named as dropped.
    assert 'temperature' not in caplog.text.split('—')[0]


def test_a_config_field_beaten_by_an_explicit_keyword_is_reported_as_dropped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`generate_recommendations(..., temperature=0.2, llm_config=cfg)` drops `cfg.temperature`.
    Warning against the constant instead called it consumed, which is the same silent drop
    this accounting exists to prevent."""
    from evaluatorq.common.structured_output import _fold_config

    cfg = LLMCallConfig(model='m', temperature=0.9, timeout_ms=1500, reasoning_effort='high')
    with caplog.at_level('WARNING'):
        settings = _fold_config(
            config=cfg,
            temperature=0.2,
            extra_kwargs=UNSET,
            extra_body=UNSET,
            reasoning_effort=UNSET,
            timeout_s=30.0,
        )
    assert settings.temperature == 0.2
    assert settings.timeout_s == 30.0
    assert 'generate_structured ignores llm_config temperature, timeout_ms' in caplog.text
    # The one field no keyword beat is still read, so it must not be named.
    assert settings.reasoning_effort == 'high'
    assert 'reasoning_effort' not in caplog.text.split('—')[0]


def test_a_config_field_no_keyword_beats_is_not_reported_as_dropped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The other half of the rule: an untouched keyword leaves the config's value in force."""
    from evaluatorq.common.structured_output import _fold_config

    with caplog.at_level('WARNING'):
        settings = _fold_config(
            config=LLMCallConfig(model='m', temperature=0.9),
            temperature=UNSET,
            extra_kwargs=UNSET,
            extra_body=UNSET,
            reasoning_effort=UNSET,
            timeout_s=UNSET,
        )
    assert settings.temperature == 0.9
    assert 'ignores llm_config' not in caplog.text


def test_the_agent_config_mirror_is_checked_against_llm_call_config() -> None:
    """A field added to `LLMCallConfig` and not mirrored is dropped from the legacy
    `AgentConfig` path in silence — which is what happened to `extra_body`. The mirror
    is verified at import time; this is the same check with a field the mirror lacks."""
    from dataclasses import fields as dataclass_fields

    from evaluatorq.simulation.agents.base import _mirror_gaps

    agent_fields = {f.name for f in dataclass_fields(AgentConfig)}
    assert _mirror_gaps(LLMCallConfig.model_fields, agent_fields) == (set(), set())
    assert _mirror_gaps([*LLMCallConfig.model_fields, 'new_knob'], agent_fields)[0] == {'new_knob'}
