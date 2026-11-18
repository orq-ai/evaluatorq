"""A run handed an Anthropic client and a model that client cannot serve stops once, before any call.

Without the check the run starts, pays for every call that comes before the first one the client refuses, and only
then stops. One test per surface: jury, red team, simulation, Insights.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from openai import AsyncOpenAI

from evaluatorq.common.llm_client import (
    LLM_CREDENTIAL_KEYS,
    ModelClientMismatchError,
    as_llm_client,
    require_models_served,
    require_models_served_by_client,
)
from evaluatorq.contracts import LLMCallConfig
from evaluatorq.llm_jury import llm_jury, llm_jury_pairwise

from tests.anthropic.test_wiring import RawAnthropic

DEFAULT_SMART = r'EVALUATORQ_SMART_MODEL \(unset, built-in default\)'


@pytest.fixture(autouse=True)
def _no_llm_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """No key in the environment: the only thing that can serve a model here is the client passed in."""
    for name in LLM_CREDENTIAL_KEYS:
        monkeypatch.delenv(name, raising=False)


def _openai() -> Any:
    return AsyncOpenAI(api_key='sk-test', max_retries=0)


# ---------------------------------------------------------------------------
# The shared rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('wrap', [lambda raw: raw, as_llm_client], ids=['raw', 'wrapped'])
def test_an_anthropic_client_refuses_another_providers_model_and_names_where_it_is_set(wrap: Any) -> None:
    client = wrap(RawAnthropic())

    with pytest.raises(ModelClientMismatchError) as info:
        require_models_served_by_client(client, {'judges[0]': 'openai/gpt-6-sol', 'judges[1]': 'claude-sonnet-5-5'})

    message = str(info.value)
    assert message.startswith('The Anthropic client passed in cannot serve judges[0]=openai/gpt-6-sol.')
    assert 'judges[1]' not in message, 'a model the client serves is not listed'
    assert 'ANTHROPIC_API_KEY' not in message, 'the environment did not choose this client'


def test_an_anthropic_client_serves_claude_ids_with_or_without_the_prefix() -> None:
    require_models_served_by_client(RawAnthropic(), {'a': 'claude-sonnet-5-5', 'b': 'anthropic/claude-haiku-5-5'})


@pytest.mark.parametrize('client', [None, 'openai'])
def test_a_client_that_is_not_anthropic_is_never_checked(client: Any) -> None:
    require_models_served_by_client(_openai() if client == 'openai' else None, {'model': 'openai/gpt-6-sol'})


def test_the_combined_check_uses_the_client_when_there_is_one(monkeypatch: pytest.MonkeyPatch) -> None:
    models = {'model': 'openai/gpt-6-sol'}
    # An OpenAI client serves it, also when the environment would have picked Anthropic.
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-ant-test')
    require_models_served(models, client=_openai())
    with pytest.raises(ModelClientMismatchError, match='ANTHROPIC_API_KEY is the only LLM key set'):
        require_models_served(models)
    with pytest.raises(ModelClientMismatchError, match='The Anthropic client passed in'):
        require_models_served(models, client=RawAnthropic())


# ---------------------------------------------------------------------------
# Jury
# ---------------------------------------------------------------------------


def _named(declare: Any) -> dict[str, str]:
    return {'name': 'correct'} if declare is llm_jury else {}


@pytest.mark.parametrize('declare', [llm_jury, llm_jury_pairwise])
def test_a_jury_on_the_default_panel_with_an_anthropic_client_stops_at_declaration(declare: Any) -> None:
    fake = RawAnthropic()

    with pytest.raises(ModelClientMismatchError, match=rf'Anthropic client passed in cannot serve {DEFAULT_SMART}=openai/'):
        declare(**_named(declare), criteria='Is the answer correct?', client=fake)

    assert fake.messages.calls == []


@pytest.mark.parametrize('declare', [llm_jury, llm_jury_pairwise])
def test_a_jury_with_an_anthropic_client_names_the_replacement_judge_it_cannot_serve(declare: Any) -> None:
    with pytest.raises(ModelClientMismatchError, match=r'replacement_judges\[0\]=openai/gpt-6-sol') as info:
        declare(
            **_named(declare),
            criteria='Is the answer correct?',
            judges=['claude-sonnet-5-5'],
            replacement_judges=['openai/gpt-6-sol'],
            client=RawAnthropic(),
        )
    assert 'serve replacement_judges[0]=' in str(info.value), 'the Claude judge in judges[0] is not listed'


@pytest.mark.parametrize('declare', [llm_jury, llm_jury_pairwise])
def test_a_jury_on_claude_judges_with_an_anthropic_client_declares(declare: Any) -> None:
    declare(**_named(declare), criteria='Is the answer correct?', judges=['claude-sonnet-5-5'], client=RawAnthropic())


# ---------------------------------------------------------------------------
# Red team
# ---------------------------------------------------------------------------


def _red_team_pre_flight(config: Any, *, llm_client: Any = None, mode: str = 'dynamic') -> None:
    from evaluatorq.redteam.contracts import Pipeline
    from evaluatorq.redteam.runner import _validate_credentials_and_deps

    _validate_credentials_and_deps(config=config, llm_client=llm_client, replay=None, resolved_mode=Pipeline(mode), dataset=None)


def test_a_default_red_team_run_with_an_anthropic_client_stops_before_any_call() -> None:
    from evaluatorq.redteam import red_team

    from evaluatorq.redteam.backends.openai import OpenAIModelTarget

    fake = RawAnthropic()
    target_client = RawAnthropic()
    target = OpenAIModelTarget(model='claude-sonnet-5-5', client=as_llm_client(target_client))

    async def run() -> None:
        await red_team(
            target,
            mode='dynamic',
            vulnerabilities=['prompt_injection'],
            max_dynamic_datapoints=1,
            max_turns=1,
            llm_client=cast('Any', fake),
        )

    import asyncio

    with pytest.raises(ModelClientMismatchError) as info:
        asyncio.run(run())

    message = str(info.value)
    assert message.startswith('The Anthropic client passed in cannot serve ')
    assert 'EVALUATORQ_SMART_MODEL (unset, built-in default) (redteam.attacker)=openai/' in message
    assert 'EVALUATORQ_SMART_MODEL (unset, built-in default) (redteam.evaluator)=openai/' in message
    assert fake.messages.calls == [], 'the Anthropic client was never called'
    assert target_client.messages.calls == [], 'the target was never called'


def test_a_red_team_role_is_checked_against_the_client_it_runs_on() -> None:
    from evaluatorq.redteam.contracts import EvaluatorConfig, LLMConfig

    # The evaluator has its own OpenAI client, so only the attacker runs on the injected Anthropic one.
    config = LLMConfig(evaluator=EvaluatorConfig(client=_openai()))
    with pytest.raises(ModelClientMismatchError) as info:
        _red_team_pre_flight(config, llm_client=RawAnthropic())
    assert '(redteam.attacker)=openai/' in str(info.value)
    assert 'redteam.evaluator' not in str(info.value)

    # The reverse: the evaluator has its own Anthropic client and the shared one is an OpenAI client. The evaluator
    # runs on its own, so its judges are the ones refused; the attacker runs on the shared client and is fine.
    config = LLMConfig(evaluator=EvaluatorConfig(client=cast('Any', RawAnthropic())))
    with pytest.raises(ModelClientMismatchError) as info:
        _red_team_pre_flight(config, llm_client=_openai())
    assert '(redteam.evaluator)=openai/' in str(info.value)
    assert 'redteam.attacker' not in str(info.value)


def test_the_evaluator_model_shorthand_is_checked_as_the_first_judge() -> None:
    from evaluatorq.redteam.contracts import EvaluatorConfig, LLMConfig

    # `EvaluatorConfig(model=...)` is folded into `judges[0]`, which is what the run calls and what the check reads.
    config = LLMConfig(evaluator=EvaluatorConfig(model='openai/some-judge', client=cast('Any', RawAnthropic())))
    with pytest.raises(ModelClientMismatchError, match=r'judges\[0\] \(--evaluator-model\)=openai/some-judge'):
        _red_team_pre_flight(config, mode='static')


def test_a_simulation_on_a_passed_client_ignores_a_stray_anthropic_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.simulation._config import require_sim_models_served

    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-ant-test')
    # The run is served by the OpenAI client that was passed in, whichever way it was passed.
    require_sim_models_served(None, generation_client=_openai())
    require_sim_models_served(LLMCallConfig(client=_openai()), generation_client=None)
    # With no client passed in, the same environment is the serving side and is checked.
    with pytest.raises(ModelClientMismatchError, match='ANTHROPIC_API_KEY is the only LLM key set'):
        require_sim_models_served(None, generation_client=None)


def test_a_static_red_team_run_does_not_check_the_attacker_it_never_calls() -> None:
    from evaluatorq.redteam.contracts import LLMConfig

    with pytest.raises(ModelClientMismatchError) as info:
        _red_team_pre_flight(LLMConfig(), llm_client=RawAnthropic(), mode='static')
    assert 'redteam.evaluator' in str(info.value)
    assert 'redteam.attacker' not in str(info.value)


def test_a_red_team_run_on_claude_roles_with_an_anthropic_client_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.redteam.contracts import LLMConfig

    monkeypatch.setenv('EVALUATORQ_SMART_MODEL', 'claude-sonnet-5-5')
    _red_team_pre_flight(LLMConfig(), llm_client=RawAnthropic())


def test_a_red_team_run_with_an_openai_client_is_not_checked() -> None:
    from evaluatorq.redteam.contracts import LLMConfig

    _red_team_pre_flight(LLMConfig(), llm_client=_openai())


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def _persona() -> Any:
    from evaluatorq.simulation.contracts import CommunicationStyle, Persona

    return Persona(
        name='Test User',
        patience=0.5,
        assertiveness=0.5,
        politeness=0.5,
        technical_level=0.5,
        communication_style=CommunicationStyle.casual,
        background='bg',
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('how', ['llm_config.client', 'generation_client'])
async def test_a_default_simulation_with_an_anthropic_client_stops_before_calling_the_target(how: str) -> None:
    from evaluatorq.simulation import simulate
    from evaluatorq.simulation.contracts import Scenario

    fake = RawAnthropic()
    target_calls: list[Any] = []

    async def target(messages: Any) -> str:
        target_calls.append(messages)
        return 'ok'

    passed = {'llm_config': LLMCallConfig(client=cast('Any', fake))} if how == 'llm_config.client' else {'generation_client': fake}
    with pytest.raises(ModelClientMismatchError, match=r'Anthropic client passed in cannot serve .*\(sim\.judge\)=openai/'):
        await simulate(
            target=target,
            personas=[_persona()],
            scenarios=[Scenario(name='S', goal='fix my bug')],
            upload_results=False,
            **cast('Any', passed),
        )

    assert target_calls == [], 'the target was never called'
    assert fake.messages.calls == [], 'the Anthropic client was never called'


def test_a_simulation_checks_only_the_roles_the_passed_client_serves() -> None:
    from evaluatorq.simulation._config import require_sim_models_served

    fake = RawAnthropic()
    config = LLMCallConfig(client=cast('Any', fake))
    agent = object()

    # Both conversation roles bring their own agent and the run writes nothing: nothing runs on the client.
    require_sim_models_served(config, generation_client=None, user_simulator=agent, judge=agent, generating=False)
    # Only the judge runs on it.
    with pytest.raises(ModelClientMismatchError) as info:
        require_sim_models_served(config, generation_client=None, user_simulator=agent, generating=False)
    assert '(sim.judge)=openai/' in str(info.value)
    assert 'sim.user' not in str(info.value)
    assert 'sim.generator' not in str(info.value)
    # A generation client wins over the config's client, as it does in the generator and the runner.
    require_sim_models_served(config, generation_client=_openai())
    with pytest.raises(ModelClientMismatchError, match='Anthropic client passed in'):
        require_sim_models_served(LLMCallConfig(client=_openai()), generation_client=fake)


def test_a_simulation_on_a_claude_model_with_an_anthropic_client_passes() -> None:
    from evaluatorq.simulation._config import require_sim_models_served

    require_sim_models_served(LLMCallConfig(model='claude-sonnet-5-5', client=cast('Any', RawAnthropic())), generation_client=None)


@pytest.mark.asyncio
async def test_generation_with_an_anthropic_client_checks_only_the_generation_model() -> None:
    from evaluatorq.simulation import generate

    fake = RawAnthropic()
    with pytest.raises(ModelClientMismatchError, match=r'Anthropic client passed in cannot serve .*\(sim\.generator\)=openai/') as info:
        await generate(agent_description='A support agent.', generation_client=cast('Any', fake))
    assert 'sim.judge' not in str(info.value)
    assert fake.messages.calls == []


# ---------------------------------------------------------------------------
# Insights
# ---------------------------------------------------------------------------


def _fail_if_reached(name: str) -> Any:
    async def stage(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail(f'{name} ran: the run did not stop before its first stage')

    return stage


@pytest.mark.asyncio
async def test_a_default_insights_run_with_an_anthropic_client_stops_before_any_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from evaluatorq.insights import pipeline
    from evaluatorq.insights.models import InsightsPopulation

    fake = RawAnthropic()
    for name in ('resolve_population', 'label_traces', 'summarize_traces', '_build_dimension'):
        monkeypatch.setattr(pipeline, name, _fail_if_reached(name))
    monkeypatch.setattr(pipeline, 'resolve_orq_client', lambda: pytest.fail('resolved an Orq client'))

    with pytest.raises(ModelClientMismatchError) as info:
        await pipeline.insights(
            InsightsPopulation(),
            dimensions=(),
            labels=(),
            runs_dir=tmp_path / 'runs',
            llm_client=cast('Any', fake),
        )

    message = str(info.value)
    assert message.startswith('The Anthropic client passed in cannot serve ')
    assert '(insights.summary)=openai/' in message
    assert 'insights.labels' not in message, 'no labels were asked for, so the label model is not checked'
    assert fake.messages.calls == []
    assert not (tmp_path / 'runs').exists(), 'nothing was written for a run that never started'


def test_insights_checks_the_label_and_query_models_only_when_the_run_calls_them() -> None:
    from evaluatorq.common.model_roles import role_model
    from evaluatorq.insights.pipeline import _require_insights_models_served

    fake = RawAnthropic()
    claude = 'claude-sonnet-5-5'
    # Only the summary model is called, and it is a Claude id.
    _require_insights_models_served(fake, summary_model=claude, label_model=None, query_models=None)
    # Labels asked for: the label model is called too.
    with pytest.raises(ModelClientMismatchError) as info:
        _require_insights_models_served(fake, summary_model=claude, label_model='openai/gpt-6-luna', query_models=None)
    assert 'classifier_model=openai/gpt-6-luna' in str(info.value)
    assert 'summary' not in str(info.value)
    # A query population: both query models are called.
    with pytest.raises(ModelClientMismatchError) as info:
        _require_insights_models_served(
            fake, summary_model=claude, label_model=claude, query_models=('openai/gpt-5.4-mini', claude)
        )
    assert 'serve compiler_model=openai/gpt-5.4-mini.' in str(info.value)
    # The classify-only default is checked as the fast model it falls back to off the router.
    default_classifier = role_model('classifier', task='insights.labels')
    with pytest.raises(ModelClientMismatchError, match='standing in for the classify-only'):
        _require_insights_models_served(fake, summary_model=claude, label_model=default_classifier, query_models=None)
    # Another kind of client is not checked.
    _require_insights_models_served(_openai(), summary_model='openai/gpt-6-sol', label_model=None, query_models=None)


@pytest.mark.asyncio
async def test_insights_left_to_an_anthropic_only_environment_stops_the_same_way(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from evaluatorq.insights import pipeline
    from evaluatorq.insights.models import InsightsPopulation

    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-ant-test')
    monkeypatch.setattr(pipeline, 'resolve_population', _fail_if_reached('resolve_population'))

    with pytest.raises(ModelClientMismatchError, match='ANTHROPIC_API_KEY is the only LLM key set'):
        await pipeline.insights(InsightsPopulation(), dimensions=(), labels=(), runs_dir=tmp_path / 'runs')


# --- the pre-flight and the run pick each role's client in the same order ---


@pytest.mark.parametrize(
    ('role', 'expected'),
    # The attacker takes the run-wide client first; the evaluator takes its own first.
    [('attacker', 'run-wide'), ('evaluator', 'own')],
)
def test_each_role_runs_on_the_client_the_run_itself_picks(role: str, expected: str) -> None:
    from evaluatorq.redteam.contracts import EvaluatorConfig, LLMConfig
    from evaluatorq.redteam.runner import _resolve_attacker_llm_client, passed_in_client

    own, run_wide = _openai(), _openai()
    config = LLMConfig(attacker=LLMCallConfig(client=own), evaluator=EvaluatorConfig(client=own))
    clients = {'own': own, 'run-wide': run_wide}

    assert passed_in_client(role, config, run_wide) is clients[expected]
    # With only one of the two set, that one is used, whichever it is.
    assert passed_in_client(role, config, None) is own
    assert passed_in_client(role, LLMConfig(), run_wide) is run_wide
    assert passed_in_client(role, LLMConfig(), None) is None
    assert passed_in_client(role, None, run_wide) is run_wide
    # The run's own resolver agrees with it.
    assert _resolve_attacker_llm_client(run_wide, config, create_if_missing=False) is passed_in_client('attacker', config, run_wide)


def test_the_attacker_is_checked_against_the_run_wide_client_when_both_are_set() -> None:
    from evaluatorq.redteam.contracts import LLMConfig

    # The run sends the attack model to the run-wide client, so that is the one that must serve it.
    on_openai_role_client = LLMConfig(attacker=LLMCallConfig(model='openai/gpt-6-sol', client=_openai()))
    with pytest.raises(ModelClientMismatchError, match='openai/gpt-6-sol'):
        _red_team_pre_flight(on_openai_role_client, llm_client=RawAnthropic())

    # The other way round the run works, so the check must pass: the Anthropic role client is never used.
    on_anthropic_role_client = LLMConfig(
        attacker=LLMCallConfig(model='openai/gpt-6-sol', client=cast('Any', RawAnthropic()))
    )
    _red_team_pre_flight(on_anthropic_role_client, llm_client=_openai(), mode='dynamic')


def test_the_evaluator_is_checked_against_its_own_client_when_both_are_set() -> None:
    from evaluatorq.redteam.contracts import EvaluatorConfig, LLMConfig

    claude_attacker = LLMCallConfig(model='claude-sonnet-5-5')
    own_openai = LLMConfig(attacker=claude_attacker, evaluator=EvaluatorConfig(model='openai/some-judge', client=_openai()))
    _red_team_pre_flight(own_openai, llm_client=RawAnthropic())

    own_anthropic = LLMConfig(
        attacker=LLMCallConfig(model='openai/gpt-6-sol'),
        evaluator=EvaluatorConfig(model='openai/some-judge', client=cast('Any', RawAnthropic())),
    )
    with pytest.raises(ModelClientMismatchError, match='openai/some-judge'):
        _red_team_pre_flight(own_anthropic, llm_client=_openai())
