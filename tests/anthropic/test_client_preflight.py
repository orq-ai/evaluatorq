"""A run handed an Anthropic client and a model that client cannot serve stops once, before any call.

Without the check the run starts, pays for every call that comes before the first one the client refuses, and only
then stops. Each surface has one end-to-end test (jury, red team, simulation, Insights) and a table of the cases its
check must tell apart.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
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
CLAUDE = 'claude-sonnet-5-5'


@pytest.fixture(autouse=True)
def _no_llm_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """No key in the environment: the only thing that can serve a model here is the client passed in."""
    for name in LLM_CREDENTIAL_KEYS:
        monkeypatch.delenv(name, raising=False)


def _openai() -> Any:
    return AsyncOpenAI(api_key='sk-test', max_retries=0)


def _anthropic() -> Any:
    return cast('Any', RawAnthropic())


def _check(call: Callable[[], object], refused: list[str] | None, not_listed: list[str] | None = None) -> None:
    """``refused=None``: ``call`` passes. Otherwise it raises, naming every ``refused`` part and none of ``not_listed``."""
    if refused is None:
        call()
        return
    with pytest.raises(ModelClientMismatchError) as info:
        call()
    message = str(info.value)
    assert all(part in message for part in refused), message
    assert not any(part in message for part in not_listed or []), message


# ---------------------------------------------------------------------------
# The shared rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('wrap', [lambda raw: raw, as_llm_client], ids=['raw', 'wrapped'])
def test_an_anthropic_client_refuses_another_providers_model_and_names_where_it_is_set(wrap: Any) -> None:
    with pytest.raises(ModelClientMismatchError) as info:
        require_models_served_by_client(wrap(RawAnthropic()), {'judges[0]': 'openai/gpt-6-sol', 'judges[1]': CLAUDE})

    message = str(info.value)
    assert message.startswith('The Anthropic client passed in cannot serve judges[0]=openai/gpt-6-sol.')
    assert 'judges[1]' not in message, 'a model the client serves is not listed'
    assert 'ANTHROPIC_API_KEY' not in message, 'the environment did not choose this client'


@pytest.mark.parametrize(
    ('client', 'model'),
    [(_anthropic, CLAUDE), (_anthropic, 'anthropic/claude-haiku-5-5'), (lambda: None, 'openai/gpt-6-sol'), (_openai, 'openai/gpt-6-sol')],
    ids=['claude id', 'prefixed claude id', 'no client', 'openai client'],
)
def test_a_model_the_client_serves_passes(client: Any, model: str) -> None:
    require_models_served_by_client(client(), {'model': model})


def test_the_combined_check_uses_the_client_when_there_is_one(monkeypatch: pytest.MonkeyPatch) -> None:
    models = {'model': 'openai/gpt-6-sol'}
    # An OpenAI client serves it, also when the environment would have picked Anthropic.
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-ant-test')
    require_models_served(models, client=_openai())
    _check(lambda: require_models_served(models), ['ANTHROPIC_API_KEY is the only LLM key set'])
    _check(lambda: require_models_served(models, client=RawAnthropic()), ['The Anthropic client passed in'])


# ---------------------------------------------------------------------------
# Jury
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('declare', [llm_jury, llm_jury_pairwise])
def test_a_jury_with_an_anthropic_client_is_checked_at_declaration(declare: Any) -> None:
    named = {'name': 'correct'} if declare is llm_jury else {}

    def jury(client: Any, **models: Any) -> None:
        declare(**named, criteria='Is the answer correct?', client=client, **models)

    fake = RawAnthropic()
    with pytest.raises(ModelClientMismatchError, match=rf'Anthropic client passed in cannot serve {DEFAULT_SMART}=openai/'):
        jury(fake)
    assert fake.messages.calls == []

    # Only the replacement judge is refused; the Claude judge in judges[0] is not listed.
    _check(
        lambda: jury(RawAnthropic(), judges=[CLAUDE], replacement_judges=['openai/gpt-6-sol']),
        ['serve replacement_judges[0]=openai/gpt-6-sol'],
    )
    jury(RawAnthropic(), judges=[CLAUDE])


# ---------------------------------------------------------------------------
# Red team
# ---------------------------------------------------------------------------


def test_a_default_red_team_run_with_an_anthropic_client_stops_before_any_call() -> None:
    from evaluatorq.redteam import red_team
    from evaluatorq.redteam.backends.openai import OpenAIModelTarget

    fake = RawAnthropic()
    target_client = RawAnthropic()
    target = OpenAIModelTarget(model=CLAUDE, client=as_llm_client(target_client))

    with pytest.raises(ModelClientMismatchError) as info:
        asyncio.run(
            red_team(
                target,
                mode='dynamic',
                vulnerabilities=['prompt_injection'],
                max_dynamic_datapoints=1,
                max_turns=1,
                llm_client=cast('Any', fake),
            )
        )

    message = str(info.value)
    assert message.startswith('The Anthropic client passed in cannot serve ')
    assert 'EVALUATORQ_SMART_MODEL (unset, built-in default) (redteam.attacker)=openai/' in message
    assert 'EVALUATORQ_SMART_MODEL (unset, built-in default) (redteam.evaluator)=openai/' in message
    assert fake.messages.calls == [], 'the Anthropic client was never called'
    assert target_client.messages.calls == [], 'the target was never called'


def _red_team_cases() -> dict[str, tuple[Callable[[], tuple[Any, Any]], str, list[str] | None, list[str]]]:
    """``id: (lambda: (config, llm_client), mode, refused, not_listed)``. A role runs on its client, picked in the run's order."""
    from evaluatorq.redteam.contracts import EvaluatorConfig, LLMConfig

    def config(**kwargs: Any) -> Any:
        return LLMConfig(**kwargs)

    return {
        # Only the attacker runs on the shared Anthropic client.
        'evaluator on its own openai client': (
            lambda: (config(evaluator=EvaluatorConfig(client=_openai())), _anthropic()),
            'dynamic', ['(redteam.attacker)=openai/'], ['redteam.evaluator'],
        ),
        'evaluator on its own anthropic client': (
            lambda: (config(evaluator=EvaluatorConfig(client=_anthropic())), _openai()),
            'dynamic', ['(redteam.evaluator)=openai/'], ['redteam.attacker'],
        ),
        # `EvaluatorConfig(model=...)` is folded into `judges[0]`, which is what the run calls.
        'evaluator model shorthand': (
            lambda: (config(evaluator=EvaluatorConfig(model='openai/some-judge', client=_anthropic())), None),
            'static', ['judges[0] (--evaluator-model)=openai/some-judge'], [],
        ),
        'static run never calls the attacker': (
            lambda: (config(), _anthropic()), 'static', ['redteam.evaluator'], ['redteam.attacker'],
        ),
        'openai client': (lambda: (config(), _openai()), 'dynamic', None, []),
        # The attacker takes the run-wide client before its own. The pre-flight once checked them the other way round.
        'attacker on run-wide anthropic over its own openai': (
            lambda: (config(attacker=LLMCallConfig(model='openai/attack-model', client=_openai())), _anthropic()),
            'dynamic', ['=openai/attack-model'], [],
        ),
    }


@pytest.mark.parametrize('case', _red_team_cases().values(), ids=_red_team_cases().keys())
def test_each_red_team_role_is_checked_against_the_client_it_runs_on(case: Any) -> None:
    from evaluatorq.redteam.contracts import Pipeline
    from evaluatorq.redteam.runner import _validate_credentials_and_deps

    build, mode, refused, not_listed = case
    config, llm_client = build()
    _check(
        lambda: _validate_credentials_and_deps(
            config=config, llm_client=llm_client, replay=None, resolved_mode=Pipeline(mode), dataset=None
        ),
        refused,
        not_listed,
    )


def test_a_red_team_run_on_claude_roles_with_an_anthropic_client_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.redteam.contracts import LLMConfig, Pipeline
    from evaluatorq.redteam.runner import _validate_credentials_and_deps

    monkeypatch.setenv('EVALUATORQ_SMART_MODEL', CLAUDE)
    _validate_credentials_and_deps(
        config=LLMConfig(), llm_client=_anthropic(), replay=None, resolved_mode=Pipeline('dynamic'), dataset=None
    )


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

    assert passed_in_client(role, config, run_wide) is {'own': own, 'run-wide': run_wide}[expected]
    # With only one of the two set, that one is used, whichever it is.
    assert passed_in_client(role, config, None) is own
    assert passed_in_client(role, LLMConfig(), run_wide) is run_wide
    assert passed_in_client(role, LLMConfig(), None) is None
    assert passed_in_client(role, None, run_wide) is run_wide
    # The run's own resolver agrees with it.
    assert _resolve_attacker_llm_client(run_wide, config, create_if_missing=False) is passed_in_client('attacker', config, run_wide)


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


AGENT = object()

SIM_CASES: dict[str, tuple[bool, Callable[[], tuple[Any, dict[str, Any]]], list[str] | None, list[str]]] = {
    # ``id: (ANTHROPIC_API_KEY set, lambda: (llm_config, kwargs), refused, not_listed)``
    'both roles bring their own agent and nothing is generated': (
        False,
        lambda: (LLMCallConfig(client=_anthropic()), {'user_simulator': AGENT, 'judge': AGENT, 'generating': False}),
        None, [],
    ),
    'only the judge runs on the client': (
        False,
        lambda: (LLMCallConfig(client=_anthropic()), {'user_simulator': AGENT, 'generating': False}),
        ['(sim.judge)=openai/'], ['sim.user', 'sim.generator'],
    ),
    'a generation client wins over the config client': (
        False, lambda: (LLMCallConfig(client=_anthropic()), {'generation_client': _openai()}), None, [],
    ),
    'an anthropic generation client wins over an openai config client': (
        False, lambda: (LLMCallConfig(client=_openai()), {'generation_client': _anthropic()}), ['Anthropic client passed in'], [],
    ),
    'a claude model on an anthropic client': (
        False, lambda: (LLMCallConfig(model=CLAUDE, client=_anthropic()), {}), None, [],
    ),
    # A passed-in client serves the run, so a stray key in the environment is not consulted.
    'a passed generation client ignores a stray anthropic key': (True, lambda: (None, {'generation_client': _openai()}), None, []),
    'a passed config client ignores a stray anthropic key': (True, lambda: (LLMCallConfig(client=_openai()), {}), None, []),
    'no client passed: the environment is checked': (True, lambda: (None, {}), ['ANTHROPIC_API_KEY is the only LLM key set'], []),
}


@pytest.mark.parametrize('case', SIM_CASES.values(), ids=SIM_CASES.keys())
def test_a_simulation_checks_the_roles_that_run_on_the_serving_client(case: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.simulation._config import require_sim_models_served

    anthropic_key, build, refused, not_listed = case
    if anthropic_key:
        monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-ant-test')
    config, kwargs = build()
    kwargs.setdefault('generation_client', None)
    _check(lambda: require_sim_models_served(config, **kwargs), refused, not_listed)


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
            InsightsPopulation(), dimensions=(), labels=(), runs_dir=tmp_path / 'runs', llm_client=cast('Any', fake)
        )

    message = str(info.value)
    assert message.startswith('The Anthropic client passed in cannot serve ')
    assert '(insights.summary)=openai/' in message
    assert 'insights.labels' not in message, 'no labels were asked for, so the label model is not checked'
    assert fake.messages.calls == []
    assert not (tmp_path / 'runs').exists(), 'nothing was written for a run that never started'


@pytest.mark.asyncio
async def test_an_insights_run_that_only_clusters_still_checks_the_model_that_merges_clusters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """No labels and no sentiment, so nothing labels a trace; clustering still sends the classifier model to the merge step."""
    from evaluatorq.insights import pipeline
    from evaluatorq.insights.models import InsightsPopulation

    fake = RawAnthropic()
    for name in ('resolve_population', 'label_traces', 'summarize_traces', '_build_dimension'):
        monkeypatch.setattr(pipeline, name, _fail_if_reached(name))
    monkeypatch.setattr(pipeline, 'resolve_orq_client', lambda: pytest.fail('resolved an Orq client'))

    with pytest.raises(ModelClientMismatchError) as info:
        await pipeline.insights(
            InsightsPopulation(),
            dimensions=('intent',),
            labels=(),
            summary_model='claude-sonnet-5-5',
            runs_dir=tmp_path / 'runs',
            llm_client=cast('Any', fake),
            embedding_client=cast('Any', object()),
        )

    message = str(info.value)
    assert '(insights.labels)=' in message
    # An explicit summary model is named by its argument, so that is the label to look for.
    assert 'summary_model' not in message and 'claude-sonnet-5-5' not in message, 'the summary model is a Claude id'
    assert fake.messages.calls == []
    assert not (tmp_path / 'runs').exists(), 'nothing was written for a run that never started'


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


def _default_classifier() -> str:
    from evaluatorq.common.model_roles import role_model

    return role_model('classifier', task='insights.labels')


INSIGHTS_CASES: dict[str, tuple[Callable[[], Any], dict[str, Any], list[str] | None, list[str]]] = {
    # ``id: (client, models, refused, not_listed)``. A model is checked only when the run calls it.
    'only the summary model, a claude id': (_anthropic, {}, None, []),
    'labels asked for': (_anthropic, {'label_model': 'openai/gpt-6-luna'}, ['classifier_model=openai/gpt-6-luna'], ['summary']),
    'a query population': (
        _anthropic, {'label_model': CLAUDE, 'query_models': ('openai/gpt-5.4-mini', CLAUDE)}, ['serve compiler_model=openai/gpt-5.4-mini.'], [],
    ),
    # The classify-only default is checked as the fast model it falls back to off the router.
    'the classify-only default': (_anthropic, {'label_model': _default_classifier}, ['standing in for the classify-only'], []),
    'another kind of client': (_openai, {'summary_model': 'openai/gpt-6-sol'}, None, []),
}


@pytest.mark.parametrize('case', INSIGHTS_CASES.values(), ids=INSIGHTS_CASES.keys())
def test_insights_checks_the_models_the_run_calls(case: Any) -> None:
    from evaluatorq.insights.pipeline import _require_insights_models_served

    client, models, refused, not_listed = case
    # A callable model is resolved here, so the role default is read under this test's environment.
    models = {'summary_model': CLAUDE, 'label_model': None, 'query_models': None, **models}
    resolved: dict[str, Any] = {key: value() if callable(value) else value for key, value in models.items()}
    _check(lambda: _require_insights_models_served(client(), **resolved), refused, not_listed)
