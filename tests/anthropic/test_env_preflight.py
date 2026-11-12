"""With only ``ANTHROPIC_API_KEY`` set, the jury and the signal classifiers stop before any work on a model Anthropic cannot serve (RES-1875)."""

from __future__ import annotations

from typing import Any

import pytest
from openai import AsyncOpenAI

from evaluatorq.common.llm_client import LLM_CREDENTIAL_KEYS, ModelClientMismatchError
from evaluatorq.llm_jury import llm_jury, llm_jury_pairwise
from evaluatorq.signals import classify_tool_doors, classify_tool_roles

from tests.signals.conftest import agent, call, traj, user


@pytest.fixture(autouse=True)
def _anthropic_only(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in LLM_CREDENTIAL_KEYS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-ant-test')


def _named(declare: Any) -> dict[str, str]:
    return {'name': 'correct'} if declare is llm_jury else {}


@pytest.mark.parametrize('declare', [llm_jury, llm_jury_pairwise])
def test_a_jury_on_the_default_panel_stops_at_declaration(declare: Any) -> None:
    with pytest.raises(ModelClientMismatchError, match=r'EVALUATORQ_SMART_MODEL \(unset, built-in default\)=openai/'):
        declare(**_named(declare), criteria='Is the answer correct?')


@pytest.mark.parametrize('declare', [llm_jury, llm_jury_pairwise])
def test_a_jury_names_the_replacement_judge_it_cannot_serve(declare: Any) -> None:
    with pytest.raises(ModelClientMismatchError, match=r'replacement_judges\[0\]=openai/gpt-6-sol'):
        declare(
            **_named(declare),
            criteria='Is the answer correct?',
            judges=['claude-sonnet-5-5'],
            replacement_judges=['openai/gpt-6-sol'],
        )


@pytest.mark.parametrize('declare', [llm_jury, llm_jury_pairwise])
def test_a_jury_on_claude_judges_or_with_its_own_client_declares(declare: Any) -> None:
    declare(**_named(declare), criteria='Is the answer correct?', judges=['claude-sonnet-5-5'])
    declare(**_named(declare), criteria='Is the answer correct?', client=AsyncOpenAI(api_key='sk-test', max_retries=0))


def _one_call() -> Any:
    return traj([user(), agent(calls=[call('frobnicate', {'path': 'x'}, 'a')])])


@pytest.mark.asyncio
async def test_door_classification_checks_the_fast_model_the_classify_only_default_falls_back_to() -> None:
    with pytest.raises(ModelClientMismatchError) as info:
        await classify_tool_doors(_one_call())
    assert 'EVALUATORQ_FAST_MODEL (unset, built-in default), standing in for the classify-only' in str(info.value)


@pytest.mark.asyncio
async def test_tool_role_classification_checks_the_fast_model_the_classify_only_default_falls_back_to() -> None:
    with pytest.raises(ModelClientMismatchError, match='EVALUATORQ_FAST_MODEL'):
        await classify_tool_roles([_one_call()])


@pytest.mark.asyncio
async def test_signal_classification_with_nothing_to_classify_needs_no_check() -> None:
    assert (await classify_tool_doors(traj([user(), agent()])))['one_way_call_count'].value == 0


@pytest.mark.asyncio
async def test_a_programmatic_simulation_stops_before_calling_the_target() -> None:
    from evaluatorq.simulation import simulate
    from evaluatorq.simulation.contracts import CommunicationStyle, Persona, Scenario

    calls: list[Any] = []

    async def target(messages: Any) -> str:
        calls.append(messages)
        return 'ok'

    with pytest.raises(ModelClientMismatchError, match=r'\(sim\.judge\)=openai/'):
        await simulate(
            target=target,
            personas=[
                Persona(
                    name='Test User',
                    patience=0.5,
                    assertiveness=0.5,
                    politeness=0.5,
                    technical_level=0.5,
                    communication_style=CommunicationStyle.casual,
                    background='bg',
                )
            ],
            scenarios=[Scenario(name='S', goal='fix my bug')],
            upload_results=False,
        )
    assert calls == []


@pytest.mark.asyncio
async def test_programmatic_generation_checks_only_the_generation_model() -> None:
    from evaluatorq.simulation import generate

    with pytest.raises(ModelClientMismatchError, match=r'\(sim\.generator\)=openai/') as info:
        await generate(agent_description='A support agent.')
    assert 'EVALUATORQ_SMART_MODEL' not in str(info.value)


def test_a_simulation_checks_only_the_roles_the_environment_serves() -> None:
    from evaluatorq.contracts import LLMCallConfig
    from evaluatorq.simulation._config import require_sim_models_served

    injected = object()
    with pytest.raises(ModelClientMismatchError) as info:
        require_sim_models_served(None, generation_client=injected, user_simulator=injected, judge=None)
    assert 'EVALUATORQ_SMART_MODEL' in str(info.value)
    assert 'EVALUATORQ_FAST_MODEL' not in str(info.value)

    require_sim_models_served(None, generation_client=injected, user_simulator=injected, judge=injected)
    require_sim_models_served(LLMCallConfig(model='claude-sonnet-5-5'), generation_client=None)
    client = AsyncOpenAI(api_key='sk-test', max_retries=0)
    require_sim_models_served(LLMCallConfig(client=client), generation_client=None)


def test_a_simulation_without_generation_does_not_check_the_generator() -> None:
    from evaluatorq.simulation._config import require_sim_models_served

    agent = object()
    require_sim_models_served(None, generation_client=None, user_simulator=agent, judge=agent, generating=False)
    with pytest.raises(ModelClientMismatchError, match=r'\(sim\.generator\)'):
        require_sim_models_served(None, generation_client=None, user_simulator=agent, judge=agent)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('kwargs', 'generating'),
    [
        ({'personas': [], 'scenarios': []}, True),
        ({'datapoints': []}, True),
        ({'datapoints': [], 'executive_summary': False, 'recommendations': False}, False),
    ],
)
async def test_simulate_checks_the_generator_only_when_the_run_uses_it(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, Any], generating: bool
) -> None:
    from evaluatorq.simulation import api, simulate

    seen: dict[str, Any] = {}

    class _Stop(Exception):
        pass

    def capture(*_args: Any, **options: Any) -> None:
        seen.update(options)
        raise _Stop

    monkeypatch.setattr(api, 'require_sim_models_served', capture)
    with pytest.raises(_Stop):
        await simulate(target=lambda messages: 'ok', upload_results=False, **kwargs)
    assert seen['generating'] is generating
