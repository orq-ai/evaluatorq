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
