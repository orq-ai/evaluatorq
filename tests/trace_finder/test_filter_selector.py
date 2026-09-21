"""Tests for catalogue-backed JEV metadata selection."""

from __future__ import annotations

from typing import Any, cast

import pytest

from evaluatorq.common.judge import ClassifyAnswer, ClassifyOutcome, ClassifyResponse, JudgeError
from evaluatorq.trace_finder.filter_selector import FilterSelectionError, select_filters
from evaluatorq.trace_finder.models import FACET_NAMES, FacetCatalogue


def catalogue() -> FacetCatalogue:
    return FacetCatalogue(
        project=('Demos',),
        model=('gpt-5.6-luna',),
        provider=('openai',),
        status=('error', 'ok'),
        product=('agents',),
        trace_type=('span.responses',),
        agent_name=(),
        tool_name=(),
    )


@pytest.mark.asyncio
async def test_select_filters_uses_one_classify_request_for_non_empty_facets(monkeypatch) -> None:
    captured: list[Any] = []

    async def fake_run_classify(**kwargs: Any) -> ClassifyOutcome:
        captured.append(kwargs)
        request = kwargs['request']
        answers = {
            name: ClassifyAnswer(type='choice', choice={
                'project': 'project_0',
                'provider': 'provider_0',
                'status': 'none',
                'model': 'none',
                'product': 'none',
                'trace_type': 'none',
            }.get(name, 'none'))
            for name in request.questions
        }
        return ClassifyOutcome(response=ClassifyResponse(answers=answers))

    monkeypatch.setattr('evaluatorq.trace_finder.filter_selector.run_classify', fake_run_classify)

    selected = await select_filters(cast(Any, object()), 'typesafe/jev-latest', catalogue(), ' OpenAI errors in Demos ')

    assert selected.project == frozenset({'Demos'})
    assert selected.provider == frozenset({'openai'})
    assert selected.status == frozenset()
    assert selected.agent_name == frozenset()
    assert len(captured) == 1
    request = captured[0]['request']
    assert request.state == {'query': 'OpenAI errors in Demos'}
    assert set(request.questions) == {str(name) for name in FACET_NAMES} - {'agent_name', 'tool_name'}
    assert all(question.kind == 'choice' for question in request.questions.values())
    assert 'span.responses' in str(request.questions['trace_type'].criteria)


@pytest.mark.asyncio
async def test_select_filters_turns_classify_errors_into_filter_selection_error(monkeypatch) -> None:
    async def fake_run_classify(**kwargs: Any) -> ClassifyOutcome:
        return ClassifyOutcome(error_kind=JudgeError.PARSE, error_message='malformed classify reply')

    monkeypatch.setattr('evaluatorq.trace_finder.filter_selector.run_classify', fake_run_classify)

    with pytest.raises(FilterSelectionError, match='malformed classify reply'):
        await select_filters(cast(Any, object()), 'typesafe/jev-latest', catalogue(), 'find errors')


@pytest.mark.asyncio
async def test_select_filters_rejects_a_choice_outside_the_catalogue(monkeypatch) -> None:
    async def fake_run_classify(**kwargs: Any) -> ClassifyOutcome:
        request = kwargs['request']
        answers = {
            name: ClassifyAnswer(type='choice', choice='invented') if name == 'project'
            else ClassifyAnswer(type='choice', choice='none')
            for name in request.questions
        }
        return ClassifyOutcome(response=ClassifyResponse(answers=answers))

    monkeypatch.setattr('evaluatorq.trace_finder.filter_selector.run_classify', fake_run_classify)

    with pytest.raises(FilterSelectionError, match='unavailable project choice'):
        await select_filters(cast(Any, object()), 'typesafe/jev-latest', catalogue(), 'invent a project')


@pytest.mark.asyncio
async def test_select_filters_rejects_empty_query(monkeypatch) -> None:
    async def should_not_run(**kwargs: Any) -> ClassifyOutcome:
        raise AssertionError('classify should not run')

    monkeypatch.setattr('evaluatorq.trace_finder.filter_selector.run_classify', should_not_run)

    with pytest.raises(ValueError, match='Enter a semantic trace query'):
        await select_filters(cast(Any, object()), 'typesafe/jev-latest', catalogue(), ' \n\t ')
