"""Tests for catalogue-backed classifier metadata selection."""

from __future__ import annotations

from typing import Any, cast

import pytest

from evaluatorq.trace_finder import filter_selector as filter_selector_module
from evaluatorq.common.judge import ClassifyAnswer, ClassifyOutcome, ClassifyResponse, JudgeError
from evaluatorq.trace_finder.filter_selector import FilterSelectionError, select_filters, select_filters_with_response
from evaluatorq.trace_finder.models import FACET_NAMES, FacetCatalogue, FacetSelection


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
async def test_filter_selector_retains_the_structured_classify_response(monkeypatch) -> None:
    response = ClassifyResponse(
        model='jev-latest',
        answers={'project': ClassifyAnswer(type='choice', choice='project_0', confidence=0.8)},
    )

    async def fake_run_classify(**kwargs: Any) -> ClassifyOutcome:
        request = kwargs['request']
        answers = {
            name: response.answers['project'] if name == 'project' else ClassifyAnswer(type='choice', choice='none')
            for name in request.questions
        }
        return ClassifyOutcome(response=response.model_copy(update={'answers': answers}))

    monkeypatch.setattr(filter_selector_module, 'run_classify', fake_run_classify)

    result = await select_filters_with_response(cast(Any, object()), 'typesafe/jev-latest', catalogue(), 'Demos')

    assert result.selection.project == frozenset({'Demos'})
    assert result.response is not None
    assert result.response.model == 'jev-latest'
    assert result.response.answers['project'].choice == 'project_0'
    assert result.response.answers['project'].confidence == 0.8


@pytest.mark.asyncio
async def test_select_filters_keeps_multiple_explicit_project_values_in_one_request(monkeypatch) -> None:
    calls: list[Any] = []

    async def fake_run_classify(**kwargs: Any) -> ClassifyOutcome:
        calls.append(kwargs['request'])
        answers = {
            name: ClassifyAnswer(type='noul', noul=0.9)
            if question.kind == 'noul'
            else ClassifyAnswer(type='choice', choice='none')
            for name, question in kwargs['request'].questions.items()
        }
        return ClassifyOutcome(response=ClassifyResponse(answers=answers))

    monkeypatch.setattr(filter_selector_module, 'run_classify', fake_run_classify)
    available = FacetCatalogue(project=('A', 'B', 'C'), provider=('openai',))

    selected = await select_filters(cast(Any, object()), 'typesafe/jev-latest', available, 'project A or project B')

    assert selected.project == frozenset({'A', 'B'})
    assert len(calls) == 1
    assert set(calls[0].questions) == {'project_0', 'project_1', 'provider'}


@pytest.mark.asyncio
async def test_select_filters_rejects_a_request_exceeding_question_limit(monkeypatch) -> None:
    async def should_not_call(**kwargs: Any) -> ClassifyOutcome:
        raise AssertionError('oversized classify request should not be sent')

    monkeypatch.setattr(filter_selector_module, 'run_classify', should_not_call)
    values = tuple(f'value_{index}' for index in range(50))
    available = FacetCatalogue(project=values, model=values, provider=values)
    query = ' '.join(values)

    with pytest.raises(FilterSelectionError, match='maximum is 100'):
        await select_filters(cast(Any, object()), 'typesafe/jev-latest', available, query)


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


@pytest.mark.asyncio
async def test_select_filters_retries_a_transient_classify_failure(monkeypatch) -> None:
    calls = 0

    async def fake_run_classify(**kwargs: Any) -> ClassifyOutcome:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError('transient')
        request = kwargs['request']
        return ClassifyOutcome(
            response=ClassifyResponse(
                answers={name: ClassifyAnswer(type='choice', choice='none') for name in request.questions}
            )
        )

    async def fake_with_retry(fn: Any, **kwargs: Any) -> Any:
        del kwargs
        try:
            return await fn()
        except RuntimeError:
            return await fn()

    monkeypatch.setattr(filter_selector_module, 'run_classify', fake_run_classify)
    monkeypatch.setattr(filter_selector_module, 'with_retry', fake_with_retry)

    selected = await select_filters(cast(Any, object()), 'typesafe/jev-latest', catalogue(), 'find errors')

    assert calls == 2
    assert selected == FacetSelection()


@pytest.mark.asyncio
async def test_select_filters_turns_exhausted_retries_into_domain_error(monkeypatch) -> None:
    async def fake_run_classify(**kwargs: Any) -> ClassifyOutcome:
        del kwargs
        raise RuntimeError('transient exhausted')

    async def fake_with_retry(fn: Any, **kwargs: Any) -> Any:
        del kwargs
        return await fn()

    monkeypatch.setattr(filter_selector_module, 'run_classify', fake_run_classify)
    monkeypatch.setattr(filter_selector_module, 'with_retry', fake_with_retry)

    with pytest.raises(FilterSelectionError, match='transient exhausted'):
        await select_filters(cast(Any, object()), 'typesafe/jev-latest', catalogue(), 'find errors')
