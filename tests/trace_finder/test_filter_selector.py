"""Tests for catalogue-backed classifier metadata selection."""

from __future__ import annotations

import json
from typing import Any, cast

import httpx
import pytest
from loguru import logger
from openai import AsyncOpenAI

from evaluatorq.trace_finder import filter_selector as filter_selector_module
from evaluatorq.common.judge import ClassifyAnswer, ClassifyOutcome, ClassifyResponse, JudgeError
from evaluatorq.trace_finder.filter_selector import FilterSelectionError, select_filters, select_filters_with_response
from evaluatorq.trace_finder.debug import cli_debug
from evaluatorq.trace_finder.models import FACET_NAMES, FILTER_OR_JUDGMENT_RULE, FacetCatalogue, FacetSelection


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
@pytest.mark.parametrize(
    ('facet', 'value'),
    [
        ('model', 'gpt-6-sol'),
        ('model', 'openai/gpt-6-sol'),
        ('provider', 'openai'),
        ('agent_name', 'Support bot'),
        ('tool_name', 'lookup.customer'),
        ('status', 'error'),
    ],
)
async def test_filter_selection_accepts_exact_catalogue_values_from_classify(facet: str, value: str) -> None:
    bodies: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={'answers': {facet: {'type': 'choice', 'choice': value}}},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
        async with AsyncOpenAI(api_key='test', base_url='https://router.test/v1', http_client=http_client) as client:
            result = await select_filters_with_response(
                client=client,
                model='typesafe/jev-latest',
                catalogue=FacetCatalogue.model_validate({facet: (value,)}),
                query=f'{facet} {value}',
            )

    assert getattr(result.selection, facet) == frozenset({value})
    assert result.response is not None
    assert result.response.answers[facet].choice == value
    assert value in bodies[0]['questions'][facet]['criteria']


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('choice', 'selected'),
    [('none', frozenset({'none'})), ('none_', frozenset({'none_'})), ('none__', frozenset())],
)
async def test_no_filter_choice_does_not_hide_literal_none_values(choice: str, selected: frozenset[str]) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={'answers': {'model': {'type': 'choice', 'choice': choice}}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
        async with AsyncOpenAI(api_key='test', base_url='https://router.test/v1', http_client=http_client) as client:
            result = await select_filters_with_response(
                client=client,
                model='typesafe/jev-latest',
                catalogue=FacetCatalogue(model=('none', 'none_')),
                query=f'model {choice}' if selected else 'any model',
            )

    assert result.selection.model == selected


@pytest.mark.asyncio
async def test_filter_request_and_response_are_debug_only(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_classify(**kwargs: Any) -> ClassifyOutcome:
        return ClassifyOutcome(
            response=ClassifyResponse(answers={'project': ClassifyAnswer(type='choice', choice='Demos')})
        )

    monkeypatch.setattr(filter_selector_module, 'run_classify', fake_run_classify)
    monkeypatch.delenv('EVALUATORQ_LOG_LEVEL', raising=False)
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), format='{message}', level='DEBUG')
    try:
        await select_filters_with_response(cast(Any, object()), 'typesafe/jev-latest', FacetCatalogue(project=('Demos',)), 'Demos')
        assert not any('Trace finder filter classify request' in message for message in messages)
        with cli_debug(active=True):
            await select_filters_with_response(
                cast(Any, object()), 'typesafe/jev-latest', FacetCatalogue(project=('Demos',)), 'Demos'
            )
    finally:
        logger.remove(sink_id)

    request_line = next(message for message in messages if 'Trace finder filter classify request' in message)
    request_body = json.loads(request_line.split(' input=', 1)[1])
    assert request_body['state'] == {'query': 'Demos', 'rule': FILTER_OR_JUDGMENT_RULE}
    assert request_body['questions']['project']['type'] == 'choice'
    assert any('Trace finder filter classify response' in message and 'Demos' in message for message in messages)


@pytest.mark.asyncio
async def test_select_filters_uses_one_classify_request_for_non_empty_facets(monkeypatch) -> None:
    captured: list[Any] = []

    async def fake_run_classify(**kwargs: Any) -> ClassifyOutcome:
        captured.append(kwargs)
        request = kwargs['request']
        answers = {
            name: ClassifyAnswer(type='choice', choice={
                'project': 'Demos',
                'provider': 'openai',
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
    assert request.state == {'query': 'OpenAI errors in Demos', 'rule': FILTER_OR_JUDGMENT_RULE}
    assert set(request.questions) == {str(name) for name in FACET_NAMES} - {'agent_name', 'tool_name'}
    assert all(question.kind == 'choice' for question in request.questions.values())


@pytest.mark.asyncio
async def test_filter_selector_retains_the_structured_classify_response(monkeypatch) -> None:
    response = ClassifyResponse(
        model='jev-latest',
        answers={'project': ClassifyAnswer(type='choice', choice='Demos', confidence=0.8)},
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
    assert result.response.answers['project'].choice == 'Demos'
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
@pytest.mark.asyncio
async def test_llm_selector_returns_only_catalogue_values_and_numeric_filters(monkeypatch) -> None:
    from datetime import UTC, datetime

    from evaluatorq.common.structured_output import StructuredResult
    from evaluatorq.trace_finder.models import NumericFilters

    calls: list[dict[str, Any]] = []

    async def fake_generate_structured(client: object, **kwargs: Any) -> Any:
        calls.append(kwargs)
        response_type = kwargs['response_format']
        return StructuredResult(
            parsed=response_type.model_validate({
                'project': ['Demos'],
                'model': ['none'],
                'provider': ['none'],
                'status': ['none'],
                'product': ['none'],
                'trace_type': ['none'],
                'agent_name': ['none'],
                'tool_name': ['none'],
                'numeric': {
                    'tokens_min': 20_000,
                    'tokens_max': None,
                    'duration_ms_min': None,
                    'duration_ms_max': None,
                },
                'start': '2025-01-01T00:00:00Z',
                'end': '2025-01-02T00:00:00Z',
            }),
            raw='',
        )

    monkeypatch.setattr(filter_selector_module, 'generate_structured', fake_generate_structured)
    result = await filter_selector_module.select_llm_filters_with_response(
        cast(Any, object()),
        'fast-model',
        FacetCatalogue(project=('Demos', 'Other')),
        'Demos over 20k tokens',
        now=datetime(2025, 1, 1, tzinfo=UTC),
    )

    assert result.selection.project == frozenset({'Demos'})
    assert result.numeric == NumericFilters(tokens_min=20_000)
    assert result.time is not None
    assert result.time.start.isoformat() == '2025-01-01T00:00:00+00:00'
    assert result.time.end.isoformat() == '2025-01-02T00:00:00+00:00'
    assert result.usage is None
    assert calls[0]['api'] == 'responses'

@pytest.mark.asyncio
async def test_llm_selector_rejects_values_outside_the_catalogue(monkeypatch) -> None:
    from datetime import UTC, datetime

    calls = 0

    async def fake_generate_structured(client: object, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return type('Result', (), {
            'parsed': {
                'project': ['invented'],
                'model': ['none'],
                'provider': ['none'],
                'status': ['none'],
                'product': ['none'],
                'trace_type': ['none'],
                'agent_name': ['none'],
                'tool_name': ['none'],
                'numeric': {
                    'tokens_min': None,
                    'tokens_max': None,
                    'duration_ms_min': None,
                    'duration_ms_max': None,
                },
                'start': None,
                'end': None,
            },
            'raw': 'invented',
            'usage': None,
        })()

    monkeypatch.setattr(filter_selector_module, 'generate_structured', fake_generate_structured)
    with pytest.raises(FilterSelectionError, match='invalid plan'):
        await filter_selector_module.select_llm_filters_with_response(
            cast(Any, object()),
            'fast-model',
            FacetCatalogue(project=('Demos',)),
            'Demos',
            now=datetime(2025, 1, 1, tzinfo=UTC),
        )
    assert calls == 2
