"""Tests for strict semantic query compilation."""

from __future__ import annotations

import json

from typing import Any, cast

import pytest
from loguru import logger
from pydantic import ValidationError

from evaluatorq.trace_finder.compiler import (
    COMPILER_INSTRUCTIONS,
    CompileError,
    CompilerWireQuery,
    classification_legend,
    compile_query,
)
from evaluatorq.trace_finder.models import FILTER_OR_JUDGMENT_RULE, CompiledQuery, ValueSelection
from evaluatorq.trace_finder.models import MAX_DIMENSIONS
from evaluatorq.trace_finder.debug import cli_debug


def choice_document() -> dict[str, Any]:
    return {
        'dimensions': [
            {
                'name': 'Support need',
                'task': {
                    'kind': 'choice',
                    'instructions': 'Classify the support need.',
                    'choice_criteria': [
                        {'label': 'billing', 'description': 'Billing help.'},
                        {'label': 'technical', 'description': 'Technical help.'},
                    ],
                    'score_criteria': None,
                    'noul_threshold': 0.5,
                },
                'selection': {'kind': 'values', 'values': ['billing']},
            }
        ],
        'numeric': {
            'tokens_min': 20_000,
            'tokens_max': None,
            'duration_ms_min': None,
            'duration_ms_max': None,
        },
        'unsupported_reason': None,
    }


@pytest.mark.parametrize('label', ['yes', 'true', 'True'])
def test_noul_text_selection_is_normalized_to_booleans(label: str) -> None:
    document = choice_document()
    document['dimensions'][0]['task'].update(kind='noul', choice_criteria=None)
    document['dimensions'][0]['selection'] = {'kind': 'values', 'values': [label]}

    dimensions, _ = CompilerWireQuery.model_validate(document).to_domain()
    compiled = dimensions[0]

    assert compiled.task.kind == 'noul'
    assert isinstance(compiled.selection, ValueSelection)
    assert compiled.selection.values == (True,)


class FakeStructuredResult:
    def __init__(self, parsed: object | None, raw: str = '') -> None:
        self.parsed = parsed
        self.raw = raw


@pytest.mark.asyncio
async def test_compile_query_uses_shared_structured_output_and_returns_numeric_filters(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_generate_structured(client: object, **kwargs: Any) -> FakeStructuredResult:
        calls.append({'client': client, **kwargs})
        return FakeStructuredResult(CompilerWireQuery.model_validate(choice_document()))

    monkeypatch.setattr('evaluatorq.trace_finder.compiler.generate_structured', fake_generate_structured)

    client = cast(Any, object())
    plan = await compile_query(client, 'compiler-model', '  find long billing requests  ')

    assert plan.dimensions[0].task.kind == 'choice'
    assert plan.numeric.tokens_min == 20_000
    assert plan.numeric.duration_ms_min is None
    assert calls[0]['client'] is client
    assert calls[0]['model'] == 'compiler-model'
    assert calls[0]['messages'] == [
        {'role': 'system', 'content': COMPILER_INSTRUCTIONS},
        {'role': 'user', 'content': 'find long billing requests'},
    ]
    assert calls[0]['label'] == 'trace_finder.compile'
    assert calls[0]['max_tokens'] == 2000
    assert calls[0]['api'] == 'responses'


@pytest.mark.asyncio
async def test_compile_query_retries_all_labels_selection(monkeypatch) -> None:
    invalid = choice_document()
    invalid['dimensions'][0]['selection']['values'] = ['billing', 'technical']
    valid = choice_document()
    responses = iter([invalid, valid])
    calls: list[dict[str, Any]] = []

    async def fake_generate_structured(client: object, **kwargs: Any) -> FakeStructuredResult:
        calls.append(kwargs)
        return FakeStructuredResult(CompilerWireQuery.model_validate(next(responses)), raw=json.dumps(invalid))

    monkeypatch.setattr('evaluatorq.trace_finder.compiler.generate_structured', fake_generate_structured)
    plan = await compile_query(cast(Any, object()), 'compiler-model', 'find billing requests')

    selection = plan.dimensions[0].selection
    assert isinstance(selection, ValueSelection)
    assert selection.values == ('billing',)
    assert len(calls) == 2
    assert calls[1]['messages'][-2]['role'] == 'assistant'
    assert 'selection must list only' in calls[1]['messages'][-1]['content']


def test_compiler_wire_rejects_both_noul_values() -> None:
    document = choice_document()
    document['dimensions'][0]['task'].update(kind='noul', choice_criteria=None)
    document['dimensions'][0]['selection'] = {'kind': 'values', 'values': [True, False]}

    with pytest.raises(ValueError, match="dimension 'Support need'.*not both true and false"):
        CompilerWireQuery.model_validate(document).to_domain()


def test_compiler_wire_rejects_whitespace_dimension_names() -> None:
    document = choice_document()
    document['dimensions'][0]['name'] = ' \t '

    with pytest.raises(ValidationError, match='dimension name must not be blank'):
        CompilerWireQuery.model_validate(document)


def test_compiler_wire_rejects_duplicate_dimension_names() -> None:
    document = choice_document()
    second = json.loads(json.dumps(document['dimensions'][0]))
    second['name'] = ' support need '
    document['dimensions'].append(second)

    with pytest.raises(ValueError, match='dimension names must be unique'):
        CompilerWireQuery.model_validate(document).to_domain()


def test_compiler_wire_rejects_more_than_maximum_dimensions() -> None:
    document = choice_document()
    document['dimensions'] = [
        {**document['dimensions'][0], 'name': f'Dimension {index}'} for index in range(MAX_DIMENSIONS + 1)
    ]

    with pytest.raises(ValueError, match=f'maximum is {MAX_DIMENSIONS}'):
        CompilerWireQuery.model_validate(document).to_domain()


@pytest.mark.asyncio
async def test_compile_query_raises_after_two_invalid_plans(monkeypatch) -> None:
    invalid = choice_document()
    invalid['dimensions'][0]['selection']['values'] = ['billing', 'technical']
    calls = 0

    async def fake_generate_structured(client: object, **kwargs: Any) -> FakeStructuredResult:
        nonlocal calls
        calls += 1
        return FakeStructuredResult(CompilerWireQuery.model_validate(invalid), raw=json.dumps(invalid))

    monkeypatch.setattr('evaluatorq.trace_finder.compiler.generate_structured', fake_generate_structured)

    with pytest.raises(CompileError, match='invalid plan.*selection must list only'):
        await compile_query(cast(Any, object()), 'compiler-model', 'find billing requests')
    assert calls == 2


@pytest.mark.asyncio
async def test_compile_query_retries_too_many_dimensions(monkeypatch) -> None:
    invalid = choice_document()
    invalid['dimensions'] = [
        {**invalid['dimensions'][0], 'name': f'Dimension {index}'} for index in range(MAX_DIMENSIONS + 1)
    ]
    valid = choice_document()
    responses = iter([invalid, valid])
    calls: list[dict[str, Any]] = []

    async def fake_generate_structured(client: object, **kwargs: Any) -> FakeStructuredResult:
        calls.append(kwargs)
        response = next(responses)
        return FakeStructuredResult(CompilerWireQuery.model_validate(response), raw=json.dumps(response))

    monkeypatch.setattr('evaluatorq.trace_finder.compiler.generate_structured', fake_generate_structured)
    plan = await compile_query(cast(Any, object()), 'compiler-model', 'find billing requests')

    assert len(plan.dimensions) == 1
    assert len(calls) == 2
    assert f'maximum is {MAX_DIMENSIONS}' in calls[1]['messages'][-1]['content']


def test_compiler_prompt_explains_match_selection() -> None:
    assert 'only answers that satisfy the request' in COMPILER_INSTRUCTIONS


@pytest.mark.asyncio
async def test_compiler_request_and_output_are_debug_only(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_generate_structured(client: object, **kwargs: Any) -> FakeStructuredResult:
        return FakeStructuredResult(CompilerWireQuery.model_validate(choice_document()))

    monkeypatch.setattr('evaluatorq.trace_finder.compiler.generate_structured', fake_generate_structured)
    monkeypatch.delenv('EVALUATORQ_LOG_LEVEL', raising=False)
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), format='{message}', level='DEBUG')
    try:
        await compile_query(cast(Any, object()), 'compiler-model', 'find billing requests')
        assert not any('Trace finder compiler request' in message for message in messages)
        with cli_debug(active=True):
            await compile_query(cast(Any, object()), 'compiler-model', 'find billing requests')
    finally:
        logger.remove(sink_id)

    assert any('Trace finder compiler request' in message and 'find billing requests' in message for message in messages)
    assert any('Trace finder compiler response' in message and 'billing' in message for message in messages)


@pytest.mark.asyncio
async def test_compile_query_reports_truncated_raw_output_when_structured_output_is_absent(monkeypatch) -> None:
    async def fake_generate_structured(client: object, **kwargs: Any) -> FakeStructuredResult:
        return FakeStructuredResult(None, 'raw compiler output ' * 100)

    monkeypatch.setattr('evaluatorq.trace_finder.compiler.generate_structured', fake_generate_structured)

    with pytest.raises(CompileError, match='raw compiler output') as error:
        await compile_query(cast(Any, object()), 'compiler-model', 'find requests')

    assert len(str(error.value).split('Raw: ', 1)[1]) <= 300


@pytest.mark.asyncio
async def test_compile_query_preserves_duration_numeric_bounds(monkeypatch) -> None:
    document = choice_document()
    document['numeric'] = {
        'tokens_min': None,
        'tokens_max': 40_000,
        'duration_ms_min': 30_000,
        'duration_ms_max': 120_000,
    }

    async def fake_generate_structured(client: object, **kwargs: Any) -> FakeStructuredResult:
        return FakeStructuredResult(CompilerWireQuery.model_validate(document))

    monkeypatch.setattr('evaluatorq.trace_finder.compiler.generate_structured', fake_generate_structured)

    plan = await compile_query(cast(Any, object()), 'compiler-model', 'slow requests under forty thousand tokens')

    assert plan.numeric.tokens_max == 40_000
    assert plan.numeric.duration_ms_min == 30_000
    assert plan.numeric.duration_ms_max == 120_000


@pytest.mark.asyncio
async def test_compile_query_enforces_strict_token_and_duration_minima(monkeypatch) -> None:
    document = choice_document()
    document['numeric']['duration_ms_min'] = 30_000

    async def fake_generate_structured(client: object, **kwargs: Any) -> FakeStructuredResult:
        return FakeStructuredResult(CompilerWireQuery.model_validate(document))

    monkeypatch.setattr('evaluatorq.trace_finder.compiler.generate_structured', fake_generate_structured)

    plan = await compile_query(cast(Any, object()), 'compiler-model', 'over 20k tokens and slower than 30 seconds')

    assert plan.numeric.tokens_min == 20_001
    assert plan.numeric.duration_ms_min == 30_001


@pytest.mark.asyncio
async def test_compile_query_enforces_strict_token_and_duration_maxima(monkeypatch) -> None:
    document = choice_document()
    document['numeric'] = {
        'tokens_min': None,
        'tokens_max': 20_000,
        'duration_ms_min': None,
        'duration_ms_max': 30_000,
    }

    async def fake_generate_structured(client: object, **kwargs: Any) -> FakeStructuredResult:
        return FakeStructuredResult(CompilerWireQuery.model_validate(document))

    monkeypatch.setattr('evaluatorq.trace_finder.compiler.generate_structured', fake_generate_structured)

    plan = await compile_query(cast(Any, object()), 'compiler-model', 'under 20k tokens and less than 30 seconds')

    assert plan.numeric.tokens_max == 19_999
    assert plan.numeric.duration_ms_max == 29_999


@pytest.mark.parametrize(
    ('query', 'bounds'),
    [
        ('under 0 tokens', {'tokens_max': 0}),
        ('over 10 tokens and under 5 tokens', {'tokens_min': 10, 'tokens_max': 10}),
    ],
)
@pytest.mark.asyncio
async def test_compile_query_reports_impossible_strict_bounds(monkeypatch, query: str, bounds: dict[str, int]) -> None:
    document = choice_document()
    document['numeric'] = {
        'tokens_min': None,
        'tokens_max': None,
        'duration_ms_min': None,
        'duration_ms_max': None,
        **bounds,
    }

    async def fake_generate_structured(client: object, **kwargs: Any) -> FakeStructuredResult:
        return FakeStructuredResult(CompilerWireQuery.model_validate(document))

    monkeypatch.setattr('evaluatorq.trace_finder.compiler.generate_structured', fake_generate_structured)

    with pytest.raises(CompileError, match='Compiler produced an invalid plan'):
        await compile_query(cast(Any, object()), 'compiler-model', query)


@pytest.mark.asyncio
async def test_compile_query_does_not_turn_conversation_wait_time_into_trace_duration(monkeypatch) -> None:
    document = choice_document()
    document['numeric'] = {
        'tokens_min': None,
        'tokens_max': None,
        'duration_ms_min': None,
        'duration_ms_max': None,
    }

    async def fake_generate_structured(client: object, **kwargs: Any) -> FakeStructuredResult:
        return FakeStructuredResult(CompilerWireQuery.model_validate(document))

    monkeypatch.setattr('evaluatorq.trace_finder.compiler.generate_structured', fake_generate_structured)

    plan = await compile_query(
        cast(Any, object()), 'compiler-model', 'find conversations mentioning a customer waiting over 30 seconds'
    )

    assert plan.numeric.duration_ms_min is None


@pytest.mark.asyncio
async def test_compile_query_tightens_fractional_duration_maximum(monkeypatch) -> None:
    document = choice_document()
    document['numeric'] = {
        'tokens_min': None,
        'tokens_max': None,
        'duration_ms_min': None,
        'duration_ms_max': 1_500,
    }

    async def fake_generate_structured(client: object, **kwargs: Any) -> FakeStructuredResult:
        return FakeStructuredResult(CompilerWireQuery.model_validate(document))

    monkeypatch.setattr('evaluatorq.trace_finder.compiler.generate_structured', fake_generate_structured)

    plan = await compile_query(cast(Any, object()), 'compiler-model', 'faster than 1.5 seconds')

    assert plan.numeric.duration_ms_max == 1_499


@pytest.mark.asyncio
async def test_compile_query_rejects_empty_query(monkeypatch) -> None:
    async def should_not_run(client: object, **kwargs: Any) -> FakeStructuredResult:
        raise AssertionError('structured output should not run')

    monkeypatch.setattr('evaluatorq.trace_finder.compiler.generate_structured', should_not_run)

    with pytest.raises(ValueError, match='Enter a semantic trace query'):
        await compile_query(cast(Any, object()), 'compiler-model', ' \n\t ')


def test_compiler_wire_schema_forbids_extra_fields() -> None:
    document = choice_document()
    document['numeric']['unexpected'] = 1

    with pytest.raises(ValidationError):
        CompilerWireQuery.model_validate(document)


def test_compiler_wire_schema_is_strict_mode_compatible() -> None:
    """OpenAI strict structured output rejects `oneOf`, which a discriminated union renders as."""
    schema = json.dumps(CompilerWireQuery.model_json_schema())
    assert 'oneOf' not in schema
    assert 'anyOf' in schema
    assert CompilerWireQuery.model_validate(choice_document()).dimensions[0].selection.kind == 'values'


def test_compiler_wire_selection_schema_forbids_extra_fields() -> None:
    document = choice_document()
    document['dimensions'][0]['selection']['unexpected'] = 1

    with pytest.raises(ValidationError):
        CompilerWireQuery.model_validate(document)


def test_compiler_prompt_keeps_descriptive_phrases_as_dimensions() -> None:
    # "all coding agents with more than 50k tokens" once compiled to zero dimensions, so the
    # filter picker found nothing and every trace over 50k matched, coding or not.
    assert 'coding agents over 50k tokens' in COMPILER_INSTRUCTIONS
    assert 'only when every part of the request' in COMPILER_INSTRUCTIONS


def test_compiler_prompt_carries_the_shared_filter_or_judgment_rule() -> None:
    # "which traces had tool errors?" once became status = error, which emptied the table before the
    # "Tool errors" dimension could run; both planners must read the same rule.
    assert FILTER_OR_JUDGMENT_RULE in COMPILER_INSTRUCTIONS
    assert '"tool errors" -> judgment' in FILTER_OR_JUDGMENT_RULE
    assert 'Aggregate questions' in FILTER_OR_JUDGMENT_RULE


@pytest.mark.asyncio
@pytest.mark.parametrize(('reason', 'expected'), [(None, None), ('  ', None), ('Costs per model.', 'Costs per model.')])
async def test_compile_query_threads_the_unsupported_reason(monkeypatch, reason: str | None, expected: str | None) -> None:
    document = choice_document()
    document['dimensions'] = []
    document['numeric'] = dict.fromkeys(document['numeric'])
    document['unsupported_reason'] = reason

    async def fake_generate_structured(client: object, **kwargs: Any) -> FakeStructuredResult:
        return FakeStructuredResult(CompilerWireQuery.model_validate(document))

    monkeypatch.setattr('evaluatorq.trace_finder.compiler.generate_structured', fake_generate_structured)

    plan = await compile_query(cast(Any, object()), 'compiler-model', 'which model costs the most?')

    assert plan.dimensions == ()
    assert plan.unsupported_reason == expected


def test_compiler_wire_schema_requires_a_nullable_unsupported_reason() -> None:
    schema = CompilerWireQuery.model_json_schema()

    assert 'unsupported_reason' in schema['required']
    document = choice_document()
    del document['unsupported_reason']
    with pytest.raises(ValidationError):
        CompilerWireQuery.model_validate(document)


def test_classification_legend_uses_dashboard_chart_tokens() -> None:
    compiled = CompiledQuery.model_validate({
        'task': {
            'kind': 'choice',
            'instructions': 'Classify.',
            'criteria': {'billing': 'Billing.', 'technical': 'Technical.'},
            'state': {},
        },
        'selection': {'kind': 'values', 'values': ['billing']},
    })

    assert [item.model_dump() for item in classification_legend(compiled)] == [
        {'label': 'billing', 'color': 'var(--chart-5)', 'kind': 'value', 'threshold': None},
        {'label': 'technical', 'color': 'var(--chart-2)', 'kind': 'value', 'threshold': None},
    ]
