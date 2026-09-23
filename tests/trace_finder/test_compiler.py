"""Tests for strict semantic query compilation."""

from __future__ import annotations

import json

from typing import Any, cast

import pytest
from pydantic import ValidationError

from evaluatorq.trace_finder.compiler import (
    COMPILER_INSTRUCTIONS,
    CompileError,
    CompilerWireQuery,
    classification_legend,
    compile_query,
)
from evaluatorq.trace_finder.models import CompiledQuery


def choice_document() -> dict[str, Any]:
    return {
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
        'numeric': {
            'tokens_min': 20_000,
            'tokens_max': None,
            'duration_ms_min': None,
            'duration_ms_max': None,
        },
    }


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

    assert plan.compiled.task.kind == 'choice'
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
    assert CompilerWireQuery.model_validate(choice_document()).selection.kind == 'values'


def test_compiler_wire_selection_schema_forbids_extra_fields() -> None:
    document = choice_document()
    document['selection']['unexpected'] = 1

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
