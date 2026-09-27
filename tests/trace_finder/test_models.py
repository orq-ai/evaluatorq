"""Public contract tests for trace-finder models and validation."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from evaluatorq.common.judge import ClassifyQuestion
from evaluatorq.trace_finder.models import (
    FACET_NAMES,
    CompiledQuery,
    FacetCatalogue,
    FacetSelection,
    NumericFilters,
    PopulationRequest,
    Snapshot,
    TraceRecord,
    ValueSelection,
)


def choice_task(*, state: dict[str, object] | None = None) -> dict[str, object]:
    return {
        'kind': 'choice',
        'instructions': 'Classify the support need.',
        'criteria': {'billing': 'Billing help.', 'technical': 'Technical help.'},
        'state': {} if state is None else state,
    }


def test_facet_catalogue_and_selection_include_the_eight_facets_in_order() -> None:
    assert FACET_NAMES == (
        'project',
        'model',
        'provider',
        'status',
        'product',
        'trace_type',
        'agent_name',
        'tool_name',
    )
    assert FacetSelection.model_fields.keys() >= set(FACET_NAMES)
    assert FacetCatalogue.model_fields.keys() >= set(FACET_NAMES)


def test_trace_record_accepts_new_optional_trace_metadata() -> None:
    trace = TraceRecord(
        schema_version=1,
        trace_id='trace-1',
        span_id='span-1',
        timestamp=datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc),
        messages=({'role': 'user', 'content': 'hello'},),
        project='alpha',
        model='gpt-5',
        provider='openai',
        status='completed',
        product='chat',
        trace_type='conversation',
        agent_name='support-agent',
        tool_names=('lookup_order',),
        total_tokens=120,
        duration_ms=42,
    )

    assert trace.agent_name == 'support-agent'
    assert trace.tool_names == ('lookup_order',)
    assert trace.total_tokens == 120
    assert trace.duration_ms == 42


def test_numeric_filters_reject_a_minimum_above_its_maximum() -> None:
    with pytest.raises(ValidationError, match='tokens_min must be less than or equal to tokens_max'):
        NumericFilters(tokens_min=10, tokens_max=5)


def test_numeric_filters_accept_independent_ranges() -> None:
    filters = NumericFilters(tokens_min=10, duration_ms_max=500)
    assert filters.tokens_min == 10
    assert filters.duration_ms_max == 500
    assert PopulationRequest().numeric == NumericFilters()


def test_population_request_accepts_configured_limit_above_500() -> None:
    assert PopulationRequest(limit=5000).limit == 5000
    with pytest.raises(ValidationError):
        PopulationRequest(limit=5001)


def test_population_request_rejects_inverted_time_range() -> None:
    with pytest.raises(ValidationError, match='start must not be after end'):
        PopulationRequest(
            start=datetime(2026, 9, 21, tzinfo=timezone.utc),
            end=datetime(2026, 9, 20, tzinfo=timezone.utc),
        )


def test_snapshot_has_no_file_path_or_filename_contract() -> None:
    snapshot = Snapshot(traces=(), capture_metadata={'source': 'orq'})
    assert snapshot.traces == ()
    assert snapshot.capture_metadata == {'source': 'orq'}
    assert 'path' not in Snapshot.model_fields
    assert 'filename' not in Snapshot.model_fields


@pytest.mark.parametrize(
    'document',
    [
        {
            'task': choice_task(),
            'selection': {'kind': 'values', 'values': ['billing']},
        },
        {
            'task': {
                'kind': 'noul',
                'instructions': 'Does the customer ask to cancel?',
                'state': {},
            },
            'selection': {'kind': 'values', 'values': [True]},
        },
        {
            'task': {
                'kind': 'score',
                'instructions': 'Score the refund request.',
                'criteria': ['No request.', 'Explicit request.'],
                'state': {},
            },
            'selection': {'kind': 'threshold', 'operator': 'gte', 'value': 0.75},
        },
    ],
)
def test_compiled_query_accepts_each_supported_classifier_task(document: dict[str, object]) -> None:
    compiled = CompiledQuery.model_validate(document)
    assert isinstance(compiled.task, ClassifyQuestion)
    assert compiled.task.state == {}


@pytest.mark.parametrize(
    ('document', 'location'),
    [
        (
            {'task': choice_task(), 'selection': {'kind': 'values', 'values': ['unknown']}},
            ('selection', 'values'),
        ),
        (
            {
                'task': {'kind': 'noul', 'instructions': 'Cancel?', 'state': {}},
                'selection': {'kind': 'values', 'values': ['true']},
            },
            ('selection', 'values'),
        ),
        (
            {'task': choice_task(), 'selection': {'kind': 'threshold', 'operator': 'gte', 'value': 0.5}},
            ('selection',),
        ),
        (
            {
                'task': {
                    'kind': 'score',
                    'instructions': 'Score.',
                    'criteria': ['low', 'high'],
                    'state': {},
                },
                'selection': {'kind': 'values', 'values': ['match']},
            },
            ('selection',),
        ),
        (
            {
                'task': {
                    'kind': 'score',
                    'instructions': 'Score.',
                    'criteria': ['low', 'high'],
                    'state': {},
                },
                'selection': {'kind': 'threshold', 'operator': 'gte', 'value': 1.1},
            },
            ('selection', 'threshold', 'value'),
        ),
        (
            {
                'task': {
                    'kind': 'choice',
                    'instructions': 'Choose.',
                    'criteria': {'one': 'One.', 'two': 'Two.', 'three': 'Three.', 'four': 'Four.', 'five': 'Five.', 'six': 'Six.'},
                    'state': {},
                },
                'selection': {'kind': 'values', 'values': ['one']},
            },
            ('task', 'criteria'),
        ),
        (
            {
                'task': {
                    'kind': 'choice',
                    'instructions': 'Choose.',
                    'criteria': {'only': 'Only choice.'},
                    'state': {},
                },
                'selection': {'kind': 'values', 'values': ['only']},
            },
            ('task', 'criteria'),
        ),
    ],
)
def test_compiled_query_reports_actionable_validation_locations(
    document: dict[str, object], location: tuple[str, ...]
) -> None:
    with pytest.raises(ValidationError) as error:
        CompiledQuery.model_validate(document)

    assert error.value.errors()[0]['loc'] == location


def test_compiled_query_rejects_non_empty_classifier_state() -> None:
    with pytest.raises(ValidationError) as error:
        CompiledQuery(
            task=ClassifyQuestion.model_validate(choice_task(state={'model_filter': 'gpt-5'})),
            selection=ValueSelection(kind='values', values=('billing',)),
        )

    assert error.value.errors()[0]['loc'] == ('task', 'state')
