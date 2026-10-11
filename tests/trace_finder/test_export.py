"""Metadata-only JSON export tests for trace-finder runs."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone

from evaluatorq.trace_finder.export import ExportTrace, RunExport, build_export, export_json
from evaluatorq.trace_finder.models import (
    CompiledQuery,
    DimensionAnswer,
    FacetSelection,
    NumericFilters,
    PopulationRequest,
    RunRequest,
    RunSnapshot,
    TraceClassification,
    TraceRecord,
)


def run() -> RunSnapshot:
    traces = tuple(
        TraceRecord(
            schema_version=1,
            trace_id=f'trace-{index}',
            span_id=f'span-{index}',
            timestamp=datetime(2026, 9, 20, 12, index, tzinfo=timezone.utc),
            messages=({'role': 'user', 'content': 'secret conversation'},),
            project='alpha',
            model='gpt-5',
            provider='openai',
            status='completed',
            product='chat',
            trace_type='conversation',
            agent_name='support',
            tool_names=('lookup', 'refund'),
            capture_metadata={
                'source': 'orq',
                'content': 'secret',
                'population_start': '2026-09-01T00:00:00+00:00',
                'population_end': '2026-09-21T00:00:00+00:00',
            },
        )
        for index in range(3)
    )
    return RunSnapshot(
        generation=42,
        state='completed',
        request=RunRequest(
            query='Find billing requests',
            mode='immediate',
            population=PopulationRequest(
                facets=FacetSelection(
                    project=frozenset({'alpha'}),
                    status=frozenset({'completed'}),
                    agent_name=frozenset({'support'}),
                    tool_name=frozenset({'lookup', 'refund'}),
                ),
                numeric=NumericFilters(tokens_min=100, duration_ms_max=500),
                limit=3,
            ),
            parallelism=17,
        ),
        dimensions=(
            CompiledQuery.model_validate({
                'name': 'Intent',
                'task': {
                    'kind': 'choice',
                    'instructions': 'Classify the customer intent.',
                    'criteria': {'billing': 'Billing request', 'technical': 'Technical request'},
                    'state': {},
                },
                'selection': {'kind': 'values', 'values': ['billing']},
            }),
        ),
        generated_filters=FacetSelection(provider=frozenset({'openai'}), agent_name=frozenset({'support'})),
        generated_numeric=NumericFilters(tokens_min=50, duration_ms_min=10),
        trace_ids=('trace-2', 'trace-1', 'trace-0'),
        traces=traces,
        results={
            'trace-2': TraceClassification(
                trace_id='trace-2',
                span_id='span-2',
                answers=(
                    DimensionAnswer(
                        value='billing',
                        confidence=0.91,
                        probabilities={'billing': 0.91, 'technical': 0.09},
                        matched=True,
                    ),
                ),
                matched=True,
                raw_result={'messages': 'private'},
            ),
            'trace-1': TraceClassification(
                trace_id='trace-1',
                span_id='span-1',
                answers=(DimensionAnswer(value='technical', matched=False),),
                matched=False,
                raw_result={'classifier_state': 'private'},
            ),
        },
        total=3,
        completed=2,
        failed=0,
        matched=1,
        percent=66.7,
        elapsed=12.5,
        rate=0.16,
    )


def test_build_export_is_conversation_free_and_contains_all_filters() -> None:
    exported = build_export(run())

    assert exported.schema_version == 2
    assert exported.query == 'Find billing requests'
    assert exported.dimensions[0].name == 'Intent'
    assert exported.filters.agent_name == ('support',)
    assert exported.filters.tool_names == ('lookup', 'refund')
    assert exported.generated_filters.provider == ('openai',)
    assert exported.generated_filters.agent_name == ('support',)
    assert exported.numeric.tokens_min == 100
    assert exported.numeric.duration_ms_max == 500
    assert exported.start == datetime(2026, 9, 1, tzinfo=timezone.utc)
    assert exported.end == datetime(2026, 9, 21, tzinfo=timezone.utc)
    assert exported.generated_numeric.tokens_min == 50
    assert exported.traces[0].trace_id == 'trace-2'
    assert exported.traces[0].agent_name == 'support'
    assert exported.traces[0].tool_names == ('lookup', 'refund')
    assert exported.traces[0].matched is True
    assert exported.traces[0].answers[0].value == 'billing'
    assert exported.traces[0].answers[0].confidence == 0.91
    assert exported.traces[1].error is None
    assert exported.traces[2].error == 'classification not completed'
    assert exported.matched_trace_ids == ['trace-2']
    encoded = export_json(run())
    assert json.loads(encoded)['traces'][0]['trace_id'] == 'trace-2'
    for secret in ('secret conversation', 'private', 'classifier_state', 'messages'):
        assert secret not in encoded


def test_matched_only_export_keeps_run_totals_and_only_positive_trace_records() -> None:
    exported = build_export(run(), matched_only=True)

    assert [trace.trace_id for trace in exported.traces] == ['trace-2']
    assert exported.matched_trace_ids == ['trace-2']
    assert exported.counts.total == 3
    assert exported.counts.matched == 1
    assert [trace['trace_id'] for trace in json.loads(export_json(run(), matched_only=True))['traces']] == ['trace-2']


def test_export_models_have_explicit_allow_lists_and_no_snapshot_file_fields() -> None:
    assert set(ExportTrace.model_fields) == {
        'trace_id',
        'span_id',
        'timestamp',
        'agent_name',
        'tool_names',
        'answers',
        'matched',
        'error',
    }
    assert 'snapshot_filename' not in RunExport.model_fields
    assert 'path' not in RunExport.model_fields


def test_filters_only_run_exports_with_no_dimensions() -> None:
    snapshot = run()
    filters_only = replace(
        snapshot,
        dimensions=(),
        results={
            trace_id: result.model_copy(update={'answers': (), 'matched': True})
            for trace_id, result in snapshot.results.items()
        },
    )

    exported = json.loads(export_json(filters_only))

    assert exported['schema_version'] == 2
    assert exported['dimensions'] == []
    assert all(trace['answers'] == [] for trace in exported['traces'])
    assert exported['matched_trace_ids'] == ['trace-2', 'trace-1']
