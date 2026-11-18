"""A small completed Insights run shared by the dashboard Insights tests."""

from __future__ import annotations

from datetime import datetime, timezone

from evaluatorq.insights.models import (
    Cluster,
    ClusterAssignment,
    DimensionResult,
    InsightsConfig,
    InsightsRun,
    LabelAnswer,
    LabelResult,
    TraceInsight,
    TraceSummary,
)
from evaluatorq.insights.presets import SENTIMENT


def build_minimal_run() -> InsightsRun:
    """Build a completed two-trace Insights run with one sentiment label and one intent dimension."""
    cluster = Cluster(
        id='base-1',
        parent_id='top-1',
        level='base',
        name='General requests',
        description='Traces about general requests.',
        size=2,
        trace_ids=['trace-1', 'trace-2'],
        example_trace_ids=['trace-1'],
    )
    top = cluster.model_copy(update={'id': 'top-1', 'parent_id': None, 'level': 'top'})
    traces = [
        TraceInsight(
            trace_id=f'trace-{index}',
            span_id=f'span-{index}',
            timestamp=datetime(2026, 9, index, tzinfo=timezone.utc),
            labels={
                'sentiment': LabelAnswer(value='positive', confidence=0.9, probabilities={'positive': 0.9}, error=None)
            },
            summary=TraceSummary(
                summary='A user asked a question.',
                request='What is the refund policy?',
                task='answer',
                topic='refunds',
                sentiment_explanation='Calm.',
            ),
            assignments={'intent': ClusterAssignment(top='top-1', base='base-1')},
        )
        for index in (1, 2)
    ]
    return InsightsRun(
        run_id='run-1',
        run_name='minimal run',
        created_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        status='completed',
        stage_failures=[],
        population={
            'mode': 'query',
            'query': 'refund policy',
            'facets': {'agent_name': ['support-bot'], 'project': ['default']},
            'numeric': {'tokens_min': 100, 'tokens_max': None, 'duration_ms_min': None, 'duration_ms_max': 2500},
            'start': '2026-08-25T00:00:00+00:00',
            'end': '2026-09-01T00:00:00+00:00',
            'limit': 500,
            'n_scanned': 2,
            'n_matched': 2,
            'n_failed_match': 0,
        },
        config=InsightsConfig(labels=[SENTIMENT], dimensions=['intent']),
        traces=traces,
        dimensions={'intent': DimensionResult(name='intent', source_field='request', clusters=[top, cluster])},
        labels={
            'sentiment': LabelResult(
                spec=SENTIMENT, counts={'positive': 2}, mean_confidence=0.9, n_low_confidence=0, n_failed=0
            )
        },
        priority=None,
        priority_reason='not available',
        counts={'n_traces': 2, 'n_failed_traces': 0},
        warnings=[],
    )
