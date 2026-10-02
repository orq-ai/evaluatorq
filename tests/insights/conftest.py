"""Shared fixtures for `evaluatorq.insights` unit tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

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


@pytest.fixture
def minimal_run() -> InsightsRun:
    """A minimal but structurally complete `InsightsRun`: 2 traces, 1 dimension (1 top + 1 base cluster), 1 label."""
    top = Cluster(
        id='top-1',
        parent_id=None,
        level='top',
        name='General requests',
        description='Traces about general requests.',
        size=2,
        trace_ids=['trace-1', 'trace-2'],
        example_trace_ids=['trace-1'],
    )
    base = Cluster(
        id='base-1',
        parent_id='top-1',
        level='base',
        name='General requests',
        description='Traces about general requests.',
        size=2,
        trace_ids=['trace-1', 'trace-2'],
        example_trace_ids=['trace-1'],
    )
    trace_1 = TraceInsight(
        trace_id='trace-1',
        span_id='span-1',
        timestamp=datetime(2026, 9, 1, tzinfo=timezone.utc),
        agent_name='support-bot',
        project='default',
        labels={'sentiment': LabelAnswer(value='positive', confidence=0.9, probabilities={'positive': 0.9}, error=None)},
        summary=TraceSummary(
            summary='A user asked a question.',
            request='What is the refund policy?',
            task='answer question',
            topic='refunds',
            sentiment_explanation='The user is calm and inquisitive.',
        ),
        assignments={'intent': ClusterAssignment(top='top-1', base='base-1')},
        coords={'intent': (0.1, 0.2, 0.3)},
    )
    trace_2 = TraceInsight(
        trace_id='trace-2',
        span_id='span-2',
        timestamp=datetime(2026, 9, 2, tzinfo=timezone.utc),
        agent_name='support-bot',
        project='default',
        labels={'sentiment': LabelAnswer(value='neutral', confidence=0.7, probabilities={'neutral': 0.7}, error=None)},
        summary=TraceSummary(
            summary='A user reported an issue.',
            request='My order is late.',
            task='resolve issue',
            topic='shipping',
            sentiment_explanation='The user is mildly annoyed.',
        ),
        assignments={'intent': ClusterAssignment(top='top-1', base='base-1')},
        coords={'intent': (0.4, 0.5, 0.6)},
    )
    return InsightsRun(
        run_id='run-1',
        run_name='minimal run',
        created_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        status='completed',
        stage_failures=[],
        population={'request': None, 'query': None, 'n_scanned': 2, 'n_matched': 2, 'n_failed_match': 0},
        config=InsightsConfig(labels=[SENTIMENT], dimensions=['intent']),
        traces=[trace_1, trace_2],
        dimensions={
            'intent': DimensionResult(
                name='intent',
                source_field='request',
                clusters=[top, base],
                n_noise=0,
                n_no_signal=0,
                n_failed=0,
                warnings=[],
            )
        },
        labels={
            'sentiment': LabelResult(
                spec=SENTIMENT,
                counts={'positive': 1, 'neutral': 1},
                mean_confidence=0.8,
                n_low_confidence=0,
                n_failed=0,
            )
        },
        priority=None,
        priority_reason='customer_satisfaction label not requested',
        counts={'n_traces': 2, 'n_failed_traces': 0},
        warnings=[],
    )
