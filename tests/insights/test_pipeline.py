from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from evaluatorq.insights import pipeline
from evaluatorq.insights.models import Cluster, DimensionName, DimensionResult, LabelAnswer, TraceSummary
from evaluatorq.insights.population import ResolvedPopulation
from evaluatorq.trace_finder.models import TraceRecord


def _trace(i: int) -> TraceRecord:
    return TraceRecord(
        schema_version=1,
        trace_id=f'trace-{i}',
        span_id=f'span-{i}',
        timestamp=datetime(2026, 9, 1, tzinfo=timezone.utc),
        messages=({'role': 'user', 'content': f'request {i}'},),
        project='p',
        model='m',
        provider='openai',
        status='ok',
        product='p',
        trace_type='chat',
    )


def _patch_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, 'resolve_llm_client', lambda client: SimpleNamespace(client=object(), owned=False))
    monkeypatch.setattr(pipeline, 'resolve_orq_client', lambda: object())

    async def close(_client):
        return None

    monkeypatch.setattr(pipeline, 'close_orq_client', close)


@pytest.mark.asyncio
async def test_happy_path_persists_completed_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_clients(monkeypatch)
    traces = [_trace(i) for i in range(30)]
    monkeypatch.setattr(pipeline, 'resolve_population', _resolve(traces))
    monkeypatch.setattr(pipeline, 'label_traces', _label(traces))
    monkeypatch.setattr(pipeline, 'summarize_traces', _summarize(traces))

    async def dimension(name, items, **kwargs):
        clusters = [
            Cluster(
                id=f'intent-b{i}',
                parent_id='intent-t0',
                level='base',
                name=f'group {i}',
                description='desc',
                size=10,
                trace_ids=[f'trace-{i * 10 + j}' for j in range(10)],
                example_trace_ids=[],
            )
            for i in range(3)
        ]
        clusters.append(
            Cluster(
                id='intent-t0',
                parent_id=None,
                level='top',
                name='all',
                description='all',
                size=30,
                trace_ids=[t.trace_id for t in items],
                example_trace_ids=[],
            )
        )
        return DimensionResult(name=name, source_field='request', clusters=clusters)

    monkeypatch.setattr(pipeline, '_build_dimension', dimension)
    run = await pipeline.insights(_population(), dimensions=('intent',), labels=(), runs_dir=tmp_path, run_name='happy')
    assert run.status == 'completed'
    assert len([c for c in run.dimensions['intent'].clusters if c.level == 'base']) == 3
    files = list(tmp_path.glob('insights_*.json'))
    assert len(files) == 1
    from evaluatorq.common.run_manifest import list_manifests

    manifest = list_manifests(tmp_path)[0]
    assert manifest.status.value == 'completed'
    assert manifest.report_path == str(files[0])


@pytest.mark.asyncio
async def test_summary_failure_is_per_trace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_clients(monkeypatch)
    traces = [_trace(i) for i in range(6)]
    monkeypatch.setattr(pipeline, 'resolve_population', _resolve(traces))
    monkeypatch.setattr(pipeline, 'label_traces', _label(traces))

    async def summaries(*args, **kwargs):
        return {t.trace_id: ('summary failed' if t.trace_id == 'trace-0' else _summary()) for t in traces}

    monkeypatch.setattr(pipeline, 'summarize_traces', summaries)
    async def dimension(*args, **kwargs):
        return _dimension_result()
    monkeypatch.setattr(pipeline, '_build_dimension', dimension)
    run = await pipeline.insights(_population(), dimensions=('intent',), labels=(), runs_dir=tmp_path)
    assert run.status == 'completed'
    assert run.counts['n_failed_traces'] == 1
    assert run.traces[0].errors['summary'] == 'summary failed'


@pytest.mark.asyncio
async def test_dimension_failure_keeps_other_dimension_and_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_clients(monkeypatch)
    traces = [_trace(i) for i in range(6)]
    monkeypatch.setattr(pipeline, 'resolve_population', _resolve(traces))
    monkeypatch.setattr(pipeline, 'label_traces', _label(traces))
    monkeypatch.setattr(pipeline, 'summarize_traces', _summarize(traces))

    async def dimension(name, *args, **kwargs):
        if name == 'failure':
            raise RuntimeError('embedding unavailable')
        return _dimension_result(name)

    monkeypatch.setattr(pipeline, '_build_dimension', dimension)
    run = await pipeline.insights(_population(), dimensions=('intent', 'failure'), labels=(), runs_dir=tmp_path)
    assert run.status == 'error'
    assert run.stage_failures[0].stage == 'dimension:failure'
    assert 'intent' in run.dimensions
    assert list(tmp_path.glob('insights_*.json'))


@pytest.mark.asyncio
async def test_empty_population_warns_and_skips_dimensions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_clients(monkeypatch)
    monkeypatch.setattr(pipeline, 'resolve_population', _resolve([]))
    async def must_not_cluster(*args, **kwargs):
        pytest.fail('must not cluster')
    monkeypatch.setattr(pipeline, '_build_dimension', must_not_cluster)
    run = await pipeline.insights(_population(), dimensions=('intent',), runs_dir=tmp_path)
    assert run.status == 'completed'
    assert 'population is empty' in run.warnings


@pytest.mark.asyncio
async def test_unknown_label_fails_before_any_call(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail('a fake or client was called before validating labels')

    monkeypatch.setattr(pipeline, 'resolve_llm_client', forbidden)
    with pytest.raises(ValueError, match='unknown Insights label'):
        await pipeline.insights(_population(), labels=('unknown-label',), runs_dir=tmp_path)


def test_noise_outlier_is_not_a_priority_member() -> None:
    from evaluatorq.insights.models import ClusterAssignment, InsightsConfig, LabelSpec, TraceInsight
    from evaluatorq.insights.priority import priority_points

    satisfaction = LabelSpec(name='customer_satisfaction', kind='score', instructions='score', criteria=['bad', 'good'])
    items = [
        TraceInsight(
            trace_id=f't{i}',
            span_id=f's{i}',
            timestamp=datetime.now(timezone.utc),
            labels={
                'customer_satisfaction': LabelAnswer(value=0.5, confidence=1, probabilities=None, error=None),
            },
        )
        for i in range(2)
    ]
    items[1].assignments['intent'] = ClusterAssignment(top='noise', base='noise')
    dimension = DimensionResult(
        name='intent',
        source_field='request',
        clusters=[
            Cluster(
                id='intent-b0',
                parent_id='intent-t0',
                level='base',
                name='inlier',
                description='',
                size=1,
                trace_ids=['t0'],
                example_trace_ids=[],
            ),
            Cluster(
                id='intent-t0',
                parent_id=None,
                level='top',
                name='all',
                description='',
                size=1,
                trace_ids=['t0'],
                example_trace_ids=[],
            ),
        ],
    )
    points, _ = priority_points(items, dimension, satisfaction_spec=satisfaction)
    assert points is not None
    assert len(points) == 1 and points[0].volume == 1
    assert 't1' not in dimension.clusters[0].trace_ids


def _population():
    from evaluatorq.insights.models import InsightsPopulation

    return InsightsPopulation()


def _resolve(traces):
    async def resolve(*args, **kwargs):
        return _resolved(traces)
    return resolve


def _label(traces):
    async def label(*args, **kwargs):
        return _labels(traces)
    return label


def _summarize(traces):
    async def summarize(*args, **kwargs):
        return _summaries(traces)
    return summarize


def _resolved(traces):
    return ResolvedPopulation(traces=traces, compiled=None, echo={'mode': 'filter'}, n_scanned=len(traces))


def _labels(traces):
    return [SimpleNamespace(trace=t, answers={}, error=None) for t in traces]


def _summary():
    return TraceSummary(
        summary='summary',
        request='request',
        task='task',
        topic='topic',
        assistant_errors=[],
        sentiment_explanation='neutral',
    )


def _summaries(traces):
    return {t.trace_id: _summary() for t in traces}


def _dimension_result(name: DimensionName = 'intent'):
    return DimensionResult(name=name, source_field='request', clusters=[])


@pytest.mark.asyncio
async def test_three_signal_traces_skip_dimension(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.insights.cache import InsightsCache
    from openai import AsyncOpenAI

    from evaluatorq.insights.models import TraceInsight

    traces = [
        TraceInsight(
            trace_id=f't{i}', span_id=f's{i}', timestamp=datetime.now(timezone.utc),
            summary=_summary(),
        )
        for i in range(3)
    ]

    async def embeddings(*args, **kwargs):
        pytest.fail('embedding should not run for three signal traces')

    monkeypatch.setattr(pipeline, 'embed_texts', embeddings)
    cache = InsightsCache(enabled=False)
    try:
        result = await pipeline._build_dimension(
            'intent', traces, client=cast('AsyncOpenAI', object()), cache=cache, embedding_model='unused', summary_model='unused',
            max_clusters=15, max_subclusters=15, outlier_zscore=None, parallelism=10,
            classifier_model='typesafe/jev-latest',
        )
    finally:
        cache.close()
    assert not result.clusters
    assert result.warnings and 'only 3 traces' in result.warnings[0]
