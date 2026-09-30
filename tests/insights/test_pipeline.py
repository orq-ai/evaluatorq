from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from evaluatorq.contracts import Usage
from evaluatorq.insights import pipeline
from evaluatorq.insights.models import (
    Cluster,
    DimensionName,
    DimensionResult,
    InsightsPopulation,
    InsightsRun,
    LabelAnswer,
    TraceInsight,
    TraceSummary,
)
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


def test_dimension_failure_count_excludes_run_wide_summary_errors() -> None:
    source = _trace(1)
    summary_failed = TraceInsight(trace_id=source.trace_id, span_id=source.span_id, timestamp=source.timestamp)
    summary_failed.errors['summary'] = 'summary model unavailable'
    dimension_failed = TraceInsight(trace_id='trace-2', span_id='span-2', timestamp=source.timestamp)
    dimension_failed.errors['dimension:intent'] = 'embedding unavailable'

    assert pipeline._dimension_failed_count([summary_failed, dimension_failed], 'intent') == 1


def _patch_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    def resolve_client(client: object, *, max_retries: int) -> SimpleNamespace:
        assert max_retries == 0
        return SimpleNamespace(client=object(), owned=False)

    monkeypatch.setattr(pipeline, 'resolve_llm_client', resolve_client)
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
    assert manifest.planned_stages == ['population', 'label', 'summary', 'dimension:intent', 'priority', 'write']
    assert manifest.stage_labels['dimension:intent'] == 'Cluster and map intent'
    assert run.cost_by_stage == {}


@pytest.mark.asyncio
async def test_finder_missing_traces_are_visible_in_run_warnings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_clients(monkeypatch)
    traces = [_trace(1)]

    async def resolve(*args, **kwargs):
        return ResolvedPopulation(
            traces=traces,
            compiled=None,
            echo={'mode': 'export', 'n_matched_ids': 4, 'n_missing_export_ids': 3},
            n_scanned=1,
        )

    monkeypatch.setattr(pipeline, 'resolve_population', resolve)
    monkeypatch.setattr(pipeline, 'label_traces', _label(traces))
    monkeypatch.setattr(pipeline, 'summarize_traces', _summarize(traces))

    run = await pipeline.insights(_population(), dimensions=(), labels=(), runs_dir=tmp_path)

    assert run.status == 'completed'
    assert '3 of 4 matched Finder traces could not be reloaded from Orq; 1 trace will be analyzed.' in run.warnings


@pytest.mark.asyncio
async def test_dashboard_finder_snapshot_records_origin_and_snapshot_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_clients(monkeypatch)
    traces = [_trace(1)]

    async def resolve(*args, **kwargs):
        return ResolvedPopulation(
            traces=traces,
            compiled=None,
            echo={'mode': 'export', 'finder_export': str(tmp_path / 'private-snapshot.json')},
            n_scanned=1,
        )

    monkeypatch.setattr(pipeline, 'resolve_population', resolve)
    monkeypatch.setattr(pipeline, 'label_traces', _label(traces))
    monkeypatch.setattr(pipeline, 'summarize_traces', _summarize(traces))
    original = tmp_path / 'finder-exports' / 'trace-finder-saved.json'
    snapshot_sha256 = 'a' * 64

    run = await pipeline.insights(
        InsightsPopulation.from_finder_export(tmp_path / 'private-snapshot.json'),
        _finder_export_source=original,
        _finder_export_sha256=snapshot_sha256,
        dimensions=(),
        labels=(),
        runs_dir=tmp_path,
    )

    assert run.status == 'completed'
    assert run.population['finder_export'] == str(original)
    assert run.population['finder_export_sha256'] == snapshot_sha256
    saved = next(tmp_path.glob('insights_*.json'))
    saved_population = InsightsRun.model_validate_json(saved.read_text(encoding='utf-8')).population
    assert saved_population['finder_export'] == str(original)
    assert saved_population['finder_export_sha256'] == snapshot_sha256


@pytest.mark.asyncio
async def test_pipeline_persists_label_and_summary_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from evaluatorq.common.run_manifest import list_manifests

    _patch_clients(monkeypatch)
    traces = [_trace(i) for i in range(3)]
    monkeypatch.setattr(pipeline, 'resolve_population', _resolve(traces))

    async def label(items, *, on_progress, **kwargs):
        for completed in range(1, len(items) + 1):
            on_progress(completed, len(items))
        return _labels(items)

    async def summarize(items, *, on_progress, **kwargs):
        for completed in range(1, len(items) + 1):
            on_progress(completed, len(items))
        return _summaries(items)

    monkeypatch.setattr(pipeline, 'label_traces', label)
    monkeypatch.setattr(pipeline, 'summarize_traces', summarize)

    run = await pipeline.insights(_population(), dimensions=(), labels=(), runs_dir=tmp_path)

    assert run.status == 'completed'
    manifest = list_manifests(tmp_path)[0]
    progress = {stage.name: (stage.completed, stage.total) for stage in manifest.stages}
    assert progress['label'] == (3, 3)
    assert progress['summary'] == (3, 3)


@pytest.mark.asyncio
async def test_pipeline_persists_per_stage_usage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_clients(monkeypatch)
    traces = [_trace(0)]
    monkeypatch.setattr(pipeline, 'resolve_population', _resolve(traces))

    async def label(*args, usage, **kwargs):
        usage.add('label', Usage(input_tokens=5, output_tokens=1, total_tokens=6, total_cost=0.02, calls=1, priced_calls=1))
        return _labels(traces)

    async def summarize(*args, usage, **kwargs):
        usage.add('summary', Usage(input_tokens=10, output_tokens=2, total_tokens=12, total_cost=0.03, calls=1, priced_calls=1))
        return _summaries(traces)

    monkeypatch.setattr(pipeline, 'label_traces', label)
    monkeypatch.setattr(pipeline, 'summarize_traces', summarize)
    run = await pipeline.insights(_population(), dimensions=(), labels=(), runs_dir=tmp_path)

    label_usage = run.cost_by_stage['label']
    summary_usage = run.cost_by_stage['summary']
    assert label_usage is not None and label_usage.total_cost == 0.02
    assert summary_usage is not None and summary_usage.total_cost == 0.03
    saved = next(tmp_path.glob('insights_*.json'))
    from evaluatorq.insights.models import InsightsRun

    restored = InsightsRun.model_validate_json(saved.read_text(encoding='utf-8'))
    assert restored.cost_by_stage == run.cost_by_stage


@pytest.mark.asyncio
async def test_classifier_model_follows_the_setting(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_clients(monkeypatch)
    monkeypatch.setenv('EVALUATORQ_CLASSIFIER_MODEL', 'acme/classifier-9')
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(tmp_path / 'none.json'))
    traces = [_trace(0)]
    seen: list[str] = []

    async def label(*args, model, **kwargs):
        seen.append(model)
        return _labels(traces)

    monkeypatch.setattr(pipeline, 'resolve_population', _resolve(traces))
    monkeypatch.setattr(pipeline, 'label_traces', label)
    monkeypatch.setattr(pipeline, 'summarize_traces', _summarize(traces))

    async def dimension(*args, **kwargs):
        return _dimension_result()

    monkeypatch.setattr(pipeline, '_build_dimension', dimension)
    run = await pipeline.insights(_population(), runs_dir=tmp_path)
    assert run.config.classifier_model == 'acme/classifier-9'
    assert seen == ['acme/classifier-9']


@pytest.mark.asyncio
async def test_summary_failure_is_per_trace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_clients(monkeypatch)
    traces = [_trace(i) for i in range(6)]
    monkeypatch.setattr(pipeline, 'resolve_population', _resolve(traces))
    async def labels(*args, **kwargs):
        outcomes = _labels(traces)
        outcomes[0].error = 'label failed'
        return outcomes

    monkeypatch.setattr(pipeline, 'label_traces', labels)

    async def summaries(*args, **kwargs):
        return {t.trace_id: ('summary failed' if t.trace_id == 'trace-0' else _summary()) for t in traces}

    monkeypatch.setattr(pipeline, 'summarize_traces', summaries)
    async def dimension(*args, **kwargs):
        return _dimension_result()
    monkeypatch.setattr(pipeline, '_build_dimension', dimension)
    run = await pipeline.insights(_population(), dimensions=('intent',), labels=(), runs_dir=tmp_path)
    assert run.status == 'completed'
    assert run.counts['n_failed_traces'] == 1
    assert run.counts['per_stage_failed'] == 2
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
@pytest.mark.parametrize('source', ['filter', 'finder'])
async def test_empty_non_query_population_does_not_require_llm_credentials(
    source: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_clients(monkeypatch)
    population = _population()
    if source == 'finder':
        population = InsightsPopulation.from_finder_export(tmp_path / 'finder.json')
    monkeypatch.setattr(pipeline, 'resolve_population', _resolve([]))

    def unavailable(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError('LLM credentials are unavailable')

    monkeypatch.setattr(pipeline, 'resolve_llm_client', unavailable)

    run = await pipeline.insights(population, dimensions=('intent',), runs_dir=tmp_path)

    assert run.status == 'completed'
    assert run.counts['n_traces'] == 0
    assert 'population is empty' in run.warnings


@pytest.mark.asyncio
async def test_saved_callback_that_raises_is_attempted_only_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from evaluatorq.common.run_manifest import list_manifests

    _patch_clients(monkeypatch)
    monkeypatch.setattr(pipeline, 'resolve_population', _resolve([]))
    callback_paths: list[Path] = []

    def fail_callback(path: Path) -> None:
        callback_paths.append(path)
        raise RuntimeError('notification failed')

    run = await pipeline.insights(_population(), dimensions=(), labels=(), runs_dir=tmp_path, _on_saved=fail_callback)

    assert len(callback_paths) == 1
    assert run.status == 'error'
    assert run.stage_failures[-1].stage == 'write'
    manifests = list_manifests(tmp_path)
    assert len(manifests) == 1
    assert manifests[0].status.value == 'error'


@pytest.mark.asyncio
async def test_unexpected_population_failure_records_the_active_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from evaluatorq.common.run_manifest import list_manifests

    _patch_clients(monkeypatch)

    async def fail_population(*args, **kwargs):
        raise RuntimeError('unexpected population failure')

    monkeypatch.setattr(pipeline, 'resolve_population', fail_population)
    run = await pipeline.insights(_population(), dimensions=(), labels=(), runs_dir=tmp_path)

    assert run.status == 'error'
    assert [(failure.stage, failure.message) for failure in run.stage_failures] == [
        ('population', 'unexpected population failure')
    ]
    manifest = list_manifests(tmp_path)[0]
    assert manifest.status.value == 'error'
    assert manifest.stage == 'population'


@pytest.mark.asyncio
async def test_unexpected_run_write_failure_is_attributed_to_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from evaluatorq.common.run_manifest import list_manifests

    _patch_clients(monkeypatch)
    monkeypatch.setattr(pipeline, 'resolve_population', _resolve([]))
    save_run = pipeline.save_run
    calls = 0

    def fail_first_write(run, directory):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError('disk full')
        return save_run(run, directory)

    monkeypatch.setattr(pipeline, 'save_run', fail_first_write)
    run = await pipeline.insights(_population(), dimensions=(), labels=(), runs_dir=tmp_path)

    assert calls == 2
    assert run.status == 'error'
    assert [(failure.stage, failure.message) for failure in run.stage_failures] == [('write', 'disk full')]
    manifest = list_manifests(tmp_path)[0]
    assert manifest.status.value == 'error'
    assert manifest.stage == 'write'


@pytest.mark.asyncio
async def test_cleanup_failure_does_not_prevent_terminal_persistence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from evaluatorq.common.run_manifest import list_manifests
    from evaluatorq.insights.population import PopulationError

    _patch_clients(monkeypatch)

    async def fail_population(*args, **kwargs):
        raise PopulationError('invalid population')

    async def fail_close(*args, **kwargs):
        raise RuntimeError('close failed')

    monkeypatch.setattr(pipeline, 'resolve_population', fail_population)
    monkeypatch.setattr(pipeline, 'close_orq_client', fail_close)
    run = await pipeline.insights(_population(), dimensions=(), labels=(), runs_dir=tmp_path)

    assert run.status == 'error'
    assert list(tmp_path.glob('insights_*.json'))
    manifests = list_manifests(tmp_path)
    assert len(manifests) == 1
    assert manifests[0].status.value == 'error'
    assert manifests[0].stage == 'population'


@pytest.mark.asyncio
@pytest.mark.parametrize('selection', ['labels', 'dimensions'])
async def test_duplicate_config_selection_fails_before_starting_manifest(
    selection: str, tmp_path: Path
) -> None:
    from evaluatorq.common.run_manifest import list_manifests
    from evaluatorq.insights.models import LabelSpec

    kwargs: dict[str, Any] = {'dimensions': ('intent',), 'labels': ()}
    if selection == 'labels':
        spec = LabelSpec(name='duplicate', kind='choice', instructions='Classify it.')
        kwargs['labels'] = (spec, spec)
    else:
        kwargs['dimensions'] = ('intent', 'intent')

    with pytest.raises(ValueError, match='unique names|must not contain duplicates'):
        await pipeline.insights(_population(), runs_dir=tmp_path, **kwargs)

    assert list_manifests(tmp_path) == []
@pytest.mark.asyncio
async def test_local_projection_omission_is_visible_in_saved_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_clients(monkeypatch)
    traces = [_trace(1)]

    async def local_population(*args, **kwargs):
        return ResolvedPopulation(
            traces=traces,
            compiled=None,
            echo={'mode': 'snapshot', 'n_projection_truncated': 1, 'n_source_messages': 2, 'n_omitted_messages': 1},
            n_scanned=1,
        )

    monkeypatch.setattr(pipeline, 'resolve_population', local_population)
    monkeypatch.setattr(pipeline, 'label_traces', _label(traces))
    monkeypatch.setattr(pipeline, 'summarize_traces', _summarize(traces))

    run = await pipeline.insights(_population(), dimensions=(), labels=(), runs_dir=tmp_path)

    assert any('1 of 1 trace exceeded' in warning and '1 of 2 whole messages' in warning for warning in run.warnings)


@pytest.mark.asyncio
async def test_unknown_label_fails_before_any_call(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail('a fake or client was called before validating labels')

    monkeypatch.setattr(pipeline, 'resolve_llm_client', forbidden)
    with pytest.raises(ValueError, match='unknown Insights label'):
        await pipeline.insights(_population(), labels=('unknown-label',), runs_dir=tmp_path)


@pytest.mark.asyncio
async def test_invalid_config_fails_before_starting_manifest(tmp_path: Path) -> None:
    from evaluatorq.common.run_manifest import list_manifests

    with pytest.raises(ValueError, match='max_clusters must be positive'):
        await pipeline.insights(_population(), dimensions=('intent',), max_clusters=0, runs_dir=tmp_path)

    assert list_manifests(tmp_path) == []


def test_noise_outlier_is_not_a_priority_member() -> None:
    from evaluatorq.insights.models import ClusterAssignment, LabelSpec, TraceInsight
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
    items[0].summary = TraceSummary(
        summary='The assistant answered the request without errors.',
        request='request',
        task=None,
        topic=None,
        assistant_errors=[],
        sentiment_explanation=None,
    )
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



@pytest.mark.asyncio
async def test_compiled_query_excludes_false_matches_before_summary_and_dimensions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_clients(monkeypatch)
    traces = [_trace(i) for i in range(3)]
    compiled = object()
    monkeypatch.setattr(pipeline, 'resolve_population', _resolve(traces, compiled=compiled))

    async def label(*args, **kwargs):
        return [
            SimpleNamespace(trace=traces[0], answers={}, matched=True, error=None),
            SimpleNamespace(trace=traces[1], answers={}, matched=False, error=None),
            SimpleNamespace(trace=traces[2], answers={}, matched=None, error='classifier response was unreadable'),
        ]

    summarized: list[str] = []
    async def summarize(selected, **kwargs):
        summarized.extend(trace.trace_id for trace in selected)
        return {trace.trace_id: _summary() for trace in selected}

    dimension_ids: list[str] = []
    async def dimension(name, selected, **kwargs):
        dimension_ids.extend(trace.trace_id for trace in selected)
        return _dimension_result(name)

    monkeypatch.setattr(pipeline, 'label_traces', label)
    monkeypatch.setattr(pipeline, 'summarize_traces', summarize)
    monkeypatch.setattr(pipeline, '_build_dimension', dimension)
    run = await pipeline.insights(_population(), dimensions=('intent',), runs_dir=tmp_path)

    assert run.status == 'completed'
    assert [trace.trace_id for trace in run.traces] == ['trace-0']
    assert summarized == ['trace-0']
    assert dimension_ids == ['trace-0']
    assert run.population['n_matched'] == 1
    assert run.population['n_failed_match'] == 1
    assert any('trace-2' in warning and 'classifier response was unreadable' in warning for warning in run.warnings)


@pytest.mark.asyncio
async def test_partial_label_answer_failures_do_not_fail_label_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from evaluatorq.insights.models import LabelAnswer, LabelSpec

    _patch_clients(monkeypatch)
    traces = [_trace(i) for i in range(6)]
    monkeypatch.setattr(pipeline, 'resolve_population', _resolve(traces))
    failed = LabelSpec(name='first', kind='choice', instructions='first')
    valid = LabelSpec(name='second', kind='choice', instructions='second')

    async def label(*args, **kwargs):
        return [
            SimpleNamespace(
                trace=trace,
                answers={
                    'first': LabelAnswer(value=None, confidence=None, probabilities=None, error='missing answer'),
                    'second': LabelAnswer(value='valid', confidence=0.9, probabilities=None, error=None),
                },
                matched=None,
                error=None,
            )
            for trace in traces
        ]

    monkeypatch.setattr(pipeline, 'label_traces', label)
    monkeypatch.setattr(pipeline, 'summarize_traces', _summarize(traces))
    async def dimension(name, *args, **kwargs):
        return _dimension_result(name)
    monkeypatch.setattr(pipeline, '_build_dimension', dimension)
    run = await pipeline.insights(_population(), labels=(failed, valid), dimensions=('intent',), runs_dir=tmp_path)

    assert run.status == 'completed'
    assert not any(failure.stage == 'label' for failure in run.stage_failures)
    assert run.labels['first'].n_failed == len(traces)
    assert run.labels['second'].counts == {'valid': len(traces)}


@pytest.mark.asyncio
async def test_all_label_requests_failing_skips_summary_and_dimensions_without_query(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from evaluatorq.insights.models import LabelSpec

    _patch_clients(monkeypatch)
    traces = [_trace(i) for i in range(6)]
    monkeypatch.setattr(pipeline, 'resolve_population', _resolve(traces))
    label_spec = LabelSpec(name='intent_label', kind='choice', instructions='classify intent')

    async def label(*args, **kwargs):
        return [
            SimpleNamespace(trace=trace, answers={}, matched=None, error='classifier unavailable')
            for trace in traces
        ]

    async def forbidden(*args, **kwargs):
        pytest.fail('summary and dimensions must not run after every label request failed')

    monkeypatch.setattr(pipeline, 'label_traces', label)
    monkeypatch.setattr(pipeline, 'summarize_traces', forbidden)
    monkeypatch.setattr(pipeline, '_build_dimension', forbidden)
    run = await pipeline.insights(
        _population(), labels=(label_spec,), dimensions=('intent',), runs_dir=tmp_path
    )

    assert run.status == 'error'
    assert [failure.stage for failure in run.stage_failures] == ['label']
    assert len(run.traces) == len(traces)
    assert all(trace.errors['label'] == 'classifier unavailable' for trace in run.traces)
    assert run.warnings == [
        'Summarize traces stage skipped because every label request failed',
        'dimension:intent stage skipped because every label request failed',
    ]
    from evaluatorq.common.run_manifest import list_manifests

    manifest = list_manifests(tmp_path)[0]
    statuses = {stage.name: stage.status.value for stage in manifest.stages}
    assert statuses['label'] == 'error'
    assert statuses['summary'] == 'error'
    assert statuses['dimension:intent'] == 'error'



@pytest.mark.asyncio
async def test_all_false_query_matches_finish_as_empty_without_summary_or_dimensions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_clients(monkeypatch)
    traces = [_trace(i) for i in range(3)]
    monkeypatch.setattr(pipeline, 'resolve_population', _resolve(traces, compiled=object()))

    async def label(*args, **kwargs):
        return [SimpleNamespace(trace=trace, answers={}, matched=False, error=None) for trace in traces]

    async def forbidden(*args, **kwargs):
        pytest.fail('empty matched population must bypass summaries and dimensions')

    monkeypatch.setattr(pipeline, 'label_traces', label)
    monkeypatch.setattr(pipeline, 'summarize_traces', forbidden)
    monkeypatch.setattr(pipeline, '_build_dimension', forbidden)
    run = await pipeline.insights(_population(), dimensions=('intent',), runs_dir=tmp_path)

    assert run.status == 'completed'
    assert run.traces == []
    assert run.warnings.count('population is empty') == 1
    assert run.counts['n_traces'] == 0
    assert run.population['n_matched'] == 0
    assert run.population['n_failed_match'] == 0
    assert run.dimensions == {}
    assert list(tmp_path.glob('insights_*.json'))
    from evaluatorq.common.run_manifest import list_manifests

    assert list_manifests(tmp_path)[0].status.value == 'completed'


@pytest.mark.asyncio
async def test_every_summary_failing_fails_the_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_clients(monkeypatch)
    traces = [_trace(i) for i in range(3)]
    monkeypatch.setattr(pipeline, 'resolve_population', _resolve(traces))
    monkeypatch.setattr(pipeline, 'label_traces', _label(traces))

    async def all_fail(*args, **kwargs):
        return {trace.trace_id: 'summary: model not enabled' for trace in traces}

    async def must_not_cluster(*args, **kwargs):
        pytest.fail('dimensions must be skipped when every summary failed')

    monkeypatch.setattr(pipeline, 'summarize_traces', all_fail)
    monkeypatch.setattr(pipeline, '_build_dimension', must_not_cluster)
    run = await pipeline.insights(_population(), runs_dir=tmp_path)
    assert run.status == 'error'
    assert [failure.stage for failure in run.stage_failures] == ['summary']
    assert 'model not enabled' in run.stage_failures[0].message


@pytest.mark.asyncio
async def test_failed_summaries_are_not_counted_as_dimension_failures_or_no_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openai import AsyncOpenAI

    from evaluatorq.insights.cache import InsightsCache
    from evaluatorq.insights.models import TraceInsight

    clean = [
        TraceInsight(
            trace_id=f'c{i}', span_id='s', timestamp=datetime.now(timezone.utc), summary=_summary()
        )
        for i in range(4)
    ]
    failed = [
        TraceInsight(
            trace_id=f'f{i}',
            span_id='s',
            timestamp=datetime.now(timezone.utc),
            errors={'summary': 'summary: boom'},
        )
        for i in range(2)
    ]
    cache = InsightsCache(enabled=False)
    try:
        result = await pipeline._build_dimension(
            'failure',
            clean + failed,
            client=cast('AsyncOpenAI', object()),
            cache=cache,
            embedding_model='unused',
            summary_model='unused',
            max_clusters=15,
            max_subclusters=15,
            outlier_zscore=None,
            parallelism=10,
            classifier_model='unused',
        )
    finally:
        cache.close()
    assert result.n_no_signal == 4
    assert result.n_failed == 0


@pytest.mark.asyncio
async def test_failure_dimension_counts_placeholder_errors_as_no_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    from openai import AsyncOpenAI

    from evaluatorq.insights.cache import InsightsCache
    from evaluatorq.insights.models import TraceInsight

    traces = [
        TraceInsight(
            trace_id=f't{i}',
            span_id='s',
            timestamp=datetime.now(timezone.utc),
            summary=_summary().model_copy(update={'assistant_errors': ['None'] if i < 6 else [f'Error {i}']}),
        )
        for i in range(9)
    ]
    monkeypatch.setattr(pipeline, 'embed_texts', lambda *args, **kwargs: pytest.fail('must not embed'))
    cache = InsightsCache(enabled=False)
    try:
        result = await pipeline._build_dimension(
            'failure',
            traces,
            client=cast('AsyncOpenAI', object()),
            cache=cache,
            embedding_model='unused',
            summary_model='unused',
            max_clusters=15,
            max_subclusters=15,
            outlier_zscore=None,
            parallelism=10,
            classifier_model='unused',
        )
    finally:
        cache.close()
    assert result.n_no_signal == 6
    assert any('only 3 traces have signal' in warning for warning in result.warnings)


def _population():
    from evaluatorq.insights.models import InsightsPopulation

    return InsightsPopulation()


def _resolve(traces, *, compiled=None):
    async def resolve(*args, **kwargs):
        return _resolved(traces, compiled=compiled)
    return resolve


def _label(traces):
    async def label(*args, **kwargs):
        return _labels(traces)
    return label


def _summarize(traces):
    async def summarize(*args, **kwargs):
        return _summaries(traces)
    return summarize


def _resolved(traces, *, compiled=None):
    return ResolvedPopulation(traces=traces, compiled=compiled, echo={'mode': 'filter'}, n_scanned=len(traces))


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


@pytest.mark.asyncio
async def test_small_sentiment_groups_cluster_and_warn_when_umap_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    from io import StringIO

    from loguru import logger
    from openai import AsyncOpenAI

    from evaluatorq.insights.cache import InsightsCache
    from evaluatorq.insights.describe import ClusterName
    from evaluatorq.insights.models import LabelAnswer, TraceInsight
    from evaluatorq.insights import reduce

    traces = [
        TraceInsight(
            trace_id=f't{i}', span_id=f's{i}', timestamp=datetime.now(timezone.utc),
            summary=_summary(),
            labels={
                'sentiment': LabelAnswer(
                    value='positive' if i < 3 else 'neutral' if i < 5 else 'negative',
                    confidence=1.0, probabilities=None, error=None,
                )
            },
        )
        for i in range(6)
    ]
    traces[0].errors['label'] = 'unrelated label failed'
    messages = StringIO()
    sink_id = logger.add(messages, level='WARNING')

    async def embeddings(texts, **kwargs):
        return {text: [1.0, 0.0] for text in set(texts)}

    async def descriptions(members, **kwargs):
        return {cluster_id: ClusterName(name=f'group {cluster_id}', description='group details') for cluster_id in members}

    async def top_descriptions(children, **kwargs):
        return {top_id: ClusterName(name=f'top {top_id}', description='top details') for top_id in children}

    def no_umap(*args, **kwargs):
        return None

    monkeypatch.setattr(pipeline, 'embed_texts', embeddings)
    monkeypatch.setattr(pipeline, 'describe_clusters', descriptions)
    monkeypatch.setattr(pipeline, 'describe_top_level', top_descriptions)
    monkeypatch.setattr(reduce, 'reduce_3d', no_umap)
    cache = InsightsCache(enabled=False)
    try:
        result = await pipeline._build_dimension(
            'sentiment', traces, client=cast('AsyncOpenAI', object()), cache=cache,
            embedding_model='unused', summary_model='unused', max_clusters=15, max_subclusters=15,
            outlier_zscore=None, parallelism=10, classifier_model='typesafe/jev-latest',
        )
    finally:
        cache.close()
        logger.remove(sink_id)

    base_clusters = [cluster for cluster in result.clusters if cluster.level == 'base']
    top_clusters = [cluster for cluster in result.clusters if cluster.level == 'top']
    assert {cluster.group: cluster.size for cluster in base_clusters} == {'positive': 3, 'neutral': 2}
    assert {cluster.group: cluster.size for cluster in top_clusters} == {'positive': 3, 'neutral': 2}
    assert all(trace.assignments['sentiment'].base != 'unclassified' for trace in traces[:5])
    assert traces[5].assignments['sentiment'].base == 'unclassified'
    assert 'dimension:sentiment' in traces[5].errors
    assert result.n_failed == 1
    assert sum('UMAP unavailable' in warning for warning in result.warnings) == 2
    assert 'UMAP unavailable' in messages.getvalue()
    assert 'marked unclassified' in messages.getvalue()


@pytest.mark.asyncio
async def test_cluster_merges_cannot_cross_top_level_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np
    from openai import AsyncOpenAI

    from evaluatorq.insights import cluster, merge, reduce
    from evaluatorq.insights.cache import InsightsCache
    from evaluatorq.insights.describe import ClusterName
    from evaluatorq.insights.models import TraceInsight

    traces = [
        TraceInsight(
            trace_id=f't{i}', span_id=f's{i}', timestamp=datetime.now(timezone.utc), summary=_summary()
        )
        for i in range(5)
    ]
    tree = cluster.ClusterTree(
        base_labels=np.asarray([0, 0, 1, 1, 1]), top_of_base={0: 0, 1: 1}, n_base=2, n_top=2
    )
    monkeypatch.setattr(cluster, 'cluster_two_level', lambda *args, **kwargs: tree)
    monkeypatch.setattr(cluster, 'nearest_neighbours', lambda _cents: {0: [1], 1: [0]})

    async def embeddings(texts, **kwargs):
        return {text: [float(index + 1), 1.0] for index, text in enumerate(texts)}

    async def descriptions(members, **kwargs):
        return {key: ClusterName(name=f'base {key}', description='details') for key in members}

    async def top_descriptions(children, **kwargs):
        return {key: ClusterName(name=f'top {key}', description='details') for key in children}

    captured_neighbours = []

    async def no_merge(names, examples, neighbours, **kwargs):
        captured_neighbours.append(neighbours)
        return SimpleNamespace(representatives={key: key for key in names}, n_pairs=0, n_failed=0)

    monkeypatch.setattr(pipeline, 'embed_texts', embeddings)
    monkeypatch.setattr(pipeline, 'describe_clusters', descriptions)
    monkeypatch.setattr(pipeline, 'describe_top_level', top_descriptions)
    monkeypatch.setattr(merge, 'merge_similar', no_merge)
    monkeypatch.setattr(reduce, 'reduce_3d', lambda _vectors: None)
    cache = InsightsCache(enabled=False)
    try:
        result = await pipeline._build_dimension(
            'intent', traces, client=cast('AsyncOpenAI', object()), cache=cache,
            embedding_model='unused', summary_model='unused', max_clusters=15, max_subclusters=15,
            outlier_zscore=None, parallelism=10, classifier_model='typesafe/jev-latest',
        )
    finally:
        cache.close()

    assert captured_neighbours == [{0: [], 1: []}]
    tops = {cluster.id: cluster for cluster in result.clusters if cluster.level == 'top'}
    bases = [cluster for cluster in result.clusters if cluster.level == 'base']
    assert {top.size for top in tops.values()} == {2, 3}
    assert all(top.size == sum(base.size for base in bases if base.parent_id == top.id) for top in tops.values())


@pytest.mark.asyncio
async def test_base_cluster_examples_are_nearest_to_centroid(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np
    from openai import AsyncOpenAI

    from evaluatorq.insights import cluster, reduce
    from evaluatorq.insights.cache import InsightsCache
    from evaluatorq.insights.describe import ClusterName
    from evaluatorq.insights.models import TraceInsight

    traces = [
        TraceInsight(
            trace_id=f't{i}', span_id=f's{i}', timestamp=datetime.now(timezone.utc),
            summary=_summary().model_copy(update={'request': f'request {i}'}),
        )
        for i in range(7)
    ]
    monkeypatch.setattr(cluster, 'cluster_two_level', lambda vectors, **kwargs: cluster.ClusterTree(
        base_labels=np.zeros(len(vectors), dtype=int), top_of_base={0: 0}, n_base=1, n_top=1
    ))
    monkeypatch.setattr(cluster, 'nearest_neighbours', lambda _cents: {0: []})
    vectors = [[-100.0], [-99.0], [0.0], [1.0], [2.0], [3.0], [4.0]]

    async def embeddings(texts, **kwargs):
        return {text: vectors[i] for i, text in enumerate(texts)}

    async def descriptions(members, **kwargs):
        return {key: ClusterName(name=f'base {key}', description='details') for key in members}

    async def top_descriptions(children, **kwargs):
        return {key: ClusterName(name=f'top {key}', description='details') for key in children}

    monkeypatch.setattr(pipeline, 'embed_texts', embeddings)
    monkeypatch.setattr(pipeline, 'describe_clusters', descriptions)
    monkeypatch.setattr(pipeline, 'describe_top_level', top_descriptions)
    monkeypatch.setattr(reduce, 'reduce_3d', lambda _vectors: None)
    cache = InsightsCache(enabled=False)
    try:
        result = await pipeline._build_dimension(
            'intent', traces, client=cast('AsyncOpenAI', object()), cache=cache,
            embedding_model='unused', summary_model='unused', max_clusters=15, max_subclusters=15,
            outlier_zscore=None, parallelism=10, classifier_model='typesafe/jev-latest',
        )
    finally:
        cache.close()
    base = next(value for value in result.clusters if value.level == 'base')
    assert base.example_trace_ids == ['t2', 't3', 't4', 't5', 't6']


@pytest.mark.parametrize(('n_failed', 'raises'), [(4, True), (2, False)])
@pytest.mark.asyncio
async def test_merge_failures_are_reported_in_dimension(
    monkeypatch: pytest.MonkeyPatch, n_failed: int, raises: bool
) -> None:
    import numpy as np
    from openai import AsyncOpenAI

    from evaluatorq.insights import cluster, merge, reduce
    from evaluatorq.insights.cache import InsightsCache
    from evaluatorq.insights.describe import ClusterName
    from evaluatorq.insights.models import TraceInsight

    traces = [
        TraceInsight(
            trace_id=f't{i}', span_id=f's{i}', timestamp=datetime.now(timezone.utc), summary=_summary()
        )
        for i in range(5)
    ]
    tree = cluster.ClusterTree(
        base_labels=np.asarray([0, 0, 1, 1, 1]), top_of_base={0: 0, 1: 0}, n_base=2, n_top=1
    )
    monkeypatch.setattr(cluster, 'cluster_two_level', lambda *args, **kwargs: tree)
    monkeypatch.setattr(cluster, 'nearest_neighbours', lambda _cents: {0: [1], 1: [0]})

    async def embeddings(texts, **kwargs):
        return {text: [float(index + 1), 1.0] for index, text in enumerate(texts)}

    async def descriptions(members, **kwargs):
        return {key: ClusterName(name=f'base {key}', description='details') for key in members}

    async def top_descriptions(children, **kwargs):
        return {key: ClusterName(name=f'top {key}', description='details') for key in children}

    async def failed_merge(names, examples, neighbours, **kwargs):
        return SimpleNamespace(
            representatives={key: key for key in names}, n_pairs=4, n_failed=n_failed
        )

    monkeypatch.setattr(pipeline, 'embed_texts', embeddings)
    monkeypatch.setattr(pipeline, 'describe_clusters', descriptions)
    monkeypatch.setattr(pipeline, 'describe_top_level', top_descriptions)
    monkeypatch.setattr(merge, 'merge_similar', failed_merge)
    monkeypatch.setattr(reduce, 'reduce_3d', lambda _vectors: None)
    cache = InsightsCache(enabled=False)
    result: DimensionResult | None = None
    try:
        if raises:
            with pytest.raises(RuntimeError, match='merge'):
                await pipeline._build_dimension(
                    'intent', traces, client=cast('AsyncOpenAI', object()), cache=cache,
                    embedding_model='unused', summary_model='unused', max_clusters=15, max_subclusters=15,
                    outlier_zscore=None, parallelism=10, classifier_model='typesafe/jev-latest',
                )
        else:
            result = await pipeline._build_dimension(
                'intent', traces, client=cast('AsyncOpenAI', object()), cache=cache,
                embedding_model='unused', summary_model='unused', max_clusters=15, max_subclusters=15,
                outlier_zscore=None, parallelism=10, classifier_model='typesafe/jev-latest',
            )
    finally:
        cache.close()

    if not raises:
        assert result is not None
        assert any('2/4 pair checks failed' in warning for warning in result.warnings)


def test_label_results_count_score_levels_by_their_own_number_and_not_asked() -> None:
    from evaluatorq.insights.models import TraceInsight
    from evaluatorq.insights.presets import SCOPE_CREEP, USER_FRUSTRATION

    def insight(i: int, labels: dict[str, LabelAnswer]) -> TraceInsight:
        return TraceInsight(trace_id=f't{i}', span_id='s', timestamp=datetime.now(timezone.utc), labels=labels)

    traces = [
        insight(0, {'user_frustration': LabelAnswer(value=0.0, confidence=None, probabilities=None, error=None), 'scope_creep': LabelAnswer(value=True, confidence=None, probabilities=None, error=None)}),
        insight(1, {'user_frustration': LabelAnswer(value=1.0, confidence=None, probabilities=None, error=None)}),
    ]

    results = pipeline._label_results(traces, [USER_FRUSTRATION, SCOPE_CREEP])

    assert results['user_frustration'].counts == {'1': 1, '5': 1}
    assert results['scope_creep'].counts == {'yes': 1}
    assert results['scope_creep'].n_not_asked == 1
