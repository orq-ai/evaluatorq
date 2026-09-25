"""Orchestrate an Insights run from population loading through JSON persistence.

The optional numerical stack is imported only after ``insights`` is called.
Individual LLM helpers own their documented retry policy; this module adds no
retry layer.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq.common.llm_client import resolve_llm_client
from evaluatorq.common.orq_client import close_orq_client, resolve_orq_client
from evaluatorq.common.run_manifest import start_manifest
from evaluatorq.insights.cache import InsightsCache
from evaluatorq.insights.describe import ClusterName, describe_clusters, describe_top_level
from evaluatorq.insights.embed import embed_texts
from evaluatorq.insights.labeling import label_traces
from evaluatorq.insights.models import (
    Cluster,
    ClusterAssignment,
    DimensionResult,
    InsightsConfig,
    InsightsRun,
    LabelResult,
    LabelSpec,
    StageFailure,
    TraceInsight,
)
from evaluatorq.insights.population import PopulationError, resolve_population
from evaluatorq.insights.presets import DIMENSION_FIELDS, LABEL_PRESETS, SENTIMENT
from evaluatorq.insights.priority import priority_points
from evaluatorq.insights.store import get_insights_runs_dir, save_run
from evaluatorq.insights.summarize import summarize_traces
from evaluatorq.trace_finder.settings import effective_settings

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from openai import AsyncOpenAI
    from orq_ai_sdk import Orq

    from evaluatorq.insights.models import DimensionName, InsightsPopulation


def _label_specs(labels: Sequence[LabelSpec | str], dimensions: Sequence[DimensionName]) -> list[LabelSpec]:
    resolved: list[LabelSpec] = []
    for label in labels:
        if isinstance(label, LabelSpec):
            resolved.append(label)
        else:
            try:
                resolved.append(LABEL_PRESETS[label])
            except KeyError as exc:
                raise ValueError(f'unknown Insights label preset: {label!r}') from exc
    if 'sentiment' in dimensions and not any(spec.name == 'sentiment' for spec in resolved):
        logger.info("Adding the 'sentiment' label because the sentiment dimension was requested")
        resolved.append(SENTIMENT)
    names = [spec.name for spec in resolved]
    if len(names) != len(set(names)):
        raise ValueError('Insights labels must have unique names')
    return resolved


def _source(trace: TraceInsight, dimension: DimensionName) -> str | None:
    summary = trace.summary
    if summary is None:
        return None
    if dimension == 'intent':
        return summary.request
    if dimension == 'failure':
        return ' ; '.join(summary.assistant_errors) or None
    return summary.sentiment_explanation


def _label_results(traces: list[TraceInsight], specs: list[LabelSpec]) -> dict[str, LabelResult]:
    results: dict[str, LabelResult] = {}
    for spec in specs:
        answers = [trace.labels.get(spec.name) for trace in traces]
        values: Counter[str] = Counter()
        confidences: list[float] = []
        low = failed = 0
        for answer in answers:
            if answer is None or answer.error is not None or answer.value is None:
                failed += 1
                continue
            if answer.confidence is not None:
                confidences.append(answer.confidence)
                low += answer.confidence < 0.6
            if spec.kind == 'noul':
                values['yes' if answer.value is True else 'no'] += 1
            elif spec.kind == 'score':
                levels = len(spec.criteria) if isinstance(spec.criteria, list) else 0
                values[str(round(float(answer.value) * max(0, levels - 1)))] += 1
            else:
                values[str(answer.value)] += 1
        results[spec.name] = LabelResult(
            spec=spec,
            counts=dict(values),
            mean_confidence=sum(confidences) / len(confidences) if confidences else None,
            n_low_confidence=low,
            n_failed=failed,
        )
    return results


def _stage(writer: Any, name: str) -> None:
    writer.start_stage(name)


def _stage_end(writer: Any, name: str, error: str | None = None) -> None:
    writer.end_stage(name, error=error)


async def _build_dimension(  # noqa: C901
    dimension: DimensionName,
    traces: list[TraceInsight],
    *,
    client: AsyncOpenAI,
    cache: InsightsCache,
    embedding_model: str,
    summary_model: str,
    max_clusters: int,
    max_subclusters: int,
    outlier_zscore: float | None,
    parallelism: int,
    classifier_model: str,
) -> DimensionResult:
    """Run one dimension's embed → cluster → describe → merge → reduce pipeline."""
    import numpy as np

    from evaluatorq.insights.cluster import centroids, cluster_two_level, nearest_neighbours
    from evaluatorq.insights.merge import merge_similar
    from evaluatorq.insights.reduce import reduce_3d

    source_field = DIMENSION_FIELDS[dimension]
    usable: list[tuple[TraceInsight, str]] = []
    n_no_signal = 0
    for trace in traces:
        text = _source(trace, dimension)
        if text is None or not text.strip():
            n_no_signal += 1
        else:
            usable.append((trace, text))
    result = DimensionResult(name=dimension, source_field=source_field, clusters=[], n_no_signal=n_no_signal)
    result.n_failed = sum(bool(trace.errors) for trace in traces)
    if len(usable) < 5:
        warning = f'dimension {dimension} skipped: only {len(usable)} traces have signal (need at least 5)'
        result.warnings.append(warning)
        logger.warning(warning)
        return result

    group_indices: dict[str | None, list[int]] = {}
    if dimension == 'sentiment':
        for i, (trace, _) in enumerate(usable):
            answer = trace.labels.get('sentiment')
            group = str(answer.value) if answer and answer.error is None and answer.value is not None else 'unknown'
            group_indices.setdefault(group, []).append(i)
    else:
        group_indices[None] = list(range(len(usable)))

    next_base = next_top = 0
    vector_map: dict[str, list[float]] = {}
    for group, indices in group_indices.items():
        if len(indices) < 5:
            warning = f'dimension {dimension} group {group!r} skipped: only {len(indices)} signal traces'
            result.warnings.append(warning)
            logger.warning(warning)
            continue
        texts = [usable[i][1] for i in indices]
        embedded = await embed_texts(texts, client=client, model=embedding_model, cache=cache)
        vectors = np.asarray([embedded[text] for text in texts], dtype=float)
        tree = cluster_two_level(
            vectors,
            max_clusters=max_clusters,
            max_subclusters=max_subclusters,
            min_cluster_size=5,
            outlier_zscore=outlier_zscore,
        )
        base_labels = tree.base_labels
        base_ids = sorted(int(value) for value in np.unique(base_labels) if value != -1)
        cents = centroids(vectors, base_labels)
        neighbours = nearest_neighbours(cents)
        members: dict[int, list[str]] = {base_id: [] for base_id in base_ids}
        examples: dict[int, list[str]] = {base_id: [] for base_id in base_ids}
        member_traces: dict[int, list[TraceInsight]] = {base_id: [] for base_id in base_ids}
        for local_i, base_id_value in enumerate(base_labels):
            base_id = int(base_id_value)
            trace, text = usable[indices[local_i]]
            if base_id == -1:
                trace.assignments[dimension] = ClusterAssignment(top='noise', base='noise')
                result.n_noise += 1
                continue
            members[base_id].append(text)
            member_traces[base_id].append(trace)
        # Five examples nearest each centroid, used for descriptions, merges and the UI.
        for base_id in base_ids:
            local_members = member_traces[base_id]
            member_positions = [i for i, lbl in enumerate(base_labels) if int(lbl) == base_id]
            center = cents[base_id]
            nearest = sorted(member_positions, key=lambda i: float(np.linalg.norm(vectors[i] - center)))[:5]
            examples[base_id] = [usable[indices[i]][1] for i in nearest]

        described = await describe_clusters(
            members,
            neighbours=neighbours,
            dimension=dimension,
            client=client,
            model=summary_model,
            parallelism=parallelism,
        )
        if described and all(isinstance(value, str) for value in described.values()):
            raise RuntimeError('describe failed for every base cluster')
        names = {key: value for key, value in described.items() if isinstance(value, ClusterName)}
        representatives = await merge_similar(
            names, examples, neighbours, client=client, model=classifier_model, parallelism=parallelism
        )
        merged: dict[int, list[int]] = {}
        for base_id in base_ids:
            merged.setdefault(representatives.get(base_id, base_id), []).append(base_id)
        merged_ids = sorted(merged)
        remapped_members: dict[int, list[TraceInsight]] = {
            rep: [trace for old in olds for trace in member_traces[old]] for rep, olds in merged.items()
        }
        merged_names: dict[int, ClusterName] = {}
        for rep, olds in merged.items():
            known = [names[old] for old in olds if old in names]
            if known:
                merged_names[rep] = known[0]
        top_children: dict[int, list[ClusterName]] = {}
        top_base_map: dict[int, int] = {}
        for base_id in base_ids:
            rep = representatives.get(base_id, base_id)
            top_id = tree.top_of_base.get(base_id, base_id)
            top_base_map[rep] = top_id
            if rep in merged_names:
                top_children.setdefault(top_id, []).append(merged_names[rep])
        top_descriptions = (
            await describe_top_level(top_children, client=client, model=summary_model, parallelism=parallelism)
            if top_children
            else {}
        )
        group_slug = f'{group}-' if group is not None else ''
        top_ids = sorted(set(top_base_map.values()))
        local_top_names: dict[int, ClusterName] = {
            top: value for top, value in top_descriptions.items() if isinstance(value, ClusterName)
        }
        base_global: dict[int, str] = {}
        top_global: dict[int, str] = {}
        for top_id in top_ids:
            top_global[top_id] = f'{dimension}-t{group_slug}{next_top}'
            next_top += 1
        for rep in merged_ids:
            base_global[rep] = f'{dimension}-b{group_slug}{next_base}'
            next_base += 1
        for rep, traces_for_base in remapped_members.items():
            top_id = top_base_map[rep]
            base_name = merged_names.get(
                rep, ClusterName(name='Unlabeled cluster', description='Cluster description unavailable.')
            )
            result.clusters.append(
                Cluster(
                    id=base_global[rep],
                    parent_id=top_global[top_id],
                    level='base',
                    name=base_name.name,
                    description=base_name.description,
                    size=len(traces_for_base),
                    trace_ids=[t.trace_id for t in traces_for_base],
                    example_trace_ids=[t.trace_id for t in traces_for_base[:5]],
                    group=group,
                )
            )
            for trace in traces_for_base:
                trace.assignments[dimension] = ClusterAssignment(top=top_global[top_id], base=base_global[rep])
        for top_id in top_ids:
            children = [rep for rep, value in top_base_map.items() if value == top_id]
            child_traces = [trace for rep in children for trace in remapped_members.get(rep, [])]
            name = local_top_names.get(
                top_id, ClusterName(name='Conversation group', description='Related conversation clusters.')
            )
            result.clusters.append(
                Cluster(
                    id=top_global[top_id],
                    parent_id=None,
                    level='top',
                    name=name.name,
                    description=name.description,
                    size=len(child_traces),
                    trace_ids=[t.trace_id for t in child_traces],
                    example_trace_ids=[t.trace_id for t in child_traces[:5]],
                    group=group,
                )
            )

        coords = reduce_3d(vectors)
        if coords is not None:
            for i, coord in enumerate(coords):
                trace = usable[indices[i]][0]
                trace.coords[dimension] = (float(coord[0]), float(coord[1]), float(coord[2]))
    return result


async def insights(  # noqa: C901
    population: InsightsPopulation,
    *,
    labels: Sequence[LabelSpec | str] = (),
    dimensions: Sequence[DimensionName] = ('intent', 'failure', 'sentiment'),
    max_clusters: int = 15,
    max_subclusters: int = 15,
    outlier_zscore: float | None = None,
    summary_model: str = 'openai/gpt-6-luna',
    classifier_model: str = 'typesafe/jev-latest',
    compiler_model: str | None = None,
    embedding_model: str = 'openai/text-embedding-3-small',
    priority_dimension: DimensionName = 'intent',
    parallelism: int = 100,
    cache: bool = True,
    run_name: str | None = None,
    runs_dir: Path | None = None,
    llm_client: AsyncOpenAI | None = None,
    orq_client: Orq | None = None,
) -> InsightsRun:
    """Run trace Insights and persist the partial or complete result with a manifest."""
    specs = _label_specs(labels, dimensions)  # resolve unknown presets before client construction or I/O
    settings = effective_settings()
    compiler_model = compiler_model or settings.compiler_model
    run_id = str(uuid.uuid4())
    name = run_name or f'insights-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}'
    directory = runs_dir or get_insights_runs_dir()
    writer = start_manifest(run_id=run_id, surface='insights', run_name=name, runs_dir=directory)
    now = datetime.now(timezone.utc)
    config = InsightsConfig(
        labels=specs,
        dimensions=list(dimensions),
        summary_model=summary_model,
        classifier_model=classifier_model,
        embedding_model=embedding_model,
        max_clusters=max_clusters,
        max_subclusters=max_subclusters,
        outlier_zscore=outlier_zscore,
        parallelism=parallelism,
        priority_dimension=priority_dimension,
        cache=cache,
    )
    run = InsightsRun(
        run_id=run_id,
        run_name=name,
        created_at=now,
        status='completed',
        stage_failures=[],
        population={},
        config=config,
        traces=[],
        dimensions={},
        labels={},
        priority=None,
        priority_reason=None,
        counts={},
        warnings=[],
    )
    resolved_llm = None
    llm_owned = False
    own_orq = orq_client is None
    resolved_orq = orq_client
    cache_store = InsightsCache(enabled=cache)
    try:
        resolved_client = resolve_llm_client(llm_client)
        resolved_llm = resolved_client.client
        llm_owned = resolved_client.owned
        if resolved_orq is None:
            resolved_orq = resolve_orq_client()
        _stage(writer, 'population')
        try:
            resolved = await resolve_population(
                population,
                orq=resolved_orq,
                client=resolved_llm,
                compiler_model=compiler_model,
                classifier_model=classifier_model,
            )
            run.population = {
                **resolved.echo,
                'n_scanned': resolved.n_scanned,
                'n_matched': len(resolved.traces) if resolved.compiled is None else 0,
                'n_failed_match': 0,
            }
            run.traces = [
                TraceInsight(
                    trace_id=trace.trace_id,
                    span_id=trace.span_id,
                    timestamp=trace.timestamp,
                    agent_name=trace.agent_name or '',
                    project=trace.project or '',
                )
                for trace in resolved.traces
            ]
            _stage_end(writer, 'population')
        except PopulationError as exc:
            message = str(exc)
            run.status = 'error'
            run.stage_failures.append(StageFailure(stage='population', message=message))
            logger.warning('Insights population stage failed: {}', message)
            _stage_end(writer, 'population', message)
            return run

        if not run.traces:
            run.warnings.append('population is empty')
            logger.warning('Insights population is empty')
        else:
            _stage(writer, 'label')
            outcomes = await label_traces(
                resolved.traces,
                labels=specs,
                compiled=resolved.compiled,
                client=resolved_llm,
                model=classifier_model,
                parallelism=parallelism,
            )
            original_by_id = {trace.trace_id: trace for trace in run.traces}
            retained_trace_ids: set[str] = set()
            n_matched = len(outcomes) if resolved.compiled is None else 0
            n_failed_match = 0
            for outcome in outcomes:
                if resolved.compiled is not None and outcome.matched is False:
                    continue
                retained_trace_ids.add(outcome.trace.trace_id)
                if resolved.compiled is not None:
                    if outcome.matched is True:
                        n_matched += 1
                    else:
                        n_failed_match += 1
                item = original_by_id[outcome.trace.trace_id]
                item.labels.update(outcome.answers)
                if outcome.error:
                    item.errors['label'] = outcome.error
                else:
                    failed_labels = [key for key, answer in outcome.answers.items() if answer.error is not None]
                    if failed_labels:
                        item.errors['label'] = 'unreadable label answer(s): ' + ', '.join(failed_labels)
                if resolved.compiled is not None and outcome.matched is None:
                    item.errors['match'] = outcome.error or 'population match could not be determined'
            run.traces = [trace for trace in run.traces if trace.trace_id in retained_trace_ids]
            run.population['n_matched'] = n_matched
            run.population['n_failed_match'] = n_failed_match
            all_label_failed = (
                bool(outcomes)
                and bool(specs or resolved.compiled is not None)
                and all(outcome.error is not None for outcome in outcomes)
            )
            label_error = 'every label request failed' if all_label_failed else None
            if label_error:
                run.status = 'error'
                run.stage_failures.append(StageFailure(stage='label', message=label_error))
                logger.warning('Insights label stage failed: {}', label_error)
            _stage_end(writer, 'label', label_error)

            if run.traces:
                _stage(writer, 'summary')
                summaries = await summarize_traces(
                    [outcome.trace for outcome in outcomes if outcome.trace.trace_id in retained_trace_ids],
                    client=resolved_llm,
                    model=summary_model,
                    cache=cache_store,
                    parallelism=parallelism,
                )
                for trace_id, summary in summaries.items():
                    item = original_by_id[trace_id]
                    if isinstance(summary, str):
                        item.errors['summary'] = summary
                    else:
                        item.summary = summary
                _stage_end(writer, 'summary')

                for dimension in dimensions:
                    stage_name = f'dimension:{dimension}'
                    _stage(writer, stage_name)
                    try:
                        result = await _build_dimension(
                            dimension,
                            run.traces,
                            client=resolved_llm,
                            cache=cache_store,
                            embedding_model=embedding_model,
                            summary_model=summary_model,
                            max_clusters=max_clusters,
                            max_subclusters=max_subclusters,
                            outlier_zscore=outlier_zscore,
                            parallelism=parallelism,
                            classifier_model=classifier_model,
                        )
                        run.dimensions[dimension] = result
                        for warning in result.warnings:
                            run.warnings.append(warning)
                        _stage_end(writer, stage_name)
                    except Exception as exc:  # noqa: BLE001 - a dimension failure must preserve other dimensions
                        message = str(exc)
                        run.status = 'error'
                        run.stage_failures.append(StageFailure(stage=stage_name, message=message, dimension=dimension))
                        logger.warning('Insights {} stage failed: {}', dimension, message)
                        _stage_end(writer, stage_name, message)

            else:
                run.warnings.append('population is empty')
                logger.warning('Insights population is empty after match filtering')
        run.labels = _label_results(run.traces, specs)
        run.counts = {
            'n_traces': len(run.traces),
            'n_failed_traces': sum(bool(trace.errors) for trace in run.traces),
            'per_stage_failed': sum(bool(trace.errors) for trace in run.traces),
        }
        _stage(writer, 'priority')
        priority_dimension_result = run.dimensions.get(priority_dimension)
        satisfaction_spec = next((spec for spec in specs if spec.name == 'customer_satisfaction'), None)
        if priority_dimension_result is not None and satisfaction_spec is not None:
            run.priority, run.priority_reason = priority_points(
                run.traces, priority_dimension_result, satisfaction_spec=satisfaction_spec
            )
        else:
            run.priority_reason = (
                f'priority dimension {priority_dimension!r} is unavailable'
                if priority_dimension_result is None
                else 'customer_satisfaction label was not requested'
            )
        _stage_end(writer, 'priority')
        _stage(writer, 'write')
        run_path = save_run(run, directory)
        _stage_end(writer, 'write')
        if run.status == 'error':
            writer.fail(run.stage_failures[-1].message, stage=run.stage_failures[-1].stage)
        else:
            writer.complete(report_path=run_path)
        return run
    except Exception as exc:  # noqa: BLE001 - setup/runtime failures must leave a persisted terminal run
        message = str(exc)
        run.status = 'error'
        run.stage_failures.append(StageFailure(stage='setup', message=message))
        logger.warning('Insights run setup failed: {}', message)
        return run
    finally:
        cache_store.close()
        if own_orq and resolved_orq is not None:
            await close_orq_client(resolved_orq)
        if llm_owned and resolved_llm is not None:
            await resolved_llm.close()
        # Ensure even a population-stage return/failure has a persisted run file and terminal manifest.
        if not writer.manifest.ended_at:
            try:
                run.counts = run.counts or {'n_traces': len(run.traces), 'n_failed_traces': 0, 'per_stage_failed': 0}
                report = save_run(run, directory)
                if run.status == 'error':
                    writer.fail(run.stage_failures[-1].message, stage=run.stage_failures[-1].stage)
                else:
                    writer.complete(report_path=report)
            except Exception as exc:  # noqa: BLE001 - persistence failure should still terminate manifest
                logger.warning('Could not persist Insights run or manifest {}: {}', run_id, exc)
                writer.fail(str(exc), stage='write')


def insights_sync(*args: Any, **kwargs: Any) -> InsightsRun:
    """Synchronous wrapper around ``insights`` for scripts without an event loop."""
    return asyncio.run(insights(*args, **kwargs))
