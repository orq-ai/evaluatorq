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

from evaluatorq import __version__
from evaluatorq.common.llm_client import resolve_llm_client
from evaluatorq.common.model_roles import role_model
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
    label_key,
    real_assistant_errors,
)
from evaluatorq.insights.population import PopulationError, resolve_population
from evaluatorq.insights.presets import CODING_LABELS, DIMENSION_FIELDS, LABEL_PRESETS, SENTIMENT
from evaluatorq.insights.priority import priority_points
from evaluatorq.insights.progress import stage_plan
from evaluatorq.insights.store import get_insights_runs_dir, save_run
from evaluatorq.insights.summarize import summarize_traces
from evaluatorq.insights.transcript import tool_stats
from evaluatorq.insights.usage import UsageLedger
from evaluatorq.trace_finder.settings import effective_settings

MIN_CLUSTER_SIZE = 5
LOW_CONFIDENCE = 0.6

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
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
    if clash := sorted(set(names) & {spec.name for spec in CODING_LABELS}):
        raise ValueError(f'Insights label names are reserved for coding analysis: {", ".join(clash)}')
    return resolved


def _source(trace: TraceInsight, dimension: DimensionName) -> str | None:
    summary = trace.summary
    if summary is None:
        return None
    if dimension == 'intent':
        return summary.request
    if dimension == 'failure':
        return ' ; '.join(real_assistant_errors(summary)) or None
    return summary.sentiment_explanation


def _label_results(traces: list[TraceInsight], specs: list[LabelSpec]) -> dict[str, LabelResult]:
    results: dict[str, LabelResult] = {}
    for spec in specs:
        answers = [trace.labels.get(spec.name) for trace in traces]
        values: Counter[str] = Counter()
        confidences: list[float] = []
        low = failed = 0
        not_asked = 0
        for answer in answers:
            if answer is None:
                # Not asked: a coding label on a trace the coding-agent check did not answer yes for.
                not_asked += 1
                continue
            if answer.error is not None or answer.value is None:
                failed += 1
                continue
            if answer.confidence is not None:
                confidences.append(answer.confidence)
                low += answer.confidence < LOW_CONFIDENCE
            values[label_key(spec, answer.value)] += 1
        results[spec.name] = LabelResult(
            spec=spec,
            counts=dict(values),
            mean_confidence=sum(confidences) / len(confidences) if confidences else None,
            n_low_confidence=low,
            n_failed=failed,
            n_not_asked=not_asked,
        )
    return results


def _stage(writer: Any, name: str) -> None:
    writer.start_stage(name)


def _stage_end(writer: Any, name: str, error: str | None = None) -> None:
    writer.end_stage(name, error=error)


def _dimension_failed_count(traces: list[TraceInsight], dimension: DimensionName) -> int:
    """Count traces unclassified by this dimension, not unrelated label or match errors."""
    key = f'dimension:{dimension}'
    return sum(key in trace.errors for trace in traces)


def _discard_dimension(traces: list[TraceInsight], dimension: DimensionName) -> None:
    """Drop the partial per-trace output of a dimension whose stage failed.

    `_build_dimension` writes assignments and coordinates onto the traces as it goes, but the
    dimension's clusters are only persisted when it returns. Without this, a failed dimension
    would leave traces pointing at cluster ids that no `DimensionResult` contains.
    """
    for trace in traces:
        trace.assignments.pop(dimension, None)
        trace.coords.pop(dimension, None)
        trace.errors.pop(f'dimension:{dimension}', None)


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
    usage: UsageLedger | None = None,
) -> DimensionResult:
    """Run one dimension's embed → cluster → describe → merge → reduce pipeline."""
    import numpy as np

    from evaluatorq.insights.cluster import centroids, cluster_two_level, nearest_neighbours
    from evaluatorq.insights.merge import merge_similar
    from evaluatorq.insights.reduce import reduce_3d

    source_field = DIMENSION_FIELDS[dimension]
    usable: list[tuple[TraceInsight, str]] = []
    n_no_signal = 0
    n_sentinel_only = 0
    for trace in traces:
        if 'summary' in trace.errors:
            continue
        if (
            dimension == 'failure'
            and trace.summary is not None
            and trace.summary.assistant_errors
            and not real_assistant_errors(trace.summary)
        ):
            n_sentinel_only += 1
        text = _source(trace, dimension)
        if text is None or not text.strip():
            n_no_signal += 1
        else:
            usable.append((trace, text))
    if n_sentinel_only:
        logger.info(
            'dimension failure: {} trace(s) listed only placeholder errors and count as no signal',
            n_sentinel_only,
        )
    result = DimensionResult(name=dimension, source_field=source_field, clusters=[], n_no_signal=n_no_signal)
    result.n_failed = _dimension_failed_count(traces, dimension)
    if len(usable) < 5:
        count = len(usable)
        noun = 'trace has' if count == 1 else 'traces have'
        warning = f'dimension {dimension} skipped: only {count} {noun} signal (need at least 5)'
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
    for group, indices in group_indices.items():
        if len(indices) == 1:
            warning = f'dimension {dimension} group {group!r} has one signal trace; marked unclassified'
            result.warnings.append(warning)
            logger.warning(warning)
            trace = usable[indices[0]][0]
            trace.assignments[dimension] = ClusterAssignment(top='unclassified', base='unclassified')
            trace.errors[f'dimension:{dimension}'] = warning
            continue
        texts = [usable[i][1] for i in indices]
        embedded = await embed_texts(texts, client=client, model=embedding_model, cache=cache, usage=usage)
        vectors = np.asarray([embedded[text] for text in texts], dtype=float)
        tree = await asyncio.to_thread(
            cluster_two_level,
            vectors,
            max_clusters=max_clusters,
            max_subclusters=max_subclusters,
            min_cluster_size=MIN_CLUSTER_SIZE,
            outlier_zscore=outlier_zscore,
        )
        base_labels = tree.base_labels
        base_ids = sorted(int(value) for value in np.unique(base_labels) if value != -1)
        cents = centroids(vectors, base_labels)
        neighbours = nearest_neighbours(cents)
        members: dict[int, list[str]] = {base_id: [] for base_id in base_ids}
        examples: dict[int, list[str]] = {base_id: [] for base_id in base_ids}
        example_ids: dict[int, list[str]] = {base_id: [] for base_id in base_ids}
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
            member_positions = [i for i, lbl in enumerate(base_labels) if int(lbl) == base_id]
            center = cents[base_id]
            nearest = sorted(member_positions, key=lambda i: float(np.linalg.norm(vectors[i] - center)))[:5]
            examples[base_id] = [usable[indices[i]][1] for i in nearest]
            example_ids[base_id] = [usable[indices[i]][0].trace_id for i in nearest]

        described = await describe_clusters(
            members,
            neighbours=neighbours,
            dimension=dimension,
            client=client,
            model=summary_model,
            parallelism=parallelism,
            usage=usage,
        )
        if described and all(isinstance(value, str) for value in described.values()):
            raise RuntimeError('describe failed for every base cluster')
        names = {key: value for key, value in described.items() if isinstance(value, ClusterName)}
        same_top_neighbours = {
            base_id: [
                neighbour
                for neighbour in neighbours.get(base_id, [])
                if tree.top_of_base.get(neighbour) == tree.top_of_base.get(base_id)
            ]
            for base_id in base_ids
        }
        merge = await merge_similar(
            names,
            examples,
            same_top_neighbours,
            client=client,
            model=classifier_model,
            parallelism=parallelism,
            usage=usage,
        )
        if merge.n_pairs and merge.n_failed == merge.n_pairs:
            raise RuntimeError(f'merge failed for every one of {merge.n_pairs} cluster pair checks')
        if merge.n_failed:
            warning = (
                f'dimension {dimension}: merge: {merge.n_failed}/{merge.n_pairs} pair checks failed; '
                'those clusters were left unmerged'
            )
            result.warnings.append(warning)
            logger.warning(warning)
        representatives = dict(merge.representatives)
        # A merge that folds a whole top group into one cluster undoes the two-subcluster split: keep them apart.
        by_top: dict[int, list[int]] = {}
        for base_id in base_ids:
            by_top.setdefault(tree.top_of_base.get(base_id, base_id), []).append(base_id)
        for siblings in by_top.values():
            if len(siblings) > 1 and len({representatives.get(b, b) for b in siblings}) == 1:
                representatives.update({b: b for b in siblings})
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
            await describe_top_level(
                top_children, client=client, model=summary_model, parallelism=parallelism, usage=usage
            )
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
                    example_trace_ids=[trace_id for old in merged[rep] for trace_id in example_ids.get(old, [])][:5],
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

        coords = await asyncio.to_thread(reduce_3d, vectors)
        if coords is not None:
            for i, coord in enumerate(coords):
                trace = usable[indices[i]][0]
                trace.coords[dimension] = (float(coord[0]), float(coord[1]), float(coord[2]))
        else:
            warning = f'dimension {dimension} group {group!r}: UMAP unavailable for {len(indices)} traces'
            result.warnings.append(warning)
            logger.warning(warning)
    result.n_failed = _dimension_failed_count(traces, dimension)
    return result


async def insights(  # noqa: C901
    population: InsightsPopulation,
    *,
    labels: Sequence[LabelSpec | str] = (),
    dimensions: Sequence[DimensionName] = ('intent', 'failure', 'sentiment'),
    max_clusters: int = 15,
    max_subclusters: int = 15,
    outlier_zscore: float | None = None,
    summary_model: str | None = None,
    classifier_model: str | None = None,
    compiler_model: str | None = None,
    embedding_model: str | None = None,
    priority_dimension: DimensionName = 'intent',
    parallelism: int = 100,
    cache: bool = True,
    coding_analysis: bool = False,
    coding_labels: Sequence[str] | None = None,
    run_name: str | None = None,
    runs_dir: Path | None = None,
    _run_id: str | None = None,
    _finder_export_source: Path | None = None,
    _finder_export_sha256: str | None = None,
    _source_name: str | None = None,
    _on_saved: Callable[[Path], None] | None = None,
    llm_client: AsyncOpenAI | None = None,
    orq_client: Orq | None = None,
) -> InsightsRun:
    """Run trace Insights and persist the partial or complete result with a manifest.

    `coding_analysis` retains the legacy full coding bundle. `coding_labels`
    selects a subset of coding labels while keeping the coding-agent gate.
    """
    specs = _label_specs(labels, dimensions)  # resolve unknown presets before client construction or I/O
    coding_by_name = {spec.name: spec for spec in CODING_LABELS[1:]}
    if coding_analysis or coding_labels is None:
        selected_coding = list(CODING_LABELS[1:]) if coding_analysis else []
    else:
        if len(set(coding_labels)) != len(coding_labels):
            raise ValueError('Insights coding labels must have unique names')
        unknown = set(coding_labels) - set(coding_by_name)
        if unknown:
            raise ValueError(f'unknown Insights coding label(s): {", ".join(sorted(unknown))}')
        selected_coding = [coding_by_name[name] for name in coding_labels]
    coding_enabled = coding_analysis or (coding_labels is not None and bool(selected_coding))
    for limit_name, limit_value in (
        ('max_clusters', max_clusters),
        ('max_subclusters', max_subclusters),
        ('parallelism', parallelism),
    ):
        if limit_value < 1:
            raise ValueError(f'{limit_name} must be positive')
    settings = effective_settings()
    compiler_model = compiler_model or role_model('fast', task='finder.compiler')
    # An explicit classifier_model covers both stages; otherwise population search
    # follows finder.classifier and labelling follows insights.labels.
    finder_classifier_model = classifier_model or role_model('classifier', task='finder.classifier')
    classifier_model = classifier_model or role_model('classifier', task='insights.labels')
    summary_model = summary_model or role_model('smart', task='insights.summary')
    embedding_model = embedding_model or role_model('embedding', task='insights.embedding')
    run_id = _run_id or str(uuid.uuid4())
    name = run_name or f'insights-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}'
    directory = runs_dir or get_insights_runs_dir()
    now = datetime.now(timezone.utc)
    config = InsightsConfig(
        labels=specs,
        dimensions=list(dimensions),
        summary_model=summary_model,
        classifier_model=classifier_model,
        embedding_model=embedding_model,
        compiler_model=compiler_model,
        max_clusters=max_clusters,
        max_subclusters=max_subclusters,
        outlier_zscore=outlier_zscore,
        parallelism=parallelism,
        priority_dimension=priority_dimension,
        cache=cache,
        coding_analysis=coding_analysis,
        coding_labels=selected_coding,
    )
    run = InsightsRun(
        evaluatorq_version=__version__,
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
    plan = stage_plan(
        population,
        [*specs, *(CODING_LABELS[:1] if coding_enabled else ()), *selected_coding],
        dimensions,
        priority_dimension=priority_dimension,
    )
    writer = start_manifest(
        run_id=run_id,
        surface='insights',
        run_name=name,
        runs_dir=directory,
        planned_stages=[stage for stage, _ in plan],
        stage_labels=dict(plan),
        parallelism=parallelism,
    )
    resolved_llm: AsyncOpenAI | None = None
    llm_owned = False
    callback_attempted = False
    own_orq = orq_client is None
    resolved_orq = orq_client
    cache_store = InsightsCache(enabled=cache)
    ledger = UsageLedger()

    def notify_saved(path: Path) -> None:
        nonlocal callback_attempted
        if _on_saved is None or callback_attempted:
            return
        # A callback may perform external side effects before raising. Attempt it
        # at most once, even when finalization has to persist the run again.
        callback_attempted = True
        _on_saved(path)

    def ensure_llm_client() -> AsyncOpenAI:
        nonlocal resolved_llm, llm_owned
        if resolved_llm is None:
            resolved_client = resolve_llm_client(llm_client, max_retries=0)
            resolved_llm = resolved_client.client
            llm_owned = resolved_client.owned
        return resolved_llm

    try:
        # Query populations need an LLM to compile and match the query. Filter
        # and Finder populations can be loaded without LLM credentials, so wait
        # until after population resolution to construct a client for analysis.
        if population.query is not None:
            ensure_llm_client()
        if resolved_orq is None and population.snapshot_path is None:
            resolved_orq = resolve_orq_client()
        _stage(writer, 'population')
        try:
            resolved = await resolve_population(
                population,
                orq=resolved_orq,
                client=resolved_llm,
                compiler_model=compiler_model,
                classifier_model=finder_classifier_model,
            )
            run.population = {
                **resolved.echo,
                'n_scanned': resolved.n_scanned,
                'n_matched': len(resolved.traces) if not resolved.compiled else 0,
                'n_failed_match': 0,
            }
            if _finder_export_source is not None and population.finder_export is not None:
                run.population['finder_export'] = str(_finder_export_source)
                if _finder_export_sha256 is not None:
                    run.population['finder_export_sha256'] = _finder_export_sha256
            if _source_name:
                run.population['source_name'] = _source_name
            truncated = resolved.echo.get('n_projection_truncated', 0)
            if truncated:
                trace_noun = 'trace' if len(resolved.traces) == 1 else 'traces'
                run.warnings.append(
                    f'Projection budget: {truncated} of {len(resolved.traces)} {trace_noun} exceeded the model projection budget; '
                    f'{resolved.echo.get("n_omitted_messages", 0)} of '
                    f'{resolved.echo.get("n_source_messages", 0)} whole messages were omitted from classification and summaries.'
                )
            run.traces = [
                TraceInsight(
                    trace_id=trace.trace_id,
                    span_id=trace.span_id,
                    timestamp=trace.timestamp,
                    agent_name=trace.agent_name or '',
                    project=trace.project or '',
                    tool_stats=tool_stats(trace),
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

        missing_export_ids = run.population.get('n_missing_export_ids', 0)
        if isinstance(missing_export_ids, int) and missing_export_ids > 0:
            matched_ids = run.population.get('n_matched_ids', 0)
            trace_word = 'trace' if len(run.traces) == 1 else 'traces'
            run.warnings.append(
                f'{missing_export_ids} of {matched_ids} matched Finder traces could not be reloaded from Orq; '
                f'{len(run.traces)} {trace_word} will be analyzed.'
            )

        if not run.traces:
            run.warnings.append('population is empty')
            logger.warning('Insights population is empty')
        else:
            client = ensure_llm_client()
            _stage(writer, 'label')
            outcomes = await label_traces(
                resolved.traces,
                labels=specs,
                compiled=resolved.compiled,
                client=client,
                model=classifier_model,
                parallelism=parallelism,
                usage=ledger,
                on_progress=lambda done, total: writer.stage_progress('label', done, total),
                coding=coding_enabled,
                coding_labels=selected_coding,
            )
            original_by_id = {trace.trace_id: trace for trace in run.traces}
            retained_trace_ids: set[str] = set()
            n_matched = len(outcomes) if not resolved.compiled else 0
            n_failed_match = 0
            for outcome in outcomes:
                if resolved.compiled and outcome.matched is False:
                    continue
                if resolved.compiled:
                    if outcome.matched is True:
                        n_matched += 1
                    else:
                        n_failed_match += 1
                item = original_by_id[outcome.trace.trace_id]
                if resolved.compiled and outcome.matched is None:
                    message = outcome.error or 'population match could not be determined'
                    item.errors['match'] = message
                    run.warnings.append(
                        f'Trace {outcome.trace.trace_id} was excluded because its population match '
                        f'could not be determined: {message}'
                    )
                    continue
                retained_trace_ids.add(outcome.trace.trace_id)
                item.labels.update(outcome.answers)
                if outcome.error:
                    item.errors['label'] = outcome.error
                else:
                    failed_labels = [key for key, answer in outcome.answers.items() if answer.error is not None]
                    if failed_labels:
                        item.errors['label'] = 'unreadable label answer(s): ' + ', '.join(failed_labels)
            run.traces = [trace for trace in run.traces if trace.trace_id in retained_trace_ids]
            run.population['n_matched'] = n_matched
            run.population['n_failed_match'] = n_failed_match
            all_label_failed = (
                bool(outcomes)
                and bool(specs or resolved.compiled)
                and all(outcome.error is not None for outcome in outcomes)
            )
            label_error = 'every label request failed' if all_label_failed else None
            if label_error:
                run.status = 'error'
                run.stage_failures.append(StageFailure(stage='label', message=label_error))
                logger.warning('Insights label stage failed: {}', label_error)
            _stage_end(writer, 'label', label_error)

            if label_error and not resolved.compiled:
                skipped_message = 'skipped because every label request failed'
                _stage(writer, 'summary')
                _stage_end(writer, 'summary', skipped_message)
                run.warnings.append(f'Summarize traces stage {skipped_message}')
                for dimension in dimensions:
                    stage_name = f'dimension:{dimension}'
                    _stage(writer, stage_name)
                    _stage_end(writer, stage_name, skipped_message)
                    run.warnings.append(f'{stage_name} stage {skipped_message}')

            if run.traces and not (label_error and not resolved.compiled):
                _stage(writer, 'summary')
                summaries = await summarize_traces(
                    [outcome.trace for outcome in outcomes if outcome.trace.trace_id in retained_trace_ids],
                    client=client,
                    model=summary_model,
                    cache=cache_store,
                    parallelism=parallelism,
                    usage=ledger,
                    on_progress=lambda done, total: writer.stage_progress('summary', done, total),
                )
                for trace_id, summary in summaries.items():
                    item = original_by_id[trace_id]
                    if isinstance(summary, str):
                        item.errors['summary'] = summary
                    else:
                        item.summary = summary
                summary_errors = [value for value in summaries.values() if isinstance(value, str)]
                summary_error = (
                    f'every summary request failed: {summary_errors[0]}'
                    if summaries and len(summary_errors) == len(summaries)
                    else None
                )
                if summary_error:
                    run.status = 'error'
                    run.stage_failures.append(StageFailure(stage='summary', message=summary_error))
                    logger.warning('Insights summary stage failed: {}', summary_error)
                _stage_end(writer, 'summary', summary_error)

                for dimension in dimensions if summary_error is None else ():
                    stage_name = f'dimension:{dimension}'
                    _stage(writer, stage_name)
                    try:
                        result = await _build_dimension(
                            dimension,
                            run.traces,
                            client=client,
                            cache=cache_store,
                            embedding_model=embedding_model,
                            summary_model=summary_model,
                            max_clusters=max_clusters,
                            max_subclusters=max_subclusters,
                            outlier_zscore=outlier_zscore,
                            parallelism=parallelism,
                            classifier_model=classifier_model,
                            usage=ledger,
                        )
                        run.dimensions[dimension] = result
                        for warning in result.warnings:
                            run.warnings.append(warning)
                        _stage_end(writer, stage_name)
                    except Exception as exc:  # noqa: BLE001 - a dimension failure must preserve other dimensions
                        message = str(exc)
                        _discard_dimension(run.traces, dimension)
                        run.status = 'error'
                        run.stage_failures.append(StageFailure(stage=stage_name, message=message, dimension=dimension))
                        logger.warning('Insights {} stage failed: {}', dimension, message)
                        _stage_end(writer, stage_name, message)

            elif not run.traces:
                run.warnings.append('population is empty')
                logger.warning('Insights population is empty after match filtering')
        run.labels = _label_results(
            run.traces, [*specs, *(CODING_LABELS[:1] if coding_enabled else ()), *selected_coding]
        )
        run.counts = {
            'n_traces': len(run.traces),
            'n_failed_traces': sum(bool(trace.errors) for trace in run.traces),
            'per_stage_failed': sum(len(trace.errors) for trace in run.traces),
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
        run.cost_by_stage = ledger.totals()
        run_path = save_run(run, directory)
        notify_saved(run_path)
        _stage_end(writer, 'write')
        if run.status == 'error':
            writer.fail(run.stage_failures[-1].message, stage=run.stage_failures[-1].stage)
        else:
            writer.complete(report_path=run_path)
        return run
    except Exception as exc:  # noqa: BLE001 - setup/runtime failures must leave a persisted terminal run
        message = str(exc)
        stage = writer.manifest.stage or 'setup'
        run.status = 'error'
        run.stage_failures.append(StageFailure(stage=stage, message=message))
        logger.warning('Insights {} stage failed: {}', stage, message)
        return run
    finally:
        try:
            cache_store.close()
        except Exception as exc:  # noqa: BLE001 - cleanup must not skip terminal persistence
            logger.warning('Could not close Insights cache for run {}: {}', run_id, exc)
        if own_orq and resolved_orq is not None:
            try:
                await close_orq_client(resolved_orq)
            except Exception as exc:  # noqa: BLE001 - cleanup must not skip terminal persistence
                logger.warning('Could not close Insights Orq client for run {}: {}', run_id, exc)
        if llm_owned and resolved_llm is not None:
            try:
                await resolved_llm.close()
            except Exception as exc:  # noqa: BLE001 - cleanup must not skip terminal persistence
                logger.warning('Could not close Insights LLM client for run {}: {}', run_id, exc)
        # Ensure even a population-stage return/failure has a persisted run file and terminal manifest.
        if not writer.manifest.ended_at:
            try:
                run.counts = run.counts or {'n_traces': len(run.traces), 'n_failed_traces': 0, 'per_stage_failed': 0}
                run.cost_by_stage = ledger.totals()
                report = save_run(run, directory)
                notify_saved(report)
                if run.status == 'error':
                    writer.fail(run.stage_failures[-1].message, stage=run.stage_failures[-1].stage)
                else:
                    writer.complete(report_path=report)
            except Exception as exc:  # noqa: BLE001 - persistence failure should still terminate manifest
                message = str(exc)
                run.status = 'error'
                run.stage_failures.append(StageFailure(stage='write', message=message))
                logger.warning('Could not persist Insights run or manifest {}: {}', run_id, message)
                writer.fail(message, stage='write')


def insights_sync(*args: Any, **kwargs: Any) -> InsightsRun:
    """Synchronous wrapper around ``insights`` for scripts without an event loop."""
    return asyncio.run(insights(*args, **kwargs))
