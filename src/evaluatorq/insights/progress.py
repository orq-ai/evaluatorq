"""Expected Insights stages and labels for manifests and the dashboard."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from evaluatorq.contracts import ManifestStatus, ManifestSurface, RunManifest, StageRecord
from evaluatorq.insights.models import InsightsStageHistory
from evaluatorq.trace_finder.models import FacetSelection, NumericFilters

if TYPE_CHECKING:
    from collections.abc import Sequence

    from evaluatorq.insights.models import DimensionName, InsightsPopulation, InsightsRun, LabelSpec

_HALTING_STAGES = frozenset({'population', 'label', 'summary'})


def _selection_labels(mode: str | None, *, has_filters: bool, has_labels: bool) -> tuple[str | None, str]:
    source = {
        'snapshot': 'Load local traces',
        'export': 'Load Finder matches',
        'query': 'Find matching traces',
        'filter': 'Filter recent traces' if has_filters else 'Load recent traces',
    }.get(mode)
    classify = (
        ('Match and classify traces' if has_labels else 'Match traces')
        if mode == 'query'
        else 'Classify traces'
        if has_labels
        else 'Keep selected traces'
        if has_filters
        else 'Prepare traces'
    )
    return source, classify


def stage_plan(
    population: InsightsPopulation,
    labels: Sequence[LabelSpec],
    dimensions: Sequence[DimensionName],
    *,
    priority_dimension: DimensionName = 'intent',
) -> list[tuple[str, str]]:
    """Describe actual pipeline stages using the selected source and analysis."""
    has_labels = bool(labels) or 'sentiment' in dimensions
    has_filters = (
        population.facets != FacetSelection()
        or population.numeric != NumericFilters()
        or population.start is not None
        or population.end is not None
        or population.window_days != 7
        or population.limit != 500
    )
    mode = (
        'snapshot'
        if population.snapshot_path is not None
        else 'export'
        if population.finder_export is not None
        else 'query'
        if population.query is not None
        else 'filter'
    )
    source, classify = _selection_labels(mode, has_filters=has_filters, has_labels=has_labels)
    if source is None:
        raise ValueError(f'Unknown Insights population mode: {mode}')
    plan = [('population', source), ('label', classify), ('summary', 'Summarize traces')]
    plan.extend((f'dimension:{name}', f'Cluster and map {name}') for name in dimensions)
    priority_ready = priority_dimension in dimensions and any(spec.name == 'customer_satisfaction' for spec in labels)
    plan.extend((
        ('priority', 'Build priority matrix' if priority_ready else 'Check priority matrix'),
        ('write', 'Save run'),
    ))
    return plan


def stage_history_for_save(manifest: RunManifest, *, failed_stage: str | None) -> InsightsStageHistory:
    """Snapshot the live manifest, closing open stages without hiding a write failure.

    An open write is completed when saving the run unless write itself failed. Other open stages
    are errored when the run failed.
    """
    now = datetime.now(tz=timezone.utc)
    stages = [stage.model_copy(deep=True) for stage in manifest.stages]
    for stage in stages:
        if stage.ended_at is not None:
            continue
        stage.status = (
            ManifestStatus.ERROR
            if failed_stage is not None and (stage.name != 'write' or failed_stage == 'write')
            else ManifestStatus.COMPLETED
        )
        stage.ended_at = now
    return InsightsStageHistory(
        planned_stages=list(manifest.planned_stages),
        stage_labels=dict(manifest.stage_labels),
        stages=stages,
    )


def _rebuilt_labels(run: InsightsRun) -> dict[str, str]:
    """Stage labels derivable from what the run file records, using ``stage_plan``'s wording."""
    config = run.config
    dimensions = config.dimensions
    has_labels = bool(config.labels or config.coding_labels or config.coding_analysis) or 'sentiment' in dimensions
    mode = run.population.get('mode')
    has_filters = (
        FacetSelection.model_validate(run.population.get('facets') or {}) != FacetSelection()
        or NumericFilters.model_validate(run.population.get('numeric') or {}) != NumericFilters()
        or run.population.get('window_days', 7) != 7
        or run.population.get('limit', 500) != 500
    )
    start, end = run.population.get('start'), run.population.get('end')
    # Resolved dates exist even for default runs; only a non-default span proves a selection.
    if isinstance(start, str) and isinstance(end, str):
        has_filters |= datetime.fromisoformat(end) - datetime.fromisoformat(start) != timedelta(days=7)
    source, classify = _selection_labels(
        mode if isinstance(mode, str) else None, has_filters=has_filters, has_labels=has_labels
    )
    priority_ready = config.priority_dimension in dimensions and any(
        spec.name == 'customer_satisfaction' for spec in config.labels
    )
    labels = {'label': classify, 'summary': 'Summarize traces'}
    if source is not None:
        labels['population'] = source
    labels.update({f'dimension:{name}': f'Cluster and map {name}' for name in dimensions})
    labels['priority'] = 'Build priority matrix' if priority_ready else 'Check priority matrix'
    labels['write'] = 'Save run'
    return labels


def run_manifest_for(run: InsightsRun) -> RunManifest:
    """Terminal manifest for a saved *run*, from its stored stage history or rebuilt from its contents."""
    failed = run.status == 'error'
    history = run.stage_history
    failures = {failure.stage: failure for failure in run.stage_failures}
    if history is not None:
        planned = list(history.planned_stages)
        labels = dict(history.stage_labels)
        stages = [stage.model_copy(deep=True) for stage in history.stages]
    else:
        planned = [
            'population',
            'label',
            'summary',
            *(f'dimension:{d}' for d in run.config.dimensions),
            'priority',
            'write',
        ]
        all_labels = _rebuilt_labels(run)
        labels = {name: all_labels[name] for name in planned if name in all_labels}
        # A failed dimension never stops the pipeline; a failed population, label or summary stage
        # stops the analysis stages after it, while priority and write still run.
        halting = [index for index, name in enumerate(planned) if name in failures and name in _HALTING_STAGES]
        first_halt = -1 if 'setup' in failures else min(halting) if halting else None
        stages = []
        for index, name in enumerate(planned):
            if name in failures:
                status = ManifestStatus.ERROR
            elif not run.traces and (
                name == 'summary'
                or name.startswith('dimension:')
                or (
                    name == 'label'
                    and not (run.population.get('mode') == 'query' and run.population.get('n_scanned', 0) > 0)
                )
            ):
                continue
            elif (
                name == 'write'
                or (name == 'priority' and first_halt not in (-1, 0))
                or first_halt is None
                or index < first_halt
            ):
                status = ManifestStatus.COMPLETED
            else:
                continue
            stages.append(StageRecord(name=name, status=status, started_at=run.created_at))
    last_failure = run.stage_failures[-1] if failed and run.stage_failures else None
    return RunManifest(
        run_id=run.run_id,
        surface=ManifestSurface.INSIGHTS,
        run_name=run.run_name,
        status=ManifestStatus.ERROR if failed else ManifestStatus.COMPLETED,
        stage=last_failure.stage if last_failure is not None else 'write',
        stages=stages,
        planned_stages=planned,
        stage_labels=labels,
        started_at=run.created_at,
        updated_at=run.created_at,
        ended_at=run.created_at,
        error=last_failure.message if last_failure is not None else None,
    )
