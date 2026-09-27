"""Expected Insights stages and labels for manifests and the dashboard."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from evaluatorq.insights.models import DimensionName, InsightsPopulation, LabelSpec

from evaluatorq.trace_finder.models import FacetSelection, NumericFilters


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
    )
    source = (
        'Load Finder matches'
        if population.finder_export is not None
        else 'Find matching traces'
        if population.query
        else 'Filter recent traces'
        if has_filters
        else 'Load recent traces'
    )
    classify = (
        'Match and classify traces'
        if population.query and has_labels
        else 'Match traces'
        if population.query
        else 'Classify traces'
        if has_labels
        else 'Keep selected traces'
        if has_filters
        else 'Prepare traces'
    )
    plan = [('population', source), ('label', classify), ('summary', 'Summarize traces')]
    plan.extend((f'dimension:{name}', f'Cluster and map {name}') for name in dimensions)
    priority_ready = priority_dimension in dimensions and any(spec.name == 'customer_satisfaction' for spec in labels)
    plan.extend((
        ('priority', 'Build priority matrix' if priority_ready else 'Check priority matrix'),
        ('write', 'Save run'),
    ))
    return plan
