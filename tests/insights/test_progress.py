from __future__ import annotations

from datetime import timedelta

import pytest
from evaluatorq.contracts import ManifestStatus
from evaluatorq.insights.models import DimensionName, InsightsConfig, InsightsPopulation, InsightsRun, StageFailure
from evaluatorq.insights.progress import run_manifest_for, stage_plan
from evaluatorq.trace_finder.models import FacetSelection


def test_filter_only_run_labels_selection_stages_accurately() -> None:
    population = InsightsPopulation(facets=FacetSelection(provider=frozenset({'openai'})))

    stages = dict(stage_plan(population, labels=(), dimensions=()))

    assert stages['population'] == 'Filter recent traces'
    assert stages['label'] == 'Keep selected traces'


def test_nondefault_limit_is_reported_as_a_population_selection() -> None:
    population = InsightsPopulation(limit=100)

    stages = dict(stage_plan(population, labels=(), dimensions=()))

    assert stages['population'] == 'Filter recent traces'
    assert stages['label'] == 'Keep selected traces'


def test_empty_query_is_still_a_query_population() -> None:
    population = InsightsPopulation(query='')

    stages = dict(stage_plan(population, labels=(), dimensions=()))

    assert stages['population'] == 'Find matching traces'
    assert stages['label'] == 'Match traces'


def _legacy_run(
    minimal_run: InsightsRun, *, dimensions: list[DimensionName], failures: list[StageFailure] | None = None
) -> InsightsRun:
    return minimal_run.model_copy(
        update={
            'config': InsightsConfig(labels=[], dimensions=dimensions),
            'status': 'error' if failures else 'completed',
            'stage_failures': failures or [],
            'stage_history': None,
        }
    )


def _statuses(manifest) -> dict[str, str]:
    return {record.name: record.status.value for record in manifest.stages}


def test_legacy_completed_run_is_rebuilt_with_every_stage_completed(minimal_run: InsightsRun) -> None:
    manifest = run_manifest_for(_legacy_run(minimal_run, dimensions=['intent', 'failure']))

    assert manifest.planned_stages == ['population', 'label', 'summary', 'dimension:intent', 'dimension:failure', 'priority', 'write']
    assert _statuses(manifest) == dict.fromkeys(manifest.planned_stages, 'completed')
    assert manifest.stage_labels['dimension:failure'] == 'Cluster and map failure'
    assert manifest.stage_labels['write'] == 'Save run'
    assert manifest.status is ManifestStatus.COMPLETED
    assert manifest.error is None


def test_legacy_dimension_failure_leaves_every_other_stage_completed(minimal_run: InsightsRun) -> None:
    failure = StageFailure(stage='dimension:failure', message='clustering broke')
    manifest = run_manifest_for(_legacy_run(minimal_run, dimensions=['intent', 'failure'], failures=[failure]))

    assert _statuses(manifest) == {
        'population': 'completed',
        'label': 'completed',
        'summary': 'completed',
        'dimension:intent': 'completed',
        'dimension:failure': 'error',
        'priority': 'completed',
        'write': 'completed',
    }
    assert manifest.status is ManifestStatus.ERROR
    assert manifest.stage == 'dimension:failure'
    assert manifest.error == 'clustering broke'


def test_legacy_summary_failure_skips_dimensions_but_priority_and_write_still_ran(minimal_run: InsightsRun) -> None:
    failure = StageFailure(stage='summary', message='model not enabled')
    manifest = run_manifest_for(_legacy_run(minimal_run, dimensions=['intent', 'failure'], failures=[failure]))

    assert _statuses(manifest) == {
        'population': 'completed',
        'label': 'completed',
        'summary': 'error',
        'priority': 'completed',
        'write': 'completed',
    }
    assert manifest.stage == 'summary'


def test_legacy_population_failure_never_reached_priority(minimal_run: InsightsRun) -> None:
    failure = StageFailure(stage='population', message='no traces')
    manifest = run_manifest_for(_legacy_run(minimal_run, dimensions=['intent'], failures=[failure]))

    assert _statuses(manifest) == {'population': 'error', 'write': 'completed'}


@pytest.mark.parametrize('selection', [
    {'facets': {'provider': ['openai']}},
    {'numeric': {'tokens_min': 0}},
    {'limit': 100},
    {'window_days': 3},
    {'start': '2026-10-01T00:00:00+00:00', 'end': '2026-10-04T00:00:00+00:00'},
])
def test_legacy_filter_selection_preserves_stage_names(
    minimal_run: InsightsRun, selection: dict[str, object]
) -> None:
    run = _legacy_run(minimal_run, dimensions=[])
    run.population = {'mode': 'filter', **selection}
    manifest = run_manifest_for(run)
    assert manifest.stage_labels['population'] == 'Filter recent traces'
    assert manifest.stage_labels['label'] == 'Keep selected traces'


def test_resolved_default_window_is_not_an_explicit_filter(minimal_run: InsightsRun) -> None:
    run = _legacy_run(minimal_run, dimensions=[])
    run.population = {
        'mode': 'filter',
        'facets': FacetSelection().model_dump(mode='json'),
        'numeric': {},
        'start': (run.created_at - timedelta(days=7)).isoformat(),
        'end': run.created_at.isoformat(),
        'limit': 500,
    }
    manifest = run_manifest_for(run)
    assert manifest.stage_labels['population'] == 'Load recent traces'
    assert manifest.stage_labels['label'] == 'Prepare traces'


@pytest.mark.parametrize('scanned, expected', [
    (0, {'population': 'completed', 'priority': 'completed', 'write': 'completed'}),
    (4, {'population': 'completed', 'label': 'completed', 'priority': 'completed', 'write': 'completed'}),
])
def test_legacy_empty_query_preserves_only_stages_that_ran(
    minimal_run: InsightsRun, scanned: int, expected: dict[str, str]
) -> None:
    run = _legacy_run(minimal_run, dimensions=['intent'])
    run.population = {'mode': 'query', 'n_scanned': scanned}
    run.traces = []
    assert _statuses(run_manifest_for(run)) == expected


def test_legacy_setup_failure_does_not_complete_analysis(minimal_run: InsightsRun) -> None:
    run = _legacy_run(minimal_run, dimensions=['intent'], failures=[StageFailure(stage='setup', message='client failed')])
    assert _statuses(run_manifest_for(run)) == {'write': 'completed'}


def test_legacy_stage_duration_is_unknown(minimal_run: InsightsRun) -> None:
    manifest = run_manifest_for(_legacy_run(minimal_run, dimensions=['intent']))
    assert all(record.duration_seconds is None for record in manifest.stages)
