from __future__ import annotations

from evaluatorq.insights.models import InsightsPopulation
from evaluatorq.insights.progress import stage_plan
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
