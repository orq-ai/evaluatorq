from __future__ import annotations

from evaluatorq.insights.models import InsightsPopulation
from evaluatorq.insights.progress import stage_plan
from evaluatorq.trace_finder.models import FacetSelection


def test_filter_only_run_labels_selection_stages_accurately() -> None:
    population = InsightsPopulation(facets=FacetSelection(provider=frozenset({'openai'})))

    stages = dict(stage_plan(population, labels=(), dimensions=()))

    assert stages['population'] == 'Filter recent traces'
    assert stages['label'] == 'Keep selected traces'
