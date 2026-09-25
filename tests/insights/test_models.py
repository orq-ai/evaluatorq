"""Unit tests for `evaluatorq.insights.models`."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from evaluatorq.insights.models import InsightsPopulation, InsightsRun
from evaluatorq.trace_finder.models import FacetSelection, NumericFilters


def test_finder_export_and_query_are_exclusive() -> None:
    with pytest.raises(ValueError, match='finder_export'):
        InsightsPopulation(query='x', finder_export=Path('a.json'))


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('facets', FacetSelection(model=frozenset({'gpt-5'}))),
        ('numeric', NumericFilters(tokens_min=10)),
        ('start', datetime(2026, 9, 1, tzinfo=timezone.utc)),
        ('end', datetime(2026, 9, 2, tzinfo=timezone.utc)),
        ('window_days', 14),
        ('limit', 100),
    ],
)
def test_finder_export_rejects_live_population_settings(field: str, value: Any) -> None:
    with pytest.raises(ValueError, match=field):
        InsightsPopulation(finder_export=Path('a.json'), **{field: value})


def test_from_finder_export_sets_path_only() -> None:
    population = InsightsPopulation.from_finder_export(Path('a.json'))
    assert population.finder_export == Path('a.json')
    assert population.query is None
    assert InsightsPopulation.model_validate(population.model_dump()) == population


def test_run_round_trips_json(minimal_run: InsightsRun) -> None:
    assert InsightsRun.model_validate_json(minimal_run.model_dump_json()) == minimal_run
