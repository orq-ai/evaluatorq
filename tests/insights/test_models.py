"""Unit tests for `evaluatorq.insights.models`."""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluatorq.insights.models import InsightsPopulation, InsightsRun


def test_finder_export_and_query_are_exclusive() -> None:
    with pytest.raises(ValueError, match='finder_export'):
        InsightsPopulation(query='x', finder_export=Path('a.json'))


def test_from_finder_export_sets_path_only() -> None:
    population = InsightsPopulation.from_finder_export(Path('a.json'))
    assert population.finder_export == Path('a.json')
    assert population.query is None


def test_run_round_trips_json(minimal_run: InsightsRun) -> None:
    assert InsightsRun.model_validate_json(minimal_run.model_dump_json()) == minimal_run
