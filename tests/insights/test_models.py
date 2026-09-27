"""Unit tests for `evaluatorq.insights.models`."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from evaluatorq.insights.models import (
    InsightsConfig,
    InsightsPopulation,
    InsightsRun,
    LabelAnswer,
    TraceSummary,
    real_assistant_errors,
)
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


@pytest.mark.parametrize('probability', [0.0, 1.0, 0, 1])
def test_label_answer_accepts_probability_edges(probability: float | int) -> None:
    answer = LabelAnswer(value='positive', confidence=None, probabilities={'positive': probability}, error=None)

    assert answer.probabilities == {'positive': float(probability)}


@pytest.mark.parametrize('probability', [float('nan'), float('inf'), float('-inf'), -0.01, 1.01])
def test_label_answer_rejects_invalid_probabilities(probability: float) -> None:
    with pytest.raises(ValidationError):
        LabelAnswer(value='positive', confidence=None, probabilities={'positive': probability}, error=None)


def test_run_loads_config_fields_from_older_saved_file(minimal_run: InsightsRun) -> None:
    import json

    payload = json.loads(minimal_run.model_dump_json())
    payload['config'].update(
        min_cluster_size=5,
        merge_threshold=0.5,
        low_confidence_threshold=0.6,
        umap_random_state=42,
    )
    loaded = InsightsRun.model_validate_json(json.dumps(payload))
    assert loaded.config.classifier_model == minimal_run.config.classifier_model


@pytest.mark.parametrize('sentinel', ['None', 'none.', ' N/A ', 'No errors', 'No errors were made.'])
def test_real_assistant_errors_drops_sentinels(sentinel: str) -> None:
    summary = TraceSummary(
        summary='s',
        request=None,
        task=None,
        topic=None,
        sentiment_explanation=None,
        assistant_errors=[sentinel, 'Quoted the wrong refund window'],
    )
    assert real_assistant_errors(summary) == ['Quoted the wrong refund window']


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('max_clusters', 0),
        ('max_clusters', -1),
        ('max_subclusters', 0),
        ('max_subclusters', -1),
        ('parallelism', 0),
        ('parallelism', -1),
        ('outlier_zscore', -0.01),
    ],
)
def test_insights_config_rejects_invalid_numeric_limits(field: str, value: int | float) -> None:
    with pytest.raises(ValidationError):
        InsightsConfig.model_validate({'labels': [], 'dimensions': ['intent'], field: value})


def test_insights_config_accepts_minimum_numeric_limits() -> None:
    config = InsightsConfig(
        labels=[],
        dimensions=['intent'],
        max_clusters=1,
        max_subclusters=1,
        parallelism=1,
        outlier_zscore=0,
    )

    assert config.max_clusters == 1
    assert config.max_subclusters == 1
    assert config.parallelism == 1
    assert config.outlier_zscore == 0
