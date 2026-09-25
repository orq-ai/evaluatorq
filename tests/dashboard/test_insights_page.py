"""Dashboard rendering and filtering for persisted Insights runs."""

# Assertions are the test framework's reporting contract.
# ruff: noqa: S101

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import pytest
from starlette.testclient import TestClient

from evaluatorq.dashboard.app import build_app
from evaluatorq.insights.models import (
    Cluster,
    ClusterAssignment,
    DimensionResult,
    InsightsConfig,
    InsightsRun,
    LabelAnswer,
    LabelResult,
    StageFailure,
    TraceInsight,
    TraceSummary,
)
from evaluatorq.insights.presets import SENTIMENT


@pytest.fixture
def minimal_run() -> InsightsRun:
    cluster = Cluster(
        id='base-1',
        parent_id='top-1',
        level='base',
        name='General requests',
        description='Traces about general requests.',
        size=2,
        trace_ids=['trace-1', 'trace-2'],
        example_trace_ids=['trace-1'],
    )
    top = cluster.model_copy(update={'id': 'top-1', 'parent_id': None, 'level': 'top'})
    traces = [
        TraceInsight(
            trace_id=f'trace-{index}',
            span_id=f'span-{index}',
            timestamp=datetime(2026, 9, index, tzinfo=timezone.utc),
            labels={
                'sentiment': LabelAnswer(value='positive', confidence=0.9, probabilities={'positive': 0.9}, error=None)
            },
            summary=TraceSummary(
                summary='A user asked a question.',
                request='What is the refund policy?',
                task='answer',
                topic='refunds',
                sentiment_explanation='Calm.',
            ),
            assignments={'intent': ClusterAssignment(top='top-1', base='base-1')},
        )
        for index in (1, 2)
    ]
    return InsightsRun(
        run_id='run-1',
        run_name='minimal run',
        created_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        status='completed',
        stage_failures=[],
        population={
            'mode': 'query',
            'query': 'refund policy',
            'facets': {'agent_name': ['support-bot'], 'project': ['default']},
            'numeric': {'tokens_min': 100, 'tokens_max': None, 'duration_ms_min': None, 'duration_ms_max': 2500},
            'start': '2026-08-25T00:00:00+00:00',
            'end': '2026-09-01T00:00:00+00:00',
            'limit': 500,
            'n_scanned': 2,
            'n_matched': 2,
            'n_failed_match': 0,
        },
        config=InsightsConfig(labels=[SENTIMENT], dimensions=['intent']),
        traces=traces,
        dimensions={'intent': DimensionResult(name='intent', source_field='request', clusters=[top, cluster])},
        labels={
            'sentiment': LabelResult(
                spec=SENTIMENT, counts={'positive': 2}, mean_confidence=0.9, n_low_confidence=0, n_failed=0
            )
        },
        priority=None,
        priority_reason='not available',
        counts={'n_traces': 2, 'n_failed_traces': 0},
        warnings=[],
    )


def _write_run(base, run: InsightsRun, name: str = 'insights_fixture.json'):
    directory = base / 'insights-runs'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(run.model_dump_json(), encoding='utf-8')
    return path


def test_insights_page_renders_list_and_population_and_label_chips(tmp_path, minimal_run, monkeypatch):
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    _write_run(tmp_path, minimal_run)
    client = TestClient(build_app())

    response = client.get('/insights')

    assert response.status_code == 200
    assert 'minimal run' in response.text
    assert 'Population' in response.text
    assert 'refund policy' in response.text
    assert 'support-bot' in response.text
    assert 'tokens ≥' in response.text
    assert '2026-08-25T00:00:00+00:00' in response.text
    assert '2026-09-01T00:00:00+00:00' in response.text
    assert '500' in response.text
    assert 'sentiment' in response.text
    assert 'General requests' in response.text
    assert 'Insights' in response.text


def test_error_run_shows_failure_stage_and_failed_trace_note(tmp_path, minimal_run, monkeypatch):
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    failed = minimal_run.model_copy(
        update={
            'status': 'error',
            'stage_failures': [StageFailure(stage='dimension:failure', message='embedding service unavailable')],
            'counts': {'n_traces': 2, 'n_failed_traces': 1},
        }
    )
    _write_run(tmp_path, failed)

    response = TestClient(build_app()).get('/insights/run-1')

    assert response.status_code == 200
    assert 'Run failed; results may be partial.' in response.text
    assert 'dimension:failure' in response.text
    assert 'embedding service unavailable' in response.text
    assert '1 traces could not be fully classified or summarized.' in response.text


def test_cluster_panel_has_description_and_example_trace_link(tmp_path, minimal_run, monkeypatch):
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    monkeypatch.setenv('ORQ_WORKSPACE', 'example-workspace')
    _write_run(tmp_path, minimal_run)

    response = TestClient(build_app()).get('/insights/run-1/cluster/base-1')

    assert response.status_code == 200
    assert 'Traces about general requests.' in response.text
    assert 'trace-1' in response.text
    assert 'example-workspace/traces?query=' in response.text


def test_traces_can_be_filtered_by_cluster(tmp_path, minimal_run, monkeypatch):
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    only_first = minimal_run.model_copy(
        update={
            'traces': [minimal_run.traces[0]],
            'dimensions': {
                'intent': minimal_run.dimensions['intent'].model_copy(
                    update={
                        'clusters': [
                            cluster.model_copy(update={'trace_ids': ['trace-1'], 'size': 1})
                            for cluster in minimal_run.dimensions['intent'].clusters
                        ]
                    }
                )
            },
        }
    )
    _write_run(tmp_path, only_first)

    response = TestClient(build_app()).get('/insights/run-1/traces?dimension=intent&cluster=base-1')

    assert response.status_code == 200
    assert 'trace-1' in response.text
    assert 'trace-2' not in response.text


def test_labels_without_any_labels_show_empty_state(tmp_path, minimal_run, monkeypatch):
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    no_labels = minimal_run.model_copy(
        update={'labels': {}, 'config': minimal_run.config.model_copy(update={'labels': []})}
    )
    _write_run(tmp_path, no_labels)

    response = TestClient(build_app()).get('/insights/run-1/tab/labels')

    assert response.status_code == 200
    assert 'No labels in this run' in response.text


def test_traces_show_active_filter_chips_and_clear_filter_navigation(tmp_path, minimal_run, monkeypatch):
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    _write_run(tmp_path, minimal_run)

    response = TestClient(build_app()).get(
        '/insights/run-1/tab/traces?dimension=intent&cluster=base-1&label=sentiment&value=positive'
    )

    assert response.status_code == 200
    assert 'intent = General requests' in response.text
    assert 'sentiment = positive' in response.text
    assert 'Clear filters' in response.text
    assert 'href="/insights/run-1/tab/traces"' in response.text
    assert 'trace-1' in response.text


def test_tab_route_serves_full_page_on_navigation_and_fragment_for_htmx(tmp_path, minimal_run, monkeypatch):
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    _write_run(tmp_path, minimal_run)
    client = TestClient(build_app())

    page_response = client.get('/insights/run-1/tab/labels')
    fragment_response = client.get('/insights/run-1/tab/labels', headers={'HX-Request': 'true'})

    assert page_response.status_code == 200
    assert page_response.text.startswith('<!DOCTYPE html>')
    assert 'class="insights-tab active" href="/insights/run-1/tab/labels"' in page_response.text
    assert 'insights-label-grid' in page_response.text
    assert fragment_response.status_code == 200
    assert not fragment_response.text.startswith('<!DOCTYPE html>')
    assert 'insights-label-grid' in fragment_response.text


def test_repeated_pages_use_the_validated_model_cache(tmp_path, minimal_run, monkeypatch):
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    _write_run(tmp_path, minimal_run)
    original = InsightsRun.model_validate
    validations = 0

    def counted(_cls: type[InsightsRun], data: object, **kwargs: Any) -> InsightsRun:
        nonlocal validations
        validations += 1
        return original(data, **kwargs)

    monkeypatch.setattr(InsightsRun, 'model_validate', classmethod(counted))
    client = TestClient(build_app())

    assert client.get('/insights').status_code == 200
    assert client.get('/insights/run-1').status_code == 200
    assert validations == 1


def test_truncated_run_stays_visible_and_insights_page_renders(tmp_path, monkeypatch):
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    directory = tmp_path / 'insights-runs'
    directory.mkdir(parents=True)
    (directory / 'insights_broken.json').write_text('{"schema_version":', encoding='utf-8')
    unknown = {
        'schema_version': 2,
        'run_id': 'unknown-version',
        'run_name': 'unknown version',
    }
    (directory / 'insights_unknown.json').write_text(json.dumps(unknown), encoding='utf-8')

    response = TestClient(build_app()).get('/insights')

    assert response.status_code == 200
    assert 'insights_broken' in response.text
    assert 'insights_unknown' in response.text
    assert 'unreadable' in response.text


def test_run_rail_keeps_newest_first_order(tmp_path, minimal_run, monkeypatch):
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    older = _write_run(tmp_path, minimal_run, 'insights_older.json')
    newer_run = minimal_run.model_copy(update={'run_id': 'run-2', 'run_name': 'newer run'})
    newer = _write_run(tmp_path, newer_run, 'insights_newer.json')
    os.utime(older, ns=(1_000_000_000, 1_000_000_000))
    os.utime(newer, ns=(2_000_000_000, 2_000_000_000))

    response = TestClient(build_app()).get('/insights')

    assert response.status_code == 200
    assert response.text.index('newer run') < response.text.index('minimal run')
