"""Map, crosstab, priority and finder handoff contracts for Insights."""

# Assertions are the test framework's reporting contract.
# ruff: noqa: S101

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from starlette.testclient import TestClient

from evaluatorq.common.reports.palette import COLORS, ORQ_SCALE_GOOD_BAD, ORQ_SCALE_HEAT, QUALITATIVE
from evaluatorq.dashboard import finder_views, insights_views
from evaluatorq.dashboard.app import build_app
from evaluatorq.insights.models import (
    Cluster,
    ClusterAssignment,
    DimensionResult,
    InsightsConfig,
    InsightsRun,
    LabelAnswer,
    LabelResult,
    LabelSpec,
    PriorityPoint,
    TraceInsight,
)


def _map_run() -> InsightsRun:
    label = LabelSpec(
        name='customer_satisfaction', kind='score', instructions='Rate satisfaction.', criteria=['low', 'mid', 'high']
    )
    sentiment = LabelSpec(
        name='sentiment',
        kind='choice',
        instructions='Classify sentiment.',
        criteria={'negative': None, 'positive': None},
    )
    clusters: list[Cluster] = [
        Cluster(
            id='top-1',
            parent_id=None,
            level='top',
            name='All requests',
            description='',
            size=10,
            trace_ids=[f'trace-{index}' for index in range(10)],
            example_trace_ids=['trace-1'],
        )
    ]
    traces: list[TraceInsight] = []
    for index in range(9):
        trace_id = f'trace-{index + 1}'
        cluster_id = f'base-{index + 1}'
        clusters.append(
            Cluster(
                id=cluster_id,
                parent_id='top-1',
                level='base',
                name=f'Cluster {index + 1}',
                description='',
                size=1,
                trace_ids=[trace_id],
                example_trace_ids=[trace_id],
            )
        )
        traces.append(
            TraceInsight(
                trace_id=trace_id,
                span_id=f'span-{index + 1}',
                timestamp=datetime(2026, 9, 1, tzinfo=timezone.utc),
                labels={
                    'customer_satisfaction': LabelAnswer(
                        value=index / 8, confidence=0.9, probabilities=None, error=None
                    ),
                    'sentiment': LabelAnswer(
                        value='positive' if index % 2 else 'negative', confidence=0.9, probabilities=None, error=None
                    ),
                },
                assignments={'intent': ClusterAssignment(top='top-1', base=cluster_id)},
                coords={'intent': (float(index), float(index + 1), float(index + 2))},
            )
        )
    traces.append(
        TraceInsight(
            trace_id='trace-10',
            span_id='span-10',
            timestamp=datetime(2026, 9, 1, tzinfo=timezone.utc),
            labels={
                'customer_satisfaction': LabelAnswer(value=0.75, confidence=0.8, probabilities=None, error=None),
                'sentiment': LabelAnswer(value='positive', confidence=0.8, probabilities=None, error=None),
            },
            assignments={'intent': ClusterAssignment(top='top-1', base='noise')},
            coords={'intent': (9.0, 10.0, 11.0)},
        )
    )
    spec = LabelSpec(
        name='customer_satisfaction', kind='score', instructions='Rate satisfaction.', criteria=['low', 'mid', 'high']
    )
    return InsightsRun(
        run_id='map-run',
        run_name='Map run',
        created_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        status='completed',
        stage_failures=[],
        population={},
        config=InsightsConfig(labels=[label, sentiment], dimensions=['intent']),
        traces=traces,
        dimensions={'intent': DimensionResult(name='intent', source_field='request', clusters=clusters)},
        labels={
            'customer_satisfaction': LabelResult(
                spec=spec, counts={}, mean_confidence=0.9, n_low_confidence=0, n_failed=0
            ),
            'sentiment': LabelResult(spec=sentiment, counts={}, mean_confidence=0.9, n_low_confidence=0, n_failed=0),
        },
        priority=None,
        priority_reason='Satisfaction data is unavailable.',
        counts={'n_traces': 10, 'n_failed_traces': 0},
        warnings=[],
    )


def _write_run(tmp_path: Path, run: InsightsRun) -> None:
    directory = tmp_path / 'insights-runs'
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'insights_map.json').write_text(run.model_dump_json(), encoding='utf-8')


def test_map_json_shape_palette_noise_diamond_and_numeric_label(tmp_path, monkeypatch):
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    run = _map_run()
    _write_run(tmp_path, run)

    response = TestClient(build_app()).get('/insights/map-run/map.json?dimension=intent')
    payload = response.json()
    points = {point['trace_id']: point for point in payload['points']}

    assert response.status_code == 200
    assert set(points['trace-1']) >= {'trace_id', 'x', 'y', 'z', 'cluster_id', 'cluster_name', 'color_value'}
    assert points['trace-1']['color'] == QUALITATIVE[0]
    assert points['trace-10']['color'] == COLORS['sand_400']
    assert points['trace-9']['symbol'] == 'diamond'
    numeric = (
        TestClient(build_app())
        .get('/insights/map-run/map.json?dimension=intent&color_by=label%3Acustomer_satisfaction')
        .json()
    )
    numeric_points = {point['trace_id']: point for point in numeric['points']}
    assert numeric_points['trace-10']['color_value'] == 0.25
    assert numeric['color_scale'] == ORQ_SCALE_GOOD_BAD
    choice = (
        TestClient(build_app()).get('/insights/map-run/map.json?dimension=intent&color_by=label%3Asentiment').json()
    )
    assert isinstance(choice['points'][0]['color_value'], int)
    assert choice['points'][0]['color'] == QUALITATIVE[0]


def test_crosstab_spec_has_both_axes_and_links_cells_to_filtered_traces():
    markup = insights_views.crosstab(_map_run(), {'row': 'dimension:intent', 'column': 'label:customer_satisfaction'})
    spec = json.loads(markup.split('data-vega-for="insights-crosstab">', 1)[1].split('</script>', 1)[0])
    axes = spec['layer'][0]['encoding']

    assert axes['x']['title'] == 'customer_satisfaction'
    assert axes['y']['title'] == 'Intent cluster'
    assert axes['color']['scale']['range'] == [color for _, color in ORQ_SCALE_HEAT]
    assert '/tab/traces?row=dimension%3Aintent' in str(spec['data']['values'][0]['href'])
    traces_html = insights_views.traces(
        _map_run(),
        row='dimension:intent',
        row_value='All requests',
        column='label:customer_satisfaction',
        column_value='0.0',
    )
    assert 'trace-1' in traces_html
    assert 'trace-2' not in traces_html


def test_priority_chart_contains_quadrant_labels_and_empty_state():
    run = _map_run().model_copy(
        update={
            'priority': [
                PriorityPoint(cluster_id='base-1', name='Cluster 1', volume=3, mean_satisfaction=0.4, error_share=0.5)
            ]
        }
    )
    html = insights_views.priority_matrix(run)
    spec = json.loads(html.split('data-vega-for="insights-priority">', 1)[1].split('</script>', 1)[0])
    assert {row['text'] for row in spec['layer'][3]['data']['values']} == {'fix first', 'watch', 'fine', 'niche'}
    assert 'Satisfaction data is unavailable.' in insights_views.priority_matrix(
        run.model_copy(update={'priority': None})
    )


def test_completed_finder_run_shows_analyze_matches_command_and_python(monkeypatch):
    monkeypatch.setattr(finder_views, 'status_indicator', lambda _snapshot: '')
    monkeypatch.setattr(finder_views, 'controls', lambda *_args, **_kwargs: '')
    monkeypatch.setattr(finder_views, 'field', lambda *_args, **_kwargs: '')
    monkeypatch.setattr(finder_views, 'table', lambda *_args, **_kwargs: '')
    monkeypatch.setattr(finder_views, 'task_panel', lambda *_args, **_kwargs: '')
    monkeypatch.setattr(finder_views, 'filter_output_panel', lambda *_args, **_kwargs: '')
    html = finder_views.body(SimpleNamespace(state='completed', compiled=object(), generation=17), object())

    assert 'Analyze matches' in html
    assert 'eq insights --from-finder trace-finder-17.json' in html
    assert 'InsightsPopulation.from_finder_export' in html
    assert 'insights_sync(population)' in html


def test_plotly_asset_exists_for_wheel_packaging():
    asset = Path(__file__).parents[2] / 'src/evaluatorq/dashboard/static/plotly-gl3d.min.js'
    assert asset.is_file()
    assert asset.read_text(encoding='utf-8').startswith('/**\n* plotly.js (gl3d - minified) v4.1.1')


def test_dimensions_has_tree_map_toggle_and_missing_coordinate_state():
    run = _map_run()
    markup = insights_views.dimensions(run)
    assert 'data-insights-view="tree"' in markup
    assert 'data-insights-view="map"' in markup
    assert 'data-map-url=' in markup
    missing_coords = run.model_copy(
        update={
            'traces': [trace.model_copy(update={'coords': {}}) for trace in run.traces],
            'dimensions': {
                'intent': run.dimensions['intent'].model_copy(update={'warnings': ['UMAP skipped for this dimension']})
            },
        }
    )
    assert 'Map unavailable: UMAP skipped for this dimension' in insights_views.dimensions(missing_coords)
    script = Path(__file__).parents[2] / 'src/evaluatorq/dashboard/static/dashboard.js'
    assert "axis('UMAP 1')" in script.read_text(encoding='utf-8')
    assert "aspectmode: 'cube'" in script.read_text(encoding='utf-8')
