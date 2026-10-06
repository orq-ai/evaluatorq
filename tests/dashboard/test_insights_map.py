"""Map, crosstab, priority and finder handoff contracts for Insights."""

# Assertions are the test framework's reporting contract.
# ruff: noqa: S101

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest
from starlette.testclient import TestClient

from evaluatorq.common.reports.palette import COLORS, ORQ_SCALE_GOOD_BAD, ORQ_SCALE_HEAT, QUALITATIVE
from evaluatorq.dashboard import insights_views
from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.trace_finder import views as finder_views
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
            trace_ids=[f'trace-{index}' for index in range(1, 11)],
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
    assert payload['color_mode'] == 'cluster'
    assert {item['cluster_id'] for item in payload['legend']} >= {'base-1', 'base-9', 'noise'}
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
    assert {point['trace_id']: point for point in choice['points']}['trace-9']['symbol'] == 'diamond'


def test_review_projection_payload_uses_full_trace_keys_and_keeps_3d_endpoint(tmp_path, monkeypatch):
    from evaluatorq.dashboard import insights_review_data

    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    monkeypatch.setattr(
        insights_review_data,
        'review_xy',
        lambda saved, _dimension: {
            f'{trace.trace_id}:{trace.span_id}': (float(index), float(index + 1))
            for index, trace in enumerate(saved.traces, start=1)
        },
    )
    run = _map_run()
    run.traces.append(run.traces[0].model_copy(update={'span_id': 'span-1-second'}))
    payload = insights_review_data.build_review_payload(run)

    map_states = cast('dict[str, object]', payload['map_states'])
    traces = cast('list[dict[str, object]]', payload['traces'])
    assert map_states['intent'] == {'available': True, 'reason': None, 'n_points': 11}
    assert traces[0]['id'] == 'trace-1:span-1'
    assert cast('dict[str, object]', traces[0]['xy'])['intent'] == [1.0, 2.0]
    assert traces[-1]['id'] == 'trace-1:span-1-second'
    assert cast('dict[str, object]', traces[-1]['xy'])['intent'] == [11.0, 12.0]

    _write_run(tmp_path, run)
    legacy = TestClient(build_app()).get('/insights/map-run/map.json?dimension=intent').json()
    assert {'x', 'y', 'z'} <= set(legacy['points'][0])


def test_review_projection_payload_matches_keys_for_dataset_rows(monkeypatch):
    from evaluatorq.dashboard import insights_review_data

    class _Array:
        def __init__(self, values: Any) -> None:
            self.values = values

        @property
        def shape(self) -> tuple[int, int]:
            return (len(self.values), len(self.values[0]) if self.values else 0)

        def __iter__(self):
            return iter(self.values)

        def __len__(self) -> int:
            return len(self.values)

    numpy = ModuleType('numpy')

    def _asarray(values: Any, **_kwargs: Any) -> _Array:
        return _Array(values)

    def _isfinite(_values: _Array) -> SimpleNamespace:
        return SimpleNamespace(all=lambda: True)

    setattr(numpy, 'asarray', _asarray)
    setattr(numpy, 'isfinite', _isfinite)

    class _Umap:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def fit_transform(self, rows: _Array) -> list[tuple[float, float]]:
            return [(float(index), float(index + 1)) for index in range(len(rows))]

    umap = ModuleType('umap')
    setattr(umap, 'UMAP', _Umap)
    monkeypatch.setitem(sys.modules, 'numpy', numpy)
    monkeypatch.setitem(sys.modules, 'umap', umap)

    run = _map_run()
    run.run_id = 'dataset-map-run'
    for trace in run.traces:
        trace.span_id = None

    payload = insights_review_data.build_review_payload(run)
    traces = cast('list[dict[str, object]]', payload['traces'])

    assert len(traces) == 10
    assert all(cast('dict[str, object]', trace['xy'])['intent'] for trace in traces)
    assert [cast('dict[str, list[float]]', trace['xy'])['intent'] for trace in traces] == [
        [float(index), float(index + 1)] for index in range(10)
    ]


def test_review_url_back_navigation_resets_omitted_map_and_compare_options():
    if shutil.which('node') is None:
        pytest.skip('Node.js is unavailable')
    script = Path(__file__).with_name('insights_review_map.cjs')
    result = subprocess.run(['node', str(script)], capture_output=True, text=True, check=True, cwd=Path(__file__).parents[2])
    assert 'URL state restoration checks passed' in result.stdout
    assert 'Map Compare folded cell filters and URL restoration checks passed' in result.stdout


def test_map_tab_switches_projections_and_colours_by_other_dimensions(tmp_path, monkeypatch):
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    run = _map_run()
    run.dimensions['failure'] = DimensionResult(
        name='failure',
        source_field='assistant_errors',
        clusters=[
            Cluster(
                id='failure-b0', parent_id=None, level='base', name='Tool failures', description='',
                size=5, trace_ids=[f'trace-{index}' for index in range(1, 6)], example_trace_ids=['trace-1'],
            ),
            Cluster(
                id='failure-b1', parent_id=None, level='base', name='Other failures', description='',
                size=5, trace_ids=[f'trace-{index}' for index in range(6, 11)], example_trace_ids=['trace-6'],
            ),
        ],
    )
    for index, trace in enumerate(run.traces):
        trace.assignments['failure'] = ClusterAssignment(
            top='failure-t0', base='failure-b0' if index < 5 else 'failure-b1'
        )
        trace.coords['failure'] = (float(index + 20), float(index + 21), float(index + 22))
        trace.agent_name = 'support-bot' if index < 5 else 'billing-bot'
    _write_run(tmp_path, run)
    client = TestClient(build_app())

    page = client.get('/insights/map-run/tab/map', headers={'HX-Request': 'true'})
    assert page.status_code == 200
    assert 'data-map-fullscreen' in page.text
    assert 'data-map-projection' in page.text
    assert 'dimension:failure' in page.text
    assert 'label:sentiment' in page.text
    assert 'value="agent"' in page.text

    recoloured = client.get('/insights/map-run/map.json?dimension=intent&color_by=dimension%3Afailure').json()
    points = {point['trace_id']: point for point in recoloured['points']}
    assert len(points) == 10
    assert points['trace-1']['x'] == 0.0
    assert points['trace-1']['cluster_id'] == 'failure-b0'
    assert points['trace-1']['cluster_name'] == 'Tool failures'
    assert points['trace-9']['symbol'] == 'diamond'
    assert recoloured['color_mode'] == 'cluster'
    agents = client.get('/insights/map-run/map.json?dimension=failure&color_by=agent').json()
    assert agents['color_mode'] == 'category'
    assert {point['label_value'] for point in agents['points']} == {'support-bot', 'billing-bot'}
    assert {point['trace_id']: point for point in agents['points']}['trace-1']['x'] == 20.0


def test_insights_map_uses_server_cluster_url_template(tmp_path, monkeypatch):
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    _write_run(tmp_path, _map_run())
    client = TestClient(build_app())

    page = client.get('/insights/map-run/tab/map', headers={'HX-Request': 'true'})
    assert 'data-cluster-detail-url-template="/insights/map-run/cluster/{cluster_id}"' in page.text


def test_insights_run_path_stem_alias_cannot_shadow_another_run_id(tmp_path, monkeypatch):
    from evaluatorq.dashboard import insights_routes

    canonical = _map_run()
    canonical.run_id = 'collision'
    alias_target = _map_run()
    alias_target.run_id = 'other'
    canonical_path = tmp_path / 'canonical.json'
    alias_path = tmp_path / 'collision.json'
    runs = {canonical_path: canonical, alias_path: alias_target}
    monkeypatch.setattr(insights_routes, 'list_run_paths', lambda _directory: [canonical_path, alias_path])
    monkeypatch.setattr(insights_routes.library, 'load_model_cached', lambda path, _validator: runs[path])
    monkeypatch.setattr(insights_routes, 'list_manifests', lambda _directory: [])

    entries, loaded, _ = insights_routes._entries(tmp_path)

    assert [entry[0] for entry in entries] == ['collision', 'other']
    assert loaded['collision'] == (canonical_path, canonical)


def test_numeric_map_colouring_keeps_unlabelled_points(tmp_path, monkeypatch):
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    run = _map_run()
    run.traces[0].labels.pop('customer_satisfaction')
    _write_run(tmp_path, run)

    payload = TestClient(build_app()).get(
        '/insights/map-run/map.json?dimension=intent&color_by=label%3Acustomer_satisfaction'
    ).json()

    assert len(payload['points']) == 9
    assert [point['trace_id'] for point in payload['missing_points']] == ['trace-1']
    assert payload['missing_points'][0]['span_id'] == 'span-1'


def test_map_json_reports_unknown_or_unsupported_colour_label(tmp_path, monkeypatch):
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    run = _map_run()
    unsupported = LabelSpec(name='intent_text', kind='noul', instructions='Summarize intent.')
    run = run.model_copy(
        update={
            'labels': {
                **run.labels,
                'intent_text': LabelResult(spec=unsupported, counts={}, mean_confidence=None, n_low_confidence=0, n_failed=0),
            }
        }
    )
    _write_run(tmp_path, run)

    client = TestClient(build_app())
    response = client.get('/insights/map-run/map.json?dimension=intent&color_by=label%3Amissing')
    unsupported_response = client.get('/insights/map-run/map.json?dimension=intent&color_by=label%3Aintent_text')

    assert response.status_code == 200
    assert response.json()['error'] == 'Unknown or unsupported colour label: missing'
    assert unsupported_response.json()['color_mode'] == 'category'
    assert {point['label_value'] for point in unsupported_response.json()['points']} == {'No value'}


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
        column_value='0',
    )
    assert 'trace-1' in traces_html
    assert 'trace-4' not in traces_html  # 0.375 is the 'mid' level; 0.0 and 0.125 both read as 'low'


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


def test_priority_chart_keeps_positive_x_domain_when_every_point_has_zero_volume():
    run = _map_run().model_copy(
        update={
            'priority': [
                PriorityPoint(cluster_id='base-1', name='Cluster 1', volume=0, mean_satisfaction=0.4, error_share=0.5)
            ]
        }
    )
    html = insights_views.priority_matrix(run)

    assert '"domain": [0, 1.0]' in html


def test_completed_finder_run_shows_runnable_analyze_matches_examples(monkeypatch, tmp_path):
    seen_snapshots = []

    def filename(snapshot):
        seen_snapshots.append(snapshot)
        return 'trace-finder-17.json'

    monkeypatch.setattr(finder_views, 'export_filename', filename)
    monkeypatch.setattr(finder_views, 'status_indicator', lambda _snapshot: '')
    monkeypatch.setattr(finder_views, 'controls', lambda *_args, **_kwargs: '')
    monkeypatch.setattr(finder_views, 'field', lambda *_args, **_kwargs: '')
    monkeypatch.setattr(finder_views, 'table', lambda *_args, **_kwargs: '')
    monkeypatch.setattr(finder_views, 'task_panel', lambda *_args, **_kwargs: '')
    monkeypatch.setattr(finder_views, 'filter_output_panel', lambda *_args, **_kwargs: '')
    monkeypatch.setattr(finder_views, 'run_status', lambda *_args, **_kwargs: '')
    snapshot = SimpleNamespace(state='completed', dimensions=(object(),), generation=17)
    html = finder_views.body(snapshot, object())

    assert 'Analyze matches' in html
    assert 'Download the completed export to use it with the CLI or Python.' in html
    assert 'Download and save trace-finder-17.json' in html
    rendered_examples = unescape(html)
    assert 'server-side copy is also saved' in html
    assert 'eq insights --from-finder trace-finder-17.json' in rendered_examples
    assert 'CLI (Windows PowerShell)' in html
    assert "eq insights --from-finder 'trace-finder-17.json'" in rendered_examples
    assert 'InsightsPopulation.from_finder_export' in html
    assert "Path('trace-finder-17.json')" in rendered_examples
    assert 'insights_sync(population)' in rendered_examples
    assert seen_snapshots == [snapshot]
    assert '/Users/' not in html
    assert '/tmp/' not in html


def test_plotly_asset_exists_in_source_tree():
    asset = Path(__file__).parents[2] / 'src/evaluatorq/dashboard/static/plotly-gl3d.min.js'
    assert asset.is_file()
    assert asset.read_text(encoding='utf-8').startswith('/**\n* plotly.js (gl3d - minified) v4.1.1')


def test_htmx_swap_initializes_maps_without_vega_embed():
    script = Path(__file__).parents[2] / 'src/evaluatorq/dashboard/static/dashboard.js'
    source = script.read_text(encoding='utf-8')
    handler = source.split("document.body.addEventListener('htmx:afterSwap', function (evt) {", 1)[1].split('\n  });', 1)[0]

    assert handler.index('if (!scope) return;') < handler.index('initInsightsMaps(scope);')
    assert handler.index('initInsightsMaps(scope);') < handler.index("if (!scope.querySelector('.vega-chart')) return;")
    assert handler.count('initInsightsMaps(scope);') == 1


def test_category_map_keeps_delimiter_values_distinct():
    if shutil.which('node') is None:
        pytest.skip('Node.js is unavailable')
    script = Path(__file__).parents[2] / 'src/evaluatorq/dashboard/static/dashboard.js'
    source = script.read_text(encoding='utf-8')
    start = source.index('  function mapTraces(payload) {')
    end = source.index('  function showInsightsMapError', start)
    payload = {
        'color_mode': 'category',
        'points': [
            {'label_value': 'a|b', 'symbol': 'circle', 'trace_id': 'first', 'color': '#111', 'x': 1},
            {'label_value': 'a', 'symbol': 'b|circle', 'trace_id': 'second', 'color': '#222', 'x': 2},
            {'label_value': 'a|b', 'symbol': 'circle', 'trace_id': 'third', 'color': '#111', 'x': 3},
            {'label_value': 'toString', 'symbol': 'circle', 'trace_id': 'fourth', 'color': '#333', 'x': 4},
        ],
    }
    javascript = source[start:end] + f'\nconsole.log(JSON.stringify(mapTraces({json.dumps(payload)})));'
    result = subprocess.run(['node', '-e', javascript], capture_output=True, text=True, check=True)
    traces = json.loads(result.stdout)

    assert [(trace['name'], trace['x']) for trace in traces] == [
        ('a|b', [1, 3]),
        ('a', [2]),
        ('toString', [4]),
    ]
    assert all(trace['showlegend'] for trace in traces)


def test_map_fetch_finishing_after_htmx_removal_does_not_render_detached_chart():
    if shutil.which('node') is None:
        pytest.skip('Node.js is unavailable')
    script = Path(__file__).parents[2] / 'src/evaluatorq/dashboard/static/dashboard.js'
    source = script.read_text(encoding='utf-8')
    start = source.index('  function drawInsightsMap(el) {')
    end = source.index('  function initInsightsMaps', start)
    javascript = source[start:end] + r'''
let releaseFetch;
let connected = true;
let renders = 0;
const el = {
  get isConnected() { return connected; },
  offsetParent: {},
  parentElement: { querySelector: function () { return null; } },
  style: {},
  __insightsMapRequestId: 0,
  closest: function () { return { querySelector: function () { return { value: 'cluster' }; } }; },
  getAttribute: function (name) { return name === 'data-map-url' ? '/map?color_by=cluster' : ''; },
};
global.window = { Plotly: { react: function () { renders += 1; } } };
global.fetch = function () { return new Promise(function (resolve) { releaseFetch = resolve; }); };
drawInsightsMap(el);
connected = false;
el.parentElement = null;
releaseFetch({ ok: true, json: function () { return Promise.resolve({ points: [], missing_points: [] }); } });
setTimeout(function () { console.log(JSON.stringify({ renders: renders })); }, 0);
'''
    result = subprocess.run(['node', '-e', javascript], capture_output=True, text=True, check=True)

    assert json.loads(result.stdout) == {'renders': 0}


def test_dashboard_script_has_no_orphaned_tree_map_toggle_handler():
    source = (Path(__file__).parents[2] / 'src/evaluatorq/dashboard/static/dashboard.js').read_text(encoding='utf-8')

    assert 'data-insights-view' not in source
    assert "closest('.insights-dimension')" not in source


def test_cluster_detail_returns_content_for_existing_detail_panel():
    markup = insights_views.cluster_detail(_map_run(), 'base-1')

    assert markup.startswith('<div class="insights-detail-kicker">Cluster · intent</div>')
    assert '<section class="insights-detail">' not in markup


def test_dimensions_has_no_map_and_map_tab_shows_missing_coordinate_state():
    run = _map_run()
    markup = insights_views.dimensions(run)
    assert 'data-insights-view' not in markup
    assert 'data-map-url=' not in markup
    assert 'data-map-url=' in insights_views.map_tab(run)
    missing_coords = run.model_copy(
        update={
            'traces': [trace.model_copy(update={'coords': {}}) for trace in run.traces],
            'dimensions': {
                'intent': run.dimensions['intent'].model_copy(update={'warnings': ['UMAP skipped for this dimension']})
            },
        }
    )
    assert 'Map unavailable: UMAP skipped for this dimension' in insights_views.map_tab(missing_coords)
    script = Path(__file__).parents[2] / 'src/evaluatorq/dashboard/static/dashboard.js'
    assert "axis('UMAP 1')" in script.read_text(encoding='utf-8')
    assert "aspectmode: 'cube'" in script.read_text(encoding='utf-8')


def test_label_card_rows_filter_to_their_traces_for_yes_no_and_score_labels():
    from evaluatorq.insights.models import LabelAnswer, label_key
    from evaluatorq.insights.presets import RISKY_ACTION, USER_FRUSTRATION

    assert label_key(RISKY_ACTION, 'deleted') == 'deleted'
    assert label_key(USER_FRUSTRATION, 0.0) == '1'
    assert label_key(USER_FRUSTRATION, 1.0) == '5'
    run = _map_run()
    trace = run.traces[0]
    trace.labels['risky_action'] = LabelAnswer(value='deleted', confidence=None, probabilities=None, error=None)
    run.labels['risky_action'] = run.labels['customer_satisfaction'].model_copy(update={'spec': RISKY_ACTION, 'counts': {'deleted': 1}})

    assert 'value=deleted' in insights_views.labels(run)
    assert trace.trace_id in insights_views.traces(run, label='risky_action', value='deleted')


def test_label_values_are_listed_in_the_labels_own_order():
    from evaluatorq.insights.models import label_order
    from evaluatorq.insights.presets import OUTCOME, RISKY_ACTION, SENTIMENT, USER_FRUSTRATION

    assert label_order(SENTIMENT) == ['positive', 'neutral', 'negative']
    assert label_order(USER_FRUSTRATION) == ['1', '2', '3', '4', '5']
    assert label_order(RISKY_ACTION)[0] == 'none'
    assert label_order(OUTCOME)[-1] == 'inconclusive'
    run = _map_run()
    run.labels['sentiment'] = run.labels['sentiment'].model_copy(
        update={'spec': SENTIMENT, 'counts': {'negative': 1, 'neutral': 1, 'positive': 1}}
    )

    card = insights_views.labels(run)

    assert card.index('value=positive') < card.index('value=neutral') < card.index('value=negative')
