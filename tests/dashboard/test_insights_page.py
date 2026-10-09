"""Dashboard rendering and filtering for persisted Insights runs."""

# Assertions are the test framework's reporting contract.
# ruff: noqa: S101

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pytest
from starlette.testclient import TestClient

from evaluatorq.common.run_manifest import start_manifest
from evaluatorq.contracts import ManifestStatus, ManifestSurface, RunManifest, StageRecord, Usage
from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.insights_views import TABS, cluster_detail, failures, header, labels, progress, trace_detail_page, traces
from evaluatorq.insights.models import (
    Cluster,
    ClusterAssignment,
    DimensionResult,
    InsightsConfig,
    InsightsRun,
    LabelAnswer,
    LabelResult,
    LabelSpec,
    StageFailure,
    TraceInsight,
    TraceSummary,
)
from evaluatorq.insights.presets import SENTIMENT
from evaluatorq.insights.usage import UsageLedger


def test_insights_header_renders_partial_cost(minimal_run) -> None:
    run = minimal_run.model_copy(
        update={
            'cost_by_stage': {
                'summary': Usage(input_tokens=10, output_tokens=2, total_tokens=12, total_cost=0.01, calls=2, priced_calls=1)
            }
        }
    )

    assert 'priced for 1 of 2 calls' in header(run)
    assert 'excludes trace selection' in header(run)


def test_insights_review_activity_synthetic_contract() -> None:
    if shutil.which('node') is None:
        pytest.skip('Node.js is unavailable')
    script = Path(__file__).with_name('insights_review_activity.cjs')
    subprocess.run(['node', str(script)], check=True, cwd=Path(__file__).parents[2])


def test_insights_review_dimensionless_run_and_safe_tooltips() -> None:
    if shutil.which('node') is None:
        pytest.skip('Node.js is unavailable')
    script = Path(__file__).with_name('insights_review_dimensionless.cjs')
    subprocess.run(['node', str(script)], check=True, cwd=Path(__file__).parents[2])


def test_insights_header_marks_priced_and_missing_usage_calls_partial(minimal_run) -> None:
    ledger = UsageLedger()
    ledger.add('label', Usage(input_tokens=10, output_tokens=2, total_tokens=12, total_cost=0.01, calls=1, priced_calls=1))
    ledger.add('label', None)
    run = minimal_run.model_copy(update={'cost_by_stage': ledger.totals()})

    assert 'priced for 1 of 2 calls' in header(run)


def test_insights_header_hides_untracked_cost(minimal_run) -> None:
    assert '$' not in header(minimal_run)


def test_local_snapshot_run_does_not_offer_orq_trace_links(minimal_run: InsightsRun) -> None:
    run = minimal_run.model_copy(update={'population': {'mode': 'snapshot', 'limit': 2}})

    assert 'trace file' in header(run)
    assert 'Open in Orq' not in cluster_detail(run, 'base-1')
    table = traces(run)
    assert '<th>Orq</th>' not in table
    assert '↗ Orq' not in table
    assert 'href="/insights/run-1/trace?trace_id=trace-1&amp;span_id=span-1"' in table
    detail = trace_detail_page(run, run.traces[0])
    assert 'A user asked a question.' in detail
    assert 'General requests' in detail
    assert 'Open full trace in Orq' not in detail


def test_run_page_shows_character_input_coverage_for_loaded_traces(minimal_run: InsightsRun) -> None:
    run = minimal_run.model_copy(update={'population': {
        'mode': 'filter',
        'n_traces': 2,
        'n_projection_truncated': 1,
        'n_source_messages': 10,
        'n_omitted_messages': 3,
        'source_chars': 120_000,
        'classifier_state_chars': 65_000,
        'summary_input_chars': 100_000,
        'classifier_state_format': 'jev',
        'classifier_question_reserve': 'applied',
        'trace_input_chars': 500_000,
    }})

    notice = failures(run)

    assert '1 of 2 traces hit a summary or classifier input cap of 500,000 characters.' in notice
    assert '3 of 10 source messages were omitted from the Jev classifier state.' in notice


def test_saved_run_uses_redesign_and_python_owned_actions(tmp_path, minimal_run, monkeypatch):
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    _write_run(tmp_path, minimal_run)
    client = TestClient(build_app())

    response = client.get('/insights/run-1')

    assert response.status_code == 200
    assert 'insights-review-root' in response.text
    assert 'href="/insights">Insights</a>' in response.text
    assert 'href="/insights/run-1/export.json">Export</a>' in response.text
    assert 'href="/insights/run-1?rerun=1">Re-run</a>' in response.text
    assert 'id="rerun"' in response.text
    assert 'data-rerun' not in client.get('/insights/run-1?rerun=1').text
    rerun = client.get('/insights/run-1/rerun', follow_redirects=False)
    assert rerun.status_code == 302
    assert rerun.headers['location'] == '/insights/run-1?rerun=1'


@pytest.mark.parametrize(
    ('tab', 'query', 'fragment'),
    [
        ('dimensions', '', '#dim=intent&view=themes'),
        ('labels', '', '#dim=intent&view=themes'),
        ('traces', 'dimension=intent&cluster=base-1', '#dim=intent&view=themes&f=c%3Abase-1'),
        ('priority', 'dimension=bogus&cluster=bad', '#dim=intent&view=themes'),
        ('map', 'dimension=intent', '#dim=intent&view=map'),
        ('crosstab', 'dimension=intent', '#dim=intent&view=compare'),
    ],
)
def test_legacy_tab_navigation_redirects_to_review(tmp_path, minimal_run, monkeypatch, tab, query, fragment):
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    _write_run(tmp_path, minimal_run)
    suffix = f'?{query}' if query else ''

    response = TestClient(build_app()).get(f'/insights/run-1/tab/{tab}{suffix}', follow_redirects=False)

    assert response.status_code == 302
    assert response.headers['location'] == f'/insights/run-1{fragment}'


def test_legacy_tab_redirect_keeps_valid_label_filter_and_discards_invalid_values(
    tmp_path, minimal_run, monkeypatch
):
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    _write_run(tmp_path, minimal_run)
    client = TestClient(build_app())

    valid = client.get(
        '/insights/run-1/tab/labels?dimension=intent&label=sentiment&value=positive', follow_redirects=False
    )
    invalid = client.get(
        '/insights/run-1/tab/labels?dimension=intent&label=sentiment&value=unknown', follow_redirects=False
    )

    assert valid.headers['location'] == '/insights/run-1#dim=intent&view=themes&f=l%3Asentiment%3Apositive'
    assert invalid.headers['location'] == '/insights/run-1#dim=intent&view=themes'


def test_launch_route_passes_selected_coding_and_custom_labels(monkeypatch):
    import re
    from unittest.mock import patch

    from evaluatorq.dashboard.auth import DashboardAuth
    from evaluatorq.insights.presets import CODING_LABELS

    client = TestClient(build_app())
    page = client.get('/insights/new')
    token = re.search(r'name="csrf" value="([^"]+)"', page.text)
    assert token is not None
    custom = {'name': 'custom_question', 'kind': 'noul', 'instructions': 'Did the agent answer?'}
    data = {
        'csrf': token.group(1),
        'source': 'orq',
        'dimensions': 'intent',
        'coding_labels': [CODING_LABELS[1].name],
        'custom_labels_json': json.dumps([custom]),
    }
    with (
        patch(
            'evaluatorq.dashboard.insights_routes.selected_dashboard_auth',
            return_value=DashboardAuth('environment', 'key', 'https://my.orq.ai'),
        ),
        patch('evaluatorq.dashboard.insights_routes.launch_insights', return_value='run-1') as launch,
    ):
        response = client.post('/insights/runs', data=data, follow_redirects=False)
        malformed = client.post('/insights/runs', data={**data, 'custom_labels_json': '{'}, follow_redirects=False)

    assert response.status_code == 303
    spec = launch.call_args.args[0]
    assert spec.coding_labels == [CODING_LABELS[1].name]
    assert [label.name for label in spec.custom_labels] == ['custom_question']
    assert malformed.status_code == 422
    assert 'role="alert"' in malformed.text


def test_projection_warning_is_not_repeated_below_coverage(minimal_run: InsightsRun) -> None:
    run = minimal_run.model_copy(update={
        'population': {
            'n_traces': 1,
            'n_projection_truncated': 1,
            'n_source_messages': 2,
            'n_omitted_messages': 1,
            'source_chars': 60_000,
            'classifier_state_chars': 30_000,
            'summary_input_chars': 30_000,
            'classifier_state_format': 'jev',
            'classifier_question_reserve': 'applied',
            'trace_input_chars': 500_000,
        },
        'warnings': ['Projection budget: 1 of 1 traces exceeded the model projection budget; 1 of 2 whole messages were omitted.'],
    })

    notice = failures(run)

    assert notice.count('Model input coverage') == 1
    assert 'Run warnings' not in notice

    earlier_run = run.model_copy(update={
        'warnings': [
            '1 of 1 traces exceeded the model projection budget; '
            '1 of 2 whole messages were omitted from classification and summaries.'
        ]
    })
    assert 'Run warnings' not in failures(earlier_run)


def test_earlier_run_does_not_invent_missing_character_coverage(minimal_run: InsightsRun) -> None:
    run = minimal_run.model_copy(update={'population': {
        'mode': 'snapshot', 'n_scanned': 2, 'n_projection_truncated': 1
    }})

    notice = failures(run)

    assert 'Input character coverage was not recorded for this earlier run.' in notice
    assert '50,000-byte' not in notice
    assert 'Whole-message counts were not saved' not in notice


def test_single_failed_trace_uses_singular_wording(minimal_run: InsightsRun) -> None:
    run = minimal_run.model_copy(update={'counts': {'n_failed_traces': 1}})

    assert '1 trace had a label' in failures(run)


def test_insights_header_marks_unknown_cost(minimal_run) -> None:
    run = minimal_run.model_copy(update={'cost_by_stage': {'summary': None}})

    assert 'cost unknown' in header(run)


def test_score_labels_use_criterion_names_and_ignore_failed_answers(minimal_run) -> None:
    score = LabelSpec(name='customer_satisfaction', kind='score', instructions='Rate satisfaction.', criteria=['low', 'mid', 'high'])
    run = minimal_run.model_copy(deep=True)
    run.config.labels.append(score)
    run.labels['customer_satisfaction'] = LabelResult(
        spec=score, counts={'2': 1, '1': 1}, mean_confidence=0.9, n_low_confidence=0, n_failed=1
    )
    run.traces[0].labels['customer_satisfaction'] = LabelAnswer(
        value=0.5, confidence=0.9, probabilities=None, error=None
    )
    run.traces[1].labels['customer_satisfaction'] = LabelAnswer(
        value=0.75, confidence=None, probabilities=None, error='invalid score'
    )

    assert '>mid</span>' in labels(run)
    assert labels(run).index('>mid</span>') < labels(run).index('>high</span>')
    assert 'mid 1/2' in cluster_detail(run, 'base-1')
    filtered = traces(run, label='customer_satisfaction', value='1')
    assert 'trace-1' in filtered
    assert 'trace-2' not in filtered
    assert 'No traces match these filters' in traces(run, label='customer_satisfaction', value='2')


def test_insights_progress_renders_stage_completed_and_total() -> None:
    now = datetime.now(timezone.utc)
    manifest = RunManifest(
        run_id='progress-run',
        surface=ManifestSurface.INSIGHTS,
        run_name='Progress run',
        status=ManifestStatus.RUNNING,
        stage='label',
        stages=[
            StageRecord(
                name='label',
                status=ManifestStatus.RUNNING,
                started_at=now,
                completed=40,
                total=100,
            )
        ],
        planned_stages=['label'],
        started_at=now,
        updated_at=now,
    )

    assert '<small>40/100</small>' in progress(manifest)


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


def test_review_page_has_mock_shell_and_no_trace_data(minimal_run: InsightsRun) -> None:
    from evaluatorq.dashboard.insights_review_views import review_page

    html = review_page(minimal_run, None)

    for element_id in ('header', 'heads', 'fbar', 'tabs', 'canvas', 'tlist', 'drawer', 'sheet'):
        assert f'id="{element_id}"' in html
    assert 'data-review-url="/insights/run-1/review-data.json"' in html
    assert 'Loading run insights' in html
    assert 'Could not load this run' in html
    assert 'What is the refund policy?' not in html
    assert 'insights-review.css' in html and 'insights-run-form.js' in html and 'insights-review.js' in html


def test_review_data_route_returns_complete_uncached_payload(tmp_path, monkeypatch, minimal_run: InsightsRun) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    monkeypatch.setenv('ORQ_WORKSPACE', 'test-workspace')
    many = minimal_run.model_copy(update={
        'traces': [
            minimal_run.traces[index % len(minimal_run.traces)].model_copy(
                update={'trace_id': f'trace-{index}', 'span_id': f'span-{index}'}
            )
            for index in range(5000)
        ]
    })
    _write_run(tmp_path, many)

    response = TestClient(build_app()).get('/insights/run-1/review-data.json')

    assert response.status_code == 200
    assert response.headers['cache-control'] == 'no-store'
    assert len(response.json()['traces']) == 5000


def test_review_data_route_reports_missing_and_unreadable_runs(tmp_path, monkeypatch, minimal_run: InsightsRun) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    _write_run(tmp_path, minimal_run, 'insights_unreadable.json').write_text('{', encoding='utf-8')
    client = TestClient(build_app())

    missing = client.get('/insights/missing/review-data.json')
    unreadable = client.get('/insights/insights_unreadable/review-data.json')

    assert missing.status_code == 404 and missing.json()['error'] == 'Insights run not found'
    assert unreadable.status_code == 422 and 'unreadable' in unreadable.json()['error'].lower()
    assert missing.headers['cache-control'] == 'no-store'


def test_default_dashboard_page_does_not_enable_review_shell() -> None:
    from evaluatorq.dashboard.shell import page

    html = page('Same title', '<p>Same body</p>')

    assert 'eq-insights-review' not in html
    assert 'insights-review.css' not in html
    assert 'insights-review.js' not in html


def test_review_page_escapes_the_run_id_in_same_origin_data_url(minimal_run: InsightsRun) -> None:
    from evaluatorq.dashboard.insights_review_views import review_page

    html = review_page(minimal_run.model_copy(update={'run_id': 'run & <one>'}), None)

    assert 'data-review-url="/insights/run%20%26%20%3Cone%3E/review-data.json"' in html


def test_review_page_does_not_embed_trace_link_template(monkeypatch, minimal_run: InsightsRun) -> None:
    from evaluatorq.dashboard.insights_review_views import review_page

    monkeypatch.setenv('ORQ_WORKSPACE', 'safe workspace')
    monkeypatch.setenv('ORQ_UI_BASE_URL', 'https://orq.example')
    html = review_page(minimal_run, None)

    assert 'data-orq-template=' not in html
    assert 'EVALQTRACEPLACEHOLDER' not in html
    assert 'EVALQSPANPLACEHOLDER' not in html
    assert 'trace-1' not in html


def test_review_client_escapes_saved_ids_and_labels_partial_cost(minimal_run: InsightsRun) -> None:
    from pathlib import Path

    from evaluatorq.dashboard.insights_review_data import build_review_payload

    script = (Path(__file__).parents[2] / 'src/evaluatorq/dashboard/static/insights-review.js').read_text(encoding='utf-8')
    hostile_id = 'trace" onmouseover="alert(1)'
    saved = minimal_run.model_copy(update={
        'traces': [minimal_run.traces[0].model_copy(update={'trace_id': hostile_id})],
        'cost_by_stage': {
            'summary': Usage(input_tokens=10, output_tokens=2, total_tokens=12, total_cost=0.25, calls=3, priced_calls=1)
        },
    })
    payload = build_review_payload(saved)
    run_data = cast(dict[str, Any], payload['run'])
    trace_data = cast(list[dict[str, Any]], payload['traces'])

    assert trace_data[0]['trace_id'] == hostile_id
    assert run_data['cost'] == 0.25
    assert run_data['cost_is_partial'] is True
    assert run_data['cost_by_stage_details']['summary'] == {
        'cost': 0.25, 'is_partial': True, 'priced_calls': 1, 'calls': 3,
    }
    assert 'data-t="${esc(t.id)}"' in script
    assert 'data-t="${esc(id)}"' in script
    assert 'data-c="${esc(c.id)}"' in script
    assert 'data-go="${esc(k.id)}"' in script
    assert 'data-backto="${esc(S.sel.prev.id)}"' in script
    assert 'data-t="${t.id}"' not in script
    assert 'data-c="${c.id}"' not in script
    assert 'run.cost_is_partial ? `Partial cost: $${run.cost.toFixed(3)}`' in script
    assert "stageCost.is_partial ? 'Partial cost: ' : ''" in script
    assert "stageCost?.cost == null" in script


def test_review_client_uses_full_trace_keys_and_label_score_keys() -> None:
    from pathlib import Path

    script = (Path(__file__).parents[2] / 'src/evaluatorq/dashboard/static/insights-review.js').read_text(encoding='utf-8')

    assert "function encodeTraceKey(id)" in script
    assert "map(encodeTraceKey).join('.')" in script
    assert "i.slice(0, 8)" not in script
    assert "return lvals(l).includes(key) ? key : null" in script
    assert "const lnum = (_name, v) => +v" in script
    assert "history.pushState(null, '', h)" in script
    assert "addEventListener('popstate', () => { fromUrl(); render(); })" in script
    assert 'const traceUrl = safeTraceUrl(t.trace_url);' in script
    assert 'const orqUrl = safeOrqUrl(t.orq_url);' in script
    assert 'function safeTraceUrl(value)' in script
    assert 'function safeOrqUrl(value)' in script


def test_review_trace_links_reject_unsafe_url_schemes() -> None:
    import subprocess
    from pathlib import Path

    script = Path(__file__).with_name('insights_review_url.cjs')
    subprocess.run(['node', str(script)], check=True)


def test_review_fallback_payloads_keep_the_same_page_structure(minimal_run: InsightsRun) -> None:
    from pathlib import Path

    from evaluatorq.dashboard.insights_review_data import build_review_payload
    from evaluatorq.dashboard.insights_review_views import review_page

    no_labels = minimal_run.model_copy(update={
        'config': minimal_run.config.model_copy(update={'labels': []}),
        'labels': {},
        'traces': [trace.model_copy(update={'labels': {}}) for trace in minimal_run.traces],
    })
    partial = minimal_run.model_copy(update={'status': 'error', 'warnings': ['A stage failed.']})
    no_summaries = minimal_run.model_copy(update={
        'traces': [trace.model_copy(update={'summary': None}) for trace in minimal_run.traces]
    })
    runs = [minimal_run, no_labels, partial, no_summaries]
    script = (Path(__file__).parents[2] / 'src/evaluatorq/dashboard/static/insights-review.js').read_text(encoding='utf-8')

    for run in runs:
        page_html = review_page(run, None)
        for element_id in ('header', 'heads', 'fbar', 'tabs', 'canvas', 'tlist', 'drawer', 'sheet'):
            assert f'id="{element_id}"' in page_html
        assert list(build_review_payload(run)['dims']) == ['intent']
    assert build_review_payload(no_labels)['labels'] == []
    assert cast(dict[str, Any], build_review_payload(partial)['run'])['status'] == 'error'
    assert all(not trace['has_summary'] for trace in build_review_payload(no_summaries)['traces'])
    assert 'No usable label results' in script
    assert 'Summaries unavailable for' in script
    assert 'Partial run' in script


def test_empty_insights_overview_points_to_new_run(tmp_path, monkeypatch):
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))

    response = TestClient(build_app()).get('/insights')

    assert response.status_code == 200
    assert 'No Insights runs yet' in response.text
    assert 'href="/insights/new"' in response.text


def test_insights_overview_shows_saved_stage_progress(tmp_path, monkeypatch):
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    writer = start_manifest(
        run_id='stage-run',
        surface='insights',
        run_name='Stage run',
        runs_dir=tmp_path / 'insights-runs',
        planned_stages=['population', 'summary', 'write'],
        stage_labels={'population': 'Load traces', 'summary': 'Summarize traces', 'write': 'Save run'},
    )
    writer.start_stage('population')
    writer.end_stage('population')
    writer.start_stage('summary')
    client = TestClient(build_app())

    running = client.get('/insights')
    assert running.status_code == 200
    assert 'Load traces: completed' in running.text
    assert 'Summarize traces: running' in running.text
    assert 'Save run: pending' in running.text
    assert '<span>Run</span><span>Stages</span><span>Status</span>' in running.text
    assert '</small></span><span class="insights-overview-stage-wrap">' in running.text
    assert 'class="insights-overview-stages"' in running.text
    assert '<span>1/3</span>' in running.text

    writer.fail('Summary failed', stage='summary')
    failed = client.get('/insights')
    assert 'Summarize traces: error' in failed.text
    assert 'Save run: skipped' in failed.text


def test_insights_overview_opens_a_dedicated_run_page(tmp_path, minimal_run, monkeypatch):
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    _write_run(tmp_path, minimal_run)
    client = TestClient(build_app())

    response = client.get('/insights')

    assert response.status_code == 200
    assert 'minimal run' in response.text
    assert 'Recent runs' in response.text
    assert 'href="/insights/run-1"' in response.text
    assert 'href="/insights/new"' in response.text
    assert 'General requests' not in response.text

    detail = client.get('/insights/run-1')
    assert detail.status_code == 200
    assert 'insights-review-root' in detail.text
    data = client.get('/insights/run-1/review-data.json').json()
    assert data['run']['population']['query'] == 'refund policy'
    assert data['run']['population']['facets']['agent_name'] == ['support-bot']
    assert data['run']['population']['limit'] == 500
    assert data['run']['created'] == '2026-09-01T00:00:00Z'
    assert 'sentiment' in [label['name'] for label in data['labels']]
    assert data['dims']['intent']['clusters'][1]['name'] == 'General requests'


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

    client = TestClient(build_app())
    response = client.get('/insights/run-1')
    data = client.get('/insights/run-1/review-data.json').json()

    assert response.status_code == 200
    assert data['run']['status'] == 'error'
    assert data['run']['stage_failures'][0]['stage'] == 'dimension:failure'
    assert data['run']['stage_failures'][0]['message'] == 'embedding service unavailable'
    assert data['run']['counts']['n_failed_traces'] == 1


def test_cluster_panel_has_description_and_example_trace_link(tmp_path, minimal_run, monkeypatch):
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    monkeypatch.setenv('ORQ_WORKSPACE', 'example-workspace')
    _write_run(tmp_path, minimal_run)

    response = TestClient(build_app()).get('/insights/run-1/cluster/base-1')

    assert response.status_code == 200
    assert 'Traces about general requests.' in response.text
    assert 'trace-1' in response.text
    assert 'example-workspace/traces/(trace:trace-1//span:span-1)' in response.text
    assert 'href="/insights/run-1/trace?trace_id=trace-1&amp;span_id=span-1"' in response.text


def test_trace_id_opens_its_saved_insights_detail(tmp_path, minimal_run, monkeypatch):
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    monkeypatch.setenv('ORQ_WORKSPACE', 'example-workspace')
    _write_run(tmp_path, minimal_run)
    client = TestClient(build_app())

    table = client.get('/insights/run-1/tab/traces', headers={'HX-Request': 'true'})
    detail = client.get('/insights/run-1/trace?trace_id=trace-1&span_id=span-1')
    missing = client.get('/insights/run-1/trace?trace_id=trace-1&span_id=missing')

    assert 'href="/insights/run-1/trace?trace_id=trace-1&amp;span_id=span-1"' in table.text
    assert detail.status_code == 200
    assert 'A user asked a question.' in detail.text
    assert 'What is the refund policy?' in detail.text
    assert 'General requests' in detail.text
    assert 'Open full trace in Orq' in detail.text
    assert 'href="/insights/run-1/tab/traces"' in detail.text
    assert missing.status_code == 404


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


def test_traces_route_forwards_crosstab_filters(tmp_path, minimal_run, monkeypatch):
    from evaluatorq.dashboard import insights_views

    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    _write_run(tmp_path, minimal_run)
    seen: dict[str, object] = {}

    def fake_traces(run, **kwargs):
        seen.update(kwargs)
        return '<p>ok</p>'

    monkeypatch.setattr(insights_views, 'traces', fake_traces)

    response = TestClient(build_app()).get(
        '/insights/run-1/traces?row=intent&row_value=a&column=sentiment&column_value=positive'
    )

    assert response.status_code == 200
    assert seen['row'] == 'intent'
    assert seen['row_value'] == 'a'
    assert seen['column'] == 'sentiment'
    assert seen['column_value'] == 'positive'


def test_labels_without_any_labels_show_empty_state(tmp_path, minimal_run, monkeypatch):
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    no_labels = minimal_run.model_copy(
        update={
            'labels': {},
            'config': minimal_run.config.model_copy(update={'labels': []}),
            'traces': [trace.model_copy(update={'labels': {}}) for trace in minimal_run.traces],
        }
    )
    _write_run(tmp_path, no_labels)

    client = TestClient(build_app())
    response = client.get('/insights/run-1/tab/labels', follow_redirects=False)
    run_page = client.get('/insights/run-1')
    data = client.get('/insights/run-1/review-data.json').json()

    assert response.status_code == 302
    assert response.headers['location'] == '/insights/run-1#dim=intent&view=themes'
    assert data['labels'] == []
    assert 'insights-review-root' in run_page.text


def test_traces_show_active_filter_chips_and_clear_filter_navigation(tmp_path, minimal_run, monkeypatch):
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    _write_run(tmp_path, minimal_run)

    response = TestClient(build_app()).get(
        '/insights/run-1/tab/traces?dimension=intent&cluster=base-1&label=sentiment&value=positive',
        follow_redirects=False,
    )

    assert response.status_code == 302
    assert response.headers['location'] == (
        '/insights/run-1#dim=intent&view=themes&f=c%3Abase-1%7Cl%3Asentiment%3Apositive'
    )


def test_tab_route_redirects_navigation_and_keeps_fragment_for_htmx(tmp_path, minimal_run, monkeypatch):
    from evaluatorq.dashboard import insights_routes

    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    _write_run(tmp_path, minimal_run)
    original_entries = insights_routes._entries
    entries_calls = 0

    def counted_entries(directory):
        nonlocal entries_calls
        entries_calls += 1
        return original_entries(directory)

    monkeypatch.setattr(insights_routes, '_entries', counted_entries)
    client = TestClient(build_app())

    page_response = client.get('/insights/run-1/tab/labels', follow_redirects=False)
    assert entries_calls == 1
    fragment_response = client.get('/insights/run-1/tab/labels', headers={'HX-Request': 'true'})
    assert entries_calls == 2

    assert page_response.status_code == 302
    assert page_response.headers['location'] == '/insights/run-1#dim=intent&view=themes'
    assert fragment_response.status_code == 200
    assert not fragment_response.text.startswith('<!DOCTYPE html>')
    assert 'insights-label-grid' in fragment_response.text


@pytest.mark.parametrize('tab', TABS)
def test_htmx_tab_fragment_moves_the_active_indicator(tmp_path, minimal_run, monkeypatch, tab):
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    _write_run(tmp_path, minimal_run)
    client = TestClient(build_app())

    fragment = client.get(f'/insights/run-1/tab/{tab}', headers={'HX-Request': 'true'}).text

    assert '<nav id="insights-tabs" class="insights-tabs" aria-label="Insights views" hx-swap-oob="true">' in fragment
    assert fragment.count('insights-tab active') == 1
    assert f'class="insights-tab active" href="/insights/run-1/tab/{tab}"' in fragment
    assert fragment.count('hx-sync="closest nav:replace"') == len(TABS)


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


def test_run_overview_keeps_newest_first_order(tmp_path, minimal_run, monkeypatch):
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    older = _write_run(tmp_path, minimal_run, 'insights_older.json')
    newer_run = minimal_run.model_copy(update={'run_id': 'run-2', 'run_name': 'newer run'})
    newer = _write_run(tmp_path, newer_run, 'insights_newer.json')
    os.utime(older, ns=(1_000_000_000, 1_000_000_000))
    os.utime(newer, ns=(2_000_000_000, 2_000_000_000))

    response = TestClient(build_app()).get('/insights')

    assert response.status_code == 200
    assert response.text.index('newer run') < response.text.index('minimal run')


def test_completed_run_shows_escaped_run_warnings(tmp_path, minimal_run, monkeypatch):
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    warning = 'Small sentiment group: <script>alert(1)</script>'
    _write_run(tmp_path, minimal_run.model_copy(update={'warnings': [warning]}))

    response = TestClient(build_app()).get('/insights/run-1')

    assert response.status_code == 200
    assert response.status_code == 200
    assert 'insights-review-root' in response.text
    assert TestClient(build_app()).get('/insights/run-1/review-data.json').json()['run']['warnings'] == [warning]


def test_review_page_shows_auth_warning_host(minimal_run: InsightsRun) -> None:
    from evaluatorq.dashboard.insights_review_views import review_page

    html = review_page(minimal_run, None)

    assert 'id="eq-auth-toast"' in html
    assert "fetch('/auth/status'" in html
    assert '<header class="app-topbar">' not in html
    assert 'class="eq-dashboard eq-insights-review"' in html
