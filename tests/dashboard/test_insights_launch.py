"""Dashboard run creation and manifest-backed Insights progress."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient

from evaluatorq.common.run_manifest import list_manifests, start_manifest
from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.insights_launch import InsightsLaunchPayload, InsightsLaunchSpec, launch_insights
from evaluatorq.insights.models import InsightsPopulation
from evaluatorq.insights.presets import CUSTOMER_SATISFACTION, SENTIMENT
from evaluatorq.insights.progress import stage_plan
from evaluatorq.insights.store import save_run
from evaluatorq.trace_finder.models import FacetCatalogue, FacetSelection
from tests.dashboard.test_insights_page import minimal_run


def test_stage_plan_follows_source_labels_and_dimensions() -> None:
    query_plan = dict(stage_plan(InsightsPopulation(query='refunds'), [SENTIMENT], ['intent', 'failure']))
    finder_plan = dict(stage_plan(InsightsPopulation.from_finder_export(Path('finder.json')), [], ['sentiment']))
    priority_plan = dict(stage_plan(InsightsPopulation(), [CUSTOMER_SATISFACTION], ['intent']))

    assert query_plan['population'] == 'Find matching traces'
    assert query_plan['label'] == 'Match and classify traces'
    assert list(query_plan).count('dimension:intent') == 1
    assert 'dimension:failure' in query_plan
    assert 'dimension:sentiment' not in query_plan
    assert finder_plan['population'] == 'Load Finder matches'
    assert finder_plan['label'] == 'Classify traces'
    assert priority_plan['priority'] == 'Build priority matrix'


def test_launch_persists_plan_before_spawning_worker(tmp_path: Path) -> None:
    spec = InsightsLaunchSpec(source='query', query='refunds', labels=['sentiment'], dimensions=['intent', 'failure'])
    with patch('evaluatorq.dashboard.insights_launch.subprocess.Popen') as spawn:
        run_id = launch_insights(spec, tmp_path)

    manifest = list_manifests(tmp_path)[0]
    assert manifest.run_id == run_id
    assert manifest.status == 'running'
    assert manifest.planned_stages == ['population', 'label', 'summary', 'dimension:intent', 'dimension:failure', 'priority', 'write']
    assert manifest.stage_labels['label'] == 'Match and classify traces'
    command = spawn.call_args.args[0]
    assert command[1:] == ['-m', 'evaluatorq.dashboard.insights_worker']
    payload = InsightsLaunchPayload.model_validate_json(spawn.call_args.kwargs['env']['EVALUATORQ_INSIGHTS_LAUNCH_REQUEST'])
    assert payload.run_id == run_id
    assert payload.spec.query == 'refunds'
    assert spawn.call_args.kwargs['start_new_session'] is True


def test_spawn_failure_is_visible_in_manifest(tmp_path: Path) -> None:
    with patch('evaluatorq.dashboard.insights_launch.subprocess.Popen', side_effect=OSError('worker unavailable')):
        run_id = launch_insights(InsightsLaunchSpec(), tmp_path)

    manifest = list_manifests(tmp_path)[0]
    assert manifest.run_id == run_id
    assert manifest.status == 'error'
    assert manifest.error is not None and 'worker unavailable' in manifest.error


def test_startup_failure_shows_in_run_page(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    directory = tmp_path / 'insights-runs'
    with patch('evaluatorq.dashboard.insights_launch.subprocess.Popen', side_effect=OSError('worker unavailable')):
        run_id = launch_insights(InsightsLaunchSpec(), directory)

    response = TestClient(build_app()).get(f'/insights/{run_id}')
    assert response.status_code == 200
    assert 'worker unavailable' in response.text
    assert '<span>Start</span><small>error</small>' in response.text
    assert 'Load recent traces' in response.text


def test_wizard_validates_source_before_launch() -> None:
    with pytest.raises(ValidationError, match='Enter a question'):
        InsightsLaunchSpec(source='query', query='  ')
    with pytest.raises(ValidationError, match='Select at least one'):
        InsightsLaunchSpec(labels=[], dimensions=[])
    with pytest.raises(ValidationError, match='Finder export already fixes'):
        InsightsLaunchSpec(source='finder', facets=FacetSelection(status=frozenset({'error'})))


def test_live_facet_selection_reaches_population_and_worker(tmp_path: Path) -> None:
    facets = FacetSelection(status=frozenset({'error'}), provider=frozenset({'openai', 'anthropic'}))
    spec = InsightsLaunchSpec(source='query', query='refunds', window_days=3, facets=facets)
    assert spec.population().facets == facets
    assert spec.population().window_days == 3
    with patch('evaluatorq.dashboard.insights_launch.subprocess.Popen') as spawn:
        launch_insights(spec, tmp_path)
    payload = InsightsLaunchPayload.model_validate_json(spawn.call_args.kwargs['env']['EVALUATORQ_INSIGHTS_LAUNCH_REQUEST'])
    assert payload.spec.population().facets == facets


def test_facet_options_use_selected_window_and_keep_values(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    client = TestClient(build_app())
    catalogue = FacetCatalogue(status=('ok',), provider=('openai',))
    load = AsyncMock(return_value=catalogue)
    with (
        patch('evaluatorq.dashboard.insights_routes.resolve_orq_client', return_value=object()),
        patch('evaluatorq.dashboard.insights_routes.load_facet_catalogue', load),
        patch('evaluatorq.dashboard.insights_routes.close_orq_client', new_callable=AsyncMock),
    ):
        response = client.get('/insights/facets?window_days=3&facet_status=error')
        cached = client.get('/insights/facets?window_days=3')
        changed = client.get('/insights/facets?window_days=4')

    assert response.status_code == cached.status_code == changed.status_code == 200
    assert 'name="facet_status" value="error" checked' in response.text
    assert 'name="facet_status" value="ok"' in response.text
    assert load.await_count == 2
    first = load.await_args_list[0].kwargs
    assert first['limit'] == 50
    assert isinstance(first['start'], datetime)
    assert isinstance(first['end'], datetime)
    assert abs(first['end'] - first['start'] - timedelta(days=3)) < timedelta(seconds=1)
    assert client.get('/insights/facets?window_days=91').status_code == 422


def test_facet_catalogue_failure_is_visible(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    client = TestClient(build_app())
    with (
        patch('evaluatorq.dashboard.insights_routes.resolve_orq_client', return_value=object()),
        patch('evaluatorq.dashboard.insights_routes.load_facet_catalogue', new_callable=AsyncMock, side_effect=RuntimeError('Orq unavailable')),
        patch('evaluatorq.dashboard.insights_routes.close_orq_client', new_callable=AsyncMock),
    ):
        response = client.get('/insights/facets?window_days=7&facet_status=error')
    assert response.status_code == 200
    assert 'Facet values are unavailable' in response.text
    assert 'name="facet_status" value="error" checked' in response.text


def test_new_run_form_rejects_bad_input_and_launches_valid_request(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    client = TestClient(build_app())
    page = client.get('/insights/new')
    assert page.status_code == 200
    assert 'New Insights run' in page.text
    token = re.search(r'name="csrf" value="([^"]+)"', page.text)
    assert token is not None

    with patch('evaluatorq.dashboard.insights_routes.launch_insights', return_value='launched-id') as launch:
        invalid = client.post('/insights/runs', data={'csrf': token.group(1), 'source': 'query', 'dimensions': 'intent'})
        assert invalid.status_code == 422
        assert 'Enter a question' in invalid.text
        forbidden = client.post('/insights/runs', data={'source': 'recent', 'dimensions': 'intent'})
        assert forbidden.status_code == 403
        valid = client.post(
            '/insights/runs',
            data={'csrf': token.group(1), 'source': 'recent', 'dimensions': 'intent', 'limit': '5', 'facet_status': 'error'},
            follow_redirects=False,
        )

    assert valid.status_code == 303
    assert valid.headers['location'] == '/insights/launched-id'
    assert launch.call_args.args[0].limit == 5
    assert launch.call_args.args[0].population().facets.status == frozenset({'error'})


def test_running_and_completed_pages_show_manifest_stages(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    directory = tmp_path / 'insights-runs'
    writer = start_manifest(
        run_id='run-1',
        surface='insights',
        run_name='Sample run',
        runs_dir=directory,
        planned_stages=['population', 'summary', 'dimension:intent', 'write'],
        stage_labels={'population': 'Load recent traces', 'summary': 'Summarize traces', 'dimension:intent': 'Cluster intent', 'write': 'Save run'},
    )
    writer.start_stage('population')
    writer.end_stage('population')
    writer.start_stage('summary')
    client = TestClient(build_app())

    running = client.get('/insights/run-1')
    overview = client.get('/insights')
    assert running.status_code == overview.status_code == 200
    assert 'Summarize traces' in running.text
    assert 'pending' in running.text
    assert 'Summarize traces' in overview.text
    assert 'location.reload()' in running.text

    writer.end_stage('summary')
    report = save_run(minimal_run.__wrapped__(), directory)
    writer.complete(report)
    completed = client.get('/insights/run-1')
    assert completed.status_code == 200
    assert 'Load recent traces' in completed.text
    assert 'Cluster intent' in completed.text
    assert 'skipped' in completed.text
