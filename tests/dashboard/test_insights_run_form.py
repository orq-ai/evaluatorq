"""The one Insights run form, shown as a page and as the run-page dialog."""

# Assertions are the test framework's reporting contract.
# ruff: noqa: S101

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from evaluatorq.common.model_catalogue import ModelInfo
from evaluatorq.common.model_roles import BUILTIN
from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.auth import DashboardAuth
from evaluatorq.dashboard.insights_launch import InsightsLaunchSpec, get_finder_exports_dir
from evaluatorq.dashboard.insights_run_form import (
    INSIGHTS_MODEL_FIELDS,
    OFFERED_DIMENSIONS,
    OFFERED_LABELS,
    RUN_PRESETS,
    RunFormValues,
    _saved_source,
    render_run_form,
)
from evaluatorq.common.run_manifest import start_manifest
from evaluatorq.contracts import ManifestStatus, StageRecord
from evaluatorq.insights.models import InsightsConfig, InsightsRun, LabelSpec
from evaluatorq.insights.presets import CODING_LABELS, LABEL_PRESETS, TASK_TYPE
from evaluatorq.insights.store import get_insights_runs_dir
from evaluatorq.trace_finder.export import (
    ExportCounts,
    ExportDimension,
    ExportFilters,
    ExportNumericFilters,
    ExportTask,
    ExportTimes,
    ExportValuesSelection,
    RunExport,
)
from evaluatorq.trace_finder.models import FacetCatalogue, FacetSelection

_AUTH = DashboardAuth('environment', 'key', 'https://my.orq.ai')


def _token(client: TestClient) -> str:
    match = re.search(r'name="csrf" value="([^"]+)"', client.get('/insights/new').text)
    assert match is not None
    return match.group(1)


def _saved_run(population: dict[str, object], **config: object) -> InsightsRun:
    custom = LabelSpec(name='needs_follow_up', kind='noul', instructions='Should a human follow up?')
    return InsightsRun(
        run_id='saved-run',
        run_name='Saved run',
        created_at=datetime(2026, 9, 1, 12, tzinfo=timezone.utc),
        status='completed',
        stage_failures=[],
        population=population,
        config=InsightsConfig(
            labels=[LABEL_PRESETS['user_frustration'], custom],
            dimensions=['intent', 'failure'],
            coding_labels=[CODING_LABELS[1]],
            **config,
        ),
        traces=[],
        dimensions={},
        labels={},
        priority=None,
        priority_reason=None,
        counts={'n_traces': 0},
        warnings=[],
    )


def test_defaults_render_three_steps_with_every_offered_choice() -> None:
    html = render_run_form(RunFormValues.defaults(), csrf='t')

    assert html.count('data-step="') == 3
    for source in ('orq', 'file'):
        assert f'name="source" value="{source}"' in html
    for retired in ('recent', 'query', 'finder', 'snapshot'):
        assert f'name="source" value="{retired}"' not in html
    assert 'name="source" value="orq" checked' in html
    assert 'Question <i>(optional)</i>' in html
    assert 'class="irf-tab-icon"' in html and 'aria-hidden="true"' in html
    assert 'Presets' in html
    for preset in RUN_PRESETS:
        assert f'data-preset="{preset.id}"' in html
    for name in (*OFFERED_LABELS, *OFFERED_DIMENSIONS, *(spec.name for spec in CODING_LABELS[1:])):
        assert f'value="{name}"' in html
    task_type = html[html.index('value="task_type"') :]
    task_type = task_type[: task_type.index('</label>')]
    assert all(key in task_type for key in TASK_TYPE.criteria or {})
    assert 'implementation' not in html
    assert 'id="insights-run-models"' in html
    assert 'id="insights-run-estimate"' in html
    assert 'name="csrf" value="t"' in html
    assert 'data-mount="page"' in html
    assert 'name="trace_file" value="" data-kind=""' in html and 'name="finder_export"' not in html
    assert 'A Finder export or a local trace file (JSON).' in html
    assert html.count('type="file"') == 1


def test_customer_satisfaction_says_the_priority_matrix_needs_it() -> None:
    html = render_run_form(RunFormValues.defaults(), csrf='t')

    toggle = html[html.index('value="customer_satisfaction"') :]
    assert 'the priority matrix needs this' in toggle[: toggle.index('</label>')]
    assert 'value="customer_satisfaction" checked' not in html


def test_defaults_are_the_find_failures_preset_on_recent_traces() -> None:
    values = RunFormValues.defaults()

    assert (values.source, values.window_days, values.limit) == ('recent', 7, 200)
    assert values.preset == RUN_PRESETS[0].id
    assert values.dimensions == RUN_PRESETS[0].dimensions
    assert values.labels == RUN_PRESETS[0].labels


def test_from_run_round_trips_a_saved_run(tmp_path: Path) -> None:
    run = _saved_run({'mode': 'query', 'query': 'refunds', 'window_days': 14, 'limit': 50, 'facets': {'agent_name': ['a']}})

    values = RunFormValues.from_run(run, tmp_path)

    assert values.source == 'query'
    assert (values.query, values.window_days, values.limit) == ('refunds', 14, 50)
    assert values.facets == FacetSelection(agent_name=frozenset({'a'}))
    assert values.dimensions == ('intent', 'failure')
    assert values.labels == ('user_frustration',)
    assert values.coding_labels == (CODING_LABELS[1].name,)
    assert [spec.name for spec in values.custom_labels] == ['needs_follow_up']
    assert values.name == 'Saved run'
    assert values.error is None


def test_from_run_clears_a_missing_uploaded_file_and_asks_for_a_fresh_one(tmp_path: Path) -> None:
    missing = RunFormValues.from_run(_saved_run({'mode': 'snapshot', 'snapshot_path': str(tmp_path / 'gone.json')}), tmp_path)
    assert missing.source == 'snapshot'
    assert missing.trace_file == ''
    assert missing.error is not None and 'fresh file' in missing.error

    present = tmp_path / 'traces.json'
    present.write_text('{}', encoding='utf-8')
    kept = RunFormValues.from_run(_saved_run({'mode': 'snapshot', 'snapshot_path': str(present)}), tmp_path)
    assert kept.trace_file == str(present)
    assert kept.error is None


@pytest.mark.parametrize(
    ('population', 'expected'),
    [
        ({}, 'recent'),
        ({'mode': 'snapshot'}, 'snapshot'),
        ({'mode': 'export', 'snapshot_path': '/x.json'}, 'snapshot'),
        ({'mode': 'export', 'finder_export': '/x.json', 'query': 'refunds'}, 'finder'),
        ({'finder_export': '/x.json'}, 'finder'),
        ({'mode': 'finder'}, 'finder'),
        ({'mode': 'query'}, 'query'),
        ({'query': 'refunds'}, 'query'),
        ({'mode': 'export', 'query': 'refunds'}, 'query'),
        ({'mode': 'export'}, 'recent'),
        ({'mode': 'filter'}, 'recent'),
        ({'mode': 'dataset'}, 'recent'),
    ],
)
def test_saved_source_maps_each_saved_population_to_its_form_source(population: dict[str, object], expected: str) -> None:
    assert _saved_source(population) == expected


def test_from_form_reads_query_parameters_and_form_data_the_same_way() -> None:
    from starlette.datastructures import FormData, QueryParams

    pairs = [
        ('source', 'orq'),
        ('query', 'refunds'),
        ('window_days', '3'),
        ('limit', '40'),
        ('facet_status', 'error'),
        ('facet_status', 'ok'),
        ('dimensions', 'failure'),
        ('labels', 'made_errors'),
        ('coding_labels', 'outcome'),
        ('custom_labels_json', json.dumps([{'name': 'q_one', 'kind': 'noul', 'instructions': 'Is it?'}])),
        ('mount', 'dialog'),
    ]
    from_query = RunFormValues.from_form(QueryParams(pairs))
    from_post = RunFormValues.from_form(FormData(pairs))

    assert from_query == from_post
    assert from_query.facets.status == frozenset({'error', 'ok'})
    assert from_query.mount == 'dialog'
    assert from_query.launch_fields()['labels'] == ['made_errors']


def test_stages_wrap_the_stage_plan_the_launch_uses() -> None:
    spec = InsightsLaunchSpec(source='query', query='refunds', labels=['made_errors'], dimensions=['intent'])

    titles = [title for _, title in spec.stages()]

    assert titles[0] == 'Find matching traces'
    assert 'Match and classify traces' in titles


def test_rejected_start_rerenders_what_the_user_entered(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    client = TestClient(build_app())
    data = {
        'csrf': _token(client),
        'source': 'orq',
        'query': 'refunds',
        'window_days': '21',
        'limit': '0',
        'facet_agent_name': 'support-bot',
        'dimensions': 'intent',
        'custom_labels_json': json.dumps([{'name': 'needs_follow_up', 'kind': 'noul', 'instructions': 'Follow up?'}]),
    }

    page = client.post('/insights/runs', data=data)
    fragment = client.post('/insights/runs', data={**data, 'mount': 'dialog'})

    assert page.status_code == fragment.status_code == 422
    for response in (page, fragment):
        assert 'greater than or equal to 1' in response.text
        assert 'name="window_days" type="number" min="1" max="90" value="21"' in response.text
        assert 'data-chip-name="facet_agent_name" data-finder-value="support-bot"' in response.text
        assert 'needs_follow_up' in response.text
    assert '<html' in page.text
    assert '<html' not in fragment.text
    assert 'data-mount="dialog"' in fragment.text
    assert client.post('/insights/runs', data={'source': 'orq', 'mount': 'dialog'}).status_code == 403


def test_new_run_dialog_is_a_fragment_with_its_own_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    client = TestClient(build_app())

    fragment = client.get('/insights/new?mount=dialog')

    assert fragment.status_code == 200
    assert '<html' not in fragment.text
    assert 'id="insights-new-form"' in fragment.text
    assert re.search(r'name="csrf" value="[^"]+"', fragment.text)
    assert 'data-mount="dialog"' in fragment.text


def test_new_run_with_rerun_prefills_from_the_saved_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    directory = tmp_path / 'insights-runs'
    directory.mkdir()
    (directory / 'insights_saved.json').write_text(
        _saved_run({'mode': 'snapshot', 'snapshot_path': str(tmp_path / 'gone.json')}).model_dump_json(), encoding='utf-8'
    )
    client = TestClient(build_app())

    fragment = client.get('/insights/new?rerun=saved-run&mount=dialog')

    assert fragment.status_code == 200
    assert 'value="Saved run"' in fragment.text
    assert 'needs_follow_up' in fragment.text
    assert 'Browse for a fresh file' in fragment.text
    assert client.get('/insights/new?rerun=unknown&mount=dialog').status_code == 404


def test_plan_route_returns_the_stage_titles_the_launch_uses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    client = TestClient(build_app())
    query = {'source': 'orq', 'query': 'refunds', 'labels': 'made_errors', 'dimensions': 'intent'}

    response = client.get('/insights/new/plan', params=query)
    expected = InsightsLaunchSpec(source='query', query='refunds', labels=['made_errors'], dimensions=['intent']).stages()

    assert response.status_code == 200
    for _, title in expected:
        assert title in response.text
    filtered = client.get(
        '/insights/new/plan', params={'source': 'orq', 'facet_agent_name': 'support-bot', 'dimensions': 'intent'}
    )
    assert 'Filter recent traces' in filtered.text


def test_plan_route_reports_the_validation_message(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    client = TestClient(build_app())

    response = client.get('/insights/new/plan', params={'source': 'file', 'dimensions': 'intent'})

    assert response.status_code == 422
    assert 'role="alert"' in response.text
    assert 'Browse to choose a trace file first' in response.text


@pytest.fixture(autouse=True)
def _no_orq_lookups(monkeypatch: pytest.MonkeyPatch) -> None:
    async def nothing(*args: object, **kwargs: object) -> tuple[None, None, bool]:
        return None, None, False

    async def unpriced(auth: object, model: str) -> None:
        return None

    monkeypatch.setattr('evaluatorq.dashboard.insights_routes._catalogue', nothing)
    monkeypatch.setattr('evaluatorq.dashboard.insights_routes.catalogue_entry', unpriced)


_COUNTS = FacetCatalogue(value_counts={'status': {'completed': 90, 'error': 30}})
_PLAN_QUERY = {'source': 'orq', 'limit': '100', 'labels': 'made_errors', 'dimensions': 'intent'}


def _patch_estimate_inputs(monkeypatch: pytest.MonkeyPatch, *, price: ModelInfo | None) -> list[str]:
    priced: list[str] = []

    async def catalogue(*args: object, **kwargs: object) -> tuple[FacetCatalogue, str, bool]:
        return _COUNTS, 'research', False

    async def entry(auth: object, model: str) -> ModelInfo | None:
        priced.append(model)
        return price

    monkeypatch.setattr('evaluatorq.dashboard.insights_routes._catalogue', catalogue)
    monkeypatch.setattr('evaluatorq.dashboard.insights_routes.catalogue_entry', entry)
    monkeypatch.setattr('evaluatorq.dashboard.insights_routes.selected_dashboard_auth', lambda app: _AUTH)
    return priced


def test_plan_route_shows_up_to_traces_cost_and_a_first_run_time(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    priced = _patch_estimate_inputs(monkeypatch, price=ModelInfo(0.01, 0.03, 'acme', False))
    client = TestClient(build_app())

    response = client.get('/insights/new/plan', params=_PLAN_QUERY)

    assert response.status_code == 200
    assert 'up to 100 traces' in response.text
    assert re.search(r'up to \$\d', response.text)
    assert 'estimated after your first run' in response.text
    assert 'Expected stages' in response.text
    spec = RunFormValues.from_form(_PLAN_QUERY)
    assert set(priced) == {spec.summary_model, spec.classifier_model, spec.embedding_model}


def test_plan_route_names_the_stage_without_a_price(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    _patch_estimate_inputs(monkeypatch, price=None)
    client = TestClient(build_app())

    response = client.get('/insights/new/plan', params=_PLAN_QUERY)

    assert 'cost unknown' in response.text
    assert 'summary has no price' in response.text


def test_plan_route_without_facet_counts_bounds_traces_by_the_limit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    client = TestClient(build_app())

    response = client.get('/insights/new/plan', params=_PLAN_QUERY)

    assert 'up to 100 traces' in response.text
    assert 'the trace limit of 100' in response.text


def test_plan_route_time_comes_from_earlier_runs_and_scales_with_parallelism(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    _patch_estimate_inputs(monkeypatch, price=ModelInfo(0.01, 0.03, 'acme', False))
    runs = get_insights_runs_dir()
    writer = start_manifest(run_id='earlier', surface='insights', run_name='earlier', runs_dir=runs, parallelism=10)
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    for name in ('label', 'summary'):
        writer.manifest.stages.append(
            StageRecord(
                name=name,
                status=ManifestStatus.COMPLETED,
                started_at=start,
                ended_at=start + timedelta(seconds=100),
                completed=100,
                total=100,
            )
        )
    writer.flush()
    client = TestClient(build_app())

    slow = client.get('/insights/new/plan', params={**_PLAN_QUERY, 'dimensions': 'intent', 'parallelism': '10'})
    fast = client.get('/insights/new/plan', params={**_PLAN_QUERY, 'dimensions': 'intent', 'parallelism': '100'})

    assert 'estimated after your first run' not in slow.text
    assert 'about 3 min' in slow.text
    assert 'about 20 s' in fast.text


def test_compact_plan_is_one_line_without_the_stage_list(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    _patch_estimate_inputs(monkeypatch, price=ModelInfo(0.01, 0.03, 'acme', False))
    client = TestClient(build_app())

    response = client.get('/insights/new/plan', params={**_PLAN_QUERY, 'compact': '1'})

    assert response.status_code == 200
    assert 'up to 100 traces' in response.text
    assert 'Expected stages' not in response.text
    assert '<table' not in response.text


def test_compact_plan_reports_an_incomplete_form_without_an_error_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    client = TestClient(build_app())

    response = client.get('/insights/new/plan', params={'source': 'file', 'dimensions': 'intent', 'compact': '1'})

    assert response.status_code == 200
    assert 'Browse to choose a trace file first' in response.text


def test_a_finder_export_gives_an_exact_trace_count(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    _patch_estimate_inputs(monkeypatch, price=ModelInfo(0.01, 0.03, 'acme', False))
    exports = get_finder_exports_dir()
    exports.mkdir(parents=True, exist_ok=True)
    export = exports / 'export.json'
    export.write_text(_finder_export(['t1', 't2', 't3']).model_dump_json(), encoding='utf-8')
    client = TestClient(build_app())

    response = client.get(
        '/insights/new/plan', params={**_PLAN_QUERY, 'source': 'file', 'trace_file': str(export), 'compact': '1'}
    )

    assert response.status_code == 200
    assert '3 traces' in response.text
    assert 'up to 3' not in response.text


def _finder_export(matched: list[str]) -> RunExport:
    return RunExport(
        query='refund requests',
        dimensions=(
            ExportDimension(
                name='intent',
                task=ExportTask(kind='choice', instructions='classify', state={}, noul_threshold=0.5),
                selection=ExportValuesSelection(kind='values', values=('refunds',)),
            ),
        ),
        generated_filters=ExportFilters(),
        filters=ExportFilters(),
        generated_numeric=ExportNumericFilters(),
        numeric=ExportNumericFilters(),
        limit=500,
        parallelism=100,
        times=ExportTimes(elapsed=0, rate=0),
        counts=ExportCounts(total=0, completed=0, failed=0, matched=0, active=0, queued=0, percent=0),
        traces=(),
        matched_trace_ids=matched,
    )


def test_start_launches_the_spec_built_from_the_submitted_values(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    client = TestClient(build_app())
    from evaluatorq.dashboard.auth import DashboardAuth

    with (
        patch(
            'evaluatorq.dashboard.insights_routes.selected_dashboard_auth',
            return_value=DashboardAuth('environment', 'key', 'https://my.orq.ai'),
        ),
        patch('evaluatorq.dashboard.insights_routes.launch_insights', return_value='run-9') as launch,
    ):
        response = client.post(
            '/insights/runs',
            data={
                'csrf': _token(client),
                'source': 'orq',
                'window_days': '5',
                'limit': '30',
                'facet_status': 'error',
                'labels': 'made_errors',
                'coding_labels': 'outcome',
                'dimensions': ['intent', 'failure'],
            },
            follow_redirects=False,
        )

    assert response.status_code == 303
    spec = launch.call_args.args[0]
    assert (spec.window_days, spec.limit, spec.labels, spec.coding_labels) == (5, 30, ['made_errors'], ['outcome'])
    assert spec.dimensions == ['intent', 'failure']
    assert spec.facets.status == frozenset({'error'})


def test_review_step_loads_a_picker_per_model_and_the_compiler_only_for_question() -> None:
    recent = render_run_form(RunFormValues.defaults(), csrf='t')
    question = render_run_form(replace(RunFormValues.defaults(), source='query', query='refunds'), csrf='t')

    for field in ('summary_model', 'classifier_model', 'embedding_model'):
        assert f'hx-get="/insights/models?field={field}&amp;{field}=' in recent
        assert 'hx-trigger="load"' in recent
    assert re.search(r'<div class="irf-field" data-compiler-model hidden><label[^>]*>Question compiler', recent)
    assert re.search(r'<div class="irf-field" data-compiler-model><label[^>]*>Question compiler', question)
    assert 'hx-get="/insights/models?field=compiler_model&amp;compiler_model=' in question
    assert set(INSIGHTS_MODEL_FIELDS) == {'summary_model', 'classifier_model', 'embedding_model', 'compiler_model'}


def test_model_defaults_come_from_the_model_roles(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(tmp_path / 'none.json'))
    monkeypatch.setenv('EVALUATORQ_SMART_MODEL', 'acme/smart-9')
    monkeypatch.setenv('EVALUATORQ_CLASSIFIER_MODEL', 'acme/classifier-9')
    monkeypatch.setenv('EVALUATORQ_COMPILER_MODEL', 'acme/compiler-9')

    values = RunFormValues.defaults()

    assert values.summary_model == 'acme/smart-9'
    assert values.embedding_model == BUILTIN['embedding']
    assert (values.classifier_model, values.compiler_model) == ('acme/classifier-9', 'acme/compiler-9')
    assert values.launch_fields()['classifier_model'] == 'acme/classifier-9'


def test_from_run_prefills_all_four_models(tmp_path: Path) -> None:
    run = _saved_run(
        {'mode': 'query', 'query': 'refunds'},
        summary_model='acme/summary',
        classifier_model='acme/classify',
        embedding_model='acme/embed',
        compiler_model='acme/compile',
    )

    values = RunFormValues.from_run(run, tmp_path)

    assert (values.summary_model, values.classifier_model, values.embedding_model, values.compiler_model) == (
        'acme/summary',
        'acme/classify',
        'acme/embed',
        'acme/compile',
    )


def test_from_run_without_a_saved_compiler_uses_the_settings_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(tmp_path / 'none.json'))
    monkeypatch.setenv('EVALUATORQ_COMPILER_MODEL', 'acme/compiler-9')

    assert RunFormValues.from_run(_saved_run({'mode': 'query'}), tmp_path).compiler_model == 'acme/compiler-9'


def test_models_route_loads_the_catalogue_kind_for_the_field(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    client = TestClient(build_app())
    seen: list[str] = []

    async def groups(auth: object, kind: str) -> dict[str, list[str]]:
        seen.append(kind)
        return {'openai': ['openai/text-embedding-3-small', 'openai/text-embedding-3-large']}

    with (
        patch('evaluatorq.dashboard.insights_routes.selected_dashboard_auth', return_value=_AUTH),
        patch('evaluatorq.dashboard.insights_routes.model_groups', groups),
    ):
        embedding = client.get('/insights/models?field=embedding_model&embedding_model=openai/text-embedding-3-large')
        summary = client.get('/insights/models?field=summary_model&summary_model=x/y')
        compiler = client.get('/insights/models?field=compiler_model')
        unknown = client.get('/insights/models?field=apply_model')

    assert seen == ['embedding', 'chat', 'chat']
    assert 'class="model-pick"' in embedding.text
    assert 'name="embedding_model" value="openai/text-embedding-3-large"' in embedding.text
    assert 'aria-label="Summary model (custom)"' in summary.text
    assert 'aria-label="Question compiler (custom)"' in compiler.text
    assert unknown.text == ''


def test_models_route_falls_back_to_a_text_input_without_a_catalogue(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    client = TestClient(build_app())

    async def none(auth: object, kind: str) -> dict[str, list[str]]:
        return {}

    with (
        patch('evaluatorq.dashboard.insights_routes.selected_dashboard_auth', return_value=_AUTH),
        patch('evaluatorq.dashboard.insights_routes.model_groups', none),
    ):
        response = client.get('/insights/models?field=classifier_model&classifier_model=acme/c')

    assert 'model-pick' not in response.text
    assert '<input id="classifier_model" name="classifier_model" type="text" value="acme/c"' in response.text


def _post_models(client: TestClient, **models: str) -> object:
    return client.post(
        '/insights/runs',
        data={'csrf': _token(client), 'source': 'orq', 'labels': 'made_errors', 'dimensions': 'intent', **models},
        follow_redirects=False,
    )


def _info(model_type: str) -> ModelInfo:
    return ModelInfo(None, None, 'acme', False, None, False, model_type)


def test_start_rejects_models_the_catalogue_says_cannot_do_the_job(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    client = TestClient(build_app())
    catalogue = {
        'acme/chat-only': _info('chat'),
        'acme/embedder': _info('embedding'),
        'acme/chat-2': _info('chat'),
    }

    async def entry(auth: object, model: str) -> ModelInfo | None:
        return catalogue.get(model)

    with (
        patch('evaluatorq.dashboard.insights_routes.selected_dashboard_auth', return_value=_AUTH),
        patch('evaluatorq.dashboard.insights_routes.catalogue_entry', entry),
        patch('evaluatorq.dashboard.insights_routes.launch_insights', return_value='run-9') as launch,
    ):
        classifier = _post_models(client, classifier_model='acme/chat-only', summary_model='acme/chat-2')
        embedding = _post_models(client, embedding_model='acme/chat-2')
        accepted = _post_models(client, embedding_model='acme/embedder', summary_model='acme/chat-2')

    assert classifier.status_code == 422
    assert 'acme/chat-only' in classifier.text
    assert 'value="acme/chat-2"' in classifier.text
    assert 'classify' in classifier.text
    assert embedding.status_code == 422
    assert 'embedding' in embedding.text
    assert accepted.status_code == 303
    assert launch.call_count == 1


def test_start_accepts_unchecked_models_and_warns_when_the_catalogue_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from loguru import logger

    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    client = TestClient(build_app())
    messages: list[str] = []
    sink = logger.add(messages.append, level='WARNING')

    async def unavailable(auth: object, model: str) -> None:
        return None

    try:
        with (
            patch('evaluatorq.dashboard.insights_routes.selected_dashboard_auth', return_value=_AUTH),
            patch('evaluatorq.dashboard.insights_routes.catalogue_entry', unavailable),
            patch('evaluatorq.dashboard.insights_routes.launch_insights', return_value='run-9') as launch,
        ):
            response = _post_models(client, classifier_model='acme/typed', embedding_model='acme/typed-embed')
    finally:
        logger.remove(sink)

    assert response.status_code == 303
    assert launch.call_args.args[0].classifier_model == 'acme/typed'
    assert any('classifier_model' in m and 'embedding_model' in m for m in messages)


def test_orq_tab_maps_to_recent_when_blank_and_query_when_asked() -> None:
    assert RunFormValues.from_form({'source': 'orq', 'query': '   '}).source == 'recent'
    asked = RunFormValues.from_form({'source': 'orq', 'query': 'refunds'})
    assert (asked.source, asked.query) == ('query', 'refunds')
    InsightsLaunchSpec.model_validate(RunFormValues.from_form({'source': 'orq', 'dimensions': 'intent'}).launch_fields())


def test_file_tab_takes_the_kind_from_the_stored_upload_name_or_the_exports_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    uploads = tmp_path / 'insights' / '.uploads'
    hex_id = 'a' * 32

    def source(trace_file: str) -> str:
        return RunFormValues.from_form({'source': 'file', 'trace_file': trace_file}).source

    assert source(str(uploads / f'finder-{hex_id}.json')) == 'finder'
    assert source(str(uploads / f'snapshot-{hex_id}.json')) == 'snapshot'
    assert source(str(get_finder_exports_dir() / 'trace-finder-7.json')) == 'finder'
    assert source(str(tmp_path / 'claude-traces.json')) == 'snapshot'
    assert source('') == 'snapshot'
    finder = RunFormValues.from_form({'source': 'file', 'trace_file': str(uploads / f'finder-{hex_id}.json')})
    assert (finder.launch_fields()['finder_export'], finder.launch_fields()['snapshot_path']) == (finder.trace_file, '')


@pytest.mark.parametrize('stale', ['recent', 'query', 'finder', 'snapshot', 'bogus'])
def test_only_the_three_tabs_are_accepted(stale: str) -> None:
    with pytest.raises(ValueError, match='Choose Orq traces, Trace file or Local sessions'):
        RunFormValues.from_form({'source': stale})


def test_fields_of_the_other_tab_are_dropped() -> None:
    on_file = RunFormValues.from_form({
        'source': 'file',
        'trace_file': 's.json',
        'query': 'refunds',
        'facet_agent_name': 'support-bot',
        'compiler_model': 'acme/compiler',
        'window_days': 'stale',
        'limit': '',
    })
    assert (on_file.query, set(on_file.facets.agent_name)) == ('', set())
    assert (on_file.window_days, on_file.limit) == (RunFormValues().window_days, 200)
    assert on_file.compiler_model != 'acme/compiler'
    on_orq = RunFormValues.from_form({'source': 'orq', 'trace_file': 's.json'})
    assert (on_orq.source, on_orq.trace_file) == ('recent', '')


def test_deeply_nested_upload_is_rejected_as_invalid_json() -> None:
    from evaluatorq.dashboard.insights_uploads import validate_upload

    with pytest.raises(ValueError, match='not valid JSON'):
        validate_upload(b'[' * 100_000)


def test_a_missing_source_is_the_orq_tab_and_limit_defaults_to_200() -> None:
    values = RunFormValues.from_form({})
    assert (values.source, values.limit) == ('recent', 200)
    assert RunFormValues().limit == 200
    assert InsightsLaunchSpec().limit == 200


@pytest.mark.parametrize(
    ('population', 'source', 'checked_tab'),
    [
        ({'mode': 'filter'}, 'recent', 'orq'),
        ({'mode': 'query', 'query': 'refunds'}, 'query', 'orq'),
        ({'mode': 'export'}, 'finder', 'file'),
        ({'mode': 'snapshot'}, 'snapshot', 'file'),
    ],
)
def test_rerun_of_each_stored_mode_opens_the_right_tab(
    tmp_path: Path, population: dict[str, object], source: str, checked_tab: str
) -> None:
    values = RunFormValues.from_run(_saved_run(population), tmp_path)

    assert values.source == source
    assert values.limit == 200
    html = render_run_form(values, csrf='t')
    assert f'name="source" value="{checked_tab}" checked' in html
    if source == 'query':
        assert 'refunds</textarea>' in html


def test_sessions_source_launches_as_a_snapshot() -> None:
    values = RunFormValues.from_form({'source': 'sessions', 'trace_file': '/runs/.uploads/snapshot-1.json', 'source_name': '2 local sessions', 'query': 'stale'})

    assert (values.source, values.query) == ('sessions', '')
    fields = values.launch_fields()
    assert fields['source'] == 'snapshot'
    assert fields['snapshot_path'] == '/runs/.uploads/snapshot-1.json'
    html = render_run_form(RunFormValues.defaults(), csrf='t')
    assert 'name="source" value="sessions"' in html
    assert 'data-sessions-search' in html and 'id="insights-sessions"' in html


def test_a_frozen_sessions_file_is_not_reported_as_the_selected_trace_snapshot() -> None:
    path = '/runs/.uploads/snapshot-1.json'
    sessions = render_run_form(
        replace(RunFormValues.defaults(), source='sessions', trace_file=path, source_name='2 local sessions'), csrf='t'
    )
    snapshot = render_run_form(replace(RunFormValues.defaults(), source='snapshot', trace_file=path), csrf='t')

    assert 'Trace snapshot is ready.' not in sessions
    assert 'Using the sessions selected earlier (2 local sessions)' in sessions
    assert 'name="source" value="sessions" checked' in sessions
    assert 'data-kind="snapshot"' in sessions
    assert 'Trace snapshot is ready.' in snapshot


def test_plan_for_a_snapshot_states_what_is_sent_and_to_which_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from evaluatorq.trace_finder.models import Snapshot
    from tests.insights.test_population import make_trace

    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    _patch_estimate_inputs(monkeypatch, price=None)
    path = tmp_path / 'traces.json'
    path.write_text(Snapshot(traces=(make_trace('a'), make_trace('b'))).model_dump_json(), encoding='utf-8')
    client = TestClient(build_app())

    response = client.get(
        '/insights/new/plan',
        params={
            'source': 'sessions',
            'trace_file': str(path),
            'source_name': '2 local sessions',
            'dimensions': 'intent',
            'summary_model': 'acme/s',
            'classifier_model': 'acme/c',
            'embedding_model': 'acme/e',
        },
    )

    assert response.status_code == 200
    assert '2 local sessions from this computer.' in response.text
    assert 'Sending 2 traces (' in response.text
    assert 'from 2 local sessions to models: summary acme/s, classifier acme/c, embedding acme/e.' in response.text
