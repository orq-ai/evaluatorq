from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from starlette.testclient import TestClient

from evaluatorq.common.orq_client import OrqProfile
from evaluatorq.dashboard import app as app_module
from evaluatorq.dashboard import apply_ui
from evaluatorq.dashboard import finder_routes
from evaluatorq.dashboard.orq_scope import OrqProject, OrqScope
from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.apply_ui import apply_model
from evaluatorq.dashboard.security import CSRF_FIELD, _CSRF_TOKEN
from evaluatorq.trace_finder import FacetCatalogue, RunSnapshot
from evaluatorq.trace_finder.settings import DashboardSettings, save_settings


_MODELS = {'compiler_model': 'compiler/custom', 'classifier_model': 'classifier/custom', 'apply_model': 'apply/custom'}


def csrf_data(values: dict[str, str] | None = None) -> dict[str, str]:
    """Build a settings POST body with the token used by the dashboard security module."""

    return {CSRF_FIELD: _CSRF_TOKEN, **(values or {})}


@pytest.fixture
def settings_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    path = tmp_path / 'dashboard-settings.json'
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(path))
    for name in (
        'EVALUATORQ_COMPILER_MODEL',
        'EVALUATORQ_CLASSIFIER_MODEL',
        'EVALUATORQ_APPLY_MODEL',
        'EVALUATORQ_FINDER_WINDOW_DAYS',
        'EVALUATORQ_FINDER_LIMIT',
        'EVALUATORQ_FINDER_PARALLELISM',
    ):
        monkeypatch.delenv(name, raising=False)
    return path


@pytest.fixture
def client(settings_file: Path, tmp_path: Path) -> TestClient:
    return TestClient(build_app(roots=[tmp_path]), follow_redirects=False)


def test_settings_post_saves_and_redirects(client: TestClient, settings_file: Path) -> None:
    response = client.post(
        '/settings',
        data=csrf_data({
            'compiler_model': 'compiler/custom',
            'classifier_model': 'classifier/custom',
            'apply_model': 'apply/custom',
            'window_days': '14',
            'limit': '42',
            'parallelism': '7',
        }),
    )

    assert response.status_code == 303
    assert response.headers['location'] == '/settings?saved=1'
    assert 'compiler/custom' in settings_file.read_text()


def test_settings_post_requires_csrf_and_same_origin(client: TestClient) -> None:
    values = {
        'compiler_model': 'compiler/custom',
        'classifier_model': 'classifier/custom',
        'apply_model': 'apply/custom',
        'window_days': '14',
        'limit': '42',
        'parallelism': '7',
    }

    assert client.post('/settings', data=values).status_code == 403
    assert client.post('/settings', data=csrf_data({**values, CSRF_FIELD: 'wrong'})).status_code == 403
    assert (
        client.post('/settings', data=csrf_data(values), headers={'sec-fetch-site': 'cross-site'}).status_code == 403
    )


def test_numeric_fields_are_not_taken_from_the_form(client: TestClient, settings_file: Path) -> None:
    response = client.post(
        '/settings',
        data=csrf_data({
            'compiler_model': 'compiler/custom',
            'classifier_model': 'classifier/custom',
            'apply_model': 'apply/custom',
            'limit': '9999',
        }),
    )

    assert response.status_code == 303
    saved = json.loads(settings_file.read_text())
    assert saved['limit'] == DashboardSettings.model_fields['limit'].default


def test_environment_overrides_are_not_persisted_by_save(
    client: TestClient, settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('EVALUATORQ_FINDER_LIMIT', '42')
    response = client.post(
        '/settings',
        data=csrf_data({'compiler_model': 'compiler/custom', 'classifier_model': 'classifier/custom', 'apply_model': 'apply/custom'}),
    )

    assert response.status_code == 303
    saved = json.loads(settings_file.read_text())
    assert saved['limit'] == DashboardSettings.model_fields['limit'].default
    assert getattr(client.app, 'state').finder_settings.limit == 42


def test_unchanged_environment_model_overrides_are_not_persisted(
    client: TestClient, settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('EVALUATORQ_COMPILER_MODEL', 'env/compiler')
    monkeypatch.setenv('EVALUATORQ_CLASSIFIER_MODEL', 'env/classifier')
    monkeypatch.setenv('EVALUATORQ_APPLY_MODEL', 'env/apply')

    response = client.post('/settings', data=csrf_data({
        'compiler_model': 'env/compiler',
        'classifier_model': 'env/classifier',
        'apply_model': 'env/apply',
    }))

    assert response.status_code == 303
    saved = json.loads(settings_file.read_text())
    assert saved['compiler_model'] == DashboardSettings.model_fields['compiler_model'].default
    assert saved['classifier_model'] == DashboardSettings.model_fields['classifier_model'].default
    assert saved['apply_model'] == DashboardSettings.model_fields['apply_model'].default


def test_explicit_model_edit_beats_environment_override(
    client: TestClient, settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('EVALUATORQ_COMPILER_MODEL', 'env/compiler')

    response = client.post('/settings', data=csrf_data({**_MODELS, 'compiler_model': 'new/compiler'}))

    assert response.status_code == 303
    assert json.loads(settings_file.read_text())['compiler_model'] == 'new/compiler'


def test_blank_model_is_rejected(client: TestClient) -> None:
    response = client.post(
        '/settings',
        data=csrf_data({
            'compiler_model': '  ',
            'classifier_model': 'classifier/custom',
            'apply_model': 'apply/custom',
            'window_days': '14',
            'limit': '42',
            'parallelism': '7',
        }),
    )

    assert response.status_code == 422
    assert 'model identifier must not be blank' in response.text


def test_saving_settings_invalidates_initialized_finder_store(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('ORQ_API_KEY', 'test-key')
    models: list[str] = []

    class Store:
        async def snapshot(self) -> RunSnapshot:
            return RunSnapshot()

        async def close(self) -> None:
            return None

    async def build_store(app: Any) -> Store:
        models.append(app.state.finder_settings.compiler_model)
        return Store()

    monkeypatch.setattr(finder_routes, '_build_store', build_store)
    assert client.get('/find').status_code == 200

    response = client.post(
        '/settings',
        data=csrf_data({
            'compiler_model': 'new/compiler',
            'classifier_model': 'classifier/custom',
            'apply_model': 'apply/custom',
            'window_days': '14',
            'limit': '42',
            'parallelism': '7',
        }),
    )
    assert response.status_code == 303
    assert client.get('/find').status_code == 200
    assert models[-1] == 'new/compiler'


def test_dashboard_shutdown_closes_finder_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[bool] = []

    class Store:
        async def snapshot(self) -> RunSnapshot:
            return RunSnapshot()

        async def close(self) -> None:
            closed.append(True)

    async def build_store(_app: Any) -> Store:
        return Store()

    monkeypatch.setenv('ORQ_API_KEY', 'test-key')
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(tmp_path / 'settings.json'))
    monkeypatch.setattr(finder_routes, '_build_store', build_store)
    app = build_app(roots=[tmp_path])
    with TestClient(app) as client:
        assert client.get('/find').status_code == 200

    assert closed == [True]


def test_settings_page_shows_saved_values(client: TestClient, settings_file: Path) -> None:
    save_settings(
        DashboardSettings(
            compiler_model='saved/compiler',
            classifier_model='saved/classifier',
            apply_model='saved/apply',
            window_days=11,
            limit=123,
            parallelism=19,
        ),
        settings_file,
    )

    response = client.get('/settings')

    assert response.status_code == 200
    assert 'value="saved/compiler"' in response.text
    assert 'value="saved/classifier"' in response.text
    assert 'value="saved/apply"' in response.text
    assert 'name="window_days"' not in response.text
    assert 'name="limit"' not in response.text
    assert 'name="parallelism"' not in response.text


def test_saved_confirmation_is_rendered_after_redirect(client: TestClient) -> None:
    response = client.get('/settings?saved=1')

    assert response.status_code == 200
    assert 'Settings saved.' in response.text


def test_project_key_scope_is_saved_and_used_for_trace_search(
    client: TestClient, settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evaluatorq.dashboard.trace_links import trace_span_url
    from evaluatorq.trace_finder.settings import load_settings

    scope = OrqScope('orq-research', 'research-id', (OrqProject('project-bauke', 'Bauke', 'research-id'),))
    monkeypatch.setattr(app_module, 'discover_orq_scope', lambda _profile: scope)
    monkeypatch.delenv('ORQ_WORKSPACE', raising=False)
    monkeypatch.delenv('ORQ_WORKSPACE_SLUG', raising=False)

    page = client.get('/settings').text
    assert '<option value="project-bauke" selected>Bauke' in page
    response = client.post('/settings', data=csrf_data({
        **_MODELS,
        'orq_workspace': 'orq-research',
        'orq_project_id': 'project-bauke',
    }))

    assert response.status_code == 303
    saved = load_settings(settings_file)
    assert (saved.orq_workspace, saved.orq_project_id, saved.orq_project_name) == (
        'orq-research', 'project-bauke', 'Bauke'
    )
    assert '/orq-research/traces?' in (trace_span_url('trace-1', 'span-1') or '')
    run = finder_routes._run_request({'query': 'Frustrated customers'}, saved)
    assert run.population.facets.project_id == 'project-bauke'
    assert '<b>Project</b><a href="/settings">Bauke</a>' in client.get('/find').text


def test_settings_rejects_project_outside_selected_key(
    client: TestClient, settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scope = OrqScope('orq-research', 'research-id', (OrqProject('project-bauke', 'Bauke', 'research-id'),))
    monkeypatch.setattr(app_module, 'discover_orq_scope', lambda _profile: scope)

    response = client.post('/settings', data=csrf_data({
        **_MODELS,
        'orq_workspace': 'orq-research',
        'orq_project_id': 'project-other',
    }))

    assert response.status_code == 422
    assert 'not available to the selected credential' in response.text
    assert not settings_file.exists()


def test_settings_rejects_workspace_outside_selected_key(
    client: TestClient, settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scope = OrqScope('orq-research', 'research-id', (OrqProject('project-bauke', 'Bauke', 'research-id'),))
    monkeypatch.setattr(app_module, 'discover_orq_scope', lambda _profile: scope)

    response = client.post('/settings', data=csrf_data({
        **_MODELS, 'orq_workspace': 'other-workspace', 'orq_project_id': 'project-bauke',
    }))

    assert response.status_code == 422
    assert 'does not match the selected credential' in response.text
    assert not settings_file.exists()


def test_apply_model_uses_saved_setting_then_environment(
    monkeypatch: pytest.MonkeyPatch, settings_file: Path
) -> None:
    save_settings(
        DashboardSettings.model_validate(
            {
                'apply_model': 'saved/apply',
            }
        ),
        settings_file,
    )

    assert apply_model() == 'saved/apply'

    monkeypatch.setenv('EVALUATORQ_APPLY_MODEL', 'env/apply')
    assert apply_model() == 'env/apply'


def _profiles() -> tuple[OrqProfile, ...]:
    return (
        OrqProfile('staging', 'key-staging', 'https://staging.orq.ai', False),
        OrqProfile('prod', 'key-prod', None, True),
    )


def test_settings_page_hides_the_profile_selector_without_cli_profiles(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, 'list_orq_profiles', lambda: ())
    html = client.get('/settings').text
    assert 'name="orq_profile"' not in html
    assert 'name="orq_workspace"' in html


def test_settings_page_offers_cli_profiles_under_advanced(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, 'list_orq_profiles', _profiles)
    html = client.get('/settings').text
    assert '<summary>Advanced</summary>' in html
    assert '<option value="">Environment (ORQ_API_KEY)</option>' in html
    assert '<option value="staging">staging (https://staging.orq.ai)</option>' in html
    assert '<option value="prod">prod</option>' in html
    assert 'key-staging' not in html
    assert '<details class="settings-advanced" open>' not in html


def test_selecting_another_profile_refreshes_its_projects_before_save(
    client: TestClient, settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_settings(DashboardSettings.model_validate({'orq_workspace': 'orq-research', 'orq_project_id': 'project-bauke'}), settings_file)
    monkeypatch.setattr(app_module, 'list_orq_profiles', _profiles)
    monkeypatch.setattr(
        app_module,
        'discover_orq_scope',
        lambda profile: OrqScope(
            'staging-workspace', 'staging-id', (OrqProject('project-staging', 'Staging', 'staging-id'),)
        ) if profile == 'staging' else OrqScope(),
    )

    html = client.get('/settings?profile=staging').text

    assert 'Profile preview. Choose a project, then Save to apply.' in html
    assert '<option value="staging" selected>staging' in html
    assert 'https://staging.orq.ai' in html
    assert 'Selected profile API key' in html
    assert 'name="orq_workspace"' in html
    assert '<option value="project-staging" selected>Staging' in html
    assert 'project-bauke' not in html
    assert json.loads(settings_file.read_text())['orq_project_id'] == 'project-bauke'


def test_profile_without_a_resolved_slug_does_not_inherit_environment_workspace(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, 'list_orq_profiles', _profiles)
    monkeypatch.setattr(
        app_module,
        'discover_orq_scope',
        lambda _profile: OrqScope(None, 'staging-id', (OrqProject('project-staging', 'Staging', 'staging-id'),)),
    )
    monkeypatch.setenv('ORQ_WORKSPACE', 'orq-research')

    html = client.get('/settings?profile=staging').text

    assert 'name="orq_workspace" type="text" value=""' in html
    assert 'could not verify this workspace slug' in html


def test_saving_a_profile_persists_it_without_changing_environment(
    client: TestClient, settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, 'list_orq_profiles', _profiles)
    monkeypatch.setenv('ORQ_API_KEY', 'from-env')
    monkeypatch.setenv('ORQ_BASE_URL', 'https://env.orq.ai')

    response = client.post('/settings', data=csrf_data({**_MODELS, 'orq_profile': 'prod'}))

    assert response.status_code == 303
    assert json.loads(settings_file.read_text())['orq_profile'] == 'prod'
    assert getattr(client.app, 'state').finder_profile == _profiles()[1]
    assert os.environ['ORQ_API_KEY'] == 'from-env'
    assert os.environ['ORQ_BASE_URL'] == 'https://env.orq.ai'
    html = client.get('/settings').text
    assert '<option value="prod" selected>prod</option>' in html
    assert '<details class="settings-advanced" open>' in html
    assert 'Selected profile API key' in html
    assert 'key-prod' not in html
    assert 'Selected profile host' in html


def test_saving_a_profile_with_a_server_keeps_it_in_app_state(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.dashboard.trace_links import trace_span_url
    from evaluatorq.trace_finder.settings import credential_fingerprint, load_settings

    monkeypatch.setattr(app_module, 'list_orq_profiles', _profiles)
    monkeypatch.setenv('ORQ_BASE_URL', 'https://environment.orq.ai')

    assert client.post('/settings', data=csrf_data({
        **_MODELS, 'orq_profile': 'staging', 'orq_workspace': 'staging-workspace',
    })).status_code == 303
    assert getattr(client.app, 'state').finder_profile == _profiles()[0]
    assert os.environ['ORQ_BASE_URL'] == 'https://environment.orq.ai'
    assert load_settings().orq_profile_host == 'https://staging.orq.ai'
    assert load_settings().orq_credential_fingerprint == credential_fingerprint('key-staging', 'https://staging.orq.ai')
    assert (trace_span_url('trace-1', 'span-1') or '').startswith('https://staging.orq.ai/')


def test_blank_profile_means_the_environment(
    client: TestClient, settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, 'list_orq_profiles', _profiles)
    monkeypatch.setenv('ORQ_API_KEY', 'from-env')

    assert client.post('/settings', data=csrf_data({**_MODELS, 'orq_profile': ''})).status_code == 303
    assert json.loads(settings_file.read_text())['orq_profile'] is None
    assert os.environ['ORQ_API_KEY'] == 'from-env'


def test_switching_back_to_environment_clears_app_profile(
    client: TestClient, settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, 'list_orq_profiles', _profiles)
    monkeypatch.setenv('ORQ_API_KEY', 'from-env')
    monkeypatch.setenv('ORQ_BASE_URL', 'https://env.orq.ai')

    assert client.post('/settings', data=csrf_data({**_MODELS, 'orq_profile': 'staging'})).status_code == 303
    assert client.post('/settings', data=csrf_data({**_MODELS, 'orq_profile': ''})).status_code == 303

    assert json.loads(settings_file.read_text())['orq_profile'] is None
    assert getattr(client.app, 'state').finder_profile is None
    assert os.environ['ORQ_API_KEY'] == 'from-env'
    assert os.environ['ORQ_BASE_URL'] == 'https://env.orq.ai'


def test_missing_profile_selector_preserves_saved_choice(
    client: TestClient, settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_settings(DashboardSettings.model_validate({'orq_profile': 'prod'}), settings_file)
    monkeypatch.setattr(app_module, 'list_orq_profiles', lambda: ())
    monkeypatch.setenv('ORQ_API_KEY', 'different-account')

    response = client.post('/settings', data=csrf_data(_MODELS))

    assert response.status_code == 303
    assert json.loads(settings_file.read_text())['orq_profile'] == 'prod'
    app = client.app
    assert finder_routes._api_available(app) is False
    assert asyncio.run(finder_routes._build_store(app)) is None
    assert 'Orq profile prod is unavailable' in client.get('/find').text
    html = client.get('/settings').text
    assert '<option value="prod" selected disabled>prod (unavailable)</option>' in html
    assert '<option value="">Environment (ORQ_API_KEY)</option>' in html

    assert client.post('/settings', data=csrf_data({**_MODELS, 'orq_profile': ''})).status_code == 303
    assert finder_routes._api_available(app) is True


def test_unknown_profile_is_rejected(client: TestClient, settings_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, 'list_orq_profiles', _profiles)

    response = client.post('/settings', data=csrf_data({**_MODELS, 'orq_profile': 'ghost'}))

    assert response.status_code == 422
    assert 'The orq CLI does not know this profile' in response.text
    assert not settings_file.exists()


def test_saved_profile_is_applied_when_the_finder_first_loads_settings(
    settings_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reload worker starts with a fresh environment; the saved profile must win again."""
    save_settings(DashboardSettings.model_validate({'orq_profile': 'prod'}), settings_file)
    monkeypatch.setattr(finder_routes, 'list_orq_profiles', _profiles)

    app = build_app(roots=[tmp_path])
    finder_routes._settings(app)
    finder_routes._settings(app)

    assert app.state.finder_profile == _profiles()[1]


def test_finder_requests_do_not_run_profile_discovery(
    settings_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_settings(DashboardSettings.model_validate({'orq_profile': 'prod'}), settings_file)
    calls: list[str] = []

    def profiles() -> tuple[OrqProfile, ...]:
        calls.append('discover')
        return _profiles()

    monkeypatch.setattr(finder_routes, 'list_orq_profiles', profiles)
    app = build_app(roots=[tmp_path])
    assert calls == ['discover']
    monkeypatch.setattr(finder_routes, 'list_orq_profiles', lambda: pytest.fail('discovery ran in request'))
    async def unavailable_store(_app: Any) -> None:
        return None

    monkeypatch.setattr(finder_routes, '_build_store', unavailable_store)

    assert TestClient(app).get('/find').status_code == 200
    assert calls == ['discover']


@pytest.mark.asyncio
async def test_stale_facet_load_cannot_restore_cache_after_settings_change(
    settings_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = build_app(roots=[tmp_path])
    started = asyncio.Event()
    release = asyncio.Event()

    async def load(*args: object, **kwargs: object) -> FacetCatalogue:
        started.set()
        await release.wait()
        return FacetCatalogue(project=('old-profile',))

    monkeypatch.setattr(finder_routes, 'resolve_orq_client', lambda *args, **kwargs: object())
    monkeypatch.setattr(finder_routes, 'load_facet_catalogue', load)
    task = asyncio.create_task(finder_routes._load_catalogue(app))
    await started.wait()
    app.state.finder_generation += 1
    release.set()

    assert await task is None
    assert not hasattr(app.state, 'finder_catalogue_cache')


@pytest.mark.asyncio
@pytest.mark.parametrize('fail', [False, True])
async def test_facet_refresh_closes_its_temporary_orq_client(
    settings_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    app = build_app(roots=[tmp_path])
    closed: list[str] = []

    class FakeOrq:
        async def __aexit__(self, *_args: object) -> None:
            closed.append('async')

        def __exit__(self, *_args: object) -> None:
            closed.append('sync')

    async def load(*args: object, **kwargs: object) -> FacetCatalogue:
        if fail:
            raise RuntimeError('facet service failed')
        return FacetCatalogue(project=('project',))

    monkeypatch.setattr(finder_routes, 'resolve_orq_client', lambda *args, **kwargs: FakeOrq())
    monkeypatch.setattr(finder_routes, 'load_facet_catalogue', load)

    result = await finder_routes._load_catalogue(app)

    assert (result is None) is fail
    assert closed == ['async', 'sync']


@pytest.mark.asyncio
@pytest.mark.parametrize('owned', [True, False])
async def test_retired_finder_closes_only_clients_it_owns(
    settings_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, owned: bool
) -> None:
    app = build_app(roots=[tmp_path])
    closed: list[str] = []

    class FakeOrq:
        async def __aexit__(self, *_args: object) -> None:
            closed.append('orq-async')

        def __exit__(self, *_args: object) -> None:
            closed.append('orq-sync')

    class FakeLLM:
        async def close(self) -> None:
            closed.append('llm')

    monkeypatch.setattr(
        finder_routes, 'resolve_llm_client', lambda **kwargs: SimpleNamespace(client=FakeLLM(), owned=owned)
    )
    monkeypatch.setattr(finder_routes, 'resolve_orq_client', lambda *args, **kwargs: FakeOrq())
    store = await finder_routes._build_store(app)
    assert store is not None

    await store.close()
    await store.close()

    assert closed == ['orq-async', 'orq-sync', *(['llm'] if owned else [])]


@pytest.mark.asyncio
async def test_finder_closes_owned_llm_when_orq_client_resolution_fails(
    settings_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = build_app(roots=[tmp_path])
    closed: list[bool] = []

    class FakeLLM:
        async def close(self) -> None:
            closed.append(True)

    def missing_orq(*args: object, **kwargs: object) -> object:
        raise ImportError('orq-ai-sdk is unavailable')

    monkeypatch.setattr(finder_routes, 'resolve_llm_client', lambda **kwargs: SimpleNamespace(client=FakeLLM(), owned=True))
    monkeypatch.setattr(finder_routes, 'resolve_orq_client', missing_orq)

    assert await finder_routes._build_store(app) is None
    assert closed == [True]


@pytest.mark.parametrize(
    ('profile_name', 'expected_host'),
    [('staging', 'https://staging.orq.ai'), ('prod', 'https://my.orq.ai')],
)
def test_finder_builds_clients_with_app_profile_not_environment(
    settings_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile_name: str, expected_host: str
) -> None:
    save_settings(DashboardSettings.model_validate({'orq_profile': profile_name}), settings_file)
    monkeypatch.setattr(finder_routes, 'list_orq_profiles', _profiles)
    monkeypatch.setenv('ORQ_API_KEY', 'from-env')
    monkeypatch.setenv('ORQ_BASE_URL', 'https://env.orq.ai')
    app = build_app(roots=[tmp_path])
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_llm(**kwargs: object) -> SimpleNamespace:
        calls.append(('llm', kwargs))
        return SimpleNamespace(client=object())

    def fake_orq(api_key: str | None = None, **kwargs: object) -> object:
        calls.append(('orq', {'api_key': api_key, **kwargs}))
        return object()

    monkeypatch.setattr(finder_routes, 'resolve_llm_client', fake_llm)
    monkeypatch.setattr(finder_routes, 'resolve_orq_client', fake_orq)
    monkeypatch.setattr(finder_routes, 'build_run_store', lambda *args, **kwargs: object())

    assert asyncio.run(finder_routes._build_store(app)) is not None
    expected_key = 'key-staging' if profile_name == 'staging' else 'key-prod'
    assert calls[0][1]['extra_api_key'] == expected_key
    assert calls[0][1]['orq_host'] == expected_host
    assert calls[1][1] == {'api_key': expected_key, 'base_url': expected_host}
    assert os.environ['ORQ_API_KEY'] == 'from-env'


def test_apply_clients_use_profile_key_and_host(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    llm_args: list[tuple[object, ...]] = []

    def fake_orq(api_key: str | None = None, **kwargs: object) -> object:
        calls.append(('orq', {'api_key': api_key, **kwargs}))
        return object()

    def fake_llm(*args: object, **kwargs: object) -> SimpleNamespace:
        llm_args.append(args)
        calls.append(('llm', kwargs))
        return SimpleNamespace(client=object())

    monkeypatch.setattr(apply_ui, 'resolve_orq_client', fake_orq)
    monkeypatch.setattr(apply_ui, 'resolve_llm_client', fake_llm)

    apply_ui._build_clients(_profiles()[0])

    assert calls[0][1] == {'api_key': 'key-staging', 'base_url': 'https://staging.orq.ai'}
    assert calls[1][1]['extra_api_key'] == 'key-staging'
    assert calls[1][1]['orq_host'] == 'https://staging.orq.ai'
    assert llm_args == [(None,)]


@pytest.mark.asyncio
async def test_concurrent_finder_requests_construct_one_store(
    settings_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = build_app(roots=[tmp_path])
    calls = 0
    entered = asyncio.Event()
    release = asyncio.Event()
    store = object()

    async def build_store(_app: Any) -> object:
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return store

    monkeypatch.setattr(finder_routes, '_build_store', build_store)
    first = asyncio.create_task(finder_routes._store(app))
    await entered.wait()
    second = asyncio.create_task(finder_routes._store(app))
    release.set()

    assert await asyncio.gather(first, second) == [store, store]
    assert calls == 1


def test_missing_optional_orq_sdk_leaves_finder_unavailable(
    settings_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = build_app(roots=[tmp_path])

    def missing_sdk(*args: object, **kwargs: object) -> object:
        raise ImportError('orq-ai-sdk is not installed')

    monkeypatch.setattr(finder_routes, 'resolve_llm_client', missing_sdk)
    monkeypatch.setattr(finder_routes, 'resolve_orq_client', missing_sdk)

    assert asyncio.run(finder_routes._build_store(app)) is None
    assert asyncio.run(finder_routes._load_catalogue(app)) is None


def test_facet_menu_degrades_when_provider_fails(
    settings_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = build_app(roots=[tmp_path])
    monkeypatch.setattr(finder_routes, 'resolve_orq_client', lambda *args, **kwargs: object())

    async def provider_failure(*args: object, **kwargs: object) -> object:
        raise RuntimeError('provider timed out')

    monkeypatch.setattr(finder_routes, 'load_facet_catalogue', provider_failure)

    assert asyncio.run(finder_routes._load_catalogue(app)) is None
    assert app.state.finder_catalogue_cache[2] is None
