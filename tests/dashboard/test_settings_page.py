from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from starlette.requests import Request
from starlette.testclient import TestClient

from evaluatorq.common.orq_client import OrqProfile
from evaluatorq.dashboard import app as app_module
from evaluatorq.dashboard import apply_ui
from evaluatorq.dashboard.trace_finder import routes as finder_routes
from evaluatorq.dashboard.trace_finder.search_views import search_page_html
from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.apply_ui import apply_model
from evaluatorq.dashboard.security import CSRF_FIELD, _CSRF_TOKEN
from evaluatorq.trace_finder import FacetCatalogue, RunSnapshot
from evaluatorq.trace_finder.settings import DashboardSettings, load_settings, save_settings


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
            'ask_ai_mode': 'review',
        }),
    )

    assert response.status_code == 303
    assert response.headers['location'] == '/settings?saved=1'
    assert 'compiler/custom' in settings_file.read_text()
    saved = json.loads(settings_file.read_text())
    assert saved['limit'] == DashboardSettings.model_fields['limit'].default
    assert saved['parallelism'] == DashboardSettings.model_fields['parallelism'].default
    assert saved['ask_ai_mode'] == 'review'


def test_ai_limits_are_per_run_trace_search_controls(client: TestClient) -> None:
    settings_html = client.get('/settings').text
    search_html = search_page_html(RunSnapshot(), DashboardSettings.model_validate({}), api_available=False)

    assert 'name="limit"' not in settings_html
    assert 'name="parallelism"' not in settings_html
    assert 'name="limit"' in search_html
    assert 'name="parallelism"' in search_html
    assert '<option value="immediate" selected>Just proceed</option>' in settings_html


def test_settings_ignores_per_run_ai_limits(client: TestClient, settings_file: Path) -> None:
    response = client.post('/settings', data=csrf_data({**_MODELS, 'limit': '5001', 'parallelism': '201'}))

    assert response.status_code == 303
    saved = json.loads(settings_file.read_text())
    assert saved['limit'] == DashboardSettings.model_fields['limit'].default
    assert saved['parallelism'] == DashboardSettings.model_fields['parallelism'].default


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


def test_invalid_hidden_numeric_field_is_ignored(client: TestClient, settings_file: Path) -> None:
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
    assert json.loads(settings_file.read_text())['limit'] == DashboardSettings.model_fields['limit'].default


def test_submitted_settings_keep_saved_explorer_columns() -> None:
    current = DashboardSettings.model_validate({'explorer_columns': ('model', 'cost')})

    values = app_module._submitted_settings_values({}, current)

    assert values['explorer_columns'] == ('model', 'cost')


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


def test_unchanged_environment_numeric_overrides_are_not_persisted(
    client: TestClient, settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('EVALUATORQ_FINDER_LIMIT', '42')
    monkeypatch.setenv('EVALUATORQ_FINDER_PARALLELISM', '9')

    response = client.post('/settings', data=csrf_data({**_MODELS, 'limit': '42', 'parallelism': '9'}))

    assert response.status_code == 303
    saved = json.loads(settings_file.read_text())
    assert saved['limit'] == DashboardSettings.model_fields['limit'].default
    assert saved['parallelism'] == DashboardSettings.model_fields['parallelism'].default


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
    closed: list[str] = []

    class Store:
        explorer = None

        def __init__(self, name: str = 'explorer') -> None:
            self.name = name

        async def snapshot(self) -> RunSnapshot:
            return RunSnapshot()

        async def close(self) -> None:
            closed.append(self.name)

    async def build_store(app: Any) -> Store:
        models.append(app.state.finder_settings.compiler_model)
        return Store(f'store-{len(models)}')

    monkeypatch.setattr(finder_routes, '_build_store', build_store)
    assert client.get('/find').status_code == 200
    assert client.get('/traces').status_code == 200
    second_browser = TestClient(client.app, raise_server_exceptions=True)
    assert second_browser.get('/traces').status_code == 200

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
    assert len(closed) == 3
    assert len(set(closed)) == 3
    state = getattr(client.app, 'state')
    assert not hasattr(state, 'finder_store')
    assert not hasattr(state, 'finder_search_store')
    assert client.get('/find/rows').status_code == 200
    assert models[-1] == 'new/compiler'
    assert client.get('/find').status_code == 200
    assert models[-1] == 'new/compiler'


@pytest.mark.asyncio
async def test_trace_store_created_during_settings_save_uses_new_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(tmp_path / 'settings.json'))
    monkeypatch.delenv('ORQ_API_KEY', raising=False)
    monkeypatch.setattr(app_module, 'list_orq_profiles', lambda: ())
    monkeypatch.setattr(finder_routes, '_api_available', lambda _app: True)
    app = build_app(roots=[tmp_path])
    detached = threading.Event()
    allow_settings_update = threading.Event()
    stale_build_started = threading.Event()
    models: list[str] = []
    stores: list[Any] = []

    class Store:
        explorer = None

        def __init__(self, model: str) -> None:
            self.model = model
            self.closed = False

        async def snapshot(self) -> RunSnapshot:
            return RunSnapshot()

        async def close(self) -> None:
            self.closed = True

    async def build_store(runtime: Any) -> Store:
        model = runtime.state.finder_settings.compiler_model
        models.append(model)
        if detached.is_set():
            stale_build_started.set()
        store = Store(model)
        stores.append(store)
        return store

    monkeypatch.setattr(finder_routes, '_build_store', build_store)
    original_detach = app.state.trace_sessions.detach_all

    async def pause_after_detach() -> list[Any]:
        result = await original_detach()
        detached.set()
        await asyncio.to_thread(allow_settings_update.wait, 10)
        return result

    monkeypatch.setattr(app.state.trace_sessions, 'detach_all', pause_after_detach)
    with TestClient(app, follow_redirects=False) as client:
        first_page = await asyncio.to_thread(client.get, '/traces')
        assert first_page.status_code == 200
        assert models == [app.state.finder_settings.compiler_model]

        save_request = asyncio.create_task(
            asyncio.to_thread(
                client.post,
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
        )
        assert await asyncio.to_thread(detached.wait, 2)
        traces_request = asyncio.create_task(asyncio.to_thread(client.get, '/traces'))
        try:
            await asyncio.sleep(0.05)
            assert not stale_build_started.is_set()
        finally:
            allow_settings_update.set()

        save_response, traces_response = await asyncio.gather(save_request, traces_request)
        assert save_response.status_code == 303, save_response.text.split('role="status">')[-1].split('</p>')[0]
        assert traces_response.status_code == 200
        assert len(models) == 2
        assert models[0] != 'new/compiler'
        assert models[1] == 'new/compiler'
        assert stores[0].closed is True
        assert stores[1].closed is False
        assert (await asyncio.to_thread(client.get, '/traces')).status_code == 200
        assert len(models) == 2


@pytest.mark.asyncio
async def test_closing_finder_stores_continues_after_one_close_fails() -> None:
    closed: list[str] = []

    class Store:
        def __init__(self, name: str, *, fail: bool = False) -> None:
            self.name = name
            self.fail = fail

        async def close(self) -> None:
            closed.append(self.name)
            if self.fail:
                raise RuntimeError(f'{self.name} close failed')

    await app_module._close_finder_stores(Store('broken', fail=True), Store('healthy'), None)

    assert set(closed) == {'broken', 'healthy'}


@pytest.mark.asyncio
async def test_closing_finder_stores_finishes_cleanup_before_reraising_cancellation() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    closed: list[str] = []

    class Store:
        def __init__(self, name: str) -> None:
            self.name = name

        async def close(self) -> None:
            started.set()
            await release.wait()
            closed.append(self.name)

    closing = asyncio.create_task(app_module._close_finder_stores(Store('first'), Store('second')))
    await started.wait()
    closing.cancel()
    await asyncio.sleep(0)
    assert not closing.done()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await closing

    assert set(closed) == {'first', 'second'}


def test_dashboard_shutdown_closes_finder_stores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[str] = []

    class Store:
        explorer = None

        def __init__(self, name: str) -> None:
            self.name = name

        async def snapshot(self) -> RunSnapshot:
            return RunSnapshot()

        async def close(self) -> None:
            closed.append(self.name)

    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(tmp_path / 'settings.json'))
    monkeypatch.setenv('ORQ_API_KEY', 'test-key')
    app = build_app(roots=[tmp_path])
    stores = iter((Store('explorer'), Store('search')))

    async def build_store(_app: Any) -> Store:
        return next(stores)

    monkeypatch.setattr(finder_routes, '_build_store', build_store)
    with TestClient(app) as client:
        assert client.get('/traces').status_code == 200
        assert client.get('/find').status_code == 200

    assert sorted(closed) == ['explorer', 'search']


def test_settings_page_shows_saved_values(client: TestClient, settings_file: Path) -> None:
    settings = DashboardSettings(
        compiler_model='saved/compiler',
        classifier_model='saved/classifier',
        apply_model='saved/apply',
        window_days=11,
        limit=123,
        parallelism=19,
    )
    save_settings(settings, settings_file)

    response = client.get('/settings')

    assert response.status_code == 200
    assert 'value="saved/compiler"' in response.text
    assert 'value="saved/classifier"' in response.text
    assert 'value="saved/apply"' in response.text
    assert 'name="window_days"' not in response.text
    assert 'name="limit"' not in response.text
    assert 'name="parallelism"' not in response.text

    search_html = search_page_html(RunSnapshot(), settings, api_available=False)
    assert 'name="limit" type="number" min="1" max="5000" value="123"' in search_html
    assert 'name="parallelism" type="number" min="1" max="200" value="19"' in search_html


def test_saved_confirmation_is_rendered_after_redirect(client: TestClient) -> None:
    response = client.get('/settings?saved=1')

    assert response.status_code == 200
    assert 'Settings saved.' in response.text


def test_environment_auth_ignores_old_and_submitted_scope(
    settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evaluatorq.dashboard.trace_links import trace_span_url
    from evaluatorq.trace_finder.settings import effective_settings

    save_settings(DashboardSettings.model_validate({
        'orq_workspace': 'old-workspace',
        'orq_project_id': 'old-project',
        'orq_project_name': 'Old project',
    }), settings_file)
    monkeypatch.setenv('ORQ_WORKSPACE', 'environment-workspace')
    fresh = TestClient(build_app(roots=[settings_file.parent]), follow_redirects=False)

    page = fresh.get('/settings').text
    assert 'name="orq_workspace"' not in page
    assert 'name="orq_project_id"' not in page
    assert 'old-project' not in fresh.get('/find').text
    assert effective_settings().orq_project_id is None
    response = fresh.post('/settings', data=csrf_data({
        **_MODELS,
        'orq_workspace': 'submitted-workspace',
        'orq_project_id': 'submitted-project',
    }))

    assert response.status_code == 303
    saved = load_settings(settings_file)
    assert (saved.orq_workspace, saved.orq_project_id, saved.orq_project_name) == (None, None, None)
    assert '/environment-workspace/traces/(trace:trace-1//span:span-1)' in (trace_span_url('trace-1', 'span-1') or '')
    run = finder_routes._run_request({'query': 'Frustrated customers'}, effective_settings())
    assert run.population.facets.project_id is None


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


def test_settings_page_shows_environment_authentication_without_cli_profiles(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, 'list_orq_profiles', lambda: ())
    html = client.get('/settings').text
    assert '<div class="panel-title">Authentication</div>' in html
    assert 'name="orq_profile"' in html
    assert 'name="orq_auth_method" value="environment" checked' in html
    assert 'name="orq_auth_method" value="cli_oauth"' in html
    assert 'name="orq_auth_method" value="stored_api_key"' in html
    assert 'name="orq_workspace"' not in html
    assert 'name="orq_project_id"' not in html


def test_settings_page_offers_cli_profiles_in_authentication(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, 'list_orq_profiles', _profiles)
    html = client.get('/settings').text
    assert '<div class="panel-title">Authentication</div>' in html
    assert '<summary>Advanced</summary>' not in html
    assert 'name="orq_auth_method" value="environment" checked' in html
    assert 'name="orq_auth_method" value="cli_profile"' in html
    assert '<option value="staging">staging (https://staging.orq.ai)</option>' in html
    assert '<option value="prod">prod</option>' in html
    assert 'key-staging' not in html


def test_authentication_choices_keep_method_fields_in_a_shared_second_step(client: TestClient) -> None:
    html = client.get('/settings').text
    choices = html.split('<div class="settings-auth-choices"', 1)[1].split('<div class="settings-auth-config">', 1)[0]
    details = html.split('<div class="settings-auth-config">', 1)[1].split('</form>', 1)[0]

    assert choices.count('class="settings-auth-choice"') == 4
    assert 'name="orq_profile"' not in choices
    assert 'name="orq_api_key_entry"' not in choices
    assert 'data-auth-method="cli_profile"' in details
    assert 'data-auth-method="cli_oauth"' in details
    assert 'data-auth-method="stored_api_key"' in details


def test_entered_api_key_is_encrypted_and_selected_method_survives_reload(
    client: TestClient, settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cryptography.fernet import Fernet

    from evaluatorq.trace_finder.secure_credentials import _encryption_key
    from evaluatorq.trace_finder.settings import load_settings, read_stored_api_key

    monkeypatch.setenv('EVALUATORQ_DASHBOARD_KEY_ENCRYPTION_KEY', Fernet.generate_key().decode())
    _encryption_key.cache_clear()
    response = client.post('/settings', data=csrf_data({
        **_MODELS,
        'orq_auth_method': 'stored_api_key',
        'orq_api_key_entry': 'entered-secret-for-test',
        'orq_stored_key_host': 'https://my.orq.ai',
    }))

    assert response.status_code == 303
    raw = settings_file.read_text()
    assert 'entered-secret-for-test' not in raw
    saved = load_settings(settings_file)
    assert saved.orq_auth_method == 'stored_api_key'
    assert read_stored_api_key(saved) == 'entered-secret-for-test'
    assert 'name="orq_auth_method" value="stored_api_key" checked' in TestClient(build_app(roots=[])).get('/settings').text
    _encryption_key.cache_clear()


def test_missing_environment_key_returns_action_for_startup_toast(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv('ORQ_API_KEY', raising=False)
    response = client.get('/auth/status')
    assert response.json()['status'] == 'action'
    assert 'ORQ_API_KEY is not set' in response.json()['message']
    assert 'id="eq-auth-toast"' in client.get('/').text


def test_rejected_key_returns_safe_startup_toast_message(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Unauthorized(Exception):
        status_code = 401

    def rejected(*_args: object, **_kwargs: object) -> object:
        raise Unauthorized('provider response includes a private value')

    monkeypatch.setenv('ORQ_API_KEY', 'never-show-this-key')
    monkeypatch.setattr(app_module, 'build_auth_clients', rejected)
    response = client.get('/auth/status')

    assert response.json()['status'] == 'action'
    assert 'was rejected' in response.json()['message']
    assert 'never-show-this-key' not in response.text
    assert 'private value' not in response.text


def test_unexpected_auth_status_failure_is_distinguished_from_outage(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('ORQ_API_KEY', 'test-key')
    monkeypatch.setattr(
        app_module,
        'build_auth_clients',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError('programming defect')),
    )

    response = client.get('/auth/status')

    assert response.status_code == 200
    assert response.json()['status'] == 'error'
    assert 'Check the dashboard logs' in response.json()['message']


def test_missing_cli_oauth_sign_in_returns_action_for_startup_toast(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, settings_file: Path
) -> None:
    save_settings(DashboardSettings.model_validate({'orq_auth_method': 'cli_oauth'}), settings_file)

    def missing_cli(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError('The orq CLI is not installed. Install it and sign in before using CLI OAuth.')

    monkeypatch.setattr(app_module, 'build_auth_clients', missing_cli)
    response = client.get('/auth/status')

    assert response.json()['status'] == 'action'
    assert 'CLI OAuth was rejected' in response.json()['message']


def test_auth_status_cleanup_closes_both_clients_and_preserves_response(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed: list[str] = []

    class Traces:
        async def list_facet_values_async(self, **_kwargs: object) -> None:
            return None

    class Orq:
        traces = Traces()

    class Llm:
        async def close(self) -> None:
            closed.append('llm')
            raise RuntimeError('llm close failed')

    async def close_orq(_orq: object) -> None:
        closed.append('orq')
        raise RuntimeError('orq close failed')

    monkeypatch.setenv('ORQ_API_KEY', 'test-key')
    monkeypatch.setattr(app_module, 'build_auth_clients', lambda *_args, **_kwargs: (Orq(), Llm()))
    monkeypatch.setattr(app_module, 'close_orq_client', close_orq)

    response = client.get('/auth/status')

    assert response.status_code == 200
    assert response.json()['status'] == 'valid'
    assert closed == ['orq', 'llm']


def test_selecting_another_profile_does_not_show_its_scope_before_save(
    client: TestClient, settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_settings(DashboardSettings.model_validate({'orq_workspace': 'orq-research', 'orq_project_id': 'project-bauke'}), settings_file)
    monkeypatch.setattr(app_module, 'list_orq_profiles', _profiles)
    html = client.get('/settings?profile=staging').text

    assert 'Profile preview. Save settings to use it.' in html
    assert '<option value="staging" selected>staging' in html
    assert 'https://staging.orq.ai' in html
    assert 'Selected profile API key' in html
    assert 'onchange="location.assign' not in html
    assert 'settings-auth-scope' not in html
    assert 'project-staging' not in html
    assert 'project-bauke' not in html
    assert json.loads(settings_file.read_text())['orq_project_id'] == 'project-bauke'


def test_saving_profile_clears_legacy_workspace_and_project(
    client: TestClient, settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, 'list_orq_profiles', _profiles)
    save_settings(DashboardSettings.model_validate({
        'orq_auth_method': 'cli_profile', 'orq_profile': 'prod',
        'orq_workspace': 'old-workspace', 'orq_project_id': 'old-project',
        'orq_project_name': 'Old project',
    }), settings_file)

    response = client.post('/settings', data=csrf_data({
        **_MODELS, 'orq_auth_method': 'cli_profile', 'orq_profile': 'staging',
        'orq_workspace': 'other-workspace', 'orq_project_id': 'other-project',
    }))

    assert response.status_code == 303
    saved = load_settings(settings_file)
    assert saved.orq_profile == 'staging'
    assert (saved.orq_workspace, saved.orq_project_id, saved.orq_project_name) == (None, None, None)


def test_saving_environment_auth_clears_saved_cli_profile(
    client: TestClient, settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, 'list_orq_profiles', _profiles)
    save_settings(
        DashboardSettings.model_validate({'orq_auth_method': 'cli_profile', 'orq_profile': 'staging'}), settings_file
    )

    response = client.post('/settings', data=csrf_data({
        **_MODELS,
        'orq_auth_method': 'environment',
    }))

    assert response.status_code == 303
    saved = load_settings(settings_file)
    assert saved.orq_auth_method == 'environment'
    assert saved.orq_profile is None


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
    assert '<div class="panel-title">Authentication</div>' in html
    assert 'Selected profile API key' in html
    assert 'key-prod' not in html
    assert 'Selected profile host' in html
    assert 'Uses the API key and host from this local Orq CLI profile.' in html


def test_saving_a_profile_with_a_server_keeps_it_in_app_state(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.dashboard.trace_links import trace_span_url
    from evaluatorq.trace_finder.settings import credential_fingerprint, load_settings

    monkeypatch.setattr(app_module, 'list_orq_profiles', _profiles)
    monkeypatch.setenv('ORQ_BASE_URL', 'https://environment.orq.ai')
    monkeypatch.setenv('ORQ_WORKSPACE', 'staging-workspace')

    assert client.post('/settings', data=csrf_data({
        **_MODELS, 'orq_profile': 'staging', 'orq_workspace': 'staging-workspace',
    })).status_code == 303
    assert getattr(client.app, 'state').finder_profile == _profiles()[0]
    assert os.environ['ORQ_BASE_URL'] == 'https://environment.orq.ai'
    assert load_settings().orq_profile_host == 'https://staging.orq.ai'
    assert load_settings().orq_workspace is None
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
    getattr(client.app, 'state').insights_facet_catalogues = {7: object()}
    assert client.post('/settings', data=csrf_data({**_MODELS, 'orq_profile': ''})).status_code == 303

    assert json.loads(settings_file.read_text())['orq_profile'] is None
    assert getattr(client.app, 'state').finder_profile is None
    assert not hasattr(getattr(client.app, 'state'), 'insights_facet_catalogues')
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
    assert 'name="orq_auth_method" value="environment"' in html
    assert 'The saved Orq CLI profile “prod” is unavailable.' in html

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


def test_apply_clients_use_resolved_dashboard_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.dashboard.auth import DashboardAuth

    auth = DashboardAuth('stored_api_key', 'saved-key', 'https://staging.orq.ai')
    settings = SimpleNamespace(orq_workspace='workspace-a', orq_project_id='project-a')
    calls: list[tuple[object, dict[str, object]]] = []

    def fake_build(resolved: object, **kwargs: object) -> tuple[object, object]:
        calls.append((resolved, kwargs))
        return object(), object()

    monkeypatch.setattr('evaluatorq.dashboard.auth.build_auth_clients', fake_build)
    apply_ui._build_clients(auth, settings)

    assert calls == [(auth, {'workspace': 'workspace-a', 'project': 'project-a'})]


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
    first = asyncio.create_task(finder_routes._store(app, session_id='session-id'))
    await entered.wait()
    second = asyncio.create_task(finder_routes._store(app, session_id='session-id'))
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


_CHOICES = {'openai': ['openai/gpt-5.6-luna'], 'typesafe': ['typesafe/jev-latest']}


def test_settings_page_renders_model_fields_before_the_catalogue_loads(client: TestClient) -> None:
    html = client.get('/settings').text

    assert '<span hx-get="/settings/models?field=compiler_model&amp;profile=&amp;auth_method=environment" hx-trigger="load"' in html
    assert '<input id="compiler_model" name="compiler_model" type="text"' in html


def test_model_field_offers_workspace_models_grouped_by_provider(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def choices(_client: object, *, classify: bool = False) -> dict[str, list[str]]:
        return _CHOICES

    monkeypatch.setattr(app_module, 'models_by_provider', choices)
    monkeypatch.setenv('ORQ_API_KEY', 'test-key')

    html = client.get('/settings/models', params={'field': 'compiler_model', 'compiler_model': 'openai/gpt-5.6-luna'}).text

    assert '<input type="hidden" name="compiler_model" value="openai/gpt-5.6-luna">' in html
    assert '<button type="button" id="compiler_model" class="model-pick-btn"' in html
    assert '<div class="hd">openai</div>' in html
    assert 'data-model="openai/gpt-5.6-luna" aria-pressed="true">gpt-5.6-luna</button>' in html

    custom = client.get('/settings/models', params={'field': 'classifier_model', 'classifier_model': 'my/finetune'}).text
    # A saved model the workspace does not list opens under Custom, text filled in.
    assert '<input class="model-custom" type="text" value="my/finetune"' in custom


def test_model_field_ignores_a_missing_profile_rather_than_using_the_environment(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def choices(_client: object, *, classify: bool = False) -> dict[str, list[str]]:
        return _CHOICES

    monkeypatch.setattr(app_module, 'models_by_provider', choices)
    monkeypatch.setenv('ORQ_API_KEY', 'test-key')

    html = client.get('/settings/models', params={'field': 'compiler_model', 'profile': 'gone'}).text

    assert 'Custom…' not in html


@pytest.mark.parametrize('method', ['stored_api_key', 'cli_oauth'])
def test_model_field_uses_selected_authentication(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    calls: list[object] = []

    class FakeLLM:
        async def close(self) -> None:
            calls.append('llm closed')

    llm = FakeLLM()

    def selected_auth(settings: DashboardSettings, **_kwargs: object) -> SimpleNamespace:
        calls.append(settings.orq_auth_method)
        return SimpleNamespace(method=settings.orq_auth_method, api_key='selected-key', base_url='https://selected.example')

    async def choices(chosen_client: object, *, classify: bool = False) -> dict[str, list[str]]:
        calls.append((chosen_client, classify))
        return _CHOICES

    async def close_orq(_client: object) -> None:
        calls.append('orq closed')

    monkeypatch.setattr(app_module, 'resolve_dashboard_auth', selected_auth)
    monkeypatch.setattr(app_module, 'models_by_provider', choices)
    monkeypatch.setattr(app_module, 'close_orq_client', close_orq)
    if method == 'cli_oauth':
        monkeypatch.setattr(app_module, 'build_auth_clients', lambda _auth: (object(), llm))
    else:
        def resolve_llm(**kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(client=llm, owned=True)

        monkeypatch.setattr(app_module, 'resolve_llm_client', resolve_llm)

    html = client.get('/settings/models', params={'field': 'compiler_model', 'auth_method': method}).text

    assert 'data-model="openai/gpt-5.6-luna"' in html
    assert method in calls
    assert (llm, False) in calls
    assert 'llm closed' in calls
    if method == 'cli_oauth':
        assert 'orq closed' in calls
    else:
        assert any(isinstance(call, dict) and call.get('extra_api_key') == 'selected-key' for call in calls)


def test_model_menu_submits_its_hidden_value(client: TestClient, settings_file: Path) -> None:
    response = client.post('/settings', data=csrf_data({**_MODELS, 'compiler_model': 'my/own-model'}))

    assert response.status_code == 303
    assert json.loads(settings_file.read_text())['compiler_model'] == 'my/own-model'


def test_model_field_stays_free_text_without_a_catalogue(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('ORQ_API_KEY', raising=False)

    html = client.get('/settings/models', params={'field': 'compiler_model', 'compiler_model': 'x/y'}).text

    assert html.startswith('<input id="compiler_model" name="compiler_model" type="text" value="x/y"')


@pytest.mark.asyncio
async def test_models_by_provider_groups_chat_and_classify_models(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.common import model_catalogue

    payload = [
        {'model_id': 'gpt-5.6-luna', 'provider': 'openai', 'model_developer': 'openai', 'model_type': 'chat', 'input_cost': 1, 'output_cost': 1},
        {'model_id': 'gpt-5.6-luna', 'provider': 'azure', 'model_developer': 'openai', 'model_type': 'chat', 'input_cost': 1, 'output_cost': 1},
        {'model_id': 'text-embedding-4', 'provider': 'openai', 'model_type': 'embedding', 'input_cost': 1, 'output_cost': 0},
        {'model_id': 'z-ai/glm-5.3-flash', 'provider': 'tensorix', 'model_type': 'chat', 'input_cost': 1, 'output_cost': 1, 'metadata': {'supports_classify': True}},
        {'model_id': 'jev-latest', 'provider': 'typesafe', 'model_type': 'classify', 'input_cost': 1, 'output_cost': 1, 'metadata': {'supports_classify': True}},
    ]

    async def load(_client: object = None) -> dict[str, object]:
        return model_catalogue._parse_catalogue(payload)

    monkeypatch.setattr(model_catalogue, '_load_catalogue', load)

    assert await model_catalogue.models_by_provider() == {
        'azure': ['azure/gpt-5.6-luna'],
        'openai': ['openai/gpt-5.6-luna'],
        'tensorix': ['tensorix/z-ai/glm-5.3-flash'],
    }
    assert await model_catalogue.models_by_provider(classify=True) == {
        'tensorix': ['tensorix/z-ai/glm-5.3-flash'],
        'typesafe': ['typesafe/jev-latest'],
    }
