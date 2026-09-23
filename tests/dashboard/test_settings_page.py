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
from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.apply_ui import apply_model
from evaluatorq.dashboard.security import CSRF_FIELD, _CSRF_TOKEN
from evaluatorq.trace_finder import RunSnapshot
from evaluatorq.trace_finder.settings import DashboardSettings, save_settings


_MODELS = {'compiler_model': 'compiler/custom', 'jev_model': 'jev/custom', 'apply_model': 'apply/custom'}


def csrf_data(values: dict[str, str] | None = None) -> dict[str, str]:
    """Build a settings POST body with the token used by the dashboard security module."""

    return {CSRF_FIELD: _CSRF_TOKEN, **(values or {})}


@pytest.fixture
def settings_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    path = tmp_path / 'dashboard-settings.json'
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(path))
    for name in (
        'EVALUATORQ_COMPILER_MODEL',
        'EVALUATORQ_JEV_MODEL',
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
            'jev_model': 'jev/custom',
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
        'jev_model': 'jev/custom',
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
            'jev_model': 'jev/custom',
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
        data=csrf_data({'compiler_model': 'compiler/custom', 'jev_model': 'jev/custom', 'apply_model': 'apply/custom'}),
    )

    assert response.status_code == 303
    saved = json.loads(settings_file.read_text())
    assert saved['limit'] == DashboardSettings.model_fields['limit'].default
    assert getattr(client.app, 'state').finder_settings.limit == 42


def test_unchanged_environment_model_overrides_are_not_persisted(
    client: TestClient, settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('EVALUATORQ_COMPILER_MODEL', 'env/compiler')
    monkeypatch.setenv('EVALUATORQ_JEV_MODEL', 'env/jev')
    monkeypatch.setenv('EVALUATORQ_APPLY_MODEL', 'env/apply')

    response = client.post('/settings', data=csrf_data({
        'compiler_model': 'env/compiler',
        'jev_model': 'env/jev',
        'apply_model': 'env/apply',
    }))

    assert response.status_code == 303
    saved = json.loads(settings_file.read_text())
    assert saved['compiler_model'] == DashboardSettings.model_fields['compiler_model'].default
    assert saved['jev_model'] == DashboardSettings.model_fields['jev_model'].default
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
            'jev_model': 'jev/custom',
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

    def build_store(app: Any) -> Store:
        models.append(app.state.finder_settings.compiler_model)
        return Store()

    monkeypatch.setattr(finder_routes, '_build_store', build_store)
    assert client.get('/find').status_code == 200

    response = client.post(
        '/settings',
        data=csrf_data({
            'compiler_model': 'new/compiler',
            'jev_model': 'jev/custom',
            'apply_model': 'apply/custom',
            'window_days': '14',
            'limit': '42',
            'parallelism': '7',
        }),
    )
    assert response.status_code == 303
    assert client.get('/find').status_code == 200
    assert models[-1] == 'new/compiler'


def test_settings_page_shows_saved_values(client: TestClient, settings_file: Path) -> None:
    save_settings(
        DashboardSettings(
            compiler_model='saved/compiler',
            jev_model='saved/jev',
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
    assert 'value="saved/jev"' in response.text
    assert 'value="saved/apply"' in response.text
    assert 'name="window_days"' not in response.text
    assert 'name="limit"' not in response.text
    assert 'name="parallelism"' not in response.text


def test_saved_confirmation_is_rendered_after_redirect(client: TestClient) -> None:
    response = client.get('/settings?saved=1')

    assert response.status_code == 200
    assert 'Settings saved.' in response.text


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
    assert '<summary>Advanced</summary>' not in html


def test_settings_page_offers_cli_profiles_under_advanced(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, 'list_orq_profiles', _profiles)
    html = client.get('/settings').text
    assert '<summary>Advanced</summary>' in html
    assert '<option value="">Environment (ORQ_API_KEY)</option>' in html
    assert '<option value="staging">staging (https://staging.orq.ai)</option>' in html
    assert '<option value="prod">prod</option>' in html
    assert 'key-staging' not in html
    assert '<details class="settings-advanced" open>' not in html


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
    monkeypatch.setattr(app_module, 'list_orq_profiles', _profiles)
    monkeypatch.delenv('ORQ_BASE_URL', raising=False)

    assert client.post('/settings', data=csrf_data({**_MODELS, 'orq_profile': 'staging'})).status_code == 303
    assert getattr(client.app, 'state').finder_profile == _profiles()[0]
    assert 'ORQ_BASE_URL' not in os.environ


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
    assert finder_routes._build_store(app) is None
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

    assert finder_routes._build_store(app) is not None
    expected_key = 'key-staging' if profile_name == 'staging' else 'key-prod'
    assert calls[0][1]['extra_api_key'] == expected_key
    assert calls[0][1]['orq_host'] == expected_host
    assert calls[1][1] == {'api_key': expected_key, 'server_url': expected_host}
    assert os.environ['ORQ_API_KEY'] == 'from-env'


def test_apply_clients_use_profile_key_and_host(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_orq(api_key: str | None = None, **kwargs: object) -> object:
        calls.append(('orq', {'api_key': api_key, **kwargs}))
        return object()

    def fake_llm(*args: object, **kwargs: object) -> SimpleNamespace:
        calls.append(('llm', kwargs))
        return SimpleNamespace(client=object())

    monkeypatch.setattr(apply_ui, 'resolve_orq_client', fake_orq)
    monkeypatch.setattr(apply_ui, 'resolve_llm_client', fake_llm)

    apply_ui._build_clients(_profiles()[0])

    assert calls[0][1] == {'api_key': 'key-staging', 'server_url': 'https://staging.orq.ai'}
    assert calls[1][1]['extra_api_key'] == 'key-staging'
    assert calls[1][1]['orq_host'] == 'https://staging.orq.ai'


def test_missing_optional_orq_sdk_leaves_finder_unavailable(
    settings_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = build_app(roots=[tmp_path])

    def missing_sdk(*args: object, **kwargs: object) -> object:
        raise ImportError('orq-ai-sdk is not installed')

    monkeypatch.setattr(finder_routes, 'resolve_llm_client', missing_sdk)
    monkeypatch.setattr(finder_routes, 'resolve_orq_client', missing_sdk)

    assert finder_routes._build_store(app) is None
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
