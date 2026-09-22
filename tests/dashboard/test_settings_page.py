from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from evaluatorq.common.orq_client import OrqProfile
from evaluatorq.dashboard import app as app_module
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


def test_saving_a_profile_persists_it_and_points_the_environment_at_it(
    client: TestClient, settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, 'list_orq_profiles', _profiles)
    monkeypatch.setenv('ORQ_API_KEY', 'from-env')
    monkeypatch.setenv('ORQ_BASE_URL', 'https://env.orq.ai')

    response = client.post('/settings', data=csrf_data({**_MODELS, 'orq_profile': 'prod'}))

    assert response.status_code == 303
    assert json.loads(settings_file.read_text())['orq_profile'] == 'prod'
    assert os.environ['ORQ_API_KEY'] == 'key-prod'
    assert 'ORQ_BASE_URL' not in os.environ, 'a profile without a server means the default host'
    html = client.get('/settings').text
    assert '<option value="prod" selected>prod</option>' in html
    assert '<details class="settings-advanced" open>' in html


def test_saving_a_profile_with_a_server_sets_the_base_url(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, 'list_orq_profiles', _profiles)
    monkeypatch.delenv('ORQ_BASE_URL', raising=False)

    assert client.post('/settings', data=csrf_data({**_MODELS, 'orq_profile': 'staging'})).status_code == 303
    assert os.environ['ORQ_API_KEY'] == 'key-staging'
    assert os.environ['ORQ_BASE_URL'] == 'https://staging.orq.ai'


def test_blank_profile_means_the_environment(
    client: TestClient, settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, 'list_orq_profiles', _profiles)
    monkeypatch.setenv('ORQ_API_KEY', 'from-env')

    assert client.post('/settings', data=csrf_data({**_MODELS, 'orq_profile': ''})).status_code == 303
    assert json.loads(settings_file.read_text())['orq_profile'] is None
    assert os.environ['ORQ_API_KEY'] == 'from-env'


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
    applied: list[str] = []
    monkeypatch.setattr(finder_routes, 'apply_orq_profile', lambda name: applied.append(name))

    app = build_app(roots=[tmp_path])
    finder_routes._settings(app)
    finder_routes._settings(app)

    assert applied == ['prod']
