from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from evaluatorq.dashboard import finder_routes
from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.apply_ui import apply_model
from evaluatorq.dashboard.security import CSRF_FIELD, _CSRF_TOKEN
from evaluatorq.trace_finder import RunSnapshot
from evaluatorq.trace_finder.settings import DashboardSettings, save_settings


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


def test_invalid_limit_rerenders_form_with_error(client: TestClient) -> None:
    response = client.post(
        '/settings',
        data=csrf_data({
            'compiler_model': 'compiler/custom',
            'jev_model': 'jev/custom',
            'apply_model': 'apply/custom',
            'window_days': '14',
            'limit': '9999',
            'parallelism': '7',
        }),
    )

    assert response.status_code == 422
    assert 'limit' in response.text
    assert 'less than or equal to 500' in response.text


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

        async def cancel(self) -> None:
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
    assert 'value="11"' in response.text
    assert 'value="123"' in response.text
    assert 'value="19"' in response.text


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
