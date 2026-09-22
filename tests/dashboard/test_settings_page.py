from __future__ import annotations

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.apply_ui import apply_model
from evaluatorq.trace_finder.settings import DashboardSettings, save_settings


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
        data={
            'compiler_model': 'compiler/custom',
            'jev_model': 'jev/custom',
            'apply_model': 'apply/custom',
            'window_days': '14',
            'limit': '42',
            'parallelism': '7',
        },
    )

    assert response.status_code == 303
    assert response.headers['location'] == '/settings?saved=1'
    assert 'compiler/custom' in settings_file.read_text()


def test_invalid_limit_rerenders_form_with_error(client: TestClient) -> None:
    response = client.post(
        '/settings',
        data={
            'compiler_model': 'compiler/custom',
            'jev_model': 'jev/custom',
            'apply_model': 'apply/custom',
            'window_days': '14',
            'limit': '9999',
            'parallelism': '7',
        },
    )

    assert response.status_code == 422
    assert 'limit' in response.text
    assert 'less than or equal to 500' in response.text


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
