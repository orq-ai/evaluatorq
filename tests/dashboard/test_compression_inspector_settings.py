from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from evaluatorq.dashboard import app as app_module
from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.security import CSRF_FIELD, _CSRF_TOKEN
from evaluatorq.dashboard.view import settings_body
from evaluatorq.trace_finder.settings import DashboardSettings, load_settings


def test_legacy_settings_default_compression_inspector_to_disabled(tmp_path: Path) -> None:
    path = tmp_path / 'dashboard-settings.json'
    path.write_text('{"window_days": 14}')

    assert load_settings(path).compression_inspector_enabled is False


@pytest.mark.parametrize(
    ('settings', 'checked'),
    [
        pytest.param(DashboardSettings.model_validate({'compression_inspector_enabled': True}), True, id='model-on'),
        pytest.param(DashboardSettings.model_validate({'compression_inspector_enabled': False}), False, id='model-off'),
        pytest.param({'compression_inspector_enabled': True}, True, id='dict-on'),
        pytest.param({'compression_inspector_enabled': False}, False, id='dict-off'),
    ],
)
def test_advanced_settings_checkbox_renders_persisted_value(settings: Any, checked: bool) -> None:
    html = settings_body([], settings)
    assert 'Compression inspector' in html
    assert 'id="compression_inspector_enabled"' in html
    assert 'name="compression_inspector_enabled"' in html
    assert 'type="checkbox"' in html
    assert 'value="true"' in html
    assert ('value="true" checked>' in html or 'value="true" checked ' in html) is checked
    assert 'Local previews may expose sensitive original content.' in html
    assert 'Previews do not call models.' in html


def test_posting_settings_toggles_inspector_and_sidebar(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings_path = tmp_path / 'dashboard-settings.json'
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(settings_path))
    monkeypatch.setenv('EVALUATORQ_USER_SETTINGS', str(tmp_path / 'user-settings.json'))
    for name in (
        'EVALUATORQ_FAST_MODEL',
        'EVALUATORQ_SMART_MODEL',
        'EVALUATORQ_CLASSIFIER_MODEL',
        'EVALUATORQ_COMPILER_MODEL',
        'EVALUATORQ_APPLY_MODEL',
        'EVALUATORQ_FINDER_WINDOW_DAYS',
        'EVALUATORQ_FINDER_LIMIT',
        'EVALUATORQ_FINDER_PARALLELISM',
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(app_module, 'list_oauth_sessions', lambda: ())
    client = TestClient(build_app(roots=[tmp_path]), follow_redirects=False)

    enabled = client.post(
        '/settings',
        data={CSRF_FIELD: _CSRF_TOKEN, 'compression_inspector_enabled': 'true'},
    )

    assert enabled.status_code == 303
    assert load_settings(settings_path).compression_inspector_enabled is True
    assert 'href="/compression-inspector"' in client.get('/').text
    inspector = client.get('/compression-inspector')
    assert inspector.status_code == 200
    assert '<a class="nav-item active" href="/compression-inspector">' in inspector.text

    disabled = client.post('/settings', data={CSRF_FIELD: _CSRF_TOKEN})

    assert disabled.status_code == 303
    assert json.loads(settings_path.read_text())['compression_inspector_enabled'] is False
    assert 'href="/compression-inspector"' not in client.get('/').text


def _client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[TestClient, Path]:
    settings_path = tmp_path / 'dashboard-settings.json'
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(settings_path))
    monkeypatch.setenv('EVALUATORQ_USER_SETTINGS', str(tmp_path / 'user-settings.json'))
    monkeypatch.setattr(app_module, 'list_oauth_sessions', lambda: ())
    return TestClient(build_app(roots=[tmp_path]), follow_redirects=False), settings_path


def test_autosave_post_saves_and_returns_the_matching_sidebar_as_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client, settings_path = _client(monkeypatch, tmp_path)

    on = client.post(
        '/settings',
        data={CSRF_FIELD: _CSRF_TOKEN, 'compression_inspector_enabled': 'true'},
        headers={'X-Autosave': '1'},
    )
    assert on.status_code == 200
    assert 'href="/compression-inspector"' in on.json()['sidebar']
    assert load_settings(settings_path).compression_inspector_enabled is True

    off = client.post('/settings', data={CSRF_FIELD: _CSRF_TOKEN}, headers={'X-Autosave': '1'})
    assert 'href="/compression-inspector"' not in off.json()['sidebar']
    assert load_settings(settings_path).compression_inspector_enabled is False


def test_model_only_autosave_preserves_compression_inspector(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client, settings_path = _client(monkeypatch, tmp_path)
    settings_path.write_text(json.dumps({'schema_version': 1, 'compression_inspector_enabled': True}))

    response = client.post(
        '/settings',
        data={CSRF_FIELD: _CSRF_TOKEN, 'smart_model': 'openai/test-smart', 'settings_changed_fields': 'smart_model'},
        headers={'X-Autosave': '1'},
    )

    assert response.status_code == 200
    saved = load_settings(settings_path)
    assert saved.compression_inspector_enabled is True
    assert saved.smart_model == 'openai/test-smart'


def test_autosave_post_reports_validation_errors_as_json_and_saves_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client, settings_path = _client(monkeypatch, tmp_path)

    response = client.post(
        '/settings',
        data={CSRF_FIELD: _CSRF_TOKEN, 'orq_auth_method': 'stored_api_key', 'orq_api_key_entry': ''},
        headers={'X-Autosave': '1'},
    )

    assert response.status_code == 422
    assert response.json()['errors']['orq_api_key_entry']
    assert not settings_path.exists()


def test_autosave_post_without_csrf_token_is_rejected_as_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client, settings_path = _client(monkeypatch, tmp_path)

    response = client.post('/settings', data={'compression_inspector_enabled': 'true'}, headers={'X-Autosave': '1'})

    assert response.status_code == 403
    assert response.json()['errors']['form']
    assert not settings_path.exists()
