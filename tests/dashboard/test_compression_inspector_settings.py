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
    advanced = html.split('<details class="settings-advanced">', 1)[1].split('</details>', 1)[0]

    assert '<summary>Advanced</summary>' in html
    assert '<details class="settings-advanced" open>' not in html
    assert 'name="compression_inspector_enabled" type="checkbox" value="true"' in advanced
    assert ('value="true" checked>' in advanced) is checked
    assert 'Local previews may expose sensitive original content.' in advanced
    assert 'Previews do not call models.' in advanced


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
