from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluatorq.contracts import DEFAULT_PIPELINE_MODEL
from evaluatorq.trace_finder.settings import (
    DashboardSettings,
    effective_settings,
    load_settings,
    save_settings,
    settings_path,
)


def test_settings_round_trip_uses_json_file(tmp_path: Path) -> None:
    path = tmp_path / 'nested' / 'dashboard-settings.json'
    settings = DashboardSettings(
        compiler_model='compiler/model',
        jev_model='jev/model',
        apply_model='apply/model',
        window_days=14,
        limit=42,
        parallelism=7,
    )

    save_settings(settings, path)

    assert load_settings(path) == settings
    assert json.loads(path.read_text()) == settings.model_dump()


def test_invalid_settings_file_warns_and_returns_defaults(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / 'dashboard-settings.json'
    path.write_text('{not json')

    with caplog.at_level('WARNING'):
        settings = load_settings(path)

    assert settings == DashboardSettings.model_validate({})
    assert 'Could not load dashboard settings' in caplog.text


def test_effective_settings_precedence_is_file_then_environment_then_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / 'dashboard-settings.json'
    save_settings(
        DashboardSettings(
            compiler_model='file/compiler',
            jev_model='file/jev',
            apply_model='file/apply',
            window_days=8,
            limit=80,
            parallelism=8,
        ),
        path,
    )
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(path))
    monkeypatch.setenv('EVALUATORQ_COMPILER_MODEL', 'env/compiler')
    monkeypatch.setenv('EVALUATORQ_JEV_MODEL', 'env/jev')
    monkeypatch.setenv('EVALUATORQ_APPLY_MODEL', 'env/apply')

    settings = effective_settings(
        {
            'compiler_model': 'override/compiler',
            'window_days': 21,
            'limit': None,
        }
    )

    assert settings.compiler_model == 'override/compiler'
    assert settings.jev_model == 'env/jev'
    assert settings.apply_model == 'env/apply'
    assert settings.window_days == 21
    assert settings.limit == 80
    assert settings.parallelism == 8


def test_settings_path_uses_environment_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / 'settings.json'
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(path))

    assert settings_path() == path


def test_default_settings_match_shared_pipeline_default() -> None:
    settings = DashboardSettings.model_validate({})
    assert settings.compiler_model == DEFAULT_PIPELINE_MODEL
    assert settings.apply_model == DEFAULT_PIPELINE_MODEL
