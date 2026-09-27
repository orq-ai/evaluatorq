from __future__ import annotations

import json
from pathlib import Path

import pytest
from loguru import logger

from evaluatorq.contracts import DEFAULT_PIPELINE_MODEL
from evaluatorq.trace_finder.settings import (
    DashboardSettings,
    effective_settings,
    load_settings,
    read_stored_api_key,
    save_settings,
    settings_path,
    store_api_key,
)
from evaluatorq.trace_finder import secure_credentials


def test_settings_round_trip_uses_json_file(tmp_path: Path) -> None:
    path = tmp_path / 'nested' / 'dashboard-settings.json'
    settings = DashboardSettings(
        compiler_model='compiler/model',
        classifier_model='classifier/model',
        apply_model='apply/model',
        window_days=14,
        limit=42,
        parallelism=7,
    )

    save_settings(settings, path)

    assert load_settings(path) == settings
    assert json.loads(path.read_text()) == settings.model_dump()


def test_save_keeps_original_error_if_temporary_cleanup_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fail_replace(self: Path, target: Path) -> None:
        raise OSError('replace failed')

    def fail_unlink(self: Path, *, missing_ok: bool = False) -> None:
        raise OSError('cleanup failed')

    monkeypatch.setattr(Path, 'replace', fail_replace)
    monkeypatch.setattr(Path, 'unlink', fail_unlink)
    with pytest.raises(OSError, match='replace failed'):
        save_settings(DashboardSettings.model_validate({}), tmp_path / 'settings.json')


def test_unknown_settings_override_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(tmp_path / 'missing.json'))

    with pytest.raises(ValueError, match='extra_forbidden'):
        effective_settings({'limt': 10})


def test_invalid_settings_file_warns_and_returns_defaults(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / 'dashboard-settings.json'
    path.write_text('{not json')

    with caplog.at_level('WARNING'):
        settings = load_settings(path)

    assert settings == DashboardSettings.model_validate({})
    assert 'Could not load dashboard settings' in caplog.text


def test_unreadable_settings_file_warns_with_path_and_returns_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / 'dashboard-settings.json'

    def raise_unreadable(*args: object, **kwargs: object) -> str:
        raise OSError('permission denied')

    monkeypatch.setattr(Path, 'read_text', raise_unreadable)
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message.record['message']), level='WARNING')
    try:
        settings = load_settings(path)
    finally:
        logger.remove(sink_id)

    assert settings == DashboardSettings.model_validate({})
    assert any(str(path) in message for message in messages)


def test_effective_settings_precedence_is_file_then_environment_then_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / 'dashboard-settings.json'
    save_settings(
        DashboardSettings(
            compiler_model='file/compiler',
            classifier_model='file/classifier',
            apply_model='file/apply',
            window_days=8,
            limit=80,
            parallelism=8,
        ),
        path,
    )
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(path))
    monkeypatch.setenv('EVALUATORQ_COMPILER_MODEL', 'env/compiler')
    monkeypatch.setenv('EVALUATORQ_CLASSIFIER_MODEL', 'env/classifier')
    monkeypatch.setenv('EVALUATORQ_APPLY_MODEL', 'env/apply')

    settings = effective_settings(
        {
            'compiler_model': 'override/compiler',
            'window_days': 21,
            'limit': None,
        }
    )

    assert settings.compiler_model == 'override/compiler'
    assert settings.classifier_model == 'env/classifier'
    assert settings.apply_model == 'env/apply'
    assert settings.window_days == 21
    assert settings.limit == 80
    assert settings.parallelism == 8


def test_effective_settings_reads_finder_limit_environment_layer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(tmp_path / 'settings.json'))
    monkeypatch.setenv('EVALUATORQ_FINDER_WINDOW_DAYS', '14')
    monkeypatch.setenv('EVALUATORQ_FINDER_LIMIT', '120')
    monkeypatch.setenv('EVALUATORQ_FINDER_PARALLELISM', '12')

    settings = effective_settings()

    assert settings.window_days == 14
    assert settings.limit == 120
    assert settings.parallelism == 12


def test_effective_settings_ignores_invalid_finder_limit_environment_layer(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(tmp_path / 'settings.json'))
    monkeypatch.setenv('EVALUATORQ_FINDER_WINDOW_DAYS', 'not-an-int')
    monkeypatch.setenv('EVALUATORQ_FINDER_LIMIT', '5001')
    monkeypatch.setenv('EVALUATORQ_FINDER_PARALLELISM', '0')

    with caplog.at_level('WARNING'):
        settings = effective_settings()

    assert settings.window_days == 7
    assert settings.limit == 500
    assert settings.parallelism == 100
    assert 'EVALUATORQ_FINDER_WINDOW_DAYS' in caplog.text
    assert 'EVALUATORQ_FINDER_LIMIT' in caplog.text
    assert 'EVALUATORQ_FINDER_PARALLELISM' in caplog.text


def test_settings_path_uses_environment_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / 'settings.json'
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(path))

    assert settings_path() == path


def test_default_settings_match_shared_pipeline_default() -> None:
    settings = DashboardSettings.model_validate({})
    assert settings.compiler_model == DEFAULT_PIPELINE_MODEL
    assert settings.apply_model == DEFAULT_PIPELINE_MODEL


def test_old_profile_config_migrates_to_cli_profile_but_explicit_auth_wins() -> None:
    migrated = DashboardSettings.model_validate({'orq_profile': 'research-bauke'})
    explicit = DashboardSettings.model_validate(
        {'orq_profile': 'research-bauke', 'orq_auth_method': 'environment'}
    )

    assert migrated.orq_auth_method == 'cli_profile'
    assert explicit.orq_auth_method == 'environment'


def test_legacy_profile_scope_is_ignored_by_effective_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / 'settings.json'
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(path))
    save_settings(DashboardSettings.model_validate({
        'orq_auth_method': 'cli_profile', 'orq_profile': 'research-bauke',
        'orq_workspace': 'old-workspace', 'orq_project_id': 'old-project',
        'orq_project_name': 'Old project',
    }), path)

    settings = effective_settings()
    assert (settings.orq_workspace, settings.orq_project_id, settings.orq_project_name) == (None, None, None)
    assert effective_settings({'orq_project_id': 'per-run-project'}).orq_project_id == 'per-run-project'


def test_user_api_key_is_encrypted_in_settings_and_can_be_read_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from cryptography.fernet import Fernet

    secure_credentials._encryption_key.cache_clear()
    monkeypatch.setattr(secure_credentials.sys, 'platform', 'linux')
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_KEY_ENCRYPTION_KEY', Fernet.generate_key().decode())
    plain_key = 'orq-secret-test-value'

    settings = store_api_key(DashboardSettings.model_validate({}), plain_key)
    path = tmp_path / 'settings.json'
    save_settings(settings, path)

    serialized = path.read_text()
    assert plain_key not in serialized
    assert settings.orq_auth_method == 'stored_api_key'
    assert settings.orq_api_key_ciphertext
    assert read_stored_api_key(load_settings(path)) == plain_key
    secure_credentials._encryption_key.cache_clear()


def test_encrypted_key_fails_closed_when_encryption_key_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    from cryptography.fernet import Fernet

    secure_credentials._encryption_key.cache_clear()
    monkeypatch.setattr(secure_credentials.sys, 'platform', 'linux')
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_KEY_ENCRYPTION_KEY', Fernet.generate_key().decode())
    encrypted = store_api_key(DashboardSettings.model_validate({}), 'orq-secret-test-value')
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_KEY_ENCRYPTION_KEY', Fernet.generate_key().decode())
    secure_credentials._encryption_key.cache_clear()

    with pytest.raises(RuntimeError, match='cannot be decrypted'):
        read_stored_api_key(encrypted)
    secure_credentials._encryption_key.cache_clear()
