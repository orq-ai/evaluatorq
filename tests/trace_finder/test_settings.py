from __future__ import annotations

import json
from pathlib import Path

import pytest
from loguru import logger

from evaluatorq.common.model_roles import BUILTIN, role_model, set_cli_models
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
        fast_model='fast/model',
        smart_model='smart/model',
        classifier_model='classifier/model',
        embedding_model='embedding/model',
        model_overrides={'apply': 'apply/model'},
        window_days=14,
        parallelism=7,
        trace_input_chars=320_000,
    )

    save_settings(settings, path)

    assert load_settings(path).model_dump() == settings.model_dump()
    assert json.loads(path.read_text()) == settings.model_dump()


def test_older_settings_files_inherit_default_trace_input_cap(tmp_path: Path) -> None:
    path = tmp_path / 'dashboard-settings.json'
    path.write_text(json.dumps({'window_days': 9}), encoding='utf-8')

    settings = load_settings(path)

    assert settings.trace_input_chars == 500_000


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

    assert settings.model_dump() == DashboardSettings.model_validate({}).model_dump()
    assert 'Could not load dashboard settings' in caplog.text


def test_save_loaded_malformed_file_fails_without_changing_bytes(tmp_path: Path) -> None:
    path = tmp_path / 'settings.json'
    original = b'{ malformed settings\n'
    path.write_bytes(original)

    with pytest.raises(ValueError):
        save_settings(load_settings(path).with_updates({'fast_model': 'edited/fast'}), path)

    assert path.read_bytes() == original


def test_invalid_known_setting_warns_with_field_and_error_type_without_value(tmp_path: Path) -> None:
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps({'window_days': 'secret-invalid-value', 'limit': 64}))
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message.record['message']), level='WARNING')
    try:
        settings = load_settings(path)
    finally:
        logger.remove(sink_id)

    assert settings.window_days == DashboardSettings.model_fields['window_days'].default
    assert settings.limit == 64
    assert any('window_days:int_parsing' in message for message in messages)
    assert all('secret-invalid-value' not in message for message in messages)


def test_invalid_non_auth_setting_keeps_valid_authentication_group(tmp_path: Path) -> None:
    path = tmp_path / 'settings.json'
    path.write_text(
        json.dumps({
            'schema_version': 1,
            'trace_input_chars': 'invalid-cap',
            'orq_auth_method': 'cli_profile',
            'orq_profile': 'saved-profile',
            'smart_model': 'saved/smart',
        })
    )

    settings = load_settings(path)

    assert settings.trace_input_chars == DashboardSettings.model_fields['trace_input_chars'].default
    assert settings.orq_auth_method == 'cli_profile'
    assert settings.orq_profile == 'saved-profile'
    assert settings.smart_model == 'saved/smart'


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
            fast_model='file/fast',
            classifier_model='file/classifier',
            model_overrides={'apply': 'file/apply'},
            window_days=8,
            limit=80,
            parallelism=8,
        ),
        path,
    )
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(path))
    monkeypatch.setenv('EVALUATORQ_CLASSIFIER_MODEL', 'env/classifier')
    monkeypatch.setenv('EVALUATORQ_APPLY_MODEL', 'env/apply')
    set_cli_models()

    settings = effective_settings(
        {
            'window_days': 21,
            'limit': None,
        }
    )

    # Models resolve through role_model, not effective_settings.
    assert settings.fast_model == 'file/fast'
    assert role_model('fast') == 'file/fast'
    assert role_model('classifier') == 'env/classifier'
    assert role_model('smart', task='apply') == 'env/apply'
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


def test_default_settings_have_no_saved_models() -> None:
    settings = DashboardSettings.model_validate({})
    assert settings.fast_model is None
    assert settings.smart_model is None
    assert settings.classifier_model is None
    assert settings.embedding_model is None
    assert settings.model_overrides == {}
    assert role_model('fast') == BUILTIN['fast']


def test_blank_role_model_means_default() -> None:
    settings = DashboardSettings.model_validate({'fast_model': '  ', 'smart_model': ' a/b '})
    assert settings.fast_model is None
    assert settings.smart_model == 'a/b'


def test_unknown_and_blank_model_overrides_are_dropped(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level('WARNING'):
        settings = DashboardSettings.model_validate(
            {'model_overrides': {'apply': 'a/b', 'bogus': 'c/d', 'sim.judge': ' '}}
        )
    assert settings.model_overrides == {'apply': 'a/b'}
    assert 'bogus' in caplog.text


def test_legacy_settings_load_without_defaults_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    path = tmp_path / 'settings.json'
    path.write_text(
        json.dumps(
            {
                'compiler_model': 'custom/compiler',
                'apply_model': 'openai/gpt-5.6-luna',
                'classifier_model': 'custom/classifier',
                'window_days': 9,
            }
        )
    )
    with caplog.at_level('WARNING'):
        settings = load_settings(path)
    assert 'using defaults' not in caplog.text
    assert settings.window_days == 9
    assert settings.model_overrides == {'finder.compiler': 'custom/compiler'}
    assert settings.classifier_model == 'custom/classifier'


def test_saved_trace_input_limit_loads_with_the_other_settings_and_survives_a_save(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    path = tmp_path / 'settings.json'
    path.write_text(
        json.dumps(
            {
                'trace_input_chars': 180000,
                'fast_model': 'saved/fast',
                'window_days': 12,
                'limit': 64,
                'ask_ai_mode': 'review',
                'orq_auth_method': 'cli_oauth',
                'orq_workspace': 'saved-workspace',
                'orq_project_id': 'saved-project',
                'orq_project_name': 'Saved project',
            }
        )
    )

    with caplog.at_level('WARNING'):
        settings = load_settings(path)

    assert 'using defaults' not in caplog.text
    assert settings.trace_input_chars == 180000
    assert settings.fast_model == 'saved/fast'
    assert settings.window_days == 12
    assert settings.limit == 64
    assert settings.ask_ai_mode == 'review'
    assert settings.orq_auth_method == 'cli_oauth'
    assert (settings.orq_workspace, settings.orq_project_id, settings.orq_project_name) == (
        'saved-workspace',
        'saved-project',
        'Saved project',
    )

    save_settings(settings, path)

    assert json.loads(path.read_text())['trace_input_chars'] == 180000


def test_trace_input_limit_does_not_allow_other_unknown_settings() -> None:
    with pytest.raises(ValueError, match='extra_forbidden'):
        DashboardSettings.model_validate({'trace_input_chars': 180000, 'limt': 64})


def test_loader_ignores_unknown_top_level_keys_and_keeps_recognized_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps({'window_days': 19, 'limit': 73, 'private_future_value': 'secret'}))
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message.record['message']), level='WARNING')
    try:
        settings = load_settings(path)
    finally:
        logger.remove(sink_id)

    assert (settings.window_days, settings.limit) == (19, 73)
    assert any('private_future_value' in message for message in messages)
    assert all('secret' not in message for message in messages)


def test_unversioned_legacy_settings_migrate_and_save_as_v1(tmp_path: Path) -> None:
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps({'compiler_model': 'custom/compiler', 'orq_profile': 'research-bauke'}))

    settings = load_settings(path)

    assert settings.schema_version == 1
    assert settings.model_overrides == {'finder.compiler': 'custom/compiler'}
    assert settings.orq_auth_method == 'cli_profile'
    save_settings(settings, path)
    saved = json.loads(path.read_text())
    assert saved['schema_version'] == 1
    assert 'compiler_model' not in saved
    assert saved['model_overrides'] == {'finder.compiler': 'custom/compiler'}


def test_unversioned_migration_preserves_invalid_overrides_and_legacy_models(tmp_path: Path) -> None:
    path = tmp_path / 'settings.json'
    raw = {'model_overrides': 'invalid-preserve-me', 'compiler_model': 'custom/compiler', 'limit': 64}
    path.write_text(json.dumps(raw))

    settings = load_settings(path).with_updates({'smart_model': 'edited/smart'})
    save_settings(settings, path)

    saved = json.loads(path.read_text())
    assert saved['schema_version'] == 1
    assert saved['model_overrides'] == 'invalid-preserve-me'
    assert saved['compiler_model'] == 'custom/compiler'
    assert saved['limit'] == 64


def test_copying_unversioned_settings_migrates_new_destination_only(tmp_path: Path) -> None:
    source = tmp_path / 'source.json'
    destination = tmp_path / 'copy.json'
    original = {
        'compiler_model': 'custom/compiler',
        'model_overrides': {'future.task': 'saved/other'},
        'trace_input_chars': 'invalid-cap',
        'future_nested': {'keep': [1, None]},
    }
    source.write_text(json.dumps(original))
    source_bytes = source.read_bytes()

    save_settings(load_settings(source), destination)

    assert source.read_bytes() == source_bytes
    copied = json.loads(destination.read_text())
    assert copied['schema_version'] == 1
    assert copied['model_overrides'] == {'finder.compiler': 'custom/compiler', 'future.task': 'saved/other'}
    assert copied['trace_input_chars'] == 'invalid-cap'
    assert copied['future_nested'] == original['future_nested']
    assert load_settings(destination).model_overrides['finder.compiler'] == 'custom/compiler'


def test_copying_settings_patches_model_overrides_and_keeps_unknown_tasks(tmp_path: Path) -> None:
    source = tmp_path / 'source.json'
    destination = tmp_path / 'copy.json'
    source.write_text(json.dumps({
        'schema_version': 1,
        'model_overrides': {'apply': 'saved/apply', 'future.task': 'opaque'},
    }))
    settings = load_settings(source).with_updates({'model_overrides': {'apply': 'edited/apply'}})

    save_settings(settings, destination)

    assert json.loads(destination.read_text())['model_overrides'] == {
        'apply': 'edited/apply',
        'future.task': 'opaque',
    }


def test_copying_invalid_auth_source_rejects_explicit_auth_edit_without_creating_file(tmp_path: Path) -> None:
    source = tmp_path / 'source.json'
    destination = tmp_path / 'copy.json'
    original = {
        'schema_version': 1,
        'orq_auth_method': 'cli_profile',
        'orq_profile': 'saved-profile',
        'orq_profile_host': 123,
    }
    source.write_text(json.dumps(original))
    settings = load_settings(source).with_updates({'orq_auth_method': 'environment'})

    with pytest.raises(ValueError):
        save_settings(settings, destination)

    assert not destination.exists()
    assert json.loads(source.read_text()) == original


def test_unrelated_settings_edit_preserves_nested_unknown_and_invalid_source_values(tmp_path: Path) -> None:
    path = tmp_path / 'settings.json'
    raw = {
        'schema_version': 1,
        'window_days': 'bad',
        'limit': 64,
        'model_overrides': {'apply': 'saved/apply', 'future.task': {'opaque': True}},
        'explorer_columns': ['model', 'future_column'],
        'future_setting': {'keep': [1, None]},
    }
    path.write_text(json.dumps(raw))
    settings = load_settings(path)
    updated = settings.with_updates({'fast_model': 'edited/fast'})

    save_settings(updated, path)

    saved = json.loads(path.read_text())
    assert saved['fast_model'] == 'edited/fast'
    assert saved['window_days'] == 'bad'
    assert saved['limit'] == 64
    assert saved['model_overrides'] == raw['model_overrides']
    assert saved['explorer_columns'] == raw['explorer_columns']
    assert saved['future_setting'] == raw['future_setting']


def test_stale_settings_save_keeps_latest_unrelated_edits(tmp_path: Path) -> None:
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps({'schema_version': 1, 'fast_model': 'old/fast', 'limit': 10}))
    stale = load_settings(path).with_updates({'fast_model': 'new/fast'})
    path.write_text(json.dumps({'schema_version': 1, 'fast_model': 'old/fast', 'limit': 64, 'new_key': 'new'}))

    save_settings(stale, path)

    saved = json.loads(path.read_text())
    assert saved['fast_model'] == 'new/fast'
    assert saved['limit'] == 64
    assert saved['new_key'] == 'new'


def test_explicit_default_repair_updates_invalid_persisted_value(tmp_path: Path) -> None:
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps({'schema_version': 1, 'trace_input_chars': 'bad'}))
    settings = load_settings(path).with_updates({'trace_input_chars': 500_000}, persistent=False)

    save_settings(settings, path, changed_fields={'trace_input_chars'})

    assert json.loads(path.read_text())['trace_input_chars'] == 500_000


def test_runtime_overrides_do_not_persist_when_an_explicit_field_is_saved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps({'schema_version': 1, 'limit': 64, 'smart_model': 'saved/smart'}))
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(path))
    monkeypatch.setenv('EVALUATORQ_FINDER_LIMIT', '42')
    settings = effective_settings().model_copy(update={'explorer_columns': ('model',)})

    save_settings(settings, path, changed_fields={'explorer_columns'})

    saved = json.loads(path.read_text())
    assert saved['limit'] == 64
    assert saved['smart_model'] == 'saved/smart'
    assert saved['explorer_columns'] == ['model']


@pytest.mark.parametrize('version', [0, -1, True, '1', 1.0])
def test_invalid_schema_versions_fall_back_to_defaults(tmp_path: Path, version: object) -> None:
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps({'schema_version': version, 'window_days': 21}))

    assert load_settings(path).model_dump() == DashboardSettings.model_validate({}).model_dump()


def test_newer_schema_warns_reads_known_values_and_preserves_version_on_save(tmp_path: Path) -> None:
    path = tmp_path / 'settings.json'
    future_values = {
        'future_field': 'opaque string',
        'future_object': {'nested': [1, 'two', {'three': True}]},
        'future_list': ['a', 2, None],
        'future_null': None,
    }
    path.write_text(json.dumps({'schema_version': 2, 'window_days': 22, **future_values}))
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message.record['message']), level='WARNING')
    try:
        settings = load_settings(path)
    finally:
        logger.remove(sink_id)

    assert settings.schema_version == 2
    assert settings.window_days == 22
    assert any('newer than supported' in message for message in messages)
    assert any(', '.join(sorted(future_values)) in message for message in messages)
    assert all('opaque string' not in message for message in messages)
    assert not set(future_values) & settings.model_dump().keys()

    updated = settings.with_updates({'window_days': 23})
    assert updated.window_days == 23
    save_settings(settings, path)
    saved = json.loads(path.read_text())
    assert saved['schema_version'] == 2
    assert saved['window_days'] == 22
    assert {key: saved[key] for key in future_values} == future_values
    save_settings(updated, path)
    saved = json.loads(path.read_text())
    assert saved['window_days'] == 23
    assert {key: saved[key] for key in future_values} == future_values


def test_model_copy_preserves_future_fields_for_explorer_columns(tmp_path: Path) -> None:
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps({'schema_version': 2, 'future_field': {'keep': True}}))

    settings = load_settings(path)
    updated = settings.model_copy(update={'explorer_columns': ('model', 'cost')})
    save_settings(updated, path)

    saved = json.loads(path.read_text())
    assert saved['explorer_columns'] == ['model', 'cost']
    assert saved['future_field'] == {'keep': True}


def test_effective_settings_updates_preserve_future_fields_on_save(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps({'schema_version': 3, 'window_days': 8, 'future_field': {'keep': True}}))
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(path))

    settings = effective_settings({'window_days': 13}).model_copy(update={'explorer_columns': ('model',)})
    save_settings(settings, path)

    saved = json.loads(path.read_text())
    assert saved['schema_version'] == 3
    assert saved['window_days'] == 8
    assert saved['explorer_columns'] == ['model']
    assert saved['future_field'] == {'keep': True}


def test_invalid_known_future_setting_falls_back_but_preserves_future_fields_on_save(tmp_path: Path) -> None:
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps({'schema_version': 2, 'window_days': 'invalid', 'future_option': {'keep': True}}))

    settings = load_settings(path)

    assert settings.window_days == 7
    assert settings.schema_version == 2
    save_settings(settings, path)

    saved = json.loads(path.read_text())
    assert saved['schema_version'] == 2
    assert saved['window_days'] == 'invalid'
    assert saved['future_option'] == {'keep': True}


@pytest.mark.parametrize('version', [None, 1])
def test_supported_and_unversioned_files_preserve_unknown_fields_on_save(tmp_path: Path, version: int | None) -> None:
    path = tmp_path / 'settings.json'
    raw = {'window_days': 18, 'future_field': 'drop me'}
    if version is not None:
        raw['schema_version'] = version
    path.write_text(json.dumps(raw))

    save_settings(load_settings(path), path)

    saved = json.loads(path.read_text())
    assert saved['schema_version'] == 1
    assert saved['window_days'] == 18
    assert saved['future_field'] == 'drop me'


def test_persisted_schema_version_round_trip_and_strict_validation(tmp_path: Path) -> None:
    path = tmp_path / 'settings.json'
    settings = DashboardSettings(window_days=17)

    save_settings(settings, path)

    assert json.loads(path.read_text())['schema_version'] == 1
    assert load_settings(path).model_dump() == settings.model_dump()
    with pytest.raises(ValueError, match='int_type'):
        DashboardSettings.model_validate({'schema_version': '1'})


def test_explorer_columns_round_trip_and_unknown_keys_drop(tmp_path: Path) -> None:
    path = tmp_path / 'settings.json'
    save_settings(DashboardSettings.model_validate({'explorer_columns': ('model', 'cost')}), path)
    assert load_settings(path).explorer_columns == ('model', 'cost')

    path.write_text('{"explorer_columns": ["model", "gone"]}', encoding='utf-8')
    assert load_settings(path).explorer_columns == ('model',)


def test_old_profile_config_migrates_to_cli_profile_but_explicit_auth_wins() -> None:
    migrated = DashboardSettings.model_validate({'orq_profile': 'research-bauke'})
    explicit = DashboardSettings.model_validate(
        {'orq_profile': 'research-bauke', 'orq_auth_method': 'environment'}
    )

    assert migrated.orq_auth_method == 'cli_profile'
    assert explicit.orq_auth_method == 'environment'


@pytest.mark.parametrize('method', ['environment', 'cli_profile', 'stored_api_key'])
def test_legacy_scope_is_ignored_by_effective_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, method: str
) -> None:
    path = tmp_path / 'settings.json'
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(path))
    save_settings(DashboardSettings.model_validate({
        'orq_auth_method': method, 'orq_profile': 'research-bauke',
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


def test_blank_task_override_does_not_block_legacy_migration() -> None:
    settings = DashboardSettings.model_validate(
        {'compiler_model': 'custom/compiler', 'model_overrides': {'finder.compiler': '  '}}
    )
    assert settings.model_overrides == {'finder.compiler': 'custom/compiler'}


def test_non_blank_task_override_still_beats_legacy_value() -> None:
    settings = DashboardSettings.model_validate(
        {'compiler_model': 'custom/compiler', 'model_overrides': {'finder.compiler': 'mine/x'}}
    )
    assert settings.model_overrides == {'finder.compiler': 'mine/x'}
