from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluatorq.common import model_roles
from evaluatorq.common.model_roles import BUILTIN, parse_overrides, role_model, set_cli_models


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(tmp_path / 'settings.json'))
    for name in ('EVALUATORQ_FAST_MODEL', 'EVALUATORQ_SMART_MODEL', 'EVALUATORQ_CLASSIFIER_MODEL',
                 'EVALUATORQ_COMPILER_MODEL', 'EVALUATORQ_APPLY_MODEL'):
        monkeypatch.delenv(name, raising=False)
    set_cli_models()
    yield
    set_cli_models()


def _write(tmp_path: Path, values: dict[str, object]) -> None:
    (tmp_path / 'settings.json').write_text(json.dumps(values))


def test_builtin_defaults() -> None:
    assert role_model('fast') == 'openai/gpt-5.6-luna'
    assert role_model('smart') == 'openai/gpt-6-luna'
    assert role_model('classifier') == 'typesafe/jev-latest'
    assert role_model('embedding') == 'openai/text-embedding-3-small'


def test_precedence_flag_env_file_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(tmp_path, {'smart_model': 'file/smart'})
    assert role_model('smart') == 'file/smart'
    monkeypatch.setenv('EVALUATORQ_SMART_MODEL', 'env/smart')
    assert role_model('smart') == 'env/smart'
    set_cli_models({'smart': 'flag/smart'})
    assert role_model('smart') == 'flag/smart'


def test_task_override_beats_role(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(tmp_path, {'model_overrides': {'apply': 'file/apply'}})
    monkeypatch.setenv('EVALUATORQ_SMART_MODEL', 'env/smart')
    assert role_model('smart', task='apply') == 'file/apply'
    assert role_model('smart', task='insights.summary') == 'env/smart'
    set_cli_models(overrides={'apply': 'flag/apply'})
    assert role_model('smart', task='apply') == 'flag/apply'


def test_blank_env_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('EVALUATORQ_FAST_MODEL', '   ')
    assert role_model('fast') == BUILTIN['fast']


def test_legacy_env_vars_are_task_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('EVALUATORQ_COMPILER_MODEL', 'legacy/compiler')
    monkeypatch.setenv('EVALUATORQ_APPLY_MODEL', 'legacy/apply')
    assert role_model('fast', task='finder.compiler') == 'legacy/compiler'
    assert role_model('smart', task='apply') == 'legacy/apply'
    assert role_model('fast') == BUILTIN['fast']


def test_unknown_task_raises() -> None:
    with pytest.raises(KeyError):
        role_model('fast', task='nope')


def test_task_of_another_role_raises() -> None:
    with pytest.raises(ValueError, match='belongs to role'):
        role_model('fast', task='apply')


def test_role_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert model_roles.role_source('smart') == 'default'
    _write(tmp_path, {'smart_model': 'file/smart'})
    assert model_roles.role_source('smart') == 'settings'
    monkeypatch.setenv('EVALUATORQ_SMART_MODEL', 'env/smart')
    assert model_roles.role_source('smart') == 'env'
    set_cli_models({'smart': 'flag/smart'})
    assert model_roles.role_source('smart') == 'flag'


def test_parse_overrides() -> None:
    assert parse_overrides(['apply=a/b', 'sim.judge = c/d']) == {'apply': 'a/b', 'sim.judge': 'c/d'}
    with pytest.raises(ValueError, match='redteam.attacker'):
        parse_overrides(['bogus=a/b'])
    with pytest.raises(ValueError, match='task=model'):
        parse_overrides(['apply'])


def test_legacy_settings_file_migrates(tmp_path: Path) -> None:
    from evaluatorq.trace_finder.settings import load_settings

    _write(tmp_path, {'compiler_model': 'openai/gpt-5.6-luna', 'apply_model': 'anthropic/claude-opus-5-5',
                      'classifier_model': 'typesafe/jev-latest'})
    settings = load_settings()
    assert settings.model_overrides == {'apply': 'anthropic/claude-opus-5-5'}
    assert settings.classifier_model is None
    assert role_model('fast', task='finder.compiler') == BUILTIN['fast']
    assert role_model('smart', task='apply') == 'anthropic/claude-opus-5-5'
