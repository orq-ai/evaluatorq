from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from evaluatorq.cli import _apply_model_flags, _register_subapps, app
from evaluatorq.common.model_roles import BUILTIN, TASKS, role_model, set_cli_models

_ENV = (
    'EVALUATORQ_FAST_MODEL',
    'EVALUATORQ_SMART_MODEL',
    'EVALUATORQ_CLASSIFIER_MODEL',
    'EVALUATORQ_COMPILER_MODEL',
    'EVALUATORQ_APPLY_MODEL',
    'EVALUATORQ_MODEL_OVERRIDES',
)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(tmp_path / 'settings.json'))
    # A blank value is ignored by the resolver, and setting it first makes monkeypatch
    # restore the variable after a test whose code writes os.environ directly.
    for name in _ENV:
        monkeypatch.setenv(name, '')
    monkeypatch.setenv('COLUMNS', '200')
    monkeypatch.setenv('NO_COLOR', '1')


@pytest.fixture(scope='module', autouse=True)
def _subapps() -> None:
    if not app.registered_groups:
        _register_subapps(app)


def _help(*args: str) -> str:
    result = CliRunner().invoke(app, [*args, '--help'])
    assert result.exit_code == 0, result.output
    return re.sub(r'\x1b\[[0-9;]*m', '', result.output)


def test_apply_model_flags_sets_roles_and_overrides() -> None:
    _apply_model_flags(fast=None, smart='a/b', classifier=None, embedding=None, overrides=['apply=c/d'])
    assert role_model('smart') == 'a/b'
    assert role_model('smart', task='apply') == 'c/d'
    assert role_model('fast') == BUILTIN['fast']


def test_apply_model_flags_exports_one_internal_variable_not_role_env_vars() -> None:
    _apply_model_flags(fast='f/x', smart=None, classifier='c/x', embedding='e/x', overrides=['apply=c/d'])
    assert json.loads(os.environ['EVALUATORQ_MODEL_OVERRIDES']) == {
        'roles': {'fast': 'f/x', 'classifier': 'c/x', 'embedding': 'e/x'},
        'overrides': {'apply': 'c/d'},
    }
    assert os.environ['EVALUATORQ_FAST_MODEL'] == ''
    assert role_model('embedding') == 'e/x'


def _worker_env(monkeypatch: pytest.MonkeyPatch, payload: object) -> None:
    """Simulate a subprocess: the environment is set, in-memory CLI state is empty."""
    monkeypatch.setenv('EVALUATORQ_MODEL_OVERRIDES', payload if isinstance(payload, str) else json.dumps(payload))
    set_cli_models()


def test_worker_resolves_role_flag_above_a_settings_task_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / 'settings.json').write_text(json.dumps({'model_overrides': {'insights.summary': 'file/task'}}))
    assert role_model('smart', task='insights.summary') == 'file/task'
    _worker_env(monkeypatch, {'roles': {'smart': 'flag/smart'}})
    assert role_model('smart', task='insights.summary') == 'flag/smart'


def test_embedding_flag_reaches_the_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    _apply_model_flags(fast=None, smart=None, classifier=None, embedding='e/x', overrides=None)
    exported = os.environ['EVALUATORQ_MODEL_OVERRIDES']
    _worker_env(monkeypatch, exported)
    assert role_model('embedding', task='insights.embedding') == 'e/x'


def test_worker_task_override_beats_role_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    _worker_env(monkeypatch, {'roles': {'smart': 'flag/smart'}, 'overrides': {'apply': 'flag/apply'}})
    assert role_model('smart', task='apply') == 'flag/apply'
    assert role_model('smart') == 'flag/smart'


@pytest.mark.parametrize(
    'payload',
    ['{not json', '[1]', {'roles': {'bogus': 'x', 'smart': ''}, 'overrides': {'bogus': 'x'}}, {'roles': 5}],
)
def test_worker_ignores_malformed_internal_variable(monkeypatch: pytest.MonkeyPatch, payload: object) -> None:
    _worker_env(monkeypatch, payload)
    assert role_model('smart', task='apply') == BUILTIN['smart']


@pytest.mark.parametrize('bad', ['bogus=x', 'apply', 'apply='])
def test_bad_model_override_exits_with_valid_tasks(bad: str) -> None:
    result = CliRunner().invoke(app, ['--model-override', bad, 'dashboard', '--help'])
    assert result.exit_code != 0
    assert 'valid tasks' in re.sub(r'\s+', ' ', result.output) or 'task=model' in result.output


def test_unknown_task_names_every_valid_task() -> None:
    result = CliRunner().invoke(app, ['--model-override', 'bogus=x', 'dashboard', '--help'])
    output = re.sub(r'[\s│╭╮╰╯─]+', ' ', result.output)
    assert 'valid tasks' in output
    assert all(task in output for task in TASKS)


def test_root_help_lists_the_model_flags() -> None:
    out = _help()
    for flag in ('--fast-model', '--smart-model', '--classifier-model', '--embedding-model', '--model-override'):
        assert flag in out


def test_redteam_run_help_points_at_the_smart_role() -> None:
    out = _help('redteam', 'run')
    assert BUILTIN['fast'] not in out
    assert 'smart model role' in re.sub(r'[\s│]+', ' ', out)


def test_dashboard_classifier_flag_help_says_a_task_override_wins() -> None:
    out = re.sub(r'[\s│]+', ' ', _help('dashboard'))
    assert 'root --model-override for a classifier task wins over it' in out


def test_root_flags_apply_before_a_subcommand_runs() -> None:
    result = CliRunner().invoke(app, ['--smart-model', 'a/b', 'dashboard', '--help'])
    assert result.exit_code == 0
    assert role_model('smart') == 'a/b'


def test_sim_model_flag_unset_leaves_roles_to_resolve() -> None:
    from evaluatorq.simulation.cli import _generation_model, _sim_llm_config

    assert 'model' not in _sim_llm_config(None).model_fields_set
    assert _generation_model(None) == BUILTIN['fast']
    assert _sim_llm_config('x/y').model == 'x/y'
    assert _generation_model('x/y') == 'x/y'
