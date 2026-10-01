from __future__ import annotations

import ast
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _table(text: str, name: str) -> str:
    match = re.search(rf'(?ms)^\[{re.escape(name)}\]\n(.*?)(?=^\[|\Z)', text)
    assert match is not None, f'missing [{name}] table'
    return match.group(1)


def _string(table: str, key: str) -> str:
    match = re.search(rf'(?m)^{re.escape(key)}\s*=\s*"([^"]+)"$', table)
    assert match is not None, f'missing {key} string'
    return match.group(1)


def _array(table: str, key: str) -> list[str]:
    match = re.search(rf'(?ms)^{re.escape(key)}\s*=\s*(\[.*?\])', table)
    assert match is not None, f'missing {key} array'
    value = ast.literal_eval(match.group(1))
    assert isinstance(value, list)
    assert all(isinstance(item, str) for item in value)
    return value


def _locked_version(lockfile: str, package_name: str) -> str:
    for package in re.findall(r'(?ms)^\[\[package\]\]\n(.*?)(?=^\[\[package\]\]|\Z)', lockfile):
        if _string(package, 'name') == package_name:
            return _string(package, 'version')
    raise AssertionError(f'missing locked package {package_name}')


def test_ty_is_added_without_removing_basedpyright() -> None:
    pyproject = (REPO_ROOT / 'pyproject.toml').read_text()
    dev_dependencies = _array(_table(pyproject, 'dependency-groups'), 'dev')

    assert 'ty==0.0.84' in dev_dependencies
    assert any(dependency.startswith('basedpyright') for dependency in dev_dependencies)
    assert '[tool.basedpyright]' in pyproject

    lockfile = (REPO_ROOT / 'uv.lock').read_text()
    assert _locked_version(lockfile, 'ty') == '0.0.84'
    assert _locked_version(lockfile, 'basedpyright')


def test_ty_checks_the_supported_project_surface() -> None:
    pyproject = (REPO_ROOT / 'pyproject.toml').read_text()
    src = _table(pyproject, 'tool.ty.src')
    environment = _table(pyproject, 'tool.ty.environment')

    assert _array(src, 'include') == ['src', 'tests', '*.py', 'docs/*.py']
    assert _array(src, 'exclude') == ['examples', 'scripts', '.venv']
    assert _string(environment, 'python-version') == '3.10'
    assert _string(environment, 'python-platform') == 'all'


def test_ty_promotes_contract_diagnostics_to_errors() -> None:
    pyproject = (REPO_ROOT / 'pyproject.toml').read_text()
    rules = _table(pyproject, 'tool.ty.rules')

    assert _string(rules, 'ignore-comment-unknown-rule') == 'error'
    assert _string(rules, 'unused-ignore-comment') == 'error'
    assert _string(rules, 'unused-type-ignore-comment') == 'error'
    assert _string(rules, 'unused-awaitable') == 'error'


def test_ty_dashboard_overrides_match_the_diagnostic_baseline() -> None:
    pyproject = (REPO_ROOT / 'pyproject.toml').read_text()
    expected = '''[[tool.ty.overrides]]
include = [ "src/evaluatorq/dashboard" ]

  [tool.ty.overrides.rules]
  invalid-argument-type = "ignore"
  invalid-return-type = "ignore"
  invalid-type-form = "ignore"
  unresolved-attribute = "ignore"

[[tool.ty.overrides]]
include = [ "tests/dashboard" ]

  [tool.ty.overrides.rules]
  invalid-argument-type = "ignore"
  invalid-return-type = "ignore"
  not-iterable = "ignore"
  unresolved-attribute = "ignore"'''

    assert pyproject.count('[[tool.ty.overrides]]') == 2
    assert expected in pyproject
