from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open('rb') as file:
        return tomllib.load(file)


def test_ty_is_added_without_removing_basedpyright() -> None:
    pyproject = _load_toml(REPO_ROOT / 'pyproject.toml')
    dev_dependencies = pyproject['dependency-groups']['dev']

    assert 'ty==0.0.84' in dev_dependencies
    assert any(dependency.startswith('basedpyright') for dependency in dev_dependencies)
    assert 'basedpyright' in pyproject['tool']

    lockfile = _load_toml(REPO_ROOT / 'uv.lock')
    locked_packages = {package['name']: package for package in lockfile['package']}
    assert locked_packages['ty']['version'] == '0.0.84'
    assert 'basedpyright' in locked_packages


def test_ty_checks_the_supported_project_surface() -> None:
    pyproject = _load_toml(REPO_ROOT / 'pyproject.toml')
    ty = pyproject['tool']['ty']

    assert ty['src']['include'] == ['src', 'tests', '*.py', 'docs/hooks.py']
    assert ty['src']['exclude'] == ['examples', 'scripts', '.venv']
    assert ty['environment'] == {
        'python-version': '3.10',
        'python-platform': 'all',
    }


def test_ty_promotes_contract_diagnostics_to_errors() -> None:
    pyproject = _load_toml(REPO_ROOT / 'pyproject.toml')
    rules = pyproject['tool']['ty']['rules']

    assert rules['ignore-comment-unknown-rule'] == 'error'
    assert rules['unused-ignore-comment'] == 'error'
    assert rules['unused-type-ignore-comment'] == 'error'
    assert rules['unused-awaitable'] == 'error'


def test_ty_dashboard_overrides_match_the_diagnostic_baseline() -> None:
    pyproject = _load_toml(REPO_ROOT / 'pyproject.toml')
    overrides = pyproject['tool']['ty']['overrides']

    assert overrides == [
        {
            'include': ['src/evaluatorq/dashboard'],
            'rules': {
                'invalid-argument-type': 'ignore',
                'invalid-return-type': 'ignore',
                'invalid-type-form': 'ignore',
                'unresolved-attribute': 'ignore',
            },
        },
        {
            'include': ['tests/dashboard'],
            'rules': {
                'invalid-argument-type': 'ignore',
                'invalid-return-type': 'ignore',
                'not-iterable': 'ignore',
                'unresolved-attribute': 'ignore',
            },
        },
    ]
