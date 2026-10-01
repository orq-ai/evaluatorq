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


def _yaml_block(text: str, header: str, indent: int) -> str:
    lines = text.splitlines()
    prefix = ' ' * indent
    matches = [index for index, line in enumerate(lines) if line == f'{prefix}{header}']
    assert len(matches) == 1, f'expected one {header} block at indentation {indent}'

    start = matches[0]
    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if line and len(line) - len(line.lstrip()) <= indent:
            end = index
            break
    return '\n'.join(lines[start:end])


def _yaml_list_blocks(text: str, indent: int) -> list[str]:
    lines = text.splitlines()
    prefix = f'{" " * indent}- '
    starts = [index for index, line in enumerate(lines) if line.startswith(prefix)]
    blocks: list[str] = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        blocks.append('\n'.join(lines[start:end]))
    return blocks


def _assert_ci_typecheck_contract(workflow: str) -> None:
    jobs = _yaml_block(workflow, 'jobs:', indent=0)
    check_job = _yaml_block(jobs, 'check:', indent=2)
    expected_name = "    name: Typecheck + test (${{ matrix.python-version }}${{ matrix.os != 'ubuntu-latest' && format(', {0}', matrix.os) || '' }})"
    assert expected_name in check_job.splitlines()

    expected_run = '        run: uv run ty check'
    assert sum(line == expected_run for line in workflow.splitlines()) == 1
    steps = _yaml_block(check_job, 'steps:', indent=4)
    ty_steps = [step for step in _yaml_list_blocks(steps, indent=6) if expected_run in step.splitlines()]
    assert len(ty_steps) == 1
    assert "        if: matrix.os == 'ubuntu-latest' && matrix.python-version == '3.10'" in ty_steps[0].splitlines()


def test_ty_is_the_only_locked_typechecker() -> None:
    pyproject = (REPO_ROOT / 'pyproject.toml').read_text()
    dev_dependencies = _array(_table(pyproject, 'dependency-groups'), 'dev')

    assert 'ty==0.0.84' in dev_dependencies
    assert not any(dependency.startswith('basedpyright') for dependency in dev_dependencies)
    assert '[tool.basedpyright]' not in pyproject

    lockfile = (REPO_ROOT / 'uv.lock').read_text()
    assert _locked_version(lockfile, 'ty') == '0.0.84'
    assert not re.search(r'(?m)^name = "(?:basedpyright|pytest-xdist)"$', lockfile)


def test_ci_runs_ty_once_on_the_python_310_ubuntu_leg() -> None:
    workflow = (REPO_ROOT / '.github/workflows/ci.yml').read_text()

    _assert_ci_typecheck_contract(workflow)
    assert 'basedpyright' not in workflow.lower()


def test_ci_typecheck_contract_rejects_a_correct_condition_only_in_a_comment() -> None:
    workflow = (REPO_ROOT / '.github/workflows/ci.yml').read_text()
    expected = "        if: matrix.os == 'ubuntu-latest' && matrix.python-version == '3.10'"
    mutated = workflow.replace(expected, "        if: matrix.os == 'ubuntu-latest'\n        # if: matrix.os == 'ubuntu-latest' && matrix.python-version == '3.10'", 1)

    try:
        _assert_ci_typecheck_contract(mutated)
    except AssertionError:
        pass
    else:
        raise AssertionError('typecheck contract accepted a condition that appeared only in a comment')


def test_ci_typecheck_contract_rejects_a_ty_command_only_in_a_comment() -> None:
    workflow = (REPO_ROOT / '.github/workflows/ci.yml').read_text()
    mutated = workflow.replace('        run: uv run ty check', '        run: uv run python -V\n        # run: uv run ty check', 1)

    try:
        _assert_ci_typecheck_contract(mutated)
    except AssertionError:
        pass
    else:
        raise AssertionError('typecheck contract accepted a ty command that appeared only in a comment')


def test_ci_typecheck_contract_rejects_the_expected_name_outside_check_job() -> None:
    workflow = (REPO_ROOT / '.github/workflows/ci.yml').read_text()
    expected = "    name: Typecheck + test (${{ matrix.python-version }}${{ matrix.os != 'ubuntu-latest' && format(', {0}', matrix.os) || '' }})"
    mutated = workflow.replace(expected, '    name: Typecheck and test', 1) + f'\n# {expected.lstrip()}\n'

    try:
        _assert_ci_typecheck_contract(mutated)
    except AssertionError:
        pass
    else:
        raise AssertionError('typecheck contract accepted the job name outside the check job')


def test_ci_keeps_full_non_integration_test_commands() -> None:
    workflow = (REPO_ROOT / '.github/workflows/ci.yml').read_text()

    assert workflow.count("run: uv run pytest -m 'not integration'") == 2
    assert "run: uv run pytest -m 'not integration'\n" in workflow
    assert "run: uv run pytest -m 'not integration' --cov --cov-report=term\n" in workflow


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
