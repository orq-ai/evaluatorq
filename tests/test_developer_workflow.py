from __future__ import annotations

import ast
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import TypedDict, cast

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


class _CollectedItem(TypedDict):
    nodeid: str
    markers: list[str]


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


def _run_scripts(yaml_text: str) -> list[str]:
    """Extract every inline or block-scalar ``run`` script from workflow YAML."""
    lines = yaml_text.splitlines()
    scripts: list[str] = []
    index = 0
    while index < len(lines):
        match = re.match(r'^(?P<indent>\s*)(?:-\s+)?run:\s*(?P<value>.*)$', lines[index])
        if match is None or lines[index].lstrip().startswith('#'):
            index += 1
            continue
        value = match.group('value').strip()
        if not re.fullmatch(r'[|>][+-]?(?:\d+)?', value):
            if len(value) >= 2 and value[0] == value[-1] == '"':
                value = json.loads(value)
            elif len(value) >= 2 and value[0] == value[-1] == "'":
                value = value[1:-1].replace("''", "'")
            scripts.append(value)
            index += 1
            continue

        header_indent = len(match.group('indent'))
        body: list[str] = []
        index += 1
        while index < len(lines):
            line = lines[index]
            indentation = len(line) - len(line.lstrip())
            if line.strip() and indentation <= header_indent:
                break
            body.append(line)
            index += 1
        nonblank_indents = [len(line) - len(line.lstrip()) for line in body if line.strip()]
        body_indent = min(nonblank_indents, default=header_indent + 1)
        scripts.append('\n'.join(line[body_indent:] if line.strip() else '' for line in body))
    return scripts


def _shell_commands(script: str) -> list[list[str]]:
    """Split a shell script into executable commands without matching quoted text."""
    lexer = shlex.shlex(script, posix=True, punctuation_chars=';&|()')
    lexer.whitespace_split = True
    lexer.commenters = '#'
    commands: list[list[str]] = []
    command: list[str] = []
    for token in lexer:
        if token and all(character in ';&|()' for character in token):
            if command:
                commands.append(command)
                command = []
        else:
            command.append(token)
    if command:
        commands.append(command)
    return commands


def _run_command(step: str) -> list[str] | None:
    scripts = _run_scripts(step)
    assert len(scripts) <= 1, 'expected at most one run script per step'
    if not scripts:
        return None
    commands = _shell_commands(scripts[0])
    return commands[0] if len(commands) == 1 else None


def _run_invocations(workflow: str, prefix: list[str]) -> list[list[str]]:
    return [
        command
        for script in _run_scripts(workflow)
        for command in _shell_commands(script)
        if command[: len(prefix)] == prefix
    ]


def _assert_failure_enforcing(block: str, *, indent: int) -> None:
    assert not re.search(
        rf'(?m)^{{indent}}continue-on-error:'.format(indent=' ' * indent),
        block,
    )


def _assert_ci_typecheck_contract(workflow: str) -> None:
    jobs = _yaml_block(workflow, 'jobs:', indent=0)
    check_job = _yaml_block(jobs, 'check:', indent=2)
    expected_name = "    name: Typecheck + test (${{ matrix.python-version }}${{ matrix.os != 'ubuntu-latest' && format(', {0}', matrix.os) || '' }})"
    assert expected_name in check_job.splitlines()
    assert '        python-version: ["3.10", "3.11", "3.12", "3.13"]' in check_job.splitlines()
    assert not any(re.match(r'^    if:', line) for line in check_job.splitlines())
    _assert_failure_enforcing(check_job, indent=4)

    steps = _yaml_block(check_job, 'steps:', indent=4)
    assert len(_run_invocations(workflow, ['uv', 'run', 'ty', 'check'])) == 1
    ty_steps = [
        step
        for step in _yaml_list_blocks(steps, indent=6)
        if (_run_command(step) or [])[:4] == ['uv', 'run', 'ty', 'check']
    ]
    assert len(ty_steps) == 1
    assert "        if: matrix.os == 'ubuntu-latest' && matrix.python-version == '3.10'" in ty_steps[0].splitlines()
    assert _run_scripts(ty_steps[0]) == ['uv run ty check']
    assert not any(re.match(r'^        shell:', line) for line in ty_steps[0].splitlines())
    _assert_failure_enforcing(ty_steps[0], indent=8)


def _assert_ci_test_contract(workflow: str) -> None:
    jobs = _yaml_block(workflow, 'jobs:', indent=0)
    check_job = _yaml_block(jobs, 'check:', indent=2)
    assert not any(re.match(r'^    if:', line) for line in check_job.splitlines())
    _assert_failure_enforcing(check_job, indent=4)
    steps = _yaml_block(check_job, 'steps:', indent=4)
    matching_steps = [
        (tuple(command), step)
        for step in _yaml_list_blocks(steps, indent=6)
        if (command := _run_command(step)) is not None
        and command[:5] == ['uv', 'run', 'pytest', '-m', 'not integration']
    ]
    bare = ('uv', 'run', 'pytest', '-m', 'not integration')
    coverage = (*bare, '--cov', '--cov-report=term')
    assert len(matching_steps) == 2
    test_steps = dict(matching_steps)
    assert set(test_steps) == {bare, coverage}
    assert "        if: matrix.python-version != '3.13' || matrix.os != 'ubuntu-latest'" in test_steps[bare].splitlines()
    assert "        if: matrix.python-version == '3.13' && matrix.os == 'ubuntu-latest'" in test_steps[coverage].splitlines()
    for step in test_steps.values():
        _assert_failure_enforcing(step, indent=8)


def _subprocess_output(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode(errors='replace')
    return value or ''


def _collected_items(tmp_path: Path, *pytest_args: str) -> list[_CollectedItem]:
    output = tmp_path / f'collection-{len(list(tmp_path.iterdir()))}.json'
    script = '''
import json
from pathlib import Path
import sys

import pytest


class CollectionRecorder:
    def pytest_collection_finish(self, session):
        items = [
            {
                "nodeid": item.nodeid,
                "markers": sorted(marker.name for marker in item.iter_markers()),
            }
            for item in session.items
        ]
        Path(sys.argv[1]).write_text(json.dumps(items))


raise SystemExit(pytest.main(sys.argv[2:], plugins=[CollectionRecorder()]))
'''
    env = os.environ.copy()
    env.pop('PYTEST_ADDOPTS', None)
    try:
        completed = subprocess.run(
            [sys.executable, '-c', script, str(output), '--collect-only', '-q', *pytest_args],
            cwd=REPO_ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as error:
        raise AssertionError(
            f'pytest collection timed out after {error.timeout} seconds\n'
            f'stdout:\n{_subprocess_output(error.stdout)}\n'
            f'stderr:\n{_subprocess_output(error.stderr)}'
        ) from error
    assert completed.returncode == 0, (
        f'pytest collection failed with exit code {completed.returncode}\n'
        f'stdout:\n{completed.stdout}\n'
        f'stderr:\n{completed.stderr}'
    )
    result = json.loads(output.read_text())
    assert isinstance(result, list)
    return cast('list[_CollectedItem]', result)


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


def test_ci_typecheck_contract_counts_ty_commands_with_options() -> None:
    workflow = (REPO_ROOT / '.github/workflows/ci.yml').read_text()
    mutated = workflow + '\n  duplicate-typecheck:\n    steps:\n      - run: uv run ty check --output-format concise\n'

    with pytest.raises(AssertionError):
        _assert_ci_typecheck_contract(mutated)


def test_run_script_extractor_covers_inline_and_block_scalars() -> None:
    workflow = '''
steps:
  - run: uv run python -V
  - run: "uv run ty check"
  - run: |
      uv run pytest
  - run: >-
      uv run ruff check
      src
'''

    assert _run_scripts(workflow) == [
        'uv run python -V',
        'uv run ty check',
        'uv run pytest',
        'uv run ruff check\nsrc',
    ]


@pytest.mark.parametrize(
    'duplicate_step',
    [
        "      - run: |\n          uv run ty check --output-format concise",
        "      - run: >\n          uv run python -V && uv run ty check --output-format concise",
        "      - run: uv run python -V; uv run ty check --output-format concise",
        '      - run: "uv run ty check --output-format concise"',
    ],
)
def test_ci_typecheck_contract_counts_every_executable_ty_invocation(duplicate_step: str) -> None:
    workflow = (REPO_ROOT / '.github/workflows/ci.yml').read_text()
    mutated = workflow + f'\n  duplicate-typecheck:\n    steps:\n{duplicate_step}\n'

    with pytest.raises(AssertionError):
        _assert_ci_typecheck_contract(mutated)


@pytest.mark.parametrize(
    ('old', 'new'),
    [
        ('        run: uv run ty check', '        run: uv run ty check || true'),
        ('        run: uv run ty check', '        run: |\n          uv run ty check\n          exit 0'),
        (
            '        run: uv run ty check',
            '        continue-on-error: true\n        run: uv run ty check',
        ),
        (
            '        run: uv run ty check',
            '        continue-on-error: false\n        run: uv run ty check',
        ),
    ],
)
def test_ci_typecheck_contract_rejects_failure_suppression(old: str, new: str) -> None:
    workflow = (REPO_ROOT / '.github/workflows/ci.yml').read_text()

    with pytest.raises(AssertionError):
        _assert_ci_typecheck_contract(workflow.replace(old, new, 1))


def test_ci_typecheck_contract_requires_python_310_matrix_leg() -> None:
    workflow = (REPO_ROOT / '.github/workflows/ci.yml').read_text()
    mutated = workflow.replace('python-version: ["3.10", "3.11", "3.12", "3.13"]', 'python-version: ["3.11", "3.12", "3.13"]', 1)

    with pytest.raises(AssertionError):
        _assert_ci_typecheck_contract(mutated)


def test_ci_typecheck_contract_rejects_disabled_check_job() -> None:
    workflow = (REPO_ROOT / '.github/workflows/ci.yml').read_text()
    mutated = workflow.replace('  check:\n', '  check:\n    if: false\n', 1)

    with pytest.raises(AssertionError):
        _assert_ci_typecheck_contract(mutated)


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

    _assert_ci_test_contract(workflow)


@pytest.mark.parametrize(
    ('old', 'new'),
    [
        ("  check:\n", "  check:\n    continue-on-error: true\n"),
        (
            "      - name: Test (unit only)\n",
            "      - name: Test (unit only)\n        continue-on-error: true\n",
        ),
        (
            "      - name: Test (unit only, with coverage)\n",
            "      - name: Test (unit only, with coverage)\n        continue-on-error: true\n",
        ),
    ],
)
def test_ci_test_contract_rejects_failure_suppression(old: str, new: str) -> None:
    workflow = (REPO_ROOT / '.github/workflows/ci.yml').read_text()

    with pytest.raises(AssertionError):
        _assert_ci_test_contract(workflow.replace(old, new, 1))


def test_ci_test_contract_rejects_commands_outside_check_job() -> None:
    workflow = (REPO_ROOT / '.github/workflows/ci.yml').read_text()
    command = "        run: uv run pytest -m 'not integration'"
    mutated = workflow.replace(command, '        run: uv run python -V', 1)
    mutated += "\n  decoy:\n    steps:\n      - run: uv run pytest -m 'not integration'\n"

    with pytest.raises(AssertionError):
        _assert_ci_test_contract(mutated)


@pytest.mark.parametrize(
    ('old_condition', 'new_condition'),
    [
        (
            "        if: matrix.python-version != '3.13' || matrix.os != 'ubuntu-latest'",
            "        if: matrix.python-version == '3.13' && matrix.os == 'ubuntu-latest'",
        ),
        (
            "        if: matrix.python-version == '3.13' && matrix.os == 'ubuntu-latest'",
            "        if: matrix.python-version != '3.13' || matrix.os != 'ubuntu-latest'",
        ),
    ],
)
def test_ci_test_contract_rejects_noncomplementary_matrix_conditions(
    old_condition: str, new_condition: str
) -> None:
    workflow = (REPO_ROOT / '.github/workflows/ci.yml').read_text()
    mutated = workflow.replace(old_condition, new_condition, 1)

    with pytest.raises(AssertionError):
        _assert_ci_test_contract(mutated)


def test_pytest_default_is_quick_and_explicit_non_integration_is_full(tmp_path: Path) -> None:
    default_items = _collected_items(tmp_path)
    full_non_integration_items = _collected_items(tmp_path, '-m', 'not integration')

    assert default_items
    assert all('slow' not in item['markers'] and 'integration' not in item['markers'] for item in default_items)
    assert all('integration' not in item['markers'] for item in full_non_integration_items)

    slow_non_integration = {
        item['nodeid'] for item in full_non_integration_items if 'slow' in item['markers']
    }
    assert slow_non_integration
    assert slow_non_integration.isdisjoint(item['nodeid'] for item in default_items)


def test_collection_subprocess_is_bounded_and_surfaces_stderr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    call_kwargs: dict[str, object] = {}

    def failed_collection(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        call_kwargs.update(kwargs)
        return subprocess.CompletedProcess(args=[], returncode=1, stdout='collection stdout', stderr='collection exploded')

    monkeypatch.setattr(subprocess, 'run', failed_collection)

    with pytest.raises(AssertionError, match='collection exploded'):
        _collected_items(tmp_path)

    assert call_kwargs['timeout'] == 30


def test_collection_timeout_surfaces_captured_stderr(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def timed_out_collection(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(
            cmd=cast('list[str]', args[0]),
            timeout=cast('float', kwargs['timeout']),
            output='partial collection output',
            stderr='collection plugin hung',
        )

    monkeypatch.setattr(subprocess, 'run', timed_out_collection)

    with pytest.raises(AssertionError, match='collection plugin hung'):
        _collected_items(tmp_path)


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
