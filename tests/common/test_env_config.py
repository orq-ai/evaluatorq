"""Tests for the declared env-var table and its readers (common/env_config).

Covers the return value AND that a WARNING is actually emitted on every invalid case, since a
silent misconfiguration is the failure this reader exists to prevent. The last section checks the
table against the code that owns layered defaults and against the docs reference tables.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from loguru import logger

from evaluatorq.common import env_config
from evaluatorq.common.env_config import (
    ENV_VARS,
    EnvVar,
    env_bool,
    env_float,
    env_int,
    env_raw,
    env_text,
    mask_secret,
    report,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

_TEST_ROWS = (
    EnvVar('X_INT', 'int', 7, 'test int', 1, 100),
    EnvVar('X_FLOAT', 'float', 2.0, 'test float', 0.0),
    EnvVar('X_BOOL', 'bool', default=True, description='test bool'),
    EnvVar('X_RAW', 'raw', 'fallback', 'test raw'),
    EnvVar('X_SECRET', 'raw', None, 'test secret', secret=True),
)


@pytest.fixture(autouse=True)
def _rows(monkeypatch: pytest.MonkeyPatch) -> None:
    for row in _TEST_ROWS:
        monkeypatch.setitem(ENV_VARS, row.name, row)
        monkeypatch.delenv(row.name, raising=False)


@pytest.fixture
def warns() -> Iterator[list[str]]:
    """Capture loguru WARNING messages (loguru does not feed pytest's caplog)."""
    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(m.record['message']), level='WARNING')
    yield messages
    logger.remove(sink_id)


# --- declared names ---
@pytest.mark.parametrize('reader', [env_raw, env_text, env_int, env_float, env_bool])
def test_undeclared_name_raises(reader) -> None:  # noqa: ANN001
    with pytest.raises(KeyError, match='not declared'):
        reader('X_NOT_DECLARED')


def test_typed_reader_refuses_a_row_of_another_kind() -> None:
    with pytest.raises(KeyError, match="declared as 'raw'"):
        env_int('X_RAW')


def test_no_row_name_is_declared_twice() -> None:
    names = [row.name for row in env_config._ROWS]
    assert len(names) == len(set(names))


# --- env_raw / env_text ---
def test_env_raw_passes_values_through_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    assert env_raw('X_RAW') is None  # no declared default applied: same as os.environ.get
    assert env_raw('X_RAW', 'caller') == 'caller'
    for raw in ('', '  ', ' padded '):
        monkeypatch.setenv('X_RAW', raw)
        assert env_raw('X_RAW', 'caller') == raw


def test_env_text_strips_and_treats_blank_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    assert env_text('X_RAW') is None
    monkeypatch.setenv('X_RAW', '   ')
    assert env_text('X_RAW') is None
    monkeypatch.setenv('X_RAW', ' value ')
    assert env_text('X_RAW') == 'value'


# --- env_int ---
def test_env_int_unset_is_silent(warns: list[str]) -> None:
    assert env_int('X_INT') == 7
    assert warns == []  # a truly-absent variable is not a misconfiguration


def test_env_int_empty_warns(monkeypatch: pytest.MonkeyPatch, warns: list[str]) -> None:
    monkeypatch.setenv('X_INT', '')
    assert env_int('X_INT') == 7
    assert any('set but empty' in m for m in warns)  # unresolved CI ${{ vars.X }} must signal


def test_env_int_whitespace_only_warns(monkeypatch: pytest.MonkeyPatch, warns: list[str]) -> None:
    monkeypatch.setenv('X_INT', '   ')
    assert env_int('X_INT') == 7
    assert any('set but empty' in m for m in warns)


def test_env_int_valid_and_stripped(monkeypatch: pytest.MonkeyPatch, warns: list[str]) -> None:
    monkeypatch.setenv('X_INT', '42')
    assert env_int('X_INT') == 42
    monkeypatch.setenv('X_INT', '  42  ')  # surrounding whitespace tolerated
    assert env_int('X_INT') == 42
    assert warns == []  # a valid value warns about nothing


def test_env_int_invalid_warns_and_defaults(monkeypatch: pytest.MonkeyPatch, warns: list[str]) -> None:
    monkeypatch.setenv('X_INT', 'notanint')
    assert env_int('X_INT') == 7  # never raises
    assert any('not an integer' in m for m in warns)


def test_env_int_out_of_range_warns_and_defaults(monkeypatch: pytest.MonkeyPatch, warns: list[str]) -> None:
    monkeypatch.setenv('X_INT', '0')
    assert env_int('X_INT') == 7
    monkeypatch.setenv('X_INT', '999')
    assert env_int('X_INT') == 7
    assert sum('must be' in m for m in warns) == 2


# --- env_float ---
def test_env_float_valid_invalid_range(monkeypatch: pytest.MonkeyPatch, warns: list[str]) -> None:
    monkeypatch.setenv('X_FLOAT', '1.5')
    assert env_float('X_FLOAT') == 1.5
    monkeypatch.setenv('X_FLOAT', 'nope')
    assert env_float('X_FLOAT') == 2.0
    monkeypatch.setenv('X_FLOAT', '-1')
    assert env_float('X_FLOAT') == 2.0
    assert any('not a number' in m for m in warns)
    assert any('must be >=' in m for m in warns)


@pytest.mark.parametrize('raw', ['nan', 'inf', '-inf', 'Infinity'])
def test_env_float_rejects_non_finite(monkeypatch: pytest.MonkeyPatch, warns: list[str], raw: str) -> None:
    monkeypatch.setenv('X_FLOAT', raw)
    assert env_float('X_FLOAT') == 2.0  # float() would accept these; the reader must not
    assert any('finite' in m for m in warns)


# --- env_bool ---
@pytest.mark.parametrize(
    ('raw', 'expected'),
    [
        ('1', True),
        ('true', True),
        ('YES', True),
        ('on', True),
        ('0', False),
        ('false', False),
        ('no', False),
        ('OFF', False),
    ],
)
def test_env_bool_recognised(monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool) -> None:
    monkeypatch.setenv('X_BOOL', raw)
    assert env_bool('X_BOOL') is expected


def test_env_bool_unset_is_silent(warns: list[str]) -> None:
    assert env_bool('X_BOOL') is True
    assert warns == []


def test_env_bool_unrecognised_warns_and_defaults(monkeypatch: pytest.MonkeyPatch, warns: list[str]) -> None:
    monkeypatch.setenv('X_BOOL', 'maybe')
    assert env_bool('X_BOOL') is True  # unrecognised -> warn + default, never raises
    assert any('not a boolean' in m for m in warns)


def test_env_bool_empty_and_whitespace_are_silent_defaults(monkeypatch: pytest.MonkeyPatch, warns: list[str]) -> None:
    for raw in ('', '   '):
        monkeypatch.setenv('X_BOOL', raw)
        assert env_bool('X_BOOL') is True  # a bool has no "empty means misconfigured" case
    assert warns == []


# --- report ---
def _reported(name: str) -> env_config.EnvReport:
    return next(row for row in report() if row.name == name)


def test_report_status(monkeypatch: pytest.MonkeyPatch, warns: list[str]) -> None:
    assert _reported('X_INT').status == 'unset'
    monkeypatch.setenv('X_INT', '42')
    assert _reported('X_INT').status == 'set'
    for raw in ('', 'abc', '0', '999'):
        monkeypatch.setenv('X_INT', raw)
        assert _reported('X_INT').status == 'rejected', raw
    monkeypatch.setenv('X_FLOAT', 'inf')
    assert _reported('X_FLOAT').status == 'rejected'
    monkeypatch.setenv('X_BOOL', '')
    assert _reported('X_BOOL').status == 'set'  # the bool reader treats empty as unset, not as a mistake
    monkeypatch.setenv('X_BOOL', 'maybe')
    assert _reported('X_BOOL').status == 'rejected'
    assert warns == []  # reporting never logs


def test_report_masks_secrets_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('X_SECRET', 'sk-proj-abcdEF1234wxyz')
    monkeypatch.setenv('X_RAW', 'plain-value')
    assert _reported('X_SECRET').value == mask_secret('sk-proj-abcdEF1234wxyz')
    assert _reported('X_RAW').value == 'plain-value'


@pytest.mark.parametrize(
    'name', ['ORQ_API_KEY', 'OPENAI_API_KEY', 'EVALUATORQ_DASHBOARD_KEY_ENCRYPTION_KEY', 'OTEL_EXPORTER_OTLP_HEADERS']
)
def test_credentials_are_declared_secret(name: str) -> None:
    assert ENV_VARS[name].secret


def test_mask_secret_shows_only_suffix() -> None:
    assert mask_secret('sk-proj-abcdEF1234wxyz') == '****************wxyz'
    assert mask_secret('abcd') == '****'


# --- the table against its owners ---
def test_role_rows_match_model_roles() -> None:
    from evaluatorq.common.model_roles import BUILTIN, LEGACY_TASK_ENV, ROLE_ENV

    for role, name in ROLE_ENV.items():
        assert ENV_VARS[name].default == BUILTIN[role]
        assert ENV_VARS[name].owner == 'evaluatorq.common.model_roles.role_model'
    for name in LEGACY_TASK_ENV.values():
        assert ENV_VARS[name].owner == 'evaluatorq.common.model_roles.role_model'


def test_finder_rows_match_settings() -> None:
    from evaluatorq.trace_finder import settings

    fields = settings.DashboardSettings.model_fields
    for field, name, lo, hi in (
        ('window_days', 'EVALUATORQ_FINDER_WINDOW_DAYS', settings.MIN_WINDOW_DAYS, settings.MAX_WINDOW_DAYS),
        ('limit', 'EVALUATORQ_FINDER_LIMIT', settings.MIN_LIMIT, settings.MAX_LIMIT),
        ('parallelism', 'EVALUATORQ_FINDER_PARALLELISM', settings.MIN_PARALLELISM, settings.MAX_PARALLELISM),
    ):
        row = ENV_VARS[name]
        assert (row.default, row.min_value, row.max_value) == (fields[field].default, lo, hi)


def test_other_duplicated_defaults_match_their_source(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.redteam.contracts import DEFAULT_TARGET_MAX_TOKENS
    from evaluatorq.trace_finder.settings import settings_path

    assert ENV_VARS['EVALUATORQ_LLM_MAX_TOKENS'].default == DEFAULT_TARGET_MAX_TOKENS
    monkeypatch.delenv('EVALUATORQ_DASHBOARD_SETTINGS', raising=False)
    assert ENV_VARS['EVALUATORQ_DASHBOARD_SETTINGS'].default == str(settings_path())


# --- the table against docs/configuration.md ---
_DOCS = Path(__file__).resolve().parents[2] / 'docs' / 'configuration.md'
# Documented but not owned: written by `eq dashboard` for its own reloader subprocess.
_DOCUMENTED_EXEMPT = frozenset({'EVALUATORQ_DASHBOARD_ROOTS'})
# Default cells written as prose rather than a value, with why.
_PROSE_DEFAULTS = {
    'ORQ_UI_BASE_URL': 'falls back through the profile host and ORQ_BASE_URL',
    'OPENAI_BASE_URL': "the openai SDK's own default",
    'EVALUATORQ_DIR': 'falls back to .evaluatorq in the working directory',
    'EVALUATORQ_SPAN_MAX_TEXT_CHARS': 'unset means no limit',
    'EVALUATORQ_REASONING_EFFORT': 'unset means the parameter is not sent',
    'ORQ_DISABLE_TRACING': 'documented as unset; reads as false',
}


def _docs_rows() -> dict[str, str]:
    text = _DOCS.read_text()
    section = text[text.index('## Full environment variable reference') : text.index('## When a value is wrong')]
    rows: dict[str, str] = {}
    for line in section.splitlines():
        if not line.startswith('| `'):
            continue
        cells = [cell.strip() for cell in line.strip('|').split('|')]
        for name in re.findall(r'`([A-Z0-9_]+)`', cells[0]):
            rows[name] = cells[2]
    return rows


def _docs_default(cell: str) -> object:
    if cell in {'—', 'unset'}:
        return None
    value = cell.strip('`')
    if value in {'true', 'false'}:
        return value == 'true'
    try:
        return float(value)
    except ValueError:
        return value


def test_docs_list_exactly_the_declared_variables() -> None:
    documented = set(_docs_rows())
    declared = {name for name in ENV_VARS if not name.startswith('X_')}
    assert documented - _DOCUMENTED_EXEMPT == declared, (
        f'missing from docs/configuration.md: {sorted(declared - documented)}; '
        f'documented but not declared in ENV_VARS: {sorted(documented - declared - _DOCUMENTED_EXEMPT)}'
    )


def test_docs_defaults_match_declared_defaults() -> None:
    for name, cell in _docs_rows().items():
        if name in _DOCUMENTED_EXEMPT or name in _PROSE_DEFAULTS:
            continue
        declared = ENV_VARS[name].default
        documented = _docs_default(cell)
        if isinstance(declared, bool) or not isinstance(declared, (int, float)):
            assert documented == declared, (name, cell)
        else:
            assert documented == float(declared), (name, cell)


def test_prose_default_exemptions_are_still_needed() -> None:
    assert set(_PROSE_DEFAULTS) <= set(_docs_rows())
    assert len(_PROSE_DEFAULTS) == 6
