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
    monkeypatch.setenv('X_BOOL', ' ')
    assert _reported('X_BOOL').status == 'unset'  # the bool reader treats blank as unset, not as a mistake
    monkeypatch.setenv('X_BOOL', 'maybe')
    assert _reported('X_BOOL').status == 'rejected'
    assert warns == []  # reporting never logs


def test_report_blank_owner_row_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('EVALUATORQ_FINDER_LIMIT', ' ')
    assert _reported('EVALUATORQ_FINDER_LIMIT').status == 'unset'  # effective_settings skips a blank value
    monkeypatch.setenv('EVALUATORQ_FINDER_LIMIT', 'abc')
    assert _reported('EVALUATORQ_FINDER_LIMIT').status == 'rejected'


@pytest.mark.parametrize(('name', 'reader'), [('X_INT', env_int), ('X_FLOAT', env_float), ('X_BOOL', env_bool)])
@pytest.mark.parametrize('raw', ['', ' ', '5', '5.5', 'abc', 'nan', '-1', '1e9', 'yes', 'off'])
def test_report_rejects_exactly_what_the_reader_warns_on(
    monkeypatch: pytest.MonkeyPatch,
    warns: list[str],
    name: str,
    reader,
    raw: str,  # noqa: ANN001
) -> None:
    monkeypatch.setenv(name, raw)
    reader(name)
    assert (_reported(name).status == 'rejected') == bool(warns), raw


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

    owner = 'evaluatorq.common.model_roles.role_model'
    for role, name in ROLE_ENV.items():
        assert ENV_VARS[name].default == BUILTIN[role]
    owned = {name for name, row in ENV_VARS.items() if row.owner == owner}
    assert owned == set(ROLE_ENV.values()) | set(LEGACY_TASK_ENV.values())


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
    assert Path(str(ENV_VARS['EVALUATORQ_DASHBOARD_SETTINGS'].default)) == settings_path()


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
}


def _docs_rows() -> dict[str, str]:
    text = _DOCS.read_text(encoding='utf-8')
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
    assert len(_PROSE_DEFAULTS) == 5


# --- call sites that moved from ``os.environ.get(name, '').strip()`` to ``env_text`` ---
# Each read used the value only for truthiness, so ``''`` becoming ``None`` must change nothing:
# blank and whitespace still read as unset, and a padded value is still stripped.
_BLANK = [None, '', '   ']


def _set(monkeypatch: pytest.MonkeyPatch, name: str, value: str | None) -> None:
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)


@pytest.mark.parametrize('value', _BLANK)
def test_settings_path_treats_blank_as_unset(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    from evaluatorq.trace_finder.settings import settings_path

    _set(monkeypatch, 'EVALUATORQ_DASHBOARD_SETTINGS', value)
    assert settings_path() == Path('.evaluatorq/dashboard-settings.json')


def test_settings_path_strips_the_value(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.trace_finder.settings import settings_path

    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', '  custom.json  ')
    assert settings_path() == Path('custom.json')


@pytest.mark.parametrize(('value', 'expected'), [(None, None), ('', None), ('   ', None), (' 42 ', 42)])
def test_finder_env_override_blank_keeps_the_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: str | None, expected: int | None
) -> None:
    from evaluatorq.trace_finder.settings import DashboardSettings, effective_settings

    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(tmp_path / 'missing.json'))
    _set(monkeypatch, 'EVALUATORQ_FINDER_LIMIT', value)
    assert effective_settings().limit == (DashboardSettings().limit if expected is None else expected)


@pytest.mark.parametrize(('value', 'expected'), [(None, None), ('', None), ('   ', None), ('  key  ', 'key')])
def test_environment_auth_strips_the_key(
    monkeypatch: pytest.MonkeyPatch, value: str | None, expected: str | None
) -> None:
    from evaluatorq.dashboard.auth import resolve_dashboard_auth
    from evaluatorq.trace_finder.settings import DashboardSettings

    _set(monkeypatch, 'ORQ_API_KEY', value)
    assert resolve_dashboard_auth(DashboardSettings(orq_auth_method='environment')).api_key == expected


@pytest.mark.parametrize(('value', 'session'), [(None, True), ('', True), ('   ', True), ('  key  ', False)])
def test_workspace_slug_blank_key_uses_the_cli_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: str | None, session: bool
) -> None:
    from evaluatorq.dashboard import orq_workspace
    from evaluatorq.trace_finder.settings import credential_fingerprint

    calls: list[tuple[str | None, str | None, bool]] = []

    def fake_cli_slug(profile: str | None, fingerprint: str | None, *, use_cli_session: bool = False) -> None:
        calls.append((profile, fingerprint, use_cli_session))

    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(tmp_path / 'missing.json'))
    for name in ('ORQ_WORKSPACE', 'ORQ_WORKSPACE_SLUG', 'ORQ_BASE_URL'):
        monkeypatch.delenv(name, raising=False)
    _set(monkeypatch, 'ORQ_API_KEY', value)
    monkeypatch.setattr(orq_workspace.shutil, 'which', lambda _: '/usr/bin/orq')
    monkeypatch.setattr(orq_workspace, '_cli_slug', fake_cli_slug)
    orq_workspace.resolve_slug()
    expected = (None, None, True) if session else (None, credential_fingerprint('key', None), False)
    assert calls == [expected]


def test_portable_encryption_key_is_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    fernet = pytest.importorskip('cryptography.fernet')
    from evaluatorq.trace_finder.secure_credentials import _encryption_key

    key = fernet.Fernet.generate_key()
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_KEY_ENCRYPTION_KEY', f'  {key.decode()}  ')
    _encryption_key.cache_clear()
    try:
        assert _encryption_key() == key
    finally:
        _encryption_key.cache_clear()


def test_public_base_url_aliases_are_the_canonical_default() -> None:
    from evaluatorq.dashboard.orq_workspace import DEFAULT_BASE_URL
    from evaluatorq.redteam.backends.registry import ORQ_DEFAULT_BASE_URL

    assert DEFAULT_BASE_URL == ORQ_DEFAULT_BASE_URL == env_config.DEFAULT_ORQ_BASE_URL
