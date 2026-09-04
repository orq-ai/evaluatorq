"""Tests for the shared env-var reader contract (common/env_config).

Covers the return value AND that a WARNING is actually emitted on every invalid case, since a
silent misconfiguration is the failure this reader exists to prevent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from loguru import logger

from evaluatorq.common.env_config import env_bool, env_float, env_int

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ('X_INT', 'X_FLOAT', 'X_BOOL'):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def warns() -> Iterator[list[str]]:
    """Capture loguru WARNING messages (loguru does not feed pytest's caplog)."""
    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(m.record['message']), level='WARNING')
    yield messages
    logger.remove(sink_id)


# --- env_int ---
def test_env_int_unset_is_silent(warns: list[str]) -> None:
    assert env_int('X_INT', 7) == 7
    assert warns == []  # a truly-absent variable is not a misconfiguration


def test_env_int_empty_warns(monkeypatch: pytest.MonkeyPatch, warns: list[str]) -> None:
    monkeypatch.setenv('X_INT', '')
    assert env_int('X_INT', 7) == 7
    assert any('set but empty' in m for m in warns)  # unresolved CI ${{ vars.X }} must signal


def test_env_int_whitespace_only_warns(monkeypatch: pytest.MonkeyPatch, warns: list[str]) -> None:
    monkeypatch.setenv('X_INT', '   ')
    assert env_int('X_INT', 7) == 7
    assert any('set but empty' in m for m in warns)


def test_env_int_valid_and_stripped(monkeypatch: pytest.MonkeyPatch, warns: list[str]) -> None:
    monkeypatch.setenv('X_INT', '42')
    assert env_int('X_INT', 7) == 42
    monkeypatch.setenv('X_INT', '  42  ')  # surrounding whitespace tolerated
    assert env_int('X_INT', 7) == 42
    assert warns == []  # a valid value warns about nothing


def test_env_int_invalid_warns_and_defaults(monkeypatch: pytest.MonkeyPatch, warns: list[str]) -> None:
    monkeypatch.setenv('X_INT', 'notanint')
    assert env_int('X_INT', 7) == 7  # never raises
    assert any('not an integer' in m for m in warns)


def test_env_int_out_of_range_warns_and_defaults(monkeypatch: pytest.MonkeyPatch, warns: list[str]) -> None:
    monkeypatch.setenv('X_INT', '0')
    assert env_int('X_INT', 7, min_value=1) == 7  # replaces the old "must be positive" check
    monkeypatch.setenv('X_INT', '999')
    assert env_int('X_INT', 7, max_value=100) == 7
    assert sum('must be' in m for m in warns) == 2


# --- env_float ---
def test_env_float_valid_invalid_range(monkeypatch: pytest.MonkeyPatch, warns: list[str]) -> None:
    monkeypatch.setenv('X_FLOAT', '1.5')
    assert env_float('X_FLOAT', 2.0) == 1.5
    monkeypatch.setenv('X_FLOAT', 'nope')
    assert env_float('X_FLOAT', 2.0) == 2.0
    monkeypatch.setenv('X_FLOAT', '-1')
    assert env_float('X_FLOAT', 2.0, min_value=0.0) == 2.0
    assert any('not a number' in m for m in warns)
    assert any('must be >=' in m for m in warns)


@pytest.mark.parametrize('raw', ['nan', 'inf', '-inf', 'Infinity'])
def test_env_float_rejects_non_finite(monkeypatch: pytest.MonkeyPatch, warns: list[str], raw: str) -> None:
    monkeypatch.setenv('X_FLOAT', raw)
    assert env_float('X_FLOAT', 2.0) == 2.0  # float() would accept these; the reader must not
    assert any('finite' in m for m in warns)


# --- env_bool ---
@pytest.mark.parametrize(('raw', 'expected'), [('1', True), ('true', True), ('YES', True), ('on', True), ('0', False), ('false', False), ('no', False), ('OFF', False)])
def test_env_bool_recognised(monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool) -> None:
    monkeypatch.setenv('X_BOOL', raw)
    assert env_bool('X_BOOL', default=not expected) is expected


def test_env_bool_unset_is_silent(warns: list[str]) -> None:
    assert env_bool('X_BOOL', default=True) is True
    assert warns == []


def test_env_bool_unrecognised_warns_and_defaults(monkeypatch: pytest.MonkeyPatch, warns: list[str]) -> None:
    monkeypatch.setenv('X_BOOL', 'maybe')
    assert env_bool('X_BOOL', default=True) is True  # unrecognised -> warn + default, never raises
    assert any('not a boolean' in m for m in warns)
