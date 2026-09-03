"""Tests for the shared env-var reader contract (common/env_config)."""

from __future__ import annotations

import pytest

from evaluatorq.common.env_config import env_bool, env_float, env_int


@pytest.fixture(autouse=True)
def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ('X_INT', 'X_FLOAT', 'X_BOOL'):
        monkeypatch.delenv(name, raising=False)


# --- env_int ---
def test_env_int_unset_and_empty_use_default(monkeypatch: pytest.MonkeyPatch) -> None:
    assert env_int('X_INT', 7) == 7
    monkeypatch.setenv('X_INT', '')
    assert env_int('X_INT', 7) == 7


def test_env_int_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('X_INT', '42')
    assert env_int('X_INT', 7) == 42


def test_env_int_invalid_warns_and_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('X_INT', 'notanint')
    assert env_int('X_INT', 7) == 7  # never raises


def test_env_int_out_of_range_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('X_INT', '0')
    assert env_int('X_INT', 7, min_value=1) == 7  # replaces the old "must be positive" check
    monkeypatch.setenv('X_INT', '999')
    assert env_int('X_INT', 7, max_value=100) == 7


# --- env_float ---
def test_env_float_valid_invalid_range(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('X_FLOAT', '1.5')
    assert env_float('X_FLOAT', 2.0) == 1.5
    monkeypatch.setenv('X_FLOAT', 'nope')
    assert env_float('X_FLOAT', 2.0) == 2.0
    monkeypatch.setenv('X_FLOAT', '-1')
    assert env_float('X_FLOAT', 2.0, min_value=0.0) == 2.0


# --- env_bool ---
@pytest.mark.parametrize(('raw', 'expected'), [('1', True), ('true', True), ('YES', True), ('on', True), ('0', False), ('false', False), ('no', False), ('OFF', False)])
def test_env_bool_recognised(monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool) -> None:
    monkeypatch.setenv('X_BOOL', raw)
    assert env_bool('X_BOOL', default=not expected) is expected


def test_env_bool_unset_and_unrecognised_use_default(monkeypatch: pytest.MonkeyPatch) -> None:
    assert env_bool('X_BOOL', default=True) is True
    monkeypatch.setenv('X_BOOL', 'maybe')
    assert env_bool('X_BOOL', default=True) is True  # unrecognised -> warn + default, never raises
