"""Tests for the env-var parsing helpers in agents/base.

A misconfigured numeric env var should fail with a message naming the offending
variable, not a bare ``ValueError`` from ``int()``/``float()``.
"""

from __future__ import annotations

import pytest

from evaluatorq.simulation.agents.base import _env_float, _env_int


def test_env_int_returns_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVALUATORQ_TEST_INT", raising=False)
    assert _env_int("EVALUATORQ_TEST_INT", 8192) == 8192


def test_env_int_parses_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVALUATORQ_TEST_INT", "4096")
    assert _env_int("EVALUATORQ_TEST_INT", 8192) == 4096


def test_env_int_raises_named_error_on_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVALUATORQ_TEST_INT", "abc")
    with pytest.raises(ValueError, match="EVALUATORQ_TEST_INT"):
        _env_int("EVALUATORQ_TEST_INT", 8192)


def test_env_float_returns_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVALUATORQ_TEST_FLOAT", raising=False)
    assert _env_float("EVALUATORQ_TEST_FLOAT", 60.0) == 60.0


def test_env_float_raises_named_error_on_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVALUATORQ_TEST_FLOAT", "60s")
    with pytest.raises(ValueError, match="EVALUATORQ_TEST_FLOAT"):
        _env_float("EVALUATORQ_TEST_FLOAT", 60.0)
