"""Tests for the env-config helpers and the recommendation limits they back."""

from __future__ import annotations

import importlib

import pytest

from evaluatorq.common.env_config import env_float, env_int


def test_env_int_unset_uses_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv('EVALUATORQ_TEST_INT', raising=False)
    assert env_int('EVALUATORQ_TEST_INT', 7) == 7


def test_env_int_reads_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('EVALUATORQ_TEST_INT', '42')
    assert env_int('EVALUATORQ_TEST_INT', 7) == 42


def test_env_int_invalid_falls_back(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('EVALUATORQ_TEST_INT', 'not-a-number')
    assert env_int('EVALUATORQ_TEST_INT', 7) == 7


def test_env_float_reads_override_and_falls_back(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('EVALUATORQ_TEST_FLOAT', '0.9')
    assert env_float('EVALUATORQ_TEST_FLOAT', 0.5) == 0.9
    monkeypatch.setenv('EVALUATORQ_TEST_FLOAT', 'x')
    assert env_float('EVALUATORQ_TEST_FLOAT', 0.5) == 0.5


def test_recommendation_limits_honor_env(monkeypatch: pytest.MonkeyPatch):
    """The module constants (read at import) reflect the env override on reload."""
    monkeypatch.setenv('EVALUATORQ_RECOMMENDATION_MAX_SUGGESTIONS', '5')
    monkeypatch.setenv('EVALUATORQ_RECOMMENDATION_MAX_TRANSCRIPT_CHARS', '1000')
    monkeypatch.setenv('EVALUATORQ_RECOMMENDATION_FACTUAL_ACCURACY_BELOW', '0.3')
    from evaluatorq.simulation.reports import recommendations

    reloaded = importlib.reload(recommendations)
    try:
        assert reloaded._MAX_SUGGESTIONS == 5
        assert reloaded._MAX_TRANSCRIPT_CHARS == 1000
        assert reloaded.FACTUAL_ACCURACY_BELOW == 0.3
    finally:
        # Restore module-level defaults so import order does not leak into other tests.
        monkeypatch.undo()
        importlib.reload(reloaded)
