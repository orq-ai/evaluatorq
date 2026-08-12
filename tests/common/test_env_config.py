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


def test_env_int_below_minimum_falls_back(monkeypatch: pytest.MonkeyPatch):
    """Review on PR #141: a parseable value is not necessarily a meaningful one."""
    monkeypatch.setenv('EVALUATORQ_TEST_INT', '0')
    assert env_int('EVALUATORQ_TEST_INT', 3, min_value=1) == 3
    monkeypatch.setenv('EVALUATORQ_TEST_INT', '-20')
    assert env_int('EVALUATORQ_TEST_INT', 3000, min_value=100) == 3000
    monkeypatch.setenv('EVALUATORQ_TEST_INT', '100')
    assert env_int('EVALUATORQ_TEST_INT', 3000, min_value=100) == 100


def test_env_float_rejects_non_finite_and_out_of_range(monkeypatch: pytest.MonkeyPatch):
    """inf makes every below-threshold check True, nan makes every comparison
    False; neither can mean anything for a 0-1 threshold."""
    for bad in ('inf', '-inf', 'nan', '1.5', '-0.1'):
        monkeypatch.setenv('EVALUATORQ_TEST_FLOAT', bad)
        assert env_float('EVALUATORQ_TEST_FLOAT', 0.5, min_value=0.0, max_value=1.0) == 0.5
    monkeypatch.setenv('EVALUATORQ_TEST_FLOAT', '0.0')
    assert env_float('EVALUATORQ_TEST_FLOAT', 0.5, min_value=0.0, max_value=1.0) == 0.0
    monkeypatch.setenv('EVALUATORQ_TEST_FLOAT', '1.0')
    assert env_float('EVALUATORQ_TEST_FLOAT', 0.5, min_value=0.0, max_value=1.0) == 1.0


def test_all_three_thresholds_honor_env_and_reject_nonsense(monkeypatch: pytest.MonkeyPatch):
    """All three trigger thresholds are asserted (the original test covered one),
    and nonsense values fall back to defaults instead of disabling triggers."""
    monkeypatch.setenv('EVALUATORQ_RECOMMENDATION_FACTUAL_ACCURACY_BELOW', '0.3')
    monkeypatch.setenv('EVALUATORQ_RECOMMENDATION_HALLUCINATION_RISK_ABOVE', '0.7')
    monkeypatch.setenv('EVALUATORQ_RECOMMENDATION_TONE_APPROPRIATENESS_BELOW', 'nan')
    monkeypatch.setenv('EVALUATORQ_RECOMMENDATION_MAX_SUGGESTIONS', '0')
    monkeypatch.setenv('EVALUATORQ_RECOMMENDATION_MAX_TRANSCRIPT_CHARS', '-20')
    from evaluatorq.simulation.reports import recommendations

    reloaded = importlib.reload(recommendations)
    try:
        assert reloaded.FACTUAL_ACCURACY_BELOW == 0.3
        assert reloaded.HALLUCINATION_RISK_ABOVE == 0.7
        assert reloaded.TONE_APPROPRIATENESS_BELOW == 0.5  # nan rejected
        assert reloaded._MAX_SUGGESTIONS == 3  # 0 rejected: would pay for the call and drop everything
        assert reloaded._MAX_TRANSCRIPT_CHARS == 3000  # negative rejected: truncates from the wrong end
    finally:
        monkeypatch.undo()
        importlib.reload(reloaded)
