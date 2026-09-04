"""Tests for how agents/base resolves its numeric env overrides.

These now route through the shared ``common.env_config`` contract: unset falls back to the default,
a valid value is used, and a MISCONFIGURED value warns and falls back to the default instead of
raising. That non-fatal fallback is a deliberate behavior change from the old private readers, which
raised a ``ValueError`` on a bad ``EVALUATORQ_LLM_TIMEOUT_S`` / ``EVALUATORQ_LLM_MAX_TOKENS``.
"""

from __future__ import annotations

import pytest

from evaluatorq.contracts import DEFAULT_TARGET_MAX_TOKENS
from evaluatorq.simulation.agents.base import _default_max_tokens, _default_timeout_s


def test_max_tokens_unset_uses_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVALUATORQ_LLM_MAX_TOKENS", raising=False)
    assert _default_max_tokens() == DEFAULT_TARGET_MAX_TOKENS


def test_max_tokens_parses_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVALUATORQ_LLM_MAX_TOKENS", "4096")
    assert _default_max_tokens() == 4096


def test_max_tokens_garbage_warns_and_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVALUATORQ_LLM_MAX_TOKENS", "abc")
    assert _default_max_tokens() == DEFAULT_TARGET_MAX_TOKENS  # no longer raises


def test_timeout_unset_and_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVALUATORQ_LLM_TIMEOUT_S", raising=False)
    assert _default_timeout_s() == 60.0
    monkeypatch.setenv("EVALUATORQ_LLM_TIMEOUT_S", "30.5")
    assert _default_timeout_s() == 30.5


def test_timeout_garbage_warns_and_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVALUATORQ_LLM_TIMEOUT_S", "60s")
    assert _default_timeout_s() == 60.0  # no longer raises
