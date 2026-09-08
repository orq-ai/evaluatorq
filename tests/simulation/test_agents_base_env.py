"""Tests for how agents/base resolves its numeric env overrides.

These now route through the shared ``common.env_config`` contract: unset falls back to the default,
a valid value is used, and a MISCONFIGURED value warns and falls back to the default instead of
raising. That non-fatal fallback is a deliberate behavior change from the old private readers, which
raised a ``ValueError`` on a bad ``EVALUATORQ_LLM_TIMEOUT_S`` / ``EVALUATORQ_LLM_MAX_TOKENS``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from loguru import logger

from evaluatorq.contracts import DEFAULT_TARGET_MAX_TOKENS
from evaluatorq.simulation.agents.base import _default_max_tokens, _default_timeout_s

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def warns() -> Iterator[list[str]]:
    """Capture loguru WARNING messages (loguru does not feed pytest's caplog)."""
    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    yield messages
    logger.remove(sink_id)


def test_max_tokens_unset_uses_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVALUATORQ_LLM_MAX_TOKENS", raising=False)
    assert _default_max_tokens() == DEFAULT_TARGET_MAX_TOKENS


def test_max_tokens_parses_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVALUATORQ_LLM_MAX_TOKENS", "4096")
    assert _default_max_tokens() == 4096


def test_max_tokens_garbage_warns_and_defaults(monkeypatch: pytest.MonkeyPatch, warns: list[str]) -> None:
    monkeypatch.setenv("EVALUATORQ_LLM_MAX_TOKENS", "abc")
    assert _default_max_tokens() == DEFAULT_TARGET_MAX_TOKENS  # no longer raises
    assert any("not an integer" in m for m in warns)


def test_timeout_unset_and_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVALUATORQ_LLM_TIMEOUT_S", raising=False)
    assert _default_timeout_s() == 60.0
    monkeypatch.setenv("EVALUATORQ_LLM_TIMEOUT_S", "30.5")
    assert _default_timeout_s() == 30.5


def test_timeout_garbage_warns_and_defaults(monkeypatch: pytest.MonkeyPatch, warns: list[str]) -> None:
    monkeypatch.setenv("EVALUATORQ_LLM_TIMEOUT_S", "60s")
    assert _default_timeout_s() == 60.0  # no longer raises
    assert any("not a number" in m for m in warns)


@pytest.mark.parametrize("raw", ["0", "-5"])
def test_max_tokens_below_min_warns_and_defaults(monkeypatch: pytest.MonkeyPatch, warns: list[str], raw: str) -> None:
    monkeypatch.setenv("EVALUATORQ_LLM_MAX_TOKENS", raw)
    assert _default_max_tokens() == DEFAULT_TARGET_MAX_TOKENS  # min_value=1 keeps it off the provider
    assert any("must be >=" in m for m in warns)


@pytest.mark.parametrize("raw", ["0", "-5", "nan"])
def test_timeout_below_min_warns_and_defaults(monkeypatch: pytest.MonkeyPatch, warns: list[str], raw: str) -> None:
    monkeypatch.setenv("EVALUATORQ_LLM_TIMEOUT_S", raw)
    assert _default_timeout_s() == 60.0
    assert warns  # "must be >=" for the numbers, "not a finite number" for nan
