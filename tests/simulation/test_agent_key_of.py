"""Unit tests for `_agent_key_of()` — the agent-key resolution helper.

Covers all three branches:
  (a) target has an `.agent_key` attribute -> return it
  (b) target's `config.model` looks like 'agent/<key>' -> parse and return <key>
  (c) neither -> return None
plus the edge case where `config.model == 'agent/'` (empty key -> None).
"""
from __future__ import annotations

from types import SimpleNamespace

from evaluatorq.simulation.api import _agent_key_of


def test_agent_key_of_returns_agent_key_attribute_when_present() -> None:
    target = SimpleNamespace(agent_key="simple-agent-1")

    assert _agent_key_of(target) == "simple-agent-1"


def test_agent_key_of_parses_config_model_agent_prefix() -> None:
    target = SimpleNamespace(config=SimpleNamespace(model="agent/simple-agent-1"))

    assert _agent_key_of(target) == "simple-agent-1"


def test_agent_key_of_returns_none_when_neither_present() -> None:
    target = SimpleNamespace(config=SimpleNamespace(model="openai/gpt-4o"))

    assert _agent_key_of(target) is None


def test_agent_key_of_returns_none_when_no_config_or_agent_key() -> None:
    target = object()

    assert _agent_key_of(target) is None


def test_agent_key_of_returns_none_for_empty_agent_key_suffix() -> None:
    target = SimpleNamespace(config=SimpleNamespace(model="agent/"))

    assert _agent_key_of(target) is None
