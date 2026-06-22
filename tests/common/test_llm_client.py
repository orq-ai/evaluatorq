"""Tests for resolve_results_base_url — the upload host resolver (RES-912)."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from evaluatorq.common.llm_client import ORQ_DEFAULT_HOST, resolve_results_base_url


def test_prefers_orq_routed_client_host_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """An Orq-routed inference client wins over ORQ_BASE_URL so results land on
    the same server inference used, even when the env points elsewhere."""
    monkeypatch.setenv("ORQ_BASE_URL", "https://my.orq.ai")
    client = SimpleNamespace(base_url="https://my.staging.orq.ai/v3/router")
    assert resolve_results_base_url(client) == "https://my.staging.orq.ai"


def test_strips_router_suffix_with_trailing_slash() -> None:
    client = SimpleNamespace(base_url="https://my.staging.orq.ai/v3/router/")
    assert resolve_results_base_url(client) == "https://my.staging.orq.ai"


def test_falls_back_to_env_when_no_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORQ_BASE_URL", "https://my.staging.orq.ai")
    assert resolve_results_base_url(None) == "https://my.staging.orq.ai"


def test_falls_back_to_default_when_no_client_and_no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ORQ_BASE_URL", raising=False)
    assert resolve_results_base_url(None) == ORQ_DEFAULT_HOST


def test_non_orq_client_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A direct-OpenAI inference client must not redirect uploads at OpenAI;
    uploads always go to Orq, resolved from env."""
    monkeypatch.setenv("ORQ_BASE_URL", "https://my.staging.orq.ai")
    client = SimpleNamespace(base_url="https://api.openai.com/v1")
    assert resolve_results_base_url(client) == "https://my.staging.orq.ai"
