"""Tests for resolve_results_base_url — the upload host resolver (RES-912)."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from evaluatorq.common.llm_client import ORQ_DEFAULT_HOST, resolve_orq_credential, resolve_results_base_url

if TYPE_CHECKING:
    from openai import AsyncOpenAI


def _client(base_url: str) -> AsyncOpenAI:
    """A duck-typed stand-in: resolve_results_base_url only reads ``.base_url``."""
    return cast("AsyncOpenAI", cast(object, SimpleNamespace(base_url=base_url)))


def test_prefers_orq_routed_client_host_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """An Orq-routed inference client wins over ORQ_BASE_URL so results land on
    the same server inference used, even when the env points elsewhere."""
    monkeypatch.setenv("ORQ_BASE_URL", "https://my.orq.ai")
    assert resolve_results_base_url(_client("https://my.staging.orq.ai/v3/router")) == "https://my.staging.orq.ai"


def test_strips_router_suffix_with_trailing_slash() -> None:
    assert resolve_results_base_url(_client("https://my.staging.orq.ai/v3/router/")) == "https://my.staging.orq.ai"


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
    assert resolve_results_base_url(_client("https://api.openai.com/v1")) == "https://my.staging.orq.ai"


def _keyed_client(base_url: str, api_key: str) -> AsyncOpenAI:
    return cast("AsyncOpenAI", cast(object, SimpleNamespace(base_url=base_url, api_key=api_key)))


def test_orq_credential_uses_an_orq_routed_clients_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORQ_API_KEY", "env-key")
    assert resolve_orq_credential(_keyed_client("https://my.staging.orq.ai/v3/router", "client-key")) == "client-key"


@pytest.mark.parametrize("base_url", ["https://api.openai.com/v1", "https://api.anthropic.com"])
def test_orq_credential_never_returns_a_provider_clients_key(monkeypatch: pytest.MonkeyPatch, base_url: str) -> None:
    """A direct provider client's key is not an Orq key, so it falls back to ORQ_API_KEY or nothing."""
    client = _keyed_client(base_url, "sk-provider-secret")
    monkeypatch.setenv("ORQ_API_KEY", "env-key")
    assert resolve_orq_credential(client) == "env-key"
    monkeypatch.delenv("ORQ_API_KEY")
    assert resolve_orq_credential(client) is None
