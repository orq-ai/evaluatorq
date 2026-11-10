"""Tests for resolve_results_base_url — the upload host resolver (RES-912)."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from evaluatorq.common.llm_client import ORQ_DEFAULT_HOST, resolve_results_base_url

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


# --- ANTHROPIC_API_KEY resolution (RES-1875) ---------------------------------


def _only_env(monkeypatch: pytest.MonkeyPatch, **keep: str) -> None:
    for name in ('ORQ_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY'):
        monkeypatch.delenv(name, raising=False)
    for name, value in keep.items():
        monkeypatch.setenv(name, value)


def test_anthropic_only_env_resolves_an_owned_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.anthropic import AnthropicLLMClient
    from evaluatorq.common.llm_client import resolve_llm_client

    _only_env(monkeypatch, ANTHROPIC_API_KEY='sk-ant-test')
    resolved = resolve_llm_client(max_retries=5)
    assert isinstance(resolved.client, AnthropicLLMClient)
    assert resolved.client.max_retries == 5
    assert resolved.owned is True
    assert resolved.routes_through_orq is False


def test_anthropic_branch_uses_the_sdk_default_retries_when_none_is_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    from openai import DEFAULT_MAX_RETRIES

    from evaluatorq.common.llm_client import resolve_llm_client

    _only_env(monkeypatch, ANTHROPIC_API_KEY='sk-ant-test')
    assert resolve_llm_client().client.max_retries == DEFAULT_MAX_RETRIES


def test_orq_and_openai_keys_still_win_over_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.common.llm_client import is_anthropic_client, resolve_llm_client

    _only_env(monkeypatch, ANTHROPIC_API_KEY='sk-ant-test', OPENAI_API_KEY='sk-openai-test')
    assert not is_anthropic_client(resolve_llm_client().client)
    _only_env(monkeypatch, ANTHROPIC_API_KEY='sk-ant-test', OPENAI_API_KEY='sk-openai-test', ORQ_API_KEY='orq-test')
    resolved = resolve_llm_client()
    assert resolved.routes_through_orq is True
    assert not is_anthropic_client(resolved.client)


def test_require_orq_never_falls_through_to_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.common.llm_client import MissingLLMCredentialsError, resolve_llm_client

    _only_env(monkeypatch, ANTHROPIC_API_KEY='sk-ant-test')
    with pytest.raises(MissingLLMCredentialsError, match='ANTHROPIC_API_KEY'):
        resolve_llm_client(require_orq=True)


def test_missing_credentials_error_names_all_three_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.common.llm_client import MissingLLMCredentialsError, resolve_llm_client

    _only_env(monkeypatch)
    with pytest.raises(MissingLLMCredentialsError) as info:
        resolve_llm_client()
    for key in ('ORQ_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY'):
        assert key in str(info.value)


def test_anthropic_key_without_the_extra_raises_the_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from evaluatorq.common.llm_client import resolve_llm_client

    _only_env(monkeypatch, ANTHROPIC_API_KEY='sk-ant-test')
    monkeypatch.setitem(sys.modules, 'anthropic', None)
    with pytest.raises(ImportError, match=r'evaluatorq\[anthropic\]'):
        resolve_llm_client()
