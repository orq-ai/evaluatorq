"""Tests for resolve_results_base_url — the upload host resolver (RES-912)."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from evaluatorq.common.llm_client import (
    LLM_CREDENTIAL_KEYS,
    ORQ_DEFAULT_HOST,
    MissingLLMCredentialsError,
    ModelClientMismatchError,
    configured_llm_provider,
    is_anthropic_client,
    require_models_served_by_provider,
    resolve_llm_client,
    resolve_orq_credential,
    resolve_results_base_url,
)

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


# --- ANTHROPIC_API_KEY resolution (RES-1875) ---------------------------------


def _only_env(monkeypatch: pytest.MonkeyPatch, **keep: str) -> None:
    for name in LLM_CREDENTIAL_KEYS:
        monkeypatch.delenv(name, raising=False)
    for name, value in keep.items():
        monkeypatch.setenv(name, value)


def test_anthropic_only_env_resolves_an_owned_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.anthropic import AnthropicLLMClient

    _only_env(monkeypatch, ANTHROPIC_API_KEY='sk-ant-test')
    resolved = resolve_llm_client(max_retries=5)
    assert isinstance(resolved.client, AnthropicLLMClient)
    assert resolved.client.max_retries == 5
    assert resolved.owned is True
    assert resolved.routes_through_orq is False


def test_anthropic_branch_uses_anthropics_own_default_retries_when_none_is_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anthropic import DEFAULT_MAX_RETRIES

    _only_env(monkeypatch, ANTHROPIC_API_KEY='sk-ant-test')
    assert resolve_llm_client().client.max_retries == DEFAULT_MAX_RETRIES


def test_anthropic_base_url_reaches_the_built_client(monkeypatch: pytest.MonkeyPatch) -> None:
    _only_env(monkeypatch, ANTHROPIC_API_KEY='sk-ant-test')
    monkeypatch.setenv('ANTHROPIC_BASE_URL', 'https://proxy.example.test')
    assert str(resolve_llm_client().client.base_url).rstrip('/') == 'https://proxy.example.test'


def test_orq_and_openai_keys_still_win_over_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    _only_env(monkeypatch, ANTHROPIC_API_KEY='sk-ant-test', OPENAI_API_KEY='sk-openai-test')
    assert not is_anthropic_client(resolve_llm_client().client)
    _only_env(monkeypatch, ANTHROPIC_API_KEY='sk-ant-test', OPENAI_API_KEY='sk-openai-test', ORQ_API_KEY='orq-test')
    resolved = resolve_llm_client()
    assert resolved.routes_through_orq is True
    assert not is_anthropic_client(resolved.client)


def test_require_orq_never_falls_through_to_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    _only_env(monkeypatch, ANTHROPIC_API_KEY='sk-ant-test')
    with pytest.raises(MissingLLMCredentialsError, match='ANTHROPIC_API_KEY'):
        resolve_llm_client(require_orq=True)


def test_missing_credentials_error_names_all_three_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    _only_env(monkeypatch)
    with pytest.raises(MissingLLMCredentialsError) as info:
        resolve_llm_client()
    for key in LLM_CREDENTIAL_KEYS:
        assert key in str(info.value)


def test_anthropic_key_without_the_extra_raises_the_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    _only_env(monkeypatch, ANTHROPIC_API_KEY='sk-ant-test')
    monkeypatch.setitem(sys.modules, 'anthropic', None)
    with pytest.raises(MissingLLMCredentialsError, match=r'evaluatorq\[anthropic\]') as info:
        resolve_llm_client()
    assert isinstance(info.value.__cause__, ImportError)


@pytest.mark.parametrize(
    ('keys', 'expected'),
    [
        ({}, None),
        ({'ORQ_API_KEY': 'o'}, 'orq'),
        ({'OPENAI_API_KEY': 'o'}, 'openai'),
        ({'ANTHROPIC_API_KEY': 'a'}, 'anthropic'),
        ({'ORQ_API_KEY': 'o', 'OPENAI_API_KEY': 'o', 'ANTHROPIC_API_KEY': 'a'}, 'orq'),
        ({'OPENAI_API_KEY': 'o', 'ANTHROPIC_API_KEY': 'a'}, 'openai'),
        ({'ORQ_API_KEY': 'o', 'ANTHROPIC_API_KEY': 'a'}, 'orq'),
    ],
)
def test_configured_llm_provider_follows_the_credential_precedence(
    monkeypatch: pytest.MonkeyPatch, keys: dict[str, str], expected: str | None
) -> None:
    _only_env(monkeypatch, **keys)
    assert configured_llm_provider() == expected


def test_configured_llm_provider_honours_require_orq_and_an_explicit_orq_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _only_env(monkeypatch, OPENAI_API_KEY='o', ANTHROPIC_API_KEY='a')
    assert configured_llm_provider(require_orq=True) is None
    assert configured_llm_provider(extra_api_key='orq-explicit', require_orq=True) == 'orq'
    assert configured_llm_provider(extra_api_key='orq-explicit') == 'orq'


def test_credential_keys_are_listed_in_precedence_order() -> None:
    assert LLM_CREDENTIAL_KEYS == ('ORQ_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY')


def test_models_the_anthropic_client_cannot_serve_stop_the_run_naming_each(monkeypatch: pytest.MonkeyPatch) -> None:
    _only_env(monkeypatch, ANTHROPIC_API_KEY='sk-ant-test')
    with pytest.raises(ModelClientMismatchError) as info:
        require_models_served_by_provider({
            'EVALUATORQ_SMART_MODEL': 'openai/gpt-6-sol',
            'EVALUATORQ_FAST_MODEL': 'claude-haiku-5-5',
            'EVALUATORQ_CLASSIFIER_MODEL': 'typesafe/jev-latest',
        })
    message = str(info.value)
    assert 'ANTHROPIC_API_KEY is the only LLM key set' in message
    assert 'EVALUATORQ_SMART_MODEL=openai/gpt-6-sol' in message
    assert 'EVALUATORQ_CLASSIFIER_MODEL=typesafe/jev-latest' in message
    assert 'EVALUATORQ_FAST_MODEL' not in message
    assert 'ORQ_API_KEY or OPENAI_API_KEY' in message


def test_claude_ids_and_other_providers_pass_the_model_check(monkeypatch: pytest.MonkeyPatch) -> None:
    _only_env(monkeypatch, ANTHROPIC_API_KEY='sk-ant-test')
    require_models_served_by_provider({'EVALUATORQ_SMART_MODEL': 'claude-opus-5-5', 'x': 'anthropic/claude-sonnet-5-5'})
    _only_env(monkeypatch, OPENAI_API_KEY='sk-openai-test', ANTHROPIC_API_KEY='sk-ant-test')
    require_models_served_by_provider({'EVALUATORQ_SMART_MODEL': 'openai/gpt-6-sol'})
