"""Retry is wired at the client-construction seam, not per call site.

Pipeline LLM calls rely on the OpenAI SDK's built-in retry (429/5xx/network,
Retry-After honored), driven by ``LLMConfig.retry_count`` through
``resolve_llm_client(max_retries=...)``. These tests pin that plumbing so a
refactor cannot silently revert a client to a different retry budget, and that
the Orq-SDK backend derives an equivalent client-level ``RetryConfig``.
"""

from __future__ import annotations

import pytest

from evaluatorq.common.llm_client import resolve_llm_client
from evaluatorq.redteam.backends.registry import create_async_llm_client
from evaluatorq.redteam.contracts import LLMConfig


@pytest.fixture(autouse=True)
def _orq_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ORQ_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_resolve_llm_client_passes_max_retries() -> None:
    client = resolve_llm_client(max_retries=4).client
    assert client.max_retries == 4


def test_resolve_llm_client_default_keeps_sdk_retries() -> None:
    """Unset means the SDK default (2 retries) — never zero, never a third layer."""
    client = resolve_llm_client().client
    assert client.max_retries == 2


def test_resolve_llm_client_openai_branch_passes_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ORQ_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = resolve_llm_client(max_retries=5).client
    assert client.max_retries == 5


def test_injected_client_is_never_reconfigured() -> None:
    from openai import AsyncOpenAI

    injected = AsyncOpenAI(api_key="user-key", max_retries=0)
    resolved = resolve_llm_client(injected, max_retries=7)
    assert resolved.client is injected
    assert resolved.client.max_retries == 0


def test_create_async_llm_client_plumbs_retry_count() -> None:
    cfg = LLMConfig(retry_count=6)
    client = create_async_llm_client(max_retries=cfg.retry_count)
    assert client.max_retries == 6


def test_retry_count_bounds_validated() -> None:
    with pytest.raises(ValueError):
        LLMConfig(retry_count=-1)
    with pytest.raises(ValueError):
        LLMConfig(retry_count=11)


def test_retry_attempts_property() -> None:
    assert LLMConfig(retry_count=2).retry_attempts == 3
    assert LLMConfig(retry_count=0).retry_attempts == 1


def test_orq_sdk_retry_config_derived_from_pipeline_config() -> None:
    pytest.importorskip("orq_ai_sdk")
    from evaluatorq.redteam.backends.orq import _orq_retry_config

    cfg = _orq_retry_config(3, [429, 500, 502, 503, 504])
    assert cfg is not None
    assert cfg.strategy == "backoff"
    assert cfg.retry_connection_errors is True
    assert cfg.status_codes_override == ["429", "500", "502", "503", "504"]
    assert cfg.backoff.max_elapsed_time == 500 + 1000 + 2000


def test_orq_sdk_retry_config_disabled_at_zero() -> None:
    pytest.importorskip("orq_ai_sdk")
    from evaluatorq.redteam.backends.orq import _orq_retry_config

    assert _orq_retry_config(0, [429]) is None
