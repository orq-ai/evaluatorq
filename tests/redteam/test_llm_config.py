# tests/redteam/test_llm_config.py
import os
from typing import TYPE_CHECKING, cast

import pytest
from evaluatorq.redteam.contracts import LLMCallConfig, LLMConfig, PIPELINE_CONFIG, DEFAULT_PIPELINE_MODEL

if TYPE_CHECKING:
    from openai import AsyncOpenAI


def test_llm_call_config_defaults():
    cfg = LLMCallConfig()
    assert cfg.model == DEFAULT_PIPELINE_MODEL
    assert cfg.temperature == 1.0
    assert cfg.max_tokens == 5000
    assert cfg.timeout_ms == 90_000
    assert cfg.extra_kwargs == {}
    assert cfg.client is None


def test_llm_call_config_custom_values():
    cfg = LLMCallConfig(model='gpt-4o', temperature=0.5, max_tokens=1000, timeout_ms=30_000)
    assert cfg.model == 'gpt-4o'
    assert cfg.temperature == 0.5
    assert cfg.max_tokens == 1000
    assert cfg.timeout_ms == 30_000


def test_llm_config_has_role_based_fields():
    cfg = LLMConfig()
    assert isinstance(cfg.attacker, LLMCallConfig)
    assert isinstance(cfg.evaluator, LLMCallConfig)
    assert cfg.attacker.model == DEFAULT_PIPELINE_MODEL
    assert cfg.evaluator.model == DEFAULT_PIPELINE_MODEL


def test_llm_config_custom_roles():
    cfg = LLMConfig(
        attacker=LLMCallConfig(model='anthropic/claude-3-5-sonnet', temperature=0.9),
        evaluator=LLMCallConfig(model='openai/gpt-4o-mini', temperature=0.0),
    )
    assert cfg.attacker.model == 'anthropic/claude-3-5-sonnet'
    assert cfg.attacker.temperature == 0.9
    assert cfg.evaluator.model == 'openai/gpt-4o-mini'
    assert cfg.evaluator.temperature == 0.0


def test_llm_config_has_retry_and_timeout_fields():
    cfg = LLMConfig()
    assert cfg.retry_count == 3
    assert cfg.cleanup_timeout_ms == 60_000


def test_llm_config_no_backend_field():
    cfg = LLMConfig()
    assert not hasattr(cfg, 'backend')


def test_llm_config_no_llm_sub_field():
    cfg = LLMConfig()
    assert not hasattr(cfg, 'llm')


def test_pipeline_config_is_llm_config():
    assert isinstance(PIPELINE_CONFIG, LLMConfig)


class _FakeClient:
    """Minimal stand-in exposing ``base_url`` for retry-gating tests."""

    def __init__(self, base_url):
        self.base_url = base_url


def _as_client(obj: object) -> "AsyncOpenAI":
    """Cast a structural stand-in to AsyncOpenAI for retry_extra_body's signature.

    retry_extra_body only reads ``base_url`` via client_routes_through_orq, so the
    fake is sufficient at runtime; this routes the cast through ``object`` to satisfy
    basedpyright (a direct _FakeClient→AsyncOpenAI cast is rejected as non-overlapping).
    """
    return cast("AsyncOpenAI", obj)


def test_retry_extra_body_populated_for_router_client():
    """A client routed through the Orq router receives ORQ-specific retry hints."""
    cfg = LLMConfig()
    body = cfg.retry_extra_body(_as_client(_FakeClient('https://my.orq.ai/v3/router')))
    assert body == {'retry': {'count': cfg.retry_count, 'on_codes': cfg.retry_on_codes}}


def test_retry_extra_body_empty_for_openai_client():
    """A plain OpenAI client must not receive the ORQ-only ``retry`` field."""
    cfg = LLMConfig()
    assert cfg.retry_extra_body(_as_client(_FakeClient('https://api.openai.com/v1'))) == {}


def test_retry_extra_body_gates_on_client_not_env(monkeypatch):
    """Gating is on the client's base_url, not on ORQ_API_KEY in the environment.

    An injected OpenAI client must not receive the ORQ-only ``retry`` field just
    because ORQ_API_KEY happens to be in the environment (it is needed for tracing).
    """
    monkeypatch.setenv('ORQ_API_KEY', 'orq-test')  # present (e.g. for tracing) but irrelevant
    cfg = LLMConfig()
    assert cfg.retry_extra_body(_as_client(_FakeClient('https://api.openai.com/v1'))) == {}


def test_retry_extra_body_empty_for_client_without_base_url():
    cfg = LLMConfig()
    assert cfg.retry_extra_body(_as_client(object())) == {}
    assert cfg.retry_extra_body(None) == {}


@pytest.mark.asyncio
async def test_red_team_accepts_legacy_config_keyword(monkeypatch):
    from evaluatorq.redteam import red_team
    from evaluatorq.redteam.exceptions import CredentialError

    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('ORQ_API_KEY', raising=False)

    with pytest.deprecated_call(match='config= is deprecated'):
        with pytest.raises(CredentialError):
            await red_team('agent:test', config=LLMConfig())
