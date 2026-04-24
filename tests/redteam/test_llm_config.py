# tests/redteam/test_llm_config.py
import os
import pytest
from evaluatorq.redteam.contracts import LLMCallConfig, LLMConfig, PIPELINE_CONFIG, DEFAULT_PIPELINE_MODEL


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


def test_retry_config_empty_when_openai_key_set(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-test')
    monkeypatch.delenv('ORQ_API_KEY', raising=False)
    cfg = LLMConfig()
    assert cfg.retry_config == {}


def test_retry_config_populated_when_only_orq_key(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.setenv('ORQ_API_KEY', 'orq-test')
    cfg = LLMConfig()
    result = cfg.retry_config
    assert 'retry' in result
    assert result['retry']['count'] == 3


def test_uses_orq_router_with_orq_key(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.setenv('ORQ_API_KEY', 'orq-test')
    cfg = LLMConfig()
    assert cfg.uses_orq_router is True


def test_uses_orq_router_false_with_openai_key(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-test')
    cfg = LLMConfig()
    assert cfg.uses_orq_router is False


@pytest.mark.asyncio
async def test_red_team_accepts_legacy_config_keyword(monkeypatch):
    from evaluatorq.redteam import red_team
    from evaluatorq.redteam.exceptions import CredentialError

    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('ORQ_API_KEY', raising=False)

    with pytest.deprecated_call(match='config= is deprecated'):
        with pytest.raises(CredentialError):
            await red_team('agent:test', config=LLMConfig())
