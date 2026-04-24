# tests/redteam/test_llm_config.py
import os
import pytest
from evaluatorq.redteam.contracts import LLMConfig, PIPELINE_CONFIG


def test_llm_config_has_pipeline_fields():
    cfg = LLMConfig()
    assert cfg.adversarial_temperature == 1.0
    assert cfg.adversarial_max_tokens == 5000
    assert cfg.strategy_generation_temperature == 1.0
    assert cfg.strategy_generation_max_tokens == 5000
    assert cfg.capability_classification_max_tokens == 5000
    assert cfg.capability_classification_temperature == 1.0
    assert cfg.tool_adaptation_max_tokens == 5000
    assert cfg.tool_adaptation_temperature == 1.0
    assert cfg.target_max_tokens == 5000
    assert cfg.llm_call_timeout_ms == 90_000
    assert cfg.target_agent_timeout_ms == 240_000
    assert cfg.cleanup_timeout_ms == 60_000
    assert cfg.retry_count == 3


def test_llm_config_has_model_fields():
    cfg = LLMConfig(attack_model='gpt-4o', evaluator_model='gpt-4o-mini')
    assert cfg.attack_model == 'gpt-4o'
    assert cfg.evaluator_model == 'gpt-4o-mini'


def test_llm_config_has_llm_kwargs():
    cfg = LLMConfig(llm_kwargs={'temperature': 0.5})
    assert cfg.llm_kwargs == {'temperature': 0.5}


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


def test_resolve_model_adds_prefix_for_orq_router():
    cfg = LLMConfig()
    assert cfg.resolve_model('gpt-5-mini', uses_orq_router=True) == 'openai/gpt-5-mini'
    assert cfg.resolve_model('openai/gpt-5-mini', uses_orq_router=True) == 'openai/gpt-5-mini'
    assert cfg.resolve_model('gpt-5-mini', uses_orq_router=False) == 'gpt-5-mini'
