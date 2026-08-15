# tests/redteam/test_llm_config.py
from typing import TYPE_CHECKING, cast

import pytest
from pydantic import ValidationError

from evaluatorq.redteam.contracts import (
    DEFAULT_PIPELINE_MODEL,
    PIPELINE_CONFIG,
    EvaluatorConfig,
    LLMCallConfig,
    LLMConfig,
)

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
    assert isinstance(cfg.evaluator, EvaluatorConfig)
    assert cfg.attacker.model == DEFAULT_PIPELINE_MODEL
    assert cfg.evaluator.model == DEFAULT_PIPELINE_MODEL


def test_llm_config_custom_roles():
    cfg = LLMConfig(
        attacker=LLMCallConfig(model='anthropic/claude-3-5-sonnet', temperature=0.9),
        evaluator=EvaluatorConfig(model='openai/gpt-4o-mini', temperature=0.0),
    )
    assert cfg.attacker.model == 'anthropic/claude-3-5-sonnet'
    assert cfg.attacker.temperature == 0.9
    assert cfg.evaluator.model == 'openai/gpt-4o-mini'
    assert cfg.evaluator.temperature == 0.0


def test_llm_config_has_retry_and_timeout_fields():
    cfg = LLMConfig()
    assert cfg.retry_count == 3
    assert cfg.cleanup_timeout_ms == 60_000
    assert cfg.max_target_retries == 2


def test_evaluator_config_min_evaluation_coverage_defaults_to_0_8():
    cfg = EvaluatorConfig()
    assert cfg.min_evaluation_coverage == 0.8


@pytest.mark.parametrize('value', [-0.1, 1.1])
def test_evaluator_config_min_evaluation_coverage_rejects_out_of_range(value):
    with pytest.raises(ValidationError):
        EvaluatorConfig(min_evaluation_coverage=value)


def test_evaluator_config_min_evaluation_coverage_accepts_none():
    cfg = EvaluatorConfig(min_evaluation_coverage=None)
    assert cfg.min_evaluation_coverage is None


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


# ---------------------------------------------------------------------------
# completion_params: extra_kwargs must merge, never collide
# ---------------------------------------------------------------------------


def test_completion_params_defaults_and_site_params():
    from evaluatorq.contracts import LLMCallConfig

    cfg = LLMCallConfig(temperature=0.7, max_tokens=1234)
    params = cfg.completion_params(model='m', messages=[{'role': 'user', 'content': 'q'}])
    assert params['temperature'] == 0.7
    assert params['max_completion_tokens'] == 1234
    assert params['model'] == 'm'


def test_completion_params_extra_kwargs_override_instead_of_typeerror():
    # Regression: splatting extra_kwargs next to explicit temperature=/
    # max_completion_tokens= keywords raised TypeError ('got multiple values')
    # the moment a user routed those keys through extra_kwargs, turning every
    # evaluation inconclusive. Merged params must let the user keys win.
    from evaluatorq.contracts import LLMCallConfig

    cfg = LLMCallConfig(temperature=0.7, extra_kwargs={'temperature': 1.0, 'max_completion_tokens': 99})
    params = cfg.completion_params(model='m', messages=[])
    assert params['temperature'] == 1.0
    assert params['max_completion_tokens'] == 99


def test_completion_params_site_params_override_field_defaults():
    from evaluatorq.contracts import LLMCallConfig

    cfg = LLMCallConfig(max_tokens=1000)
    params = cfg.completion_params(max_completion_tokens=1500)
    assert params['max_completion_tokens'] == 1500


def test_no_call_site_splats_extra_kwargs_next_to_explicit_sampling_kwargs():
    """Grep-level guard: the collision pattern must not come back.

    A call carrying explicit temperature=/max_completion_tokens= keywords AND a
    **...extra_kwargs splat raises TypeError on a duplicate key. All call sites
    must go through LLMCallConfig.completion_params (or an equivalent dict
    merge) instead.
    """
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / 'src' / 'evaluatorq'
    offenders = []
    for path in src.rglob('*.py'):
        text = path.read_text()
        for m in re.finditer(r'\.(?:create|parse)\(\n(?:[^()]*?\n)*?[^()]*?\*\*[\w.]*extra_kwargs', text):
            window = m.group(0)
            if 'temperature=' in window or 'max_completion_tokens=' in window:
                offenders.append(str(path.relative_to(src)))
    assert offenders == [], f'explicit sampling kwargs next to **extra_kwargs in: {offenders}'


def test_openai_backend_factory_forwards_pipeline_timeout():
    from unittest.mock import MagicMock

    from evaluatorq.redteam.backends.openai import OpenAIBackend
    from evaluatorq.redteam.backends.registry import _create_openai_backend

    cfg = LLMConfig(target_agent_timeout_ms=123_456)
    backend = _create_openai_backend(llm_client=MagicMock(), target_config=None, pipeline_config=cfg)
    assert isinstance(backend, OpenAIBackend)
    assert backend._timeout_ms == 123_456


def test_completion_params_rejects_structural_extra_kwargs():
    """extra_kwargs tunes sampling/provider options; silently replacing
    model/messages/response_format/extra_body would break the call it rides
    on (e.g. dropping a required JSON response format)."""
    from evaluatorq.contracts import LLMCallConfig

    cfg = LLMCallConfig(model='m', extra_kwargs={'response_format': None, 'temperature': 1})
    with pytest.raises(ValueError, match='structural'):
        cfg.completion_params(model='m', messages=[])


def test_completion_params_sampling_extra_kwargs_still_pass():
    from evaluatorq.contracts import LLMCallConfig

    cfg = LLMCallConfig(model='m', extra_kwargs={'top_p': 0.9})
    params = cfg.completion_params(model='m', messages=[])
    assert params['top_p'] == 0.9
