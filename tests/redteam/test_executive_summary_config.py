"""The red-team executive summary sends the evaluator's knobs, and only the ones set.

`generate_executive_summary` reads `model_fields_set`, so a field this call site
hands over as an explicit ``None`` is not "unset" to it — it is "send null", which
reasoning-class models answer with a 400. Attribute reads made every unset field
explicit; `set_values` is what keeps them unset.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from evaluatorq.redteam.contracts import EvaluatorConfig, LLMConfig
from evaluatorq.redteam.runner import _executive_summary_config


def test_an_unset_evaluator_field_stays_unset() -> None:
    cfg = _executive_summary_config(LLMConfig(), model='judge/model', client=None)
    assert cfg.model == 'judge/model'
    assert 'temperature' not in cfg.model_fields_set
    assert 'reasoning_effort' not in cfg.model_fields_set
    assert 'extra_kwargs' not in cfg.model_fields_set


def test_a_set_evaluator_field_reaches_the_summary_call() -> None:
    """``temperature=0.0`` is the case a truthiness or ``is not None`` check gets wrong."""
    config = LLMConfig(evaluator=EvaluatorConfig(temperature=0.0, reasoning_effort='high'))
    cfg = _executive_summary_config(config, model='judge/model', client=None)
    assert cfg.temperature == 0.0
    assert 'temperature' in cfg.model_fields_set
    assert cfg.reasoning_effort == 'high'


def test_the_router_retry_policy_rides_along_only_for_an_orq_client() -> None:
    """``retry`` is an Orq router field; a plain OpenAI endpoint rejects it."""
    plain = MagicMock()
    plain.base_url = 'https://api.openai.com/v1'
    assert _executive_summary_config(LLMConfig(), model='m', client=plain).extra_body == {}

    orq = MagicMock()
    orq.base_url = 'https://my.orq.ai/v3/router'
    body = _executive_summary_config(LLMConfig(retry_count=4), model='m', client=orq).extra_body
    assert body['retry']['count'] == 4


def test_the_fields_the_summary_reads_are_forwarded() -> None:
    """`generate_executive_summary` reads `max_tokens` and `timeout_ms`. Narrowing the
    evaluator config clears `model_fields_set`, so a field dropped here can never be
    named by the callee's own warning — it has to survive the narrowing or be logged."""
    config = LLMConfig(evaluator=EvaluatorConfig(max_tokens=2000, timeout_ms=300_000))
    cfg = _executive_summary_config(config, model='m', client=None)
    assert cfg.max_tokens == 2000
    assert cfg.timeout_ms == 300_000


def test_a_caller_key_wins_the_router_body_merge(caplog: pytest.LogCaptureFixture) -> None:
    """The retry policy is this call site's default; a key the caller set on the evaluator
    config wins it, and the fields the summary cannot read are named rather than dropped."""
    orq = MagicMock()
    orq.base_url = 'https://my.orq.ai/v3/router'
    config = LLMConfig(
        retry_count=4,
        evaluator=EvaluatorConfig(extra_body={'my_router_flag': 1}, api='responses'),
    )
    with caplog.at_level('WARNING'):
        cfg = _executive_summary_config(config, model='m', client=orq)
    assert cfg.extra_body == {'retry': {'count': 4, 'on_codes': [429, 500, 502, 503, 504]}, 'my_router_flag': 1}
    assert 'red_team executive summary ignores llm_config api' in caplog.text
