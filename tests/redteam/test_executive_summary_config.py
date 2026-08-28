"""The red-team executive summary sends the evaluator's knobs, and only the ones set.

`generate_executive_summary` reads `model_fields_set`, so a field this call site
hands over as an explicit ``None`` is not "unset" to it — it is "send null", which
reasoning-class models answer with a 400. Attribute reads made every unset field
explicit; `set_values` is what keeps them unset.
"""

from __future__ import annotations

from unittest.mock import MagicMock

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
