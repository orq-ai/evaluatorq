"""OWASP evaluator entities carry no sampling default unless the caller sets one."""

from evaluatorq.redteam.frameworks.owasp.llm_evaluators import get_llm01_prompt_injection_evaluator
from evaluatorq.redteam.frameworks.owasp.models import (
    EvaluatorModelConfig,
    LlmEvaluatorEntity,
    LlmEvaluatorOutputFormat,
)


def test_entity_temperature_defaults_to_none() -> None:
    entity = LlmEvaluatorEntity(
        model=EvaluatorModelConfig(id='openai/gpt-5.6-luna'),
        prompt='judge',
        output_type=LlmEvaluatorOutputFormat.BOOLEAN,
    )

    assert entity.temperature is None
    assert entity.model_dump()['temperature'] is None


def test_built_owasp_evaluator_sets_no_temperature() -> None:
    entity = get_llm01_prompt_injection_evaluator('openai/gpt-5.6-luna')

    assert entity.temperature is None
