"""A target-level error must never reach the judge — in either jury.

Judging an errored generation would score "the agent said nothing" as if the agent
had genuinely answered that way.
"""

from __future__ import annotations

import importlib
from unittest.mock import patch

import pytest

from evaluatorq.contracts import AgentResponse, AgentResponseError
from evaluatorq.llm_jury import DEFAULT_PAIRWISE_TEMPLATE, llm_jury, llm_jury_pairwise
from evaluatorq.types import DataPoint, EvaluationResult

llm_jury_mod = importlib.import_module('evaluatorq.llm_jury')


def _errored() -> AgentResponse:
    return AgentResponse(output=[], error=AgentResponseError(message='upstream timeout', error_type='timeout'))


@pytest.mark.asyncio
async def test_pointwise_skips_judges_when_target_errored():
    evaluator = llm_jury(name='correctness', judges=['openai/gpt-5.4-mini'], criteria='Is it correct?')

    with patch.object(llm_jury_mod, 'resolve_llm_client') as resolve_client:
        result = await evaluator['scorer']({
            'data': DataPoint(inputs={'question': 'q'}),
            'output': _errored(),
        })

    resolve_client.assert_not_called()

    # Scorer is typed EvaluationResult | dict[str, Any]; this path returns the model.
    assert isinstance(result, EvaluationResult)
    assert result.pass_ is None
    assert result.value == 'inconclusive'
    assert 'upstream timeout' in (result.explanation or '')


@pytest.mark.asyncio
async def test_pointwise_skips_judges_for_empty_error_message():
    evaluator = llm_jury(name='correctness', judges=['openai/gpt-5.4-mini'], criteria='Is it correct?')

    with patch.object(llm_jury_mod, 'resolve_llm_client') as resolve_client:
        result = await evaluator['scorer']({
            'data': DataPoint(inputs={'question': 'q'}),
            'output': AgentResponse(
                output=[], error=AgentResponseError(message='', error_type='target_error')
            ),
        })

    resolve_client.assert_not_called()
    assert isinstance(result, EvaluationResult)
    assert result.value == 'inconclusive'


@pytest.mark.asyncio
async def test_pointwise_skips_judges_for_dict_error():
    evaluator = llm_jury(name='correctness', judges=['openai/gpt-5.4-mini'], criteria='Is it correct?')

    with patch.object(llm_jury_mod, 'resolve_llm_client') as resolve_client:
        result = await evaluator['scorer']({
            'data': DataPoint(inputs={'question': 'q'}),
            'output': {'response': 'ignored', 'error': {'message': 'upstream timeout'}},
        })

    resolve_client.assert_not_called()
    assert isinstance(result, EvaluationResult)
    assert result.value == 'inconclusive'


@pytest.mark.asyncio
@pytest.mark.parametrize('errored_side', ['a', 'b'])
async def test_pairwise_skips_judges_when_a_side_errored(errored_side: str):
    comparator = llm_jury_pairwise(judges=['openai/gpt-5.4-mini'])

    with patch.object(llm_jury_mod, 'resolve_llm_client') as resolve_client:
        comparison = await comparator.compare(
            question='which is better?',
            response_a=_errored() if errored_side == 'a' else 'a fine answer',
            response_b=_errored() if errored_side == 'b' else 'another fine answer',
        )

    resolve_client.assert_not_called()
    assert comparison.winner == 'inconclusive'
    assert comparison.votes == []


@pytest.mark.asyncio
async def test_pairwise_prompt_override_replaces_default_template():
    custom = '# Only A\n{{response_a.output.response}}'
    comparator = llm_jury_pairwise(judges=['openai/gpt-5.4-mini'], prompt=custom)
    assert comparator._template == custom

    default = llm_jury_pairwise(judges=['openai/gpt-5.4-mini'])
    assert default._template == DEFAULT_PAIRWISE_TEMPLATE
