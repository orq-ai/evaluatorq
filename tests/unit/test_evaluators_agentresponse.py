import pytest

from evaluatorq.contracts import AgentResponse
from evaluatorq.evaluators import string_contains_evaluator
from evaluatorq.types import DataPoint


@pytest.mark.asyncio
async def test_contains_matches_agentresponse_text():
    ev = string_contains_evaluator()
    data = DataPoint(inputs={}, expected_output='answer')
    result = await ev['scorer']({'data': data, 'output': AgentResponse(text='the answer is here')})
    assert isinstance(result, dict)
    assert result['pass_'] is True  # matched on .text, not pydantic repr
