"""Scorer must not judge target-error turns as RESISTANT (false success).

Regression for the pii-guardrail-timeout bug: a target 502 (guardrail timed out)
is synthesized into a turn whose visible text is ``"[ERROR: ...]"`` and carries
``AgentResponseError``. A single such error does not set the run-level
``output.error``, so it previously flowed straight into the LLM judge, which saw
no harmful content and scored the attack ``passed=True`` (RESISTANT) — a timeout
counted as a defense. The scorer must drop error turns before judging.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from evaluatorq.contracts import AgentResponse, AgentResponseError, TextOutputItem
from evaluatorq.redteam.adaptive.pipeline import create_dynamic_evaluator
from evaluatorq.redteam.contracts import AttackEvaluationResult, AttackOutput, Turn, Vulnerability

VULN = Vulnerability.GOAL_HIJACKING.value


def _error_turn(attack: str, code: str | None = 'orq.http.502') -> Turn:
    return Turn(
        attacker=AgentResponse(output=[TextOutputItem(text=attack, annotations=[])]),
        target=AgentResponse(
            output=[TextOutputItem(text='[ERROR: guardrail timeout]', annotations=[])],
            error=AgentResponseError(message='guardrail timeout', error_type='target_error', code=code),
        ),
    )


def _clean_turn(attack: str, reply: str) -> Turn:
    return Turn(
        attacker=AgentResponse(output=[TextOutputItem(text=attack, annotations=[])]),
        target=AgentResponse(output=[TextOutputItem(text=reply, annotations=[])]),
    )


def _params(output: AttackOutput) -> dict:
    return {'data': SimpleNamespace(inputs={}), 'output': output}


def _resistant() -> AttackEvaluationResult:
    return AttackEvaluationResult(passed=True, explanation='resisted', evaluator_id='goal_hijacking')


@pytest.mark.asyncio
async def test_all_error_turns_scored_as_error_not_resistant():
    """When every turn is a target error, judge is not called and value='error'."""
    output = AttackOutput(vulnerability=VULN, turns=[_error_turn('a'), _error_turn('b')])
    judge = AsyncMock(return_value=_resistant())

    with patch('evaluatorq.redteam.adaptive.pipeline.OWASPEvaluator') as cls:
        cls.return_value.evaluate_vulnerability = judge
        scorer = create_dynamic_evaluator(llm_client=AsyncMock())['scorer']
        result = await scorer(_params(output))

    judge.assert_not_called()
    assert result.value == 'error'


@pytest.mark.asyncio
async def test_all_error_turns_with_missing_codes_does_not_crash():
    """AgentResponseError.code is Optional; a None code must not crash the summary."""
    output = AttackOutput(vulnerability=VULN, turns=[_error_turn('a', code=None), _error_turn('b', code='orq.http.502')])
    judge = AsyncMock(return_value=_resistant())

    with patch('evaluatorq.redteam.adaptive.pipeline.OWASPEvaluator') as cls:
        cls.return_value.evaluate_vulnerability = judge
        scorer = create_dynamic_evaluator(llm_client=AsyncMock())['scorer']
        result = await scorer(_params(output))

    judge.assert_not_called()
    assert result.value == 'error'
    assert 'orq.http.502' in result.explanation


@pytest.mark.asyncio
async def test_empty_content_turn_scored_as_error():
    """A turn with no error marker but blank target text (e.g. empty-body 200 where a
    guardrail silently blocked) is not a real reply and must not be judged as resistant."""
    blank = Turn(
        attacker=AgentResponse(output=[TextOutputItem(text='attack', annotations=[])]),
        target=AgentResponse(output=[TextOutputItem(text='   ', annotations=[])]),
    )
    output = AttackOutput(vulnerability=VULN, turns=[blank])
    judge = AsyncMock(return_value=_resistant())

    with patch('evaluatorq.redteam.adaptive.pipeline.OWASPEvaluator') as cls:
        cls.return_value.evaluate_vulnerability = judge
        scorer = create_dynamic_evaluator(llm_client=AsyncMock())['scorer']
        result = await scorer(_params(output))

    judge.assert_not_called()
    assert result.value == 'error'


@pytest.mark.asyncio
async def test_error_turn_dropped_before_judging():
    """A mix of error + real turns judges only the real turn's messages."""
    output = AttackOutput(
        vulnerability=VULN,
        turns=[_error_turn('turn1'), _clean_turn('turn2', 'benign reply')],
    )
    judge = AsyncMock(return_value=_resistant())

    with patch('evaluatorq.redteam.adaptive.pipeline.OWASPEvaluator') as cls:
        cls.return_value.evaluate_vulnerability = judge
        scorer = create_dynamic_evaluator(llm_client=AsyncMock())['scorer']
        await scorer(_params(output))

    judge.assert_called_once()
    kwargs = judge.call_args.kwargs
    assert [m['content'] for m in kwargs['messages']] == ['turn2']
    assert [i.text for i in kwargs['output_messages']] == ['benign reply']
