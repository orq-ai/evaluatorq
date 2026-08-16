"""Bridge-level tests: OWASP panel scoring keeps failed-judge cost and cause.

`create_owasp_evaluator`'s panel path (``judge_fn`` + ``run_jury``) must behave
like `evaluatorq.redteam.adaptive.evaluator.AdaptiveEvaluator`:

- a judge call that errors still reports the tokens it burned (they were billed
  whether or not the verdict parsed), and
- when the whole panel fails to reach quorum, the resulting inconclusive
  ``EvaluationResult`` names a cause via ``EVAL_ERROR_RAW_OUTPUT_KEY`` instead of
  reporting a bare, unexplained "no verdict".
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evaluatorq import DataPoint, EvaluationResult
from evaluatorq.common.judge import EvaluatorResponsePayload, JudgeError, JudgeOutcome
from evaluatorq.contracts import EVAL_ERROR_RAW_OUTPUT_KEY, TokenUsage


def _run_judge_path() -> str:
    return 'evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge.run_judge'


class TestOwaspBridgePanelErrorVisibility:
    @pytest.mark.asyncio
    async def test_single_judge_failure_keeps_usage_and_structured_cause(self) -> None:
        """The default single-judge path must preserve billed failure metadata."""
        from evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge import create_owasp_evaluator

        mock_evaluator_entity = MagicMock()
        mock_evaluator_entity.prompt = 'response: {{output.response}}'
        failing_outcome = JudgeOutcome(
            error_kind=JudgeError.TIMEOUT,
            error_message='timed out',
            timeout_ms=30_000,
            token_usage=TokenUsage(input_tokens=5, output_tokens=3, total_tokens=8, calls=1),
        )

        with (
            patch(
                'evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge.get_evaluator_for_category',
                return_value=mock_evaluator_entity,
            ),
            patch(_run_judge_path(), new=AsyncMock(return_value=failing_outcome)),
        ):
            evaluator_config = create_owasp_evaluator(
                evaluator_model='judge-a',
                llm_client=AsyncMock(),
            )
            result: EvaluationResult = await evaluator_config['scorer']({
                'data': DataPoint(inputs={'category': 'ASI01', 'messages': []}),
                'output': {'response': 'target output'},
            })

        assert result.pass_ is None
        assert result.token_usage is not None
        assert result.token_usage.total_tokens == 8
        assert result.raw_output is not None
        payload = result.raw_output[EVAL_ERROR_RAW_OUTPUT_KEY]
        assert payload['code'] == 'timeout'
        assert payload['stage'] == 'evaluation'
        assert payload['details']['timeout_ms'] == 30_000

    @pytest.mark.asyncio
    async def test_failed_judge_prediction_keeps_token_usage(self) -> None:
        """A judge call that errors must not drop the tokens it already spent.

        Two repetitions of a single-judge panel both fail with a known
        token_usage on the JudgeOutcome; the resulting EvaluationResult must
        still report that cost instead of silently zeroing it out.
        """
        from evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge import create_owasp_evaluator

        mock_evaluator_entity = MagicMock()
        mock_evaluator_entity.prompt = 'response: {{output.response}}'

        failing_outcome = JudgeOutcome(
            error_kind=JudgeError.UNKNOWN,
            error_message='boom',
            token_usage=TokenUsage(input_tokens=5, output_tokens=3, total_tokens=8, calls=1),
        )
        mock_run_judge = AsyncMock(return_value=failing_outcome)

        with (
            patch(
                'evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge.get_evaluator_for_category',
                return_value=mock_evaluator_entity,
            ),
            patch(_run_judge_path(), mock_run_judge),
        ):
            evaluator_config = create_owasp_evaluator(
                evaluator_model='judge-a',
                llm_client=AsyncMock(),
                judge_repetitions=2,
            )
            result: EvaluationResult = await evaluator_config['scorer']({
                'data': DataPoint(inputs={'category': 'ASI01', 'messages': []}),
                'output': {'response': 'target output'},
            })

        assert mock_run_judge.await_count == 2
        assert result.token_usage is not None
        # Both failed repetitions' usage must be summed, not dropped.
        assert result.token_usage.total_tokens == 16
        assert result.token_usage.calls == 2

    @pytest.mark.asyncio
    async def test_quorum_failure_names_a_cause(self) -> None:
        """All-judges-failed panel must surface a structured cause, not a bare inconclusive."""
        from evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge import create_owasp_evaluator

        mock_evaluator_entity = MagicMock()
        mock_evaluator_entity.prompt = 'response: {{output.response}}'

        failing_outcome = JudgeOutcome(
            error_kind=JudgeError.API_STATUS,
            error_message='rate limited',
            token_usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2, calls=1),
        )
        mock_run_judge = AsyncMock(return_value=failing_outcome)

        with (
            patch(
                'evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge.get_evaluator_for_category',
                return_value=mock_evaluator_entity,
            ),
            patch(_run_judge_path(), mock_run_judge),
        ):
            evaluator_config = create_owasp_evaluator(
                evaluator_model='judge-a',
                llm_client=AsyncMock(),
                judges=['judge-b'],
            )
            result: EvaluationResult = await evaluator_config['scorer']({
                'data': DataPoint(inputs={'category': 'ASI01', 'messages': []}),
                'output': {'response': 'target output'},
            })

        assert result.pass_ is None
        assert result.raw_output is not None
        assert EVAL_ERROR_RAW_OUTPUT_KEY in result.raw_output
        payload = result.raw_output[EVAL_ERROR_RAW_OUTPUT_KEY]
        assert payload['message'] == 'rate limited'
        assert payload['stage'] == 'evaluation'

    @pytest.mark.asyncio
    async def test_successful_judge_call_is_unaffected(self) -> None:
        """Sanity check: a healthy panel call still returns a decisive verdict
        with no error payload — the fix must not regress the happy path."""
        from evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge import create_owasp_evaluator

        mock_evaluator_entity = MagicMock()
        mock_evaluator_entity.prompt = 'response: {{output.response}}'

        success_outcome = JudgeOutcome(
            payload=EvaluatorResponsePayload(value=True, explanation='Resistant'),
            token_usage=TokenUsage(input_tokens=2, output_tokens=2, total_tokens=4, calls=1),
        )
        mock_run_judge = AsyncMock(return_value=success_outcome)

        with (
            patch(
                'evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge.get_evaluator_for_category',
                return_value=mock_evaluator_entity,
            ),
            patch(_run_judge_path(), mock_run_judge),
        ):
            evaluator_config = create_owasp_evaluator(
                evaluator_model='judge-a',
                llm_client=AsyncMock(),
                judge_repetitions=2,
            )
            result: EvaluationResult = await evaluator_config['scorer']({
                'data': DataPoint(inputs={'category': 'ASI01', 'messages': []}),
                'output': {'response': 'target output'},
            })

        assert result.pass_ is True
        assert result.raw_output is not None
        assert EVAL_ERROR_RAW_OUTPUT_KEY not in result.raw_output
