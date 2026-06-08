"""Evaluator wrapper for OWASP vulnerability detection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq.redteam.backends.registry import create_async_llm_client
from evaluatorq.redteam.contracts import (
    DEFAULT_PIPELINE_MODEL,
    PIPELINE_CONFIG,
    AttackEvaluationResult,
    LLMCallConfig,
    OutputMessage,
    Vulnerability,
)
from evaluatorq.redteam.frameworks.owasp.evaluators import get_evaluator_for_category, get_evaluator_for_vulnerability
from evaluatorq.redteam.judge import JudgeError, build_eval_replacements, run_judge
from evaluatorq.redteam.vulnerability_registry import resolve_category_safe

if TYPE_CHECKING:
    from collections.abc import Sequence

    from openai import AsyncOpenAI

    from evaluatorq.redteam.contracts import Message


class OWASPEvaluator:
    """Wrapper for OWASP vulnerability evaluators."""

    def __init__(
        self,
        evaluator_model: str = DEFAULT_PIPELINE_MODEL,
        llm_client: AsyncOpenAI | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        cfg: LLMCallConfig | None = None,
    ):
        """Initialize the evaluator with the given model and optional async LLM client."""
        base_cfg = cfg or PIPELINE_CONFIG.evaluator
        # Fold constructor-level llm_kwargs into the cfg's extra_kwargs so they reach
        # run_judge (which forwards cfg.extra_kwargs). Without this, llm_kwargs threaded
        # via create_dynamic_evaluator would be silently dropped on the judge call.
        self.llm_kwargs = {**base_cfg.extra_kwargs, **(llm_kwargs or {})}
        self._cfg = base_cfg.model_copy(update={'extra_kwargs': self.llm_kwargs})
        self.evaluator_model = evaluator_model
        self.client = llm_client or base_cfg.client or create_async_llm_client()
        logger.debug(f'Initialized OWASPEvaluator with model: {evaluator_model}')

    async def evaluate_vulnerability(
        self,
        vuln: Vulnerability,
        messages: list[dict[str, Any]] | list[Message],
        output_messages: Sequence[OutputMessage],
    ) -> AttackEvaluationResult:
        """Evaluate an agent response for a specific vulnerability.

        This is the preferred path when a Vulnerability enum is already known —
        it bypasses category normalization and resolves directly via the
        VULNERABILITY_EVALUATOR_REGISTRY.
        """
        evaluator = get_evaluator_for_vulnerability(vuln, model_id=self.evaluator_model)
        if evaluator is None:
            logger.warning(f'No evaluator found for vulnerability {vuln.value}')
            return AttackEvaluationResult(
                passed=None,
                explanation=f'No evaluator available for vulnerability {vuln.value}',
                evaluator_id='none',
                raw_output=None,
            )

        return await self._run_evaluator(
            evaluator=evaluator,
            evaluator_id=vuln.value,
            messages=messages,
            output_messages=output_messages,
            span_attributes={
                'orq.redteam.llm_purpose': 'evaluation',
                'orq.redteam.vulnerability': vuln.value,
            },
        )

    async def evaluate(
        self,
        category: str,
        messages: list[dict[str, Any]] | list[Message],
        output_messages: Sequence[OutputMessage],
    ) -> AttackEvaluationResult:
        """Evaluate an agent response for vulnerability.

        Resolves the category to a Vulnerability enum when possible and delegates
        to evaluate_vulnerability() for the preferred vulnerability-first path.
        Falls back to direct category lookup for unrecognized category codes.
        """
        category_code = category.removeprefix('OWASP-')

        vuln = resolve_category_safe(category_code)
        if vuln is not None:
            return await self.evaluate_vulnerability(vuln, messages, output_messages=output_messages)

        # Fallback: category not in the registry — try the category-keyed lookup directly
        evaluator = get_evaluator_for_category(category, model_id=self.evaluator_model)
        if evaluator is None:
            logger.warning(f'No evaluator found for category {category}')
            return AttackEvaluationResult(
                passed=None,
                explanation=f'No evaluator available for category {category}',
                evaluator_id='none',
                raw_output=None,
            )

        return await self._run_evaluator(
            evaluator=evaluator,
            evaluator_id=category_code,
            messages=messages,
            output_messages=output_messages,
            span_attributes={
                'orq.redteam.llm_purpose': 'evaluation',
                'orq.redteam.category': category,
                'orq.redteam.vulnerability': '',
            },
        )

    async def _run_evaluator(
        self,
        evaluator: Any,
        evaluator_id: str,
        messages: list[dict[str, Any]] | list[Message],
        output_messages: Sequence[OutputMessage],
        span_attributes: dict[str, str],
    ) -> AttackEvaluationResult:
        """Execute an evaluator entity against a conversation and return a typed result."""
        outcome = await run_judge(
            client=self.client,
            model=self.evaluator_model,
            cfg=self._cfg,
            prompt_template=evaluator.prompt,
            replacements=build_eval_replacements(
                input_messages=messages,
                output_messages=output_messages,
                expected_output=None,
                system_instructions=None,
            ),
            span_attributes=span_attributes,
        )

        if outcome.error_kind in (JudgeError.API_CONNECTION, JudgeError.API_STATUS):
            if outcome.error_exc is not None:
                raise outcome.error_exc
        if outcome.error_kind is JudgeError.TIMEOUT:
            return AttackEvaluationResult(
                passed=None,
                explanation=f'Evaluation timed out after {outcome.timeout_ms}ms',
                evaluator_id=evaluator_id,
                raw_output={'error': 'timeout', 'timeout_ms': outcome.timeout_ms},
            )
        if outcome.error_kind is not None or outcome.payload is None:
            return AttackEvaluationResult(
                passed=None,
                explanation=f'Evaluation error: {outcome.error_message}',
                evaluator_id=evaluator_id,
                raw_output={'error': outcome.error_message, 'raw_content': outcome.raw_content},
            )

        return AttackEvaluationResult(
            passed=outcome.payload.value,
            explanation=outcome.payload.explanation,
            evaluator_id=evaluator_id,
            token_usage=outcome.token_usage,
            raw_output={
                'value': outcome.payload.value,
                'explanation': outcome.payload.explanation,
                'raw_content': outcome.raw_content,
            },
        )


async def evaluate_attack(
    category: str,
    messages: list[dict[str, Any]] | list[Message],
    output_messages: Sequence[OutputMessage],
    evaluator_model: str = DEFAULT_PIPELINE_MODEL,
    *,
    vulnerability: Vulnerability | None = None,
) -> AttackEvaluationResult:
    """Convenience function to evaluate a single attack.

    When vulnerability is provided, uses the vulnerability-first path directly,
    skipping category resolution. Falls back to category-based resolution otherwise.
    """
    evaluator = OWASPEvaluator(evaluator_model=evaluator_model)
    if vulnerability is not None:
        return await evaluator.evaluate_vulnerability(vulnerability, messages, output_messages=output_messages)
    return await evaluator.evaluate(category, messages, output_messages=output_messages)
