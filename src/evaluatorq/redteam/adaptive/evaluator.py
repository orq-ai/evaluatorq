"""Evaluator wrapper for OWASP vulnerability detection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq.common.judge import JudgeError, JudgeOutcome, build_eval_replacements, run_judge
from evaluatorq.common.jury import (
    Prediction,
    VerdictKind,
    _panel_composition_messages,
    run_jury,
)
from evaluatorq.common.jury import provider_family as provider_family
from evaluatorq.redteam.backends.registry import create_async_llm_client
from evaluatorq.redteam.contracts import (
    DEFAULT_PIPELINE_MODEL,
    PIPELINE_CONFIG,
    AttackEvaluationResult,
    EvaluatorConfig,
    LLMCallConfig,
    OutputMessage,
    Vulnerability,
)
from evaluatorq.redteam.frameworks.owasp.evaluators import get_evaluator_for_category, get_evaluator_for_vulnerability
from evaluatorq.redteam.vulnerability_registry import resolve_category_safe

if TYPE_CHECKING:
    from collections.abc import Sequence

    from openai import AsyncOpenAI

    from evaluatorq.redteam.contracts import Message


class OWASPEvaluator:
    """Wrapper for OWASP vulnerability evaluators.

    Supports a panel of judge models and repeated predictions per judge with
    majority-vote aggregation (RES-739). With the defaults (single model, single
    repetition) it behaves exactly like a single-judge, single-pass evaluator.
    """

    def __init__(
        self,
        evaluator_model: str = DEFAULT_PIPELINE_MODEL,
        llm_client: AsyncOpenAI | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        cfg: LLMCallConfig | EvaluatorConfig | None = None,
        judges: list[str] | None = None,
        repetitions: int = 1,
        replacement_judges: list[str] | None = None,
        min_successful_judges: int = 1,
        target_models: list[str] | None = None,
        *,
        strict_panel: bool = False,
    ):
        """Initialize the evaluator with the given model and optional async LLM client.

        ``judges`` are additional panel models evaluated alongside ``evaluator_model``;
        ``repetitions`` is the number of majority-vote passes run per judge;
        ``replacement_judges`` stand in for configured judges that fail entirely;
        ``min_successful_judges`` is the floor below which the verdict is inconclusive.

        ``target_models`` are the model IDs under test, used only to warn (or, with
        ``strict_panel``, hard-error) when a panel judge shares the target's provider
        family — same-family self-judging biases verdicts toward RESISTANT and
        under-counts vulnerabilities. Pass it only when the target model is known.
        """
        base_cfg = cfg or PIPELINE_CONFIG.evaluator
        # Fold constructor-level llm_kwargs into the cfg's extra_kwargs so they reach
        # run_judge (which forwards cfg.extra_kwargs). Without this, llm_kwargs threaded
        # via create_dynamic_evaluator would be silently dropped on the judge call.
        self.llm_kwargs = {**base_cfg.extra_kwargs, **(llm_kwargs or {})}
        self._cfg = base_cfg.model_copy(update={'extra_kwargs': self.llm_kwargs})
        self._call_cfg = self._cfg.as_call_config() if isinstance(self._cfg, EvaluatorConfig) else self._cfg
        self.evaluator_model = evaluator_model
        self.client = llm_client or base_cfg.client or create_async_llm_client()
        # Panel = primary model + extra judges, de-duplicated, primary first.
        panel: list[str] = [evaluator_model]
        for j in judges or []:
            if j and j not in panel:
                panel.append(j)
        self.panel = panel
        # Replacement judges not already configured as judges, de-duplicated
        # within the list too — a repeated stand-in would cast two independent
        # votes from one model and could manufacture a false consensus.
        seen_replacements: set[str] = set(panel)
        deduped_replacements: list[str] = []
        for r in replacement_judges or []:
            if r and r not in seen_replacements:
                deduped_replacements.append(r)
                seen_replacements.add(r)
        self.replacement_judges = deduped_replacements
        self.repetitions = max(1, repetitions)
        self.min_successful_judges = max(1, min_successful_judges)
        # Validate against the *effective* panel, not the raw judge count. The
        # LLMConfig validator dedups the same way; de-duplication here (primary
        # model appearing in judges, or repeats within judges) can shrink the panel,
        # which would leave min_successful_judges permanently unsatisfiable and force
        # every verdict to inconclusive.
        if self.min_successful_judges > len(self.panel):
            raise ValueError(
                f'min_successful_judges ({self.min_successful_judges}) exceeds the effective '
                f'panel size ({len(self.panel)}) after de-duplication; panel={self.panel}'
            )
        # Composition warnings (judge diversity + self-judge / family bias). These
        # are advisory by default so existing configs keep running; strict_panel
        # turns them into hard errors for callers who want fail-closed composition.
        for issue in _panel_composition_messages(self.panel, target_models or [], strict=strict_panel):
            if strict_panel:
                raise ValueError(issue)
            logger.warning(issue)
        logger.debug(
            f'Initialized OWASPEvaluator with panel={self.panel}, repetitions={self.repetitions}, '
            f'replacements={self.replacement_judges}, min_successful={self.min_successful_judges}'
        )

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
                'orq.llm.purpose': 'evaluation',
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
                'orq.llm.purpose': 'evaluation',
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
        """Execute an evaluator entity against a conversation and return a typed result.

        Runs every judge in ``self.panel`` ``self.repetitions`` times, takes a
        per-judge majority vote, then a panel majority for the final verdict. Ties
        resolve fail-closed (VULNERABLE) and are flagged on the jury result. With a
        single judge and a single repetition this reduces to one LLM call and the
        ``jury`` field stays ``None`` (pre-RES-739 behaviour).
        """
        replacements = build_eval_replacements(
            input_messages=messages,
            output_messages=output_messages,
            expected_output=None,
            system_instructions=None,
        )

        # Fast path: single judge, single pass, no replacements — identical to the
        # single-judge evaluator, including its fail-loud error policy.
        if len(self.panel) == 1 and self.repetitions == 1 and not self.replacement_judges:
            outcome = await run_judge(
                client=self.client,
                model=self.evaluator_model,
                cfg=self._call_cfg,
                prompt_template=evaluator.prompt,
                replacements=replacements,
                span_attributes=span_attributes,
            )
            return self._single_outcome_to_result(outcome, evaluator_id)

        # A lone judge with no replacements has no redundancy to absorb an
        # outage. Mirror the single-judge fast path's fail-loud policy: infra
        # errors must abort the run rather than degrade every datapoint to an
        # inconclusive verdict (e.g. an invalid API key with judge_repetitions>1).
        no_redundancy = len(self.panel) == 1 and not self.replacement_judges

        async def judge_fn(model: str) -> Prediction:
            outcome = await run_judge(
                client=self.client,
                model=model,
                cfg=self._call_cfg,
                prompt_template=evaluator.prompt,
                replacements=replacements,
                span_attributes=span_attributes,
            )
            if outcome.error_kind in (JudgeError.API_CONNECTION, JudgeError.API_STATUS) and outcome.error_exc is not None:
                raise outcome.error_exc
            if outcome.error_kind is not None or outcome.payload is None:
                return Prediction(error=outcome.error_message or (outcome.error_kind.value if outcome.error_kind else 'error'))
            return Prediction(
                value=outcome.payload.value,
                explanation=outcome.payload.explanation,
                token_usage=outcome.token_usage,
                abstained=outcome.payload.abstain,
            )

        deliberation = await run_jury(
            judge_fn=judge_fn,
            panel=self.panel,
            repetitions=self.repetitions,
            replacement_judges=self.replacement_judges,
            min_successful_judges=self.min_successful_judges,
            verdict_kind=VerdictKind.CATEGORICAL,
            tie_break=lambda _values: False,
            tie_break_label='fail-closed to VULNERABLE',
            propagate_errors=no_redundancy,
        )
        final_passed = deliberation.verdict if isinstance(deliberation.verdict, bool) else None
        explanation = deliberation.explanation

        return AttackEvaluationResult(
            passed=final_passed,
            explanation=explanation,
            evaluator_id=evaluator_id,
            token_usage=deliberation.token_usage,
            raw_output={'value': final_passed, 'explanation': explanation},
            jury=deliberation.jury,
        )

    def _single_outcome_to_result(self, outcome: JudgeOutcome, evaluator_id: str) -> AttackEvaluationResult:
        """Map a single JudgeOutcome to a result, preserving the fail-loud policy.

        Infrastructure errors (API connection/status) re-raise so the run surfaces
        them; timeout and parse/unknown errors degrade to an inconclusive verdict.
        """
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

        passed = outcome.payload.value if isinstance(outcome.payload.value, bool) else None
        return AttackEvaluationResult(
            passed=passed,
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
