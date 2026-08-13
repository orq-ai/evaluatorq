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
from evaluatorq.contracts import EVAL_ERROR_RAW_OUTPUT_KEY
from evaluatorq.redteam.backends.registry import create_async_llm_client
from evaluatorq.redteam.contracts import (
    DEFAULT_PIPELINE_MODEL,
    PIPELINE_CONFIG,
    AttackEvaluationResult,
    EvaluatorConfig,
    LLMCallConfig,
    OutputMessage,
    PipelineStage,
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
        # max_retries=0 on a client we build: run_judge owns retry via with_retry, and
        # the two layers would multiply into 15 requests for one judgement. An injected
        # client keeps whatever budget its owner gave it.
        self.client = llm_client or base_cfg.client or create_async_llm_client(max_retries=0)
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

        # Every judge failure is recorded, none re-raised — see
        # _single_outcome_to_result for why fail-loud was dropped. The last
        # failure's kind is kept so the panel result can name a cause instead of
        # reporting only "quorum not met".
        last_error: JudgeOutcome | None = None

        async def judge_fn(model: str) -> Prediction:
            nonlocal last_error
            outcome = await run_judge(
                client=self.client,
                model=model,
                cfg=self._call_cfg,
                prompt_template=evaluator.prompt,
                replacements=replacements,
                span_attributes=span_attributes,
            )
            if outcome.error_kind is not None or outcome.payload is None:
                last_error = outcome
                return Prediction(
                    error=outcome.error_message or (outcome.error_kind.value if outcome.error_kind else 'error'),
                    token_usage=outcome.token_usage,
                    endpoint=outcome.endpoint,
                )
            return Prediction(
                value=outcome.payload.value,
                explanation=outcome.payload.explanation,
                token_usage=outcome.token_usage,
                abstained=outcome.payload.abstain,
                endpoint=outcome.endpoint,
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
            propagate_errors=False,
        )
        final_passed = deliberation.verdict if isinstance(deliberation.verdict, bool) else None
        explanation = deliberation.explanation

        raw_output: dict[str, Any] = {
            'value': final_passed,
            'explanation': explanation,
            # Same key as the single-judge fast path's raw_output['endpoint'] (see
            # _single_outcome_to_result) so a consumer reads one key regardless of
            # which path produced the result. 'mixed' when the panel split across
            # both endpoints, None when nothing recorded one.
            'endpoint': deliberation.endpoint,
        }
        # No verdict from a panel means the quorum was not met, i.e. judges failed.
        # Without this the panel path produced passed=None with no recorded cause,
        # so a whole-panel outage was invisible to the error rollup and the CLI hint.
        if final_passed is None and last_error is not None:
            raw_output[EVAL_ERROR_RAW_OUTPUT_KEY] = self._judge_error_payload(last_error, evaluator_id)

        return AttackEvaluationResult(
            passed=final_passed,
            explanation=explanation,
            evaluator_id=evaluator_id,
            token_usage=deliberation.token_usage,
            raw_output=raw_output,
            jury=deliberation.jury,
        )

    @staticmethod
    def _judge_error_payload(outcome: JudgeOutcome, evaluator_id: str) -> dict[str, Any]:
        """Serialize a failed judge call into the shape converters lift to ``RunError``.

        ``stage`` is always 'evaluation': the attack itself ran, so this is not an
        execution error and must not be conflated with one. ``code`` is the JudgeError
        kind, which is what makes 'every judge call was blocked' legible as a single
        cause in the error rollup rather than N unrelated one-off failures.
        """
        return {
            'message': outcome.error_message or (outcome.error_kind.value if outcome.error_kind else 'unknown'),
            'error_type': outcome.error_kind.value if outcome.error_kind else 'unknown',
            'stage': PipelineStage.EVALUATION.value,
            'code': outcome.error_kind.value if outcome.error_kind else None,
            'details': {
                'evaluator_id': evaluator_id,
                # Truncated: the point is to identify the cause, not to store the payload
                # twice — the untruncated content stays under raw_output['raw_content'].
                'raw_content': (outcome.raw_content or '')[:500] or None,
                'timeout_ms': outcome.timeout_ms,
            },
        }

    def _single_outcome_to_result(self, outcome: JudgeOutcome, evaluator_id: str) -> AttackEvaluationResult:
        """Map a single JudgeOutcome to a result — every failure kind is structured.

        Infrastructure errors (API connection/status) used to re-raise on a
        fail-loud policy, but nothing upstream honoured it:
        ``evaluatorq.processings.process_evaluator`` catches every exception into
        ``EvaluatorScore.error``, so the run kept going *and* the cause was discarded. Returning a structured inconclusive
        result instead keeps the diagnosis attached to the attack, and the
        run-level coverage gate (``min_evaluation_coverage``) is what now makes
        a judge outage fail the run.
        """
        if outcome.error_kind is JudgeError.TIMEOUT:
            return AttackEvaluationResult(
                passed=None,
                explanation=f'Evaluation timed out after {outcome.timeout_ms}ms',
                evaluator_id=evaluator_id,
                token_usage=outcome.token_usage,
                raw_output={
                    'error': 'timeout',
                    'timeout_ms': outcome.timeout_ms,
                    EVAL_ERROR_RAW_OUTPUT_KEY: self._judge_error_payload(outcome, evaluator_id),
                },
            )
        if outcome.error_kind is not None or outcome.payload is None:
            return AttackEvaluationResult(
                passed=None,
                explanation=f'Evaluation error: {outcome.error_message}',
                evaluator_id=evaluator_id,
                token_usage=outcome.token_usage,
                raw_output={
                    'error': outcome.error_message,
                    'raw_content': outcome.raw_content,
                    EVAL_ERROR_RAW_OUTPUT_KEY: self._judge_error_payload(outcome, evaluator_id),
                },
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
                # Which endpoint actually served this verdict. Recorded here rather
                # than as a first-class field because it is a diagnostic: when a
                # judge reports no cost, this is what separates "the model is not in
                # the catalogue" from "Responses 400'd and fell back to chat".
                'endpoint': outcome.endpoint,
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
