"""Converters between pipeline-specific formats and unified result models."""

import ast
import json
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any

from loguru import logger
from pydantic import ValidationError

from evaluatorq.redteam.contracts import (
    AgentContext,
    AgentInfo,
    AttackInfo,
    AttackStrategy,
    AttackTechnique,
    CategorySummary,
    DeliveryMethod,
    EvaluatedRow,
    ExecutionDetails,
    Framework,
    JobOutputPayload,
    OWASP_CATEGORY_NAMES,
    Pipeline,
    RedTeamReport,
    RedTeamResult,
    ReportSummary,
    TokenUsage,
    UnifiedEvaluationResult,
    infer_framework,
    normalize_category,
)

from evaluatorq.redteam.frameworks.owasp.evaluators import get_evaluator_metadata_for_category


# ---------------------------------------------------------------------------
# Static pipeline converters
# ---------------------------------------------------------------------------


def _agent_display_name(agent_context: AgentContext | None, *, fallback: str | None = None) -> str | None:
    """Best-effort stable agent label for reports."""
    if agent_context is not None:
        return agent_context.display_name or agent_context.key or fallback
    return fallback


def _coerce_job_output_payload(raw_output: Any, _depth: int = 0) -> JobOutputPayload:
    """Normalize evaluatorq job output into a typed payload."""
    if _depth > 5:
        return JobOutputPayload()

    def _normalize_output_dict(d: dict[str, Any]) -> dict[str, Any]:
        wrapped = d.get('output')
        if isinstance(wrapped, dict):
            merged = dict(wrapped)
            for k, v in d.items():
                if k != 'output' and k not in merged:
                    merged[k] = v
            return merged
        if isinstance(wrapped, str):
            nested = _coerce_job_output_payload(wrapped, _depth + 1).model_dump(mode='json')
            if nested:
                merged = dict(nested)
                for k, v in d.items():
                    if k != 'output' and k not in merged:
                        merged[k] = v
                return merged
        return d

    if isinstance(raw_output, dict):
        return JobOutputPayload.model_validate(_normalize_output_dict(raw_output))
    if raw_output is None:
        return JobOutputPayload()

    model_dump = getattr(raw_output, 'model_dump', None)
    if callable(model_dump):
        with suppress(ValidationError, TypeError):
            dumped = model_dump(mode='json')
            if isinstance(dumped, dict):
                return JobOutputPayload.model_validate(dumped)

    if isinstance(raw_output, str):
        text = raw_output.strip()
        if text.startswith('{') and text.endswith('}'):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return JobOutputPayload.model_validate(_normalize_output_dict(parsed))
                if isinstance(parsed, str):
                    return _coerce_job_output_payload(parsed, _depth + 1)
            except json.JSONDecodeError:
                with suppress(ValidationError, TypeError, ValueError, SyntaxError):
                    parsed = ast.literal_eval(text)
                    if isinstance(parsed, dict):
                        return JobOutputPayload.model_validate(_normalize_output_dict(parsed))
                    if isinstance(parsed, str):
                        return _coerce_job_output_payload(parsed, _depth + 1)
        return JobOutputPayload()

    out: dict[str, Any] = {}
    for key in (
        'conversation',
        'final_response',
        'turns',
        'objective_achieved',
        'duration_seconds',
        'token_usage',
        'token_usage_adversarial',
        'token_usage_target',
        'system_prompt',
        'error',
        'error_type',
        'error_stage',
        'error_code',
        'error_details',
        'error_turn',
        'truncated_turns',
    ):
        if hasattr(raw_output, key):
            out[key] = getattr(raw_output, key)
    return JobOutputPayload.model_validate(out)


def _coerce_job_output_text(raw_output: Any) -> str:
    """Extract best-effort response text from evaluatorq output."""
    payload = _coerce_job_output_payload(raw_output)
    if payload.final_response is not None:
        return payload.final_response
    if payload.response is not None:
        return payload.response
    if payload.output is not None:
        return payload.output
    if isinstance(raw_output, str):
        text = raw_output.strip()
        if text.startswith('{') and text.endswith('}'):
            return ''
        return raw_output
    if raw_output is None:
        return ''
    return str(raw_output)


def _normalize_attack_technique(value: str | None) -> AttackTechnique:
    """Normalize free-form/static dataset techniques to supported enum values."""
    raw = str(value or '').strip()
    if not raw:
        return AttackTechnique.INDIRECT_INJECTION
    lowered = raw.lower().replace('_', '-').replace(' ', '-')
    supported = {str(item.value): item for item in AttackTechnique}
    if lowered in supported:
        return supported[lowered]
    logger.debug(f'Unknown attack technique {raw!r}, defaulting to direct-injection')
    return AttackTechnique.DIRECT_INJECTION


def _safe_int(value: Any) -> int:
    """Convert a value to int, returning 0 on failure."""
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _normalize_token_usage(raw: Any) -> TokenUsage | None:
    """Normalize flexible usage payloads to TokenUsage."""
    if isinstance(raw, TokenUsage):
        return raw
    if not isinstance(raw, dict):
        return None
    prompt = _safe_int(raw.get('prompt_tokens', raw.get('prompt', 0)))
    completion = _safe_int(raw.get('completion_tokens', raw.get('completion', 0)))
    total = _safe_int(raw.get('total_tokens', raw.get('total', prompt + completion)))
    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
    )


def _coerce_score_passed(value: Any) -> bool | None:
    """Normalize evaluator score values to bool/None."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == 'true':
            return True
        if normalized == 'false':
            return False
        if normalized in {'error', ''}:
            return None
    return None


def static_sample_to_result(
    sample: dict[str, Any],
    agent_model: str | None = None,
    agent_key: str | None = None,
) -> RedTeamResult:
    """Convert a static pipeline sample dict to a unified RedTeamResult.

    Args:
        sample: Dict with 'input', 'messages', 'response', 'evaluation_result' keys.
        agent_model: Model used for inference (e.g., 'azure/gpt-5-mini').

    Returns:
        RedTeamResult
    """
    normalized_sample = dict(sample)
    raw_input = normalized_sample.get('input')
    if isinstance(raw_input, dict):
        normalized_input = dict(raw_input)
        # Normalize free-form dataset techniques before strict schema validation.
        normalized_input['attack_technique'] = _normalize_attack_technique(normalized_input.get('attack_technique'))
        raw_dm = normalized_input.get('delivery_method', '')
        valid_dms = {str(dm.value) for dm in DeliveryMethod}
        if raw_dm and raw_dm not in valid_dms:
            logger.debug(f'Unknown delivery_method {raw_dm!r}, defaulting to direct-request')
            normalized_input['delivery_method'] = 'direct-request'
        normalized_sample['input'] = normalized_input
    raw_eval = normalized_sample.get('evaluation_result')
    if isinstance(raw_eval, dict):
        normalized_eval = dict(raw_eval)
        normalized_eval['passed'] = _coerce_score_passed(normalized_eval.get('passed'))
        normalized_eval['value'] = _coerce_score_passed(normalized_eval.get('value'))
        normalized_sample['evaluation_result'] = normalized_eval
    row = EvaluatedRow.model_validate(normalized_sample)
    inp = row.input
    raw_category = inp.category
    category = normalize_category(raw_category)

    # Wrap singular delivery_method into list
    dm = inp.delivery_method
    delivery_methods: list[DeliveryMethod] = [DeliveryMethod(dm)] if dm else []
    evaluator_meta = get_evaluator_metadata_for_category(category) or {}

    attack = AttackInfo(
        id=inp.id,
        category=category,
        framework=infer_framework(category),
        attack_technique=_normalize_attack_technique(inp.attack_technique),
        delivery_methods=delivery_methods,
        turn_type=inp.turn_type,
        severity=inp.severity,
        scope=inp.scope,
        source=inp.source,
        evaluator_id=inp.evaluator_id or evaluator_meta.get('evaluator_id') or category,
        evaluator_name=inp.evaluator_name or evaluator_meta.get('evaluator_name') or OWASP_CATEGORY_NAMES.get(category),
        additional_metadata=inp.additional_metadata,
    )

    # Build evaluation from dict
    eval_dict = row.evaluation_result.model_dump(mode='json')
    has_eval = any([
        eval_dict.get('passed') is not None,
        eval_dict.get('value') is not None,
        bool(eval_dict.get('explanation')),
        bool(eval_dict.get('error')),
        eval_dict.get('token_usage') is not None,
    ])
    evaluation: UnifiedEvaluationResult | None = None
    if has_eval:
        evaluation_usage = _normalize_token_usage(eval_dict.get('token_usage'))
        evaluation = UnifiedEvaluationResult(
            passed=eval_dict.get('passed'),
            explanation=eval_dict.get('explanation', ''),
            evaluator_id=eval_dict.get('evaluator_id', evaluator_meta.get('evaluator_id', category)),
            evaluator_name=eval_dict.get(
                'evaluator_name', evaluator_meta.get('evaluator_name', OWASP_CATEGORY_NAMES.get(category))
            ),
            token_usage=evaluation_usage,
        )

    # Determine vulnerability status
    vulnerable = evaluation is not None and evaluation.passed is False

    agent = AgentInfo(key=agent_key, model=agent_model)

    return RedTeamResult(
        attack=attack,
        agent=agent,
        messages=row.messages,
        response=row.response,
        evaluation=evaluation,
        vulnerable=vulnerable,
        execution=None,
        error=row.error,
        error_type=row.error_type,
        error_stage=getattr(row, 'error_stage', None),
        error_code=getattr(row, 'error_code', None),
        error_details=getattr(row, 'error_details', None),
    )


def static_results_to_report(
    results: list[dict[str, Any]],
    agent_model: str | None = None,
    agent_key: str | None = None,
    description: str | None = None,
) -> RedTeamReport:
    """Bulk-convert static pipeline results to a unified RedTeamReport.

    Args:
        results: List of static sample dicts.
        agent_model: Model used for inference.
        description: Optional report description.

    Returns:
        RedTeamReport with computed summary.
    """
    unified = [static_sample_to_result(s, agent_model=agent_model, agent_key=agent_key) for s in results]
    summary = compute_report_summary(unified)
    categories = sorted({r.attack.category for r in unified})

    return RedTeamReport(
        created_at=datetime.now(tz=timezone.utc).astimezone(),
        description=description,
        pipeline=Pipeline.STATIC,
        framework=_dominant_framework(unified),
        categories_tested=categories,
        tested_agents=[agent_key] if agent_key else [],
        total_results=len(unified),
        results=unified,
        summary=summary,
    )


def dynamic_evaluatorq_results_to_report(
    *,
    agent_context: AgentContext,
    categories_tested: list[str],
    results: list[Any],
    duration_seconds: float | None = None,
    description: str | None = None,
) -> RedTeamReport:
    """Convert evaluatorq dynamic results to a unified RedTeamReport."""
    unified: list[RedTeamResult] = []

    for result in results:
        data_point = getattr(result, 'data_point', None)
        if data_point is None:
            logger.debug('Skipping result with no data_point')
            continue

        inputs = getattr(data_point, 'inputs', {}) or {}
        strategy_payload = inputs.get('strategy') or {}
        strategy = AttackStrategy.model_validate(strategy_payload)
        category = normalize_category(inputs.get('category', strategy.category))

        source = 'llm_generated_strategy' if strategy.is_generated else 'hardcoded_strategy'

        job_output = JobOutputPayload()
        eval_passed: bool | None = None
        eval_explanation = ''
        job_result = None

        job_results = getattr(result, 'job_results', None) or []
        if job_results:
            job_result = job_results[0]
            raw_output = getattr(job_result, 'output', None)
            job_output = _coerce_job_output_payload(raw_output)

            evaluator_scores = getattr(job_result, 'evaluator_scores', None) or []
            if evaluator_scores:
                score = evaluator_scores[0].score
                eval_passed = _coerce_score_passed(score.value)
                eval_explanation = score.explanation or ''

        error = getattr(result, 'error', None) or job_output.error
        error_type = job_output.error_type
        error_stage = job_output.error_stage
        error_code = job_output.error_code
        error_details = job_output.error_details
        vulnerable = eval_passed is False

        token_usage = None
        raw_usage = job_output.token_usage
        if isinstance(raw_usage, TokenUsage):
            token_usage = raw_usage
        evaluator_meta = get_evaluator_metadata_for_category(category) or {}
        evaluator_id = evaluator_meta.get('evaluator_id', category)
        evaluator_name = evaluator_meta.get('evaluator_name', OWASP_CATEGORY_NAMES.get(category))

        attack = AttackInfo(
            id=inputs.get('id', f'{category}-{strategy.name}'),
            category=category,
            framework=infer_framework(category),
            attack_technique=strategy.attack_technique,
            delivery_methods=strategy.delivery_methods,
            turn_type=strategy.turn_type,
            severity=strategy.severity,
            source=source,
            strategy_name=strategy.name,
            objective=inputs.get('objective'),
            evaluator_id=evaluator_id,
            evaluator_name=evaluator_name,
        )

        evaluation = UnifiedEvaluationResult(
            passed=eval_passed,
            explanation=eval_explanation,
            evaluator_id=evaluator_id,
            evaluator_name=evaluator_name,
        )

        execution = ExecutionDetails(
            turns=job_output.turns or 1,
            duration_seconds=job_output.duration_seconds,
            objective_achieved=job_output.objective_achieved,
            token_usage=token_usage,
        )

        agent = AgentInfo(
            key=agent_context.key,
            model=agent_context.model,
            display_name=agent_context.display_name,
        )

        unified.append(
            RedTeamResult(
                attack=attack,
                agent=agent,
                messages=job_output.conversation,
                response=_coerce_job_output_text(job_result.output if job_result is not None else job_output),
                evaluation=evaluation,
                vulnerable=vulnerable,
                execution=execution,
                error=error,
                error_type=error_type,
                error_stage=error_stage,
                error_code=error_code,
                error_details=error_details,
            )
        )

    summary = compute_report_summary(unified)
    categories = [normalize_category(c) for c in categories_tested]

    return RedTeamReport(
        created_at=datetime.now(tz=timezone.utc).astimezone(),
        description=description,
        pipeline=Pipeline.DYNAMIC,
        framework=_dominant_framework(unified),
        categories_tested=categories,
        tested_agents=[name] if (name := _agent_display_name(agent_context)) else [],
        total_results=len(unified),
        agent_context=agent_context,
        results=unified,
        summary=summary,
        duration_seconds=duration_seconds,
    )


def static_evaluatorq_results_to_reports(
    *,
    results: list[Any],
    agent_model: str | None = None,
    agent_key: str | None = None,
    description: str | None = None,
) -> dict[str, RedTeamReport]:
    """Convert evaluatorq static results into per-job unified reports."""
    samples_by_job: dict[str, list[dict[str, Any]]] = {}

    for result in results:
        data_point = getattr(result, 'data_point', None)
        inputs = getattr(data_point, 'inputs', {}) if data_point is not None else {}
        normalized_category = normalize_category(inputs.get('category', ''))
        evaluator_meta = get_evaluator_metadata_for_category(normalized_category) or {}
        base_sample = {
            'input': {
                'id': inputs.get('id') or f"{normalized_category}-{inputs.get('attack_technique', 'unknown')}",
                'category': inputs.get('category', ''),
                'attack_technique': inputs.get('attack_technique', 'indirect-injection'),
                'delivery_method': inputs.get('delivery_method', 'direct-request'),
                'severity': inputs.get('severity', 'medium'),
                'scope': inputs.get('scope', 'application'),
                'framework': inputs.get('framework', 'OWASP-AGENTIC'),
                'turn_type': inputs.get('turn_type', 'single'),
                'source': inputs.get('source', 'orq_dataset'),
                'evaluator_id': evaluator_meta.get('evaluator_id', normalized_category),
                'evaluator_name': evaluator_meta.get(
                    'evaluator_name', OWASP_CATEGORY_NAMES.get(inputs.get('category', ''))
                ) or '',
            },
            'messages': inputs.get('messages', []),
        }

        dp_error = getattr(result, 'error', None)
        job_results = getattr(result, 'job_results', None) or []
        for job_result in job_results:
            job_name = job_result.job_name

            output_dict = _coerce_job_output_payload(job_result.output)
            response = _coerce_job_output_text(job_result.output)
            output_usage = _normalize_token_usage(output_dict.token_usage)

            eval_result = {}
            scores = job_result.evaluator_scores or []
            if scores:
                score = scores[0].score
                eval_result = {
                    'passed': _coerce_score_passed(score.value),
                    'explanation': score.explanation or '',
                    'evaluator_id': evaluator_meta.get('evaluator_id', normalized_category),
                    'evaluator_name': evaluator_meta.get(
                        'evaluator_name', OWASP_CATEGORY_NAMES.get(inputs.get('category', ''))
                    ) or '',
                    'token_usage': output_usage,
                }
            elif output_usage is not None:
                eval_result = {
                    'token_usage': output_usage,
                }

            sample = {
                **base_sample,
                'response': response,
                'error': dp_error or job_result.error,
                'evaluation_result': eval_result,
            }
            samples_by_job.setdefault(job_name, []).append(sample)

    reports: dict[str, RedTeamReport] = {}
    for job_name, samples in samples_by_job.items():
        reports[job_name] = static_results_to_report(
            samples,
            agent_model=agent_model,
            agent_key=agent_key,
            description=f'{description or "Static red teaming"} ({job_name})',
        )

    return reports


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_report_summary(results: list[RedTeamResult]) -> ReportSummary:
    """Compute summary statistics from unified results."""
    if not results:
        return ReportSummary()

    total = len(results)
    evaluated = [r for r in results if r.evaluation is not None and isinstance(r.evaluation.passed, bool)]
    evaluated_total = len(evaluated)
    unevaluated_total = total - evaluated_total
    coverage = evaluated_total / total if total > 0 else 0.0
    vulns = sum(1 for r in evaluated if r.evaluation and r.evaluation.passed is False)
    resistant = evaluated_total - vulns
    resistance = resistant / evaluated_total if evaluated_total > 0 else 1.0

    # Group by category
    by_cat: dict[str, list[RedTeamResult]] = {}
    for r in results:
        cat = r.attack.category
        by_cat.setdefault(cat, []).append(r)

    cat_summaries: dict[str, CategorySummary] = {}
    for cat, cat_results in by_cat.items():
        cat_evaluated = [r for r in cat_results if r.evaluation is not None and isinstance(r.evaluation.passed, bool)]
        cat_evaluated_total = len(cat_evaluated)
        cat_vulns = sum(1 for r in cat_evaluated if r.evaluation and r.evaluation.passed is False)
        cat_resistant = cat_evaluated_total - cat_vulns
        cat_total = len(cat_results)
        cat_turns = sum(r.execution.turns for r in cat_results if r.execution)
        cat_summaries[cat] = CategorySummary(
            category=cat,
            category_name=OWASP_CATEGORY_NAMES.get(cat, cat),
            total_attacks=cat_total,
            total_conversations=cat_total,
            total_turns=cat_turns,
            vulnerabilities_found=cat_vulns,
            resistance_rate=cat_resistant / cat_evaluated_total if cat_evaluated_total > 0 else 1.0,
            strategies_used=list({r.attack.strategy_name for r in cat_results if r.attack.strategy_name}),
        )

    # Group by technique
    by_technique: dict[str, int] = {}
    for r in results:
        if r.evaluation is not None and r.evaluation.passed is False:
            tech = r.attack.attack_technique
            by_technique[tech] = by_technique.get(tech, 0) + 1

    # Count errors by type
    errors_by_type: dict[str, int] = {}
    total_errors = 0
    for r in results:
        if r.error:
            total_errors += 1
            etype = r.error_type or 'unknown'
            errors_by_type[etype] = errors_by_type.get(etype, 0) + 1

    total_turns = sum(r.execution.turns for r in results if r.execution)

    return ReportSummary(
        total_attacks=total,
        evaluated_attacks=evaluated_total,
        unevaluated_attacks=unevaluated_total,
        evaluation_coverage=coverage,
        total_conversations=total,
        total_turns=total_turns,
        vulnerabilities_found=vulns,
        resistance_rate=resistance,
        total_errors=total_errors,
        errors_by_type=errors_by_type,
        by_category=cat_summaries,
        by_technique=by_technique,
    )


def _dominant_framework(results: list[RedTeamResult]) -> Framework | None:
    """Return the dominant framework, or None if mixed/empty."""
    if not results:
        return None
    frameworks = {r.attack.framework for r in results}
    if len(frameworks) == 1:
        return frameworks.pop()
    return None
