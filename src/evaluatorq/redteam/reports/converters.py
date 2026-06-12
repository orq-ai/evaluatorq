"""Converters between pipeline-specific formats and unified result models."""

import json
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any, cast

from loguru import logger as _converters_logger
from pydantic import ValidationError

from evaluatorq.redteam.contracts import (
    JURY_RAW_OUTPUT_KEY,
    OWASP_CATEGORY_NAMES,
    AgentContext,
    AgentInfo,
    AttackInfo,
    AttackSource,
    AttackStrategy,
    AttackTechnique,
    CategorySummary,
    DeliveryMethod,
    DeliveryMethodSummary,
    DomainSummary,
    EvaluatedRow,
    ExecutionDetails,
    Framework,
    FrameworkSummary,
    JobOutputPayload,
    JuryReliability,
    JuryResult,
    Pipeline,
    RedTeamReport,
    RedTeamResult,
    ReportSummary,
    SeveritySummary,
    TechniqueSummary,
    TokenUsage,
    TurnTypeSummary,
    UnifiedEvaluationResult,
    Vulnerability,
    VulnerabilitySummary,
    classify_error_type,
    infer_framework,
    normalize_category,
)
from evaluatorq.redteam.frameworks.owasp.evaluators import get_evaluator_metadata_for_category
from evaluatorq.redteam.runtime.jobs import _normalize_usage as _normalize_token_usage
from evaluatorq.redteam.vulnerability_registry import (
    VULNERABILITY_DEFS,
    get_framework_categories,
    get_vulnerability_name,
    resolve_category_safe,
)

# Error classification lives in evaluatorq.redteam.contracts so the orchestrator
# can share it without importing the report layer. Alias kept for call sites here.
_classify_error = classify_error_type


def _extract_jury(raw_output: Any) -> JuryResult | None:
    """Lift the panel-of-judges breakdown out of the scorer's ``raw_output``.

    The dynamic scorer stashes the jury under ``raw_output[JURY_RAW_OUTPUT_KEY]``
    (the generic EvaluationResult has no typed field for it); here it is
    reconstructed onto the typed report model so per-judge votes + agreement reach
    the report (RES-739). Returns None for single-judge runs or malformed payloads.
    """
    if not isinstance(raw_output, dict):
        return None
    jury_data = raw_output.get(JURY_RAW_OUTPUT_KEY)
    if not isinstance(jury_data, dict):
        return None
    try:
        return JuryResult.model_validate(jury_data)
    except ValidationError as e:
        _converters_logger.warning(f'Discarding malformed jury payload in report conversion: {e}')
        return None


def _raw_output_without_jury(raw_output: Any) -> dict[str, Any] | None:
    """Return ``raw_output`` minus the stashed jury, which is lifted to the typed
    ``jury`` field — so the per-judge breakdown is not serialized twice (RES-739).
    """
    if not isinstance(raw_output, dict) or JURY_RAW_OUTPUT_KEY not in raw_output:
        return raw_output
    return {k: v for k, v in raw_output.items() if k != JURY_RAW_OUTPUT_KEY}


# ---------------------------------------------------------------------------
# Static pipeline converters
# ---------------------------------------------------------------------------


def _coerce_job_output_payload(raw_output: Any) -> JobOutputPayload:
    """Normalize evaluatorq job output into a typed payload."""

    def _flatten_turns(d: dict[str, Any]) -> dict[str, Any]:
        """If ``turns`` is a list of new-shape Turn dicts, translate to the legacy
        wire fields (turns:int, conversation:list[Message-dict], final_response)
        so JobOutputPayload validates.
        """
        turns_val = d.get('turns')
        if not isinstance(turns_val, list) or not turns_val:
            return d
        first = turns_val[0]
        if not (isinstance(first, dict) and 'attacker' in first and 'target' in first):
            return d
        conversation: list[dict[str, Any]] = []
        final_response_text = ''
        for t in turns_val:
            attacker = t.get('attacker') or {}
            target = t.get('target') or {}
            conversation.append({'role': 'user', 'content': attacker.get('generated_prompt', '')})
            target_text = ''
            for item in target.get('output') or []:
                if not isinstance(item, dict):
                    continue
                if item.get('type') == 'output_text':
                    target_text = item.get('text', '') or target_text
            conversation.append({'role': 'assistant', 'content': target_text})
            final_response_text = target_text or final_response_text
        out = dict(d)
        out['turns'] = len(turns_val)
        out.setdefault('conversation', conversation)
        out.setdefault('final_response', final_response_text)
        return out

    def _normalize_output_dict(d: dict[str, Any]) -> dict[str, Any]:
        """Unwrap nested 'output' keys and merge top-level fields into a flat dict."""
        wrapped = d.get('output')
        if isinstance(wrapped, dict):
            merged = dict(wrapped)
            for k, v in d.items():
                if k != 'output' and k not in merged:
                    merged[k] = v
            return _flatten_turns(merged)
        if isinstance(wrapped, str):
            nested = _coerce_job_output_payload(wrapped).model_dump(mode='json')
            if nested:
                merged = dict(nested)
                for k, v in d.items():
                    if k != 'output' and k not in merged:
                        merged[k] = v
                return _flatten_turns(merged)
        return _flatten_turns(d)

    if isinstance(raw_output, dict):
        return JobOutputPayload.model_validate(_normalize_output_dict(raw_output))
    if raw_output is None:
        return JobOutputPayload()

    model_dump = getattr(raw_output, 'model_dump', None)
    if callable(model_dump):
        with suppress(Exception):
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
                    return _coerce_job_output_payload(parsed)
            except json.JSONDecodeError as exc:
                _converters_logger.debug(f'JSON parse failed for job output: {exc}')
        return JobOutputPayload()

    out: dict[str, Any] = {}
    for key in (
        'conversation',
        'final_response',
        'turns',
        'objective_achieved',
        'objective_rationale',
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
    supported = {str(item.value) for item in AttackTechnique}
    if lowered in supported:
        return AttackTechnique(lowered)
    # Common jailbreak-like labels in LLM datasets map to direct injection.
    _converters_logger.warning('Unknown attack technique %r, defaulting to direct-injection', raw)
    return AttackTechnique.DIRECT_INJECTION


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
    vuln = resolve_category_safe(category)
    vulnerability_str = vuln.value if vuln else ''

    # Wrap singular delivery_method into list
    dm = inp.delivery_method
    delivery_methods: list[DeliveryMethod] = [DeliveryMethod(dm)] if dm else []
    evaluator_meta = get_evaluator_metadata_for_category(category) or {}

    attack = AttackInfo(
        id=inp.id,
        vulnerability=vulnerability_str,
        category=category,
        framework=cast('Framework', infer_framework(category)),
        attack_technique=_normalize_attack_technique(inp.attack_technique),
        delivery_methods=delivery_methods,
        turn_type=inp.turn_type,
        severity=inp.severity,
        vulnerability_domain=inp.vulnerability_domain,
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
        eval_dict.get('raw_output') is not None,
    ])
    evaluation: UnifiedEvaluationResult | None = None
    if has_eval:
        evaluation_usage = _normalize_token_usage(eval_dict.get('token_usage'))
        evaluation = UnifiedEvaluationResult(
            value=eval_dict.get('value'),
            passed=eval_dict.get('passed'),
            explanation=eval_dict.get('explanation', ''),
            evaluator_id=eval_dict.get('evaluator_id', evaluator_meta.get('evaluator_id', category)),
            evaluator_name=eval_dict.get(
                'evaluator_name', evaluator_meta.get('evaluator_name', OWASP_CATEGORY_NAMES.get(category))
            ),
            token_usage=evaluation_usage,
            raw_output=_raw_output_without_jury(eval_dict.get('raw_output')),
            jury=_extract_jury(eval_dict.get('raw_output')),
        )

    # Determine vulnerability status
    vulnerable = evaluation is not None and evaluation.passed is False

    agent = AgentInfo(key=agent_key, model=agent_model)

    # Target-generation cost (one inference) lives on execution, kept separate from
    # the judge cost on the evaluation. ``execution_token_usage`` is an optional
    # passthrough set by the dynamic-datapoint static converter.
    execution_usage = _normalize_token_usage(normalized_sample.get('execution_token_usage'))

    return RedTeamResult(
        attack=attack,
        agent=agent,
        messages=row.messages,
        response=row.response,
        evaluation=evaluation,
        vulnerable=vulnerable,
        execution=ExecutionDetails(turns=1, max_turns=1, token_usage=execution_usage),
        error=row.error,
        error_type=_classify_error(row.error, existing_type=row.error_type),
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
    results: list[Any],
    duration_seconds: float | None = None,
    description: str | None = None,
    categories_tested: list[str] | None = None,  # deprecated, derived from results
) -> RedTeamReport:
    """Convert evaluatorq dynamic results to a unified RedTeamReport."""
    unified: list[RedTeamResult] = []

    for result in results:
        data_point = getattr(result, 'data_point', None)
        if data_point is None:
            continue

        inputs = getattr(data_point, 'inputs', {}) or {}
        strategy_payload = inputs.get('strategy', {})
        strategy = AttackStrategy.model_validate(strategy_payload)
        category = normalize_category(inputs.get('category', strategy.category))
        vulnerability_str = inputs.get('vulnerability', '')
        if not vulnerability_str:
            vuln = resolve_category_safe(category)
            vulnerability_str = vuln.value if vuln else ''

        source = AttackSource.GENERATED_DYNAMIC if strategy.is_generated else AttackSource.TEMPLATE_DYNAMIC

        job_output = JobOutputPayload()
        eval_passed: bool | None = None
        eval_value: Any = None
        eval_explanation = ''
        evaluation_usage: TokenUsage | None = None
        evaluation_raw: dict[str, Any] | None = None
        job_result = None

        job_results = getattr(result, 'job_results', None) or []
        if job_results:
            job_result = job_results[0]
            raw_output = getattr(job_result, 'output', None)
            job_output = _coerce_job_output_payload(raw_output)

            evaluator_scores = getattr(job_result, 'evaluator_scores', None) or []
            if evaluator_scores:
                score = evaluator_scores[0].score
                eval_value = score.value
                eval_passed = _coerce_score_passed(score.value)
                eval_explanation = score.explanation or ''
                # The scorer forwards the LLM judge's cost + raw response on the
                # optional token_usage/raw_output EvaluationResult fields (serialized
                # in local dumps, stripped at the Orq send boundary — see
                # evaluatorq.send_results); read them off the live score so the report
                # surfaces and aggregates evaluator token usage.
                # getattr-guarded: ``score`` is an Any-typed external (evaluatorq)
                # result, and these two are optional metadata that other scorers omit.
                evaluation_usage = _normalize_token_usage(getattr(score, 'token_usage', None))
                evaluation_raw = getattr(score, 'raw_output', None)

        error = getattr(result, 'error', None) or job_output.error
        error_type = _classify_error(error, existing_type=job_output.error_type)
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

        vuln_key = strategy.vulnerability or resolve_category_safe(category)
        vuln_def = VULNERABILITY_DEFS.get(vuln_key) if vuln_key else None
        attack = AttackInfo(
            id=inputs.get('id', f'{category}-{strategy.name}'),
            vulnerability=vulnerability_str,
            category=category,
            framework=cast('Framework', infer_framework(category)),
            attack_technique=strategy.attack_technique,
            delivery_methods=strategy.delivery_methods,
            turn_type=strategy.turn_type,
            severity=strategy.severity,
            vulnerability_domain=vuln_def.domain if vuln_def else None,
            source=source,
            strategy_name=strategy.name,
            objective=inputs.get('objective'),
            evaluator_id=evaluator_id,
            evaluator_name=evaluator_name,
        )

        evaluation = UnifiedEvaluationResult(
            value=eval_value,
            passed=eval_passed,
            explanation=eval_explanation,
            evaluator_id=evaluator_id,
            evaluator_name=evaluator_name,
            token_usage=evaluation_usage,
            raw_output=_raw_output_without_jury(evaluation_raw),
            jury=_extract_jury(evaluation_raw),
        )

        execution = ExecutionDetails(
            turns=job_output.turns or 1,
            max_turns=job_output.max_turns,
            duration_seconds=job_output.duration_seconds,
            objective_achieved=job_output.objective_achieved,
            objective_rationale=job_output.objective_rationale,
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
    # Derive categories from actual results; fall back to caller-provided list
    # when all results were skipped (e.g. missing data_point).
    categories = sorted({r.attack.category for r in unified})
    if not categories and categories_tested:
        _converters_logger.warning(
            'No results produced data points; falling back to %d requested categories.',
            len(categories_tested),
        )
        categories = sorted({normalize_category(c) for c in categories_tested})

    return RedTeamReport(
        created_at=datetime.now(tz=timezone.utc).astimezone(),
        description=description,
        pipeline=Pipeline.DYNAMIC,
        framework=_dominant_framework(unified),
        categories_tested=categories,
        tested_agents=[agent_context.key or agent_context.display_name or 'unknown'],
        total_results=len(unified),
        agent_contexts={(agent_context.key or agent_context.display_name or 'unknown'): agent_context},
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
        vuln = resolve_category_safe(normalized_category)
        base_sample = {
            'input': {
                'id': inputs.get('id', ''),
                'vulnerability': vuln.value if vuln else '',
                'category': inputs.get('category', ''),
                'attack_technique': inputs.get('attack_technique', 'indirect-injection'),
                'delivery_method': inputs.get('delivery_method', 'direct-request'),
                'severity': inputs.get('severity', 'medium'),
                'scope': inputs.get('scope', 'application'),
                'framework': inputs.get('framework', 'OWASP-AGENTIC'),
                'turn_type': inputs.get('turn_type', 'single'),
                'source': inputs.get('source', AttackSource.ORQ_DATASET),
                'evaluator_id': evaluator_meta.get('evaluator_id', normalized_category),
                'evaluator_name': evaluator_meta.get(
                    'evaluator_name', OWASP_CATEGORY_NAMES.get(inputs.get('category', ''))
                ),
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
                # The judge's own cost goes on the evaluation; the target's
                # generation cost (output_usage) is routed to execution below so the
                # two stay separable (and each correctly reports a single call).
                eval_result = {
                    'value': score.value,
                    'passed': _coerce_score_passed(score.value),
                    'explanation': score.explanation or '',
                    'evaluator_id': evaluator_meta.get('evaluator_id', normalized_category),
                    'evaluator_name': evaluator_meta.get(
                        'evaluator_name', OWASP_CATEGORY_NAMES.get(inputs.get('category', ''))
                    ),
                    'token_usage': getattr(score, 'token_usage', None),
                    'raw_output': getattr(score, 'raw_output', None),
                }

            raw_error = dp_error or job_result.error or output_dict.error
            sample = {
                **base_sample,
                'response': response,
                'error': raw_error,
                'error_type': _classify_error(raw_error, existing_type=output_dict.error_type),
                'error_stage': output_dict.error_stage,
                'error_code': output_dict.error_code,
                'error_details': output_dict.error_details,
                # Target-generation usage → execution slot (kept apart from the judge).
                'execution_token_usage': output_usage.model_dump(mode='json') if output_usage else None,
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


def _rate(numerator: int, denominator: int, *, default: float = 1.0) -> float:
    """Safe division returning *default* when denominator is zero."""
    return numerator / denominator if denominator > 0 else default


def _is_evaluated(r: RedTeamResult) -> bool:
    """Return True if the result has a definitive boolean evaluation outcome."""
    return r.evaluation is not None and isinstance(r.evaluation.passed, bool)


def _is_vulnerable(r: RedTeamResult) -> bool:
    """Return True if the result was evaluated and the agent failed (is vulnerable)."""
    return r.evaluation is not None and r.evaluation.passed is False


def _aggregate_token_usage(results: list[RedTeamResult]) -> TokenUsage | None:
    """Sum token usage across all results, covering both execution and evaluator calls.

    Walks both ``r.execution.token_usage`` (orchestrator: adversarial + target)
    and ``r.evaluation.token_usage`` (per-attack evaluator) so the report's
    grand total reflects the full LLM cost of the run.
    """
    total = TokenUsage()
    found = False
    for r in results:
        if r.execution and r.execution.token_usage:
            total = total + r.execution.token_usage
            found = True
        if r.evaluation and r.evaluation.token_usage:
            total = total + r.evaluation.token_usage
            found = True
    return total if found else None


def _krippendorff_alpha_binary(units: list[list[bool]]) -> tuple[float | None, int]:
    """Nominal Krippendorff's alpha for binary verdicts across samples.

    ``units`` is one list of judge verdicts per sample. Only samples with >=2
    verdicts are pairable. For two nominal categories the coincidence form reduces
    to a closed expression:

        alpha = 1 - (n - 1) * O01 / (n0 * n1)

    where ``O01 = sum over units of c0*c1/(m-1)`` (m verdicts on the unit, c0/c1
    the per-category counts), and ``n0``/``n1`` are the total per-category verdicts
    across pairable units. Returns ``(alpha, pairable_sample_count)``; alpha is
    ``None`` when undefined — fewer than two pairable verdicts, or every verdict
    identical (n0*n1 == 0, no expected disagreement to correct against).
    """
    o01 = 0.0
    n0 = 0
    n1 = 0
    pairable = 0
    for ratings in units:
        m = len(ratings)
        if m < 2:
            continue
        pairable += 1
        c1 = sum(1 for r in ratings if r)
        c0 = m - c1
        o01 += (c0 * c1) / (m - 1)
        n0 += c0
        n1 += c1
    n = n0 + n1
    if n < 2 or n0 * n1 == 0:
        return None, pairable
    return 1.0 - (n - 1) * o01 / (n0 * n1), pairable


def _compute_jury_reliability(results: list[RedTeamResult]) -> JuryReliability | None:
    """Aggregate per-sample jury verdicts into a run-level reliability statistic.

    Returns None when no sample carried a multi-judge jury (single-judge runs),
    so the field stays absent for the common path.
    """
    units: list[list[bool]] = []
    for r in results:
        jury = r.evaluation.jury if r.evaluation else None
        if jury is None:
            continue
        verdicts = [v.value for v in jury.votes if v.success and not v.abstained and isinstance(v.value, bool)]
        if len(verdicts) >= 2:
            units.append(verdicts)
    if not units:
        return None
    alpha, samples = _krippendorff_alpha_binary(units)
    return JuryReliability(krippendorff_alpha=alpha, samples=samples)


def compute_report_summary(results: list[RedTeamResult]) -> ReportSummary:
    """Compute summary statistics from unified results."""
    if not results:
        return ReportSummary()

    total = len(results)
    evaluated = [r for r in results if _is_evaluated(r)]
    evaluated_total = len(evaluated)
    unevaluated_total = total - evaluated_total
    coverage = _rate(evaluated_total, total, default=0.0)
    vulns = sum(1 for r in evaluated if _is_vulnerable(r))
    resistant = evaluated_total - vulns
    resistance = _rate(resistant, evaluated_total, default=0.0)

    total_turns = sum(r.execution.turns for r in results if r.execution)

    # ── Group by category ──────────────────────────────────────────────
    by_cat: dict[str, list[RedTeamResult]] = {}
    for r in results:
        by_cat.setdefault(r.attack.category, []).append(r)

    cat_summaries: dict[str, CategorySummary] = {}
    for cat, cat_results in by_cat.items():
        cat_eval = [r for r in cat_results if _is_evaluated(r)]
        cat_eval_total = len(cat_eval)
        cat_total = len(cat_results)
        cat_vulns = sum(1 for r in cat_eval if _is_vulnerable(r))
        cat_resistant = cat_eval_total - cat_vulns
        cat_turns = sum(r.execution.turns for r in cat_results if r.execution)
        cat_errors = sum(1 for r in cat_results if r.error)
        cat_summaries[cat] = CategorySummary(
            category=cat,
            category_name=OWASP_CATEGORY_NAMES.get(cat, cat),
            total_attacks=cat_total,
            evaluated_attacks=cat_eval_total,
            unevaluated_attacks=cat_total - cat_eval_total,
            evaluation_coverage=_rate(cat_eval_total, cat_total, default=0.0),
            total_conversations=cat_total,
            total_turns=cat_turns,
            vulnerabilities_found=cat_vulns,
            vulnerability_rate=_rate(cat_vulns, cat_eval_total, default=0.0),
            resistance_rate=_rate(cat_resistant, cat_eval_total, default=0.0),
            total_errors=cat_errors,
            strategies_used=list({r.attack.strategy_name for r in cat_results if r.attack.strategy_name}),
        )

    # ── Group by vulnerability ────────────────────────────────────────
    by_vuln: dict[str, list[RedTeamResult]] = {}
    unresolved_vuln_count = 0
    for r in results:
        v = r.attack.vulnerability
        if v:
            by_vuln.setdefault(v, []).append(r)
        else:
            unresolved_vuln_count += 1

    if unresolved_vuln_count > 0:
        _converters_logger.warning(
            '%d result(s) have no vulnerability identifier and are excluded from by_vulnerability.',
            unresolved_vuln_count,
        )

    vuln_summaries: dict[str, VulnerabilitySummary] = {}
    for v, v_results in by_vuln.items():
        try:
            vuln_enum = Vulnerability(v)
        except ValueError:
            _converters_logger.warning('Skipping unknown vulnerability %r in by_vulnerability grouping.', v)
            continue
        vdef = VULNERABILITY_DEFS.get(vuln_enum)
        v_eval = [r for r in v_results if _is_evaluated(r)]
        v_eval_total = len(v_eval)
        v_vulns = sum(1 for r in v_eval if _is_vulnerable(r))
        v_resistant = v_eval_total - v_vulns
        vuln_summaries[v] = VulnerabilitySummary(
            vulnerability=v,
            vulnerability_name=get_vulnerability_name(vuln_enum),
            domain=vdef.domain.value if vdef else '',
            total_attacks=len(v_results),
            vulnerabilities_found=v_vulns,
            resistance_rate=_rate(v_resistant, v_eval_total, default=0.0),
            strategies_used=list({r.attack.strategy_name for r in v_results if r.attack.strategy_name}),
            framework_categories=get_framework_categories(vuln_enum),
        )

    # ── Group by technique ─────────────────────────────────────────────
    tech_totals: dict[str, int] = {}
    tech_eval_totals: dict[str, int] = {}
    tech_vulns: dict[str, int] = {}
    for r in results:
        tech = r.attack.attack_technique
        tech_totals[tech] = tech_totals.get(tech, 0) + 1
        if _is_evaluated(r):
            tech_eval_totals[tech] = tech_eval_totals.get(tech, 0) + 1
        if _is_vulnerable(r):
            tech_vulns[tech] = tech_vulns.get(tech, 0) + 1
    by_technique: dict[str, TechniqueSummary] = {}
    for tech, t_total in tech_totals.items():
        t_eval = tech_eval_totals.get(tech, 0)
        t_vulns = tech_vulns.get(tech, 0)
        t_resistant = t_eval - t_vulns
        by_technique[tech] = TechniqueSummary(
            total_attacks=t_total,
            vulnerabilities_found=t_vulns,
            resistance_rate=_rate(t_resistant, t_eval, default=0.0),
            vulnerability_rate=_rate(t_vulns, t_eval, default=0.0),
        )

    # ── Group by severity ──────────────────────────────────────────────
    sev_totals: dict[str, int] = {}
    sev_eval_totals: dict[str, int] = {}
    sev_vulns: dict[str, int] = {}
    for r in results:
        sev = r.attack.severity
        sev_totals[sev] = sev_totals.get(sev, 0) + 1
        if _is_evaluated(r):
            sev_eval_totals[sev] = sev_eval_totals.get(sev, 0) + 1
        if _is_vulnerable(r):
            sev_vulns[sev] = sev_vulns.get(sev, 0) + 1
    by_severity: dict[str, SeveritySummary] = {}
    for sev, s_total in sev_totals.items():
        s_eval = sev_eval_totals.get(sev, 0)
        s_vulns = sev_vulns.get(sev, 0)
        s_resistant = s_eval - s_vulns
        by_severity[sev] = SeveritySummary(
            total_attacks=s_total,
            vulnerabilities_found=s_vulns,
            resistance_rate=_rate(s_resistant, s_eval, default=0.0),
            vulnerability_rate=_rate(s_vulns, s_eval, default=0.0),
        )

    # ── Group by delivery method ───────────────────────────────────────
    dm_totals: dict[str, int] = {}
    dm_eval_totals: dict[str, int] = {}
    dm_vulns: dict[str, int] = {}
    for r in results:
        for dm in r.attack.delivery_methods:
            dm_totals[dm] = dm_totals.get(dm, 0) + 1
            if _is_evaluated(r):
                dm_eval_totals[dm] = dm_eval_totals.get(dm, 0) + 1
            if _is_vulnerable(r):
                dm_vulns[dm] = dm_vulns.get(dm, 0) + 1
    by_delivery_method: dict[str, DeliveryMethodSummary] = {}
    for dm, d_total in dm_totals.items():
        d_eval = dm_eval_totals.get(dm, 0)
        d_vulns = dm_vulns.get(dm, 0)
        d_resistant = d_eval - d_vulns
        by_delivery_method[dm] = DeliveryMethodSummary(
            total_attacks=d_total,
            vulnerabilities_found=d_vulns,
            resistance_rate=_rate(d_resistant, d_eval, default=0.0),
            vulnerability_rate=_rate(d_vulns, d_eval, default=0.0),
        )

    # ── Group by turn type ─────────────────────────────────────────────
    tt_totals: dict[str, int] = {}
    tt_eval_totals: dict[str, int] = {}
    tt_vulns: dict[str, int] = {}
    tt_turns: dict[str, int] = {}
    for r in results:
        tt = r.attack.turn_type
        tt_totals[tt] = tt_totals.get(tt, 0) + 1
        if _is_evaluated(r):
            tt_eval_totals[tt] = tt_eval_totals.get(tt, 0) + 1
        if r.execution:
            tt_turns[tt] = tt_turns.get(tt, 0) + r.execution.turns
        if _is_vulnerable(r):
            tt_vulns[tt] = tt_vulns.get(tt, 0) + 1
    by_turn_type: dict[str, TurnTypeSummary] = {}
    for tt, t_total in tt_totals.items():
        t_eval = tt_eval_totals.get(tt, 0)
        t_vulns = tt_vulns.get(tt, 0)
        t_resistant = t_eval - t_vulns
        by_turn_type[tt] = TurnTypeSummary(
            total_attacks=t_total,
            vulnerabilities_found=t_vulns,
            resistance_rate=_rate(t_resistant, t_eval, default=0.0),
            vulnerability_rate=_rate(t_vulns, t_eval, default=0.0),
            average_turns=tt_turns.get(tt, 0) / t_total if t_total > 0 else 0.0,
        )

    # ── Group by vulnerability domain ──────────────────────────────────
    domain_totals: dict[str, int] = {}
    domain_eval_totals: dict[str, int] = {}
    domain_vulns: dict[str, int] = {}
    for r in results:
        sc = r.attack.vulnerability_domain
        if sc is not None:
            domain_totals[sc] = domain_totals.get(sc, 0) + 1
            if _is_evaluated(r):
                domain_eval_totals[sc] = domain_eval_totals.get(sc, 0) + 1
            if _is_vulnerable(r):
                domain_vulns[sc] = domain_vulns.get(sc, 0) + 1
    by_domain: dict[str, DomainSummary] = {}
    for sc, s_total in domain_totals.items():
        s_eval = domain_eval_totals.get(sc, 0)
        s_vulns = domain_vulns.get(sc, 0)
        s_resistant = s_eval - s_vulns
        by_domain[sc] = DomainSummary(
            total_attacks=s_total,
            vulnerabilities_found=s_vulns,
            resistance_rate=_rate(s_resistant, s_eval, default=0.0),
            vulnerability_rate=_rate(s_vulns, s_eval, default=0.0),
        )

    # ── Group by framework ─────────────────────────────────────────────
    fw_totals: dict[str, int] = {}
    fw_eval_totals: dict[str, int] = {}
    fw_vulns: dict[str, int] = {}
    for r in results:
        fw = r.attack.framework
        fw_totals[fw] = fw_totals.get(fw, 0) + 1
        if _is_evaluated(r):
            fw_eval_totals[fw] = fw_eval_totals.get(fw, 0) + 1
        if _is_vulnerable(r):
            fw_vulns[fw] = fw_vulns.get(fw, 0) + 1
    by_framework: dict[str, FrameworkSummary] = {}
    for fw, f_total in fw_totals.items():
        f_eval = fw_eval_totals.get(fw, 0)
        f_vulns = fw_vulns.get(fw, 0)
        f_resistant = f_eval - f_vulns
        by_framework[fw] = FrameworkSummary(
            total_attacks=f_total,
            vulnerabilities_found=f_vulns,
            resistance_rate=_rate(f_resistant, f_eval, default=0.0),
            vulnerability_rate=_rate(f_vulns, f_eval, default=0.0),
        )

    # ── Count errors by type ───────────────────────────────────────────
    errors_by_type: dict[str, int] = {}
    total_errors = 0
    for r in results:
        if r.error:
            total_errors += 1
            # Prefer error_code (specific) over error_type (generic)
            etype = r.error_code or r.error_type or 'unknown'
            errors_by_type[etype] = errors_by_type.get(etype, 0) + 1

    return ReportSummary(
        total_attacks=total,
        evaluated_attacks=evaluated_total,
        unevaluated_attacks=unevaluated_total,
        evaluation_coverage=coverage,
        total_conversations=total,
        total_turns=total_turns,
        average_turns_per_attack=total_turns / total if total > 0 else 0.0,
        vulnerabilities_found=vulns,
        vulnerability_rate=_rate(vulns, evaluated_total, default=0.0),
        resistance_rate=resistance,
        total_errors=total_errors,
        errors_by_type=errors_by_type,
        token_usage_total=_aggregate_token_usage(results),
        by_vulnerability=vuln_summaries,
        by_category=cat_summaries,
        by_technique=by_technique,
        by_severity=by_severity,
        by_delivery_method=by_delivery_method,
        by_turn_type=by_turn_type,
        by_domain=by_domain,
        by_framework=by_framework,
        jury_reliability=_compute_jury_reliability(results),
    )


def _dominant_framework(results: list[RedTeamResult]) -> Framework | None:
    """Return the dominant framework, or None if mixed/empty."""
    if not results:
        return None
    frameworks = {r.attack.framework for r in results}
    if len(frameworks) == 1:
        return frameworks.pop()
    return None


def merge_reports(
    *reports: RedTeamReport,
    description: str | None = None,
) -> RedTeamReport:
    """Merge multiple RedTeamReports into a single unified report.

    Concatenates results, unions categories_tested and tested_agents,
    resolves framework/pipeline labels, and recomputes the summary.

    Args:
        *reports: One or more RedTeamReport instances to merge.
        description: Optional description for the merged report.

    Returns:
        A new RedTeamReport combining all inputs.

    Raises:
        ValueError: If no reports are provided.
    """
    if not reports:
        msg = 'merge_reports() requires at least one report'
        raise ValueError(msg)

    if len(reports) == 1:
        return reports[0]

    all_results: list[RedTeamResult] = []
    frameworks: set[Framework | None] = set()
    pipelines: set[Pipeline] = set()
    merged_agent_contexts: dict[str, AgentContext] = {}

    for report in reports:
        all_results.extend(report.results)
        frameworks.add(report.framework)
        pipelines.add(report.pipeline)
        # Collect agent_contexts from all sub-reports
        if report.agent_contexts:
            merged_agent_contexts.update(report.agent_contexts)

    # Resolve framework: use single value if unanimous, else None
    resolved_framework: Framework | None = None
    non_none_frameworks = {f for f in frameworks if f is not None}
    if len(non_none_frameworks) == 1:
        resolved_framework = non_none_frameworks.pop()

    # Resolve pipeline: use single value if unanimous, else 'hybrid'
    resolved_pipeline: Pipeline = Pipeline.HYBRID
    if len(pipelines) == 1:
        resolved_pipeline = pipelines.pop()

    summary = compute_report_summary(all_results)

    # Derive categories from actual results, not sub-report metadata
    all_categories = sorted({r.attack.category for r in all_results})
    # Union agent_contexts keys with sub-report tested_agents so that
    # agents from static reports (which lack agent_context) are not lost.
    all_agent_keys: set[str] = set(merged_agent_contexts.keys())
    for report in reports:
        all_agent_keys.update(report.tested_agents)
    all_agents = sorted(all_agent_keys)

    return RedTeamReport(
        created_at=datetime.now(tz=timezone.utc).astimezone(),
        description=description or 'Merged report',
        pipeline=resolved_pipeline,
        framework=resolved_framework,
        categories_tested=all_categories,
        tested_agents=all_agents,
        total_results=len(all_results),
        agent_contexts=merged_agent_contexts,
        results=all_results,
        summary=summary,
    )
