"""EvaluatorQ adapter for replaying projected traces through the classifier."""

from __future__ import annotations

import json
import math
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq.common.judge import JudgeOutcome, judge_error_payload, run_judge
from evaluatorq.common.llm_call import classify_request_body
from evaluatorq.contracts import JuryRepetition, JuryResult, JuryVote, LLMCallConfig
from evaluatorq.evaluatorq import evaluatorq
from evaluatorq.types import DataPoint, DataPointResult, EvaluationResult, Evaluator, ScorerParameter

from .debug import enabled as debug_enabled
from .models import CompiledQuery, TraceClassification, TraceProjection, TraceRecord

if TYPE_CHECKING:
    from openai import AsyncOpenAI


def build_datapoint(trace: TraceRecord, projection: TraceProjection) -> DataPoint:
    """Build a no-inference replay row while retaining the projected classifier state."""

    return DataPoint(
        inputs={
            'trace_id': trace.trace_id,
            'span_id': trace.span_id,
            'classifier_state': projection.payload,
            'messages': [{'role': 'assistant', 'content': 'classifier state prepared'}],
        }
    )


def build_classifier_evaluator(compiled: CompiledQuery, *, model: str, client: AsyncOpenAI) -> Evaluator:
    """Configure one merged EvaluatorQ classify judge over the exact projected state."""

    async def score(params: ScorerParameter) -> EvaluationResult:
        state = params['data'].inputs['classifier_state']
        trace_id = params['data'].inputs['trace_id']
        question = compiled.task.model_copy(update={'state': state})
        if debug_enabled():
            logger.debug(
                'Trace finder trace classifier request trace_id={} model={} input={}',
                trace_id,
                model,
                json.dumps(classify_request_body(model, question.state, {'verdict': question}), ensure_ascii=False),
            )
        outcome = await run_judge(
            client=client,
            model=model,
            cfg=LLMCallConfig(model=model, timeout_ms=90_000),
            prompt_template='',
            replacements={},
            span_attributes={'orq.llm.purpose': 'judge'},
            classify=question,
        )
        if debug_enabled():
            logger.debug(
                'Trace finder trace classifier response trace_id={} model={} endpoint={} output={} error={}',
                trace_id,
                model,
                outcome.endpoint,
                outcome.raw_content or (json.dumps(outcome.raw_output) if outcome.raw_output is not None else ''),
                outcome.error_message,
            )
        return _outcome_result(outcome, model=model)

    return {'name': 'classifier', 'scorer': score}


def _outcome_result(outcome: JudgeOutcome, *, model: str) -> EvaluationResult:
    """Keep the familiar single-seat jury result shape around a direct classify outcome."""

    payload = outcome.payload
    payload_value = payload.value if payload is not None else None
    payload_explanation = payload.explanation if payload is not None else None
    mechanical_success = outcome.error_kind is None and payload is not None
    abstained = mechanical_success and bool(payload and payload.abstain)
    decisive = mechanical_success and not abstained and payload_value is not None
    success = decisive or abstained
    message = outcome.error_message or (None if success else 'The classifier returned no verdict')
    repetition = JuryRepetition(
        value=payload_value if decisive else None,
        explanation=payload_explanation,
        raw_output=outcome.raw_output,
    )
    vote = JuryVote(
        model=model,
        success=success,
        abstained=abstained,
        value=payload_value if decisive else None,
        explanation=payload_explanation or '',
        error=message if not success else None,
        repetitions=[repetition],
        repetitions_failed=0 if success else 1,
    )
    jury = JuryResult(
        judges_configured=1,
        judges_succeeded=1 if decisive else 0,
        judges_failed=0 if success else 1,
        inconclusive=not decisive,
        votes=[vote],
        raw_agreement=1.0 if decisive else None,
    )
    raw_output: dict[str, Any] = {'jury': jury.model_dump(mode='json')}
    if not success:
        raw_output['evaluation_error'] = (
            judge_error_payload(outcome, 'classifier')
            if outcome.error_kind is not None
            else {
                'message': message,
                'error_type': 'no_verdict',
                'stage': 'evaluation',
                'code': 'no_verdict',
                'details': {'evaluator_id': 'classifier'},
            }
        )
    result_value = payload_value if success and payload_value is not None else 'inconclusive'
    return EvaluationResult(
        value=result_value,
        explanation=payload_explanation if payload is not None else message,
        token_usage=outcome.token_usage,
        raw_output=raw_output,
    )


def matches_selection(value: object, compiled: CompiledQuery) -> bool:
    """Return whether a conclusive classifier verdict satisfies the compiled inclusion rule."""

    if value is None:
        return False
    selection = compiled.selection
    if selection.kind == 'values':
        return any(type(value) is type(candidate) and value == candidate for candidate in selection.values)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return value >= selection.value if selection.operator == 'gte' else value <= selection.value


def parse_datapoint_result(result: DataPointResult, compiled: CompiledQuery) -> TraceClassification:
    """Collapse EvaluatorQ's result tree into one terminal UI classification."""

    raw_result = result.model_dump(mode='json', by_alias=True)
    inputs = result.data_point.inputs
    trace_id = str(inputs.get('trace_id', ''))
    span_id = str(inputs.get('span_id', ''))
    try:
        if result.error:
            raise _TerminalResultError(result.error)
        jobs = result.job_results
        if jobs is None or len(jobs) != 1:
            raise _TerminalResultError('expected exactly one job result')
        job = jobs[0]
        if job.error:
            raise _TerminalResultError(job.error)
        scores = job.evaluator_scores
        if scores is None or len(scores) != 1:
            raise _TerminalResultError('expected exactly one evaluator score')
        evaluator_score = scores[0]
        if evaluator_score.error:
            raise _TerminalResultError(evaluator_score.error)

        score = evaluator_score.score
        value = _validate_verdict(score.value, compiled, score.raw_output)
        raw_answer = _raw_answer(score.raw_output)
        confidence, probabilities = _classification_details(raw_answer)
        return TraceClassification(
            trace_id=trace_id,
            span_id=span_id,
            value=value,
            confidence=confidence,
            probabilities=probabilities,
            matched=matches_selection(value, compiled),
            summary=score.explanation,
            raw_result=raw_result,
        )
    except _TerminalResultError as error:
        logger.warning(
            'Classifier result parsing failed for trace_id={} span_id={}: {}',
            trace_id,
            span_id,
            error,
        )
        return TraceClassification(
            trace_id=trace_id,
            span_id=span_id,
            error=str(error),
            raw_result=raw_result,
        )


def _validate_verdict(value: object, compiled: CompiledQuery, raw_output: dict[str, Any] | None) -> bool | float | str:
    """Reject jury failures and values outside the exact compiled task contract."""
    if raw_output:
        failure = raw_output.get('evaluation_error')
        if failure:
            message = failure.get('message') if isinstance(failure, Mapping) else str(failure)
            raise _TerminalResultError(message or 'Classifier evaluation failed')
        jury = raw_output.get('jury')
        if isinstance(jury, Mapping) and (jury.get('inconclusive') or jury.get('judges_failed')):
            errors: list[str] = []
            votes = jury.get('votes', [])
            if isinstance(votes, list):
                for vote in votes:
                    if isinstance(vote, Mapping):
                        if vote.get('abstained') is True:
                            raise _TerminalResultError('judge abstained')
                        error = vote.get('error')
                        if isinstance(error, str) and error:
                            errors.append(error)
            raise _TerminalResultError('; '.join(errors) or 'classifier jury returned an inconclusive verdict')
    kind = compiled.task.kind
    criteria = compiled.task.criteria
    if kind == 'choice' and isinstance(value, str) and isinstance(criteria, dict) and value in criteria:
        return value
    if kind == 'noul' and isinstance(value, bool):
        return value
    if kind == 'score' and isinstance(value, int | float) and not isinstance(value, bool) and 0 <= value <= 1:
        return float(value)
    raise _TerminalResultError(f'Invalid {kind} verdict: {value!r}')


def _raw_answer(raw_output: dict[str, Any] | None) -> dict[str, Any] | None:
    """Read the first and only classifier repetition from the documented Jury result tree."""

    if raw_output is None:
        return None
    try:
        jury = raw_output['jury']
        if not isinstance(jury, Mapping):
            raise TypeError
        votes = jury.get('votes')
        if not isinstance(votes, list):
            raise TypeError
        vote = votes[0]
        if not isinstance(vote, Mapping):
            raise TypeError
        repetitions = vote.get('repetitions')
        if not isinstance(repetitions, list):
            raise TypeError
        repetition = repetitions[0]
        if not isinstance(repetition, Mapping):
            raise TypeError
        answer = repetition.get('raw_output')
    except (IndexError, KeyError, TypeError) as error:
        raise _TerminalResultError('malformed classifier result tree') from error
    if answer is None:
        return None
    if not isinstance(answer, Mapping):
        raise _TerminalResultError('malformed classifier result tree')
    return dict(answer)


def _classification_details(raw_answer: dict[str, Any] | None) -> tuple[float | None, dict[str, float] | None]:
    """Keep optional confidence information absent when the classifier did not report it."""

    if raw_answer is None:
        return None, None
    confidence = raw_answer.get('confidence')
    probabilities = raw_answer.get('probabilities')
    if confidence is not None and (
        isinstance(confidence, bool)
        or not isinstance(confidence, int | float)
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        raise _TerminalResultError('malformed classifier confidence')
    if probabilities is not None:
        if not isinstance(probabilities, dict) or any(
            not isinstance(label, str)
            or isinstance(probability, bool)
            or not isinstance(probability, int | float)
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
            for label, probability in probabilities.items()
        ):
            raise _TerminalResultError('malformed classifier probabilities')
        probabilities = {label: float(probability) for label, probability in probabilities.items()}
    return (None if confidence is None else float(confidence)), probabilities


async def run_classifier(
    traces: tuple[TraceRecord, ...],
    projections: dict[str, TraceProjection],
    compiled: CompiledQuery,
    *,
    model: str,
    client: AsyncOpenAI,
    parallelism: int,
    on_complete: Callable[[TraceClassification], Awaitable[None]],
) -> list[TraceClassification]:
    """Evaluate all selected traces, forwarding each terminal result exactly once."""

    evaluator = build_classifier_evaluator(compiled, model=model, client=client)
    classifications: list[TraceClassification] = []

    async def complete(result: DataPointResult) -> None:
        classification = parse_datapoint_result(result, compiled)
        await on_complete(classification)
        classifications.append(classification)

    await evaluatorq(
        'classifier-trace-finder',
        data=[build_datapoint(trace, projections[trace.trace_id]) for trace in traces],
        evaluators=[evaluator],
        inference=False,
        datapoint_parallelism=parallelism,
        llm_parallelism=parallelism,
        on_datapoint_complete=complete,
        print_results=False,
        _send_results=False,
    )
    return classifications


class _TerminalResultError(ValueError):
    """Expected terminal failures that still deserve a trace classification."""
