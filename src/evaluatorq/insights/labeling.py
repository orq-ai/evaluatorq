"""Label pass: one `/classify` request per trace, carrying every requested label plus the optional population-match question.

One retry layer: `with_retry` wraps `run_classify`, retried only for
`JudgeError.TIMEOUT`/`JudgeError.API_CONNECTION` outcomes (mirroring their
underlying `error_exc`, since `run_classify` itself never raises — see
`_classify_with_retry`). `trace_finder/classifier.py` does not retry at all,
so there is nothing else to mirror there; this is the one retry layer for
this call path (`run_classify` already disarms the client's own retries).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from evaluatorq.common.judge import (
    ClassifyAnswer,
    ClassifyOutcome,
    ClassifyQuestion,
    ClassifyRequest,
    JudgeError,
    run_classify,
)
from evaluatorq.common.retry import with_retry
from evaluatorq.contracts import LLMCallConfig
from evaluatorq.insights.models import LabelAnswer, LabelSpec
from evaluatorq.trace_finder.classifier import matches_selection
from evaluatorq.trace_finder.projection import project_trace

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from openai import AsyncOpenAI

    from evaluatorq.trace_finder.models import CompiledQuery, TraceRecord

MATCH_KEY = '__match__'

# Only these two kinds carry a live `error_exc` we can safely re-raise for `with_retry`
# to classify; PARSE/UNKNOWN failures are not transient and must not be retried.
_RETRYABLE_ERROR_KINDS = frozenset({JudgeError.TIMEOUT, JudgeError.API_CONNECTION})


@dataclass
class LabelOutcome:
    """One trace's labels from a single `/classify` round trip, plus the optional population-match verdict."""

    trace: TraceRecord
    answers: dict[str, LabelAnswer]
    matched: bool | None
    error: str | None


def _default_cfg(model: str) -> LLMCallConfig:
    """Mirror `trace_finder.classifier.build_classifier_evaluator`'s default classify config."""
    return LLMCallConfig(model=model, timeout_ms=90_000)


def _map_answer(question: ClassifyQuestion, answer: ClassifyAnswer) -> LabelAnswer:
    """Read the field matching `question.kind`, mirroring `judge._classify_verdict` so labels and the finder agree.

    `noul` -> the threshold-derived boolean verdict (`noul_threshold` is exactly
    what turns the probability into that boolean, matching the finder's own
    classify -> value mapping), with the raw probability preserved as
    `probabilities={'true': answer.noul, 'false': 1 - answer.noul}` rather than
    discarded. `choice` -> the label string plus its probability distribution.
    `score` -> the level index normalized to [0, 1]. Any shape mismatch (wrong
    `answer.type`, an out-of-criteria choice, an out-of-range score) is
    unreadable and reported as `error`, never guessed; the caller logs it.
    """
    if answer.type != question.kind:
        return LabelAnswer(
            value=None,
            confidence=None,
            probabilities=None,
            error=f'classify answer type {answer.type!r} does not match question kind {question.kind!r}',
        )
    if question.kind == 'noul':
        if answer.noul is None:
            return LabelAnswer(
                value=None, confidence=None, probabilities=None, error='noul answer is missing its probability'
            )
        return LabelAnswer(
            value=answer.noul >= question.noul_threshold,
            confidence=answer.confidence,
            probabilities={'true': answer.noul, 'false': 1 - answer.noul},
            error=None,
        )
    if question.kind == 'choice':
        if answer.choice is None or not isinstance(question.criteria, dict) or answer.choice not in question.criteria:
            return LabelAnswer(
                value=None,
                confidence=None,
                probabilities=None,
                error=f'choice answer {answer.choice!r} is missing or not in the question criteria',
            )
        return LabelAnswer(
            value=answer.choice, confidence=answer.confidence, probabilities=answer.probabilities, error=None
        )
    # kind == 'score'
    if answer.score is None or not isinstance(question.criteria, list):
        return LabelAnswer(
            value=None,
            confidence=None,
            probabilities=None,
            error='score answer is missing or the question criteria is not an ordered level list',
        )
    top = len(question.criteria) - 1
    if top <= 0 or not 0.0 <= answer.score <= top:
        return LabelAnswer(
            value=None, confidence=None, probabilities=None, error=f'score {answer.score!r} is outside the level range'
        )
    return LabelAnswer(
        value=min(1.0, max(0.0, answer.score / top)), confidence=answer.confidence, probabilities=None, error=None
    )


async def _classify_with_retry(
    *,
    client: AsyncOpenAI,
    model: str,
    cfg: LLMCallConfig,
    request: ClassifyRequest,
) -> ClassifyOutcome:
    """Call `run_classify`, retrying only a transient failure.

    `run_classify` never raises — every failure comes back as a `ClassifyOutcome`
    with `error_kind` set. To reuse `with_retry` (the canonical backoff loop)
    without a second, hand-rolled retry layer, a retryable outcome re-raises its
    own `error_exc` so `with_retry`'s exception classifier can see it; the final
    attempt's outcome (success or not) is returned as-is.
    """
    holder: list[ClassifyOutcome] = []

    async def attempt() -> ClassifyOutcome:
        outcome = await run_classify(client=client, model=model, cfg=cfg, request=request)
        holder.append(outcome)
        if outcome.error_kind in _RETRYABLE_ERROR_KINDS and outcome.error_exc is not None:
            raise outcome.error_exc
        return outcome

    try:
        return await with_retry(attempt, label='insights label classify')
    except Exception:  # noqa: BLE001 - the retryable branch above always leaves a recorded outcome behind
        return holder[-1]


async def _label_one(
    trace: TraceRecord,
    *,
    labels: Sequence[LabelSpec],
    compiled: CompiledQuery | None,
    client: AsyncOpenAI,
    model: str,
    cfg: LLMCallConfig,
    semaphore: asyncio.Semaphore,
) -> LabelOutcome:
    state = project_trace(trace).payload

    questions: dict[str, ClassifyQuestion] = {}
    if compiled is not None:
        questions[MATCH_KEY] = compiled.task.model_copy(update={'state': state})
    for label in labels:
        questions[label.name] = label.to_question(state)

    if not questions:
        return LabelOutcome(trace=trace, answers={}, matched=None, error=None)

    request = ClassifyRequest(state=state, questions=questions)

    async with semaphore:
        outcome = await _classify_with_retry(client=client, model=model, cfg=cfg, request=request)

    if outcome.error_kind is not None or outcome.response is None:
        message = outcome.error_message or (
            outcome.error_kind.value if outcome.error_kind else 'classify reply produced no response'
        )
        logger.warning('Insights label classify failed for trace {}: {}', trace.trace_id, message)
        failed = LabelAnswer(value=None, confidence=None, probabilities=None, error=message)
        return LabelOutcome(trace=trace, answers={label.name: failed for label in labels}, matched=None, error=message)

    answers: dict[str, LabelAnswer] = {}
    for label in labels:
        answer = outcome.response.answers.get(label.name)
        if answer is None:
            logger.warning(
                'Insights label classify reply for trace {} is missing the {!r} answer', trace.trace_id, label.name
            )
            answers[label.name] = LabelAnswer(
                value=None, confidence=None, probabilities=None, error=f'no answer returned for label {label.name!r}'
            )
            continue
        mapped = _map_answer(questions[label.name], answer)
        if mapped.error is not None:
            logger.warning(
                'Insights label classify answer for trace {} label {!r} is unreadable: {}',
                trace.trace_id,
                label.name,
                mapped.error,
            )
        answers[label.name] = mapped

    matched: bool | None = None
    if compiled is not None:
        match_answer = outcome.response.answers.get(MATCH_KEY)
        if match_answer is None:
            logger.warning(
                'Insights label classify reply for trace {} is missing the population-match answer', trace.trace_id
            )
        else:
            mapped_match = _map_answer(questions[MATCH_KEY], match_answer)
            if mapped_match.error is not None:
                logger.warning(
                    'Insights label classify population-match answer for trace {} is unreadable: {}',
                    trace.trace_id,
                    mapped_match.error,
                )
                matched = None
            else:
                matched = matches_selection(mapped_match.value, compiled)

    return LabelOutcome(trace=trace, answers=answers, matched=matched, error=None)


async def label_traces(
    traces: Sequence[TraceRecord],
    *,
    labels: Sequence[LabelSpec],
    compiled: CompiledQuery | None,
    client: AsyncOpenAI,
    model: str,
    parallelism: int = 100,
    cfg: LLMCallConfig | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[LabelOutcome]:
    """Label every trace with one `/classify` request each, bounded by `parallelism`.

    Per trace: project the trace to its classifier state, ask every label's
    question plus (when `compiled` is given) the population-match question in
    a single request, and skip the call entirely when neither is present. A
    per-trace failure never raises — it comes back as a `LabelOutcome` with
    `error` set and every answer failed, per the "per-trace failures never
    fail a run" house rule.
    """
    resolved_cfg = cfg if cfg is not None else _default_cfg(model)
    if parallelism <= 0:
        raise ValueError('parallelism must be greater than zero')
    semaphore = asyncio.Semaphore(parallelism)
    total = len(traces)
    completed = 0
    completed_lock = asyncio.Lock()

    async def run_one(trace: TraceRecord) -> LabelOutcome:
        nonlocal completed
        try:
            outcome = await _label_one(
                trace,
                labels=labels,
                compiled=compiled,
                client=client,
                model=model,
                cfg=resolved_cfg,
                semaphore=semaphore,
            )
        except Exception as exc:  # noqa: BLE001 - an unexpected trace shape must not fail the whole pass
            message = str(exc) or type(exc).__name__
            logger.warning('Insights label processing failed for trace {}: {}', trace.trace_id, message)
            failed = LabelAnswer(value=None, confidence=None, probabilities=None, error=message)
            outcome = LabelOutcome(
                trace=trace,
                answers={label.name: failed for label in labels},
                matched=None,
                error=message,
            )
        finally:
            if on_progress is not None:
                async with completed_lock:
                    completed += 1
                    current = completed
                try:
                    on_progress(current, total)
                except Exception as exc:  # noqa: BLE001 - progress reporting must not abort result collection
                    logger.warning('Insights label progress callback failed: {}', exc)
        return outcome

    results: list[LabelOutcome | None] = [None] * total
    next_index = 0

    async def worker() -> None:
        nonlocal next_index
        while next_index < total:
            index = next_index
            next_index += 1
            results[index] = await run_one(traces[index])

    await asyncio.gather(*(worker() for _ in range(min(parallelism, total))))
    return [outcome for outcome in results if outcome is not None]
