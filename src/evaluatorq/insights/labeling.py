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
from evaluatorq.insights.presets import CODING_AGENT, CODING_CONVERSATION_LABELS, CODING_TOOL_LABELS
from evaluatorq.insights.transcript import conversation_view, tool_activity_chunks, tool_inventory
from evaluatorq.trace_finder.classifier import matches_selection

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from openai import AsyncOpenAI

    from evaluatorq.insights.usage import UsageLedger
    from evaluatorq.trace_finder.models import CompiledQuery, TraceRecord

MATCH_KEY = '__match__'


def _match_key(index: int, count: int) -> str:
    return MATCH_KEY if count == 1 else f'{MATCH_KEY[:-2]}_{index}__'


# Only these two kinds carry a live `error_exc` we can safely re-raise for `with_retry`
# to classify; PARSE/UNKNOWN failures are not transient and must not be retried.
_RETRYABLE_ERROR_KINDS = frozenset({JudgeError.TIMEOUT, JudgeError.API_CONNECTION, JudgeError.API_STATUS})


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
    usage: UsageLedger | None = None,
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
        try:
            outcome = await run_classify(client=client, model=model, cfg=cfg, request=request)
        except Exception:
            if usage is not None:
                usage.add('label', None)
            raise
        holder.append(outcome)
        if usage is not None:
            usage.add('label', outcome.token_usage)
        if outcome.error_kind in _RETRYABLE_ERROR_KINDS and outcome.error_exc is not None:
            raise outcome.error_exc
        return outcome

    try:
        return await with_retry(attempt, max_attempts=cfg.retry_count + 1, label='insights label classify')
    except Exception:
        if not holder:
            raise
        return holder[-1]


async def _ask(
    trace: TraceRecord,
    state: str,
    specs: Sequence[LabelSpec],
    *,
    client: AsyncOpenAI,
    model: str,
    cfg: LLMCallConfig,
    semaphore: asyncio.Semaphore,
    usage: UsageLedger | None,
    what: str,
) -> tuple[dict[str, LabelAnswer], str | None]:
    """Ask `specs` about `state` in one `/classify` call; a failed call fails every answer, never raises."""
    questions = {spec.name: spec.to_question(state) for spec in specs}
    async with semaphore:
        outcome = await _classify_with_retry(
            client=client, model=model, cfg=cfg, request=ClassifyRequest(state=state, questions=questions), usage=usage
        )
    if outcome.error_kind is not None or outcome.response is None:
        message = outcome.error_message or (
            outcome.error_kind.value if outcome.error_kind else 'classify reply produced no response'
        )
        logger.warning('Insights {} classify failed for trace {}: {}', what, trace.trace_id, message)
        failed = LabelAnswer(value=None, confidence=None, probabilities=None, error=message)
        return {spec.name: failed for spec in specs}, message
    return _read_answers(trace, questions, outcome.response.answers), None


# ponytail: the two chunked tool labels, named here; move onto LabelSpec if other labels ever chunk.
# An error counts as unfixed only if it is still broken at the end, so the last chunk decides it;
# for every other chunked label, a finding (yes, or a choice other than `none`) in any chunk stands.
_LAST_CHUNK_DECIDES = frozenset({'unfixed_error'})
_NO_FINDING = (False, 'none')


def _merge_chunks(name: str, answers: list[LabelAnswer]) -> LabelAnswer:
    """Merge one label's answers over consecutive chunks of a trace into one answer."""
    if len(answers) == 1 or name in _LAST_CHUNK_DECIDES:
        return answers[-1]
    answered = [answer for answer in answers if answer.error is None]
    found = [answer for answer in answered if answer.value not in _NO_FINDING]
    if found or len(answered) == len(answers):
        # Most confident finding wins; with none, the most confident clean answer.
        return max(
            found or answered,
            key=lambda answer: answer.confidence or max((answer.probabilities or {}).values(), default=0.0),
        )
    # No chunk found anything and one failed: the failed chunk may have held the finding.
    return next(answer for answer in answers if answer.error is not None)


def _read_answers(
    trace: TraceRecord, questions: dict[str, ClassifyQuestion], answers: dict[str, ClassifyAnswer]
) -> dict[str, LabelAnswer]:
    read: dict[str, LabelAnswer] = {}
    for name, question in questions.items():
        if name == MATCH_KEY or name.startswith('__match_'):
            continue
        answer = answers.get(name)
        if answer is None:
            logger.warning(
                'Insights label classify reply for trace {} is missing the {!r} answer', trace.trace_id, name
            )
            read[name] = LabelAnswer(
                value=None, confidence=None, probabilities=None, error=f'no answer returned for label {name!r}'
            )
            continue
        mapped = _map_answer(question, answer)
        if mapped.error is not None:
            logger.warning(
                'Insights label classify answer for trace {} label {!r} is unreadable: {}',
                trace.trace_id,
                name,
                mapped.error,
            )
        read[name] = mapped
    return read


def _population_match(
    trace: TraceRecord,
    questions: dict[str, ClassifyQuestion],
    raw: dict[str, ClassifyAnswer],
    keys: Sequence[str],
    compiled: Sequence[CompiledQuery],
) -> bool | None:
    """Return true only when every selected dimension matches; unreadable answers stay unknown."""
    matches: list[bool] = []
    for key, dimension in zip(keys, compiled, strict=True):
        answer = raw.get(key)
        if answer is None:
            logger.warning(
                'Insights label classify reply for trace {} is missing population-match answer {!r}',
                trace.trace_id,
                dimension.name,
            )
            continue
        mapped = _map_answer(questions[key], answer)
        if mapped.error is not None:
            logger.warning(
                'Insights label classify population-match answer for trace {} dimension {!r} is unreadable: {}',
                trace.trace_id,
                dimension.name,
                mapped.error,
            )
            continue
        matches.append(matches_selection(mapped.value, dimension))
    return all(matches) if len(matches) == len(compiled) else None


async def _label_one(
    trace: TraceRecord,
    *,
    labels: Sequence[LabelSpec],
    compiled: Sequence[CompiledQuery] | None,
    client: AsyncOpenAI,
    model: str,
    cfg: LLMCallConfig,
    semaphore: asyncio.Semaphore,
    usage: UsageLedger | None = None,
    coding: bool = False,
    coding_labels: Sequence[LabelSpec] | None = None,
) -> LabelOutcome:
    """Label one trace.

    With `coding` on, a first call asks `CODING_AGENT` over the trace's tool
    inventory. When it answers yes, the conversation call also asks the coding
    conversation labels, and a third call asks the coding tool labels over the
    tool activity (inputs, statuses, outputs). A coding check that fails leaves
    the coding labels unasked and says so in a warning; it never guesses yes.
    """
    answers: dict[str, LabelAnswer] = {}
    selected_coding = (
        list(coding_labels)
        if coding_labels is not None
        else [
            *CODING_CONVERSATION_LABELS,
            *CODING_TOOL_LABELS,
        ]
    )
    conversation_coding = [spec for spec in selected_coding if spec in CODING_CONVERSATION_LABELS]
    tool_coding = [spec for spec in selected_coding if spec in CODING_TOOL_LABELS]
    is_coding = False
    if coding:
        detected, _ = await _ask(
            trace,
            tool_inventory(trace),
            [CODING_AGENT],
            client=client,
            model=model,
            cfg=cfg,
            semaphore=semaphore,
            usage=usage,
            what='coding-agent check',
        )
        answers.update(detected)
        verdict = detected[CODING_AGENT.name]
        is_coding = verdict.error is None and verdict.value is True
        if verdict.error is not None:
            logger.warning(
                'Insights coding-agent check failed for trace {}; coding labels were not asked: {}',
                trace.trace_id,
                verdict.error,
            )

    state = conversation_view(trace)
    conversation_specs = [*labels, *(conversation_coding if is_coding else ())]
    questions: dict[str, ClassifyQuestion] = {spec.name: spec.to_question(state) for spec in conversation_specs}
    match_keys: list[str] = []
    if compiled:
        for index, dimension in enumerate(compiled):
            key = _match_key(index, len(compiled))
            match_keys.append(key)
            questions[key] = dimension.task.model_copy(update={'state': state})

    async def conversation_call() -> tuple[dict[str, ClassifyAnswer] | None, str | None]:
        if not questions:
            return None, None
        async with semaphore:
            outcome = await _classify_with_retry(
                client=client,
                model=model,
                cfg=cfg,
                request=ClassifyRequest(state=state, questions=questions),
                usage=usage,
            )
        if outcome.error_kind is not None or outcome.response is None:
            return None, outcome.error_message or (
                outcome.error_kind.value if outcome.error_kind else 'classify reply produced no response'
            )
        return outcome.response.answers, None

    async def tool_call() -> dict[str, LabelAnswer]:
        if not is_coding or not tool_coding:
            return {}
        chunks = tool_activity_chunks(trace)
        asked = await asyncio.gather(
            *(
                _ask(
                    trace,
                    chunk,
                    tool_coding,
                    client=client,
                    model=model,
                    cfg=cfg,
                    semaphore=semaphore,
                    usage=usage,
                    what='coding tool-activity',
                )
                for chunk in chunks
            )
        )
        return {
            spec.name: _merge_chunks(spec.name, [answers[spec.name] for answers, _ in asked]) for spec in tool_coding
        }

    (raw, error), tool_answers = await asyncio.gather(conversation_call(), tool_call())
    answers.update(tool_answers)

    if error is not None:
        logger.warning('Insights label classify failed for trace {}: {}', trace.trace_id, error)
        failed = LabelAnswer(value=None, confidence=None, probabilities=None, error=error)
        answers.update({spec.name: failed for spec in conversation_specs})
        return LabelOutcome(trace=trace, answers=answers, matched=None, error=error)
    if raw is None:
        return LabelOutcome(trace=trace, answers=answers, matched=None, error=None)

    answers.update(_read_answers(trace, questions, raw))

    matched: bool | None = None
    if compiled:
        matched = _population_match(trace, questions, raw, match_keys, compiled)

    return LabelOutcome(trace=trace, answers=answers, matched=matched, error=None)


async def label_traces(
    traces: Sequence[TraceRecord],
    *,
    labels: Sequence[LabelSpec],
    compiled: Sequence[CompiledQuery] | None,
    client: AsyncOpenAI,
    model: str,
    parallelism: int = 100,
    cfg: LLMCallConfig | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    usage: UsageLedger | None = None,
    coding: bool = False,
    coding_labels: Sequence[LabelSpec] | None = None,
) -> list[LabelOutcome]:
    """Label every trace, bounded by `parallelism` concurrent `/classify` calls.

    Per trace: render the conversation with `transcript.conversation_view`, ask
    every label's question plus (when `compiled` is given) the population-match
    question in a single request, and skip that call when neither is present.
    With `coding` on, each trace also gets the coding-agent check and, when it
    answers yes, the coding labels (see `_label_one`). A
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
                usage=usage,
                coding=coding,
                coding_labels=coding_labels,
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
