"""Generic judge-panel orchestration and verdict aggregation."""

from __future__ import annotations

import asyncio
import re
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

from loguru import logger
from pydantic import BaseModel

from evaluatorq.common.tracing import (
    current_otel_context,
    set_span_attrs,
    set_span_error,
    truncate_for_span,
    with_span,
)
from evaluatorq.contracts import (
    EVAL_ERROR_RAW_OUTPUT_KEY,
    JURY_RAW_OUTPUT_KEY,
    JuryRepetition,
    JuryResult,
    JuryStats,
    JuryVote,
    StrEnum,
    TokenUsage,
)

if TYPE_CHECKING:
    from opentelemetry.trace import Span

VerdictValue = bool | float | str
TieBreak = Callable[[list[VerdictValue]], VerdictValue | None]

_UNSWAP = {'A': 'B', 'B': 'A'}


def _unswap(value: VerdictValue | None) -> VerdictValue | None:
    """Map a verdict from the swapped ordering back to the canonical A/B frame.

    Lives here rather than in ``pairwise`` because the judge span needs it too,
    and ``pairwise`` already imports from this module. Pure and per-ordering:
    only *flip detection* needs both orderings, un-swapping does not.

    Anything that is not ``'A'`` or ``'B'`` passes through unchanged, which is
    required rather than incidental: ``'tie'`` is symmetric and must not flip,
    and a non-comparative panel's bool / float / str verdicts have no slot frame
    to be mapped out of.
    """
    if value is None:
        return None
    return _UNSWAP.get(str(value), value)


# A custom panel aggregator sees ALL per-judge votes — including abstained and
# failed ones (filter on .success / .abstained yourself) — plus model and
# replacement, so it can weight or quorum. It returns the consensus verdict, or
# None for "no consensus" (inconclusive). The built-in keyword aggregators
# instead operate on decisive votes only (see _decisive_values). The runner
# derives stats / agreement / pass downstream regardless.
Aggregator = Callable[[list[JuryVote]], VerdictValue | None]
NumericAggName = Literal['mean_std', 'median', 'min', 'max']
AggregatorName = Literal['mode', 'majority', 'mean_std', 'median', 'min', 'max']
AggregatorSpec = AggregatorName | Aggregator


class VerdictKind(StrEnum):
    CATEGORICAL = 'categorical'
    NUMERIC = 'numeric'


class Prediction(BaseModel):
    """One judge pass returned by a caller-provided judge function."""

    value: VerdictValue | None = None
    explanation: str = ''
    token_usage: TokenUsage | None = None
    error: str | None = None
    abstained: bool = False

    @property
    def decisive(self) -> bool:
        return self.error is None and not self.abstained and self.value is not None


class JuryDeliberation(BaseModel):
    """Final verdict plus the serializable jury result."""

    verdict: VerdictValue | None = None
    explanation: str = ''
    jury: JuryResult
    token_usage: TokenUsage | None = None


def _sum_usage(usages: list[TokenUsage]) -> TokenUsage | None:
    if not usages:
        return None
    total = usages[0]
    for usage in usages[1:]:
        total = total + usage
    return total


def _plurality_vote(values: Sequence[VerdictValue]) -> tuple[VerdictValue | None, bool]:
    if not values:
        return None, False
    counts = Counter(values)
    top_count = max(counts.values())
    winners = [value for value, count in counts.items() if count == top_count]
    if len(winners) > 1:
        return None, True
    return winners[0], False


def _numeric_reduce(values: Sequence[VerdictValue], how: NumericAggName) -> float | None:
    """Reduce numeric verdicts to one float. ``mean_std`` returns the mean (the
    std rides along in `_jury_stats`); ``median``/``min``/``max`` are exact."""
    nums = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not nums:
        return None
    if how == 'median':
        ordered = sorted(nums)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2
    if how == 'min':
        return min(nums)
    if how == 'max':
        return max(nums)
    return sum(nums) / len(nums)


def _strict_majority(values: Sequence[VerdictValue]) -> VerdictValue | None:
    """Most common value only if it holds a strict >50% majority, else None."""
    if not values:
        return None
    value, count = Counter(values).most_common(1)[0]
    return value if count * 2 > len(values) else None


def _decisive_values(votes: Sequence[JuryVote]) -> list[VerdictValue]:
    return [v.value for v in votes if v.value is not None and not v.abstained and v.success]


# Built-in panel aggregators, each conforming to the public ``Aggregator``
# schema (list[JuryVote] -> verdict | None). Custom callables plug in the same
# way. ``mode`` here ignores ties (the runner handles tie_break separately).
def _agg_mode(votes: list[JuryVote]) -> VerdictValue | None:
    verdict, _tie = _plurality_vote(_decisive_values(votes))
    return verdict


def _agg_majority(votes: list[JuryVote]) -> VerdictValue | None:
    return _strict_majority(_decisive_values(votes))


def _make_numeric_agg(how: NumericAggName) -> Aggregator:
    def agg(votes: list[JuryVote]) -> VerdictValue | None:
        return _numeric_reduce(_decisive_values(votes), how)

    return agg


_AGGREGATORS: dict[str, Aggregator] = {
    'mode': _agg_mode,
    'majority': _agg_majority,
    'mean_std': _make_numeric_agg('mean_std'),
    'median': _make_numeric_agg('median'),
    'min': _make_numeric_agg('min'),
    'max': _make_numeric_agg('max'),
}

# Single source of truth for the keyword -> verdict-kind partition. Keep in sync
# with _AGGREGATORS and the AggregatorName literal; test_aggregator_registry_parity
# pins all three together.
_AGG_KIND: dict[str, VerdictKind] = {
    'mode': VerdictKind.CATEGORICAL,
    'majority': VerdictKind.CATEGORICAL,
    'mean_std': VerdictKind.NUMERIC,
    'median': VerdictKind.NUMERIC,
    'min': VerdictKind.NUMERIC,
    'max': VerdictKind.NUMERIC,
}


def validate_aggregator(aggregator: AggregatorSpec | None, verdict_kind: VerdictKind) -> None:
    """Reject a keyword aggregator that doesn't match the verdict kind.

    ``None`` (default) and custom callables always pass — a callable is trusted
    to handle whatever values its panel produces.
    """
    if aggregator is None or callable(aggregator):
        return
    if aggregator not in _AGG_KIND:
        raise ValueError(f'Unknown aggregator {aggregator!r}; expected one of {sorted(_AGG_KIND)} or a callable.')
    if _AGG_KIND[aggregator] is not verdict_kind:
        allowed = sorted(n for n, k in _AGG_KIND.items() if k is verdict_kind)
        raise ValueError(
            f'aggregator={aggregator!r} is {_AGG_KIND[aggregator].value}-only; '
            f'verdict_kind={verdict_kind.value!r} needs one of {allowed}.'
        )


def _jury_stats(values: Sequence[VerdictValue]) -> JuryStats | None:
    if not values:
        return None
    if all(isinstance(v, bool) for v in values):
        nums = [1.0 if v else 0.0 for v in values]
    elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        nums = [float(v) for v in values]
    else:
        return None
    mean = sum(nums) / len(nums)
    variance = sum((n - mean) ** 2 for n in nums) / len(nums)
    return JuryStats(mean=mean, std=variance**0.5)


def _agreement_rate(values: Sequence[VerdictValue]) -> float | None:
    if not values:
        return None
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        return None
    counts = Counter(values)
    return max(counts.values()) / len(values)


def _jury_explanation(votes: Sequence[JuryVote]) -> str:
    for vote in reversed(votes):
        if vote.error:
            return f'No judge produced a usable verdict; last error: {vote.error}'
    if any(v.abstained for v in votes):
        return 'No judge produced a usable verdict; all decisive judges abstained.'
    return 'No judge produced a usable verdict.'


def append_jury_summary(explanation: str | None, jury: JuryResult | None) -> str:
    """Append a compact jury summary to a scorer explanation."""
    base = explanation or ''
    if jury is None:
        return base
    parts = [f'{jury.judges_succeeded}/{jury.judges_configured} judges']
    # Omitted rather than rendered 'n/a' when there is nothing to report: a single
    # cyclic vote and a numeric panel both have no cross-judge agreement, and a
    # placeholder in a fixed slot reads as a measurement that came back empty.
    if jury.raw_agreement is not None:
        parts.append(f'raw agreement {jury.raw_agreement:.0%}')
    if jury.tie:
        parts.append('TIE (tie-break applied)')
    if jury.inconclusive:
        parts.append('INCONCLUSIVE')
    summary = f'[jury: {", ".join(parts)}]'
    return f'{base} {summary}' if base else summary


def attach_jury_raw_output(
    raw_output: dict[str, Any] | None,
    jury: JuryResult | None,
    *,
    error_payload: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Stash a serialized jury — and, when supplied, the judge's own failure — on an evaluator's ``raw_output``.

    The single writer for every scorer that carries a `JuryResult` through
    `EvaluationResult.raw_output` so the report converters can lift it back onto a typed
    field: `evaluatorq.llm_jury`, the adaptive pipeline's dynamic scorer, and the OWASP
    static bridge. All three hand-built this payload and had already diverged over
    whether the error key was written at all.

    ``raw_output`` is returned unchanged when ``jury`` is ``None``, ``None`` included: a
    pre-existing ``None`` must stay ``null`` in serialized reports because consumers
    distinguish "no raw output" from "an empty one". Otherwise the caller's own keys are
    preserved and only the two lifted keys are written.

    ``error_payload`` is built by the caller instead of derived here because the two
    callers that have one build it from different material — `common.judge.judge_error_payload`
    needs a `JudgeOutcome`, which the plain jury path never has (its votes carry a bare
    error string). Pass ``None`` whenever the panel reached a verdict.
    """
    if jury is None:
        if error_payload is not None:
            logger.warning('attach_jury_raw_output: no jury to attach, dropping the judge error payload with it')
        return raw_output
    merged = dict(raw_output or {})
    merged[JURY_RAW_OUTPUT_KEY] = jury.model_dump(mode='json')
    if error_payload is not None:
        merged[EVAL_ERROR_RAW_OUTPUT_KEY] = error_payload
    return merged


def as_semaphore(max_concurrency: int | asyncio.Semaphore | None) -> asyncio.Semaphore | None:
    """Normalize a concurrency limit to a semaphore (or None for unbounded).

    An existing semaphore passes through unchanged so several jury runs can
    share one budget — ``run_pairwise`` relies on this to bound its two
    concurrent orderings with a single limit.
    """
    if max_concurrency is None:
        return None
    if isinstance(max_concurrency, asyncio.Semaphore):
        return max_concurrency
    if max_concurrency < 1:
        raise ValueError(f'max_concurrency ({max_concurrency}) must be >= 1.')
    return asyncio.Semaphore(max_concurrency)


async def _call_prediction(
    judge_fn: Callable[[str], Awaitable[Prediction]],
    model: str,
    *,
    propagate_errors: bool = False,
    semaphore: asyncio.Semaphore | None = None,
) -> Prediction:
    try:
        if semaphore is None:
            return await judge_fn(model)
        async with semaphore:
            return await judge_fn(model)
    except Exception as exc:
        # When the caller has no redundancy to fall back on (a lone judge, no
        # replacements), let the error abort the run instead of silently
        # degrading to an inconclusive verdict across every datapoint.
        if propagate_errors:
            raise
        logger.warning('jury judge_fn raised: {}', exc)
        return Prediction(error=str(exc))


async def _judge_vote(
    *,
    model: str,
    judge_fn: Callable[[str], Awaitable[Prediction]],
    repetitions: int,
    verdict_kind: VerdictKind,
    tie_break: TieBreak | None,
    replacement: bool,
    numeric_how: NumericAggName,
    propagate_errors: bool = False,
    semaphore: asyncio.Semaphore | None = None,
    parent_context: object | None = None,
    label_swapped: bool | None = None,
) -> tuple[JuryVote, list[TokenUsage]]:
    """Run one judge (its repetitions) inside a per-judge span, then stamp the
    resolved ``JuryVote`` onto the span (RES-985). The span is a no-op when
    tracing is disabled; the vote/usages are unchanged either way."""
    async with with_span('orq.judge', parent_context=parent_context) as span:
        start = time.monotonic()
        # Identity up front so a propagate_errors=True abort still leaves a span
        # that says which judge died.
        set_span_attrs(
            span,
            {
                'judge.name': model,
                'judge.model': model,
                'judge.replacement': replacement,
                # Only set in comparative mode, where the same judge votes twice.
                'judge.label_swapped': label_swapped,
            },
        )
        vote, usages = await _compute_judge_vote(
            model=model,
            judge_fn=judge_fn,
            repetitions=repetitions,
            verdict_kind=verdict_kind,
            tie_break=tie_break,
            replacement=replacement,
            numeric_how=numeric_how,
            propagate_errors=propagate_errors,
            semaphore=semaphore,
        )
        _record_judge_span(span, vote, latency_ms=(time.monotonic() - start) * 1000.0, label_swapped=label_swapped)
    return vote, usages


def _record_judge_span(span: Span | None, vote: JuryVote, *, latency_ms: float, label_swapped: bool | None) -> None:
    """Set the outcome ``judge.*`` attributes on a judge span from its vote.

    Identity (``judge.name`` / ``judge.model`` / ``judge.replacement`` /
    ``judge.label_swapped``) is already stamped by `_judge_vote` at
    span-open and cannot change here, so it is not re-written. The verdict is
    stringified so bool / float / str verdicts share one attribute type.

    ``judge.verdict`` is the single verdict attribute, and it is always in the
    canonical frame. In comparative mode the labels a judge returns mean *slot*,
    not response — a judge naming the same response in both orderings
    necessarily says 'A' once and 'B' once — so a raw-frame attribute would make
    "how often did this judge pick response A" need a per-span join against
    ``judge.label_swapped`` first. The raw text the verdict was parsed from is
    still one level down, on the ``chat`` child's ``gen_ai.output.messages``,
    which is why a second raw-frame attribute here would only restate what the
    trace already holds (see the alias-removal note in ``docs/tracing.md``).

    Un-swapping is per-ordering and pure; only *flip detection* needs both
    orderings, and that lands on the parent as ``jury.flipped_judges``.

    Token usage and cost are not recorded here: they belong on the underlying
    ``chat`` spans, which the consumer rolls up.
    """
    canonical = _unswap(vote.value) if label_swapped else vote.value
    set_span_attrs(
        span,
        {
            'judge.verdict': None if canonical is None else str(canonical),
            'judge.success': vote.success,
            'judge.abstained': vote.abstained,
            'judge.latency_ms': round(latency_ms, 3),
            # Provider errors are unbounded; cap like every other text attribute.
            'judge.error': None if vote.error is None else truncate_for_span(vote.error),
            'judge.repetitions_failed': vote.repetitions_failed,
        },
    )
    if not vote.success:
        # The failure is swallowed (the panel carries on), but the span must not read as OK.
        set_span_error(span, vote.error or f'judge {vote.model} produced no usable verdict')


async def _compute_judge_vote(
    *,
    model: str,
    judge_fn: Callable[[str], Awaitable[Prediction]],
    repetitions: int,
    verdict_kind: VerdictKind,
    tie_break: TieBreak | None,
    replacement: bool,
    numeric_how: NumericAggName,
    propagate_errors: bool = False,
    semaphore: asyncio.Semaphore | None = None,
) -> tuple[JuryVote, list[TokenUsage]]:
    predictions = await asyncio.gather(*[
        _call_prediction(judge_fn, model, propagate_errors=propagate_errors, semaphore=semaphore)
        for _ in range(max(1, repetitions))
    ])
    usages = [p.token_usage for p in predictions if p.token_usage is not None]
    decisive = [p for p in predictions if p.decisive]
    abstained = bool(predictions) and not decisive and any(p.abstained for p in predictions)
    repetitions_raw = [
        JuryRepetition(value=p.value if p.decisive else None, explanation=p.explanation or None) for p in predictions
    ]
    # A pass has failed if it is neither decisive nor a CLEAN abstention: an error, or a
    # mechanically-unusable value=None / abstained=False pass. Counting only p.error let the
    # second case pass as a free abstention and keep consistency at 1.0 (RES-1251 review, item 6).
    # A pass that both errored and set abstained is a failure, not a clean abstention, so
    # `p.error is not None` overrides the abstained flag (RES-1251 review, item 21 - latent today,
    # nothing builds that combination, but the predicate should not let it through). decisive
    # already means "error is None and not abstained and value is not None", so a tie or a genuine
    # clean abstention is never counted. Shared jury layer, so it tightens redteam/simulation too.
    failed_count = sum(1 for p in predictions if not p.decisive and (not p.abstained or p.error is not None))

    if failed_count > 0 and failed_count < len(predictions):
        logger.warning('judge {} had {}/{} repetitions fail', model, failed_count, len(predictions))

    if not decisive:
        if abstained:
            explanation = next((p.explanation for p in predictions if p.abstained and p.explanation), '')
            return (
                JuryVote(
                    model=model,
                    replacement=replacement,
                    success=True,
                    abstained=True,
                    explanation=explanation,
                    repetitions=repetitions_raw,
                    repetitions_failed=failed_count,
                ),
                usages,
            )
        error = next((p.error for p in predictions if p.error), 'no successful prediction')
        return (
            JuryVote(
                model=model,
                replacement=replacement,
                success=False,
                error=error,
                repetitions=repetitions_raw,
                repetitions_failed=failed_count,
            ),
            usages,
        )

    values = [p.value for p in decisive if p.value is not None]
    tie = False
    if verdict_kind is VerdictKind.NUMERIC:
        value = _numeric_reduce(values, numeric_how)
    else:
        value, tie = _plurality_vote(values)
        if tie and tie_break is not None:
            value = tie_break(values)
    if value is None:
        return (
            JuryVote(
                model=model,
                replacement=replacement,
                success=True,
                abstained=True,
                explanation='Judge repetitions tied without a decisive tie-break.',
                repetitions=repetitions_raw,
                repetitions_failed=failed_count,
            ),
            usages,
        )
    representative = next(
        (p.explanation for p in decisive if p.value == value and p.explanation), decisive[0].explanation
    )
    return (
        JuryVote(
            model=model,
            replacement=replacement,
            success=True,
            value=value,
            explanation=representative,
            repetitions=repetitions_raw,
            repetitions_failed=failed_count,
        ),
        usages,
    )


async def run_jury(
    *,
    judge_fn: Callable[[str], Awaitable[Prediction]],
    panel: Sequence[str],
    repetitions: int = 1,
    replacement_judges: Sequence[str] | None = None,
    min_successful_judges: int = 1,
    verdict_kind: VerdictKind = VerdictKind.CATEGORICAL,
    tie_break: TieBreak | None = None,
    aggregator: AggregatorSpec | None = None,
    tie_break_label: str | None = None,
    propagate_errors: bool = False,
    max_concurrency: int | asyncio.Semaphore | None = None,
) -> JuryDeliberation:
    """Run a generic panel of judges and aggregate their verdicts.

    ``aggregator`` selects the panel consensus rule: a keyword (``mode`` /
    ``majority`` for categorical, ``mean_std`` / ``median`` / ``min`` / ``max``
    for numeric) or a custom ``Aggregator`` callable. ``None`` defaults to
    ``mode`` (categorical) or ``mean_std`` (numeric). The same numeric rule
    collapses a single judge's repetitions; ``tie_break`` applies only to
    ``mode`` plurality ties.

    ``propagate_errors`` re-raises a judge_fn exception instead of recording it
    as a failed vote. Callers set this when the panel has no redundancy (a lone
    judge with no replacements) so an outage aborts loudly rather than producing
    inconclusive verdicts on every datapoint.

    ``max_concurrency`` caps how many ``judge_fn`` calls run at once across the
    whole panel (judges x repetitions, replacements included). Pass an int for
    a run-local cap, or an existing ``asyncio.Semaphore`` to share one budget
    across several jury runs. ``None`` (default) keeps the fan-out unbounded.

    The deliberation runs inside an ``orq.jury`` span with the panel's
    aggregate attributes; each judge opens a child ``orq.judge`` span, and the
    judge's own LLM calls nest under that (RES-985). The full hierarchy is
    ``orq.evaluation`` -> ``orq.jury`` -> ``orq.judge`` -> ``chat {model}``.

    Comparative mode needs one jury span across BOTH label orderings, so it
    drives `_run_jury_core` directly rather than calling this function
    twice; see `evaluatorq.pairwise.run_pairwise`.
    """
    async with with_span('orq.jury') as jury_span:
        # Captured once and threaded into each judge so per-judge spans parent
        # to this jury span deterministically, regardless of gather scheduling.
        deliberation = await _run_jury_core(
            judge_fn=judge_fn,
            panel=panel,
            repetitions=repetitions,
            replacement_judges=replacement_judges,
            min_successful_judges=min_successful_judges,
            verdict_kind=verdict_kind,
            tie_break=tie_break,
            aggregator=aggregator,
            tie_break_label=tie_break_label,
            propagate_errors=propagate_errors,
            max_concurrency=max_concurrency,
            parent_context=current_otel_context(),
        )
        record_jury_span(
            jury_span,
            deliberation,
            aggregator=aggregator,
            verdict_kind=verdict_kind,
            min_successful_judges=min_successful_judges,
        )
    return deliberation


def _aggregator_name(aggregator: AggregatorSpec | None, verdict_kind: VerdictKind) -> str:
    """Display name for the consensus rule — the same defaulting _run_jury_core
    applies, with custom callables collapsed to ``'custom'``."""
    if isinstance(aggregator, str):
        return aggregator
    if callable(aggregator):
        return 'custom'
    return 'mean_std' if verdict_kind is VerdictKind.NUMERIC else 'mode'


def record_jury_span(
    span: Span | None,
    deliberation: JuryDeliberation,
    *,
    aggregator: AggregatorSpec | None,
    verdict_kind: VerdictKind,
    min_successful_judges: int,
) -> None:
    """Stamp panel-level attributes on the ``orq.jury`` span (RES-985).

    Comparative mode computes its aggregates from reconciled pair votes rather
    than a ``JuryResult``, so it stamps its own set (see
    ``pairwise._record_pairwise_span``) using the same ``jury.*`` vocabulary.
    ``jury.verdict`` is stringified so bool / float / str verdicts share one
    attribute type, matching ``judge.verdict``.

    No token usage or cost here: those are recorded once, on the underlying
    ``chat`` spans, and rolled up by the consumer.
    """
    j = deliberation.jury
    set_span_attrs(
        span,
        {
            'jury.verdict': None if deliberation.verdict is None else str(deliberation.verdict),
            'jury.aggregator': _aggregator_name(aggregator, verdict_kind),
            'jury.min_successful_judges': min_successful_judges,
            'jury.raw_agreement': j.raw_agreement,
            'jury.judges_succeeded': j.judges_succeeded,
            'jury.judges_configured': j.judges_configured,
            'jury.judges_failed': j.judges_failed,
            'jury.replacements_used': j.replacements_used,
            'jury.tie': j.tie,
            'jury.inconclusive': j.inconclusive,
        },
    )


async def _run_jury_core(
    *,
    judge_fn: Callable[[str], Awaitable[Prediction]],
    panel: Sequence[str],
    repetitions: int = 1,
    replacement_judges: Sequence[str] | None = None,
    min_successful_judges: int = 1,
    verdict_kind: VerdictKind = VerdictKind.CATEGORICAL,
    tie_break: TieBreak | None = None,
    aggregator: AggregatorSpec | None = None,
    tie_break_label: str | None = None,
    propagate_errors: bool = False,
    max_concurrency: int | asyncio.Semaphore | None = None,
    parent_context: object | None = None,
    label_swapped: bool | None = None,
    replacement: bool = False,
) -> JuryDeliberation:
    """Panel deliberation + aggregation (see `run_jury` for semantics).

    Split out so `run_jury` owns only the span; ``parent_context`` is the
    jury span's context, threaded into each judge span. ``replacement=True``
    marks the WHOLE panel as stand-ins — comparative mode promotes replacements
    at the pair level and runs them through a second core call, so without it
    those judge spans would claim to be configured judges."""
    if aggregator is None:
        aggregator = 'mean_std' if verdict_kind is VerdictKind.NUMERIC else 'mode'
    # Per-judge repetition collapse reuses the numeric keyword when one is set,
    # else falls back to mean (a custom callable only runs at the panel level).
    numeric_how: NumericAggName = 'mean_std'
    if isinstance(aggregator, str) and _AGG_KIND.get(aggregator) is VerdictKind.NUMERIC:
        numeric_how = cast('NumericAggName', aggregator)
    agg_fn: Aggregator = aggregator if callable(aggregator) else _AGGREGATORS[aggregator]

    semaphore = as_semaphore(max_concurrency)
    resolved_panel = resolve_panel(panel)
    # Dedup the replacement pool against the panel AND within itself; a repeated
    # stand-in (e.g. ['mistral-large', 'mistral-large']) would otherwise cast two
    # independent votes from one model and could manufacture a false consensus.
    seen: set[str] = set(resolved_panel)
    replacement_pool: list[str] = []
    for r in replacement_judges or []:
        if r and r not in seen:
            replacement_pool.append(r)
            seen.add(r)

    judge_results = await asyncio.gather(*[
        _judge_vote(
            model=model,
            judge_fn=judge_fn,
            repetitions=repetitions,
            verdict_kind=verdict_kind,
            tie_break=tie_break,
            replacement=replacement,
            numeric_how=numeric_how,
            propagate_errors=propagate_errors,
            semaphore=semaphore,
            parent_context=parent_context,
            label_swapped=label_swapped,
        )
        for model in resolved_panel
    ])

    votes: list[JuryVote] = []
    usages: list[TokenUsage] = []
    for vote, vote_usages in judge_results:
        votes.append(vote)
        usages.extend(vote_usages)

    failures = sum(1 for vote in votes if not vote.success)
    stand_ins = replacement_pool[:failures]
    if stand_ins:
        replacement_results = await asyncio.gather(*[
            _judge_vote(
                model=model,
                judge_fn=judge_fn,
                repetitions=repetitions,
                verdict_kind=verdict_kind,
                tie_break=tie_break,
                replacement=True,
                numeric_how=numeric_how,
                semaphore=semaphore,
                parent_context=parent_context,
                label_swapped=label_swapped,
            )
            for model in stand_ins
        ])
        for vote, vote_usages in replacement_results:
            votes.append(vote)
            usages.extend(vote_usages)

    decisive_votes = [v for v in votes if v.success and not v.abstained and v.value is not None]
    decisive_values = [v.value for v in decisive_votes if v.value is not None]
    inconclusive = len(decisive_votes) < max(1, min_successful_judges)
    tie = False
    verdict: VerdictValue | None = None

    if not inconclusive:
        # ``mode`` is special-cased so plurality ties route through tie_break and
        # set the tie flag; every other built-in keyword and custom callable is a
        # plain decisive-votes -> verdict reduction (no tie concept).
        if aggregator == 'mode':
            verdict, tie = _plurality_vote(decisive_values)
            if tie and tie_break is not None:
                verdict = tie_break(decisive_values)
        else:
            # Pass ALL votes — custom callables may want abstained/failed votes
            # for quorum/weighting; built-ins re-filter to decisive internally.
            verdict = agg_fn(votes)
        if verdict is None:
            inconclusive = True
            tie = False

    # Log degraded / collapsed jury states loudly (A4).
    if not decisive_votes:
        logger.error(
            'jury collapsed: 0/{} judges produced a usable verdict ({} failed)',
            len(resolved_panel),
            failures,
        )
    elif inconclusive:
        logger.warning(
            'jury inconclusive: {}/{} decisive, need {}',
            len(decisive_votes),
            len(resolved_panel),
            max(1, min_successful_judges),
        )

    if inconclusive:
        if decisive_votes:
            explanation = (
                f'Inconclusive: only {len(decisive_votes)} of {max(1, min_successful_judges)} '
                'required judges returned a usable verdict.'
            )
        else:
            explanation = _jury_explanation(votes)
    else:
        representative = next((v for v in decisive_votes if v.value == verdict), None)
        explanation = representative.explanation if representative else ''
        if tie:
            tie_label = tie_break_label if tie_break_label is not None else 'tie-break applied'
            explanation = f'[TIE — {tie_label}] {explanation}'

    jury = JuryResult(
        # replacement=True means the caller already promoted this whole panel as
        # stand-ins (comparative mode): none of them is a configured judge, so
        # they land in replacements_used instead. The caller derives the real
        # panel-level aggregates itself; these keep JuryResult self-consistent.
        judges_configured=0 if replacement else len(resolved_panel),
        judges_succeeded=len(decisive_votes),
        judges_failed=failures if not replacement else 0,
        replacements_used=len(stand_ins) + (len(resolved_panel) if replacement else 0),
        tie=tie,
        inconclusive=inconclusive,
        votes=votes,
        stats=None if inconclusive else _jury_stats(decisive_values),
        raw_agreement=None if inconclusive else _agreement_rate(decisive_values),
    )
    return JuryDeliberation(verdict=verdict, explanation=explanation, jury=jury, token_usage=_sum_usage(usages))


_FAMILY_MARKERS: tuple[tuple[str, str], ...] = (
    ('claude', 'anthropic'),
    ('chatgpt', 'openai'),
    ('gpt', 'openai'),
    ('o1', 'openai'),
    ('o3', 'openai'),
    ('o4', 'openai'),
    ('gemini', 'google'),
    ('palm', 'google'),
    ('llama', 'meta'),
    ('mixtral', 'mistral'),
    ('mistral', 'mistral'),
    ('command', 'cohere'),
    ('grok', 'xai'),
    ('deepseek', 'deepseek'),
    ('qwen', 'alibaba'),
    ('glm', 'zhipu'),
    ('minimax', 'minimax'),
)
_KNOWN_FAMILIES: frozenset[str] = frozenset(fam for _, fam in _FAMILY_MARKERS)


def provider_family(model_id: str) -> str:
    ident = (model_id or '').strip().lower()
    if not ident:
        return 'unknown'
    tokens = [t for t in re.split(r'[/\-_.: ]+', ident) if t]
    if not tokens:
        return 'unknown'
    if tokens[0] in _KNOWN_FAMILIES:
        return tokens[0]
    # Match a marker as a whole token, or as a prefix immediately followed by a
    # version DIGIT (gpt4o, o1, claude3). The digit guard is what stops the old
    # substring trap where a short marker bled into an unrelated word
    # (palmyra->palm, command->...): 'palmyra'.startswith('palm') is True but the
    # next char 'y' is alphabetic, so it no longer maps to google.
    for marker, family in _FAMILY_MARKERS:
        for tok in tokens:
            if tok == marker or (tok.startswith(marker) and len(tok) > len(marker) and tok[len(marker)].isdigit()):
                return family
    return 'unknown'


def _panel_composition_messages(panel: list[str], target_models: list[str], *, strict: bool = False) -> list[str]:
    messages: list[str] = []
    families = {provider_family(m) for m in panel}
    known = families - {'unknown'}
    if len(panel) > 1 and 'unknown' not in families and len(known) == 1:
        messages.append(
            f'Panel judges are all from a single provider family ({next(iter(known))}): {panel}. '
            'Correlated judges do not add the diversity a jury is meant to provide; '
            'prefer an odd, mixed-provider panel.'
        )
    target_families = {provider_family(m) for m in target_models} - {'unknown'}
    shared = known & target_families
    # For a single-judge run there is no diversity decision to act on, so the
    # advisory warning is pure noise (it would fire on the default gpt-4o-mini
    # eval vs gpt-4o target). Still surface it when the user opted into
    # strict_panel — there a self-judging lone judge is a configuration error.
    if shared and (len(panel) > 1 or strict):
        offenders = [m for m in panel if provider_family(m) in shared]
        shared_label = ', '.join(sorted(shared))
        messages.append(
            f'Judge(s) {offenders} share the target provider family ({shared_label}). '
            "Same-family self-judging may bias verdicts toward the target's own provider; "
            'prefer judges from a different provider than the target.'
        )
    return messages


def resolve_panel(panel: Sequence[str]) -> list[str]:
    """Dedup panel preserving insertion order, then validate non-empty."""
    resolved: list[str] = []
    for model in panel:
        if model and model not in resolved:
            resolved.append(model)
    if not resolved:
        raise ValueError('judge panel must contain at least one model')
    return resolved
