"""Generic judge-panel orchestration and verdict aggregation."""

from __future__ import annotations

import asyncio
import re
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from typing import Literal, cast

from loguru import logger
from pydantic import BaseModel

from evaluatorq.contracts import JuryResult, JuryStats, JuryVote, StrEnum, TokenUsage

VerdictValue = bool | float | str
TieBreak = Callable[[list[VerdictValue]], VerdictValue | None]

# A custom panel aggregator sees the full per-judge votes (model, abstained,
# replacement, value) so it can weight or quorum, and returns the consensus
# verdict — or None for "no consensus" (inconclusive). The runner still derives
# stats / agreement / pass downstream.
Aggregator = Callable[[list[JuryVote]], VerdictValue | None]
CategoricalAggName = Literal['mode', 'majority']
NumericAggName = Literal['mean_std', 'median', 'min', 'max']
AggregatorName = Literal['mode', 'majority', 'mean_std', 'median', 'min', 'max']
AggregatorSpec = AggregatorName | Aggregator
_CATEGORICAL_AGGS: frozenset[str] = frozenset(('mode', 'majority'))
_NUMERIC_AGGS: frozenset[str] = frozenset(('mean_std', 'median', 'min', 'max'))


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
    std rides along in :func:`_jury_stats`); ``median``/``min``/``max`` are exact."""
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


def validate_aggregator(aggregator: AggregatorSpec | None, verdict_kind: VerdictKind) -> None:
    """Reject a keyword aggregator that doesn't match the verdict kind.

    ``None`` (default) and custom callables always pass — a callable is trusted
    to handle whatever values its panel produces.
    """
    if aggregator is None or callable(aggregator):
        return
    if aggregator not in _AGGREGATORS:
        raise ValueError(f'Unknown aggregator {aggregator!r}; expected one of {sorted(_AGGREGATORS)} or a callable.')
    if verdict_kind is VerdictKind.NUMERIC and aggregator not in _NUMERIC_AGGS:
        raise ValueError(f'aggregator={aggregator!r} is categorical-only; numeric verdict_kind needs one of {sorted(_NUMERIC_AGGS)}.')
    if verdict_kind is VerdictKind.CATEGORICAL and aggregator not in _CATEGORICAL_AGGS:
        raise ValueError(f'aggregator={aggregator!r} is numeric-only; categorical verdict_kind needs one of {sorted(_CATEGORICAL_AGGS)}.')


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
    rate = f'{jury.raw_agreement:.0%}' if jury.raw_agreement is not None else 'n/a'
    flags: list[str] = []
    if jury.tie:
        flags.append('TIE (tie-break applied)')
    if jury.inconclusive:
        flags.append('INCONCLUSIVE')
    suffix = f', {", ".join(flags)}' if flags else ''
    summary = f'[jury: {jury.judges_succeeded}/{jury.judges_configured} judges, raw agreement {rate}{suffix}]'
    return f'{base} {summary}' if base else summary


async def _call_prediction(
    judge_fn: Callable[[str], Awaitable[Prediction]], model: str, *, propagate_errors: bool = False
) -> Prediction:
    try:
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
) -> tuple[JuryVote, list[TokenUsage]]:
    predictions = await asyncio.gather(
        *[_call_prediction(judge_fn, model, propagate_errors=propagate_errors) for _ in range(max(1, repetitions))]
    )
    usages = [p.token_usage for p in predictions if p.token_usage is not None]
    decisive = [p for p in predictions if p.decisive]
    abstained = bool(predictions) and not decisive and any(p.abstained for p in predictions)
    repetitions_raw = [p.value if p.decisive else None for p in predictions]
    failed_count = sum(1 for p in predictions if p.error is not None)

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
    """
    if aggregator is None:
        aggregator = 'mean_std' if verdict_kind is VerdictKind.NUMERIC else 'mode'
    # Per-judge repetition collapse reuses the numeric keyword when one is set,
    # else falls back to mean (a custom callable only runs at the panel level).
    numeric_how: NumericAggName = 'mean_std'
    if isinstance(aggregator, str) and aggregator in _NUMERIC_AGGS:
        numeric_how = cast('NumericAggName', aggregator)
    agg_fn: Aggregator = aggregator if callable(aggregator) else _AGGREGATORS[aggregator]

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
            replacement=False,
            numeric_how=numeric_how,
            propagate_errors=propagate_errors,
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
            verdict = agg_fn(decisive_votes)
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
        judges_configured=len(resolved_panel),
        judges_succeeded=len(decisive_votes),
        judges_failed=failures,
        replacements_used=len(stand_ins),
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
            'Same-family self-judging may bias verdicts toward the target\'s own provider; '
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
