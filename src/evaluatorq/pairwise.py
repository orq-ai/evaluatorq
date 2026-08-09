"""Pairwise (preference) judging over the shared jury.

A pairwise comparison asks judges to pick between two responses, A and B. To
control for position bias each judge is run in both orderings; this module holds
the ordering-independent core: reconciling a judge's two verdicts into one
consistency-gated vote (RES-760, ADR-24).
"""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from evaluatorq.common.jury import (
    Prediction,
    VerdictValue,
    _agreement_rate,
    _plurality_vote,
    _run_jury_core,
    _sum_usage,
    as_semaphore,
    resolve_panel,
)
from evaluatorq.common.tracing import current_otel_context, set_span_attrs, with_span
from evaluatorq.contracts import TokenUsage  # noqa: TC001  # runtime-needed: pydantic field type on PairwiseComparison

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence
    from typing import Any

    from opentelemetry.trace import Span

    from evaluatorq.contracts import JuryVote

    # A judge comparing two responses in the order shown: (first, second, model) -> Prediction.
    # The first two args are opaque per-side payloads forwarded verbatim to judge_fn (a bare
    # string, or richer caller-defined side data); value is in {'A', 'B', 'tie'} referring to
    # the first or second response.
    PairwiseJudgeFn = Callable[[Any, Any, str], Awaitable[Prediction]]


def reconcile_pair(first: VerdictValue | None, second: VerdictValue | None) -> tuple[VerdictValue | None, bool]:
    """Reconcile a judge's verdicts from both orderings into one vote.

    Both verdicts are in the canonical A/B frame (the caller un-swaps the
    second ordering). Returns ``(vote, flipped)``: a consistent judge votes for
    the value it gave both times; an inconsistent one abstains (``vote=None``)
    and is recorded as a flip. A missing verdict abstains without a flip.
    """
    if first is None or second is None:
        return None, False
    if first == second:
        return first, False
    return None, True


def pairwise_consensus(votes: Sequence[VerdictValue | None]) -> str:
    """Reduce reconciled per-judge votes to a consensus winner.

    Abstained judges (``None``) are dropped. Returns the strict plurality of the
    remaining votes (``'A'``, ``'B'``, or ``'tie'``), or ``'inconclusive'`` when
    no judge had a preference or the vote splits with no plurality.
    """
    decisive = [v for v in votes if v is not None]
    winner, _tie = _plurality_vote(decisive)
    if winner is None:
        return 'inconclusive'
    return str(winner)


class PairwiseVote(BaseModel):
    """One judge's reconciled verdict for a single A-vs-B comparison."""

    model: str = Field(description='Judge model ID')
    vote: str | None = Field(description="Reconciled vote: 'A', 'B', 'tie', or None if the judge abstained")
    flipped: bool = Field(default=False, description='True if the judge disagreed with itself across orderings')
    completed: bool = Field(
        default=True,
        description='True if both orderings produced a decisive verdict, so a flip was actually possible',
    )
    replacement: bool = Field(default=False, description='True if this judge stood in for a failed configured judge')
    explanation: str = Field(default='', description='Explanation from the reconciled decisive ordering')


class PairwiseComparison(BaseModel):
    """The panel's result for a single A-vs-B comparison."""

    winner: str = Field(description="Consensus winner: 'A', 'B', 'tie', or 'inconclusive'")
    votes: list[PairwiseVote] = Field(default_factory=list, description='Per-judge reconciled votes')
    token_usage: TokenUsage | None = Field(
        default=None, description='Summed token usage across both orderings and any replacements'
    )


class JudgeStats(BaseModel):
    """Per-judge behaviour rolled up across a set of comparisons."""

    model: str
    a_rate: float | None = Field(
        description="Share of the judge's decisive picks that went to A; None if it never picked a side"
    )
    b_rate: float | None = Field(
        description="Share of the judge's decisive picks that went to B; None if it never picked a side"
    )
    position_bias: float = Field(
        description='Flips over completed pairs (both orderings decisive); 0.0 when no pair was flippable'
    )
    tie_rate: float = Field(description='Ties over total comparisons the judge saw')


class PairwiseReport(BaseModel):
    """Cross-comparison rollup of a pairwise run."""

    comparisons: int = Field(description='Number of comparisons in the run')
    a_win_rate: float | None = Field(
        description='Consensus A-wins over comparisons decided A or B; None if none were decisive'
    )
    b_win_rate: float | None = Field(
        description='Consensus B-wins over comparisons decided A or B; None if none were decisive'
    )
    tie_rate: float = Field(description='Comparisons whose consensus was a tie, over all comparisons')
    inconclusive_rate: float = Field(
        description='Comparisons with no consensus (panel too degraded to decide), over all comparisons'
    )
    mean_agreement: float | None = Field(
        description='Mean inter-judge agreement (modal vote share) across comparisons; None if none were decisive'
    )
    per_judge: list[JudgeStats] = Field(default_factory=list, description='Per-judge behaviour breakdown')


def _rate(count: int, total: int) -> float | None:
    return count / total if total else None


def build_report(comparisons: Sequence[PairwiseComparison]) -> PairwiseReport:
    """Roll a set of pairwise comparisons up into headline and per-judge metrics."""
    total = len(comparisons)
    winners = Counter(c.winner for c in comparisons)
    decisive = winners['A'] + winners['B']

    per_judge: dict[str, list[PairwiseVote]] = {}
    for comparison in comparisons:
        for vote in comparison.votes:
            per_judge.setdefault(vote.model, []).append(vote)

    judge_stats = [
        JudgeStats(
            model=model,
            a_rate=_rate(sum(v.vote == 'A' for v in votes), sum(v.vote in ('A', 'B') for v in votes)),
            b_rate=_rate(sum(v.vote == 'B' for v in votes), sum(v.vote in ('A', 'B') for v in votes)),
            # Flips only over pairs where flipping was possible; a failed or missing
            # ordering never had the chance to flip and would dilute the rate.
            position_bias=_rate(sum(v.flipped for v in votes), sum(v.completed for v in votes)) or 0.0,
            tie_rate=sum(v.vote == 'tie' for v in votes) / len(votes),
        )
        for model, votes in per_judge.items()
    ]

    # A single decisive vote scores agreement 1.0 (highest exactly when the panel is
    # most degraded); only comparisons with >=2 decisive votes carry real agreement.
    agreements = [
        rate
        for c in comparisons
        if len([v for v in c.votes if v.vote is not None]) >= 2
        and (rate := _agreement_rate([v.vote for v in c.votes if v.vote is not None])) is not None
    ]

    return PairwiseReport(
        comparisons=total,
        a_win_rate=_rate(winners['A'], decisive),
        b_win_rate=_rate(winners['B'], decisive),
        tie_rate=winners['tie'] / total if total else 0.0,
        inconclusive_rate=winners['inconclusive'] / total if total else 0.0,
        mean_agreement=sum(agreements) / len(agreements) if agreements else None,
        per_judge=judge_stats,
    )


_UNSWAP = {'A': 'B', 'B': 'A'}
_PAIRWISE_VALUES = frozenset({'A', 'B', 'tie'})


def _decisive_value(vote: JuryVote) -> VerdictValue | None:
    """The judge's verdict, or None if it failed or abstained.

    Raises ``ValueError`` on a decisive verdict outside the pairwise contract
    (``'A'``/``'B'``/``'tie'``) — ``run_pairwise`` is a public export, so a
    caller-supplied ``judge_fn`` returning e.g. a bool must not leak into the
    winner as ``'True'``.
    """
    if not vote.success or vote.abstained:
        return None
    if vote.value not in _PAIRWISE_VALUES:
        raise ValueError(f"pairwise judge returned {vote.value!r}; expected 'A', 'B', or 'tie'")
    return vote.value


def _unswap(value: VerdictValue | None) -> VerdictValue | None:
    """Map a verdict from the swapped ordering back to the canonical A/B frame."""
    if value is None:
        return None
    return _UNSWAP.get(str(value), value)


def _reconciled_explanation(
    value: VerdictValue | None, first_vote: JuryVote | None, second_vote: JuryVote | None
) -> str:
    """Explanation for a reconciled vote, from whichever ordering yielded ``value``."""
    if value is None:
        return ''
    if first_vote is not None and _decisive_value(first_vote) == value and first_vote.explanation:
        return first_vote.explanation
    if second_vote is not None and _unswap(_decisive_value(second_vote)) == value and second_vote.explanation:
        return second_vote.explanation
    return (first_vote.explanation if first_vote else '') or (second_vote.explanation if second_vote else '')


async def run_pairwise(
    *,
    judge_fn: PairwiseJudgeFn,
    panel: Sequence[str],
    response_a: Any,
    response_b: Any,
    swap: bool = True,
    repetitions: int = 1,
    replacement_judges: Sequence[str] | None = None,
    min_successful_judges: int = 1,
    propagate_errors: bool = False,
    max_concurrency: int | asyncio.Semaphore | None = None,
) -> PairwiseComparison:
    """Run a panel over one A-vs-B comparison and reconcile it into a verdict.

    Each judge runs through the shared :func:`run_jury`. When ``swap`` is on
    (default) every judge is also run with A and B exchanged; the two verdicts
    are un-swapped and passed through :func:`reconcile_pair`, so a judge that
    just follows slot order abstains and is recorded as a flip.

    Both orderings run concurrently. Replacement judges are promoted at the pair
    level — a stand-in is run in *both* orderings so it can cast a real
    reconciled vote, and a primary that fails in either ordering is what it
    stands in for. The winner is the plurality of the reconciled votes, or
    ``'inconclusive'`` when fewer than ``min_successful_judges`` cast a decisive
    reconciled vote.

    ``max_concurrency`` caps how many judge calls run at once across the whole
    comparison (judges x orderings x repetitions, replacements included). Pass
    an existing ``asyncio.Semaphore`` to share one budget across several
    comparisons. ``None`` (default) keeps the fan-out unbounded.

    Tracing: the whole comparison is ONE ``orq.pairwise_jury`` span (RES-985). Each
    ordering drives ``_run_jury_core`` rather than ``run_jury`` so it doesn't
    mint a second jury span whose aggregates describe half a comparison; every
    judge span hangs off this one, tagged ``judge.label_swapped``.
    """
    # Normalized once so both orderings (and the replacement pass) draw from
    # the same budget rather than each minting their own.
    semaphore = as_semaphore(max_concurrency)
    resolved_panel = resolve_panel(panel)
    jury_ctx: object | None = None

    async def _ordering(
        models: Sequence[str], *, swapped: bool, replacement: bool
    ) -> tuple[dict[str, JuryVote], TokenUsage | None]:
        async def _fn(model: str) -> Prediction:
            return await (
                judge_fn(response_b, response_a, model) if swapped else judge_fn(response_a, response_b, model)
            )

        # _run_jury_core, not run_jury: this comparison owns ONE orq.pairwise_jury span
        # across both orderings, and run_jury would open a second one per
        # ordering. Everything below the span (fan-out, per-judge spans,
        # repetition collapse) is identical.
        #
        # Replacements are handled here at the pair level, not inside the core,
        # so a stand-in gets a fair shot in both orderings rather than being
        # promoted independently per ordering (and then silently dropped in
        # reconciliation).
        deliberation = await _run_jury_core(
            judge_fn=_fn,
            panel=models,
            repetitions=repetitions,
            replacement_judges=None,
            min_successful_judges=1,
            propagate_errors=propagate_errors,
            max_concurrency=semaphore,
            label_swapped=swapped,
            parent_context=jury_ctx,
            replacement=replacement,
        )
        return {v.model: v for v in deliberation.jury.votes}, deliberation.token_usage

    async def _both(
        models: Sequence[str], *, replacement: bool = False
    ) -> tuple[dict[str, JuryVote], dict[str, JuryVote], list[TokenUsage]]:
        if not swap:
            first_votes, first_usage = await _ordering(models, swapped=False, replacement=replacement)
            return first_votes, {}, [u for u in (first_usage,) if u]
        (first_votes, first_usage), (second_votes, second_usage) = await asyncio.gather(
            _ordering(models, swapped=False, replacement=replacement),
            _ordering(models, swapped=True, replacement=replacement),
        )
        return first_votes, second_votes, [u for u in (first_usage, second_usage) if u]

    async with with_span('orq.pairwise_jury') as jury_span:
        # Captured before any judge runs so every judge span (both orderings,
        # replacements included) parents to this one comparison-level span.
        jury_ctx = current_otel_context()
        first_votes, second_votes, usages = await _both(resolved_panel)

        def _failed(model: str) -> bool:
            first = first_votes.get(model)
            if first is None or not first.success:
                return True
            if swap:
                second = second_votes.get(model)
                return second is None or not second.success
            return False

        num_failed = sum(1 for model in resolved_panel if _failed(model))
        seen = set(resolved_panel)
        stand_ins: list[str] = []
        for candidate in replacement_judges or []:
            if candidate and candidate not in seen:
                stand_ins.append(candidate)
                seen.add(candidate)
        stand_ins = stand_ins[:num_failed]

        stand_in_set = set(stand_ins)
        if stand_ins:
            rep_first, rep_second, rep_usages = await _both(stand_ins, replacement=True)
            first_votes.update(rep_first)
            second_votes.update(rep_second)
            usages.extend(rep_usages)

        votes: list[PairwiseVote] = []
        for model in (*resolved_panel, *stand_ins):
            first_vote = first_votes.get(model)
            second_vote = second_votes.get(model)
            first_value = _decisive_value(first_vote) if first_vote else None
            if swap:
                second_value = _unswap(_decisive_value(second_vote)) if second_vote else None
                vote, flipped = reconcile_pair(first_value, second_value)
                completed = first_value is not None and second_value is not None
            else:
                vote, flipped, completed = first_value, False, False
            votes.append(
                PairwiseVote(
                    model=model,
                    vote=_as_str(vote),
                    flipped=flipped,
                    completed=completed,
                    replacement=model in stand_in_set,
                    explanation=_reconciled_explanation(vote, first_vote, second_vote),
                )
            )

        decisive = [v.vote for v in votes if v.vote is not None]
        winner = 'inconclusive' if len(decisive) < max(1, min_successful_judges) else pairwise_consensus(decisive)
        comparison = PairwiseComparison(winner=winner, votes=votes, token_usage=_sum_usage(usages))
        _record_pairwise_span(
            jury_span,
            comparison,
            panel_size=len(resolved_panel),
            min_successful_judges=min_successful_judges,
            swap=swap,
        )
    return comparison


def _as_str(value: VerdictValue | None) -> str | None:
    return None if value is None else str(value)


def _record_pairwise_span(
    span: Span | None,
    comparison: PairwiseComparison,
    *,
    panel_size: int,
    min_successful_judges: int,
    swap: bool,
) -> None:
    """Stamp the comparison-level attributes on the ``orq.pairwise_jury`` span.

    Deliberately reuses the ``jury.*`` namespace so one dashboard reads both
    modes: ``jury.verdict`` is the winner, ``judges_succeeded`` counts judges
    that cast a reconciled vote. ``jury.flipped`` / ``jury.flipped_judges`` are
    comparative-only — judges that contradicted themselves across the two
    orderings, i.e. position bias.

    ``jury.judges_failed`` counts only judges with NO reconciled vote and no
    flip: a flipped judge answered both times, so that is position bias, not a
    failure. Counting it as failed would give one attribute two different
    meanings across the two modes.
    """
    flipped = [v.model for v in comparison.votes if v.flipped]
    decisive = [v.vote for v in comparison.votes if v.vote is not None]
    set_span_attrs(
        span,
        {
            'jury.verdict': comparison.winner,
            'jury.aggregator': 'pairwise_plurality',
            'jury.min_successful_judges': min_successful_judges,
            'jury.raw_agreement': _agreement_rate(decisive),
            'jury.judges_configured': panel_size,
            'jury.judges_succeeded': len(decisive),
            'jury.judges_failed': sum(
                1 for v in comparison.votes if not v.replacement and v.vote is None and not v.flipped
            ),
            'jury.replacements_used': sum(1 for v in comparison.votes if v.replacement),
            'jury.inconclusive': comparison.winner == 'inconclusive',
            'jury.flipped': len(flipped),
            # Which judges, not just how many: a per-judge attribute is impossible
            # (a judge span closes before the other ordering reconciles against it).
            'jury.flipped_judges': ','.join(sorted(flipped)) or None,
            'jury.swap': swap,
        },
    )
