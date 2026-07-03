"""Pairwise (preference) judging over the shared jury.

A pairwise comparison asks judges to pick between two responses, A and B. To
control for position bias each judge is run in both orderings; this module holds
the ordering-independent core: reconciling a judge's two verdicts into one
consistency-gated vote (RES-760, ADR-24).
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from evaluatorq.common.jury import Prediction, VerdictValue, _plurality_vote, run_jury

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from evaluatorq.contracts import JuryVote

    # A judge comparing two responses in the order shown: (first, second, model) -> Prediction
    # with value in {'A', 'B', 'tie'} referring to the first or second response.
    PairwiseJudgeFn = Callable[[str, str, str], Awaitable[Prediction]]


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


class PairwiseComparison(BaseModel):
    """The panel's result for a single A-vs-B comparison."""

    winner: str = Field(description="Consensus winner: 'A', 'B', 'tie', or 'inconclusive'")
    votes: list[PairwiseVote] = Field(default_factory=list, description='Per-judge reconciled votes')


class JudgeStats(BaseModel):
    """Per-judge behaviour rolled up across a set of comparisons."""

    model: str
    a_rate: float | None = Field(
        description="Share of the judge's decisive picks that went to A; None if it never picked a side"
    )
    b_rate: float | None = Field(
        description="Share of the judge's decisive picks that went to B; None if it never picked a side"
    )
    position_bias: float = Field(description='Flips over total comparisons the judge saw')
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
            position_bias=sum(v.flipped for v in votes) / len(votes),
            tie_rate=sum(v.vote == 'tie' for v in votes) / len(votes),
        )
        for model, votes in per_judge.items()
    ]

    return PairwiseReport(
        comparisons=total,
        a_win_rate=_rate(winners['A'], decisive),
        b_win_rate=_rate(winners['B'], decisive),
        tie_rate=winners['tie'] / total if total else 0.0,
        per_judge=judge_stats,
    )


_UNSWAP = {'A': 'B', 'B': 'A'}


def _decisive_value(vote: JuryVote) -> VerdictValue | None:
    """The judge's verdict, or None if it failed or abstained."""
    if not vote.success or vote.abstained:
        return None
    return vote.value


def _unswap(value: VerdictValue | None) -> VerdictValue | None:
    """Map a verdict from the swapped ordering back to the canonical A/B frame."""
    if value is None:
        return None
    return _UNSWAP.get(str(value), value)


async def run_pairwise(
    *,
    judge_fn: PairwiseJudgeFn,
    panel: Sequence[str],
    response_a: str,
    response_b: str,
    swap: bool = True,
    repetitions: int = 1,
    replacement_judges: Sequence[str] | None = None,
    min_successful_judges: int = 1,
    propagate_errors: bool = False,
) -> PairwiseComparison:
    """Run a panel over one A-vs-B comparison and reconcile it into a verdict.

    Each judge runs through the shared :func:`run_jury`. When ``swap`` is on
    (default) every judge is also run with A and B exchanged; the two verdicts
    are un-swapped and passed through :func:`reconcile_pair`, so a judge that
    just follows slot order abstains and is recorded as a flip. The winner is
    the plurality of the reconciled votes.
    """

    async def _judge_first(model: str) -> Prediction:
        return await judge_fn(response_a, response_b, model)

    first = await run_jury(
        judge_fn=_judge_first,
        panel=panel,
        repetitions=repetitions,
        replacement_judges=replacement_judges,
        min_successful_judges=min_successful_judges,
        propagate_errors=propagate_errors,
    )
    first_votes = {v.model: v for v in first.jury.votes}

    if not swap:
        votes = [
            PairwiseVote(model=model, vote=_as_str(_decisive_value(vote)), flipped=False)
            for model, vote in first_votes.items()
        ]
        return PairwiseComparison(winner=pairwise_consensus([v.vote for v in votes]), votes=votes)

    async def _judge_second(model: str) -> Prediction:
        return await judge_fn(response_b, response_a, model)

    second = await run_jury(
        judge_fn=_judge_second,
        panel=panel,
        repetitions=repetitions,
        replacement_judges=replacement_judges,
        min_successful_judges=min_successful_judges,
        propagate_errors=propagate_errors,
    )
    second_votes = {v.model: v for v in second.jury.votes}

    votes = []
    for model, first_vote in first_votes.items():
        second_vote = second_votes.get(model)
        first_value = _decisive_value(first_vote)
        second_value = _unswap(_decisive_value(second_vote)) if second_vote else None
        vote, flipped = reconcile_pair(first_value, second_value)
        votes.append(PairwiseVote(model=model, vote=_as_str(vote), flipped=flipped))

    return PairwiseComparison(winner=pairwise_consensus([v.vote for v in votes]), votes=votes)


def _as_str(value: VerdictValue | None) -> str | None:
    return None if value is None else str(value)
