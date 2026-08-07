"""Pairwise (preference) judging over the shared jury.

A pairwise comparison asks judges to pick between two responses, A and B. To
control for position bias each judge is run in both orderings; this module holds
the ordering-independent core: reconciling a judge's two verdicts into one
consistency-gated vote (RES-760, ADR-24).
"""

from __future__ import annotations

import asyncio
import math
import statistics
from collections import Counter
from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel, Field

from evaluatorq.common.jury import (
    Prediction,
    VerdictValue,
    _agreement_rate,
    _plurality_vote,
    _sum_usage,
    as_semaphore,
    resolve_panel,
    run_jury,
)
from evaluatorq.contracts import TokenUsage  # noqa: TC001  # runtime-needed: pydantic field type on PairwiseComparison

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence
    from typing import Any

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
    vote: Literal['A', 'B', 'tie'] | None = Field(
        description="Reconciled vote: 'A', 'B', 'tie', or None if the judge abstained"
    )
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
    sigma: float | None = Field(
        default=None,
        description='BT-sigma discriminator (smaller = more reliable); set only when the report was built '
        "with aggregation='bt-sigma'",
    )


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
    bt_sigma: BTSigmaAggregation | None = Field(
        default=None,
        description="Reliability-weighted rollup; set only when the report was built with aggregation='bt-sigma'",
    )


def _rate(count: int, total: int) -> float | None:
    return count / total if total else None


class BTSigmaAggregation(BaseModel):
    """Reliability-weighted rollup of a pairwise run (BT-sigma, arXiv:2602.16610).

    Fitted unsupervised on the run's own reconciled votes: hard BT-sigma jointly
    infers the A-vs-B skill gap and a discriminator per judge, then re-derives
    each comparison's winner as a reliability-weighted vote (weight ``1/sigma``)
    instead of uniform plurality. Down-weights noisy judges; needs no labels.

    Two caveats of the two-item collapse, both handled here:

    - With only two items a judge's sigma is pinned by its own vote
      distribution, so a judge whose decisive votes are unanimous gets an
      arbitrarily small sigma that measures one-sidedness (e.g. position or
      verbosity bias), not reliability. Such judges are excluded from the
      weighting — they vote with a neutral weight, their sigma is dropped from
      ``judge_sigmas``, and ``fit_warnings`` names them.
    - ``p_a_beats_b``/``skill_gap`` come from the pooled global fit, while
      ``winners`` come from the per-comparison weighted vote. These are two
      different estimators and need not agree; read ``p_a_beats_b`` as the
      run-level headline and ``winners`` as the per-row calls.
    """

    p_a_beats_b: float = Field(description='Fitted global probability that A beats B (logistic of the skill gap)')
    skill_gap: float = Field(description='Fitted skill difference s_A - s_B')
    judge_sigmas: dict[str, float] = Field(
        default_factory=dict,
        description='Per-judge discriminator; smaller = sharper/more reliable. Empty on single-judge fallback.',
    )
    winners: list[str] = Field(
        default_factory=list,
        description='Reliability-weighted winner per comparison, aligned with the input order',
    )
    a_win_rate: float | None = Field(description='Weighted A-wins over decisive comparisons; None if none decisive')
    b_win_rate: float | None = Field(description='Weighted B-wins over decisive comparisons; None if none decisive')
    tie_rate: float = Field(description='Weighted-tie comparisons over all comparisons')
    inconclusive_rate: float = Field(description='Comparisons no judge decided, over all comparisons')
    converged: bool = Field(
        default=True,
        description='False when the fit stopped at the iteration cap; treat sigmas and weighted winners '
        'with suspicion then (the cap is also reported in fit_warnings)',
    )
    fit_warnings: list[str] = Field(default_factory=list, description='Degradations applied during the BT fit')


_VOTE_TO_P = {'A': 1.0, 'B': 0.0, 'tie': 0.5}
# Below this many decisive votes a judge's sigma is mostly prior, not evidence;
# the fit still runs but fit_warnings flags the judge.
_MIN_VOTES_FOR_SIGMA = 5


def _uniform_plurality_aggregation(
    comparisons: Sequence[PairwiseComparison], warnings: list[str], *, converged: bool = True
) -> BTSigmaAggregation:
    winners = [c.winner for c in comparisons]
    counts = Counter(winners)
    decisive = counts['A'] + counts['B']
    total = len(comparisons)
    return BTSigmaAggregation(
        p_a_beats_b=0.5,
        skill_gap=0.0,
        winners=winners,
        a_win_rate=_rate(counts['A'], decisive),
        b_win_rate=_rate(counts['B'], decisive),
        tie_rate=counts['tie'] / total if total else 0.0,
        inconclusive_rate=counts['inconclusive'] / total if total else 0.0,
        converged=converged,
        fit_warnings=warnings,
    )


def bt_sigma_aggregation(comparisons: Sequence[PairwiseComparison]) -> BTSigmaAggregation:
    """Fit hard BT-sigma over a run's reconciled votes and re-derive winners.

    The two "items" are the run's A and B sides; every decisive reconciled vote
    is one comparison between them. Reconciliation has already symmetrised
    position bias (both orderings per judge), satisfying the model's
    commutativity requirement. Categorical votes make hard BT-sigma the natural
    variant, which is also the paper's most robust one under inconsistency.

    Identical votes are collapsed into weighted records before the fit (a judge
    has at most three distinct judgements here: A, B, tie), so the fit cost
    stays flat no matter how many comparisons the run holds.

    Judges whose decisive votes are unanimous are excluded from the reliability
    weighting: in the two-item collapse their sigma measures one-sidedness, not
    reliability (see :class:`BTSigmaAggregation`). They vote with the median
    weight of the remaining judges (or uniformly when no judge remains), and
    ``fit_warnings`` names them.
    """
    from evaluatorq.ranking import JudgedComparison, comparisons_per_judge, fit_bt

    if any(v.vote is not None and not v.completed for c in comparisons for v in c.votes):
        return _uniform_plurality_aggregation(
            comparisons,
            ['single-ordering data: position-bias symmetrisation unavailable; used uniform plurality instead'],
        )

    # judge -> vote -> count. Collapsing keeps the optimum identical (the
    # likelihood is additive over identical records) while the per-iteration
    # cost drops from O(votes) to O(judges x 3).
    vote_counts: dict[str, Counter[str]] = {}
    for c in comparisons:
        for v in c.votes:
            if v.vote is not None:
                vote_counts.setdefault(v.model, Counter())[v.vote] += 1
    records = [
        JudgedComparison(judge=judge, item_a='A', item_b='B', p_a=_VOTE_TO_P[vote], weight=float(n))
        for judge, counts in sorted(vote_counts.items())
        for vote, n in sorted(counts.items())
    ]
    if not records:
        return BTSigmaAggregation(
            p_a_beats_b=0.5,
            skill_gap=0.0,
            winners=['inconclusive'] * len(comparisons),
            a_win_rate=None,
            b_win_rate=None,
            tie_rate=0.0,
            inconclusive_rate=1.0 if comparisons else 0.0,
            fit_warnings=['no decisive votes to fit on'],
        )

    fit = fit_bt(records, judge_sigma=True, hard=True)
    fit_warnings = list(fit.warnings)
    if not fit.converged:
        fit_warnings.append('non-converged fit: used uniform plurality winners instead')
        return _uniform_plurality_aggregation(comparisons, fit_warnings, converged=False)

    # Two-item degeneracy guard: a unanimous judge's sigma is a closed-form
    # function of its own one-sidedness (its marginal pins the fit), so 1/sigma
    # would hand the most degenerate judge on the panel an unbounded weight.
    one_sided = {judge for judge, counts in vote_counts.items() if len(counts) == 1}
    # Single-judge fallback leaves sigmas empty; weight uniformly then.
    sigmas = {judge: sigma for judge, sigma in fit.sigmas.items() if judge not in one_sided}
    weights = {judge: fit.reliability(judge) for judge in sigmas}
    if fit.sigmas and one_sided:
        neutral = statistics.median(weights.values()) if weights else 1.0
        for judge in sorted(one_sided):
            weights[judge] = neutral
            direction = next(iter(vote_counts[judge]))
            fit_warnings.append(
                f"judge '{judge}' voted '{direction}' on every decisive comparison; with two items its sigma "
                'measures one-sidedness, not reliability - weighted neutrally and excluded from judge_sigmas'
            )
    if fit.sigmas:
        for judge, n in sorted(comparisons_per_judge(records).items()):
            if n < _MIN_VOTES_FOR_SIGMA and judge not in one_sided:
                fit_warnings.append(
                    f"judge '{judge}' has only {int(n)} decisive vote(s); its sigma is mostly prior, not evidence"
                )

    winners: list[str] = []
    for c in comparisons:
        w: dict[str, float] = {'A': 0.0, 'B': 0.0, 'tie': 0.0}
        for v in c.votes:
            if v.vote is None:
                continue
            w[v.vote] += weights.get(v.model, 1.0)
        if not any(w.values()):
            winners.append('inconclusive')
            continue
        top = max(w.values())
        # isclose, not ==: mathematically equal weights can differ in the last
        # bits (asymmetric comparison counts), and a genuine tie must stay a
        # tie rather than crown whichever side rounded up.
        leaders = [k for k, val in w.items() if math.isclose(val, top, rel_tol=1e-9)]
        # Mirrors plurality semantics: a unique leader wins, a split is inconclusive.
        winners.append(leaders[0] if len(leaders) == 1 else 'inconclusive')

    counts = Counter(winners)
    decisive = counts['A'] + counts['B']
    total = len(comparisons)
    return BTSigmaAggregation(
        p_a_beats_b=1.0 / (1.0 + math.exp(-(fit.skills['A'] - fit.skills['B']))),
        skill_gap=fit.skills['A'] - fit.skills['B'],
        judge_sigmas=sigmas,
        winners=winners,
        a_win_rate=_rate(counts['A'], decisive),
        b_win_rate=_rate(counts['B'], decisive),
        tie_rate=counts['tie'] / total if total else 0.0,
        inconclusive_rate=counts['inconclusive'] / total if total else 0.0,
        converged=fit.converged,
        fit_warnings=fit_warnings,
    )


def build_report(comparisons: Sequence[PairwiseComparison], *, aggregation: str = 'plurality') -> PairwiseReport:
    """Roll a set of pairwise comparisons up into headline and per-judge metrics.

    ``aggregation='plurality'`` (default) keeps the existing uniform plurality
    consensus. ``aggregation='bt-sigma'`` additionally fits hard BT-sigma over
    the run and attaches the reliability-weighted rollup (``report.bt_sigma``)
    plus each judge's discriminator (``JudgeStats.sigma``); the headline
    plurality rates are unchanged so the two aggregations stay comparable."""
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

    bt_block: BTSigmaAggregation | None = None
    if aggregation == 'bt-sigma':
        bt_block = bt_sigma_aggregation(comparisons)
        for stats in judge_stats:
            stats.sigma = bt_block.judge_sigmas.get(stats.model)
    elif aggregation != 'plurality':
        msg = f"unknown aggregation {aggregation!r}; expected 'plurality' or 'bt-sigma'"
        raise ValueError(msg)

    return PairwiseReport(
        comparisons=total,
        a_win_rate=_rate(winners['A'], decisive),
        b_win_rate=_rate(winners['B'], decisive),
        tie_rate=winners['tie'] / total if total else 0.0,
        inconclusive_rate=winners['inconclusive'] / total if total else 0.0,
        mean_agreement=sum(agreements) / len(agreements) if agreements else None,
        per_judge=judge_stats,
        bt_sigma=bt_block,
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
    """
    # Normalized once so both orderings (and the replacement pass) draw from
    # the same budget rather than each minting their own.
    semaphore = as_semaphore(max_concurrency)
    resolved_panel = resolve_panel(panel)

    async def _ordering(models: Sequence[str], *, swapped: bool) -> tuple[dict[str, JuryVote], TokenUsage | None]:
        async def _fn(model: str) -> Prediction:
            return await (
                judge_fn(response_b, response_a, model) if swapped else judge_fn(response_a, response_b, model)
            )

        # Replacements are handled here at the pair level, not inside run_jury, so
        # a stand-in gets a fair shot in both orderings rather than being promoted
        # independently per ordering (and then silently dropped in reconciliation).
        deliberation = await run_jury(
            judge_fn=_fn,
            panel=models,
            repetitions=repetitions,
            replacement_judges=None,
            min_successful_judges=1,
            propagate_errors=propagate_errors,
            max_concurrency=semaphore,
        )
        return {v.model: v for v in deliberation.jury.votes}, deliberation.token_usage

    async def _both(models: Sequence[str]) -> tuple[dict[str, JuryVote], dict[str, JuryVote], list[TokenUsage]]:
        if not swap:
            first_votes, first_usage = await _ordering(models, swapped=False)
            return first_votes, {}, [u for u in (first_usage,) if u]
        (first_votes, first_usage), (second_votes, second_usage) = await asyncio.gather(
            _ordering(models, swapped=False),
            _ordering(models, swapped=True),
        )
        return first_votes, second_votes, [u for u in (first_usage, second_usage) if u]

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
        rep_first, rep_second, rep_usages = await _both(stand_ins)
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
                vote=_as_vote(vote),
                flipped=flipped,
                completed=completed,
                replacement=model in stand_in_set,
                explanation=_reconciled_explanation(vote, first_vote, second_vote),
            )
        )

    decisive = [v.vote for v in votes if v.vote is not None]
    winner = 'inconclusive' if len(decisive) < max(1, min_successful_judges) else pairwise_consensus(decisive)
    return PairwiseComparison(winner=winner, votes=votes, token_usage=_sum_usage(usages))


def _as_vote(value: VerdictValue | None) -> Literal['A', 'B', 'tie'] | None:
    """Narrow a reconciled verdict to the pairwise vote vocabulary.

    ``_decisive_value`` has already rejected anything outside the contract, so
    the cast is safe; keeping the check here too guards the type against a
    future caller that skips that gate.
    """
    if value is None:
        return None
    s = str(value)
    if s not in _PAIRWISE_VALUES:
        raise ValueError(f"pairwise vote {s!r} outside the contract; expected 'A', 'B', or 'tie'")
    return cast("Literal['A', 'B', 'tie']", s)
