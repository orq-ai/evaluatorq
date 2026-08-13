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
    _run_jury_core,
    _sum_usage,
    _unswap,
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


class RepetitionObservation(BaseModel):
    """One raw repetition pass of one judge in one ordering, canonicalized.

    ``verdict`` is in the ORIGINAL A/B orientation regardless of ordering: a
    'B' returned under the swapped ordering is recorded as 'A' here, so the
    observations are directly comparable across orderings. ``None`` is a pass
    that produced no usable verdict: a genuine abstention, an error, or an
    off-contract value. The per-vote ``repetition_failures`` count says how many
    of those Nones were errors or off-contract (as opposed to clean
    abstentions); those two lower reliability while an abstention does not
    (RES-1251). Nothing is silently collapsed.
    """

    ordering: Literal['ab', 'ba'] = Field(description="'ab' = original slots, 'ba' = swapped ordering")
    repetition: int = Field(ge=0, description='Call order within the ordering, 0-based')
    verdict: Literal['A', 'B', 'tie'] | None = Field(
        default=None, description='Canonicalized verdict; None = abstained or failed pass'
    )


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
    observations: list[RepetitionObservation] = Field(
        default_factory=list,
        description='Canonicalized per-repetition votes across both orderings (RES-1251); empty on '
        'runs saved before repetition capture existed',
    )
    repetition_failures: int = Field(
        default=0, ge=0, description='Repetition passes that raised an error, summed over both orderings'
    )


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
    consistency: float | None = Field(
        default=None,
        description='Within-datapoint repetition consistency in [0, 1] (RES-1251): on a repetition run this '
        'is the reliability the winner weights actually came from, so surface it next to sigma rather '
        'than leaving the reader to infer reliability from a pooled sigma that did not decide anything. '
        'None when the run had no usable repeats for this judge.',
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
    repetition_consistency: dict[str, float] = Field(
        default_factory=dict,
        description='Per-judge within-datapoint repetition consistency in [0, 1] (RES-1251); when non-empty, '
        'the winner weights came from THESE, not from the global-fit sigmas. Measures how often a judge '
        'agrees with itself on repeated passes of the same prompt - NOT task difficulty and NOT overall '
        'judge quality.',
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


def repetition_consistency(comparisons: Sequence[PairwiseComparison]) -> dict[str, float]:
    """Per-judge within-datapoint repetition consistency, in [0, 1] (RES-1251).

    The unit of analysis is one (judge, comparison, ordering) group: repeated
    passes of the SAME prompt. A group needs >= 2 decisive repetitions;
    its consistency is the mean pairwise agreement among them, then discounted
    by the share of that vote's passes that errored or came back off-contract
    (``repetition_failures``) so a flaky judge scores below a clean one; a
    genuine abstention is not counted as a failure and does not penalise. A
    comparison contributes at most ONE observation per judge (its groups
    averaged), so repetition count never multiplies a judge's evidence, and
    different datapoints are never compared to each other - which is exactly why
    this is interpretable as judge reliability where the global two-item fit was not.

    Empty dict when no judge has any qualifying group (e.g. repetitions=1 or a
    legacy run without observations).

    Known trade-off (RES-1251, for sign-off): the per-judge mean over groups is
    UNWEIGHTED, so at the recommended R=2 - where a group's agreement is only 0.0
    or 1.0 - a judge with one lucky group counts the same as a judge with many.
    Shrinkage toward the panel mean by group count would damp that; deferred as a
    modelling decision rather than changed unilaterally.
    """
    per_judge: dict[str, list[float]] = {}
    for c in comparisons:
        for v in c.votes:
            groups: dict[str, list[str]] = {}
            for o in v.observations:
                if o.verdict is not None:
                    groups.setdefault(o.ordering, []).append(o.verdict)
            group_scores = []
            for vals in groups.values():
                if len(vals) >= 2:
                    pairs = [a == b for i, a in enumerate(vals) for b in vals[i + 1 :]]
                    group_scores.append(sum(pairs) / len(pairs))
            if not group_scores:
                continue
            agreement = sum(group_scores) / len(group_scores)
            # A judge that errors or returns off-contract on some passes is less
            # reliable than one that answers cleanly, so discount agreement by the
            # share of THIS vote's passes that failed (RES-1251 review). Failures
            # are the None observations counted in ``repetition_failures``; a
            # genuine abstention is a None that is NOT a failure, so it is not
            # penalised - declining honestly should not cost reliability.
            n_obs = len(v.observations)
            completion = max(0.0, (n_obs - v.repetition_failures) / n_obs) if n_obs else 1.0
            per_judge.setdefault(v.model, []).append(agreement * completion)
    return {judge: sum(xs) / len(xs) for judge, xs in sorted(per_judge.items())}


# Floor for consistency-derived weights: a judge that never agreed with itself
# still casts a (heavily discounted) vote rather than being erased outright.
_MIN_CONSISTENCY_WEIGHT = 0.05


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
    # Consistency weights do not depend on the pooled fit, so a non-converged
    # fit only loses the run-level headline - repetition evidence still weights
    # the winners (RES-1251). Without repeats, fall back as before.
    rep_consistency = repetition_consistency(comparisons)
    if not fit.converged:
        if not rep_consistency:
            fit_warnings.append('non-converged fit: used uniform plurality winners instead')
            return _uniform_plurality_aggregation(comparisons, fit_warnings, converged=False)
        fit_warnings.append(
            'non-converged fit: p_a_beats_b/skill_gap unavailable (reported neutral); winners still '
            'weighted by repetition consistency'
        )

    # Two-item degeneracy guard: a unanimous judge's sigma is a closed-form
    # function of its own one-sidedness (its marginal pins the fit), so 1/sigma
    # would hand the most degenerate judge on the panel an unbounded weight.
    one_sided = {judge for judge, counts in vote_counts.items() if len(counts) == 1}
    # Single-judge fallback leaves sigmas empty; weight uniformly then.
    sigmas = {judge: sigma for judge, sigma in fit.sigmas.items() if fit.converged and judge not in one_sided}
    weights = {judge: fit.reliability(judge) for judge in sigmas}
    if fit.sigmas and one_sided:
        neutral = statistics.median(weights.values()) if weights else 1.0
        for judge in sorted(one_sided):
            weights[judge] = neutral
            direction = next(iter(vote_counts[judge]))
            if rep_consistency:
                # The repetition block below replaces every weight, so the
                # neutral assignment never survives; only the sigma exclusion
                # is real on this path.
                fit_warnings.append(
                    f"judge '{judge}' voted '{direction}' on every decisive comparison; with two items its sigma "
                    'measures one-sidedness, not reliability - excluded from judge_sigmas; its weight comes '
                    'from repetition consistency'
                )
            else:
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

    # RES-1251: when per-repetition observations exist, reliability weights come
    # from within-datapoint consistency instead of the global two-item fit, so
    # datapoint heterogeneity structurally cannot masquerade as judge noise.
    # The pooled fit still supplies the run-level headline (p_a_beats_b) - two
    # different questions, two different estimators.
    if rep_consistency:
        weights = {judge: max(c, _MIN_CONSISTENCY_WEIGHT) for judge, c in rep_consistency.items()}
        voted = {v.model for c in comparisons for v in c.votes if v.vote is not None}
        fallback = statistics.median(weights.values()) if weights else 1.0
        for judge in sorted(voted - set(weights)):
            weights[judge] = fallback
            # Not a fixed neutral: it is the median consistency of the MEASURED
            # judges, so name the number rather than call it neutral (review). A
            # lone measured judge makes this the same as that judge's weight.
            fit_warnings.append(
                f"judge '{judge}' has no repeated decisive observations; weighted at the median measured "
                f'consistency ({fallback:.2f})'
            )
        # Mirrors the estimator's per-ordering-group rule: a vote only has
        # repeats if some single ordering holds >= 2 decisive passes. Two
        # decisive passes split across orderings carry no consistency evidence.
        n_dp = sum(
            1
            for c in comparisons
            if any(
                any(n >= 2 for n in Counter(o.ordering for o in v.observations if o.verdict is not None).values())
                for v in c.votes
            )
        )
        fit_warnings.append(
            f'reliability weights from within-datapoint repetition consistency '
            f'({len(rep_consistency)} judge(s), {n_dp} datapoint(s) with repeats)'
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
        p_a_beats_b=1.0 / (1.0 + math.exp(-(fit.skills['A'] - fit.skills['B']))) if fit.converged else 0.5,
        skill_gap=fit.skills['A'] - fit.skills['B'] if fit.converged else 0.0,
        judge_sigmas=sigmas,
        winners=winners,
        a_win_rate=_rate(counts['A'], decisive),
        b_win_rate=_rate(counts['B'], decisive),
        tie_rate=counts['tie'] / total if total else 0.0,
        inconclusive_rate=counts['inconclusive'] / total if total else 0.0,
        converged=fit.converged,
        repetition_consistency=rep_consistency,
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
            stats.consistency = bt_block.repetition_consistency.get(stats.model)
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

        def _observations(model: str) -> tuple[list[RepetitionObservation], int]:
            """Canonicalized per-repetition votes from both orderings' JuryVotes.

            The jury already retains raw per-repetition verdicts; this keeps
            them on the PairwiseVote (swapped-ordering entries un-swapped) so
            reliability estimation can use repeated observations instead of
            only the collapsed vote (RES-1251)."""
            out: list[RepetitionObservation] = []
            failures = 0
            for ordering, jv, swapped in (
                ('ab', first_votes.get(model), False),
                ('ba', second_votes.get(model), True),
            ):
                if jv is None:
                    continue
                failures += jv.repetitions_failed
                for i, raw in enumerate(jv.repetitions):
                    value = raw if not swapped else _unswap(raw)
                    if value is None:
                        verdict: Literal['A', 'B', 'tie'] | None = None  # genuine abstention: a valid no-decision pass
                    elif str(value) in _PAIRWISE_VALUES:
                        verdict = cast("Literal['A', 'B', 'tie']", str(value))
                    else:
                        # Decisive-looking but off-contract (RES-1251 review): a
                        # malformed pass, not a silently dropped None. Count it
                        # like an error so it lowers reliability rather than
                        # passing as a free abstention.
                        verdict = None
                        failures += 1
                    out.append(
                        RepetitionObservation(
                            ordering=cast("Literal['ab', 'ba']", ordering), repetition=i, verdict=verdict
                        )
                    )
            return out, failures

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
            observations, repetition_failures = _observations(model)
            votes.append(
                PairwiseVote(
                    model=model,
                    vote=_as_vote(vote),
                    flipped=flipped,
                    completed=completed,
                    replacement=model in stand_in_set,
                    explanation=_reconciled_explanation(vote, first_vote, second_vote),
                    observations=observations,
                    repetition_failures=repetition_failures,
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
