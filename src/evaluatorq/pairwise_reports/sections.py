"""Section builders for pairwise run reports.

Renderer-agnostic: these produce :class:`~evaluatorq.contracts.ReportSection`
objects holding plain data, which ``export_html`` dispatches on. Same split as
``simulation.reports.sections`` / ``redteam.reports.sections``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from evaluatorq.contracts import ReportSection

if TYPE_CHECKING:
    from evaluatorq.pairwise import PairwiseVote
    from evaluatorq.pairwise_run import PairwiseEntry, PairwiseRun

# A judge that contradicts itself across orderings has no real preference, so
# its votes are noise rather than signal. Flag at the point where that starts
# to matter for reading the consensus.
POSITION_BIAS_WARN = 0.15

# Above this share of undecidable comparisons the win rates are being computed
# over a thin slice of the run, so the card says so rather than reading as a
# neutral fact. Compared with ``>``, unlike POSITION_BIAS_WARN above: each keeps
# the operator it shipped with, and exactly 10% inconclusive is not yet worth a
# warning where a bias of exactly 0.15 is.
INCONCLUSIVE_WARN = 0.1


def vote_label(vote: str | None, run: PairwiseRun) -> str:
    """Render a raw vote value using the run's side labels."""
    if vote == 'A':
        return run.label_a
    if vote == 'B':
        return run.label_b
    if vote == 'tie':
        return 'tie'
    return 'abstained'


def winner_label(winner: str, run: PairwiseRun) -> str:
    if winner == 'inconclusive':
        return 'inconclusive'
    return vote_label(winner, run)


def _is_split(entry: PairwiseEntry) -> bool:
    """True when the panel did not speak with one voice on this comparison.

    Either the judges landed on more than one decisive value, or the panel was
    too degraded to decide at all. These are the rows worth opening.
    """
    if entry.comparison.winner == 'inconclusive':
        return True
    decisive = {v.vote for v in entry.comparison.votes if v.vote is not None}
    return len(decisive) > 1


def _build_consensus_section(run: PairwiseRun) -> ReportSection:
    report = run.rollup()
    total = report.comparisons
    # Win rates are computed over decided comparisons only, so a mostly
    # inconclusive run can still post a strong-looking rate. Carry the decided
    # count through so the renderer can say so out loud.
    #
    # Counted from the entries rather than reconstructed from the rates:
    # ``PairwiseReport`` exposes no decided count, and inverting tie_rate +
    # inconclusive_rate would make the caption depend on that float arithmetic
    # round-tripping. Counting is exact by construction.
    decided = sum(1 for e in run.entries if e.comparison.winner in ('A', 'B'))
    return ReportSection(
        kind='pairwise_consensus',
        title='Consensus',
        data={
            'label_a': run.label_a,
            'label_b': run.label_b,
            'a_win_rate': report.a_win_rate,
            'b_win_rate': report.b_win_rate,
            'tie_rate': report.tie_rate,
            'inconclusive_rate': report.inconclusive_rate,
            'mean_agreement': report.mean_agreement,
            'comparisons': total,
            'decided': decided,
        },
    )


def observed_swap(run: PairwiseRun) -> bool:
    """Whether the data shows both orderings actually landed for some judge.

    ``run.swap`` is recorded intent and can disagree with what ran. The data
    settles it: ``pairwise.py`` sets ``completed=False`` on every vote when
    swapping is off, so a single completed vote anywhere proves swapping
    happened. The implication only runs one way — a swapped run whose every
    pair failed one ordering also shows nothing — which is why this reads as
    "observed" rather than as a correction of the flag.
    """
    return any(v.completed for e in run.entries for v in e.comparison.votes)


def _build_judges_section(run: PairwiseRun) -> ReportSection:
    report = run.rollup()
    replacements = {v.model for e in run.entries for v in e.comparison.votes if v.replacement}
    # Whether this judge ever had a pair where a flip was actually possible.
    # ``position_bias`` divides flips by completed pairs and falls back to 0.0
    # when that denominator is empty, so a judge whose every pair failed one
    # ordering would otherwise read as perfectly steady. ``swap`` being on is
    # not enough: the orderings still have to have both landed.
    measured = {
        stat.model: any(v.completed for e in run.entries for v in e.comparison.votes if v.model == stat.model)
        for stat in report.per_judge
    }
    rows: list[dict[str, Any]] = [
        {
            'model': stat.model,
            'a_rate': stat.a_rate,
            'b_rate': stat.b_rate,
            'tie_rate': stat.tie_rate,
            # Gated on the data alone. ``run.swap`` is redundant here — a run
            # with swapping off leaves every vote incomplete, so ``measured`` is
            # already False — and consulting it would let a mis-set flag decide
            # what the votes have already settled.
            'position_bias': stat.position_bias if measured[stat.model] else None,
            'biased': measured[stat.model] and stat.position_bias >= POSITION_BIAS_WARN,
            # Carried so the renderer can tell the unmeasurable cases apart:
            # the run never swapped, or it swapped but this judge never landed
            # both orderings of any pair.
            'measured': measured[stat.model],
            'replacement': stat.model in replacements,
        }
        for stat in report.per_judge
    ]
    return ReportSection(
        kind='pairwise_judges',
        title='Judges',
        data={
            'label_a': run.label_a,
            'label_b': run.label_b,
            'swap': run.swap,
            'observed_swap': any(measured.values()),
            'rows': rows,
        },
    )


def _vote_cell(vote: PairwiseVote | None, model: str, run: PairwiseRun) -> dict[str, Any]:
    """One judge's cell for a comparison row.

    ``vote`` is None when this judge cast nothing for this comparison, which is
    distinct from casting an abstention; both render as "abstained" but only the
    latter carries an explanation.
    """
    if vote is None:
        return {
            'model': model,
            'vote': None,
            'label': 'abstained',
            'flipped': False,
            'explanation': '',
            'present': False,
        }
    return {
        'model': model,
        'vote': vote.vote,
        'label': vote_label(vote.vote, run),
        'flipped': bool(vote.flipped),
        'explanation': vote.explanation,
        'present': True,
    }


def judge_order(run: PairwiseRun) -> list[str]:
    """Every judge that voted anywhere in the run, configured panel first.

    A fixed column order across all rows is what makes a split panel read as a
    ragged column while scanning. It has to be the union, not the first
    comparison's votes: a replacement judge is promoted per pair, so it can
    appear only partway through a run.
    """
    order: list[str] = list(run.judges)
    for entry in run.entries:
        for vote in entry.comparison.votes:
            if vote.model not in order:
                order.append(vote.model)
    return order


def _build_comparisons_section(run: PairwiseRun) -> ReportSection:
    order = judge_order(run)
    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(run.entries, start=1):
        votes = {v.model: v for v in entry.comparison.votes}
        rows.append({
            'index': index,
            'question': entry.question,
            'response_a': entry.response_a,
            'response_b': entry.response_b,
            'winner': entry.comparison.winner,
            'winner_label': winner_label(entry.comparison.winner, run),
            'split': _is_split(entry),
            'votes': [_vote_cell(votes.get(model), model, run) for model in order],
        })

    return ReportSection(
        kind='pairwise_comparisons',
        title='Comparisons',
        data={
            'label_a': run.label_a,
            'label_b': run.label_b,
            'judges': order,
            'rows': rows,
        },
    )


def build_report_sections(run: PairwiseRun) -> list[ReportSection]:
    """Build every section of a pairwise run report, in display order."""
    return [
        _build_consensus_section(run),
        _build_judges_section(run),
        _build_comparisons_section(run),
    ]
